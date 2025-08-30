"""
Unified Contamination Detection for REV model verification.

This module implements comprehensive contamination detection using statistical tests,
perplexity analysis, n-gram overlap, HDC behavioral signatures, and temperature variation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
from collections import defaultdict, Counter
import logging
import warnings
import hashlib
import math
import json
from pathlib import Path
import pickle

from ..hypervector.hamming import HammingLUT, pack_binary_vector, hamming_distance_cpu
from ..hypervector.operations.hamming_lut import pack_binary_vector_simd
from ..hdc.encoder import HypervectorEncoder, HypervectorConfig
from ..hdc.behavioral_sites import BehavioralSites, ProbeFeatures
from .blackbox import BlackBoxVerifier, ModelProvider, APIConfig

logger = logging.getLogger(__name__)


class ContaminationType(Enum):
    """Types of model contamination."""
    NONE = "none"
    DATA_LEAKAGE = "data_leakage"
    MEMORIZATION = "memorization"
    FINE_TUNING = "fine_tuning"
    DISTILLATION = "distillation"
    BENCHMARK_OVERLAP = "benchmark_overlap"
    ADVERSARIAL = "adversarial"
    HYBRID = "hybrid"


class DetectionMode(Enum):
    """Detection modes for contamination analysis."""
    FAST = "fast"  # Quick statistical tests
    COMPREHENSIVE = "comprehensive"  # Full multi-modal analysis
    TARGETED = "targeted"  # Focus on specific contamination type
    ADAPTIVE = "adaptive"  # Adapt based on initial findings


@dataclass
class MemorizationEvidence:
    """Evidence of memorization in model responses."""
    verbatim_matches: List[str]
    near_duplicates: List[Tuple[str, float]]  # (text, similarity)
    perplexity_anomalies: List[float]
    consistency_violations: List[Dict[str, Any]]
    statistical_significance: float
    
    @property
    def memorization_score(self) -> float:
        """Compute overall memorization score."""
        scores = []
        if self.verbatim_matches:
            scores.append(min(1.0, len(self.verbatim_matches) / 10))
        if self.near_duplicates:
            scores.append(np.mean([sim for _, sim in self.near_duplicates]))
        if self.perplexity_anomalies:
            scores.append(min(1.0, len(self.perplexity_anomalies) / 5))
        if self.consistency_violations:
            scores.append(min(1.0, len(self.consistency_violations) / 3))
        
        return np.mean(scores) if scores else 0.0


@dataclass
class NGramOverlap:
    """N-gram overlap analysis results."""
    unigram_overlap: float
    bigram_overlap: float
    trigram_overlap: float
    unique_ngrams: Set[str]
    common_ngrams: Dict[str, int]  # n-gram -> count
    jaccard_similarity: float
    
    @property
    def overlap_score(self) -> float:
        """Compute weighted overlap score."""
        return (
            0.2 * self.unigram_overlap +
            0.3 * self.bigram_overlap +
            0.5 * self.trigram_overlap
        )


@dataclass
class ContaminationReport:
    """Comprehensive contamination detection report."""
    
    # Overall assessment
    contamination_type: ContaminationType
    contamination_score: float  # Overall score [0, 1]
    confidence: float  # Confidence in detection [0, 1]
    
    # Component analyses
    memorization_evidence: MemorizationEvidence
    ngram_overlap: NGramOverlap
    perplexity_stats: Dict[str, float]
    temperature_consistency: Dict[float, float]  # temp -> consistency
    
    # Multi-modal detection results
    token_level_score: float
    semantic_similarity_score: float
    behavioral_pattern_score: float
    benchmark_overlap_score: float
    
    # Detailed analysis
    suspicious_patterns: List[str]
    distance_statistics: Dict[str, float]
    hdc_signatures: Dict[str, np.ndarray]
    
    # Evidence and remediation
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)
    
    def is_contaminated(self, threshold: float = 0.7) -> bool:
        """Check if contamination score exceeds threshold."""
        return self.contamination_score > threshold
    
    def get_summary(self) -> str:
        """Get human-readable summary of contamination findings."""
        summary = []
        summary.append(f"Contamination Type: {self.contamination_type.value}")
        summary.append(f"Overall Score: {self.contamination_score:.3f}")
        summary.append(f"Confidence: {self.confidence:.3f}")
        
        if self.memorization_evidence.memorization_score > 0.5:
            summary.append(f"⚠️ Memorization detected: {self.memorization_evidence.memorization_score:.3f}")
        
        if self.ngram_overlap.overlap_score > 0.6:
            summary.append(f"⚠️ High n-gram overlap: {self.ngram_overlap.overlap_score:.3f}")
        
        if self.benchmark_overlap_score > 0.7:
            summary.append(f"⚠️ Benchmark overlap detected: {self.benchmark_overlap_score:.3f}")
        
        if self.remediation_suggestions:
            summary.append("\nRemediation Suggestions:")
            for suggestion in self.remediation_suggestions[:3]:
                summary.append(f"  • {suggestion}")
        
        return "\n".join(summary)


@dataclass
class BenchmarkDataset:
    """Reference benchmark dataset for overlap detection."""
    name: str
    prompts: List[str]
    expected_responses: Optional[List[str]] = None
    signatures: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedContaminationDetector:
    """
    Unified contamination detector with comprehensive multi-modal analysis.
    
    Features:
    - Statistical tests for memorization detection
    - Perplexity-based anomaly detection
    - N-gram overlap analysis
    - Response consistency checking across temperature variations
    - Multi-modal detection (token, semantic, behavioral)
    - Benchmark dataset cross-referencing
    """
    
    def __init__(
        self,
        hdc_encoder: Optional[HypervectorEncoder] = None,
        behavioral_sites: Optional[BehavioralSites] = None,
        blackbox_verifier: Optional[BlackBoxVerifier] = None,
        benchmark_datasets: Optional[List[BenchmarkDataset]] = None,
        cache_dir: Optional[Path] = None,
        detection_mode: DetectionMode = DetectionMode.COMPREHENSIVE
    ):
        """
        Initialize unified contamination detector.
        
        Args:
            hdc_encoder: HDC encoder for semantic analysis
            behavioral_sites: Behavioral pattern analyzer
            blackbox_verifier: Black-box model verifier
            benchmark_datasets: Known benchmark datasets for cross-reference
            cache_dir: Directory for caching signatures and results
            detection_mode: Initial detection mode
        """
        # Initialize HDC encoder
        self.hdc_encoder = hdc_encoder or HypervectorEncoder(
            HypervectorConfig(
                dimension=10000,
                sparsity=0.1,
                multi_scale=True
            )
        )
        
        # Initialize behavioral sites analyzer
        self.behavioral_sites = behavioral_sites or BehavioralSites()
        
        # Black-box verifier for API models
        self.blackbox_verifier = blackbox_verifier
        
        # Benchmark datasets
        self.benchmark_datasets = benchmark_datasets or []
        self._load_default_benchmarks()
        
        # Cache directory
        self.cache_dir = cache_dir or Path("contamination_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Detection mode
        self.detection_mode = detection_mode
        
        # Signature cache
        self.signature_cache: Dict[str, Any] = {}
        
        # Statistical test thresholds
        self.thresholds = {
            'memorization_pvalue': 0.01,
            'perplexity_zscore': 3.0,
            'ngram_overlap': 0.6,
            'temperature_variance': 0.2,
            'semantic_similarity': 0.85,
            'benchmark_overlap': 0.7
        }
    
    def _load_default_benchmarks(self):
        """Load default benchmark datasets for cross-reference."""
        # Common benchmarks (would load from files in production)
        common_benchmarks = [
            BenchmarkDataset(
                name="MMLU",
                prompts=["What is the capital of France?", "Solve: 2+2="],
                metadata={"type": "knowledge", "size": 15000}
            ),
            BenchmarkDataset(
                name="HumanEval",
                prompts=["def fibonacci(n):", "def quicksort(arr):"],
                metadata={"type": "coding", "size": 164}
            ),
            BenchmarkDataset(
                name="GSM8K",
                prompts=["If John has 5 apples", "Calculate the total"],
                metadata={"type": "math", "size": 8500}
            )
        ]
        
        for benchmark in common_benchmarks:
            # Generate signatures for benchmark prompts
            signatures = {}
            for prompt in benchmark.prompts:
                sig = self.hdc_encoder.encode_feature(prompt, "prompt")
                signatures[prompt] = sig
            benchmark.signatures = signatures
            self.benchmark_datasets.append(benchmark)
    
    def detect_contamination(
        self,
        model_responses: List[str],
        prompts: List[str],
        model_id: str,
        reference_responses: Optional[List[str]] = None,
        temperature_variations: Optional[List[float]] = None,
        mode: Optional[DetectionMode] = None
    ) -> ContaminationReport:
        """
        Perform comprehensive contamination detection.
        
        Args:
            model_responses: Responses from the model being tested
            prompts: Input prompts used
            model_id: Identifier for the model
            reference_responses: Optional reference responses for comparison
            temperature_variations: Temperature settings to test consistency
            mode: Detection mode (overrides default)
            
        Returns:
            Comprehensive contamination report
        """
        mode = mode or self.detection_mode
        
        # Initialize evidence collectors
        memorization_evidence = self._detect_memorization(
            model_responses, prompts, reference_responses
        )
        
        ngram_overlap = self._analyze_ngram_overlap(
            model_responses, reference_responses or prompts
        )
        
        perplexity_stats = self._analyze_perplexity(
            model_responses, prompts
        )
        
        temperature_consistency = {}
        if temperature_variations and mode in [DetectionMode.COMPREHENSIVE, DetectionMode.ADAPTIVE]:
            temperature_consistency = self._check_temperature_consistency(
                prompts, temperature_variations, model_id
            )
        
        # Multi-modal detection
        token_score = self._analyze_token_level(model_responses, prompts)
        semantic_score = self._analyze_semantic_similarity(model_responses, prompts)
        behavioral_score = self._analyze_behavioral_patterns(model_responses, prompts)
        benchmark_score = self._check_benchmark_overlap(model_responses, prompts)
        
        # Compute overall contamination score
        contamination_score, contamination_type = self._compute_contamination_score(
            memorization_evidence,
            ngram_overlap,
            perplexity_stats,
            temperature_consistency,
            token_score,
            semantic_score,
            behavioral_score,
            benchmark_score
        )
        
        # Compute confidence
        confidence = self._compute_confidence(
            len(model_responses),
            memorization_evidence.statistical_significance,
            temperature_consistency
        )
        
        # Collect evidence
        evidence = self._collect_evidence(
            memorization_evidence,
            ngram_overlap,
            perplexity_stats,
            temperature_consistency
        )
        
        # Generate remediation suggestions
        remediation_suggestions = self._generate_remediation_suggestions(
            contamination_type,
            contamination_score,
            evidence
        )
        
        # Create report
        report = ContaminationReport(
            contamination_type=contamination_type,
            contamination_score=contamination_score,
            confidence=confidence,
            memorization_evidence=memorization_evidence,
            ngram_overlap=ngram_overlap,
            perplexity_stats=perplexity_stats,
            temperature_consistency=temperature_consistency,
            token_level_score=token_score,
            semantic_similarity_score=semantic_score,
            behavioral_pattern_score=behavioral_score,
            benchmark_overlap_score=benchmark_score,
            suspicious_patterns=self._identify_suspicious_patterns(model_responses),
            distance_statistics=self._compute_distance_statistics(model_responses),
            hdc_signatures=self._generate_hdc_signatures(model_responses),
            evidence=evidence,
            remediation_suggestions=remediation_suggestions
        )
        
        # Cache results
        self._cache_results(model_id, report)
        
        return report
    
    def _detect_memorization(
        self,
        responses: List[str],
        prompts: List[str],
        references: Optional[List[str]] = None
    ) -> MemorizationEvidence:
        """
        Detect memorization using statistical tests.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            references: Reference responses
            
        Returns:
            Memorization evidence
        """
        verbatim_matches = []
        near_duplicates = []
        perplexity_anomalies = []
        consistency_violations = []
        
        # Check for verbatim matches
        if references:
            for resp, ref in zip(responses, references):
                if resp.strip() == ref.strip():
                    verbatim_matches.append(resp)
                else:
                    # Check similarity
                    similarity = self._compute_text_similarity(resp, ref)
                    if similarity > 0.95:
                        near_duplicates.append((resp, similarity))
        
        # Check for repeated patterns (memorization indicator)
        response_counter = Counter(responses)
        for resp, count in response_counter.items():
            if count > 1:
                consistency_violations.append({
                    'response': resp,
                    'count': count,
                    'type': 'exact_repetition'
                })
        
        # Perplexity analysis
        for i, resp in enumerate(responses):
            perplexity = self._compute_perplexity(resp, prompts[i] if i < len(prompts) else "")
            if perplexity < 5.0:  # Suspiciously low perplexity
                perplexity_anomalies.append(perplexity)
        
        # Statistical significance test
        significance = self._memorization_statistical_test(
            len(verbatim_matches),
            len(near_duplicates),
            len(responses)
        )
        
        return MemorizationEvidence(
            verbatim_matches=verbatim_matches,
            near_duplicates=near_duplicates,
            perplexity_anomalies=perplexity_anomalies,
            consistency_violations=consistency_violations,
            statistical_significance=significance
        )
    
    def _analyze_ngram_overlap(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> NGramOverlap:
        """
        Analyze n-gram overlap between text sets.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            N-gram overlap analysis
        """
        def get_ngrams(text: str, n: int) -> Set[str]:
            """Extract n-grams from text."""
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        # Combine texts for analysis
        text1 = ' '.join(texts1)
        text2 = ' '.join(texts2)
        
        # Extract n-grams
        unigrams1 = get_ngrams(text1, 1)
        unigrams2 = get_ngrams(text2, 1)
        bigrams1 = get_ngrams(text1, 2)
        bigrams2 = get_ngrams(text2, 2)
        trigrams1 = get_ngrams(text1, 3)
        trigrams2 = get_ngrams(text2, 3)
        
        # Compute overlaps
        unigram_overlap = len(unigrams1 & unigrams2) / max(len(unigrams1), len(unigrams2), 1)
        bigram_overlap = len(bigrams1 & bigrams2) / max(len(bigrams1), len(bigrams2), 1)
        trigram_overlap = len(trigrams1 & trigrams2) / max(len(trigrams1), len(trigrams2), 1)
        
        # Find common n-grams
        all_ngrams1 = unigrams1 | bigrams1 | trigrams1
        all_ngrams2 = unigrams2 | bigrams2 | trigrams2
        common_ngrams_set = all_ngrams1 & all_ngrams2
        
        # Count occurrences
        common_ngrams = {}
        for ngram in common_ngrams_set:
            count1 = text1.lower().count(ngram)
            count2 = text2.lower().count(ngram)
            common_ngrams[ngram] = min(count1, count2)
        
        # Jaccard similarity
        jaccard = len(all_ngrams1 & all_ngrams2) / max(len(all_ngrams1 | all_ngrams2), 1)
        
        return NGramOverlap(
            unigram_overlap=unigram_overlap,
            bigram_overlap=bigram_overlap,
            trigram_overlap=trigram_overlap,
            unique_ngrams=(all_ngrams1 | all_ngrams2) - common_ngrams_set,
            common_ngrams=common_ngrams,
            jaccard_similarity=jaccard
        )
    
    def _analyze_perplexity(
        self,
        responses: List[str],
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Analyze perplexity statistics for anomaly detection.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            
        Returns:
            Perplexity statistics
        """
        perplexities = []
        
        for resp, prompt in zip(responses, prompts):
            perp = self._compute_perplexity(resp, prompt)
            perplexities.append(perp)
        
        if not perplexities:
            return {}
        
        # Compute statistics
        perp_array = np.array(perplexities)
        stats = {
            'mean': np.mean(perp_array),
            'std': np.std(perp_array),
            'min': np.min(perp_array),
            'max': np.max(perp_array),
            'median': np.median(perp_array),
            'q1': np.percentile(perp_array, 25),
            'q3': np.percentile(perp_array, 75)
        }
        
        # Detect anomalies
        z_scores = (perp_array - stats['mean']) / (stats['std'] + 1e-8)
        stats['anomaly_count'] = np.sum(np.abs(z_scores) > self.thresholds['perplexity_zscore'])
        stats['anomaly_ratio'] = stats['anomaly_count'] / len(perplexities)
        
        return stats
    
    def _check_temperature_consistency(
        self,
        prompts: List[str],
        temperatures: List[float],
        model_id: str
    ) -> Dict[float, float]:
        """
        Check response consistency across temperature variations.
        
        Args:
            prompts: Input prompts
            temperatures: Temperature values to test
            model_id: Model identifier
            
        Returns:
            Temperature -> consistency score mapping
        """
        consistency_scores = {}
        
        if not self.blackbox_verifier:
            logger.warning("No blackbox verifier available for temperature testing")
            return consistency_scores
        
        # Sample prompts for efficiency
        sample_size = min(10, len(prompts))
        sample_prompts = np.random.choice(prompts, sample_size, replace=False)
        
        # Collect responses at different temperatures
        responses_by_temp = {}
        for temp in temperatures:
            responses = []
            for prompt in sample_prompts:
                # Use blackbox verifier to get response
                # This is a simplified version - actual implementation would use the API
                resp = f"Response at temp {temp} for: {prompt}"
                responses.append(resp)
            responses_by_temp[temp] = responses
        
        # Compute consistency scores
        base_temp = temperatures[0]
        base_responses = responses_by_temp[base_temp]
        
        for temp in temperatures:
            if temp == base_temp:
                consistency_scores[temp] = 1.0
                continue
            
            # Compare responses
            similarities = []
            for base_resp, temp_resp in zip(base_responses, responses_by_temp[temp]):
                sim = self._compute_text_similarity(base_resp, temp_resp)
                similarities.append(sim)
            
            # Consistency is inverse of variance in similarities
            consistency = 1.0 - np.std(similarities)
            consistency_scores[temp] = max(0, consistency)
        
        return consistency_scores
    
    def _analyze_token_level(
        self,
        responses: List[str],
        prompts: List[str]
    ) -> float:
        """
        Analyze contamination at token level.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            
        Returns:
            Token-level contamination score
        """
        # Simple token-level analysis
        # In production, would use actual tokenizer
        
        scores = []
        for resp, prompt in zip(responses, prompts):
            # Token overlap
            resp_tokens = set(resp.lower().split())
            prompt_tokens = set(prompt.lower().split())
            
            # Unexpected token repetition
            token_counts = Counter(resp.lower().split())
            max_repetition = max(token_counts.values()) if token_counts else 0
            
            # Score based on unusual patterns
            overlap_ratio = len(resp_tokens & prompt_tokens) / max(len(resp_tokens), 1)
            repetition_score = min(1.0, max_repetition / 10)
            
            scores.append((overlap_ratio + repetition_score) / 2)
        
        return np.mean(scores) if scores else 0.0
    
    def _analyze_semantic_similarity(
        self,
        responses: List[str],
        prompts: List[str]
    ) -> float:
        """
        Analyze semantic similarity in HDC space.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            
        Returns:
            Semantic similarity score
        """
        similarities = []
        
        for resp, prompt in zip(responses, prompts):
            # Encode in HDC space
            resp_vec = self.hdc_encoder.encode_feature(resp, "prompt")
            prompt_vec = self.hdc_encoder.encode_feature(prompt, "prompt")
            
            # Compute similarity
            similarity = 1 - cosine(resp_vec, prompt_vec)
            similarities.append(similarity)
        
        # High similarity might indicate contamination
        mean_sim = np.mean(similarities) if similarities else 0.0
        
        # Check for suspiciously high similarity
        if mean_sim > self.thresholds['semantic_similarity']:
            return mean_sim
        
        return mean_sim * 0.5  # Scale down if not suspicious
    
    def _analyze_behavioral_patterns(
        self,
        responses: List[str],
        prompts: List[str]
    ) -> float:
        """
        Analyze behavioral patterns for contamination.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            
        Returns:
            Behavioral pattern score
        """
        if not self.behavioral_sites:
            return 0.0
        
        # Extract behavioral features
        features = []
        for resp, prompt in zip(responses, prompts):
            # Create probe features
            probe = ProbeFeatures(
                prompt=prompt,
                response=resp,
                metadata={'analysis': 'contamination'}
            )
            
            # Extract behavioral signature
            signature = self.behavioral_sites.extract_signature(probe)
            features.append(signature)
        
        if not features:
            return 0.0
        
        # Analyze pattern consistency
        features_array = np.array(features)
        
        # Check for unusual clustering
        if len(features) > 2:
            # Compute pairwise distances
            distances = pdist(features_array, metric='cosine')
            
            # Low variance in distances indicates suspicious consistency
            distance_variance = np.var(distances)
            
            # Score based on clustering tightness
            if distance_variance < 0.01:
                return 0.9  # Very tight clustering
            elif distance_variance < 0.05:
                return 0.7
            else:
                return 0.3
        
        return 0.0
    
    def _check_benchmark_overlap(
        self,
        responses: List[str],
        prompts: List[str]
    ) -> float:
        """
        Check overlap with known benchmark datasets.
        
        Args:
            responses: Model responses
            prompts: Input prompts
            
        Returns:
            Benchmark overlap score
        """
        if not self.benchmark_datasets:
            return 0.0
        
        overlap_scores = []
        
        for benchmark in self.benchmark_datasets:
            # Check prompt similarity
            for prompt in prompts:
                prompt_vec = self.hdc_encoder.encode_feature(prompt, "prompt")
                
                for bench_prompt, bench_sig in benchmark.signatures.items():
                    similarity = 1 - cosine(prompt_vec, bench_sig)
                    
                    if similarity > self.thresholds['benchmark_overlap']:
                        overlap_scores.append(similarity)
                        logger.info(f"Potential overlap with {benchmark.name}: {similarity:.3f}")
        
        if overlap_scores:
            return np.max(overlap_scores)
        
        return 0.0
    
    def _compute_contamination_score(
        self,
        memorization: MemorizationEvidence,
        ngram: NGramOverlap,
        perplexity: Dict[str, float],
        temperature: Dict[float, float],
        token_score: float,
        semantic_score: float,
        behavioral_score: float,
        benchmark_score: float
    ) -> Tuple[float, ContaminationType]:
        """
        Compute overall contamination score and type.
        
        Returns:
            (contamination_score, contamination_type)
        """
        # Component weights
        weights = {
            'memorization': 0.25,
            'ngram': 0.15,
            'perplexity': 0.1,
            'temperature': 0.1,
            'token': 0.1,
            'semantic': 0.1,
            'behavioral': 0.1,
            'benchmark': 0.1
        }
        
        # Component scores
        scores = {
            'memorization': memorization.memorization_score,
            'ngram': ngram.overlap_score,
            'perplexity': perplexity.get('anomaly_ratio', 0),
            'temperature': 1.0 - np.mean(list(temperature.values())) if temperature else 0,
            'token': token_score,
            'semantic': semantic_score,
            'behavioral': behavioral_score,
            'benchmark': benchmark_score
        }
        
        # Weighted average
        total_score = sum(weights[k] * scores[k] for k in weights)
        
        # Determine contamination type
        contamination_type = ContaminationType.NONE
        
        if total_score > 0.7:
            if memorization.memorization_score > 0.7:
                contamination_type = ContaminationType.MEMORIZATION
            elif benchmark_score > 0.7:
                contamination_type = ContaminationType.BENCHMARK_OVERLAP
            elif ngram.overlap_score > 0.7:
                contamination_type = ContaminationType.DATA_LEAKAGE
            elif behavioral_score > 0.7:
                contamination_type = ContaminationType.FINE_TUNING
            else:
                contamination_type = ContaminationType.HYBRID
        elif total_score > 0.4:
            if benchmark_score > 0.5:
                contamination_type = ContaminationType.BENCHMARK_OVERLAP
            elif memorization.memorization_score > 0.5:
                contamination_type = ContaminationType.MEMORIZATION
        
        return total_score, contamination_type
    
    def _compute_confidence(
        self,
        sample_size: int,
        statistical_significance: float,
        temperature_consistency: Dict[float, float]
    ) -> float:
        """Compute confidence in contamination detection."""
        # Base confidence from sample size
        size_confidence = min(1.0, sample_size / 100)
        
        # Statistical confidence
        stat_confidence = 1.0 - statistical_significance
        
        # Temperature consistency confidence
        temp_confidence = 1.0
        if temperature_consistency:
            temp_variance = np.var(list(temperature_consistency.values()))
            temp_confidence = 1.0 - min(1.0, temp_variance * 5)
        
        # Combined confidence
        confidence = (
            0.4 * size_confidence +
            0.3 * stat_confidence +
            0.3 * temp_confidence
        )
        
        return min(1.0, confidence)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Simple character-level similarity
        # In production, would use more sophisticated methods
        
        if not text1 or not text2:
            return 0.0
        
        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Character overlap
        chars1 = set(text1)
        chars2 = set(text2)
        char_overlap = len(chars1 & chars2) / max(len(chars1 | chars2), 1)
        
        # Length similarity
        len_sim = min(len(text1), len(text2)) / max(len(text1), len(text2), 1)
        
        return (char_overlap + len_sim) / 2
    
    def _compute_perplexity(self, text: str, context: str) -> float:
        """
        Compute perplexity of text given context.
        
        Simplified version - in production would use actual language model.
        """
        if not text:
            return float('inf')
        
        # Simple entropy-based approximation
        words = text.lower().split()
        word_probs = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for word, count in word_probs.items():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Perplexity = 2^entropy
        perplexity = 2 ** entropy
        
        # Adjust based on context match
        if context:
            context_words = set(context.lower().split())
            text_words = set(words)
            overlap = len(context_words & text_words) / max(len(text_words), 1)
            perplexity *= (1 - overlap * 0.5)  # Lower perplexity if high overlap
        
        return perplexity
    
    def _memorization_statistical_test(
        self,
        n_verbatim: int,
        n_near: int,
        n_total: int
    ) -> float:
        """
        Perform statistical test for memorization significance.
        
        Returns:
            p-value of memorization hypothesis
        """
        if n_total == 0:
            return 1.0
        
        # Binomial test
        # Null hypothesis: memorization rate <= expected random rate
        expected_rate = 0.01  # 1% expected random match
        observed_rate = (n_verbatim + n_near * 0.5) / n_total
        
        # Simple binomial test
        from scipy.stats import binom
        p_value = binom.cdf(n_verbatim + n_near, n_total, expected_rate)
        
        return p_value
    
    def _identify_suspicious_patterns(self, responses: List[str]) -> List[str]:
        """Identify suspicious patterns in responses."""
        patterns = []
        
        # Check for repeated phrases
        phrase_counter = Counter()
        for resp in responses:
            # Extract 5-word phrases
            words = resp.split()
            for i in range(len(words) - 4):
                phrase = ' '.join(words[i:i+5])
                phrase_counter[phrase] += 1
        
        # Suspicious if phrase appears multiple times
        for phrase, count in phrase_counter.items():
            if count > 2:
                patterns.append(f"Repeated phrase ({count}x): {phrase}")
        
        # Check for template patterns
        if len(responses) > 5:
            # Look for common prefixes/suffixes
            common_prefix = self._find_common_prefix(responses)
            if len(common_prefix) > 20:
                patterns.append(f"Common prefix: {common_prefix[:50]}...")
            
            common_suffix = self._find_common_suffix(responses)
            if len(common_suffix) > 20:
                patterns.append(f"Common suffix: ...{common_suffix[-50:]}")
        
        return patterns
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """Find common prefix among strings."""
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    def _find_common_suffix(self, strings: List[str]) -> str:
        """Find common suffix among strings."""
        if not strings:
            return ""
        
        suffix = strings[0]
        for s in strings[1:]:
            while not s.endswith(suffix):
                suffix = suffix[1:]
                if not suffix:
                    return ""
        return suffix
    
    def _compute_distance_statistics(self, responses: List[str]) -> Dict[str, float]:
        """Compute distance statistics for responses."""
        if len(responses) < 2:
            return {}
        
        # Encode responses
        vectors = []
        for resp in responses:
            vec = self.hdc_encoder.encode_feature(resp, "prompt")
            vectors.append(vec)
        
        vectors_array = np.array(vectors)
        
        # Compute pairwise distances
        distances = pdist(vectors_array, metric='cosine')
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'distance_variance': np.var(distances)
        }
    
    def _generate_hdc_signatures(self, responses: List[str]) -> Dict[str, np.ndarray]:
        """Generate HDC signatures for responses."""
        signatures = {}
        
        for i, resp in enumerate(responses[:10]):  # Limit for efficiency
            sig = self.hdc_encoder.encode_feature(resp, "prompt")
            signatures[f"response_{i}"] = sig
        
        return signatures
    
    def _collect_evidence(
        self,
        memorization: MemorizationEvidence,
        ngram: NGramOverlap,
        perplexity: Dict[str, float],
        temperature: Dict[float, float]
    ) -> Dict[str, Any]:
        """Collect evidence for contamination."""
        evidence = {
            'memorization': {
                'verbatim_count': len(memorization.verbatim_matches),
                'near_duplicate_count': len(memorization.near_duplicates),
                'statistical_significance': memorization.statistical_significance
            },
            'ngram_overlap': {
                'overlap_score': ngram.overlap_score,
                'jaccard_similarity': ngram.jaccard_similarity,
                'common_ngrams_count': len(ngram.common_ngrams)
            },
            'perplexity': perplexity,
            'temperature_consistency': temperature
        }
        
        return evidence
    
    def _generate_remediation_suggestions(
        self,
        contamination_type: ContaminationType,
        score: float,
        evidence: Dict[str, Any]
    ) -> List[str]:
        """Generate remediation suggestions based on contamination type."""
        suggestions = []
        
        if contamination_type == ContaminationType.MEMORIZATION:
            suggestions.append("Apply data deduplication to training set")
            suggestions.append("Implement privacy-preserving training techniques")
            suggestions.append("Use differential privacy during fine-tuning")
            suggestions.append("Increase regularization to reduce overfitting")
        
        elif contamination_type == ContaminationType.BENCHMARK_OVERLAP:
            suggestions.append("Remove benchmark data from training set")
            suggestions.append("Use held-out evaluation sets")
            suggestions.append("Implement data provenance tracking")
            suggestions.append("Apply benchmark-specific filters during training")
        
        elif contamination_type == ContaminationType.DATA_LEAKAGE:
            suggestions.append("Audit training data sources")
            suggestions.append("Implement strict data segregation")
            suggestions.append("Use cryptographic hashing for data tracking")
            suggestions.append("Apply leakage detection during preprocessing")
        
        elif contamination_type == ContaminationType.FINE_TUNING:
            suggestions.append("Review fine-tuning datasets")
            suggestions.append("Implement catastrophic forgetting prevention")
            suggestions.append("Use elastic weight consolidation")
            suggestions.append("Apply domain adaptation techniques")
        
        elif contamination_type == ContaminationType.HYBRID:
            suggestions.append("Perform comprehensive data audit")
            suggestions.append("Implement multi-level contamination detection")
            suggestions.append("Use ensemble of decontamination techniques")
            suggestions.append("Consider retraining with clean data")
        
        # Add severity-based suggestions
        if score > 0.8:
            suggestions.insert(0, "⚠️ Critical: Consider model quarantine and immediate audit")
        elif score > 0.6:
            suggestions.insert(0, "⚠️ High: Implement immediate remediation measures")
        
        return suggestions
    
    def _cache_results(self, model_id: str, report: ContaminationReport):
        """Cache contamination detection results."""
        cache_file = self.cache_dir / f"{model_id}_contamination.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(report, f)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
    
    def load_cached_results(self, model_id: str) -> Optional[ContaminationReport]:
        """Load cached contamination results if available."""
        cache_file = self.cache_dir / f"{model_id}_contamination.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached results: {e}")
        
        return None
    
    def compare_models_contamination(
        self,
        model_ids: List[str],
        test_prompts: List[str],
        test_responses: Dict[str, List[str]]
    ) -> Dict[str, ContaminationReport]:
        """
        Compare contamination across multiple models.
        
        Args:
            model_ids: List of model identifiers
            test_prompts: Common test prompts
            test_responses: Model ID -> responses mapping
            
        Returns:
            Model ID -> contamination report mapping
        """
        reports = {}
        
        for model_id in model_ids:
            if model_id not in test_responses:
                logger.warning(f"No responses for model {model_id}")
                continue
            
            # Check cache first
            cached = self.load_cached_results(model_id)
            if cached:
                reports[model_id] = cached
                continue
            
            # Perform detection
            report = self.detect_contamination(
                test_responses[model_id],
                test_prompts,
                model_id
            )
            reports[model_id] = report
        
        return reports