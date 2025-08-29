"""
Unified Contamination Detection for REV model verification.

This module implements contamination detection using Hamming distance matrices,
HDC behavioral signatures, and genealogy tracing for model relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
from collections import defaultdict, Counter
import logging
import warnings

from ..hypervector.hamming import HammingLUT, pack_binary_vector, hamming_distance_cpu
from ..hypervector.operations.hamming_lut import pack_binary_vector_simd
from ..hdc.encoder import UnifiedHDCEncoder, HypervectorConfig
from ..hdc.behavioral_sites import BehavioralSites, ProbeFeatures
from .blackbox import BlackBoxVerifier, ModelProvider, APIConfig

logger = logging.getLogger(__name__)


class ContaminationType(Enum):
    """Types of model contamination."""
    NONE = "none"
    DATA_LEAKAGE = "data_leakage"
    FINE_TUNING = "fine_tuning"
    DISTILLATION = "distillation"
    ADVERSARIAL = "adversarial"
    HYBRID = "hybrid"


@dataclass
class ContaminationResult:
    """Result of contamination detection."""
    
    contamination_type: ContaminationType
    contamination_score: float  # Overall score [0, 1]
    confidence: float  # Confidence in detection [0, 1]
    
    # Component scores
    hamming_anomaly_score: float
    behavioral_similarity_score: float
    fine_tuning_artifact_score: float
    
    # Detailed analysis
    suspicious_models: List[str]
    genealogy_graph: Optional[nx.DiGraph]
    distance_statistics: Dict[str, float]
    artifact_patterns: List[str]
    
    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def is_contaminated(self, threshold: float = 0.7) -> bool:
        """Check if contamination score exceeds threshold."""
        return self.contamination_score > threshold


@dataclass 
class ModelSignature:
    """Signature for a model used in contamination detection."""
    
    model_id: str
    hypervector: np.ndarray  # HDC signature
    response_patterns: List[str]  # Sample responses
    logit_distribution: Optional[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedContaminationDetector:
    """
    Unified contamination detector combining Hamming distance and HDC analysis.
    
    Detects various forms of model contamination including:
    - Data leakage from training
    - Fine-tuning artifacts
    - Model distillation
    - Adversarial modifications
    """
    
    def __init__(
        self,
        hamming_computer: Optional[HammingLUT] = None,
        hdc_encoder: Optional[UnifiedHDCEncoder] = None,
        behavioral_sites: Optional[BehavioralSites] = None,
        blackbox_verifier: Optional[BlackBoxVerifier] = None,
        cache_signatures: bool = True
    ):
        """
        Initialize contamination detector.
        
        Args:
            hamming_computer: Hamming distance computer
            hdc_encoder: HDC encoder for signatures
            behavioral_sites: Behavioral site analyzer
            blackbox_verifier: Black-box API verifier
            cache_signatures: Whether to cache model signatures
        """
        # Initialize components
        self.hamming_computer = hamming_computer or HammingLUT(enable_simd=True)
        
        self.hdc_encoder = hdc_encoder or UnifiedHDCEncoder(
            HypervectorConfig(
                dimension=10000,
                encoding_mode="unified",
                multi_scale=True
            )
        )
        
        self.behavioral_sites = behavioral_sites or BehavioralSites(
            encoder=self.hdc_encoder
        )
        
        self.blackbox_verifier = blackbox_verifier
        
        # Caching
        self.cache_signatures = cache_signatures
        self.signature_cache: Dict[str, ModelSignature] = {}
        
        # Contamination patterns database
        self.known_patterns = self._load_contamination_patterns()
        
        # Statistics tracking
        self.detection_history: List[ContaminationResult] = []
    
    def detect_contamination(
        self,
        model,
        reference_models: List[Any],
        model_id: str = "target",
        reference_ids: Optional[List[str]] = None,
        challenges: Optional[List[str]] = None,
        use_api: bool = False,
        api_configs: Optional[Dict[str, APIConfig]] = None
    ) -> ContaminationResult:
        """
        Detect contamination in a model compared to reference models.
        
        Args:
            model: Target model to check for contamination
            reference_models: List of reference models for comparison
            model_id: Identifier for target model
            reference_ids: Identifiers for reference models
            challenges: Test challenges for behavioral analysis
            use_api: Whether to use API-based verification
            api_configs: API configurations if using API mode
            
        Returns:
            ContaminationResult with detection details
        """
        if reference_ids is None:
            reference_ids = [f"ref_{i}" for i in range(len(reference_models))]
        
        if challenges is None:
            challenges = self._generate_default_challenges()
        
        logger.info(f"Starting contamination detection for model '{model_id}'")
        
        # Step 1: Generate model signatures
        target_signature = self._generate_model_signature(
            model, model_id, challenges, use_api, 
            api_configs.get(model_id) if api_configs else None
        )
        
        reference_signatures = []
        for ref_model, ref_id in zip(reference_models, reference_ids):
            ref_signature = self._generate_model_signature(
                ref_model, ref_id, challenges, use_api,
                api_configs.get(ref_id) if api_configs else None
            )
            reference_signatures.append(ref_signature)
        
        # Step 2: Compute Hamming distance matrix
        distance_matrix = self._compute_distance_matrix(
            target_signature, reference_signatures
        )
        
        # Step 3: Analyze distance anomalies
        hamming_anomaly_score, distance_stats = self.analyze_distance_distribution(
            distance_matrix
        )
        
        # Step 4: Analyze behavioral similarity via HDC
        behavioral_similarity_score = self._analyze_behavioral_similarity(
            target_signature, reference_signatures
        )
        
        # Step 5: Detect fine-tuning artifacts
        fine_tuning_score, artifact_patterns = self.detect_fine_tuning_artifacts(
            target_signature, reference_signatures
        )
        
        # Step 6: Trace contamination genealogy
        genealogy_graph, suspicious_models = self.trace_contamination_genealogy(
            target_signature, reference_signatures, distance_matrix
        )
        
        # Step 7: Aggregate contamination score
        contamination_score, contamination_type = self.aggregate_contamination_score(
            hamming_anomaly_score,
            behavioral_similarity_score,
            fine_tuning_score,
            distance_stats
        )
        
        # Calculate confidence based on evidence strength
        confidence = self._calculate_confidence(
            distance_stats,
            len(artifact_patterns),
            len(suspicious_models)
        )
        
        result = ContaminationResult(
            contamination_type=contamination_type,
            contamination_score=contamination_score,
            confidence=confidence,
            hamming_anomaly_score=hamming_anomaly_score,
            behavioral_similarity_score=behavioral_similarity_score,
            fine_tuning_artifact_score=fine_tuning_score,
            suspicious_models=suspicious_models,
            genealogy_graph=genealogy_graph,
            distance_statistics=distance_stats,
            artifact_patterns=artifact_patterns,
            evidence={
                'distance_matrix': distance_matrix,
                'target_signature': target_signature,
                'num_references': len(reference_models)
            }
        )
        
        self.detection_history.append(result)
        logger.info(f"Contamination detection complete: {contamination_type.value} "
                   f"(score: {contamination_score:.3f}, confidence: {confidence:.3f})")
        
        return result
    
    def _generate_model_signature(
        self,
        model,
        model_id: str,
        challenges: List[str],
        use_api: bool = False,
        api_config: Optional[APIConfig] = None
    ) -> ModelSignature:
        """
        Generate comprehensive signature for a model.
        
        Args:
            model: Model to analyze
            model_id: Model identifier
            challenges: Test challenges
            use_api: Whether to use API
            api_config: API configuration
            
        Returns:
            ModelSignature with HDC and response data
        """
        # Check cache
        if self.cache_signatures and model_id in self.signature_cache:
            return self.signature_cache[model_id]
        
        responses = []
        features_list = []
        
        if use_api and self.blackbox_verifier and api_config:
            # Use API-based extraction
            for challenge in challenges[:20]:  # Limit for API calls
                response_data = self.blackbox_verifier.get_model_response(
                    prompt=challenge,
                    config=api_config
                )
                responses.append(response_data.get('text', ''))
                
                if 'logits' in response_data:
                    features_list.append(response_data['logits'])
        else:
            # Direct model access
            for challenge in challenges:
                # Get model response (implementation depends on model type)
                response = self._get_model_response(model, challenge)
                responses.append(response['text'])
                
                if 'features' in response:
                    features_list.append(response['features'])
        
        # Generate HDC hypervector from responses
        if features_list:
            combined_features = np.concatenate(features_list)
            hypervector = self.hdc_encoder.encode_adaptive(
                combined_features,
                context="consensus",
                behavioral_site=model_id
            ).numpy()
        else:
            # Fallback: encode text responses
            text_features = self._extract_text_features(responses)
            hypervector = self.hdc_encoder.encode_adaptive(
                text_features,
                context="consensus"
            ).numpy()
        
        # Extract response patterns
        response_patterns = self._extract_response_patterns(responses)
        
        # Compute logit distribution if available
        logit_distribution = None
        if features_list:
            all_logits = np.concatenate([f for f in features_list if len(f) > 0])
            logit_distribution = np.histogram(all_logits, bins=50)[0]
            logit_distribution = logit_distribution / logit_distribution.sum()
        
        signature = ModelSignature(
            model_id=model_id,
            hypervector=hypervector,
            response_patterns=response_patterns,
            logit_distribution=logit_distribution,
            metadata={'num_challenges': len(challenges)}
        )
        
        # Cache signature
        if self.cache_signatures:
            self.signature_cache[model_id] = signature
        
        return signature
    
    def _compute_distance_matrix(
        self,
        target_signature: ModelSignature,
        reference_signatures: List[ModelSignature]
    ) -> np.ndarray:
        """
        Compute Hamming distance matrix between signatures.
        
        Args:
            target_signature: Target model signature
            reference_signatures: Reference model signatures
            
        Returns:
            Distance matrix
        """
        n_refs = len(reference_signatures)
        distance_matrix = np.zeros((n_refs + 1, n_refs + 1))
        
        # Pack hypervectors for efficient Hamming distance
        all_signatures = [target_signature] + reference_signatures
        packed_vectors = []
        
        for sig in all_signatures:
            # Convert to binary and pack
            binary_vec = (sig.hypervector > np.median(sig.hypervector)).astype(np.bool_)
            packed = pack_binary_vector_simd(binary_vec)
            packed_vectors.append(packed)
        
        # Compute pairwise distances
        for i in range(len(packed_vectors)):
            for j in range(i + 1, len(packed_vectors)):
                distance = self.hamming_computer.distance(
                    packed_vectors[i], packed_vectors[j]
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def analyze_distance_distribution(
        self,
        distance_matrix: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze statistical distribution of Hamming distances.
        
        Args:
            distance_matrix: Pairwise distance matrix
            
        Returns:
            Tuple of (anomaly_score, statistics_dict)
        """
        # Extract upper triangle (excluding diagonal)
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        if len(distances) == 0:
            return 0.0, {}
        
        # Compute statistics
        stats_dict = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'q25': np.percentile(distances, 25),
            'q75': np.percentile(distances, 75)
        }
        
        # Detect anomalies using IQR method
        iqr = stats_dict['q75'] - stats_dict['q25']
        lower_bound = stats_dict['q25'] - 1.5 * iqr
        upper_bound = stats_dict['q75'] + 1.5 * iqr
        
        # Count anomalies
        anomalies = distances[(distances < lower_bound) | (distances > upper_bound)]
        anomaly_ratio = len(anomalies) / len(distances) if len(distances) > 0 else 0
        
        # Check for suspicious clustering (too similar)
        very_similar = distances[distances < stats_dict['mean'] - 2 * stats_dict['std']]
        clustering_score = len(very_similar) / len(distances) if len(distances) > 0 else 0
        
        # Combined anomaly score
        anomaly_score = min(1.0, anomaly_ratio + clustering_score * 2)
        
        stats_dict['anomaly_ratio'] = anomaly_ratio
        stats_dict['clustering_score'] = clustering_score
        
        return anomaly_score, stats_dict
    
    def _analyze_behavioral_similarity(
        self,
        target_signature: ModelSignature,
        reference_signatures: List[ModelSignature]
    ) -> float:
        """
        Analyze behavioral similarity using HDC signatures.
        
        Args:
            target_signature: Target model signature
            reference_signatures: Reference signatures
            
        Returns:
            Behavioral similarity score [0, 1]
        """
        if not reference_signatures:
            return 0.0
        
        similarities = []
        
        for ref_sig in reference_signatures:
            # Compute cosine similarity between hypervectors
            cos_sim = np.dot(target_signature.hypervector, ref_sig.hypervector) / (
                np.linalg.norm(target_signature.hypervector) * 
                np.linalg.norm(ref_sig.hypervector) + 1e-8
            )
            similarities.append(cos_sim)
        
        # High similarity to any reference is suspicious
        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Score based on both max and average
        behavioral_score = 0.7 * max_similarity + 0.3 * avg_similarity
        
        # Adjust for very high similarity (potential contamination)
        if max_similarity > 0.95:
            behavioral_score = min(1.0, behavioral_score * 1.2)
        
        return behavioral_score
    
    def detect_fine_tuning_artifacts(
        self,
        target_signature: ModelSignature,
        reference_signatures: List[ModelSignature]
    ) -> Tuple[float, List[str]]:
        """
        Detect fine-tuning artifacts in response patterns.
        
        Args:
            target_signature: Target model signature
            reference_signatures: Reference signatures
            
        Returns:
            Tuple of (artifact_score, detected_patterns)
        """
        detected_patterns = []
        artifact_indicators = []
        
        # Check for response pattern anomalies
        target_patterns = set(target_signature.response_patterns)
        
        # 1. Check for overly specific patterns (memorization)
        for pattern in target_patterns:
            if self._is_memorized_pattern(pattern):
                detected_patterns.append(f"memorized: {pattern[:50]}...")
                artifact_indicators.append(1.0)
        
        # 2. Check for format inconsistencies
        ref_patterns = set()
        for ref_sig in reference_signatures:
            ref_patterns.update(ref_sig.response_patterns)
        
        unique_to_target = target_patterns - ref_patterns
        if len(unique_to_target) > len(target_patterns) * 0.3:
            detected_patterns.append("format_deviation")
            artifact_indicators.append(0.7)
        
        # 3. Check logit distribution if available
        if target_signature.logit_distribution is not None:
            entropy = stats.entropy(target_signature.logit_distribution)
            
            # Low entropy suggests overfitting
            if entropy < 2.0:
                detected_patterns.append("low_entropy_distribution")
                artifact_indicators.append(0.8)
            
            # Compare to reference distributions
            for ref_sig in reference_signatures:
                if ref_sig.logit_distribution is not None:
                    kl_div = stats.entropy(
                        target_signature.logit_distribution,
                        ref_sig.logit_distribution
                    )
                    if kl_div > 1.0:
                        detected_patterns.append(f"distribution_shift_from_{ref_sig.model_id}")
                        artifact_indicators.append(min(1.0, kl_div / 2.0))
        
        # 4. Check for repetitive patterns
        pattern_counts = Counter(target_signature.response_patterns)
        repetitions = sum(1 for count in pattern_counts.values() if count > 2)
        if repetitions > len(pattern_counts) * 0.2:
            detected_patterns.append("repetitive_responses")
            artifact_indicators.append(0.6)
        
        # Calculate overall artifact score
        artifact_score = np.mean(artifact_indicators) if artifact_indicators else 0.0
        
        return artifact_score, detected_patterns[:10]  # Limit patterns returned
    
    def trace_contamination_genealogy(
        self,
        target_signature: ModelSignature,
        reference_signatures: List[ModelSignature],
        distance_matrix: np.ndarray
    ) -> Tuple[Optional[nx.DiGraph], List[str]]:
        """
        Trace contamination genealogy through model relationships.
        
        Args:
            target_signature: Target model signature
            reference_signatures: Reference signatures
            distance_matrix: Distance matrix
            
        Returns:
            Tuple of (genealogy_graph, suspicious_models)
        """
        if len(reference_signatures) < 2:
            return None, []
        
        # Create directed graph for genealogy
        G = nx.DiGraph()
        
        # Add nodes
        all_models = [target_signature.model_id] + [
            sig.model_id for sig in reference_signatures
        ]
        G.add_nodes_from(all_models)
        
        # Perform hierarchical clustering on distance matrix
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Get clusters
        clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')
        
        # Find models in same cluster as target
        target_cluster = clusters[0]
        suspicious_models = []
        
        for i, (model_id, cluster) in enumerate(zip(all_models[1:], clusters[1:]), 1):
            if cluster == target_cluster:
                suspicious_models.append(model_id)
                
                # Add edge indicating potential relationship
                similarity = 1.0 - (distance_matrix[0, i] / distance_matrix.max())
                G.add_edge(model_id, target_signature.model_id, weight=similarity)
        
        # Infer parent-child relationships based on distances
        for i in range(1, len(all_models)):
            for j in range(i + 1, len(all_models)):
                dist = distance_matrix[i, j]
                
                # Very close models might have parent-child relationship
                if dist < distance_matrix.mean() - distance_matrix.std():
                    # Infer direction based on other relationships
                    similarity = 1.0 - (dist / distance_matrix.max())
                    
                    # Add bidirectional edge, will be refined later
                    G.add_edge(all_models[i], all_models[j], weight=similarity)
        
        # Refine to DAG if possible
        if nx.is_directed_acyclic_graph(G):
            return G, suspicious_models
        
        # Remove cycles to create DAG
        try:
            G = self._remove_cycles(G)
        except:
            pass
        
        return G, suspicious_models
    
    def aggregate_contamination_score(
        self,
        hamming_anomaly_score: float,
        behavioral_similarity_score: float,
        fine_tuning_artifact_score: float,
        distance_stats: Dict[str, float]
    ) -> Tuple[float, ContaminationType]:
        """
        Aggregate scores into final contamination score and type.
        
        Args:
            hamming_anomaly_score: Hamming distance anomaly score
            behavioral_similarity_score: HDC behavioral similarity
            fine_tuning_artifact_score: Fine-tuning artifact score
            distance_stats: Distance statistics
            
        Returns:
            Tuple of (contamination_score, contamination_type)
        """
        # Weight components based on reliability
        weights = {
            'hamming': 0.3,
            'behavioral': 0.4,
            'fine_tuning': 0.3
        }
        
        # Compute weighted score
        contamination_score = (
            weights['hamming'] * hamming_anomaly_score +
            weights['behavioral'] * behavioral_similarity_score +
            weights['fine_tuning'] * fine_tuning_artifact_score
        )
        
        # Determine contamination type based on dominant signal
        scores = {
            'hamming': hamming_anomaly_score,
            'behavioral': behavioral_similarity_score,
            'fine_tuning': fine_tuning_artifact_score
        }
        
        dominant = max(scores, key=scores.get)
        
        # Map to contamination type
        if contamination_score < 0.3:
            contamination_type = ContaminationType.NONE
        elif dominant == 'hamming' and distance_stats.get('clustering_score', 0) > 0.2:
            contamination_type = ContaminationType.DATA_LEAKAGE
        elif dominant == 'fine_tuning':
            contamination_type = ContaminationType.FINE_TUNING
        elif dominant == 'behavioral' and behavioral_similarity_score > 0.9:
            contamination_type = ContaminationType.DISTILLATION
        elif contamination_score > 0.8:
            contamination_type = ContaminationType.ADVERSARIAL
        else:
            contamination_type = ContaminationType.HYBRID
        
        return contamination_score, contamination_type
    
    def _load_contamination_patterns(self) -> Dict[str, List[str]]:
        """Load known contamination patterns database."""
        return {
            'memorization': [
                'exact repetition',
                'verbatim quotes',
                'specific numbers repeated'
            ],
            'format_artifacts': [
                'unusual formatting',
                'consistent typos',
                'specific tokens'
            ],
            'behavioral': [
                'response style changes',
                'capability gaps',
                'knowledge cutoff violations'
            ]
        }
    
    def _generate_default_challenges(self) -> List[str]:
        """Generate default challenge prompts for testing."""
        return [
            "What is machine learning?",
            "Explain neural networks",
            "Describe gradient descent",
            "What is overfitting?",
            "How does attention work?",
            "Explain transformers",
            "What is fine-tuning?",
            "Describe BERT architecture",
            "What is transfer learning?",
            "Explain backpropagation",
            "What are embeddings?",
            "Describe RNN vs CNN",
            "What is regularization?",
            "Explain batch normalization",
            "What is dropout?",
            "Describe loss functions",
            "What is cross-validation?",
            "Explain precision and recall",
            "What is F1 score?",
            "Describe ROC curves"
        ]
    
    def _get_model_response(self, model, challenge: str) -> Dict[str, Any]:
        """Get response from model (implementation depends on model type)."""
        # Placeholder - actual implementation depends on model interface
        return {
            'text': f"Model response to: {challenge}",
            'features': np.random.randn(100)
        }
    
    def _extract_text_features(self, responses: List[str]) -> np.ndarray:
        """Extract features from text responses."""
        # Simple feature extraction - can be enhanced
        features = []
        for response in responses:
            # Length, word count, unique words, etc.
            features.extend([
                len(response),
                len(response.split()),
                len(set(response.split())),
                response.count('.'),
                response.count(',')
            ])
        return np.array(features, dtype=np.float32)
    
    def _extract_response_patterns(self, responses: List[str]) -> List[str]:
        """Extract patterns from responses."""
        patterns = []
        for response in responses[:20]:  # Limit number of patterns
            # Extract first sentence or up to 100 chars
            pattern = response.split('.')[0][:100]
            patterns.append(pattern)
        return patterns
    
    def _is_memorized_pattern(self, pattern: str) -> bool:
        """Check if pattern shows signs of memorization."""
        # Check for exact quotes, URLs, specific numbers, etc.
        indicators = [
            len(pattern) > 200,  # Very long exact response
            pattern.count('"') > 4,  # Many quotes
            'http' in pattern,  # URLs
            any(year in pattern for year in ['2021', '2022', '2023', '2024']),  # Specific years
            pattern.count('.') > 10  # Many sentences verbatim
        ]
        return sum(indicators) >= 2
    
    def _calculate_confidence(
        self,
        distance_stats: Dict[str, float],
        num_artifacts: int,
        num_suspicious: int
    ) -> float:
        """Calculate confidence in contamination detection."""
        # Base confidence on amount of evidence
        evidence_score = min(1.0, (num_artifacts + num_suspicious) / 10)
        
        # Adjust based on statistical significance
        if 'std' in distance_stats and distance_stats['std'] > 0:
            z_score = abs(distance_stats['mean'] - distance_stats['median']) / distance_stats['std']
            stat_confidence = min(1.0, z_score / 3.0)
        else:
            stat_confidence = 0.5
        
        # Combined confidence
        confidence = 0.6 * evidence_score + 0.4 * stat_confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _remove_cycles(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles from graph to create DAG."""
        # Simple cycle removal - can be enhanced
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                # Remove edge with lowest weight in cycle
                min_weight = float('inf')
                min_edge = None
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(u, v):
                        weight = G[u][v].get('weight', 1.0)
                        if weight < min_weight:
                            min_weight = weight
                            min_edge = (u, v)
                if min_edge:
                    G.remove_edge(*min_edge)
        return G
    
    def get_contamination_report(
        self,
        result: ContaminationResult
    ) -> str:
        """
        Generate human-readable contamination report.
        
        Args:
            result: Contamination detection result
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("CONTAMINATION DETECTION REPORT")
        report.append("="*60)
        report.append(f"\nContamination Type: {result.contamination_type.value.upper()}")
        report.append(f"Overall Score: {result.contamination_score:.3f}")
        report.append(f"Confidence: {result.confidence:.3f}")
        report.append(f"Contaminated: {'YES' if result.is_contaminated() else 'NO'}")
        
        report.append("\n" + "-"*40)
        report.append("COMPONENT SCORES:")
        report.append(f"  Hamming Anomaly:     {result.hamming_anomaly_score:.3f}")
        report.append(f"  Behavioral Similarity: {result.behavioral_similarity_score:.3f}")
        report.append(f"  Fine-tuning Artifacts: {result.fine_tuning_artifact_score:.3f}")
        
        if result.suspicious_models:
            report.append("\n" + "-"*40)
            report.append("SUSPICIOUS MODELS:")
            for model in result.suspicious_models:
                report.append(f"  - {model}")
        
        if result.artifact_patterns:
            report.append("\n" + "-"*40)
            report.append("DETECTED ARTIFACTS:")
            for pattern in result.artifact_patterns[:5]:
                report.append(f"  - {pattern}")
        
        report.append("\n" + "-"*40)
        report.append("DISTANCE STATISTICS:")
        for key, value in result.distance_statistics.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)