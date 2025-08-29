"""
Decision Aggregation for REV verification.

This module implements per-challenge equality indicators, distance score aggregation,
sequential test integration, and first-divergence localization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import hashlib
from scipy import stats

from ..core.sequential import SPRTResult, SequentialState, sequential_verify
from ..hypervector.similarity import AdvancedSimilarity, SimilarityMetric
from .decision import Verdict, StepRecord


@dataclass
class ChallengeResult:
    """Result for a single challenge."""
    
    challenge_id: str
    prompt: str
    model_a_response: str
    model_b_response: str
    distance_score: float
    equality_indicator: bool  # True if responses are functionally equivalent
    hypervector_similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergencePoint:
    """Point where models first diverge significantly."""
    
    challenge_index: int
    challenge_id: str
    divergence_score: float
    confidence: float
    context: str  # What type of divergence (semantic, syntactic, etc.)


@dataclass
class AggregatedDecision:
    """Aggregated decision across all challenges."""
    
    verdict: Verdict
    confidence: float
    total_challenges: int
    equal_challenges: int
    divergent_challenges: int
    mean_distance: float
    std_distance: float
    first_divergence: Optional[DivergencePoint]
    sequential_result: Optional[SPRTResult]
    per_challenge_results: List[ChallengeResult]


class AggregationMethod(Enum):
    """Methods for aggregating distance scores."""
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"
    TRIMMED_MEAN = "trimmed_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    MAJORITY_VOTE = "majority_vote"


class DecisionAggregator:
    """
    Aggregates verification decisions across multiple challenges.
    
    Implements:
    - Per-challenge equality indicators
    - Distance score aggregation with multiple methods
    - Sequential test integration for early stopping
    - First-divergence localization for debugging
    """
    
    def __init__(
        self,
        equality_threshold: float = 0.95,
        divergence_threshold: float = 0.7,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        use_sequential_testing: bool = True,
        alpha: float = 0.05,
        beta: float = 0.10
    ):
        """
        Initialize decision aggregator.
        
        Args:
            equality_threshold: Similarity threshold for equality
            divergence_threshold: Threshold for significant divergence
            aggregation_method: Method for aggregating scores
            use_sequential_testing: Whether to use sequential testing
            alpha: Type I error rate for sequential testing
            beta: Type II error rate for sequential testing
        """
        self.equality_threshold = equality_threshold
        self.divergence_threshold = divergence_threshold
        self.aggregation_method = aggregation_method
        self.use_sequential_testing = use_sequential_testing
        self.alpha = alpha
        self.beta = beta
        
        # Initialize similarity computer
        self.similarity_computer = AdvancedSimilarity()
        
        # Storage for results
        self.challenge_results: List[ChallengeResult] = []
        self.first_divergence: Optional[DivergencePoint] = None
        
        # Sequential testing state
        self.sequential_state = SequentialState() if use_sequential_testing else None
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 0.0
        self.n_samples = 0
    
    def process_challenge(
        self,
        challenge_id: str,
        prompt: str,
        response_a: Union[str, np.ndarray],
        response_b: Union[str, np.ndarray],
        hypervector_a: Optional[np.ndarray] = None,
        hypervector_b: Optional[np.ndarray] = None
    ) -> ChallengeResult:
        """
        Process a single challenge and compute equality indicator.
        
        Args:
            challenge_id: Unique identifier for challenge
            prompt: Challenge prompt
            response_a: Response from model A (text or embedding)
            response_b: Response from model B (text or embedding)
            hypervector_a: Optional hypervector for model A
            hypervector_b: Optional hypervector for model B
            
        Returns:
            ChallengeResult with equality indicator and scores
        """
        # Compute distance score
        if isinstance(response_a, str) and isinstance(response_b, str):
            # Text responses - use string similarity
            distance_score = self._compute_text_distance(response_a, response_b)
        else:
            # Embedding responses
            if isinstance(response_a, str):
                response_a = self._text_to_embedding(response_a)
            if isinstance(response_b, str):
                response_b = self._text_to_embedding(response_b)
            
            distance_score = 1.0 - self.similarity_computer._compute_similarity(
                response_a, response_b, SimilarityMetric.COSINE
            )
        
        # Compute hypervector similarity if provided
        hv_similarity = 0.0
        if hypervector_a is not None and hypervector_b is not None:
            hv_similarity = self.similarity_computer._compute_similarity(
                hypervector_a, hypervector_b, SimilarityMetric.COSINE
            )
        
        # Determine equality indicator
        equality_indicator = (1.0 - distance_score) >= self.equality_threshold
        
        # Create result
        result = ChallengeResult(
            challenge_id=challenge_id,
            prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            model_a_response=str(response_a)[:100] + "..." if len(str(response_a)) > 100 else str(response_a),
            model_b_response=str(response_b)[:100] + "..." if len(str(response_b)) > 100 else str(response_b),
            distance_score=distance_score,
            equality_indicator=equality_indicator,
            hypervector_similarity=hv_similarity,
            metadata={
                "response_length_a": len(str(response_a)),
                "response_length_b": len(str(response_b))
            }
        )
        
        # Store result
        self.challenge_results.append(result)
        
        # Update running statistics
        self._update_statistics(distance_score)
        
        # Check for first divergence
        if not equality_indicator and self.first_divergence is None:
            if distance_score > self.divergence_threshold:
                self.first_divergence = DivergencePoint(
                    challenge_index=len(self.challenge_results) - 1,
                    challenge_id=challenge_id,
                    divergence_score=distance_score,
                    confidence=1.0 - distance_score,
                    context=self._analyze_divergence_context(response_a, response_b)
                )
        
        return result
    
    def _compute_text_distance(self, text_a: str, text_b: str) -> float:
        """Compute distance between text responses."""
        # Normalize texts
        text_a = text_a.lower().strip()
        text_b = text_b.lower().strip()
        
        # Exact match
        if text_a == text_b:
            return 0.0
        
        # Levenshtein distance (normalized)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, text_a, text_b).ratio()
        
        return 1.0 - similarity
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        # Simple hash-based embedding (can be replaced with actual embeddings)
        text_hash = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(text_hash[:4], 'big'))
        embedding = np.random.randn(1000)
        np.random.seed(None)
        return embedding / np.linalg.norm(embedding)
    
    def _update_statistics(self, score: float):
        """Update running statistics with Welford's algorithm."""
        self.n_samples += 1
        delta = score - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = score - self.running_mean
        self.running_var += delta * delta2
    
    def _analyze_divergence_context(self, response_a: Any, response_b: Any) -> str:
        """Analyze the context/type of divergence."""
        if isinstance(response_a, str) and isinstance(response_b, str):
            len_diff = abs(len(response_a) - len(response_b))
            
            if len_diff > len(response_a) * 0.5:
                return "length_divergence"
            elif response_a[:50] != response_b[:50]:
                return "early_semantic_divergence"
            elif response_a[-50:] != response_b[-50:]:
                return "late_semantic_divergence"
            else:
                return "middle_content_divergence"
        else:
            return "embedding_divergence"
    
    def aggregate_scores(
        self,
        scores: Optional[List[float]] = None,
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Aggregate distance scores using specified method.
        
        Args:
            scores: List of scores to aggregate (or use stored results)
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated score
        """
        if scores is None:
            scores = [r.distance_score for r in self.challenge_results]
        
        if not scores:
            return 0.0
        
        if self.aggregation_method == AggregationMethod.MEAN:
            return float(np.mean(scores))
            
        elif self.aggregation_method == AggregationMethod.MEDIAN:
            return float(np.median(scores))
            
        elif self.aggregation_method == AggregationMethod.WEIGHTED_MEAN:
            if weights is None:
                # Use inverse variance weighting
                variances = [(s - self.running_mean) ** 2 for s in scores]
                weights = [1.0 / (v + 1e-8) for v in variances]
            
            weight_sum = sum(weights)
            if weight_sum > 0:
                return sum(s * w for s, w in zip(scores, weights)) / weight_sum
            else:
                return float(np.mean(scores))
                
        elif self.aggregation_method == AggregationMethod.TRIMMED_MEAN:
            # Remove top and bottom 10%
            trimmed = stats.trim_mean(scores, 0.1)
            return float(trimmed)
            
        elif self.aggregation_method == AggregationMethod.GEOMETRIC_MEAN:
            # Geometric mean (add small epsilon to avoid zero)
            scores_positive = [s + 1e-8 for s in scores]
            return float(stats.gmean(scores_positive))
            
        elif self.aggregation_method == AggregationMethod.MAJORITY_VOTE:
            # Count how many are below equality threshold
            equal_count = sum(1 for s in scores if s < (1.0 - self.equality_threshold))
            return 0.0 if equal_count > len(scores) / 2 else 1.0
            
        else:
            return float(np.mean(scores))
    
    def integrate_sequential_test(
        self,
        scores: Optional[List[float]] = None
    ) -> SPRTResult:
        """
        Integrate with sequential testing framework.
        
        Args:
            scores: Distance scores to test
            
        Returns:
            Sequential test result
        """
        if scores is None:
            scores = [r.distance_score for r in self.challenge_results]
        
        if not self.use_sequential_testing or not scores:
            return None
        
        # Run sequential test
        result = sequential_verify(
            scores=scores,
            alpha=self.alpha,
            beta=self.beta,
            delta=0.1,  # Effect size
            state=self.sequential_state
        )
        
        return result
    
    def localize_first_divergence(
        self,
        window_size: int = 5
    ) -> Optional[DivergencePoint]:
        """
        Localize the first significant divergence point.
        
        Args:
            window_size: Size of sliding window for detection
            
        Returns:
            First divergence point or None
        """
        if self.first_divergence:
            return self.first_divergence
        
        if len(self.challenge_results) < window_size:
            return None
        
        # Sliding window detection
        for i in range(len(self.challenge_results) - window_size + 1):
            window_scores = [
                r.distance_score 
                for r in self.challenge_results[i:i+window_size]
            ]
            
            # Check if window shows consistent divergence
            if all(s > self.divergence_threshold for s in window_scores):
                # Found divergence point
                divergence_result = self.challenge_results[i]
                
                self.first_divergence = DivergencePoint(
                    challenge_index=i,
                    challenge_id=divergence_result.challenge_id,
                    divergence_score=divergence_result.distance_score,
                    confidence=np.mean(window_scores),
                    context="sustained_divergence"
                )
                
                return self.first_divergence
        
        return None
    
    def make_decision(
        self,
        min_challenges: int = 10,
        max_challenges: Optional[int] = None
    ) -> AggregatedDecision:
        """
        Make final aggregated decision.
        
        Args:
            min_challenges: Minimum challenges before decision
            max_challenges: Maximum challenges to consider
            
        Returns:
            Aggregated decision with all statistics
        """
        n_challenges = len(self.challenge_results)
        
        if n_challenges < min_challenges:
            verdict = Verdict.UNDECIDED
            confidence = 0.0
        else:
            # Get scores for analysis
            scores = [r.distance_score for r in self.challenge_results[:max_challenges]]
            
            # Aggregate scores
            aggregated_score = self.aggregate_scores(scores)
            
            # Count equality indicators
            equal_count = sum(
                1 for r in self.challenge_results[:max_challenges]
                if r.equality_indicator
            )
            divergent_count = n_challenges - equal_count
            
            # Run sequential test if enabled
            sequential_result = None
            if self.use_sequential_testing:
                sequential_result = self.integrate_sequential_test(scores)
                
                # Use sequential test result if available
                if sequential_result and sequential_result.decision != "continue":
                    if sequential_result.decision == "accept_h0":
                        verdict = Verdict.SAME
                    else:
                        verdict = Verdict.DIFFERENT
                    confidence = sequential_result.confidence
                else:
                    # Fall back to threshold-based decision
                    if aggregated_score < (1.0 - self.equality_threshold):
                        verdict = Verdict.SAME
                    elif aggregated_score > self.divergence_threshold:
                        verdict = Verdict.DIFFERENT
                    else:
                        verdict = Verdict.UNDECIDED
                    
                    # Confidence based on score distribution
                    if n_challenges > 1:
                        std_dev = np.sqrt(self.running_var / n_challenges)
                        z_score = abs(aggregated_score - 0.5) / (std_dev + 1e-8)
                        confidence = min(1.0, z_score / 3.0)  # Normalize to [0, 1]
                    else:
                        confidence = 0.0
            else:
                # Threshold-based decision without sequential testing
                if equal_count > n_challenges * 0.8:
                    verdict = Verdict.SAME
                elif divergent_count > n_challenges * 0.5:
                    verdict = Verdict.DIFFERENT
                else:
                    verdict = Verdict.UNDECIDED
                
                confidence = abs(equal_count - divergent_count) / n_challenges
        
        # Try to localize first divergence
        if self.first_divergence is None:
            self.localize_first_divergence()
        
        return AggregatedDecision(
            verdict=verdict,
            confidence=confidence,
            total_challenges=n_challenges,
            equal_challenges=sum(1 for r in self.challenge_results if r.equality_indicator),
            divergent_challenges=sum(1 for r in self.challenge_results if not r.equality_indicator),
            mean_distance=self.running_mean,
            std_distance=np.sqrt(self.running_var / n_challenges) if n_challenges > 0 else 0.0,
            first_divergence=self.first_divergence,
            sequential_result=sequential_result,
            per_challenge_results=self.challenge_results[:max_challenges] if max_challenges else self.challenge_results
        )
    
    def reset(self):
        """Reset aggregator state for new comparison."""
        self.challenge_results.clear()
        self.first_divergence = None
        self.running_mean = 0.0
        self.running_var = 0.0
        self.n_samples = 0
        if self.sequential_state:
            self.sequential_state = SequentialState()