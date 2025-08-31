"""
Adaptive encoder with dynamic sparsity based on statistical testing of variance.
This ensures efficient generation of discriminative hypervectors.
"""

import numpy as np
import torch
from scipy import stats
from typing import Optional, Tuple, List, Dict, Union, Any
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AdjustmentStrategy(Enum):
    """Strategies for sparsity adjustment."""
    CONSERVATIVE = "conservative"  # Small adjustments
    AGGRESSIVE = "aggressive"      # Large adjustments
    ADAPTIVE = "adaptive"          # Adjusts based on history


@dataclass
class EncodingStats:
    """Statistics from encoding process."""
    final_sparsity: float
    mean_variance: float
    mean_discrimination: float
    sparsity_changes: int
    actual_density: float
    convergence_iterations: int
    quality_score: float


class AdaptiveSparsityEncoder:
    """
    Encoder with dynamic sparsity based on variance testing.
    Adjusts sparsity to maintain discriminative power while minimizing density.
    """
    
    def __init__(self, 
                 dimension: int = 10000,
                 initial_sparsity: float = 0.01,
                 min_sparsity: float = 0.001,
                 max_sparsity: float = 0.2,  # Increased from 0.1 to 0.2 for better fingerprinting
                 variance_threshold: float = 0.01,
                 discrimination_threshold: float = 0.3,
                 adjustment_strategy: AdjustmentStrategy = AdjustmentStrategy.ADAPTIVE,
                 convergence_tolerance: float = 0.001,
                 max_iterations: int = 100):
        """
        Initialize adaptive encoder.
        
        Args:
            dimension: Hypervector dimension
            initial_sparsity: Starting sparsity level
            min_sparsity: Minimum allowed sparsity
            max_sparsity: Maximum allowed sparsity
            variance_threshold: Minimum variance for discrimination
            discrimination_threshold: Minimum similarity difference for discrimination
            adjustment_strategy: Strategy for adjusting sparsity
            convergence_tolerance: Tolerance for convergence detection
            max_iterations: Maximum adjustment iterations
        """
        self.dimension = dimension
        self.current_sparsity = initial_sparsity
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.variance_threshold = variance_threshold
        self.discrimination_threshold = discrimination_threshold
        self.adjustment_strategy = adjustment_strategy
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # History for adaptive adjustment
        self.variance_history: List[float] = []
        self.discrimination_history: List[float] = []
        self.sparsity_history: List[float] = [initial_sparsity]
        
        # Adjustment parameters
        self.adjustment_rates = {
            AdjustmentStrategy.CONSERVATIVE: (0.95, 1.05),
            AdjustmentStrategy.AGGRESSIVE: (0.8, 1.25),
            AdjustmentStrategy.ADAPTIVE: (0.9, 1.1)  # Will be adapted
        }
        
        # For adaptive strategy
        self.success_count = 0
        self.failure_count = 0
        
        # Cache for performance
        self._position_cache: Dict[bytes, List[int]] = {}
        
    def test_variance(self, vectors: np.ndarray) -> Tuple[float, bool]:
        """
        Test variance of encoded vectors.
        
        Args:
            vectors: Array of encoded vectors [n_vectors, dimension]
            
        Returns:
            Tuple of (variance, passes_test)
        """
        if len(vectors) < 2:
            return 0.0, False
        
        # Calculate variance across vectors
        variance = np.var(vectors, axis=0).mean()
        
        # Also check variance between vectors (inter-vector variance)
        inter_variance = np.var(np.mean(vectors, axis=1))
        
        # Combined variance score
        combined_variance = 0.7 * variance + 0.3 * inter_variance
        
        # Test if variance is sufficient
        passes = combined_variance >= self.variance_threshold
        
        self.variance_history.append(combined_variance)
        
        return combined_variance, passes
    
    def test_discrimination(self, vectors: np.ndarray) -> Tuple[float, bool]:
        """
        Test discriminative power of vectors.
        
        Args:
            vectors: Array of encoded vectors
            
        Returns:
            Tuple of (discrimination_score, passes_test)
        """
        if len(vectors) < 2:
            return 0.0, False
        
        # Calculate pairwise similarities
        similarities = []
        n_vectors = len(vectors)
        
        # Sample if too many vectors for efficiency
        if n_vectors > 50:
            sample_indices = np.random.choice(n_vectors, 50, replace=False)
            sample_vectors = vectors[sample_indices]
        else:
            sample_vectors = vectors
        
        for i in range(len(sample_vectors)):
            for j in range(i + 1, len(sample_vectors)):
                sim = self._cosine_similarity(sample_vectors[i], sample_vectors[j])
                similarities.append(sim)
        
        if not similarities:
            return 0.0, False
        
        # Discrimination score is 1 - mean similarity
        # (we want low similarity between different vectors)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Penalize if similarities are too uniform (low std)
        discrimination = (1.0 - mean_sim) * (1.0 + min(std_sim, 0.2))
        
        passes = discrimination >= self.discrimination_threshold
        
        self.discrimination_history.append(discrimination)
        
        return discrimination, passes
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def adjust_sparsity(self, 
                        variance_score: float, 
                        discrimination_score: float) -> float:
        """
        Adjust sparsity based on test results.
        
        Args:
            variance_score: Current variance score
            discrimination_score: Current discrimination score
            
        Returns:
            New sparsity level
        """
        old_sparsity = self.current_sparsity
        
        # Get adjustment rates based on strategy
        if self.adjustment_strategy == AdjustmentStrategy.ADAPTIVE:
            # Adapt rates based on history
            if self.success_count > self.failure_count * 2:
                # Doing well, be more conservative
                decrease_rate, increase_rate = 0.95, 1.05
            elif self.failure_count > self.success_count * 2:
                # Struggling, be more aggressive
                decrease_rate, increase_rate = 0.85, 1.15
            else:
                # Balanced
                decrease_rate, increase_rate = 0.9, 1.1
        else:
            decrease_rate, increase_rate = self.adjustment_rates[self.adjustment_strategy]
        
        # Decision logic
        variance_pass = variance_score >= self.variance_threshold
        discrimination_pass = discrimination_score >= self.discrimination_threshold
        
        if variance_pass and discrimination_pass:
            # Both tests pass - try to reduce sparsity (fewer active elements)
            self.current_sparsity *= decrease_rate
            self.success_count += 1
            
        elif not variance_pass and not discrimination_pass:
            # Both tests fail - increase sparsity significantly
            self.current_sparsity *= increase_rate ** 1.5
            self.failure_count += 1
            
        elif not variance_pass:
            # Low variance - increase sparsity
            self.current_sparsity *= increase_rate * 1.2
            self.failure_count += 1
            
        elif not discrimination_pass:
            # Poor discrimination - increase sparsity
            self.current_sparsity *= increase_rate
            self.failure_count += 1
        
        # Clamp to bounds
        self.current_sparsity = np.clip(self.current_sparsity, self.min_sparsity, self.max_sparsity)
        
        self.sparsity_history.append(self.current_sparsity)
        
        # Log significant changes
        if abs(old_sparsity - self.current_sparsity) > 0.001:
            logger.info(f"Adjusted sparsity: {old_sparsity:.3%} -> {self.current_sparsity:.3%}")
            logger.info(f"  Variance: {variance_score:.4f} (threshold: {self.variance_threshold})")
            logger.info(f"  Discrimination: {discrimination_score:.4f} (threshold: {self.discrimination_threshold})")
        
        return self.current_sparsity
    
    def encode_adaptive(self, 
                        features: List[np.ndarray],
                        test_samples: int = 10,
                        auto_converge: bool = True) -> Tuple[List[np.ndarray], EncodingStats]:
        """
        Encode features with adaptive sparsity.
        
        Args:
            features: List of feature arrays to encode
            test_samples: Number of samples to use for testing
            auto_converge: Whether to iterate until convergence
            
        Returns:
            Tuple of (encoded_vectors, statistics)
        """
        encoded_vectors = []
        test_vectors = []
        convergence_iterations = 0
        
        # Auto-convergence mode
        if auto_converge and len(features) >= test_samples:
            # Find optimal sparsity before full encoding
            self._find_optimal_sparsity(features[:min(50, len(features))])
            convergence_iterations = len(self.sparsity_history)
        
        # Encode all features
        for i, feature in enumerate(features):
            # Encode with current sparsity
            vector = self._encode_with_sparsity(feature, self.current_sparsity)
            encoded_vectors.append(vector)
            
            # Collect test samples
            if i < test_samples:
                test_vectors.append(vector)
            
            # Periodically test and adjust if not in auto-converge mode
            if not auto_converge and (i + 1) % test_samples == 0 and len(test_vectors) >= 2:
                # Run tests
                variance, var_pass = self.test_variance(np.array(test_vectors))
                discrimination, disc_pass = self.test_discrimination(np.array(test_vectors))
                
                # Adjust sparsity
                self.adjust_sparsity(variance, discrimination)
                
                # Reset test vectors
                test_vectors = []
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(encoded_vectors)
        
        # Final statistics
        stats = EncodingStats(
            final_sparsity=self.current_sparsity,
            mean_variance=np.mean(self.variance_history) if self.variance_history else 0,
            mean_discrimination=np.mean(self.discrimination_history) if self.discrimination_history else 0,
            sparsity_changes=len(set(self.sparsity_history)) - 1,
            actual_density=np.mean([np.count_nonzero(v) / len(v) for v in encoded_vectors]),
            convergence_iterations=convergence_iterations,
            quality_score=quality_score
        )
        
        return encoded_vectors, stats
    
    def _find_optimal_sparsity(self, sample_features: List[np.ndarray]):
        """
        Find optimal sparsity through iterative testing.
        
        Args:
            sample_features: Sample features for testing
        """
        prev_sparsity = self.current_sparsity
        
        for iteration in range(self.max_iterations):
            # Encode samples with current sparsity
            test_vectors = [self._encode_with_sparsity(f, self.current_sparsity) 
                          for f in sample_features]
            
            # Test variance and discrimination
            variance, var_pass = self.test_variance(np.array(test_vectors))
            discrimination, disc_pass = self.test_discrimination(np.array(test_vectors))
            
            # Check convergence
            if abs(self.current_sparsity - prev_sparsity) < self.convergence_tolerance:
                if var_pass and disc_pass:
                    logger.info(f"Converged to optimal sparsity: {self.current_sparsity:.3%}")
                    break
            
            prev_sparsity = self.current_sparsity
            
            # Adjust sparsity
            self.adjust_sparsity(variance, discrimination)
    
    def _encode_with_sparsity(self, feature: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Encode feature with specified sparsity using deterministic hashing.
        
        Args:
            feature: Input feature array
            sparsity: Sparsity level to use
            
        Returns:
            Sparse encoded vector
        """
        vector = np.zeros(self.dimension, dtype=np.float32)
        
        # Number of active positions
        n_active = max(1, int(self.dimension * sparsity))
        
        # Get or generate positions
        feature_hash = hashlib.sha256(feature.tobytes()).digest()
        
        if feature_hash in self._position_cache:
            positions = self._position_cache[feature_hash][:n_active]
        else:
            # Generate unique positions deterministically
            positions = self._generate_positions(feature_hash, n_active)
            self._position_cache[feature_hash] = positions
        
        # Set values at active positions using Box-Muller transform
        for i, pos in enumerate(positions[:n_active]):
            # Generate two uniform values from hash
            val_hash1 = hashlib.sha256(feature_hash + (i * 2).to_bytes(4, 'big')).digest()
            val_hash2 = hashlib.sha256(feature_hash + (i * 2 + 1).to_bytes(4, 'big')).digest()
            
            u1 = int.from_bytes(val_hash1[:4], 'big') / (2**32)
            u2 = int.from_bytes(val_hash2[:4], 'big') / (2**32)
            
            # Box-Muller transform for Gaussian
            if u1 > 0:  # Avoid log(0)
                z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
                vector[pos] = np.clip(z / 3, -1, 1)  # Clip to [-1, 1]
        
        # L2 normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def _generate_positions(self, feature_hash: bytes, n_positions: int) -> List[int]:
        """Generate unique positions deterministically."""
        positions = set()
        counter = 0
        max_attempts = n_positions * 10
        
        while len(positions) < n_positions and counter < max_attempts:
            pos_hash = hashlib.sha256(feature_hash + counter.to_bytes(4, 'big')).digest()
            position = int.from_bytes(pos_hash[:4], 'big') % self.dimension
            positions.add(position)
            counter += 1
        
        return list(positions)
    
    def _calculate_quality_score(self, vectors: List[np.ndarray]) -> float:
        """
        Calculate overall quality score of encoded vectors.
        
        Args:
            vectors: List of encoded vectors
            
        Returns:
            Quality score between 0 and 1
        """
        if len(vectors) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(100, len(vectors))
        sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample_vectors = [vectors[i] for i in sample_indices]
        
        # Test variance and discrimination
        variance, var_pass = self.test_variance(np.array(sample_vectors))
        discrimination, disc_pass = self.test_discrimination(np.array(sample_vectors))
        
        # Calculate density efficiency (prefer lower density)
        actual_density = np.mean([np.count_nonzero(v) / len(v) for v in sample_vectors])
        density_score = 1.0 - min(actual_density / self.max_sparsity, 1.0)
        
        # Combined quality score
        quality = (
            0.3 * (variance / (self.variance_threshold * 2)) +  # Variance contribution
            0.4 * (discrimination / (self.discrimination_threshold * 2)) +  # Discrimination contribution
            0.3 * density_score  # Efficiency contribution
        )
        
        return np.clip(quality, 0, 1)
    
    def run_statistical_tests(self, vectors: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive statistical tests on vectors.
        
        Args:
            vectors: Array of vectors to test
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Density test
        densities = [np.count_nonzero(v) / len(v) for v in vectors]
        results['density'] = {
            'mean': float(np.mean(densities)),
            'std': float(np.std(densities)),
            'min': float(np.min(densities)),
            'max': float(np.max(densities)),
            'target': self.current_sparsity
        }
        
        # Variance test
        variance = np.var(vectors, axis=0)
        results['variance'] = {
            'mean': float(np.mean(variance)),
            'std': float(np.std(variance)),
            'passes_threshold': np.mean(variance) >= self.variance_threshold,
            'threshold': self.variance_threshold
        }
        
        # Discrimination test
        if len(vectors) >= 2:
            similarities = []
            sample_size = min(100, len(vectors))
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    sim = self._cosine_similarity(vectors[sample_indices[i]], 
                                                 vectors[sample_indices[j]])
                    similarities.append(sim)
            
            results['discrimination'] = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'discrimination_score': float(1.0 - np.mean(similarities)),
                'passes_threshold': (1.0 - np.mean(similarities)) >= self.discrimination_threshold,
                'threshold': self.discrimination_threshold
            }
        
        # Kolmogorov-Smirnov test for distribution
        if len(vectors) >= 2:
            # Test if values follow expected distribution
            all_values = vectors[vectors != 0]  # Only non-zero values
            if len(all_values) > 0:
                ks_stat, ks_pvalue = stats.kstest(all_values, 'norm')
                results['distribution'] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'is_normal': ks_pvalue > 0.05
                }
        
        # Entropy test
        if len(vectors) > 0:
            # Calculate entropy of active positions
            active_positions = np.sum(vectors != 0, axis=0)
            position_probs = active_positions / np.sum(active_positions)
            position_probs = position_probs[position_probs > 0]  # Remove zeros
            entropy = -np.sum(position_probs * np.log2(position_probs))
            max_entropy = np.log2(self.dimension)
            
            results['entropy'] = {
                'value': float(entropy),
                'normalized': float(entropy / max_entropy),
                'max_possible': float(max_entropy)
            }
        
        # Convergence analysis
        if len(self.sparsity_history) > 1:
            recent_changes = np.diff(self.sparsity_history[-10:])
            results['convergence'] = {
                'converged': np.max(np.abs(recent_changes)) < self.convergence_tolerance,
                'iterations': len(self.sparsity_history) - 1,
                'final_sparsity': self.current_sparsity,
                'total_change': abs(self.sparsity_history[-1] - self.sparsity_history[0])
            }
        
        return results
    
    def reset(self):
        """Reset encoder to initial state."""
        self.current_sparsity = self.sparsity_history[0] if self.sparsity_history else 0.01
        self.variance_history.clear()
        self.discrimination_history.clear()
        self.sparsity_history = [self.current_sparsity]
        self.success_count = 0
        self.failure_count = 0
        self._position_cache.clear()
        
    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Encode a list of tokens into a hypervector.
        
        Args:
            tokens: List of tokens to encode
            
        Returns:
            Hypervector representation
        """
        # Create a combined feature vector from tokens
        # Use hash of each token to create deterministic features
        hypervector = np.zeros(self.dimension)
        
        for token in tokens:
            # Hash token to get deterministic indices
            token_hash = hash(token)
            np.random.seed(abs(token_hash) % (2**32))
            
            # Select random dimensions to activate
            num_active = int(self.dimension * self.current_sparsity)
            active_dims = np.random.choice(self.dimension, num_active, replace=False)
            
            # Add contribution from this token
            for dim in active_dims:
                hypervector[dim] += np.random.randn()
        
        # Normalize
        if np.linalg.norm(hypervector) > 0:
            hypervector = hypervector / np.linalg.norm(hypervector)
        
        return hypervector
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of encoder state and history."""
        return {
            'current_sparsity': self.current_sparsity,
            'sparsity_range': (self.min_sparsity, self.max_sparsity),
            'history_length': len(self.sparsity_history),
            'mean_variance': np.mean(self.variance_history) if self.variance_history else 0,
            'mean_discrimination': np.mean(self.discrimination_history) if self.discrimination_history else 0,
            'success_rate': self.success_count / max(1, self.success_count + self.failure_count),
            'adjustment_strategy': self.adjustment_strategy.value,
            'cache_size': len(self._position_cache)
        }