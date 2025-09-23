"""
Improved Dual Library System for REV with Cross-Size Model Matching
Handles models of different sizes within the same family
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from scipy import interpolate
from scipy.stats import pearsonr, spearmanr
from numpy.linalg import norm

logger = logging.getLogger(__name__)


@dataclass
class ModelIdentification:
    """Result of model identification"""
    identified_family: Optional[str]
    confidence: float
    method: str  # "name_match", "fingerprint_match", "unknown"
    reference_model: Optional[str] = None  # e.g., "gpt2", "llama-7b"
    notes: str = ""


class DualLibrarySystem:
    """
    Improved matching algorithm that handles cross-size model comparison.
    Dual Library System for REV with enhanced cross-size matching.
    - Reference Library: Base fingerprints from smallest models of each family
    - Active Library: Continuously updated with new runs
    """

    # Known model family patterns
    FAMILY_PATTERNS = {
        "gpt": ["gpt", "gpt2", "gpt-neo", "gpt-j", "distilgpt"],
        "llama": ["llama", "alpaca", "vicuna", "guanaco"],
        "mistral": ["mistral", "mixtral"],
        "qwen": ["qwen", "qwen2"],
        "yi": ["yi-"],
        "falcon": ["falcon"],
        "bloom": ["bloom", "bloomz"],
        "opt": ["opt-"],
        "pythia": ["pythia"],
        "dolly": ["dolly"],
        "stablelm": ["stablelm", "stable-lm"],
    }

    # Reference models (smallest of each family)
    REFERENCE_MODELS = {
        "gpt": "gpt2",  # 124M
        "llama": "llama-2-7b",  # 7B
        "mistral": "mistral-7b",  # 7B
        "qwen": "qwen-1.8b",  # 1.8B
        "yi": "yi-6b",  # 6B
        "falcon": "falcon-1b",  # 1B
        "bloom": "bloom-560m",  # 560M
        "opt": "opt-125m",  # 125M
        "pythia": "pythia-70m",  # 70M
    }

    def __init__(self,
                 reference_path: str = "fingerprint_library/reference_library.json",
                 active_path: str = "fingerprint_library/active_library.json"):
        """Initialize dual library system with caching."""
        self.reference_path = Path(reference_path)
        self.active_path = Path(active_path)
        self.interpolation_cache = {}  # Cache for expensive interpolation calculations

        # Load libraries
        self.reference_library = self._load_library(self.reference_path)
        self.active_library = self._load_library(self.active_path)

    def load_library(self, library_type: str) -> Dict:
        """Load library by type (for compatibility)."""
        if library_type == "reference":
            return self._load_library(self.reference_path)
        elif library_type == "active":
            return self._load_library(self.active_path)
        else:
            return {}

    def _load_library(self, path: Path) -> Dict:
        """Load a library from disk."""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load library {path}: {e}")
        return {}

    def identify_from_behavioral_analysis(self, fingerprint_dict: Dict, verbose: bool = False) -> ModelIdentification:
        """
        Match behavioral fingerprint using relative layer positions and improved similarity metrics.
        Handles models of different sizes within the same family.
        """
        print(f"\n[MATCHING] Starting cross-size behavioral analysis matching")

        # Load reference models
        reference_models = []
        for fp_id, ref_data in self.reference_library.get("fingerprints", {}).items():
            reference_models.append(ref_data)

        if not reference_models:
            return ModelIdentification(
                identified_family=None,
                confidence=0.0,
                method="no_reference_library",
                notes="No reference library available"
            )

        # Extract test data
        test_variance = fingerprint_dict.get('variance_profile', [])
        test_sites = fingerprint_dict.get('restriction_sites', [])
        test_layers = fingerprint_dict.get('layer_count', 0)
        test_sampled = fingerprint_dict.get('layers_sampled', [])

        # If no layer count, try to infer from sampled layers
        if not test_layers and test_sampled:
            test_layers = max(test_sampled) + 1
        elif not test_layers:
            test_layers = 80  # Default for large models

        print(f"[MATCHING] Test model: {test_layers} layers, {len(test_variance)} variance samples")
        print(f"[MATCHING] Test sampled layers: {test_sampled[:5]}..." if test_sampled else "[MATCHING] No sampled layer info")

        best_match = None
        best_score = 0.0
        match_results = []

        for ref_model in reference_models:
            ref_variance = ref_model.get('variance_profile', [])
            ref_behavioral = ref_model.get('behavioral_profile', [])

            # Use behavioral_profile if variance_profile not found
            if not ref_variance and ref_behavioral:
                ref_variance = ref_behavioral

            ref_sites = ref_model.get('restriction_sites', [])
            ref_layers = ref_model.get('layer_count', ref_model.get('total_layers', 32))
            ref_family = ref_model.get('model_family', ref_model.get('family', 'unknown'))

            if verbose:
                print(f"  Testing against reference: {ref_family} ({ref_layers} layers)")
                print(f"    Reference variance profile length: {len(ref_variance)}")
                print(f"    Reference restriction sites: {len(ref_sites)}")

            # Try to determine sampled layers for reference
            ref_sampled = ref_model.get('layers_sampled', [])
            if not ref_sampled and ref_variance:
                # Assume evenly distributed if not specified
                ref_sampled = list(range(len(ref_variance)))

            print(f"\n[MATCHING] Comparing with {ref_family} ({ref_layers} layers)")

            # Skip if no variance profile
            if not ref_variance or not test_variance:
                print(f"  Skipping - missing variance profiles")
                continue

            # === METHOD 1: Interpolated Variance Profile Comparison ===
            # Convert both profiles to relative depth (0.0 to 1.0)

            # For test model: convert layer indices to relative positions
            if test_sampled:
                test_relative_positions = [layer / test_layers for layer in test_sampled[:len(test_variance)]]
            else:
                # Assume evenly distributed sampling
                test_relative_positions = np.linspace(0, 1, len(test_variance)).tolist()

            # For reference model: convert layer indices to relative positions
            if ref_sampled:
                ref_relative_positions = [layer / ref_layers for layer in ref_sampled[:len(ref_variance)]]
            else:
                ref_relative_positions = np.linspace(0, 1, len(ref_variance)).tolist()

            print(f"  Test relative positions: {test_relative_positions[:3]}...")
            print(f"  Ref relative positions: {ref_relative_positions[:3]}...")

            # Interpolate both profiles to common relative positions (0 to 1 in 50 steps)
            common_positions = np.linspace(0, 1, 50)

            try:
                # Ensure we have enough points for interpolation
                if len(test_relative_positions) < 2 or len(ref_relative_positions) < 2:
                    print(f"  Not enough data points for interpolation")
                    continue

                # Normalize variances to account for different model scales
                test_variance_normalized = self._normalize_variance_profile(test_variance)
                ref_variance_normalized = self._normalize_variance_profile(ref_variance)

                if verbose:
                    print(f"    Original test variance range: {min(test_variance):.3f} - {max(test_variance):.3f}")
                    print(f"    Normalized test variance range: {min(test_variance_normalized):.3f} - {max(test_variance_normalized):.3f}")
                    print(f"    Original ref variance range: {min(ref_variance):.3f} - {max(ref_variance):.3f}")
                    print(f"    Normalized ref variance range: {min(ref_variance_normalized):.3f} - {max(ref_variance_normalized):.3f}")

                # Check interpolation cache
                test_cache_key = f"test_{hash(tuple(test_variance_normalized))}"
                ref_cache_key = f"ref_{hash(tuple(ref_variance_normalized))}"

                if test_cache_key in self.interpolation_cache:
                    test_interpolated = self.interpolation_cache[test_cache_key]
                    if verbose:
                        print(f"    Using cached test interpolation")
                else:
                    # Interpolate test variance profile
                    test_interp = interpolate.interp1d(
                        test_relative_positions[:len(test_variance_normalized)],
                        test_variance_normalized[:len(test_relative_positions)],
                        kind='linear',
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                    test_interpolated = test_interp(common_positions)
                    self.interpolation_cache[test_cache_key] = test_interpolated
                    if verbose:
                        print(f"    Computed and cached test interpolation")

                if ref_cache_key in self.interpolation_cache:
                    ref_interpolated = self.interpolation_cache[ref_cache_key]
                    if verbose:
                        print(f"    Using cached reference interpolation")
                else:
                    # Interpolate reference variance profile
                    ref_interp = interpolate.interp1d(
                        ref_relative_positions[:len(ref_variance_normalized)],
                        ref_variance_normalized[:len(ref_relative_positions)],
                        kind='linear',
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                    ref_interpolated = ref_interp(common_positions)
                    self.interpolation_cache[ref_cache_key] = ref_interpolated
                    if verbose:
                        print(f"    Computed and cached reference interpolation")

                # === NEW: TOPOLOGICAL SIMILARITY METRICS ===

                # 1. Cosine Similarity (captures shape regardless of magnitude)

                # Center the profiles (remove mean to focus on shape)
                test_centered = test_interpolated - np.mean(test_interpolated)
                ref_centered = ref_interpolated - np.mean(ref_interpolated)

                # Calculate cosine similarity
                cosine_sim = np.dot(test_centered, ref_centered) / (norm(test_centered) * norm(ref_centered) + 1e-8)
                cosine_sim = max(0, cosine_sim)  # Ensure non-negative

                print(f"  Cosine similarity (shape): {cosine_sim:.3f}")

                # 2. Normalized Profile Correlation
                # Normalize both profiles to [0, 1] range to compare patterns
                test_min, test_max = test_interpolated.min(), test_interpolated.max()
                ref_min, ref_max = ref_interpolated.min(), ref_interpolated.max()

                if test_max - test_min > 1e-6 and ref_max - ref_min > 1e-6:
                    test_normalized = (test_interpolated - test_min) / (test_max - test_min)
                    ref_normalized = (ref_interpolated - ref_min) / (ref_max - ref_min)

                    # Pearson correlation on normalized profiles
                    normalized_corr, _ = pearsonr(test_normalized, ref_normalized)
                    if np.isnan(normalized_corr):
                        normalized_corr = 0.0
                else:
                    normalized_corr = 0.0

                print(f"  Normalized correlation: {normalized_corr:.3f}")

                # 3. Dynamic Time Warping Distance (captures similar patterns even if shifted)
                # Simple DTW implementation for behavioral patterns
                def simple_dtw_distance(x, y, window=10):
                    """Calculate DTW distance between two sequences."""
                    n, m = len(x), len(y)

                    # Create cost matrix with window constraint
                    dtw_matrix = np.full((n + 1, m + 1), np.inf)
                    dtw_matrix[0, 0] = 0

                    for i in range(1, n + 1):
                        for j in range(max(1, i - window), min(m + 1, i + window)):
                            cost = abs(x[i-1] - y[j-1])
                            dtw_matrix[i, j] = cost + min(
                                dtw_matrix[i-1, j],    # insertion
                                dtw_matrix[i, j-1],    # deletion
                                dtw_matrix[i-1, j-1]   # match
                            )

                    # Normalize by path length
                    dtw_distance = dtw_matrix[n, m] / (n + m)
                    return dtw_distance

                # Calculate DTW distance and convert to similarity
                dtw_distance = simple_dtw_distance(test_normalized if 'test_normalized' in locals() else test_interpolated,
                                                   ref_normalized if 'ref_normalized' in locals() else ref_interpolated)

                # Convert distance to similarity (lower distance = higher similarity)
                dtw_similarity = max(0, 1.0 - dtw_distance / 0.5)  # Normalize assuming max meaningful distance is 0.5

                print(f"  DTW similarity (pattern): {dtw_similarity:.3f}")

                # 4. Topology Signature Matching
                # Extract key topological features
                def extract_topology_signature(profile):
                    """Extract topological features from variance profile."""
                    # Find peaks and valleys
                    peaks = []
                    valleys = []

                    for i in range(1, len(profile) - 1):
                        if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                            peaks.append(i / len(profile))  # Relative position
                        elif profile[i] < profile[i-1] and profile[i] < profile[i+1]:
                            valleys.append(i / len(profile))

                    # Calculate gradient statistics
                    gradients = np.diff(profile)

                    return {
                        'num_peaks': len(peaks),
                        'num_valleys': len(valleys),
                        'peak_positions': peaks[:5],  # First 5 peaks
                        'valley_positions': valleys[:5],  # First 5 valleys
                        'mean_gradient': np.mean(np.abs(gradients)),
                        'gradient_variance': np.var(gradients),
                        'smoothness': np.mean(np.abs(np.diff(gradients)))  # Second derivative
                    }

                test_topology = extract_topology_signature(test_interpolated)
                ref_topology = extract_topology_signature(ref_interpolated)

                # Compare topological features
                topology_score = 0.0
                weights = {
                    'num_peaks': 0.15,
                    'num_valleys': 0.15,
                    'peak_positions': 0.25,
                    'valley_positions': 0.25,
                    'mean_gradient': 0.1,
                    'smoothness': 0.1
                }

                # Compare number of peaks/valleys (similar complexity)
                if ref_topology['num_peaks'] > 0:
                    peak_similarity = min(test_topology['num_peaks'], ref_topology['num_peaks']) / max(test_topology['num_peaks'], ref_topology['num_peaks'], 1)
                    topology_score += weights['num_peaks'] * peak_similarity

                if ref_topology['num_valleys'] > 0:
                    valley_similarity = min(test_topology['num_valleys'], ref_topology['num_valleys']) / max(test_topology['num_valleys'], ref_topology['num_valleys'], 1)
                    topology_score += weights['num_valleys'] * valley_similarity

                # Compare peak/valley positions (similar pattern structure)
                if test_topology['peak_positions'] and ref_topology['peak_positions']:
                    position_distances = []
                    for test_peak in test_topology['peak_positions']:
                        if ref_topology['peak_positions']:
                            min_dist = min(abs(test_peak - ref_peak) for ref_peak in ref_topology['peak_positions'])
                            position_distances.append(min_dist)
                    if position_distances:
                        peak_position_similarity = max(0, 1.0 - np.mean(position_distances))
                        topology_score += weights['peak_positions'] * peak_position_similarity

                # Compare gradient characteristics (similar rate of change)
                if ref_topology['mean_gradient'] > 0:
                    gradient_ratio = min(test_topology['mean_gradient'], ref_topology['mean_gradient']) / max(test_topology['mean_gradient'], ref_topology['mean_gradient'])
                    topology_score += weights['mean_gradient'] * gradient_ratio

                if ref_topology['smoothness'] > 0:
                    smoothness_ratio = min(test_topology['smoothness'], ref_topology['smoothness']) / max(test_topology['smoothness'], ref_topology['smoothness'], 1e-8)
                    topology_score += weights['smoothness'] * smoothness_ratio

                print(f"  Topology signature match: {topology_score:.3f}")

                # 5. Fourier Transform Similarity (frequency domain comparison)
                # This captures periodic patterns in the behavioral profile
                test_fft = np.abs(np.fft.fft(test_centered))[:50]  # First 50 frequency components
                ref_fft = np.abs(np.fft.fft(ref_centered))[:50]

                # Normalize FFT magnitudes
                test_fft = test_fft / (np.sum(test_fft) + 1e-8)
                ref_fft = ref_fft / (np.sum(ref_fft) + 1e-8)

                # Compare frequency distributions
                fft_similarity = 1.0 - np.sum(np.abs(test_fft - ref_fft)) / 2.0  # Convert L1 distance to similarity

                print(f"  Fourier similarity (periodicity): {fft_similarity:.3f}")

                # Original metrics (kept but with lower weight)
                pearson_corr, _ = pearsonr(test_interpolated, ref_interpolated)
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0

                spearman_corr, _ = spearmanr(test_interpolated, ref_interpolated)
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0

                mae = np.mean(np.abs(test_interpolated - ref_interpolated))
                mae_similarity = max(0, 1.0 - (mae / 0.2))

                print(f"  Pearson correlation: {pearson_corr:.3f}")
                print(f"  Spearman correlation: {spearman_corr:.3f}")
                print(f"  MAE similarity: {mae_similarity:.3f}")

            except Exception as e:
                print(f"  Interpolation/calculation failed: {e}")
                cosine_sim = normalized_corr = dtw_similarity = topology_score = fft_similarity = 0.0
                pearson_corr = spearman_corr = mae_similarity = 0.0

            # === METHOD 2: Restriction Site Pattern Matching ===
            site_similarity = 0.0

            if test_sites and ref_sites:
                # Convert restriction sites to relative positions
                test_site_positions = []
                for site in test_sites[:10]:  # Use up to 10 sites
                    if isinstance(site, dict) and 'layer' in site:
                        relative_pos = site['layer'] / test_layers
                        test_site_positions.append(relative_pos)

                ref_site_positions = []
                for site in ref_sites[:10]:
                    if isinstance(site, dict) and 'layer' in site:
                        relative_pos = site['layer'] / ref_layers
                        ref_site_positions.append(relative_pos)

                if test_site_positions and ref_site_positions:
                    # Find closest matches for each test site
                    site_distances = []
                    for test_pos in test_site_positions:
                        # Find closest reference site
                        min_dist = min(abs(test_pos - ref_pos) for ref_pos in ref_site_positions)
                        site_distances.append(min_dist)

                    # Convert distances to similarity (closer = higher similarity)
                    # Average distance of 0.1 (10% of model depth) = 0.5 similarity
                    avg_distance = np.mean(site_distances)
                    site_similarity = max(0, 1.0 - (avg_distance * 5))

                    print(f"  Restriction site similarity: {site_similarity:.3f}")

            # === METHOD 3: Behavioral Pattern Matching ===
            # Check if variance profiles follow similar patterns (high/low regions)
            pattern_similarity = 0.0

            if len(test_variance) > 3 and len(ref_variance) > 3:
                # Identify high variance regions (above median)
                test_median = np.median(test_variance)
                ref_median = np.median(ref_variance)

                test_high_regions = [i/len(test_variance) for i, v in enumerate(test_variance) if v > test_median]
                ref_high_regions = [i/len(ref_variance) for i, v in enumerate(ref_variance) if v > ref_median]

                if test_high_regions and ref_high_regions:
                    # Check overlap of high variance regions
                    overlap_count = 0
                    for test_region in test_high_regions:
                        for ref_region in ref_high_regions:
                            if abs(test_region - ref_region) < 0.15:  # Within 15% relative position
                                overlap_count += 1
                                break

                    pattern_similarity = overlap_count / max(len(test_high_regions), len(ref_high_regions))
                    print(f"  Pattern similarity: {pattern_similarity:.3f}")

            # === METHOD 4: Syntactic Feature Matching ===
            syntactic_similarity = 0.0

            # Extract response-based features if available
            test_responses = fingerprint_dict.get('response_text', '')
            ref_responses = ref_model.get('response_text', '')

            if test_responses and ref_responses:
                # Token diversity comparison (simplified)
                test_tokens = set(test_responses.lower().split())
                ref_tokens = set(ref_responses.lower().split())
                if test_tokens and ref_tokens:
                    token_overlap = len(test_tokens & ref_tokens) / max(len(test_tokens), len(ref_tokens))
                    syntactic_similarity = token_overlap
                    print(f"  Syntactic similarity: {syntactic_similarity:.3f}")

            # === METHOD 5: Semantic Feature Matching ===
            semantic_similarity = 0.0

            # Compare embedding statistics if available
            test_embeddings = fingerprint_dict.get('embedding_stats', {})
            ref_embeddings = ref_model.get('embedding_stats', {})

            if test_embeddings and ref_embeddings:
                # Compare mean and std of embeddings
                test_mean = test_embeddings.get('mean', 0)
                ref_mean = ref_embeddings.get('mean', 0)
                if test_mean and ref_mean:
                    semantic_similarity = 1.0 - abs(test_mean - ref_mean) / max(abs(test_mean), abs(ref_mean))
                    print(f"  Semantic similarity: {semantic_similarity:.3f}")

            # === METHOD 6: Behavioral Feature Matching ===
            behavioral_similarity = 0.0

            # Compare response consistency and patterns
            test_consistency = fingerprint_dict.get('response_consistency', 0)
            ref_consistency = ref_model.get('response_consistency', 0)

            if test_consistency and ref_consistency:
                behavioral_similarity = 1.0 - abs(test_consistency - ref_consistency)
                print(f"  Behavioral consistency match: {behavioral_similarity:.3f}")

            # === METHOD 7: Architectural Feature Matching ===
            architectural_similarity = 0.0

            # Compare layer count and model dimensions
            test_dims = fingerprint_dict.get('hidden_size', 0)
            ref_dims = ref_model.get('hidden_size', 0)

            # Layer count similarity (already have this)
            layer_ratio = min(test_layers, ref_layers) / max(test_layers, ref_layers) if test_layers > 0 else 0

            # Dimension similarity if available
            dim_ratio = 1.0
            if test_dims and ref_dims:
                dim_ratio = min(test_dims, ref_dims) / max(test_dims, ref_dims)

            architectural_similarity = (layer_ratio + dim_ratio) / 2
            print(f"  Architectural similarity: {architectural_similarity:.3f}")

            # === COMBINE SCORES WITH TOPOLOGY-FOCUSED WEIGHTS ===
            # Prioritize shape/topology over absolute value matching
            scores = {
                # Topology-based metrics (60% total weight) - these capture SHAPE
                'cosine': (cosine_sim, 0.20),               # Shape similarity
                'normalized_corr': (normalized_corr, 0.15), # Pattern correlation
                'dtw': (dtw_similarity, 0.10),              # Pattern matching with shifts
                'topology': (topology_score, 0.10),         # Structural features
                'fourier': (fft_similarity, 0.05),          # Periodic patterns

                # Value-based metrics (15% total weight) - less important
                'pearson': (pearson_corr, 0.05),            # Linear correlation
                'spearman': (spearman_corr, 0.05),          # Rank correlation
                'mae': (mae_similarity, 0.05),              # Absolute difference

                # Structural methods (25% total weight)
                'sites': (site_similarity, 0.15),           # Restriction sites
                'pattern': (pattern_similarity, 0.10),      # High/low patterns
            }

            # Calculate weighted average
            total_weight = sum(weight for _, weight in scores.values())
            weighted_sum = sum(score * weight for score, weight in scores.values())
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Boost if multiple topology metrics agree
            topology_metrics = [cosine_sim, normalized_corr, dtw_similarity, topology_score]
            if sum(m > 0.6 for m in topology_metrics) >= 2:  # At least 2 topology metrics show good match
                final_score = min(1.0, final_score * 1.15)
                print(f"  Topology agreement boost applied")

            print(f"  FINAL SCORE: {final_score:.3f}")

            match_results.append({
                'family': ref_family,
                'score': final_score,
                'components': scores,
                'ref_layers': ref_layers,
                'size_ratio': ref_layers / test_layers if test_layers > 0 else 0
            })

            if final_score > best_score:
                best_score = final_score
                best_match = ref_model

        # Sort results by score
        match_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n[MATCHING] Top matches:")
        for i, result in enumerate(match_results[:3]):
            print(f"  {i+1}. {result['family']}: {result['score']:.3f} (size ratio: {result['size_ratio']:.2f})")

        # === DYNAMIC THRESHOLD BASED ON MODEL SIZE DIFFERENCE ===
        # Lower threshold for models of different sizes
        if best_match:
            ref_layers = best_match.get('layer_count', best_match.get('total_layers', 32))
            size_ratio = min(ref_layers, test_layers) / max(ref_layers, test_layers) if test_layers > 0 else 1.0

            # Adjust threshold based on size difference
            # Same size (ratio=1.0): threshold=0.45
            # 2x size difference (ratio=0.5): threshold=0.30
            # 3x size difference (ratio=0.33): threshold=0.25
            base_threshold = 0.45
            threshold_adjustment = (1.0 - size_ratio) * 0.25
            adjusted_threshold = max(0.20, base_threshold - threshold_adjustment)

            print(f"\n[MATCHING] Size ratio: {size_ratio:.2f}, Adjusted threshold: {adjusted_threshold:.3f}")

            if best_score >= adjusted_threshold:
                # Enhanced confidence calculation with better boosting
                confidence_boost = 1.0
                ref_family = best_match.get('model_family', best_match.get('family', ''))

                # Progressive boost based on base confidence level
                if best_score < 0.5:
                    base_boost = 1.4  # Stronger boost for lower scores that pass threshold
                elif best_score < 0.7:
                    base_boost = 1.2
                else:
                    base_boost = 1.1

                confidence_boost *= base_boost

                # Special boost for known architecture patterns
                if 'llama' in ref_family.lower():
                    if test_layers == 80:  # Exact match for Llama-70B
                        confidence_boost *= 1.2
                    elif test_layers > 32:  # Other large Llama models
                        confidence_boost *= 1.15

                # Common size transition pairs get additional boost
                common_transitions = [
                    (32, 80),   # 7B -> 70B
                    (32, 40),   # 7B -> 13B
                    (6, 12),    # DistilGPT -> GPT2
                    (32, 64),   # Small -> Medium
                ]

                size_pair = (ref_layers, test_layers)
                if size_pair in common_transitions or (test_layers, ref_layers) in common_transitions:
                    confidence_boost *= 1.1

                # Calculate final confidence
                final_confidence = min(0.95, best_score * confidence_boost)

                # Determine confidence label
                if final_confidence > 0.85:
                    confidence_label = "high_confidence"
                elif final_confidence > 0.65:
                    confidence_label = "moderate_confidence"
                elif final_confidence > 0.45:
                    confidence_label = "low_confidence"
                else:
                    confidence_label = "uncertain"

                return ModelIdentification(
                    identified_family=ref_family,
                    confidence=final_confidence,
                    method="cross_size_behavioral_matching",
                    reference_model=best_match.get('reference_model', 'unknown'),
                    notes=f"Matched across size difference ({ref_layers} vs {test_layers} layers) - {confidence_label}"
                )

        # Return actual score even if below threshold (for debugging)
        if best_match:
            return ModelIdentification(
                identified_family=None,
                confidence=best_score,
                method="below_threshold",
                notes=f"Best match: {best_match.get('model_family', 'unknown')} at {best_score:.3f}"
            )

        # No match found
        return ModelIdentification(
            identified_family=None,
            confidence=0.0,
            method="no_match",
            notes="No behavioral patterns matched"
        )


    def identify_model(self, model_path: str) -> ModelIdentification:
        """
        Identify a model through LIGHT behavioral analysis.

        This does a QUICK probe (not deep analysis) to identify the model family
        so we can use reference libraries for 15-20x speedup.
        """
        model_name = Path(model_path).name.lower()

        # SECURITY: We do behavioral verification, but LIGHTWEIGHT
        logger.info(f"Starting LIGHT behavioral probe for {model_name}")
        logger.info("Quick topology scan for reference matching...")

        # For now, mark as requiring light probe
        # The actual light probe will be done by the pipeline
        return ModelIdentification(
            identified_family=None,
            confidence=0.0,
            method="needs_light_probe",
            reference_model=None,
            notes="Light behavioral probe needed for family identification"
        )

    def get_reference_fingerprint(self, family: str) -> Optional[Dict]:
        """Get the reference fingerprint for a model family."""
        for fp_id, fp_data in self.reference_library.get("fingerprints", {}).items():
            if fp_data.get("model_family") == family:
                return fp_data
        return None

    def get_testing_strategy(self, identification: ModelIdentification, test_layer_count: int = None, variance_profile: List[float] = None) -> Dict:
        """
        Get testing strategy based on identification.

        Args:
            identification: Model identification result
            test_layer_count: Number of layers in test model (for scaling)
            variance_profile: Actual variance measurements from light probe

        Returns:
            Dict with testing configuration
        """
        # If we have a behavioral match with good confidence, use reference
        if identification.method == "cross_size_behavioral_matching" and identification.confidence > 0.3:
            # We have a family identification - check for reference
            reference_fp = self.get_reference_fingerprint(identification.identified_family)
            if reference_fp:
                # Extract restriction sites from reference
                restriction_sites = reference_fp.get("restriction_sites", [])
                focus_layers = []

                # Get reference and test model layer counts for scaling
                ref_layer_count = reference_fp.get("layer_count", reference_fp.get("behavioral_topology", {}).get("total_layers", 32))

                # Try to extract test layer count from notes if not provided
                if test_layer_count is None:
                    # Parse from notes string like "Matched across size difference (32 vs 80 layers)"
                    import re
                    notes_match = re.search(r'\((\d+) vs (\d+) layers\)', identification.notes or '')
                    if notes_match:
                        test_layer_count = int(notes_match.group(2))
                    else:
                        test_layer_count = 80  # Default fallback

                # If we have actual variance measurements, use them to find high-variance layers
                if variance_profile and len(variance_profile) > 0:
                    # Find layers with highest variance from light probe
                    variance_with_layer = [(i, v) for i, v in enumerate(variance_profile)]
                    # Sort by variance descending
                    variance_with_layer.sort(key=lambda x: x[1], reverse=True)

                    # Take top N highest variance layers (matching number of restriction sites)
                    num_sites = len(restriction_sites) if restriction_sites else 7
                    high_variance_layers = [layer for layer, _ in variance_with_layer[:num_sites]]

                    # Also get scaled positions from reference for comparison
                    reference_scaled = []
                    if restriction_sites:
                        for site in restriction_sites:
                            if isinstance(site, dict) and "layer" in site:
                                ref_layer = site["layer"]
                                scaled_layer = int(round((ref_layer / ref_layer_count) * test_layer_count))
                                scaled_layer = min(scaled_layer, test_layer_count - 1)
                                reference_scaled.append(scaled_layer)
                            elif isinstance(site, int):
                                scaled_layer = int(round((site / ref_layer_count) * test_layer_count))
                                scaled_layer = min(scaled_layer, test_layer_count - 1)
                                reference_scaled.append(scaled_layer)

                    # Combine high-variance layers with reference-predicted layers
                    # Prioritize layers that appear in both
                    combined_layers = set(high_variance_layers) | set(reference_scaled)

                    # Sort combined layers by variance (prioritize high variance)
                    focus_layers = sorted(combined_layers,
                                         key=lambda l: variance_profile[l] if l < len(variance_profile) else 0,
                                         reverse=True)[:15]  # Take top 15 for focused analysis

                    logger.info(f"[STRATEGY] Selected {len(focus_layers)} high-variance layers from light probe")
                    logger.info(f"[STRATEGY] High-variance layers: {high_variance_layers[:5]}")
                    logger.info(f"[STRATEGY] Reference-predicted layers: {reference_scaled[:5]}")
                    logger.info(f"[STRATEGY] Final selected layers: {sorted(focus_layers)[:15]}")

                elif restriction_sites:
                    # Fallback to just scaling if no variance data available
                    for site in restriction_sites:
                        if isinstance(site, dict) and "layer" in site:
                            ref_layer = site["layer"]
                            # Scale layer position proportionally
                            scaled_layer = int(round((ref_layer / ref_layer_count) * test_layer_count))
                            # Ensure we don't exceed model boundaries
                            scaled_layer = min(scaled_layer, test_layer_count - 1)
                            focus_layers.append(scaled_layer)
                        elif isinstance(site, int):
                            # Scale integer layer position
                            scaled_layer = int(round((site / ref_layer_count) * test_layer_count))
                            scaled_layer = min(scaled_layer, test_layer_count - 1)
                            focus_layers.append(scaled_layer)

                # Always include first and last layers for boundary behavior
                if 0 not in focus_layers:
                    focus_layers.insert(0, 0)
                if (test_layer_count - 1) not in focus_layers:
                    focus_layers.append(test_layer_count - 1)

                # Remove duplicates and sort
                focus_layers = sorted(list(set(focus_layers)))

                return {
                    "strategy": "targeted",
                    "reference_model": identification.reference_model,
                    "candidate_layers": focus_layers[:15],  # Candidate positions to probe for actual sites
                    "baseline_measurements": restriction_sites,  # Reference measurements for comparison
                    "skip_layers": reference_fp.get("stable_layers", []),
                    "cassettes": reference_fp.get("recommended_cassettes", ["syntactic", "semantic"]),
                    "expected_patterns": reference_fp.get("behavioral_patterns", {}),
                    "challenges": reference_fp.get("challenges_processed", 10),
                    "notes": f"Using baseline from {identification.reference_model} (scaled from {ref_layer_count} to {test_layer_count} layers)"
                }
            else:
                # Family identified but no reference available
                return {
                    "strategy": "standard",
                    "challenges": 15,
                    "notes": f"Family {identification.identified_family} identified but no reference available"
                }

        elif identification.method == "needs_light_probe":
            # Do LIGHT probe first (not deep analysis)
            return {
                "strategy": "light_probe",
                "challenges": 5,  # Just 5 quick probes
                "sample_layers": "adaptive",  # Sample based on layer count
                "notes": "Light probe for family identification (enables 15-20x speedup)"
            }

        elif identification.method == "below_threshold":
            # Low confidence, need more probing
            return {
                "strategy": "exploratory",
                "challenges": 20,
                "notes": f"Low confidence match ({identification.confidence:.1%}), exploratory analysis needed"
            }

        else:
            # Default fallback
            return {
                "strategy": "exploratory",
                "challenges": 20,
                "notes": "Full exploratory analysis"
            }

    def _normalize_variance_profile(self, variance_profile):
        """
        Normalize variance profile to [0, 1] range for cross-model comparison.

        This is critical for comparing models of different scales (e.g., GPT2 vs Llama).
        Different models have vastly different variance magnitudes, so normalization
        ensures we're comparing patterns/shapes rather than absolute values.

        Args:
            variance_profile: List or array of variance values

        Returns:
            Normalized variance profile in [0, 1] range
        """
        if not variance_profile or len(variance_profile) == 0:
            return []

        # Convert to numpy array for easier manipulation
        profile = np.array(variance_profile)

        # Handle edge case where all values are the same
        min_val = profile.min()
        max_val = profile.max()

        if max_val - min_val < 1e-10:  # All values essentially the same
            # Return array of 0.5 (middle of range) to indicate no variance
            return np.full_like(profile, 0.5).tolist()

        # Standard min-max normalization to [0, 1]
        normalized = (profile - min_val) / (max_val - min_val)

        return normalized.tolist()

    def _interpolate_profile(self, positions, values, common_positions):
        """
        Helper to interpolate a profile to common positions.

        Args:
            positions: Original positions (0 to 1)
            values: Original values
            common_positions: Target positions to interpolate to

        Returns:
            Interpolated values at common positions
        """
        if len(positions) < 2 or len(values) < 2:
            # Not enough points for interpolation
            return np.zeros_like(common_positions)

        # Ensure arrays are properly sized
        positions = np.array(positions[:len(values)])
        values = np.array(values[:len(positions)])

        # Create interpolator
        interp_func = interpolate.interp1d(
            positions,
            values,
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )

        return interp_func(common_positions)

    def add_to_active_library(self, fingerprint_data: Dict, model_info: Dict):
        """Add a new fingerprint to the active library."""
        from datetime import datetime

        # Generate ID
        fp_id = f"{model_info['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Add to active library
        if "fingerprints" not in self.active_library:
            self.active_library["fingerprints"] = {}

        self.active_library["fingerprints"][fp_id] = {
            **fingerprint_data,
            **model_info,
            "timestamp": datetime.now().isoformat()
        }

        # Update metadata
        if "metadata" not in self.active_library:
            self.active_library["metadata"] = {}
        self.active_library["metadata"]["last_updated"] = datetime.now().isoformat()
        self.active_library["metadata"]["num_fingerprints"] = len(self.active_library["fingerprints"])

        # Save
        self._save_library(self.active_library, self.active_path)
        logger.info(f"Added fingerprint {fp_id} to active library")

    def _save_library(self, library: Dict, path: Path):
        """Save a library to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(library, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save library to {path}: {e}")


def create_dual_library():
    """Create a dual library system instance with improved cross-size matching."""
    return DualLibrarySystem()


def identify_and_strategize(model_path: str) -> Tuple[ModelIdentification, Dict]:
    """
    Identify a model and get its testing strategy.

    Returns:
        (identification, strategy)
    """
    library = create_dual_library()
    identification = library.identify_model(model_path)
    strategy = library.get_testing_strategy(identification)

    return identification, strategy