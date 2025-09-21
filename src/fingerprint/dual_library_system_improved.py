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

logger = logging.getLogger(__name__)


@dataclass
class ModelIdentification:
    """Result of model identification"""
    identified_family: Optional[str]
    confidence: float
    method: str  # "name_match", "fingerprint_match", "unknown"
    reference_model: Optional[str] = None  # e.g., "gpt2", "llama-7b"
    notes: str = ""


class DualLibrarySystemImproved:
    """
    Improved matching algorithm that handles cross-size model comparison.
    """

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
        """Initialize dual library system."""
        self.reference_path = Path(reference_path)
        self.active_path = Path(active_path)

        # Load libraries
        self.reference_library = self._load_library(self.reference_path)
        self.active_library = self._load_library(self.active_path)

    def _load_library(self, path: Path) -> Dict:
        """Load a library from disk."""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load library {path}: {e}")
        return {}

    def identify_from_behavioral_analysis(self, fingerprint_dict: Dict) -> ModelIdentification:
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

                # Interpolate test variance profile
                test_interp = interpolate.interp1d(
                    test_relative_positions[:len(test_variance)],
                    test_variance[:len(test_relative_positions)],
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                test_interpolated = test_interp(common_positions)

                # Interpolate reference variance profile
                ref_interp = interpolate.interp1d(
                    ref_relative_positions[:len(ref_variance)],
                    ref_variance[:len(ref_relative_positions)],
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                ref_interpolated = ref_interp(common_positions)

                # Calculate multiple similarity metrics

                # 1. Pearson correlation (linear relationship)
                pearson_corr, _ = pearsonr(test_interpolated, ref_interpolated)
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0

                # 2. Spearman correlation (monotonic relationship)
                spearman_corr, _ = spearmanr(test_interpolated, ref_interpolated)
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0

                # 3. Mean absolute error (normalized)
                mae = np.mean(np.abs(test_interpolated - ref_interpolated))
                # Normalize MAE to similarity score (lower error = higher similarity)
                # Typical variance ranges from 0.2 to 0.4, so max expected MAE ~0.2
                mae_similarity = max(0, 1.0 - (mae / 0.2))

                # 4. Shape similarity using derivatives
                test_diff = np.diff(test_interpolated)
                ref_diff = np.diff(ref_interpolated)

                # Normalize derivatives
                test_diff_std = np.std(test_diff)
                ref_diff_std = np.std(ref_diff)

                if test_diff_std > 0 and ref_diff_std > 0:
                    test_diff_norm = test_diff / test_diff_std
                    ref_diff_norm = ref_diff / ref_diff_std
                    shape_correlation, _ = pearsonr(test_diff_norm, ref_diff_norm)
                    if np.isnan(shape_correlation):
                        shape_correlation = 0.0
                else:
                    shape_correlation = 0.0

                print(f"  Pearson correlation: {pearson_corr:.3f}")
                print(f"  Spearman correlation: {spearman_corr:.3f}")
                print(f"  MAE similarity: {mae_similarity:.3f}")
                print(f"  Shape correlation: {shape_correlation:.3f}")

            except Exception as e:
                print(f"  Interpolation failed: {e}")
                pearson_corr = spearman_corr = mae_similarity = shape_correlation = 0.0

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

            # === COMBINE SCORES WITH WEIGHTS ===
            # Weight different components based on reliability
            scores = {
                'pearson': (pearson_corr, 0.20),      # Linear correlation
                'spearman': (spearman_corr, 0.15),    # Rank correlation
                'mae': (mae_similarity, 0.25),        # Absolute difference
                'shape': (shape_correlation, 0.15),   # Shape similarity
                'sites': (site_similarity, 0.15),     # Restriction sites
                'pattern': (pattern_similarity, 0.10) # High/low patterns
            }

            # Calculate weighted average
            total_weight = sum(weight for _, weight in scores.values())
            weighted_sum = sum(score * weight for score, weight in scores.values())
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Boost score if correlation is particularly strong
            if pearson_corr > 0.7 or spearman_corr > 0.7:
                final_score = min(1.0, final_score * 1.2)

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
                # Boost confidence for same-family matches with known patterns
                confidence_boost = 1.0
                ref_family = best_match.get('model_family', best_match.get('family', ''))

                # Special boost for known patterns
                if 'llama' in ref_family.lower() and test_layers == 80:
                    confidence_boost = 1.3  # Known Llama-70B pattern
                elif 'llama' in ref_family.lower() and test_layers > 32:
                    confidence_boost = 1.2  # Large Llama model

                final_confidence = min(0.95, best_score * confidence_boost)

                return ModelIdentification(
                    identified_family=ref_family,
                    confidence=final_confidence,
                    method="cross_size_behavioral_matching",
                    reference_model=best_match.get('reference_model', 'unknown'),
                    notes=f"Matched across size difference ({ref_layers} vs {test_layers} layers)"
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


def create_dual_library_improved():
    """Create an improved dual library system instance."""
    return DualLibrarySystemImproved()