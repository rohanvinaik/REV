"""
Advanced matching enhancements for REV cross-size model comparison
Implements gradient-based interpolation, quality metrics, and ensemble matching
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class IdentificationResult:
    """Enhanced identification result with diagnostics"""
    family: Optional[str]
    confidence: float
    method: str
    diagnostics: Optional[Dict] = None
    recommendations: Optional[List[str]] = None


class MatchingCache:
    """Intelligent caching for expensive computations"""

    def __init__(self):
        self.interpolation_cache = {}  # Cache interpolated profiles
        self.similarity_cache = {}     # Cache computed similarities
        self.feature_cache = {}        # Cache extracted features
        self.max_cache_size = 100      # Prevent unbounded growth

    def get_or_compute_interpolation(self, profile: List[float], positions: List[float],
                                    cache_key: str, interpolation_func) -> np.ndarray:
        """Get cached interpolation or compute and cache"""
        if cache_key not in self.interpolation_cache:
            # Evict oldest if cache is full
            if len(self.interpolation_cache) >= self.max_cache_size:
                # Remove first (oldest) item
                self.interpolation_cache.pop(next(iter(self.interpolation_cache)))

            self.interpolation_cache[cache_key] = interpolation_func(profile, positions)

        return self.interpolation_cache[cache_key]

    def get_or_compute_similarity(self, test_data: Any, ref_data: Any,
                                 cache_key: str, similarity_func) -> float:
        """Get cached similarity or compute and cache"""
        if cache_key not in self.similarity_cache:
            if len(self.similarity_cache) >= self.max_cache_size:
                self.similarity_cache.pop(next(iter(self.similarity_cache)))

            self.similarity_cache[cache_key] = similarity_func(test_data, ref_data)

        return self.similarity_cache[cache_key]


def adaptive_interpolation(variance_profile: List[float], layers: List[int]) -> List[int]:
    """
    Smart interpolation that focuses on high-gradient regions
    Uses gradient analysis to identify regions needing interpolation
    """
    if len(variance_profile) < 3:
        return []  # Not enough data for gradient analysis

    # Calculate variance gradient
    gradients = np.gradient(variance_profile)

    # Find regions with high rate of change (top 25% gradients)
    threshold = np.percentile(np.abs(gradients), 75)

    high_gradient_regions = []
    for i in range(len(gradients) - 1):
        if abs(gradients[i]) > threshold:
            if i < len(layers) - 1:
                high_gradient_regions.append((layers[i], layers[i + 1]))

    # Only interpolate where it matters
    interpolation_points = []
    for start, end in high_gradient_regions:
        if end - start > 2:  # Worth interpolating if gap > 2 layers
            # Golden ratio search for smooth curves
            golden = 0.618
            mid1 = int(start + (end - start) * (1 - golden))
            mid2 = int(start + (end - start) * golden)

            # Add unique points only
            if mid1 not in interpolation_points and mid1 not in layers:
                interpolation_points.append(mid1)
            if mid2 not in interpolation_points and mid2 not in layers:
                interpolation_points.append(mid2)

    return sorted(interpolation_points)


def assess_reference_quality(reference: Dict) -> float:
    """
    Score reference library entries for reliability
    Returns quality score between 0 and 1
    """
    quality_score = 0.0

    # More challenges = better reference (30% weight)
    challenges = reference.get('challenges_processed', 0)
    quality_score += min(1.0, challenges / 400) * 0.3

    # More restriction sites = better behavioral coverage (20% weight)
    sites = reference.get('restriction_sites', [])
    quality_score += min(1.0, len(sites) / 10) * 0.2

    # Non-zero divergence values = real analysis (30% weight)
    real_divergences = 0
    for site in sites:
        if isinstance(site, dict):
            delta = abs(site.get('divergence_delta', 0))
            if delta > 0.001:
                real_divergences += 1

    if len(sites) > 0:
        quality_score += (real_divergences / len(sites)) * 0.3

    # Variance profile coverage (20% weight)
    variance_profile = reference.get('variance_profile', [])
    if variance_profile:
        variance_range = max(variance_profile) - min(variance_profile)
        # Typical good range is ~0.2
        quality_score += min(1.0, variance_range / 0.2) * 0.2

    return quality_score


def hierarchical_matching(fingerprint: Dict, references: List[Dict]) -> IdentificationResult:
    """
    Multi-level matching with fallbacks
    Tries multiple strategies in order of preference
    """
    from .dual_library_system import DualLibrarySystem

    # Create temporary library instance for matching
    library = DualLibrarySystem()

    # Level 1: Exact architectural match (same layer count, high similarity)
    for ref in references:
        ref_layers = ref.get('layer_count', ref.get('total_layers', 0))
        test_layers = fingerprint.get('layer_count', 0)

        if ref_layers == test_layers:
            # Use existing matching algorithm
            result = library.identify_from_behavioral_analysis(fingerprint)
            if result.confidence > 0.8:
                return IdentificationResult(
                    family=result.identified_family,
                    confidence=result.confidence,
                    method="exact_architecture_match"
                )

    # Level 2: Family-level patterns (relaxed threshold)
    best_family_match = None
    best_family_confidence = 0.0

    for ref in references:
        # Check for family patterns
        result = library.identify_from_behavioral_analysis(fingerprint)
        if result.confidence > 0.6 and result.confidence > best_family_confidence:
            best_family_match = result
            best_family_confidence = result.confidence

    if best_family_match:
        return IdentificationResult(
            family=best_family_match.identified_family,
            confidence=best_family_confidence,
            method="family_pattern_match"
        )

    # Level 3: Behavioral clustering (very relaxed)
    cluster_result = find_behavioral_cluster(fingerprint, references)
    if cluster_result and cluster_result.confidence > 0.4:
        return cluster_result

    # Level 4: Size-based heuristics
    size_match = find_size_based_match(fingerprint, references)

    return size_match or IdentificationResult(None, 0.0, "no_match")


def find_behavioral_cluster(fingerprint: Dict, references: List[Dict]) -> Optional[IdentificationResult]:
    """
    Find the best matching behavioral cluster
    Groups models by similar variance patterns
    """
    test_variance = fingerprint.get('variance_profile', [])
    if not test_variance:
        return None

    # Simple clustering based on variance statistics
    test_mean = np.mean(test_variance)
    test_std = np.std(test_variance)

    best_match = None
    best_distance = float('inf')
    best_family = None

    for ref in references:
        ref_variance = ref.get('variance_profile', [])
        if not ref_variance:
            continue

        ref_mean = np.mean(ref_variance)
        ref_std = np.std(ref_variance)

        # Euclidean distance in mean-std space
        distance = np.sqrt((test_mean - ref_mean)**2 + (test_std - ref_std)**2)

        if distance < best_distance:
            best_distance = distance
            best_family = ref.get('model_family', ref.get('family', 'unknown'))

    if best_distance < 0.1:  # Threshold for cluster membership
        confidence = 1.0 - (best_distance / 0.1) * 0.6  # Scale to 0.4-1.0
        return IdentificationResult(
            family=best_family,
            confidence=confidence,
            method="behavioral_clustering"
        )

    return None


def find_size_based_match(fingerprint: Dict, references: List[Dict]) -> Optional[IdentificationResult]:
    """
    Use size-based heuristics as last resort
    Maps common layer counts to families
    """
    test_layers = fingerprint.get('layer_count', 0)

    # Common size patterns
    size_patterns = {
        (6, 12): 'gpt',      # Small GPT models
        (24, 32): 'gpt',     # Medium GPT models
        (32, 40): 'llama',   # 7B-13B Llama
        (80, 80): 'llama',   # 70B Llama
        (32, 32): 'mistral', # 7B Mistral
        (6, 6): 'pythia',    # Small Pythia
    }

    for size_range, family in size_patterns.items():
        if size_range[0] <= test_layers <= size_range[1]:
            # Low confidence for size-only matching
            return IdentificationResult(
                family=family,
                confidence=0.35,
                method="size_heuristic",
                diagnostics={"matched_range": size_range}
            )

    return None


def ensemble_matching(fingerprint: Dict, references: List[Dict], cache: Optional[MatchingCache] = None) -> IdentificationResult:
    """
    Combine multiple matching strategies for robustness
    Uses weighted voting across different matchers
    """
    if cache is None:
        cache = MatchingCache()

    # Import matcher functions (would need to be implemented)
    # For now, we'll use the hierarchical matcher as primary

    matchers = [
        ('hierarchical', hierarchical_matching, 0.4),
        ('variance', variance_based_matcher, 0.3),
        ('pattern', pattern_based_matcher, 0.2),
        ('statistical', statistical_matcher, 0.1)
    ]

    results = []
    family_votes = {}

    for name, matcher, weight in matchers:
        try:
            if name == 'hierarchical':
                result = matcher(fingerprint, references)
            else:
                # Placeholder for other matchers
                result = IdentificationResult(None, 0.0, name)

            if result and result.family:
                if result.family not in family_votes:
                    family_votes[result.family] = 0.0
                family_votes[result.family] += result.confidence * weight

            results.append((name, result))
        except Exception as e:
            print(f"Matcher {name} failed: {e}")
            continue

    if family_votes:
        best_family = max(family_votes, key=family_votes.get)
        combined_confidence = min(0.95, family_votes[best_family])

        return IdentificationResult(
            family=best_family,
            confidence=combined_confidence,
            method="ensemble",
            diagnostics={"votes": family_votes, "matchers": results}
        )

    return IdentificationResult(None, 0.0, "no_consensus")


def generate_matching_diagnostic(test_fp: Dict, ref_fp: Dict, similarity_scores: Dict) -> Dict:
    """
    Generate detailed diagnostic for debugging matches
    Provides actionable recommendations
    """
    test_layers = test_fp.get('layer_count', 0)
    ref_layers = ref_fp.get('layer_count', ref_fp.get('total_layers', 0))

    diagnostic = {
        'timestamp': datetime.now().isoformat(),
        'test_model': {
            'layers': test_layers,
            'variance_mean': np.mean(test_fp.get('variance_profile', [0])),
            'variance_std': np.std(test_fp.get('variance_profile', [0])),
            'sites_count': len(test_fp.get('restriction_sites', []))
        },
        'reference_model': {
            'family': ref_fp.get('model_family', ref_fp.get('family', 'unknown')),
            'layers': ref_layers,
            'quality_score': assess_reference_quality(ref_fp),
            'challenges_processed': ref_fp.get('challenges_processed', 0)
        },
        'matching_details': {
            'size_ratio': min(test_layers, ref_layers) / max(test_layers, ref_layers) if test_layers and ref_layers else 0,
            'variance_correlation': similarity_scores.get('variance', 0),
            'site_overlap': similarity_scores.get('sites', 0),
            'pattern_match': similarity_scores.get('pattern', 0),
            'final_score': similarity_scores.get('final', 0)
        },
        'recommendations': []
    }

    # Add specific recommendations based on analysis

    # Size ratio recommendations
    if diagnostic['matching_details']['size_ratio'] < 0.3:
        diagnostic['recommendations'].append(
            f"Large size mismatch ({test_layers} vs {ref_layers} layers). "
            "Consider building size-specific reference for better matching"
        )

    # Reference quality recommendations
    if diagnostic['reference_model']['quality_score'] < 0.5:
        diagnostic['recommendations'].append(
            f"Reference quality is low ({diagnostic['reference_model']['quality_score']:.2f}). "
            "Rebuild with: python run_rev.py /path/to/model --build-reference --enable-prompt-orchestration"
        )

    # Challenges recommendations
    if diagnostic['reference_model']['challenges_processed'] < 250:
        diagnostic['recommendations'].append(
            f"Reference has only {diagnostic['reference_model']['challenges_processed']} challenges. "
            "Rebuild with --enable-prompt-orchestration for 250+ challenges"
        )

    # Correlation recommendations
    if diagnostic['matching_details']['variance_correlation'] < 0.3:
        diagnostic['recommendations'].append(
            "Low variance correlation suggests different model architectures. "
            "Verify model family or expand reference library"
        )

    # Sites recommendations
    if diagnostic['test_model']['sites_count'] < 3:
        diagnostic['recommendations'].append(
            "Too few restriction sites detected. "
            "Run deeper analysis with --enable-prompt-orchestration"
        )

    return diagnostic


# Placeholder functions for ensemble matcher
def variance_based_matcher(fingerprint: Dict, references: List[Dict]) -> IdentificationResult:
    """Placeholder for variance-based matching"""
    return IdentificationResult(None, 0.0, "variance_based")


def pattern_based_matcher(fingerprint: Dict, references: List[Dict]) -> IdentificationResult:
    """Placeholder for pattern-based matching"""
    return IdentificationResult(None, 0.0, "pattern_based")


def statistical_matcher(fingerprint: Dict, references: List[Dict]) -> IdentificationResult:
    """Placeholder for statistical matching"""
    return IdentificationResult(None, 0.0, "statistical")


def save_diagnostic_report(diagnostic: Dict, output_path: str = "diagnostic_report.json"):
    """Save diagnostic report to file"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(diagnostic, f, indent=2, default=str)

    print(f"Diagnostic report saved to: {output_path}")

    # Print recommendations to console
    if diagnostic.get('recommendations'):
        print("\n📋 Recommendations:")
        for i, rec in enumerate(diagnostic['recommendations'], 1):
            print(f"  {i}. {rec}")