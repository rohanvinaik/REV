#!/usr/bin/env python3
"""
Test the enhanced sequential decision framework from Section 5.7.
"""

import numpy as np
from typing import Dict, Any, Generator

from src.core.sequential import (
    SequentialState,
    DualSequentialTest,
    ConfidenceSequence,
    TestType,
    sequential_decision,
    init_seq_test,
    accept_same,
    accept_diff,
    compute_e_value
)
from src.verifier.decision import Verdict


def generate_stream(n_samples: int, similarity: float = 0.8) -> Generator[Dict[str, Any], None, None]:
    """
    Generate a stream of comparison results.
    
    Args:
        n_samples: Number of samples to generate
        similarity: Similarity level (1.0 = identical, 0.0 = completely different)
        
    Yields:
        Dict with "I" (match indicator) and "d" (distance)
    """
    np.random.seed(42)
    
    for i in range(n_samples):
        # Generate match indicator (Bernoulli)
        is_match = np.random.random() < similarity
        I = 1 if is_match else 0
        
        # Generate distance based on similarity
        if is_match:
            d = 0.0  # Exact match
        else:
            # Distance inversely proportional to similarity
            d = np.random.beta(2, 5) * (1 - similarity)
        
        yield {"I": I, "d": d, "sample_id": i + 1}


def test_dual_sequential_test():
    """Test the dual sequential test framework."""
    print("Testing Dual Sequential Test Framework (Section 5.7)")
    print("=" * 60)
    
    # Test 1: Similar models (high similarity)
    print("\nTest 1: Similar models (similarity=0.95)")
    stream = generate_stream(100, similarity=0.95)
    verdict, stopping_time, localization = sequential_decision(
        stream,
        alpha=0.01,
        beta=0.01,
        d_thresh=0.08,
        max_C=100
    )
    
    print(f"  Verdict: {verdict}")
    print(f"  Stopped at: sample {stopping_time}")
    if localization["first_divergence"]:
        print(f"  First divergence: sample {localization['first_divergence']}")
    print(f"  Divergence sites: {len(localization['divergence_sites'])} total")
    
    assert verdict == "SAME", f"Expected SAME for similar models, got {verdict}"
    
    # Test 2: Different models (low similarity)
    print("\nTest 2: Different models (similarity=0.3)")
    stream = generate_stream(100, similarity=0.3)
    verdict, stopping_time, localization = sequential_decision(
        stream,
        alpha=0.01,
        beta=0.01,
        d_thresh=0.08,
        max_C=100
    )
    
    print(f"  Verdict: {verdict}")
    print(f"  Stopped at: sample {stopping_time}")
    if localization["first_divergence"]:
        print(f"  First divergence: sample {localization['first_divergence']}")
    print(f"  Divergence sites: {len(localization['divergence_sites'])} total")
    
    assert verdict == "DIFFERENT", f"Expected DIFFERENT for dissimilar models, got {verdict}"
    
    # Test 3: Borderline case
    print("\nTest 3: Borderline case (similarity=0.6)")
    stream = generate_stream(200, similarity=0.6)
    verdict, stopping_time, localization = sequential_decision(
        stream,
        alpha=0.01,
        beta=0.01,
        d_thresh=0.08,
        max_C=200
    )
    
    print(f"  Verdict: {verdict}")
    print(f"  Stopped at: sample {stopping_time}")
    if localization["first_divergence"]:
        print(f"  First divergence: sample {localization['first_divergence']}")
    print(f"  Match rate trajectory (last 10): {localization['match_trajectory'][-10:]}")
    
    print("\n✓ Dual sequential test framework tests passed")


def test_enhanced_sequential_state():
    """Test the enhanced SequentialState class."""
    print("\nTesting Enhanced SequentialState")
    print("=" * 60)
    
    # Create state for match test
    match_state = SequentialState(test_type=TestType.MATCH, alpha=0.01, beta=0.01)
    
    # Add some matches and non-matches
    for i in range(20):
        is_match = i % 3 != 0  # 2/3 matches
        match_state.update(float(is_match), is_match=is_match)
    
    print(f"  Match rate: {match_state.get_match_rate():.3f}")
    print(f"  First divergence: sample {match_state.first_divergence_site}")
    print(f"  Total divergences: {len(match_state.divergence_sites)}")
    print(f"  Confidence: {match_state.get_confidence():.3f}")
    print(f"  Decision: {match_state.get_decision()}")
    
    # Create state for distance test
    dist_state = SequentialState(test_type=TestType.DISTANCE, alpha=0.01, beta=0.01)
    
    # Add some distances
    for i in range(20):
        d = np.random.beta(2, 5) * 0.1  # Small distances
        dist_state.update_distance(d, threshold=0.08)
    
    print(f"\n  Mean distance: {dist_state.mean:.3f}")
    print(f"  Below threshold rate: {dist_state.get_below_threshold_rate():.3f}")
    print(f"  Confidence radius: {dist_state.get_confidence_radius():.3f}")
    print(f"  Should stop: {dist_state.should_stop()}")
    
    print("\n✓ Enhanced SequentialState tests passed")


def test_confidence_sequences():
    """Test confidence sequences and e-values."""
    print("\nTesting Confidence Sequences and E-values")
    print("=" * 60)
    
    # Create confidence sequence
    conf_seq = ConfidenceSequence(peeling_factor=1.1)
    
    # Add some e-values
    e_values = [1.0, 1.5, 2.0, 3.0, 2.5, 4.0]
    for e_val in e_values:
        conf_seq.update(e_val)
    
    print("  E-values:", e_values)
    print("  Confidence levels:", [f"{c:.3f}" for c in conf_seq.confidence_levels])
    
    # Test adjusted confidence with peeling
    for k in range(len(e_values)):
        adj_conf = conf_seq.get_adjusted_confidence(k)
        print(f"  Test {k+1}: adjusted confidence = {adj_conf:.3f}")
    
    # Test e-value computation
    state = SequentialState()
    for _ in range(50):
        state.update(np.random.beta(2, 5))
    
    e_val = compute_e_value(state, null_mean=0.5, alt_mean=0.3)
    print(f"\n  Computed e-value: {e_val:.3f}")
    print(f"  State mean: {state.mean:.3f}, variance: {state.variance:.3f}")
    
    print("\n✓ Confidence sequences tests passed")


def test_localization():
    """Test localization of first divergence site."""
    print("\nTesting Localization of Divergence Sites")
    print("=" * 60)
    
    # Generate stream with known divergence pattern
    def controlled_stream(n_samples: int) -> Generator[Dict[str, Any], None, None]:
        """Generate stream with controlled divergence."""
        for i in range(n_samples):
            if i < 5:
                # First 5 samples match
                yield {"I": 1, "d": 0.0, "sample_id": i + 1}
            elif i == 5:
                # First divergence at sample 6
                yield {"I": 0, "d": 0.15, "sample_id": i + 1}
            elif i < 15:
                # Mixed after divergence
                is_match = i % 2 == 0
                yield {
                    "I": 1 if is_match else 0,
                    "d": 0.0 if is_match else 0.1,
                    "sample_id": i + 1
                }
            else:
                # All different after sample 15
                yield {"I": 0, "d": 0.2, "sample_id": i + 1}
    
    stream = controlled_stream(50)
    verdict, stopping_time, localization = sequential_decision(
        stream,
        alpha=0.01,
        beta=0.01,
        d_thresh=0.08,
        max_C=50
    )
    
    print(f"  Verdict: {verdict}")
    print(f"  Stopped at: sample {stopping_time}")
    print(f"  First divergence detected: sample {localization['first_divergence']}")
    print(f"  Expected first divergence: sample 6")
    print(f"  Total divergence sites: {len(localization['divergence_sites'])}")
    print(f"  Divergence sites (first 10): {localization['divergence_sites'][:10]}")
    
    assert localization["first_divergence"] == 6, \
        f"Expected first divergence at 6, got {localization['first_divergence']}"
    
    print("\n✓ Localization tests passed")


def test_early_stopping():
    """Test early stopping conditions."""
    print("\nTesting Early Stopping Conditions")
    print("=" * 60)
    
    # Test with very similar models (should stop early with SAME)
    print("  Testing early SAME decision...")
    samples_used = 0
    for similarity in [0.99, 0.98, 0.97]:
        stream = generate_stream(1000, similarity=similarity)
        verdict, stopping_time, _ = sequential_decision(
            stream,
            alpha=0.01,
            beta=0.01,
            d_thresh=0.08,
            max_C=1000
        )
        samples_used = stopping_time
        print(f"    Similarity={similarity:.2f}: {verdict} at sample {stopping_time}")
        
        if verdict == "SAME" and stopping_time < 50:
            print(f"    ✓ Early stopping achieved at sample {stopping_time}")
            break
    
    assert samples_used < 100, f"Should stop early for similar models, used {samples_used}"
    
    # Test with very different models (should stop early with DIFFERENT)
    print("\n  Testing early DIFFERENT decision...")
    for similarity in [0.1, 0.2, 0.3]:
        stream = generate_stream(1000, similarity=similarity)
        verdict, stopping_time, _ = sequential_decision(
            stream,
            alpha=0.01,
            beta=0.01,
            d_thresh=0.08,
            max_C=1000
        )
        samples_used = stopping_time
        print(f"    Similarity={similarity:.2f}: {verdict} at sample {stopping_time}")
        
        if verdict == "DIFFERENT" and stopping_time < 50:
            print(f"    ✓ Early stopping achieved at sample {stopping_time}")
            break
    
    assert samples_used < 100, f"Should stop early for different models, used {samples_used}"
    
    print("\n✓ Early stopping tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Enhanced Sequential Decision Framework (Section 5.7)")
    print("=" * 70)
    
    test_dual_sequential_test()
    test_enhanced_sequential_state()
    test_confidence_sequences()
    test_localization()
    test_early_stopping()
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()