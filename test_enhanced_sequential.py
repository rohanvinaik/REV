#!/usr/bin/env python3
"""
Comprehensive tests for enhanced sequential testing framework.
Tests edge cases, statistical properties, and advanced features.
"""

import numpy as np
import math
from typing import Dict, Any, List, Generator
import pytest

from src.core.sequential import (
    SequentialState,
    DualSequentialTest,
    HybridSequentialTest,
    TestType,
    Verdict,
    ConfidenceSequence,
    PowerAnalysis,
    sequential_decision,
    compute_e_value,
    create_sequential_tester,
    analyze_sequential_results
)


def generate_test_stream(
    n_samples: int,
    match_prob: float = 0.8,
    mean_distance: float = 0.05,
    std_distance: float = 0.02,
    seed: int = 42
) -> Generator[Dict[str, Any], None, None]:
    """Generate test stream with controlled properties."""
    np.random.seed(seed)
    
    for i in range(n_samples):
        # Generate match indicator
        is_match = np.random.random() < match_prob
        I = 1.0 if is_match else 0.0
        
        # Generate distance (correlated with match)
        if is_match:
            d = np.random.normal(mean_distance * 0.5, std_distance)
        else:
            d = np.random.normal(mean_distance * 2, std_distance * 2)
        
        d = max(0, d)  # Ensure non-negative
        
        yield {"I": I, "d": d, "sample_id": i + 1}


class TestSequentialState:
    """Test SequentialState class with various edge cases."""
    
    def test_initialization(self):
        """Test proper initialization of sequential state."""
        state = SequentialState(TestType.MATCH, alpha=0.05, beta=0.10)
        
        assert state.n == 0
        assert state.mean == 0.0
        assert state.variance == 0.0
        assert state.test_type == TestType.MATCH
        assert state.alpha == 0.05
        assert state.beta == 0.10
        assert state.first_divergence_site is None
        
    def test_single_observation_update(self):
        """Test update with single observation."""
        state = SequentialState(TestType.MATCH)
        state.update(1.0, is_match=True)
        
        assert state.n == 1
        assert state.mean == 1.0
        assert state.n_matches == 1
        assert state.n_mismatches == 0
        assert len(state.e_values) == 1
        
    def test_variance_computation(self):
        """Test variance computation with multiple observations."""
        state = SequentialState(TestType.DISTANCE)
        values = [0.1, 0.2, 0.15, 0.25, 0.18]
        
        for v in values:
            state.update(v)
        
        # Check mean and variance
        expected_mean = np.mean(values)
        expected_var = np.var(values, ddof=1)
        
        assert abs(state.mean - expected_mean) < 1e-10
        assert abs(state.variance - expected_var) < 1e-10
        
    def test_e_value_computation(self):
        """Test e-value computation for different scenarios."""
        # Test for MATCH type
        state_match = SequentialState(TestType.MATCH)
        for _ in range(10):
            state_match.update(1.0, is_match=True)
        
        # All matches should give high e-value
        assert state_match.e_value_product > 1.0
        
        # Test for DISTANCE type
        state_dist = SequentialState(TestType.DISTANCE)
        state_dist.current_threshold = 0.1
        for _ in range(10):
            state_dist.update(0.05)  # Below threshold
        
        # Low distances should give high e-value
        assert state_dist.e_value_product > 1.0
        
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        state = SequentialState(TestType.MATCH, alpha=0.05)
        
        # Add observations
        for i in range(100):
            state.update(1.0 if i % 4 != 0 else 0.0)  # 75% match rate
        
        ci_lower, ci_upper = state.get_confidence_interval()
        
        # Check CI contains true mean
        assert ci_lower <= state.mean <= ci_upper
        # Check CI is reasonable width
        assert ci_upper - ci_lower < 0.2
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold adjustment."""
        state = SequentialState(
            TestType.DISTANCE,
            adaptive_threshold=True
        )
        
        # Generate observations with changing distribution
        for i in range(50):
            if i < 25:
                state.update(0.1 + np.random.normal(0, 0.01))
            else:
                state.update(0.2 + np.random.normal(0, 0.01))
        
        # Threshold should adapt
        assert len(state.threshold_history) > 0
        # Later threshold should be higher
        if len(state.threshold_history) > 10:
            assert state.threshold_history[-1] > state.threshold_history[0]
            
    def test_stopping_boundaries(self):
        """Test SPRT stopping boundaries."""
        state = SequentialState(TestType.MATCH, alpha=0.01, beta=0.01)
        
        # Strong evidence for SAME
        for _ in range(50):
            state.update(1.0, is_match=True)
        
        assert state.should_stop()
        assert state.get_decision() == Verdict.SAME
        
        # Reset and test DIFFERENT
        state = SequentialState(TestType.MATCH, alpha=0.01, beta=0.01)
        for _ in range(50):
            state.update(0.0, is_match=False)
        
        assert state.should_stop()
        assert state.get_decision() == Verdict.DIFFERENT
        
    def test_localization_tracking(self):
        """Test divergence localization."""
        state = SequentialState(TestType.MATCH)
        
        # Add matches then divergences
        for i in range(10):
            state.update(1.0, is_match=True)
        
        # First divergence at position 11
        state.update(0.0, is_match=False)
        assert state.first_divergence_site == 11
        
        # More divergences
        for i in range(5):
            state.update(0.0, is_match=False)
        
        assert len(state.divergence_sites) == 6
        assert state.divergence_sites[0] == 11
        
    def test_edge_case_empty_state(self):
        """Test behavior with empty state."""
        state = SequentialState(TestType.MATCH)
        
        # Should handle empty state gracefully
        assert state.get_confidence() == 0.0
        assert state.get_match_rate() == 0.5
        assert state.get_confidence_interval() == (0.0, 1.0)
        assert not state.should_stop()
        
    def test_edge_case_single_value_variance(self):
        """Test variance with single value."""
        state = SequentialState(TestType.DISTANCE)
        state.update(0.5)
        
        # Variance should be 0 with single observation
        assert state.variance == 0.0
        assert state.std == 0.0
        
    def test_extreme_values(self):
        """Test handling of extreme values."""
        state = SequentialState(TestType.DISTANCE)
        
        # Test very large values
        state.update(1e6)
        assert not math.isnan(state.mean)
        assert not math.isinf(state.mean)
        
        # Test very small values
        state.update(1e-10)
        assert state.mean > 0
        
    def test_history_management(self):
        """Test history buffer management."""
        state = SequentialState(TestType.MATCH, history_size=10)
        
        # Add more than history size
        for i in range(20):
            state.update(float(i))
        
        # History should be limited
        assert len(state.history) == 10
        # Should contain most recent values
        assert list(state.history) == list(range(10, 20))


class TestConfidenceSequence:
    """Test ConfidenceSequence class."""
    
    def test_e_value_update(self):
        """Test e-value updates."""
        conf_seq = ConfidenceSequence()
        
        e_values = [1.5, 2.0, 1.8, 2.5]
        for e in e_values:
            conf_seq.update(e)
        
        assert len(conf_seq.e_values) == 4
        assert len(conf_seq.confidence_levels) == 4
        
        # Confidence should increase with higher e-values
        assert conf_seq.confidence_levels[-1] > 0.5
        
    def test_peeling_adjustment(self):
        """Test peeling for multiple testing."""
        conf_seq = ConfidenceSequence(peeling_factor=1.2)
        
        # Add some e-values
        for _ in range(5):
            conf_seq.update(2.0)
        
        # Test adjusted confidence decreases with k
        conf_0 = conf_seq.get_adjusted_confidence(0)
        conf_2 = conf_seq.get_adjusted_confidence(2)
        conf_4 = conf_seq.get_adjusted_confidence(4)
        
        assert conf_0 > conf_2 > conf_4
        
    def test_confidence_radius(self):
        """Test confidence radius computation."""
        conf_seq = ConfidenceSequence()
        
        # Low e-value should give large radius
        conf_seq.update(1.1)
        radius_1 = conf_seq.get_current_radius()
        
        # High e-value should give smaller radius
        conf_seq.update(10.0)
        radius_2 = conf_seq.get_current_radius()
        
        assert radius_2 < radius_1


class TestPowerAnalysis:
    """Test PowerAnalysis utilities."""
    
    def test_expected_sample_size(self):
        """Test expected sample size computation."""
        power = PowerAnalysis()
        
        # Larger effect size should need fewer samples
        n_small = power.compute_expected_sample_size(0.05, 0.20, 0.1, TestType.MATCH)
        n_large = power.compute_expected_sample_size(0.05, 0.20, 0.3, TestType.MATCH)
        
        assert n_large < n_small
        
        # Stricter alpha should need more samples
        n_loose = power.compute_expected_sample_size(0.10, 0.20, 0.2, TestType.MATCH)
        n_strict = power.compute_expected_sample_size(0.01, 0.20, 0.2, TestType.MATCH)
        
        assert n_strict > n_loose
        
    def test_power_computation(self):
        """Test statistical power computation."""
        power = PowerAnalysis()
        
        # Power should increase with sample size
        power_10 = power.compute_power(10, 0.05, 0.5, TestType.MATCH)
        power_100 = power.compute_power(100, 0.05, 0.5, TestType.MATCH)
        
        assert power_100 > power_10
        
        # Power should be between 0 and 1
        assert 0 <= power_10 <= 1
        assert 0 <= power_100 <= 1
        
    def test_edge_cases(self):
        """Test edge cases in power analysis."""
        power = PowerAnalysis()
        
        # Zero samples should give zero power
        assert power.compute_power(0, 0.05, 0.5) == 0.0
        
        # Zero effect size should need maximum samples
        n = power.compute_expected_sample_size(0.05, 0.20, 0.0, TestType.MATCH)
        assert n == 10000  # Maximum


class TestDualSequentialTest:
    """Test DualSequentialTest class."""
    
    def test_dual_update(self):
        """Test updating both tests simultaneously."""
        S_match = SequentialState(TestType.MATCH)
        S_dist = SequentialState(TestType.DISTANCE)
        dual = DualSequentialTest(S_match, S_dist)
        
        dual.update(1.0, 0.05, 0.1)
        
        assert dual.S_match.n == 1
        assert dual.S_dist.n == 1
        assert dual.S_match.n_matches == 1
        assert dual.S_dist.below_threshold_count == 1
        
    def test_combined_verdict(self):
        """Test verdict combination logic."""
        S_match = SequentialState(TestType.MATCH)
        S_dist = SequentialState(TestType.DISTANCE)
        dual = DualSequentialTest(S_match, S_dist)
        
        # Both agree on SAME
        for _ in range(50):
            dual.update(1.0, 0.01, 0.1)
        
        # Force decision
        dual.S_match.log_likelihood_ratio = 10.0
        dual.S_dist.log_likelihood_ratio = 10.0
        dual._update_combined_verdict()
        
        assert dual.combined_verdict == Verdict.SAME
        
    def test_conservative_combination(self):
        """Test conservative verdict combination."""
        S_match = SequentialState(TestType.MATCH)
        S_dist = SequentialState(TestType.DISTANCE)
        dual = DualSequentialTest(S_match, S_dist)
        
        # Conflicting evidence
        S_match.log_likelihood_ratio = 10.0  # Says SAME
        S_dist.log_likelihood_ratio = -10.0  # Says DIFFERENT
        
        dual._update_combined_verdict()
        
        # Should be conservative
        assert dual.combined_verdict == Verdict.DIFFERENT


class TestHybridSequentialTest:
    """Test HybridSequentialTest class."""
    
    def test_multi_hypothesis(self):
        """Test multi-hypothesis testing."""
        hybrid = HybridSequentialTest(
            alpha=0.05,
            beta=0.10,
            enable_multi_hypothesis=True
        )
        
        # Add uncertain evidence
        for i in range(20):
            hybrid.update(
                match_indicator=float(i % 2),  # Alternating
                distance=0.1,
                threshold=0.1
            )
        
        verdict = hybrid._get_combined_verdict()
        
        # Should be uncertain with mixed evidence
        assert verdict in [Verdict.UNCERTAIN, Verdict.UNDECIDED]
        
    def test_power_analysis_integration(self):
        """Test integrated power analysis."""
        hybrid = HybridSequentialTest(
            alpha=0.05,
            beta=0.20,
            power_analysis=True
        )
        
        # Add observations
        for _ in range(50):
            hybrid.update(0.8, 0.05, 0.1)
        
        analysis = hybrid.get_analysis()
        
        assert "power_analysis" in analysis
        assert "expected_n" in analysis["power_analysis"]
        assert "current_power" in analysis["power_analysis"]
        
        # Power should be reasonable
        assert 0 < analysis["power_analysis"]["current_power"] <= 1
        
    def test_verdict_history(self):
        """Test verdict history tracking."""
        hybrid = HybridSequentialTest()
        
        # Generate changing verdicts
        for i in range(30):
            if i < 10:
                hybrid.update(0.0, 0.2, 0.1)  # DIFFERENT evidence
            else:
                hybrid.update(1.0, 0.01, 0.1)  # SAME evidence
        
        # Should have recorded verdict changes
        assert len(hybrid.verdict_history) > 1
        
        # Check history format
        for n, verdict in hybrid.verdict_history:
            assert isinstance(n, int)
            assert isinstance(verdict, Verdict)
            
    def test_reset_functionality(self):
        """Test reset method."""
        hybrid = HybridSequentialTest()
        
        # Add some data
        for _ in range(10):
            hybrid.update(1.0, 0.05, 0.1)
        
        assert hybrid.match_test.n == 10
        
        # Reset
        hybrid.reset()
        
        assert hybrid.match_test.n == 0
        assert len(hybrid.verdict_history) == 0
        assert len(hybrid.confidence_history) == 0


class TestSequentialDecision:
    """Test main sequential_decision function."""
    
    def test_same_verdict(self):
        """Test detection of SAME models."""
        stream = generate_test_stream(100, match_prob=0.95, mean_distance=0.02)
        verdict, stopping_time, localization = sequential_decision(
            stream,
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1
        )
        
        assert verdict == "SAME"
        assert stopping_time <= 100
        assert localization["match_rate"] > 0.8
        
    def test_different_verdict(self):
        """Test detection of DIFFERENT models."""
        stream = generate_test_stream(100, match_prob=0.2, mean_distance=0.2)
        verdict, stopping_time, localization = sequential_decision(
            stream,
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1
        )
        
        assert verdict == "DIFFERENT"
        assert stopping_time <= 100
        assert localization["match_rate"] < 0.5
        
    def test_early_stopping(self):
        """Test early stopping with strong evidence."""
        # Very strong SAME evidence
        def strong_same_stream():
            for i in range(1000):
                yield {"I": 1.0, "d": 0.001, "sample_id": i}
        
        verdict, stopping_time, _ = sequential_decision(
            strong_same_stream(),
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1
        )
        
        assert verdict == "SAME"
        assert stopping_time < 50  # Should stop very early
        
    def test_max_samples_limit(self):
        """Test maximum samples limit."""
        # Ambiguous evidence
        def ambiguous_stream():
            for i in range(100):
                yield {"I": float(i % 2), "d": 0.1, "sample_id": i}
        
        verdict, stopping_time, _ = sequential_decision(
            ambiguous_stream(),
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1,
            max_C=50
        )
        
        assert stopping_time == 50
        assert verdict == "UNDECIDED"
        
    def test_localization_info(self):
        """Test localization information."""
        # Stream with known divergence
        def diverging_stream():
            for i in range(100):
                if i < 20:
                    yield {"I": 1.0, "d": 0.01, "sample_id": i}
                else:
                    yield {"I": 0.0, "d": 0.2, "sample_id": i}
        
        verdict, _, localization = sequential_decision(
            diverging_stream(),
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1
        )
        
        assert localization["first_divergence"] is not None
        assert len(localization["divergence_sites"]) > 0
        assert "match_trajectory" in localization
        assert "confidence_match" in localization
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold in sequential decision."""
        stream = generate_test_stream(100, mean_distance=0.1, std_distance=0.05)
        
        verdict, _, localization, dual_test = sequential_decision(
            stream,
            alpha=0.01,
            beta=0.01,
            d_thresh=0.1,
            adaptive=True,
            return_full_state=True
        )
        
        # Check adaptive threshold was used
        assert len(dual_test.S_dist.threshold_history) > 0
        
    def test_return_full_state(self):
        """Test returning full test state."""
        stream = generate_test_stream(50)
        
        result_with_state = sequential_decision(
            stream,
            return_full_state=True
        )
        
        assert len(result_with_state) == 4
        verdict, stopping_time, localization, dual_test = result_with_state
        
        assert isinstance(dual_test, DualSequentialTest)
        assert dual_test.S_match.n == stopping_time


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sequential_tester(self):
        """Test factory function."""
        # Single test
        single = create_sequential_tester("single", alpha=0.05)
        assert isinstance(single, SequentialState)
        
        # Dual test
        dual = create_sequential_tester("dual", alpha=0.05, beta=0.10)
        assert isinstance(dual, DualSequentialTest)
        
        # Hybrid test
        hybrid = create_sequential_tester(
            "hybrid",
            multi_hypothesis=True,
            power_analysis=True
        )
        assert isinstance(hybrid, HybridSequentialTest)
        
        # Invalid type
        with pytest.raises(ValueError):
            create_sequential_tester("invalid")
            
    def test_analyze_sequential_results(self):
        """Test results analysis function."""
        # Single state
        state = SequentialState(TestType.MATCH)
        for _ in range(10):
            state.update(1.0)
        
        analysis = analyze_sequential_results(state)
        assert "n" in analysis
        assert "mean" in analysis
        assert "verdict" in analysis
        
        # Dual test
        dual = DualSequentialTest(
            SequentialState(TestType.MATCH),
            SequentialState(TestType.DISTANCE)
        )
        
        analysis = analyze_sequential_results(dual)
        assert "verdict" in analysis
        assert "match_analysis" in analysis
        assert "distance_analysis" in analysis
        
        # Hybrid test
        hybrid = HybridSequentialTest()
        analysis = analyze_sequential_results(hybrid)
        assert "verdict" in analysis
        assert "match_summary" in analysis
        assert "distance_summary" in analysis
        
    def test_compute_e_value_function(self):
        """Test standalone e-value computation."""
        state = SequentialState(TestType.MATCH)
        
        # Add some matches
        for _ in range(10):
            state.update(1.0, is_match=True)
        
        e_val = compute_e_value(state, null_mean=0.5)
        
        # Should be > 1 for evidence against null
        assert e_val > 1.0
        
        # Test with specified alternative
        e_val_alt = compute_e_value(state, null_mean=0.5, alt_mean=0.9)
        assert e_val_alt > 0


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        state = SequentialState(TestType.DISTANCE)
        
        # Very small variance
        for _ in range(100):
            state.update(1.0)  # All same value
        
        # Should not crash or produce NaN
        assert not math.isnan(state.mean)
        assert state.variance == 0.0
        
        # LLR update should handle zero variance
        state._update_llr(1.0)
        assert not math.isnan(state.log_likelihood_ratio)
        
    def test_overflow_protection(self):
        """Test protection against overflow."""
        state = SequentialState(TestType.MATCH)
        
        # Many observations
        for _ in range(10000):
            state.update(1.0)
        
        # E-value product might overflow but should be handled
        assert not math.isinf(state.log_likelihood_ratio)
        
    def test_empty_stream(self):
        """Test handling of empty stream."""
        def empty_stream():
            return
            yield  # Never yields anything
        
        verdict, stopping_time, localization = sequential_decision(
            empty_stream(),
            max_C=10
        )
        
        assert verdict == "UNDECIDED"
        assert stopping_time == 1  # Minimum stopping time is 1
        
    def test_malformed_stream_data(self):
        """Test handling of malformed stream data."""
        def malformed_stream():
            yield {"I": 1.0}  # Missing "d"
            yield {"d": 0.1}  # Missing "I"
            yield {}  # Missing both
            yield {"I": 1.0, "d": 0.1}  # Valid
        
        # Should handle gracefully with defaults
        verdict, stopping_time, _ = sequential_decision(
            malformed_stream(),
            max_C=4
        )
        
        assert stopping_time == 4
        
    def test_concurrent_state_updates(self):
        """Test state consistency with rapid updates."""
        state = SequentialState(TestType.MATCH)
        
        # Rapid updates
        values = np.random.random(1000)
        for v in values:
            state.update(v)
        
        # Check consistency
        assert state.n == 1000
        assert abs(state.mean - np.mean(values)) < 1e-10
        
        # History should be maintained correctly
        assert len(state.history) <= state.history.maxlen


def run_all_tests():
    """Run all test classes."""
    import sys
    
    test_classes = [
        TestSequentialState(),
        TestConfidenceSequence(),
        TestPowerAnalysis(),
        TestDualSequentialTest(),
        TestHybridSequentialTest(),
        TestSequentialDecision(),
        TestUtilityFunctions(),
        TestEdgeCasesAndRobustness()
    ]
    
    print("=" * 70)
    print("Running Enhanced Sequential Testing Framework Tests")
    print("=" * 70)
    
    failed = 0
    passed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        
        # Get all test methods
        methods = [m for m in dir(test_class) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: Unexpected error: {e}")
                failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()