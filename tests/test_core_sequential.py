"""
Unit tests for core sequential testing components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.sequential import (
    sequential_verify,
    SPRTResult,
    SequentialState,
    eb_radius
)
from src.core.boundaries import SequentialTest


class TestSequentialVerify:
    """Test sequential verification functions."""
    
    def test_sequential_verify_accept_h0(self):
        """Test accepting null hypothesis (models are same)."""
        # Generate similar scores
        np.random.seed(42)
        scores = np.random.normal(0.95, 0.01, 100)  # High similarity scores
        
        result = sequential_verify(
            scores=scores.tolist(),
            alpha=0.05,
            beta=0.10,
            delta=0.1
        )
        
        assert isinstance(result, SPRTResult)
        assert result.decision in ["accept_h0", "continue"]
        assert result.n_samples == len(scores)
        assert 0 <= result.confidence <= 1
    
    def test_sequential_verify_reject_h0(self):
        """Test rejecting null hypothesis (models are different)."""
        # Generate dissimilar scores
        np.random.seed(42)
        scores = np.random.normal(0.3, 0.05, 100)  # Low similarity scores
        
        result = sequential_verify(
            scores=scores.tolist(),
            alpha=0.05,
            beta=0.10,
            delta=0.1
        )
        
        assert isinstance(result, SPRTResult)
        assert result.decision in ["reject_h0", "continue"]
        assert result.n_samples == len(scores)
    
    def test_sequential_verify_continue(self):
        """Test continuing when more samples needed."""
        # Generate borderline scores
        scores = [0.5, 0.6, 0.4, 0.5]  # Ambiguous scores
        
        result = sequential_verify(
            scores=scores,
            alpha=0.05,
            beta=0.10,
            delta=0.1
        )
        
        assert result.decision == "continue"
        assert result.n_samples == len(scores)
    
    def test_sequential_state_persistence(self):
        """Test that state persists across calls."""
        state = SequentialState()
        
        # First batch
        scores1 = [0.9, 0.85, 0.88]
        result1 = sequential_verify(scores1, state=state)
        
        # Second batch - should continue from previous state
        scores2 = [0.91, 0.89]
        result2 = sequential_verify(scores2, state=state)
        
        assert result2.n_samples == len(scores1) + len(scores2)
        assert state.n == len(scores1) + len(scores2)
    
    def test_empty_scores(self):
        """Test handling of empty score list."""
        result = sequential_verify(scores=[])
        
        assert result.decision == "continue"
        assert result.n_samples == 0
        assert result.mean == 0
    
    def test_single_score(self):
        """Test handling of single score."""
        result = sequential_verify(scores=[0.8])
        
        assert result.decision == "continue"  # Need more samples
        assert result.n_samples == 1
        assert result.mean == 0.8
    
    @pytest.mark.parametrize("alpha,beta", [
        (0.01, 0.01),
        (0.05, 0.05),
        (0.10, 0.10),
        (0.01, 0.10),
    ])
    def test_different_error_rates(self, alpha, beta):
        """Test with different Type I and Type II error rates."""
        np.random.seed(42)
        scores = np.random.normal(0.7, 0.1, 50)
        
        result = sequential_verify(
            scores=scores.tolist(),
            alpha=alpha,
            beta=beta
        )
        
        assert isinstance(result, SPRTResult)
        assert result.alpha == alpha
        assert result.beta == beta


class TestEBRadius:
    """Test Empirical-Bernstein radius calculation."""
    
    def test_eb_radius_basic(self):
        """Test basic EB radius calculation."""
        radius = eb_radius(
            t=100,
            n_total=1000,
            delta=0.05,
            sigma_sq=0.1,
            alpha=1.0
        )
        
        assert radius > 0
        assert radius < 1  # Should be reasonable for these parameters
    
    def test_eb_radius_increases_with_variance(self):
        """Test that radius increases with variance."""
        radius_low = eb_radius(t=100, n_total=1000, delta=0.05, sigma_sq=0.01)
        radius_high = eb_radius(t=100, n_total=1000, delta=0.05, sigma_sq=0.5)
        
        assert radius_high > radius_low
    
    def test_eb_radius_decreases_with_samples(self):
        """Test that radius decreases with more samples."""
        radius_few = eb_radius(t=10, n_total=1000, delta=0.05, sigma_sq=0.1)
        radius_many = eb_radius(t=500, n_total=1000, delta=0.05, sigma_sq=0.1)
        
        assert radius_many < radius_few
    
    def test_eb_radius_edge_cases(self):
        """Test edge cases for EB radius."""
        # Zero variance
        radius_zero_var = eb_radius(t=100, n_total=1000, delta=0.05, sigma_sq=0.0)
        assert radius_zero_var >= 0
        
        # Very small t
        radius_small_t = eb_radius(t=1, n_total=1000, delta=0.05, sigma_sq=0.1)
        assert radius_small_t > 0
        
        # t equals n_total
        radius_full = eb_radius(t=1000, n_total=1000, delta=0.05, sigma_sq=0.1)
        assert radius_full >= 0


class TestSequentialTest:
    """Test SequentialTest class."""
    
    def test_sequential_test_initialization(self):
        """Test SequentialTest initialization."""
        test = SequentialTest(alpha=0.05, beta=0.10, n_max=1000)
        
        assert test.alpha == 0.05
        assert test.beta == 0.10
        assert test.n_max == 1000
        assert test.state is not None
    
    def test_sequential_test_update(self):
        """Test updating sequential test with new samples."""
        test = SequentialTest(alpha=0.05, beta=0.10)
        
        # Add samples one by one
        decisions = []
        for score in [0.9, 0.85, 0.88, 0.92, 0.89]:
            decision = test.update(score)
            decisions.append(decision)
        
        assert all(d in ["accept_h0", "reject_h0", "continue"] for d in decisions)
        assert test.state.n == 5
    
    def test_sequential_test_batch_update(self):
        """Test batch update of sequential test."""
        test = SequentialTest(alpha=0.05, beta=0.10)
        
        scores = [0.3, 0.35, 0.32, 0.28, 0.31]
        decision = test.batch_update(scores)
        
        assert decision in ["accept_h0", "reject_h0", "continue"]
        assert test.state.n == len(scores)
    
    def test_sequential_test_reset(self):
        """Test resetting sequential test."""
        test = SequentialTest(alpha=0.05, beta=0.10)
        
        # Add some samples
        test.batch_update([0.5, 0.6, 0.4])
        assert test.state.n == 3
        
        # Reset
        test.reset()
        assert test.state.n == 0
        assert test.state.sum_x == 0
        assert test.state.sum_xx == 0


class TestSequentialState:
    """Test SequentialState class."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        state = SequentialState()
        
        assert state.n == 0
        assert state.sum_x == 0.0
        assert state.sum_xx == 0.0
        assert state.mean == 0.0
        assert state.var == 0.0
    
    def test_state_update(self):
        """Test state update with Welford's algorithm."""
        state = SequentialState()
        
        # Update with known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            state.update(val)
        
        assert state.n == 5
        assert state.mean == pytest.approx(3.0)
        assert state.var == pytest.approx(2.5)  # Sample variance
    
    def test_state_numerical_stability(self):
        """Test numerical stability with large values."""
        state = SequentialState()
        
        # Large values with small variance
        large_base = 1e10
        values = [large_base + i*0.1 for i in range(100)]
        
        for val in values:
            state.update(val)
        
        # Check that mean is computed accurately
        expected_mean = np.mean(values)
        assert abs(state.mean - expected_mean) / expected_mean < 1e-10
        
        # Check variance computation
        expected_var = np.var(values, ddof=1)
        assert abs(state.var - expected_var) / expected_var < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])