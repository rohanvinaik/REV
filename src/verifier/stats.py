import math
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from ..core.boundaries import EnhancedStatisticalFramework, VerificationMode


@dataclass
class Welford:
    """Online algorithm for computing mean and variance with numerical stability"""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of diffs from current mean

    def push(self, x: float) -> None:
        """Add new observation and update statistics"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        """Compute sample variance using Bessel's correction"""
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)


def eb_halfwidth(var: float, n: int, delta: float) -> float:
    """
    Empirical-Bernstein half-width for bounded [0,1] scores.
    
    Formula: U_n(delta) = sqrt(2 * var * log(1/delta) / n) + 7*log(1/delta) / (3*(n-1))
    
    This provides anytime-valid confidence bounds for sequential testing.
    
    Args:
        var: Sample variance
        n: Number of samples
        delta: Confidence parameter (smaller = wider bounds)
        
    Returns:
        Half-width of confidence interval
    """
    if n <= 1:
        return float("inf")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1)")
    
    log_term = math.log(1.0 / delta)
    variance_term = math.sqrt(max(0.0, 2.0 * var * log_term / n))
    concentration_term = 7.0 * log_term / (3.0 * (n - 1))
    
    return variance_term + concentration_term


def spending_schedule(alpha: float, n: int, mode: Optional[VerificationMode] = None) -> float:
    """
    Conservative spending schedule for anytime bounds in REV.
    
    Uses delta_n = alpha / (n * (n + 1)) which ensures sum_n delta_n <= alpha
    while providing reasonable power for sequential testing.
    
    Adapts based on verification mode for optimal power.
    
    Args:
        alpha: Overall significance level
        n: Current sample size
        mode: Optional verification mode for adaptive scheduling
        
    Returns:
        Spending amount for step n
    """
    if mode == VerificationMode.CONSENSUS:
        # More aggressive spending for consensus mode
        return alpha / (n * math.sqrt(n + 1))
    elif mode == VerificationMode.UNIFIED:
        # Balanced spending
        return alpha / (n * (n + 1) ** 0.75)
    else:
        # Conservative spending for sequential mode
        return alpha / (n * (n + 1))


class UnifiedStatistics:
    """
    Unified statistics tracking for REV+HBT verification.
    
    Combines Welford's algorithm with variance tensor analysis
    for comprehensive statistical tracking.
    """
    
    def __init__(
        self,
        mode: VerificationMode = VerificationMode.UNIFIED,
        alpha: float = 0.05,
        beta: float = 0.10
    ):
        """
        Initialize unified statistics.
        
        Args:
            mode: Verification mode
            alpha: Type I error rate
            beta: Type II error rate
        """
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        
        # Core statistics
        self.welford = Welford()
        self.framework = EnhancedStatisticalFramework(
            mode=mode,
            alpha=alpha,
            beta=beta
        )
        
        # Variance tracking
        self.variance_tensors: list[np.ndarray] = []
        self.signature_history: list[Dict[str, np.ndarray]] = []
    
    def update(
        self,
        observation: float,
        variance_tensor: Optional[np.ndarray] = None,
        signatures: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Update statistics with new observation.
        
        Args:
            observation: New observation value in [0, 1]
            variance_tensor: Optional variance tensor from HBT
            signatures: Optional segment signatures
            
        Returns:
            Dictionary with updated statistics and confidence
        """
        # Update Welford statistics
        self.welford.push(observation)
        
        # Store variance tensor if provided
        if variance_tensor is not None:
            self.variance_tensors.append(variance_tensor)
        
        # Store signatures if provided
        if signatures is not None:
            self.signature_history.append(signatures)
        
        # Update enhanced framework
        result = self.framework.update_boundaries(
            observation=observation,
            variance_data=variance_tensor
        )
        
        # Add Welford statistics
        result['welford_mean'] = self.welford.mean
        result['welford_var'] = self.welford.var
        
        # Compute EB halfwidth with adaptive spending
        delta_n = spending_schedule(self.alpha, self.welford.n, self.mode)
        result['eb_halfwidth'] = eb_halfwidth(self.welford.var, self.welford.n, delta_n)
        
        return result
    
    def get_unified_confidence(self) -> Tuple[float, Dict[str, float]]:
        """
        Get unified confidence score.
        
        Returns:
            Tuple of (unified_confidence, components)
        """
        # Extract latest variance tensor if available
        variance_tensor = None
        if self.variance_tensors:
            variance_tensor = self.variance_tensors[-1]
        
        # Extract latest signatures if available
        signatures = None
        if self.signature_history:
            signatures = self.signature_history[-1]
        
        return self.framework.compute_unified_confidence(
            variance_tensor=variance_tensor,
            segment_signatures=signatures
        )
    
    def make_decision(
        self,
        threshold: float = 0.5,
        min_samples: int = 10
    ) -> Tuple[str, float]:
        """
        Make verification decision.
        
        Args:
            threshold: Decision threshold
            min_samples: Minimum samples required
            
        Returns:
            Tuple of (decision, confidence)
        """
        return self.framework.make_decision(threshold, min_samples)
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """
        Get current confidence interval.
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.welford.n < 1:
            return (0.0, 1.0)
        
        # Use adaptive spending schedule
        delta_n = spending_schedule(self.alpha, self.welford.n, self.mode)
        halfwidth = eb_halfwidth(self.welford.var, self.welford.n, delta_n)
        
        lower = max(0.0, self.welford.mean - halfwidth)
        upper = min(1.0, self.welford.mean + halfwidth)
        
        return (lower, upper)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics summary.
        
        Returns:
            Dictionary with all relevant statistics
        """
        unified_conf, components = self.get_unified_confidence()
        lower, upper = self.get_confidence_interval()
        
        return {
            'n_samples': self.welford.n,
            'mean': self.welford.mean,
            'variance': self.welford.var,
            'confidence_interval': (lower, upper),
            'unified_confidence': unified_conf,
            'confidence_components': components,
            'mode': self.mode.value,
            'n_variance_tensors': len(self.variance_tensors),
            'n_signatures': len(self.signature_history)
        }
    
    def reset(self):
        """Reset all statistics."""
        self.welford = Welford()
        self.framework.reset()
        self.variance_tensors.clear()
        self.signature_history.clear()