import math
from dataclasses import dataclass


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


def spending_schedule(alpha: float, n: int) -> float:
    """
    Conservative spending schedule for anytime bounds in REV.
    
    Uses delta_n = alpha / (n * (n + 1)) which ensures sum_n delta_n <= alpha
    while providing reasonable power for sequential testing.
    
    Args:
        alpha: Overall significance level
        n: Current sample size
        
    Returns:
        Spending amount for step n
    """
    return alpha / (n * (n + 1))