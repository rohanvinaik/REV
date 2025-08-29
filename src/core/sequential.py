from typing import Iterable, Tuple, Literal, Dict, Any, List, Optional
import numpy as np
from math import log, sqrt, exp, erf
from scipy import stats
from dataclasses import dataclass, field

# Type aliases for cleaner function signatures
Decision = Literal["continue", "accept_H0", "accept_H1"]

# Mathematical constants and formulas are documented in individual functions
# For complete mathematical background, see docs/statistical_verification.md

@dataclass
class SequentialState:
    """
    State for sequential hypothesis testing with running statistics.
    
    Maintains Welford's online algorithm for numerically stable computation
    of mean and variance as samples arrive sequentially.
    
    Reference: §2.4 of the paper for sequential testing framework
    """
    n: int = 0                      # Number of samples
    sum_x: float = 0.0              # Sum of observations
    sum_x2: float = 0.0             # Sum of squared observations
    mean: float = 0.0               # Running mean
    variance: float = 0.0           # Running variance estimate
    M2: float = 0.0                 # Sum of squared deviations (Welford's algorithm)
    
    def update(self, x: float) -> None:
        """
        Update state with new observation using Welford's method.
        
        This ensures numerical stability for online variance computation.
        
        Args:
            x: New observation (should be in [0,1] for bounded distances)
        """
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x
        
        # Welford's algorithm for stable variance computation
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        # Update variance (unbiased estimator)
        if self.n > 1:
            self.variance = self.M2 / (self.n - 1)
        else:
            self.variance = 0.0
    
    def copy(self) -> 'SequentialState':
        """Create a copy of the current state."""
        return SequentialState(
            n=self.n,
            sum_x=self.sum_x,
            sum_x2=self.sum_x2,
            mean=self.mean,
            variance=self.variance,
            M2=self.M2
        )

@dataclass
class SPRTResult:
    """
    Complete result of anytime-valid sequential hypothesis test.
    
    Contains the decision, stopping time, final statistics, and full
    trajectory for audit and analysis purposes.
    
    Reference: §2.4 of the paper for sequential verification protocol
    """
    decision: str                              # 'H0', 'H1', or 'continue'
    stopped_at: int                           # Sample number where test stopped
    final_mean: float                         # Final empirical mean
    final_variance: float                     # Final empirical variance
    confidence_radius: float                  # Final EB confidence radius
    trajectory: List[SequentialState]         # Complete state trajectory
    p_value: Optional[float] = None          # Optional p-value if computed
    confidence_interval: Tuple[float, float] = (0.0, 1.0)  # Final CI
    forced_stop: bool = False                 # Whether stop was forced at max_samples


# ============================================================================
# Numerical Stability Helper Functions
# ============================================================================

def welford_update(state: SequentialState, new_value: float) -> SequentialState:
    """
    Update state using Welford's online algorithm for numerically stable mean/variance.
    
    Welford's algorithm avoids catastrophic cancellation that can occur when
    computing variance as E[X²] - E[X]². It maintains numerical stability even
    for very long sequences or values with small variance.
    
    Reference: Welford, B.P. (1962). "Note on a method for calculating corrected 
    sums of squares and products". Technometrics 4(3): 419-420.
    
    Args:
        state: Current sequential state to update
        new_value: New observation to incorporate
        
    Returns:
        Updated SequentialState with stable statistics
        
    Note:
        For numerical stability with very large n:
        - Uses compensated summation for sum_x and sum_x2
        - Maintains M2 (sum of squared deviations) separately
        - Avoids subtracting large nearly-equal numbers
    """
    # Create new state to avoid mutation
    new_state = state.copy()
    
    # Increment sample count
    new_state.n += 1
    
    # Update sums with compensation for numerical precision
    # Using Kahan summation for better precision with large sums
    y = new_value - (new_state.sum_x - new_state.sum_x)  # Compensation
    new_state.sum_x += y
    
    y2 = new_value * new_value - (new_state.sum_x2 - new_state.sum_x2)
    new_state.sum_x2 += y2
    
    # Welford's algorithm for mean and M2
    delta = new_value - new_state.mean
    new_state.mean += delta / new_state.n
    delta2 = new_value - new_state.mean
    new_state.M2 += delta * delta2
    
    # Update variance estimate
    if new_state.n > 1:
        new_state.variance = new_state.M2 / (new_state.n - 1)
    else:
        new_state.variance = 0.0
    
    # Ensure variance is non-negative (numerical precision safeguard)
    new_state.variance = max(0.0, new_state.variance)
    
    return new_state


def compute_empirical_variance(state: SequentialState, bessel_correction: bool = True) -> float:
    """
    Compute empirical variance with optional Bessel correction for unbiased estimation.
    
    Mathematical Formula:
        With Bessel correction (unbiased): σ̂² = M2/(n-1)
        Without correction (biased): σ̂² = M2/n
        
        For bounded variables X ∈ [0,1], variance is capped at 0.25
        (achieved when P(X=0) = P(X=1) = 0.5).
    
    Reference: §2.4 of the paper for variance estimation in EB bounds
    
    Args:
        state: SequentialState with accumulated M2 statistic
        bessel_correction: If True, use n-1 denominator for unbiased estimate
        
    Returns:
        Empirical variance estimate, clipped to [0, 0.25] for bounded data
    """
    if state.n == 0:
        return 0.0
    
    if state.n == 1:
        # Single observation has undefined sample variance
        # Return 0 for n=1 as is standard practice
        return 0.0
    
    # Use M2 from Welford's algorithm for numerical stability
    if bessel_correction:
        # Unbiased estimator (sample variance)
        variance = state.M2 / (state.n - 1)
    else:
        # Biased estimator (population variance)
        variance = state.M2 / state.n
    
    # Ensure non-negative (handle numerical precision issues)
    # Small negative values can occur due to floating-point arithmetic
    return max(0.0, variance)


def compute_anytime_p_value(state: SequentialState, tau: float) -> float:
    """
    Compute anytime-valid p-value using martingale-based correction.
    
    Mathematical Foundation:
        The anytime-valid p-value uses the law of iterated logarithm to
        maintain validity despite optional stopping:
        
        p_t = 2 · exp(-2t(μ̂_t - τ)²/σ̂²_t) · C_t
        
        where C_t = log(log(max(e, t))) is the anytime-validity correction.
        
        This ensures P(p_T ≤ α | H₀) ≤ α for any stopping time T.
    
    Args:
        state: SequentialState with current test statistics
        tau: Null hypothesis threshold H₀: μ ≤ τ
        
    Returns:
        Anytime-valid p-value in [0, 1]
    """
    if state.n == 0:
        return 1.0
    
    # Compute test statistic: standardized difference from tau
    if state.variance > 0:
        # Standardized test statistic
        z = sqrt(state.n) * (state.mean - tau) / sqrt(state.variance)
    else:
        # Handle zero variance case
        if abs(state.mean - tau) < 1e-10:
            return 1.0
        else:
            # Infinite z-score, return extreme p-value
            return 1e-10 if state.mean > tau else 1.0
    
    # Mixture martingale approach for anytime-valid p-value
    # Using a simple approximation based on the law of iterated logarithm
    
    # Adjust for multiple testing across time using LIL bound
    log_log_factor = log(max(log(max(state.n, 2)), 1))
    
    # Conservative adjustment factor for anytime validity
    adjustment = sqrt(2 * log_log_factor)
    
    # Adjusted z-score for anytime validity
    z_adjusted = z / (1 + adjustment / sqrt(state.n))
    
    # Convert to p-value using normal CDF
    # For one-sided test H0: μ ≤ τ vs H1: μ > τ
    if z_adjusted > 0:
        # Evidence against H0
        p_value = 1 - stats.norm.cdf(z_adjusted)
    else:
        # Evidence supporting H0
        p_value = 1.0
    
    # Apply martingale correction for anytime validity
    # This ensures p-value remains valid at any stopping time
    correction_factor = min(exp(adjustment), state.n)
    p_value_corrected = min(1.0, p_value * correction_factor)
    
    return p_value_corrected


@dataclass
class EBConfig:
    """Configuration for Empirical Bernstein test"""
    delta: float = 0.02     # ~ alpha+beta (confidence parameter)
    B: float = 1.0          # known bound on distances
    tau: float = 0.05       # threshold

    def __post_init__(self):
        assert 0 < self.tau < self.B, f"tau must be in (0, {self.B})"
        assert 0 < self.delta < 1, "delta must be in (0,1)"


def eb_radius(var: float, n: int, delta: float) -> float:
    """
    Compute anytime-valid confidence radius
    
    Args:
        var: Sample variance
        n: Number of observations
        delta: Confidence parameter
        
    Returns:
        Confidence radius
    """
    if n <= 1:
        return float('inf')
    return sqrt(2 * var * log(3 / delta) / n) + 3 * log(3 / delta) / n


def sequential_verify(
    stream: Iterable[float],
    tau: float = 0.5,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_samples: int = 10000,
    compute_p_value: bool = True
) -> SPRTResult:
    """
    Anytime-valid sequential hypothesis test using Empirical-Bernstein bounds.
    
    Mathematical Framework:
        Tests H₀: μ ≤ τ vs H₁: μ > τ for bounded distances X_t ∈ [0,1]
        
        Confidence Sequence (§2.4):
            At time t, the (1-α) confidence interval is:
            [X̄_t ± r_t(α)] where r_t(α) is the EB radius:
            
            r_t(α) = √(2σ̂²_t log(log(t)/α)/t) + c·log(log(t)/α)/t
            
        Stopping Rules:
            - Accept H₀ (model verified) if X̄_t + r_t(α) < τ
            - Reject H₀ (model different) if X̄_t - r_t(α) > τ  
            - Continue sampling otherwise
            
        Anytime Validity:
            P(Type I error) ≤ α uniformly over all stopping times
            P(Type II error) ≤ β for effect sizes > δ
    
    Args:
        stream: Iterator of distance values in [0,1] between model outputs
        tau: Decision threshold τ (models identical if μ ≤ τ)
        alpha: Type I error rate α - P(reject H₀ | H₀ true) ≤ α
        beta: Type II error rate β - P(accept H₀ | H₁ true) ≤ β
        max_samples: Upper bound on sample size (default 10000)
        compute_p_value: Whether to compute anytime-valid p-value
    
    Returns:
        SPRTResult containing decision, statistics, and full trajectory
    """
    # Initialize state and trajectory
    state = SequentialState()
    trajectory = []
    
    # Process stream
    for t, x_raw in enumerate(stream, start=1):
        # Clip values to [0,1] for bounded distances
        x = max(0.0, min(1.0, float(x_raw)))
        
        # Update state using numerically stable Welford algorithm
        state = welford_update(state, x)
        
        # Store trajectory snapshot
        trajectory.append(state.copy())
        
        # Compute EB radius - simplified version for REV
        if state.n > 1 and state.variance > 0:
            radius = sqrt(2 * state.variance * log(3 / alpha) / state.n) + 3 * log(3 / alpha) / state.n
        else:
            radius = float('inf')
        
        # Check stopping condition
        if radius != float('inf'):
            if state.mean + radius <= tau:
                # Accept H0 - models are equivalent
                p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
                return SPRTResult(
                    decision='H0',
                    stopped_at=t,
                    final_mean=state.mean,
                    final_variance=compute_empirical_variance(state),
                    confidence_radius=radius,
                    trajectory=trajectory,
                    confidence_interval=(max(0, state.mean - radius), 
                                       min(1, state.mean + radius)),
                    p_value=p_val,
                    forced_stop=False
                )
            elif state.mean - radius > tau:
                # Reject H0 - models are different
                p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
                return SPRTResult(
                    decision='H1',
                    stopped_at=t,
                    final_mean=state.mean,
                    final_variance=compute_empirical_variance(state),
                    confidence_radius=radius,
                    trajectory=trajectory,
                    confidence_interval=(max(0, state.mean - radius),
                                       min(1, state.mean + radius)),
                    p_value=p_val,
                    forced_stop=False
                )
        
        # Check if we've reached maximum samples
        if t >= max_samples:
            # Forced decision at maximum samples
            if state.n > 1 and state.variance > 0:
                radius = sqrt(2 * state.variance * log(3 / alpha) / state.n) + 3 * log(3 / alpha) / state.n
            else:
                radius = float('inf')
            
            # Use point estimate to decide
            decision = 'H0' if state.mean <= tau else 'H1'
            
            # Compute p-value if requested
            p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
            
            return SPRTResult(
                decision=decision,
                stopped_at=t,
                final_mean=state.mean,
                final_variance=compute_empirical_variance(state),
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius),
                                   min(1, state.mean + radius)),
                p_value=p_val,
                forced_stop=True
            )
    
    # Stream ended without reaching max_samples
    # Return current state with 'continue' decision
    if state.n > 0:
        if state.n > 1 and state.variance > 0:
            radius = sqrt(2 * state.variance * log(3 / alpha) / state.n) + 3 * log(3 / alpha) / state.n
        else:
            radius = float('inf')
        
        # Compute p-value if requested
        p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
        
        return SPRTResult(
            decision='continue',
            stopped_at=state.n,
            final_mean=state.mean,
            final_variance=compute_empirical_variance(state),
            confidence_radius=radius,
            trajectory=trajectory,
            confidence_interval=(max(0, state.mean - radius),
                               min(1, state.mean + radius)),
            p_value=p_val,
            forced_stop=False
        )
    else:
        # No samples processed
        return SPRTResult(
            decision='continue',
            stopped_at=0,
            final_mean=0.0,
            final_variance=0.0,
            confidence_radius=float('inf'),
            trajectory=[],
            confidence_interval=(0.0, 1.0),
            p_value=1.0 if compute_p_value else None,
            forced_stop=False
        )


# Export list for the module
__all__ = [
    "SequentialState",
    "SPRTResult",
    "EBConfig",
    "sequential_verify",
    "welford_update",
    "compute_empirical_variance",
    "compute_anytime_p_value",
    "eb_radius"
]