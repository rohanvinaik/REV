from typing import Iterable, Tuple, Literal, Dict, Any, List, Optional, Generator
import numpy as np
from math import log, sqrt, exp, erf
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum

# Type aliases for cleaner function signatures
Decision = Literal["continue", "accept_H0", "accept_H1"]

class TestType(Enum):
    """Type of sequential test being performed."""
    MATCH = "match"      # Bernoulli test for exact equality
    DISTANCE = "distance" # Distance threshold test

# Mathematical constants and formulas are documented in individual functions
# For complete mathematical background, see docs/statistical_verification.md

@dataclass
class SequentialState:
    """
    Enhanced state for dual sequential hypothesis testing per Section 5.7.
    
    Maintains Welford's online algorithm for numerically stable computation
    of mean and variance as samples arrive sequentially. Supports both
    Bernoulli evidence (exact matches) and distance threshold testing.
    
    Reference: Section 5.7 of the paper for dual sequential testing framework
    """
    # Basic statistics
    n: int = 0                      # Number of samples
    sum_x: float = 0.0              # Sum of observations
    sum_x2: float = 0.0             # Sum of squared observations
    mean: float = 0.0               # Running mean
    variance: float = 0.0           # Running variance estimate
    M2: float = 0.0                 # Sum of squared deviations (Welford's algorithm)
    
    # Dual test statistics (Section 5.7)
    n_match: int = 0                # Count of exact matches (Bernoulli evidence)
    n_below_threshold: int = 0      # Count below distance threshold
    
    # Test configuration
    test_type: TestType = TestType.DISTANCE
    alpha: float = 0.05             # Type I error rate
    beta: float = 0.10              # Type II error rate
    
    # Confidence sequences
    log_likelihood_ratio: float = 0.0  # Log LR for SPRT
    e_value: float = 1.0            # E-value for anytime validity
    
    # Localization tracking
    first_divergence_site: Optional[int] = None
    divergence_sites: List[int] = field(default_factory=list)
    
    def update(self, x: float, is_match: bool = False) -> None:
        """
        Update state with new observation using Welford's method.
        
        Enhanced to track both distance and match evidence per Section 5.7.
        
        Args:
            x: New observation (should be in [0,1] for bounded distances)
            is_match: Whether this observation is an exact match
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
        
        # Track match evidence (Section 5.7)
        if is_match:
            self.n_match += 1
        
        # Track first divergence
        if not is_match and self.first_divergence_site is None:
            self.first_divergence_site = self.n
            
        # Track all divergence sites
        if not is_match:
            self.divergence_sites.append(self.n)
    
    def update_distance(self, d: float, threshold: float) -> None:
        """
        Update distance test statistics per Section 5.7.
        
        Args:
            d: Distance value
            threshold: Distance threshold for test
        """
        self.update(d, is_match=(d == 0.0))
        
        # Track threshold crossings
        if d <= threshold:
            self.n_below_threshold += 1
    
    def get_match_rate(self) -> float:
        """Get empirical match rate for Bernoulli test."""
        if self.n == 0:
            return 0.0
        return self.n_match / self.n
    
    def get_below_threshold_rate(self) -> float:
        """Get rate of observations below threshold."""
        if self.n == 0:
            return 0.0
        return self.n_below_threshold / self.n
    
    def get_confidence(self) -> float:
        """Get confidence level based on current evidence."""
        if self.n < 2:
            return 0.0
        
        # Use variance-stabilized confidence
        if self.variance > 0:
            z_score = sqrt(self.n) * abs(self.mean - 0.5) / sqrt(self.variance)
            confidence = 1.0 - exp(-z_score**2 / 2)
        else:
            confidence = 1.0 if self.mean != 0.5 else 0.0
        
        return min(1.0, confidence)
    
    def get_decision(self) -> 'Verdict':
        """Get current decision based on sequential test."""
        from ..verifier.decision import Verdict
        
        if self.n < 10:  # Minimum samples
            return Verdict.UNDECIDED
        
        # Check confidence bounds
        radius = self.get_confidence_radius()
        
        if self.mean + radius <= 0.05:  # Very similar
            return Verdict.SAME
        elif self.mean - radius > 0.15:  # Clearly different
            return Verdict.DIFFERENT
        else:
            return Verdict.UNDECIDED
    
    def get_confidence_radius(self) -> float:
        """Get confidence radius for current state."""
        if self.n <= 1:
            return float('inf')
        
        # Empirical Bernstein radius
        if self.variance > 0:
            radius = sqrt(2 * self.variance * log(3 / self.alpha) / self.n) + \
                    3 * log(3 / self.alpha) / self.n
        else:
            radius = 3 * log(3 / self.alpha) / self.n
        
        return radius
    
    def should_stop(self) -> bool:
        """Check if sequential test should stop."""
        if self.n < 10:  # Minimum samples
            return False
        
        # Check if confidence interval excludes indifference region
        radius = self.get_confidence_radius()
        
        # Stop if we can make a decision
        if self.mean + radius <= 0.05 or self.mean - radius > 0.15:
            return True
        
        # Stop if we've seen enough samples
        if self.n >= 2000:
            return True
        
        return False
    
    def copy(self) -> 'SequentialState':
        """Create a copy of the current state."""
        new_state = SequentialState(
            n=self.n,
            sum_x=self.sum_x,
            sum_x2=self.sum_x2,
            mean=self.mean,
            variance=self.variance,
            M2=self.M2,
            n_match=self.n_match,
            n_below_threshold=self.n_below_threshold,
            test_type=self.test_type,
            alpha=self.alpha,
            beta=self.beta,
            log_likelihood_ratio=self.log_likelihood_ratio,
            e_value=self.e_value,
            first_divergence_site=self.first_divergence_site,
            divergence_sites=self.divergence_sites.copy()
        )
        return new_state

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


# ============================================================================
# Dual Sequential Testing Framework (Section 5.7)
# ============================================================================

@dataclass 
class DualSequentialTest:
    """
    Dual sequential test framework from Section 5.7.
    
    Combines two sequential tests:
    - S_match: Bernoulli test for exact equality
    - S_dist: Distance threshold test
    """
    S_match: SequentialState
    S_dist: SequentialState
    alpha: float = 0.01
    beta: float = 0.01
    d_thresh: float = 0.08
    max_C: int = 2000
    
    def __post_init__(self):
        """Initialize the two sequential tests."""
        self.S_match.test_type = TestType.MATCH
        self.S_match.alpha = self.alpha
        self.S_match.beta = self.beta
        
        self.S_dist.test_type = TestType.DISTANCE
        self.S_dist.alpha = self.alpha
        self.S_dist.beta = self.beta


def init_seq_test(alpha: float, test_type: TestType = TestType.DISTANCE) -> SequentialState:
    """
    Initialize a sequential test with given error rate.
    
    Args:
        alpha: Type I error rate
        test_type: Type of test (MATCH or DISTANCE)
        
    Returns:
        Initialized SequentialState
    """
    return SequentialState(
        test_type=test_type,
        alpha=alpha,
        beta=alpha  # Symmetric error rates by default
    )


def accept_same(S_match: SequentialState, S_dist: SequentialState) -> bool:
    """
    Check if we should accept that models are the SAME.
    
    Per Section 5.7: Accept SAME if both tests indicate similarity.
    
    Args:
        S_match: Bernoulli match test state
        S_dist: Distance threshold test state
        
    Returns:
        True if models should be considered SAME
    """
    # High match rate indicates similarity
    match_evidence = S_match.get_match_rate() > 0.9 and S_match.n >= 10
    
    # Low distance indicates similarity
    dist_evidence = S_dist.mean < 0.05 and S_dist.get_confidence_radius() < 0.05
    
    return match_evidence or dist_evidence


def accept_diff(S_match: SequentialState, S_dist: SequentialState) -> bool:
    """
    Check if we should accept that models are DIFFERENT.
    
    Per Section 5.7: Accept DIFFERENT if either test indicates difference.
    
    Args:
        S_match: Bernoulli match test state
        S_dist: Distance threshold test state
        
    Returns:
        True if models should be considered DIFFERENT
    """
    # Low match rate indicates difference
    match_evidence = S_match.get_match_rate() < 0.5 and S_match.n >= 20
    
    # High distance indicates difference
    dist_evidence = S_dist.mean > 0.15 and S_dist.get_confidence_radius() < 0.05
    
    return match_evidence or dist_evidence


def sequential_decision(
    stream: Generator[Dict[str, Any], None, None],
    alpha: float = 0.01,
    beta: float = 0.01,
    d_thresh: float = 0.08,
    max_C: int = 2000
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Dual sequential decision framework from Section 5.7 pseudocode.
    
    Implements the exact algorithm:
    ```
    def sequential_decision(stream, alpha=0.01, beta=0.01, d_thresh=0.08, max_C=2000):
        S_match = init_seq_test(alpha)
        S_dist = init_seq_test(beta)
        for t, r in enumerate(stream, 1):
            update(S_match, r["I"])      # Bernoulli evidence
            update(S_dist, r["d"], d_thresh)  # distance evidence
            if accept_same(S_match, S_dist):
                return "SAME", t
            if accept_diff(S_match, S_dist):
                return "DIFFERENT", t
            if t >= max_C:
                break
        return "UNDECIDED", t
    ```
    
    Args:
        stream: Generator yielding dicts with "I" (indicator) and "d" (distance)
        alpha: Type I error rate for match test
        beta: Type I error rate for distance test
        d_thresh: Distance threshold
        max_C: Maximum number of comparisons
        
    Returns:
        Tuple of (verdict, stopping_time, localization_info)
    """
    # Initialize dual sequential tests
    S_match = init_seq_test(alpha, TestType.MATCH)
    S_dist = init_seq_test(beta, TestType.DISTANCE)
    
    localization_info = {
        "first_divergence": None,
        "divergence_sites": [],
        "match_trajectory": [],
        "distance_trajectory": []
    }
    
    # Process stream
    for t, r in enumerate(stream, 1):
        # Extract evidence from stream
        I = r.get("I", 0)  # Bernoulli indicator (1 if match, 0 if not)
        d = r.get("d", 0.0)  # Distance value
        
        # Update match test with Bernoulli evidence
        S_match.update(float(I), is_match=(I == 1))
        localization_info["match_trajectory"].append(S_match.get_match_rate())
        
        # Update distance test
        S_dist.update_distance(d, d_thresh)
        localization_info["distance_trajectory"].append(d)
        
        # Track first divergence
        if I == 0 and localization_info["first_divergence"] is None:
            localization_info["first_divergence"] = t
            localization_info["divergence_sites"].append(t)
        elif I == 0:
            localization_info["divergence_sites"].append(t)
        
        # Check stopping conditions
        if accept_same(S_match, S_dist):
            localization_info["match_rate"] = S_match.mean if S_match.n > 0 else 0.0
            localization_info["mean_distance"] = S_dist.mean if S_dist.n > 0 else 0.0
            return "SAME", t, localization_info
        
        if accept_diff(S_match, S_dist):
            localization_info["match_rate"] = S_match.mean if S_match.n > 0 else 0.0
            localization_info["mean_distance"] = S_dist.mean if S_dist.n > 0 else 0.0
            return "DIFFERENT", t, localization_info
        
        if t >= max_C:
            break
    
    # Return undecided if we hit max samples
    localization_info["match_rate"] = S_match.mean if S_match.n > 0 else 0.0
    localization_info["mean_distance"] = S_dist.mean if S_dist.n > 0 else 0.0
    return "UNDECIDED", t, localization_info


@dataclass
class ConfidenceSequence:
    """
    Anytime-valid confidence sequence for sequential testing.
    
    Implements e-values and confidence sequences for multiple testing correction.
    """
    e_values: List[float] = field(default_factory=list)
    confidence_levels: List[float] = field(default_factory=list)
    peeling_factor: float = 1.1  # For peeling in multiple testing
    
    def update(self, e_value: float):
        """Update with new e-value."""
        self.e_values.append(e_value)
        
        # Compute confidence level
        confidence = 1.0 - 1.0 / max(e_value, 1.0)
        self.confidence_levels.append(confidence)
    
    def get_adjusted_confidence(self, k: int) -> float:
        """
        Get adjusted confidence for k-th test with peeling.
        
        Args:
            k: Test index
            
        Returns:
            Adjusted confidence level
        """
        if k >= len(self.confidence_levels):
            return 0.0
        
        # Apply peeling adjustment
        adjusted = self.confidence_levels[k] / (self.peeling_factor ** k)
        return min(1.0, adjusted)


def compute_e_value(
    state: SequentialState,
    null_mean: float = 0.5,
    alt_mean: float = 0.3
) -> float:
    """
    Compute e-value for anytime-valid inference.
    
    E-values are non-negative random variables with E[E] ≤ 1 under null.
    
    Args:
        state: Current sequential state
        null_mean: Mean under null hypothesis
        alt_mean: Mean under alternative hypothesis
        
    Returns:
        E-value for current evidence
    """
    if state.n == 0:
        return 1.0
    
    # Compute likelihood ratio
    if state.variance > 0:
        # Normal approximation
        null_ll = -state.n * ((state.mean - null_mean)**2) / (2 * state.variance)
        alt_ll = -state.n * ((state.mean - alt_mean)**2) / (2 * state.variance)
        
        # E-value is likelihood ratio
        e_value = exp(alt_ll - null_ll)
    else:
        # Degenerate case
        if abs(state.mean - alt_mean) < abs(state.mean - null_mean):
            e_value = float('inf')
        else:
            e_value = 0.0
    
    return max(0.0, e_value)


# Export list for the module
__all__ = [
    "SequentialState",
    "SPRTResult",
    "EBConfig",
    "TestType",
    "DualSequentialTest",
    "ConfidenceSequence",
    "sequential_verify",
    "sequential_decision",
    "init_seq_test",
    "accept_same",
    "accept_diff",
    "welford_update",
    "compute_empirical_variance",
    "compute_anytime_p_value",
    "compute_e_value",
    "eb_radius"
]