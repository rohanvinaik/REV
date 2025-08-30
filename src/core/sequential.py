"""
Enhanced Sequential Testing Framework for REV (Section 5.7)

This module implements anytime-valid sequential testing with e-values,
confidence sequences, and advanced statistical features for robust
model comparison in the REV verification system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Union, Generator
import numpy as np
from collections import deque
import math
from scipy import stats
from scipy.special import loggamma


class TestType(Enum):
    """Type of sequential test being performed."""
    MATCH = "match"  # Bernoulli test for match/no-match
    DISTANCE = "distance"  # Continuous test for distance metrics
    HYBRID = "hybrid"  # Combined test using both types


class Verdict(Enum):
    """Possible verdicts from sequential testing."""
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNCERTAIN = "UNCERTAIN"
    UNDECIDED = "UNDECIDED"  # Not enough evidence yet


@dataclass
class ConfidenceSequence:
    """
    Anytime-valid confidence sequence using e-values.
    
    Implements the peeling method for multiple testing correction
    and maintains confidence bounds that are valid at any stopping time.
    """
    peeling_factor: float = 1.1  # Factor for peeling (ρ in paper)
    confidence_levels: List[float] = field(default_factory=list)
    e_values: List[float] = field(default_factory=list)
    confidence_radii: List[float] = field(default_factory=list)
    
    def update(self, e_value: float) -> None:
        """Update confidence sequence with new e-value."""
        self.e_values.append(e_value)
        
        # Compute confidence level using e-value
        # CI_t = 1 - 1/E_t for anytime-valid confidence
        confidence = 1.0 - 1.0 / max(e_value, 1.0)
        self.confidence_levels.append(confidence)
        
        # Compute confidence radius using Ville's inequality
        n = len(self.e_values)
        radius = math.sqrt(2 * math.log(e_value) / n) if n > 0 and e_value > 1 else 1.0
        self.confidence_radii.append(radius)
    
    def get_adjusted_confidence(self, k: int) -> float:
        """
        Get adjusted confidence for k-th test (peeling).
        
        Args:
            k: Index of test (0-based)
            
        Returns:
            Adjusted confidence level accounting for multiple testing
        """
        if k >= len(self.confidence_levels):
            return 0.0
        
        # Apply peeling correction: α_k = α * ρ^k
        base_confidence = self.confidence_levels[k]
        adjustment = self.peeling_factor ** k
        
        return min(base_confidence / adjustment, 1.0)
    
    def get_current_radius(self) -> float:
        """Get current confidence radius."""
        return self.confidence_radii[-1] if self.confidence_radii else 1.0
    
    def get_confidence_radius(self, alpha: float = 0.05) -> float:
        """
        Get confidence radius based on e-values.
        
        Args:
            alpha: Significance level
            
        Returns:
            Confidence radius
        """
        if self.n == 0:
            return float('inf')
        
        # Use maximum e-value for tighter bounds
        max_e = max(self.e_values) if self.e_values else 1.0
        
        # Ville's inequality based radius
        if max_e > 1:
            radius = math.sqrt(2 * math.log(max_e) / (self.n * alpha))
        else:
            radius = math.sqrt(2 * math.log(1/alpha) / self.n)
        
        return min(radius, 1.0)  # Cap at 1.0 for reasonable bounds


@dataclass
class PowerAnalysis:
    """Power analysis utilities for sequential testing."""
    
    @staticmethod
    def compute_expected_sample_size(
        alpha: float,
        beta: float,
        effect_size: float,
        test_type: TestType = TestType.MATCH
    ) -> int:
        """
        Compute expected sample size for given power.
        
        Args:
            alpha: Type I error rate
            beta: Type II error rate
            effect_size: Expected effect size (Cohen's d for continuous)
            test_type: Type of test
            
        Returns:
            Expected number of samples needed
        """
        if test_type == TestType.MATCH:
            # For Bernoulli test, use Wald's approximation
            p0 = 0.5  # Null hypothesis
            p1 = 0.5 + effect_size  # Alternative
            
            # Log likelihood ratio components
            if p1 <= p0 or p1 >= 1.0:
                return 10000  # Maximum
            
            llr_1 = p1 * math.log(p1/p0) + (1-p1) * math.log((1-p1)/(1-p0))
            llr_0 = p0 * math.log(p0/p1) + (1-p0) * math.log((1-p0)/(1-p1))
            
            # Wald's approximation
            A = math.log((1-beta)/alpha)
            B = math.log(beta/(1-alpha))
            
            n_1 = A / llr_1 if llr_1 > 1e-10 else 10000  # Expected size under H1
            n_0 = B / llr_0 if abs(llr_0) > 1e-10 else 10000  # Expected size under H0
            
            result = (n_0 + n_1) / 2
            return min(int(result) if not math.isinf(result) else 10000, 10000)  # Cap at 10000
        
        else:  # DISTANCE test
            # Use standard power analysis for t-test
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(1 - beta)
            
            n = ((z_alpha + z_beta) / effect_size) ** 2
            return max(int(n), 5)
    
    @staticmethod
    def compute_power(
        n: int,
        alpha: float,
        effect_size: float,
        test_type: TestType = TestType.MATCH
    ) -> float:
        """
        Compute statistical power for given sample size.
        
        Args:
            n: Sample size
            alpha: Type I error rate
            effect_size: Effect size
            test_type: Type of test
            
        Returns:
            Statistical power (1 - beta)
        """
        if n <= 0:
            return 0.0
        
        if test_type == TestType.MATCH:
            # Approximate power for binomial test
            se = math.sqrt(0.25 / n)  # Standard error under null
            z_alpha = stats.norm.ppf(1 - alpha/2)
            
            # Non-centrality parameter
            ncp = effect_size * math.sqrt(n)
            
            # Power using normal approximation
            power = stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
            return min(power, 1.0)
        
        else:  # DISTANCE test
            # Power for t-test
            df = n - 1
            ncp = effect_size * math.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            # Non-central t-distribution
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
            return min(power, 1.0)


class SequentialState:
    """
    Enhanced sequential state with e-value tracking and advanced features.
    
    Maintains running statistics, confidence sequences, and history
    for anytime-valid sequential testing with optimal stopping.
    """
    
    def __init__(
        self,
        test_type: TestType = TestType.MATCH,
        alpha: float = 0.01,
        beta: float = 0.01,
        adaptive_threshold: bool = True,
        history_size: int = 1000
    ):
        """
        Initialize sequential state.
        
        Args:
            test_type: Type of sequential test
            alpha: Type I error rate (false positive)
            beta: Type II error rate (false negative)
            adaptive_threshold: Whether to adapt thresholds based on variance
            history_size: Maximum history to maintain
        """
        self.test_type = test_type
        self.alpha = alpha
        self.beta = beta
        self.adaptive_threshold = adaptive_threshold
        
        # Running statistics
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.mean = 0.0
        self.variance = 0.0
        self.std = 0.0
        
        # For Bernoulli test
        self.n_matches = 0
        self.n_mismatches = 0
        
        # For distance test
        self.min_distance = float('inf')
        self.max_distance = 0.0
        self.below_threshold_count = 0
        
        # Confidence sequences
        self.confidence_seq = ConfidenceSequence()
        
        # E-values and likelihood ratios
        self.log_likelihood_ratio = 0.0
        self.e_value_product = 1.0
        self.e_values: List[float] = []
        
        # SPRT boundaries
        self.upper_boundary = math.log((1 - beta) / alpha)
        self.lower_boundary = math.log(beta / (1 - alpha))
        
        # History tracking
        self.history = deque(maxlen=history_size)
        self.decision_history: List[Tuple[int, Verdict]] = []
        
        # Localization tracking
        self.first_divergence_site: Optional[int] = None
        self.divergence_sites: List[int] = []
        self.match_trajectory: List[float] = []
        
        # Adaptive threshold parameters
        self.threshold_history: List[float] = []
        self.current_threshold = 0.5  # Initial threshold
        
        # Confidence interval tracking
        self.ci_lower: List[float] = []
        self.ci_upper: List[float] = []
    
    def update(self, value: float, is_match: Optional[bool] = None) -> None:
        """
        Update state with new observation.
        
        Args:
            value: Observed value (distance or match indicator)
            is_match: Optional explicit match indicator for hybrid tests
        """
        self.n += 1
        self.sum += value
        self.sum_sq += value * value
        
        # Update running statistics (Welford's algorithm)
        delta = value - self.mean
        self.mean += delta / self.n
        if self.n > 1:
            self.variance = (self.sum_sq - self.sum * self.sum / self.n) / (self.n - 1)
            self.std = math.sqrt(max(self.variance, 0))
        
        # Store in history
        self.history.append(value)
        
        # Update test-specific statistics
        if self.test_type == TestType.MATCH:
            if is_match or value > 0.5:  # Treat as match
                self.n_matches += 1
            else:
                self.n_mismatches += 1
                if self.first_divergence_site is None:
                    self.first_divergence_site = self.n
                self.divergence_sites.append(self.n)
            
            # Update match trajectory
            match_rate = self.n_matches / self.n
            self.match_trajectory.append(match_rate)
        
        elif self.test_type == TestType.DISTANCE:
            self.min_distance = min(self.min_distance, value)
            self.max_distance = max(self.max_distance, value)
            
            # Track divergences based on adaptive threshold
            if self.adaptive_threshold:
                self._update_adaptive_threshold()
            
            if value < self.current_threshold:
                self.below_threshold_count += 1
            else:
                if self.first_divergence_site is None:
                    self.first_divergence_site = self.n
                self.divergence_sites.append(self.n)
        
        # Compute e-value
        e_val = self._compute_e_value(value)
        self.e_values.append(e_val)
        self.e_value_product *= e_val
        
        # Update confidence sequence
        self.confidence_seq.update(e_val)
        
        # Update log-likelihood ratio for SPRT
        self._update_llr(value)
        
        # Update confidence intervals
        self._update_confidence_intervals()
    
    def update_distance(self, distance: float, threshold: float) -> None:
        """
        Update with distance observation.
        
        Args:
            distance: Observed distance
            threshold: Distance threshold for comparison
        """
        self.current_threshold = threshold
        is_below = distance < threshold
        self.update(distance, is_match=is_below)
    
    def _compute_e_value(self, value: float) -> float:
        """
        Compute e-value for current observation.
        
        E-values provide anytime-valid inference without requiring
        a fixed sample size or stopping rule.
        """
        if self.n < 2:
            return 1.0
        
        if self.test_type == TestType.MATCH:
            # Binomial e-value
            p_null = 0.5  # Null: random guessing
            p_alt = self.mean  # Alternative: observed rate
            
            if p_alt <= 0 or p_alt >= 1:
                return 1.0
            
            # Likelihood ratio e-value
            if value > 0.5:  # Match
                e_val = (p_alt / p_null) if p_alt > p_null else 1.0
            else:  # Mismatch
                e_val = ((1 - p_alt) / (1 - p_null)) if p_alt < p_null else 1.0
            
            return max(e_val, 1.0)
        
        else:  # DISTANCE test
            # Gaussian e-value with empirical variance
            if self.variance <= 0:
                return 1.0
            
            # Null: high distance (different)
            # Alt: low distance (same)
            z_score = (self.current_threshold - value) / (self.std + 1e-10)
            
            # Convert to e-value using mixture approach
            e_val = math.exp(z_score - z_score**2 / 2)
            
            return max(e_val, 1.0)
    
    def _update_llr(self, value: float) -> None:
        """Update log-likelihood ratio for SPRT."""
        if self.test_type == TestType.MATCH:
            # Bernoulli LLR
            p0 = 0.5  # Null
            p1 = 0.9  # Alternative (high match rate for SAME)
            
            if value > 0.5:  # Match
                llr_increment = math.log(p1 / p0)
            else:
                llr_increment = math.log((1 - p1) / (1 - p0))
            
            self.log_likelihood_ratio += llr_increment
        
        else:  # DISTANCE test
            # Gaussian LLR with known variance
            if self.variance <= 0:
                return
            
            # H0: μ = threshold (different)
            # H1: μ = 0 (same)
            llr_increment = (value * self.current_threshold - self.current_threshold**2 / 2) / self.variance
            self.log_likelihood_ratio += llr_increment
    
    def _update_adaptive_threshold(self) -> None:
        """Update threshold adaptively based on observed variance."""
        if not self.adaptive_threshold or self.n < 10:
            return
        
        # Use empirical quantiles for robust threshold estimation
        recent = list(self.history)[-100:]  # Last 100 observations
        if len(recent) < 10:
            return
        
        # Set threshold at empirical quantile
        q25 = np.percentile(recent, 25)
        q75 = np.percentile(recent, 75)
        iqr = q75 - q25
        
        # Adaptive threshold: median + k * IQR
        median = np.median(recent)
        k = 0.5  # Tuning parameter
        
        new_threshold = median + k * iqr
        
        # Smooth update
        alpha_smooth = 0.1
        self.current_threshold = (1 - alpha_smooth) * self.current_threshold + alpha_smooth * new_threshold
        self.threshold_history.append(self.current_threshold)
    
    def _update_confidence_intervals(self) -> None:
        """Update confidence intervals for the mean."""
        if self.n < 2:
            self.ci_lower.append(0.0)
            self.ci_upper.append(1.0)
            return
        
        # Use t-distribution for small samples
        confidence_level = 1 - self.alpha
        
        if self.n < 30:
            # t-distribution
            df = self.n - 1
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
            margin = t_crit * self.std / math.sqrt(self.n)
        else:
            # Normal approximation
            z_crit = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_crit * self.std / math.sqrt(self.n)
        
        # Anytime-valid adjustment using e-values
        radius = self.confidence_seq.get_current_radius()
        margin = min(margin, radius)
        
        self.ci_lower.append(max(self.mean - margin, 0))
        self.ci_upper.append(min(self.mean + margin, 1))
    
    def get_confidence(self) -> float:
        """Get current confidence level."""
        if not self.confidence_seq.confidence_levels:
            return 0.0
        return self.confidence_seq.confidence_levels[-1]
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """Get current confidence interval for the mean."""
        if not self.ci_lower:
            return (0.0, 1.0)
        return (self.ci_lower[-1], self.ci_upper[-1])
    
    def get_confidence_radius(self) -> float:
        """Get current confidence radius."""
        return self.confidence_seq.get_current_radius()
    
    def get_match_rate(self) -> float:
        """Get current match rate (for MATCH test)."""
        if self.n == 0:
            return 0.5
        return self.n_matches / self.n
    
    def get_below_threshold_rate(self) -> float:
        """Get rate of observations below threshold (for DISTANCE test)."""
        if self.n == 0:
            return 0.5
        return self.below_threshold_count / self.n
    
    def should_stop(self) -> bool:
        """Check if we should stop based on SPRT boundaries."""
        return (self.log_likelihood_ratio >= self.upper_boundary or 
                self.log_likelihood_ratio <= self.lower_boundary)
    
    def get_decision(self) -> Verdict:
        """
        Get current decision based on accumulated evidence.
        
        Returns:
            Current verdict (SAME/DIFFERENT/UNCERTAIN/UNDECIDED)
        """
        # Check SPRT boundaries
        if self.log_likelihood_ratio >= self.upper_boundary:
            verdict = Verdict.SAME
        elif self.log_likelihood_ratio <= self.lower_boundary:
            verdict = Verdict.DIFFERENT
        elif self.n < 10:
            verdict = Verdict.UNDECIDED
        else:
            # Use confidence intervals for uncertain region
            ci_lower, ci_upper = self.get_confidence_interval()
            
            if self.test_type == TestType.MATCH:
                if ci_lower > 0.7:  # High match rate
                    verdict = Verdict.SAME
                elif ci_upper < 0.3:  # Low match rate
                    verdict = Verdict.DIFFERENT
                else:
                    verdict = Verdict.UNCERTAIN
            else:  # DISTANCE
                if ci_upper < self.current_threshold * 0.5:  # Very low distance
                    verdict = Verdict.SAME
                elif ci_lower > self.current_threshold * 1.5:  # Very high distance
                    verdict = Verdict.DIFFERENT
                else:
                    verdict = Verdict.UNCERTAIN
        
        # Track decision history
        if self.n > 0 and (not self.decision_history or 
                           self.decision_history[-1][1] != verdict):
            self.decision_history.append((self.n, verdict))
        
        return verdict
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current state."""
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "confidence": self.get_confidence(),
            "confidence_interval": self.get_confidence_interval(),
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "e_value_product": self.e_value_product,
            "verdict": self.get_decision().value,
            "match_rate": self.get_match_rate() if self.test_type == TestType.MATCH else None,
            "below_threshold_rate": self.get_below_threshold_rate() if self.test_type == TestType.DISTANCE else None,
            "first_divergence": self.first_divergence_site,
            "num_divergences": len(self.divergence_sites),
            "current_threshold": self.current_threshold,
            "should_stop": self.should_stop()
        }


@dataclass
class DualSequentialTest:
    """
    Dual sequential test combining Bernoulli and distance evidence.
    
    Implements the algorithm from Section 5.7 with both S_match and S_dist tests.
    """
    S_match: SequentialState  # Bernoulli test
    S_dist: SequentialState   # Distance test
    combined_verdict: Verdict = Verdict.UNDECIDED
    stopping_time: Optional[int] = None
    
    def update(self, match_indicator: float, distance: float, threshold: float) -> None:
        """Update both tests with new observation."""
        self.S_match.update(match_indicator, is_match=match_indicator > 0.5)
        self.S_dist.update_distance(distance, threshold)
        
        # Update combined verdict
        self._update_combined_verdict()
    
    def _update_combined_verdict(self) -> None:
        """Combine evidence from both tests."""
        match_verdict = self.S_match.get_decision()
        dist_verdict = self.S_dist.get_decision()
        
        # Conservative combination: both must agree
        if match_verdict == Verdict.SAME and dist_verdict == Verdict.SAME:
            self.combined_verdict = Verdict.SAME
            if self.stopping_time is None:
                self.stopping_time = self.S_match.n
        elif match_verdict == Verdict.DIFFERENT or dist_verdict == Verdict.DIFFERENT:
            self.combined_verdict = Verdict.DIFFERENT
            if self.stopping_time is None:
                self.stopping_time = self.S_match.n
        elif match_verdict == Verdict.UNDECIDED or dist_verdict == Verdict.UNDECIDED:
            self.combined_verdict = Verdict.UNDECIDED
        else:
            self.combined_verdict = Verdict.UNCERTAIN
    
    def should_stop(self) -> bool:
        """Check if either test has reached a stopping boundary."""
        return self.S_match.should_stop() or self.S_dist.should_stop()


def compute_e_value(
    state: SequentialState,
    null_mean: float = 0.5,
    alt_mean: Optional[float] = None
) -> float:
    """
    Compute e-value for given state.
    
    Args:
        state: Current sequential state
        null_mean: Mean under null hypothesis
        alt_mean: Mean under alternative (uses observed if None)
        
    Returns:
        E-value for current evidence
    """
    if state.n == 0:
        return 1.0
    
    if alt_mean is None:
        alt_mean = state.mean
    
    if state.test_type == TestType.MATCH:
        # Binomial e-value
        n_success = state.n_matches
        n_fail = state.n_mismatches
        
        # Likelihood under alternative
        l_alt = (alt_mean ** n_success) * ((1 - alt_mean) ** n_fail)
        
        # Likelihood under null
        l_null = (null_mean ** n_success) * ((1 - null_mean) ** n_fail)
        
        if l_null <= 0:
            return float('inf')
        
        return l_alt / l_null
    
    else:  # DISTANCE
        # Gaussian e-value
        if state.variance <= 0:
            return 1.0
        
        # Standardized difference
        z = (null_mean - alt_mean) * math.sqrt(state.n) / state.std
        
        # E-value using mixture method
        e_val = math.exp(z * state.mean * math.sqrt(state.n) / state.std - z**2 / 2)
        
        return max(e_val, 1.0)


def init_seq_test(
    alpha: float = 0.01,
    test_type: TestType = TestType.MATCH,
    adaptive: bool = True
) -> SequentialState:
    """
    Initialize sequential test with given parameters.
    
    Args:
        alpha: Significance level
        test_type: Type of test
        adaptive: Whether to use adaptive thresholds
        
    Returns:
        Initialized sequential state
    """
    return SequentialState(
        test_type=test_type,
        alpha=alpha,
        beta=alpha,  # Symmetric error rates
        adaptive_threshold=adaptive
    )


def accept_same(S_match: SequentialState, S_dist: SequentialState) -> bool:
    """
    Check if we should accept SAME hypothesis.
    
    Args:
        S_match: Bernoulli test state
        S_dist: Distance test state
        
    Returns:
        True if evidence supports SAME verdict
    """
    # Check if sufficient evidence
    if S_match.n < 5:
        return False
    
    match_same = S_match.log_likelihood_ratio >= S_match.upper_boundary
    dist_same = S_dist.log_likelihood_ratio >= S_dist.upper_boundary
    
    # Check: both tests should indicate SAME or strong evidence
    high_match_rate = S_match.get_match_rate() > 0.85
    low_distance = S_dist.mean < S_dist.current_threshold * 0.8
    
    return (match_same and dist_same) or (match_same and low_distance) or (dist_same and high_match_rate)


def accept_diff(S_match: SequentialState, S_dist: SequentialState) -> bool:
    """
    Check if we should accept DIFFERENT hypothesis.
    
    Args:
        S_match: Bernoulli test state
        S_dist: Distance test state
        
    Returns:
        True if evidence supports DIFFERENT verdict
    """
    # Check if sufficient evidence
    if S_match.n < 5:
        return False
        
    match_diff = S_match.log_likelihood_ratio <= S_match.lower_boundary
    dist_diff = S_dist.log_likelihood_ratio <= S_dist.lower_boundary
    
    # More stringent check for DIFFERENT: need strong evidence
    very_low_match = S_match.get_match_rate() < 0.2
    very_high_dist = S_dist.mean > S_dist.current_threshold * 2.0
    
    return (match_diff and dist_diff) or (very_low_match and very_high_dist)


def sequential_decision(
    stream: Generator[Dict[str, Any], None, None],
    alpha: float = 0.01,
    beta: float = 0.01,
    d_thresh: float = 0.08,
    max_C: int = 2000,
    adaptive: bool = True,
    return_full_state: bool = False
) -> Union[Tuple[str, int, Dict], Tuple[str, int, Dict, DualSequentialTest]]:
    """
    Main sequential decision function from Section 5.7.
    
    Implements the dual sequential test with both Bernoulli (S_match)
    and distance (S_dist) components for robust model comparison.
    
    Args:
        stream: Generator yielding dicts with "I" (indicator) and "d" (distance)
        alpha: Type I error rate
        beta: Type II error rate
        d_thresh: Distance threshold
        max_C: Maximum number of comparisons
        adaptive: Whether to use adaptive thresholds
        return_full_state: Whether to return the full test state
        
    Returns:
        Tuple of (verdict, stopping_time, localization_info, [optional: test_state])
    """
    # Initialize dual sequential test
    S_match = init_seq_test(alpha, TestType.MATCH, adaptive)
    S_dist = init_seq_test(beta, TestType.DISTANCE, adaptive)
    
    dual_test = DualSequentialTest(S_match, S_dist)
    
    # Process stream
    t = 0
    verdict = "UNDECIDED"
    
    for t, r in enumerate(stream, 1):
        # Extract match indicator and distance
        I = r.get("I", 0)
        d = r.get("d", 1.0)
        
        # Update both tests
        dual_test.update(I, d, d_thresh)
        
        # Check stopping conditions
        if accept_same(S_match, S_dist):
            verdict = "SAME"
            break
        
        if accept_diff(S_match, S_dist):
            verdict = "DIFFERENT"
            break
        
        if t >= max_C:
            verdict = "UNDECIDED"
            break
    
    # Handle empty stream
    if t == 0:
        t = 1  # Minimum stopping time
    
    # Prepare localization info
    localization_info = {
        "first_divergence": S_match.first_divergence_site or S_dist.first_divergence_site,
        "divergence_sites": sorted(set(S_match.divergence_sites + S_dist.divergence_sites))[:10],
        "match_trajectory": S_match.match_trajectory[-10:] if S_match.match_trajectory else [],
        "match_rate": S_match.get_match_rate(),
        "mean_distance": S_dist.mean,
        "confidence_match": S_match.get_confidence(),
        "confidence_dist": S_dist.get_confidence(),
        "samples_used": t
    }
    
    if return_full_state:
        return verdict, t, localization_info, dual_test
    else:
        return verdict, t, localization_info


class HybridSequentialTest:
    """
    Advanced hybrid sequential test combining multiple evidence types.
    
    Supports multi-hypothesis testing with uncertain regions and
    provides comprehensive analysis capabilities.
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.01,
        enable_multi_hypothesis: bool = True,
        power_analysis: bool = True
    ):
        """
        Initialize hybrid sequential test.
        
        Args:
            alpha: Type I error rate
            beta: Type II error rate
            enable_multi_hypothesis: Support for UNCERTAIN verdict
            power_analysis: Enable power analysis features
        """
        self.alpha = alpha
        self.beta = beta
        self.enable_multi_hypothesis = enable_multi_hypothesis
        
        # Initialize component tests
        self.match_test = SequentialState(TestType.MATCH, alpha, beta)
        self.distance_test = SequentialState(TestType.DISTANCE, alpha, beta)
        
        # Power analysis
        self.power_analyzer = PowerAnalysis() if power_analysis else None
        self.expected_n = None
        self.current_power = 0.0
        
        # Multi-hypothesis boundaries
        self.uncertain_lower = -math.log(2)  # Log(0.5)
        self.uncertain_upper = math.log(2)   # Log(2)
        
        # Results tracking
        self.verdict_history: List[Tuple[int, Verdict]] = []
        self.confidence_history: List[float] = []
    
    def update(
        self,
        match_indicator: float,
        distance: float,
        threshold: float = 0.1
    ) -> Verdict:
        """
        Update with new observation and return current verdict.
        
        Args:
            match_indicator: Binary match indicator (0 or 1)
            distance: Continuous distance measure
            threshold: Distance threshold
            
        Returns:
            Current verdict
        """
        # Update component tests
        self.match_test.update(match_indicator, is_match=match_indicator > 0.5)
        self.distance_test.update_distance(distance, threshold)
        
        # Update power analysis
        if self.power_analyzer:
            self._update_power_analysis()
        
        # Get combined verdict
        verdict = self._get_combined_verdict()
        
        # Track history
        n = self.match_test.n
        if not self.verdict_history or self.verdict_history[-1][1] != verdict:
            self.verdict_history.append((n, verdict))
        
        confidence = (self.match_test.get_confidence() + 
                     self.distance_test.get_confidence()) / 2
        self.confidence_history.append(confidence)
        
        return verdict
    
    def _get_combined_verdict(self) -> Verdict:
        """Get combined verdict with multi-hypothesis support."""
        match_llr = self.match_test.log_likelihood_ratio
        dist_llr = self.distance_test.log_likelihood_ratio
        
        # Combined log-likelihood ratio (weighted average)
        weight_match = 0.6  # Can be tuned
        weight_dist = 0.4
        combined_llr = weight_match * match_llr + weight_dist * dist_llr
        
        # Multi-hypothesis testing
        if self.enable_multi_hypothesis:
            if combined_llr >= self.match_test.upper_boundary:
                return Verdict.SAME
            elif combined_llr <= self.match_test.lower_boundary:
                return Verdict.DIFFERENT
            elif self.uncertain_lower <= combined_llr <= self.uncertain_upper:
                return Verdict.UNCERTAIN
            else:
                return Verdict.UNDECIDED
        else:
            # Standard binary decision
            if combined_llr >= self.match_test.upper_boundary:
                return Verdict.SAME
            elif combined_llr <= self.match_test.lower_boundary:
                return Verdict.DIFFERENT
            else:
                return Verdict.UNDECIDED
    
    def _update_power_analysis(self) -> None:
        """Update power analysis based on observed effect size."""
        if self.match_test.n < 10:
            return
        
        # Estimate effect size from data
        match_effect = abs(self.match_test.mean - 0.5)
        dist_effect = abs(self.distance_test.mean - self.distance_test.current_threshold) / (
            self.distance_test.std + 1e-10)
        
        avg_effect = (match_effect + dist_effect) / 2
        
        # Update expected sample size
        self.expected_n = self.power_analyzer.compute_expected_sample_size(
            self.alpha, self.beta, avg_effect, TestType.HYBRID
        )
        
        # Update current power
        self.current_power = self.power_analyzer.compute_power(
            self.match_test.n, self.alpha, avg_effect, TestType.HYBRID
        )
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of current state."""
        analysis = {
            "verdict": self._get_combined_verdict().value,
            "n_samples": self.match_test.n,
            "match_summary": self.match_test.get_summary(),
            "distance_summary": self.distance_test.get_summary(),
            "verdict_history": [
                {"n": n, "verdict": v.value} 
                for n, v in self.verdict_history
            ],
            "average_confidence": np.mean(self.confidence_history) if self.confidence_history else 0.0
        }
        
        # Add power analysis if enabled
        if self.power_analyzer:
            analysis["power_analysis"] = {
                "expected_n": self.expected_n,
                "current_power": self.current_power,
                "efficiency": self.match_test.n / self.expected_n if self.expected_n else None
            }
        
        return analysis
    
    def should_stop(self) -> bool:
        """Check if test should stop."""
        verdict = self._get_combined_verdict()
        return verdict in [Verdict.SAME, Verdict.DIFFERENT]
    
    def reset(self) -> None:
        """Reset test state for new comparison."""
        self.match_test = SequentialState(TestType.MATCH, self.alpha, self.beta)
        self.distance_test = SequentialState(TestType.DISTANCE, self.alpha, self.beta)
        self.verdict_history.clear()
        self.confidence_history.clear()
        self.expected_n = None
        self.current_power = 0.0


# Utility functions for integration

def create_sequential_tester(
    test_type: str = "dual",
    alpha: float = 0.01,
    beta: float = 0.01,
    **kwargs
) -> Union[SequentialState, DualSequentialTest, HybridSequentialTest]:
    """
    Factory function to create appropriate sequential tester.
    
    Args:
        test_type: Type of test ("single", "dual", "hybrid")
        alpha: Type I error rate
        beta: Type II error rate
        **kwargs: Additional parameters
        
    Returns:
        Configured sequential tester
    """
    if test_type == "single":
        return SequentialState(
            test_type=kwargs.get("evidence_type", TestType.MATCH),
            alpha=alpha,
            beta=beta,
            adaptive_threshold=kwargs.get("adaptive", True)
        )
    elif test_type == "dual":
        S_match = init_seq_test(alpha, TestType.MATCH, kwargs.get("adaptive", True))
        S_dist = init_seq_test(beta, TestType.DISTANCE, kwargs.get("adaptive", True))
        return DualSequentialTest(S_match, S_dist)
    elif test_type == "hybrid":
        return HybridSequentialTest(
            alpha=alpha,
            beta=beta,
            enable_multi_hypothesis=kwargs.get("multi_hypothesis", True),
            power_analysis=kwargs.get("power_analysis", True)
        )
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def analyze_sequential_results(
    test: Union[SequentialState, DualSequentialTest, HybridSequentialTest]
) -> Dict[str, Any]:
    """
    Analyze results from sequential testing.
    
    Args:
        test: Completed sequential test
        
    Returns:
        Comprehensive analysis dictionary
    """
    if isinstance(test, SequentialState):
        return test.get_summary()
    elif isinstance(test, DualSequentialTest):
        return {
            "verdict": test.combined_verdict.value,
            "stopping_time": test.stopping_time,
            "match_analysis": test.S_match.get_summary(),
            "distance_analysis": test.S_dist.get_summary()
        }
    elif isinstance(test, HybridSequentialTest):
        return test.get_analysis()
    else:
        return {"error": "Unknown test type"}