"""Anytime-valid confidence sequence boundaries for sequential decision making."""

import math
import numpy as np
from typing import Literal, Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass, field
from enum import Enum


class VerificationMode(Enum):
    """Verification mode for boundary calculations."""
    SEQUENTIAL = "sequential"  # REV-style sequential testing
    CONSENSUS = "consensus"    # HBT-style consensus verification
    UNIFIED = "unified"        # Combined REV+HBT approach


@dataclass
class CSState:
    """
    State for confidence sequence computation using Welford's online algorithm.
    
    Maintains running statistics for bounded values Z ∈ [0,1].
    Uses Welford's algorithm for numerically stable online mean/variance computation.
    """
    
    n: int = 0                    # Number of samples seen
    mean: float = 0.0             # Running mean
    M2: float = 0.0               # Sum of squared deviations (for variance)
    min_val: float = float('inf') # Minimum value seen
    max_val: float = float('-inf')# Maximum value seen
    sum_val: float = 0.0          # Sum of all values
    
    def update(self, z: float) -> None:
        """
        Update statistics with a new observation using Welford's algorithm.
        
        Args:
            z: New observation value (should be in [0,1])
        """
        if not (0.0 <= z <= 1.0):
            raise ValueError(f"Value {z} not in [0,1]")
        
        self.n += 1
        delta = z - self.mean
        self.mean += delta / self.n
        delta2 = z - self.mean
        self.M2 += delta * delta2
        
        # Update additional statistics
        self.min_val = min(self.min_val, z)
        self.max_val = max(self.max_val, z)
        self.sum_val += z
    
    @property
    def variance(self) -> float:
        """Compute sample variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        """Compute sample standard deviation."""
        return math.sqrt(self.variance)
    
    @property
    def empirical_variance(self) -> float:
        """
        Compute empirical variance for confidence sequences.
        Uses unbiased estimator with Bessel's correction.
        """
        if self.n < 1:
            return 0.0
        # For bounded [0,1] values, we can use the sample variance
        # but cap it at 0.25 (maximum possible variance for [0,1])
        return min(self.variance, 0.25)
    
    def copy(self) -> 'CSState':
        """Create a copy of the current state."""
        return CSState(
            n=self.n,
            mean=self.mean,
            M2=self.M2,
            min_val=self.min_val,
            max_val=self.max_val,
            sum_val=self.sum_val
        )


def log_log_correction(t: int, alpha: float) -> float:
    """
    Implement the log(log(t)) correction factor for anytime validity.
    
    This correction ensures the confidence sequence remains valid
    at all stopping times (anytime-valid property).
    
    Args:
        t: Current time/sample count (must be >= 1)
        alpha: Significance level (e.g., 0.05 for 95% confidence)
    
    Returns:
        The log-log correction factor: log(log(max(e, t)) / α)
    """
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    # Use max(e, t) to ensure log(log(t)) >= 0
    # Since log(e) = 1, log(log(e)) = log(1) = 0
    e = math.e
    t_corrected = max(e, t)
    
    # Compute log(log(t) / α)
    log_log_term = math.log(math.log(t_corrected) / alpha)
    
    return log_log_term


def eb_radius(state: CSState, alpha: float, c: float = 1.0) -> float:
    """
    Compute Empirical-Bernstein confidence radius with log-log term.
    
    The EB radius formula:
    r_t(α) = sqrt(2 * σ²_t * log(log(t) / α) / t) + c * log(log(t) / α) / t
    
    Where:
    - σ²_t is the empirical variance estimate at time t
    - The log-log term ensures anytime validity
    - c is a constant (typically 1.0) for the bias/concentration term
    
    Args:
        state: Current CSState with running statistics
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        c: Constant for bias term (default 1.0, can be tuned for tightness)
    
    Returns:
        Confidence radius for the empirical mean
    """
    if state.n < 1:
        return float('inf')
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    t = state.n
    sigma_squared_t = state.empirical_variance
    
    # Get log-log correction factor
    log_log_term = log_log_correction(t, alpha)
    
    # Compute radius components according to paper formula
    # First term: sqrt(2 * σ²_t * log(log(t) / α) / t)
    variance_term = math.sqrt(2 * sigma_squared_t * log_log_term / t)
    
    # Second term: c * log(log(t) / α) / t
    bias_term = c * log_log_term / t
    
    # Total radius
    radius = variance_term + bias_term
    
    return radius


def eb_confidence_interval(mean: float, variance: float, n: int, alpha: float, c: float = 1.0) -> Tuple[float, float]:
    """
    Return (lower, upper) confidence bounds using Empirical-Bernstein radius.
    
    Args:
        mean: Sample mean of observations
        variance: Sample variance
        n: Number of samples
        alpha: Significance level
        c: Constant for bias term in EB radius
    
    Returns:
        (lower, upper): Tuple of confidence bounds, clipped to [0, 1]
    """
    if n < 1:
        return (0.0, 1.0)  # Uninformative interval
    
    if not (0.0 <= mean <= 1.0):
        raise ValueError(f"Mean must be in [0,1], got {mean}")
    if variance < 0:
        raise ValueError(f"Variance must be non-negative, got {variance}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    # Create a temporary state for radius computation
    temp_state = CSState()
    temp_state.n = n
    temp_state.mean = mean
    # Set M2 such that variance = M2 / (n - 1) matches input variance
    # But cap variance at 0.25 (max for [0,1] bounded variables)
    capped_variance = min(variance, 0.25)
    if n > 1:
        temp_state.M2 = capped_variance * (n - 1)
    else:
        temp_state.M2 = 0.0
    
    # Compute EB radius
    radius = eb_radius(temp_state, alpha, c)
    
    # Compute confidence bounds
    lower = mean - radius
    upper = mean + radius
    
    # Clip to [0, 1] since we know values are bounded
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    
    return (lower, upper)


def decide_one_sided(
    state: CSState,
    threshold: float,
    alpha: float,
    hypothesis: Literal["H0", "H1"] = "H0"
) -> Literal["accept_id", "reject_id", "continue"]:
    """
    Make sequential decision based on confidence sequence.
    
    For one-sided testing:
    - H0: μ ≤ threshold (null hypothesis: identity/genuine model)
    - H1: μ > threshold (alternative: different/adversarial model)
    
    Args:
        state: Current CSState with observations
        threshold: Decision threshold (boundary value)
        alpha: Significance level for confidence sequence
        hypothesis: Which hypothesis we're testing ("H0" or "H1")
    
    Returns:
        "accept_id": Accept identity (H0 true, model is genuine)
        "reject_id": Reject identity (H1 true, model is adversarial)
        "continue": Need more samples to decide
    """
    if state.n < 1:
        return "continue"
    
    # Get confidence radius
    radius = eb_radius(state, alpha)
    
    # Compute confidence bounds
    lower_bound = state.mean - radius
    upper_bound = state.mean + radius
    
    # Ensure bounds respect [0,1] constraint
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)
    
    if hypothesis == "H0":
        # Testing H0: μ ≤ threshold
        # Reject H0 if lower bound > threshold (entire CI above threshold)
        if lower_bound > threshold:
            return "reject_id"  # Strong evidence against H0
        # Accept H0 if upper bound ≤ threshold (entire CI below/at threshold)
        elif upper_bound <= threshold:
            return "accept_id"  # Strong evidence for H0
        else:
            return "continue"  # CI contains threshold, need more data
    
    elif hypothesis == "H1":
        # Testing H1: μ > threshold
        # Accept H1 (reject H0) if lower bound > threshold
        if lower_bound > threshold:
            return "reject_id"  # Strong evidence for H1
        # Reject H1 (accept H0) if upper bound ≤ threshold
        elif upper_bound <= threshold:
            return "accept_id"  # Strong evidence against H1
        else:
            return "continue"  # CI contains threshold, need more data
    
    else:
        raise ValueError(f"hypothesis must be 'H0' or 'H1', got {hypothesis}")


class SequentialTest:
    """
    Wrapper class for sequential testing with confidence sequences.
    
    Maintains state and provides high-level interface for sequential decisions.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        alpha: float = 0.05,
        max_samples: Optional[int] = None,
        hypothesis: Literal["H0", "H1"] = "H0"
    ):
        """
        Initialize sequential test.
        
        Args:
            threshold: Decision threshold
            alpha: Significance level
            max_samples: Maximum samples before forced decision
            hypothesis: Which hypothesis to test
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_samples = max_samples
        self.hypothesis = hypothesis
        self.state = CSState()
        self.decision_history = []
    
    def update(self, z: float) -> Literal["accept_id", "reject_id", "continue"]:
        """
        Update with new observation and return decision.
        
        Args:
            z: New observation in [0,1]
        
        Returns:
            Current decision
        """
        self.state.update(z)
        
        # Check if we've reached max samples
        if self.max_samples and self.state.n >= self.max_samples:
            # Forced decision based on mean
            if self.state.mean <= self.threshold:
                decision = "accept_id"
            else:
                decision = "reject_id"
        else:
            decision = decide_one_sided(
                self.state, self.threshold, self.alpha, self.hypothesis
            )
        
        self.decision_history.append({
            'n': self.state.n,
            'mean': self.state.mean,
            'radius': eb_radius(self.state, self.alpha),
            'decision': decision
        })
        
        return decision
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """Get current confidence interval."""
        if self.state.n < 1:
            return (0.0, 1.0)
        
        radius = eb_radius(self.state, self.alpha)
        lower = max(0.0, self.state.mean - radius)
        upper = min(1.0, self.state.mean + radius)
        return (lower, upper)
    
    def reset(self):
        """Reset test state."""
        self.state = CSState()
        self.decision_history = []


class EnhancedStatisticalFramework:
    """
    Enhanced statistical framework combining REV's Empirical-Bernstein bounds
    with HBT's variance tensor analysis for unified confidence computation.
    """
    
    def __init__(
        self,
        mode: VerificationMode = VerificationMode.UNIFIED,
        rev_weight: float = 0.6,
        hbt_weight: float = 0.4,
        alpha: float = 0.05,
        beta: float = 0.10
    ):
        """
        Initialize enhanced statistical framework.
        
        Args:
            mode: Verification mode (sequential, consensus, or unified)
            rev_weight: Weight for REV confidence (default 0.6)
            hbt_weight: Weight for HBT confidence (default 0.4)
            alpha: Type I error rate
            beta: Type II error rate
        """
        self.mode = mode
        self.rev_weight = rev_weight
        self.hbt_weight = hbt_weight
        self.alpha = alpha
        self.beta = beta
        
        # Validate weights
        if abs(rev_weight + hbt_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {rev_weight + hbt_weight}")
        
        # State tracking
        self.cs_state = CSState()
        self.variance_history: List[np.ndarray] = []
        self.confidence_history: List[float] = []
        
        # Adaptive parameters based on mode
        self._setup_adaptive_parameters()
    
    def _setup_adaptive_parameters(self):
        """Setup adaptive alpha/beta based on verification mode."""
        if self.mode == VerificationMode.SEQUENTIAL:
            # REV mode: tighter sequential bounds
            self.adaptive_alpha = self.alpha
            self.adaptive_beta = self.beta
            self.c_factor = 1.0  # Standard EB constant
        
        elif self.mode == VerificationMode.CONSENSUS:
            # HBT mode: looser bounds for consensus
            self.adaptive_alpha = self.alpha * 1.5
            self.adaptive_beta = self.beta * 1.5
            self.c_factor = 0.8  # Reduced bias term
        
        else:  # UNIFIED
            # Balanced parameters
            self.adaptive_alpha = self.alpha * 1.2
            self.adaptive_beta = self.beta * 1.2
            self.c_factor = 0.9
    
    def compute_unified_confidence(
        self,
        state: Optional[CSState] = None,
        variance_tensor: Optional[np.ndarray] = None,
        segment_signatures: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute unified confidence combining REV's EB bounds and HBT's variance analysis.
        
        Args:
            state: Optional CSState for EB calculation (uses internal if None)
            variance_tensor: Optional variance tensor from HBT encoding
            segment_signatures: Optional segment signatures for variance extraction
            
        Returns:
            Tuple of (unified_confidence, confidence_components)
            where confidence_components contains individual scores
        """
        if state is None:
            state = self.cs_state
        
        components = {}
        
        # 1. Calculate REV's Empirical-Bernstein confidence
        if state.n > 0:
            eb_conf = self._compute_eb_confidence(state)
            components['eb_confidence'] = eb_conf
        else:
            eb_conf = 0.5  # Neutral confidence
            components['eb_confidence'] = eb_conf
        
        # 2. Calculate HBT's variance tensor confidence
        if variance_tensor is not None:
            var_conf = self.analyze_variance_tensor(variance_tensor)
            components['variance_confidence'] = var_conf
        elif segment_signatures is not None:
            # Extract variance from signatures
            variance_tensor = self._extract_variance_from_signatures(segment_signatures)
            var_conf = self.analyze_variance_tensor(variance_tensor)
            components['variance_confidence'] = var_conf
        else:
            var_conf = 0.5  # Neutral confidence
            components['variance_confidence'] = var_conf
        
        # 3. Combine with weights
        if self.mode == VerificationMode.SEQUENTIAL:
            # REV mode: primarily use EB confidence
            unified_confidence = eb_conf
        elif self.mode == VerificationMode.CONSENSUS:
            # HBT mode: primarily use variance confidence
            unified_confidence = var_conf
        else:  # UNIFIED
            # Weighted combination
            unified_confidence = (
                self.rev_weight * eb_conf +
                self.hbt_weight * var_conf
            )
        
        # Store in history
        self.confidence_history.append(unified_confidence)
        components['unified'] = unified_confidence
        
        return unified_confidence, components
    
    def _compute_eb_confidence(self, state: CSState) -> float:
        """
        Compute confidence from Empirical-Bernstein bounds.
        
        Args:
            state: CSState with observations
            
        Returns:
            Confidence score in [0, 1]
        """
        if state.n < 1:
            return 0.5
        
        # Get EB radius
        radius = eb_radius(state, self.adaptive_alpha, self.c_factor)
        
        # Convert radius to confidence
        # Smaller radius = higher confidence
        # Normalize by maximum possible radius (0.5 for [0,1] bounded)
        max_radius = 0.5
        normalized_radius = min(radius, max_radius) / max_radius
        
        # Confidence is inverse of normalized radius
        confidence = 1.0 - normalized_radius
        
        # Apply sample size adjustment
        # More samples = higher confidence
        sample_factor = min(1.0, state.n / 100.0)  # Saturates at 100 samples
        adjusted_confidence = confidence * (0.5 + 0.5 * sample_factor)
        
        return min(1.0, max(0.0, adjusted_confidence))
    
    def analyze_variance_tensor(
        self,
        variance_tensor: Union[np.ndarray, List[float]],
        stability_window: int = 10
    ) -> float:
        """
        Analyze variance tensor to compute confidence based on stability.
        
        Args:
            variance_tensor: Variance data from segment signatures
            stability_window: Window size for stability analysis
            
        Returns:
            Normalized confidence score in [0, 1]
        """
        if isinstance(variance_tensor, list):
            variance_tensor = np.array(variance_tensor)
        
        if len(variance_tensor) == 0:
            return 0.5  # Neutral confidence
        
        # Store in history
        self.variance_history.append(variance_tensor)
        
        # 1. Compute variance stability over time
        if len(self.variance_history) >= stability_window:
            recent_variances = self.variance_history[-stability_window:]
            
            # Calculate coefficient of variation for stability
            var_means = [np.mean(v) for v in recent_variances]
            var_stds = [np.std(v) for v in recent_variances]
            
            mean_of_means = np.mean(var_means)
            std_of_means = np.std(var_means)
            
            if mean_of_means > 1e-6:
                cv = std_of_means / mean_of_means
                # Lower CV = more stable = higher confidence
                stability_score = np.exp(-cv)  # Exponential decay
            else:
                stability_score = 1.0  # Very low variance is stable
        else:
            # Not enough history for stability analysis
            stability_score = 0.5 + 0.05 * len(self.variance_history)
        
        # 2. Analyze current variance magnitude
        current_var_mean = np.mean(variance_tensor)
        current_var_std = np.std(variance_tensor)
        
        # Lower variance = higher confidence
        # Normalize by expected variance range [0, 0.25] for bounded [0,1] values
        magnitude_score = 1.0 - min(1.0, current_var_mean / 0.25)
        
        # 3. Check for variance convergence
        if len(self.variance_history) > 1:
            # Compare current to previous
            prev_var = self.variance_history[-2]
            var_change = np.abs(np.mean(variance_tensor) - np.mean(prev_var))
            
            # Small change = convergence = higher confidence
            convergence_score = np.exp(-10 * var_change)  # Sensitive to small changes
        else:
            convergence_score = 0.5
        
        # 4. Combine scores with weights
        confidence = (
            0.4 * stability_score +
            0.3 * magnitude_score +
            0.3 * convergence_score
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_variance_from_signatures(
        self,
        segment_signatures: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract variance tensor from segment signatures.
        
        Args:
            segment_signatures: Dictionary of signatures
            
        Returns:
            Variance tensor
        """
        variances = []
        
        for sig_name, signature in segment_signatures.items():
            if isinstance(signature, np.ndarray):
                # Compute variance of signature components
                sig_var = np.var(signature)
                variances.append(sig_var)
            elif hasattr(signature, '__iter__'):
                # Handle list or other iterables
                sig_var = np.var(list(signature))
                variances.append(sig_var)
        
        return np.array(variances) if variances else np.array([0.0])
    
    def update_boundaries(
        self,
        observation: float,
        variance_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Update boundaries with new observation and optional variance data.
        
        Args:
            observation: New observation value in [0, 1]
            variance_data: Optional variance tensor
            
        Returns:
            Dictionary with updated boundaries and confidence
        """
        # Update CS state
        self.cs_state.update(observation)
        
        # Compute unified confidence
        unified_conf, components = self.compute_unified_confidence(
            variance_tensor=variance_data
        )
        
        # Get confidence interval
        radius = eb_radius(self.cs_state, self.adaptive_alpha, self.c_factor)
        lower = max(0.0, self.cs_state.mean - radius)
        upper = min(1.0, self.cs_state.mean + radius)
        
        return {
            'mean': self.cs_state.mean,
            'variance': self.cs_state.variance,
            'confidence_interval': (lower, upper),
            'radius': radius,
            'unified_confidence': unified_conf,
            'confidence_components': components,
            'n_samples': self.cs_state.n,
            'mode': self.mode.value
        }
    
    def make_decision(
        self,
        threshold: float = 0.5,
        min_samples: int = 10
    ) -> Tuple[Literal["accept", "reject", "continue"], float]:
        """
        Make decision based on current boundaries and confidence.
        
        Args:
            threshold: Decision threshold
            min_samples: Minimum samples before making decision
            
        Returns:
            Tuple of (decision, confidence)
        """
        if self.cs_state.n < min_samples:
            return "continue", 0.0
        
        # Get unified confidence
        unified_conf, _ = self.compute_unified_confidence()
        
        # Get confidence bounds
        radius = eb_radius(self.cs_state, self.adaptive_alpha, self.c_factor)
        lower = max(0.0, self.cs_state.mean - radius)
        upper = min(1.0, self.cs_state.mean + radius)
        
        # Make decision based on mode
        if self.mode == VerificationMode.CONSENSUS:
            # Consensus mode: require high confidence
            if unified_conf > 0.9:
                if self.cs_state.mean <= threshold:
                    return "accept", unified_conf
                else:
                    return "reject", unified_conf
        
        # Sequential or unified mode: use boundaries
        if upper <= threshold:
            return "accept", unified_conf
        elif lower > threshold:
            return "reject", unified_conf
        else:
            return "continue", unified_conf
    
    def reset(self):
        """Reset all state."""
        self.cs_state = CSState()
        self.variance_history.clear()
        self.confidence_history.clear()


# Export list for the module
__all__ = [
    "CSState",
    "VerificationMode",
    "log_log_correction",
    "eb_radius",
    "eb_confidence_interval",
    "decide_one_sided",
    "SequentialTest",
    "EnhancedStatisticalFramework"
]