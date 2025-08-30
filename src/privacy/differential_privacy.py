"""
Differential privacy mechanisms for REV model verification.

This module implements differential privacy for protecting model signatures
and activation patterns during REV verification while maintaining utility.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch


class PrivacyLevel(Enum):
    """Predefined privacy levels for REV verification"""
    LOW = (1.0, 1e-5)      # (epsilon, delta) 
    MEDIUM = (0.1, 1e-6)
    HIGH = (0.01, 1e-7)
    VERY_HIGH = (0.001, 1e-8)


@dataclass
class PrivacyBudget:
    """Privacy budget tracking for REV sessions"""
    epsilon: float
    delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    def can_spend(self, eps: float, delta: float) -> bool:
        """Check if we can spend privacy budget"""
        return (self.consumed_epsilon + eps <= self.epsilon and
                self.consumed_delta + delta <= self.delta)
    
    def spend(self, eps: float, delta: float) -> None:
        """Spend privacy budget"""
        if not self.can_spend(eps, delta):
            raise ValueError("Insufficient privacy budget")
        self.consumed_epsilon += eps
        self.consumed_delta += delta


class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanism for REV hypervector signatures.
    
    This protects model activations and signatures during verification
    by adding calibrated noise while preserving verification accuracy.
    """
    
    def __init__(
        self, 
        privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
        sensitivity: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize DP mechanism.
        
        Args:
            privacy_level: Target privacy level
            sensitivity: L2 sensitivity of the function
            seed: Random seed for reproducibility
        """
        self.epsilon, self.delta = privacy_level.value
        self.sensitivity = sensitivity
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def add_gaussian_noise(
        self, 
        data: torch.Tensor, 
        epsilon_spend: float,
        delta_spend: float
    ) -> torch.Tensor:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy.
        
        Args:
            data: Input tensor to add noise to
            epsilon_spend: Privacy budget to spend (epsilon)
            delta_spend: Privacy budget to spend (delta)
            
        Returns:
            Noisy tensor with DP guarantees
        """
        # Compute noise scale using Gaussian mechanism
        # sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / delta_spend)) * self.sensitivity / epsilon_spend
        
        # Add Gaussian noise
        noise = torch.normal(mean=0.0, std=sigma, size=data.shape)
        return data + noise
    
    def add_laplace_noise(self, data: torch.Tensor, epsilon_spend: float) -> torch.Tensor:
        """
        Add Laplace noise for epsilon-differential privacy.
        
        Args:
            data: Input tensor
            epsilon_spend: Privacy budget to spend
            
        Returns:
            Noisy tensor with pure DP guarantees
        """
        # Laplace scale parameter
        scale = self.sensitivity / epsilon_spend
        
        # Add Laplace noise
        noise = torch.from_numpy(
            np.random.laplace(scale=scale, size=data.shape)
        ).float()
        
        return data + noise
    
    def privatize_hypervector(
        self, 
        hypervector: torch.Tensor,
        privacy_budget: PrivacyBudget,
        mechanism: str = "gaussian"
    ) -> torch.Tensor:
        """
        Add DP noise to hypervector signature for REV.
        
        Args:
            hypervector: Model signature hypervector
            privacy_budget: Available privacy budget
            mechanism: "gaussian" or "laplace"
            
        Returns:
            Privatized hypervector
        """
        # Use a fraction of available budget
        eps_spend = min(0.1, privacy_budget.epsilon - privacy_budget.consumed_epsilon)
        delta_spend = min(1e-7, privacy_budget.delta - privacy_budget.consumed_delta)
        
        if eps_spend <= 0:
            raise ValueError("Insufficient epsilon budget")
        
        if mechanism == "gaussian":
            if delta_spend <= 0:
                raise ValueError("Insufficient delta budget for Gaussian mechanism")
            noisy_hv = self.add_gaussian_noise(hypervector, eps_spend, delta_spend)
            privacy_budget.spend(eps_spend, delta_spend)
        elif mechanism == "laplace":
            noisy_hv = self.add_laplace_noise(hypervector, eps_spend)
            privacy_budget.spend(eps_spend, 0.0)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        # Renormalize if needed
        if torch.norm(noisy_hv) > 0:
            noisy_hv = noisy_hv / torch.norm(noisy_hv)
        
        return noisy_hv
    
    def privatize_distance(
        self, 
        distance: float,
        privacy_budget: PrivacyBudget,
        max_distance: float = 1.0
    ) -> float:
        """
        Add DP noise to distance measurements.
        
        Args:
            distance: Original distance
            privacy_budget: Available privacy budget  
            max_distance: Maximum possible distance (for sensitivity)
            
        Returns:
            Privatized distance
        """
        # Sensitivity is max_distance for distance queries
        old_sensitivity = self.sensitivity
        self.sensitivity = max_distance
        
        # Spend small amount of budget
        eps_spend = min(0.05, privacy_budget.epsilon - privacy_budget.consumed_epsilon)
        
        if eps_spend <= 0:
            raise ValueError("Insufficient privacy budget")
        
        # Use Laplace mechanism for scalar values
        noisy_distance = self.add_laplace_noise(
            torch.tensor([distance]), eps_spend
        ).item()
        
        # Restore original sensitivity
        self.sensitivity = old_sensitivity
        
        # Clip to valid range
        noisy_distance = max(0.0, min(max_distance, noisy_distance))
        
        privacy_budget.spend(eps_spend, 0.0)
        
        return noisy_distance
    
    def compose_privacy(self, mechanisms: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Compute privacy parameters under composition.
        
        Uses advanced composition theorem for tighter bounds.
        
        Args:
            mechanisms: List of (epsilon, delta) pairs
            
        Returns:
            Composed (epsilon, delta) parameters
        """
        if not mechanisms:
            return (0.0, 0.0)
        
        # Separate pure DP and approximate DP mechanisms
        pure_dp = [(eps, d) for eps, d in mechanisms if d == 0]
        approx_dp = [(eps, d) for eps, d in mechanisms if d > 0]
        
        # Simple composition for pure DP
        pure_epsilon = sum(eps for eps, _ in pure_dp)
        
        # Advanced composition for approximate DP
        if approx_dp:
            epsilons = [eps for eps, _ in approx_dp]
            deltas = [d for _, d in approx_dp]
            
            # Use advanced composition theorem
            # For k mechanisms with (ε_i, δ_i)-DP:
            # Total is (ε', kδ + δ')-DP where ε' is from advanced composition
            k = len(approx_dp)
            base_delta = sum(deltas)
            
            # Advanced composition bound
            epsilon_sum = sum(epsilons)
            epsilon_squared_sum = sum(e**2 for e in epsilons)
            
            # For any δ' > 0, the composition is (ε', kδ + δ')-DP where:
            # ε' = sqrt(2k * ln(1/δ')) * max(ε_i) + sum(ε_i)
            # We use δ' = min(deltas) for balance
            delta_prime = min(deltas) if deltas else 1e-8
            max_epsilon = max(epsilons)
            
            advanced_epsilon = (
                np.sqrt(2 * k * np.log(1 / delta_prime)) * max_epsilon + 
                epsilon_sum
            )
            
            # Choose better of simple and advanced composition
            simple_epsilon = epsilon_sum
            total_epsilon = min(simple_epsilon, advanced_epsilon) + pure_epsilon
            total_delta = base_delta + delta_prime
        else:
            total_epsilon = pure_epsilon
            total_delta = 0.0
        
        return (total_epsilon, total_delta)


def create_privacy_budget(
    privacy_level: PrivacyLevel,
    num_queries: int = 100
) -> PrivacyBudget:
    """
    Create privacy budget for REV verification session.
    
    Args:
        privacy_level: Target privacy level
        num_queries: Expected number of queries
        
    Returns:
        Privacy budget allocated for session
    """
    epsilon, delta = privacy_level.value
    return PrivacyBudget(epsilon=epsilon, delta=delta)


def validate_privacy_parameters(
    epsilon: float, 
    delta: float,
    min_epsilon: float = 1e-6,
    max_delta: float = 1e-3
) -> bool:
    """
    Validate privacy parameters are reasonable.
    
    Args:
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        min_epsilon: Minimum acceptable epsilon
        max_delta: Maximum acceptable delta
        
    Returns:
        True if parameters are valid
    """
    return (epsilon >= min_epsilon and 
            delta >= 0 and 
            delta <= max_delta)


def estimate_utility_loss(
    epsilon: float,
    delta: float, 
    data_dimension: int,
    sensitivity: float = 1.0
) -> float:
    """
    Estimate utility loss from DP noise addition.
    
    Args:
        epsilon: Privacy parameter
        delta: Privacy parameter  
        data_dimension: Dimensionality of data
        sensitivity: L2 sensitivity
        
    Returns:
        Estimated relative utility loss
    """
    # Rough estimate based on Gaussian noise scale
    if delta > 0:
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        # Relative noise grows with sqrt(dimension)
        relative_noise = sigma * np.sqrt(data_dimension)
    else:
        # Laplace mechanism
        scale = sensitivity / epsilon
        relative_noise = scale * np.sqrt(data_dimension)
    
    return min(1.0, relative_noise)  # Cap at 100% loss


class RenyiDifferentialPrivacy:
    """
    Rényi differential privacy for tighter composition bounds.
    
    Provides better composition properties than standard DP for
    iterative algorithms like those in REV verification.
    """
    
    def __init__(self, alpha: float = 2.0):
        """
        Initialize RDP mechanism.
        
        Args:
            alpha: Rényi divergence order (alpha > 1)
        """
        if alpha <= 1:
            raise ValueError("Alpha must be > 1 for RDP")
        self.alpha = alpha
    
    def gaussian_rdp(self, sensitivity: float, sigma: float) -> float:
        """
        Compute RDP parameter for Gaussian mechanism.
        
        Args:
            sensitivity: L2 sensitivity
            sigma: Noise standard deviation
            
        Returns:
            RDP epsilon at order alpha
        """
        return (self.alpha * sensitivity**2) / (2 * sigma**2)
    
    def compose_rdp(self, epsilons: List[float]) -> float:
        """
        Compose multiple RDP mechanisms.
        
        Args:
            epsilons: List of RDP parameters at same order
            
        Returns:
            Composed RDP parameter
        """
        return sum(epsilons)
    
    def rdp_to_dp(self, rdp_epsilon: float, delta: float) -> float:
        """
        Convert RDP to standard (epsilon, delta)-DP.
        
        Args:
            rdp_epsilon: RDP parameter at order alpha
            delta: Target delta for conversion
            
        Returns:
            Standard DP epsilon
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        # Conversion formula from RDP to DP
        epsilon = rdp_epsilon + np.log(1 / delta) / (self.alpha - 1)
        return epsilon
    
    def adaptive_noise_schedule(
        self,
        num_iterations: int,
        total_epsilon: float,
        total_delta: float,
        sensitivity: float = 1.0
    ) -> List[float]:
        """
        Compute adaptive noise schedule for iterative algorithms.
        
        Args:
            num_iterations: Number of iterations
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
            sensitivity: Query sensitivity
            
        Returns:
            List of noise scales for each iteration
        """
        # Convert total budget to RDP
        rdp_budget = total_epsilon - np.log(1 / total_delta) * (self.alpha - 1)
        
        # Allocate budget per iteration (can be non-uniform)
        # Use decreasing noise for later iterations (more accurate as we converge)
        weights = np.array([1 / np.sqrt(i + 1) for i in range(num_iterations)])
        weights = weights / weights.sum()
        
        rdp_per_iter = rdp_budget * weights
        
        # Convert to noise scales
        noise_scales = []
        for rdp_eps in rdp_per_iter:
            # Solve for sigma: rdp_eps = (alpha * sensitivity^2) / (2 * sigma^2)
            sigma = sensitivity * np.sqrt(self.alpha / (2 * rdp_eps))
            noise_scales.append(sigma)
        
        return noise_scales


class ConcentratedDifferentialPrivacy:
    """
    Concentrated Differential Privacy (CDP) for optimal composition.
    
    Provides near-optimal composition for homogeneous mechanisms,
    particularly useful for REV's repeated similarity computations.
    """
    
    def __init__(self, rho: float):
        """
        Initialize CDP mechanism.
        
        Args:
            rho: CDP parameter (zero-concentrated DP)
        """
        self.rho = rho
    
    def gaussian_cdp(self, sensitivity: float, sigma: float) -> float:
        """
        Compute CDP parameter for Gaussian mechanism.
        
        Args:
            sensitivity: L2 sensitivity
            sigma: Noise standard deviation
            
        Returns:
            CDP rho parameter
        """
        return (sensitivity**2) / (2 * sigma**2)
    
    def compose_cdp(self, rhos: List[float]) -> float:
        """
        Compose multiple CDP mechanisms.
        
        Args:
            rhos: List of CDP parameters
            
        Returns:
            Composed CDP parameter
        """
        return sum(rhos)
    
    def cdp_to_dp(self, rho: float, delta: float) -> float:
        """
        Convert CDP to (epsilon, delta)-DP.
        
        Args:
            rho: CDP parameter
            delta: Target delta
            
        Returns:
            Standard DP epsilon
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        # Conversion from zCDP to (ε, δ)-DP
        epsilon = rho + 2 * np.sqrt(rho * np.log(1 / delta))
        return epsilon


class PrivacyAccountant:
    """
    Advanced privacy accounting for complex REV verification workflows.
    
    Tracks privacy consumption across different mechanisms and provides
    optimal budget allocation strategies.
    """
    
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        accounting_method: str = "rdp"
    ):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
            accounting_method: "standard", "rdp", or "cdp"
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.accounting_method = accounting_method
        
        # Track consumed privacy
        self.mechanisms: List[Dict[str, any]] = []
        
        # Initialize specialized accountants
        if accounting_method == "rdp":
            self.rdp = RenyiDifferentialPrivacy(alpha=2.0)
            self.rdp_consumed = 0.0
        elif accounting_method == "cdp":
            self.cdp = ConcentratedDifferentialPrivacy(rho=0.0)
            self.cdp_consumed = 0.0
        else:
            self.consumed_epsilon = 0.0
            self.consumed_delta = 0.0
    
    def record_mechanism(
        self,
        mechanism_type: str,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        sigma: Optional[float] = None,
        sensitivity: float = 1.0
    ) -> None:
        """
        Record a privacy mechanism usage.
        
        Args:
            mechanism_type: Type of mechanism ("gaussian", "laplace", etc.)
            epsilon: DP epsilon (if known)
            delta: DP delta (if known)
            sigma: Noise scale (for Gaussian)
            sensitivity: Query sensitivity
        """
        mechanism = {
            "type": mechanism_type,
            "epsilon": epsilon,
            "delta": delta,
            "sigma": sigma,
            "sensitivity": sensitivity,
            "timestamp": np.random.rand()  # Placeholder for actual timestamp
        }
        
        self.mechanisms.append(mechanism)
        
        # Update consumption based on accounting method
        if self.accounting_method == "rdp" and sigma is not None:
            rdp_eps = self.rdp.gaussian_rdp(sensitivity, sigma)
            self.rdp_consumed += rdp_eps
        elif self.accounting_method == "cdp" and sigma is not None:
            cdp_rho = self.cdp.gaussian_cdp(sensitivity, sigma)
            self.cdp_consumed += cdp_rho
        else:
            if epsilon is not None:
                self.consumed_epsilon += epsilon
            if delta is not None:
                self.consumed_delta += delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        if self.accounting_method == "rdp":
            # Convert RDP to DP
            consumed_eps = self.rdp.rdp_to_dp(self.rdp_consumed, self.total_delta)
            remaining_eps = max(0, self.total_epsilon - consumed_eps)
            remaining_delta = self.total_delta  # Delta doesn't compose in RDP
        elif self.accounting_method == "cdp":
            # Convert CDP to DP
            consumed_eps = self.cdp.cdp_to_dp(self.cdp_consumed, self.total_delta)
            remaining_eps = max(0, self.total_epsilon - consumed_eps)
            remaining_delta = self.total_delta
        else:
            remaining_eps = max(0, self.total_epsilon - self.consumed_epsilon)
            remaining_delta = max(0, self.total_delta - self.consumed_delta)
        
        return (remaining_eps, remaining_delta)
    
    def optimal_noise_for_remaining_queries(
        self,
        num_queries: int,
        sensitivity: float = 1.0
    ) -> float:
        """
        Compute optimal noise scale for remaining queries.
        
        Args:
            num_queries: Number of remaining queries
            sensitivity: Query sensitivity
            
        Returns:
            Optimal noise standard deviation
        """
        remaining_eps, remaining_delta = self.get_remaining_budget()
        
        if remaining_eps <= 0:
            return float('inf')  # No budget left
        
        # Allocate budget equally among remaining queries
        eps_per_query = remaining_eps / num_queries
        delta_per_query = remaining_delta / num_queries
        
        # Compute noise scale for Gaussian mechanism
        if delta_per_query > 0:
            sigma = np.sqrt(2 * np.log(1.25 / delta_per_query)) * sensitivity / eps_per_query
        else:
            # Use Laplace if no delta budget
            sigma = sensitivity / eps_per_query
        
        return sigma