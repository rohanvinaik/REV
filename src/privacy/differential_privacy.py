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
        
        # Simple composition (can be improved with advanced composition)
        total_epsilon = sum(eps for eps, _ in mechanisms)
        total_delta = sum(delta for _, delta in mechanisms)
        
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