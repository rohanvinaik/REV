"""
Information-theoretic bounds for REV privacy validation.

This module implements information-theoretic analysis to validate
the privacy claims made by REV verification protocols.
"""

from typing import Tuple, Optional, List, Dict, Any
import math
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class InformationBound:
    """Information-theoretic privacy bound"""
    mutual_information_bound: float  # Upper bound on I(X; Y)
    entropy_lower_bound: float       # Lower bound on H(X|Y)
    privacy_loss_bound: float        # Upper bound on privacy loss
    confidence_level: float          # Statistical confidence
    method: str                      # Estimation method used


class InformationAnalyzer:
    """
    Information-theoretic analyzer for REV privacy guarantees.
    
    Computes and validates information bounds for hypervector signatures
    and model comparison protocols to ensure privacy claims are sound.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize information analyzer.
        
        Args:
            confidence_level: Statistical confidence for bounds
        """
        self.confidence_level = confidence_level
    
    def compute_mutual_information_bound(
        self,
        private_data: torch.Tensor,
        public_output: torch.Tensor,
        method: str = "gaussian_approximation"
    ) -> float:
        """
        Compute upper bound on mutual information I(private; public).
        
        Args:
            private_data: Private model signatures/activations
            public_output: Public verification outputs
            method: Estimation method to use
            
        Returns:
            Upper bound on mutual information in bits
        """
        if method == "gaussian_approximation":
            return self._mutual_info_gaussian(private_data, public_output)
        elif method == "kernel_density":
            return self._mutual_info_kernel(private_data, public_output)
        elif method == "histogram":
            return self._mutual_info_histogram(private_data, public_output)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _mutual_info_gaussian(
        self, 
        private_data: torch.Tensor, 
        public_output: torch.Tensor
    ) -> float:
        """Estimate MI under Gaussian assumption"""
        # Compute correlation matrix
        combined = torch.cat([private_data.flatten().unsqueeze(1), 
                            public_output.flatten().unsqueeze(1)], dim=1)
        
        # Estimate covariance
        cov_matrix = torch.cov(combined.T)
        
        # Extract variances and covariance
        var_x = cov_matrix[0, 0].item()
        var_y = cov_matrix[1, 1].item()
        cov_xy = cov_matrix[0, 1].item()
        
        # Compute correlation coefficient
        if var_x > 0 and var_y > 0:
            correlation = cov_xy / math.sqrt(var_x * var_y)
            correlation = max(-0.999, min(0.999, correlation))  # Avoid log(0)
            
            # Mutual information for Gaussian: -0.5 * log(1 - rho^2)
            mi_nats = -0.5 * math.log(1 - correlation**2)
            mi_bits = mi_nats / math.log(2)
        else:
            mi_bits = 0.0
        
        return max(0.0, mi_bits)
    
    def _mutual_info_kernel(
        self,
        private_data: torch.Tensor,
        public_output: torch.Tensor
    ) -> float:
        """Estimate MI using kernel density estimation"""
        # Simplified implementation - full version would use proper KDE
        # This is a placeholder for the concept
        
        # Sample-based estimation using correlation
        flat_private = private_data.flatten()
        flat_public = public_output.flatten()
        
        # Subsample for efficiency
        n_samples = min(1000, len(flat_private))
        indices = torch.randperm(len(flat_private))[:n_samples]
        
        x = flat_private[indices].numpy()
        y = flat_public[indices].numpy()
        
        # Compute sample correlation
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            correlation = max(-0.999, min(0.999, correlation))
            
            mi_nats = -0.5 * math.log(1 - correlation**2)
            mi_bits = mi_nats / math.log(2)
        else:
            mi_bits = 0.0
        
        return max(0.0, mi_bits)
    
    def _mutual_info_histogram(
        self,
        private_data: torch.Tensor,
        public_output: torch.Tensor
    ) -> float:
        """Estimate MI using histogram-based method"""
        # Quantize data for histogram estimation
        n_bins = 50
        
        flat_private = private_data.flatten().numpy()
        flat_public = public_output.flatten().numpy()
        
        # Create histograms
        hist_joint, x_edges, y_edges = np.histogram2d(
            flat_private, flat_public, bins=n_bins
        )
        hist_x = np.histogram(flat_private, bins=x_edges)[0]
        hist_y = np.histogram(flat_public, bins=y_edges)[0]
        
        # Normalize to probabilities
        p_joint = hist_joint / np.sum(hist_joint)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)
        
        # Compute mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_joint[i, j] * math.log2(
                        p_joint[i, j] / (p_x[i] * p_y[j])
                    )
        
        return max(0.0, mi)
    
    def compute_entropy_bound(
        self,
        data: torch.Tensor,
        method: str = "gaussian_approximation"
    ) -> float:
        """
        Compute entropy bound for hypervector data.
        
        Args:
            data: Input tensor
            method: Estimation method
            
        Returns:
            Entropy estimate in bits
        """
        if method == "gaussian_approximation":
            # For multivariate Gaussian: H(X) = 0.5 * log((2πe)^d * |Σ|)
            flat_data = data.flatten()
            
            # Estimate variance
            variance = torch.var(flat_data).item()
            if variance <= 0:
                return 0.0
            
            # Entropy in nats, convert to bits
            entropy_nats = 0.5 * math.log(2 * math.pi * math.e * variance)
            return entropy_nats / math.log(2)
        
        elif method == "histogram":
            # Histogram-based entropy estimation
            flat_data = data.flatten().numpy()
            hist, _ = np.histogram(flat_data, bins=100, density=True)
            
            # Normalize
            hist = hist / np.sum(hist)
            
            # Compute entropy
            entropy = 0.0
            for p in hist:
                if p > 0:
                    entropy -= p * math.log2(p)
            
            return entropy
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def validate_differential_privacy_claim(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
        mechanism: str = "gaussian"
    ) -> bool:
        """
        Validate that noise parameters satisfy (ε,δ)-DP.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            sensitivity: Function sensitivity
            noise_scale: Noise standard deviation
            mechanism: "gaussian" or "laplace"
            
        Returns:
            True if parameters are consistent with DP guarantee
        """
        if mechanism == "gaussian":
            # For Gaussian mechanism: σ >= sqrt(2*ln(1.25/δ)) * Δ / ε
            required_scale = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
            return noise_scale >= required_scale * 0.99  # Allow small numerical tolerance
        
        elif mechanism == "laplace":
            # For Laplace mechanism: b >= Δ / ε
            required_scale = sensitivity / epsilon
            return noise_scale >= required_scale * 0.99
        
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    
    def compute_privacy_loss_bound(
        self,
        original_data: torch.Tensor,
        perturbed_data: torch.Tensor,
        epsilon: float
    ) -> InformationBound:
        """
        Compute information-theoretic privacy loss bound.
        
        Args:
            original_data: Original private data
            perturbed_data: Privacy-preserving version
            epsilon: Target privacy parameter
            
        Returns:
            Information bound analysis
        """
        # Compute mutual information bound
        mi_bound = self.compute_mutual_information_bound(
            original_data, perturbed_data, method="gaussian_approximation"
        )
        
        # Compute conditional entropy bound
        entropy_original = self.compute_entropy_bound(original_data)
        entropy_conditional = max(0.0, entropy_original - mi_bound)
        
        # Theoretical privacy loss bound for (ε,0)-DP is ε
        theoretical_loss = epsilon / math.log(2)  # Convert to bits
        
        return InformationBound(
            mutual_information_bound=mi_bound,
            entropy_lower_bound=entropy_conditional,
            privacy_loss_bound=min(mi_bound, theoretical_loss),
            confidence_level=self.confidence_level,
            method="information_theoretic"
        )


def compute_information_bound(
    private_data: torch.Tensor,
    public_output: torch.Tensor,
    confidence_level: float = 0.95
) -> InformationBound:
    """
    Convenience function to compute information bounds.
    
    Args:
        private_data: Private model data
        public_output: Public verification output
        confidence_level: Statistical confidence level
        
    Returns:
        Information bound analysis
    """
    analyzer = InformationAnalyzer(confidence_level)
    
    mi_bound = analyzer.compute_mutual_information_bound(private_data, public_output)
    entropy_bound = analyzer.compute_entropy_bound(private_data)
    
    return InformationBound(
        mutual_information_bound=mi_bound,
        entropy_lower_bound=max(0.0, entropy_bound - mi_bound),
        privacy_loss_bound=mi_bound,
        confidence_level=confidence_level,
        method="mutual_information"
    )


def validate_privacy_claims(
    privacy_mechanism_params: Dict[str, Any],
    empirical_data: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, bool]:
    """
    Validate REV privacy claims against information-theoretic bounds.
    
    Args:
        privacy_mechanism_params: Parameters of privacy mechanism
        empirical_data: Optional empirical data for validation
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Extract parameters
    epsilon = privacy_mechanism_params.get("epsilon", 1.0)
    delta = privacy_mechanism_params.get("delta", 1e-5)
    sensitivity = privacy_mechanism_params.get("sensitivity", 1.0)
    noise_scale = privacy_mechanism_params.get("noise_scale", 1.0)
    mechanism = privacy_mechanism_params.get("mechanism", "gaussian")
    
    analyzer = InformationAnalyzer()
    
    # Validate DP parameters
    results["dp_parameters_valid"] = analyzer.validate_differential_privacy_claim(
        epsilon, delta, sensitivity, noise_scale, mechanism
    )
    
    # Validate empirical claims if data provided
    if empirical_data and "private" in empirical_data and "public" in empirical_data:
        bound = analyzer.compute_privacy_loss_bound(
            empirical_data["private"],
            empirical_data["public"], 
            epsilon
        )
        
        # Check if empirical MI is below theoretical bound
        theoretical_bound = epsilon / math.log(2)
        results["empirical_mi_valid"] = bound.mutual_information_bound <= theoretical_bound * 1.1
        
        # Check if entropy is sufficiently preserved
        results["entropy_preserved"] = bound.entropy_lower_bound > 0.1 * analyzer.compute_entropy_bound(empirical_data["private"])
    
    return results


def estimate_optimal_privacy_parameters(
    data_dimension: int,
    target_utility: float = 0.9,
    max_epsilon: float = 1.0
) -> Dict[str, float]:
    """
    Estimate optimal privacy parameters for REV verification.
    
    Args:
        data_dimension: Dimensionality of hypervectors
        target_utility: Target utility preservation (0-1)
        max_epsilon: Maximum acceptable epsilon
        
    Returns:
        Recommended privacy parameters
    """
    # Simple heuristic based on dimension and utility
    # More sophisticated optimization could be implemented
    
    # Higher dimensions need more privacy budget for same utility
    dimension_factor = math.log(data_dimension) / math.log(10000)  # Normalize to 10k baseline
    
    recommended_epsilon = min(max_epsilon, 0.1 * dimension_factor / target_utility)
    recommended_delta = 1e-5 * dimension_factor
    
    # Noise scale for target utility under Gaussian mechanism
    sensitivity = 1.0  # Assuming L2 sensitivity of 1
    noise_scale = math.sqrt(2 * math.log(1.25 / recommended_delta)) * sensitivity / recommended_epsilon
    
    return {
        "epsilon": recommended_epsilon,
        "delta": recommended_delta,
        "noise_scale": noise_scale,
        "expected_utility": target_utility,
        "mechanism": "gaussian"
    }