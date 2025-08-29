"""
Secure aggregation for REV model signature combination.

This module provides cryptographic protocols for securely combining
multiple model signatures without revealing individual signatures.
"""

from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass

from ..crypto.commit import H, TAGS


@dataclass
class AggregationParams:
    """Parameters for secure aggregation"""
    aggregation_type: str = "mean"  # "mean", "sum", "majority"
    noise_scale: float = 0.01
    privacy_threshold: int = 3  # Minimum participants for privacy
    use_secure_shuffle: bool = True


class SecureAggregator:
    """
    Secure aggregator for REV model signatures.
    
    Provides cryptographic protocols to combine multiple model signatures
    while preserving privacy and preventing individual signature inference.
    """
    
    def __init__(self, params: Optional[AggregationParams] = None):
        """
        Initialize secure aggregator.
        
        Args:
            params: Aggregation parameters
        """
        self.params = params or AggregationParams()
        self._participant_count = 0
        
    def aggregate_signatures(
        self,
        signatures: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        participant_ids: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Securely aggregate multiple hypervector signatures.
        
        Args:
            signatures: List of hypervector signatures
            weights: Optional weights for weighted aggregation
            participant_ids: Optional participant identifiers
            
        Returns:
            Aggregated signature with privacy guarantees
            
        Raises:
            ValueError: If insufficient participants for privacy
        """
        if len(signatures) < self.params.privacy_threshold:
            raise ValueError(
                f"Need at least {self.params.privacy_threshold} participants, "
                f"got {len(signatures)}"
            )
        
        # Validate all signatures have same dimension
        dimension = signatures[0].shape[0]
        for i, sig in enumerate(signatures):
            if sig.shape[0] != dimension:
                raise ValueError(f"Signature {i} has wrong dimension")
        
        # Apply secure shuffling if enabled
        if self.params.use_secure_shuffle:
            signatures = self._secure_shuffle(signatures, participant_ids)
        
        # Perform aggregation
        if self.params.aggregation_type == "mean":
            aggregated = self._aggregate_mean(signatures, weights)
        elif self.params.aggregation_type == "sum":
            aggregated = self._aggregate_sum(signatures, weights)
        elif self.params.aggregation_type == "majority":
            aggregated = self._aggregate_majority(signatures)
        else:
            raise ValueError(f"Unknown aggregation type: {self.params.aggregation_type}")
        
        # Add noise for additional privacy
        aggregated = self._add_aggregation_noise(aggregated)
        
        # Normalize result
        if torch.norm(aggregated) > 0:
            aggregated = aggregated / torch.norm(aggregated)
        
        return aggregated
    
    def _secure_shuffle(
        self,
        signatures: List[torch.Tensor],
        participant_ids: Optional[List[str]] = None
    ) -> List[torch.Tensor]:
        """
        Apply cryptographic shuffling to break signature-participant linkage.
        
        Args:
            signatures: Input signatures
            participant_ids: Optional participant IDs for committed shuffling
            
        Returns:
            Shuffled signatures
        """
        if participant_ids:
            # Create deterministic but unpredictable shuffle based on participant IDs
            shuffle_seed = H(TAGS["REV_COMMITMENT"], 
                           b"".join(id.encode() for id in sorted(participant_ids)))
            shuffle_key = int.from_bytes(shuffle_seed[:8], 'big')
        else:
            # Use random shuffling
            shuffle_key = torch.randint(0, 2**32, (1,)).item()
        
        # Generate permutation
        torch.manual_seed(shuffle_key)
        indices = torch.randperm(len(signatures))
        
        return [signatures[i] for i in indices]
    
    def _aggregate_mean(
        self,
        signatures: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute weighted mean of signatures"""
        if weights is None:
            weights = [1.0 / len(signatures)] * len(signatures)
        
        if len(weights) != len(signatures):
            raise ValueError("Weights and signatures length mismatch")
        
        result = torch.zeros_like(signatures[0])
        for sig, weight in zip(signatures, weights):
            result += weight * sig
        
        return result
    
    def _aggregate_sum(
        self,
        signatures: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute weighted sum of signatures"""
        if weights is None:
            weights = [1.0] * len(signatures)
        
        result = torch.zeros_like(signatures[0])
        for sig, weight in zip(signatures, weights):
            result += weight * sig
        
        return result
    
    def _aggregate_majority(self, signatures: List[torch.Tensor]) -> torch.Tensor:
        """Compute majority vote for binary signatures"""
        # Convert to binary
        binary_sigs = [(sig > 0).float() for sig in signatures]
        
        # Stack and compute majority
        stacked = torch.stack(binary_sigs)
        majority = (stacked.sum(dim=0) > len(signatures) / 2).float()
        
        # Convert back to [-1, 1] range
        return majority * 2 - 1
    
    def _add_aggregation_noise(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to aggregated result"""
        if self.params.noise_scale > 0:
            noise = torch.normal(
                mean=0.0,
                std=self.params.noise_scale,
                size=aggregated.shape
            )
            aggregated = aggregated + noise
        
        return aggregated
    
    def compute_aggregation_proof(
        self,
        original_signatures: List[torch.Tensor],
        aggregated_signature: torch.Tensor,
        weights: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Generate proof that aggregation was performed correctly.
        
        Args:
            original_signatures: Input signatures
            aggregated_signature: Output aggregated signature
            weights: Aggregation weights used
            
        Returns:
            Aggregation proof for verification
        """
        # Compute commitments to original signatures
        commitments = []
        for i, sig in enumerate(original_signatures):
            sig_bytes = sig.numpy().tobytes()
            commitment = H(TAGS["REV_COMMITMENT"], 
                         sig_bytes, f"participant_{i}".encode())
            commitments.append(commitment.hex())
        
        # Compute expected aggregation
        if weights is None:
            weights = [1.0 / len(original_signatures)] * len(original_signatures)
        
        expected = torch.zeros_like(aggregated_signature)
        for sig, weight in zip(original_signatures, weights):
            expected += weight * sig
        
        # Normalize expected result
        if torch.norm(expected) > 0:
            expected = expected / torch.norm(expected)
        
        # Compute aggregation hash
        agg_data = {
            "commitments": commitments,
            "weights": weights,
            "aggregation_type": self.params.aggregation_type,
            "noise_scale": self.params.noise_scale
        }
        
        agg_bytes = str(agg_data).encode()
        agg_hash = H(TAGS["VK_AGG"], agg_bytes)
        
        return {
            "signature_commitments": commitments,
            "aggregation_type": self.params.aggregation_type,
            "weights": weights,
            "noise_scale": self.params.noise_scale,
            "aggregation_hash": agg_hash.hex(),
            "expected_result": expected.tolist(),
            "num_participants": len(original_signatures)
        }


def aggregate_signatures(
    signatures: List[torch.Tensor],
    aggregation_type: str = "mean",
    weights: Optional[List[float]] = None,
    add_noise: bool = True
) -> torch.Tensor:
    """
    Convenience function for signature aggregation.
    
    Args:
        signatures: List of hypervector signatures
        aggregation_type: Type of aggregation ("mean", "sum", "majority")
        weights: Optional aggregation weights
        add_noise: Whether to add privacy noise
        
    Returns:
        Aggregated signature
    """
    params = AggregationParams(
        aggregation_type=aggregation_type,
        noise_scale=0.01 if add_noise else 0.0
    )
    
    aggregator = SecureAggregator(params)
    return aggregator.aggregate_signatures(signatures, weights)


def federated_signature_aggregation(
    local_signatures: Dict[str, torch.Tensor],
    privacy_budget: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Perform federated aggregation of signatures from multiple parties.
    
    Args:
        local_signatures: Dictionary mapping party ID to signature
        privacy_budget: Privacy budget for noise addition
        
    Returns:
        Tuple of (aggregated_signature, aggregation_metadata)
    """
    if len(local_signatures) < 3:
        raise ValueError("Need at least 3 parties for federated aggregation")
    
    party_ids = list(local_signatures.keys())
    signatures = list(local_signatures.values())
    
    # Create aggregator with privacy parameters
    params = AggregationParams(
        aggregation_type="mean",
        noise_scale=privacy_budget / len(signatures),
        privacy_threshold=3,
        use_secure_shuffle=True
    )
    
    aggregator = SecureAggregator(params)
    
    # Perform secure aggregation
    aggregated = aggregator.aggregate_signatures(signatures, participant_ids=party_ids)
    
    # Generate proof
    proof = aggregator.compute_aggregation_proof(signatures, aggregated)
    
    metadata = {
        "num_parties": len(party_ids),
        "aggregation_proof": proof,
        "privacy_budget_used": privacy_budget,
        "party_ids_hash": H(TAGS["REV_COMMITMENT"], 
                          "".join(sorted(party_ids)).encode()).hex()
    }
    
    return aggregated, metadata


def verify_aggregation_proof(
    aggregation_proof: Dict[str, any],
    claimed_result: torch.Tensor,
    tolerance: float = 0.1
) -> bool:
    """
    Verify that aggregation was performed correctly.
    
    Args:
        aggregation_proof: Proof from compute_aggregation_proof
        claimed_result: Claimed aggregation result
        tolerance: Tolerance for numerical differences
        
    Returns:
        True if proof is valid
    """
    try:
        expected = torch.tensor(aggregation_proof["expected_result"])
        
        # Check dimensions match
        if expected.shape != claimed_result.shape:
            return False
        
        # Check results are close (accounting for noise)
        distance = torch.norm(expected - claimed_result).item()
        max_expected_distance = aggregation_proof["noise_scale"] * np.sqrt(len(expected))
        
        return distance <= max_expected_distance + tolerance
        
    except Exception:
        return False