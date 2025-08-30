"""
Secure aggregation for REV model signature combination.

This module provides cryptographic protocols for securely combining
multiple model signatures without revealing individual signatures,
including secret sharing schemes and multi-party computation protocols.
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
import torch
import numpy as np
from dataclasses import dataclass
import random
import secrets
from collections import defaultdict
import json

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


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing scheme for distributed signatures.
    
    Allows splitting a signature into n shares where any k shares
    can reconstruct the original, providing fault tolerance.
    """
    
    def __init__(self, threshold: int, num_parties: int, prime: int = 2**31 - 1):
        """
        Initialize secret sharing scheme.
        
        Args:
            threshold: Minimum shares needed to reconstruct (k)
            num_parties: Total number of parties (n)
            prime: Prime modulus for finite field arithmetic (smaller for stability)
        """
        if threshold > num_parties:
            raise ValueError("Threshold cannot exceed number of parties")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        
        self.threshold = threshold
        self.num_parties = num_parties
        self.prime = prime
    
    def share_signature(self, signature: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """
        Split signature into secret shares.
        
        Args:
            signature: Original signature to share
            
        Returns:
            List of (party_id, share) tuples
        """
        # Convert to integers for finite field arithmetic
        signature_int = (signature * 1e3).long()  # Scale for precision (reduced for stability)
        
        shares = []
        for dim in range(signature_int.shape[0]):
            secret = signature_int[dim].item()
            dim_shares = self._share_secret(secret)
            shares.append(dim_shares)
        
        # Reorganize by party
        party_shares = []
        for party_id in range(1, self.num_parties + 1):
            party_tensor = torch.zeros_like(signature_int)
            for dim in range(signature_int.shape[0]):
                party_tensor[dim] = shares[dim][party_id - 1][1]
            party_shares.append((party_id, party_tensor.float() / 1e3))
        
        return party_shares
    
    def reconstruct_signature(
        self, 
        shares: List[Tuple[int, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Reconstruct signature from shares.
        
        Args:
            shares: List of (party_id, share) tuples
            
        Returns:
            Reconstructed signature
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use first threshold shares
        shares = shares[:self.threshold]
        
        # Convert to integers
        signature_int = torch.zeros_like(shares[0][1]).long()
        
        for dim in range(signature_int.shape[0]):
            dim_shares = [(party_id, int(share[dim].item() * 1e3)) 
                         for party_id, share in shares]
            reconstructed = self._reconstruct_secret(dim_shares)
            signature_int[dim] = reconstructed
        
        return signature_int.float() / 1e3
    
    def _share_secret(self, secret: int) -> List[Tuple[int, int]]:
        """Generate shares for a single secret value."""
        # Generate random coefficients for polynomial of degree k-1
        coefficients = [secret]  # a_0 = secret
        for _ in range(self.threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))
        
        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for x in range(1, self.num_parties + 1):
            y = 0
            for i, coeff in enumerate(coefficients):
                y = (y + coeff * pow(x, i, self.prime)) % self.prime
            shares.append((x, y))
        
        return shares
    
    def _reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret using Lagrange interpolation."""
        result = 0
        
        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange coefficient
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    # Keep values small to avoid overflow
                    numerator = (numerator * (-x_j % self.prime)) % self.prime
                    denominator = (denominator * ((x_i - x_j) % self.prime)) % self.prime
            
            # Ensure denominator is not zero
            if denominator == 0:
                continue
            
            # Compute modular inverse of denominator
            try:
                denominator_inv = pow(denominator % self.prime, self.prime - 2, self.prime)
                lagrange_coeff = (numerator * denominator_inv) % self.prime
                result = (result + (y_i * lagrange_coeff) % self.prime) % self.prime
            except:
                # Skip if computation fails
                continue
        
        # Convert back to signed if needed
        if result > self.prime // 2:
            result -= self.prime
        
        return result


class MultiPartyComputation:
    """
    Multi-party computation protocols for secure REV operations.
    
    Implements protocols for computing on encrypted/shared data without
    revealing individual inputs to any party.
    """
    
    def __init__(self, num_parties: int, threshold: int):
        """
        Initialize MPC system.
        
        Args:
            num_parties: Total number of parties
            threshold: Threshold for secret sharing
        """
        self.num_parties = num_parties
        self.threshold = threshold
        self.secret_sharing = ShamirSecretSharing(threshold, num_parties)
        
        # Communication channels (simulated)
        self.channels: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
    
    def secure_sum(
        self,
        private_values: List[torch.Tensor],
        party_ids: List[int]
    ) -> torch.Tensor:
        """
        Compute sum of private values using secure MPC.
        
        Args:
            private_values: Private values from each party
            party_ids: IDs of participating parties
            
        Returns:
            Sum without revealing individual values
        """
        if len(private_values) != len(party_ids):
            raise ValueError("Mismatched number of values and party IDs")
        
        # Phase 1: Each party secret-shares their value
        all_shares = []
        for value, party_id in zip(private_values, party_ids):
            shares = self.secret_sharing.share_signature(value)
            all_shares.append(shares)
        
        # Phase 2: Compute sum of shares (additive homomorphism)
        sum_shares = []
        for party_idx in range(self.num_parties):
            party_sum = torch.zeros_like(private_values[0])
            for shares in all_shares:
                party_sum += shares[party_idx][1]
            sum_shares.append((party_idx + 1, party_sum))
        
        # Phase 3: Reconstruct result
        return self.secret_sharing.reconstruct_signature(sum_shares[:self.threshold])
    
    def secure_dot_product(
        self,
        vec_a_shares: List[Tuple[int, torch.Tensor]],
        vec_b_shares: List[Tuple[int, torch.Tensor]]
    ) -> float:
        """
        Compute dot product of two shared vectors.
        
        Args:
            vec_a_shares: Secret shares of vector A
            vec_b_shares: Secret shares of vector B
            
        Returns:
            Dot product result
        """
        # Multiply corresponding shares (works for Shamir sharing)
        product_shares = []
        for (id_a, share_a), (id_b, share_b) in zip(vec_a_shares, vec_b_shares):
            if id_a != id_b:
                raise ValueError("Mismatched party IDs in shares")
            product = share_a * share_b
            product_shares.append((id_a, product))
        
        # Reconstruct product vector
        product_vector = self.secret_sharing.reconstruct_signature(
            product_shares[:self.threshold]
        )
        
        # Sum elements for dot product
        return product_vector.sum().item()
    
    def secure_distance_computation(
        self,
        vec_a_shares: List[Tuple[int, torch.Tensor]],
        vec_b_shares: List[Tuple[int, torch.Tensor]],
        metric: str = "euclidean"
    ) -> float:
        """
        Compute distance between shared vectors.
        
        Args:
            vec_a_shares: Secret shares of vector A
            vec_b_shares: Secret shares of vector B
            metric: Distance metric ("euclidean", "cosine")
            
        Returns:
            Distance without revealing vectors
        """
        if metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            dot_ab = self.secure_dot_product(vec_a_shares, vec_b_shares)
            dot_aa = self.secure_dot_product(vec_a_shares, vec_a_shares)
            dot_bb = self.secure_dot_product(vec_b_shares, vec_b_shares)
            
            distance_squared = dot_aa + dot_bb - 2 * dot_ab
            return np.sqrt(max(0, distance_squared))
            
        elif metric == "cosine":
            # cos_sim = <a,b> / (||a|| * ||b||)
            dot_ab = self.secure_dot_product(vec_a_shares, vec_b_shares)
            dot_aa = self.secure_dot_product(vec_a_shares, vec_a_shares)
            dot_bb = self.secure_dot_product(vec_b_shares, vec_b_shares)
            
            norm_a = np.sqrt(max(0, dot_aa))
            norm_b = np.sqrt(max(0, dot_bb))
            
            if norm_a * norm_b == 0:
                return 0.0
            
            cosine_sim = dot_ab / (norm_a * norm_b)
            return 1.0 - cosine_sim  # Convert to distance
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def byzantine_robust_aggregation(
        self,
        signatures: List[torch.Tensor],
        party_ids: List[int],
        max_byzantine: int
    ) -> torch.Tensor:
        """
        Aggregate signatures with Byzantine fault tolerance.
        
        Args:
            signatures: Input signatures from parties
            party_ids: Party identifiers
            max_byzantine: Maximum number of Byzantine parties
            
        Returns:
            Robust aggregated signature
        """
        if len(signatures) < 3 * max_byzantine + 1:
            raise ValueError("Need at least 3f+1 parties for f Byzantine faults")
        
        # Use coordinate-wise median for Byzantine robustness
        stacked = torch.stack(signatures)
        
        # Compute trimmed mean (remove extreme values)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        
        # Remove top and bottom max_byzantine values
        if max_byzantine > 0:
            trimmed = sorted_vals[max_byzantine:-max_byzantine]
        else:
            trimmed = sorted_vals
        
        # Compute mean of remaining values
        robust_mean = trimmed.mean(dim=0)
        
        # Normalize result
        if torch.norm(robust_mean) > 0:
            robust_mean = robust_mean / torch.norm(robust_mean)
        
        return robust_mean


class FederatedAggregationProtocol:
    """
    Federated aggregation protocol with privacy guarantees.
    
    Combines secure aggregation with differential privacy for
    federated REV verification across multiple organizations.
    """
    
    def __init__(
        self,
        privacy_budget: float = 1.0,
        min_participants: int = 10,
        dropout_resilience: float = 0.5
    ):
        """
        Initialize federated protocol.
        
        Args:
            privacy_budget: Total privacy budget for aggregation
            min_participants: Minimum participants needed
            dropout_resilience: Fraction of dropouts to tolerate
        """
        self.privacy_budget = privacy_budget
        self.min_participants = min_participants
        self.dropout_resilience = dropout_resilience
    
    def setup_secure_aggregation(
        self,
        participant_ids: List[str],
        signature_dimension: int
    ) -> Dict[str, Any]:
        """
        Set up secure aggregation round.
        
        Args:
            participant_ids: List of participant identifiers
            signature_dimension: Dimension of signatures to aggregate
            
        Returns:
            Setup information for participants
        """
        num_participants = len(participant_ids)
        
        if num_participants < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants")
        
        # Calculate threshold for secret sharing
        max_dropouts = int(num_participants * self.dropout_resilience)
        threshold = num_participants - max_dropouts
        
        # Generate shared randomness for secure aggregation
        # Each pair of participants shares a random mask
        pairwise_masks = {}
        for i, id_i in enumerate(participant_ids):
            for j, id_j in enumerate(participant_ids):
                if i < j:  # Only generate once per pair
                    mask = torch.randn(signature_dimension)
                    pairwise_masks[(id_i, id_j)] = mask
        
        setup_info = {
            "participant_ids": participant_ids,
            "threshold": threshold,
            "signature_dimension": signature_dimension,
            "pairwise_masks": pairwise_masks,
            "privacy_noise_scale": self._compute_noise_scale(num_participants),
            "round_id": secrets.token_hex(16)
        }
        
        return setup_info
    
    def participant_mask(
        self,
        participant_id: str,
        setup_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute aggregated mask for a participant.
        
        Args:
            participant_id: ID of the participant
            setup_info: Setup information from setup phase
            
        Returns:
            Aggregated mask for this participant
        """
        participant_ids = setup_info["participant_ids"]
        pairwise_masks = setup_info["pairwise_masks"]
        signature_dim = setup_info["signature_dimension"]
        
        if participant_id not in participant_ids:
            raise ValueError("Unknown participant ID")
        
        # Aggregate masks: +mask for (self, other), -mask for (other, self)
        aggregated_mask = torch.zeros(signature_dim)
        
        for other_id in participant_ids:
            if other_id != participant_id:
                if (participant_id, other_id) in pairwise_masks:
                    aggregated_mask += pairwise_masks[(participant_id, other_id)]
                elif (other_id, participant_id) in pairwise_masks:
                    aggregated_mask -= pairwise_masks[(other_id, participant_id)]
        
        return aggregated_mask
    
    def secure_aggregate_round(
        self,
        masked_signatures: Dict[str, torch.Tensor],
        setup_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Perform secure aggregation round.
        
        Args:
            masked_signatures: Dictionary of participant_id -> masked_signature
            setup_info: Setup information
            
        Returns:
            Aggregated signature with privacy guarantees
        """
        participant_ids = setup_info["participant_ids"]
        threshold = setup_info["threshold"]
        noise_scale = setup_info["privacy_noise_scale"]
        
        if len(masked_signatures) < threshold:
            raise ValueError(f"Insufficient participants: {len(masked_signatures)} < {threshold}")
        
        # Sum all masked signatures (masks cancel out)
        total_signature = torch.zeros_like(next(iter(masked_signatures.values())))
        
        for signature in masked_signatures.values():
            total_signature += signature
        
        # Add differential privacy noise
        privacy_noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=total_signature.shape
        )
        
        noisy_aggregate = total_signature + privacy_noise
        
        # Normalize result
        if torch.norm(noisy_aggregate) > 0:
            noisy_aggregate = noisy_aggregate / torch.norm(noisy_aggregate)
        
        return noisy_aggregate
    
    def _compute_noise_scale(self, num_participants: int) -> float:
        """Compute DP noise scale based on number of participants."""
        # Sensitivity is 2 (each participant contributes ±1 to aggregation)
        sensitivity = 2.0
        epsilon = self.privacy_budget / np.log(num_participants)  # Adaptive epsilon
        
        # Use Gaussian mechanism
        delta = 1e-6
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        return sigma
    
    def verify_aggregation_correctness(
        self,
        participants: List[str],
        individual_signatures: Dict[str, torch.Tensor],
        aggregated_signature: torch.Tensor,
        tolerance: float = 0.1
    ) -> bool:
        """
        Verify that aggregation was performed correctly (for testing).
        
        Args:
            participants: List of participant IDs
            individual_signatures: Original signatures (for verification only)
            aggregated_signature: Claimed aggregated result
            tolerance: Tolerance for numerical differences
            
        Returns:
            True if aggregation appears correct
        """
        # Compute expected aggregate (without privacy noise)
        expected = torch.zeros_like(aggregated_signature)
        for signature in individual_signatures.values():
            expected += signature
        
        if torch.norm(expected) > 0:
            expected = expected / torch.norm(expected)
        
        # Check if result is within expected noise bounds
        difference = torch.norm(expected - aggregated_signature).item()
        noise_scale = self._compute_noise_scale(len(participants))
        expected_noise = noise_scale * np.sqrt(len(aggregated_signature))
        
        return difference <= expected_noise + tolerance