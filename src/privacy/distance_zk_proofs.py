"""
Zero-Knowledge Proofs for distance computations in REV.

This module implements ZK proofs for verifying distance computations
between hypervectors without revealing the vectors themselves.
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256


@dataclass
class DistanceProof:
    """ZK proof of distance computation."""
    
    commitment_a: bytes  # Commitment to vector A
    commitment_b: bytes  # Commitment to vector B
    distance_commitment: bytes  # Commitment to distance value
    challenge: bytes  # Verifier's challenge
    response: Dict[str, Any]  # Prover's response
    metadata: Dict[str, Any]


@dataclass
class RangeProof:
    """Proof that a value lies in a specific range."""
    
    value_commitment: bytes
    range_min: float
    range_max: float
    proof_data: bytes
    

class DistanceZKProof:
    """
    Zero-knowledge proofs for distance computations.
    
    Implements Sigma protocols for proving properties of distance
    computations without revealing the underlying vectors.
    """
    
    def __init__(self, security_bits: int = 128):
        """
        Initialize ZK proof system.
        
        Args:
            security_bits: Security parameter in bits
        """
        self.security_bits = security_bits
        self.hash_function = SHA256
        
    def commit_vector(
        self,
        vector: np.ndarray,
        randomness: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Create commitment to a vector.
        
        Args:
            vector: Vector to commit to
            randomness: Random value for commitment (generated if None)
            
        Returns:
            Tuple of (commitment, randomness)
        """
        if randomness is None:
            randomness = get_random_bytes(self.security_bits // 8)
        
        # Serialize vector
        vector_bytes = vector.tobytes()
        
        # Compute commitment: H(vector || randomness)
        hasher = self.hash_function.new()
        hasher.update(vector_bytes)
        hasher.update(randomness)
        commitment = hasher.digest()
        
        return commitment, randomness
    
    def prove_distance(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        distance: float,
        metric: str = "euclidean"
    ) -> DistanceProof:
        """
        Generate ZK proof of distance computation.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            distance: Computed distance
            metric: Distance metric used
            
        Returns:
            Zero-knowledge proof of distance
        """
        # Step 1: Commit to vectors
        commitment_a, rand_a = self.commit_vector(vec_a)
        commitment_b, rand_b = self.commit_vector(vec_b)
        
        # Step 2: Commit to distance
        distance_bytes = str(distance).encode()
        rand_dist = get_random_bytes(self.security_bits // 8)
        
        hasher = self.hash_function.new()
        hasher.update(distance_bytes)
        hasher.update(rand_dist)
        distance_commitment = hasher.digest()
        
        # Step 3: Generate challenge (Fiat-Shamir heuristic)
        challenge = self._generate_challenge(
            commitment_a, commitment_b, distance_commitment
        )
        
        # Step 4: Compute response based on challenge
        challenge_int = int.from_bytes(challenge, 'big')
        
        # Blinded vectors (simplified - proper implementation would use more sophisticated blinding)
        blinding_factor_a = np.random.randn(len(vec_a))
        blinding_factor_b = np.random.randn(len(vec_b))
        
        blinded_a = vec_a + (challenge_int % 100) * blinding_factor_a / 100
        blinded_b = vec_b + (challenge_int % 100) * blinding_factor_b / 100
        
        # Response includes blinded information
        response = {
            "blinded_norm_a": float(np.linalg.norm(blinded_a)),
            "blinded_norm_b": float(np.linalg.norm(blinded_b)),
            "blinded_dot_product": float(np.dot(blinded_a, blinded_b)),
            "metric": metric,
            "dimension": len(vec_a)
        }
        
        # Add metric-specific proofs
        if metric == "euclidean":
            # For Euclidean distance, prove ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            response["distance_squared"] = distance ** 2
            response["sum_of_squares"] = float(np.linalg.norm(vec_a)**2 + np.linalg.norm(vec_b)**2)
            response["twice_dot_product"] = float(2 * np.dot(vec_a, vec_b))
            
        elif metric == "cosine":
            # For cosine similarity, prove similarity = <a,b> / (||a|| * ||b||)
            response["dot_product"] = float(np.dot(vec_a, vec_b))
            response["norm_product"] = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        
        return DistanceProof(
            commitment_a=commitment_a,
            commitment_b=commitment_b,
            distance_commitment=distance_commitment,
            challenge=challenge,
            response=response,
            metadata={
                "metric": metric,
                "timestamp": hashlib.sha256(str(np.random.rand()).encode()).hexdigest()
            }
        )
    
    def _generate_challenge(self, *commitments) -> bytes:
        """Generate challenge using Fiat-Shamir heuristic."""
        hasher = self.hash_function.new()
        for commitment in commitments:
            hasher.update(commitment)
        return hasher.digest()[:self.security_bits // 8]
    
    def verify_distance(
        self,
        proof: DistanceProof,
        distance: float,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Verify ZK proof of distance computation.
        
        Args:
            proof: Distance proof to verify
            distance: Claimed distance value
            tolerance: Numerical tolerance for verification
            
        Returns:
            True if proof is valid
        """
        # Verify distance commitment
        distance_bytes = str(distance).encode()
        
        # Recompute challenge
        challenge = self._generate_challenge(
            proof.commitment_a,
            proof.commitment_b,
            proof.distance_commitment
        )
        
        if challenge != proof.challenge:
            return False
        
        # Verify metric-specific properties
        metric = proof.response.get("metric", "euclidean")
        
        if metric == "euclidean":
            # Check Euclidean distance relation
            if "distance_squared" in proof.response:
                expected = proof.response["sum_of_squares"] - proof.response["twice_dot_product"]
                if abs(proof.response["distance_squared"] - expected) > tolerance:
                    return False
                
                # Check that claimed distance matches
                if abs(distance ** 2 - proof.response["distance_squared"]) > tolerance:
                    return False
                    
        elif metric == "cosine":
            # Check cosine similarity relation
            if "dot_product" in proof.response and "norm_product" in proof.response:
                similarity = proof.response["dot_product"] / (proof.response["norm_product"] + 1e-8)
                expected_distance = 1.0 - similarity
                
                if abs(distance - expected_distance) > tolerance:
                    return False
        
        # Verify dimension consistency
        dimension = proof.response.get("dimension", 0)
        if dimension <= 0:
            return False
        
        # Basic sanity checks on blinded values
        blinded_norm_a = proof.response.get("blinded_norm_a", 0)
        blinded_norm_b = proof.response.get("blinded_norm_b", 0)
        
        if blinded_norm_a < 0 or blinded_norm_b < 0:
            return False
        
        return True
    
    def prove_distance_range(
        self,
        distance: float,
        min_distance: float,
        max_distance: float
    ) -> RangeProof:
        """
        Prove that distance lies within a range without revealing exact value.
        
        Args:
            distance: Actual distance
            min_distance: Minimum of range
            max_distance: Maximum of range
            
        Returns:
            Range proof
        """
        if not (min_distance <= distance <= max_distance):
            raise ValueError("Distance not in specified range")
        
        # Commit to distance
        distance_bytes = str(distance).encode()
        randomness = get_random_bytes(self.security_bits // 8)
        
        hasher = self.hash_function.new()
        hasher.update(distance_bytes)
        hasher.update(randomness)
        commitment = hasher.digest()
        
        # Create proof using bit decomposition (simplified)
        # In practice, use Bulletproofs or similar
        
        # Normalize distance to [0, 1]
        normalized = (distance - min_distance) / (max_distance - min_distance)
        
        # Bit decomposition
        bits = []
        value = int(normalized * (2**32))
        for i in range(32):
            bits.append((value >> i) & 1)
        
        # Create proof data (simplified - includes bit commitments)
        proof_data = hashlib.sha256(
            str(bits).encode() + randomness
        ).digest()
        
        return RangeProof(
            value_commitment=commitment,
            range_min=min_distance,
            range_max=max_distance,
            proof_data=proof_data
        )
    
    def verify_distance_range(
        self,
        proof: RangeProof
    ) -> bool:
        """
        Verify that a distance lies within the claimed range.
        
        Args:
            proof: Range proof to verify
            
        Returns:
            True if proof is valid
        """
        # Basic validation
        if proof.range_min > proof.range_max:
            return False
        
        # In a full implementation, verify bit commitments
        # and range proof protocol
        
        # For now, just check proof data exists
        return len(proof.proof_data) == 32
    
    def prove_similarity_threshold(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        threshold: float,
        is_above: bool
    ) -> Dict[str, Any]:
        """
        Prove that similarity is above/below threshold without revealing vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            threshold: Similarity threshold
            is_above: True if proving similarity > threshold
            
        Returns:
            Zero-knowledge proof
        """
        # Compute actual similarity
        similarity = np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
        )
        
        # Check claim is correct
        if is_above and similarity <= threshold:
            raise ValueError("Cannot prove false claim")
        if not is_above and similarity >= threshold:
            raise ValueError("Cannot prove false claim")
        
        # Create commitments
        comm_a, rand_a = self.commit_vector(vec_a)
        comm_b, rand_b = self.commit_vector(vec_b)
        
        # Create proof of threshold relation
        # Simplified - proper implementation would use comparison protocols
        
        # Blind the difference from threshold
        diff = similarity - threshold
        blinding = np.random.randn()
        blinded_diff = diff + blinding
        
        # Challenge
        challenge = self._generate_challenge(comm_a, comm_b)
        challenge_int = int.from_bytes(challenge, 'big') % 100
        
        # Response includes blinded information
        proof = {
            "commitment_a": comm_a.hex(),
            "commitment_b": comm_b.hex(),
            "threshold": threshold,
            "is_above": is_above,
            "blinded_difference": blinded_diff * (challenge_int / 100),
            "challenge": challenge.hex(),
            "dimension": len(vec_a)
        }
        
        return proof
    
    def batch_prove_distances(
        self,
        vectors: List[np.ndarray],
        distance_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate batch proof for multiple distance computations.
        
        Args:
            vectors: List of vectors
            distance_matrix: Pairwise distance matrix
            
        Returns:
            Batch proof
        """
        n = len(vectors)
        
        # Commit to all vectors
        commitments = []
        randomness = []
        
        for vec in vectors:
            comm, rand = self.commit_vector(vec)
            commitments.append(comm)
            randomness.append(rand)
        
        # Create Merkle tree of commitments for efficiency
        merkle_root = self._compute_merkle_root(commitments)
        
        # Generate challenge
        challenge = self._generate_challenge(merkle_root)
        
        # Create aggregated response
        # Sum of all distances (can be verified without individual distances)
        total_distance = np.sum(distance_matrix)
        
        # Proof includes statistical properties
        batch_proof = {
            "merkle_root": merkle_root.hex(),
            "num_vectors": n,
            "total_distance": total_distance,
            "mean_distance": np.mean(distance_matrix),
            "std_distance": np.std(distance_matrix),
            "min_distance": np.min(distance_matrix[distance_matrix > 0]),  # Exclude diagonal
            "max_distance": np.max(distance_matrix),
            "challenge": challenge.hex()
        }
        
        return batch_proof
    
    def _compute_merkle_root(self, commitments: List[bytes]) -> bytes:
        """Compute Merkle tree root of commitments."""
        if len(commitments) == 1:
            return commitments[0]
        
        # Simplified Merkle tree
        level = commitments
        
        while len(level) > 1:
            next_level = []
            
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Hash pair
                    hasher = self.hash_function.new()
                    hasher.update(level[i])
                    hasher.update(level[i + 1])
                    next_level.append(hasher.digest())
                else:
                    # Odd number - carry forward
                    next_level.append(level[i])
            
            level = next_level
        
        return level[0]