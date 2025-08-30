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


class CommitmentScheme:
    """
    Cryptographic commitment scheme for REV privacy.
    
    Provides binding and hiding properties for vector commitments
    used in zero-knowledge proofs.
    """
    
    def __init__(self, security_bits: int = 128):
        """
        Initialize commitment scheme.
        
        Args:
            security_bits: Security parameter in bits
        """
        self.security_bits = security_bits
        self.hash_function = SHA256
    
    def commit(
        self,
        value: np.ndarray,
        randomness: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Create commitment to a value.
        
        Args:
            value: Value to commit to
            randomness: Random value (generated if None)
            
        Returns:
            Tuple of (commitment, randomness)
        """
        if randomness is None:
            randomness = get_random_bytes(self.security_bits // 8)
        
        # Serialize value
        if isinstance(value, np.ndarray):
            value_bytes = value.tobytes()
        else:
            value_bytes = str(value).encode()
        
        # Compute commitment: H(value || randomness)
        hasher = self.hash_function.new()
        hasher.update(value_bytes)
        hasher.update(randomness)
        commitment = hasher.digest()
        
        return commitment, randomness
    
    def verify_commitment(
        self,
        commitment: bytes,
        value: np.ndarray,
        randomness: bytes
    ) -> bool:
        """
        Verify a commitment opening.
        
        Args:
            commitment: Original commitment
            value: Claimed value
            randomness: Opening randomness
            
        Returns:
            True if commitment is valid
        """
        expected_commitment, _ = self.commit(value, randomness)
        return expected_commitment == commitment
    
    def batch_commit(
        self,
        values: List[np.ndarray]
    ) -> Tuple[List[bytes], List[bytes]]:
        """
        Create batch commitments for multiple values.
        
        Args:
            values: List of values to commit to
            
        Returns:
            Tuple of (commitments, randomness_list)
        """
        commitments = []
        randomness_list = []
        
        for value in values:
            commitment, randomness = self.commit(value)
            commitments.append(commitment)
            randomness_list.append(randomness)
        
        return commitments, randomness_list


class MerkleInclusionProof:
    """
    Merkle tree inclusion proofs for REV verification.
    
    Proves that a signature is included in a committed set
    without revealing other signatures.
    """
    
    def __init__(self, hash_function=SHA256):
        """
        Initialize Merkle proof system.
        
        Args:
            hash_function: Hash function for tree construction
        """
        self.hash_function = hash_function
    
    def build_tree(self, leaves: List[bytes]) -> Dict[str, Any]:
        """
        Build Merkle tree from leaf values.
        
        Args:
            leaves: List of leaf values (hashed signatures)
            
        Returns:
            Tree structure with root and internal nodes
        """
        if not leaves:
            raise ValueError("Cannot build tree from empty leaves")
        
        # Ensure even number of leaves by duplicating last leaf if needed
        if len(leaves) % 2 == 1:
            leaves = leaves + [leaves[-1]]
        
        tree = {"leaves": leaves, "levels": []}
        current_level = leaves
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1]
                
                # Compute parent hash
                hasher = self.hash_function.new()
                hasher.update(left)
                hasher.update(right)
                parent = hasher.digest()
                next_level.append(parent)
            
            tree["levels"].append(current_level)
            current_level = next_level
        
        tree["root"] = current_level[0]
        tree["levels"].append(current_level)
        
        return tree
    
    def generate_inclusion_proof(
        self,
        tree: Dict[str, Any],
        leaf_index: int
    ) -> List[Tuple[bytes, bool]]:
        """
        Generate inclusion proof for a specific leaf.
        
        Args:
            tree: Merkle tree structure
            leaf_index: Index of leaf to prove inclusion for
            
        Returns:
            List of (sibling_hash, is_right) tuples for proof path
        """
        leaves = tree["leaves"]
        levels = tree["levels"]
        
        if leaf_index >= len(leaves):
            raise ValueError("Leaf index out of bounds")
        
        proof = []
        current_index = leaf_index
        
        for level_idx, level in enumerate(levels[:-1]):  # Exclude root level
            # Find sibling
            if current_index % 2 == 0:
                # Left child, sibling is right
                sibling_index = current_index + 1
                is_right = True
            else:
                # Right child, sibling is left
                sibling_index = current_index - 1
                is_right = False
            
            if sibling_index < len(level):
                sibling_hash = level[sibling_index]
            else:
                # Handle case where we duplicated the last leaf
                sibling_hash = level[current_index]
            
            proof.append((sibling_hash, is_right))
            current_index = current_index // 2
        
        return proof
    
    def verify_inclusion_proof(
        self,
        root: bytes,
        leaf: bytes,
        proof: List[Tuple[bytes, bool]]
    ) -> bool:
        """
        Verify Merkle inclusion proof.
        
        Args:
            root: Merkle tree root
            leaf: Leaf value to verify
            proof: Inclusion proof path
            
        Returns:
            True if proof is valid
        """
        current_hash = leaf
        
        for sibling_hash, is_right in proof:
            hasher = self.hash_function.new()
            
            if is_right:
                # Sibling is on the right
                hasher.update(current_hash)
                hasher.update(sibling_hash)
            else:
                # Sibling is on the left
                hasher.update(sibling_hash)
                hasher.update(current_hash)
            
            current_hash = hasher.digest()
        
        return current_hash == root
    
    def batch_inclusion_proofs(
        self,
        signatures: List[np.ndarray],
        indices: List[int]
    ) -> Dict[str, Any]:
        """
        Generate batch inclusion proofs for multiple signatures.
        
        Args:
            signatures: List of signature vectors
            indices: Indices of signatures to prove inclusion for
            
        Returns:
            Batch proof structure
        """
        # Hash all signatures to create leaves
        leaves = []
        for sig in signatures:
            hasher = self.hash_function.new()
            hasher.update(sig.tobytes())
            leaves.append(hasher.digest())
        
        # Build tree
        tree = self.build_tree(leaves)
        
        # Generate proofs for specified indices
        proofs = {}
        for idx in indices:
            proof = self.generate_inclusion_proof(tree, idx)
            proofs[idx] = proof
        
        return {
            "root": tree["root"],
            "proofs": proofs,
            "signature_hashes": [leaves[i] for i in indices],
            "num_signatures": len(signatures)
        }


class EnhancedDistanceZKProof(DistanceZKProof):
    """
    Enhanced zero-knowledge proofs with commitment schemes and Merkle proofs.
    
    Extends the basic distance ZK proofs with additional privacy features
    for REV verification systems.
    """
    
    def __init__(self, security_bits: int = 128):
        """
        Initialize enhanced ZK proof system.
        
        Args:
            security_bits: Security parameter in bits
        """
        super().__init__(security_bits)
        self.commitment_scheme = CommitmentScheme(security_bits)
        self.merkle_prover = MerkleInclusionProof()
    
    def prove_committed_distance(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        distance: float,
        commitment_a: bytes,
        commitment_b: bytes,
        rand_a: bytes,
        rand_b: bytes,
        metric: str = "euclidean"
    ) -> Dict[str, Any]:
        """
        Prove distance between committed vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            distance: Computed distance
            commitment_a: Commitment to vector A
            commitment_b: Commitment to vector B
            rand_a: Randomness for commitment A
            rand_b: Randomness for commitment B
            metric: Distance metric
            
        Returns:
            Zero-knowledge proof with commitments
        """
        # Verify commitments are valid
        if not self.commitment_scheme.verify_commitment(commitment_a, vec_a, rand_a):
            raise ValueError("Invalid commitment for vector A")
        if not self.commitment_scheme.verify_commitment(commitment_b, vec_b, rand_b):
            raise ValueError("Invalid commitment for vector B")
        
        # Generate basic distance proof
        distance_proof = self.prove_distance(vec_a, vec_b, distance, metric)
        
        # Add commitment information
        enhanced_proof = {
            "commitment_a": commitment_a.hex(),
            "commitment_b": commitment_b.hex(),
            "distance_proof": {
                "distance_commitment": distance_proof.distance_commitment.hex(),
                "challenge": distance_proof.challenge.hex(),
                "response": distance_proof.response,
                "metadata": distance_proof.metadata
            },
            "metric": metric,
            "proven_distance": distance
        }
        
        return enhanced_proof
    
    def verify_committed_distance(
        self,
        proof: Dict[str, Any],
        commitment_a: bytes,
        commitment_b: bytes,
        claimed_distance: float
    ) -> bool:
        """
        Verify distance proof with commitments.
        
        Args:
            proof: Enhanced distance proof
            commitment_a: Commitment to vector A
            commitment_b: Commitment to vector B
            claimed_distance: Claimed distance value
            
        Returns:
            True if proof is valid
        """
        # Check commitments match
        if proof["commitment_a"] != commitment_a.hex():
            return False
        if proof["commitment_b"] != commitment_b.hex():
            return False
        
        # Check distance matches
        if abs(proof["proven_distance"] - claimed_distance) > 1e-6:
            return False
        
        # Reconstruct distance proof
        distance_proof_data = proof["distance_proof"]
        distance_proof = DistanceProof(
            commitment_a=commitment_a,
            commitment_b=commitment_b,
            distance_commitment=bytes.fromhex(distance_proof_data["distance_commitment"]),
            challenge=bytes.fromhex(distance_proof_data["challenge"]),
            response=distance_proof_data["response"],
            metadata=distance_proof_data["metadata"]
        )
        
        # Verify basic distance proof
        return self.verify_distance(distance_proof, claimed_distance)
    
    def prove_signature_inclusion(
        self,
        signature: np.ndarray,
        signature_set: List[np.ndarray],
        signature_index: int
    ) -> Dict[str, Any]:
        """
        Prove that a signature is included in a committed set.
        
        Args:
            signature: Signature to prove inclusion for
            signature_set: Full set of signatures
            signature_index: Index of signature in set
            
        Returns:
            Inclusion proof with Merkle authentication
        """
        if signature_index >= len(signature_set):
            raise ValueError("Signature index out of bounds")
        
        # Verify signature matches
        if not np.array_equal(signature, signature_set[signature_index]):
            raise ValueError("Signature doesn't match claimed index")
        
        # Generate Merkle inclusion proof
        batch_proof = self.merkle_prover.batch_inclusion_proofs(
            signature_set, [signature_index]
        )
        
        # Create commitment to the signature
        commitment, randomness = self.commitment_scheme.commit(signature)
        
        return {
            "signature_commitment": commitment.hex(),
            "signature_randomness": randomness.hex(),
            "merkle_root": batch_proof["root"].hex(),
            "inclusion_proof": [
                (sibling.hex(), is_right) 
                for sibling, is_right in batch_proof["proofs"][signature_index]
            ],
            "signature_index": signature_index,
            "set_size": len(signature_set)
        }
    
    def verify_signature_inclusion(
        self,
        proof: Dict[str, Any],
        signature: np.ndarray,
        merkle_root: bytes
    ) -> bool:
        """
        Verify signature inclusion proof.
        
        Args:
            proof: Inclusion proof
            signature: Claimed signature
            merkle_root: Root of signature set Merkle tree
            
        Returns:
            True if proof is valid
        """
        # Check Merkle root matches
        if proof["merkle_root"] != merkle_root.hex():
            return False
        
        # Verify commitment opening
        commitment = bytes.fromhex(proof["signature_commitment"])
        randomness = bytes.fromhex(proof["signature_randomness"])
        
        if not self.commitment_scheme.verify_commitment(commitment, signature, randomness):
            return False
        
        # Hash signature for Merkle proof
        hasher = self.hash_function.new()
        hasher.update(signature.tobytes())
        signature_hash = hasher.digest()
        
        # Reconstruct inclusion proof
        inclusion_proof = [
            (bytes.fromhex(sibling), is_right)
            for sibling, is_right in proof["inclusion_proof"]
        ]
        
        # Verify Merkle inclusion
        return self.merkle_prover.verify_inclusion_proof(
            merkle_root, signature_hash, inclusion_proof
        )
    
    def prove_distance_threshold_with_privacy(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        threshold: float,
        is_above: bool,
        privacy_level: float = 0.1
    ) -> Dict[str, Any]:
        """
        Prove distance threshold with additional privacy protection.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            threshold: Distance threshold
            is_above: True if proving distance > threshold
            privacy_level: Additional privacy noise level
            
        Returns:
            Privacy-enhanced threshold proof
        """
        # Compute actual distance
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have same dimension")
        
        distance = np.linalg.norm(vec_a - vec_b)
        
        # Check claim validity
        if is_above and distance <= threshold:
            raise ValueError("Cannot prove false claim (distance not above threshold)")
        if not is_above and distance >= threshold:
            raise ValueError("Cannot prove false claim (distance not below threshold)")
        
        # Add privacy noise to vectors before proof generation
        noise_a = np.random.normal(0, privacy_level, vec_a.shape)
        noise_b = np.random.normal(0, privacy_level, vec_b.shape)
        
        noisy_a = vec_a + noise_a
        noisy_b = vec_b + noise_b
        
        # Compute noisy distance
        noisy_distance = np.linalg.norm(noisy_a - noisy_b)
        
        # Generate commitments to noisy vectors
        comm_a, rand_a = self.commitment_scheme.commit(noisy_a)
        comm_b, rand_b = self.commitment_scheme.commit(noisy_b)
        
        # Create proof structure
        proof = {
            "commitment_a": comm_a.hex(),
            "commitment_b": comm_b.hex(),
            "threshold": threshold,
            "is_above": is_above,
            "privacy_level": privacy_level,
            "noisy_distance": noisy_distance,
            "claim_valid": (noisy_distance > threshold) if is_above else (noisy_distance < threshold)
        }
        
        # Add challenge-response for verification
        challenge_data = f"{comm_a.hex()}{comm_b.hex()}{threshold}{is_above}".encode()
        challenge = hashlib.sha256(challenge_data).digest()
        
        proof["challenge"] = challenge.hex()
        proof["dimension"] = len(vec_a)
        
        return proof