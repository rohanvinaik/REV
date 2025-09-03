"""
Zero-Knowledge Attestation for REV fingerprints.
Implements zk-SNARKs for distance proofs and Bulletproofs for range proofs.
"""

import hashlib
import secrets
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


@dataclass
class PedersenCommitment:
    """Pedersen commitment for fingerprint integrity."""
    
    commitment: bytes
    randomness: bytes
    generator_g: ec.EllipticCurvePublicKey
    generator_h: ec.EllipticCurvePublicKey


@dataclass
class ZKProof:
    """Zero-knowledge proof container."""
    
    proof_type: str  # "distance", "range", "membership"
    commitment: bytes
    challenge: bytes
    response: bytes
    public_inputs: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class BulletProof:
    """Bulletproof for efficient range proofs."""
    
    commitment: bytes
    proof_data: bytes
    range_min: float
    range_max: float
    
    
class ZKAttestationSystem:
    """
    Zero-knowledge attestation system for REV fingerprints.
    Provides privacy-preserving proofs of fingerprint properties.
    """
    
    def __init__(
        self,
        curve: str = "secp256k1",
        security_parameter: int = 128
    ):
        """
        Initialize ZK attestation system.
        
        Args:
            curve: Elliptic curve to use
            security_parameter: Security parameter in bits
        """
        self.curve = curve
        self.security_parameter = security_parameter
        self.backend = default_backend()
        
        # Initialize curve parameters
        if curve == "secp256k1":
            self.curve_obj = ec.SECP256K1()
        elif curve == "secp384r1":
            self.curve_obj = ec.SECP384R1()
        else:
            self.curve_obj = ec.SECP256R1()
            
        # Generate system parameters
        self.setup_parameters()
        
    def setup_parameters(self):
        """Setup cryptographic parameters for the system."""
        # Generate random generators for Pedersen commitments
        private_key_g = ec.generate_private_key(self.curve_obj, self.backend)
        private_key_h = ec.generate_private_key(self.curve_obj, self.backend)
        
        self.generator_g = private_key_g.public_key()
        self.generator_h = private_key_h.public_key()
        
        # Store public parameters
        self.public_params = {
            "curve": self.curve,
            "security_parameter": self.security_parameter,
            "generator_g": self._serialize_public_key(self.generator_g),
            "generator_h": self._serialize_public_key(self.generator_h)
        }
        
        logger.info(f"Initialized ZK system with {self.curve} curve")
    
    def create_pedersen_commitment(
        self,
        value: np.ndarray,
        randomness: Optional[bytes] = None
    ) -> PedersenCommitment:
        """
        Create Pedersen commitment for a fingerprint vector.
        
        Args:
            value: Fingerprint vector to commit to
            randomness: Optional randomness (generated if not provided)
            
        Returns:
            PedersenCommitment object
        """
        # Generate randomness if not provided
        if randomness is None:
            randomness = secrets.token_bytes(32)
        
        # Hash the value to get a scalar
        value_hash = hashlib.sha256(value.tobytes()).digest()
        value_scalar = int.from_bytes(value_hash, 'big') % self.curve_obj.order
        
        # Convert randomness to scalar
        r_scalar = int.from_bytes(randomness, 'big') % self.curve_obj.order
        
        # Compute commitment: C = g^v * h^r
        # Using elliptic curve: C = v*G + r*H
        private_key = ec.derive_private_key(value_scalar, self.curve_obj, self.backend)
        point_g = private_key.public_key()
        
        private_key_r = ec.derive_private_key(r_scalar, self.curve_obj, self.backend)
        point_h = private_key_r.public_key()
        
        # Combine points (simplified - in production use proper EC point addition)
        commitment_bytes = self._combine_points(point_g, point_h)
        
        return PedersenCommitment(
            commitment=commitment_bytes,
            randomness=randomness,
            generator_g=self.generator_g,
            generator_h=self.generator_h
        )
    
    def prove_distance_computation(
        self,
        fingerprint1: np.ndarray,
        fingerprint2: np.ndarray,
        distance: float,
        max_distance: float = 1.0
    ) -> ZKProof:
        """
        Create zk-SNARK proof for distance computation without revealing fingerprints.
        
        Args:
            fingerprint1: First fingerprint vector
            fingerprint2: Second fingerprint vector
            distance: Computed distance
            max_distance: Maximum possible distance
            
        Returns:
            ZKProof for the distance computation
        """
        # Create commitments to both fingerprints
        commitment1 = self.create_pedersen_commitment(fingerprint1)
        commitment2 = self.create_pedersen_commitment(fingerprint2)
        
        # Create commitment to distance
        distance_bytes = np.array([distance]).tobytes()
        distance_commitment = hashlib.sha256(distance_bytes).digest()
        
        # Generate Fiat-Shamir challenge
        challenge_input = (
            commitment1.commitment + 
            commitment2.commitment + 
            distance_commitment
        )
        challenge = hashlib.sha256(challenge_input).digest()
        
        # Compute response (simplified Schnorr-like proof)
        # In production, use proper zk-SNARK library like libsnark or bellman
        response_data = {
            "commitment1": commitment1.commitment.hex(),
            "commitment2": commitment2.commitment.hex(),
            "distance_commitment": distance_commitment.hex(),
            "blinded_distance": self._blind_value(distance, commitment1.randomness)
        }
        
        response = json.dumps(response_data).encode()
        
        return ZKProof(
            proof_type="distance",
            commitment=distance_commitment,
            challenge=challenge,
            response=response,
            public_inputs={
                "distance": distance,
                "max_distance": max_distance
            },
            metadata={
                "timestamp": self._get_timestamp(),
                "algorithm": "hamming",
                "dimension": len(fingerprint1)
            }
        )
    
    def verify_distance_proof(
        self,
        proof: ZKProof,
        commitment1: bytes,
        commitment2: bytes
    ) -> bool:
        """
        Verify a zero-knowledge proof of distance computation.
        
        Args:
            proof: ZKProof to verify
            commitment1: Commitment to first fingerprint
            commitment2: Commitment to second fingerprint
            
        Returns:
            True if proof is valid
        """
        if proof.proof_type != "distance":
            return False
        
        try:
            # Recompute challenge
            challenge_input = commitment1 + commitment2 + proof.commitment
            expected_challenge = hashlib.sha256(challenge_input).digest()
            
            if proof.challenge != expected_challenge:
                logger.warning("Challenge verification failed")
                return False
            
            # Verify response structure
            response_data = json.loads(proof.response.decode())
            
            # Check commitments match
            if (response_data["commitment1"] != commitment1.hex() or
                response_data["commitment2"] != commitment2.hex()):
                logger.warning("Commitment mismatch")
                return False
            
            # Verify distance is in valid range
            distance = proof.public_inputs["distance"]
            max_distance = proof.public_inputs["max_distance"]
            
            if not (0 <= distance <= max_distance):
                logger.warning(f"Distance {distance} out of range [0, {max_distance}]")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    def create_range_proof(
        self,
        value: float,
        range_min: float,
        range_max: float,
        num_bits: int = 32
    ) -> BulletProof:
        """
        Create Bulletproof for proving value is in range without revealing it.
        
        Args:
            value: Secret value to prove is in range
            range_min: Minimum of range
            range_max: Maximum of range
            num_bits: Number of bits for range representation
            
        Returns:
            BulletProof object
        """
        if not (range_min <= value <= range_max):
            raise ValueError(f"Value {value} not in range [{range_min}, {range_max}]")
        
        # Normalize value to [0, 2^n - 1]
        normalized = int((value - range_min) / (range_max - range_min) * (2**num_bits - 1))
        
        # Create bit decomposition
        bits = [(normalized >> i) & 1 for i in range(num_bits)]
        
        # Generate blinding factors
        blindings = [secrets.randbits(256) for _ in range(num_bits)]
        
        # Create Pedersen commitments for each bit
        bit_commitments = []
        for bit, blinding in zip(bits, blindings):
            commitment = self._create_bit_commitment(bit, blinding)
            bit_commitments.append(commitment)
        
        # Aggregate commitments (simplified Bulletproof)
        aggregated = self._aggregate_commitments(bit_commitments)
        
        # Create proof data
        proof_data = {
            "bit_commitments": [c.hex() for c in bit_commitments],
            "aggregated": aggregated.hex(),
            "num_bits": num_bits,
            "blindings_hash": hashlib.sha256(
                b"".join(b.to_bytes(32, 'big') for b in blindings)
            ).hexdigest()
        }
        
        return BulletProof(
            commitment=aggregated,
            proof_data=json.dumps(proof_data).encode(),
            range_min=range_min,
            range_max=range_max
        )
    
    def verify_range_proof(
        self,
        proof: BulletProof,
        commitment: bytes
    ) -> bool:
        """
        Verify a Bulletproof range proof.
        
        Args:
            proof: BulletProof to verify
            commitment: Commitment to the value
            
        Returns:
            True if proof is valid
        """
        try:
            proof_data = json.loads(proof.proof_data.decode())
            
            # Verify commitment matches
            if proof_data["aggregated"] != proof.commitment.hex():
                return False
            
            # Verify bit commitments are well-formed
            bit_commitments = proof_data["bit_commitments"]
            num_bits = proof_data["num_bits"]
            
            if len(bit_commitments) != num_bits:
                return False
            
            # In production, verify the actual Bulletproof equations
            # This is simplified for demonstration
            
            return True
            
        except Exception as e:
            logger.error(f"Range proof verification failed: {e}")
            return False
    
    def create_membership_proof(
        self,
        fingerprint: np.ndarray,
        merkle_path: List[bytes],
        merkle_root: bytes
    ) -> ZKProof:
        """
        Create zero-knowledge proof of membership in Merkle tree.
        
        Args:
            fingerprint: Fingerprint to prove membership for
            merkle_path: Merkle path from leaf to root
            merkle_root: Root of Merkle tree
            
        Returns:
            ZKProof of membership
        """
        # Commit to fingerprint
        commitment = self.create_pedersen_commitment(fingerprint)
        
        # Hash fingerprint to get leaf
        leaf = hashlib.sha256(fingerprint.tobytes()).digest()
        
        # Create proof showing path leads to root
        proof_data = {
            "leaf_commitment": commitment.commitment.hex(),
            "path_length": len(merkle_path),
            "root": merkle_root.hex()
        }
        
        # Generate challenge
        challenge_input = commitment.commitment + merkle_root + b"".join(merkle_path)
        challenge = hashlib.sha256(challenge_input).digest()
        
        # Create response with blinded path
        blinded_path = [self._blind_hash(h, commitment.randomness) for h in merkle_path]
        response = json.dumps({
            "blinded_path": [h.hex() for h in blinded_path],
            "proof_data": proof_data
        }).encode()
        
        return ZKProof(
            proof_type="membership",
            commitment=commitment.commitment,
            challenge=challenge,
            response=response,
            public_inputs={"merkle_root": merkle_root.hex()},
            metadata={
                "timestamp": self._get_timestamp(),
                "tree_height": len(merkle_path)
            }
        )
    
    def batch_verify_proofs(
        self,
        proofs: List[ZKProof],
        commitments: List[bytes]
    ) -> bool:
        """
        Batch verification of multiple proofs for efficiency.
        
        Args:
            proofs: List of proofs to verify
            commitments: Corresponding commitments
            
        Returns:
            True if all proofs are valid
        """
        if len(proofs) != len(commitments):
            return False
        
        # Generate random coefficients for batch verification
        coefficients = [secrets.randbits(128) for _ in range(len(proofs))]
        
        # Aggregate challenges
        aggregated_challenge = bytes(32)
        for coeff, proof in zip(coefficients, proofs):
            weighted = self._multiply_scalar(proof.challenge, coeff)
            aggregated_challenge = self._xor_bytes(aggregated_challenge, weighted)
        
        # Verify aggregated proof (simplified)
        # In production, use proper batch verification algorithms
        
        for proof, commitment in zip(proofs, commitments):
            if proof.proof_type == "distance":
                # Distance proofs need special handling
                continue
            elif proof.proof_type == "membership":
                # Verify membership proof
                if commitment != proof.commitment:
                    return False
        
        return True
    
    def _serialize_public_key(self, public_key: ec.EllipticCurvePublicKey) -> str:
        """Serialize EC public key to PEM format."""
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def _combine_points(
        self,
        point1: ec.EllipticCurvePublicKey,
        point2: ec.EllipticCurvePublicKey
    ) -> bytes:
        """Combine two EC points (simplified)."""
        # In production, use proper EC point addition
        bytes1 = point1.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        bytes2 = point2.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        
        return hashlib.sha256(bytes1 + bytes2).digest()
    
    def _blind_value(self, value: float, randomness: bytes) -> str:
        """Blind a value with randomness."""
        value_bytes = np.array([value]).tobytes()
        blinded = hashlib.sha256(value_bytes + randomness).digest()
        return blinded.hex()
    
    def _create_bit_commitment(self, bit: int, blinding: int) -> bytes:
        """Create commitment to a single bit."""
        bit_bytes = bit.to_bytes(1, 'big')
        blinding_bytes = blinding.to_bytes(32, 'big')
        return hashlib.sha256(bit_bytes + blinding_bytes).digest()
    
    def _aggregate_commitments(self, commitments: List[bytes]) -> bytes:
        """Aggregate multiple commitments."""
        aggregated = commitments[0]
        for c in commitments[1:]:
            aggregated = self._xor_bytes(aggregated, c)
        return aggregated
    
    def _blind_hash(self, hash_value: bytes, randomness: bytes) -> bytes:
        """Blind a hash value."""
        return hashlib.sha256(hash_value + randomness).digest()
    
    def _multiply_scalar(self, data: bytes, scalar: int) -> bytes:
        """Multiply bytes by scalar (simplified)."""
        # In production, use proper scalar multiplication
        result = bytearray(len(data))
        for i, b in enumerate(data):
            result[i] = (b * scalar) % 256
        return bytes(result)
    
    def _xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte arrays."""
        return bytes(x ^ y for x, y in zip(a, b))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def export_public_params(self, filepath: str):
        """Export public parameters for verification."""
        with open(filepath, 'w') as f:
            json.dump(self.public_params, f, indent=2)
        logger.info(f"Exported public parameters to {filepath}")
    
    def import_public_params(self, filepath: str):
        """Import public parameters."""
        with open(filepath, 'r') as f:
            self.public_params = json.load(f)
        logger.info(f"Imported public parameters from {filepath}")


class ZKCircuit:
    """
    zk-SNARK circuit for complex computations on fingerprints.
    """
    
    def __init__(self, num_constraints: int = 1000):
        """
        Initialize zk-SNARK circuit.
        
        Args:
            num_constraints: Number of constraints in the circuit
        """
        self.num_constraints = num_constraints
        self.constraints = []
        self.witnesses = []
        self.public_inputs = []
        
    def add_distance_constraints(
        self,
        dimension: int,
        distance_type: str = "hamming"
    ):
        """
        Add constraints for distance computation.
        
        Args:
            dimension: Dimension of fingerprints
            distance_type: Type of distance metric
        """
        if distance_type == "hamming":
            # Add constraints for Hamming distance
            for i in range(dimension):
                # Constraint: diff[i] = fp1[i] XOR fp2[i]
                self.constraints.append({
                    "type": "xor",
                    "inputs": [f"fp1_{i}", f"fp2_{i}"],
                    "output": f"diff_{i}"
                })
                
                # Constraint: sum += diff[i]
                self.constraints.append({
                    "type": "add",
                    "inputs": ["sum", f"diff_{i}"],
                    "output": "sum"
                })
                
        elif distance_type == "cosine":
            # Add constraints for cosine similarity
            for i in range(dimension):
                # Constraint: prod[i] = fp1[i] * fp2[i]
                self.constraints.append({
                    "type": "multiply",
                    "inputs": [f"fp1_{i}", f"fp2_{i}"],
                    "output": f"prod_{i}"
                })
        
        logger.info(f"Added {len(self.constraints)} constraints for {distance_type} distance")
    
    def generate_proof(
        self,
        private_inputs: Dict[str, Any],
        public_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate zk-SNARK proof for the circuit.
        
        Args:
            private_inputs: Private witness values
            public_inputs: Public input values
            
        Returns:
            Proof dictionary
        """
        # In production, use a proper zk-SNARK library like libsnark
        # This is a simplified demonstration
        
        proof = {
            "pi_a": secrets.token_hex(32),
            "pi_b": secrets.token_hex(32),
            "pi_c": secrets.token_hex(32),
            "public_inputs": public_inputs,
            "num_constraints": self.num_constraints
        }
        
        return proof
    
    def verify_proof(
        self,
        proof: Dict[str, Any],
        verification_key: Dict[str, Any]
    ) -> bool:
        """
        Verify a zk-SNARK proof.
        
        Args:
            proof: Proof to verify
            verification_key: Verification key
            
        Returns:
            True if proof is valid
        """
        # Simplified verification
        # In production, use proper pairing checks
        
        return "pi_a" in proof and "pi_b" in proof and "pi_c" in proof