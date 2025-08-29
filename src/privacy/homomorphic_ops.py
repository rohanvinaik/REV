"""
Homomorphic-friendly operations for privacy-preserving verification.

This module implements operations that can be performed on encrypted data,
federated evaluation protocols, and secure aggregation mechanisms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import hashlib
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
import struct


@dataclass
class HomomorphicConfig:
    """Configuration for homomorphic operations."""
    
    key_size: int = 2048
    precision_bits: int = 32
    scaling_factor: int = 1000000  # For fixed-point arithmetic
    modulus: int = 2**32 - 1
    enable_packing: bool = True  # Pack multiple values per ciphertext
    batch_size: int = 64


@dataclass 
class EncryptedVector:
    """Encrypted hypervector representation."""
    
    ciphertexts: List[bytes]
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    public_key_hash: str = ""


@dataclass
class SecureAggregationShare:
    """Share for secure multi-party aggregation."""
    
    participant_id: str
    share_data: bytes
    commitment: bytes
    round_number: int


class HomomorphicOperations:
    """
    Homomorphic-friendly operations for privacy-preserving REV verification.
    
    Implements operations that can be performed on encrypted data without
    decryption, enabling privacy-preserving model comparison.
    """
    
    def __init__(self, config: Optional[HomomorphicConfig] = None):
        """
        Initialize homomorphic operations.
        
        Args:
            config: Homomorphic operation configuration
        """
        self.config = config or HomomorphicConfig()
        
        # Generate key pair for encryption
        self.key = RSA.generate(self.config.key_size)
        self.public_key = self.key.publickey()
        self.cipher = PKCS1_OAEP.new(self.public_key)
        self.decipher = PKCS1_OAEP.new(self.key)
        
        # Cache for encrypted values
        self.encrypted_cache = {}
    
    def encode_for_homomorphic(
        self,
        vector: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode vector for homomorphic operations.
        
        Args:
            vector: Input hypervector
            normalize: Whether to normalize values
            
        Returns:
            Encoded vector suitable for homomorphic operations
        """
        # Normalize to [-1, 1] if needed
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        # Convert to fixed-point representation
        scaled = (vector * self.config.scaling_factor).astype(np.int64)
        
        # Apply modulus for overflow protection
        encoded = scaled % self.config.modulus
        
        return encoded
    
    def encrypt_vector(
        self,
        vector: np.ndarray,
        public_key: Optional[RSA.RsaKey] = None
    ) -> EncryptedVector:
        """
        Encrypt a hypervector.
        
        Args:
            vector: Vector to encrypt
            public_key: Public key for encryption (or use default)
            
        Returns:
            Encrypted vector
        """
        if public_key is None:
            public_key = self.public_key
            
        cipher = PKCS1_OAEP.new(public_key)
        
        # Encode vector for homomorphic operations
        encoded = self.encode_for_homomorphic(vector)
        
        ciphertexts = []
        
        if self.config.enable_packing:
            # Pack multiple values per ciphertext
            for i in range(0, len(encoded), self.config.batch_size):
                batch = encoded[i:i+self.config.batch_size]
                
                # Serialize batch
                packed = b''.join(
                    struct.pack('>q', int(val)) for val in batch
                )
                
                # Encrypt packed values
                # Note: RSA has size limits, may need to chunk further
                max_chunk = (self.config.key_size // 8) - 42  # OAEP overhead
                
                for j in range(0, len(packed), max_chunk):
                    chunk = packed[j:j+max_chunk]
                    ciphertext = cipher.encrypt(chunk)
                    ciphertexts.append(ciphertext)
        else:
            # Encrypt each value separately
            for val in encoded:
                data = struct.pack('>q', int(val))
                ciphertext = cipher.encrypt(data)
                ciphertexts.append(ciphertext)
        
        # Compute public key hash for verification
        key_hash = hashlib.sha256(
            public_key.export_key()
        ).hexdigest()
        
        return EncryptedVector(
            ciphertexts=ciphertexts,
            dimension=len(vector),
            metadata={
                "packed": self.config.enable_packing,
                "batch_size": self.config.batch_size
            },
            public_key_hash=key_hash
        )
    
    def decrypt_vector(
        self,
        encrypted: EncryptedVector,
        private_key: Optional[RSA.RsaKey] = None
    ) -> np.ndarray:
        """
        Decrypt an encrypted vector.
        
        Args:
            encrypted: Encrypted vector
            private_key: Private key for decryption
            
        Returns:
            Decrypted vector
        """
        if private_key is None:
            decipher = self.decipher
        else:
            decipher = PKCS1_OAEP.new(private_key)
        
        decoded_values = []
        
        if encrypted.metadata.get("packed", False):
            # Unpack values
            batch_size = encrypted.metadata.get("batch_size", self.config.batch_size)
            
            decrypted_data = b''
            for ciphertext in encrypted.ciphertexts:
                decrypted_data += decipher.decrypt(ciphertext)
            
            # Deserialize values
            for i in range(0, len(decrypted_data), 8):
                if i + 8 <= len(decrypted_data):
                    val = struct.unpack('>q', decrypted_data[i:i+8])[0]
                    decoded_values.append(val)
        else:
            # Decrypt each value
            for ciphertext in encrypted.ciphertexts:
                decrypted = decipher.decrypt(ciphertext)
                val = struct.unpack('>q', decrypted)[0]
                decoded_values.append(val)
        
        # Convert back from fixed-point
        decoded_array = np.array(decoded_values[:encrypted.dimension])
        vector = decoded_array.astype(np.float64) / self.config.scaling_factor
        
        return vector
    
    def homomorphic_add(
        self,
        encrypted_a: EncryptedVector,
        encrypted_b: EncryptedVector
    ) -> EncryptedVector:
        """
        Add two encrypted vectors (approximation).
        
        Note: This is a simplified version. True homomorphic addition
        would require a proper HE scheme like Paillier or CKKS.
        
        Args:
            encrypted_a: First encrypted vector
            encrypted_b: Second encrypted vector
            
        Returns:
            Encrypted sum (approximation)
        """
        # For demonstration - in practice, use proper HE library
        # This shows the interface but doesn't provide true HE
        
        if encrypted_a.dimension != encrypted_b.dimension:
            raise ValueError("Vectors must have same dimension")
        
        # Combine ciphertexts (simplified - not true HE)
        combined = []
        for ca, cb in zip(encrypted_a.ciphertexts, encrypted_b.ciphertexts):
            # XOR as a simple combining operation (not secure!)
            combined_bytes = bytes(a ^ b for a, b in zip(ca, cb))
            combined.append(combined_bytes)
        
        return EncryptedVector(
            ciphertexts=combined,
            dimension=encrypted_a.dimension,
            metadata={"operation": "add"},
            public_key_hash=encrypted_a.public_key_hash
        )
    
    def homomorphic_multiply_constant(
        self,
        encrypted: EncryptedVector,
        constant: float
    ) -> EncryptedVector:
        """
        Multiply encrypted vector by a constant.
        
        Args:
            encrypted: Encrypted vector
            constant: Scalar constant
            
        Returns:
            Encrypted result
        """
        # Encode constant
        encoded_const = int(constant * self.config.scaling_factor) % self.config.modulus
        
        # For demonstration - proper HE would maintain encryption
        modified_ciphertexts = []
        for ciphertext in encrypted.ciphertexts:
            # Simple transformation (not secure!)
            modified = bytes((b * encoded_const) % 256 for b in ciphertext)
            modified_ciphertexts.append(modified)
        
        return EncryptedVector(
            ciphertexts=modified_ciphertexts,
            dimension=encrypted.dimension,
            metadata={"operation": "scalar_multiply", "constant": constant},
            public_key_hash=encrypted.public_key_hash
        )
    
    def compute_encrypted_distance(
        self,
        encrypted_a: EncryptedVector,
        encrypted_b: EncryptedVector,
        metric: str = "euclidean"
    ) -> float:
        """
        Compute distance between encrypted vectors.
        
        Note: This returns a plaintext distance. In true HE,
        the result would also be encrypted.
        
        Args:
            encrypted_a: First encrypted vector
            encrypted_b: Second encrypted vector
            metric: Distance metric
            
        Returns:
            Distance value
        """
        # For demonstration - decrypt and compute
        # In practice, use HE-friendly distance approximations
        
        vec_a = self.decrypt_vector(encrypted_a)
        vec_b = self.decrypt_vector(encrypted_b)
        
        if metric == "euclidean":
            distance = np.linalg.norm(vec_a - vec_b)
        elif metric == "cosine":
            similarity = np.dot(vec_a, vec_b) / (
                np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
            )
            distance = 1.0 - similarity
        else:
            distance = np.linalg.norm(vec_a - vec_b)
        
        return float(distance)
    
    def generate_secure_aggregation_shares(
        self,
        vector: np.ndarray,
        num_participants: int,
        participant_id: str,
        round_number: int = 0
    ) -> List[SecureAggregationShare]:
        """
        Generate shares for secure multi-party aggregation.
        
        Args:
            vector: Vector to share
            num_participants: Number of participants
            participant_id: ID of this participant
            round_number: Protocol round number
            
        Returns:
            List of shares for distribution
        """
        # Additive secret sharing
        shares = []
        encoded = self.encode_for_homomorphic(vector)
        
        # Generate random shares
        random_shares = []
        for i in range(num_participants - 1):
            random_share = np.random.randint(
                0, self.config.modulus,
                size=len(encoded),
                dtype=np.int64
            )
            random_shares.append(random_share)
        
        # Last share makes sum equal to original
        last_share = encoded.copy()
        for random_share in random_shares:
            last_share = (last_share - random_share) % self.config.modulus
        random_shares.append(last_share)
        
        # Create share objects with commitments
        for i, share_data in enumerate(random_shares):
            # Serialize share
            serialized = share_data.tobytes()
            
            # Create commitment (hash of share)
            commitment = hashlib.sha256(serialized).digest()
            
            share = SecureAggregationShare(
                participant_id=f"{participant_id}_{i}",
                share_data=serialized,
                commitment=commitment,
                round_number=round_number
            )
            shares.append(share)
        
        return shares
    
    def aggregate_shares(
        self,
        shares: List[SecureAggregationShare],
        dimension: int
    ) -> np.ndarray:
        """
        Aggregate shares to recover sum.
        
        Args:
            shares: List of shares from all participants
            dimension: Vector dimension
            
        Returns:
            Aggregated vector
        """
        # Verify commitments
        for share in shares:
            expected_commitment = hashlib.sha256(share.share_data).digest()
            if share.commitment != expected_commitment:
                raise ValueError(f"Invalid commitment for share {share.participant_id}")
        
        # Deserialize and sum shares
        aggregated = np.zeros(dimension, dtype=np.int64)
        
        for share in shares:
            # Deserialize share data
            share_array = np.frombuffer(share.share_data, dtype=np.int64)
            if len(share_array) != dimension:
                share_array = share_array[:dimension]
            
            # Add to aggregate
            aggregated = (aggregated + share_array) % self.config.modulus
        
        # Convert back from fixed-point
        result = aggregated.astype(np.float64) / self.config.scaling_factor
        
        return result


class FederatedProtocol:
    """
    Federated evaluation protocol for distributed verification.
    """
    
    def __init__(
        self,
        num_participants: int,
        threshold: int,
        homomorphic_ops: Optional[HomomorphicOperations] = None
    ):
        """
        Initialize federated protocol.
        
        Args:
            num_participants: Number of participants
            threshold: Minimum participants for aggregation
            homomorphic_ops: Homomorphic operations instance
        """
        self.num_participants = num_participants
        self.threshold = threshold
        self.homomorphic_ops = homomorphic_ops or HomomorphicOperations()
        
        # Storage for protocol state
        self.round_number = 0
        self.collected_shares = {}
        self.participant_keys = {}
    
    def register_participant(
        self,
        participant_id: str,
        public_key: RSA.RsaKey
    ):
        """Register a participant with their public key."""
        self.participant_keys[participant_id] = public_key
    
    def initiate_round(self) -> int:
        """Initiate a new protocol round."""
        self.round_number += 1
        self.collected_shares[self.round_number] = []
        return self.round_number
    
    def submit_encrypted_vector(
        self,
        participant_id: str,
        encrypted_vector: EncryptedVector,
        round_number: int
    ) -> bool:
        """
        Submit an encrypted vector for aggregation.
        
        Args:
            participant_id: ID of submitting participant
            encrypted_vector: Encrypted vector
            round_number: Protocol round
            
        Returns:
            Success status
        """
        if round_number != self.round_number:
            return False
        
        if participant_id not in self.participant_keys:
            return False
        
        # Verify the vector was encrypted with correct key
        expected_hash = hashlib.sha256(
            self.participant_keys[participant_id].export_key()
        ).hexdigest()
        
        if encrypted_vector.public_key_hash != expected_hash:
            return False
        
        # Store for aggregation
        self.collected_shares[round_number].append({
            "participant_id": participant_id,
            "vector": encrypted_vector
        })
        
        return True
    
    def aggregate_round(
        self,
        round_number: int,
        decrypt_key: Optional[RSA.RsaKey] = None
    ) -> Optional[np.ndarray]:
        """
        Aggregate submissions for a round.
        
        Args:
            round_number: Round to aggregate
            decrypt_key: Key for decryption (if authorized)
            
        Returns:
            Aggregated result or None
        """
        if round_number not in self.collected_shares:
            return None
        
        shares = self.collected_shares[round_number]
        
        if len(shares) < self.threshold:
            return None
        
        # For demonstration - decrypt and aggregate
        # In practice, use HE or MPC for aggregation
        if decrypt_key is None:
            decrypt_key = self.homomorphic_ops.key
        
        aggregated = None
        for share_data in shares:
            vector = self.homomorphic_ops.decrypt_vector(
                share_data["vector"],
                decrypt_key
            )
            
            if aggregated is None:
                aggregated = vector
            else:
                aggregated += vector
        
        # Average
        if aggregated is not None:
            aggregated /= len(shares)
        
        return aggregated