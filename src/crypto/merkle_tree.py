"""
Enhanced Merkle tree implementation with sparse trees and HSM support.
Provides efficient membership proofs and batch verification.
"""

import hashlib
import math
import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class MerkleNode:
    """Node in a Merkle tree."""
    
    hash: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    index: Optional[int] = None
    value: Optional[bytes] = None


@dataclass 
class MerkleProof:
    """Proof of membership in Merkle tree."""
    
    leaf_hash: bytes
    leaf_index: int
    siblings: List[Tuple[bytes, str]]  # (hash, 'left' or 'right')
    root: bytes


@dataclass
class SparseMerkleProof:
    """Proof for sparse Merkle tree."""
    
    key: bytes
    value: Optional[bytes]
    siblings: List[bytes]
    inclusion: bool  # True for inclusion, False for non-inclusion


class MerkleTree:
    """
    Standard Merkle tree with batch verification optimizations.
    """
    
    def __init__(
        self,
        hash_function: str = "sha256",
        enable_caching: bool = True
    ):
        """
        Initialize Merkle tree.
        
        Args:
            hash_function: Hash function to use
            enable_caching: Enable caching of intermediate hashes
        """
        self.hash_function = hash_function
        self.enable_caching = enable_caching
        self.root = None
        self.leaves = []
        self.nodes = {}
        self.cache = {} if enable_caching else None
        self.lock = threading.Lock()
        
    def build(self, data: List[bytes]):
        """
        Build Merkle tree from data.
        
        Args:
            data: List of data elements
        """
        with self.lock:
            if not data:
                self.root = None
                return
            
            # Create leaf nodes
            self.leaves = []
            for i, item in enumerate(data):
                leaf_hash = self._hash(item)
                node = MerkleNode(
                    hash=leaf_hash,
                    is_leaf=True,
                    index=i,
                    value=item
                )
                self.leaves.append(node)
                self.nodes[leaf_hash] = node
            
            # Build tree level by level
            current_level = self.leaves[:]
            
            while len(current_level) > 1:
                next_level = []
                
                # Process pairs
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    
                    if i + 1 < len(current_level):
                        right = current_level[i + 1]
                    else:
                        # Duplicate last node if odd number
                        right = left
                    
                    # Create parent node
                    parent_hash = self._hash_pair(left.hash, right.hash)
                    parent = MerkleNode(
                        hash=parent_hash,
                        left=left,
                        right=right
                    )
                    
                    next_level.append(parent)
                    self.nodes[parent_hash] = parent
                
                current_level = next_level
            
            self.root = current_level[0] if current_level else None
    
    def get_proof(self, index: int) -> Optional[MerkleProof]:
        """
        Get membership proof for element at index.
        
        Args:
            index: Index of element
            
        Returns:
            MerkleProof or None if index invalid
        """
        with self.lock:
            if index < 0 or index >= len(self.leaves):
                return None
            
            leaf = self.leaves[index]
            siblings = []
            
            # Traverse from leaf to root
            current_nodes = self.leaves[:]
            current_index = index
            
            while len(current_nodes) > 1:
                next_nodes = []
                
                for i in range(0, len(current_nodes), 2):
                    if i == current_index or i + 1 == current_index:
                        # This pair contains our node
                        if current_index % 2 == 0:
                            # Current is left, sibling is right
                            if i + 1 < len(current_nodes):
                                siblings.append((current_nodes[i + 1].hash, 'right'))
                        else:
                            # Current is right, sibling is left
                            siblings.append((current_nodes[i].hash, 'left'))
                    
                    # Create parent for next level
                    if i + 1 < len(current_nodes):
                        next_nodes.append(current_nodes[i])  # Placeholder
                    else:
                        next_nodes.append(current_nodes[i])
                
                current_nodes = next_nodes
                current_index = current_index // 2
            
            return MerkleProof(
                leaf_hash=leaf.hash,
                leaf_index=index,
                siblings=siblings,
                root=self.root.hash if self.root else b''
            )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a membership proof.
        
        Args:
            proof: MerkleProof to verify
            
        Returns:
            True if proof is valid
        """
        current_hash = proof.leaf_hash
        
        for sibling_hash, position in proof.siblings:
            if position == 'left':
                current_hash = self._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = self._hash_pair(current_hash, sibling_hash)
        
        return current_hash == proof.root
    
    def batch_verify(self, proofs: List[MerkleProof]) -> bool:
        """
        Batch verification of multiple proofs.
        
        Args:
            proofs: List of proofs to verify
            
        Returns:
            True if all proofs are valid
        """
        if not proofs:
            return True
        
        # Group proofs by root for efficiency
        proofs_by_root = defaultdict(list)
        for proof in proofs:
            proofs_by_root[proof.root].append(proof)
        
        # Verify each group in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for root, root_proofs in proofs_by_root.items():
                future = executor.submit(self._verify_group, root_proofs)
                futures.append(future)
            
            # Check all results
            for future in as_completed(futures):
                if not future.result():
                    return False
        
        return True
    
    def _verify_group(self, proofs: List[MerkleProof]) -> bool:
        """Verify a group of proofs with same root."""
        for proof in proofs:
            if not self.verify_proof(proof):
                return False
        return True
    
    def _hash(self, data: bytes) -> bytes:
        """Hash data using configured function."""
        if self.enable_caching and data in self.cache:
            return self.cache[data]
        
        if self.hash_function == "sha256":
            result = hashlib.sha256(data).digest()
        elif self.hash_function == "sha3_256":
            result = hashlib.sha3_256(data).digest()
        else:
            result = hashlib.sha256(data).digest()
        
        if self.enable_caching:
            self.cache[data] = result
        
        return result
    
    def _hash_pair(self, left: bytes, right: bytes) -> bytes:
        """Hash a pair of nodes."""
        # Sort to ensure consistent ordering
        if left <= right:
            combined = left + right
        else:
            combined = right + left
        
        return self._hash(combined)


class SparseMerkleTree:
    """
    Sparse Merkle tree for efficient membership and non-membership proofs.
    """
    
    def __init__(
        self,
        height: int = 256,
        hash_function: str = "sha256",
        default_value: bytes = b''
    ):
        """
        Initialize sparse Merkle tree.
        
        Args:
            height: Height of the tree (determines key space)
            hash_function: Hash function to use
            default_value: Default value for empty leaves
        """
        self.height = height
        self.hash_function = hash_function
        self.default_value = default_value
        
        # Storage for non-default nodes
        self.nodes: Dict[bytes, bytes] = {}
        
        # Precompute default hashes for each level
        self.default_hashes = self._compute_default_hashes()
        
        # Root hash
        self.root = self.default_hashes[0]
        
        self.lock = threading.Lock()
    
    def _compute_default_hashes(self) -> List[bytes]:
        """Compute default hashes for each level."""
        defaults = []
        current = self._hash(self.default_value)
        
        for _ in range(self.height):
            defaults.append(current)
            current = self._hash(current + current)
        
        return defaults[::-1]  # Reverse so index 0 is root level
    
    def update(self, key: bytes, value: bytes):
        """
        Update a key-value pair in the tree.
        
        Args:
            key: Key (must be hash_size bytes)
            value: Value to store
        """
        with self.lock:
            # Convert key to path (bit sequence)
            path = self._key_to_path(key)
            
            # Update nodes from leaf to root
            current_hash = self._hash(value)
            self.nodes[self._node_key(path, self.height)] = current_hash
            
            for level in range(self.height - 1, -1, -1):
                bit = path[level]
                node_key = self._node_key(path[:level], level)
                
                if bit == 0:
                    left_hash = current_hash
                    right_key = self._sibling_key(path[:level + 1], level + 1, 1)
                    right_hash = self.nodes.get(right_key, self.default_hashes[level + 1])
                else:
                    left_key = self._sibling_key(path[:level + 1], level + 1, 0)
                    left_hash = self.nodes.get(left_key, self.default_hashes[level + 1])
                    right_hash = current_hash
                
                current_hash = self._hash(left_hash + right_hash)
                self.nodes[node_key] = current_hash
            
            self.root = current_hash
    
    def get_proof(self, key: bytes) -> SparseMerkleProof:
        """
        Get membership/non-membership proof for key.
        
        Args:
            key: Key to get proof for
            
        Returns:
            SparseMerkleProof
        """
        with self.lock:
            path = self._key_to_path(key)
            siblings = []
            
            # Collect siblings along the path
            for level in range(self.height):
                bit = path[level]
                sibling_bit = 1 - bit
                sibling_key = self._sibling_key(path[:level + 1], level + 1, sibling_bit)
                sibling_hash = self.nodes.get(sibling_key, self.default_hashes[level + 1])
                siblings.append(sibling_hash)
            
            # Get leaf value
            leaf_key = self._node_key(path, self.height)
            if leaf_key in self.nodes:
                # Inclusion proof
                leaf_value = self._unhash_value(self.nodes[leaf_key])
                return SparseMerkleProof(
                    key=key,
                    value=leaf_value,
                    siblings=siblings,
                    inclusion=True
                )
            else:
                # Non-inclusion proof
                return SparseMerkleProof(
                    key=key,
                    value=None,
                    siblings=siblings,
                    inclusion=False
                )
    
    def verify_proof(self, proof: SparseMerkleProof) -> bool:
        """
        Verify a sparse Merkle proof.
        
        Args:
            proof: Proof to verify
            
        Returns:
            True if proof is valid
        """
        path = self._key_to_path(proof.key)
        
        # Start from leaf
        if proof.inclusion and proof.value is not None:
            current_hash = self._hash(proof.value)
        else:
            current_hash = self.default_hashes[self.height]
        
        # Traverse to root
        for level in range(self.height - 1, -1, -1):
            bit = path[level]
            sibling = proof.siblings[level]
            
            if bit == 0:
                current_hash = self._hash(current_hash + sibling)
            else:
                current_hash = self._hash(sibling + current_hash)
        
        return current_hash == self.root
    
    def _key_to_path(self, key: bytes) -> List[int]:
        """Convert key to binary path."""
        path = []
        for byte in key:
            for i in range(8):
                path.append((byte >> (7 - i)) & 1)
        return path[:self.height]
    
    def _node_key(self, path: List[int], level: int) -> bytes:
        """Generate storage key for node."""
        path_bytes = bytes(path) if path else b''
        return hashlib.sha256(path_bytes + level.to_bytes(2, 'big')).digest()
    
    def _sibling_key(self, path: List[int], level: int, bit: int) -> bytes:
        """Generate key for sibling node."""
        sibling_path = path[:-1] + [bit]
        return self._node_key(sibling_path, level)
    
    def _hash(self, data: bytes) -> bytes:
        """Hash data."""
        if self.hash_function == "sha256":
            return hashlib.sha256(data).digest()
        elif self.hash_function == "sha3_256":
            return hashlib.sha3_256(data).digest()
        else:
            return hashlib.sha256(data).digest()
    
    def _unhash_value(self, hash_value: bytes) -> bytes:
        """Extract original value from hash (simplified)."""
        # In practice, would need to store mapping
        return hash_value


class HSMIntegratedMerkleTree(MerkleTree):
    """
    Merkle tree with Hardware Security Module integration.
    """
    
    def __init__(
        self,
        hash_function: str = "sha256",
        enable_caching: bool = True,
        hsm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HSM-integrated Merkle tree.
        
        Args:
            hash_function: Hash function
            enable_caching: Enable caching
            hsm_config: HSM configuration
        """
        super().__init__(hash_function, enable_caching)
        self.hsm_config = hsm_config or {}
        self.hsm_client = self._init_hsm()
    
    def _init_hsm(self):
        """Initialize HSM client."""
        try:
            # Try to import HSM library (e.g., PKCS#11)
            # This is a placeholder - actual implementation would use real HSM library
            if self.hsm_config.get("type") == "softhsm":
                logger.info("Initializing SoftHSM for testing")
                return SoftHSMClient(self.hsm_config)
            elif self.hsm_config.get("type") == "aws_cloudhsm":
                logger.info("Initializing AWS CloudHSM")
                return AWSCloudHSMClient(self.hsm_config)
            else:
                logger.info("No HSM configured, using software implementation")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize HSM: {e}")
            return None
    
    def _hash(self, data: bytes) -> bytes:
        """Hash using HSM if available."""
        if self.hsm_client:
            try:
                return self.hsm_client.hash(data, self.hash_function)
            except Exception as e:
                logger.warning(f"HSM hash failed, falling back to software: {e}")
        
        # Fallback to software
        return super()._hash(data)
    
    def sign_root(self, key_id: str) -> bytes:
        """
        Sign the root hash using HSM.
        
        Args:
            key_id: HSM key identifier
            
        Returns:
            Signature bytes
        """
        if not self.root:
            raise ValueError("Tree has no root")
        
        if self.hsm_client:
            return self.hsm_client.sign(self.root.hash, key_id)
        else:
            # Software signing fallback
            import hmac
            key = key_id.encode()  # Simplified
            return hmac.new(key, self.root.hash, hashlib.sha256).digest()
    
    def verify_signature(
        self,
        signature: bytes,
        root_hash: bytes,
        key_id: str
    ) -> bool:
        """
        Verify root signature.
        
        Args:
            signature: Signature to verify
            root_hash: Root hash that was signed
            key_id: Key identifier
            
        Returns:
            True if signature is valid
        """
        if self.hsm_client:
            return self.hsm_client.verify(signature, root_hash, key_id)
        else:
            # Software verification fallback
            import hmac
            key = key_id.encode()
            expected = hmac.new(key, root_hash, hashlib.sha256).digest()
            return hmac.compare_digest(signature, expected)


class SoftHSMClient:
    """Mock SoftHSM client for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def hash(self, data: bytes, algorithm: str) -> bytes:
        """Hash using HSM."""
        # Mock implementation
        return hashlib.sha256(data).digest()
    
    def sign(self, data: bytes, key_id: str) -> bytes:
        """Sign data using HSM key."""
        # Mock implementation
        import hmac
        return hmac.new(key_id.encode(), data, hashlib.sha256).digest()
    
    def verify(self, signature: bytes, data: bytes, key_id: str) -> bool:
        """Verify signature."""
        # Mock implementation
        import hmac
        expected = hmac.new(key_id.encode(), data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)


class AWSCloudHSMClient:
    """Mock AWS CloudHSM client."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def hash(self, data: bytes, algorithm: str) -> bytes:
        """Hash using CloudHSM."""
        # Would use boto3 and CloudHSM SDK
        return hashlib.sha256(data).digest()
    
    def sign(self, data: bytes, key_id: str) -> bytes:
        """Sign using CloudHSM key."""
        # Would use CloudHSM signing API
        import hmac
        return hmac.new(key_id.encode(), data, hashlib.sha256).digest()
    
    def verify(self, signature: bytes, data: bytes, key_id: str) -> bool:
        """Verify using CloudHSM."""
        # Would use CloudHSM verification API
        import hmac
        expected = hmac.new(key_id.encode(), data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)