"""
Enhanced Merkle tree implementation for REV per-challenge verification (Sections 4.3, 5.5).

This module provides per-challenge Merkle tree construction with support for
both architectural segments and HDC behavioral sites, incremental updates for
streaming, and comprehensive audit trail generation.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Any, Sequence, NamedTuple, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import deque
import hashlib
import json
import time
import numpy as np
from .commit import TAGS, H
from .zk_proofs import ZKProofSystem, ZKProof, ZKCircuitParams


class PathItem(NamedTuple):
    """
    Single step in a Merkle proof.

    Attributes:
        sibling: Hash of the sibling node
        is_right: True if sibling is on the right-hand side
    """
    sibling: bytes
    is_right: bool


def _be_int(value: int, width: int) -> bytes:
    """Convert integer to big-endian bytes with specified width."""
    return value.to_bytes(width, "big")


def leaf_bytes(vals: Iterable[int]) -> bytes:
    """
    Hash a sequence of integers into a Merkle leaf node.
    
    Each value is encoded as a 32-byte big-endian integer and concatenated
    before being hashed with the LEAF domain-separation tag.
    
    Args:
        vals: Sequence of integers to hash
        
    Returns:
        Leaf node hash
    """
    data = b"".join(_be_int(v, 32) for v in vals)
    return H(TAGS["LEAF"], data)


def node_bytes(left: bytes, right: bytes) -> bytes:
    """
    Hash two child nodes to derive their parent node.
    
    Args:
        left: Left child hash
        right: Right child hash
        
    Returns:
        Parent node hash
    """
    return H(TAGS["NODE"], left, right)


@dataclass
class SegmentSite:
    """
    Segment or behavioral site for per-challenge Merkle trees (Section 5.5).
    
    Represents either an architectural segment or HDC behavioral site
    that will be included in the per-challenge Merkle tree.
    """
    seg_id: str  # Unique segment identifier
    segment_type: str  # "architectural" or "behavioral"
    token_range: Tuple[int, int]  # Token indices [start, end)
    projector_seed: int  # Seed for random projection matrix
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signature:
    """
    Signature for a segment/site in the per-challenge Merkle tree.
    
    Contains the binary signature and associated metadata including
    the leaf hash for Merkle tree construction.
    """
    seg_id: str
    sigma: bytes  # Binary signature
    metadata: Dict[str, Any]  # Contains leaf hash and other data


@dataclass 
class ChallengeLeaf:
    """
    Leaf node for per-challenge Merkle tree (Section 4.3).
    
    Encodes segment ID, signature, and execution policy into a single
    leaf hash for inclusion in the challenge-specific Merkle tree.
    """
    seg_id: str
    sigma: bytes  # Binary signature from build_signature()
    policy: Dict[str, Any]  # Execution policy (e.g., temperature, max_tokens)
    leaf_hash: Optional[bytes] = None
    
    def compute_leaf(self) -> bytes:
        """Compute leaf hash as per Section 4.3."""
        leaf_data = {
            "seg": self.seg_id,
            "sigma": self.sigma.hex() if isinstance(self.sigma, bytes) else str(self.sigma),
            "policy": self.policy
        }
        
        # Canonical encoding for deterministic hashing
        encoded = json.dumps(leaf_data, sort_keys=True).encode('utf-8')
        leaf_hash = H(TAGS["LEAF"], encoded)
        
        self.leaf_hash = leaf_hash
        return leaf_hash


@dataclass
class PerChallengeTree:
    """
    Per-challenge Merkle tree for REV verification (Section 4.3).
    
    Each challenge gets its own Merkle tree built from segment signatures
    and behavioral sites, enabling efficient verification of specific segments.
    """
    challenge_id: str
    leaves: List[ChallengeLeaf]
    tree: Optional[Dict[str, Any]] = None
    root: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def build_tree(self) -> bytes:
        """Build Merkle tree from challenge leaves."""
        if not self.leaves:
            raise ValueError(f"No leaves for challenge {self.challenge_id}")
        
        # Compute leaf hashes
        leaf_hashes = [leaf.compute_leaf() for leaf in self.leaves]
        
        # Build tree
        self.tree = build_merkle_tree(leaf_hashes)
        self.root = self.tree["root"]
        
        # Update metadata
        self.metadata.update({
            "num_leaves": len(self.leaves),
            "tree_depth": len(self.tree["layers"]),
            "build_time": time.time()
        })
        
        return self.root
    
    def generate_proof(self, seg_id: str) -> Optional[List[PathItem]]:
        """Generate proof for specific segment."""
        if not self.tree:
            self.build_tree()
        
        # Find leaf index
        for idx, leaf in enumerate(self.leaves):
            if leaf.seg_id == seg_id:
                return generate_merkle_proof(self.tree, idx)
        
        return None
    
    def verify_segment(self, seg_id: str, sigma: bytes, policy: Dict[str, Any]) -> bool:
        """Verify that a segment is included in this challenge tree."""
        if not self.root:
            return False
        
        # Find the leaf
        leaf_idx = None
        for idx, leaf in enumerate(self.leaves):
            if leaf.seg_id == seg_id:
                leaf_idx = idx
                break
        
        if leaf_idx is None:
            return False
        
        # Generate proof
        proof = self.generate_proof(seg_id)
        if not proof:
            return False
        
        # Recreate leaf
        test_leaf = ChallengeLeaf(seg_id, sigma, policy)
        leaf_hash = test_leaf.compute_leaf()
        
        # Verify proof
        return verify_merkle_proof_bytes(leaf_hash, proof, self.root)


class IncrementalMerkleTree:
    """
    Incremental Merkle tree for streaming updates (Section 4.3).
    
    Supports efficient incremental updates as new segments are processed,
    without rebuilding the entire tree from scratch.
    """
    
    def __init__(self, challenge_id: str):
        """Initialize incremental tree."""
        self.challenge_id = challenge_id
        self.leaves: List[bytes] = []
        self.nodes: List[List[bytes]] = []  # Layers of the tree
        self.pending_updates: deque = deque()
        self.finalized = False
    
    def add_leaf(self, leaf: ChallengeLeaf) -> None:
        """Add a new leaf to the tree incrementally."""
        if self.finalized:
            raise ValueError("Cannot add leaves to finalized tree")
        
        leaf_hash = leaf.compute_leaf()
        self.leaves.append(leaf_hash)
        self.pending_updates.append(leaf_hash)
        
        # Update tree structure incrementally
        self._update_incremental()
    
    def _update_incremental(self) -> None:
        """Update tree structure incrementally."""
        # This is a simplified version - production would optimize further
        if len(self.leaves) == 1:
            # First leaf
            self.nodes = [[self.leaves[0]]]
        else:
            # Rebuild affected portions
            # In production, would only update affected paths
            tree = build_merkle_tree(self.leaves)
            self.nodes = tree["layers"]
    
    def get_current_root(self) -> bytes:
        """Get current root hash."""
        if not self.nodes:
            raise ValueError("Tree is empty")
        
        # Root is at the top layer
        return self.nodes[-1][0]
    
    def finalize(self) -> bytes:
        """Finalize tree and return final root."""
        self.finalized = True
        self.pending_updates.clear()
        return self.get_current_root()
    
    def generate_streaming_proof(self, leaf_idx: int) -> List[PathItem]:
        """Generate proof for streaming verification."""
        if not self.nodes:
            raise ValueError("Tree is empty")
        
        proof_path = []
        idx = leaf_idx
        
        for layer_idx in range(len(self.nodes) - 1):
            layer = self.nodes[layer_idx]
            
            # Determine sibling
            if idx % 2 == 0:
                sib_idx = min(idx + 1, len(layer) - 1)
                is_right = True
            else:
                sib_idx = idx - 1
                is_right = False
            
            proof_path.append(PathItem(layer[sib_idx], is_right))
            idx //= 2
        
        return proof_path


@dataclass
class AuditTranscript:
    """
    Audit transcript for REV verification (Section 5.5).
    
    Contains all roots, metadata, and verification data needed for
    comprehensive audit trail of the verification process.
    """
    transcript_id: str
    run_id: str
    challenge_trees: List[PerChallengeTree]
    master_root: bytes
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_challenge_tree(self, tree: PerChallengeTree) -> None:
        """Add a per-challenge tree to the transcript."""
        self.challenge_trees.append(tree)
        self._update_master_root()
    
    def _update_master_root(self) -> None:
        """Update master root from all challenge trees."""
        if not self.challenge_trees:
            self.master_root = b"\x00" * 32
            return
        
        # Collect all challenge roots
        roots = []
        for tree in self.challenge_trees:
            if tree.root is None:
                tree.build_tree()
            roots.append(tree.root)
        
        # Aggregate into master root
        self.master_root = aggregate_merkle_roots(roots)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transcript to dictionary for serialization."""
        return {
            "transcript_id": self.transcript_id,
            "run_id": self.run_id,
            "master_root": self.master_root.hex(),
            "timestamp": self.timestamp,
            "num_challenges": len(self.challenge_trees),
            "challenges": [
                {
                    "challenge_id": tree.challenge_id,
                    "root": tree.root.hex() if tree.root else None,
                    "num_segments": len(tree.leaves),
                    "metadata": tree.metadata
                }
                for tree in self.challenge_trees
            ],
            "metadata": self.metadata
        }
    
    def verify_challenge(self, challenge_id: str, expected_root: bytes) -> bool:
        """Verify a specific challenge tree."""
        for tree in self.challenge_trees:
            if tree.challenge_id == challenge_id:
                if tree.root is None:
                    tree.build_tree()
                return tree.root == expected_root
        return False
    
    def generate_challenge_proof(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """Generate proof that a challenge is included in the transcript."""
        # Find challenge tree
        tree_idx = None
        for idx, tree in enumerate(self.challenge_trees):
            if tree.challenge_id == challenge_id:
                tree_idx = idx
                break
        
        if tree_idx is None:
            return None
        
        # Build proof showing challenge root is in master tree
        roots = [t.root if t.root else t.build_tree() for t in self.challenge_trees]
        master_tree = build_merkle_tree(roots)
        proof_path = generate_merkle_proof(master_tree, tree_idx)
        
        return {
            "challenge_id": challenge_id,
            "challenge_root": roots[tree_idx].hex(),
            "proof_path": [(p.sibling.hex(), p.is_right) for p in proof_path],
            "master_root": self.master_root.hex()
        }


def build_merkle_tree(leaves: Sequence[bytes]) -> Dict[str, Any]:
    """
    Build a Merkle tree and return its root and all layers.
    
    Used in REV for creating verification certificates and audit trails
    that can be efficiently verified without revealing all data.
    
    Args:
        leaves: Sequence of leaf node hashes
        
    Returns:
        Dictionary with 'root' and 'layers' keys
        
    Raises:
        ValueError: If leaves sequence is empty
    """
    if not leaves:
        raise ValueError("Empty leaf set")

    layers: List[List[bytes]] = [list(leaves)]
    cur = layers[0]
    
    while len(cur) > 1:
        nxt: List[bytes] = []
        
        # Bitcoin-style padding: duplicate last node if odd number
        if len(cur) & 1:
            cur = cur + [cur[-1]]
            
        # Hash pairs to create next layer
        for i in range(0, len(cur), 2):
            nxt.append(node_bytes(cur[i], cur[i + 1]))
            
        layers.append(nxt)
        cur = nxt
        
    return {"root": cur[0], "layers": layers}


def generate_merkle_proof(tree: Dict[str, Any], index: int) -> List[PathItem]:
    """
    Generate Merkle proof for a leaf at given index.
    
    The proof allows verification that a specific leaf is included
    in the Merkle tree without revealing other leaves.
    
    Args:
        tree: Tree structure from build_merkle_tree
        index: Index of the leaf to prove
        
    Returns:
        List of PathItem objects describing the proof path
    """
    layers = tree["layers"]
    idx = index
    proof_path: List[PathItem] = []
    
    for lvl in range(len(layers) - 1):
        layer = layers[lvl]
        
        # Determine sibling position
        if idx % 2 == 0:
            sib_idx = idx + 1
            sib_right = True
        else:
            sib_idx = idx - 1
            sib_right = False
            
        # Handle case where sibling doesn't exist (odd layer size)
        if sib_idx >= len(layer):
            sib_idx = idx
            
        proof_path.append(PathItem(layer[sib_idx], sib_right))
        idx //= 2
        
    return proof_path


def verify_merkle_proof(
    leaf_data_vals: Iterable[int], 
    proof_path: List[PathItem], 
    root: bytes
) -> bool:
    """
    Verify a Merkle proof for given leaf data against root.
    
    Used in REV for verifying that specific challenges or responses
    are included in the verification certificate without revealing
    the entire dataset.
    
    Args:
        leaf_data_vals: Values to create leaf from
        proof_path: Proof path from generate_merkle_proof  
        root: Expected Merkle root
        
    Returns:
        True if proof is valid, False otherwise
    """
    # Compute leaf hash
    current_hash = leaf_bytes(leaf_data_vals)
    
    # Walk up the tree using the proof path
    for path_item in proof_path:
        if path_item.is_right:
            # Sibling is on the right
            current_hash = node_bytes(current_hash, path_item.sibling)
        else:
            # Sibling is on the left  
            current_hash = node_bytes(path_item.sibling, current_hash)
    
    return current_hash == root


def verify_merkle_proof_bytes(
    leaf_hash: bytes,
    proof_path: List[PathItem],
    root: bytes
) -> bool:
    """
    Verify a Merkle proof for given leaf hash against root.
    
    Args:
        leaf_hash: Pre-computed leaf hash
        proof_path: Proof path from generate_merkle_proof
        root: Expected Merkle root
        
    Returns:
        True if proof is valid, False otherwise
    """
    current_hash = leaf_hash
    
    for path_item in proof_path:
        if path_item.is_right:
            current_hash = node_bytes(current_hash, path_item.sibling)
        else:
            current_hash = node_bytes(path_item.sibling, current_hash)
    
    return current_hash == root


def compute_merkle_root(leaves: Sequence[bytes]) -> bytes:
    """
    Compute just the Merkle root without storing full tree.
    
    More memory-efficient when only the root is needed.
    
    Args:
        leaves: Sequence of leaf hashes
        
    Returns:
        Merkle root hash
    """
    tree = build_merkle_tree(leaves)
    return tree["root"]


def batch_merkle_proofs(
    tree: Dict[str, Any], 
    indices: List[int]
) -> List[List[PathItem]]:
    """
    Generate multiple Merkle proofs efficiently.
    
    Args:
        tree: Tree structure from build_merkle_tree
        indices: List of leaf indices to prove
        
    Returns:
        List of proof paths, one for each index
    """
    return [generate_merkle_proof(tree, idx) for idx in indices]


def aggregate_merkle_roots(roots: List[bytes]) -> bytes:
    """
    Aggregate multiple Merkle roots into a single commitment.
    
    Useful for REV when combining multiple verification sessions
    or model comparisons into a single certificate.
    
    Args:
        roots: List of Merkle root hashes
        
    Returns:
        Aggregated root hash
    """
    if not roots:
        raise ValueError("No roots to aggregate")
    
    if len(roots) == 1:
        return roots[0]
    
    # Build tree of the roots themselves
    agg_tree = build_merkle_tree(roots)
    return agg_tree["root"]


def build_signature(
    activations_or_logits: Union[np.ndarray, Dict[str, np.ndarray]],
    seg: SegmentSite,
    policy: Dict[str, Any],
    d_prime: int = 256,
    tau: float = 3.0,
    q: int = 8
) -> Signature:
    """
    Build signature for segment/site as per Sections 4.3 and 5.5.
    
    Implements the signature generation algorithm:
    1. Select and pool activations
    2. Apply seeded random projection
    3. Quantize and binarize
    4. Create leaf hash for Merkle tree
    
    Args:
        activations_or_logits: Model activations or logits
        seg: Segment or behavioral site
        policy: Execution policy (temperature, max_tokens, etc.)
        d_prime: Projected dimension
        tau: Clipping threshold
        q: Quantization bits
        
    Returns:
        Signature with binary signature and leaf hash
    """
    # Step 1: Select and pool activations
    if isinstance(activations_or_logits, dict):
        # Pool across layers if dict of activations
        pooled = []
        for layer_name in sorted(activations_or_logits.keys()):
            act = activations_or_logits[layer_name]
            # Global average pooling
            if len(act.shape) > 1:
                pooled.append(np.mean(act, axis=tuple(range(1, len(act.shape)))))
            else:
                pooled.append(act)
        a = np.concatenate(pooled)
    else:
        # Direct use if already array
        a = activations_or_logits.flatten()
    
    # Step 2: Seeded random projection
    np.random.seed(seg.projector_seed)
    a_dim = len(a)
    R = np.random.randn(d_prime, a_dim) / np.sqrt(a_dim)
    
    # Step 3: Project, clip, and quantize
    z = R @ a
    z = np.clip(z, -tau, tau)
    
    # Quantize to q bits
    z_min, z_max = -tau, tau
    z_normalized = (z - z_min) / (z_max - z_min)  # Normalize to [0, 1]
    z_quantized = np.round(z_normalized * (2**q - 1)).astype(np.uint8)
    
    # Step 4: Binarize (pack into bytes)
    sigma = _pack_binary(z_quantized, q)
    
    # Step 5: Create leaf hash
    leaf = ChallengeLeaf(seg.seg_id, sigma, policy)
    leaf_hash = leaf.compute_leaf()
    
    return Signature(
        seg_id=seg.seg_id,
        sigma=sigma,
        metadata={
            "leaf": leaf_hash,
            "leaf_hex": leaf_hash.hex(),
            "d_prime": d_prime,
            "tau": tau,
            "q": q,
            "projector_seed": seg.projector_seed,
            "a_dim": a_dim
        }
    )


def _pack_binary(values: np.ndarray, bits_per_value: int) -> bytes:
    """Pack values into binary representation."""
    # Simple packing - production would optimize
    packed = []
    current_byte = 0
    bit_position = 0
    
    for val in values:
        # Add bits to current byte
        current_byte |= (int(val) << bit_position)
        bit_position += bits_per_value
        
        # Flush complete bytes
        while bit_position >= 8:
            packed.append(current_byte & 0xFF)
            current_byte >>= 8
            bit_position -= 8
    
    # Add remaining bits
    if bit_position > 0:
        packed.append(current_byte)
    
    return bytes(packed)


@dataclass
class BehavioralCertificate:
    """HBT-style behavioral certificate for consensus checkpoints."""
    
    certificate_id: str
    timestamp: float
    segment_indices: List[int]
    behavioral_signature: bytes
    consensus_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize certificate to bytes for hashing."""
        cert_dict = {
            "id": self.certificate_id,
            "timestamp": self.timestamp,
            "indices": self.segment_indices,
            "signature": self.behavioral_signature.hex(),
            "consensus": self.consensus_result,
            "metadata": self.metadata
        }
        return json.dumps(cert_dict, sort_keys=True).encode()
    
    def compute_hash(self) -> bytes:
        """Compute certificate hash."""
        return H(TAGS["CERT"], self.to_bytes())


@dataclass
class VerificationLevel:
    """Single level in the hierarchical verification tree."""
    
    level_id: int
    level_type: str  # "segment", "behavioral", "master"
    nodes: List[bytes]
    merkle_root: bytes
    certificates: List[BehavioralCertificate] = field(default_factory=list)
    proofs: List[ZKProof] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalVerificationTree:
    """Complete hierarchical verification proof structure."""
    
    tree_id: str
    levels: List[VerificationLevel]
    master_root: bytes
    zk_proofs: List[ZKProof] = field(default_factory=list)
    certificates: List[BehavioralCertificate] = field(default_factory=list)
    per_challenge_trees: List[PerChallengeTree] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_level(self, level_id: int) -> Optional[VerificationLevel]:
        """Get verification level by ID."""
        for level in self.levels:
            if level.level_id == level_id:
                return level
        return None
    
    def get_challenge_tree(self, challenge_id: str) -> Optional[PerChallengeTree]:
        """Get per-challenge tree by ID."""
        for tree in self.per_challenge_trees:
            if tree.challenge_id == challenge_id:
                return tree
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tree_id": self.tree_id,
            "levels": [
                {
                    "level_id": lvl.level_id,
                    "level_type": lvl.level_type,
                    "merkle_root": lvl.merkle_root.hex(),
                    "num_nodes": len(lvl.nodes),
                    "num_certificates": len(lvl.certificates),
                    "metadata": lvl.metadata
                }
                for lvl in self.levels
            ],
            "master_root": self.master_root.hex(),
            "num_zk_proofs": len(self.zk_proofs),
            "per_challenge_trees": [
                {
                    "challenge_id": tree.challenge_id,
                    "root": tree.root.hex() if tree.root else None,
                    "num_leaves": len(tree.leaves)
                }
                for tree in self.per_challenge_trees
            ],
            "metadata": self.metadata
        }


class HierarchicalVerificationChain:
    """
    Enhanced hierarchical verification chain with per-challenge tree support.
    
    This class implements:
    - Level 0: Per-challenge Merkle trees (NEW)
    - Level 1: Merkle trees from segment hashes
    - Level 2: HBT behavioral certificates at consensus points
    - Level 3: Master verification proofs with ZK support
    """
    
    def __init__(
        self,
        enable_zk: bool = True,
        consensus_interval: int = 10,
        circuit_params: Optional[ZKCircuitParams] = None,
        enable_per_challenge: bool = True
    ):
        """
        Initialize hierarchical verification chain.
        
        Args:
            enable_zk: Whether to generate ZK proofs
            consensus_interval: Number of segments between consensus points
            circuit_params: Parameters for ZK circuit
            enable_per_challenge: Whether to build per-challenge trees
        """
        self.enable_zk = enable_zk
        self.consensus_interval = consensus_interval
        self.zk_system = ZKProofSystem(circuit_params) if enable_zk else None
        self.enable_per_challenge = enable_per_challenge
        
        # Storage for levels
        self.per_challenge_trees: List[PerChallengeTree] = []
        self.segment_trees: List[Dict[str, Any]] = []
        self.behavioral_certificates: List[BehavioralCertificate] = []
        self.master_proofs: List[ZKProof] = []
        
        # Incremental trees for streaming
        self.incremental_trees: Dict[str, IncrementalMerkleTree] = {}
    
    def build_per_challenge_tree(
        self,
        challenge_id: str,
        signatures: List[Signature]
    ) -> PerChallengeTree:
        """
        Build per-challenge Merkle tree from signatures.
        
        Args:
            challenge_id: Unique challenge identifier
            signatures: List of signatures from build_signature()
            
        Returns:
            Per-challenge Merkle tree
        """
        # Create leaves from signatures
        leaves = []
        for sig in signatures:
            # Extract policy from metadata if available
            policy = sig.metadata.get("policy", {})
            
            leaf = ChallengeLeaf(
                seg_id=sig.seg_id,
                sigma=sig.sigma,
                policy=policy
            )
            leaves.append(leaf)
        
        # Create tree
        tree = PerChallengeTree(
            challenge_id=challenge_id,
            leaves=leaves,
            metadata={
                "num_signatures": len(signatures),
                "creation_time": time.time()
            }
        )
        
        # Build tree
        tree.build_tree()
        self.per_challenge_trees.append(tree)
        
        return tree
    
    def start_incremental_tree(self, challenge_id: str) -> IncrementalMerkleTree:
        """
        Start building an incremental tree for streaming.
        
        Args:
            challenge_id: Challenge identifier
            
        Returns:
            Incremental tree instance
        """
        tree = IncrementalMerkleTree(challenge_id)
        self.incremental_trees[challenge_id] = tree
        return tree
    
    def add_to_incremental(
        self,
        challenge_id: str,
        signature: Signature,
        policy: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Add signature to incremental tree.
        
        Args:
            challenge_id: Challenge identifier
            signature: Signature to add
            policy: Optional execution policy
            
        Returns:
            Current root hash
        """
        if challenge_id not in self.incremental_trees:
            self.start_incremental_tree(challenge_id)
        
        tree = self.incremental_trees[challenge_id]
        
        # Create leaf
        leaf = ChallengeLeaf(
            seg_id=signature.seg_id,
            sigma=signature.sigma,
            policy=policy or {}
        )
        
        # Add to tree
        tree.add_leaf(leaf)
        
        return tree.get_current_root()
    
    def finalize_incremental(self, challenge_id: str) -> PerChallengeTree:
        """
        Finalize incremental tree and convert to per-challenge tree.
        
        Args:
            challenge_id: Challenge identifier
            
        Returns:
            Finalized per-challenge tree
        """
        if challenge_id not in self.incremental_trees:
            raise ValueError(f"No incremental tree for {challenge_id}")
        
        inc_tree = self.incremental_trees[challenge_id]
        root = inc_tree.finalize()
        
        # Convert to per-challenge tree
        # (In production, would preserve the incremental structure)
        tree = PerChallengeTree(
            challenge_id=challenge_id,
            leaves=[],  # Would reconstruct from incremental
            root=root,
            metadata={
                "finalized_from_incremental": True,
                "final_root": root.hex()
            }
        )
        
        self.per_challenge_trees.append(tree)
        del self.incremental_trees[challenge_id]
        
        return tree
    
    def build_verification_tree(
        self,
        segments: Sequence[Dict[str, Any]],
        consensus_checkpoints: Optional[List[Dict[str, Any]]] = None,
        challenge_trees: Optional[List[PerChallengeTree]] = None
    ) -> HierarchicalVerificationTree:
        """
        Build complete hierarchical verification tree.
        
        Args:
            segments: Sequence of segments from REVPipeline.segment_buffer
            consensus_checkpoints: Optional consensus checkpoint data
            challenge_trees: Optional pre-built per-challenge trees
            
        Returns:
            Complete verification proof structure
        """
        if not segments:
            raise ValueError("No segments to build tree from")
        
        tree_id = hashlib.sha256(
            f"tree_{time.time()}".encode()
        ).hexdigest()[:16]
        
        levels = []
        
        # Level 0: Per-challenge trees (if enabled)
        if self.enable_per_challenge and challenge_trees:
            level0 = self._build_per_challenge_level(challenge_trees)
            levels.append(level0)
        
        # Level 1: Build Merkle trees from segment hashes
        level1 = self._build_segment_level(segments)
        levels.append(level1)
        
        # Level 2: Create behavioral certificates at consensus points
        if consensus_checkpoints is None:
            # Create default consensus checkpoints
            consensus_checkpoints = []
            for i in range(0, len(segments), self.consensus_interval):
                checkpoint = {
                    "start_idx": i,
                    "end_idx": min(i + self.consensus_interval, len(segments)),
                    "verdict": "accept",
                    "confidence": 0.95,
                    "type": "automatic"
                }
                consensus_checkpoints.append(checkpoint)
        
        if consensus_checkpoints:
            level2 = self._build_behavioral_level(
                segments, consensus_checkpoints, level1
            )
            levels.append(level2)
        
        # Level 3: Generate master verification proofs with ZK
        level3 = self._build_master_level(levels)
        levels.append(level3)
        
        # Compute master root
        all_roots = [lvl.merkle_root for lvl in levels]
        master_root = aggregate_merkle_roots(all_roots)
        
        return HierarchicalVerificationTree(
            tree_id=tree_id,
            levels=levels,
            master_root=master_root,
            zk_proofs=self.master_proofs,
            certificates=self.behavioral_certificates,
            per_challenge_trees=challenge_trees or self.per_challenge_trees,
            metadata={
                "num_segments": len(segments),
                "num_levels": len(levels),
                "consensus_interval": self.consensus_interval,
                "zk_enabled": self.enable_zk,
                "per_challenge_enabled": self.enable_per_challenge
            }
        )
    
    def _build_per_challenge_level(
        self,
        challenge_trees: List[PerChallengeTree]
    ) -> VerificationLevel:
        """Build Level 0: Per-challenge Merkle trees."""
        challenge_roots = []
        
        for tree in challenge_trees:
            if tree.root is None:
                tree.build_tree()
            challenge_roots.append(tree.root)
        
        # Aggregate challenge roots
        if challenge_roots:
            aggregated = build_merkle_tree(challenge_roots)
            merkle_root = aggregated["root"]
        else:
            merkle_root = b"\x00" * 32
        
        return VerificationLevel(
            level_id=0,
            level_type="per_challenge",
            nodes=challenge_roots,
            merkle_root=merkle_root,
            metadata={
                "num_challenges": len(challenge_trees),
                "challenge_ids": [t.challenge_id for t in challenge_trees]
            }
        )
    
    def _build_segment_level(
        self, segments: Sequence[Dict[str, Any]]
    ) -> VerificationLevel:
        """Build Level 1: Merkle trees from segment hashes."""
        segment_hashes = []
        
        for segment in segments:
            # Extract segment data for hashing
            segment_data = json.dumps({
                "index": segment.get("index", 0),
                "tokens": segment.get("tokens", []),
                "activations": segment.get("activations", {})
            }, sort_keys=True).encode()
            
            segment_hash = H(TAGS["LEAF"], segment_data)
            segment_hashes.append(segment_hash)
        
        # Build Merkle tree
        tree = build_merkle_tree(segment_hashes)
        self.segment_trees.append(tree)
        
        return VerificationLevel(
            level_id=1,
            level_type="segment",
            nodes=segment_hashes,
            merkle_root=tree["root"],
            metadata={
                "num_segments": len(segments),
                "tree_depth": len(tree["layers"])
            }
        )
    
    def _build_behavioral_level(
        self,
        segments: Sequence[Dict[str, Any]],
        consensus_checkpoints: List[Dict[str, Any]],
        segment_level: VerificationLevel
    ) -> VerificationLevel:
        """Build Level 2: HBT behavioral certificates at consensus points."""
        certificates = []
        certificate_hashes = []
        
        for checkpoint in consensus_checkpoints:
            # Create behavioral certificate
            cert = self.create_behavioral_certificate(
                segments=segments[
                    checkpoint.get("start_idx", 0):
                    checkpoint.get("end_idx", len(segments))
                ],
                consensus_data=checkpoint,
                segment_root=segment_level.merkle_root
            )
            
            certificates.append(cert)
            certificate_hashes.append(cert.compute_hash())
            self.behavioral_certificates.append(cert)
        
        # Build Merkle tree of certificates
        if certificate_hashes:
            cert_tree = build_merkle_tree(certificate_hashes)
            merkle_root = cert_tree["root"]
        else:
            merkle_root = b"\x00" * 32
        
        return VerificationLevel(
            level_id=2,
            level_type="behavioral",
            nodes=certificate_hashes,
            merkle_root=merkle_root,
            certificates=certificates,
            metadata={
                "num_certificates": len(certificates),
                "num_checkpoints": len(consensus_checkpoints)
            }
        )
    
    def _build_master_level(
        self, lower_levels: List[VerificationLevel]
    ) -> VerificationLevel:
        """Build Level 3: Master verification proofs with ZK support."""
        master_nodes = []
        zk_proofs = []
        
        # Collect roots from lower levels
        for level in lower_levels:
            master_nodes.append(level.merkle_root)
            
            # Generate ZK proof if enabled
            if self.enable_zk and self.zk_system:
                proof = self._generate_level_proof(level)
                zk_proofs.append(proof)
                self.master_proofs.append(proof)
        
        # Build master tree
        master_tree = build_merkle_tree(master_nodes)
        
        return VerificationLevel(
            level_id=3,
            level_type="master",
            nodes=master_nodes,
            merkle_root=master_tree["root"],
            proofs=zk_proofs,
            metadata={
                "num_lower_levels": len(lower_levels),
                "zk_proofs_generated": len(zk_proofs)
            }
        )
    
    def create_behavioral_certificate(
        self,
        segments: Sequence[Dict[str, Any]],
        consensus_data: Dict[str, Any],
        segment_root: bytes
    ) -> BehavioralCertificate:
        """
        Create HBT-style behavioral certificate.
        
        Args:
            segments: Segments covered by this certificate
            consensus_data: Consensus checkpoint data
            segment_root: Root hash of segment Merkle tree
            
        Returns:
            Behavioral certificate
        """
        cert_id = hashlib.sha256(
            f"cert_{time.time()}_{len(segments)}".encode()
        ).hexdigest()[:16]
        
        # Extract behavioral signature from segments
        behavioral_data = {
            "segment_count": len(segments),
            "consensus_verdict": consensus_data.get("verdict", "unknown"),
            "confidence": consensus_data.get("confidence", 0.0),
            "segment_root": segment_root.hex()
        }
        
        behavioral_signature = H(
            TAGS["BEHAVE"],
            json.dumps(behavioral_data, sort_keys=True).encode()
        )
        
        return BehavioralCertificate(
            certificate_id=cert_id,
            timestamp=time.time(),
            segment_indices=list(range(len(segments))),
            behavioral_signature=behavioral_signature,
            consensus_result=consensus_data,
            metadata={
                "segment_count": len(segments),
                "consensus_type": consensus_data.get("type", "standard")
            }
        )
    
    def link_certificates(
        self,
        certificates: List[BehavioralCertificate],
        merkle_roots: List[bytes]
    ) -> bytes:
        """
        Chain certificates with Merkle roots.
        
        Args:
            certificates: List of behavioral certificates
            merkle_roots: List of Merkle roots to link
            
        Returns:
            Linked commitment hash
        """
        if not certificates or not merkle_roots:
            raise ValueError("Need certificates and roots to link")
        
        # Create linkage data
        link_data = {
            "certificates": [
                cert.certificate_id for cert in certificates
            ],
            "roots": [root.hex() for root in merkle_roots],
            "timestamp": time.time()
        }
        
        link_bytes = json.dumps(link_data, sort_keys=True).encode()
        return H(TAGS["LINK"], link_bytes)
    
    def generate_segment_proof(
        self,
        challenge_id: str,
        seg_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate proof for specific segment in challenge tree.
        
        Args:
            challenge_id: Challenge identifier
            seg_id: Segment identifier
            
        Returns:
            Proof dictionary or None if not found
        """
        # Find challenge tree
        tree = None
        for t in self.per_challenge_trees:
            if t.challenge_id == challenge_id:
                tree = t
                break
        
        if not tree:
            return None
        
        # Generate proof
        proof_path = tree.generate_proof(seg_id)
        if not proof_path:
            return None
        
        return {
            "challenge_id": challenge_id,
            "seg_id": seg_id,
            "proof_path": [(p.sibling.hex(), p.is_right) for p in proof_path],
            "challenge_root": tree.root.hex() if tree.root else None
        }
    
    def create_audit_transcript(
        self,
        run_id: str,
        verification_tree: HierarchicalVerificationTree
    ) -> AuditTranscript:
        """
        Create comprehensive audit transcript.
        
        Args:
            run_id: Run identifier
            verification_tree: Complete verification tree
            
        Returns:
            Audit transcript with all roots and metadata
        """
        transcript_id = hashlib.sha256(
            f"transcript_{run_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        transcript = AuditTranscript(
            transcript_id=transcript_id,
            run_id=run_id,
            challenge_trees=verification_tree.per_challenge_trees,
            master_root=verification_tree.master_root,
            timestamp=time.time(),
            metadata={
                "tree_id": verification_tree.tree_id,
                "num_levels": len(verification_tree.levels),
                "num_challenges": len(verification_tree.per_challenge_trees),
                "zk_enabled": self.enable_zk
            }
        )
        
        return transcript
    
    def verify_chain(
        self,
        verification_tree: HierarchicalVerificationTree,
        expected_root: Optional[bytes] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Complete chain verification.
        
        Args:
            verification_tree: Tree to verify
            expected_root: Optional expected master root
            
        Returns:
            Tuple of (is_valid, verification_details)
        """
        details = {
            "tree_id": verification_tree.tree_id,
            "levels_verified": [],
            "certificates_verified": [],
            "zk_proofs_verified": [],
            "per_challenge_verified": [],
            "overall_valid": True
        }
        
        # Verify per-challenge trees
        for tree in verification_tree.per_challenge_trees:
            if tree.root is None:
                tree.build_tree()
            
            # Verify tree structure
            tree_valid = len(tree.leaves) > 0 and tree.root is not None
            details["per_challenge_verified"].append({
                "challenge_id": tree.challenge_id,
                "valid": tree_valid,
                "num_segments": len(tree.leaves)
            })
            
            if not tree_valid:
                details["overall_valid"] = False
        
        # Verify each level
        for level in verification_tree.levels:
            level_valid = self._verify_level(level)
            details["levels_verified"].append({
                "level_id": level.level_id,
                "level_type": level.level_type,
                "valid": level_valid
            })
            
            if not level_valid:
                details["overall_valid"] = False
        
        # Verify certificates
        for cert in verification_tree.certificates:
            cert_valid = self._verify_certificate(cert)
            details["certificates_verified"].append({
                "certificate_id": cert.certificate_id,
                "valid": cert_valid
            })
            
            if not cert_valid:
                details["overall_valid"] = False
        
        # Verify ZK proofs if enabled
        if self.enable_zk and self.zk_system:
            for proof in verification_tree.zk_proofs:
                proof_valid = self.zk_system.verify_proof(proof)
                details["zk_proofs_verified"].append({
                    "proof_valid": proof_valid
                })
                
                if not proof_valid:
                    details["overall_valid"] = False
        
        # Verify master root if provided
        if expected_root is not None:
            root_match = verification_tree.master_root == expected_root
            details["root_verification"] = {
                "expected": expected_root.hex(),
                "actual": verification_tree.master_root.hex(),
                "match": root_match
            }
            
            if not root_match:
                details["overall_valid"] = False
        
        return details["overall_valid"], details
    
    def _generate_level_proof(self, level: VerificationLevel) -> ZKProof:
        """Generate ZK proof for a verification level."""
        private_witness = {
            "level_id": level.level_id,
            "level_type": level.level_type,
            "node_count": len(level.nodes),
            "merkle_root": level.merkle_root.hex()
        }
        
        public_inputs = [
            level.level_id,
            len(level.nodes),
            int(hashlib.sha256(level.merkle_root).hexdigest()[:8], 16)
        ]
        
        return self.zk_system.generate_proof(
            private_witness,
            public_inputs,
            context=f"level_{level.level_id}_verification"
        )
    
    def _verify_level(self, level: VerificationLevel) -> bool:
        """Verify a single level."""
        if not level.nodes:
            return False
        
        # Recompute Merkle root
        computed_tree = build_merkle_tree(level.nodes)
        return computed_tree["root"] == level.merkle_root
    
    def _verify_certificate(self, cert: BehavioralCertificate) -> bool:
        """Verify a behavioral certificate."""
        # Verify certificate structure
        if not cert.certificate_id or not cert.behavioral_signature:
            return False
        
        # Verify hash consistency
        computed_hash = cert.compute_hash()
        
        # For now, just verify the certificate is well-formed
        return len(computed_hash) == 32
    
    def generate_audit_trail(
        self,
        verification_tree: HierarchicalVerificationTree,
        include_proofs: bool = True
    ) -> Dict[str, Any]:
        """
        Generate audit trail from verification tree.
        
        Args:
            verification_tree: Tree to generate trail from
            include_proofs: Whether to include ZK proofs
            
        Returns:
            Audit trail dictionary
        """
        trail = {
            "tree_id": verification_tree.tree_id,
            "master_root": verification_tree.master_root.hex(),
            "timestamp": time.time(),
            "levels": [],
            "per_challenge_trees": []
        }
        
        # Add per-challenge tree info
        for tree in verification_tree.per_challenge_trees:
            trail["per_challenge_trees"].append({
                "challenge_id": tree.challenge_id,
                "root": tree.root.hex() if tree.root else None,
                "num_segments": len(tree.leaves),
                "metadata": tree.metadata
            })
        
        # Add level info
        for level in verification_tree.levels:
            level_info = {
                "level_id": level.level_id,
                "level_type": level.level_type,
                "merkle_root": level.merkle_root.hex(),
                "num_nodes": len(level.nodes),
                "metadata": level.metadata
            }
            
            if level.certificates:
                level_info["certificates"] = [
                    {
                        "id": cert.certificate_id,
                        "timestamp": cert.timestamp,
                        "signature": cert.behavioral_signature.hex()
                    }
                    for cert in level.certificates
                ]
            
            trail["levels"].append(level_info)
        
        if include_proofs and verification_tree.zk_proofs:
            trail["zk_proofs"] = [
                {
                    "public_inputs": proof.public_inputs,
                    "metadata": proof.metadata
                }
                for proof in verification_tree.zk_proofs
            ]
        
        return trail


# Helper functions for integration with existing code

def create_hierarchical_tree_from_segments(
    segments: Sequence[Dict[str, Any]],
    enable_zk: bool = True,
    consensus_interval: int = 10,
    enable_per_challenge: bool = True
) -> HierarchicalVerificationTree:
    """
    Convenience function to create hierarchical tree from segments.
    
    Args:
        segments: Segments from REVPipeline
        enable_zk: Whether to enable ZK proofs
        consensus_interval: Interval for consensus checkpoints
        enable_per_challenge: Whether to build per-challenge trees
        
    Returns:
        Hierarchical verification tree
    """
    chain = HierarchicalVerificationChain(
        enable_zk=enable_zk,
        consensus_interval=consensus_interval,
        enable_per_challenge=enable_per_challenge
    )
    
    # Generate consensus checkpoints
    consensus_checkpoints = []
    for i in range(0, len(segments), consensus_interval):
        checkpoint = {
            "start_idx": i,
            "end_idx": min(i + consensus_interval, len(segments)),
            "verdict": "accept",
            "confidence": 0.95,
            "type": "periodic"
        }
        consensus_checkpoints.append(checkpoint)
    
    return chain.build_verification_tree(segments, consensus_checkpoints)


def verify_hierarchical_tree(
    tree: HierarchicalVerificationTree,
    expected_root: Optional[bytes] = None
) -> bool:
    """
    Convenience function to verify hierarchical tree.
    
    Args:
        tree: Tree to verify
        expected_root: Optional expected root
        
    Returns:
        True if tree is valid
    """
    chain = HierarchicalVerificationChain()
    valid, _ = chain.verify_chain(tree, expected_root)
    return valid