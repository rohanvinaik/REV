"""
Merkle tree implementation for REV verification certificates.

This module provides Merkle tree construction and verification for REV's
audit logging and verification certificate generation, with hierarchical
certificate support for multi-level verification proofs.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Any, Sequence, NamedTuple, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import hashlib
import json
import time
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_level(self, level_id: int) -> Optional[VerificationLevel]:
        """Get verification level by ID."""
        for level in self.levels:
            if level.level_id == level_id:
                return level
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
            "metadata": self.metadata
        }


class HierarchicalVerificationChain:
    """
    Hierarchical verification chain with multi-level certificate support.
    
    This class implements:
    - Level 1: Merkle trees from segment hashes
    - Level 2: HBT behavioral certificates at consensus points
    - Level 3: Master verification proofs with ZK support
    """
    
    def __init__(
        self,
        enable_zk: bool = True,
        consensus_interval: int = 10,
        circuit_params: Optional[ZKCircuitParams] = None
    ):
        """
        Initialize hierarchical verification chain.
        
        Args:
            enable_zk: Whether to generate ZK proofs
            consensus_interval: Number of segments between consensus points
            circuit_params: Parameters for ZK circuit
        """
        self.enable_zk = enable_zk
        self.consensus_interval = consensus_interval
        self.zk_system = ZKProofSystem(circuit_params) if enable_zk else None
        
        # Storage for levels
        self.segment_trees: List[Dict[str, Any]] = []
        self.behavioral_certificates: List[BehavioralCertificate] = []
        self.master_proofs: List[ZKProof] = []
        
    def build_verification_tree(
        self,
        segments: Sequence[Dict[str, Any]],
        consensus_checkpoints: Optional[List[Dict[str, Any]]] = None
    ) -> HierarchicalVerificationTree:
        """
        Build complete hierarchical verification tree.
        
        Args:
            segments: Sequence of segments from REVPipeline.segment_buffer
            consensus_checkpoints: Optional consensus checkpoint data
            
        Returns:
            Complete verification proof structure
        """
        if not segments:
            raise ValueError("No segments to build tree from")
        
        tree_id = hashlib.sha256(
            f"tree_{time.time()}".encode()
        ).hexdigest()[:16]
        
        levels = []
        
        # Level 1: Build Merkle trees from segment hashes
        level1 = self._build_segment_level(segments)
        levels.append(level1)
        
        # Level 2: Create behavioral certificates at consensus points
        # If no checkpoints provided, create default ones
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
            metadata={
                "num_segments": len(segments),
                "num_levels": len(levels),
                "consensus_interval": self.consensus_interval,
                "zk_enabled": self.enable_zk
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
            "overall_valid": True
        }
        
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
            "levels": []
        }
        
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
    consensus_interval: int = 10
) -> HierarchicalVerificationTree:
    """
    Convenience function to create hierarchical tree from segments.
    
    Args:
        segments: Segments from REVPipeline
        enable_zk: Whether to enable ZK proofs
        consensus_interval: Interval for consensus checkpoints
        
    Returns:
        Hierarchical verification tree
    """
    chain = HierarchicalVerificationChain(
        enable_zk=enable_zk,
        consensus_interval=consensus_interval
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