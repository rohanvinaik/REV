"""Cryptographic primitives for REV verification."""

from .commit import H, hexH, TAGS
from .merkle import (
    build_merkle_tree, 
    verify_merkle_proof, 
    PathItem,
    BehavioralCertificate,
    VerificationLevel,
    HierarchicalVerificationTree,
    HierarchicalVerificationChain,
    create_hierarchical_tree_from_segments,
    verify_hierarchical_tree
)
from .zk_proofs import ZKProofSystem, generate_zk_proof, verify_zk_proof

__all__ = [
    "H", "hexH", "TAGS",
    "build_merkle_tree", "verify_merkle_proof", "PathItem", 
    "BehavioralCertificate", "VerificationLevel", 
    "HierarchicalVerificationTree", "HierarchicalVerificationChain",
    "create_hierarchical_tree_from_segments", "verify_hierarchical_tree",
    "ZKProofSystem", "generate_zk_proof", "verify_zk_proof"
]