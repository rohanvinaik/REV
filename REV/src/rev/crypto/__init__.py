"""Cryptographic primitives for REV verification."""

from .commit import H, hexH, TAGS
from .merkle import build_merkle_tree, verify_merkle_proof, PathItem
from .zk_proofs import ZKProofSystem, generate_zk_proof, verify_zk_proof

__all__ = [
    "H", "hexH", "TAGS",
    "build_merkle_tree", "verify_merkle_proof", "PathItem", 
    "ZKProofSystem", "generate_zk_proof", "verify_zk_proof"
]