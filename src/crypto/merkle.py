"""
Merkle tree implementation for REV verification certificates.

This module provides Merkle tree construction and verification for REV's
audit logging and verification certificate generation.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Any, Sequence, NamedTuple
from .commit import TAGS, H


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