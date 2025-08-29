"""
Commitment and hashing utilities for REV verification.

This module provides domain-separated hashing and commitment schemes
for REV's commit-reveal protocol and secure verification.
"""

from __future__ import annotations
import hashlib

# Domain separation tags for REV protocol
TAGS = {
    "LEAF": b"\x00LEAF",
    "NODE": b"\x01NODE", 
    "VK_AGG": b"VK_AGG",
    "SUBPROOF": b"SUBPROOF",
    "ACC": b"ACC",
    "PROOF_ID": b"PROOF_ID",
    "TRACE": b"TRACE",
    "CONS": b"CONS",
    "INT": b"INT",  # integrity binding
    "EMPTY_VK": b"EMPTY_VK",
    # REV-specific tags
    "REV_CHALLENGE": b"REV_CHALLENGE",
    "REV_RESPONSE": b"REV_RESPONSE", 
    "REV_COMMITMENT": b"REV_COMMITMENT",
    "REV_MODEL_SIG": b"REV_MODEL_SIG",
    "REV_VERIFY": b"REV_VERIFY",
    # Hierarchical verification tags
    "CERT": b"CERT",  # Behavioral certificate
    "BEHAVE": b"BEHAVE",  # Behavioral signature
    "LINK": b"LINK",  # Certificate linkage
}


def _len_prefix(b: bytes) -> bytes:
    """Return a 4-byte big-endian length prefix for bytes."""
    return len(b).to_bytes(4, "big")


def H(tag: bytes, *parts: bytes) -> bytes:
    """
    Domain-separated SHA-256 hash for REV protocol.
    
    The hash is computed as H(tag || len(tag) || len(part_i)||part_i ...)
    This provides collision-resistant domain separation across different
    uses of the hash function in REV.
    
    Args:
        tag: Domain separation tag  
        *parts: Data parts to hash
        
    Returns:
        32-byte hash digest
    """
    h = hashlib.sha256()
    h.update(_len_prefix(tag))
    h.update(tag)
    for part in parts:
        h.update(_len_prefix(part))
        h.update(part)
    return h.digest()


def hexH(tag: bytes, *parts: bytes) -> str:
    """Return the hexadecimal representation of domain-separated hash."""
    return H(tag, *parts).hex()


def commit_value(value: bytes, nonce: bytes) -> bytes:
    """
    Create a commitment to a value with nonce.
    
    Used in REV's commit-reveal protocol for secure challenge/response.
    
    Args:
        value: Value to commit to
        nonce: Random nonce for hiding
        
    Returns:
        Commitment hash
    """
    return H(TAGS["REV_COMMITMENT"], value, nonce)


def reveal_commitment(value: bytes, nonce: bytes, commitment: bytes) -> bool:
    """
    Verify a commitment reveal.
    
    Args:
        value: Revealed value
        nonce: Revealed nonce  
        commitment: Original commitment
        
    Returns:
        True if reveal is valid
    """
    expected_commitment = commit_value(value, nonce)
    return expected_commitment == commitment


def hash_model_signature(model_id: str, signature_data: bytes) -> bytes:
    """
    Hash a model signature for REV verification.
    
    Args:
        model_id: Identifier for the model
        signature_data: Serialized signature data
        
    Returns:
        Model signature hash
    """
    return H(TAGS["REV_MODEL_SIG"], model_id.encode(), signature_data)


def hash_challenge(challenge_data: bytes, context: bytes = b"") -> bytes:
    """
    Hash challenge data for REV protocol.
    
    Args:
        challenge_data: Challenge prompt or data
        context: Optional context information
        
    Returns:
        Challenge hash
    """
    return H(TAGS["REV_CHALLENGE"], challenge_data, context)


def hash_response(response_data: bytes, challenge_hash: bytes) -> bytes:
    """
    Hash response data bound to specific challenge.
    
    Args:
        response_data: Model response data
        challenge_hash: Hash of the challenge
        
    Returns:
        Response hash
    """
    return H(TAGS["REV_RESPONSE"], response_data, challenge_hash)