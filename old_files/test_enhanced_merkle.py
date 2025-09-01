#!/usr/bin/env python3
"""
Test enhanced Merkle tree functionality for per-challenge verification (Sections 4.3, 5.5).
"""

import numpy as np
import hashlib
import json
import time
from typing import List, Dict, Any

from src.crypto.merkle import (
    SegmentSite,
    Signature,
    ChallengeLeaf,
    PerChallengeTree,
    IncrementalMerkleTree,
    AuditTranscript,
    HierarchicalVerificationChain,
    build_signature,
    build_merkle_tree,
    verify_merkle_proof_bytes
)
from src.rev_pipeline import REVPipeline, Segment, ExecutionPolicy


def test_per_challenge_merkle_tree():
    """Test per-challenge Merkle tree construction."""
    print("Testing Per-Challenge Merkle Tree Construction (Section 4.3)")
    print("=" * 60)
    
    # Create test segments
    segments = []
    for i in range(5):
        seg = SegmentSite(
            seg_id=f"seg_{i}",
            segment_type="architectural",
            token_range=(i * 10, (i + 1) * 10),
            projector_seed=12345 + i,
            metadata={"layer": i}
        )
        segments.append(seg)
    
    # Create signatures for segments
    signatures = []
    policy = {"temperature": 0.0, "max_tokens": 512}
    
    for seg in segments:
        # Generate dummy activations
        activations = np.random.randn(256)
        
        # Build signature
        sig = build_signature(
            activations_or_logits=activations,
            seg=seg,
            policy=policy
        )
        signatures.append(sig)
        print(f"  Segment {seg.seg_id}: signature size = {len(sig.sigma)} bytes")
    
    # Build per-challenge tree
    leaves = []
    for sig in signatures:
        leaf = ChallengeLeaf(
            seg_id=sig.seg_id,
            sigma=sig.sigma,
            policy=policy
        )
        leaves.append(leaf)
    
    tree = PerChallengeTree(
        challenge_id="test_challenge_001",
        leaves=leaves
    )
    
    root = tree.build_tree()
    print(f"\n  Challenge tree root: {root.hex()[:32]}...")
    print(f"  Tree depth: {len(tree.tree['layers'])}")
    print(f"  Number of leaves: {len(tree.leaves)}")
    
    # Test segment verification
    test_seg = segments[2]
    test_sig = signatures[2]
    
    is_valid = tree.verify_segment(
        seg_id=test_seg.seg_id,
        sigma=test_sig.sigma,
        policy=policy
    )
    
    assert is_valid, "Segment verification should succeed"
    print(f"\n  ✓ Segment {test_seg.seg_id} verified successfully")
    
    # Test invalid segment
    is_invalid = tree.verify_segment(
        seg_id="invalid_seg",
        sigma=b"invalid",
        policy=policy
    )
    
    assert not is_invalid, "Invalid segment should not verify"
    print(f"  ✓ Invalid segment rejected correctly")
    
    print("\n✓ Per-challenge Merkle tree tests passed")


def test_build_signature():
    """Test build_signature() function implementation."""
    print("\nTesting build_signature() Function (Section 4.3)")
    print("=" * 60)
    
    # Create test segment
    seg = SegmentSite(
        seg_id="test_seg_42",
        segment_type="behavioral",
        token_range=(100, 200),
        projector_seed=0xDEADBEEF,
        metadata={"test": True}
    )
    
    # Test with array activations
    activations_array = np.random.randn(1024).astype(np.float32)
    policy = {"temperature": 0.7, "max_tokens": 256, "top_p": 0.9}
    
    sig1 = build_signature(
        activations_or_logits=activations_array,
        seg=seg,
        policy=policy,
        d_prime=128,
        tau=2.0,
        q=4
    )
    
    print(f"  Array input signature:")
    print(f"    Sigma size: {len(sig1.sigma)} bytes")
    print(f"    Leaf hash: {sig1.metadata['leaf_hex'][:32]}...")
    print(f"    Projected dim: {sig1.metadata['d_prime']}")
    print(f"    Original dim: {sig1.metadata['a_dim']}")
    
    # Test with dict activations (multi-layer)
    activations_dict = {
        "layer_0": np.random.randn(256),
        "layer_1": np.random.randn(512),
        "layer_2": np.random.randn(256)
    }
    
    sig2 = build_signature(
        activations_or_logits=activations_dict,
        seg=seg,
        policy=policy
    )
    
    print(f"\n  Dict input signature:")
    print(f"    Sigma size: {len(sig2.sigma)} bytes")
    print(f"    Combined dim: {sig2.metadata['a_dim']}")
    
    # Test determinism with same seed
    seg_same = SegmentSite(
        seg_id="test_seg_42",
        segment_type="behavioral",
        token_range=(100, 200),
        projector_seed=0xDEADBEEF,  # Same seed
        metadata={"test": True}
    )
    
    sig3 = build_signature(
        activations_or_logits=activations_array,
        seg=seg_same,
        policy=policy,
        d_prime=128,
        tau=2.0,
        q=4
    )
    
    # Should produce same projection matrix due to same seed
    # But different signature due to different random state
    print(f"\n  Determinism test:")
    print(f"    Same projector seed: {seg.projector_seed == seg_same.projector_seed}")
    print(f"    Signature lengths match: {len(sig1.sigma) == len(sig3.sigma)}")
    
    print("\n✓ build_signature() tests passed")


def test_incremental_merkle_tree():
    """Test incremental Merkle tree for streaming."""
    print("\nTesting Incremental Merkle Tree (Streaming)")
    print("=" * 60)
    
    # Create incremental tree
    inc_tree = IncrementalMerkleTree("streaming_challenge")
    
    # Add leaves incrementally
    policy = {"temperature": 0.0}
    roots = []
    
    for i in range(10):
        leaf = ChallengeLeaf(
            seg_id=f"stream_seg_{i}",
            sigma=f"sigma_{i}".encode(),
            policy=policy
        )
        
        inc_tree.add_leaf(leaf)
        current_root = inc_tree.get_current_root()
        roots.append(current_root)
        
        print(f"  After leaf {i}: root = {current_root.hex()[:16]}...")
    
    # Verify roots change as tree grows
    unique_roots = len(set(roots))
    print(f"\n  Unique roots during construction: {unique_roots}/10")
    assert unique_roots > 1, "Roots should change as tree grows"
    
    # Finalize tree
    final_root = inc_tree.finalize()
    print(f"  Final root: {final_root.hex()[:32]}...")
    
    # Try to add after finalization (should fail)
    try:
        inc_tree.add_leaf(ChallengeLeaf("late", b"late", {}))
        assert False, "Should not allow additions after finalization"
    except ValueError as e:
        print(f"  ✓ Correctly rejected post-finalization addition: {e}")
    
    # Test streaming proof generation
    proof = inc_tree.generate_streaming_proof(3)
    print(f"  Proof for leaf 3: {len(proof)} path items")
    
    print("\n✓ Incremental Merkle tree tests passed")


def test_audit_transcript():
    """Test audit transcript generation."""
    print("\nTesting Audit Transcript Generation (Section 5.5)")
    print("=" * 60)
    
    # Create multiple challenge trees
    challenge_trees = []
    
    for c in range(3):
        leaves = []
        for s in range(4):
            leaf = ChallengeLeaf(
                seg_id=f"c{c}_s{s}",
                sigma=f"sig_c{c}_s{s}".encode(),
                policy={"challenge": c}
            )
            leaves.append(leaf)
        
        tree = PerChallengeTree(
            challenge_id=f"challenge_{c}",
            leaves=leaves
        )
        tree.build_tree()
        challenge_trees.append(tree)
    
    # Create audit transcript
    transcript = AuditTranscript(
        transcript_id="audit_001",
        run_id="test_run_2024",
        challenge_trees=challenge_trees,
        master_root=b"\x00" * 32,  # Will be updated
        timestamp=time.time()
    )
    
    # Update master root
    transcript._update_master_root()
    
    print(f"  Transcript ID: {transcript.transcript_id}")
    print(f"  Run ID: {transcript.run_id}")
    print(f"  Number of challenges: {len(transcript.challenge_trees)}")
    print(f"  Master root: {transcript.master_root.hex()[:32]}...")
    
    # Test challenge verification
    expected_root = challenge_trees[1].root
    is_valid = transcript.verify_challenge("challenge_1", expected_root)
    assert is_valid, "Challenge verification should succeed"
    print(f"\n  ✓ Challenge 1 verified with correct root")
    
    # Test challenge proof generation
    proof = transcript.generate_challenge_proof("challenge_1")
    assert proof is not None, "Should generate proof"
    print(f"  ✓ Generated proof for challenge 1")
    print(f"    Proof path length: {len(proof['proof_path'])}")
    
    # Test serialization
    transcript_dict = transcript.to_dict()
    print(f"\n  Serialized transcript:")
    print(f"    Keys: {list(transcript_dict.keys())}")
    print(f"    Challenges: {transcript_dict['num_challenges']}")
    
    print("\n✓ Audit transcript tests passed")


def test_hierarchical_verification_chain():
    """Test hierarchical verification chain with per-challenge trees."""
    print("\nTesting Hierarchical Verification Chain")
    print("=" * 60)
    
    # Create verification chain
    chain = HierarchicalVerificationChain(
        enable_zk=False,  # Disable ZK for testing
        consensus_interval=5,
        enable_per_challenge=True
    )
    
    # Create test segments
    segments = []
    for i in range(10):
        segment = {
            "index": i,
            "tokens": list(range(i * 10, (i + 1) * 10)),
            "activations": {"layer_0": np.random.randn(64).tolist()}
        }
        segments.append(segment)
    
    # Build per-challenge trees
    challenge_trees = []
    for c in range(2):
        signatures = []
        for s in range(5):
            seg = SegmentSite(
                seg_id=f"c{c}_seg{s}",
                segment_type="architectural",
                token_range=(s * 10, (s + 1) * 10),
                projector_seed=c * 1000 + s
            )
            
            sig = build_signature(
                activations_or_logits=np.random.randn(128),
                seg=seg,
                policy={"challenge": c}
            )
            signatures.append(sig)
        
        tree = chain.build_per_challenge_tree(f"challenge_{c}", signatures)
        challenge_trees.append(tree)
    
    # Build complete verification tree
    verification_tree = chain.build_verification_tree(
        segments=segments,
        challenge_trees=challenge_trees
    )
    
    print(f"  Tree ID: {verification_tree.tree_id}")
    print(f"  Number of levels: {len(verification_tree.levels)}")
    print(f"  Master root: {verification_tree.master_root.hex()[:32]}...")
    
    # Verify tree structure
    for level in verification_tree.levels:
        print(f"\n  Level {level.level_id} ({level.level_type}):")
        print(f"    Nodes: {len(level.nodes)}")
        print(f"    Root: {level.merkle_root.hex()[:16]}...")
    
    # Test verification
    is_valid, details = chain.verify_chain(verification_tree)
    print(f"\n  Verification result: {'VALID' if is_valid else 'INVALID'}")
    print(f"  Levels verified: {len(details['levels_verified'])}")
    print(f"  Per-challenge verified: {len(details['per_challenge_verified'])}")
    
    assert is_valid, "Verification tree should be valid"
    
    # Test segment proof generation
    proof = chain.generate_segment_proof("challenge_0", "c0_seg2")
    assert proof is not None, "Should generate segment proof"
    print(f"\n  ✓ Generated proof for segment c0_seg2")
    
    # Create audit transcript
    audit = chain.create_audit_transcript("test_run", verification_tree)
    print(f"  ✓ Created audit transcript: {audit.transcript_id}")
    
    # Generate audit trail
    trail = chain.generate_audit_trail(verification_tree)
    print(f"  ✓ Generated audit trail with {len(trail['levels'])} levels")
    
    print("\n✓ Hierarchical verification chain tests passed")


def test_rev_pipeline_integration():
    """Test integration with REVPipeline."""
    print("\nTesting REVPipeline Integration")
    print("=" * 60)
    
    # Create pipeline
    pipeline = REVPipeline(
        segment_size=10,
        buffer_size=4
    )
    
    # Set execution policy
    policy = ExecutionPolicy(
        temperature=0.0,
        max_tokens=100,
        dtype="fp16",
        checkpoint_activations=True
    )
    pipeline.set_execution_policy(policy)
    
    # Create test segments
    segments = []
    for i in range(5):
        segment = Segment(
            segment_id=i,
            tokens=list(range(i * 10, (i + 1) * 10)),
            start_idx=i * 10,
            end_idx=(i + 1) * 10,
            overlap_group=i // 2
        )
        segment.checkpoint_data = {
            "activations": {"layer_0": np.random.randn(128)}
        }
        segments.append(segment)
    
    # Build per-challenge tree
    tree = pipeline.build_per_challenge_tree(
        challenge_id="pipeline_test",
        segments=segments,
        use_incremental=False
    )
    
    print(f"  Challenge ID: {tree.challenge_id}")
    print(f"  Root: {tree.root.hex()[:32]}...")
    print(f"  Number of segments: {len(tree.leaves)}")
    
    # Test with incremental construction
    tree_inc = pipeline.build_per_challenge_tree(
        challenge_id="pipeline_test_inc",
        segments=segments,
        use_incremental=True
    )
    
    print(f"\n  Incremental tree root: {tree_inc.root.hex()[:32]}...")
    
    # Verify signatures were stored
    if hasattr(pipeline, 'segment_signatures'):
        print(f"  Stored signatures: {len(pipeline.segment_signatures)}")
    
    print("\n✓ REVPipeline integration tests passed")


def test_proof_verification():
    """Test Merkle proof generation and verification."""
    print("\nTesting Merkle Proof Generation and Verification")
    print("=" * 60)
    
    # Create tree with known leaves
    leaves = []
    for i in range(8):
        leaf = ChallengeLeaf(
            seg_id=f"proof_seg_{i}",
            sigma=f"proof_sig_{i}".encode(),
            policy={"index": i}
        )
        leaves.append(leaf)
    
    tree = PerChallengeTree(
        challenge_id="proof_test",
        leaves=leaves
    )
    root = tree.build_tree()
    
    # Generate and verify proofs for all segments
    for i, leaf in enumerate(leaves):
        proof = tree.generate_proof(leaf.seg_id)
        assert proof is not None, f"Should generate proof for seg {i}"
        
        # Manually verify proof
        leaf_hash = leaf.compute_leaf()
        is_valid = verify_merkle_proof_bytes(leaf_hash, proof, root)
        assert is_valid, f"Proof for seg {i} should be valid"
        
        print(f"  Segment {leaf.seg_id}: proof valid ✓ (path length: {len(proof)})")
    
    # Test proof for non-existent segment
    invalid_proof = tree.generate_proof("non_existent")
    assert invalid_proof is None, "Should return None for non-existent segment"
    print("\n  ✓ Correctly handled non-existent segment")
    
    print("\n✓ Proof verification tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Enhanced Merkle Trees for Per-Challenge Verification")
    print("Sections 4.3 and 5.5 of REV Paper")
    print("=" * 70)
    
    test_per_challenge_merkle_tree()
    test_build_signature()
    test_incremental_merkle_tree()
    test_audit_transcript()
    test_hierarchical_verification_chain()
    test_rev_pipeline_integration()
    test_proof_verification()
    
    print("\n" + "=" * 70)
    print("All enhanced Merkle tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()