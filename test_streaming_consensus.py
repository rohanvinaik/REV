#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced Streaming Consensus Verifier.

Tests Byzantine fault tolerance, adaptive mode selection, checkpointing,
and memory-bounded streaming verification.
"""

import numpy as np
import torch
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
import json

from src.verifier.streaming_consensus import (
    StreamingConsensusVerifier,
    VerificationMode,
    ConsensusMode,
    ValidatorReputation,
    ConsensusCheckpoint,
    StreamingVerificationState
)
from src.verifier.decision import Verdict
from src.rev_pipeline import Segment
from src.hdc.encoder import HypervectorConfig


@dataclass
class MockSegment(Segment):
    """Mock segment for testing."""
    def __init__(self, segment_id: int, similarity: float = 0.8):
        super().__init__(
            segment_id=segment_id,
            tokens=list(range(50)),
            start_idx=0,
            end_idx=50,
            signatures={}
        )
        self.similarity = similarity


class TestStreamingConsensusVerifier:
    """Test the enhanced StreamingConsensusVerifier."""
    
    def test_initialization(self):
        """Test verifier initialization with various configurations."""
        # Default initialization
        verifier = StreamingConsensusVerifier()
        assert verifier.verification_mode == VerificationMode.ADAPTIVE
        assert verifier.consensus_mode == ConsensusMode.BALANCED
        assert verifier.early_stop_threshold == 0.95
        assert verifier.buffer_size == 8
        
        # Custom configuration
        verifier = StreamingConsensusVerifier(
            verification_mode=VerificationMode.REV_ONLY,
            consensus_mode=ConsensusMode.ROBUST,
            early_stop_threshold=0.9,
            max_segments=500,
            buffer_size=16,
            num_validators=7,
            byzantine_threshold=0.4
        )
        assert verifier.verification_mode == VerificationMode.REV_ONLY
        assert verifier.consensus_mode == ConsensusMode.ROBUST
        assert verifier.buffer_size == 16
        assert len(verifier.verification_state.validator_reputations) == 7
        
        print("✓ Initialization tests passed")
    
    def test_adaptive_mode_selection(self):
        """Test adaptive verification mode selection."""
        verifier = StreamingConsensusVerifier(
            verification_mode=VerificationMode.ADAPTIVE
        )
        
        # Initial phase - should use UNIFIED
        mode = verifier._select_verification_mode()
        assert mode == VerificationMode.UNIFIED
        
        # Simulate processing segments
        verifier.verification_state.segments_processed = 20
        
        # High confidence - should use REV_ONLY
        verifier.verification_state.current_confidence = 0.9
        mode = verifier._select_verification_mode()
        assert mode == VerificationMode.REV_ONLY
        
        # Low confidence - should use HBT_ONLY
        verifier.verification_state.current_confidence = 0.3
        mode = verifier._select_verification_mode()
        assert mode == VerificationMode.HBT_ONLY
        
        # Stable similarity - should use REV_ONLY
        verifier.verification_state.similarity_history = [0.8] * 20
        verifier.verification_state.current_confidence = 0.6
        mode = verifier._select_verification_mode()
        assert mode == VerificationMode.REV_ONLY
        
        # High variance - should use HBT_ONLY
        verifier.verification_state.similarity_history = [0.2, 0.9] * 10
        mode = verifier._select_verification_mode()
        assert mode == VerificationMode.HBT_ONLY
        
        print("✓ Adaptive mode selection tests passed")
    
    def test_adaptive_consensus_frequency(self):
        """Test adaptive consensus frequency adjustment."""
        verifier = StreamingConsensusVerifier(
            consensus_mode=ConsensusMode.ADAPTIVE
        )
        
        # Low confidence - frequent consensus
        verifier.verification_state.current_confidence = 0.3
        freq = verifier._adapt_consensus_frequency()
        assert freq == 2
        
        # Medium confidence - balanced
        verifier.verification_state.current_confidence = 0.75
        freq = verifier._adapt_consensus_frequency()
        assert freq == 4
        
        # High confidence - less frequent
        verifier.verification_state.current_confidence = 0.95
        freq = verifier._adapt_consensus_frequency()
        assert freq == 8
        
        # Non-adaptive mode - should return fixed frequency
        verifier = StreamingConsensusVerifier(
            consensus_mode=ConsensusMode.FAST
        )
        freq = verifier._adapt_consensus_frequency()
        assert freq == 8
        
        print("✓ Adaptive consensus frequency tests passed")
    
    def test_validator_reputation(self):
        """Test validator reputation tracking and weighted voting."""
        reputation = ValidatorReputation("validator_0")
        
        # Initial reputation
        assert reputation.reputation_score == 1.0
        assert reputation.weight == 1.0
        
        # Successful validation aligned with consensus
        reputation.update(success=True, consensus_aligned=True)
        assert reputation.reputation_score > 1.0
        assert reputation.successful_validations == 1
        
        # Successful but not aligned
        old_score = reputation.reputation_score
        reputation.update(success=True, consensus_aligned=False)
        assert reputation.reputation_score < old_score
        assert reputation.successful_validations == 2
        
        # Failed validation
        reputation.update(success=False, consensus_aligned=False)
        assert reputation.failed_validations == 1
        assert reputation.reputation_score < old_score
        
        # Reputation bounds
        for _ in range(50):
            reputation.update(success=False, consensus_aligned=False)
        assert reputation.reputation_score >= 0.1  # Minimum bound
        
        reputation2 = ValidatorReputation("validator_1")
        for _ in range(50):
            reputation2.update(success=True, consensus_aligned=True)
        assert reputation2.reputation_score <= 2.0  # Maximum bound
        
        print("✓ Validator reputation tests passed")
    
    def test_byzantine_consensus(self):
        """Test Byzantine fault tolerance in consensus."""
        verifier = StreamingConsensusVerifier(
            num_validators=5,
            byzantine_threshold=0.33
        )
        
        # Create validator votes
        validator_votes = {
            "validator_0": {"behavioral": 0.8, "architectural": 0.85},
            "validator_1": {"behavioral": 0.75, "architectural": 0.8},
            "validator_2": {"behavioral": 0.9, "architectural": 0.9},
            "validator_3": {"behavioral": 0.2, "architectural": 0.3},  # Byzantine
            "validator_4": {"behavioral": 0.85, "architectural": 0.88}
        }
        
        validator_weights = {
            f"validator_{i}": 1.0 for i in range(5)
        }
        
        # Check consensus with 1 Byzantine validator (20%)
        consensus = verifier._check_byzantine_consensus(validator_votes, validator_weights)
        assert consensus is True  # Should reach consensus
        
        # Add another Byzantine validator (40% > threshold)
        validator_votes["validator_1"] = {"behavioral": 0.1, "architectural": 0.2}
        consensus = verifier._check_byzantine_consensus(validator_votes, validator_weights)
        assert consensus is False  # Should not reach consensus
        
        # Test with weighted voting
        validator_weights["validator_2"] = 2.0  # High reputation validator
        consensus = verifier._check_byzantine_consensus(validator_votes, validator_weights)
        assert consensus is True  # Should reach consensus with weighted voting
        
        print("✓ Byzantine consensus tests passed")
    
    def test_checkpoint_creation_and_resume(self):
        """Test checkpoint creation and resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            verifier = StreamingConsensusVerifier(
                checkpoint_dir=checkpoint_dir
            )
            
            # Create state
            state = verifier.verification_state
            state.segments_processed = 100
            state.current_confidence = 0.85
            state.similarity_history = [0.8] * 50
            state.early_stopped = False
            
            # Add a checkpoint
            import uuid
            checkpoint = ConsensusCheckpoint(
                checkpoint_id=str(uuid.uuid4())[:8],
                segment_ids=[1, 2, 3, 4],
                consensus_result=None,
                timestamp=time.time(),
                merkle_root=b"test_merkle",
                confidence=0.85,
                segment_buffer_snapshot=[],
                verification_mode=VerificationMode.UNIFIED,
                validator_reputations={},
                metadata={"test": "data"}
            )
            state.consensus_checkpoints.append(checkpoint)
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / "test_checkpoint.json"
            state.save_checkpoint(checkpoint_path)
            
            # Verify files created
            assert checkpoint_path.exists()
            assert checkpoint_path.with_suffix('.state').exists()
            
            # Load checkpoint
            loaded_state = StreamingVerificationState.load_checkpoint(checkpoint_path)
            
            # Verify loaded state
            assert loaded_state.segments_processed == 100
            assert loaded_state.current_confidence == 0.85
            assert len(loaded_state.similarity_history) == 50
            assert len(loaded_state.consensus_checkpoints) == 1
            assert loaded_state.consensus_checkpoints[0].checkpoint_id == checkpoint.checkpoint_id
            
            print("✓ Checkpoint creation and resume tests passed")
    
    def test_streaming_verification(self):
        """Test streaming verification with mock segments."""
        verifier = StreamingConsensusVerifier(
            verification_mode=VerificationMode.UNIFIED,
            consensus_mode=ConsensusMode.FAST,
            early_stop_threshold=0.9,
            max_segments=50
        )
        
        # Create mock segment generators
        def create_segment_generator(n_segments: int, similarity: float) -> Generator:
            for i in range(n_segments):
                yield MockSegment(i, similarity)
        
        gen_a = create_segment_generator(30, 0.9)
        gen_b = create_segment_generator(30, 0.9)
        
        # Run streaming verification
        results = []
        for update in verifier.stream_verify(gen_a, gen_b):
            results.append(update)
            if update['type'] == 'checkpoint':
                print(f"  Checkpoint at segment {update['segment_count']}, "
                      f"confidence: {update['current_confidence']:.3f}")
        
        # Get final result
        final_result = results[-1] if results else None
        
        # Verify we got checkpoints
        checkpoints = [r for r in results if r['type'] == 'checkpoint']
        assert len(checkpoints) > 0
        
        # Verify status updates
        status_updates = [r for r in results if r['type'] == 'status']
        assert len(status_updates) > 0
        
        print("✓ Streaming verification tests passed")
    
    def test_early_stopping(self):
        """Test early stopping conditions."""
        verifier = StreamingConsensusVerifier(
            verification_mode=VerificationMode.UNIFIED,
            early_stop_threshold=0.85
        )
        
        # Setup state for early stopping - high confidence
        verifier.verification_state.current_confidence = 0.9
        verifier.verification_state.sequential_state.n = 25
        verifier.verification_state.sequential_state.mean = 0.2  # High similarity
        verifier.verification_state.sequential_state.variance = 0.005  # Low variance
        
        # Create strong consensus checkpoints
        for i in range(3):
            checkpoint = ConsensusCheckpoint(
                checkpoint_id=f"cp_{i}",
                segment_ids=[],
                consensus_result=type('obj', (object,), {
                    'consensus_reached': True,
                    'behavioral_agreement': 0.95,
                    'architectural_agreement': 0.93,
                    'confidence_score': 0.92
                })(),
                timestamp=time.time(),
                merkle_root=b"test",
                confidence=0.92,
                segment_buffer_snapshot=[],
                verification_mode=VerificationMode.UNIFIED,
                validator_reputations={},
                metadata={}
            )
            verifier.consensus_points.append(checkpoint)
        
        # Should trigger early stopping
        should_stop = verifier._should_early_stop(checkpoint)
        assert should_stop is True
        
        # Test early stopping with low similarity (different models)
        verifier2 = StreamingConsensusVerifier()
        verifier2.verification_state.sequential_state.n = 25
        verifier2.verification_state.sequential_state.mean = 0.8  # Low similarity
        verifier2.verification_state.sequential_state.variance = 0.008
        
        checkpoint2 = ConsensusCheckpoint(
            checkpoint_id="test",
            segment_ids=[],
            consensus_result=None,
            timestamp=time.time(),
            merkle_root=b"test",
            confidence=0.5,
            segment_buffer_snapshot=[],
            verification_mode=VerificationMode.UNIFIED,
            validator_reputations={},
            metadata={}
        )
        
        should_stop2 = verifier2._should_early_stop(checkpoint2)
        assert should_stop2 is True
        
        print("✓ Early stopping tests passed")
    
    def test_verdict_determination(self):
        """Test verdict determination logic."""
        verifier = StreamingConsensusVerifier()
        
        # Not enough samples - UNDECIDED
        verifier.verification_state.sequential_state.n = 5
        verdict = verifier._determine_current_verdict()
        assert verdict == Verdict.UNDECIDED
        
        # High confidence, high similarity - SAME
        verifier.verification_state.sequential_state.n = 20
        verifier.verification_state.sequential_state.mean = 0.25
        verifier.verification_state.current_confidence = 0.92
        verdict = verifier._determine_current_verdict()
        assert verdict == Verdict.SAME
        
        # High confidence, low similarity - DIFFERENT
        verifier.verification_state.sequential_state.mean = 0.75
        verdict = verifier._determine_current_verdict()
        assert verdict == Verdict.DIFFERENT
        
        # Medium confidence boundaries
        verifier.verification_state.current_confidence = 0.75
        verifier.verification_state.sequential_state.mean = 0.5
        verdict = verifier._determine_current_verdict()
        assert verdict == Verdict.UNDECIDED
        
        print("✓ Verdict determination tests passed")
    
    def test_memory_bounded_streaming(self):
        """Test memory-bounded streaming with buffer management."""
        buffer_size = 4
        verifier = StreamingConsensusVerifier(
            buffer_size=buffer_size
        )
        
        # Add segments to buffers
        for i in range(10):
            seg_a = MockSegment(i, 0.8)
            seg_b = MockSegment(i, 0.8)
            verifier.verification_state.segment_buffer_a.append(seg_a)
            verifier.verification_state.segment_buffer_b.append(seg_b)
        
        # Buffers should be bounded
        assert len(verifier.verification_state.segment_buffer_a) <= buffer_size * 2
        assert len(verifier.verification_state.segment_buffer_b) <= buffer_size * 2
        
        # Latest segments should be in buffer
        latest_ids = [seg.segment_id for seg in verifier.verification_state.segment_buffer_a]
        assert 9 in latest_ids
        assert 0 not in latest_ids  # Oldest should be evicted
        
        print("✓ Memory-bounded streaming tests passed")
    
    def test_verification_summary(self):
        """Test verification summary generation."""
        verifier = StreamingConsensusVerifier(
            verification_mode=VerificationMode.ADAPTIVE,
            num_validators=3
        )
        
        # Empty summary
        summary = verifier.get_verification_summary()
        assert summary['segments_processed'] == 0
        assert summary['checkpoints'] == 0
        
        # Add some data
        verifier.verification_state.segments_processed = 100
        verifier.verification_state.similarity_history = [0.7, 0.8, 0.75]
        verifier.verification_state.mode_history = [
            VerificationMode.UNIFIED,
            VerificationMode.REV_ONLY,
            VerificationMode.REV_ONLY
        ]
        
        # Add checkpoints
        for i in range(2):
            checkpoint = ConsensusCheckpoint(
                checkpoint_id=f"cp_{i}",
                segment_ids=[],
                consensus_result=type('obj', (object,), {
                    'consensus_reached': True,
                    'behavioral_agreement': 0.85,
                    'architectural_agreement': 0.88,
                    'confidence_score': 0.86
                })(),
                timestamp=time.time(),
                merkle_root=b"test",
                confidence=0.86,
                segment_buffer_snapshot=[],
                verification_mode=VerificationMode.UNIFIED,
                validator_reputations={},
                metadata={}
            )
            verifier.consensus_points.append(checkpoint)
        
        # Update validator reputations
        verifier.verification_state.validator_reputations["validator_0"].reputation_score = 1.5
        verifier.verification_state.validator_reputations["validator_1"].reputation_score = 0.8
        
        summary = verifier.get_verification_summary()
        
        assert summary['segments_processed'] == 100
        assert summary['checkpoints'] == 2
        assert summary['mean_similarity'] == np.mean([0.7, 0.8, 0.75])
        assert summary['mode_distribution']['rev_only'] == 2
        assert summary['best_validator']['reputation'] == 1.5
        assert summary['worst_validator']['reputation'] == 0.8
        assert summary['byzantine_consensus_rate'] == 1.0
        
        print("✓ Verification summary tests passed")


class TestIntegration:
    """Integration tests for the streaming consensus system."""
    
    def test_end_to_end_verification(self):
        """Test complete verification pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            # Create verifier with all features
            verifier = StreamingConsensusVerifier(
                verification_mode=VerificationMode.ADAPTIVE,
                consensus_mode=ConsensusMode.ADAPTIVE,
                early_stop_threshold=0.9,
                max_segments=100,
                buffer_size=8,
                checkpoint_dir=checkpoint_dir,
                num_validators=5,
                byzantine_threshold=0.3
            )
            
            # Create segment generators simulating similar models
            def create_similar_segments(n: int) -> Generator:
                for i in range(n):
                    # Mostly similar with occasional differences
                    similarity = 0.85 if i % 10 != 0 else 0.4
                    yield MockSegment(i, similarity)
            
            gen_a = create_similar_segments(50)
            gen_b = create_similar_segments(50)
            
            # Run verification
            checkpoints = []
            final_result = None
            
            for update in verifier.stream_verify(gen_a, gen_b):
                if update['type'] == 'checkpoint':
                    checkpoints.append(update)
                    print(f"  Checkpoint {update['checkpoint_id']}: "
                          f"segment {update['segment_count']}, "
                          f"confidence {update['current_confidence']:.3f}, "
                          f"mode {update['verification_mode']}")
                elif update['type'] == 'status':
                    print(f"  Status: {update['segments_processed']} segments, "
                          f"mode {update['current_mode']}")
                final_result = update
            
            # Verify results
            assert len(checkpoints) > 0
            assert verifier.verification_state.segments_processed > 0
            
            # Check summary
            summary = verifier.get_verification_summary()
            print(f"\nFinal Summary:")
            print(f"  Segments: {summary['segments_processed']}")
            print(f"  Checkpoints: {summary['checkpoints']}")
            print(f"  Verdict: {summary['final_verdict']}")
            print(f"  Confidence: {summary['final_confidence']:.3f}")
            print(f"  Byzantine consensus rate: {summary['byzantine_consensus_rate']:.3f}")
            
            # Verify checkpoints were saved
            saved_checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
            assert len(saved_checkpoints) > 0
            
            print("✓ End-to-end verification test passed")
    
    def test_resume_from_checkpoint(self):
        """Test resuming verification from saved checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            # First run - partial verification
            verifier1 = StreamingConsensusVerifier(
                checkpoint_dir=checkpoint_dir,
                max_segments=20
            )
            
            gen_a1 = (MockSegment(i, 0.8) for i in range(20))
            gen_b1 = (MockSegment(i, 0.8) for i in range(20))
            
            checkpoint_path = None
            for update in verifier1.stream_verify(gen_a1, gen_b1):
                if update['type'] == 'checkpoint':
                    checkpoint_path = Path(update['checkpoint_saved'])
            
            assert checkpoint_path and checkpoint_path.exists()
            segments_processed_1 = verifier1.verification_state.segments_processed
            
            # Second run - resume from checkpoint
            verifier2 = StreamingConsensusVerifier(
                checkpoint_dir=checkpoint_dir,
                max_segments=40
            )
            
            gen_a2 = (MockSegment(i, 0.8) for i in range(20, 40))
            gen_b2 = (MockSegment(i, 0.8) for i in range(20, 40))
            
            resumed = False
            for update in verifier2.stream_verify(
                gen_a2, gen_b2, 
                resume_from=checkpoint_path
            ):
                if update['type'] == 'resumed':
                    resumed = True
                    assert update['segments_processed'] == segments_processed_1
            
            assert resumed
            assert verifier2.verification_state.segments_processed > segments_processed_1
            
            print("✓ Resume from checkpoint test passed")
    
    def test_byzantine_fault_scenario(self):
        """Test system behavior with Byzantine validators."""
        verifier = StreamingConsensusVerifier(
            num_validators=7,
            byzantine_threshold=0.3  # Can tolerate up to 30% Byzantine
        )
        
        # Simulate Byzantine validators by manipulating reputations
        # Make 2 validators Byzantine (28% < 30% threshold)
        verifier.verification_state.validator_reputations["validator_0"].reputation_score = 0.1
        verifier.verification_state.validator_reputations["validator_1"].reputation_score = 0.1
        
        # Good validators maintain high reputation
        for i in range(2, 7):
            verifier.verification_state.validator_reputations[f"validator_{i}"].reputation_score = 1.5
        
        # Create mock segments
        gen_a = (MockSegment(i, 0.85) for i in range(20))
        gen_b = (MockSegment(i, 0.85) for i in range(20))
        
        # Run verification
        consensus_reached_count = 0
        for update in verifier.stream_verify(gen_a, gen_b):
            if update['type'] == 'checkpoint':
                if update.get('consensus_reached', False):
                    consensus_reached_count += 1
        
        # Should still reach consensus despite Byzantine validators
        assert consensus_reached_count > 0
        
        # Check validator reputations evolved correctly
        summary = verifier.get_verification_summary()
        assert summary['best_validator']['reputation'] > summary['worst_validator']['reputation']
        
        print("✓ Byzantine fault scenario test passed")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("Streaming Consensus Verifier Tests")
    print("="*70)
    
    # Basic tests
    print("\nTesting StreamingConsensusVerifier...")
    verifier_tests = TestStreamingConsensusVerifier()
    verifier_tests.test_initialization()
    verifier_tests.test_adaptive_mode_selection()
    verifier_tests.test_adaptive_consensus_frequency()
    verifier_tests.test_validator_reputation()
    verifier_tests.test_byzantine_consensus()
    verifier_tests.test_checkpoint_creation_and_resume()
    verifier_tests.test_streaming_verification()
    verifier_tests.test_early_stopping()
    verifier_tests.test_verdict_determination()
    verifier_tests.test_memory_bounded_streaming()
    verifier_tests.test_verification_summary()
    
    # Integration tests
    print("\nTesting Integration...")
    integration_tests = TestIntegration()
    integration_tests.test_end_to_end_verification()
    integration_tests.test_resume_from_checkpoint()
    integration_tests.test_byzantine_fault_scenario()
    
    print("\n" + "="*70)
    print("All Streaming Consensus tests passed successfully! ✓")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()