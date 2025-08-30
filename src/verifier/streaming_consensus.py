"""
Streaming Consensus Verifier - Combines REV streaming with HBT consensus.

This module implements memory-bounded streaming verification with periodic
Byzantine consensus checkpoints for robust LLM comparison with adaptive mode selection.
"""

from typing import Dict, List, Tuple, Optional, Any, Generator, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import time
import pickle
import json
import os
from enum import Enum
from pathlib import Path
import logging

from ..rev_pipeline import REVPipeline, Segment
from ..consensus.byzantine import ConsensusNetwork, ConsensusResult, ByzantineValidator
from ..core.sequential import SequentialState, DualSequentialTest
from ..verifier.decision import Verdict, StepRecord, RunResult
from ..hdc.encoder import HypervectorEncoder, HypervectorConfig
from ..crypto.merkle import build_merkle_tree, leaf_bytes

logger = logging.getLogger(__name__)


class VerificationMode(Enum):
    """Verification mode for adaptive selection."""
    REV_ONLY = "rev_only"  # Fast verification using REV
    HBT_ONLY = "hbt_only"  # Maximum robustness using HBT
    UNIFIED = "unified"  # Balanced approach
    ADAPTIVE = "adaptive"  # Automatic mode switching


class ConsensusMode(Enum):
    """Consensus verification modes."""
    FAST = "fast"  # Consensus every 8 segments
    BALANCED = "balanced"  # Consensus every 4 segments (default)
    ROBUST = "robust"  # Consensus every 2 segments
    ADAPTIVE = "adaptive"  # Adaptive frequency based on confidence


@dataclass
class ValidatorReputation:
    """Track validator reputation for weighted voting."""
    validator_id: str
    successful_validations: int = 0
    failed_validations: int = 0
    total_validations: int = 0
    reputation_score: float = 1.0
    
    def update(self, success: bool, consensus_aligned: bool):
        """Update reputation based on validation outcome."""
        self.total_validations += 1
        if success:
            self.successful_validations += 1
            if consensus_aligned:
                # Bonus for aligning with consensus
                self.reputation_score = min(2.0, self.reputation_score * 1.05)
            else:
                # Small penalty for diverging from consensus when successful
                self.reputation_score *= 0.98
        else:
            self.failed_validations += 1
            self.reputation_score *= 0.95
        
        # Ensure reputation stays in bounds
        self.reputation_score = max(0.1, min(2.0, self.reputation_score))
    
    @property
    def weight(self) -> float:
        """Get voting weight based on reputation."""
        return self.reputation_score


@dataclass
class ConsensusCheckpoint:
    """Enhanced checkpoint for consensus verification with resume support."""
    
    checkpoint_id: str
    segment_ids: List[int]
    consensus_result: ConsensusResult
    timestamp: float
    merkle_root: bytes
    confidence: float
    segment_buffer_snapshot: List[Segment]
    verification_mode: VerificationMode
    validator_reputations: Dict[str, ValidatorReputation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_strong_consensus(self, threshold: float = 0.9) -> bool:
        """Check if this checkpoint has strong consensus."""
        return (
            self.consensus_result.consensus_reached and
            self.confidence >= threshold
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'segment_ids': self.segment_ids,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root.hex() if isinstance(self.merkle_root, bytes) else self.merkle_root,
            'confidence': self.confidence,
            'verification_mode': self.verification_mode.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsensusCheckpoint':
        """Create checkpoint from dictionary."""
        # Simplified reconstruction for resuming
        return cls(
            checkpoint_id=data['checkpoint_id'],
            segment_ids=data['segment_ids'],
            consensus_result=None,  # Will be reconstructed
            timestamp=data['timestamp'],
            merkle_root=bytes.fromhex(data['merkle_root']),
            confidence=data['confidence'],
            segment_buffer_snapshot=[],  # Will be reconstructed
            verification_mode=VerificationMode(data['verification_mode']),
            validator_reputations={},  # Will be reconstructed
            metadata=data.get('metadata', {})
        )


@dataclass 
class StreamingVerificationState:
    """Enhanced state for streaming consensus verification with resume support."""
    
    segments_processed: int = 0
    consensus_checkpoints: List[ConsensusCheckpoint] = field(default_factory=list)
    current_confidence: float = 0.0
    sequential_state: SequentialState = field(default_factory=SequentialState)
    early_stopped: bool = False
    final_verdict: Optional[Verdict] = None
    segment_buffer_a: deque = field(default_factory=lambda: deque(maxlen=8))
    segment_buffer_b: deque = field(default_factory=lambda: deque(maxlen=8))
    similarity_history: List[float] = field(default_factory=list)
    mode_history: List[VerificationMode] = field(default_factory=list)
    validator_reputations: Dict[str, ValidatorReputation] = field(default_factory=dict)
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update running confidence with smoothing."""
        alpha = 0.3  # Smoothing factor
        self.current_confidence = (
            alpha * new_confidence + 
            (1 - alpha) * self.current_confidence
        )
    
    def save_checkpoint(self, path: Path) -> None:
        """Save state for resuming."""
        checkpoint_data = {
            'segments_processed': self.segments_processed,
            'checkpoints': [cp.to_dict() for cp in self.consensus_checkpoints],
            'current_confidence': self.current_confidence,
            'early_stopped': self.early_stopped,
            'final_verdict': self.final_verdict.value if self.final_verdict else None,
            'similarity_history': self.similarity_history,
            'mode_history': [m.value for m in self.mode_history]
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Save sequential state separately (binary)
        state_path = path.with_suffix('.state')
        with open(state_path, 'wb') as f:
            pickle.dump(self.sequential_state, f)
    
    @classmethod
    def load_checkpoint(cls, path: Path) -> 'StreamingVerificationState':
        """Load state from checkpoint."""
        with open(path, 'r') as f:
            checkpoint_data = json.load(f)
        
        state = cls()
        state.segments_processed = checkpoint_data['segments_processed']
        state.consensus_checkpoints = [
            ConsensusCheckpoint.from_dict(cp) 
            for cp in checkpoint_data['checkpoints']
        ]
        state.current_confidence = checkpoint_data['current_confidence']
        state.early_stopped = checkpoint_data['early_stopped']
        state.final_verdict = Verdict(checkpoint_data['final_verdict']) if checkpoint_data['final_verdict'] else None
        state.similarity_history = checkpoint_data['similarity_history']
        state.mode_history = [VerificationMode(m) for m in checkpoint_data['mode_history']]
        
        # Load sequential state
        state_path = path.with_suffix('.state')
        if state_path.exists():
            with open(state_path, 'rb') as f:
                state.sequential_state = pickle.load(f)
        
        return state


class StreamingConsensusVerifier:
    """
    Enhanced streaming consensus verifier combining REV and HBT approaches
    with Byzantine fault tolerance and adaptive mode selection.
    
    Features:
    - Memory-bounded streaming with configurable buffer size
    - Checkpoint creation at regular/adaptive intervals
    - Early stopping based on confidence thresholds
    - Support for resuming from checkpoints
    - Byzantine consensus with weighted voting
    - Adaptive mode selection for optimal performance
    """
    
    def __init__(
        self,
        rev_pipeline: Optional[REVPipeline] = None,
        consensus_network: Optional[ConsensusNetwork] = None,
        verification_mode: VerificationMode = VerificationMode.ADAPTIVE,
        consensus_mode: ConsensusMode = ConsensusMode.BALANCED,
        early_stop_threshold: float = 0.95,
        max_segments: int = 1000,
        buffer_size: int = 8,
        encoder_config: Optional[HypervectorConfig] = None,
        checkpoint_dir: Optional[Path] = None,
        num_validators: int = 5,
        byzantine_threshold: float = 0.33
    ):
        """
        Initialize enhanced streaming consensus verifier.
        
        Args:
            rev_pipeline: REV pipeline for segment processing
            consensus_network: Byzantine consensus network
            verification_mode: Initial verification mode
            consensus_mode: Frequency of consensus checkpoints
            early_stop_threshold: Confidence threshold for early stopping
            max_segments: Maximum segments to process
            buffer_size: Size of segment buffers
            encoder_config: Configuration for HDC encoding
            checkpoint_dir: Directory for saving checkpoints
            num_validators: Number of validators in consensus
            byzantine_threshold: Maximum fraction of Byzantine validators
        """
        # Initialize REV pipeline
        self.rev_pipeline = rev_pipeline or REVPipeline(
            segment_size=512,
            buffer_size=buffer_size
        )
        
        # Initialize consensus network with Byzantine fault tolerance
        self.consensus_network = consensus_network or ConsensusNetwork(
            num_validators=num_validators,
            behavioral_threshold=0.85,
            architectural_threshold=0.90,
            batch_size=buffer_size,
            byzantine_threshold=byzantine_threshold
        )
        
        # Initialize encoder for adaptive encoding
        encoder_config = encoder_config or HypervectorConfig(
            dimension=10000,
            encoding_mode="hybrid",
            multi_scale=True,
            privacy_mode=True
        )
        self.encoder = HypervectorEncoder(encoder_config)
        
        # Configuration
        self.verification_mode = verification_mode
        self.consensus_mode = consensus_mode
        self.early_stop_threshold = early_stop_threshold
        self.max_segments = max_segments
        self.buffer_size = buffer_size
        self.byzantine_threshold = byzantine_threshold
        
        # Set consensus frequency based on mode
        self.consensus_frequency = self._get_consensus_frequency()
        
        # State tracking
        self.verification_state = StreamingVerificationState()
        self.consensus_points: List[ConsensusCheckpoint] = []
        
        # Initialize validator reputations
        for i in range(num_validators):
            validator_id = f"validator_{i}"
            self.verification_state.validator_reputations[validator_id] = ValidatorReputation(validator_id)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Mode selection callbacks
        self.mode_callbacks = {
            VerificationMode.REV_ONLY: self._verify_rev_only,
            VerificationMode.HBT_ONLY: self._verify_hbt_only,
            VerificationMode.UNIFIED: self._verify_unified,
            VerificationMode.ADAPTIVE: self._verify_adaptive
        }
    
    def _get_consensus_frequency(self) -> int:
        """Get consensus frequency based on mode."""
        if self.consensus_mode == ConsensusMode.ADAPTIVE:
            # Start with balanced, will adapt based on confidence
            return 4
        
        return {
            ConsensusMode.FAST: 8,
            ConsensusMode.BALANCED: 4,
            ConsensusMode.ROBUST: 2
        }[self.consensus_mode]
    
    def _adapt_consensus_frequency(self) -> int:
        """Adapt consensus frequency based on current confidence."""
        if self.consensus_mode != ConsensusMode.ADAPTIVE:
            return self.consensus_frequency
        
        confidence = self.verification_state.current_confidence
        
        if confidence > 0.9:
            # High confidence - less frequent consensus
            return 8
        elif confidence > 0.7:
            # Medium confidence - balanced
            return 4
        else:
            # Low confidence - frequent consensus
            return 2
    
    def _select_verification_mode(self) -> VerificationMode:
        """Select verification mode based on current state."""
        if self.verification_mode != VerificationMode.ADAPTIVE:
            return self.verification_mode
        
        # Adaptive mode selection based on confidence and history
        confidence = self.verification_state.current_confidence
        segments = self.verification_state.segments_processed
        
        if segments < 10:
            # Initial phase - use unified approach
            return VerificationMode.UNIFIED
        
        # Check recent similarity variance
        if len(self.verification_state.similarity_history) >= 10:
            recent_variance = np.var(self.verification_state.similarity_history[-10:])
            
            if recent_variance < 0.01:
                # Very stable - use fast REV only
                return VerificationMode.REV_ONLY
            elif recent_variance > 0.1:
                # High variance - need robust HBT
                return VerificationMode.HBT_ONLY
        
        # Default based on confidence
        if confidence > 0.85:
            return VerificationMode.REV_ONLY
        elif confidence < 0.5:
            return VerificationMode.HBT_ONLY
        else:
            return VerificationMode.UNIFIED
    
    def stream_verify(
        self,
        model_stream_a: Generator[Segment, None, None],
        model_stream_b: Generator[Segment, None, None],
        challenges: Optional[List[str]] = None,
        tokenizer_a=None,
        tokenizer_b=None,
        resume_from: Optional[Path] = None
    ) -> Generator[Dict[str, Any], None, RunResult]:
        """
        Perform streaming verification with consensus checkpoints.
        
        Args:
            model_stream_a: Generator of segments from model A
            model_stream_b: Generator of segments from model B
            challenges: Optional list of challenge prompts
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            resume_from: Path to checkpoint to resume from
            
        Yields:
            Verification updates at each checkpoint
            
        Returns:
            Final RunResult with verdict
        """
        # Resume from checkpoint if provided
        if resume_from and resume_from.exists():
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.verification_state = StreamingVerificationState.load_checkpoint(resume_from)
            yield {
                'type': 'resumed',
                'segments_processed': self.verification_state.segments_processed,
                'checkpoints_loaded': len(self.verification_state.consensus_checkpoints)
            }
        
        start_time = time.time()
        step_records = []
        segment_count = self.verification_state.segments_processed
        
        try:
            while segment_count < self.max_segments:
                # Get next segments from both models
                try:
                    segment_a = next(model_stream_a)
                    segment_b = next(model_stream_b)
                except StopIteration:
                    # End of stream
                    break
                
                # Add to buffers
                self.verification_state.segment_buffer_a.append(segment_a)
                self.verification_state.segment_buffer_b.append(segment_b)
                segment_count += 1
                self.verification_state.segments_processed = segment_count
                
                # Select verification mode
                current_mode = self._select_verification_mode()
                self.verification_state.mode_history.append(current_mode)
                
                # Process segments based on mode
                similarity = self.mode_callbacks[current_mode](segment_a, segment_b)
                self.verification_state.similarity_history.append(similarity)
                
                # Update sequential state
                self.verification_state.sequential_state.update(1.0 - similarity)
                
                # Adaptive consensus frequency
                current_frequency = self._adapt_consensus_frequency()
                
                # Check for consensus checkpoint
                if segment_count % current_frequency == 0:
                    checkpoint = self._perform_consensus_checkpoint(current_mode)
                    
                    # Save checkpoint for resume capability
                    checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
                    self.verification_state.save_checkpoint(checkpoint_path)
                    
                    # Create step record
                    step_record = StepRecord(
                        index=len(step_records) + 1,
                        prompt=f"Segments {segment_count - current_frequency + 1}-{segment_count}",
                        ref_output=f"Model A: {len(self.verification_state.segment_buffer_a)} segments",
                        cand_output=f"Model B: {len(self.verification_state.segment_buffer_b)} segments",
                        score=1.0 - similarity,
                        mean=self.verification_state.sequential_state.mean,
                        var=self.verification_state.sequential_state.variance,
                        halfwidth=np.sqrt(self.verification_state.sequential_state.variance),
                        delta_n=0.05,
                        verdict_so_far=self._determine_current_verdict().value
                    )
                    step_records.append(step_record)
                    
                    # Yield checkpoint update
                    yield {
                        'type': 'checkpoint',
                        'checkpoint_id': checkpoint.checkpoint_id,
                        'segment_count': segment_count,
                        'checkpoint': checkpoint,
                        'current_confidence': self.verification_state.current_confidence,
                        'mean_similarity': np.mean(self.verification_state.similarity_history),
                        'verdict_so_far': self._determine_current_verdict(),
                        'consensus_reached': checkpoint.consensus_result.consensus_reached if checkpoint.consensus_result else False,
                        'verification_mode': current_mode.value,
                        'consensus_frequency': current_frequency,
                        'checkpoint_saved': str(checkpoint_path)
                    }
                    
                    # Check for early stopping
                    if self._should_early_stop(checkpoint):
                        self.verification_state.early_stopped = True
                        self.verification_state.final_verdict = self._determine_current_verdict()
                        logger.info(f"Early stopping at segment {segment_count} with verdict {self.verification_state.final_verdict}")
                        break
                
                # Periodic status update
                if segment_count % 10 == 0:
                    yield {
                        'type': 'status',
                        'segments_processed': segment_count,
                        'current_confidence': self.verification_state.current_confidence,
                        'buffer_sizes': (
                            len(self.verification_state.segment_buffer_a),
                            len(self.verification_state.segment_buffer_b)
                        ),
                        'current_mode': current_mode.value
                    }
        
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error during verification: {e}")
            yield {
                'type': 'error',
                'error': str(e),
                'segments_processed': segment_count
            }
        
        # Final consensus if buffer has segments
        if self.verification_state.segment_buffer_a:
            final_checkpoint = self._perform_consensus_checkpoint(
                self.verification_state.mode_history[-1] if self.verification_state.mode_history else VerificationMode.UNIFIED
            )
            yield {
                'type': 'final_checkpoint',
                'checkpoint': final_checkpoint,
                'segments_processed': segment_count
            }
        
        # Determine final verdict
        if self.verification_state.final_verdict is None:
            self.verification_state.final_verdict = self._determine_final_verdict()
        
        # Create final result
        result = RunResult(
            verdict=self.verification_state.final_verdict,
            steps=step_records,
            n_used=segment_count,
            params={
                'verification_mode': self.verification_mode.value,
                'consensus_mode': self.consensus_mode.value,
                'consensus_frequency': self.consensus_frequency,
                'early_stop_threshold': self.early_stop_threshold,
                'total_checkpoints': len(self.consensus_points),
                'early_stopped': self.verification_state.early_stopped,
                'final_confidence': self.verification_state.current_confidence,
                'mean_similarity': np.mean(self.verification_state.similarity_history) if self.verification_state.similarity_history else 0,
                'time_taken': time.time() - start_time,
                'mode_switches': len(set(self.verification_state.mode_history)),
                'validator_reputations': {
                    vid: rep.reputation_score 
                    for vid, rep in self.verification_state.validator_reputations.items()
                }
            }
        )
        
        return result
    
    def _verify_rev_only(self, segment_a: Segment, segment_b: Segment) -> float:
        """
        Fast verification using REV only.
        
        Args:
            segment_a: Segment from model A
            segment_b: Segment from model B
            
        Returns:
            Similarity score
        """
        # Generate lightweight signatures
        sig_a = self._generate_rev_signature(segment_a)
        sig_b = self._generate_rev_signature(segment_b)
        
        # Fast Hamming similarity
        similarity = self.encoder.popcount_hamming(sig_a, sig_b)
        normalized_sim = 1.0 - (similarity / len(sig_a))
        
        return normalized_sim
    
    def _verify_hbt_only(self, segment_a: Segment, segment_b: Segment) -> float:
        """
        Maximum robustness using HBT only.
        
        Args:
            segment_a: Segment from model A
            segment_b: Segment from model B
            
        Returns:
            Similarity score
        """
        # Generate comprehensive signatures at multiple levels
        sigs_a = self._generate_hbt_signatures(segment_a)
        sigs_b = self._generate_hbt_signatures(segment_b)
        
        # Multi-level similarity with Byzantine consensus
        similarities = []
        for level in sigs_a.keys():
            if level in sigs_b:
                sim = self._compute_robust_similarity(sigs_a[level], sigs_b[level])
                similarities.append(sim)
        
        # Weighted average with outlier removal
        if len(similarities) > 2:
            # Remove outliers
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            filtered = [s for s in similarities if abs(s - mean_sim) < 2 * std_sim]
            return np.mean(filtered) if filtered else mean_sim
        
        return np.mean(similarities) if similarities else 0.0
    
    def _verify_unified(self, segment_a: Segment, segment_b: Segment) -> float:
        """
        Balanced verification using both REV and HBT.
        
        Args:
            segment_a: Segment from model A
            segment_b: Segment from model B
            
        Returns:
            Similarity score
        """
        # REV similarity (fast)
        rev_sim = self._verify_rev_only(segment_a, segment_b)
        
        # HBT similarity (robust)
        hbt_sim = self._verify_hbt_only(segment_a, segment_b)
        
        # Weighted combination based on confidence
        confidence = self.verification_state.current_confidence
        rev_weight = confidence  # Higher confidence -> more weight to fast REV
        hbt_weight = 1.0 - confidence
        
        combined_sim = (rev_weight * rev_sim + hbt_weight * hbt_sim) / (rev_weight + hbt_weight)
        
        return combined_sim
    
    def _verify_adaptive(self, segment_a: Segment, segment_b: Segment) -> float:
        """
        Adaptive verification with automatic mode selection.
        
        Args:
            segment_a: Segment from model A
            segment_b: Segment from model B
            
        Returns:
            Similarity score
        """
        # Select optimal mode
        mode = self._select_verification_mode()
        
        # Use appropriate verification
        if mode == VerificationMode.REV_ONLY:
            return self._verify_rev_only(segment_a, segment_b)
        elif mode == VerificationMode.HBT_ONLY:
            return self._verify_hbt_only(segment_a, segment_b)
        else:
            return self._verify_unified(segment_a, segment_b)
    
    def _generate_rev_signature(self, segment: Segment) -> np.ndarray:
        """Generate lightweight REV signature."""
        # Use cached signature if available
        if hasattr(segment, 'rev_signature'):
            return segment.rev_signature
        
        # Generate using BLAKE2b hashing
        segment_str = f"segment_{segment.segment_id}"
        signature = self.encoder.encode_feature(segment_str, "token_window")
        
        # Cache for reuse
        segment.rev_signature = signature
        return signature
    
    def _generate_hbt_signatures(self, segment: Segment) -> Dict[str, np.ndarray]:
        """Generate comprehensive HBT signatures."""
        # Use cached signatures if available
        if hasattr(segment, 'hbt_signatures'):
            return segment.hbt_signatures
        
        signatures = {}
        segment_str = f"segment_{segment.segment_id}"
        
        # Generate at multiple zoom levels
        for level in ["corpus", "prompt", "span", "token_window"]:
            signatures[level] = self.encoder.encode_feature(segment_str, level)
        
        # Cache for reuse
        segment.hbt_signatures = signatures
        return signatures
    
    def _compute_robust_similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Compute robust similarity with outlier detection."""
        # Multiple similarity metrics
        hamming = 1.0 - (self.encoder.popcount_hamming(sig_a, sig_b) / len(sig_a))
        cosine = np.dot(sig_a, sig_b) / (np.linalg.norm(sig_a) * np.linalg.norm(sig_b))
        
        # Return average
        return (hamming + cosine) / 2.0
    
    def _perform_consensus_checkpoint(self, current_mode: VerificationMode) -> ConsensusCheckpoint:
        """
        Perform Byzantine consensus on current segment buffers with weighted voting.
        
        Args:
            current_mode: Current verification mode
            
        Returns:
            ConsensusCheckpoint with results
        """
        import uuid
        checkpoint_id = str(uuid.uuid4())[:8]
        
        # Prepare signatures for consensus
        signatures_a = {}
        signatures_b = {}
        
        for i, (seg_a, seg_b) in enumerate(zip(
            self.verification_state.segment_buffer_a,
            self.verification_state.segment_buffer_b
        )):
            # Generate signatures based on mode
            if current_mode in [VerificationMode.REV_ONLY, VerificationMode.ADAPTIVE]:
                sig_a = self._generate_rev_signature(seg_a)
                sig_b = self._generate_rev_signature(seg_b)
                signatures_a[f"seg_{i}_rev"] = sig_a
                signatures_b[f"seg_{i}_rev"] = sig_b
            
            if current_mode in [VerificationMode.HBT_ONLY, VerificationMode.UNIFIED]:
                sigs_a = self._generate_hbt_signatures(seg_a)
                sigs_b = self._generate_hbt_signatures(seg_b)
                for level, sig in sigs_a.items():
                    signatures_a[f"seg_{i}_{level}"] = sig
                for level, sig in sigs_b.items():
                    signatures_b[f"seg_{i}_{level}"] = sig
        
        # Perform weighted consensus validation
        validator_votes = {}
        validator_weights = {}
        
        for validator_id, reputation in self.verification_state.validator_reputations.items():
            # Simulate validator vote (in real system, this would be actual validator)
            vote = self._simulate_validator_vote(signatures_a, signatures_b, validator_id)
            validator_votes[validator_id] = vote
            validator_weights[validator_id] = reputation.weight
        
        # Compute weighted consensus
        weighted_behavioral = sum(
            vote['behavioral'] * validator_weights[vid]
            for vid, vote in validator_votes.items()
        ) / sum(validator_weights.values())
        
        weighted_architectural = sum(
            vote['architectural'] * validator_weights[vid]
            for vid, vote in validator_votes.items()
        ) / sum(validator_weights.values())
        
        # Check for Byzantine consensus
        consensus_reached = self._check_byzantine_consensus(
            validator_votes, validator_weights
        )
        
        # Update validator reputations
        consensus_value = weighted_behavioral > 0.5
        for validator_id, vote in validator_votes.items():
            aligned = (vote['behavioral'] > 0.5) == consensus_value
            self.verification_state.validator_reputations[validator_id].update(
                success=True,  # Assume successful validation
                consensus_aligned=aligned
            )
        
        # Create consensus result
        consensus_result = ConsensusResult(
            consensus_reached=consensus_reached,
            behavioral_agreement=weighted_behavioral,
            architectural_agreement=weighted_architectural,
            confidence_score=min(weighted_behavioral, weighted_architectural)
        )
        
        # Build Merkle tree for checkpoint
        segment_ids = [seg.segment_id for seg in self.verification_state.segment_buffer_a]
        leaves = [leaf_bytes([sid]) for sid in segment_ids]
        merkle_tree = build_merkle_tree(leaves)
        
        # Update confidence
        self.verification_state.update_confidence(consensus_result.confidence_score)
        
        # Create checkpoint
        checkpoint = ConsensusCheckpoint(
            checkpoint_id=checkpoint_id,
            segment_ids=segment_ids,
            consensus_result=consensus_result,
            timestamp=time.time(),
            merkle_root=merkle_tree['root'],
            confidence=consensus_result.confidence_score,
            segment_buffer_snapshot=list(self.verification_state.segment_buffer_a),
            verification_mode=current_mode,
            validator_reputations=dict(self.verification_state.validator_reputations),
            metadata={
                'weighted_behavioral': weighted_behavioral,
                'weighted_architectural': weighted_architectural,
                'validator_weights': validator_weights
            }
        )
        
        self.consensus_points.append(checkpoint)
        self.verification_state.consensus_checkpoints.append(checkpoint)
        
        return checkpoint
    
    def _simulate_validator_vote(
        self,
        signatures_a: Dict[str, np.ndarray],
        signatures_b: Dict[str, np.ndarray],
        validator_id: str
    ) -> Dict[str, float]:
        """
        Simulate a validator's vote (in production, this would be actual validator).
        
        Args:
            signatures_a: Signatures from model A
            signatures_b: Signatures from model B
            validator_id: ID of the validator
            
        Returns:
            Vote dictionary with behavioral and architectural scores
        """
        # Add some randomness to simulate different validator perspectives
        np.random.seed(hash(validator_id) % 2**32)
        noise = np.random.normal(0, 0.05)
        
        # Compute similarities
        similarities = []
        for key in signatures_a.keys():
            if key in signatures_b:
                sim = self._compute_robust_similarity(
                    signatures_a[key], signatures_b[key]
                )
                similarities.append(sim)
        
        base_similarity = np.mean(similarities) if similarities else 0.5
        
        # Add validator-specific noise
        behavioral = max(0, min(1, base_similarity + noise))
        architectural = max(0, min(1, base_similarity + noise * 0.5))
        
        return {
            'behavioral': behavioral,
            'architectural': architectural
        }
    
    def _check_byzantine_consensus(
        self,
        validator_votes: Dict[str, Dict[str, float]],
        validator_weights: Dict[str, float]
    ) -> bool:
        """
        Check if Byzantine consensus is reached.
        
        Args:
            validator_votes: Votes from all validators
            validator_weights: Weights for each validator
            
        Returns:
            True if consensus reached despite Byzantine validators
        """
        if not validator_votes:
            return False
        
        # Count validators agreeing on verdict (behavioral > 0.5 means SAME)
        same_votes = sum(
            validator_weights[vid]
            for vid, vote in validator_votes.items()
            if vote['behavioral'] > 0.5
        )
        
        different_votes = sum(
            validator_weights[vid]
            for vid, vote in validator_votes.items()
            if vote['behavioral'] <= 0.5
        )
        
        total_weight = sum(validator_weights.values())
        
        # Need more than (1 - byzantine_threshold) agreement
        required_weight = (1 - self.byzantine_threshold) * total_weight
        
        return max(same_votes, different_votes) >= required_weight
    
    def _should_early_stop(self, checkpoint: ConsensusCheckpoint) -> bool:
        """
        Determine if early stopping criteria are met.
        
        Args:
            checkpoint: Latest consensus checkpoint
            
        Returns:
            True if should stop early
        """
        # Check confidence threshold
        if self.verification_state.current_confidence >= self.early_stop_threshold:
            # Require at least 3 consecutive strong consensus checkpoints
            if len(self.consensus_points) >= 3:
                recent_checkpoints = self.consensus_points[-3:]
                all_strong = all(
                    cp.is_strong_consensus(self.early_stop_threshold)
                    for cp in recent_checkpoints
                )
                if all_strong:
                    return True
        
        # Check if models are clearly same/different
        if self.verification_state.sequential_state.n > 20:
            mean = self.verification_state.sequential_state.mean
            variance = self.verification_state.sequential_state.variance
            
            # High confidence bounds
            if variance < 0.01:  # Low variance
                if mean < 0.3:
                    # Very high similarity - models are same
                    return True
                elif mean > 0.7:
                    # Very low similarity - models are different
                    return True
        
        # Check mode stability for adaptive mode
        if self.verification_mode == VerificationMode.ADAPTIVE:
            if len(self.verification_state.mode_history) >= 20:
                recent_modes = self.verification_state.mode_history[-20:]
                if len(set(recent_modes)) == 1:
                    # Mode has stabilized
                    if self.verification_state.current_confidence > 0.8:
                        return True
        
        return False
    
    def _determine_current_verdict(self) -> Verdict:
        """
        Determine current verdict based on state.
        
        Returns:
            Current verdict
        """
        if self.verification_state.sequential_state.n < 10:
            return Verdict.UNDECIDED
        
        mean_distance = self.verification_state.sequential_state.mean
        confidence = self.verification_state.current_confidence
        
        # Use stricter thresholds for high confidence
        if confidence > 0.9:
            if mean_distance < 0.35:
                return Verdict.SAME
            elif mean_distance > 0.65:
                return Verdict.DIFFERENT
        
        # Medium confidence thresholds
        elif confidence > 0.7:
            if mean_distance < 0.3:
                return Verdict.SAME
            elif mean_distance > 0.7:
                return Verdict.DIFFERENT
        
        # Low confidence - more conservative
        else:
            if mean_distance < 0.25:
                return Verdict.SAME
            elif mean_distance > 0.75:
                return Verdict.DIFFERENT
        
        return Verdict.UNDECIDED
    
    def _determine_final_verdict(self) -> Verdict:
        """
        Determine final verdict with Byzantine consensus consideration.
        
        Returns:
            Final verdict
        """
        if not self.consensus_points:
            return Verdict.UNDECIDED
        
        # Weight recent checkpoints more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.consensus_points)))
        weights /= weights.sum()
        
        # Compute weighted metrics
        weighted_confidence = sum(
            w * cp.confidence 
            for w, cp in zip(weights, self.consensus_points)
        )
        
        # Get weighted behavioral agreement from metadata
        weighted_behavioral = 0.0
        for w, cp in zip(weights, self.consensus_points):
            if cp.metadata and 'weighted_behavioral' in cp.metadata:
                weighted_behavioral += w * cp.metadata['weighted_behavioral']
        
        # Count Byzantine consensus successes
        byzantine_success_rate = sum(
            1 for cp in self.consensus_points
            if cp.consensus_result and cp.consensus_result.consensus_reached
        ) / len(self.consensus_points)
        
        # Determine verdict with Byzantine consideration
        mean_similarity = np.mean(self.verification_state.similarity_history) if self.verification_state.similarity_history else 0.5
        
        # Require higher confidence if Byzantine consensus is weak
        confidence_threshold = 0.85 if byzantine_success_rate > 0.8 else 0.9
        
        if weighted_confidence > confidence_threshold:
            if mean_similarity > 0.65 and weighted_behavioral > 0.7:
                return Verdict.SAME
            elif mean_similarity < 0.35 or weighted_behavioral < 0.3:
                return Verdict.DIFFERENT
        
        # Fallback to sequential state with Byzantine adjustment
        adjusted_threshold = 0.5 - (0.05 * (1 - byzantine_success_rate))
        
        if self.verification_state.sequential_state.mean < adjusted_threshold:
            return Verdict.SAME
        elif self.verification_state.sequential_state.mean > (1 - adjusted_threshold):
            return Verdict.DIFFERENT
        
        return Verdict.UNDECIDED
    
    def create_segment_generators(
        self,
        model_a,
        model_b,
        challenges: List[str],
        tokenizer_a=None,
        tokenizer_b=None
    ) -> Tuple[Generator[Segment, None, None], Generator[Segment, None, None]]:
        """
        Create segment generators from models and challenges.
        
        Args:
            model_a: First model
            model_b: Second model
            challenges: List of challenge prompts
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            
        Returns:
            Tuple of (generator_a, generator_b)
        """
        def generate_segments(model, challenges, tokenizer):
            """Generate segments for a model."""
            segment_id = 0
            for challenge in challenges:
                # Simulate model processing (in production, use actual model)
                # This is a placeholder implementation
                tokens = list(range(100))  # Dummy tokens
                
                # Create segments
                for i in range(0, len(tokens), 50):
                    segment = Segment(
                        segment_id=segment_id,
                        tokens=tokens[i:i+50],
                        start_idx=i,
                        end_idx=min(i+50, len(tokens)),
                        signatures={}
                    )
                    segment_id += 1
                    yield segment
        
        gen_a = generate_segments(model_a, challenges, tokenizer_a)
        gen_b = generate_segments(model_b, challenges, tokenizer_b)
        
        return gen_a, gen_b
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of verification results.
        
        Returns:
            Dictionary with verification statistics
        """
        if not self.consensus_points:
            return {
                'segments_processed': self.verification_state.segments_processed,
                'checkpoints': 0,
                'final_verdict': None,
                'confidence': 0.0,
                'mode': self.verification_mode.value
            }
        
        strong_checkpoints = sum(
            1 for cp in self.consensus_points 
            if cp.is_strong_consensus()
        )
        
        # Mode distribution
        mode_counts = {}
        for mode in self.verification_state.mode_history:
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1
        
        # Validator performance
        best_validator = max(
            self.verification_state.validator_reputations.values(),
            key=lambda r: r.reputation_score
        )
        worst_validator = min(
            self.verification_state.validator_reputations.values(),
            key=lambda r: r.reputation_score
        )
        
        return {
            'segments_processed': self.verification_state.segments_processed,
            'checkpoints': len(self.consensus_points),
            'strong_checkpoints': strong_checkpoints,
            'final_verdict': self.verification_state.final_verdict.value if self.verification_state.final_verdict else None,
            'final_confidence': self.verification_state.current_confidence,
            'mean_similarity': np.mean(self.verification_state.similarity_history) if self.verification_state.similarity_history else 0,
            'early_stopped': self.verification_state.early_stopped,
            'verification_mode': self.verification_mode.value,
            'consensus_mode': self.consensus_mode.value,
            'mode_distribution': mode_counts,
            'byzantine_consensus_rate': sum(
                1 for cp in self.consensus_points
                if cp.consensus_result and cp.consensus_result.consensus_reached
            ) / len(self.consensus_points) if self.consensus_points else 0,
            'best_validator': {
                'id': best_validator.validator_id,
                'reputation': best_validator.reputation_score,
                'success_rate': best_validator.successful_validations / best_validator.total_validations if best_validator.total_validations > 0 else 0
            },
            'worst_validator': {
                'id': worst_validator.validator_id,
                'reputation': worst_validator.reputation_score,
                'success_rate': worst_validator.successful_validations / worst_validator.total_validations if worst_validator.total_validations > 0 else 0
            },
            'checkpoints_saved': len(list(self.checkpoint_dir.glob("checkpoint_*.json")))
        }