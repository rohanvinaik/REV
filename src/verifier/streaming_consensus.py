"""
Streaming Consensus Verifier - Combines REV streaming with HBT consensus.

This module implements memory-bounded streaming verification with periodic
Byzantine consensus checkpoints for robust LLM comparison.
"""

from typing import Dict, List, Tuple, Optional, Any, Generator, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import time
from enum import Enum

from ..rev_pipeline import REVPipeline, Segment
from ..consensus.byzantine import ConsensusNetwork, ConsensusResult, ByzantineValidator
from ..core.sequential import SequentialState, SPRTResult
from ..verifier.decision import Verdict, StepRecord, RunResult
from ..hdc.encoder import UnifiedHDCEncoder, HypervectorConfig
from ..crypto.merkle import build_merkle_tree, leaf_bytes


class ConsensusMode(Enum):
    """Consensus verification modes."""
    FAST = "fast"  # Consensus every 8 segments
    BALANCED = "balanced"  # Consensus every 4 segments (default)
    ROBUST = "robust"  # Consensus every 2 segments


@dataclass
class ConsensusCheckpoint:
    """Checkpoint for consensus verification."""
    
    segment_ids: List[int]
    consensus_result: ConsensusResult
    timestamp: float
    merkle_root: bytes
    confidence: float
    segment_buffer_snapshot: List[Segment]
    
    def is_strong_consensus(self, threshold: float = 0.9) -> bool:
        """Check if this checkpoint has strong consensus."""
        return (
            self.consensus_result.consensus_reached and
            self.confidence >= threshold
        )


@dataclass 
class StreamingVerificationState:
    """State for streaming consensus verification."""
    
    segments_processed: int = 0
    consensus_checkpoints: List[ConsensusCheckpoint] = field(default_factory=list)
    current_confidence: float = 0.0
    sequential_state: SequentialState = field(default_factory=SequentialState)
    early_stopped: bool = False
    final_verdict: Optional[Verdict] = None
    segment_buffer_a: deque = field(default_factory=lambda: deque(maxlen=4))
    segment_buffer_b: deque = field(default_factory=lambda: deque(maxlen=4))
    similarity_history: List[float] = field(default_factory=list)
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update running confidence with smoothing."""
        alpha = 0.3  # Smoothing factor
        self.current_confidence = (
            alpha * new_confidence + 
            (1 - alpha) * self.current_confidence
        )


class StreamingConsensusVerifier:
    """
    Streaming consensus verifier combining REV and HBT approaches.
    
    Provides memory-bounded streaming verification with periodic Byzantine
    consensus checkpoints for robust model comparison.
    """
    
    def __init__(
        self,
        rev_pipeline: Optional[REVPipeline] = None,
        consensus_network: Optional[ConsensusNetwork] = None,
        consensus_mode: ConsensusMode = ConsensusMode.BALANCED,
        early_stop_threshold: float = 0.95,
        max_segments: int = 1000,
        encoder_config: Optional[HypervectorConfig] = None
    ):
        """
        Initialize streaming consensus verifier.
        
        Args:
            rev_pipeline: REV pipeline for segment processing
            consensus_network: Byzantine consensus network
            consensus_mode: Frequency of consensus checkpoints
            early_stop_threshold: Confidence threshold for early stopping
            max_segments: Maximum segments to process
            encoder_config: Configuration for HDC encoding
        """
        # Initialize REV pipeline
        self.rev_pipeline = rev_pipeline or REVPipeline(
            segment_size=512,
            buffer_size=4
        )
        
        # Initialize consensus network
        self.consensus_network = consensus_network or ConsensusNetwork(
            num_validators=4,
            behavioral_threshold=0.85,
            architectural_threshold=0.90,
            batch_size=4
        )
        
        # Initialize unified encoder for adaptive encoding
        encoder_config = encoder_config or HypervectorConfig(
            dimension=10000,
            encoding_mode="hybrid",
            multi_scale=True
        )
        self.encoder = UnifiedHDCEncoder(encoder_config, auto_mode=True)
        
        # Configuration
        self.consensus_mode = consensus_mode
        self.early_stop_threshold = early_stop_threshold
        self.max_segments = max_segments
        
        # Set consensus frequency based on mode
        self.consensus_frequency = {
            ConsensusMode.FAST: 8,
            ConsensusMode.BALANCED: 4,
            ConsensusMode.ROBUST: 2
        }[consensus_mode]
        
        # State tracking
        self.verification_state = StreamingVerificationState()
        self.consensus_points: List[ConsensusCheckpoint] = []
    
    def stream_verify(
        self,
        model_stream_a: Generator[Segment, None, None],
        model_stream_b: Generator[Segment, None, None],
        challenges: Optional[List[str]] = None,
        tokenizer_a=None,
        tokenizer_b=None
    ) -> Generator[Dict[str, Any], None, RunResult]:
        """
        Perform streaming verification with consensus checkpoints.
        
        Args:
            model_stream_a: Generator of segments from model A
            model_stream_b: Generator of segments from model B
            challenges: Optional list of challenge prompts
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            
        Yields:
            Verification updates at each checkpoint
            
        Returns:
            Final RunResult with verdict
        """
        start_time = time.time()
        step_records = []
        
        # Process segments in parallel from both streams
        segment_count = 0
        
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
                
                # Generate signatures for segments
                segment_a.signatures = self._generate_signatures(segment_a, "model_a")
                segment_b.signatures = self._generate_signatures(segment_b, "model_b")
                
                # Compute segment similarity
                similarity = self._compute_segment_similarity(segment_a, segment_b)
                self.verification_state.similarity_history.append(similarity)
                
                # Update sequential state
                self.verification_state.sequential_state.update(1.0 - similarity)
                
                # Check for consensus checkpoint
                if segment_count % self.consensus_frequency == 0:
                    checkpoint = self._perform_consensus_checkpoint()
                    
                    # Create step record
                    step_record = StepRecord(
                        index=len(step_records) + 1,
                        prompt=f"Segments {segment_count - self.consensus_frequency + 1}-{segment_count}",
                        ref_output=f"Model A: {len(self.verification_state.segment_buffer_a)} segments",
                        cand_output=f"Model B: {len(self.verification_state.segment_buffer_b)} segments",
                        score=1.0 - similarity,
                        mean=self.verification_state.sequential_state.mean,
                        var=self.verification_state.sequential_state.variance,
                        halfwidth=np.sqrt(self.verification_state.sequential_state.variance),
                        delta_n=0.05,  # Default spending schedule
                        verdict_so_far=self._determine_current_verdict().value
                    )
                    step_records.append(step_record)
                    
                    # Yield checkpoint update
                    yield {
                        'type': 'checkpoint',
                        'segment_count': segment_count,
                        'checkpoint': checkpoint,
                        'current_confidence': self.verification_state.current_confidence,
                        'mean_similarity': np.mean(self.verification_state.similarity_history),
                        'verdict_so_far': self._determine_current_verdict(),
                        'consensus_reached': checkpoint.consensus_result.consensus_reached,
                        'behavioral_agreement': checkpoint.consensus_result.behavioral_agreement,
                        'architectural_agreement': checkpoint.consensus_result.architectural_agreement
                    }
                    
                    # Check for early stopping
                    if self._should_early_stop(checkpoint):
                        self.verification_state.early_stopped = True
                        self.verification_state.final_verdict = self._determine_current_verdict()
                        break
                
                # Periodic status update (every 10 segments)
                if segment_count % 10 == 0:
                    yield {
                        'type': 'status',
                        'segments_processed': segment_count,
                        'current_confidence': self.verification_state.current_confidence,
                        'buffer_sizes': (
                            len(self.verification_state.segment_buffer_a),
                            len(self.verification_state.segment_buffer_b)
                        )
                    }
        
        except Exception as e:
            # Handle errors gracefully
            yield {
                'type': 'error',
                'error': str(e),
                'segments_processed': segment_count
            }
        
        # Final consensus if buffer has segments
        if self.verification_state.segment_buffer_a:
            final_checkpoint = self._perform_consensus_checkpoint()
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
                'consensus_mode': self.consensus_mode.value,
                'consensus_frequency': self.consensus_frequency,
                'early_stop_threshold': self.early_stop_threshold,
                'total_checkpoints': len(self.consensus_points),
                'early_stopped': self.verification_state.early_stopped,
                'final_confidence': self.verification_state.current_confidence,
                'mean_similarity': np.mean(self.verification_state.similarity_history),
                'time_taken': time.time() - start_time
            }
        )
        
        return result
    
    def _generate_signatures(self, segment: Segment, model_id: str) -> Dict[str, np.ndarray]:
        """
        Generate HDC signatures for a segment.
        
        Args:
            segment: Segment to process
            model_id: Identifier for the model
            
        Returns:
            Dictionary of signatures
        """
        signatures = {}
        
        # Convert tokens to features
        features = torch.tensor(segment.tokens, dtype=torch.float32)
        
        # Generate signatures at multiple zoom levels
        for zoom_level in ["token_window", "span"]:
            # Use adaptive encoding based on context
            context = "streaming" if zoom_level == "token_window" else "consensus"
            
            encoded = self.encoder.encode_adaptive(
                features,
                context=context,
                behavioral_site=f"{model_id}_segment_{segment.segment_id}",
                zoom_level=zoom_level
            )
            
            signatures[f"{zoom_level}_signature"] = encoded.numpy()
        
        return signatures
    
    def _compute_segment_similarity(self, segment_a: Segment, segment_b: Segment) -> float:
        """
        Compute similarity between two segments.
        
        Args:
            segment_a: First segment
            segment_b: Second segment
            
        Returns:
            Similarity score in [0, 1]
        """
        if not segment_a.signatures or not segment_b.signatures:
            return 0.0
        
        similarities = []
        
        for sig_name in segment_a.signatures.keys():
            if sig_name in segment_b.signatures:
                sig_a = torch.from_numpy(segment_a.signatures[sig_name])
                sig_b = torch.from_numpy(segment_b.signatures[sig_name])
                
                # Use Hamming similarity for binary comparison
                sim = self.encoder.compute_hamming_similarity(sig_a, sig_b, normalize=True)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _perform_consensus_checkpoint(self) -> ConsensusCheckpoint:
        """
        Perform Byzantine consensus on current segment buffers.
        
        Returns:
            ConsensusCheckpoint with results
        """
        # Prepare signatures for consensus
        signatures_a = {}
        signatures_b = {}
        
        for i, (seg_a, seg_b) in enumerate(zip(
            self.verification_state.segment_buffer_a,
            self.verification_state.segment_buffer_b
        )):
            if seg_a.signatures:
                for k, v in seg_a.signatures.items():
                    signatures_a[f"seg_{i}_{k}"] = v
            if seg_b.signatures:
                for k, v in seg_b.signatures.items():
                    signatures_b[f"seg_{i}_{k}"] = v
        
        # Perform consensus validation
        consensus_result = self.consensus_network.validate_segments(
            self.verification_state.segment_buffer_a,
            pipeline_signatures={'model_a': signatures_a, 'model_b': signatures_b}
        )
        
        # Build Merkle tree for checkpoint
        segment_ids = [seg.segment_id for seg in self.verification_state.segment_buffer_a]
        leaves = [leaf_bytes([sid]) for sid in segment_ids]
        merkle_tree = build_merkle_tree(leaves)
        
        # Update confidence
        self.verification_state.update_confidence(consensus_result.confidence_score)
        
        # Create checkpoint
        checkpoint = ConsensusCheckpoint(
            segment_ids=segment_ids,
            consensus_result=consensus_result,
            timestamp=time.time(),
            merkle_root=merkle_tree['root'],
            confidence=consensus_result.confidence_score,
            segment_buffer_snapshot=list(self.verification_state.segment_buffer_a)
        )
        
        self.consensus_points.append(checkpoint)
        return checkpoint
    
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
        
        # Check if models are clearly different
        if self.verification_state.sequential_state.n > 20:
            if self.verification_state.sequential_state.mean < 0.3:
                # Very high similarity - models are same
                return True
            elif self.verification_state.sequential_state.mean > 0.8:
                # Very low similarity - models are different
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
        
        # High confidence consensus
        if confidence > 0.9:
            if mean_distance < 0.4:
                return Verdict.SAME
            elif mean_distance > 0.6:
                return Verdict.DIFFERENT
        
        # Medium confidence
        elif confidence > 0.7:
            if mean_distance < 0.3:
                return Verdict.SAME
            elif mean_distance > 0.7:
                return Verdict.DIFFERENT
        
        # Low confidence - need more data
        return Verdict.UNDECIDED
    
    def _determine_final_verdict(self) -> Verdict:
        """
        Determine final verdict based on all checkpoints.
        
        Returns:
            Final verdict
        """
        if not self.consensus_points:
            return Verdict.UNDECIDED
        
        # Weight recent checkpoints more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.consensus_points)))
        weights /= weights.sum()
        
        # Compute weighted confidence and agreement
        weighted_confidence = sum(
            w * cp.confidence 
            for w, cp in zip(weights, self.consensus_points)
        )
        
        weighted_behavioral = sum(
            w * cp.consensus_result.behavioral_agreement
            for w, cp in zip(weights, self.consensus_points)
        )
        
        weighted_architectural = sum(
            w * cp.consensus_result.architectural_agreement
            for w, cp in zip(weights, self.consensus_points)
        )
        
        # Determine verdict based on weighted metrics
        mean_similarity = np.mean(self.verification_state.similarity_history)
        
        if weighted_confidence > 0.85:
            if mean_similarity > 0.7 and weighted_behavioral > 0.75:
                return Verdict.SAME
            elif mean_similarity < 0.4 or weighted_behavioral < 0.5:
                return Verdict.DIFFERENT
        
        # Fallback to sequential state
        if self.verification_state.sequential_state.mean < 0.45:
            return Verdict.SAME
        elif self.verification_state.sequential_state.mean > 0.55:
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
            for challenge in challenges:
                # Use REV pipeline to process challenge
                result = self.rev_pipeline.process_challenge(
                    model, challenge, tokenizer
                )
                
                # Yield segments from the result
                for segment_data in result['segment_signatures']:
                    # Reconstruct segment from data
                    segment = Segment(
                        segment_id=segment_data['id'],
                        tokens=[],  # Tokens already processed
                        start_idx=0,
                        end_idx=0,
                        signatures=segment_data.get('signatures', {})
                    )
                    yield segment
        
        gen_a = generate_segments(model_a, challenges, tokenizer_a)
        gen_b = generate_segments(model_b, challenges, tokenizer_b)
        
        return gen_a, gen_b
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get summary of verification results.
        
        Returns:
            Dictionary with verification statistics
        """
        if not self.consensus_points:
            return {
                'segments_processed': self.verification_state.segments_processed,
                'checkpoints': 0,
                'final_verdict': None,
                'confidence': 0.0
            }
        
        strong_checkpoints = sum(
            1 for cp in self.consensus_points 
            if cp.is_strong_consensus()
        )
        
        return {
            'segments_processed': self.verification_state.segments_processed,
            'checkpoints': len(self.consensus_points),
            'strong_checkpoints': strong_checkpoints,
            'final_verdict': self.verification_state.final_verdict.value if self.verification_state.final_verdict else None,
            'final_confidence': self.verification_state.current_confidence,
            'mean_similarity': np.mean(self.verification_state.similarity_history),
            'early_stopped': self.verification_state.early_stopped,
            'consensus_mode': self.consensus_mode.value,
            'behavioral_agreement': np.mean([
                cp.consensus_result.behavioral_agreement 
                for cp in self.consensus_points
            ]),
            'architectural_agreement': np.mean([
                cp.consensus_result.architectural_agreement
                for cp in self.consensus_points
            ])
        }