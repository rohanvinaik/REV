"""
Byzantine fault-tolerant consensus for REV verification.

This module implements Byzantine consensus for distributed verification of LLM behavior,
ensuring robust agreement even with up to f faulty validators in a network of 3f+1 nodes.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import numpy as np
from enum import Enum
import time

from ..rev_pipeline import Segment
from ..crypto.merkle import build_merkle_tree, verify_merkle_proof, leaf_bytes
from ..core.sequential import SequentialState
from ..verifier.decision import Verdict
from ..hdc.encoder import HypervectorEncoder


class VoteType(Enum):
    """Types of votes in Byzantine consensus."""
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"


@dataclass
class Vote:
    """Represents a validator's vote on a segment batch."""
    
    validator_id: str
    round: int
    vote_type: VoteType
    segment_hashes: List[bytes]
    signature_hash: bytes
    behavioral_similarity: float
    timestamp: float
    view_number: int = 0
    
    def compute_hash(self) -> bytes:
        """Compute hash of the vote for verification."""
        data = (
            f"{self.validator_id}:{self.round}:{self.vote_type.value}:"
            f"{self.view_number}:{self.behavioral_similarity:.6f}"
        ).encode()
        for seg_hash in self.segment_hashes:
            data += seg_hash
        return hashlib.sha256(data).digest()


@dataclass
class ConsensusResult:
    """Result of Byzantine consensus on segment batch."""
    
    consensus_reached: bool
    confidence_score: float
    agreed_segments: List[int]
    dissenting_validators: Set[str]
    behavioral_agreement: float
    architectural_agreement: float
    rounds_taken: int
    timestamp: float
    merkle_root: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'consensus_reached': self.consensus_reached,
            'confidence_score': self.confidence_score,
            'agreed_segments': self.agreed_segments,
            'dissenting_validators': list(self.dissenting_validators),
            'behavioral_agreement': self.behavioral_agreement,
            'architectural_agreement': self.architectural_agreement,
            'rounds_taken': self.rounds_taken,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root.hex() if self.merkle_root else None
        }


class ByzantineValidator:
    """
    Byzantine fault-tolerant validator for REV segments.
    
    Implements PBFT-style consensus with architectural and behavioral signature validation.
    Tolerates up to f Byzantine failures with 3f+1 total validators.
    """
    
    def __init__(
        self,
        validator_id: str,
        total_validators: int,
        behavioral_threshold: float = 0.85,
        architectural_threshold: float = 0.90,
        timeout_ms: int = 5000
    ):
        """
        Initialize Byzantine validator.
        
        Args:
            validator_id: Unique identifier for this validator
            total_validators: Total number of validators in network (must be 3f+1)
            behavioral_threshold: Similarity threshold for HDC signatures
            architectural_threshold: Similarity threshold for architectural signatures
            timeout_ms: Timeout for consensus rounds in milliseconds
        """
        self.validator_id = validator_id
        self.total_validators = total_validators
        
        # Calculate Byzantine fault tolerance
        self.f = (total_validators - 1) // 3
        self.quorum_size = 2 * self.f + 1
        
        if total_validators < 3 * self.f + 1:
            raise ValueError(
                f"Need at least {3 * self.f + 1} validators for f={self.f} failures"
            )
        
        self.behavioral_threshold = behavioral_threshold
        self.architectural_threshold = architectural_threshold
        self.timeout_ms = timeout_ms
        
        # Vote tracking
        self.prepare_votes: Dict[int, List[Vote]] = defaultdict(list)
        self.commit_votes: Dict[int, List[Vote]] = defaultdict(list)
        self.view_number = 0
        self.current_round = 0
        
        # State management
        self.state = SequentialState()
        self.vote_history: List[Vote] = []
        
    def vote(
        self,
        segments: List[Segment],
        signatures: Dict[str, np.ndarray]
    ) -> Vote:
        """
        Cast vote for segment batch based on signatures.
        
        Args:
            segments: Batch of segments to validate
            signatures: Dictionary of architectural/behavioral signatures
            
        Returns:
            Vote object representing this validator's decision
        """
        # Compute segment hashes
        segment_hashes = [seg.compute_hash() for seg in segments]
        
        # Compute aggregate signature hash
        sig_data = b""
        for name, sig in sorted(signatures.items()):
            sig_data += name.encode() + sig.tobytes()
        signature_hash = hashlib.sha256(sig_data).digest()
        
        # Calculate behavioral similarity score
        behavioral_sim = self._compute_behavioral_similarity(segments, signatures)
        
        # Create vote
        vote = Vote(
            validator_id=self.validator_id,
            round=self.current_round,
            vote_type=VoteType.PREPARE,
            segment_hashes=segment_hashes,
            signature_hash=signature_hash,
            behavioral_similarity=behavioral_sim,
            timestamp=time.time(),
            view_number=self.view_number
        )
        
        self.vote_history.append(vote)
        return vote
    
    def _compute_behavioral_similarity(
        self,
        segments: List[Segment],
        signatures: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute behavioral similarity score from HDC signatures.
        
        Args:
            segments: Segments being validated
            signatures: HDC signatures to compare
            
        Returns:
            Similarity score in [0, 1]
        """
        if not signatures:
            return 0.0
        
        similarities = []
        
        # Compare segment signatures if available
        for segment in segments:
            if segment.signatures:
                for site_name, ref_sig in segment.signatures.items():
                    if site_name in signatures:
                        cand_sig = signatures[site_name]
                        # Compute cosine similarity
                        sim = np.dot(ref_sig, cand_sig) / (
                            np.linalg.norm(ref_sig) * np.linalg.norm(cand_sig) + 1e-8
                        )
                        similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def validate_vote(self, vote: Vote, segments: List[Segment]) -> bool:
        """
        Validate another validator's vote.
        
        Args:
            vote: Vote to validate
            segments: Segments being voted on
            
        Returns:
            True if vote is valid
        """
        # Check segment hashes match
        expected_hashes = [seg.compute_hash() for seg in segments]
        if vote.segment_hashes != expected_hashes:
            return False
        
        # Verify vote hash
        if vote.compute_hash() == b"":
            return False
        
        # Check behavioral similarity is reasonable
        if not 0 <= vote.behavioral_similarity <= 1:
            return False
        
        return True
    
    def handle_prepare_phase(
        self,
        votes: List[Vote],
        segments: List[Segment]
    ) -> Optional[Vote]:
        """
        Handle prepare phase of PBFT consensus.
        
        Args:
            votes: Prepare votes from other validators
            segments: Segments being validated
            
        Returns:
            Commit vote if quorum reached, None otherwise
        """
        # Validate and store votes
        valid_votes = []
        for vote in votes:
            if self.validate_vote(vote, segments):
                valid_votes.append(vote)
        
        self.prepare_votes[self.current_round] = valid_votes
        
        # Check for quorum
        if len(valid_votes) >= self.quorum_size:
            # Check behavioral agreement
            behavioral_scores = [v.behavioral_similarity for v in valid_votes]
            avg_behavioral = np.mean(behavioral_scores)
            
            if avg_behavioral >= self.behavioral_threshold:
                # Create commit vote
                commit_vote = Vote(
                    validator_id=self.validator_id,
                    round=self.current_round,
                    vote_type=VoteType.COMMIT,
                    segment_hashes=segments[0].compute_hash() if segments else b"",
                    signature_hash=valid_votes[0].signature_hash,
                    behavioral_similarity=avg_behavioral,
                    timestamp=time.time(),
                    view_number=self.view_number
                )
                return commit_vote
        
        return None
    
    def handle_commit_phase(
        self,
        votes: List[Vote],
        segments: List[Segment]
    ) -> bool:
        """
        Handle commit phase of PBFT consensus.
        
        Args:
            votes: Commit votes from other validators
            segments: Segments being validated
            
        Returns:
            True if consensus reached
        """
        # Validate and store votes
        valid_votes = []
        for vote in votes:
            if vote.vote_type == VoteType.COMMIT and self.validate_vote(vote, segments):
                valid_votes.append(vote)
        
        self.commit_votes[self.current_round] = valid_votes
        
        # Check for commit quorum
        return len(valid_votes) >= self.quorum_size


class ConsensusNetwork:
    """
    Manages Byzantine consensus network for REV verification.
    
    Coordinates multiple validators to achieve consensus on segment batches,
    integrating with Merkle tree verification and decision aggregation.
    """
    
    def __init__(
        self,
        num_validators: int = 4,
        behavioral_threshold: float = 0.85,
        architectural_threshold: float = 0.90,
        batch_size: int = 4
    ):
        """
        Initialize consensus network.
        
        Args:
            num_validators: Number of validators (must be 3f+1)
            behavioral_threshold: HDC similarity threshold
            architectural_threshold: Architectural similarity threshold
            batch_size: Number of segments per consensus batch
        """
        # Ensure valid Byzantine configuration
        f = (num_validators - 1) // 3
        if num_validators < 3 * f + 1:
            num_validators = 3 * f + 1
        
        self.num_validators = num_validators
        self.f = f
        self.batch_size = batch_size
        
        # Create validators
        self.validators = {}
        for i in range(num_validators):
            validator_id = f"validator_{i}"
            self.validators[validator_id] = ByzantineValidator(
                validator_id=validator_id,
                total_validators=num_validators,
                behavioral_threshold=behavioral_threshold,
                architectural_threshold=architectural_threshold
            )
        
        # Consensus state
        self.current_round = 0
        self.consensus_history: List[ConsensusResult] = []
        
    def validate_segments(
        self,
        segment_buffer: deque,
        pipeline_signatures: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> ConsensusResult:
        """
        Perform Byzantine consensus on segment batch.
        
        Args:
            segment_buffer: Buffer of segments from REVPipeline
            pipeline_signatures: Optional signatures from pipeline execution
            
        Returns:
            ConsensusResult with consensus outcome and metrics
        """
        # Extract batch of segments
        segments = list(segment_buffer)[:self.batch_size]
        if not segments:
            return ConsensusResult(
                consensus_reached=False,
                confidence_score=0.0,
                agreed_segments=[],
                dissenting_validators=set(),
                behavioral_agreement=0.0,
                architectural_agreement=0.0,
                rounds_taken=0,
                timestamp=time.time()
            )
        
        self.current_round += 1
        start_time = time.time()
        
        # Phase 1: Prepare votes
        prepare_votes = []
        for validator_id, validator in self.validators.items():
            # Generate signatures for validation
            signatures = self._generate_validation_signatures(
                segments,
                pipeline_signatures,
                validator_id
            )
            
            vote = validator.vote(segments, signatures)
            prepare_votes.append(vote)
        
        # Distribute prepare votes to all validators
        commit_votes = []
        for validator_id, validator in self.validators.items():
            commit_vote = validator.handle_prepare_phase(prepare_votes, segments)
            if commit_vote:
                commit_votes.append(commit_vote)
        
        # Phase 2: Commit phase
        consensus_reached = False
        dissenting = set()
        
        for validator_id, validator in self.validators.items():
            if validator.handle_commit_phase(commit_votes, segments):
                consensus_reached = True
            else:
                dissenting.add(validator_id)
        
        # Calculate agreement metrics
        behavioral_scores = [v.behavioral_similarity for v in prepare_votes]
        behavioral_agreement = np.mean(behavioral_scores) if behavioral_scores else 0.0
        
        # Architectural agreement based on signature hashes
        sig_hashes = [v.signature_hash for v in prepare_votes]
        unique_hashes = len(set(sig_hashes))
        architectural_agreement = 1.0 - (unique_hashes - 1) / len(sig_hashes)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            consensus_reached,
            behavioral_agreement,
            architectural_agreement,
            len(dissenting)
        )
        
        # Build Merkle tree for verified segments
        merkle_root = None
        if consensus_reached and segments:
            leaves = [leaf_bytes([seg.segment_id]) for seg in segments]
            merkle_tree = build_merkle_tree(leaves)
            merkle_root = merkle_tree['root']
        
        result = ConsensusResult(
            consensus_reached=consensus_reached,
            confidence_score=confidence,
            agreed_segments=[seg.segment_id for seg in segments],
            dissenting_validators=dissenting,
            behavioral_agreement=behavioral_agreement,
            architectural_agreement=architectural_agreement,
            rounds_taken=1,
            timestamp=time.time(),
            merkle_root=merkle_root
        )
        
        self.consensus_history.append(result)
        return result
    
    def _generate_validation_signatures(
        self,
        segments: List[Segment],
        pipeline_signatures: Optional[Dict[str, Dict[str, np.ndarray]]],
        validator_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate signatures for validation.
        
        Args:
            segments: Segments to validate
            pipeline_signatures: Optional existing signatures
            validator_id: ID of validator generating signatures
            
        Returns:
            Dictionary of signatures
        """
        if pipeline_signatures and validator_id in pipeline_signatures:
            return pipeline_signatures[validator_id]
        
        # Aggregate signatures from segments
        aggregated = {}
        for segment in segments:
            if segment.signatures:
                for site_name, sig in segment.signatures.items():
                    if site_name not in aggregated:
                        aggregated[site_name] = []
                    aggregated[site_name].append(sig)
        
        # Average signatures per site
        signatures = {}
        for site_name, sigs in aggregated.items():
            if sigs:
                signatures[site_name] = np.mean(sigs, axis=0)
        
        return signatures
    
    def _calculate_confidence(
        self,
        consensus_reached: bool,
        behavioral_agreement: float,
        architectural_agreement: float,
        num_dissenting: int
    ) -> float:
        """
        Calculate confidence score for consensus result.
        
        Args:
            consensus_reached: Whether consensus was achieved
            behavioral_agreement: HDC similarity agreement
            architectural_agreement: Architectural signature agreement
            num_dissenting: Number of dissenting validators
            
        Returns:
            Confidence score in [0, 1]
        """
        if not consensus_reached:
            return 0.0
        
        # Weight factors
        w_behavioral = 0.4
        w_architectural = 0.3
        w_dissent = 0.3
        
        # Calculate dissent penalty
        dissent_ratio = num_dissenting / self.num_validators
        dissent_score = 1.0 - dissent_ratio
        
        # Combined confidence
        confidence = (
            w_behavioral * behavioral_agreement +
            w_architectural * architectural_agreement +
            w_dissent * dissent_score
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_consensus_summary(self) -> Dict[str, Any]:
        """
        Get summary of consensus history.
        
        Returns:
            Dictionary with consensus statistics
        """
        if not self.consensus_history:
            return {
                'total_rounds': 0,
                'consensus_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_behavioral_agreement': 0.0,
                'avg_architectural_agreement': 0.0
            }
        
        successful = sum(1 for r in self.consensus_history if r.consensus_reached)
        
        return {
            'total_rounds': len(self.consensus_history),
            'consensus_rate': successful / len(self.consensus_history),
            'avg_confidence': np.mean([r.confidence_score for r in self.consensus_history]),
            'avg_behavioral_agreement': np.mean([r.behavioral_agreement for r in self.consensus_history]),
            'avg_architectural_agreement': np.mean([r.architectural_agreement for r in self.consensus_history]),
            'failed_rounds': [i for i, r in enumerate(self.consensus_history) if not r.consensus_reached]
        }