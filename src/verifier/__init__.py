"""Model verification pipeline for REV."""

from .modes import ModeParams, TestingMode
from .stats import Welford, eb_halfwidth, spending_schedule, UnifiedStatistics
from .scoring import bounded_difference, normalize
from .decision import EnhancedSequentialTester, Verdict, StepRecord, RunResult
from .streaming_consensus import (
    StreamingConsensusVerifier,
    ConsensusCheckpoint,
    StreamingVerificationState,
    ConsensusMode
)
from .contamination import (
    UnifiedContaminationDetector,
    ContaminationResult,
    ContaminationType,
    ModelSignature
)

__all__ = [
    "ModeParams", 
    "TestingMode",
    "Welford",
    "UnifiedStatistics",
    "eb_halfwidth", 
    "spending_schedule",
    "bounded_difference",
    "normalize",
    "EnhancedSequentialTester",
    "Verdict",
    "StepRecord", 
    "RunResult",
    "StreamingConsensusVerifier",
    "ConsensusCheckpoint",
    "StreamingVerificationState",
    "ConsensusMode",
    "UnifiedContaminationDetector",
    "ContaminationResult",
    "ContaminationType",
    "ModelSignature"
]