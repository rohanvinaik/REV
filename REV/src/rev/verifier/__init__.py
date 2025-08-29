"""Model verification pipeline for REV."""

from .modes import ModeParams, TestingMode
from .stats import Welford, eb_halfwidth, spending_schedule
from .scoring import bounded_difference, normalize
from .decision import EnhancedSequentialTester, Verdict, StepRecord, RunResult

__all__ = [
    "ModeParams", 
    "TestingMode",
    "Welford",
    "eb_halfwidth", 
    "spending_schedule",
    "bounded_difference",
    "normalize",
    "EnhancedSequentialTester",
    "Verdict",
    "StepRecord", 
    "RunResult"
]