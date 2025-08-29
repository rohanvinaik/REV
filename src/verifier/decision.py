from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Callable, Iterable

from .modes import ModeParams
from .stats import Welford, eb_halfwidth, spending_schedule
from .scoring import bounded_difference


class Verdict(str, Enum):
    """Verification verdict for REV model comparison"""
    SAME = "SAME"           # Models are functionally equivalent
    DIFFERENT = "DIFFERENT" # Models produce meaningfully different outputs
    UNDECIDED = "UNDECIDED" # Need more samples to reach decision


@dataclass
class StepRecord:
    """Record of a single verification step in REV"""
    index: int              # Step number (1-indexed)
    prompt: str            # Input prompt used
    ref_output: str        # Reference model output
    cand_output: str       # Candidate model output
    score: float           # Distance score between outputs
    mean: float            # Running mean of scores
    var: float             # Running variance of scores
    halfwidth: float       # Confidence interval half-width
    delta_n: float         # Spending schedule value
    verdict_so_far: str    # Current verdict at this step


@dataclass
class RunResult:
    """Complete result of REV verification run"""
    verdict: Verdict                # Final decision
    steps: list[StepRecord]         # Complete trajectory
    n_used: int                     # Number of samples used
    params: dict                    # Parameters used for verification


class EnhancedSequentialTester:
    """
    Sequential tester for REV memory-bounded verification.
    
    This implementation provides:
    - Streaming execution framework for memory efficiency
    - Anytime-valid confidence bounds using empirical Bernstein
    - Adaptive stopping rules for early termination
    - Comprehensive audit logging for verification certificates
    
    The tester processes prompt-response pairs sequentially and maintains
    running statistics with Welford's algorithm for numerical stability.
    """

    def __init__(
        self,
        params: ModeParams,
        score_fn: Callable[[str, str], float] = bounded_difference,
    ) -> None:
        """
        Initialize REV sequential tester.
        
        Args:
            params: Testing mode parameters (significance levels, thresholds)
            score_fn: Function to compute distance between model outputs
        """
        self.params = params
        self.score_fn = score_fn

    def _same_rule(self, mean: float, h: float) -> bool:
        """
        Check if models should be considered equivalent.
        
        Models are SAME if:
        1. Upper confidence bound is below gamma threshold
        2. Confidence interval is narrow enough (precision requirement)
        
        Args:
            mean: Current empirical mean of distance scores
            h: Current confidence interval half-width
            
        Returns:
            True if models should be considered equivalent
        """
        upper_bound_ok = (mean + h) <= self.params.gamma
        precision_ok = h <= self.params.eta * self.params.gamma
        return upper_bound_ok and precision_ok

    def _diff_rule(self, mean: float, h: float) -> bool:
        """
        Check if models should be considered different.
        
        Models are DIFFERENT if:
        1. Mean distance exceeds delta_star threshold
        2. Relative margin (h/mean) is small enough for confidence
        
        Args:
            mean: Current empirical mean of distance scores
            h: Current confidence interval half-width
            
        Returns:
            True if models should be considered different
        """
        if mean < self.params.delta_star:
            return False
        
        relative_margin = h / max(mean, 1e-12)
        return relative_margin <= self.params.eps_diff

    def run(
        self,
        prompts: Iterable[str],
        ref_generate: Callable[[str], str],
        cand_generate: Callable[[str], str],
    ) -> RunResult:
        """
        Run REV verification with streaming execution.
        
        This is the core verification loop that:
        1. Streams prompts to both models
        2. Computes distances between outputs
        3. Updates running statistics
        4. Checks stopping conditions
        5. Returns comprehensive audit trail
        
        Args:
            prompts: Iterable of challenge prompts
            ref_generate: Reference model generation function
            cand_generate: Candidate model generation function
            
        Returns:
            RunResult with verdict and complete execution trace
        """
        w = Welford()  # Numerically stable online statistics
        steps: list[StepRecord] = []
        verdict: Verdict = Verdict.UNDECIDED

        for i, prompt in enumerate(prompts, start=1):
            # Generate outputs (should use deterministic decoding for reproducibility)
            ref_out = ref_generate(prompt)
            cand_out = cand_generate(prompt)

            # Compute distance score in [0,1]
            score = self.score_fn(ref_out, cand_out)
            w.push(score)

            # Compute anytime-valid confidence bounds
            delta_n = spending_schedule(self.params.alpha, w.n)
            h = eb_halfwidth(w.var, w.n, delta_n)

            # Check stopping conditions (respect minimum sample requirement)
            if w.n >= self.params.n_min:
                if self._same_rule(w.mean, h):
                    verdict = Verdict.SAME
                elif self._diff_rule(w.mean, h):
                    verdict = Verdict.DIFFERENT
                else:
                    verdict = Verdict.UNDECIDED
            else:
                verdict = Verdict.UNDECIDED

            # Record step for audit trail
            steps.append(
                StepRecord(
                    index=i,
                    prompt=prompt,
                    ref_output=ref_out,
                    cand_output=cand_out,
                    score=score,
                    mean=w.mean,
                    var=w.var,
                    halfwidth=h,
                    delta_n=delta_n,
                    verdict_so_far=verdict.value,
                )
            )

            # Stop if decision reached or maximum samples exceeded
            if verdict in (Verdict.SAME, Verdict.DIFFERENT) or w.n >= self.params.n_max:
                break

        return RunResult(
            verdict=verdict,
            steps=steps,
            n_used=len(steps),
            params={
                **asdict(self.params),
                "score_function": self.score_fn.__name__,
                "note": "REV verification with Empirical-Bernstein confidence sequences",
            },
        )