"""
Parallel Verification Pipeline for REV+HBT.

This module implements parallel execution of REV sequential tests and HBT consensus
validation using thread pools and process pools for optimal performance.
"""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Dict, List, Tuple, Optional, Any, Generator, Union
from dataclasses import dataclass, field
import numpy as np
import torch
import queue
import time
import logging
from collections import deque, defaultdict
from contextlib import contextmanager
import pickle

from .segment_runner import SegmentRunner, SegmentConfig, KVCache
from ..rev_pipeline import REVPipeline, Segment
from ..consensus.byzantine import ConsensusNetwork, ConsensusResult, ByzantineValidator
from ..verifier.decision_aggregator import (
    DecisionAggregator, 
    AggregatedDecision,
    ChallengeResult,
    AggregationMethod
)
from ..verifier.decision import Verdict, StepRecord
from ..core.sequential import SequentialState, sequential_verify
from ..crypto.merkle import build_merkle_tree, leaf_bytes
from ..hdc.encoder import UnifiedHDCEncoder, HypervectorConfig
from ..challenges.prompt_generator import DeterministicPromptGenerator

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel verification."""
    
    thread_pool_size: int = 4  # For REV segment processing
    process_pool_size: int = 8  # For HBT consensus validation
    batch_size: int = 16  # Challenges per batch
    segment_queue_size: int = 100  # Max segments in queue
    consensus_batch_size: int = 4  # Segments per consensus round
    enable_gpu: bool = True
    memory_limit_gb: float = 8.0
    timeout_seconds: float = 300.0
    use_shared_memory: bool = True  # For inter-process communication


@dataclass
class VerificationTask:
    """Task for parallel verification."""
    
    task_id: str
    challenge: str
    model_a_id: str
    model_b_id: str
    task_type: str  # "sequential" or "consensus"
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelResult:
    """Result from parallel verification."""
    
    task_id: str
    verdict: Verdict
    confidence: float
    execution_time: float
    task_type: str
    sequential_result: Optional[Dict[str, Any]] = None
    consensus_result: Optional[ConsensusResult] = None
    error: Optional[str] = None


class ThreadSafeResourceManager:
    """
    Manages thread-safe access to shared resources.
    
    Provides synchronized access to merkle trees, segment buffers,
    and other shared data structures.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._segment_buffers: Dict[str, deque] = {}
        self._merkle_trees: Dict[str, Any] = {}
        self._cache_data: Dict[str, Any] = {}
        
    @contextmanager
    def segment_buffer_access(self, buffer_id: str):
        """Thread-safe access to segment buffer."""
        with self._lock:
            if buffer_id not in self._segment_buffers:
                self._segment_buffers[buffer_id] = deque(maxlen=4)
            yield self._segment_buffers[buffer_id]
    
    @contextmanager
    def merkle_tree_access(self, tree_id: str):
        """Thread-safe access to merkle tree."""
        with self._lock:
            if tree_id not in self._merkle_trees:
                self._merkle_trees[tree_id] = {}
            yield self._merkle_trees[tree_id]
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Thread-safe cache retrieval."""
        with self._lock:
            return self._cache_data.get(key)
    
    def set_cache(self, key: str, value: Any):
        """Thread-safe cache update."""
        with self._lock:
            self._cache_data[key] = value
    
    def clear_buffer(self, buffer_id: str):
        """Clear a specific segment buffer."""
        with self._lock:
            if buffer_id in self._segment_buffers:
                self._segment_buffers[buffer_id].clear()


class ParallelVerificationPipeline:
    """
    Parallel verification pipeline combining REV and HBT approaches.
    
    Uses thread pools for REV segment processing and process pools for
    HBT Byzantine consensus validation to maximize throughput.
    """
    
    def __init__(
        self,
        config: Optional[ParallelConfig] = None,
        segment_runner: Optional[SegmentRunner] = None,
        rev_pipeline: Optional[REVPipeline] = None,
        consensus_network: Optional[ConsensusNetwork] = None
    ):
        """
        Initialize parallel verification pipeline.
        
        Args:
            config: Parallel execution configuration
            segment_runner: Segment execution runner
            rev_pipeline: REV pipeline for sequential testing
            consensus_network: Byzantine consensus network
        """
        self.config = config or ParallelConfig()
        
        # Initialize components
        self.segment_runner = segment_runner or SegmentRunner(
            SegmentConfig(max_memory_gb=self.config.memory_limit_gb)
        )
        
        self.rev_pipeline = rev_pipeline or REVPipeline(
            segment_size=512,
            buffer_size=4
        )
        
        self.consensus_network = consensus_network or ConsensusNetwork(
            num_validators=4,
            batch_size=self.config.consensus_batch_size
        )
        
        # Initialize thread and process pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="rev_worker"
        )
        
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.process_pool_size,
            mp_context=mp.get_context('spawn')
        )
        
        # Resource management
        self.resource_manager = ThreadSafeResourceManager()
        
        # Task queues
        self.sequential_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.consensus_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # Results tracking
        self.results: Dict[str, ParallelResult] = {}
        self.results_lock = threading.Lock()
        
        # Decision aggregator
        self.aggregator = DecisionAggregator(
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            similarity_metric="hypervector"
        )
        
        # HDC encoder for signatures
        self.encoder = UnifiedHDCEncoder(
            HypervectorConfig(
                dimension=10000,
                encoding_mode="hybrid"
            )
        )
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0
        }
        
        # Prompt generator for challenges
        self.prompt_generator = None  # Initialize when needed
    
    async def verify_parallel(
        self,
        model_a,
        model_b,
        challenges: List[str],
        tokenizer_a=None,
        tokenizer_b=None,
        batch_size: Optional[int] = None
    ) -> AggregatedDecision:
        """
        Perform parallel verification of models.
        
        Args:
            model_a: First model to verify
            model_b: Second model to verify
            challenges: List of challenge prompts
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            batch_size: Optional batch size override
            
        Returns:
            Aggregated decision across all challenges
        """
        batch_size = batch_size or self.config.batch_size
        
        # Create verification tasks
        tasks = []
        for i, challenge_batch in enumerate(self._batch_challenges(challenges, batch_size)):
            for j, challenge in enumerate(challenge_batch):
                task_id = f"task_{i}_{j}"
                
                # Create sequential task for REV
                seq_task = VerificationTask(
                    task_id=f"{task_id}_seq",
                    challenge=challenge,
                    model_a_id="model_a",
                    model_b_id="model_b",
                    task_type="sequential",
                    priority=i
                )
                tasks.append(seq_task)
                
                # Create consensus task for HBT
                cons_task = VerificationTask(
                    task_id=f"{task_id}_cons",
                    challenge=challenge,
                    model_a_id="model_a",
                    model_b_id="model_b",
                    task_type="consensus",
                    priority=i + 100  # Lower priority than sequential
                )
                tasks.append(cons_task)
        
        # Submit tasks to appropriate pools
        futures = []
        for task in tasks:
            if task.task_type == "sequential":
                # Submit to thread pool for REV processing
                future = asyncio.create_task(
                    self._run_sequential_task(task, model_a, model_b, tokenizer_a, tokenizer_b)
                )
            else:
                # Submit to process pool for HBT consensus
                future = asyncio.create_task(
                    self._run_consensus_task(task, model_a, model_b, tokenizer_a, tokenizer_b)
                )
            futures.append((task.task_id, future))
        
        # Gather results
        results = await asyncio.gather(*[f for _, f in futures], return_exceptions=True)
        
        # Process results
        task_results = {}
        for (task_id, _), result in zip(futures, results):
            if isinstance(result, Exception):
                logger.error(f"Task {task_id} failed: {result}")
                task_results[task_id] = ParallelResult(
                    task_id=task_id,
                    verdict=Verdict.UNDECIDED,
                    confidence=0.0,
                    execution_time=0.0,
                    task_type="error",
                    error=str(result)
                )
            else:
                task_results[task_id] = result
        
        # Merge parallel results
        aggregated = self.merge_parallel_results(task_results, challenges)
        
        return aggregated
    
    async def _run_sequential_task(
        self,
        task: VerificationTask,
        model_a,
        model_b,
        tokenizer_a,
        tokenizer_b
    ) -> ParallelResult:
        """
        Run REV sequential test in thread pool.
        
        Args:
            task: Verification task
            model_a: First model
            model_b: Second model
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            
        Returns:
            Parallel result with sequential test outcome
        """
        start_time = time.time()
        
        def run_sequential():
            """Execute sequential test in thread."""
            try:
                # Process challenge through REV pipeline
                with self.resource_manager.segment_buffer_access(task.task_id) as buffer:
                    # Generate segments for both models
                    segments_a = self._generate_segments(
                        model_a, task.challenge, tokenizer_a
                    )
                    segments_b = self._generate_segments(
                        model_b, task.challenge, tokenizer_b
                    )
                    
                    # Run sequential test
                    state = SequentialState()
                    for seg_a, seg_b in zip(segments_a, segments_b):
                        # Compute similarity
                        similarity = self._compute_segment_similarity(seg_a, seg_b)
                        distance = 1.0 - similarity
                        
                        # Update sequential state
                        state.update(distance)
                        buffer.append((seg_a, seg_b))
                    
                    # Get verdict
                    if state.n < 10:
                        verdict = Verdict.UNDECIDED
                    elif state.mean < 0.4:
                        verdict = Verdict.SAME
                    elif state.mean > 0.6:
                        verdict = Verdict.DIFFERENT
                    else:
                        verdict = Verdict.UNDECIDED
                    
                    confidence = 1.0 - state.variance if state.n > 1 else 0.0
                    
                    return {
                        'verdict': verdict,
                        'confidence': confidence,
                        'mean_distance': state.mean,
                        'variance': state.variance,
                        'n_samples': state.n
                    }
            
            except Exception as e:
                logger.error(f"Sequential task failed: {e}")
                raise
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.thread_pool, run_sequential)
        
        return ParallelResult(
            task_id=task.task_id,
            verdict=result['verdict'],
            confidence=result['confidence'],
            execution_time=time.time() - start_time,
            task_type="sequential",
            sequential_result=result
        )
    
    async def _run_consensus_task(
        self,
        task: VerificationTask,
        model_a,
        model_b,
        tokenizer_a,
        tokenizer_b
    ) -> ParallelResult:
        """
        Run HBT Byzantine consensus in process pool.
        
        Args:
            task: Verification task
            model_a: First model
            model_b: Second model
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            
        Returns:
            Parallel result with consensus outcome
        """
        start_time = time.time()
        
        def run_consensus():
            """Execute consensus validation in process."""
            try:
                # Generate segments
                segments_a = list(self._generate_segments(
                    model_a, task.challenge, tokenizer_a
                ))[:self.config.consensus_batch_size]
                
                segments_b = list(self._generate_segments(
                    model_b, task.challenge, tokenizer_b
                ))[:self.config.consensus_batch_size]
                
                # Prepare segment buffer for consensus
                segment_buffer = deque(maxlen=4)
                for seg_a, seg_b in zip(segments_a, segments_b):
                    # Combine segment info for consensus
                    combined_seg = Segment(
                        segment_id=seg_a.segment_id,
                        tokens=seg_a.tokens,
                        start_idx=seg_a.start_idx,
                        end_idx=seg_a.end_idx,
                        signatures={
                            'model_a': seg_a.signatures,
                            'model_b': seg_b.signatures
                        }
                    )
                    segment_buffer.append(combined_seg)
                
                # Run Byzantine consensus
                consensus_result = self.consensus_network.validate_segments(
                    segment_buffer
                )
                
                # Determine verdict from consensus
                if consensus_result.consensus_reached:
                    if consensus_result.behavioral_agreement > 0.7:
                        verdict = Verdict.SAME
                    elif consensus_result.behavioral_agreement < 0.3:
                        verdict = Verdict.DIFFERENT
                    else:
                        verdict = Verdict.UNDECIDED
                else:
                    verdict = Verdict.UNDECIDED
                
                return consensus_result, verdict
            
            except Exception as e:
                logger.error(f"Consensus task failed: {e}")
                raise
        
        # Run in process pool
        loop = asyncio.get_event_loop()
        consensus_result, verdict = await loop.run_in_executor(
            self.process_pool, run_consensus
        )
        
        return ParallelResult(
            task_id=task.task_id,
            verdict=verdict,
            confidence=consensus_result.confidence_score,
            execution_time=time.time() - start_time,
            task_type="consensus",
            consensus_result=consensus_result
        )
    
    def merge_parallel_results(
        self,
        task_results: Dict[str, ParallelResult],
        challenges: List[str]
    ) -> AggregatedDecision:
        """
        Merge results from parallel REV and HBT execution.
        
        Args:
            task_results: Dictionary of task results
            challenges: Original challenge list
            
        Returns:
            Aggregated decision combining all results
        """
        # Separate sequential and consensus results
        sequential_results = []
        consensus_results = []
        
        for task_id, result in task_results.items():
            if result.task_type == "sequential":
                sequential_results.append(result)
            elif result.task_type == "consensus":
                consensus_results.append(result)
        
        # Aggregate sequential results
        seq_verdicts = [r.verdict for r in sequential_results if r.verdict != Verdict.UNDECIDED]
        seq_confidences = [r.confidence for r in sequential_results]
        
        if seq_verdicts:
            # Majority vote for sequential
            seq_same = sum(1 for v in seq_verdicts if v == Verdict.SAME)
            seq_diff = sum(1 for v in seq_verdicts if v == Verdict.DIFFERENT)
            
            if seq_same > seq_diff:
                seq_verdict = Verdict.SAME
            elif seq_diff > seq_same:
                seq_verdict = Verdict.DIFFERENT
            else:
                seq_verdict = Verdict.UNDECIDED
            
            seq_confidence = np.mean(seq_confidences) if seq_confidences else 0.0
        else:
            seq_verdict = Verdict.UNDECIDED
            seq_confidence = 0.0
        
        # Aggregate consensus results
        cons_verdicts = [r.verdict for r in consensus_results if r.verdict != Verdict.UNDECIDED]
        cons_confidences = [r.confidence for r in consensus_results]
        
        if cons_verdicts:
            # Weighted vote for consensus
            cons_same = sum(
                r.confidence for r in consensus_results 
                if r.verdict == Verdict.SAME
            )
            cons_diff = sum(
                r.confidence for r in consensus_results
                if r.verdict == Verdict.DIFFERENT
            )
            
            if cons_same > cons_diff:
                cons_verdict = Verdict.SAME
            elif cons_diff > cons_same:
                cons_verdict = Verdict.DIFFERENT
            else:
                cons_verdict = Verdict.UNDECIDED
            
            cons_confidence = np.mean(cons_confidences) if cons_confidences else 0.0
        else:
            cons_verdict = Verdict.UNDECIDED
            cons_confidence = 0.0
        
        # Combine REV and HBT results
        if seq_verdict == cons_verdict and seq_verdict != Verdict.UNDECIDED:
            # Strong agreement
            final_verdict = seq_verdict
            final_confidence = 0.7 * seq_confidence + 0.3 * cons_confidence
        elif seq_verdict != Verdict.UNDECIDED and cons_verdict == Verdict.UNDECIDED:
            # Trust sequential
            final_verdict = seq_verdict
            final_confidence = seq_confidence * 0.8
        elif cons_verdict != Verdict.UNDECIDED and seq_verdict == Verdict.UNDECIDED:
            # Trust consensus
            final_verdict = cons_verdict
            final_confidence = cons_confidence * 0.8
        elif seq_verdict != cons_verdict:
            # Disagreement - need more analysis
            if seq_confidence > cons_confidence + 0.2:
                final_verdict = seq_verdict
            elif cons_confidence > seq_confidence + 0.2:
                final_verdict = cons_verdict
            else:
                final_verdict = Verdict.UNDECIDED
            final_confidence = abs(seq_confidence - cons_confidence)
        else:
            final_verdict = Verdict.UNDECIDED
            final_confidence = 0.0
        
        # Create challenge results
        challenge_results = []
        for i, challenge in enumerate(challenges[:len(task_results) // 2]):
            # Find corresponding results
            seq_result = next(
                (r for r in sequential_results if f"task_{i}" in r.task_id),
                None
            )
            cons_result = next(
                (r for r in consensus_results if f"task_{i}" in r.task_id),
                None
            )
            
            if seq_result and seq_result.sequential_result:
                distance = seq_result.sequential_result.get('mean_distance', 0.5)
            else:
                distance = 0.5
            
            challenge_results.append(ChallengeResult(
                challenge_id=f"challenge_{i}",
                prompt=challenge[:100],
                model_a_response="[response_a]",
                model_b_response="[response_b]",
                distance_score=distance,
                equality_indicator=distance < 0.4,
                hypervector_similarity=1.0 - distance
            ))
        
        # Calculate statistics
        distances = [cr.distance_score for cr in challenge_results]
        mean_distance = np.mean(distances) if distances else 0.5
        std_distance = np.std(distances) if distances else 0.0
        
        return AggregatedDecision(
            verdict=final_verdict,
            confidence=final_confidence,
            total_challenges=len(challenges),
            equal_challenges=sum(1 for cr in challenge_results if cr.equality_indicator),
            divergent_challenges=sum(1 for cr in challenge_results if not cr.equality_indicator),
            mean_distance=mean_distance,
            std_distance=std_distance,
            first_divergence=None,  # Could be computed if needed
            sequential_result=None,  # Could include full SPRT result
            per_challenge_results=challenge_results
        )
    
    def _generate_segments(
        self,
        model,
        challenge: str,
        tokenizer
    ) -> Generator[Segment, None, None]:
        """
        Generate segments for a model and challenge.
        
        Args:
            model: Model to process
            challenge: Challenge prompt
            tokenizer: Tokenizer for the model
            
        Yields:
            Segments with signatures
        """
        # Use REV pipeline to generate segments
        result = self.rev_pipeline.process_challenge(model, challenge, tokenizer)
        
        for seg_data in result.get('segment_signatures', []):
            segment = Segment(
                segment_id=seg_data.get('id', 0),
                tokens=[],  # Already processed
                start_idx=0,
                end_idx=0,
                signatures=seg_data.get('signatures', {})
            )
            yield segment
    
    def _compute_segment_similarity(self, seg_a: Segment, seg_b: Segment) -> float:
        """
        Compute similarity between two segments.
        
        Args:
            seg_a: First segment
            seg_b: Second segment
            
        Returns:
            Similarity score in [0, 1]
        """
        if not seg_a.signatures or not seg_b.signatures:
            return 0.5
        
        similarities = []
        for key in seg_a.signatures:
            if key in seg_b.signatures:
                sig_a = seg_a.signatures[key]
                sig_b = seg_b.signatures[key]
                
                if isinstance(sig_a, np.ndarray) and isinstance(sig_b, np.ndarray):
                    # Compute cosine similarity
                    sim = np.dot(sig_a.flatten(), sig_b.flatten()) / (
                        np.linalg.norm(sig_a) * np.linalg.norm(sig_b) + 1e-8
                    )
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _batch_challenges(
        self,
        challenges: List[str],
        batch_size: int
    ) -> Generator[List[str], None, None]:
        """
        Batch challenges for parallel processing.
        
        Args:
            challenges: List of challenges
            batch_size: Size of each batch
            
        Yields:
            Batches of challenges
        """
        for i in range(0, len(challenges), batch_size):
            yield challenges[i:i + batch_size]
    
    def generate_challenges_batch(
        self,
        model_a_id: str,
        model_b_id: str,
        n_challenges: int,
        master_key: Optional[bytes] = None,
        seed: int = 42
    ) -> List[str]:
        """
        Generate batch of challenges using prompt generator.
        
        Args:
            model_a_id: Identifier for first model
            model_b_id: Identifier for second model
            n_challenges: Number of challenges to generate
            master_key: Master key for deterministic generation
            seed: Random seed
            
        Returns:
            List of challenge prompts
        """
        if self.prompt_generator is None:
            # Initialize prompt generator
            if master_key is None:
                master_key = b"default_rev_verification_key"
            self.prompt_generator = DeterministicPromptGenerator(master_key)
        
        # Generate challenges
        challenges_data = self.prompt_generator.generate_challenges(
            ref_model_id=model_a_id,
            cand_model_id=model_b_id,
            n=n_challenges,
            namespace="parallel_verification",
            seed=seed
        )
        
        # Extract prompts
        return [c.get('prompt', '') for c in challenges_data]
    
    async def run_complete_verification(
        self,
        model_a,
        model_b,
        n_challenges: int = 100,
        tokenizer_a=None,
        tokenizer_b=None,
        model_a_id: str = "model_a",
        model_b_id: str = "model_b",
        master_key: Optional[bytes] = None,
        seed: int = 42
    ) -> AggregatedDecision:
        """
        Run complete parallel verification with generated challenges.
        
        This is a convenience method that:
        1. Generates challenges using the prompt generator
        2. Runs parallel verification (REV + HBT)
        3. Returns aggregated decision
        
        Args:
            model_a: First model to verify
            model_b: Second model to verify
            n_challenges: Number of challenges to generate
            tokenizer_a: Tokenizer for model A
            tokenizer_b: Tokenizer for model B
            model_a_id: Identifier for model A
            model_b_id: Identifier for model B
            master_key: Master key for deterministic generation
            seed: Random seed
            
        Returns:
            Aggregated decision across all challenges
        """
        # Generate challenges
        logger.info(f"Generating {n_challenges} challenges for verification")
        challenges = self.generate_challenges_batch(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            n_challenges=n_challenges,
            master_key=master_key,
            seed=seed
        )
        
        # Update statistics
        self.stats['tasks_submitted'] = n_challenges * 2  # Sequential + consensus
        
        # Run parallel verification
        logger.info(f"Starting parallel verification with {self.config.thread_pool_size} threads and {self.config.process_pool_size} processes")
        start_time = time.time()
        
        decision = await self.verify_parallel(
            model_a=model_a,
            model_b=model_b,
            challenges=challenges,
            tokenizer_a=tokenizer_a,
            tokenizer_b=tokenizer_b
        )
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['tasks_completed'] = sum(
            1 for r in self.results.values() 
            if r.verdict != Verdict.UNDECIDED
        )
        self.stats['avg_execution_time'] = total_time / max(1, self.stats['tasks_completed'])
        
        logger.info(f"Verification complete: {decision.verdict.value} with confidence {decision.confidence:.3f}")
        logger.info(f"Total time: {total_time:.2f}s, Average per task: {self.stats['avg_execution_time']:.2f}s")
        
        return decision
    
    async def shutdown(self):
        """Shutdown parallel pools and clean up resources."""
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        # Clear resources
        self.resource_manager._segment_buffers.clear()
        self.resource_manager._merkle_trees.clear()
        self.resource_manager._cache_data.clear()
        
        logger.info("Parallel verification pipeline shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'tasks_submitted': self.stats['tasks_submitted'],
            'tasks_completed': self.stats['tasks_completed'],
            'tasks_failed': self.stats['tasks_failed'],
            'avg_execution_time': self.stats['avg_execution_time'],
            'thread_pool_size': self.config.thread_pool_size,
            'process_pool_size': self.config.process_pool_size,
            'active_threads': self.thread_pool._threads.__len__() if hasattr(self.thread_pool, '_threads') else 0,
            'queued_tasks': self.sequential_queue.qsize() + self.consensus_queue.qsize()
        }