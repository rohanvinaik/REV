"""
Parallel Execution Engine for REV System

Enables parallel processing of multiple prompts/models while respecting memory limits.
"""

import os
import time
import psutil
import queue
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import logging

from src.utils.error_handling import REVException, CircuitBreaker
from src.utils.logging_config import LoggingConfig


@dataclass
class MemoryConfig:
    """Memory configuration for parallel execution."""
    total_limit_gb: float = 36.0  # Total memory limit
    per_process_gb: float = 2.0    # Memory per process
    buffer_gb: float = 2.0          # Safety buffer
    
    @property
    def max_processes(self) -> int:
        """Calculate maximum number of parallel processes."""
        available = self.total_limit_gb - self.buffer_gb
        return max(1, int(available / self.per_process_gb))


@dataclass
class ProcessTask:
    """Task for parallel processing."""
    task_id: str
    task_type: str  # 'prompt', 'model', 'fingerprint'
    payload: Dict[str, Any]
    priority: int = 0
    memory_estimate_gb: float = 2.0
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        return self.priority > other.priority  # Higher priority first


class MemoryMonitor:
    """Monitor and manage memory usage across processes."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.lock = threading.Lock()
        self.allocated_gb = 0.0
        self.process_memory = {}  # pid -> memory_gb
        self.logger = logging.getLogger(__name__)
    
    def can_allocate(self, memory_gb: float) -> bool:
        """Check if memory can be allocated."""
        with self.lock:
            system_mem = psutil.virtual_memory()
            system_available_gb = system_mem.available / (1024**3)
            
            # Check both configured limit and system availability
            within_limit = (self.allocated_gb + memory_gb) <= self.config.total_limit_gb
            system_has_space = memory_gb <= (system_available_gb - self.config.buffer_gb)
            
            return within_limit and system_has_space
    
    def allocate(self, pid: int, memory_gb: float) -> bool:
        """Allocate memory for a process."""
        with self.lock:
            if self.can_allocate(memory_gb):
                self.allocated_gb += memory_gb
                self.process_memory[pid] = memory_gb
                self.logger.info(f"Allocated {memory_gb:.1f}GB for process {pid}. "
                               f"Total: {self.allocated_gb:.1f}/{self.config.total_limit_gb:.1f}GB")
                return True
            return False
    
    def release(self, pid: int):
        """Release memory from a process."""
        with self.lock:
            if pid in self.process_memory:
                released = self.process_memory.pop(pid)
                self.allocated_gb -= released
                self.logger.info(f"Released {released:.1f}GB from process {pid}. "
                               f"Total: {self.allocated_gb:.1f}/{self.config.total_limit_gb:.1f}GB")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        with self.lock:
            system_mem = psutil.virtual_memory()
            return {
                "allocated_gb": self.allocated_gb,
                "limit_gb": self.config.total_limit_gb,
                "available_gb": self.config.total_limit_gb - self.allocated_gb,
                "processes": len(self.process_memory),
                "system_available_gb": system_mem.available / (1024**3),
                "system_percent": system_mem.percent
            }


class ParallelExecutor:
    """
    Parallel execution engine with memory management.
    
    Supports parallel processing of:
    - Multiple prompts on single model
    - Multiple models with same prompts
    - Batch fingerprint generation
    """
    
    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        """
        Initialize parallel executor.
        
        Args:
            memory_config: Memory configuration (default: 36GB total, 2GB per process)
        """
        self.memory_config = memory_config or MemoryConfig()
        self.memory_monitor = MemoryMonitor(self.memory_config)
        
        # Determine optimal worker count
        cpu_count = mp.cpu_count()
        max_by_memory = self.memory_config.max_processes
        self.max_workers = min(cpu_count, max_by_memory)
        
        # Task queue and results
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.errors = {}
        
        # Executor management
        self.executor = None
        self.futures = {}
        self.shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ParallelExecutor initialized: {self.max_workers} workers, "
                        f"{self.memory_config.total_limit_gb}GB memory limit")
    
    def process_prompts_parallel(self, 
                                 model_path: str,
                                 prompts: List[str],
                                 batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple prompts in parallel on a single model.
        
        Args:
            model_path: Path to model
            prompts: List of prompts to process
            batch_size: Prompts per batch (auto-determined if None)
            
        Returns:
            List of results for each prompt
        """
        if batch_size is None:
            # Determine optimal batch size based on memory
            batch_size = max(1, self.max_workers // 2)
        
        self.logger.info(f"Processing {len(prompts)} prompts in parallel "
                        f"(batch_size={batch_size}, workers={self.max_workers})")
        
        # Create tasks
        tasks = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            task = ProcessTask(
                task_id=f"prompt_batch_{i}",
                task_type="prompt",
                payload={
                    "model_path": model_path,
                    "prompts": batch,
                    "batch_index": i
                },
                memory_estimate_gb=self.memory_config.per_process_gb
            )
            tasks.append(task)
        
        # Execute tasks
        results = self._execute_tasks(tasks)
        
        # Combine results in order
        combined = []
        for i in range(0, len(prompts), batch_size):
            batch_result = results.get(f"prompt_batch_{i}", {})
            combined.extend(batch_result.get("responses", []))
        
        return combined
    
    def process_models_parallel(self,
                               model_paths: List[str],
                               prompts: List[str],
                               challenges: int = 30) -> Dict[str, Any]:
        """
        Process multiple models in parallel.
        
        Args:
            model_paths: List of model paths
            prompts: Prompts to test each model
            challenges: Number of challenges per model
            
        Returns:
            Dictionary mapping model path to results
        """
        self.logger.info(f"Processing {len(model_paths)} models in parallel")
        
        # Create tasks for each model
        tasks = []
        for i, model_path in enumerate(model_paths):
            task = ProcessTask(
                task_id=f"model_{i}_{Path(model_path).name}",
                task_type="model",
                payload={
                    "model_path": model_path,
                    "prompts": prompts[:challenges],
                    "challenges": challenges
                },
                priority=i,  # Process in order
                memory_estimate_gb=self._estimate_model_memory(model_path)
            )
            tasks.append(task)
        
        # Execute with memory management
        results = self._execute_tasks(tasks)
        
        # Map results back to model paths
        model_results = {}
        for i, model_path in enumerate(model_paths):
            task_id = f"model_{i}_{Path(model_path).name}"
            model_results[model_path] = results.get(task_id, {"error": "No result"})
        
        return model_results
    
    def generate_fingerprints_parallel(self,
                                      model_paths: List[str],
                                      fingerprint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fingerprints for multiple models in parallel.
        
        Args:
            model_paths: List of model paths
            fingerprint_config: Configuration for fingerprint generation
            
        Returns:
            Dictionary mapping model path to fingerprint
        """
        self.logger.info(f"Generating fingerprints for {len(model_paths)} models in parallel")
        
        tasks = []
        for i, model_path in enumerate(model_paths):
            task = ProcessTask(
                task_id=f"fingerprint_{i}_{Path(model_path).name}",
                task_type="fingerprint",
                payload={
                    "model_path": model_path,
                    "config": fingerprint_config
                },
                memory_estimate_gb=self._estimate_model_memory(model_path)
            )
            tasks.append(task)
        
        results = self._execute_tasks(tasks)
        
        # Map results
        fingerprints = {}
        for i, model_path in enumerate(model_paths):
            task_id = f"fingerprint_{i}_{Path(model_path).name}"
            fingerprints[model_path] = results.get(task_id, {})
        
        return fingerprints
    
    def _execute_tasks(self, tasks: List[ProcessTask]) -> Dict[str, Any]:
        """
        Execute tasks with memory management and parallel processing.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary of task_id -> result
        """
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks as memory becomes available
            future_to_task = {}
            pending_tasks = list(tasks)
            active_futures = set()
            
            while pending_tasks or active_futures:
                # Submit new tasks if memory available
                while pending_tasks and len(active_futures) < self.max_workers:
                    task = pending_tasks[0]
                    
                    # Check memory availability
                    if self.memory_monitor.can_allocate(task.memory_estimate_gb):
                        pending_tasks.pop(0)
                        
                        # Allocate memory and submit task
                        pid = os.getpid()  # Would be child process PID
                        self.memory_monitor.allocate(pid, task.memory_estimate_gb)
                        
                        future = executor.submit(self._process_task, task)
                        future_to_task[future] = task
                        active_futures.add(future)
                        
                        self.logger.info(f"Submitted task {task.task_id} "
                                       f"({len(active_futures)}/{self.max_workers} active)")
                    else:
                        # Wait for memory to be available
                        self.logger.debug(f"Waiting for memory to process {task.task_id}")
                        break
                
                # Wait for some tasks to complete
                if active_futures:
                    done_futures = []
                    for future in as_completed(active_futures, timeout=1.0):
                        try:
                            task = future_to_task[future]
                            result = future.result(timeout=1.0)
                            results[task.task_id] = result
                            
                            self.logger.info(f"Completed task {task.task_id}")
                            
                        except Exception as e:
                            task = future_to_task.get(future)
                            if task:
                                results[task.task_id] = {"error": str(e)}
                                self.logger.error(f"Task {task.task_id} failed: {e}")
                        
                        finally:
                            done_futures.append(future)
                            # Release memory
                            pid = os.getpid()  # Would be child process PID
                            self.memory_monitor.release(pid)
                    
                    # Remove completed futures
                    for future in done_futures:
                        active_futures.discard(future)
                        future_to_task.pop(future, None)
                
                # Brief sleep to prevent busy waiting
                if pending_tasks and not active_futures:
                    time.sleep(0.1)
        
        return results
    
    @staticmethod
    def _process_task(task: ProcessTask) -> Dict[str, Any]:
        """
        Process a single task (runs in separate process).
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Import here to avoid circular dependencies in child process
        from run_rev import REVUnified
        
        try:
            if task.task_type == "prompt":
                # Process prompt batch
                payload = task.payload
                rev = REVUnified(
                    memory_limit_gb=task.memory_estimate_gb,
                    debug=False
                )
                
                responses = []
                for prompt in payload["prompts"]:
                    # Process each prompt
                    response = rev.process_prompt(
                        model_path=payload["model_path"],
                        prompt=prompt
                    )
                    responses.append(response)
                
                rev.cleanup()
                return {"responses": responses, "success": True}
            
            elif task.task_type == "model":
                # Process entire model
                payload = task.payload
                rev = REVUnified(
                    memory_limit_gb=task.memory_estimate_gb,
                    debug=False
                )
                
                result = rev.process_model(
                    model_path=payload["model_path"],
                    challenges=payload.get("challenges", 30)
                )
                
                rev.cleanup()
                return result
            
            elif task.task_type == "fingerprint":
                # Generate fingerprint
                payload = task.payload
                rev = REVUnified(
                    memory_limit_gb=task.memory_estimate_gb,
                    unified_fingerprints=True,
                    **payload.get("config", {})
                )
                
                result = rev.generate_fingerprint(
                    model_path=payload["model_path"]
                )
                
                rev.cleanup()
                return result
            
            else:
                return {"error": f"Unknown task type: {task.task_type}"}
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _estimate_model_memory(self, model_path: str) -> float:
        """
        Estimate memory requirements for a model.
        
        Args:
            model_path: Path to model
            
        Returns:
            Estimated memory in GB
        """
        try:
            model_path = Path(model_path)
            
            # Check model size on disk
            if model_path.exists():
                size_bytes = sum(f.stat().st_size for f in model_path.rglob('*'))
                size_gb = size_bytes / (1024**3)
                
                # Heuristic: memory ≈ 1.5x model size for inference
                estimated = min(size_gb * 1.5, self.memory_config.per_process_gb * 2)
                
                return estimated
            
        except Exception:
            pass
        
        # Default estimate
        return self.memory_config.per_process_gb
    
    def get_status(self) -> Dict[str, Any]:
        """Get current executor status."""
        return {
            "max_workers": self.max_workers,
            "memory_status": self.memory_monitor.get_status(),
            "active_tasks": len(self.futures),
            "completed_tasks": len(self.results),
            "failed_tasks": len(self.errors)
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.shutdown_event.set()
        if self.executor:
            self.executor.shutdown(wait=wait)
        self.logger.info("ParallelExecutor shutdown complete")


class AdaptiveParallelExecutor(ParallelExecutor):
    """
    Adaptive parallel executor that adjusts parallelism based on system load.
    """
    
    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        super().__init__(memory_config)
        
        self.performance_history = []
        self.optimal_workers = self.max_workers
        self.adaptation_interval = 10  # Adjust every N tasks
        
    def adapt_workers(self):
        """Adapt number of workers based on performance."""
        if len(self.performance_history) < self.adaptation_interval:
            return
        
        # Calculate recent performance
        recent_perf = self.performance_history[-self.adaptation_interval:]
        avg_time = np.mean([p["duration"] for p in recent_perf])
        avg_memory = np.mean([p["memory_peak"] for p in recent_perf])
        
        # Adjust workers
        if avg_memory > self.memory_config.total_limit_gb * 0.9:
            # Reduce workers if memory pressure
            self.optimal_workers = max(1, self.optimal_workers - 1)
            self.logger.info(f"Reducing workers to {self.optimal_workers} due to memory pressure")
            
        elif avg_memory < self.memory_config.total_limit_gb * 0.5:
            # Increase workers if memory available
            max_possible = self.memory_config.max_processes
            self.optimal_workers = min(max_possible, self.optimal_workers + 1)
            self.logger.info(f"Increasing workers to {self.optimal_workers} due to available memory")
        
        # Update max_workers
        self.max_workers = self.optimal_workers
    
    def record_performance(self, task_id: str, duration: float, memory_peak: float):
        """Record task performance for adaptation."""
        self.performance_history.append({
            "task_id": task_id,
            "duration": duration,
            "memory_peak": memory_peak,
            "timestamp": time.time()
        })
        
        # Trigger adaptation
        if len(self.performance_history) % self.adaptation_interval == 0:
            self.adapt_workers()


class BatchProcessor:
    """
    High-level batch processing interface.
    """
    
    def __init__(self, memory_limit_gb: float = 36.0):
        """
        Initialize batch processor.
        
        Args:
            memory_limit_gb: Total memory limit for parallel processing
        """
        self.memory_config = MemoryConfig(total_limit_gb=memory_limit_gb)
        self.executor = AdaptiveParallelExecutor(self.memory_config)
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self,
                     model_paths: List[str],
                     prompts: List[str],
                     mode: str = "cross_product",
                     **kwargs) -> Dict[str, Any]:
        """
        Process a batch of models and prompts.
        
        Args:
            model_paths: List of model paths
            prompts: List of prompts
            mode: Processing mode:
                  - 'cross_product': Each model with all prompts
                  - 'paired': model[i] with prompts[i]
                  - 'broadcast': All models with same prompts
            **kwargs: Additional arguments
            
        Returns:
            Batch processing results
        """
        self.logger.info(f"Processing batch: {len(model_paths)} models, "
                        f"{len(prompts)} prompts, mode={mode}")
        
        start_time = time.time()
        
        if mode == "cross_product":
            # Each model processes all prompts
            results = {}
            for model_path in model_paths:
                model_results = self.executor.process_prompts_parallel(
                    model_path, prompts, **kwargs
                )
                results[model_path] = model_results
                
        elif mode == "paired":
            # Pair each model with corresponding prompts
            if len(model_paths) != len(prompts):
                raise ValueError("Paired mode requires equal number of models and prompts")
            
            results = {}
            for model_path, prompt in zip(model_paths, prompts):
                model_results = self.executor.process_prompts_parallel(
                    model_path, [prompt], **kwargs
                )
                results[model_path] = model_results[0] if model_results else None
                
        elif mode == "broadcast":
            # All models process the same prompts
            results = self.executor.process_models_parallel(
                model_paths, prompts, **kwargs
            )
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        duration = time.time() - start_time
        
        return {
            "results": results,
            "statistics": {
                "models_processed": len(model_paths),
                "prompts_processed": len(prompts),
                "duration_seconds": duration,
                "throughput": len(model_paths) * len(prompts) / duration,
                "executor_status": self.executor.get_status()
            }
        }
    
    def shutdown(self):
        """Shutdown batch processor."""
        self.executor.shutdown()


# Convenience functions
def parallel_process_prompts(model_path: str,
                            prompts: List[str],
                            memory_limit_gb: float = 36.0,
                            **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function for parallel prompt processing.
    
    Args:
        model_path: Path to model
        prompts: List of prompts
        memory_limit_gb: Memory limit
        **kwargs: Additional arguments
        
    Returns:
        List of results
    """
    config = MemoryConfig(total_limit_gb=memory_limit_gb)
    executor = ParallelExecutor(config)
    
    try:
        return executor.process_prompts_parallel(model_path, prompts, **kwargs)
    finally:
        executor.shutdown()


def parallel_process_models(model_paths: List[str],
                          challenges: int = 30,
                          memory_limit_gb: float = 36.0,
                          **kwargs) -> Dict[str, Any]:
    """
    Convenience function for parallel model processing.
    
    Args:
        model_paths: List of model paths
        challenges: Number of challenges per model
        memory_limit_gb: Memory limit
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of results
    """
    config = MemoryConfig(total_limit_gb=memory_limit_gb)
    executor = ParallelExecutor(config)
    
    # Generate prompts for challenges
    from src.orchestration.prompt_orchestrator import PromptOrchestrator
    orchestrator = PromptOrchestrator()
    prompts = orchestrator.generate_prompts(n=challenges)
    
    try:
        return executor.process_models_parallel(model_paths, prompts, challenges, **kwargs)
    finally:
        executor.shutdown()