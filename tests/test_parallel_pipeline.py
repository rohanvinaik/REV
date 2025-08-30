"""
Tests for the advanced parallel execution pipeline.

Tests all components including work stealing, memory management,
GPU/CPU hybrid execution, and optimization features.
"""

import pytest
import asyncio
import threading
import time
import pickle
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.executor.parallel_pipeline import (
    ParallelPipeline,
    PipelineConfig,
    MemoryConfig,
    GPUConfig,
    OptimizationConfig,
    SegmentTask,
    TaskResult,
    TaskPriority,
    TaskStatus,
    ExecutionMode,
    MemoryManager,
    WorkStealingQueue,
    ResourceMonitor,
    ProgressTracker,
    BatchProcessor,
    create_pipeline,
    process_segments_parallel
)
from src.executor.segment_runner import SegmentConfig
from src.rev_pipeline import Segment


class TestMemoryManager:
    """Test memory management functionality."""
    
    @pytest.fixture
    def memory_config(self):
        return MemoryConfig(
            max_memory_gb=4.0,
            memory_pool_size_mb=64
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        return MemoryManager(memory_config)
    
    def test_memory_pool_initialization(self, memory_manager):
        """Test memory pool is properly initialized."""
        assert len(memory_manager.memory_pool) == 64  # 64MB / 1MB chunks
        assert all(len(chunk) == 1024*1024 for chunk in memory_manager.memory_pool)
    
    def test_memory_allocation_and_deallocation(self, memory_manager):
        """Test memory allocation and deallocation."""
        # Allocate 2MB
        memory_view = memory_manager.allocate_memory(2 * 1024 * 1024)
        assert memory_view is not None
        assert len(memory_view) == 2 * 1024 * 1024
        assert len(memory_manager.memory_pool) == 62  # 2 chunks allocated
        
        # Deallocate
        memory_manager.deallocate_memory(memory_view)
        assert len(memory_manager.memory_pool) == 64  # Back to original
    
    def test_memory_usage_stats(self, memory_manager):
        """Test memory usage statistics."""
        stats = memory_manager.get_memory_usage()
        
        assert 'rss_gb' in stats
        assert 'vms_gb' in stats
        assert 'percent' in stats
        assert 'available_gb' in stats
        assert 'pool_available_mb' in stats
        
        assert stats['pool_available_mb'] == 64
        assert stats['rss_gb'] >= 0
        assert stats['percent'] >= 0
    
    def test_activation_checkpointing(self, memory_manager):
        """Test activation checkpointing functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Create mock activation tensor
        activations = torch.randn(100, 100)
        task_id = "test_task_123"
        
        # Checkpoint activations
        checkpoint_id = memory_manager.checkpoint_activations(task_id, activations)
        
        assert checkpoint_id.startswith("checkpoint_test_task_123")
        assert checkpoint_id in memory_manager.checkpointed_activations
        
        # Restore activations
        restored = memory_manager.restore_activations(checkpoint_id)
        assert restored is not None
        assert torch.allclose(activations.cpu(), restored, atol=1e-6)
        
        # Cleanup
        memory_manager.cleanup_checkpoints([task_id])
        assert checkpoint_id not in memory_manager.checkpointed_activations
    
    def test_shared_kv_cache(self, memory_manager):
        """Test shared KV cache functionality."""
        from src.executor.segment_runner import KVCache
        
        cache_key = "model_abc_layer_5"
        kv_cache = Mock(spec=KVCache)
        
        # Set cache
        memory_manager.set_shared_kv_cache(cache_key, kv_cache)
        
        # Get cache
        retrieved = memory_manager.get_shared_kv_cache(cache_key)
        assert retrieved is kv_cache
        
        # Non-existent cache
        assert memory_manager.get_shared_kv_cache("non_existent") is None


class TestWorkStealingQueue:
    """Test work stealing queue functionality."""
    
    @pytest.fixture
    def work_queue(self):
        return WorkStealingQueue("worker_1")
    
    @pytest.fixture
    def sample_tasks(self):
        tasks = []
        for i in range(5):
            task = SegmentTask(
                task_id=f"task_{i}",
                segment_id=f"seg_{i}",
                model_id="test_model",
                prompt="test prompt",
                segment_data=b"test_data",
                priority=TaskPriority.NORMAL if i < 3 else TaskPriority.LOW
            )
            tasks.append(task)
        return tasks
    
    def test_task_insertion_and_retrieval(self, work_queue, sample_tasks):
        """Test basic task operations."""
        # Insert tasks
        for task in sample_tasks:
            work_queue.put_task(task)
        
        assert work_queue.size() == 5
        
        # Retrieve tasks (should be in priority order)
        retrieved_tasks = []
        while work_queue.size() > 0:
            task = work_queue.get_task()
            if task:
                retrieved_tasks.append(task)
        
        assert len(retrieved_tasks) == 5
        
        # Check priority ordering (NORMAL should come before LOW)
        normal_tasks = [t for t in retrieved_tasks if t.priority == TaskPriority.NORMAL]
        low_tasks = [t for t in retrieved_tasks if t.priority == TaskPriority.LOW]
        
        assert len(normal_tasks) == 3
        assert len(low_tasks) == 2
    
    def test_work_stealing(self, sample_tasks):
        """Test work stealing between queues."""
        queue1 = WorkStealingQueue("worker_1")
        queue2 = WorkStealingQueue("worker_2")
        
        # Add all tasks to queue1
        for task in sample_tasks:
            queue1.put_task(task)
        
        assert queue1.size() == 5
        assert queue2.size() == 0
        
        # Steal task from queue1
        stolen_task = queue1.steal_task()
        
        assert stolen_task is not None
        assert queue1.size() == 4  # One task stolen
        
        # Can't steal from empty queue
        stolen_from_empty = queue2.steal_task()
        assert stolen_from_empty is None
        
        # Can't steal from queue with only one task
        for _ in range(3):
            queue1.get_task()  # Remove tasks until only one left
        
        no_steal = queue1.steal_task()
        assert no_steal is None
        assert queue1.size() == 1


class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    @pytest.fixture
    def config(self):
        return PipelineConfig(stats_collection_interval=0.1)
    
    @pytest.fixture
    def monitor(self, config):
        return ResourceMonitor(config)
    
    def test_stats_collection(self, monitor):
        """Test basic stats collection."""
        stats = monitor.get_current_stats()
        
        required_keys = [
            'timestamp', 'cpu_percent', 'memory_rss_gb',
            'memory_vms_gb', 'memory_percent', 'num_threads', 'open_files'
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
    
    def test_monitoring_lifecycle(self, monitor):
        """Test monitoring start/stop."""
        assert not monitor.monitoring
        
        monitor.start_monitoring()
        assert monitor.monitoring
        assert monitor.monitor_thread is not None
        
        # Let it collect some stats
        time.sleep(0.3)
        
        monitor.stop_monitoring()
        assert not monitor.monitoring
        
        # Should have collected some history
        assert len(monitor.stats_history) > 0
    
    def test_stats_history(self, monitor):
        """Test stats history functionality."""
        monitor.start_monitoring()
        time.sleep(0.3)
        monitor.stop_monitoring()
        
        # Get recent history
        recent_stats = monitor.get_stats_history(seconds=10)
        assert len(recent_stats) > 0
        
        # All stats should be recent
        cutoff = time.time() - 10
        for stats in recent_stats:
            assert stats['timestamp'] > cutoff
    
    def test_overload_detection(self, monitor):
        """Test system overload detection."""
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.cpu_percent.return_value = 95
            mock_process.return_value.memory_percent.return_value = 95
            
            assert monitor.is_system_overloaded()
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.cpu_percent.return_value = 50
            mock_process.return_value.memory_percent.return_value = 50
            
            assert not monitor.is_system_overloaded()


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        return ProgressTracker()
    
    def test_basic_progress_tracking(self, tracker):
        """Test basic progress operations."""
        tracker.set_total_tasks(10)
        
        progress = tracker.get_progress()
        assert progress['total_tasks'] == 10
        assert progress['completed_tasks'] == 0
        assert progress['progress_percent'] == 0
        
        # Start some tasks
        for i in range(3):
            tracker.start_task(f"task_{i}")
        
        progress = tracker.get_progress()
        assert progress['running_tasks'] == 3
        
        # Complete some tasks
        for i in range(2):
            result = TaskResult(
                task_id=f"task_{i}",
                status=TaskStatus.COMPLETED
            )
            tracker.complete_task(f"task_{i}", result)
        
        progress = tracker.get_progress()
        assert progress['completed_tasks'] == 2
        assert progress['running_tasks'] == 1
        assert progress['progress_percent'] == 20.0
        assert progress['success_rate'] == 100.0
    
    def test_progress_callbacks(self, tracker):
        """Test progress callbacks."""
        callback_data = []
        
        def test_callback(progress_data):
            callback_data.append(progress_data)
        
        tracker.register_callback(test_callback)
        tracker.set_total_tasks(5)
        
        # Start a task
        tracker.start_task("task_1")
        assert len(callback_data) > 0
        
        # Complete a task
        result = TaskResult(task_id="task_1", status=TaskStatus.COMPLETED)
        tracker.complete_task("task_1", result)
        
        assert len(callback_data) >= 2
        assert callback_data[-1]['completed_tasks'] == 1


class TestBatchProcessor:
    """Test dynamic batching functionality."""
    
    @pytest.fixture
    def config(self):
        return OptimizationConfig(
            max_batch_size=3,
            batch_timeout_ms=50
        )
    
    @pytest.fixture
    def processor(self, config):
        return BatchProcessor(config)
    
    @pytest.fixture
    def sample_tasks(self):
        return [
            SegmentTask(
                task_id=f"task_{i}",
                segment_id=f"seg_{i}",
                model_id="test_model",
                prompt="test prompt",
                segment_data=b"test_data"
            )
            for i in range(5)
        ]
    
    def test_batch_size_trigger(self, processor, sample_tasks):
        """Test batching triggered by size."""
        # Add tasks one by one
        ready_batch = None
        
        for i, task in enumerate(sample_tasks[:2]):
            ready_batch = processor.add_task(task)
            assert ready_batch is None  # Not ready yet
        
        # Third task should trigger batch
        ready_batch = processor.add_task(sample_tasks[2])
        assert ready_batch is not None
        assert len(ready_batch) == 3
    
    def test_batch_timeout_trigger(self, processor, sample_tasks):
        """Test batching triggered by timeout."""
        # Add one task
        ready_batch = processor.add_task(sample_tasks[0])
        assert ready_batch is None
        
        # Wait for timeout
        time.sleep(0.1)
        
        # Check pending batches
        pending = processor.get_pending_batches()
        assert len(pending) == 1
        assert "default" in pending
        assert len(pending["default"]) == 1
    
    def test_multiple_batch_keys(self, processor, sample_tasks):
        """Test multiple batch keys."""
        # Add tasks to different batches
        batch1 = processor.add_task(sample_tasks[0], "batch_1")
        batch2 = processor.add_task(sample_tasks[1], "batch_2")
        
        assert batch1 is None
        assert batch2 is None
        
        # Add more tasks to each batch
        for task in sample_tasks[2:4]:
            processor.add_task(task, "batch_1")
        
        # batch_1 should be ready now (3 tasks)
        pending = processor.get_pending_batches()
        
        # Should have ready batch_1 and pending batch_2
        total_ready = sum(len(batch) for batch in pending.values())
        assert total_ready >= 3  # batch_1 should be ready


class TestParallelPipeline:
    """Test the main parallel pipeline."""
    
    @pytest.fixture
    def config(self):
        return PipelineConfig(
            thread_pool_size=2,
            process_pool_size=1,
            memory=MemoryConfig(max_memory_gb=2.0),
            gpu=GPUConfig(enable_gpu=False),  # Disable GPU for testing
            optimization=OptimizationConfig(
                enable_dynamic_batching=False,  # Simplify for testing
                enable_prefetching=False
            )
        )
    
    @pytest.fixture
    def pipeline(self, config):
        return ParallelPipeline(config)
    
    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment for testing."""
        segment = Segment(
            segment_id=1,
            tokens=[1, 2, 3, 4, 5],
            start_idx=0,
            end_idx=4,
            signatures={'test': np.array([1, 2, 3, 4])}
        )
        return pickle.dumps(segment)
    
    @pytest.fixture
    def sample_task(self, sample_segment):
        return SegmentTask(
            task_id="test_task_123",
            segment_id="seg_1",
            model_id="test_model",
            prompt="test prompt",
            segment_data=sample_segment,
            priority=TaskPriority.NORMAL
        )
    
    def test_pipeline_lifecycle(self, pipeline):
        """Test pipeline start/shutdown."""
        assert pipeline.thread_pool is None
        
        pipeline.start()
        assert pipeline.thread_pool is not None
        assert pipeline.process_pool is not None
        assert len(pipeline.work_queues) > 0
        assert pipeline.resource_monitor.monitoring
        
        pipeline.shutdown()
        assert pipeline.shutdown_event.is_set()
        assert not pipeline.resource_monitor.monitoring
    
    def test_task_submission(self, pipeline, sample_task):
        """Test task submission and tracking."""
        pipeline.start()
        
        task_id = pipeline.submit_task(sample_task)
        assert task_id == sample_task.task_id
        assert task_id in pipeline.pending_tasks
        assert pipeline.stats['tasks_submitted'] == 1
        
        pipeline.shutdown()
    
    def test_batch_submission(self, pipeline, sample_segment):
        """Test batch task submission."""
        pipeline.start()
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = SegmentTask(
                task_id=f"batch_task_{i}",
                segment_id=f"seg_{i}",
                model_id="test_model",
                prompt=f"prompt_{i}",
                segment_data=sample_segment
            )
            tasks.append(task)
        
        task_ids = pipeline.submit_batch(tasks)
        assert len(task_ids) == 3
        assert pipeline.progress_tracker.total_tasks == 3
        assert pipeline.stats['tasks_submitted'] == 3
        
        pipeline.shutdown()
    
    def test_task_cancellation(self, pipeline, sample_task):
        """Test task cancellation."""
        pipeline.start()
        
        task_id = pipeline.submit_task(sample_task)
        
        # Cancel the task
        success = pipeline.cancel_task(task_id)
        assert success
        assert task_id in pipeline.cancelled_tasks
        assert task_id not in pipeline.pending_tasks
        
        pipeline.shutdown()
    
    @patch('src.executor.segment_runner.SegmentRunner')
    def test_task_execution(self, mock_runner_class, pipeline, sample_task):
        """Test task execution."""
        # Mock the SegmentRunner
        mock_runner = Mock()
        mock_runner.process_segment.return_value = {"result": "test_result"}
        mock_runner_class.return_value = mock_runner
        
        pipeline.start()
        
        # Execute task directly
        result = pipeline.execute_task(sample_task)
        
        assert result.task_id == sample_task.task_id
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"result": "test_result"}
        assert result.execution_time > 0
        
        pipeline.shutdown()
    
    def test_execution_mode_determination(self, pipeline, sample_task):
        """Test execution mode determination."""
        pipeline.start()
        
        # Test auto mode with different system conditions
        mode = pipeline._determine_execution_mode(sample_task)
        assert mode in [ExecutionMode.CPU_ONLY, ExecutionMode.HYBRID]
        
        # Test explicit modes
        sample_task.execution_mode = ExecutionMode.CPU_ONLY
        mode = pipeline._determine_execution_mode(sample_task)
        assert mode == ExecutionMode.CPU_ONLY
        
        pipeline.shutdown()
    
    def test_statistics_collection(self, pipeline):
        """Test statistics collection."""
        pipeline.start()
        
        stats = pipeline.get_statistics()
        
        required_keys = [
            'tasks_submitted', 'tasks_completed', 'tasks_failed',
            'memory_usage', 'system_stats', 'progress',
            'pending_tasks', 'running_tasks', 'work_queue_sizes'
        ]
        
        for key in required_keys:
            assert key in stats
        
        pipeline.shutdown()
    
    def test_detailed_report(self, pipeline):
        """Test detailed reporting."""
        pipeline.start()
        
        report = pipeline.get_detailed_report()
        
        assert 'task_status_breakdown' in report
        assert 'execution_mode_breakdown' in report
        assert 'configuration' in report
        assert 'resource_history' in report
        
        # Check configuration details
        config = report['configuration']
        assert 'thread_pool_size' in config
        assert 'memory_config' in config
        assert 'optimization_config' in config
        
        pipeline.shutdown()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_pipeline_function(self):
        """Test pipeline creation utility."""
        pipeline = create_pipeline(
            thread_pool_size=4,
            process_pool_size=2,
            memory_gb=4.0,
            enable_gpu=False,
            enable_optimizations=True
        )
        
        assert isinstance(pipeline, ParallelPipeline)
        assert pipeline.config.thread_pool_size == 4
        assert pipeline.config.process_pool_size == 2
        assert pipeline.config.memory.max_memory_gb == 4.0
        assert pipeline.config.gpu.enable_gpu == False
        assert pipeline.config.optimization.enable_dynamic_batching == True
    
    @pytest.mark.asyncio
    @patch('src.executor.segment_runner.SegmentRunner')
    async def test_process_segments_parallel_function(self, mock_runner_class):
        """Test parallel segment processing utility."""
        # Mock the SegmentRunner
        mock_runner = Mock()
        mock_runner.process_segment.return_value = {"result": "processed"}
        mock_runner_class.return_value = mock_runner
        
        pipeline = create_pipeline(
            thread_pool_size=2,
            process_pool_size=1,
            enable_gpu=False
        )
        
        # Create sample segments
        segments = []
        prompts = []
        for i in range(3):
            segment = Segment(
                segment_id=i,
                tokens=[1, 2, 3],
                start_idx=0,
                end_idx=2,
                signatures={'test': np.array([1, 2, 3])}
            )
            segments.append(pickle.dumps(segment))
            prompts.append(f"prompt_{i}")
        
        # Process segments
        results = await process_segments_parallel(
            pipeline=pipeline,
            segments=segments,
            model_id="test_model",
            prompts=prompts,
            priority=TaskPriority.HIGH
        )
        
        assert len(results) == 3
        for result in results:
            assert result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        
        pipeline.shutdown()


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def integration_config(self):
        return PipelineConfig(
            thread_pool_size=3,
            process_pool_size=2,
            memory=MemoryConfig(max_memory_gb=2.0),
            gpu=GPUConfig(enable_gpu=False),
            optimization=OptimizationConfig(
                enable_dynamic_batching=True,
                max_batch_size=2,
                batch_timeout_ms=100,
                enable_prefetching=False
            )
        )
    
    @patch('src.executor.segment_runner.SegmentRunner')
    def test_end_to_end_processing(self, mock_runner_class, integration_config):
        """Test end-to-end pipeline processing."""
        # Mock the SegmentRunner
        mock_runner = Mock()
        mock_runner.process_segment.return_value = {"processed": True, "score": 0.85}
        mock_runner_class.return_value = mock_runner
        
        pipeline = ParallelPipeline(integration_config)
        pipeline.start()
        
        # Create multiple tasks with different priorities
        tasks = []
        for i in range(5):
            segment = Segment(
                segment_id=i,
                tokens=list(range(i*10, (i+1)*10)),
                start_idx=i*10,
                end_idx=(i+1)*10-1,
                signatures={'layer_0': np.random.randn(128)}
            )
            
            task = SegmentTask(
                task_id=f"integration_task_{i}",
                segment_id=f"seg_{i}",
                model_id="integration_model",
                prompt=f"Integration test prompt {i}",
                segment_data=pickle.dumps(segment),
                priority=TaskPriority.HIGH if i < 2 else TaskPriority.NORMAL
            )
            tasks.append(task)
        
        # Submit all tasks
        task_ids = pipeline.submit_batch(tasks)
        
        # Wait for completion
        start_time = time.time()
        completed_results, pending = pipeline.wait_for_completion(
            task_ids, timeout=10.0
        )
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(pending) == 0  # All tasks should complete
        assert len(completed_results) == 5
        
        successful = [r for r in completed_results if r.status == TaskStatus.COMPLETED]
        assert len(successful) == 5  # All should succeed with mocked runner
        
        # Check that high priority tasks were processed
        high_priority_results = [
            r for r in completed_results 
            if 'integration_task_0' in r.task_id or 'integration_task_1' in r.task_id
        ]
        assert len(high_priority_results) == 2
        
        # Verify statistics
        stats = pipeline.get_statistics()
        assert stats['tasks_submitted'] == 5
        assert stats['tasks_completed'] == 5
        assert stats['tasks_failed'] == 0
        
        # Check resource monitoring worked
        assert len(pipeline.resource_monitor.stats_history) > 0
        
        print(f"Integration test completed in {execution_time:.2f}s")
        print(f"Average task time: {execution_time/5:.3f}s")
        
        pipeline.shutdown()
    
    def test_error_handling_and_recovery(self, integration_config):
        """Test error handling and recovery."""
        pipeline = ParallelPipeline(integration_config)
        pipeline.start()
        
        # Create a task with invalid data to trigger an error
        bad_task = SegmentTask(
            task_id="bad_task",
            segment_id="bad_seg",
            model_id="test_model",
            prompt="test prompt",
            segment_data=b"invalid_pickle_data",  # This will cause unpickling to fail
            priority=TaskPriority.NORMAL
        )
        
        task_id = pipeline.submit_task(bad_task)
        
        # Wait for completion
        results, pending = pipeline.wait_for_completion([task_id], timeout=5.0)
        
        assert len(results) == 1
        result = results[0]
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        
        # Verify stats reflect the failure
        stats = pipeline.get_statistics()
        assert stats['tasks_failed'] >= 1
        
        pipeline.shutdown()
    
    def test_memory_management_under_load(self, integration_config):
        """Test memory management under heavy load."""
        # Configure for aggressive memory management
        integration_config.memory.max_memory_gb = 1.0  # Small limit
        integration_config.memory.gc_threshold = 0.5   # Aggressive GC
        
        pipeline = ParallelPipeline(integration_config)
        pipeline.start()
        
        # Create many tasks to stress memory
        tasks = []
        for i in range(20):
            # Create larger segments to use more memory
            segment = Segment(
                segment_id=i,
                tokens=list(range(1000)),  # Larger token list
                start_idx=0,
                end_idx=999,
                signatures={
                    'layer_0': np.random.randn(512),  # Larger signatures
                    'layer_1': np.random.randn(512),
                    'layer_2': np.random.randn(512),
                }
            )
            
            task = SegmentTask(
                task_id=f"memory_task_{i}",
                segment_id=f"mem_seg_{i}",
                model_id="memory_test_model",
                prompt=f"Memory test {i}",
                segment_data=pickle.dumps(segment)
            )
            tasks.append(task)
        
        # Submit tasks
        task_ids = pipeline.submit_batch(tasks)
        
        # Monitor memory during execution
        initial_memory = pipeline.memory_manager.get_memory_usage()
        
        # Cancel some tasks to test cleanup
        for i in range(5):
            pipeline.cancel_task(f"memory_task_{i}")
        
        # Wait for remaining tasks
        results, pending = pipeline.wait_for_completion(
            task_ids[5:], timeout=15.0
        )
        
        final_memory = pipeline.memory_manager.get_memory_usage()
        
        # Memory should be managed (not grow unboundedly)
        memory_growth = final_memory['rss_gb'] - initial_memory['rss_gb']
        assert memory_growth < 2.0  # Shouldn't grow by more than 2GB
        
        # Should have processed most tasks
        successful = len([r for r in results if r.status == TaskStatus.COMPLETED])
        assert successful >= 10  # At least 2/3 should succeed
        
        print(f"Memory growth during test: {memory_growth:.2f}GB")
        
        pipeline.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])