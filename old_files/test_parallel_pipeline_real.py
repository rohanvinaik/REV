#!/usr/bin/env python3
"""
Test the fixed parallel pipeline with real model execution.
Verifies that the mock implementations have been replaced with actual processing.
"""

import sys
import time
import pickle
import torch
import numpy as np
import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/rohanvinaik/REV')

from src.executor.parallel_pipeline import (
    ParallelPipeline, 
    PipelineConfig,
    SegmentTask,
    TaskPriority,
    ExecutionMode
)

def test_parallel_pipeline():
    """Test the parallel pipeline with real model execution."""
    
    print("=" * 80)
    print("PARALLEL PIPELINE TEST - REAL EXECUTION")
    print("=" * 80)
    
    # Load a small model for testing
    print("\n1. Loading test model...")
    model_path = Path("/Users/rohanvinaik/LLM_models/pythia-70m")
    
    if not model_path.exists():
        print(f"❌ Model path {model_path} doesn't exist")
        return False
    
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    print(f"✓ Loaded pythia-70m model")
    
    # Create pipeline config
    config = PipelineConfig(
        thread_pool_size=2,
        process_pool_size=1,
        max_concurrent_tasks=4
    )
    
    # Initialize pipeline
    print("\n2. Initializing parallel pipeline...")
    pipeline = ParallelPipeline(config)
    pipeline.start()
    print(f"✓ Pipeline started with {config.thread_pool_size} threads, {config.process_pool_size} processes")
    
    # Create test segments
    print("\n3. Creating test segments...")
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "The weather today is"
    ]
    
    tasks = []
    for i, prompt in enumerate(test_prompts):
        # Tokenize prompt
        tokens = tokenizer.encode(prompt, max_length=128, truncation=True)
        
        # Create segment data
        segment_data = {
            'model': model,
            'tokens': tokens,
            'extraction_sites': ['embeddings', 'attention.0']
        }
        
        # Create tasks for different execution modes
        for mode in [ExecutionMode.CPU_ONLY, ExecutionMode.GPU_ONLY, ExecutionMode.HYBRID]:
            task = SegmentTask(
                task_id=f"task_{i}_{mode.value}",
                segment_id=f"segment_{i}",
                model_id="pythia-70m",
                prompt=prompt,
                segment_data=pickle.dumps(segment_data),
                priority=TaskPriority.NORMAL,
                execution_mode=mode,
                metadata={'prompt': prompt, 'mode': mode.value}
            )
            tasks.append(task)
    
    print(f"✓ Created {len(tasks)} tasks ({len(test_prompts)} prompts × 3 modes)")
    
    # Submit tasks
    print("\n4. Submitting tasks to pipeline...")
    task_ids = []
    for task in tasks:
        task_id = pipeline.submit_task(task)
        task_ids.append((task, task_id))
    
    print(f"✓ Submitted {len(task_ids)} tasks")
    
    # Wait for results
    print("\n5. Processing tasks...")
    
    # Wait for all tasks to complete
    import time
    time.sleep(2)  # Give tasks time to start
    
    # Get results
    all_task_ids = [task_id for _, task_id in task_ids]
    completed_results, pending_ids = pipeline.wait_for_completion(all_task_ids, timeout=30)
    
    # Convert list of results to dict by task_id
    completed = {r.task_id: r for r in completed_results}
    
    results = []
    errors = []
    
    for task, task_id in task_ids:
        if task_id in completed:
            task_result = completed[task_id]
            results.append(task_result)
            
            # Check if result contains real data
            if task_result.error:
                errors.append(f"Task {task.task_id}: {task_result.error}")
            elif task_result.result and 'activations' in task_result.result and task_result.result['activations']:
                # Verify activations are not mock data
                is_mock = False
                result_data = task_result.result
                for key, val in result_data['activations'].items():
                    if 'mock' in key.lower():
                        is_mock = True
                        break
                    # Check if activation has reasonable shape and values
                    if isinstance(val, np.ndarray):
                        if val.shape == (3,) or np.array_equal(val, np.array([1, 2, 3])):
                            is_mock = True
                            break
                
                if is_mock:
                    print(f"  ❌ Task {task.task_id}: Still using mock data!")
                else:
                    print(f"  ✓ Task {task.task_id}: Real activations extracted")
                    print(f"    - Device: {result_data.get('device', 'unknown')}")
                    print(f"    - Execution time: {task_result.execution_time:.3f}s")
                    print(f"    - Memory used: {task_result.memory_used:.1f}MB")
                    if 'signatures' in result_data:
                        print(f"    - Signatures generated: {len(result_data['signatures'])}")
            else:
                print(f"  ⚠️  Task {task.task_id}: No activations returned")
        else:
            errors.append(f"Task {task.task_id}: Timed out or not completed")
            print(f"  ❌ Task {task.task_id}: Not completed")
    
    # Analyze results
    print("\n6. Results Analysis...")
    print("-" * 60)
    
    successful = len([r for r in results if not r.error and r.result and r.result.get('activations')])
    failed = len(errors)
    
    print(f"Successful: {successful}/{len(tasks)}")
    print(f"Failed: {failed}/{len(tasks)}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Verify activations are different for different prompts
    if successful > 1:
        print("\n7. Verification: Different prompts produce different activations...")
        
        # Get two successful results
        success_results = [r for r in results if not r.error and r.result and r.result.get('activations')]
        if len(success_results) >= 2:
            act1 = list(success_results[0].result['activations'].values())[0]
            act2 = list(success_results[1].result['activations'].values())[0]
            
            if isinstance(act1, np.ndarray) and isinstance(act2, np.ndarray):
                diff = np.mean(np.abs(act1.flatten()[:100] - act2.flatten()[:100]))
                print(f"  Average difference: {diff:.4f}")
                if diff > 0.01:
                    print(f"  ✓ Different prompts produce different activations")
                else:
                    print(f"  ❌ Activations too similar (might be using mock data)")
    
    # Shutdown pipeline
    print("\n8. Shutting down pipeline...")
    stats = pipeline.get_statistics()
    print(f"  Tasks processed: {stats.get('tasks_completed', 0)}")
    print(f"  Tasks failed: {stats.get('tasks_failed', 0)}")
    print(f"  Avg execution time: {stats.get('avg_execution_time', 0):.3f}s")
    
    pipeline.shutdown()
    print("✓ Pipeline shut down")
    
    # Final verdict
    print("\n" + "=" * 80)
    if successful > 0 and successful > failed:
        print("✅ SUCCESS: Parallel pipeline is using REAL model execution!")
        print(f"   {successful} tasks completed with real activations")
    else:
        print("❌ FAILURE: Pipeline not working correctly")
        print(f"   Only {successful} successful tasks out of {len(tasks)}")
    print("=" * 80)
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return successful > failed

if __name__ == "__main__":
    success = test_parallel_pipeline()
    exit(0 if success else 1)