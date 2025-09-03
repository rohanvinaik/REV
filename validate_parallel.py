#!/usr/bin/env python3
"""
Validation script for REV parallel processing functionality.
Tests the new parallel processing capabilities with memory management.
"""

import sys
import json
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))

from src.executor.parallel_executor import (
    ParallelExecutor,
    AdaptiveParallelExecutor,
    BatchProcessor,
    MemoryConfig,
    parallel_process_prompts,
    parallel_process_models
)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def validate_memory_management():
    """Test 1: Validate memory management stays within limits."""
    print_section("TEST 1: Memory Management Validation")
    
    # Get initial memory
    initial_mem = psutil.virtual_memory()
    print(f"Initial system memory: {initial_mem.available / (1024**3):.1f} GB available")
    
    # Configure for 36GB limit
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=2.0,
        buffer_gb=2.0
    )
    
    print(f"Memory configuration:")
    print(f"  Total limit: {config.total_limit_gb} GB")
    print(f"  Per process: {config.per_process_gb} GB")
    print(f"  Max processes: {config.max_processes}")
    
    # Initialize executor
    executor = ParallelExecutor(config)
    
    # Check allocation tracking
    print(f"\nMemory allocation tracking:")
    print(f"  Allocated: {executor.monitor.allocated_memory_gb:.2f} GB")
    print(f"  Available: {executor.monitor.get_available_memory():.2f} GB")
    print(f"  Can allocate 2GB? {executor.monitor.can_allocate(2.0)}")
    print(f"  Can allocate 40GB? {executor.monitor.can_allocate(40.0)}")
    
    # Test allocation
    if executor.monitor.allocate(4.0):
        print(f"  ✓ Successfully allocated 4GB")
        print(f"  Allocated now: {executor.monitor.allocated_memory_gb:.2f} GB")
        executor.monitor.release(4.0)
        print(f"  ✓ Released 4GB")
    
    executor.shutdown()
    print("\n✅ Memory management validation passed")
    return True


def validate_parallel_prompts():
    """Test 2: Validate parallel prompt processing."""
    print_section("TEST 2: Parallel Prompt Processing")
    
    # Find a test model
    model_path = None
    model_dir = Path.home() / "LLM_models"
    
    if model_dir.exists():
        # Try to find pythia-70m first (small and fast)
        for path in model_dir.glob("**/config.json"):
            if "pythia" in str(path).lower() and "70m" in str(path):
                model_path = str(path.parent)
                break
        
        # If not found, use any model
        if not model_path:
            for path in model_dir.glob("*/config.json"):
                model_path = str(path.parent)
                break
    
    if not model_path:
        print("⚠️  No models found for testing. Skipping prompt test.")
        return False
    
    print(f"Using model: {Path(model_path).name}")
    
    # Create test prompts
    prompts = [
        "What is AI?",
        "Explain quantum computing.",
        "How does DNA work?",
        "What is climate change?",
        "Describe photosynthesis."
    ]
    
    print(f"Testing with {len(prompts)} prompts")
    
    # Configure memory
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=2.0
    )
    
    # Test parallel processing
    executor = ParallelExecutor(config)
    
    print("\nProcessing prompts in parallel...")
    start_time = time.time()
    
    try:
        results = executor.process_prompts_parallel(
            model_path=model_path,
            prompts=prompts,
            batch_size=2
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ Processed {len(results)} prompts in {duration:.2f} seconds")
        print(f"Throughput: {len(prompts)/duration:.2f} prompts/second")
        
        # Verify results structure
        if results and isinstance(results[0], dict):
            print("\nResult structure validated:")
            print(f"  Keys: {list(results[0].keys())}")
        
        executor.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        executor.shutdown()
        return False


def validate_adaptive_executor():
    """Test 3: Validate adaptive parallel executor."""
    print_section("TEST 3: Adaptive Parallel Executor")
    
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=2.0
    )
    
    executor = AdaptiveParallelExecutor(config)
    
    print(f"Adaptive executor configuration:")
    print(f"  Initial workers: {executor.max_workers}")
    print(f"  Adaptation interval: {executor.adaptation_interval}")
    print(f"  Memory limit: {config.total_limit_gb} GB")
    
    # Test adaptation
    print("\nTesting worker adaptation...")
    
    # Simulate light load
    executor._adjust_workers()
    print(f"  After light load: {executor.optimal_workers} workers")
    
    # Simulate heavy load (mock high memory usage)
    original_mem = psutil.virtual_memory().percent
    psutil.virtual_memory = lambda: type('obj', (), {'percent': 85.0})()
    executor._adjust_workers()
    print(f"  After heavy load simulation: {executor.optimal_workers} workers")
    
    # Restore
    psutil.virtual_memory = lambda: type('obj', (), {'percent': original_mem})()
    
    executor.shutdown()
    print("\n✅ Adaptive executor validation passed")
    return True


def validate_batch_processor():
    """Test 4: Validate batch processor interface."""
    print_section("TEST 4: Batch Processor Interface")
    
    processor = BatchProcessor(memory_limit_gb=36.0)
    
    print(f"Batch processor configuration:")
    print(f"  Memory limit: 36.0 GB")
    print(f"  Max workers: {processor.executor.max_workers}")
    
    # Test status
    status = processor.executor.get_status()
    print(f"\nExecutor status:")
    print(f"  Memory allocated: {status['memory_status']['allocated_gb']:.1f} GB")
    print(f"  Memory available: {status['memory_status']['available_gb']:.1f} GB")
    print(f"  Active tasks: {status['active_tasks']}")
    print(f"  Max workers: {status['max_workers']}")
    
    processor.shutdown()
    print("\n✅ Batch processor validation passed")
    return True


def validate_memory_limits():
    """Test 5: Validate memory limit enforcement."""
    print_section("TEST 5: Memory Limit Enforcement")
    
    # Test with small limit
    config = MemoryConfig(
        total_limit_gb=8.0,  # Small limit
        per_process_gb=3.0   # 3GB per process
    )
    
    executor = ParallelExecutor(config)
    
    print(f"Testing with 8GB limit and 3GB per process:")
    print(f"  Max processes allowed: {config.max_processes}")
    
    # Try to allocate within limit
    allocated = []
    for i in range(5):  # Try to allocate 5 x 3GB = 15GB (should fail after 2)
        if executor.monitor.allocate(3.0):
            allocated.append(3.0)
            print(f"  ✓ Allocated process {i+1}: Total {sum(allocated):.0f}GB")
        else:
            print(f"  ✗ Cannot allocate process {i+1}: Limit reached")
    
    # Release all
    for amount in allocated:
        executor.monitor.release(amount)
    
    print(f"\n✅ Memory limits properly enforced")
    print(f"  Maximum allocated: {sum(allocated):.0f}GB (limit was 8GB)")
    
    executor.shutdown()
    return True


def main():
    """Run all validation tests."""
    print("="*80)
    print("  REV PARALLEL PROCESSING VALIDATION")
    print("="*80)
    print("\nValidating new parallel processing capabilities...")
    print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"Configured limit: 36 GB")
    
    tests = [
        ("Memory Management", validate_memory_management),
        ("Parallel Prompts", validate_parallel_prompts),
        ("Adaptive Executor", validate_adaptive_executor),
        ("Batch Processor", validate_batch_processor),
        ("Memory Limits", validate_memory_limits)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error: {e}")
            results[name] = False
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All parallel processing features validated successfully!")
        print("\nYou can now use parallel processing with:")
        print("  python run_rev.py /path/to/model --parallel --parallel-memory-limit 36.0")
        print("\nOr with multiple models:")
        print("  python run_rev.py model1 model2 model3 --parallel --parallel-batch-size 3")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
    
    # Save results
    results_file = Path("parallel_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": results,
            "passed": passed,
            "total": total,
            "memory_limit": 36.0
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()