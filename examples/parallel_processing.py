#!/usr/bin/env python3
"""
Parallel Processing Example for REV System

Demonstrates how to process multiple models and prompts in parallel
with configurable memory limits.
"""

import sys
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.executor.parallel_executor import (
    ParallelExecutor, 
    AdaptiveParallelExecutor,
    BatchProcessor,
    MemoryConfig,
    parallel_process_prompts,
    parallel_process_models
)


def display_memory_status():
    """Display current memory status."""
    mem = psutil.virtual_memory()
    print(f"\n💾 System Memory Status:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Used: {mem.used / (1024**3):.1f} GB ({mem.percent:.1f}%)")


def example_parallel_prompts():
    """
    Example 1: Process multiple prompts on a single model in parallel.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Parallel Prompt Processing")
    print("=" * 80)
    
    # Display initial memory
    display_memory_status()
    
    # Model path (update with your actual model)
    model_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    
    # Generate test prompts
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Describe the water cycle.",
        "What is artificial intelligence?",
        "How do vaccines work?",
        "Explain climate change.",
        "What is DNA?",
        "How does the internet work?"
    ]
    
    print(f"\nModel: {Path(model_path).name}")
    print(f"Prompts: {len(prompts)}")
    
    # Configure memory (36GB total, 2GB per process)
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=2.0,
        buffer_gb=2.0
    )
    
    print(f"\nMemory Configuration:")
    print(f"  Total limit: {config.total_limit_gb} GB")
    print(f"  Per process: {config.per_process_gb} GB")
    print(f"  Max parallel: {config.max_processes} processes")
    
    # Initialize executor
    executor = ParallelExecutor(config)
    
    print("\n🚀 Starting parallel processing...")
    start_time = time.time()
    
    # Process prompts in parallel
    results = executor.process_prompts_parallel(
        model_path=model_path,
        prompts=prompts,
        batch_size=3  # Process 3 prompts per batch
    )
    
    duration = time.time() - start_time
    
    # Display results
    print(f"\n✅ Processing complete in {duration:.2f} seconds")
    print(f"  Results: {len(results)} responses")
    print(f"  Throughput: {len(prompts)/duration:.2f} prompts/second")
    
    # Show sample results
    print("\nSample Results:")
    for i, (prompt, result) in enumerate(zip(prompts[:3], results[:3])):
        print(f"\n  Prompt {i+1}: {prompt}")
        if isinstance(result, dict) and result.get("response"):
            response = str(result["response"])[:100] + "..." if len(str(result["response"])) > 100 else str(result["response"])
            print(f"  Response: {response}")
    
    # Cleanup
    executor.shutdown()
    
    # Display final memory
    display_memory_status()


def example_parallel_models():
    """
    Example 2: Process multiple models in parallel.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Parallel Model Processing")
    print("=" * 80)
    
    # Find available models
    model_dir = Path.home() / "LLM_models"
    model_paths = []
    
    if model_dir.exists():
        # Find models with config.json
        for config_path in model_dir.glob("*/config.json"):
            model_paths.append(str(config_path.parent))
            if len(model_paths) >= 3:  # Limit to 3 models for demo
                break
    
    if not model_paths:
        print("⚠️  No models found. Please update model paths.")
        model_paths = [
            "/path/to/model1",
            "/path/to/model2",
            "/path/to/model3"
        ]
    
    print(f"\nModels to process: {len(model_paths)}")
    for path in model_paths:
        print(f"  • {Path(path).name}")
    
    # Configure memory
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=4.0  # More memory per model
    )
    
    print(f"\nMemory Configuration:")
    print(f"  Total limit: {config.total_limit_gb} GB")
    print(f"  Per process: {config.per_process_gb} GB")
    print(f"  Max parallel: {config.max_processes} models")
    
    # Generate test prompts
    prompts = [
        "What is AI?",
        "Explain gravity.",
        "How do computers work?"
    ]
    
    print(f"\nChallenges per model: {len(prompts)}")
    
    # Process models in parallel
    print("\n🚀 Processing models in parallel...")
    start_time = time.time()
    
    results = parallel_process_models(
        model_paths=model_paths,
        challenges=len(prompts),
        memory_limit_gb=36.0
    )
    
    duration = time.time() - start_time
    
    # Display results
    print(f"\n✅ Processing complete in {duration:.2f} seconds")
    print(f"  Models processed: {len(results)}")
    print(f"  Total operations: {len(model_paths) * len(prompts)}")
    print(f"  Throughput: {len(model_paths) * len(prompts) / duration:.2f} ops/second")
    
    # Show results
    print("\nModel Results:")
    for model_path, result in results.items():
        model_name = Path(model_path).name
        print(f"\n  {model_name}:")
        if isinstance(result, dict):
            if result.get("success"):
                print(f"    ✅ Success")
                print(f"    Confidence: {result.get('confidence', 0):.2%}")
            else:
                print(f"    ❌ Error: {result.get('error', 'Unknown')}")


def example_adaptive_parallel():
    """
    Example 3: Adaptive parallel processing that adjusts to system load.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Adaptive Parallel Processing")
    print("=" * 80)
    
    # Configure adaptive executor
    config = MemoryConfig(
        total_limit_gb=36.0,
        per_process_gb=2.0
    )
    
    executor = AdaptiveParallelExecutor(config)
    
    print(f"\nAdaptive Executor Configuration:")
    print(f"  Initial workers: {executor.max_workers}")
    print(f"  Memory limit: {config.total_limit_gb} GB")
    print(f"  Adaptation interval: {executor.adaptation_interval} tasks")
    
    # Simulate varying workload
    model_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    
    print("\n🔄 Simulating varying workload...")
    
    # Phase 1: Light load
    print("\nPhase 1: Light load (5 prompts)")
    prompts_light = ["Test prompt " + str(i) for i in range(5)]
    
    start = time.time()
    results1 = executor.process_prompts_parallel(model_path, prompts_light)
    phase1_time = time.time() - start
    
    print(f"  Completed in {phase1_time:.2f}s")
    print(f"  Workers: {executor.optimal_workers}")
    
    # Phase 2: Heavy load
    print("\nPhase 2: Heavy load (20 prompts)")
    prompts_heavy = ["Complex prompt " + str(i) for i in range(20)]
    
    start = time.time()
    results2 = executor.process_prompts_parallel(model_path, prompts_heavy)
    phase2_time = time.time() - start
    
    print(f"  Completed in {phase2_time:.2f}s")
    print(f"  Workers: {executor.optimal_workers}")
    
    # Phase 3: Normal load
    print("\nPhase 3: Normal load (10 prompts)")
    prompts_normal = ["Normal prompt " + str(i) for i in range(10)]
    
    start = time.time()
    results3 = executor.process_prompts_parallel(model_path, prompts_normal)
    phase3_time = time.time() - start
    
    print(f"  Completed in {phase3_time:.2f}s")
    print(f"  Workers: {executor.optimal_workers}")
    
    # Show adaptation summary
    print("\n📊 Adaptation Summary:")
    print(f"  Total prompts: {len(prompts_light) + len(prompts_heavy) + len(prompts_normal)}")
    print(f"  Total time: {phase1_time + phase2_time + phase3_time:.2f}s")
    print(f"  Average throughput: {35/(phase1_time + phase2_time + phase3_time):.2f} prompts/s")
    
    # Cleanup
    executor.shutdown()


def example_batch_processor():
    """
    Example 4: High-level batch processing interface.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Processor Interface")
    print("=" * 80)
    
    # Initialize batch processor with 36GB limit
    processor = BatchProcessor(memory_limit_gb=36.0)
    
    print(f"\nBatch Processor Configuration:")
    print(f"  Memory limit: 36.0 GB")
    print(f"  Max workers: {processor.executor.max_workers}")
    
    # Find models
    model_paths = []
    model_dir = Path.home() / "LLM_models"
    
    if model_dir.exists():
        for config_path in list(model_dir.glob("*/config.json"))[:2]:
            model_paths.append(str(config_path.parent))
    
    if len(model_paths) < 2:
        model_paths = ["/path/to/model1", "/path/to/model2"]
    
    # Generate prompts
    prompts = [
        "What is the meaning of life?",
        "How does gravity work?",
        "What is consciousness?"
    ]
    
    print(f"\nBatch Configuration:")
    print(f"  Models: {len(model_paths)}")
    print(f"  Prompts: {len(prompts)}")
    
    # Test different modes
    modes = ["cross_product", "broadcast"]
    
    for mode in modes:
        print(f"\n📦 Mode: {mode}")
        print("-" * 40)
        
        start = time.time()
        
        batch_results = processor.process_batch(
            model_paths=model_paths,
            prompts=prompts,
            mode=mode
        )
        
        duration = time.time() - start
        
        # Display results
        stats = batch_results["statistics"]
        print(f"  Duration: {stats['duration_seconds']:.2f}s")
        print(f"  Throughput: {stats['throughput']:.2f} ops/s")
        print(f"  Models processed: {stats['models_processed']}")
        print(f"  Prompts processed: {stats['prompts_processed']}")
    
    # Show executor status
    status = processor.executor.get_status()
    print(f"\n📊 Final Status:")
    print(f"  Memory allocated: {status['memory_status']['allocated_gb']:.1f} GB")
    print(f"  Memory available: {status['memory_status']['available_gb']:.1f} GB")
    print(f"  Active tasks: {status['active_tasks']}")
    print(f"  Completed tasks: {status['completed_tasks']}")
    
    # Cleanup
    processor.shutdown()


def main():
    """Main function to run all examples."""
    print("=" * 80)
    print("REV PARALLEL PROCESSING EXAMPLES")
    print("=" * 80)
    print("\nThis demonstrates parallel processing with memory management.")
    print("Default configuration: 36GB total memory limit")
    
    # Check system resources
    display_memory_status()
    
    cpu_count = psutil.cpu_count()
    print(f"\n🖥️  System CPUs: {cpu_count}")
    
    # Run examples
    examples = [
        ("Parallel Prompts", example_parallel_prompts),
        ("Parallel Models", example_parallel_models),
        ("Adaptive Processing", example_adaptive_parallel),
        ("Batch Processor", example_batch_processor)
    ]
    
    print("\nSelect an example to run:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. Run all examples")
    print(f"  0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == "0":
        print("Exiting...")
        return
    elif choice == str(len(examples) + 1):
        # Run all examples
        for name, func in examples:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print('='*80)
            try:
                func()
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("Invalid choice")
        except (ValueError, IndexError):
            print("Invalid choice")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()