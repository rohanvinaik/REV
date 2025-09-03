#!/usr/bin/env python3
"""
Memory-Bounded Execution Example

Demonstrates how REV handles massive models that exceed available memory.
"""

import sys
import psutil
import time
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from run_rev import REVUnified
from src.utils.run_rev_recovery import PipelineRecovery


def monitor_memory() -> Dict[str, float]:
    """Get current memory usage statistics."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent,
        'swap_used_gb': swap.used / (1024**3)
    }


def demonstrate_memory_bounded(model_path: str):
    """
    Demonstrate memory-bounded execution for large models.
    
    Args:
        model_path: Path to a large model
    """
    print("=" * 70)
    print("MEMORY-BOUNDED EXECUTION DEMONSTRATION")
    print("=" * 70)
    
    # Show system memory
    mem_start = monitor_memory()
    print("\n💾 System Memory Status:")
    print(f"  Total: {mem_start['total_gb']:.1f} GB")
    print(f"  Available: {mem_start['available_gb']:.1f} GB")
    print(f"  Used: {mem_start['used_gb']:.1f} GB ({mem_start['percent']:.1f}%)")
    
    # Check model size
    model_path = Path(model_path)
    if model_path.exists():
        model_size = sum(f.stat().st_size for f in model_path.rglob('*')) / (1024**3)
        print(f"\n📦 Model Size: {model_size:.1f} GB")
        
        if model_size > mem_start['available_gb']:
            print(f"⚠️  Model exceeds available memory by "
                  f"{model_size - mem_start['available_gb']:.1f} GB")
            print("✅ REV will handle this using segmented execution")
    
    # Configure memory-bounded execution
    print("\n⚙️  Configuration:")
    memory_configs = [
        (1.0, "Minimal (1GB) - Slowest but works everywhere"),
        (2.0, "Conservative (2GB) - Good for 16GB systems"),
        (4.0, "Balanced (4GB) - Good for 32GB+ systems"),
        (8.0, "Performance (8GB) - Good for 64GB+ systems")
    ]
    
    for limit, description in memory_configs:
        if limit <= mem_start['available_gb'] * 0.5:  # Use 50% of available
            print(f"  Using: {limit:.1f} GB - {description}")
            memory_limit = limit
            break
    else:
        memory_limit = 1.0
        print(f"  Forcing: 1.0 GB (low memory system)")
    
    # Initialize REV with memory limit
    print("\n🚀 Starting Memory-Bounded Analysis:")
    print("-" * 50)
    
    rev = REVUnified(
        memory_limit_gb=memory_limit,
        enable_memory_mapping=True,  # Use mmap for large files
        checkpoint_dir="checkpoints/memory_demo",
        debug=True
    )
    
    try:
        # Monitor memory during execution
        print("\n📊 Memory Usage During Execution:")
        
        start_time = time.time()
        
        # Process model with periodic memory monitoring
        result = rev.process_model_with_monitoring(
            model_path=str(model_path),
            challenges=5,  # Quick demo
            monitor_interval=1.0  # Check memory every second
        )
        
        elapsed = time.time() - start_time
        
        # Show results
        print("\n✅ Analysis Completed Successfully!")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Model Family: {result.get('model_family', 'Unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        
        # Memory statistics
        if 'memory_stats' in result:
            stats = result['memory_stats']
            print(f"\n📊 Memory Statistics:")
            print(f"  Peak Usage: {stats['peak_gb']:.2f} GB")
            print(f"  Average Usage: {stats['avg_gb']:.2f} GB")
            print(f"  Stayed Under Limit: {'✅' if stats['peak_gb'] <= memory_limit else '❌'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        rev.cleanup()
        
        # Final memory check
        mem_end = monitor_memory()
        print(f"\n💾 Final Memory Status:")
        print(f"  Memory Released: {mem_start['used_gb'] - mem_end['used_gb']:.1f} GB")


def demonstrate_checkpoint_recovery(model_path: str):
    """
    Demonstrate checkpoint recovery for interrupted runs.
    """
    print("\n" + "=" * 70)
    print("CHECKPOINT RECOVERY DEMONSTRATION")
    print("=" * 70)
    
    checkpoint_dir = Path("checkpoints/recovery_demo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Checkpoint Directory: {checkpoint_dir}")
    
    # Simulate interrupted run
    print("\n🔄 Phase 1: Starting analysis (will interrupt)...")
    
    rev = REVUnified(
        memory_limit_gb=2.0,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=2,  # Save every 2 challenges
        debug=True
    )
    
    try:
        # Process only first few challenges
        result = rev.process_model(
            model_path,
            challenges=10,
            interrupt_after=3  # Simulate interruption
        )
    except KeyboardInterrupt:
        print("\n⚠️  Run interrupted! (simulated)")
    
    # Check for checkpoint
    checkpoints = list(checkpoint_dir.glob("*.checkpoint"))
    if checkpoints:
        print(f"\n✅ Found {len(checkpoints)} checkpoint(s)")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"  Latest: {latest.name}")
        
        # Resume from checkpoint
        print("\n🔄 Phase 2: Resuming from checkpoint...")
        
        recovery = PipelineRecovery(checkpoint_dir=str(checkpoint_dir))
        
        # Load checkpoint
        state = recovery.load_checkpoint(latest)
        if state:
            print(f"  Resumed at: Challenge {state['progress']['current_challenge']}/10")
            print(f"  Completed: {state['progress']['percent_complete']:.1f}%")
            
            # Continue processing
            rev_resumed = REVUnified(
                memory_limit_gb=2.0,
                checkpoint_dir=str(checkpoint_dir),
                resume_from_checkpoint=str(latest)
            )
            
            result = rev_resumed.process_model(
                model_path,
                challenges=10  # Will continue from checkpoint
            )
            
            print(f"\n✅ Analysis completed!")
            print(f"  Final confidence: {result.get('confidence', 0):.2%}")
    else:
        print("\n❌ No checkpoints found")


def demonstrate_adaptive_memory(model_path: str):
    """
    Demonstrate adaptive memory management based on system load.
    """
    print("\n" + "=" * 70)
    print("ADAPTIVE MEMORY MANAGEMENT")
    print("=" * 70)
    
    print("\nThis demonstrates how REV adapts to changing memory conditions.\n")
    
    # Initialize with adaptive configuration
    rev = REVUnified(
        memory_limit_gb=4.0,  # Initial limit
        adaptive_memory=True,  # Enable adaptation
        min_memory_gb=1.0,     # Minimum limit
        max_memory_gb=8.0      # Maximum limit
    )
    
    print("📊 Adaptive Configuration:")
    print(f"  Initial Limit: 4.0 GB")
    print(f"  Minimum: 1.0 GB")
    print(f"  Maximum: 8.0 GB")
    print(f"  Adaptation: Enabled")
    
    # Simulate memory pressure scenarios
    scenarios = [
        ("Normal", 0.5),
        ("High Load", 0.8),
        ("Critical", 0.95),
        ("Recovery", 0.3)
    ]
    
    print("\n🔄 Testing Memory Adaptation:")
    print("-" * 50)
    
    for scenario, memory_pressure in scenarios:
        print(f"\n📌 Scenario: {scenario} (Memory {memory_pressure*100:.0f}% full)")
        
        # Simulate memory pressure
        mem = monitor_memory()
        simulated_available = mem['total_gb'] * (1 - memory_pressure)
        
        # REV would adapt here
        if memory_pressure > 0.9:
            adapted_limit = 1.0  # Minimum
        elif memory_pressure > 0.7:
            adapted_limit = 2.0  # Reduced
        elif memory_pressure < 0.4:
            adapted_limit = 8.0  # Increased
        else:
            adapted_limit = 4.0  # Normal
        
        print(f"  Adapted Limit: {adapted_limit:.1f} GB")
        print(f"  Strategy: ", end="")
        
        if adapted_limit <= 1.0:
            print("Survival mode - Single layer at a time")
        elif adapted_limit <= 2.0:
            print("Conservative - Small segments")
        elif adapted_limit >= 8.0:
            print("Performance - Large segments")
        else:
            print("Balanced - Standard segments")


def compare_memory_strategies():
    """
    Compare different memory strategies and their trade-offs.
    """
    print("\n" + "=" * 70)
    print("MEMORY STRATEGY COMPARISON")
    print("=" * 70)
    
    strategies = [
        {
            "name": "Minimal",
            "memory_gb": 1.0,
            "segment_size": 32,
            "speed": "Slow",
            "reliability": "Very High",
            "use_case": "Systems with <16GB RAM or running multiple models"
        },
        {
            "name": "Conservative",
            "memory_gb": 2.0,
            "segment_size": 64,
            "speed": "Moderate",
            "reliability": "High",
            "use_case": "Standard 16-32GB systems"
        },
        {
            "name": "Balanced",
            "memory_gb": 4.0,
            "segment_size": 128,
            "speed": "Good",
            "reliability": "Good",
            "use_case": "Systems with 32-64GB RAM"
        },
        {
            "name": "Performance",
            "memory_gb": 8.0,
            "segment_size": 256,
            "speed": "Fast",
            "reliability": "Moderate",
            "use_case": "High-end systems with 64GB+ RAM"
        },
        {
            "name": "Maximum",
            "memory_gb": 16.0,
            "segment_size": 512,
            "speed": "Very Fast",
            "reliability": "Low",
            "use_case": "Dedicated servers with 128GB+ RAM"
        }
    ]
    
    print("\n📊 Strategy Comparison Table:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Memory':<10} {'Segment':<10} {'Speed':<10} {'Reliability':<12}")
    print("-" * 70)
    
    for s in strategies:
        print(f"{s['name']:<15} {s['memory_gb']:<10.1f} {s['segment_size']:<10} "
              f"{s['speed']:<10} {s['reliability']:<12}")
    
    print("\n📝 Recommendations:")
    for s in strategies:
        print(f"\n  {s['name']}:")
        print(f"    Use when: {s['use_case']}")
        print(f"    Command: python run_rev.py model/ --memory-limit {s['memory_gb']}")


def main():
    """Main function."""
    
    # Example model path (update with your model)
    model_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    
    # Try to find a model if default doesn't exist
    if not Path(model_path).exists():
        print("⚠️  Default model not found, searching for alternatives...")
        
        search_paths = [
            Path.home() / "LLM_models",
            Path.home() / ".cache" / "huggingface" / "hub"
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                configs = list(search_path.glob("*/config.json")) + \
                         list(search_path.glob("*/snapshots/*/config.json"))
                if configs:
                    model_path = str(configs[0].parent)
                    print(f"✅ Using model: {model_path}\n")
                    break
        else:
            print("❌ No models found. Please specify a model path.\n")
            # Continue with examples anyway
    
    # Run demonstrations
    if Path(model_path).exists():
        demonstrate_memory_bounded(model_path)
        demonstrate_checkpoint_recovery(model_path)
        demonstrate_adaptive_memory(model_path)
    
    # Always show comparison
    compare_memory_strategies()
    
    print("\n" + "=" * 70)
    print("Memory demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()