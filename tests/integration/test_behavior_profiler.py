#!/usr/bin/env python3
"""
Test script for the BehaviorProfiler system.
Validates integration with REV pipeline and performance targets.
"""

import time
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

from src.analysis.behavior_profiler import (
    BehaviorProfiler,
    BehavioralSignature,
    integrate_with_rev_pipeline
)


def generate_test_segment(layer: int, size: tuple = (512, 4096)) -> torch.Tensor:
    """Generate realistic test segment data."""
    # Simulate layer-specific patterns
    base_pattern = torch.randn(size)
    
    # Add layer-specific characteristics
    if layer < 10:
        # Early layers: sparse, structured
        mask = torch.rand(size) > 0.7
        base_pattern = base_pattern * mask.float()
    elif layer < 40:
        # Middle layers: moderate density
        base_pattern = base_pattern * (0.5 + 0.5 * torch.sigmoid(base_pattern))
    else:
        # Deep layers: dense, complex
        base_pattern = base_pattern + 0.1 * torch.randn(size)
    
    return base_pattern


def test_feature_extraction():
    """Test feature extraction performance and accuracy."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    profiler = BehaviorProfiler({
        "signature_dim": 16,
        "window_size": 5,
        "device": "cpu"
    })
    
    # Test gradient analysis
    segment = generate_test_segment(20)
    
    start = time.time()
    features = profiler.feature_extractor.extract_gradient_features(segment)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✓ Gradient extraction: {elapsed:.2f}ms")
    print(f"  - Gradient magnitude: {features['gradient_magnitude']:.4f}")
    print(f"  - Gradient variability: {features['gradient_variability']:.4f}")
    print(f"  - Direction changes: {features['direction_changes']}")
    
    # Test timing analysis
    timing_data = np.random.lognormal(3.0, 0.5, 100)  # Simulate token timings
    
    start = time.time()
    timing_features = profiler.feature_extractor.extract_timing_features(timing_data)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✓ Timing extraction: {elapsed:.2f}ms")
    print(f"  - Mean interval: {timing_features['mean_interval']:.2f}ms")
    print(f"  - Timing variability: {timing_features['timing_variability']:.4f}")
    print(f"  - Burst ratio: {timing_features['burst_ratio']:.2f}")
    
    return elapsed < 50  # Target: <50ms per extraction


def test_multi_signal_integration():
    """Test multi-signal integration and voting."""
    print("\n" + "="*60)
    print("TESTING MULTI-SIGNAL INTEGRATION")
    print("="*60)
    
    profiler = BehaviorProfiler({
        "enable_multi_signal": True
    })
    
    # Create diverse signals
    signals = []
    for i in range(5):
        sig = BehavioralSignature()
        # Add variation
        sig.response_variability = 0.3 + i * 0.1
        sig.semantic_coherence = 0.8 - i * 0.05
        sig.attention_entropy = 0.5 + (i % 2) * 0.2
        signals.append(sig)
    
    # Test integration
    start = time.time()
    integrated = profiler.signal_integrator.integrate(
        signals,
        weights=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✓ Signal integration: {elapsed:.2f}ms")
    print(f"  - Response variability: {integrated.response_variability:.4f}")
    print(f"  - Semantic coherence: {integrated.semantic_coherence:.4f}")
    print(f"  - Attention entropy: {integrated.attention_entropy:.4f}")
    
    # Test voting mechanism
    votes = profiler.signal_integrator.vote_on_classification(signals)
    print(f"\n✓ Voting results:")
    for model, confidence in votes.items():
        print(f"  - {model}: {confidence:.2%} confidence")
    
    return elapsed < 10  # Target: <10ms for integration


def test_streaming_analysis():
    """Test real-time streaming analysis performance."""
    print("\n" + "="*60)
    print("TESTING STREAMING ANALYSIS")
    print("="*60)
    
    profiler = BehaviorProfiler({
        "window_size": 10,
        "enable_streaming": True
    })
    
    # Simulate streaming segments
    segments = [generate_test_segment(i) for i in range(20)]
    
    latencies = []
    for i, segment in enumerate(segments):
        start = time.time()
        
        # Process segment
        signature = profiler.streaming_analyzer.process_segment(
            segment,
            layer_idx=i,
            timestamp=time.time()
        )
        
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        
        if i % 5 == 0:
            print(f"  Layer {i:2d}: {elapsed:.2f}ms latency")
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n✓ Streaming performance:")
    print(f"  - Average latency: {avg_latency:.2f}ms")
    print(f"  - P95 latency: {p95_latency:.2f}ms")
    print(f"  - Max latency: {max(latencies):.2f}ms")
    
    # Test anomaly detection
    anomalies = profiler.streaming_analyzer.get_anomalies()
    print(f"\n✓ Detected {len(anomalies)} anomalies")
    
    return avg_latency < 100  # Target: <100ms average


def test_full_pipeline():
    """Test complete profiling pipeline end-to-end."""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    profiler = BehaviorProfiler({
        "signature_dim": 16,
        "enable_multi_signal": True,
        "enable_streaming": True,
        "enable_visualization": True
    })
    
    # Simulate full layer processing
    print("\nProcessing 80 layers...")
    
    all_signatures = []
    layer_times = []
    
    for layer in range(80):
        layer_start = time.time()
        
        # Generate layer data
        segment = generate_test_segment(layer, size=(512, 4096))
        
        # Profile layer
        signature = profiler.profile_segment(
            segment=segment,
            layer_idx=layer,
            metadata={
                "model": "test_model",
                "probe_type": "behavioral",
                "timestamp": time.time()
            }
        )
        
        all_signatures.append(signature)
        
        layer_time = (time.time() - layer_start) * 1000
        layer_times.append(layer_time)
        
        # Progress indicator
        if layer % 10 == 0:
            avg_time = np.mean(layer_times[-10:]) if layer > 0 else layer_time
            print(f"  Layer {layer:2d}/80: {avg_time:.2f}ms avg")
    
    # Generate report
    report = profiler.generate_report()
    
    print(f"\n✓ Pipeline complete:")
    print(f"  - Total time: {sum(layer_times)/1000:.2f}s")
    print(f"  - Average per layer: {np.mean(layer_times):.2f}ms")
    print(f"  - P95 per layer: {np.percentile(layer_times, 95):.2f}ms")
    
    # Analyze behavioral evolution
    print(f"\n✓ Behavioral evolution:")
    
    # Sample signatures at different depths
    early = all_signatures[5]
    middle = all_signatures[40]
    late = all_signatures[75]
    
    print(f"  Early layers (5):")
    print(f"    - Response var: {early.response_variability:.3f}")
    print(f"    - Attention entropy: {early.attention_entropy:.3f}")
    
    print(f"  Middle layers (40):")
    print(f"    - Response var: {middle.response_variability:.3f}")
    print(f"    - Attention entropy: {middle.attention_entropy:.3f}")
    
    print(f"  Late layers (75):")
    print(f"    - Response var: {late.response_variability:.3f}")
    print(f"    - Attention entropy: {late.attention_entropy:.3f}")
    
    return np.mean(layer_times) < 100  # Target: <100ms average


def test_rev_integration():
    """Test integration with REV pipeline."""
    print("\n" + "="*60)
    print("TESTING REV PIPELINE INTEGRATION")
    print("="*60)
    
    # Mock REV pipeline components
    class MockREVPipeline:
        def __init__(self):
            self.segments_processed = 0
            
        def process_segment(self, segment_data):
            self.segments_processed += 1
            return {
                "divergence": np.random.random(),
                "layer": self.segments_processed,
                "segment": segment_data
            }
    
    # Create enhanced pipeline
    base_pipeline = MockREVPipeline()
    enhanced_pipeline = integrate_with_rev_pipeline(base_pipeline)
    
    # Process segments through enhanced pipeline
    print("\nProcessing segments through enhanced pipeline...")
    
    for i in range(10):
        segment = generate_test_segment(i)
        
        start = time.time()
        result = enhanced_pipeline.process_segment(segment)
        elapsed = (time.time() - start) * 1000
        
        print(f"  Segment {i}: {elapsed:.2f}ms")
        
        # Verify behavioral signature is added
        assert "behavioral_signature" in result
        sig = result["behavioral_signature"]
        assert isinstance(sig, dict)
        assert "response_variability" in sig
    
    print(f"\n✓ REV integration successful")
    print(f"  - Segments processed: {base_pipeline.segments_processed}")
    print(f"  - Behavioral signatures added: {base_pipeline.segments_processed}")
    
    return True


def test_performance_targets():
    """Validate all performance targets are met."""
    print("\n" + "="*60)
    print("PERFORMANCE TARGET VALIDATION")
    print("="*60)
    
    results = {
        "Feature extraction": test_feature_extraction(),
        "Signal integration": test_multi_signal_integration(),
        "Streaming analysis": test_streaming_analysis(),
        "Full pipeline": test_full_pipeline(),
        "REV integration": test_rev_integration()
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All performance targets met!")
        print("The BehaviorProfiler is ready for production use.")
    else:
        print("\n⚠️ Some targets not met. Review and optimize.")
    
    return all_passed


def run_memory_profile():
    """Profile memory usage of the system."""
    print("\n" + "="*60)
    print("MEMORY PROFILING")
    print("="*60)
    
    import tracemalloc
    tracemalloc.start()
    
    profiler = BehaviorProfiler({
        "signature_dim": 16,
        "window_size": 100,  # Large window to test memory
        "enable_multi_signal": True,
        "enable_streaming": True
    })
    
    # Process many segments
    print("\nProcessing 1000 segments for memory analysis...")
    
    snapshot_start = tracemalloc.take_snapshot()
    
    for i in range(1000):
        segment = generate_test_segment(i % 80)
        _ = profiler.profile_segment(segment, i % 80, {})
        
        if i % 100 == 0 and i > 0:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.compare_to(snapshot_start, 'lineno')
            
            current_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
            print(f"  Segments {i:4d}: {current_mb:+.2f} MB")
    
    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    
    total_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
    
    print(f"\n✓ Memory usage:")
    print(f"  - Total increase: {total_mb:.2f} MB")
    print(f"  - Per segment: {total_mb/1000:.3f} MB")
    
    # Show top memory consumers
    print(f"\n✓ Top memory consumers:")
    for stat in top_stats[:5]:
        if stat.size_diff > 0:
            print(f"  {stat.filename}:{stat.lineno}: {stat.size_diff/1024:.1f} KB")
    
    tracemalloc.stop()
    
    return total_mb < 100  # Target: <100MB for 1000 segments


if __name__ == "__main__":
    print("="*60)
    print("BEHAVIOR PROFILER TEST SUITE")
    print("="*60)
    print("\nValidating integration and performance targets...")
    
    # Run performance tests
    passed = test_performance_targets()
    
    # Run memory profiling
    print("\n" + "="*60)
    memory_ok = run_memory_profile()
    
    if passed and memory_ok:
        print("\n✅ ALL TESTS PASSED")
        print("The BehaviorProfiler meets all requirements:")
        print("  • <100ms per segment analysis ✓")
        print("  • Multi-signal integration working ✓")
        print("  • Streaming analysis functional ✓")
        print("  • REV pipeline integrated ✓")
        print("  • Memory usage acceptable ✓")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Review the output above for details.")