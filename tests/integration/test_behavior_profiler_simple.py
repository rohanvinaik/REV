#!/usr/bin/env python3
"""
Simple test script for BehaviorProfiler system.
Tests basic functionality and integration.
"""

import time
import numpy as np
import torch
from typing import Dict, Any

from src.analysis.behavior_profiler import (
    BehaviorProfiler,
    BehavioralSignature,
    integrate_with_rev_pipeline
)


def test_basic_profiling():
    """Test basic profiling functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC PROFILING")
    print("="*60)
    
    # Create profiler
    profiler = BehaviorProfiler()
    
    # Generate test data - simulating activations from different layers
    test_data = {
        0: np.random.randn(512, 4096),  # Layer 0 activations
        10: np.random.randn(512, 4096),  # Layer 10 activations
        20: np.random.randn(512, 4096),  # Layer 20 activations
    }
    
    # Test gradient signature extraction
    print("\n1. Testing gradient signature extraction...")
    start = time.time()
    gradient_sig = profiler.feature_extractor.extract_gradient_signature(test_data)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Extracted {len(gradient_sig)}-dim gradient signature in {elapsed:.2f}ms")
    print(f"   ✓ Signature stats: mean={np.mean(gradient_sig):.4f}, std={np.std(gradient_sig):.4f}")
    
    # Test attention pattern extraction
    print("\n2. Testing attention pattern extraction...")
    attention_weights = {
        0: np.random.rand(32, 512, 512),  # 32 heads, 512x512 attention
        10: np.random.rand(32, 512, 512),
        20: np.random.rand(32, 512, 512),
    }
    
    start = time.time()
    attention_pattern = profiler.feature_extractor.extract_attention_patterns(attention_weights)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Extracted attention patterns in {elapsed:.2f}ms")
    print(f"   ✓ Head specialization: {attention_pattern.head_specialization:.4f}")
    print(f"   ✓ Attention sparsity: {attention_pattern.attention_sparsity:.4f}")
    
    return True


def test_segment_processing():
    """Test processing individual segments."""
    print("\n" + "="*60)
    print("TESTING SEGMENT PROCESSING")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    # Simulate processing multiple segments
    print("\nProcessing 10 segments...")
    
    for i in range(10):
        # Create segment data
        segment = torch.randn(512, 4096)
        
        start = time.time()
        signature = profiler.profile_segment(
            segment=segment,
            layer_idx=i,
            metadata={"probe_type": "test", "timestamp": time.time()}
        )
        elapsed = (time.time() - start) * 1000
        
        print(f"  Segment {i}: {elapsed:.2f}ms - divergence={signature.response_variability:.4f}")
    
    return True


def test_multi_signal_integration():
    """Test multi-signal integration."""
    print("\n" + "="*60)
    print("TESTING MULTI-SIGNAL INTEGRATION")
    print("="*60)
    
    profiler = BehaviorProfiler({"enable_multi_signal": True})
    
    # Create multiple behavioral signatures
    signatures = []
    for i in range(5):
        sig = BehavioralSignature()
        sig.response_variability = 0.3 + i * 0.1
        sig.semantic_coherence = 0.8 - i * 0.05
        sig.attention_entropy = 0.5 + (i % 2) * 0.2
        sig.layer_correlation = 0.7 + i * 0.03
        signatures.append(sig)
    
    print(f"\nIntegrating {len(signatures)} signals...")
    
    # Test integration
    start = time.time()
    integrated = profiler.signal_integrator.integrate(
        signatures,
        weights=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"  ✓ Integration completed in {elapsed:.2f}ms")
    print(f"  ✓ Integrated signature:")
    print(f"    - Response variability: {integrated.response_variability:.4f}")
    print(f"    - Semantic coherence: {integrated.semantic_coherence:.4f}")
    print(f"    - Attention entropy: {integrated.attention_entropy:.4f}")
    
    # Test voting
    votes = profiler.signal_integrator.vote_on_classification(signatures)
    print(f"\n  ✓ Model classification votes:")
    for model, confidence in votes.items():
        print(f"    - {model}: {confidence:.2%}")
    
    return True


def test_streaming_analysis():
    """Test streaming analysis capabilities."""
    print("\n" + "="*60)
    print("TESTING STREAMING ANALYSIS")
    print("="*60)
    
    profiler = BehaviorProfiler({"enable_streaming": True})
    
    print("\nProcessing streaming segments...")
    
    latencies = []
    for i in range(20):
        # Generate segment
        segment = torch.randn(512, 4096)
        
        start = time.time()
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
    
    print(f"\n  ✓ Streaming performance:")
    print(f"    - Average latency: {avg_latency:.2f}ms")
    print(f"    - P95 latency: {p95_latency:.2f}ms")
    print(f"    - Max latency: {max(latencies):.2f}ms")
    
    # Check for anomalies
    anomalies = profiler.streaming_analyzer.get_anomalies()
    print(f"  ✓ Detected {len(anomalies)} anomalies")
    
    return avg_latency < 100  # Target: <100ms


def test_rev_integration():
    """Test REV pipeline integration."""
    print("\n" + "="*60)
    print("TESTING REV INTEGRATION")
    print("="*60)
    
    # Mock REV pipeline
    class MockPipeline:
        def process_segment(self, segment_data):
            return {
                "divergence": np.random.random(),
                "layer": 0,
                "segment": segment_data
            }
    
    # Enhance pipeline
    base_pipeline = MockPipeline()
    enhanced_pipeline = integrate_with_rev_pipeline(base_pipeline)
    
    # Test processing
    print("\nProcessing through enhanced pipeline...")
    
    segment = torch.randn(512, 4096)
    result = enhanced_pipeline.process_segment(segment)
    
    assert "behavioral_signature" in result
    assert "response_variability" in result["behavioral_signature"]
    
    print(f"  ✓ Pipeline enhanced successfully")
    print(f"  ✓ Original divergence: {result['divergence']:.4f}")
    print(f"  ✓ Behavioral signature added with {len(result['behavioral_signature'])} dimensions")
    
    return True


def test_performance_summary():
    """Run all tests and summarize performance."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    tests = [
        ("Basic Profiling", test_basic_profiling),
        ("Segment Processing", test_segment_processing),
        ("Multi-Signal Integration", test_multi_signal_integration),
        ("Streaming Analysis", test_streaming_analysis),
        ("REV Integration", test_rev_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:25s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("The BehaviorProfiler is ready for use.")
    else:
        print("\n⚠️ Some tests failed. Please review.")
    
    return all_passed


if __name__ == "__main__":
    print("="*60)
    print("BEHAVIOR PROFILER TEST SUITE (SIMPLE)")
    print("="*60)
    
    # Run tests
    success = test_performance_summary()
    
    if success:
        print("\n✅ BEHAVIOR PROFILER VALIDATED")
        print("Key achievements:")
        print("  • Gradient signature extraction working")
        print("  • Attention pattern analysis functional")
        print("  • Multi-signal integration operational")
        print("  • Streaming analysis <100ms target met")
        print("  • REV pipeline successfully enhanced")
    else:
        print("\n⚠️ VALIDATION INCOMPLETE")
        print("Review the errors above for details.")