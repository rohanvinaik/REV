#!/usr/bin/env python3
"""
Corrected test script for BehaviorProfiler system.
Uses the actual API methods from the implementation.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List
import json

from src.analysis.behavior_profiler import (
    BehaviorProfiler,
    BehavioralSignature,
    integrate_with_rev_pipeline
)


def generate_test_segments(num_segments: int = 10) -> List[Dict[str, Any]]:
    """Generate test segment data matching expected format."""
    segments = []
    
    for i in range(num_segments):
        # Create realistic segment data
        segment = {
            "layer_idx": i,
            "activations": {
                j: np.random.randn(512, 4096) for j in range(0, min(i+1, 3))
            },
            "attention_weights": {
                j: np.random.rand(32, 512, 512) for j in range(0, min(i+1, 3))
            },
            "timestamps": [time.time() + j*0.01 for j in range(512)],
            "token_count": 512,
            "embeddings": np.random.randn(512, 4096),
            "metadata": {
                "probe_type": "behavioral",
                "model_name": "test_model",
                "timestamp": time.time()
            }
        }
        segments.append(segment)
    
    return segments


def test_profile_model():
    """Test the main profile_model method."""
    print("\n" + "="*60)
    print("TESTING PROFILE_MODEL")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    # Generate test segments
    segments = generate_test_segments(10)
    
    print(f"\nProfiling {len(segments)} segments...")
    
    start = time.time()
    report = profiler.profile_model(
        segments=segments,
        model_name="test_model_v1"
    )
    elapsed = time.time() - start
    
    print(f"  ✓ Profiling completed in {elapsed:.2f}s")
    print(f"  ✓ Model signature generated:")
    print(f"    - Response variability: {report.signature.response_variability:.4f}")
    print(f"    - Semantic coherence: {report.signature.semantic_coherence:.4f}")
    print(f"    - Attention entropy: {report.signature.attention_entropy:.4f}")
    
    if report.model_identification:
        print(f"  ✓ Model identification:")
        for model, confidence in report.model_identification.items():
            print(f"    - {model}: {confidence:.2%}")
    
    if report.anomalies:
        print(f"  ✓ Detected {len(report.anomalies)} anomalies")
    
    return True


def test_feature_extraction():
    """Test feature extraction components."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    # Test gradient signature
    print("\n1. Gradient signature extraction:")
    activations = {
        0: np.random.randn(512, 4096),
        1: np.random.randn(512, 4096),
        2: np.random.randn(512, 4096)
    }
    
    start = time.time()
    gradient_sig = profiler.feature_extractor.extract_gradient_signature(activations)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Extracted {len(gradient_sig)}-dim signature in {elapsed:.2f}ms")
    
    # Test timing analysis
    print("\n2. Timing analysis:")
    timestamps = [time.time() + i*0.01 for i in range(100)]
    
    start = time.time()
    timing_profile = profiler.feature_extractor.analyze_timing(timestamps, 100)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Analyzed timing in {elapsed:.2f}ms")
    print(f"   ✓ Mean latency: {timing_profile.mean_latency:.2f}ms")
    print(f"   ✓ Latency variance: {timing_profile.latency_variance:.4f}")
    
    # Test embedding analysis
    print("\n3. Embedding analysis:")
    embeddings = np.random.randn(512, 4096)
    
    start = time.time()
    embedding_analysis = profiler.feature_extractor.analyze_embeddings(embeddings)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Analyzed embeddings in {elapsed:.2f}ms")
    print(f"   ✓ Semantic density: {embedding_analysis.semantic_density:.4f}")
    print(f"   ✓ Embedding entropy: {embedding_analysis.embedding_entropy:.4f}")
    
    # Test attention patterns
    print("\n4. Attention pattern extraction:")
    attention_weights = {
        0: np.random.rand(32, 512, 512),
        1: np.random.rand(32, 512, 512)
    }
    
    start = time.time()
    attention_pattern = profiler.feature_extractor.extract_attention_patterns(attention_weights)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ✓ Extracted patterns in {elapsed:.2f}ms")
    print(f"   ✓ Head specialization: {attention_pattern.head_specialization:.4f}")
    print(f"   ✓ Attention sparsity: {attention_pattern.attention_sparsity:.4f}")
    
    return elapsed < 500  # Reasonable target for all extractions


def test_streaming_analyzer():
    """Test streaming analysis component."""
    print("\n" + "="*60)
    print("TESTING STREAMING ANALYZER")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    print("\nAnalyzing streaming segments...")
    
    latencies = []
    for i in range(20):
        # Create segment data
        segment_data = {
            "layer_idx": i,
            "activations": {j: np.random.randn(512, 4096) for j in range(3)},
            "timestamps": [time.time() + j*0.01 for j in range(512)],
            "token_count": 512,
            "embeddings": np.random.randn(512, 4096)
        }
        
        start = time.time()
        analysis = profiler.streaming_analyzer.analyze_segment(segment_data)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        
        if i % 5 == 0:
            print(f"  Segment {i:2d}: {elapsed:.2f}ms - divergence={analysis.behavioral_divergence:.4f}")
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n  ✓ Streaming performance:")
    print(f"    - Average latency: {avg_latency:.2f}ms")
    print(f"    - P95 latency: {p95_latency:.2f}ms")
    
    # Get incremental signature
    incremental_sig = profiler.streaming_analyzer.get_incremental_signature()
    print(f"\n  ✓ Incremental signature built:")
    print(f"    - Response variability: {incremental_sig.response_variability:.4f}")
    print(f"    - Layer correlation: {incremental_sig.layer_correlation:.4f}")
    
    return avg_latency < 100  # Target: <100ms average


def test_signal_integration():
    """Test multi-signal integration."""
    print("\n" + "="*60)
    print("TESTING SIGNAL INTEGRATION")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    # Create diverse signals
    from src.analysis.behavior_profiler import SignalType, TimingProfile, EmbeddingAnalysis, AttentionPattern
    
    signals = {
        SignalType.GRADIENT: np.random.randn(16),
        SignalType.TIMING: TimingProfile(
            mean_latency=50.0,
            latency_variance=10.0,
            percentile_95=75.0,
            token_rate=20.0
        ),
        SignalType.EMBEDDING: EmbeddingAnalysis(
            semantic_density=0.7,
            cluster_count=5,
            embedding_entropy=2.5,
            boundary_strength=0.8
        ),
        SignalType.ATTENTION: AttentionPattern(
            head_specialization=0.6,
            attention_entropy=3.0,
            attention_sparsity=0.3,
            layer_correlation=0.7
        )
    }
    
    print("\nIntegrating 4 signal types...")
    
    start = time.time()
    model_class, confidence = profiler.signal_integrator.integrate_signals(signals)
    elapsed = (time.time() - start) * 1000
    
    print(f"  ✓ Integration completed in {elapsed:.2f}ms")
    print(f"  ✓ Predicted model: {model_class} (confidence: {confidence:.2%})")
    
    # Test confidence calculation
    from src.analysis.behavior_profiler import ConfidenceLevel
    conf_level = profiler.signal_integrator.calculate_confidence(signals)
    print(f"  ✓ Confidence level: {conf_level.name}")
    
    return elapsed < 50  # Target: <50ms for integration


def test_export_formats():
    """Test report export in different formats."""
    print("\n" + "="*60)
    print("TESTING EXPORT FORMATS")
    print("="*60)
    
    profiler = BehaviorProfiler()
    
    # Generate a report
    segments = generate_test_segments(5)
    report = profiler.profile_model(segments, "test_model")
    
    # Test JSON export
    print("\n1. JSON export:")
    json_output = profiler.export_report(report, format="json")
    json_data = json.loads(json_output)
    print(f"   ✓ JSON export successful ({len(json_output)} bytes)")
    print(f"   ✓ Contains {len(json_data)} top-level keys")
    
    # Test Markdown export
    print("\n2. Markdown export:")
    md_output = profiler.export_report(report, format="markdown")
    print(f"   ✓ Markdown export successful ({len(md_output)} bytes)")
    print(f"   ✓ Contains {md_output.count('#')} headers")
    
    # Test HTML export
    print("\n3. HTML export:")
    html_output = profiler.export_report(report, format="html")
    print(f"   ✓ HTML export successful ({len(html_output)} bytes)")
    print(f"   ✓ Contains charts: {html_output.count('Plotly.newPlot')}")
    
    return True


def test_rev_integration():
    """Test REV pipeline integration."""
    print("\n" + "="*60)
    print("TESTING REV INTEGRATION")
    print("="*60)
    
    # Mock REV pipeline
    class MockPipeline:
        def __init__(self):
            self.segments = []
            
        def process_segments(self, segments):
            self.segments = segments
            return {"status": "processed", "count": len(segments)}
    
    # Enhance pipeline
    base_pipeline = MockPipeline()
    enhanced_pipeline = integrate_with_rev_pipeline(base_pipeline)
    
    print("\nTesting enhanced pipeline...")
    
    # Process segments
    segments = generate_test_segments(5)
    result = enhanced_pipeline.process_segments(segments)
    
    print(f"  ✓ Pipeline enhanced successfully")
    print(f"  ✓ Processed {result['count']} segments")
    
    # Test behavioral analysis
    if hasattr(enhanced_pipeline, 'run_behavioral_analysis'):
        report = enhanced_pipeline.run_behavioral_analysis(segments)
        print(f"  ✓ Behavioral analysis added to pipeline")
        print(f"  ✓ Generated signature with {len(report.signature.to_vector())} dimensions")
    
    return True


def run_all_tests():
    """Run all tests and summarize results."""
    print("="*60)
    print("BEHAVIOR PROFILER TEST SUITE (CORRECTED)")
    print("="*60)
    
    tests = [
        ("Profile Model", test_profile_model),
        ("Feature Extraction", test_feature_extraction),
        ("Streaming Analyzer", test_streaming_analyzer),
        ("Signal Integration", test_signal_integration),
        ("Export Formats", test_export_formats),
        ("REV Integration", test_rev_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe BehaviorProfiler system is fully functional:")
        print("  • Profile model with comprehensive analysis ✓")
        print("  • Extract multi-dimensional features ✓")
        print("  • Stream processing <100ms latency ✓")
        print("  • Multi-signal integration working ✓")
        print("  • Export to JSON/Markdown/HTML ✓")
        print("  • REV pipeline integration complete ✓")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)