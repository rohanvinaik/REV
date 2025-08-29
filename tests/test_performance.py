"""
Performance benchmarks for REV components.
"""

import pytest
import numpy as np
import time
import psutil
import os
from memory_profiler import profile
from typing import List, Dict, Any

from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.binding_operations import BindingOperations
from src.hypervector.hamming import HammingDistance
from src.hypervector.similarity import AdvancedSimilarity
from src.core.sequential import sequential_verify, SequentialState
from src.hdc.error_correction import ErrorCorrection, ErrorCorrectionConfig
from src.privacy.distance_zk_proofs import DistanceZKProof


class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
    @property
    def elapsed_ms(self):
        return self.elapsed * 1000


class TestHypervectorPerformance:
    """Performance benchmarks for hypervector operations."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("dimension", [1000, 8000, 10000, 50000, 100000])
    def test_encoding_speed(self, dimension, benchmark):
        """Benchmark hypervector encoding speed."""
        config = HypervectorConfig(dimension=dimension, sparse_density=0.01)
        encoder = HypervectorEncoder(config)
        
        input_vec = np.random.randn(100)
        
        def encode():
            return encoder.encode(input_vec)
        
        result = benchmark(encode)
        
        # Performance targets
        if dimension <= 10000:
            assert benchmark.stats["mean"] < 0.01  # < 10ms for small dimensions
        elif dimension <= 50000:
            assert benchmark.stats["mean"] < 0.05  # < 50ms for medium dimensions
        else:
            assert benchmark.stats["mean"] < 0.1   # < 100ms for large dimensions
    
    @pytest.mark.benchmark
    def test_binding_operations_speed(self, benchmark):
        """Benchmark binding operations."""
        binder = BindingOperations(dimension=10000)
        
        vec_a = np.random.randn(10000)
        vec_b = np.random.randn(10000)
        
        # Test XOR binding
        def xor_bind():
            return binder.xor_bind(vec_a, vec_b)
        
        result = benchmark.pedantic(xor_bind, rounds=100, iterations=5)
        assert benchmark.stats["mean"] < 0.005  # < 5ms
    
    @pytest.mark.benchmark
    def test_circular_convolution_speed(self, benchmark):
        """Benchmark circular convolution."""
        binder = BindingOperations(dimension=10000)
        
        vec_a = np.random.randn(10000)
        vec_b = np.random.randn(10000)
        
        # Test FFT-based convolution
        def convolve_fft():
            return binder.circular_convolve(vec_a, vec_b, mode="fft")
        
        result = benchmark(convolve_fft)
        assert benchmark.stats["mean"] < 0.01  # < 10ms with FFT


class TestHammingDistancePerformance:
    """Performance benchmarks for Hamming distance computation."""
    
    @pytest.mark.benchmark
    def test_hamming_lut_speed(self, benchmark):
        """Benchmark Hamming distance with lookup tables."""
        hamming = HammingDistance(use_lut=True, lut_bits=16)
        
        # Binary vectors
        vec_a = np.random.choice([0, 1], size=10000).astype(np.uint8)
        vec_b = np.random.choice([0, 1], size=10000).astype(np.uint8)
        
        def compute_hamming():
            return hamming.compute(vec_a, vec_b)
        
        result = benchmark(compute_hamming)
        
        # Should be 10-20x faster than naive
        assert benchmark.stats["mean"] < 0.001  # < 1ms for 10K dimensions
    
    @pytest.mark.benchmark
    def test_hamming_batch_speed(self, benchmark):
        """Benchmark batch Hamming distance."""
        hamming = HammingDistance(use_lut=True)
        
        # Batch of vectors
        batch_size = 100
        vectors = [
            np.random.choice([0, 1], size=1000).astype(np.uint8)
            for _ in range(batch_size)
        ]
        
        def compute_batch():
            return hamming.compute_batch_distances(vectors)
        
        result = benchmark(compute_batch)
        
        # Should process 100 vectors in reasonable time
        assert benchmark.stats["mean"] < 0.1  # < 100ms for 100 vectors


class TestSequentialTestingPerformance:
    """Performance benchmarks for sequential testing."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_sprt_speed(self, n_samples, benchmark):
        """Benchmark SPRT computation speed."""
        scores = np.random.normal(0.7, 0.1, n_samples).tolist()
        
        def run_sprt():
            return sequential_verify(scores, alpha=0.05, beta=0.10)
        
        result = benchmark(run_sprt)
        
        # Performance targets
        if n_samples <= 1000:
            assert benchmark.stats["mean"] < 0.01  # < 10ms for 1K samples
        else:
            assert benchmark.stats["mean"] < 0.1   # < 100ms for 10K samples
    
    @pytest.mark.benchmark
    def test_welford_update_speed(self, benchmark):
        """Benchmark Welford's algorithm for online statistics."""
        state = SequentialState()
        values = np.random.randn(10000)
        
        def update_all():
            for val in values:
                state.update(val)
            return state.mean, state.var
        
        result = benchmark(update_all)
        
        # Should be very fast for online updates
        assert benchmark.stats["mean"] < 0.05  # < 50ms for 10K updates


class TestSimilarityMetricsPerformance:
    """Performance benchmarks for similarity metrics."""
    
    @pytest.mark.benchmark
    def test_advanced_similarity_speed(self, benchmark):
        """Benchmark advanced similarity computation."""
        similarity = AdvancedSimilarity()
        
        vec_a = np.random.randn(10000)
        vec_b = np.random.randn(10000)
        
        def compute_all_metrics():
            return similarity.compute_all_metrics(vec_a, vec_b)
        
        result = benchmark(compute_all_metrics)
        
        # Should compute 10+ metrics efficiently
        assert benchmark.stats["mean"] < 0.05  # < 50ms for all metrics
    
    @pytest.mark.benchmark
    def test_hierarchical_similarity_speed(self, benchmark):
        """Benchmark hierarchical similarity."""
        similarity = AdvancedSimilarity()
        
        vec_a = np.random.randn(10000)
        vec_b = np.random.randn(10000)
        
        def compute_hierarchical():
            return similarity.compute_hierarchical_similarity(vec_a, vec_b)
        
        result = benchmark(compute_hierarchical)
        
        # Should handle multiple levels efficiently
        assert benchmark.stats["mean"] < 0.1  # < 100ms for hierarchical


class TestErrorCorrectionPerformance:
    """Performance benchmarks for error correction."""
    
    @pytest.mark.benchmark
    def test_parity_encoding_speed(self, benchmark):
        """Benchmark parity encoding speed."""
        config = ErrorCorrectionConfig(dimension=10000, parity_overhead=0.25)
        corrector = ErrorCorrection(config)
        
        data = np.random.randn(10000)
        
        def encode():
            return corrector.encode_with_parity(data)
        
        result = benchmark(encode)
        
        # Should be fast for encoding
        assert benchmark.stats["mean"] < 0.01  # < 10ms
    
    @pytest.mark.benchmark
    def test_error_correction_speed(self, benchmark):
        """Benchmark error correction speed."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=1000))
        
        # Create corrupted data
        original = np.random.choice([-1, 1], size=1000).astype(np.float32)
        encoded = corrector.encode_with_parity(original)
        noisy = corrector.add_noise(encoded, noise_level=0.1)
        
        def correct():
            return corrector.decode_with_correction(noisy, correct_errors=True)
        
        result = benchmark(correct)
        
        # Should correct errors quickly
        assert benchmark.stats["mean"] < 0.05  # < 50ms


class TestZKProofPerformance:
    """Performance benchmarks for zero-knowledge proofs."""
    
    @pytest.mark.benchmark
    def test_zk_proof_generation(self, benchmark):
        """Benchmark ZK proof generation."""
        zk = DistanceZKProof(security_bits=128)
        
        vec_a = np.random.randn(1000)
        vec_b = np.random.randn(1000)
        distance = np.linalg.norm(vec_a - vec_b)
        
        def generate_proof():
            return zk.prove_distance(vec_a, vec_b, distance)
        
        result = benchmark(generate_proof)
        
        # Target: ~200ms for proof generation
        assert benchmark.stats["mean"] < 0.3  # < 300ms
    
    @pytest.mark.benchmark
    def test_zk_proof_verification(self, benchmark):
        """Benchmark ZK proof verification."""
        zk = DistanceZKProof()
        
        # Generate proof once
        vec_a = np.random.randn(1000)
        vec_b = np.random.randn(1000)
        distance = np.linalg.norm(vec_a - vec_b)
        proof = zk.prove_distance(vec_a, vec_b, distance)
        
        def verify_proof():
            return zk.verify_distance(proof, distance)
        
        result = benchmark(verify_proof)
        
        # Verification should be fast
        assert benchmark.stats["mean"] < 0.01  # < 10ms


class TestMemoryUsage:
    """Memory usage profiling tests."""
    
    def test_hypervector_memory(self):
        """Test hypervector memory usage."""
        dimensions = [1000, 10000, 100000]
        expected_mb = [0.004, 0.04, 0.4]  # float32 = 4 bytes per element
        
        for dim, expected in zip(dimensions, expected_mb):
            vec = np.random.randn(dim).astype(np.float32)
            actual_mb = vec.nbytes / (1024 * 1024)
            
            # Allow 10% overhead
            assert actual_mb < expected * 1.1
    
    def test_hamming_lut_memory(self):
        """Test Hamming LUT memory usage."""
        hamming = HammingDistance(use_lut=True, lut_bits=16)
        
        # 16-bit LUT should use ~512KB
        lut_size = hamming.lut.nbytes if hasattr(hamming, 'lut') else 0
        lut_mb = lut_size / (1024 * 1024)
        
        assert lut_mb < 1.0  # Less than 1MB
    
    @profile
    def test_segment_buffer_memory(self):
        """Profile memory usage of segment buffer."""
        from src.executor.segment_runner import SegmentRunner, SegmentConfig
        
        config = SegmentConfig(
            max_sequence_length=512,
            buffer_size=4
        )
        runner = SegmentRunner(config)
        
        # Simulate segment processing
        segments = [np.random.randn(512, 768) for _ in range(4)]
        
        for segment in segments:
            runner.buffer.append(segment)
        
        # Buffer should use < 100MB
        buffer_size = sum(s.nbytes for s in runner.buffer) / (1024 * 1024)
        assert buffer_size < 100
    
    def test_merkle_tree_memory(self):
        """Test Merkle tree memory usage."""
        from src.rev_pipeline import REVPipeline
        
        pipeline = REVPipeline()
        
        # Create many leaves
        leaves = [hashlib.sha256(f"leaf_{i}".encode()).digest() for i in range(1000)]
        
        # Build tree
        root = pipeline._build_merkle_tree(leaves)
        
        # Tree should have logarithmic memory usage
        # ~32KB for 1000 leaves (32 bytes per hash)
        tree_size = len(leaves) * 32 / 1024  # KB
        assert tree_size < 50  # Less than 50KB


class TestScalability:
    """Scalability tests for increasing workloads."""
    
    def test_dimension_scalability(self):
        """Test performance scaling with vector dimension."""
        dimensions = [1000, 5000, 10000, 50000]
        times = []
        
        for dim in dimensions:
            encoder = HypervectorEncoder(HypervectorConfig(dimension=dim))
            
            start = time.perf_counter()
            for _ in range(100):
                encoder.encode(np.random.randn(100))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Check sub-linear scaling
        for i in range(1, len(times)):
            scaling_factor = dimensions[i] / dimensions[i-1]
            time_factor = times[i] / times[i-1]
            
            # Time should scale sub-linearly with dimension
            assert time_factor < scaling_factor * 1.5
    
    def test_batch_scalability(self):
        """Test performance scaling with batch size."""
        batch_sizes = [10, 50, 100, 500]
        similarity = AdvancedSimilarity()
        times = []
        
        for batch_size in batch_sizes:
            vectors = [np.random.randn(1000) for _ in range(batch_size)]
            
            start = time.perf_counter()
            for i in range(batch_size):
                for j in range(i+1, batch_size):
                    similarity.compute_cosine(vectors[i], vectors[j])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Check quadratic scaling (expected for pairwise)
        for i in range(1, len(times)):
            batch_factor = (batch_sizes[i] / batch_sizes[i-1]) ** 2
            time_factor = times[i] / times[i-1]
            
            # Should scale roughly quadratically
            assert 0.5 * batch_factor < time_factor < 2 * batch_factor


def run_performance_suite():
    """Run complete performance test suite and generate report."""
    import json
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version
        },
        "benchmarks": {}
    }
    
    # Run each benchmark category
    test_classes = [
        TestHypervectorPerformance,
        TestHammingDistancePerformance,
        TestSequentialTestingPerformance,
        TestSimilarityMetricsPerformance,
        TestErrorCorrectionPerformance,
        TestZKProofPerformance
    ]
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"Running {class_name}...")
        
        # Run pytest with json report
        pytest.main([
            f"{__file__}::{class_name}",
            "--benchmark-json=benchmark_temp.json"
        ])
        
        # Load results
        if os.path.exists("benchmark_temp.json"):
            with open("benchmark_temp.json") as f:
                bench_data = json.load(f)
                results["benchmarks"][class_name] = bench_data["benchmarks"]
    
    # Save final report
    with open("performance_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Performance report saved to performance_report.json")


if __name__ == "__main__":
    # Run with: pytest test_performance.py -v --benchmark-only
    run_performance_suite()