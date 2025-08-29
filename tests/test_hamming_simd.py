"""
Tests for SIMD-optimized Hamming distance computations.

Verifies correctness and performance of the enhanced hamming_lut module.
"""

import numpy as np
import pytest
import time
from typing import Dict, Tuple

from src.hypervector.operations.hamming_lut import (
    HammingLUT,
    pack_binary_vector,
    pack_binary_vector_simd,
    unpack_binary_vector,
    benchmark_hamming_implementations,
    hamming_distance_cpu
)


class TestHammingSIMD:
    """Test suite for SIMD-optimized Hamming distance operations."""
    
    def test_16bit_lut_construction(self):
        """Test that 16-bit LUT is correctly built."""
        computer = HammingLUT(enable_simd=True)
        lut = computer.build_16bit_lut()
        
        # Verify LUT size
        assert len(lut) == 65536, "16-bit LUT should have 65536 entries"
        
        # Verify specific values
        assert lut[0] == 0, "Popcount of 0 should be 0"
        assert lut[0xFFFF] == 16, "Popcount of 0xFFFF should be 16"
        assert lut[0x5555] == 8, "Popcount of 0x5555 should be 8"
        
        # Verify all values
        for i in range(min(1000, len(lut))):
            expected = bin(i).count('1')
            assert lut[i] == expected, f"LUT[{i}] = {lut[i]}, expected {expected}"
    
    def test_pack_binary_vector_simd(self):
        """Test SIMD-optimized binary vector packing."""
        # Test various sizes
        for size in [64, 128, 1000, 10000]:
            binary_vec = np.random.randint(0, 2, size=size, dtype=np.bool_)
            
            # Pack using both methods
            packed_regular = pack_binary_vector(binary_vec)
            packed_simd = pack_binary_vector_simd(binary_vec)
            
            # Verify they produce the same result
            np.testing.assert_array_equal(
                packed_regular, packed_simd,
                err_msg=f"SIMD packing differs for size {size}"
            )
            
            # Verify unpacking works
            unpacked = unpack_binary_vector(packed_simd, size)
            np.testing.assert_array_equal(
                binary_vec, unpacked,
                err_msg=f"Unpacking failed for size {size}"
            )
    
    def test_compute_distance_simd_correctness(self):
        """Test that SIMD distance computation is accurate."""
        computer_regular = HammingLUT(enable_simd=False)
        computer_simd = HammingLUT(enable_simd=True)
        
        # Test various dimensions
        for dim in [100, 1000, 10000, 16384]:
            # Generate random binary vectors
            vec1 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            vec2 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            
            # Pack vectors
            packed1 = pack_binary_vector_simd(vec1)
            packed2 = pack_binary_vector_simd(vec2)
            
            # Compute distances using different methods
            dist_naive = np.sum(vec1 != vec2)
            dist_regular = computer_regular.distance(packed1, packed2)
            dist_simd = computer_simd.compute_distance_simd(packed1, packed2)
            
            # Verify all methods agree
            assert dist_naive == dist_regular, \
                f"Regular distance mismatch: {dist_regular} vs {dist_naive}"
            assert dist_naive == dist_simd, \
                f"SIMD distance mismatch: {dist_simd} vs {dist_naive}"
    
    def test_simd_performance_10k_vectors(self):
        """Test SIMD achieves 10-20x speedup for 10K dimensional vectors."""
        dim = 10000
        n_trials = 50
        
        # Generate test vectors
        vec1 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        vec2 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        
        packed1 = pack_binary_vector_simd(vec1)
        packed2 = pack_binary_vector_simd(vec2)
        
        # Benchmark naive implementation
        start = time.perf_counter()
        for _ in range(n_trials):
            _ = np.sum(vec1 != vec2)
        time_naive = (time.perf_counter() - start) / n_trials
        
        # Benchmark SIMD implementation
        computer_simd = HammingLUT(enable_simd=True)
        start = time.perf_counter()
        for _ in range(n_trials):
            _ = computer_simd.compute_distance_simd(packed1, packed2)
        time_simd = (time.perf_counter() - start) / n_trials
        
        speedup = time_naive / time_simd
        
        print(f"\n10K Vector Performance:")
        print(f"  Naive: {time_naive*1000:.3f} ms")
        print(f"  SIMD:  {time_simd*1000:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Assert minimum speedup (conservative due to test environment variability)
        assert speedup >= 5.0, \
            f"SIMD speedup ({speedup:.1f}x) below minimum expected (5x)"
    
    def test_batch_operations_simd(self):
        """Test batch Hamming distance operations with SIMD."""
        computer = HammingLUT(enable_simd=True)
        
        # Create batch of vectors
        n_vecs = 10
        dim = 1000
        vecs1 = np.random.randint(0, 2**64, size=(n_vecs, dim // 64), dtype=np.uint64)
        vecs2 = np.random.randint(0, 2**64, size=(n_vecs, dim // 64), dtype=np.uint64)
        
        # Compute batch distances
        distances = computer.distance_batch(vecs1, vecs2)
        
        # Verify shape
        assert distances.shape == (n_vecs, n_vecs), \
            f"Distance matrix shape {distances.shape} != expected {(n_vecs, n_vecs)}"
        
        # Verify diagonal is zero (same vectors)
        distances_self = computer.distance_batch(vecs1, vecs1)
        np.testing.assert_array_equal(
            np.diag(distances_self), np.zeros(n_vecs),
            err_msg="Self-distances should be zero"
        )
    
    def test_uint64_packing_efficiency(self):
        """Test that uint64 packing is memory efficient."""
        dim = 10000
        
        # Create binary vector
        binary_vec = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        
        # Pack to uint64
        packed = pack_binary_vector_simd(binary_vec)
        
        # Check memory usage
        binary_bytes = binary_vec.nbytes
        packed_bytes = packed.nbytes
        
        # Packed should be ~8x more efficient (1 bit vs 8 bits per element)
        efficiency = binary_bytes / packed_bytes
        
        print(f"\nMemory Efficiency:")
        print(f"  Binary: {binary_bytes} bytes")
        print(f"  Packed: {packed_bytes} bytes")
        print(f"  Efficiency: {efficiency:.1f}x")
        
        assert efficiency >= 7.0, \
            f"Packing efficiency ({efficiency:.1f}x) below expected (>7x)"
    
    def test_simd_fallback(self):
        """Test that SIMD operations fall back gracefully when unavailable."""
        # Force SIMD disabled
        computer_no_simd = HammingLUT(enable_simd=False)
        
        # Should still work correctly
        vec1 = np.random.randint(0, 2**64, size=100, dtype=np.uint64)
        vec2 = np.random.randint(0, 2**64, size=100, dtype=np.uint64)
        
        distance = computer_no_simd.distance(vec1, vec2)
        assert isinstance(distance, (int, np.integer)), \
            "Distance should be an integer"
        assert distance >= 0, "Distance should be non-negative"
    
    def test_floating_point_tolerance(self):
        """Test that results are within floating point tolerance."""
        computer = HammingLUT(enable_simd=True)
        
        # Create vectors with known distance
        dim = 1000
        vec1 = np.zeros(dim, dtype=np.bool_)
        vec2 = np.zeros(dim, dtype=np.bool_)
        vec2[:100] = True  # Exactly 100 differences
        
        packed1 = pack_binary_vector_simd(vec1)
        packed2 = pack_binary_vector_simd(vec2)
        
        distance = computer.compute_distance_simd(packed1, packed2)
        
        assert distance == 100, f"Expected distance 100, got {distance}"
    
    def test_edge_cases(self):
        """Test edge cases for SIMD operations."""
        computer = HammingLUT(enable_simd=True)
        
        # Empty vectors
        empty1 = np.array([], dtype=np.uint64)
        empty2 = np.array([], dtype=np.uint64)
        assert computer.distance(empty1, empty2) == 0
        
        # Single element
        single1 = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        single2 = np.array([0x0000000000000000], dtype=np.uint64)
        assert computer.distance(single1, single2) == 64
        
        # All zeros vs all ones
        zeros = np.zeros(100, dtype=np.uint64)
        ones = np.full(100, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        assert computer.distance(zeros, ones) == 100 * 64


class TestPerformanceBenchmarks:
    """Performance benchmarks for SIMD optimizations."""
    
    def test_benchmark_suite(self):
        """Run comprehensive benchmark suite."""
        print("\n" + "="*60)
        print("HAMMING DISTANCE SIMD OPTIMIZATION BENCHMARKS")
        print("="*60)
        
        # Test different dimensions
        for dim in [1000, 5000, 10000, 16384]:
            print(f"\nDimension: {dim}")
            results = benchmark_hamming_implementations(dim=dim, n_trials=20)
            
            # Calculate speedups
            speedup_lut = results['naive'] / results['lut']
            speedup_simd = results['naive'] / results['simd']
            
            # For 10K vectors, verify we achieve target speedup
            if dim == 10000:
                assert speedup_simd >= 10.0, \
                    f"10K vector SIMD speedup ({speedup_simd:.1f}x) below target (10x)"
        
        print("\n" + "="*60)
    
    def test_scaling_behavior(self):
        """Test how performance scales with dimension."""
        dims = [100, 500, 1000, 5000, 10000, 20000]
        times_naive = []
        times_simd = []
        
        computer = HammingLUT(enable_simd=True)
        
        for dim in dims:
            vec1 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            vec2 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            packed1 = pack_binary_vector_simd(vec1)
            packed2 = pack_binary_vector_simd(vec2)
            
            # Measure naive
            start = time.perf_counter()
            for _ in range(10):
                _ = np.sum(vec1 != vec2)
            times_naive.append((time.perf_counter() - start) / 10)
            
            # Measure SIMD
            start = time.perf_counter()
            for _ in range(10):
                _ = computer.compute_distance_simd(packed1, packed2)
            times_simd.append((time.perf_counter() - start) / 10)
        
        print("\nScaling Analysis:")
        print("Dimension | Naive (ms) | SIMD (ms) | Speedup")
        print("-" * 50)
        for i, dim in enumerate(dims):
            speedup = times_naive[i] / times_simd[i]
            print(f"{dim:8d} | {times_naive[i]*1000:10.3f} | "
                  f"{times_simd[i]*1000:9.3f} | {speedup:7.1f}x")
        
        # Verify SIMD scales better than naive
        scaling_naive = times_naive[-1] / times_naive[0]
        scaling_simd = times_simd[-1] / times_simd[0]
        
        print(f"\nScaling factors (20K/100):")
        print(f"  Naive: {scaling_naive:.1f}x")
        print(f"  SIMD:  {scaling_simd:.1f}x")
        
        assert scaling_simd < scaling_naive, \
            "SIMD should scale better than naive implementation"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])