"""
Tests for optimized Hamming distance operations.

This module tests the enhanced Hamming distance implementations including:
- Multiple algorithm implementations
- Adaptive algorithm selection
- Memory-mapped file support
- Advanced similarity metrics
- Performance benchmarks
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import time
from typing import Dict, List

from src.hypervector.hamming import (
    HammingDistanceOptimized,
    AlgorithmType,
    PerformanceProfile,
    create_optimized_computer,
    pack_binary_vector_simd,
    hamming_distance_cpu
)


class TestHammingDistanceOptimized:
    """Test suite for optimized Hamming distance operations."""
    
    @pytest.fixture
    def computer(self):
        """Create optimized Hamming distance computer."""
        return HammingDistanceOptimized(
            enable_gpu=False,  # Disable GPU for consistent testing
            enable_simd=True,
            cache_size=10,
            adaptive=True
        )
    
    @pytest.fixture
    def test_vectors(self):
        """Generate test vectors."""
        np.random.seed(42)
        dim = 1000
        vec1 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        vec2 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        vec3 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
        
        # Pack vectors
        packed1 = pack_binary_vector_simd(vec1)
        packed2 = pack_binary_vector_simd(vec2)
        packed3 = pack_binary_vector_simd(vec3)
        
        return {
            'binary': (vec1, vec2, vec3),
            'packed': (packed1, packed2, packed3),
            'dimension': dim
        }
    
    def test_basic_distance_computation(self, computer, test_vectors):
        """Test basic distance computation."""
        packed1, packed2, _ = test_vectors['packed']
        
        # Compute distance
        distance = computer.distance(packed1, packed2)
        
        # Verify with reference implementation
        ref_distance = hamming_distance_cpu(packed1, packed2)
        assert distance == ref_distance
    
    def test_algorithm_selection(self, computer):
        """Test adaptive algorithm selection."""
        # Small vectors should use LUT
        small_vec1 = pack_binary_vector_simd(np.random.randint(0, 2, 100, dtype=np.bool_))
        small_vec2 = pack_binary_vector_simd(np.random.randint(0, 2, 100, dtype=np.bool_))
        algo = computer._select_algorithm(small_vec1, small_vec2)
        assert algo in [AlgorithmType.LUT_16BIT, AlgorithmType.SIMD_NUMPY]
        
        # Large vectors should use SIMD or parallel
        large_vec1 = pack_binary_vector_simd(np.random.randint(0, 2, 100000, dtype=np.bool_))
        large_vec2 = pack_binary_vector_simd(np.random.randint(0, 2, 100000, dtype=np.bool_))
        algo = computer._select_algorithm(large_vec1, large_vec2)
        assert algo in [AlgorithmType.SIMD_NUMPY, AlgorithmType.NUMBA_PARALLEL]
    
    def test_cache_functionality(self, computer, test_vectors):
        """Test distance caching."""
        packed1, packed2, _ = test_vectors['packed']
        
        # First computation
        start = time.perf_counter()
        distance1 = computer.distance(packed1, packed2)
        time1 = time.perf_counter() - start
        
        # Second computation (should be cached)
        start = time.perf_counter()
        distance2 = computer.distance(packed1, packed2)
        time2 = time.perf_counter() - start
        
        assert distance1 == distance2
        # Cache hit should be faster (though timing can be noisy)
        # Just verify cache was used
        assert len(computer.distance_cache) > 0
    
    def test_distance_matrix(self, computer):
        """Test distance matrix computation."""
        # Create test vectors
        n_vecs = 5
        dim = 100
        vectors = []
        for _ in range(n_vecs):
            vec = np.random.randint(0, 2, dim, dtype=np.bool_)
            vectors.append(pack_binary_vector_simd(vec))
        
        vectors = np.array(vectors)
        
        # Compute distance matrix
        distances = computer.distance_matrix(vectors, block_size=2)
        
        # Verify shape
        assert distances.shape == (n_vecs, n_vecs)
        
        # Verify diagonal is zero (self-distances)
        assert np.all(np.diag(distances) == 0)
        
        # Verify symmetry
        assert np.allclose(distances, distances.T)
    
    def test_similarity_metrics(self, computer):
        """Test different similarity metrics."""
        # Create test vectors
        vec1 = np.array([0b1010101010101010], dtype=np.uint64)
        vec2 = np.array([0b1100110011001100], dtype=np.uint64)
        
        vecs = np.array([vec1, vec2])
        
        # Test Hamming similarity
        ham_sim = computer.similarity_matrix(vecs, metric="hamming", max_distance=64)
        assert ham_sim.shape == (2, 2)
        assert ham_sim[0, 0] == 1.0  # Self-similarity
        assert 0 <= ham_sim[0, 1] <= 1.0
        
        # Test Jaccard similarity
        jac_sim = computer.similarity_matrix(vecs, metric="jaccard")
        assert jac_sim.shape == (2, 2)
        assert jac_sim[0, 0] == 1.0
        
        # Test Cosine similarity
        cos_sim = computer.similarity_matrix(vecs, metric="cosine")
        assert cos_sim.shape == (2, 2)
        assert abs(cos_sim[0, 0] - 1.0) < 1e-6
        
        # Test Dice similarity
        dice_sim = computer.similarity_matrix(vecs, metric="dice")
        assert dice_sim.shape == (2, 2)
        assert dice_sim[0, 0] == 1.0
    
    def test_format_conversion(self, computer):
        """Test vector format conversion."""
        # Start with binary
        binary = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 8, dtype=np.bool_)
        
        # Convert to uint64
        uint64 = computer.convert_format(binary, "uint64")
        assert uint64.dtype == np.uint64
        
        # Convert back to binary
        binary2 = computer.convert_format(uint64, "bool")
        assert np.array_equal(binary, binary2[:len(binary)])
        
        # Convert to uint8
        uint8 = computer.convert_format(binary, "uint8")
        assert uint8.dtype == np.uint8
        assert np.array_equal(binary, uint8.astype(np.bool_))
        
        # Convert to list
        list_format = computer.convert_format(binary, "list")
        assert isinstance(list_format, list)
        assert len(list_format) == len(binary)
    
    def test_save_load_vectors(self, computer):
        """Test saving and loading vectors."""
        # Create test vectors
        vectors = np.random.randint(0, 2**32, size=(10, 16), dtype=np.uint64)
        metadata = {
            'description': 'Test vectors',
            'created': '2024-01-01'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_vectors.bin'
            
            # Save vectors
            computer.save_vectors(vectors, str(filepath), metadata)
            
            # Verify files exist
            assert filepath.exists()
            assert filepath.with_suffix('.meta.json').exists()
            
            # Load vectors
            loaded_vectors, loaded_metadata = computer.load_vectors(str(filepath))
            
            # Verify content
            assert np.array_equal(vectors, loaded_vectors)
            assert loaded_metadata['description'] == metadata['description']
            assert loaded_metadata['shape'] == list(vectors.shape)
    
    def test_memory_mapped_vectors(self, computer):
        """Test memory-mapped file support."""
        # Create large test file
        vectors = np.random.randint(0, 2**32, size=(100, 16), dtype=np.uint64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath1 = Path(tmpdir) / 'vec1.bin'
            filepath2 = Path(tmpdir) / 'vec2.bin'
            
            # Save vectors
            vectors[0].tofile(filepath1)
            vectors[1].tofile(filepath2)
            
            # Compute distance using memory-mapped files
            distance = computer.distance(str(filepath1), str(filepath2))
            
            # Verify with direct computation
            ref_distance = hamming_distance_cpu(vectors[0], vectors[1])
            assert distance == ref_distance
    
    def test_bit_parallel_algorithm(self, computer):
        """Test bit-parallel Hamming distance."""
        vec1 = np.array([0xFFFFFFFFFFFFFFFF, 0x0], dtype=np.uint64)
        vec2 = np.array([0x0, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        
        distance = computer._distance_bit_parallel(vec1, vec2)
        assert distance == 128  # All bits different
        
        # Test identical vectors
        distance = computer._distance_bit_parallel(vec1, vec1)
        assert distance == 0
    
    def test_performance_report(self, computer, test_vectors):
        """Test performance reporting."""
        packed1, packed2, packed3 = test_vectors['packed']
        
        # Run some computations
        computer.distance(packed1, packed2, algorithm=AlgorithmType.LUT_16BIT)
        computer.distance(packed2, packed3, algorithm=AlgorithmType.SIMD_NUMPY)
        
        # Get report
        report = computer.get_performance_report()
        
        # Verify structure
        assert 'cache' in report
        assert 'system' in report
        assert report['system']['simd_enabled'] == True
        
        # Check algorithm times
        if AlgorithmType.LUT_16BIT.value in report:
            assert report[AlgorithmType.LUT_16BIT.value]['n_calls'] >= 1
    
    def test_cleanup(self, computer):
        """Test resource cleanup."""
        # Create some resources
        vec1 = pack_binary_vector_simd(np.random.randint(0, 2, 100, dtype=np.bool_))
        vec2 = pack_binary_vector_simd(np.random.randint(0, 2, 100, dtype=np.bool_))
        
        _ = computer.distance(vec1, vec2)
        assert len(computer.distance_cache) > 0
        
        # Clean up
        computer.cleanup()
        
        # Verify cleanup
        assert len(computer.distance_cache) == 0
        assert len(computer.mmap_handles) == 0


class TestPerformanceProfile:
    """Test performance profiling and adaptive configuration."""
    
    def test_create_optimized_computer(self):
        """Test factory function for creating optimized computer."""
        # Create with auto-detection
        computer = create_optimized_computer()
        assert isinstance(computer, HammingDistanceOptimized)
        assert computer.adaptive == True
        
        # Create with custom profile
        profile = PerformanceProfile(
            vector_size=1000000,
            batch_size=1000,
            available_memory=16 * 1024 * 1024 * 1024,
            has_gpu=False,
            has_numba=True
        )
        computer = create_optimized_computer(profile)
        assert computer.enable_gpu == False
    
    def test_benchmark_functionality(self):
        """Test benchmarking functionality."""
        computer = HammingDistanceOptimized(
            enable_gpu=False,
            enable_simd=True,
            adaptive=False
        )
        
        # Run benchmark on small dimensions for speed
        results = computer.benchmark(
            dimensions=[100, 500],
            algorithms=[AlgorithmType.NAIVE, AlgorithmType.LUT_16BIT]
        )
        
        # Verify structure
        assert AlgorithmType.NAIVE.value in results
        assert AlgorithmType.LUT_16BIT.value in results
        
        # Verify measurements exist
        for algo in results:
            assert 100 in results[algo]
            assert 500 in results[algo]
            assert results[algo][100] > 0  # Time should be positive
        
        # LUT should be faster than naive
        assert results[AlgorithmType.LUT_16BIT.value][500] < results[AlgorithmType.NAIVE.value][500]


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from raw vectors to similarity scores."""
        # Create computer
        computer = create_optimized_computer()
        
        # Generate synthetic hypervectors
        n_models = 3
        n_features = 10000
        hypervectors = []
        
        for _ in range(n_models):
            # Simulate sparse hypervector (1% density)
            vec = np.random.random(n_features) < 0.01
            hypervectors.append(vec)
        
        # Convert to packed format
        packed_vectors = []
        for hv in hypervectors:
            packed = pack_binary_vector_simd(hv)
            packed_vectors.append(packed)
        
        packed_array = np.array(packed_vectors)
        
        # Compute distance matrix
        distances = computer.distance_matrix(packed_array)
        
        # Convert to similarities
        similarities = computer.similarity_matrix(
            packed_array,
            metric="hamming",
            max_distance=n_features
        )
        
        # Verify properties
        assert distances.shape == (n_models, n_models)
        assert similarities.shape == (n_models, n_models)
        assert np.all(np.diag(distances) == 0)
        assert np.all(np.diag(similarities) == 1.0)
        
        # Find closest matches
        for i in range(n_models):
            closest_idx = np.argmin(distances[i, :])
            assert closest_idx == i  # Should match itself
        
        # Clean up
        computer.cleanup()
    
    def test_large_scale_computation(self):
        """Test handling of large-scale computations."""
        computer = HammingDistanceOptimized(
            enable_gpu=False,
            enable_simd=True,
            cache_size=100,
            adaptive=True
        )
        
        # Create larger dataset
        n_vectors = 50
        dimension = 5000
        
        vectors = []
        for _ in range(n_vectors):
            vec = np.random.randint(0, 2, dimension, dtype=np.bool_)
            vectors.append(pack_binary_vector_simd(vec))
        
        vectors = np.array(vectors)
        
        # Compute with blocking for efficiency
        distances = computer.distance_matrix(
            vectors,
            block_size=10,
            symmetric=True
        )
        
        # Verify results
        assert distances.shape == (n_vectors, n_vectors)
        assert np.allclose(distances, distances.T)  # Symmetric
        
        # Test different similarity metrics
        for metric in ["hamming", "jaccard", "cosine", "dice"]:
            similarities = computer.similarity_matrix(
                vectors[:10],  # Use subset for speed
                metric=metric
            )
            assert similarities.shape == (10, 10)
            assert np.all(similarities >= 0) and np.all(similarities <= 1)
        
        # Get performance report
        report = computer.get_performance_report()
        assert 'system' in report
        
        computer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])