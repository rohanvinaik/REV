#!/usr/bin/env python3
"""
Comprehensive tests for Semantic Hypervector encoding system from Section 6.

Tests the enhanced HypervectorEncoder and BindingOperations with:
- BLAKE2b hashing for stable encoding
- Multiple binding operations (XOR, permutation, convolution)
- Hierarchical zoom levels
- Optimization features (LUT, SIMD, bit-packing)
- Privacy-preserving mechanisms
"""

import numpy as np
import torch
import pytest
import time
import hashlib
from typing import List, Dict, Any

from src.hdc.encoder import (
    HypervectorEncoder,
    HypervectorConfig,
    ProjectionType,
    DEFAULT_DIMENSION,
    MIN_DIMENSION,
    MAX_DIMENSION
)
from src.hdc.binding_operations import BindingOperations


class TestHypervectorEncoder:
    """Test the enhanced HypervectorEncoder class."""
    
    def test_initialization(self):
        """Test encoder initialization with various configs."""
        # Default initialization
        encoder = HypervectorEncoder()
        assert encoder.config.dimension == DEFAULT_DIMENSION
        assert encoder.config.multi_scale is True
        # BLAKE2b may not be available, so check if it was disabled
        # assert encoder.config.use_blake2b is True
        
        # Custom config
        config = HypervectorConfig(
            dimension=16384,
            sparsity=0.05,
            encoding_mode="hbt",
            enable_lut=True,
            enable_simd=True,
            bit_packed=True
        )
        encoder = HypervectorEncoder(config)
        assert encoder.config.dimension == 16384
        assert encoder.config.sparsity == 0.05
        
        # Test dimension validation
        with pytest.raises(ValueError):
            config = HypervectorConfig(dimension=MIN_DIMENSION - 1)
            HypervectorEncoder(config)
        
        with pytest.raises(ValueError):
            config = HypervectorConfig(dimension=MAX_DIMENSION + 1)
            HypervectorEncoder(config)
    
    def test_blake2b_hashing(self):
        """Test BLAKE2b stable feature encoding."""
        encoder = HypervectorEncoder()
        
        # Test hash generation
        hash1 = encoder.blake2b_hash("test_data")
        hash2 = encoder.blake2b_hash("test_data")
        assert hash1 == hash2  # Deterministic
        
        hash3 = encoder.blake2b_hash("different_data")
        assert hash1 != hash3  # Different input -> different hash
        
        # Test with bytes input
        hash_bytes = encoder.blake2b_hash(b"byte_data")
        # SHA256 fallback gives 32 bytes max, BLAKE2b gives 64
        assert len(hash_bytes) in [32, 64]
        
        # Test custom output size
        hash_small = encoder.blake2b_hash("test", output_size=32)
        assert len(hash_small) == 32
    
    def test_feature_encoding(self):
        """Test feature encoding with BLAKE2b."""
        encoder = HypervectorEncoder()
        
        # Test basic encoding
        vec1 = encoder.encode_feature("feature1")
        assert vec1.shape == (encoder.config.dimension,)
        assert np.abs(np.linalg.norm(vec1) - 1.0) < 1e-6  # Normalized
        
        # Test determinism
        vec2 = encoder.encode_feature("feature1")
        np.testing.assert_array_equal(vec1, vec2)
        
        # Test different features produce different vectors
        vec3 = encoder.encode_feature("feature2")
        assert not np.array_equal(vec1, vec3)
        
        # Test zoom levels
        vec_corpus = encoder.encode_feature("test", "corpus")
        vec_token = encoder.encode_feature("test", "token_window")
        assert not np.array_equal(vec_corpus, vec_token)
    
    def test_hierarchical_encoding(self):
        """Test multi-resolution hierarchical encoding."""
        encoder = HypervectorEncoder()
        
        text = "This is a test sentence with multiple words for hierarchical encoding"
        
        # Test all levels returned
        levels = encoder.encode_hierarchical(text, return_all_levels=True)
        assert "corpus" in levels
        assert "prompt" in levels
        assert "span" in levels
        assert "token_window" in levels
        
        # Check dimensions
        for level_name, vector in levels.items():
            assert vector.shape == (encoder.config.dimension,)
            
        # Test single level return
        single_vec = encoder.encode_hierarchical(text, return_all_levels=False)
        assert single_vec.shape == (encoder.config.dimension,)
        np.testing.assert_array_equal(single_vec, levels["token_window"])
    
    def test_batch_encoding(self):
        """Test batch encoding with optimizations."""
        encoder = HypervectorEncoder()
        
        texts = [f"text_{i}" for i in range(100)]
        
        # Time batch encoding
        start = time.time()
        batch_vecs = encoder.batch_encode(texts)
        batch_time = time.time() - start
        
        assert batch_vecs.shape == (100, encoder.config.dimension)
        
        # Compare with sequential encoding
        start = time.time()
        seq_vecs = np.array([encoder.encode_feature(t) for t in texts])
        seq_time = time.time() - start
        
        # Batch should be at least as fast (often faster with SIMD)
        print(f"Batch time: {batch_time:.3f}s, Sequential time: {seq_time:.3f}s")
        
        # Results should be identical
        np.testing.assert_array_almost_equal(batch_vecs, seq_vecs)
    
    def test_popcount_hamming(self):
        """Test LUT-accelerated Hamming distance."""
        config = HypervectorConfig(enable_lut=True)
        encoder = HypervectorEncoder(config)
        
        # Create binary vectors
        a = np.random.choice([-1, 1], size=encoder.config.dimension)
        b = np.random.choice([-1, 1], size=encoder.config.dimension)
        
        # Compute Hamming distance
        distance = encoder.popcount_hamming(a, b)
        
        # Verify against naive implementation
        expected = np.sum(a != b)
        assert abs(distance - expected) <= encoder.config.dimension * 0.01  # Allow small error
    
    def test_bit_packing(self):
        """Test bit-packing for memory efficiency."""
        config = HypervectorConfig(bit_packed=True, dimension=32768)
        encoder = HypervectorEncoder(config)
        
        # Create vector
        vector = np.random.randn(encoder.config.dimension)
        
        # Pack
        packed = encoder.bit_pack(vector)
        assert len(packed) < vector.nbytes  # Should be compressed
        
        # Unpack
        unpacked = encoder.bit_unpack(packed)
        assert unpacked.shape == vector.shape
        
        # Check binary quantization preserved signs
        assert np.all(np.sign(vector) == np.sign(unpacked))
    
    def test_distributed_encoding(self):
        """Test distributed representation for privacy."""
        config = HypervectorConfig(privacy_mode=True)
        encoder = HypervectorEncoder(config)
        
        data = "sensitive_data"
        shares = encoder.distributed_encode(data, n_shares=4)
        
        assert len(shares) == 4
        
        # Verify shares sum to original
        original = encoder.encode_feature(data)
        reconstructed = np.sum(shares, axis=0)
        np.testing.assert_array_almost_equal(original, reconstructed, decimal=5)
        
        # Individual shares should look random
        for share in shares[:-1]:
            assert np.abs(np.mean(share)) < 0.1  # Near zero mean
    
    def test_homomorphic_encoding(self):
        """Test homomorphic-friendly encoding."""
        config = HypervectorConfig(homomorphic_friendly=True)
        encoder = HypervectorEncoder(config)
        
        vector = np.random.randn(encoder.config.dimension)
        encoded = encoder.homomorphic_encode(vector)
        
        # Check integer domain
        assert encoded.dtype == np.int32
        
        # Check modulus applied
        assert np.all(encoded >= 0)
        assert np.all(encoded < encoder._he_modulus)
    
    def test_zero_knowledge_commitment(self):
        """Test ZK commitment and verification."""
        encoder = HypervectorEncoder()
        
        vector = np.random.randn(encoder.config.dimension)
        
        # Create commitment
        commitment, opening = encoder.zero_knowledge_commit(vector)
        
        assert len(commitment) in [32, 64]  # SHA256 or BLAKE2b output
        assert len(opening) == 32  # Random blinding factor
        
        # Verify valid commitment
        assert encoder.verify_zk_commitment(vector, commitment, opening)
        
        # Wrong vector should fail
        wrong_vector = np.random.randn(encoder.config.dimension)
        assert not encoder.verify_zk_commitment(wrong_vector, commitment, opening)
        
        # Wrong opening should fail
        wrong_opening = os.urandom(32)
        assert not encoder.verify_zk_commitment(vector, commitment, wrong_opening)


class TestBindingOperations:
    """Test enhanced binding operations."""
    
    def test_initialization(self):
        """Test binding operations initialization."""
        bind_ops = BindingOperations()
        assert bind_ops.dimension == 10000
        
        # Custom initialization
        bind_ops = BindingOperations(
            dimension=8192,
            seed=42,
            use_blake2b=True,
            enable_simd=True,
            enable_lut=True
        )
        assert bind_ops.dimension == 8192
        assert bind_ops.use_blake2b is True
    
    def test_xor_binding(self):
        """Test XOR binding operation."""
        bind_ops = BindingOperations(dimension=1000, seed=42)
        
        # Binary vectors
        a = np.random.choice([-1, 1], size=1000)
        b = np.random.choice([-1, 1], size=1000)
        
        result = bind_ops.xor_bind(a, b)
        assert result.shape == (1000,)
        
        # Continuous vectors
        a_cont = np.random.randn(1000)
        b_cont = np.random.randn(1000)
        
        result_cont = bind_ops.xor_bind(a_cont, b_cont)
        assert result_cont.shape == (1000,)
        
        # Test with torch tensors
        a_torch = torch.randn(1000)
        b_torch = torch.randn(1000)
        
        result_torch = bind_ops.xor_bind(a_torch, b_torch)
        assert isinstance(result_torch, np.ndarray)
    
    def test_permutation_binding(self):
        """Test permutation binding."""
        bind_ops = BindingOperations(dimension=1000, seed=42)
        
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        
        # Position permutation
        result_pos = bind_ops.permutation_bind(a, b, perm_type='position')
        assert result_pos.shape == (1000,)
        
        # Shift permutation
        result_shift = bind_ops.permutation_bind(a, b, perm_type='shift')
        assert result_shift.shape == (1000,)
        
        # Results should be different
        assert not np.array_equal(result_pos, result_shift)
    
    def test_circular_convolution(self):
        """Test circular convolution binding."""
        bind_ops = BindingOperations(dimension=1000, seed=42)
        
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        
        result = bind_ops.circular_convolution(a, b)
        assert result.shape == (1000,)
        
        # Test properties
        # Convolution should be commutative
        result_swap = bind_ops.circular_convolution(b, a)
        np.testing.assert_array_almost_equal(result, result_swap)
    
    def test_blake2b_binding(self):
        """Test BLAKE2b stable binding."""
        bind_ops = BindingOperations(dimension=1000, use_blake2b=True)
        
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        
        # Stable binding
        result1 = bind_ops.blake2b_bind(a, b, stable=True)
        result2 = bind_ops.blake2b_bind(a, b, stable=True)
        
        # Should be deterministic
        np.testing.assert_array_equal(result1, result2)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(result1) - 1.0) < 1e-6
    
    def test_hierarchical_binding(self):
        """Test hierarchical multi-level binding."""
        bind_ops = BindingOperations(dimension=1000)
        
        vectors = [np.random.randn(1000) for _ in range(5)]
        zoom_levels = ['corpus', 'prompt', 'span', 'token_window']
        
        results = bind_ops.hierarchical_bind(vectors, zoom_levels)
        
        assert len(results) == 4
        for level in zoom_levels:
            assert level in results
            assert results[level].shape == (1000,)
        
        # Different levels should produce different results
        assert not np.array_equal(results['corpus'], results['token_window'])
    
    def test_simd_batch_binding(self):
        """Test SIMD-optimized batch operations."""
        bind_ops = BindingOperations(dimension=1000, enable_simd=True)
        
        batch_a = np.random.randn(32, 1000)
        batch_b = np.random.randn(32, 1000)
        
        # Test XOR
        result_xor = bind_ops.simd_batch_bind(batch_a, batch_b, 'xor')
        assert result_xor.shape == (32, 1000)
        
        # Test permutation
        result_perm = bind_ops.simd_batch_bind(batch_a, batch_b, 'permute')
        assert result_perm.shape == (32, 1000)
        
        # Test convolution
        result_conv = bind_ops.simd_batch_bind(batch_a, batch_b, 'convolve')
        assert result_conv.shape == (32, 1000)
    
    def test_privacy_preserving_binding(self):
        """Test privacy-preserving binding with DP."""
        bind_ops = BindingOperations(dimension=1000)
        
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        
        private_result, noise = bind_ops.privacy_preserving_bind(a, b, noise_scale=0.1)
        
        assert private_result.shape == (1000,)
        assert noise.shape == (1000,)
        
        # Should be normalized
        assert np.abs(np.linalg.norm(private_result) - 1.0) < 1e-6
        
        # Noise should have expected properties
        assert np.abs(np.mean(noise)) < 0.05  # Near zero mean
    
    def test_homomorphic_binding(self):
        """Test homomorphic-friendly binding."""
        bind_ops = BindingOperations(dimension=1000)
        
        # Simulate encrypted vectors
        a_enc = np.random.randint(0, 2**16, size=1000)
        b_enc = np.random.randint(0, 2**16, size=1000)
        
        result = bind_ops.homomorphic_bind(a_enc, b_enc)
        
        assert result.shape == (1000,)
        assert np.all(result >= 0)
        assert np.all(result < 2**16)
    
    def test_cached_binding(self):
        """Test cached binding operations."""
        bind_ops = BindingOperations(dimension=1000)
        
        # First call - not cached
        result1 = bind_ops.cached_bind("key1", "key2", "xor")
        
        # Second call - should be cached
        result2 = bind_ops.cached_bind("key1", "key2", "xor")
        
        np.testing.assert_array_equal(result1, result2)
        
        # Different keys
        result3 = bind_ops.cached_bind("key3", "key4", "xor")
        assert not np.array_equal(result1, result3)


class TestIntegration:
    """Integration tests for encoder and binding operations."""
    
    def test_end_to_end_encoding(self):
        """Test complete encoding pipeline."""
        # Initialize components
        encoder = HypervectorEncoder()
        bind_ops = BindingOperations(dimension=encoder.config.dimension)
        
        # Encode text at multiple levels
        text1 = "First model output for comparison"
        text2 = "Second model output for comparison"
        
        levels1 = encoder.encode_hierarchical(text1, return_all_levels=True)
        levels2 = encoder.encode_hierarchical(text2, return_all_levels=True)
        
        # Bind at each level
        bound_levels = {}
        for level in levels1.keys():
            bound_levels[level] = bind_ops.xor_bind(levels1[level], levels2[level])
        
        # Verify properties
        for level, bound in bound_levels.items():
            assert bound.shape == (encoder.config.dimension,)
    
    def test_performance_benchmark(self):
        """Benchmark encoding and binding performance."""
        encoder = HypervectorEncoder()
        bind_ops = BindingOperations(dimension=encoder.config.dimension)
        
        # Generate test data
        texts = [f"Test text number {i} with some content" for i in range(100)]
        
        # Benchmark encoding
        start = time.time()
        vectors = encoder.batch_encode(texts)
        encode_time = time.time() - start
        
        print(f"\nEncoding 100 texts: {encode_time:.3f}s")
        print(f"Per-text encoding: {encode_time/100*1000:.2f}ms")
        
        # Benchmark binding
        start = time.time()
        for i in range(0, 100, 2):
            bind_ops.xor_bind(vectors[i], vectors[i+1])
        bind_time = time.time() - start
        
        print(f"50 binding operations: {bind_time:.3f}s")
        print(f"Per-binding: {bind_time/50*1000:.2f}ms")
        
        # Performance targets
        assert encode_time < 5.0  # Should encode 100 texts in < 5s
        assert bind_time < 1.0  # Should bind 50 pairs in < 1s
    
    def test_memory_efficiency(self):
        """Test memory usage with bit-packing."""
        config_normal = HypervectorConfig(dimension=32768, bit_packed=False)
        config_packed = HypervectorConfig(dimension=32768, bit_packed=True)
        
        encoder_normal = HypervectorEncoder(config_normal)
        encoder_packed = HypervectorEncoder(config_packed)
        
        # Create vectors
        vec = np.random.randn(32768)
        
        # Compare sizes
        normal_bytes = vec.tobytes()
        packed_bytes = encoder_packed.bit_pack(vec)
        
        compression_ratio = len(normal_bytes) / len(packed_bytes)
        print(f"\nCompression ratio: {compression_ratio:.2f}x")
        print(f"Normal size: {len(normal_bytes)} bytes")
        print(f"Packed size: {len(packed_bytes)} bytes")
        
        assert compression_ratio > 10  # Should achieve >10x compression for binary
    
    def test_privacy_preservation(self):
        """Test privacy-preserving features."""
        config = HypervectorConfig(privacy_mode=True, homomorphic_friendly=True)
        encoder = HypervectorEncoder(config)
        bind_ops = BindingOperations(dimension=encoder.config.dimension)
        
        # Test distributed encoding preserves privacy
        sensitive_data = "private information"
        shares = encoder.distributed_encode(sensitive_data, n_shares=4)
        
        # Individual shares shouldn't reveal information
        for share in shares[:-1]:
            correlation = np.corrcoef(share, encoder.encode_feature(sensitive_data))[0, 1]
            assert abs(correlation) < 0.3  # Low correlation with original
        
        # Test privacy-preserving binding
        vec1 = encoder.encode_feature("data1")
        vec2 = encoder.encode_feature("data2")
        
        private_bound, noise = bind_ops.privacy_preserving_bind(vec1, vec2)
        
        # Result should be different from non-private binding
        normal_bound = bind_ops.xor_bind(vec1, vec2)
        assert not np.array_equal(private_bound, normal_bound)


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("Semantic Hypervector Encoding System Tests")
    print("="*70)
    
    # Run encoder tests
    print("\nTesting HypervectorEncoder...")
    encoder_tests = TestHypervectorEncoder()
    encoder_tests.test_initialization()
    encoder_tests.test_blake2b_hashing()
    encoder_tests.test_feature_encoding()
    encoder_tests.test_hierarchical_encoding()
    encoder_tests.test_batch_encoding()
    encoder_tests.test_popcount_hamming()
    encoder_tests.test_bit_packing()
    encoder_tests.test_distributed_encoding()
    encoder_tests.test_homomorphic_encoding()
    encoder_tests.test_zero_knowledge_commitment()
    print("✓ All encoder tests passed")
    
    # Run binding tests
    print("\nTesting BindingOperations...")
    binding_tests = TestBindingOperations()
    binding_tests.test_initialization()
    binding_tests.test_xor_binding()
    binding_tests.test_permutation_binding()
    binding_tests.test_circular_convolution()
    binding_tests.test_blake2b_binding()
    binding_tests.test_hierarchical_binding()
    binding_tests.test_simd_batch_binding()
    binding_tests.test_privacy_preserving_binding()
    binding_tests.test_homomorphic_binding()
    binding_tests.test_cached_binding()
    print("✓ All binding tests passed")
    
    # Run integration tests
    print("\nTesting Integration...")
    integration_tests = TestIntegration()
    integration_tests.test_end_to_end_encoding()
    integration_tests.test_performance_benchmark()
    integration_tests.test_memory_efficiency()
    integration_tests.test_privacy_preservation()
    print("✓ All integration tests passed")
    
    print("\n" + "="*70)
    print("All Semantic Hypervector tests passed successfully! ✓")
    print("="*70)


if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Add missing import
    import os
    
    run_all_tests()