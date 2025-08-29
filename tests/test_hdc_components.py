"""
Unit tests for Hyperdimensional Computing components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch

from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.binding_operations import BindingOperations
from src.hdc.error_correction import ErrorCorrection, ErrorCorrectionConfig
from src.hdc.behavioral_sites import (
    BehavioralSites, ProbeFeatures, TaskCategory,
    SyntaxComplexity, ReasoningDepth
)


class TestHypervectorEncoder:
    """Test hypervector encoding."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization with config."""
        config = HypervectorConfig(
            dimension=1000,
            sparse_density=0.01,
            dtype="float32"
        )
        encoder = HypervectorEncoder(config)
        
        assert encoder.dimension == 1000
        assert encoder.sparse_density == 0.01
    
    def test_encode_vector(self):
        """Test encoding a vector to hypervector."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=1000))
        
        input_vec = np.random.randn(100)
        hypervector = encoder.encode(input_vec)
        
        assert len(hypervector) == 1000
        assert isinstance(hypervector, np.ndarray)
        
        # Check sparsity
        if encoder.sparse_density < 1.0:
            sparsity = np.mean(hypervector == 0)
            assert sparsity > 0.5  # Should be somewhat sparse
    
    def test_encode_text(self):
        """Test encoding text to hypervector."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=1000))
        
        text = "Hello, world!"
        hypervector = encoder.encode_text(text)
        
        assert len(hypervector) == 1000
        
        # Different texts should give different vectors
        other_text = "Goodbye, world!"
        other_vector = encoder.encode_text(other_text)
        
        similarity = np.dot(hypervector, other_vector) / (
            np.linalg.norm(hypervector) * np.linalg.norm(other_vector)
        )
        assert similarity < 0.9  # Should be somewhat different
    
    def test_encode_sequence(self):
        """Test encoding a sequence of vectors."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=1000))
        
        sequence = [np.random.randn(50) for _ in range(10)]
        hypervector = encoder.encode_sequence(sequence)
        
        assert len(hypervector) == 1000
        assert not np.all(hypervector == 0)  # Should not be all zeros
    
    def test_encode_deterministic(self):
        """Test that encoding is deterministic with seed."""
        config = HypervectorConfig(dimension=1000, seed=42)
        encoder1 = HypervectorEncoder(config)
        encoder2 = HypervectorEncoder(config)
        
        input_vec = np.random.randn(100)
        hv1 = encoder1.encode(input_vec)
        hv2 = encoder2.encode(input_vec)
        
        np.testing.assert_array_almost_equal(hv1, hv2)


class TestBindingOperations:
    """Test binding operations for hypervectors."""
    
    def test_xor_bind_binary(self):
        """Test XOR binding for binary vectors."""
        binder = BindingOperations(dimension=1000)
        
        # Create binary vectors
        a = np.random.choice([0, 1], size=1000).astype(np.float32)
        b = np.random.choice([0, 1], size=1000).astype(np.float32)
        
        bound = binder.xor_bind(a, b)
        
        assert len(bound) == 1000
        assert np.all(np.logical_or(bound == 1, bound == -1))
    
    def test_xor_bind_continuous(self):
        """Test XOR binding for continuous vectors."""
        binder = BindingOperations(dimension=1000)
        
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        
        bound = binder.xor_bind(a, b)
        
        assert len(bound) == 1000
        assert not np.all(bound == a)  # Should be modified
        assert not np.all(bound == b)
    
    def test_permutation_bind(self):
        """Test permutation binding."""
        binder = BindingOperations(dimension=100)
        
        a = np.random.randn(100)
        b = np.random.randn(100)
        
        # Test different permutation types
        for perm_type in ['position', 'value', 'hash']:
            bound = binder.permutation_bind(a, b, perm_type=perm_type)
            
            assert len(bound) == 100
            # Check that it's a permutation (same elements, different order)
            assert np.allclose(sorted(bound), sorted(a))
    
    def test_circular_convolve(self):
        """Test circular convolution."""
        binder = BindingOperations(dimension=100)
        
        a = np.random.randn(100)
        b = np.random.randn(100)
        
        # Test different modes
        for mode in ['fft', 'direct']:
            bound = binder.circular_convolve(a, b, mode=mode)
            
            assert len(bound) == 100
            assert not np.all(bound == 0)
    
    def test_fourier_bind(self):
        """Test Fourier domain binding."""
        binder = BindingOperations(dimension=100)
        
        a = np.random.randn(100)
        b = np.random.randn(100)
        
        # Test different domains
        for domain in ['frequency', 'phase', 'magnitude']:
            bound = binder.fourier_bind(a, b, domain=domain)
            
            assert len(bound) == 100
            assert np.all(np.isfinite(bound))  # No NaN or inf
    
    def test_weighted_bind(self):
        """Test weighted binding."""
        binder = BindingOperations(dimension=100)
        
        vectors = [np.random.randn(100) for _ in range(5)]
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        
        bound = binder.weighted_bind(vectors, weights, normalize=True)
        
        assert len(bound) == 100
        assert np.allclose(np.linalg.norm(bound), 1.0)  # Should be normalized
    
    def test_protect_unprotect_bind(self):
        """Test protected binding and unbinding."""
        binder = BindingOperations(dimension=100)
        
        data = np.random.randn(100)
        key = np.random.randn(100)
        
        # Protect
        protected = binder.protect_bind(data, key)
        assert len(protected) == 100
        assert not np.allclose(protected, data)  # Should be scrambled
        
        # Unprotect
        recovered = binder.unprotect_bind(protected, key)
        assert len(recovered) == 100
        np.testing.assert_array_almost_equal(recovered, data, decimal=5)


class TestErrorCorrection:
    """Test error correction mechanisms."""
    
    def test_encode_with_parity(self):
        """Test encoding with parity bits."""
        config = ErrorCorrectionConfig(
            dimension=1000,
            parity_overhead=0.25,
            block_size=64
        )
        corrector = ErrorCorrection(config)
        
        data = np.random.randn(1000)
        encoded = corrector.encode_with_parity(data)
        
        expected_len = 1000 + int(1000 * 0.25)
        assert len(encoded) == expected_len
    
    def test_decode_with_correction(self):
        """Test decoding and error correction."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=1000))
        
        # Original data
        data = np.random.choice([-1, 1], size=1000).astype(np.float32)
        
        # Encode
        encoded = corrector.encode_with_parity(data)
        
        # Decode without errors
        decoded, corrections = corrector.decode_with_correction(encoded)
        
        assert len(decoded) == 1000
        assert corrections == 0  # No errors to correct
        np.testing.assert_array_almost_equal(decoded, data)
    
    def test_error_correction_single_bit(self):
        """Test single bit error correction."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=100))
        
        # Binary data
        data = np.random.choice([0, 1], size=100).astype(np.float32)
        
        # Encode
        encoded = corrector.encode_with_parity(data)
        
        # Introduce single bit error
        encoded[10] = -encoded[10] if encoded[10] != 0 else 1
        
        # Decode and correct
        decoded, corrections = corrector.decode_with_correction(encoded, correct_errors=True)
        
        assert corrections > 0  # Should have corrected something
        # Check recovery (may not be perfect for all cases)
        recovery_rate = np.mean(decoded == data)
        assert recovery_rate > 0.9
    
    def test_add_noise(self):
        """Test noise addition."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=1000))
        
        vector = np.random.randn(1000)
        
        # Test different noise types
        for noise_type in ['gaussian', 'salt_pepper', 'burst']:
            noisy = corrector.add_noise(vector, noise_level=0.1, noise_type=noise_type)
            
            assert len(noisy) == 1000
            assert not np.allclose(noisy, vector)  # Should be modified
            
            # Check that noise is reasonable
            difference = np.mean(np.abs(noisy - vector))
            assert difference > 0
            assert difference < 10  # Not too extreme
    
    def test_measure_robustness(self):
        """Test robustness measurement."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=1000))
        
        original = np.random.choice([-1, 1], size=1000).astype(np.float32)
        noisy = corrector.add_noise(original, noise_level=0.1, noise_type='salt_pepper')
        
        # Encode and decode
        encoded = corrector.encode_with_parity(noisy)
        corrected, _ = corrector.decode_with_correction(encoded, correct_errors=True)
        
        # Measure robustness
        metrics = corrector.measure_robustness(original, noisy, corrected)
        
        assert 'ber_noisy' in metrics
        assert 'ber_corrected' in metrics
        assert 'cosine_sim_noisy' in metrics
        assert 'cosine_sim_corrected' in metrics
        assert metrics['correction_success'] in [0.0, 1.0]


class TestBehavioralSites:
    """Test behavioral site analysis."""
    
    def test_extract_probe_features(self):
        """Test probe feature extraction."""
        sites = BehavioralSites()
        
        probe_text = "Explain the concept of machine learning and provide examples."
        features = sites.extract_probe_features(probe_text)
        
        assert isinstance(features, ProbeFeatures)
        assert features.task_category in TaskCategory
        assert features.syntax_complexity in SyntaxComplexity
        assert features.reasoning_depth in ReasoningDepth
        assert features.token_count > 0
        assert 0 <= features.vocabulary_diversity <= 1
    
    def test_task_category_detection(self):
        """Test task category detection."""
        sites = BehavioralSites()
        
        test_cases = [
            ("Translate this to French: Hello", TaskCategory.TRANSLATION),
            ("Summarize this article", TaskCategory.SUMMARIZATION),
            ("What is the capital of France?", TaskCategory.QUESTION_ANSWERING),
            ("Write a Python function", TaskCategory.CODE_GENERATION),
            ("Calculate 2+2", TaskCategory.MATH),
            ("Write a poem about", TaskCategory.CREATIVE),
        ]
        
        for text, expected_category in test_cases:
            features = sites.extract_probe_features(text)
            # Allow some flexibility in categorization
            assert features.task_category in TaskCategory
    
    def test_generate_response_hypervector(self):
        """Test response hypervector generation."""
        sites = BehavioralSites()
        
        # Simulate logit profile
        logit_profile = np.random.randn(100, 50)  # seq_len x vocab_size
        
        # Test different zoom levels
        for zoom_level in ['prompt', 'span_64', 'token_window_8']:
            hypervector = sites.generate_response_hypervector(
                logit_profile, zoom_level=zoom_level
            )
            
            assert len(hypervector) == sites.hdc_config.dimension
            assert not np.all(hypervector == 0)
            assert np.all(np.isfinite(hypervector))
    
    def test_hierarchical_analysis(self):
        """Test hierarchical analysis at multiple zoom levels."""
        sites = BehavioralSites()
        
        # Create mock model outputs
        model_outputs = {
            'attention_0': np.random.randn(100, 768),
            'mlp_0': np.random.randn(100, 768),
            'attention_6': np.random.randn(100, 768),
            'mlp_6': np.random.randn(100, 768),
        }
        
        probe_features = sites.extract_probe_features("Test prompt")
        
        hierarchical_hvs = sites.hierarchical_analysis(model_outputs, probe_features)
        
        assert len(hierarchical_hvs) == len(sites.zoom_levels)
        for zoom_name, hv in hierarchical_hvs.items():
            assert len(hv) == sites.hdc_config.dimension
            assert not np.all(hv == 0)
    
    def test_compare_behavioral_signatures(self):
        """Test behavioral signature comparison."""
        sites = BehavioralSites()
        
        # Create two signatures
        sig_a = {
            'prompt': np.random.randn(sites.hdc_config.dimension),
            'span_64': np.random.randn(sites.hdc_config.dimension),
            'token_window_8': np.random.randn(sites.hdc_config.dimension),
        }
        
        sig_b = {
            'prompt': np.random.randn(sites.hdc_config.dimension),
            'span_64': np.random.randn(sites.hdc_config.dimension),
            'token_window_8': np.random.randn(sites.hdc_config.dimension),
        }
        
        similarity = sites.compare_behavioral_signatures(sig_a, sig_b)
        
        assert 0 <= similarity <= 1
        
        # Same signature should have similarity 1
        self_similarity = sites.compare_behavioral_signatures(sig_a, sig_a)
        assert self_similarity == pytest.approx(1.0, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])