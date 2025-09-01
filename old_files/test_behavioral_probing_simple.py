#!/usr/bin/env python3
"""
Simplified test for behavioral probing functionality
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import REV components
from src.models.true_segment_execution import BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig


class TestBehavioralProbingSimple:
    """Simplified test suite focusing on core behavioral probing logic"""
    
    def test_behavioral_response_creation(self):
        """Test BehavioralResponse dataclass creation and validation"""
        # Create test tensors
        hidden_states = torch.randn(1, 10, 4096)
        attention_patterns = torch.randn(1, 32, 10, 10)
        token_predictions = torch.randn(1, 10, 50000)
        layer_activations = {
            0: torch.randn(1, 10, 4096),
            8: torch.randn(1, 10, 4096),
            16: torch.randn(1, 10, 4096)
        }
        statistical_signature = {
            'mean_activation': 0.5,
            'std_activation': 0.2,
            'activation_entropy': 3.4,
            'sparsity_ratio': 0.1,
            'max_activation': 2.3
        }
        
        # Create BehavioralResponse
        response = BehavioralResponse(
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
            token_predictions=token_predictions,
            layer_activations=layer_activations,
            statistical_signature=statistical_signature
        )
        
        # Validate structure
        assert response.hidden_states is not None
        assert response.hidden_states.shape == (1, 10, 4096)
        assert response.attention_patterns is not None
        assert response.attention_patterns.shape == (1, 32, 10, 10)
        assert response.token_predictions is not None
        assert response.token_predictions.shape == (1, 10, 50000)
        assert len(response.layer_activations) == 3
        assert 'mean_activation' in response.statistical_signature
        assert 'activation_entropy' in response.statistical_signature
    
    def test_behavioral_divergence_calculation_methods(self):
        """Test individual divergence calculation methods"""
        # Create two different behavioral responses
        response1 = BehavioralResponse(
            hidden_states=torch.randn(1, 10, 4096),
            attention_patterns=torch.randn(1, 32, 10, 10),
            token_predictions=torch.randn(1, 10, 50000),
            statistical_signature={'mean_activation': 0.5, 'activation_entropy': 3.0}
        )
        
        response2 = BehavioralResponse(
            hidden_states=torch.randn(1, 10, 4096),
            attention_patterns=torch.randn(1, 32, 10, 10),
            token_predictions=torch.randn(1, 10, 50000),
            statistical_signature={'mean_activation': 0.7, 'activation_entropy': 3.5}
        )
        
        # Create a minimal config and executor instance for testing methods
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Test divergence calculation
            divergence = executor.compute_behavioral_divergence(response1, response2)
            
            # Validate divergence metrics exist
            expected_metrics = ['kl_divergence', 'cosine_similarity', 'wasserstein_distance', 'l2_distance']
            for metric in expected_metrics:
                assert metric in divergence, f"Missing expected metric: {metric}"
            
            # All metrics should be finite numbers (except KL divergence which can be inf)
            for metric, value in divergence.items():
                if metric == 'kl_divergence' and np.isinf(value):
                    continue  # KL divergence can be inf for different distributions
                assert np.isfinite(value), f"{metric} should be finite, got {value}"
            
            # Cosine similarity should be between -1 and 1
            assert -1 <= divergence['cosine_similarity'] <= 1
            
            # L2 distance should be non-negative  
            assert divergence['l2_distance'] >= 0
    
    def test_statistical_signature_calculation(self):
        """Test statistical signature generation from tensor data"""
        # Use LayerSegmentExecutor
        
        # Create test hidden states with known statistical properties
        hidden_states = torch.tensor([[[1.0, 2.0, 0.0, 3.0, 0.0]]], dtype=torch.float32)
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Test _compute_statistical_signature method
            signature = executor._compute_statistical_signature(hidden_states)
            
            # Verify signature contains expected metrics
            expected_metrics = ['mean_activation', 'std_activation', 'activation_entropy', 'sparsity_ratio', 'max_activation']
            for metric in expected_metrics:
                assert metric in signature, f"Missing metric: {metric}"
                assert np.isfinite(signature[metric]), f"{metric} should be finite"
            
            # Verify known values
            assert abs(signature['mean_activation'] - 1.2) < 0.1  # (1+2+0+3+0)/5 = 1.2
            assert abs(signature['max_activation'] - 3.0) < 0.1   # max value is 3.0
            assert abs(signature['sparsity_ratio'] - 0.4) < 0.1   # 2/5 zeros = 0.4
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation between probability distributions"""
        # Use LayerSegmentExecutor
        
        # Create probability distributions (should sum to 1)
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.4, 0.4, 0.2])
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            kl_div = executor._compute_kl_divergence(p, q)
            
            # KL divergence should be non-negative and finite
            assert kl_div >= 0
            assert np.isfinite(kl_div)
            
            # KL(p||p) should be 0
            kl_self = executor._compute_kl_divergence(p, p)
            assert abs(kl_self) < 1e-6
    
    def test_restriction_site_identification_logic(self):
        """Test the core logic of restriction site identification"""
        # Use LayerSegmentExecutor
        
        # Mock behavioral responses with different divergence levels
        mock_responses = []
        for i in range(32):  # 32 layers
            response = BehavioralResponse(
                hidden_states=torch.randn(1, 10, 4096),
                statistical_signature={'mean_activation': 0.5 + 0.1 * (i % 8)}  # Pattern every 8 layers
            )
            mock_responses.append(response)
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Mock divergence calculation to return pattern-based values
            def mock_divergence(resp1, resp2):
                sig1 = resp1.statistical_signature['mean_activation']
                sig2 = resp2.statistical_signature['mean_activation']
                divergence = abs(sig1 - sig2)
                return {
                    'kl_divergence': divergence,
                    'cosine_similarity': 1.0 - divergence,
                    'wasserstein_distance': divergence,
                    'weighted_score': divergence
                }
            
            executor.compute_behavioral_divergence = mock_divergence
            
            # Test restriction site identification with known pattern
            sample_challenges = [{'id': 'test', 'problem': 'test problem'}]
            
            # Mock execute_behavioral_probe to return our mock responses
            def mock_probe(text, up_to_layer):
                return mock_responses[up_to_layer]
            
            executor.execute_behavioral_probe = mock_probe
            
            sites = executor.probe_for_restriction_sites(
                challenges=sample_challenges,
                num_layers=32,
                divergence_threshold=0.05
            )
            
            # Should identify sites where divergence exceeds threshold
            assert len(sites) > 0
            assert all(0 < site < 32 for site in sites)  # Sites should be valid layer indices
            assert sites == sorted(sites)  # Sites should be sorted
    
    def test_fallback_to_hardcoded_sites(self):
        """Test fallback behavior when behavioral probing fails"""
        # Use LayerSegmentExecutor
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Test with empty challenges (should trigger fallback)
            sites = executor.probe_for_restriction_sites(
                challenges=[],
                num_layers=32,
                divergence_threshold=0.3
            )
            
            # Should return hardcoded fallback sites
            expected_fallback = [8, 16, 24]  # Every 8th layer
            assert sites == expected_fallback
    
    def test_adaptive_threshold_behavior(self):
        """Test adaptive threshold adjustment logic"""
        # Use LayerSegmentExecutor
        
        # Test the _adjust_threshold_adaptively method
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Test threshold adjustment with different scenarios
            high_divergences = [0.8, 0.9, 0.7, 0.85, 0.75]  # All high
            low_divergences = [0.1, 0.05, 0.08, 0.12, 0.06]  # All low
            mixed_divergences = [0.1, 0.8, 0.15, 0.7, 0.2]   # Mixed
            
            # High divergences should lower threshold
            new_threshold_high = executor._adjust_threshold_adaptively(high_divergences, 0.5)
            assert new_threshold_high < 0.5
            
            # Low divergences should raise threshold  
            new_threshold_low = executor._adjust_threshold_adaptively(low_divergences, 0.5)
            assert new_threshold_low > 0.5 or new_threshold_low == 0.5  # Might stay same if already reasonable
            
            # Mixed should be somewhere in between
            new_threshold_mixed = executor._adjust_threshold_adaptively(mixed_divergences, 0.5)
            assert 0.1 <= new_threshold_mixed <= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])