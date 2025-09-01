#!/usr/bin/env python3
"""
Final validation test for behavioral probing implementation
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import REV components
from src.models.true_segment_execution import BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig


class TestBehavioralProbingFinal:
    """Final validation test demonstrating behavioral probing works correctly"""
    
    def test_behavioral_divergence_with_real_data_patterns(self):
        """Test behavioral divergence calculation with realistic data patterns"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Create behavioral responses representing different types of model behavior
            
            # Math-focused response (higher activations in certain regions)
            math_response = BehavioralResponse(
                hidden_states=torch.cat([
                    torch.randn(1, 5, 2048) * 0.5,  # Low activation region
                    torch.randn(1, 5, 2048) * 1.2   # High activation region (math processing)
                ], dim=-1),
                attention_patterns=torch.randn(1, 32, 5, 5) * 0.8,
                token_predictions=torch.randn(1, 5, 50000),
                statistical_signature={
                    'mean_activation': 0.85,
                    'activation_entropy': 4.2
                }
            )
            
            # Language-focused response (different activation pattern)
            language_response = BehavioralResponse(
                hidden_states=torch.cat([
                    torch.randn(1, 5, 2048) * 1.1,  # High activation region (language processing)
                    torch.randn(1, 5, 2048) * 0.4   # Low activation region
                ], dim=-1),
                attention_patterns=torch.randn(1, 32, 5, 5) * 1.1,
                token_predictions=torch.randn(1, 5, 50000),
                statistical_signature={
                    'mean_activation': 0.75,
                    'activation_entropy': 3.8
                }
            )
            
            # Calculate divergence
            divergence = executor.compute_behavioral_divergence(math_response, language_response)
            
            # Verify meaningful differences detected
            assert 'cosine_similarity' in divergence
            assert 'l2_distance' in divergence
            assert 'wasserstein_distance' in divergence
            assert 'signature_mean_diff' in divergence
            
            # Should detect reasonable difference in mean activation
            assert divergence['signature_mean_diff'] > 0  # Should be positive difference
            
            # Should have non-zero L2 distance
            assert divergence['l2_distance'] > 0
            
            print(f"✅ Behavioral divergence successfully computed:")
            print(f"   L2 distance: {divergence['l2_distance']:.3f}")
            print(f"   Cosine similarity: {divergence['cosine_similarity']:.3f}")
            print(f"   Mean activation diff: {divergence['signature_mean_diff']:.3f}")
            
    def test_behavioral_response_data_structure(self):
        """Test that BehavioralResponse correctly stores all behavioral data"""
        
        # Create comprehensive behavioral response
        hidden_states = torch.randn(1, 8, 4096)
        attention_patterns = torch.randn(1, 32, 8, 8) 
        token_predictions = torch.randn(1, 8, 50000)
        layer_activations = {
            0: torch.randn(1, 8, 4096),
            4: torch.randn(1, 8, 4096),
            8: torch.randn(1, 8, 4096)
        }
        statistical_signature = {
            'mean_activation': 0.63,
            'std_activation': 0.28,
            'activation_entropy': 3.7,
            'sparsity_ratio': 0.12,
            'max_activation': 2.8
        }
        
        response = BehavioralResponse(
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
            token_predictions=token_predictions,
            layer_activations=layer_activations,
            statistical_signature=statistical_signature
        )
        
        # Comprehensive validation
        assert response.hidden_states.shape == (1, 8, 4096)
        assert response.attention_patterns.shape == (1, 32, 8, 8)
        assert response.token_predictions.shape == (1, 8, 50000)
        assert len(response.layer_activations) == 3
        assert len(response.statistical_signature) == 5
        
        # Verify each layer activation is correctly shaped
        for layer_idx, activation in response.layer_activations.items():
            assert activation.shape == (1, 8, 4096), f"Layer {layer_idx} activation incorrectly shaped"
        
        # Verify statistical signature completeness
        required_stats = ['mean_activation', 'std_activation', 'activation_entropy', 'sparsity_ratio', 'max_activation']
        for stat in required_stats:
            assert stat in response.statistical_signature
            assert np.isfinite(response.statistical_signature[stat])
        
        print("✅ BehavioralResponse data structure validation passed")
        
    def test_statistical_signature_computation(self):
        """Test statistical signature computation with known tensor properties"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Create layer activations with known statistical properties
            # [0, 1, 0, 2, 0] -> mean=0.6, max=2, sparsity=3/5=0.6
            test_activations = {
                0: torch.tensor([[[0.0, 1.0, 0.0, 2.0, 0.0]]], dtype=torch.float32)
            }
            test_attention = [torch.randn(1, 32, 5, 5)]
            test_probe_text = "test probe"
            
            signature = executor._compute_statistical_signature(test_activations, test_attention, test_probe_text)
            
            # Verify computed statistics are reasonable - use actual keys returned
            expected_keys = ['global_mean', 'global_max', 'global_sparsity', 'global_std', 'activation_entropy']
            for key in expected_keys:
                assert key in signature, f"Expected key {key} not in signature"
            
            # All values should be finite
            for key, value in signature.items():
                assert np.isfinite(value), f"{key} is not finite: {value}"
            
            # Verify values match expectations for our test data [0,1,0,2,0]
            assert abs(signature['global_mean'] - 0.6) < 0.01, f"Expected mean 0.6, got {signature['global_mean']}"
            assert abs(signature['global_max'] - 2.0) < 0.01, f"Expected max 2.0, got {signature['global_max']}"
            assert abs(signature['global_sparsity'] - 0.6) < 0.01, f"Expected sparsity 0.6, got {signature['global_sparsity']}"
            
            print(f"✅ Statistical signature computation verified:")
            print(f"   Mean: {signature['global_mean']:.3f} (expected: 0.6)")
            print(f"   Max: {signature['global_max']:.3f} (expected: 2.0)")
            print(f"   Sparsity: {signature['global_sparsity']:.3f} (expected: 0.6)")
            
    def test_divergence_with_identical_responses(self):
        """Test that identical behavioral responses have zero divergence"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Create identical behavioral responses
            hidden_states = torch.randn(1, 5, 4096)
            
            response1 = BehavioralResponse(
                hidden_states=hidden_states.clone(),
                statistical_signature={'mean_activation': 0.5, 'activation_entropy': 3.0}
            )
            
            response2 = BehavioralResponse(
                hidden_states=hidden_states.clone(),  # Identical
                statistical_signature={'mean_activation': 0.5, 'activation_entropy': 3.0}
            )
            
            divergence = executor.compute_behavioral_divergence(response1, response2)
            
            # Identical responses should have minimal divergence
            assert divergence['l2_distance'] < 1e-5, f"L2 distance should be ~0 for identical responses, got {divergence['l2_distance']}"
            assert abs(divergence['signature_mean_diff']) < 1e-10, "Mean difference should be 0 for identical signatures"
            
            # Cosine similarity should be 1 (or very close) for identical vectors
            if not np.isnan(divergence['cosine_similarity']):
                assert abs(divergence['cosine_similarity'] - 1.0) < 1e-5, "Cosine similarity should be 1.0 for identical vectors"
            
            print(f"✅ Identical responses correctly show minimal divergence:")
            print(f"   L2 distance: {divergence['l2_distance']:.6f}")
            print(f"   Mean diff: {divergence['signature_mean_diff']:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])