#!/usr/bin/env python3
"""
Integration test demonstrating complete behavioral probing functionality
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import REV components
from src.models.true_segment_execution import BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig


class TestBehavioralProbingIntegration:
    """Integration test demonstrating the complete behavioral probing pipeline"""
    
    def test_complete_behavioral_probing_pipeline(self):
        """Test that behavioral probing correctly identifies restriction sites with PoT challenges"""
        
        # Create sample PoT challenges representing different types of reasoning
        pot_challenges = [
            {
                'id': 'math_001',
                'problem': 'Calculate the sum of prime numbers less than 20',
                'solution_steps': ['Find primes: 2,3,5,7,11,13,17,19', 'Sum: 77'],
                'expected_output': '77',
                'category': 'arithmetic'
            },
            {
                'id': 'logic_001', 
                'problem': 'If all birds can fly and penguins are birds, can penguins fly?',
                'solution_steps': ['Premise 1: All birds can fly', 'Premise 2: Penguins are birds', 'But penguins cannot fly'],
                'expected_output': 'False - the premise is incorrect',
                'category': 'logical_reasoning'
            }
        ]
        
        # Create executor with mocked initialization
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Mock device manager
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Mock the tokenizer
            executor.tokenizer = MagicMock()
            executor.tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Simple token sequence
            executor.tokenizer.decode.return_value = "mock decoded output"
            
            # Create behavioral responses that simulate different model behaviors for different challenges
            def create_mock_behavioral_response(challenge_type, layer_idx):
                """Create mock behavioral response that varies by challenge type and layer"""
                
                # Simulate different activation patterns for different reasoning types
                if challenge_type == 'arithmetic':
                    # Math problems activate certain patterns
                    base_activation = 0.6 + 0.1 * (layer_idx % 8)  # Pattern every 8 layers
                    hidden_states = torch.randn(1, 5, 4096) * base_activation + base_activation
                elif challenge_type == 'logical_reasoning':
                    # Logic problems have different activation patterns
                    base_activation = 0.4 + 0.15 * (layer_idx % 6)  # Pattern every 6 layers  
                    hidden_states = torch.randn(1, 5, 4096) * base_activation + base_activation
                else:
                    hidden_states = torch.randn(1, 5, 4096)
                
                # Create attention patterns that vary by layer
                attention_patterns = torch.randn(1, 32, 5, 5) * (0.5 + 0.02 * layer_idx)
                
                # Token predictions with layer-dependent confidence
                token_predictions = torch.randn(1, 5, 50000) * (1.0 + 0.05 * layer_idx)
                
                return BehavioralResponse(
                    hidden_states=hidden_states,
                    attention_patterns=attention_patterns,
                    token_predictions=token_predictions,
                    statistical_signature={
                        'mean_activation': float(torch.mean(hidden_states)),
                        'activation_entropy': 3.0 + 0.1 * layer_idx
                    }
                )
            
            # Mock execute_behavioral_probe to return different responses for different challenges
            def mock_execute_probe(probe_text, up_to_layer):
                # Determine challenge type from probe text
                if 'prime' in probe_text.lower() or 'sum' in probe_text.lower():
                    challenge_type = 'arithmetic'
                elif 'birds' in probe_text.lower() or 'penguins' in probe_text.lower():
                    challenge_type = 'logical_reasoning'
                else:
                    challenge_type = 'other'
                
                return create_mock_behavioral_response(challenge_type, up_to_layer)
            
            executor.execute_behavioral_probe = mock_execute_probe
            
            # Test restriction site identification
            restriction_sites = executor.probe_for_restriction_sites(
                challenges=pot_challenges,
                num_layers=24,  # 24-layer model
                divergence_threshold=0.2
            )
            
            # Validate results
            assert len(restriction_sites) > 0, "Should identify at least some restriction sites"
            assert len(restriction_sites) < 24, "Should not mark every layer as a restriction site"
            
            # All sites should be valid layer indices
            for site in restriction_sites:
                assert 0 < site < 24, f"Site {site} should be valid layer index (1-23)"
            
            # Sites should be sorted
            assert restriction_sites == sorted(restriction_sites), "Restriction sites should be sorted"
            
            print(f"✅ Successfully identified {len(restriction_sites)} behavioral restriction sites: {restriction_sites}")
            
            # Verify that behavioral divergence calculation works
            math_response = create_mock_behavioral_response('arithmetic', 8)
            logic_response = create_mock_behavioral_response('logical_reasoning', 8)
            
            divergence = executor.compute_behavioral_divergence(math_response, logic_response)
            
            # Should detect differences between different reasoning types
            assert divergence['cosine_similarity'] != 1.0, "Different reasoning types should have different behavioral signatures"
            assert divergence['l2_distance'] > 0, "Should have non-zero L2 distance between different behaviors"
            
            print(f"✅ Behavioral divergence correctly computed: {divergence['l2_distance']:.3f} L2 distance")
            
    def test_behavioral_probing_fallback_mechanism(self):
        """Test that fallback to hardcoded sites works when behavioral probing fails"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            
            # Test with empty challenges (should trigger fallback)
            restriction_sites = executor.probe_for_restriction_sites(
                challenges=[],  # No challenges provided
                num_layers=24,
                divergence_threshold=0.3
            )
            
            # Should fallback to hardcoded layer boundaries
            expected_fallback = [6, 12, 18]  # Every 6th layer for 24-layer model
            assert restriction_sites == expected_fallback, f"Expected fallback sites {expected_fallback}, got {restriction_sites}"
            
            print(f"✅ Fallback mechanism works correctly: {restriction_sites}")
            
    def test_behavioral_response_validation(self):
        """Test that BehavioralResponse objects contain all expected components"""
        
        # Create a realistic behavioral response
        hidden_states = torch.randn(1, 10, 4096)
        attention_patterns = torch.randn(1, 32, 10, 10)
        token_predictions = torch.randn(1, 10, 50000)
        layer_activations = {
            4: torch.randn(1, 10, 4096),
            8: torch.randn(1, 10, 4096), 
            12: torch.randn(1, 10, 4096)
        }
        statistical_signature = {
            'mean_activation': 0.65,
            'std_activation': 0.32,
            'activation_entropy': 4.2,
            'sparsity_ratio': 0.15,
            'max_activation': 3.1
        }
        
        response = BehavioralResponse(
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
            token_predictions=token_predictions,
            layer_activations=layer_activations,
            statistical_signature=statistical_signature
        )
        
        # Validate all components are present and correctly shaped
        assert response.hidden_states.shape == (1, 10, 4096)
        assert response.attention_patterns.shape == (1, 32, 10, 10)
        assert response.token_predictions.shape == (1, 10, 50000)
        assert len(response.layer_activations) == 3
        assert len(response.statistical_signature) == 5
        
        # Verify statistical signature contains expected metrics
        required_metrics = ['mean_activation', 'std_activation', 'activation_entropy', 'sparsity_ratio', 'max_activation']
        for metric in required_metrics:
            assert metric in response.statistical_signature
            assert np.isfinite(response.statistical_signature[metric])
        
        print("✅ BehavioralResponse validation passed")
        
    def test_restriction_site_confidence_scoring(self):
        """Test that restriction sites are identified with appropriate confidence scores"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Create two very different behavioral responses
            high_divergence_response1 = BehavioralResponse(
                hidden_states=torch.zeros(1, 5, 4096),  # All zeros
                statistical_signature={'mean_activation': 0.0, 'activation_entropy': 0.1}
            )
            high_divergence_response2 = BehavioralResponse(
                hidden_states=torch.ones(1, 5, 4096),   # All ones
                statistical_signature={'mean_activation': 1.0, 'activation_entropy': 5.0}
            )
            
            # Calculate divergence between very different responses
            divergence = executor.compute_behavioral_divergence(high_divergence_response1, high_divergence_response2)
            
            # Should detect high divergence
            assert divergence['l2_distance'] > 100, "Should detect high L2 distance between very different responses"
            assert abs(divergence['cosine_similarity']) < 0.1, "Cosine similarity should be low for very different responses"
            
            print(f"✅ High divergence correctly detected: L2={divergence['l2_distance']:.1f}, Cosine={divergence['cosine_similarity']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])