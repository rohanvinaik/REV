#!/usr/bin/env python3
"""
Test behavioral probing implementation with actual PoT challenges
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, create_autospec
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Import REV components
from src.models.true_segment_execution import REVTrueExecution, BehavioralResponse
from src.hdc.behavioral_sites import BehavioralSites
from src.core.device_manager import DeviceManager
from src.challenges.pot_challenge_generator import PoTChallengeGenerator


class TestBehavioralProbing:
    """Test suite for behavioral probing implementation"""
    
    @pytest.fixture
    def device_manager(self):
        """Create device manager for testing"""
        return DeviceManager()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = MagicMock()
        model.config.num_hidden_layers = 32
        model.config.hidden_size = 4096
        model.config.num_attention_heads = 32
        
        # Mock forward pass behavior
        def mock_forward(*args, **kwargs):
            batch_size = 1
            seq_len = 10
            hidden_size = 4096
            
            # Create mock outputs
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
            mock_output.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_size) for _ in range(33)  # 32 layers + embeddings
            ]
            mock_output.attentions = [
                torch.randn(batch_size, 32, seq_len, seq_len) for _ in range(32)  # 32 layers
            ]
            mock_output.logits = torch.randn(batch_size, seq_len, 50000)  # vocab_size
            return mock_output
        
        model.return_value = mock_forward
        model.forward = mock_forward
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
        tokenizer.decode.return_value = "decoded text"
        tokenizer.pad_token_id = 0
        return tokenizer
    
    @pytest.fixture
    def true_executor(self, mock_model, mock_tokenizer, device_manager):
        """Create REVTrueExecution instance"""
        # Mock the model loading since we're testing behavioral logic
        with patch('src.models.true_segment_execution.REVTrueExecution._load_model_and_tokenizer') as mock_load:
            mock_load.return_value = (mock_model, mock_tokenizer)
            executor = REVTrueExecution(
                model_path="/fake/path/to/model",
                max_memory_gb=4.0
            )
            # Override the loaded model and tokenizer
            executor.tokenizer = mock_tokenizer
            return executor
    
    @pytest.fixture
    def behavioral_sites(self, device_manager):
        """Create BehavioralSites instance"""
        return BehavioralSites(
            hypervector_dim=10000,
            device_manager=device_manager
        )
    
    @pytest.fixture
    def sample_pot_challenges(self):
        """Generate sample PoT challenges for testing"""
        return [
            {
                'id': 'pot_001',
                'problem': 'Calculate the sum of prime numbers less than 20',
                'solution_steps': [
                    'Find all prime numbers less than 20: 2, 3, 5, 7, 11, 13, 17, 19',
                    'Sum them: 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77'
                ],
                'expected_output': '77',
                'difficulty': 'medium'
            },
            {
                'id': 'pot_002', 
                'problem': 'Find the area of a circle with radius 5',
                'solution_steps': [
                    'Use formula A = π × r²',
                    'A = π × 5² = π × 25 = 25π ≈ 78.54'
                ],
                'expected_output': '78.54',
                'difficulty': 'easy'
            }
        ]
    
    def test_behavioral_response_creation(self, true_executor):
        """Test BehavioralResponse dataclass creation"""
        hidden_states = torch.randn(1, 10, 4096)
        attention_patterns = torch.randn(1, 32, 10, 10)
        token_predictions = torch.randn(1, 10, 50000)
        
        response = BehavioralResponse(
            hidden_states=hidden_states,
            attention_patterns=attention_patterns,
            token_predictions=token_predictions,
            statistical_signature={'mean_activation': 0.5, 'std_activation': 0.2}
        )
        
        assert response.hidden_states is not None
        assert response.attention_patterns is not None
        assert response.token_predictions is not None
        assert 'mean_activation' in response.statistical_signature
        assert 'std_activation' in response.statistical_signature
    
    def test_execute_behavioral_probe(self, true_executor, sample_pot_challenges):
        """Test behavioral probe execution with PoT challenges"""
        challenge = sample_pot_challenges[0]
        probe_text = f"Problem: {challenge['problem']}\nSolution: {challenge['solution_steps'][0]}"
        
        # Execute probe up to layer 16
        response = true_executor.execute_behavioral_probe(probe_text, up_to_layer=16)
        
        # Verify response structure
        assert isinstance(response, BehavioralResponse)
        assert response.hidden_states is not None
        assert response.hidden_states.shape[-1] == 4096  # hidden_size
        
        # Check attention patterns
        if response.attention_patterns is not None:
            assert len(response.attention_patterns.shape) == 4  # [batch, heads, seq, seq]
        
        # Check token predictions
        if response.token_predictions is not None:
            assert response.token_predictions.shape[-1] == 50000  # vocab_size
        
        # Verify statistical signature
        assert response.statistical_signature is not None
        assert 'mean_activation' in response.statistical_signature
        assert 'activation_entropy' in response.statistical_signature
    
    def test_behavioral_divergence_calculation(self, true_executor, sample_pot_challenges):
        """Test behavioral divergence calculation between different challenges"""
        challenge1 = sample_pot_challenges[0]  # Math problem
        challenge2 = sample_pot_challenges[1]  # Geometry problem
        
        # Execute probes for both challenges
        probe_text1 = f"Problem: {challenge1['problem']}"
        probe_text2 = f"Problem: {challenge2['problem']}"
        
        response1 = true_executor.execute_behavioral_probe(probe_text1, up_to_layer=16)
        response2 = true_executor.execute_behavioral_probe(probe_text2, up_to_layer=16)
        
        # Calculate divergence
        divergence = true_executor.compute_behavioral_divergence(response1, response2)
        
        # Verify divergence metrics
        assert 'kl_divergence' in divergence
        assert 'cosine_similarity' in divergence
        assert 'wasserstein_distance' in divergence
        assert 'weighted_score' in divergence
        
        # All metrics should be finite numbers
        for metric, value in divergence.items():
            assert np.isfinite(value), f"{metric} should be finite, got {value}"
        
        # Weighted score should be between 0 and 1
        assert 0 <= divergence['weighted_score'] <= 1
    
    def test_probe_for_restriction_sites(self, true_executor, sample_pot_challenges):
        """Test restriction site identification using behavioral probing"""
        
        # Execute restriction site probing
        restriction_sites = true_executor.probe_for_restriction_sites(
            challenges=sample_pot_challenges,
            num_layers=32,
            divergence_threshold=0.3
        )
        
        # Verify restriction sites found
        assert len(restriction_sites) > 0, "Should find at least some restriction sites"
        
        # All sites should be valid layer indices
        for site in restriction_sites:
            assert 0 <= site < 32, f"Site {site} should be valid layer index"
        
        # Sites should be sorted
        assert restriction_sites == sorted(restriction_sites)
        
        # Should not include layer 0 (input embeddings) or last layer
        assert 0 not in restriction_sites
        assert 31 not in restriction_sites
    
    def test_behavioral_sites_integration(self, true_executor, behavioral_sites, sample_pot_challenges):
        """Test integration between true execution and behavioral sites"""
        challenge = sample_pot_challenges[0]
        probe_text = f"Problem: {challenge['problem']}"
        
        # Execute behavioral probe
        response = true_executor.execute_behavioral_probe(probe_text, up_to_layer=16)
        
        # Extract features using behavioral sites
        features = behavioral_sites.extract_probe_features(
            behavioral_response=response,
            challenge_data=challenge
        )
        
        # Verify feature extraction
        assert 'activation_hypervector' in features
        assert 'attention_hypervector' in features
        assert 'semantic_embedding' in features
        
        # Check hypervector dimensions
        assert features['activation_hypervector'].shape[0] == 10000
        assert features['attention_hypervector'].shape[0] == 10000
        assert features['semantic_embedding'].shape[0] == 10000
    
    def test_fallback_to_hardcoded_sites(self, true_executor):
        """Test fallback to hardcoded sites when behavioral probing fails"""
        
        # Create scenario where behavioral probing fails (empty challenges)
        empty_challenges = []
        
        restriction_sites = true_executor.probe_for_restriction_sites(
            challenges=empty_challenges,
            num_layers=32,
            divergence_threshold=0.3
        )
        
        # Should fallback to hardcoded layer boundaries
        expected_hardcoded = [8, 16, 24]  # Every 8th layer
        assert restriction_sites == expected_hardcoded
    
    def test_adaptive_threshold_adjustment(self, true_executor, sample_pot_challenges):
        """Test adaptive threshold adjustment in restriction site identification"""
        
        # Test with very high threshold (should reduce automatically)
        restriction_sites_high = true_executor.probe_for_restriction_sites(
            challenges=sample_pot_challenges,
            num_layers=32,
            divergence_threshold=0.9  # Very high threshold
        )
        
        # Test with low threshold
        restriction_sites_low = true_executor.probe_for_restriction_sites(
            challenges=sample_pot_challenges,
            num_layers=32,
            divergence_threshold=0.1  # Low threshold
        )
        
        # Low threshold should find more sites than high threshold
        assert len(restriction_sites_low) >= len(restriction_sites_high)
        
        # Both should find at least some sites (or fallback to hardcoded)
        assert len(restriction_sites_high) >= 3  # At least hardcoded fallback
        assert len(restriction_sites_low) >= 3
    
    def test_statistical_signature_generation(self, true_executor, sample_pot_challenges):
        """Test statistical signature generation from neural activations"""
        challenge = sample_pot_challenges[0]
        probe_text = f"Problem: {challenge['problem']}"
        
        response = true_executor.execute_behavioral_probe(probe_text, up_to_layer=16)
        
        # Verify statistical signature contains expected metrics
        signature = response.statistical_signature
        assert signature is not None
        
        expected_metrics = [
            'mean_activation',
            'std_activation', 
            'activation_entropy',
            'sparsity_ratio',
            'max_activation'
        ]
        
        for metric in expected_metrics:
            assert metric in signature, f"Missing metric: {metric}"
            assert np.isfinite(signature[metric]), f"{metric} should be finite"
    
    def test_memory_bounded_execution(self, true_executor, sample_pot_challenges):
        """Test that behavioral probing works within memory bounds"""
        challenge = sample_pot_challenges[0]
        probe_text = f"Problem: {challenge['problem']}"
        
        # Monitor memory usage during execution
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Execute probe
        response = true_executor.execute_behavioral_probe(probe_text, up_to_layer=16)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (< 500MB for mock model)
        assert memory_increase < 500 * 1024 * 1024, f"Memory increase too large: {memory_increase / 1024 / 1024:.1f}MB"
        
        # Verify execution completed successfully
        assert response is not None
        assert response.hidden_states is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])