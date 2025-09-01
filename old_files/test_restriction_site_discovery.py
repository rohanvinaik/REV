#!/usr/bin/env python3
"""
Test sophisticated restriction site discovery mechanism
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import REV components
from src.models.true_segment_execution import (
    BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig, RestrictionSite
)


class TestRestrictionSiteDiscovery:
    """Test suite for sophisticated restriction site discovery"""
    
    def test_sophisticated_restriction_site_discovery(self):
        """Test that the new discovery mechanism finds appropriate sites with behavioral analysis"""
        
        # Create diverse PoT challenges representing different reasoning types
        diverse_probes = [
            "Calculate the sum of prime numbers less than 20",
            "If all birds can fly and penguins are birds, can penguins fly?",
            "Translate 'hello world' into French and explain the grammar",
            "Solve: 2x + 5 = 17 for x",
            "What is the logical fallacy in: 'All cats are animals, so all animals are cats'?",
            "Write a function to find the factorial of a number",
            "Explain the difference between correlation and causation",
            "What is 15% of 240?"
        ]
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.n_layers = 24  # 24-layer model
            executor.config = config
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Mock profile_layer_behavior to return realistic profiles with behavioral patterns
            def mock_profile_layer_behavior(layer_idx, probe_prompts):
                # Simulate different behavioral patterns at different layers
                
                # Early layers (0-7): Basic token processing
                if layer_idx < 8:
                    base_activation = 0.4 + 0.05 * layer_idx
                    attention_concentration = 0.3 + 0.1 * layer_idx
                    reasoning_preference = {'linguistic': 0.6, 'general': 0.4}
                
                # Middle layers (8-15): Complex reasoning emerges
                elif layer_idx < 16:
                    base_activation = 0.7 + 0.1 * (layer_idx % 4)  # Pattern every 4 layers
                    attention_concentration = 0.8 + 0.1 * np.sin(layer_idx)
                    reasoning_preference = {'mathematical': 0.4, 'logical': 0.4, 'linguistic': 0.2}
                
                # Later layers (16-23): Output generation  
                else:
                    base_activation = 0.6 + 0.15 * (layer_idx % 3)  # Pattern every 3 layers
                    attention_concentration = 0.9 - 0.1 * (layer_idx - 16) / 8
                    reasoning_preference = {'general': 0.5, 'linguistic': 0.3, 'logical': 0.2}
                
                return {
                    'layer_idx': layer_idx,
                    'statistical_aggregates': {
                        'mean_activation_across_probes': base_activation,
                        'std_activation_across_probes': 0.2,
                        'activation_range': 2.0,
                        'probe_consistency': 0.1,
                        'entropy_variance': 0.3,
                        'sparsity_mean': 0.15
                    },
                    'behavioral_features': {
                        'attention_concentration': attention_concentration,
                        'activation_diversity': 0.5 + 0.1 * (layer_idx % 5),
                        'response_stability': 0.7 + 0.1 * np.cos(layer_idx),
                        'reasoning_type_preference': reasoning_preference
                    },
                    'divergence_signatures': {
                        'mean_divergence': 0.3 + 0.1 * (layer_idx % 6),
                        'max_divergence': 0.5 + 0.2 * (layer_idx % 4),
                        'std_divergence': 0.1
                    },
                    'probe_responses': [
                        {'probe_id': i, 'probe_text': probe[:50], 'response_signature': {'mean_activation': base_activation}}
                        for i, probe in enumerate(probe_prompts[:3])
                    ]
                }
            
            executor.profile_layer_behavior = mock_profile_layer_behavior
            
            # Execute restriction site discovery
            restriction_sites = executor.identify_all_restriction_sites(diverse_probes)
            
            # Validate results meet test expectations
            assert len(restriction_sites) >= 6, f"Should find at least 6 sites, found {len(restriction_sites)}"
            assert len(restriction_sites) <= 12, f"Should find at most 12 sites, found {len(restriction_sites)}"
            
            # Check divergence scores are in expected range (0.2-0.5)
            behavioral_sites = [s for s in restriction_sites if s.site_type == "behavioral_divergence"]
            assert len(behavioral_sites) >= 3, f"Should find at least 3 behavioral sites, found {len(behavioral_sites)}"
            
            for site in behavioral_sites:
                assert 0.2 <= site.behavioral_divergence <= 0.8, \
                    f"Site at layer {site.layer_idx} has divergence {site.behavioral_divergence:.3f}, expected 0.2-0.8"
                assert 0.0 <= site.confidence_score <= 1.0, \
                    f"Confidence score {site.confidence_score} should be between 0 and 1"
            
            # Verify sites are properly sorted and distributed
            layer_indices = [s.layer_idx for s in restriction_sites]
            assert layer_indices == sorted(layer_indices), "Sites should be sorted by layer index"
            
            # Check distribution (no layer should appear twice)
            assert len(set(layer_indices)) == len(layer_indices), "No duplicate layer indices"
            
            print(f"✅ Successfully discovered {len(restriction_sites)} restriction sites")
            print(f"✅ Behavioral sites: {len(behavioral_sites)}")
            print(f"✅ Divergence range: {min(s.behavioral_divergence for s in behavioral_sites):.3f} - {max(s.behavioral_divergence for s in behavioral_sites):.3f}")
            
            # Log discovered sites
            for i, site in enumerate(restriction_sites):
                print(f"   Site {i+1}: Layer {site.layer_idx} ({site.site_type}, div={site.behavioral_divergence:.3f})")
    
    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold calculation with realistic divergence distributions"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Create realistic divergence matrix with natural boundaries
            n_layers = 16
            divergence_matrix = np.zeros((n_layers, n_layers))
            
            # Simulate behavioral boundaries at layers 4, 8, 12 with higher divergences
            boundary_layers = [4, 8, 12]
            
            for i in range(n_layers):
                for j in range(n_layers):
                    if i != j:
                        base_divergence = 0.1 + 0.05 * abs(i - j)  # Distance-based divergence
                        
                        # Add spikes at boundary layers
                        if min(i, j) in boundary_layers or max(i, j) in boundary_layers:
                            base_divergence += 0.3  # Boundary boost
                        
                        # Add some noise
                        noise = np.random.normal(0, 0.02)
                        divergence_matrix[i, j] = max(0, base_divergence + noise)
            
            # Test adaptive threshold calculation
            adaptive_threshold = executor._compute_adaptive_threshold(divergence_matrix, target_sites=9)
            
            # Threshold should be reasonable
            assert 0.1 <= adaptive_threshold <= 0.8, f"Adaptive threshold {adaptive_threshold:.3f} outside reasonable range"
            
            # Should be higher than baseline divergence but lower than boundary spikes
            baseline_div = np.mean(divergence_matrix[divergence_matrix > 0])
            boundary_div = np.max(divergence_matrix)
            
            assert baseline_div < adaptive_threshold < boundary_div, \
                f"Threshold {adaptive_threshold:.3f} should be between baseline {baseline_div:.3f} and boundary {boundary_div:.3f}"
            
            print(f"✅ Adaptive threshold: {adaptive_threshold:.3f}")
            print(f"✅ Baseline divergence: {baseline_div:.3f}")
            print(f"✅ Boundary divergence: {boundary_div:.3f}")
    
    def test_local_maxima_detection(self):
        """Test local maxima detection for boundary identification"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Create divergence matrix with clear local maxima
            n_layers = 12
            divergence_matrix = np.zeros((n_layers, n_layers))
            
            # Create pattern: low divergence normally, high at layers 3, 6, 9
            for i in range(n_layers):
                for j in range(n_layers):
                    if i != j:
                        dist = abs(i - j)
                        base_div = 0.1 + 0.02 * dist
                        
                        # Spikes at layers 3, 6, 9
                        if abs(i - 3) <= 1 or abs(j - 3) <= 1:
                            base_div += 0.4
                        if abs(i - 6) <= 1 or abs(j - 6) <= 1:
                            base_div += 0.5
                        if abs(i - 9) <= 1 or abs(j - 9) <= 1:
                            base_div += 0.3
                        
                        divergence_matrix[i, j] = base_div
            
            # Test local maxima detection
            threshold = 0.25
            local_maxima = executor._find_local_maxima(divergence_matrix, threshold)
            
            # Should detect the boundary layers
            expected_boundaries = [3, 6, 9]
            
            # Allow some tolerance in detection
            detected_boundaries = set(local_maxima)
            expected_boundaries_set = set(expected_boundaries)
            
            intersection = detected_boundaries.intersection(expected_boundaries_set)
            assert len(intersection) >= 2, f"Should detect at least 2 of {expected_boundaries}, detected {local_maxima}"
            
            print(f"✅ Detected boundaries: {local_maxima}")
            print(f"✅ Expected boundaries: {expected_boundaries}")
            print(f"✅ Correctly identified: {list(intersection)}")
    
    def test_fallback_mechanism(self):
        """Test fallback to hardcoded sites when behavioral discovery fails"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.n_layers = 32
            executor.config = config
            
            # Mock profile_layer_behavior to always fail
            def mock_failing_profile(layer_idx, probe_prompts):
                return {'error': 'Mocked failure'}
            
            executor.profile_layer_behavior = mock_failing_profile
            
            # Should fallback to hardcoded sites
            restriction_sites = executor.identify_all_restriction_sites(["test probe"])
            
            # Validate fallback behavior
            assert len(restriction_sites) >= 6, f"Fallback should produce at least 6 sites, got {len(restriction_sites)}"
            assert len(restriction_sites) <= 10, f"Fallback should produce at most 10 sites, got {len(restriction_sites)}"
            
            # All sites should be hardcoded_fallback type
            for site in restriction_sites:
                assert site.site_type == "hardcoded_fallback"
                assert site.confidence_score == 0.0  # Zero confidence for hardcoded
                assert site.behavioral_divergence == 0.3  # Default divergence
            
            print(f"✅ Fallback mechanism produces {len(restriction_sites)} hardcoded sites")
            
    def test_behavioral_profile_generation(self):
        """Test comprehensive behavioral profile generation"""
        
        config = SegmentExecutionConfig(model_path="/fake/path")
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            executor.device_manager = MagicMock()
            executor.device_manager.get_device.return_value = torch.device('cpu')
            
            # Mock execute_behavioral_probe to return varied responses
            def mock_execute_probe(probe_text, up_to_layer):
                # Vary response based on probe type and layer
                if 'math' in probe_text.lower() or 'calculate' in probe_text.lower():
                    activation_multiplier = 1.2  # Math probes activate more
                elif 'logic' in probe_text.lower() or 'fallacy' in probe_text.lower():
                    activation_multiplier = 1.0   # Logic probes normal activation
                else:
                    activation_multiplier = 0.8   # Other probes activate less
                
                layer_factor = 1.0 + 0.1 * up_to_layer  # Layer-dependent activation
                
                hidden_states = torch.randn(1, 5, 4096) * activation_multiplier * layer_factor
                
                return BehavioralResponse(
                    hidden_states=hidden_states,
                    attention_patterns=torch.randn(1, 32, 5, 5),
                    statistical_signature={
                        'mean_activation': float(torch.mean(hidden_states)),
                        'activation_entropy': 3.0 + 0.5 * up_to_layer,
                        'sparsity_ratio': 0.1 + 0.02 * up_to_layer
                    }
                )
            
            executor.execute_behavioral_probe = mock_execute_probe
            
            # Test profiling for different layers
            diverse_probes = [
                "Calculate 15 + 27",
                "Is this statement logical: All dogs bark, Rover barks, therefore Rover is a dog?",
                "Translate: Bonjour"
            ]
            
            # Profile early layer (should show basic patterns)
            early_profile = executor.profile_layer_behavior(2, diverse_probes)
            
            # Profile middle layer (should show complex patterns)  
            middle_profile = executor.profile_layer_behavior(12, diverse_probes)
            
            # Validate profile structure
            for profile in [early_profile, middle_profile]:
                assert 'statistical_aggregates' in profile
                assert 'behavioral_features' in profile
                assert 'divergence_signatures' in profile
                assert 'probe_responses' in profile
                
                # Check statistical aggregates
                stats = profile['statistical_aggregates']
                required_stats = ['mean_activation_across_probes', 'std_activation_across_probes', 
                                'activation_range', 'probe_consistency', 'entropy_variance']
                for stat in required_stats:
                    assert stat in stats
                    assert np.isfinite(stats[stat])
                
                # Check behavioral features
                features = profile['behavioral_features']
                required_features = ['attention_concentration', 'activation_diversity', 
                                   'response_stability', 'reasoning_type_preference']
                for feature in required_features:
                    assert feature in features
            
            # Middle layer should have different characteristics than early layer
            early_attention = early_profile['behavioral_features']['attention_concentration']
            middle_attention = middle_profile['behavioral_features']['attention_concentration']
            
            assert abs(early_attention - middle_attention) > 0.1, \
                "Different layers should have different attention patterns"
            
            print(f"✅ Early layer attention concentration: {early_attention:.3f}")
            print(f"✅ Middle layer attention concentration: {middle_attention:.3f}")
            print(f"✅ Behavioral profiles successfully generated with distinct characteristics")
    
    def test_divergence_matrix_computation(self):
        """Test divergence matrix computation between behavioral profiles"""
        
        config = SegmentExecutionConfig(model_path="/fake/path") 
        
        with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
            executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
            
            # Create mock behavioral profiles with distinct characteristics
            behavioral_profiles = {}
            
            for layer_idx in range(6):  # 6 layers for testing
                # Create distinct profiles
                if layer_idx < 2:
                    # Early layers: linguistic focus
                    reasoning_pref = {'linguistic': 0.8, 'general': 0.2}
                    attention_conc = 0.4
                elif layer_idx < 4:
                    # Middle layers: mathematical focus
                    reasoning_pref = {'mathematical': 0.7, 'logical': 0.3}
                    attention_conc = 0.8
                else:
                    # Later layers: balanced
                    reasoning_pref = {'general': 0.5, 'linguistic': 0.3, 'logical': 0.2}
                    attention_conc = 0.6
                
                behavioral_profiles[layer_idx] = {
                    'statistical_aggregates': {
                        'mean_activation_across_probes': 0.5 + 0.1 * layer_idx,
                        'std_activation_across_probes': 0.2,
                        'activation_range': 2.0,
                        'probe_consistency': 0.1,
                        'entropy_variance': 0.3
                    },
                    'behavioral_features': {
                        'attention_concentration': attention_conc,
                        'activation_diversity': 0.5,
                        'response_stability': 0.7,
                        'reasoning_type_preference': reasoning_pref
                    }
                }
            
            # Compute divergence matrix
            divergence_matrix = executor.compute_divergence_matrix(behavioral_profiles)
            
            # Validate matrix properties
            assert divergence_matrix.shape == (6, 6), "Matrix should be square"
            assert np.allclose(divergence_matrix, divergence_matrix.T), "Matrix should be symmetric"
            assert np.allclose(np.diag(divergence_matrix), 0), "Diagonal should be zero"
            
            # Validate divergence values
            for i in range(6):
                for j in range(6):
                    if i != j:
                        div_val = divergence_matrix[i, j]
                        assert 0 <= div_val <= 2.0, f"Divergence [{i},{j}] = {div_val} outside reasonable range"
            
            # Different layer types should have higher divergence
            linguistic_mathematical_div = divergence_matrix[0, 2]  # Layer 0 vs Layer 2
            within_type_div = divergence_matrix[0, 1]  # Layer 0 vs Layer 1 (both linguistic)
            
            assert linguistic_mathematical_div > within_type_div, \
                "Cross-type divergence should be higher than within-type"
            
            print(f"✅ Divergence matrix computed successfully: {divergence_matrix.shape}")
            print(f"✅ Cross-type divergence: {linguistic_mathematical_div:.3f}")
            print(f"✅ Within-type divergence: {within_type_div:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])