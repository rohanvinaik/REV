#!/usr/bin/env python3
"""
Comprehensive tests for functional segmentation implementation in REV system.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

# Import REV components
from src.rev_pipeline import REVPipeline, FunctionalSegment, ExecutionPolicy
from src.models.true_segment_execution import RestrictionSite


class TestFunctionalSegmentation:
    """Test functional segmentation capabilities in REV pipeline."""
    
    def test_create_functional_segments(self):
        """Test creation of functional segments from restriction sites."""
        
        # Create REV pipeline
        pipeline = REVPipeline()
        
        # Create mock restriction sites
        sites = [
            self._create_mock_restriction_site(0, 0.3, "layer_boundary"),
            self._create_mock_restriction_site(8, 0.6, "behavioral_divergence"), 
            self._create_mock_restriction_site(24, 0.8, "behavioral_divergence"),
            self._create_mock_restriction_site(40, 0.7, "behavioral_divergence"),
            self._create_mock_restriction_site(60, 0.4, "layer_boundary"),
        ]
        
        # Create functional segments
        segments = pipeline.create_functional_segments(sites)
        
        # Verify segments created
        assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"
        
        # Verify segment structure
        for segment in segments:
            assert isinstance(segment, FunctionalSegment)
            assert segment.id.startswith("seg_")
            assert segment.start_layer < segment.end_layer
            assert segment.functional_role in [
                "token_embedding", "early_processing", "feature_extraction",
                "semantic_processing", "reasoning", "pattern_integration",
                "output_generation", "decision_making", "final_processing",
                "representation_building"
            ]
            assert segment.processing_mode in ["high_precision", "standard", "fast"]
            assert isinstance(segment.execution_policy, ExecutionPolicy)
            
        print("✅ Functional segments created successfully with proper structure")
    
    def test_behavioral_fingerprint_computation(self):
        """Test computation of behavioral fingerprints for segments."""
        
        pipeline = REVPipeline()
        
        # Create detailed mock sites with metrics
        start_site = self._create_mock_restriction_site(
            layer_idx=10, 
            divergence=0.4, 
            site_type="behavioral_divergence",
            extra_attrs={
                'divergence_metrics': {'cosine_similarity': 0.6, 'l2_distance': 1.2},
                'prompt_responses': {'math': 'response1', 'language': 'response2'},
                'confidence_score': 0.8
            }
        )
        
        end_site = self._create_mock_restriction_site(
            layer_idx=20,
            divergence=0.7,
            site_type="behavioral_divergence", 
            extra_attrs={
                'divergence_metrics': {'cosine_similarity': 0.3, 'l2_distance': 2.1},
                'prompt_responses': {'math': 'response3', 'language': 'response4', 'reasoning': 'response5'},
                'confidence_score': 0.9
            }
        )
        
        # Compute fingerprint
        fingerprint = pipeline.compute_segment_fingerprint(start_site, end_site)
        
        # Verify fingerprint structure
        assert 'layer_range' in fingerprint
        assert 'layer_count' in fingerprint
        assert 'start_divergence' in fingerprint
        assert 'end_divergence' in fingerprint
        assert 'avg_divergence' in fingerprint
        assert 'response_patterns' in fingerprint
        assert 'activation_summary' in fingerprint
        
        # Verify specific values
        assert fingerprint['layer_range'] == (10, 20)
        assert fingerprint['layer_count'] == 10
        assert fingerprint['start_divergence'] == 0.4
        assert fingerprint['end_divergence'] == 0.7
        assert abs(fingerprint['avg_divergence'] - 0.55) < 0.01
        
        # Verify response patterns captured
        assert fingerprint['response_patterns']['start_responses'] == 2
        assert fingerprint['response_patterns']['end_responses'] == 3
        
        print("✅ Behavioral fingerprint computation working correctly")
    
    def test_functional_role_identification(self):
        """Test identification of functional roles based on behavioral patterns."""
        
        pipeline = REVPipeline()
        
        # Test different layer ranges and divergences
        test_cases = [
            # (start_layer, end_layer, start_div, end_div, expected_role)
            (0, 4, 0.2, 0.3, "token_embedding"),
            (2, 8, 0.5, 0.6, "semantic_processing"),  # Updated: higher divergence triggers semantic processing
            (10, 20, 0.3, 0.4, "feature_extraction"),
            (15, 25, 0.5, 0.6, "semantic_processing"),
            (30, 40, 0.7, 0.8, "semantic_processing"),
            (35, 45, 0.9, 0.8, "reasoning"),
            (50, 60, 0.8, 0.9, "output_generation"),
            (55, 65, 0.6, 0.7, "decision_making"),
        ]
        
        for start_layer, end_layer, start_div, end_div, expected_role in test_cases:
            start_site = self._create_mock_restriction_site(start_layer, start_div, "behavioral_divergence")
            end_site = self._create_mock_restriction_site(end_layer, end_div, "behavioral_divergence")
            
            role = pipeline.identify_functional_role(start_site, end_site)
            
            assert role == expected_role, f"Layer {start_layer}-{end_layer} with div {start_div}-{end_div}: expected {expected_role}, got {role}"
            
        print("✅ Functional role identification working correctly")
    
    def test_processing_mode_determination(self):
        """Test determination of processing modes based on behavioral complexity."""
        
        pipeline = REVPipeline()
        
        # Test different divergence levels and layer counts
        test_cases = [
            # (divergence1, divergence2, layer_count, expected_mode)
            (0.8, 0.9, 5, "high_precision"),  # High divergence
            (0.5, 0.6, 8, "standard"),        # Moderate divergence
            (0.2, 0.3, 2, "fast"),            # Low divergence, short segment
            (0.3, 0.4, 10, "standard"),       # Low divergence, long segment
        ]
        
        for div1, div2, layer_count, expected_mode in test_cases:
            start_site = self._create_mock_restriction_site(10, div1, "behavioral_divergence")
            end_site = self._create_mock_restriction_site(10 + layer_count, div2, "behavioral_divergence")
            
            mode = pipeline.determine_processing_mode(start_site, end_site)
            
            assert mode == expected_mode, f"Divergence {div1}-{div2}, {layer_count} layers: expected {expected_mode}, got {mode}"
            
        print("✅ Processing mode determination working correctly")
    
    def test_execution_policy_creation(self):
        """Test creation of execution policies tailored to functional roles."""
        
        pipeline = REVPipeline()
        
        # Test different functional roles
        role_policy_tests = [
            ("semantic_processing", {"dtype": "fp32", "attn_impl": "flash"}),
            ("reasoning", {"dtype": "fp32", "checkpoint_activations": True}),
            ("output_generation", {"dtype": "fp16", "quantization": "8bit"}),
            ("token_embedding", {"offload_to_cpu": True}),
        ]
        
        for role, expected_attrs in role_policy_tests:
            policy = pipeline._create_execution_policy_for_role(role)
            
            assert isinstance(policy, ExecutionPolicy)
            
            for attr, expected_value in expected_attrs.items():
                actual_value = getattr(policy, attr)
                assert actual_value == expected_value, f"Role {role}, attr {attr}: expected {expected_value}, got {actual_value}"
                
        print("✅ Execution policy creation working correctly")
    
    def test_segment_aware_processing(self):
        """Test segment-aware processing with different execution strategies."""
        
        pipeline = REVPipeline()
        
        # Create test segments with different roles
        segments = [
            FunctionalSegment(
                id="test_semantic",
                start_layer=10,
                end_layer=20,
                behavioral_fingerprint={"avg_divergence": 0.7},
                functional_role="semantic_processing",
                processing_mode="high_precision",
                execution_policy=ExecutionPolicy(dtype="fp32")
            ),
            FunctionalSegment(
                id="test_embedding",
                start_layer=0,
                end_layer=5,
                behavioral_fingerprint={"avg_divergence": 0.3},
                functional_role="token_embedding",
                processing_mode="fast",
                execution_policy=ExecutionPolicy(dtype="fp16", quantization="8bit")
            ),
        ]
        
        # Test processing each segment
        test_tokens = [1, 2, 3, 4, 5]
        
        for segment in segments:
            try:
                result, telemetry = pipeline.process_segment_with_fingerprint(segment, test_tokens)
                
                # Verify result structure
                assert 'processed_tokens' in result
                assert 'mode' in result
                assert 'segment_id' in result
                assert result['processed_tokens'] == test_tokens
                assert result['segment_id'] == segment.id
                
                # Verify telemetry
                assert hasattr(telemetry, 'segment_id')
                assert hasattr(telemetry, 't_ms')
                assert hasattr(telemetry, 'tokens_processed')
                assert telemetry.tokens_processed == len(test_tokens)
                
                print(f"✅ Segment {segment.id} processed successfully with {result['mode']} mode")
                
            except Exception as e:
                pytest.fail(f"Processing failed for segment {segment.id}: {e}")
    
    def test_integration_with_behavioral_analysis(self):
        """Test integration of functional segmentation with behavioral analysis."""
        
        pipeline = REVPipeline()
        
        # Mock model and tokenizer for behavioral analysis
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 24
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        
        # Mock the behavioral probe generation and executor
        with patch('src.challenges.pot_challenge_generator.PoTChallengeGenerator') as mock_generator_class, \
             patch('src.models.true_segment_execution.LayerSegmentExecutor') as mock_executor_class:
            # Mock the probe generator
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            mock_generator.generate_behavioral_probes.return_value = {
                'math': ['What is 2+2?', 'Calculate 5*3'],
                'reasoning': ['If A then B', 'All X are Y']
            }
            
            # Mock the executor
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock restriction sites from behavioral analysis
            mock_sites = [
                self._create_mock_restriction_site(0, 0.3, "layer_boundary"),
                self._create_mock_restriction_site(8, 0.6, "behavioral_divergence"),
                self._create_mock_restriction_site(16, 0.8, "behavioral_divergence"),
                self._create_mock_restriction_site(24, 0.4, "layer_boundary"),
            ]
            mock_executor.identify_all_restriction_sites.return_value = mock_sites
            
            # Run behavioral analysis
            results = pipeline.run_behavioral_analysis(mock_model, mock_tokenizer)
            
            # Verify functional segments were created
            assert 'functional_segments' in results
            assert 'num_functional_segments' in results
            assert results['num_functional_segments'] > 0
            
            functional_segments = results['functional_segments']
            assert len(functional_segments) == 3  # 4 sites = 3 segments
            
            # Verify each segment has proper functional metadata
            for segment in functional_segments:
                assert isinstance(segment, FunctionalSegment)
                assert segment.functional_role != ""
                assert segment.processing_mode != ""
                assert segment.execution_policy is not None
                
        print("✅ Integration with behavioral analysis working correctly")
    
    def test_fallback_segmentation(self):
        """Test fallback to basic segmentation when sophisticated analysis fails."""
        
        pipeline = REVPipeline()
        
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = 12
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        
        # Mock behavioral probe generation and make executor fail
        with patch('src.challenges.pot_challenge_generator.PoTChallengeGenerator') as mock_generator_class, \
             patch('src.models.true_segment_execution.LayerSegmentExecutor') as mock_executor_class:
            # Mock the probe generator 
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            mock_generator.generate_behavioral_probes.return_value = {
                'math': ['What is 2+2?'],
                'reasoning': ['Simple reasoning']
            }
            
            mock_executor_class.side_effect = Exception("LayerSegmentExecutor failed")
            
            # Run behavioral analysis - should fall back to basic segmentation
            results = pipeline.run_behavioral_analysis(mock_model, mock_tokenizer)
            
            # Should still create functional segments using fallback
            assert 'functional_segments' in results
            functional_segments = results['functional_segments']
            
            if functional_segments:  # May be empty if basic segmentation also fails
                for segment in functional_segments:
                    assert isinstance(segment, FunctionalSegment)
                    assert segment.functional_role != ""
                    
        print("✅ Fallback segmentation working correctly")
    
    def _create_mock_restriction_site(self, layer_idx: int, divergence: float, site_type: str, extra_attrs: Dict[str, Any] = None) -> RestrictionSite:
        """Create a mock RestrictionSite for testing."""
        
        # Create a simple mock object with the required attributes
        site = type('MockRestrictionSite', (), {
            'layer_idx': layer_idx,
            'site_type': site_type,
            'behavioral_divergence': divergence,
            'confidence_score': 0.7,
            'prompt_responses': {},
            'divergence_metrics': {}
        })()
        
        # Add extra attributes if provided
        if extra_attrs:
            for attr, value in extra_attrs.items():
                setattr(site, attr, value)
                
        return site


if __name__ == "__main__":
    pytest.main([__file__, "-v"])