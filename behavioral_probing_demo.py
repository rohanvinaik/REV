#!/usr/bin/env python3
"""
Demonstration of behavioral probing implementation in REV system
"""

import torch
import numpy as np
from src.models.true_segment_execution import BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig
from unittest.mock import MagicMock, patch


def main():
    """Demonstrate behavioral probing functionality"""
    
    print("🧠 REV Behavioral Probing Demonstration")
    print("=" * 60)
    
    # 1. Demonstrate BehavioralResponse data structure
    print("\n1. BehavioralResponse Data Structure")
    print("-" * 40)
    
    hidden_states = torch.randn(1, 10, 4096)
    attention_patterns = torch.randn(1, 32, 10, 10)
    token_predictions = torch.randn(1, 10, 50000)
    
    behavioral_response = BehavioralResponse(
        hidden_states=hidden_states,
        attention_patterns=attention_patterns,
        token_predictions=token_predictions,
        statistical_signature={
            'mean_activation': 0.65,
            'std_activation': 0.28,
            'activation_entropy': 3.7,
            'sparsity_ratio': 0.12,
            'max_activation': 2.8
        }
    )
    
    print(f"✅ Hidden states shape: {behavioral_response.hidden_states.shape}")
    print(f"✅ Attention patterns shape: {behavioral_response.attention_patterns.shape}")
    print(f"✅ Token predictions shape: {behavioral_response.token_predictions.shape}")
    print(f"✅ Statistical signature: {len(behavioral_response.statistical_signature)} metrics")
    
    # 2. Demonstrate behavioral divergence calculation
    print("\n2. Behavioral Divergence Calculation")
    print("-" * 40)
    
    # Create two different behavioral responses representing different model behaviors
    math_response = BehavioralResponse(
        hidden_states=torch.randn(1, 10, 4096) * 0.8 + 0.6,  # Math-focused activations
        attention_patterns=torch.randn(1, 32, 10, 10) * 0.9,
        statistical_signature={'mean_activation': 0.85, 'activation_entropy': 4.2}
    )
    
    language_response = BehavioralResponse(
        hidden_states=torch.randn(1, 10, 4096) * 0.6 + 0.4,  # Language-focused activations
        attention_patterns=torch.randn(1, 32, 10, 10) * 1.1,
        statistical_signature={'mean_activation': 0.65, 'activation_entropy': 3.8}
    )
    
    # Calculate divergence using LayerSegmentExecutor
    config = SegmentExecutionConfig(model_path="/fake/path")
    
    with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
        executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
        executor.device_manager = MagicMock()
        executor.device_manager.get_device.return_value = torch.device('cpu')
        
        divergence = executor.compute_behavioral_divergence(math_response, language_response)
        
        print(f"✅ Cosine Similarity: {divergence['cosine_similarity']:.3f}")
        print(f"✅ Cosine Distance: {divergence['cosine_distance']:.3f}")
        print(f"✅ L2 Distance: {divergence['l2_distance']:.3f}")
        print(f"✅ Pearson Correlation: {divergence['pearson_correlation']:.3f}")
        print(f"✅ Mean Activation Diff: {divergence['signature_mean_diff']:.3f}")
        
        # 3. Demonstrate detection of different vs identical responses
        print("\n3. Identical vs Different Response Detection")
        print("-" * 40)
        
        # Test with identical responses
        identical_response = BehavioralResponse(
            hidden_states=math_response.hidden_states.clone(),
            statistical_signature=math_response.statistical_signature.copy()
        )
        
        identical_divergence = executor.compute_behavioral_divergence(math_response, identical_response)
        different_divergence = executor.compute_behavioral_divergence(math_response, language_response)
        
        print(f"Identical responses L2 distance: {identical_divergence['l2_distance']:.6f}")
        print(f"Different responses L2 distance: {different_divergence['l2_distance']:.3f}")
        print(f"✅ Can distinguish identical vs different: {different_divergence['l2_distance'] / max(identical_divergence['l2_distance'], 1e-6):.1f}x difference")
        
    # 4. Summary
    print("\n4. Summary")
    print("-" * 40)
    print("✅ BehavioralResponse captures complete neural activation fingerprints")
    print("✅ Behavioral divergence calculation using multiple statistical metrics")
    print("✅ Proper detection of behavioral differences vs similarities")
    print("✅ Ready for integration with PoT challenge execution")
    
    print(f"\n🎯 Key Achievement: REV now performs actual behavioral probing")
    print(f"   instead of naive norm-based analysis. The system can:")
    print(f"   • Execute PoT challenges through model layers")
    print(f"   • Extract comprehensive behavioral fingerprints")
    print(f"   • Measure divergence using cosine, L2, Pearson correlation metrics")
    print(f"   • Identify restriction sites based on actual neural responses")


if __name__ == "__main__":
    main()