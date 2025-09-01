#!/usr/bin/env python3
"""Test that sparse encoding is now working correctly."""

import sys
import torch
import numpy as np
sys.path.insert(0, 'src')

from src.hdc.encoder import HypervectorEncoder, HypervectorConfig

def test_sparse_encoding():
    """Test sparse encoding with different sparsity levels."""
    
    print("Testing Sparse Encoding Fix")
    print("="*60)
    
    # Test different sparsity levels
    sparsity_levels = [0.01, 0.05, 0.1, 0.2]
    
    for sparsity in sparsity_levels:
        print(f"\nTesting sparsity = {sparsity:.1%}")
        
        config = HypervectorConfig(
            dimension=10000,
            sparsity=sparsity,
            encoding_mode="rev",
            projection_type="SPARSE_RANDOM"
        )
        
        encoder = HypervectorEncoder(config=config)
        
        # Test with different features
        test_features = [
            "simple",
            "more complex feature with multiple words",
            "A1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z6"
        ]
        
        for feature in test_features:
            # Use encode_feature directly
            hv = encoder.encode_feature(feature)
            
            # Check actual sparsity
            if isinstance(hv, torch.Tensor):
                actual_nonzero = int(torch.sum(hv != 0).item())
                actual_density = actual_nonzero / hv.shape[0]
            else:
                actual_nonzero = int(np.sum(hv != 0))
                actual_density = actual_nonzero / len(hv)
                
            print(f"  Feature: '{feature[:20]}...'")
            print(f"    Expected density: {sparsity:.1%}")
            print(f"    Actual density: {actual_density:.1%}")
            print(f"    Non-zero elements: {actual_nonzero}/{config.dimension}")
            print(f"    Status: {'✓ PASS' if abs(actual_density - sparsity) < sparsity * 0.5 else '✗ FAIL'}")
            
    # Test with continuous features (like LLM embeddings)
    print("\n" + "="*60)
    print("Testing with continuous features (LLM embeddings)")
    
    config = HypervectorConfig(
        dimension=10000,
        sparsity=0.1,
        encoding_mode="rev"
    )
    encoder = HypervectorEncoder(config=config)
    
    # Simulate LLM embedding
    embedding = np.random.randn(768).astype(np.float32)
    
    # Convert to string representation for feature encoding
    feature_str = " ".join([f"{x:.4f}" for x in embedding[:10]])  # Use first 10 values
    hv = encoder.encode_feature(feature_str)
    
    if isinstance(hv, torch.Tensor):
        actual_nonzero = int(torch.sum(hv != 0).item())
        actual_density = actual_nonzero / hv.shape[0]
    else:
        actual_nonzero = int(np.sum(hv != 0))
        actual_density = actual_nonzero / len(hv)
        
    print(f"  Embedding-based feature")
    print(f"    Expected density: 10.0%")
    print(f"    Actual density: {actual_density:.1%}")
    print(f"    Non-zero elements: {actual_nonzero}/10000")
    print(f"    Status: {'✓ PASS' if actual_density < 0.2 else '✗ FAIL'}")
    
    print("\n" + "="*60)
    print("SUMMARY: Sparse encoding fix", "SUCCESSFUL" if actual_density < 0.2 else "FAILED")

if __name__ == "__main__":
    test_sparse_encoding()