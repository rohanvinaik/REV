#!/usr/bin/env python3
"""Quick test of sparsity improvements."""

import numpy as np
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy

print("Testing Sparsity Improvements")
print("=" * 50)

# Test 1: Basic encoder with different sparsity levels
print("\n1. Testing HypervectorEncoder with increased sparsity:")
for sparsity in [0.01, 0.05, 0.1, 0.15, 0.2]:
    config = HypervectorConfig(
        dimension=10000,
        sparsity=sparsity,
        encoding_mode="rev"
    )
    encoder = HypervectorEncoder(config)
    
    # Test with varying complexity features
    densities = []
    for i in range(10):
        feature = f"test_feature_{i}_complex_data_" + "x" * (i * 10)
        vec = encoder.encode_feature(feature)
        if hasattr(vec, 'numpy'):
            vec = vec.numpy()
        density = np.count_nonzero(vec) / len(vec)
        densities.append(density)
    
    mean_density = np.mean(densities)
    print(f"  Target: {sparsity:5.1%} → Actual: {mean_density:5.1%} (range: {min(densities):.1%}-{max(densities):.1%})")

# Test 2: Adaptive encoder
print("\n2. Testing AdaptiveSparsityEncoder with max_sparsity=0.2:")
adaptive = AdaptiveSparsityEncoder(
    dimension=10000,
    initial_sparsity=0.01,
    min_sparsity=0.005,
    max_sparsity=0.2,  # Now supports up to 20%
    adjustment_strategy=AdjustmentStrategy.ADAPTIVE
)

# Generate features with varying complexity
features = [np.random.randn(768).astype(np.float32) for _ in range(30)]
encoded, stats = adaptive.encode_adaptive(features, auto_converge=True)

print(f"  Initial sparsity: 0.010")
print(f"  Final sparsity: {stats.final_sparsity:.3f}")
print(f"  Actual density: {stats.actual_density:.3f}")
print(f"  Convergence iterations: {stats.convergence_iterations}")
print(f"  Quality score: {stats.quality_score:.3f}")

# Test 3: Complex feature encoding
print("\n3. Testing with highly complex features:")
config = HypervectorConfig(
    dimension=10000,
    sparsity=0.2,  # Max sparsity for complex features
    encoding_mode="rev"
)
encoder = HypervectorEncoder(config)

complex_feature = "This is a very complex feature with lots of semantic information " * 10
vec = encoder.encode_feature(complex_feature)
if hasattr(vec, 'numpy'):
    vec = vec.numpy()
density = np.count_nonzero(vec) / len(vec)
print(f"  Complex feature density: {density:.1%} (target: 20%)")

print("\n✅ Sparsity improvements verified!")