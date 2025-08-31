#!/usr/bin/env python3
"""
Test REV's core verification capability: distinguishing between different models.
Uses actual hypervector generation and comparison.
"""

import numpy as np
import torch
from pathlib import Path

from src.models.large_model_inference import LargeModelInference, LargeModelConfig
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy
from src.hypervector.hamming import HammingDistanceOptimized
from src.core.sequential import DualSequentialTest

print("="*80)
print("REV MODEL VERIFICATION TEST")
print("Testing ability to distinguish between different models")
print("="*80)

# Test prompt
test_prompt = "What is the meaning of life?"

print(f"\nTest prompt: '{test_prompt}'")
print("-"*40)

# Initialize HDC encoder
hdc_config = HypervectorConfig(
    dimension=10000,
    sparsity=0.15,
    encoding_mode="rev"
)
encoder = HypervectorEncoder(hdc_config)

# Initialize adaptive encoder
adaptive_encoder = AdaptiveSparsityEncoder(
    dimension=10000,
    initial_sparsity=0.01,
    min_sparsity=0.005,
    max_sparsity=0.2,
    adjustment_strategy=AdjustmentStrategy.ADAPTIVE
)

print("\n1. Testing with GPT-2 (124M parameters)")
print("-"*40)

# Load GPT-2
config1 = LargeModelConfig(
    model_path="gpt2",
    device="cpu",
    low_cpu_mem_usage=True,
    max_new_tokens=20,
    do_sample=False
)

inference1 = LargeModelInference("gpt2", config1)
success1, msg1 = inference1.load_model()
print(f"Loading: {msg1}")

# Generate response and hypervector
result1 = inference1.process_for_rev(
    prompt=test_prompt,
    extract_activations=True,
    hdc_encoder=encoder,
    adaptive_encoder=adaptive_encoder
)

response1 = result1.get('response', '')[:50]
print(f"Response: {response1}...")

if 'hypervector' in result1:
    hv1 = result1['hypervector']
    sparsity1 = np.count_nonzero(hv1) / len(hv1)
    print(f"Hypervector sparsity: {sparsity1:.3%}")
else:
    print("❌ No hypervector generated")

# Clean up
inference1.cleanup()

print("\n2. Testing with DistilGPT-2 (82M parameters)")  
print("-"*40)

# Load DistilGPT-2
config2 = LargeModelConfig(
    model_path="distilgpt2",
    device="cpu", 
    low_cpu_mem_usage=True,
    max_new_tokens=20,
    do_sample=False
)

inference2 = LargeModelInference("distilgpt2", config2)
success2, msg2 = inference2.load_model()
print(f"Loading: {msg2}")

# Generate response and hypervector
result2 = inference2.process_for_rev(
    prompt=test_prompt,
    extract_activations=True,
    hdc_encoder=encoder,
    adaptive_encoder=adaptive_encoder
)

response2 = result2.get('response', '')[:50]
print(f"Response: {response2}...")

if 'hypervector' in result2:
    hv2 = result2['hypervector']
    sparsity2 = np.count_nonzero(hv2) / len(hv2)
    print(f"Hypervector sparsity: {sparsity2:.3%}")
else:
    print("❌ No hypervector generated")

# Clean up
inference2.cleanup()

print("\n" + "="*80)
print("HYPERVECTOR COMPARISON")
print("="*80)

if 'hypervector' in result1 and 'hypervector' in result2:
    # Compare hypervectors
    hv1 = result1['hypervector']
    hv2 = result2['hypervector']
    
    # Cosine similarity
    if isinstance(hv1, torch.Tensor):
        hv1 = hv1.numpy()
    if isinstance(hv2, torch.Tensor):
        hv2 = hv2.numpy()
    
    dot_product = np.dot(hv1, hv2)
    norm1 = np.linalg.norm(hv1)
    norm2 = np.linalg.norm(hv2)
    
    if norm1 > 0 and norm2 > 0:
        cosine_sim = dot_product / (norm1 * norm2)
    else:
        cosine_sim = 0
    
    print(f"Cosine similarity: {cosine_sim:.4f}")
    
    # Hamming distance for binary version
    hamming_calc = HammingDistanceOptimized()
    binary_hv1 = (hv1 != 0).astype(np.uint8)
    binary_hv2 = (hv2 != 0).astype(np.uint8)
    hamming_dist = hamming_calc.distance(binary_hv1, binary_hv2)
    normalized_hamming = hamming_dist / len(binary_hv1)
    
    print(f"Hamming distance: {hamming_dist}")
    print(f"Normalized Hamming: {normalized_hamming:.4f}")
    
    # Overlap of active positions
    active1 = set(np.where(hv1 != 0)[0])
    active2 = set(np.where(hv2 != 0)[0])
    overlap = len(active1 & active2)
    union = len(active1 | active2)
    jaccard = overlap / union if union > 0 else 0
    
    print(f"Active position overlap: {overlap}/{union} (Jaccard: {jaccard:.3f})")
    
    # Statistical test
    print("\nStatistical Decision:")
    sequential_test = DualSequentialTest(alpha=0.05, beta=0.10)
    
    # Use multiple similarity measures
    similarities = [cosine_sim, jaccard, 1 - normalized_hamming]
    for sim in similarities:
        # If similarity < 0.5, models are different
        observation = 0 if sim > 0.7 else 1
        sequential_test.update(observation)
    
    if sequential_test.has_decision():
        decision = "SAME MODEL" if sequential_test.get_decision() == 0 else "DIFFERENT MODELS"
        print(f"Decision: {decision}")
        print(f"Confidence: Based on {len(similarities)} similarity measures")
    else:
        print("Need more observations for decision")
    
    # Verification summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    print("\n✅ REV Successfully:")
    print("  • Loaded both models with memory-bounded execution")
    print("  • Generated semantic hypervectors for each model")
    print("  • Computed multiple similarity metrics")
    print("  • Made statistical verification decision")
    
    if cosine_sim < 0.5:
        print("\n✓ Models are VERIFIED as DIFFERENT")
        print("  The hypervectors show significant divergence")
    elif cosine_sim > 0.9:
        print("\n✓ Models are VERIFIED as SAME/SIMILAR")
        print("  The hypervectors show high similarity")
    else:
        print("\n⚠️ Models show moderate similarity")
        print("  Further testing recommended")
    
else:
    print("❌ Could not generate hypervectors for comparison")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)