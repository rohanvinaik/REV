#!/usr/bin/env python3
"""
Minimal test to prove models are really being executed, not mocked.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import psutil

# Path to models
LLM_MODELS_PATH = Path("/Users/rohanvinaik/LLM_models")

def verify_real_execution():
    """Verify that we're using real models, not mock data."""
    
    print("=" * 60)
    print("VERIFICATION: Real Model Execution Test")
    print("=" * 60)
    
    # Test 1: Memory footprint
    print("\n1. MEMORY FOOTPRINT TEST")
    print("-" * 40)
    
    mem_before = psutil.Process().memory_info().rss / (1024**2)
    print(f"Memory before loading: {mem_before:.1f} MB")
    
    # Load a real model
    model_path = LLM_MODELS_PATH / "pythia-70m"
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float32)  # Use float32 to avoid NaN
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mem_after_load = psutil.Process().memory_info().rss / (1024**2)
    print(f"Memory after loading: {mem_after_load:.1f} MB")
    print(f"Memory increase: {mem_after_load - mem_before:.1f} MB")
    
    if mem_after_load - mem_before > 50:  # Should increase by at least 50MB
        print("✅ PASS: Significant memory increase indicates real model loaded")
    else:
        print("❌ FAIL: No significant memory increase")
    
    # Test 2: Deterministic outputs
    print("\n2. DETERMINISTIC OUTPUT TEST")
    print("-" * 40)
    
    prompt1 = "The capital of France is"
    prompt2 = "The weather today is"
    
    tokens1 = tokenizer(prompt1, return_tensors='pt')
    tokens2 = tokenizer(prompt2, return_tensors='pt')
    
    with torch.no_grad():
        output1 = model(**tokens1, output_hidden_states=True)
        output2 = model(**tokens2, output_hidden_states=True)
    
    # Extract activations
    act1 = output1.hidden_states[-1].numpy()
    act2 = output2.hidden_states[-1].numpy()
    
    # Compare
    diff = np.abs(act1[:, :4, :] - act2[:, :4, :]).mean()
    print(f"Activation difference between prompts: {diff:.4f}")
    
    if diff > 0.1:
        print("✅ PASS: Different prompts produce different activations")
    else:
        print("❌ FAIL: Activations too similar (might be random)")
    
    # Test 3: Reproducibility
    print("\n3. REPRODUCIBILITY TEST")
    print("-" * 40)
    
    with torch.no_grad():
        output1a = model(**tokens1, output_hidden_states=True)
        output1b = model(**tokens1, output_hidden_states=True)
    
    act1a = output1a.hidden_states[-1].numpy()
    act1b = output1b.hidden_states[-1].numpy()
    
    diff_same = np.abs(act1a - act1b).max()
    print(f"Max difference for same input: {diff_same:.8f}")
    
    if diff_same < 1e-5:
        print("✅ PASS: Same input produces identical outputs (deterministic)")
    else:
        print("❌ FAIL: Outputs not reproducible")
    
    # Test 4: Activation statistics
    print("\n4. ACTIVATION STATISTICS TEST")
    print("-" * 40)
    
    # Get statistics
    mean = act1.mean()
    std = act1.std()
    min_val = act1.min()
    max_val = act1.max()
    
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    
    # Check if values look like real activations (not uniform random)
    if abs(mean) < 1.0 and std > 0.5 and std < 10:
        print("✅ PASS: Statistics match expected neural network activations")
    else:
        print("❌ FAIL: Statistics don't match expected patterns")
    
    # Test 5: Layer progression
    print("\n5. LAYER PROGRESSION TEST")
    print("-" * 40)
    
    layer_norms = []
    for i, layer_act in enumerate(output1.hidden_states):
        norm = np.linalg.norm(layer_act.numpy())
        layer_norms.append(norm)
        print(f"Layer {i} norm: {norm:.2f}")
    
    # Check if norms change across layers
    norm_changes = [abs(layer_norms[i+1] - layer_norms[i]) for i in range(len(layer_norms)-1)]
    avg_change = np.mean(norm_changes)
    
    if avg_change > 0.1:
        print(f"✅ PASS: Layer norms change across depth (avg change: {avg_change:.2f})")
    else:
        print("❌ FAIL: Layer norms too similar")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return True

if __name__ == "__main__":
    verify_real_execution()