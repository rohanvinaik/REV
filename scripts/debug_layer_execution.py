#!/usr/bin/env python3
"""
Debug script to test layer execution and divergence calculation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

def test_layer_execution():
    """Test layer execution with proper dtype handling"""
    
    # Simulate the scenario
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    
    logger.info(f"Testing on device: {device}, dtype: {dtype}")
    
    # Create sample inputs (like layer 0 outputs)
    batch_size = 1
    seq_len = 40
    hidden_size = 8192
    
    # Layer 0 activations
    layer_0_acts = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    logger.info(f"Layer 0 activations: shape={layer_0_acts.shape}, dtype={layer_0_acts.dtype}, device={layer_0_acts.device}")
    
    # Simulate layer 1 execution failure and fallback
    try:
        # This simulates what happens when layer execution fails
        # We try to do some operation that might fail with dtype mismatch
        
        # Create a simple linear layer (simulating part of transformer layer)
        linear = nn.Linear(hidden_size, hidden_size, bias=False)
        linear = linear.to(device=device, dtype=dtype)
        
        # Try forward pass
        hidden_states = layer_0_acts.clone()
        
        # Simulate the error: mixing float16 and float32
        if device.type == "mps":
            logger.info("Simulating MPS failure - falling back to CPU")
            # Move to CPU but with wrong dtype handling
            hidden_states_cpu = hidden_states.cpu()  # Still float16
            linear_cpu = linear.cpu()  # Module on CPU
            
            # This might cause dtype issues
            try:
                output = linear_cpu(hidden_states_cpu.unsqueeze(0))
                logger.info(f"CPU execution succeeded: output dtype={output.dtype}")
            except Exception as e:
                logger.error(f"CPU execution failed: {e}")
                
                # Fix: ensure proper dtype
                linear_cpu = linear_cpu.float()  # Convert to float32
                hidden_states_cpu = hidden_states_cpu.float()
                output = linear_cpu(hidden_states_cpu.unsqueeze(0))
                logger.info(f"CPU execution with float32 succeeded: output dtype={output.dtype}")
                
                # Convert back to original dtype and device
                output = output.to(dtype=dtype, device=device).squeeze(0)
        else:
            output = linear(hidden_states.unsqueeze(0)).squeeze(0)
        
        layer_1_acts = output
        
    except Exception as e:
        logger.error(f"Layer execution failed: {e}")
        # Fallback: use unchanged activations
        layer_1_acts = layer_0_acts.clone()
    
    # Test divergence calculation
    logger.info("\n=== Testing Divergence Calculation ===")
    
    # Ensure both tensors are on the same device
    logger.info(f"Layer 0 device: {layer_0_acts.device}, Layer 1 device: {layer_1_acts.device}")
    
    # If layer_1_acts == layer_0_acts (due to fallback), divergence will be 0
    cos_sim = torch.nn.functional.cosine_similarity(
        layer_0_acts.flatten().unsqueeze(0),
        layer_1_acts.flatten().unsqueeze(0)
    ).item()
    
    divergence = 1.0 - cos_sim
    logger.info(f"Cosine divergence: {divergence:.6f}")
    
    # Check if they're identical
    if torch.allclose(layer_0_acts, layer_1_acts):
        logger.warning("⚠️ Layer 1 activations are identical to Layer 0 (no transformation applied)")
    else:
        logger.info("✓ Layer 1 activations differ from Layer 0")
        
        # Calculate actual divergence metrics
        l2_dist = torch.norm(layer_1_acts - layer_0_acts).item()
        logger.info(f"L2 distance: {l2_dist:.3f}")
        
        mean_diff = abs(layer_1_acts.mean().item() - layer_0_acts.mean().item())
        logger.info(f"Mean difference: {mean_diff:.6f}")

def test_proper_layer_transform():
    """Test with a proper layer transformation"""
    logger.info("\n=== Testing Proper Layer Transform ===")
    
    device = torch.device("cpu")  # Use CPU to avoid MPS issues
    dtype = torch.float32
    
    seq_len = 40
    hidden_size = 512  # Smaller for testing
    
    # Create layer 0 activations
    layer_0_acts = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    
    # Apply a simple transformation (like a layer would)
    transform = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size)
    ).to(device=device, dtype=dtype)
    
    with torch.no_grad():
        layer_1_acts = transform(layer_0_acts) + layer_0_acts  # Residual connection
    
    # Calculate divergence
    cos_sim = torch.nn.functional.cosine_similarity(
        layer_0_acts.flatten().unsqueeze(0),
        layer_1_acts.flatten().unsqueeze(0)
    ).item()
    
    divergence = 1.0 - cos_sim
    logger.info(f"Divergence after proper transform: {divergence:.6f}")
    
    # This should show non-zero divergence
    assert divergence > 0.01, "Divergence should be non-zero after transformation"
    logger.info("✓ Proper transformation produces non-zero divergence")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("DEBUGGING LAYER EXECUTION AND DIVERGENCE")
    logger.info("="*60)
    
    test_layer_execution()
    test_proper_layer_transform()
    
    logger.info("\n" + "="*60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("="*60)