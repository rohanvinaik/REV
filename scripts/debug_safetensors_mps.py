#!/usr/bin/env python3
"""
Debug safetensors loading to MPS issue
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

def test_safetensors_loading():
    """Test loading weights from safetensors directly to MPS"""
    logger.info("="*60)
    logger.info("TEST: Safetensors to MPS Loading")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    model_path = Path("/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
    weight_file = model_path / "model-00001-of-00030.safetensors"
    
    if not weight_file.exists():
        logger.error(f"Weight file not found: {weight_file}")
        return False
    
    logger.info(f"Loading from: {weight_file.name}")
    
    # Test 1: Load to CPU first (should work)
    try:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            tensor_name = "model.layers.1.mlp.gate_proj.weight"
            if tensor_name in f.keys():
                # Load to CPU
                tensor_cpu = f.get_tensor(tensor_name)
                logger.info(f"✓ Loaded to CPU: shape={tensor_cpu.shape}, dtype={tensor_cpu.dtype}")
                
                # Then move to MPS
                tensor_mps = tensor_cpu.to(device=device, dtype=dtype)
                logger.info(f"✓ Moved to MPS: device={tensor_mps.device}, dtype={tensor_mps.dtype}")
                
                # Use it
                x = torch.randn(100, tensor_mps.shape[1], device=device, dtype=dtype)
                y = x @ tensor_mps.T
                logger.info(f"✓ Used tensor: output shape={y.shape}")
    except Exception as e:
        logger.error(f"✗ CPU->MPS failed: {e}")
        return False
    
    # Test 2: Try loading directly to MPS (might cause issues)
    try:
        logger.info("\nTrying direct load to MPS...")
        with safe_open(weight_file, framework="pt", device="mps") as f:
            tensor_name = "model.layers.1.mlp.up_proj.weight"
            if tensor_name in f.keys():
                # Load directly to MPS
                tensor_mps = f.get_tensor(tensor_name)
                logger.info(f"Loaded directly to MPS: shape={tensor_mps.shape}, device={tensor_mps.device}")
                
                # Try to use it
                x = torch.randn(100, tensor_mps.shape[1], device=device, dtype=dtype)
                y = x @ tensor_mps.T
                logger.info(f"✓ Direct MPS load worked!")
    except Exception as e:
        logger.error(f"✗ Direct MPS load failed: {e}")
        # This might be the issue!
    
    # Test 3: Load to CPU and convert dtype before moving to MPS
    try:
        logger.info("\nTrying CPU->dtype->MPS...")
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            tensor_name = "model.layers.1.mlp.down_proj.weight"
            if tensor_name in f.keys():
                # Load to CPU
                tensor_cpu = f.get_tensor(tensor_name)
                
                # Convert dtype on CPU first
                tensor_cpu = tensor_cpu.to(dtype)
                logger.info(f"Converted dtype on CPU: dtype={tensor_cpu.dtype}")
                
                # Then move to MPS
                tensor_mps = tensor_cpu.to(device)
                logger.info(f"✓ Moved to MPS: device={tensor_mps.device}, dtype={tensor_mps.dtype}")
                
                # Use it
                x = torch.randn(tensor_mps.shape[0], 100, device=device, dtype=dtype)
                y = tensor_mps @ x
                logger.info(f"✓ Used tensor: output shape={y.shape}")
    except Exception as e:
        logger.error(f"✗ CPU->dtype->MPS failed: {e}")
        return False
    
    return True

def test_layer_with_loaded_weights():
    """Test using loaded weights in a layer"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Layer with Loaded Weights")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    model_path = Path("/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
    weight_file = model_path / "model-00001-of-00030.safetensors"
    
    # Load weights for layer 1
    weights = {}
    layer_prefix = "model.layers.1"
    
    with safe_open(weight_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(layer_prefix):
                tensor = f.get_tensor(key)
                # Move to MPS with dtype conversion
                tensor = tensor.to(device=device, dtype=dtype)
                weights[key] = tensor
                logger.info(f"Loaded {key}: shape={tensor.shape}")
    
    logger.info(f"\nLoaded {len(weights)} weights")
    
    # Create a simple layer and assign weights
    try:
        hidden_size = 8192
        intermediate_size = 28672
        
        # Create linear layers
        gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Move to device
        gate_proj = gate_proj.to(device=device, dtype=dtype)
        up_proj = up_proj.to(device=device, dtype=dtype)
        down_proj = down_proj.to(device=device, dtype=dtype)
        
        # Assign loaded weights
        if f"{layer_prefix}.mlp.gate_proj.weight" in weights:
            gate_proj.weight.data = weights[f"{layer_prefix}.mlp.gate_proj.weight"]
        if f"{layer_prefix}.mlp.up_proj.weight" in weights:
            up_proj.weight.data = weights[f"{layer_prefix}.mlp.up_proj.weight"]
        if f"{layer_prefix}.mlp.down_proj.weight" in weights:
            down_proj.weight.data = weights[f"{layer_prefix}.mlp.down_proj.weight"]
        
        logger.info("✓ Assigned weights to layers")
        
        # Test forward pass
        x = torch.randn(1, 10, hidden_size, device=device, dtype=dtype)
        gate_out = gate_proj(x)
        up_out = up_proj(x)
        mlp_out = down_proj(torch.nn.functional.silu(gate_out) * up_out)
        
        logger.info(f"✓ Forward pass succeeded: output shape={mlp_out.shape}")
        
    except Exception as e:
        logger.error(f"✗ Layer execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    logger.info("="*60)
    logger.info("SAFETENSORS MPS DEBUGGING")
    logger.info("="*60)
    
    test_safetensors_loading()
    test_layer_with_loaded_weights()

if __name__ == "__main__":
    main()