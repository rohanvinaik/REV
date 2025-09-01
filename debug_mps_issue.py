#!/usr/bin/env python3
"""
Debug MPS placeholder storage allocation issue
"""

import torch
import torch.nn as nn
import numpy as np
import gc
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

def test_basic_mps():
    """Test basic MPS functionality"""
    logger.info("="*60)
    logger.info("TEST 1: Basic MPS Operations")
    logger.info("="*60)
    
    if not torch.backends.mps.is_available():
        logger.error("MPS not available!")
        return False
    
    device = torch.device("mps")
    
    # Test 1: Basic tensor creation
    try:
        x = torch.randn(10, 10, device=device, dtype=torch.float32)
        logger.info("✓ Can create float32 tensor on MPS")
    except Exception as e:
        logger.error(f"✗ Failed to create float32 tensor: {e}")
        return False
    
    # Test 2: Float16 support
    try:
        x_half = torch.randn(10, 10, device=device, dtype=torch.float16)
        logger.info("✓ Can create float16 tensor on MPS")
    except Exception as e:
        logger.error(f"✗ Failed to create float16 tensor: {e}")
        return False
    
    # Test 3: Basic operations
    try:
        y = x @ x.T
        logger.info("✓ Can perform matrix multiplication on MPS")
    except Exception as e:
        logger.error(f"✗ Failed matrix multiplication: {e}")
        return False
    
    return True

def test_weight_loading():
    """Test loading weights from disk to MPS"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Weight Loading to MPS")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # Find a weight file
    model_path = Path("/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
    weight_files = list(model_path.glob("*.safetensors"))
    
    if not weight_files:
        logger.error("No weight files found")
        return False
    
    logger.info(f"Found {len(weight_files)} weight files")
    
    # Try loading a weight file
    from safetensors import safe_open
    weight_file = weight_files[0]
    logger.info(f"Testing with: {weight_file.name}")
    
    try:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            # Get first tensor
            tensor_name = list(f.keys())[0]
            tensor_cpu = f.get_tensor(tensor_name)
            logger.info(f"Loaded tensor '{tensor_name}': shape={tensor_cpu.shape}, dtype={tensor_cpu.dtype}")
            
            # Try moving to MPS
            tensor_mps = tensor_cpu.to(device=device, dtype=dtype)
            logger.info(f"✓ Moved to MPS: device={tensor_mps.device}, dtype={tensor_mps.dtype}")
            
            # Try using it
            result = tensor_mps.sum()
            logger.info(f"✓ Can perform operations: sum={result.item():.3f}")
            
    except Exception as e:
        logger.error(f"✗ Weight loading failed: {e}")
        return False
    
    return True

def test_layer_execution():
    """Test executing a transformer layer on MPS"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Layer Execution on MPS")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # Create simple layer components
    hidden_size = 4096
    batch_size = 1
    seq_len = 10
    
    # Test LayerNorm
    try:
        layer_norm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        
        # Forward pass
        y = layer_norm(x)
        logger.info(f"✓ LayerNorm forward pass: input={x.shape}, output={y.shape}")
        
    except Exception as e:
        logger.error(f"✗ LayerNorm failed: {e}")
        return False
    
    # Test Linear layer
    try:
        linear = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        
        # Forward pass
        y = linear(x)
        logger.info(f"✓ Linear forward pass: input={x.shape}, output={y.shape}")
        
    except Exception as e:
        logger.error(f"✗ Linear failed: {e}")
        return False
    
    return True

def test_memory_allocation():
    """Test memory allocation patterns that might cause placeholder issues"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Memory Allocation Patterns")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # Test 1: Creating tensors without data
    try:
        # This might create placeholder storage
        x = torch.empty(100, 100, device=device, dtype=dtype)
        logger.info(f"Created empty tensor: {x.shape}")
        
        # Try to use it
        y = x + 1
        logger.info("✓ Can use empty tensor after creation")
        
    except Exception as e:
        logger.error(f"✗ Empty tensor issue: {e}")
    
    # Test 2: Tensor views and reshaping
    try:
        x = torch.randn(100, 100, device=device, dtype=dtype)
        x_view = x.view(10000)
        y = x_view.sum()
        logger.info("✓ Can use tensor views")
        
    except Exception as e:
        logger.error(f"✗ Tensor view issue: {e}")
    
    # Test 3: In-place operations
    try:
        x = torch.randn(100, 100, device=device, dtype=dtype)
        x += 1  # In-place
        y = x.sum()
        logger.info("✓ In-place operations work")
        
    except Exception as e:
        logger.error(f"✗ In-place operation issue: {e}")
    
    # Test 4: Cloning and detaching
    try:
        x = torch.randn(100, 100, device=device, dtype=dtype)
        x_clone = x.clone().detach()
        y = x_clone.sum()
        logger.info("✓ Clone and detach work")
        
    except Exception as e:
        logger.error(f"✗ Clone/detach issue: {e}")
    
    return True

def test_llama_layer_simulation():
    """Simulate the exact scenario from our layer execution"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Llama Layer Simulation")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # Simulate loading weights
    hidden_size = 8192
    intermediate_size = 28672
    
    try:
        # Create layer components
        class LlamaRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)
        
        # Create components
        input_layernorm = LlamaRMSNorm(hidden_size).to(device=device, dtype=dtype)
        gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device=device, dtype=dtype)
        up_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device=device, dtype=dtype)
        down_proj = nn.Linear(intermediate_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        
        # Test input
        batch_size = 1
        seq_len = 40
        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        
        logger.info(f"Input: shape={hidden_states.shape}, device={hidden_states.device}, dtype={hidden_states.dtype}")
        
        # Forward pass (like in the actual layer)
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        
        # MLP
        gate_out = gate_proj(hidden_states)
        up_out = up_proj(hidden_states)
        hidden_states = down_proj(torch.nn.functional.silu(gate_out) * up_out)
        hidden_states = residual + hidden_states
        
        logger.info(f"✓ Full layer forward pass succeeded!")
        logger.info(f"Output: shape={hidden_states.shape}, device={hidden_states.device}, dtype={hidden_states.dtype}")
        
    except Exception as e:
        logger.error(f"✗ Layer simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_weight_loading_issue():
    """Test the specific weight loading pattern that might cause issues"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Weight Loading Pattern Issue")
    logger.info("="*60)
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # Simulate loading multiple weights and using them
    try:
        weights = {}
        
        # Simulate loading weights (like in load_layer_weights)
        for i in range(5):
            # Create a "loaded" weight
            w = torch.randn(1000, 1000, device="cpu", dtype=torch.float32)
            # Move to MPS
            w_mps = w.to(device=device, dtype=dtype)
            weights[f"weight_{i}"] = w_mps
            
        logger.info(f"✓ Loaded {len(weights)} weights to MPS")
        
        # Try using them
        x = torch.randn(1000, device=device, dtype=dtype)
        for name, w in weights.items():
            y = w @ x
            logger.info(f"✓ Used weight '{name}': output shape={y.shape}")
        
        # Clean up (like in the actual code)
        del weights
        gc.collect()
        logger.info("✓ Cleanup successful")
        
    except Exception as e:
        logger.error(f"✗ Weight loading pattern failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    logger.info("="*60)
    logger.info("MPS DEBUGGING SUITE")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        logger.info(f"MPS device: {torch.device('mps')}")
    logger.info("")
    
    # Run tests
    tests = [
        ("Basic MPS", test_basic_mps),
        ("Weight Loading", test_weight_loading),
        ("Layer Execution", test_layer_execution),
        ("Memory Allocation", test_memory_allocation),
        ("Llama Layer Simulation", test_llama_layer_simulation),
        ("Weight Loading Pattern", test_weight_loading_issue),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    logger.info(f"\nPassed: {passed}/{len(results)} tests")

if __name__ == "__main__":
    main()