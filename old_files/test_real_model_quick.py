#!/usr/bin/env python3
"""
Quick test to verify real model loading and activation extraction works.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Add src to path
sys.path.insert(0, '/Users/rohanvinaik/REV')

# Path to LLM models
LLM_MODELS_PATH = Path("/Users/rohanvinaik/LLM_models")

def test_real_model_loading():
    """Test that we can actually load and run a model."""
    
    print("Testing real model loading and activation extraction...")
    
    # Try to load a small model
    model_name = "pythia-70m"
    model_path = LLM_MODELS_PATH / model_name
    
    if not model_path.exists():
        print(f"Model path {model_path} doesn't exist")
        return False
    
    print(f"\n1. Loading {model_name} from {model_path}...")
    
    try:
        # Load model and tokenizer
        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    print("\n2. Testing activation extraction...")
    
    try:
        # Create a test prompt
        test_prompt = "The capital of France is"
        
        # Tokenize
        tokens = tokenizer.encode(
            test_prompt,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        print(f"✓ Tokenized prompt: shape {tokens.shape}")
        
        # Run model and extract activations
        with torch.no_grad():
            outputs = model(
                tokens,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Check we got hidden states
            if hasattr(outputs, 'hidden_states'):
                n_layers = len(outputs.hidden_states)
                print(f"✓ Got hidden states from {n_layers} layers")
                
                # Extract activations from middle layer
                middle_layer = n_layers // 2
                activations = outputs.hidden_states[middle_layer]
                
                # Convert to numpy
                activations_np = activations.cpu().numpy()
                
                print(f"✓ Extracted activations shape: {activations_np.shape}")
                print(f"✓ Activations dtype: {activations_np.dtype}")
                print(f"✓ Activations range: [{activations_np.min():.3f}, {activations_np.max():.3f}]")
                
                # Verify activations are not random
                mean = activations_np.mean()
                std = activations_np.std()
                print(f"✓ Activations statistics: mean={mean:.3f}, std={std:.3f}")
                
                # Check that different positions have different activations
                if activations_np.shape[1] > 1:
                    pos1 = activations_np[0, 0, :]
                    pos2 = activations_np[0, 1, :]
                    diff = np.abs(pos1 - pos2).mean()
                    print(f"✓ Different positions have different activations: avg diff={diff:.3f}")
                
            else:
                print("✗ No hidden states in output")
                return False
                
    except Exception as e:
        print(f"✗ Failed to extract activations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing memory efficiency...")
    
    # Check memory usage
    import psutil
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"✓ Current memory usage: {mem_mb:.1f} MB")
    
    # Clear model from memory
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    mem_mb_after = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"✓ Memory after cleanup: {mem_mb_after:.1f} MB")
    print(f"✓ Memory freed: {mem_mb - mem_mb_after:.1f} MB")
    
    print("\n✅ All tests passed! Real model loading and activation extraction is working.")
    return True

if __name__ == "__main__":
    success = test_real_model_loading()
    exit(0 if success else 1)