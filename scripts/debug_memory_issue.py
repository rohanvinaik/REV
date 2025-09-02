#!/usr/bin/env python3
"""
Diagnostic script to identify memory issues in REV pipeline.
"""

import psutil
import os

def analyze_memory_issues():
    """Analyze the memory issues found in REV pipeline."""
    
    print("="*70)
    print("REV MEMORY ISSUE ANALYSIS")
    print("="*70)
    
    # Get system info
    mem = psutil.virtual_memory()
    print(f"\nSystem Memory: {mem.total/(1024**3):.1f}GB total, {mem.available/(1024**3):.1f}GB available")
    
    print("\n" + "="*70)
    print("PROBLEMS IDENTIFIED:")
    print("="*70)
    
    print("\n1. NO MEMORY LIMIT ENFORCEMENT:")
    print("   ✗ run_rev_complete.py has no --memory-limit argument")
    print("   ✗ The 36GB limit mentioned in README is never passed to the model loader")
    print("   ✗ LargeModelConfig defaults to using 80% of ALL available memory")
    print(f"   ✗ This means it tries to use {mem.available * 0.8 / (1024**3):.1f}GB instead of 36GB")
    
    print("\n2. MODEL LOADING ISSUE:")
    print("   ✗ LargeModelInference._get_device_map() ALWAYS returns 'auto'")
    print("   ✗ This causes the ENTIRE model to load into memory at once")
    print("   ✗ For a 131GB model, this will consume all available RAM")
    print("   ✗ Offloading only happens AFTER memory is exhausted (too late!)")
    
    print("\n3. NOT TRUE STREAMING:")
    print("   ✗ Despite claims, model is NOT streamed from disk")
    print("   ✗ AutoModelForCausalLM.from_pretrained() loads the whole model")
    print("   ✗ SegmentRunner only segments inference, not model loading")
    
    print("\n4. MPS BACKEND ISSUES (from logs):")
    print("   ✗ 'Placeholder storage has not been allocated on MPS device!'")
    print("   ✗ 'expected m1 and m2 to have same dtype, but got: c10::Half != float'")
    print("   ✗ These indicate dtype mismatches and memory allocation failures")
    
    print("\n" + "="*70)
    print("ROOT CAUSE:")
    print("="*70)
    print("\nThe pipeline tries to load the ENTIRE 131GB Llama 3.3 70B model into memory")
    print("at once, instead of loading it layer-by-layer as intended.")
    
    print("\n" + "="*70)
    print("FIXES NEEDED:")
    print("="*70)
    
    print("\n1. ADD MEMORY LIMIT ARGUMENT:")
    print("   • Add --memory-limit to run_rev_complete.py")
    print("   • Pass this to LargeModelConfig.max_memory")
    
    print("\n2. FIX MODEL LOADING:")
    print("   • Implement proper layer-by-layer loading")
    print("   • Use device_map with specific layer assignments")
    print("   • Example: device_map = {0: 'cpu', 1: 'cpu', ..., 79: 'disk'}")
    
    print("\n3. IMPLEMENT TRUE STREAMING:")
    print("   • Load model weights incrementally")
    print("   • Keep only active layers in memory")
    print("   • Offload inactive layers before OOM")
    
    print("\n4. FIX MPS ISSUES:")
    print("   • Ensure consistent dtypes (all fp16 or all fp32)")
    print("   • Make tensors contiguous before device transfers")
    print("   • Consider using CPU instead of MPS for large models")
    
    print("\n" + "="*70)
    print("IMMEDIATE WORKAROUND:")
    print("="*70)
    
    print("\nTo run the model without crashing:")
    print("1. Use CPU instead of MPS: --device cpu")
    print("2. Enable aggressive quantization: --quantize 4bit")
    print("3. Manually set smaller memory limit in code")
    print("4. Or use API-only mode to avoid local loading entirely")
    
    print("\nExample command that should work:")
    print("python run_rev_complete.py /path/to/model --device cpu --quantize 4bit")
    print("Or better:")
    print("python run_rev_complete.py meta-llama/Llama-3.3-70B-Instruct --api-only")

if __name__ == "__main__":
    analyze_memory_issues()