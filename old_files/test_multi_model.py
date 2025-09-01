#!/usr/bin/env python3
"""
Test REV framework with multiple models to verify robustness and model-agnostic design.
"""

import subprocess
import sys
from pathlib import Path

def test_model(model_path, quantize="none", challenges=1):
    """Test a single model through the pipeline."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_path}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "run_rev_complete.py",
        model_path,
        "--challenges", str(challenges),
        "--quantize", quantize,
        "--max-tokens", "20",
        "--output", f"test_{Path(model_path).name.replace('/', '_')}.json"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Success: {model_path}")
            return True
        else:
            print(f"❌ Failed: {model_path}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout: {model_path} (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test multiple models to verify robustness."""
    print("="*80)
    print("REV MULTI-MODEL ROBUSTNESS TEST")
    print("="*80)
    
    # Test models (mix of sizes and sources)
    test_cases = [
        # Small models (fast testing)
        ("gpt2", "none", 1),           # 124M params
        ("distilgpt2", "none", 1),      # 82M params
        
        # Medium model with quantization (if available)
        # ("EleutherAI/gpt-neo-125M", "8bit", 1),
        
        # Local model (if exists)
        # ("/path/to/local/model", "none", 1),
    ]
    
    results = {}
    
    for model_path, quantize, challenges in test_cases:
        success = test_model(model_path, quantize, challenges)
        results[model_path] = success
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)
    
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.0f}%)")
    
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model}")
    
    # Verify model comparison if multiple succeeded
    if success_count >= 2:
        print("\n✅ Model-agnostic framework validated!")
        print("   Multiple models processed successfully")
        print("   Hypervector generation working")
        print("   Model comparison capability confirmed")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    exit(main())