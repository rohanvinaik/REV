#!/usr/bin/env python3
"""
Basic Model Verification Example

This script demonstrates the simplest way to verify a model using REV.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from run_rev import REVUnified


def verify_model(model_path: str, challenges: int = 30):
    """
    Basic model verification example.
    
    Args:
        model_path: Path to model directory containing config.json
        challenges: Number of verification challenges (5-500)
    """
    print(f"Verifying model: {model_path}")
    print(f"Using {challenges} challenges\n")
    
    # Initialize REV system
    rev = REVUnified(
        memory_limit_gb=4.0,  # Limit memory usage
        debug=False           # Set True for verbose output
    )
    
    try:
        # Process model
        result = rev.process_model(model_path, challenges=challenges)
        
        # Display results
        print("=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Model Family: {result['model_family']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Architecture: {result.get('architecture', 'Unknown')}")
        
        if 'restriction_sites' in result:
            print(f"\nRestriction Sites Found: {len(result['restriction_sites'])}")
            for site in result['restriction_sites'][:3]:  # Show first 3
                print(f"  Layer {site['layer_idx']}: "
                      f"divergence={site['behavioral_divergence']:.3f}")
        
        if 'metrics' in result:
            print(f"\nPerformance Metrics:")
            print(f"  Processing Time: {result['metrics'].get('time', 0):.1f}s")
            print(f"  Memory Used: {result['metrics'].get('memory_gb', 0):.1f}GB")
        
        # Verification decision
        print("\n" + "=" * 60)
        if result['confidence'] > 0.85:
            print("✅ Model VERIFIED with high confidence")
        elif result['confidence'] > 0.60:
            print("⚠️  Model verified with medium confidence")
        else:
            print("❌ Model verification uncertain (low confidence)")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    finally:
        # Cleanup
        rev.cleanup()
    
    return True


def main():
    """Main function with example usage."""
    
    # Example 1: Verify a local model
    print("Example 1: Basic Verification")
    print("-" * 60)
    
    # Update with your actual model path
    model_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    
    if Path(model_path).exists():
        verify_model(model_path, challenges=10)
    else:
        print(f"Model not found at {model_path}")
        print("Please update the path to point to your model directory")
    
    print("\n" * 2)
    
    # Example 2: Quick diagnostic (5 challenges)
    print("Example 2: Quick Diagnostic")
    print("-" * 60)
    
    if Path(model_path).exists():
        print("Running quick diagnostic with minimal challenges...")
        verify_model(model_path, challenges=5)
    
    print("\n" * 2)
    
    # Example 3: Show how to find HuggingFace cache models
    print("Example 3: Finding HuggingFace Cache Models")
    print("-" * 60)
    
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        print(f"HuggingFace cache found at: {hf_cache}")
        
        # Find all config.json files
        configs = list(hf_cache.glob("*/snapshots/*/config.json"))
        if configs:
            print(f"Found {len(configs)} cached models:")
            for config in configs[:3]:  # Show first 3
                model_dir = config.parent
                print(f"  {model_dir}")
            
            # Try to verify the first one
            if configs:
                print(f"\nVerifying first cached model...")
                verify_model(str(configs[0].parent), challenges=5)
        else:
            print("No models found in HuggingFace cache")
    else:
        print(f"HuggingFace cache not found at {hf_cache}")
    
    # Alternative cache location
    alt_cache = Path.home() / "LLM_models"
    if alt_cache.exists():
        print(f"\nAlternative model directory found at: {alt_cache}")
        models = list(alt_cache.glob("*/config.json"))
        if models:
            print(f"Found {len(models)} models")


if __name__ == "__main__":
    main()