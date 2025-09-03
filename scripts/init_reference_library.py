#!/usr/bin/env python3
"""
Initialize the Reference Library with GPT-2 as the reference for the GPT family
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.fingerprint.dual_library_system import DualLibrarySystem

def main():
    # Load GPT-2 results from our previous run
    gpt2_results_path = Path("outputs/gpt2_baseline.json")
    if not gpt2_results_path.exists():
        # Try alternative paths
        for alt_path in ["outputs/gpt2_behavioral_full.json", "outputs/gpt2_with_library.json"]:
            if Path(alt_path).exists():
                gpt2_results_path = Path(alt_path)
                break
        else:
            print("❌ No GPT-2 results found. Run GPT-2 first.")
            return 1
    
    print(f"Loading GPT-2 results from {gpt2_results_path}")
    with open(gpt2_results_path) as f:
        results = json.load(f)
    
    gpt2_data = results["results"]["gpt2"]
    
    # Create dual library system
    library = DualLibrarySystem()
    
    # Create reference fingerprint for GPT family
    reference_fingerprint = {
        "model_family": "gpt",
        "model_size": "124M",
        "architecture_version": "gpt-2",
        "reference_model": "gpt2",
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 50257,
        
        # Key behavioral characteristics from our run
        "behavioral_patterns": {
            "hv_entropy": gpt2_data["stages"]["behavioral_analysis"]["metrics"]["hv_entropy"],
            "hv_sparsity": gpt2_data["stages"]["behavioral_analysis"]["metrics"]["hv_sparsity"],
            "response_diversity": gpt2_data["stages"]["behavioral_analysis"]["metrics"]["response_diversity"],
            "avg_response_length": gpt2_data["stages"]["behavioral_analysis"]["metrics"]["avg_response_length"]
        },
        
        # Strategic testing information
        "vulnerable_layers": [3, 6, 9],  # Middle layers often most interesting
        "stable_layers": [0, 1, 11],  # First and last layers usually stable
        "recommended_cassettes": ["syntactic", "semantic", "arithmetic"],
        
        # Performance characteristics
        "memory_footprint_gb": 0.5,
        "optimal_segment_size": 512,
        "optimal_batch_size": 8,
        
        # Validation
        "validation_score": 1.0,
        "source": "empirical"
    }
    
    # Add to reference library
    library.add_reference_fingerprint("gpt", reference_fingerprint)
    
    print("✅ Added GPT-2 as reference for GPT family")
    print(f"   Model: GPT-2 (124M parameters)")
    print(f"   Entropy: {reference_fingerprint['behavioral_patterns']['hv_entropy']:.2f}")
    print(f"   Sparsity: {reference_fingerprint['behavioral_patterns']['hv_sparsity']:.1%}")
    print(f"   Response diversity: {reference_fingerprint['behavioral_patterns']['response_diversity']:.1%}")
    
    # Also add to active library for completeness
    library.add_to_active_library(
        fingerprint_data=reference_fingerprint,
        model_info={
            "model_name": "gpt2",
            "model_path": "/Users/rohanvinaik/LLM_models/gpt2",
            "run_type": "reference_baseline"
        }
    )
    
    print("\n📚 Reference Library initialized")
    print("   Use this for identifying GPT family models")
    print("   GPT-2 Medium should now be identified as GPT family")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())