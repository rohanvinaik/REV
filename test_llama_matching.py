#!/usr/bin/env python3
"""Test the updated cosine similarity matching with existing Llama 70B data."""

import json
import sys
sys.path.append('/Users/rohanvinaik/REV')

from src.fingerprint.dual_library_system import DualLibrarySystem

# Llama 70B light probe data from the previous test
llama_70b_variance_profile = [
    0.53125, 0.46875, 0.28125, 0.125, 0.0625, 0.0625, 0.03125, 0.0625,
    0.0625, 0.09375, 0.0625, 0.09375, 0.03125, 0.0625, 0.0625, 0.09375,
    0.0625, 0.125, 0.03125, 0.125, 0.0625, 0.15625, 0.09375, 0.1875,
    0.09375, 0.1875, 0.125, 0.21875, 0.125, 0.21875, 0.125, 0.28125,
    0.15625, 0.3125, 0.15625, 0.3125, 0.21875, 0.375, 0.21875, 0.375,
    0.25, 0.40625, 0.25, 0.4375, 0.28125, 0.46875, 0.28125, 0.46875,
    0.3125, 0.5, 0.34375, 0.53125, 0.34375, 0.5625, 0.375, 0.5625,
    0.40625, 0.59375, 0.40625, 0.625, 0.4375, 0.65625, 0.4375, 0.6875,
    0.46875, 0.6875, 0.5, 0.71875, 0.53125, 0.75, 0.53125, 0.78125,
    0.5625, 0.8125, 0.59375, 0.84375, 0.625, 0.84375, 0.65625, 0.875
]

def test_matching():
    print("Testing updated cosine similarity matching with Llama 70B data...")
    print(f"Llama 70B has {len(llama_70b_variance_profile)} layers")
    
    # Initialize the dual library system
    library = DualLibrarySystem()
    
    # Load reference library to verify structure
    try:
        with open('/Users/rohanvinaik/REV/fingerprint_library/reference_library.json', 'r') as f:
            ref_lib = json.load(f)
            
        # Check if llama reference exists
        llama_refs = [k for k in ref_lib.get('fingerprints', {}).keys() if 'llama' in k.lower()]
        print(f"\nFound {len(llama_refs)} Llama references in library:")
        for ref_name in llama_refs:
            ref_data = ref_lib['fingerprints'][ref_name]
            if 'behavioral_topology' in ref_data:
                num_layers = ref_data['behavioral_topology'].get('total_layers', 0)
                print(f"  - {ref_name}: {num_layers} layers")
    except Exception as e:
        print(f"Error loading reference library: {e}")
    
    # Create test behavioral data
    behavioral_data = {
        'variance_profile': llama_70b_variance_profile,
        'model_name': 'llama-3.3-70b-instruct',
        'total_layers': 80,
        'layer_profiles': {
            str(i): {
                'mean': llama_70b_variance_profile[i],
                'std': 0.1,
                'samples': 5
            } for i in range(len(llama_70b_variance_profile))
        }
    }
    
    # Test identification
    print("\n" + "="*60)
    print("Testing identification with updated cosine similarity...")
    print("="*60)
    
    result = library.identify_from_behavioral_analysis(behavioral_data)
    
    print(f"\nIdentification Result:")
    print(f"  Family: {result.identified_family}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Method: {result.method}")
    print(f"  Reference Model: {result.reference_model}")
    
    if hasattr(result, 'notes') and result.notes:
        print(f"\nNotes: {result.notes}")
    
    # Show similarity scores for all references
    if hasattr(library, '_compute_topology_similarity'):
        print("\n" + "="*60)
        print("Similarity scores for all references:")
        print("="*60)
        
        for ref_name in llama_refs:
            ref_data = ref_lib['fingerprints'][ref_name]
            
            # Extract variance profile from reference
            ref_profile = ref_data.get('variance_profile')
            if not ref_profile and 'behavioral_topology' in ref_data:
                topo = ref_data['behavioral_topology']
                layer_profiles = topo.get('layer_profiles', {})
                if layer_profiles:
                    ref_profile = []
                    for i in range(topo.get('total_layers', 0)):
                        if str(i) in layer_profiles:
                            ref_profile.append(layer_profiles[str(i)].get('mean', 0.5))
            
            if ref_profile:
                similarity = library._compute_topology_similarity(
                    llama_70b_variance_profile,
                    ref_profile
                )
                print(f"  {ref_name}: {similarity:.1%} (cosine similarity)")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    # Success criteria
    if result.confidence > 0.7:
        print("\n✅ SUCCESS: Llama 70B correctly identified with high confidence!")
    elif result.confidence > 0.5:
        print("\n⚠️  PARTIAL SUCCESS: Llama family identified but confidence could be higher")
    else:
        print("\n❌ FAILED: Llama 70B not properly identified")
    
    return result

if __name__ == "__main__":
    test_matching()