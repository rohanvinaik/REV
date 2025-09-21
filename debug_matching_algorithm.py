#!/usr/bin/env python3

"""
Debug script to test the matching algorithm directly.
This will help identify why matching returns 0% confidence.
"""

import sys
import os
import json
import traceback

# Add the project root to path
sys.path.insert(0, '/Users/rohanvinaik/REV')
sys.path.insert(0, '/Users/rohanvinaik/REV/src')

from src.fingerprint.dual_library_system import create_dual_library
from src.hdc.unified_fingerprint import UnifiedFingerprint

def debug_matching():
    """Test the matching algorithm with actual data from the light probe."""

    print("🔍 DEBUG: Testing matching algorithm...")
    print("-" * 60)

    # First, load and check the reference library
    print("\n📚 Checking reference library...")
    try:
        with open('fingerprint_library/reference_library.json', 'r') as f:
            ref_data = json.load(f)

        fingerprints = ref_data.get('fingerprints', {})
        print(f"Found {len(fingerprints)} reference fingerprints:")

        for name, fp_data in fingerprints.items():
            family = fp_data.get('model_family', 'unknown')
            sites = fp_data.get('restriction_sites', [])
            print(f"  - {name}: family={family}, sites={len(sites)}")

            # Check for llama family
            if 'llama' in family.lower():
                print(f"    ✅ Found LLAMA reference!")
                print(f"    Sites: {sites[:3]}...")  # Show first 3 sites

    except Exception as e:
        print(f"❌ Error loading reference library: {e}")
        return

    print("\n" + "-" * 60)
    print("\n🧪 Creating test fingerprint (mimics light probe output)...")

    # Create test data that mimics what the light probe produces
    test_data = {
        'model_name': 'llama-3.3-70b-instruct',
        'model_family': 'llama',  # This should match!
        'layer_count': 80,
        'layers_sampled': [0, 7, 14, 21, 28, 35, 43, 50, 57, 64, 71, 79],
        'restriction_sites': [
            {
                'layer': 43,
                'divergence_delta': 0.024,  # ~21.4% change
                'percent_change': 21.43,
                'before': 0.330,
                'after': 0.354
            }
        ],
        'variance_profile': [0.310, 0.304, 0.305, 0.302, 0.311, 0.304, 0.324, 0.309, 0.326, 0.305, 0.318, 0.287],
        'behavioral_profile': [0.310, 0.304, 0.305, 0.302, 0.311, 0.304, 0.324, 0.309, 0.326, 0.305, 0.318, 0.287],
        'divergence_stats': {
            'mean_divergence': 0.309,
            'variance_profile': [0.310, 0.304, 0.305, 0.302, 0.311, 0.304, 0.324, 0.309, 0.326, 0.305, 0.318, 0.287],
            'behavioral_profile': [0.310, 0.304, 0.305, 0.302, 0.311, 0.304, 0.324, 0.309, 0.326, 0.305, 0.318, 0.287]
        },
        'prompt_text': 'Complete this sentence: The we...',
        'response_text': 'ather today is beautiful...'  # Mock response
    }

    print(f"Test fingerprint created:")
    print(f"  - Family: {test_data['model_family']}")
    print(f"  - Layers: {test_data['layer_count']}")
    print(f"  - Sites: {len(test_data['restriction_sites'])}")
    print(f"  - Variance profile: {test_data['variance_profile'][:3]}...")

    print("\n" + "-" * 60)
    print("\n🔬 Testing dual library matching...")

    try:
        # Create the dual library system
        library = create_dual_library()
        print("✅ Dual library created successfully")

        # Add debug output to the library's identify method
        original_identify = library.identify_from_behavioral_analysis

        def debug_identify(fingerprint_data):
            print("\n[DEBUG] Inside identify_from_behavioral_analysis")
            print(f"[DEBUG] Input type: {type(fingerprint_data)}")

            if isinstance(fingerprint_data, dict):
                print(f"[DEBUG] Keys in fingerprint_data: {list(fingerprint_data.keys())}")
                print(f"[DEBUG] model_family: {fingerprint_data.get('model_family', 'NOT FOUND')}")
                print(f"[DEBUG] restriction_sites: {fingerprint_data.get('restriction_sites', 'NOT FOUND')[:1] if fingerprint_data.get('restriction_sites') else 'EMPTY'}")

            # Call original method
            result = original_identify(fingerprint_data)

            print(f"\n[DEBUG] Result from identify:")
            print(f"  - Family: {result.identified_family}")
            print(f"  - Confidence: {result.confidence:.2%}")
            print(f"  - Source: {result.identification_source}")

            return result

        # Replace with debug version
        library.identify_from_behavioral_analysis = debug_identify

        # Test the matching
        print("\n🎯 Calling identify_from_behavioral_analysis...")
        result = library.identify_from_behavioral_analysis(test_data)

        print("\n" + "=" * 60)
        print("📊 FINAL RESULT:")
        print("=" * 60)
        print(f"Identified Family: {result.identified_family}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Source: {result.identification_source}")

        if result.confidence == 0.0:
            print("\n❌ STILL GETTING 0% CONFIDENCE!")
            print("The matching algorithm is failing to find similarities.")

            # Let's manually check what the library has
            print("\n🔍 Checking library contents...")
            if hasattr(library, 'reference_library'):
                ref_lib = library.reference_library
                print(f"Reference library has {len(ref_lib.fingerprints)} entries")

                # Check if there's a llama reference
                for name, fp in ref_lib.fingerprints.items():
                    if 'llama' in name.lower() or (hasattr(fp, 'model_family') and 'llama' in str(fp.model_family).lower()):
                        print(f"  Found potential llama match: {name}")

        else:
            print("\n✅ SUCCESS! Got non-zero confidence!")

    except Exception as e:
        print(f"\n❌ Exception during matching: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_matching()