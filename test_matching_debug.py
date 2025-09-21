#!/usr/bin/env python3

"""
Minimal script to debug the 0% matching issue.
This isolates the matching function to see what exception is being thrown.
"""

import sys
import os
import traceback

# Add the project root to path
sys.path.insert(0, '/Users/rohanvinaik/REV')
sys.path.insert(0, '/Users/rohanvinaik/REV/src')

from src.fingerprint.dual_library_system import create_dual_library
from src.hdc.unified_fingerprint import UnifiedFingerprint

def test_matching_debug():
    """Test the matching function with a minimal fingerprint to see what fails."""

    print("🔍 Testing matching function...")

    try:
        # Create a minimal fingerprint like the light probe would
        test_fingerprint = UnifiedFingerprint(
            model_name="llama-3.3-70b-instruct",
            model_family="llama",  # This should match
            layer_count=80,
            layers_sampled=[0, 7, 14, 21, 28, 35, 43, 50, 57, 64, 71, 79],
            restriction_sites=[
                {
                    "layer": 0,
                    "variance": 0.290,
                    "divergence_delta": 0.0,
                    "behavioral_transition": False
                },
                {
                    "layer": 7,
                    "variance": 0.304,
                    "divergence_delta": 4.8,  # Should be significant
                    "behavioral_transition": True
                }
            ],
            prompt_text="Complete this sentence: The we...",
            response_text="weather today is wonderful...",  # Mock response
            divergence_stats={
                "mean_divergence": 0.297,
                "variance_profile": [0.290, 0.287, 0.320, 0.285, 0.325, 0.294, 0.296, 0.312, 0.292, 0.354, 0.287, 0.284],
                "behavioral_profile": [0.290, 0.287, 0.320, 0.285, 0.325, 0.294, 0.296, 0.312, 0.292, 0.354, 0.287, 0.284]
            }
        )

        print(f"✅ Created test fingerprint: {test_fingerprint.model_family} with {len(test_fingerprint.restriction_sites)} sites")

        # Now test the library matching
        print("📚 Creating dual library...")
        library = create_dual_library()
        print(f"✅ Library created successfully")

        print("🔍 Testing identify_from_behavioral_analysis...")
        result = library.identify_from_behavioral_analysis(test_fingerprint)

        print(f"✅ MATCHING RESULT:")
        print(f"   Family: {result.identified_family}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Source: {result.identification_source}")

        if result.confidence == 0.0:
            print("❌ Still getting 0% confidence - something is wrong with the matching logic")
        else:
            print("🎉 Matching is working!")

    except Exception as e:
        print(f"❌ EXCEPTION CAUGHT: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()

        # Let's examine what specifically failed
        print(f"\n🔍 Exception type: {type(e).__name__}")
        print(f"🔍 Exception args: {e.args}")

if __name__ == "__main__":
    test_matching_debug()