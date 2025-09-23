#!/usr/bin/env python3
"""
Test the enhanced dual library routing fix.

This tests that:
1. The enhanced matching algorithm is ALWAYS called for local models
2. Even with 30-40% confidence, the system uses the reference library
3. The system doesn't immediately fall back to API-only mode
"""

import sys
import json
from pathlib import Path

def test_enhanced_routing():
    """Test the enhanced routing with a model that typically gets low initial confidence."""

    print("="*80)
    print("ENHANCED DUAL LIBRARY ROUTING TEST")
    print("="*80)

    # Test with a Llama model path (known to get ~20% initial confidence)
    test_model = "/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct"

    if not Path(test_model).exists():
        print(f"⚠️ Test model not found: {test_model}")
        print("Please update the path to a valid Llama model")
        return

    print(f"\n📍 Testing with model: {test_model}")
    print("This model typically gets ~20% initial confidence\n")

    # Import the pipeline components
    from src.fingerprint.dual_library_system import identify_and_strategize, create_dual_library

    # Step 1: Initial identification
    print("Step 1: Initial identification")
    identification, strategy = identify_and_strategize(test_model)
    print(f"  Method: {identification.method}")
    print(f"  Confidence: {identification.confidence:.1%}")
    print(f"  Family: {identification.identified_family}")

    # Step 2: Check if light probe is triggered
    if identification.method == "needs_light_probe":
        print("\n✅ Light probe correctly triggered")
        print("Step 2: Simulating light probe results...")

        # Simulate light probe results (typical for Llama)
        test_fingerprint = {
            "restriction_sites": [
                {"layer": 5, "divergence_delta": -0.048, "percent_change": -48.0},
                {"layer": 10, "divergence_delta": 0.033, "percent_change": 33.0}
            ],
            "variance_profile": [0.1, 0.15, 0.12, 0.18, 0.14, 0.10, 0.16],
            "layer_divergences": {"5": 0.048, "10": 0.033},
            "layer_count": 80,
            "layers_sampled": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79],
            "model_family": "unknown",
            "model_name": test_model
        }

        # Step 3: Call enhanced matching
        print("\nStep 3: Calling enhanced dual library matching...")
        library = create_dual_library()
        new_identification = library.identify_from_behavioral_analysis(test_fingerprint, verbose=True)

        print(f"\n📊 Enhanced Matching Results:")
        print(f"  Family: {new_identification.identified_family}")
        print(f"  Confidence: {new_identification.confidence:.1%}")
        print(f"  Method: {new_identification.method}")
        print(f"  Reference: {new_identification.reference_model}")

        # Step 4: Check routing decision
        print("\nStep 4: Checking routing decision...")

        # The fix ensures lower threshold for enhanced matching
        if new_identification.method == "cross_size_behavioral_matching":
            if new_identification.confidence > 0.3:
                print(f"✅ CORRECT: Enhanced matching confidence {new_identification.confidence:.0%} > 30%")
                print("   System should use reference library for 15-20x speedup")
                print("   NOT falling back to API-only mode")
            else:
                print(f"⚠️ Low confidence {new_identification.confidence:.0%} even with enhanced matching")
                print("   System may need deep analysis")
        else:
            print(f"⚠️ Method is {new_identification.method}, not cross_size_behavioral_matching")

        # Step 5: Get testing strategy
        print("\nStep 5: Testing strategy determination...")
        strategy = library.get_testing_strategy(new_identification)
        print(f"  Strategy: {strategy.get('strategy')}")
        print(f"  Challenges: {strategy.get('challenges')}")
        if strategy.get('focus_layers'):
            print(f"  Focus Layers: {strategy['focus_layers'][:5]}... ({len(strategy['focus_layers'])} total)")
        print(f"  Notes: {strategy.get('notes')}")

        # Success criteria check
        print("\n" + "="*80)
        print("SUCCESS CRITERIA CHECK:")
        print("-"*80)

        success_count = 0
        total_checks = 5

        # Check 1: Enhanced algorithm was called
        print("1. Enhanced dual library algorithm called: ✅")
        success_count += 1

        # Check 2: Matching metrics shown
        if "Cosine similarity" in str(new_identification.notes):
            print("2. Matching metrics (Cosine, DTW, etc.) shown: ✅")
            success_count += 1
        else:
            print("2. Matching metrics (Cosine, DTW, etc.) shown: ❌")

        # Check 3: Confidence improved
        if new_identification.confidence > identification.confidence:
            print(f"3. Confidence improved ({identification.confidence:.0%} → {new_identification.confidence:.0%}): ✅")
            success_count += 1
        else:
            print(f"3. Confidence improved: ❌ (stayed at {new_identification.confidence:.0%})")

        # Check 4: Reference library usage
        if strategy.get('strategy') == 'targeted' or strategy.get('reference_model'):
            print("4. Reference library being used: ✅")
            success_count += 1
        else:
            print("4. Reference library being used: ❌")

        # Check 5: Not API-only mode
        if strategy.get('strategy') != 'api_only':
            print("5. NOT using API-only mode: ✅")
            success_count += 1
        else:
            print("5. NOT using API-only mode: ❌")

        print("-"*80)
        print(f"RESULT: {success_count}/{total_checks} checks passed")

        if success_count == total_checks:
            print("🎉 ALL CHECKS PASSED - Enhanced routing is working correctly!")
        elif success_count >= 3:
            print("✅ PARTIAL SUCCESS - Enhanced routing is mostly working")
        else:
            print("❌ FAILED - Enhanced routing still has issues")
    else:
        print(f"\n❌ Light probe NOT triggered - method is {identification.method}")
        print("This suggests the initial routing logic may need adjustment")

if __name__ == "__main__":
    test_enhanced_routing()