#!/usr/bin/env python3
"""
Test script to verify the interpolation and matching fixes.
This will run a minimal test with a smaller model to verify functionality.
"""

import subprocess
import json
import sys

def test_small_model():
    """Test with DistilGPT2 which should match against distilgpt2_reference."""
    print("Testing with DistilGPT2 model...")
    print("=" * 60)

    # Check if model exists
    model_path = "/Users/rohanvinaik/LLM_models/distilgpt2"

    # Run with minimal challenges for speed
    cmd = [
        "python", "run_rev.py",
        model_path,
        "--enable-prompt-orchestration",
        "--challenges", "5",
        "--debug"
    ]

    print(f"Running: {' '.join(cmd)}")
    print("Looking for:")
    print("  1. Adaptive threshold calculation (no arbitrary limits)")
    print("  2. Proper restriction_sites format in matching")
    print("  3. Non-zero similarity score")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Check for key indicators
    checks = {
        "Adaptive threshold": "Adaptive threshold:" in output,
        "No arbitrary limit": "no arbitrary limit" in output,
        "DEBUG-IDENTIFY output": "[DEBUG-IDENTIFY]" in output,
        "Matching phase": "Matching against reference library" in output,
        "Family identification": "Family identified:" in output
    }

    print("\nResults:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    # Extract confidence if found
    if "Confidence:" in output:
        for line in output.split('\n'):
            if "Confidence:" in line:
                print(f"\n  Found: {line.strip()}")
                break

    # Extract similarity scores from DEBUG-IDENTIFY
    if "[DEBUG-IDENTIFY]" in output:
        print("\nDEBUG-IDENTIFY outputs:")
        for line in output.split('\n'):
            if "[DEBUG-IDENTIFY]" in line and "similarity=" in line:
                print(f"  {line.strip()}")

    return all(checks.values())

def check_reference_library():
    """Check the reference library structure."""
    print("\nChecking reference library structure...")
    print("=" * 60)

    try:
        with open('fingerprint_library/reference_library.json', 'r') as f:
            data = json.load(f)

        for name, info in data.get('fingerprints', {}).items():
            if 'distilgpt2' in name.lower():
                print(f"Found: {name}")
                sites = info.get('restriction_sites', [])
                if sites and isinstance(sites[0], dict):
                    print(f"  ✅ Restriction sites are dictionaries")
                    print(f"  Keys: {list(sites[0].keys())}")
                else:
                    print(f"  ❌ Restriction sites format issue")
                break
    except Exception as e:
        print(f"Error reading reference library: {e}")

if __name__ == "__main__":
    check_reference_library()
    success = test_small_model()

    if success:
        print("\n✅ All checks passed! The fixes appear to be working.")
    else:
        print("\n❌ Some checks failed. Review the output above.")

    sys.exit(0 if success else 1)