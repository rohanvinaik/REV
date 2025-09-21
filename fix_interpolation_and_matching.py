#!/usr/bin/env python3
"""
Fix for incomplete interpolation and broken matching in REV pipeline.

Issues:
1. Interpolation only does max 3 additional probes (should find ALL transition points)
2. Matching returns 0% even for Llama models against Llama reference
"""

import json

def analyze_reference_library():
    """Check what's in the reference library."""
    with open('fingerprint_library/reference_library.json', 'r') as f:
        data = json.load(f)

    print("=== Reference Library Analysis ===")
    for name, info in data.get('fingerprints', {}).items():
        print(f"\n{name}:")
        print(f"  Family: {info.get('model_family', 'unknown')}")
        print(f"  Challenges: {info.get('challenges_processed', 0)}")

        # Check restriction sites format
        sites = info.get('restriction_sites', [])
        if sites:
            print(f"  Restriction sites: {len(sites)} sites")
            if isinstance(sites[0], dict):
                print(f"    Format: dict with keys: {list(sites[0].keys())}")
                print(f"    First site: {sites[0]}")
            else:
                print(f"    Format: simple list - {sites[:5]}")

        # Check variance profile
        var_profile = info.get('variance_profile', [])
        if var_profile:
            print(f"  Variance profile: {len(var_profile)} points")
            print(f"    First 5: {var_profile[:5]}")

        # Check behavioral patterns
        bp = info.get('behavioral_patterns', {})
        if bp:
            print(f"  Behavioral patterns keys: {list(bp.keys())}")

print("\n" + "="*60)
print("INTERPOLATION FIX - run_rev.py lines 731-733")
print("="*60)
print("""
CURRENT CODE (lines 731-733):
    # Sort by transition strength and probe top transitions only
    transitions.sort(key=lambda x: x[1], reverse=True)
    additional_layers = [layer for layer, _ in transitions[:3]]  # Max 3 additional probes

FIXED CODE:
    # Sort by transition strength
    transitions.sort(key=lambda x: x[1], reverse=True)

    # Probe ALL significant transitions (not just top 3)
    # But limit to max 10 to avoid excessive probing on very large models
    significant_transitions = [t for t in transitions if t[1] > 0.1]  # 10% change threshold
    additional_layers = [layer for layer, _ in significant_transitions[:10]]  # Up to 10 probes

    # If we have very few transitions, lower the threshold
    if len(additional_layers) < 5 and transitions:
        # Add more transitions with lower threshold
        medium_transitions = [t for t in transitions if t[1] > 0.05 and t[0] not in additional_layers]
        additional_layers.extend([layer for layer, _ in medium_transitions[:5-len(additional_layers)]])
""")

print("\n" + "="*60)
print("MATCHING FIX - src/fingerprint/dual_library_system.py")
print("="*60)
print("""
The issue is that the reference library has restriction_sites as simple dictionaries
but the matching expects specific format. The fix is to ensure compatibility.

ADDITIONAL DEBUG needed in identify_from_behavioral_analysis (line ~200):
    print(f"[DEBUG-IDENTIFY] {fp_id[:30]}: similarity={similarity:.3f}, family={ref_data.get('model_family')}")
    print(f"    ref_sites={ref_sites[:3] if ref_sites else 'None'}...")
    print(f"    ref_profile={len(ref_profile) if ref_profile else 0} points")
    print(f"    input_sites={restriction_sites[:3] if restriction_sites else 'None'}...")
    print(f"    input_profile={len(variance_profile) if variance_profile else 0} points")

And in _compute_topology_similarity, ensure we handle different formats properly.
""")

# Analyze the reference library
analyze_reference_library()

print("\n" + "="*60)
print("RECOMMENDED IMMEDIATE FIX")
print("="*60)
print("""
1. Edit run_rev.py line 733:
   Change: additional_layers = [layer for layer, _ in transitions[:3]]
   To:     additional_layers = [layer for layer, _ in transitions[:10] if _ > 0.05]

2. Edit run_rev.py line 719:
   Change: if relative_change > 0.05 and i < len(sample_layers) - 1:
   To:     if relative_change > 0.02 and i < len(sample_layers) - 1:  # Lower threshold

3. Add debug output to see what's being compared in the matching phase
""")