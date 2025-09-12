#!/usr/bin/env python3
"""
Fix model_family field in reference library to use actual family names.
"""

import json

# Load reference library
with open('fingerprint_library/reference_library.json', 'r') as f:
    data = json.load(f)

print("Fixing reference library family names...")
print("=" * 60)

# Fix family names
for name, info in data['fingerprints'].items():
    old_family = info.get('model_family', 'unknown')
    
    # Determine correct family based on model name
    if 'pythia' in name.lower() or 'a39f36b100fe8a5377810d56c3f4789b9c53ac42' in name:
        new_family = 'pythia'
    elif 'distilgpt2' in name.lower() or 'gpt2' in name.lower():
        new_family = 'gpt'
    elif 'gpt-neo' in name.lower():
        new_family = 'gpt'
    else:
        new_family = old_family
    
    if old_family != new_family:
        print(f"✅ Fixed {name}: '{old_family}' → '{new_family}'")
        info['model_family'] = new_family
    else:
        print(f"⚠️  Unchanged {name}: '{old_family}'")

# Save fixed library
with open('fingerprint_library/reference_library.json', 'w') as f:
    json.dump(data, f, indent=2)

print("=" * 60)
print("Reference library family names fixed!")
