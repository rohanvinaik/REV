#!/usr/bin/env python3
"""
Clean up reference library by removing duplicate and inadequate entries.
Keeps only the best reference for each model family.
"""

import json
import sys
from pathlib import Path

def clean_reference_library(library_path="fingerprint_library/reference_library.json"):
    """Clean up the reference library."""
    
    # Load current library
    with open(library_path, 'r') as f:
        data = json.load(f)
    
    print("Current reference library status:")
    print("=" * 60)
    
    # Analyze current entries
    families = {}
    for name, info in data['fingerprints'].items():
        if 'reference' in name:
            family = info.get('model_family', 'unknown')
            challenges = info.get('challenges_processed', 0)
            print(f"{family:30s}: {challenges:4d} challenges ({name})")
            
            # Track best entry per family
            if family not in families or challenges > families[family]['challenges']:
                families[family] = {
                    'name': name,
                    'challenges': challenges,
                    'info': info
                }
    
    print("\n" + "=" * 60)
    print("Cleaning reference library...")
    print("=" * 60)
    
    # Keep only the best reference per family with adequate challenges
    MIN_CHALLENGES = 250  # Minimum acceptable challenge count
    
    cleaned_fingerprints = {}
    removed_count = 0
    kept_count = 0
    
    for family, best in families.items():
        if best['challenges'] >= MIN_CHALLENGES:
            # Keep this reference
            cleaned_fingerprints[best['name']] = best['info']
            print(f"✅ KEPT: {family} with {best['challenges']} challenges")
            kept_count += 1
        else:
            print(f"❌ REMOVED: {family} with only {best['challenges']} challenges (< {MIN_CHALLENGES})")
            removed_count += 1
    
    # Update the library
    data['fingerprints'] = cleaned_fingerprints
    
    # Save cleaned library
    backup_path = library_path + ".backup"
    print(f"\n📁 Backing up original to: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(json.load(open(library_path)), f, indent=2)
    
    print(f"📁 Saving cleaned library to: {library_path}")
    with open(library_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  References kept: {kept_count}")
    print(f"  References removed: {removed_count}")
    print(f"  Minimum challenge threshold: {MIN_CHALLENGES}")
    print("=" * 60)
    
    return kept_count, removed_count

if __name__ == "__main__":
    kept, removed = clean_reference_library()
    sys.exit(0 if kept > 0 else 1)