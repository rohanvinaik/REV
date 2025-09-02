#!/usr/bin/env python3
"""
One-shot progress checker for REV pipeline test
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

LOG_FILE = "/Users/rohanvinaik/REV/rev_70b_pot_probes.log"
START_TIME = datetime(2025, 8, 31, 20, 40, 0)
TOTAL_LAYERS = 80
PROBES_PER_LAYER = 4

def create_progress_bar(current, total, width=50):
    """Create a visual progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"

def main():
    if not Path(LOG_FILE).exists():
        print("❌ Log file not found!")
        return
    
    # Read last 1000 lines
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()[-1000:]
    
    # Find all successful probes
    layer_probes = {}
    current_layer = 0
    
    for line in lines:
        if "PROBE SUCCESS: Layer" in line:
            match = re.search(r'Layer\s+(\d+).*Divergence:\s+([\d.]+)', line)
            if match:
                layer = int(match.group(1))
                divergence = float(match.group(2))
                current_layer = max(current_layer, layer)
                
                if layer not in layer_probes:
                    layer_probes[layer] = []
                layer_probes[layer].append(divergence)
        
        if "Profiling layer" in line:
            match = re.search(r'Profiling layer (\d+)', line)
            if match:
                current_layer = max(current_layer, int(match.group(1)))
    
    # Calculate progress
    completed_layers = sum(1 for probes in layer_probes.values() if len(probes) >= 4)
    current_probes = len(layer_probes.get(current_layer, []))
    total_probes = sum(len(probes) for probes in layer_probes.values())
    
    # Time calculations
    now = datetime.now()
    elapsed = (now - START_TIME).total_seconds() / 3600
    progress = completed_layers + (current_probes / PROBES_PER_LAYER)
    
    print("\n" + "="*70)
    print("🔬 REV PIPELINE PROGRESS - Llama 3.3 70B")
    print("="*70)
    
    print(f"\n📍 Current: Layer {current_layer}/{TOTAL_LAYERS} - Probe {current_probes}/{PROBES_PER_LAYER}")
    print(f"⏱️  Elapsed: {elapsed:.1f} hours")
    
    print(f"\n📊 OVERALL: {create_progress_bar(progress, TOTAL_LAYERS)}")
    print(f"   Complete: {completed_layers} layers | Processing: Layer {current_layer}")
    print(f"   Probes: {total_probes}/{TOTAL_LAYERS * PROBES_PER_LAYER}")
    
    # Recent divergences
    if layer_probes:
        print(f"\n🎯 DIVERGENCE SCORES:")
        for layer in sorted(layer_probes.keys())[-3:]:
            divs = layer_probes[layer]
            avg = sum(divs)/len(divs) if divs else 0
            print(f"   Layer {layer}: {avg:.3f} (n={len(divs)})")
    
    # Time estimate
    if progress > 0:
        rate = elapsed / progress
        remaining = (TOTAL_LAYERS - progress) * rate
        eta = now + timedelta(hours=remaining)
        print(f"\n⏰ ESTIMATES:")
        print(f"   Rate: {rate*60:.1f} min/layer")
        print(f"   Remaining: {remaining:.1f} hours")
        print(f"   ETA: {eta.strftime('%m/%d %I:%M %p')}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()