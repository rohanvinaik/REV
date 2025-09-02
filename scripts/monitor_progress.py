#!/usr/bin/env python3
"""
Simple visual progress monitor for REV pipeline test
Updates every 30 seconds with current status
"""

import time
import re
from datetime import datetime, timedelta
from pathlib import Path

LOG_FILE = "/Users/rohanvinaik/REV/rev_70b_pot_probes.log"
START_TIME = datetime(2025, 8, 31, 20, 40, 0)
TOTAL_LAYERS = 80
PROBES_PER_LAYER = 4

def parse_log():
    """Parse the log file to extract progress information."""
    if not Path(LOG_FILE).exists():
        return None
    
    # Read last 1000 lines for efficiency
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()[-1000:]
    
    # Find all successful probes
    probes = []
    current_layer = 0
    layer_probes = {}
    
    for line in lines:
        # Check for successful probes
        if "PROBE SUCCESS: Layer" in line:
            match = re.search(r'Layer\s+(\d+).*Divergence:\s+([\d.]+).*Probe:\s+(.+?)\.\.\.', line)
            if match:
                layer = int(match.group(1))
                divergence = float(match.group(2))
                probe_type = match.group(3)[:30]  # First 30 chars of probe
                
                if layer not in layer_probes:
                    layer_probes[layer] = []
                layer_probes[layer].append({
                    'divergence': divergence,
                    'type': probe_type
                })
                current_layer = max(current_layer, layer)
        
        # Check for "Profiling layer X"
        if "Profiling layer" in line:
            match = re.search(r'Profiling layer (\d+)', line)
            if match:
                current_layer = max(current_layer, int(match.group(1)))
    
    return {
        'current_layer': current_layer,
        'layer_probes': layer_probes,
        'last_update': datetime.now()
    }

def create_progress_bar(current, total, width=50):
    """Create a visual progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"

def format_time(seconds):
    """Format seconds into readable time."""
    if seconds < 0:
        return "calculating..."
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    
    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h {minutes}m"
    return f"{hours:.0f}h {minutes:.0f}m"

def display_monitor():
    """Display the progress monitor."""
    data = parse_log()
    if not data:
        print("❌ Log file not found!")
        return
    
    # Clear screen (works on Unix-like systems)
    print("\033[2J\033[H")
    
    # Header
    print("=" * 70)
    print("🔬 REV PIPELINE PROGRESS MONITOR - Llama 3.3 70B")
    print("=" * 70)
    print()
    
    # Time information
    now = datetime.now()
    elapsed = (now - START_TIME).total_seconds()
    elapsed_str = format_time(elapsed)
    
    print(f"⏱️  Started: {START_TIME.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️  Current: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️  Elapsed: {elapsed_str}")
    print()
    
    # Current status
    current_layer = data['current_layer']
    layer_probes = data['layer_probes']
    
    # Count completed layers and probes
    completed_layers = 0
    total_probes_done = 0
    
    for layer, probes in layer_probes.items():
        if len(probes) >= 4:
            completed_layers += 1
        total_probes_done += len(probes)
    
    # If current layer has incomplete probes, show partial progress
    if current_layer in layer_probes:
        current_layer_probes = len(layer_probes[current_layer])
    else:
        current_layer_probes = 0
    
    print(f"📍 Current Layer: {current_layer} of {TOTAL_LAYERS}")
    print(f"📊 Probes on Layer {current_layer}: {current_layer_probes}/{PROBES_PER_LAYER}")
    print()
    
    # Overall progress
    overall_progress = (completed_layers + (current_layer_probes / PROBES_PER_LAYER)) / TOTAL_LAYERS
    print("📈 OVERALL PROGRESS:")
    print(f"   {create_progress_bar(overall_progress * TOTAL_LAYERS, TOTAL_LAYERS)}")
    print(f"   Layers Complete: {completed_layers}/{TOTAL_LAYERS}")
    print(f"   Total Probes: {total_probes_done}/{TOTAL_LAYERS * PROBES_PER_LAYER}")
    print()
    
    # Current layer detail
    if current_layer in layer_probes and layer_probes[current_layer]:
        print(f"🔍 LAYER {current_layer} DETAILS:")
        for i, probe in enumerate(layer_probes[current_layer], 1):
            status = "✅"
            div = probe['divergence']
            probe_type = probe['type']
            print(f"   {status} Probe {i}: Divergence {div:.3f} - {probe_type}")
        
        # Show pending probes
        for i in range(len(layer_probes[current_layer]) + 1, PROBES_PER_LAYER + 1):
            if i == len(layer_probes[current_layer]) + 1:
                print(f"   ⏳ Probe {i}: Processing...")
            else:
                print(f"   ⏸️  Probe {i}: Pending")
        print()
    
    # Recent divergence values
    print("📊 RECENT DIVERGENCE SCORES:")
    recent_layers = sorted(layer_probes.keys())[-3:]
    for layer in recent_layers:
        if layer in layer_probes and layer_probes[layer]:
            divs = [p['divergence'] for p in layer_probes[layer]]
            avg_div = sum(divs) / len(divs) if divs else 0
            print(f"   Layer {layer}: {avg_div:.3f} (avg of {len(divs)} probes)")
    print()
    
    # Time estimates
    if completed_layers > 0 or current_layer_probes > 0:
        progress_decimal = completed_layers + (current_layer_probes / PROBES_PER_LAYER)
        if progress_decimal > 0:
            avg_time_per_layer = elapsed / progress_decimal
            remaining_layers = TOTAL_LAYERS - progress_decimal
            remaining_seconds = remaining_layers * avg_time_per_layer
            completion_time = now + timedelta(seconds=remaining_seconds)
            
            print("⏰ TIME ESTIMATES:")
            print(f"   Avg per layer: {format_time(avg_time_per_layer)}")
            print(f"   Remaining: {format_time(remaining_seconds)}")
            print(f"   Est. Completion: {completion_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("⏰ TIME ESTIMATES: Calculating...")
    
    print()
    print("=" * 70)
    print("Refreshes every 30 seconds. Press Ctrl+C to exit.")

def main():
    """Main monitoring loop."""
    print("Starting REV Pipeline Progress Monitor...")
    print("Reading from:", LOG_FILE)
    print()
    
    try:
        while True:
            display_monitor()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())