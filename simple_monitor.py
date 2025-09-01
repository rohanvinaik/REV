#!/usr/bin/env python3
"""
Simple progress monitor for REV pipeline - prints updates without clearing screen.
"""

import sys
import time
import re
from datetime import datetime, timedelta

def monitor_log(log_file="llama70b_test_fixed.log"):
    """Monitor log file and print progress updates."""
    
    start_time = time.time()
    layers_processed = set()
    divergence_scores = []
    last_layer = 0
    
    print("=" * 80)
    print("REV PIPELINE MONITOR - Llama 3.3 70B")
    print("=" * 80)
    print()
    
    try:
        with open(log_file, 'r') as f:
            # Read existing content
            for line in f:
                if "PROBE SUCCESS" in line:
                    match = re.search(r"Layer\s+(\d+)\s+\|\s+Divergence:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)ms", line)
                    if match:
                        layer = int(match.group(1))
                        divergence = float(match.group(2))
                        exec_time = float(match.group(3)) / 1000  # Convert to seconds
                        layers_processed.add(layer)
                        divergence_scores.append(divergence)
                        last_layer = max(last_layer, layer)
            
            # Print initial status
            print(f"Catching up... Found {len(layers_processed)} layers already processed")
            if divergence_scores:
                print(f"Average divergence so far: {sum(divergence_scores)/len(divergence_scores):.3f}")
            print("\nMonitoring new progress...\n")
            
            # Monitor new lines
            while True:
                line = f.readline()
                if line:
                    # Check for layer processing start
                    if "Profiling layer" in line:
                        match = re.search(r"Profiling layer (\d+)", line)
                        if match:
                            layer = int(match.group(1))
                            elapsed = time.time() - start_time
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Layer {layer} (Elapsed: {elapsed/60:.1f}m)")
                    
                    # Check for successful probe
                    elif "PROBE SUCCESS" in line:
                        match = re.search(r"Layer\s+(\d+)\s+\|\s+Divergence:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)ms\s+\|\s+Device:\s+(\w+)\s+\|\s+Probe:\s+(.+?)\.{3}", line)
                        if match:
                            layer = int(match.group(1))
                            divergence = float(match.group(2))
                            exec_time = float(match.group(3)) / 1000
                            device = match.group(4)
                            probe_type = match.group(5)[:30]  # First 30 chars of probe
                            
                            layers_processed.add(layer)
                            divergence_scores.append(divergence)
                            
                            # Calculate stats
                            avg_div = sum(divergence_scores) / len(divergence_scores)
                            layers_done = len(layers_processed)
                            
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Layer {layer:2d} | Div: {divergence:.3f} | Time: {exec_time:5.1f}s | Probe: {probe_type}")
                            
                            # Print summary every 4 probes (one complete layer)
                            if len(divergence_scores) % 4 == 0:
                                print(f"  └─ Layers complete: {layers_done} | Avg divergence: {avg_div:.3f}")
                                
                                # Estimate time remaining (38 layers typical for behavioral profiling)
                                if layers_done > 0:
                                    elapsed = time.time() - start_time
                                    avg_time_per_layer = elapsed / layers_done
                                    remaining = max(0, 38 - layers_done)
                                    eta_seconds = remaining * avg_time_per_layer
                                    
                                    if eta_seconds > 0:
                                        if eta_seconds < 3600:
                                            eta_str = f"{eta_seconds/60:.0f}m"
                                        else:
                                            eta_str = f"{eta_seconds/3600:.1f}h"
                                        print(f"  └─ Progress: {layers_done}/38 layers | ETA: {eta_str}")
                                print()
                    
                    # Check for errors
                    elif "ERROR" in line or "Failed" in line:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  {line.strip()}")
                    
                    # Check for stage changes
                    elif "[Stage" in line and "]" in line:
                        match = re.search(r"\[Stage (\d+)/(\d+)\] (.+)", line)
                        if match:
                            stage = f"Stage {match.group(1)}/{match.group(2)}: {match.group(3)}"
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📍 {stage}")
                            print()
                            
                else:
                    time.sleep(1)  # Wait for new lines
                    
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Pipeline continues in background.")
        print(f"Final stats: {len(layers_processed)} layers, Avg divergence: {sum(divergence_scores)/len(divergence_scores):.3f}" if divergence_scores else "")
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found!")

if __name__ == "__main__":
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else "llama70b_test_fixed.log"
    monitor_log(log_file)