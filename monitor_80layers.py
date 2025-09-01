#!/usr/bin/env python3
"""
Progress monitor for full 80-layer REV behavioral profiling.
Shows real-time progress, time estimates, and divergence patterns.
"""

import sys
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict
import os

class Layer80Monitor:
    def __init__(self, log_file="llama70b_full_80layers.log"):
        self.log_file = log_file
        self.start_time = None
        self.total_layers = 80
        self.probes_per_layer = 4
        
        # Track completion by layer
        self.layer_probes = defaultdict(list)  # layer -> list of (divergence, time_ms)
        self.current_layer = -1
        self.last_update = time.time()
        
    def parse_log_line(self, line):
        """Extract progress information from log lines."""
        
        # Track current layer being profiled
        if "Profiling layer" in line and "BEHAVIORAL" in line:
            match = re.search(r"Profiling layer (\d+)$", line)
            if match:
                self.current_layer = int(match.group(1))
        
        # Track successful probes
        elif "PROBE SUCCESS" in line:
            match = re.search(r"Layer\s+(\d+)\s+\|\s+Divergence:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)ms", line)
            if match:
                layer = int(match.group(1))
                divergence = float(match.group(2))
                time_ms = float(match.group(3))
                self.layer_probes[layer].append((divergence, time_ms))
    
    def get_progress_stats(self):
        """Calculate current progress statistics."""
        # Count completed layers (those with 4 probes)
        completed_layers = sum(1 for probes in self.layer_probes.values() if len(probes) >= 4)
        
        # Count total probes
        total_probes = sum(len(probes) for probes in self.layer_probes.values())
        
        # Get all divergence scores
        all_divergences = [div for probes in self.layer_probes.values() for div, _ in probes]
        
        # Calculate average time per probe
        all_times = [t for probes in self.layer_probes.values() for _, t in probes]
        avg_time_ms = sum(all_times) / len(all_times) if all_times else 0
        
        return {
            'completed_layers': completed_layers,
            'total_probes': total_probes,
            'all_divergences': all_divergences,
            'avg_probe_time_ms': avg_time_ms
        }
    
    def format_time(self, seconds):
        """Format seconds into readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def calculate_eta(self, stats):
        """Calculate estimated time to completion."""
        if not self.start_time or stats['total_probes'] == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        probes_done = stats['total_probes']
        probes_total = self.total_layers * self.probes_per_layer  # 80 * 4 = 320
        
        if probes_done > 0:
            avg_time_per_probe = elapsed / probes_done
            remaining_probes = probes_total - probes_done
            eta_seconds = remaining_probes * avg_time_per_probe
            
            if eta_seconds > 0:
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                return f"{self.format_time(eta_seconds)} (at {eta_time.strftime('%H:%M')})"
        
        return "Unknown"
    
    def display(self):
        """Display current progress."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        stats = self.get_progress_stats()
        
        print("=" * 80)
        print("🔬 REV 70B BEHAVIORAL PROFILING - FULL 80 LAYERS".center(80))
        print("=" * 80)
        print()
        
        # Progress bar for layers
        completed = stats['completed_layers']
        bar_width = 50
        filled = int(bar_width * completed / self.total_layers)
        bar = "█" * filled + "░" * (bar_width - filled)
        percentage = (completed / self.total_layers) * 100
        
        print(f"📊 Layer Progress: {completed}/{self.total_layers} layers complete ({percentage:.1f}%)")
        print(f"   [{bar}]")
        if self.current_layer >= 0 and self.current_layer not in [l for l, p in self.layer_probes.items() if len(p) >= 4]:
            layer_probe_count = len(self.layer_probes.get(self.current_layer, []))
            print(f"   Currently: Layer {self.current_layer} (probe {layer_probe_count + 1}/4)")
        print()
        
        # Probe progress
        probes_done = stats['total_probes']
        probes_total = 320  # 80 layers * 4 probes
        probe_percentage = (probes_done / probes_total) * 100
        print(f"🔍 Probe Progress: {probes_done}/{probes_total} probes ({probe_percentage:.1f}%)")
        print()
        
        # Divergence analysis
        if stats['all_divergences']:
            divs = stats['all_divergences']
            avg_div = sum(divs) / len(divs)
            min_div = min(divs)
            max_div = max(divs)
            
            print(f"📈 Divergence Analysis:")
            print(f"   Average: {avg_div:.3f}")
            print(f"   Range: {min_div:.3f} - {max_div:.3f}")
            
            # Show last 5 layers with their average divergence
            if len(self.layer_probes) > 0:
                print(f"\n   Recent layers:")
                recent_layers = sorted(self.layer_probes.keys())[-5:]
                for layer in recent_layers:
                    probes = self.layer_probes[layer]
                    if probes:
                        layer_avg = sum(d for d, _ in probes) / len(probes)
                        status = "✓" if len(probes) >= 4 else f"{len(probes)}/4"
                        print(f"     Layer {layer:2d}: {layer_avg:.3f} ({status})")
        print()
        
        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"⏱️  Timing:")
            print(f"   Elapsed: {self.format_time(elapsed)}")
            print(f"   ETA: {self.calculate_eta(stats)}")
            
            if stats['avg_probe_time_ms'] > 0:
                avg_probe_sec = stats['avg_probe_time_ms'] / 1000
                print(f"   Avg per probe: {self.format_time(avg_probe_sec)}")
                avg_layer_time = avg_probe_sec * 4
                print(f"   Avg per layer: {self.format_time(avg_layer_time)}")
        print()
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024**3)
            sys_mem = psutil.virtual_memory()
            print(f"💾 Memory:")
            print(f"   Process: {mem_gb:.1f}GB")
            print(f"   System: {sys_mem.used/(1024**3):.1f}/{sys_mem.total/(1024**3):.0f}GB ({sys_mem.percent:.0f}%)")
        except:
            pass
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring (pipeline continues in background)")
    
    def monitor(self):
        """Main monitoring loop."""
        print("Starting 80-Layer Behavioral Profiling Monitor...")
        print(f"Watching: {self.log_file}")
        
        self.start_time = time.time()
        
        try:
            with open(self.log_file, 'r') as f:
                # Process existing lines
                for line in f:
                    self.parse_log_line(line.strip())
                
                # Initial display
                self.display()
                
                # Monitor new lines
                while True:
                    line = f.readline()
                    if line:
                        self.parse_log_line(line.strip())
                        
                        # Update display on significant events or every 2 seconds
                        if (time.time() - self.last_update > 2.0 or 
                            "PROBE SUCCESS" in line or 
                            "Profiling layer" in line):
                            self.display()
                            self.last_update = time.time()
                    else:
                        time.sleep(0.5)
                        
                        # Still update periodically
                        if time.time() - self.last_update > 5.0:
                            self.display()
                            self.last_update = time.time()
                            
        except KeyboardInterrupt:
            print("\n\nMonitor stopped. Pipeline continues in background.")
            
            # Show final stats
            stats = self.get_progress_stats()
            print(f"\nFinal Summary:")
            print(f"  Layers completed: {stats['completed_layers']}/80")
            print(f"  Total probes: {stats['total_probes']}/320")
            if stats['all_divergences']:
                avg = sum(stats['all_divergences']) / len(stats['all_divergences'])
                print(f"  Average divergence: {avg:.3f}")
            
            print(f"\nTo resume monitoring: python {__file__}")
            print(f"To check logs: tail -f {self.log_file}")
            
        except FileNotFoundError:
            print(f"Error: Log file '{self.log_file}' not found!")
            print("Make sure the pipeline is running.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor 80-layer behavioral profiling")
    parser.add_argument("--log", default="llama70b_full_80layers.log", 
                       help="Log file to monitor")
    args = parser.parse_args()
    
    monitor = Layer80Monitor(args.log)
    monitor.monitor()

if __name__ == "__main__":
    main()