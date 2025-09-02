#!/usr/bin/env python3
"""
Real-time progress monitor for REV pipeline execution.
Displays progress bars, estimated time, and key metrics.
"""

import sys
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import os

class REVProgressMonitor:
    def __init__(self, log_file="llama70b_test_fixed.log"):
        self.log_file = log_file
        self.start_time = None
        self.current_layer = 0
        self.total_layers = 80
        self.layers_processed = set()
        self.current_phase = "Initializing"
        self.divergence_scores = []
        self.last_update = time.time()
        
    def parse_log_line(self, line):
        """Parse a log line for relevant information."""
        # Check for layer processing
        if "Profiling layer" in line:
            match = re.search(r"Profiling layer (\d+)", line)
            if match:
                self.current_layer = int(match.group(1))
                self.current_phase = f"Profiling Layer {self.current_layer}"
                
        # Check for successful probe
        elif "PROBE SUCCESS" in line:
            match = re.search(r"Layer\s+(\d+)\s+\|\s+Divergence:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)", line)
            if match:
                layer = int(match.group(1))
                divergence = float(match.group(2))
                exec_time = float(match.group(3))
                self.layers_processed.add(layer)
                self.divergence_scores.append(divergence)
                
        # Check for stage changes
        elif "[Stage" in line:
            match = re.search(r"\[Stage (\d+)/(\d+)\] (.+)", line)
            if match:
                stage_num = match.group(1)
                total_stages = match.group(2)
                stage_name = match.group(3).strip("...")
                self.current_phase = f"Stage {stage_num}/{total_stages}: {stage_name}"
                
        # Check for behavioral discovery
        elif "BEHAVIORAL-DISCOVERY" in line:
            if "Starting sophisticated restriction site discovery" in line:
                self.current_phase = "Behavioral Discovery"
            elif "Using" in line and "PoT challenges" in line:
                match = re.search(r"Using (\d+) PoT challenges across (\d+) layers", line)
                if match:
                    self.total_layers = int(match.group(2))
                    
        # Check for errors
        elif "ERROR" in line or "Failed" in line:
            self.current_phase = f"⚠️ Error encountered"
            
    def format_time(self, seconds):
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def calculate_eta(self):
        """Calculate estimated time of arrival."""
        if not self.start_time or len(self.layers_processed) == 0:
            return "Calculating..."
            
        elapsed = time.time() - self.start_time
        layers_done = len(self.layers_processed)
        
        # For behavioral profiling, we typically profile ~38 layers (not all 80)
        estimated_total_layers = min(38, self.total_layers)
        
        if layers_done > 0:
            avg_time_per_layer = elapsed / layers_done
            remaining_layers = estimated_total_layers - layers_done
            eta_seconds = remaining_layers * avg_time_per_layer
            
            if eta_seconds > 0:
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                return f"{self.format_time(eta_seconds)} ({eta_time.strftime('%H:%M')})"
        
        return "Unknown"
    
    def draw_progress_bar(self, current, total, width=40):
        """Draw a progress bar."""
        if total == 0:
            return "[" + "?" * width + "]"
            
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total) * 100
        return f"[{bar}] {percentage:.1f}%"
    
    def display(self):
        """Display the current progress."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("REV PIPELINE PROGRESS MONITOR".center(80))
        print("=" * 80)
        print()
        
        # Model info
        print(f"🤖 Model: Llama 3.3 70B Instruct")
        print(f"📁 Path: /Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
        print(f"⚙️  Mode: TRUE Segment Execution (Memory-Bounded)")
        print()
        
        # Current phase
        print(f"📍 Current Phase: {self.current_phase}")
        print()
        
        # Layer progress
        layers_done = len(self.layers_processed)
        estimated_total = min(38, self.total_layers)  # Behavioral profiling typically does ~38 layers
        
        print(f"📊 Layer Analysis Progress:")
        print(f"   Layers Processed: {layers_done}/{estimated_total}")
        print(f"   {self.draw_progress_bar(layers_done, estimated_total)}")
        
        if self.current_layer > 0 and self.current_layer not in self.layers_processed:
            print(f"   Currently Processing: Layer {self.current_layer}")
        print()
        
        # Divergence scores
        if self.divergence_scores:
            avg_divergence = sum(self.divergence_scores) / len(self.divergence_scores)
            min_div = min(self.divergence_scores)
            max_div = max(self.divergence_scores)
            print(f"📈 Divergence Scores:")
            print(f"   Average: {avg_divergence:.3f}")
            print(f"   Range: {min_div:.3f} - {max_div:.3f}")
            print(f"   Latest: {self.divergence_scores[-1]:.3f}")
            print()
        
        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"⏱️  Time:")
            print(f"   Elapsed: {self.format_time(elapsed)}")
            print(f"   ETA: {self.calculate_eta()}")
            
            if layers_done > 0:
                avg_time = elapsed / layers_done
                print(f"   Avg per layer: {self.format_time(avg_time)}")
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
        print("Press Ctrl+C to stop monitoring (pipeline will continue running)")
    
    def monitor(self):
        """Main monitoring loop."""
        print("Starting REV Pipeline Monitor...")
        print(f"Watching: {self.log_file}")
        print()
        
        # Start time tracking
        self.start_time = time.time()
        
        # Open log file and seek to end
        try:
            with open(self.log_file, 'r') as f:
                # Read existing content to catch up
                for line in f:
                    self.parse_log_line(line.strip())
                
                # Display initial state
                self.display()
                
                # Monitor new lines
                while True:
                    line = f.readline()
                    if line:
                        self.parse_log_line(line.strip())
                        
                        # Update display every second or on significant events
                        if time.time() - self.last_update > 1.0:
                            self.display()
                            self.last_update = time.time()
                    else:
                        # No new line, sleep briefly
                        time.sleep(0.5)
                        
                        # Still update display periodically
                        if time.time() - self.last_update > 5.0:
                            self.display()
                            self.last_update = time.time()
                            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped. Pipeline continues running in background.")
            print(f"To check pipeline output: tail -f {self.log_file}")
        except FileNotFoundError:
            print(f"Error: Log file '{self.log_file}' not found!")
            print("Make sure the pipeline is running and creating the log file.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor REV pipeline progress")
    parser.add_argument("--log", default="llama70b_test_fixed.log", 
                       help="Log file to monitor")
    args = parser.parse_args()
    
    monitor = REVProgressMonitor(args.log)
    monitor.monitor()

if __name__ == "__main__":
    main()