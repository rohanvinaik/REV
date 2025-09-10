#!/usr/bin/env python3
"""
Reference Build Progress Tracker
Monitors running reference library builds and estimates completion times
"""

import os
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import re

class BuildTracker:
    def __init__(self):
        self.builds = {
            'pythia-70m': {'layers': 6, 'probes': 259, 'start': '14:37:22'},
            'distilgpt2': {'layers': 6, 'probes': 257, 'start': '12:01:33'},  
            'gpt-neo-125m': {'layers': 12, 'probes': 257, 'start': '12:01:36'},
            'llama-2-7b': {'layers': 32, 'probes': 301, 'start': '12:01:36'},
            'mistral': {'layers': 32, 'probes': 301, 'start': '12:01:36'},
            'falcon-7b': {'layers': 32, 'probes': 301, 'start': '12:01:37'},
            'phi-2': {'layers': 32, 'probes': 301, 'start': '12:01:37'},
        }
        
    def get_running_processes(self) -> List[Tuple[str, int]]:
        """Get list of running reference build processes"""
        try:
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True
            )
            
            processes = []
            for line in result.stdout.split('\n'):
                if 'run_rev.py' in line and '--build-reference' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = int(parts[1])
                        # Extract model name from path
                        for part in parts:
                            if '/LLM_models/' in part:
                                model = part.split('/')[-1]
                                processes.append((model, pid))
                                break
            return processes
        except:
            return []
    
    def estimate_completion(self, model_key: str, current_time: datetime) -> str:
        """Estimate completion time based on model complexity"""
        if model_key not in self.builds:
            return "Unknown"
            
        build = self.builds[model_key]
        total_probes = build['layers'] * build['probes']
        
        # Rough estimates based on observed performance
        # ~50ms per probe on average for small models, ~100ms for large
        if build['layers'] <= 12:
            ms_per_probe = 50
        else:
            ms_per_probe = 100
            
        total_seconds = (total_probes * ms_per_probe) / 1000
        
        # Add overhead for orchestration and file I/O (20%)
        total_seconds *= 1.2
        
        # Parse start time
        start_parts = build['start'].split(':')
        start_datetime = current_time.replace(
            hour=int(start_parts[0]),
            minute=int(start_parts[1]),
            second=int(start_parts[2])
        )
        
        # If start time is after current time, it started yesterday
        if start_datetime > current_time:
            start_datetime -= timedelta(days=1)
            
        elapsed = (current_time - start_datetime).total_seconds()
        remaining = max(0, total_seconds - elapsed)
        
        eta = current_time + timedelta(seconds=remaining)
        
        progress = min(100, (elapsed / total_seconds) * 100)
        
        return {
            'progress': progress,
            'eta': eta.strftime('%H:%M:%S'),
            'remaining_hours': remaining / 3600,
            'total_probes': total_probes
        }
    
    def display_status(self):
        """Display current status of all builds"""
        os.system('clear')
        current_time = datetime.now()
        
        print("=" * 80)
        print(f"REV REFERENCE BUILD TRACKER - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        running = self.get_running_processes()
        running_models = [r[0] for r in running]
        
        print(f"Active Builds: {len(running)}")
        print()
        print(f"{'Model':<20} {'Layers':<8} {'Probes':<8} {'Total':<10} {'Progress':<12} {'ETA':<10} {'Remaining'}")
        print("-" * 80)
        
        for model_key, info in self.builds.items():
            # Check if this model is running
            is_running = any(model_key in m for m in running_models)
            
            if is_running:
                status = self.estimate_completion(model_key, current_time)
                total_probes = info['layers'] * info['probes']
                
                # Progress bar
                bar_width = 20
                filled = int(bar_width * status['progress'] / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                print(f"{model_key:<20} {info['layers']:<8} {info['probes']:<8} "
                      f"{total_probes:<10,} [{bar}] {status['progress']:>5.1f}% "
                      f"{status['eta']:<10} {status['remaining_hours']:.1f}h")
            else:
                print(f"{model_key:<20} {'STOPPED or COMPLETED'}")
        
        print()
        print("Legend:")
        print("  • Probes are applied to EACH layer")
        print("  • Total = Layers × Probes (total evaluations)")
        print("  • Times are estimates based on ~50-100ms per probe")
        print()
        print("Press Ctrl+C to exit")

def main():
    tracker = BuildTracker()
    
    try:
        while True:
            tracker.display_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\nTracker stopped.")

if __name__ == "__main__":
    main()