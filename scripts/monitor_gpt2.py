#!/usr/bin/env python3
"""
Real-time monitoring script for GPT-2 behavioral fingerprinting
"""

import os
import time
import json
import psutil
from datetime import datetime
from pathlib import Path

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    system = psutil.virtual_memory()
    
    return {
        'process_gb': process.memory_info().rss / (1024**3),
        'system_percent': system.percent,
        'system_available_gb': system.available / (1024**3),
        'system_total_gb': system.total / (1024**3)
    }

def parse_log_progress(log_file):
    """Parse progress from log file"""
    if not os.path.exists(log_file):
        return None
    
    progress = {
        'current_stage': 'Unknown',
        'current_layer': None,
        'layers_completed': 0,
        'challenges_completed': 0,
        'errors': [],
        'last_update': None
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        # Check for stage markers
        if '[Stage' in line:
            stage_match = line.split('[Stage')[1].split(']')[0]
            progress['current_stage'] = stage_match.strip()
            
        # Check for layer progress
        if 'Processing layer' in line or 'Layer' in line:
            try:
                if 'Layer' in line:
                    parts = line.split('Layer')
                    if len(parts) > 1:
                        layer_num = parts[1].split()[0].strip(':')
                        if layer_num.isdigit():
                            progress['current_layer'] = int(layer_num)
            except:
                pass
                
        # Check for segment execution
        if '[SEGMENT-EXEC]' in line:
            if 'Loaded weights for layer' in line:
                try:
                    layer_num = int(line.split('layer')[1].strip())
                    progress['layers_completed'] = max(progress.get('layers_completed', 0), layer_num)
                except:
                    pass
                    
        # Check for challenge completion
        if 'Challenge' in line and 'completed' in line:
            progress['challenges_completed'] += 1
            
        # Check for errors
        if 'ERROR' in line or 'Failed' in line:
            progress['errors'].append(line.strip())
            
        # Get timestamp
        if '2025-' in line:
            try:
                timestamp = line.split(' - ')[0]
                progress['last_update'] = timestamp
            except:
                pass
    
    return progress

def display_dashboard(log_file, output_file):
    """Display monitoring dashboard"""
    os.system('clear')
    
    print("=" * 80)
    print(" GPT-2 BEHAVIORAL FINGERPRINTING MONITOR ".center(80))
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Memory stats
    mem = get_memory_usage()
    print("📊 MEMORY USAGE")
    print(f"  Process: {mem['process_gb']:.2f} GB")
    print(f"  System: {mem['system_percent']:.1f}% ({mem['system_total_gb'] - mem['system_available_gb']:.1f}/{mem['system_total_gb']:.1f} GB)")
    print()
    
    # Progress from log
    progress = parse_log_progress(log_file)
    if progress:
        print("📈 PROCESSING PROGRESS")
        print(f"  Stage: {progress['current_stage']}")
        if progress['current_layer'] is not None:
            print(f"  Current Layer: {progress['current_layer']}")
        print(f"  Layers Completed: {progress['layers_completed']}")
        print(f"  Challenges: {progress['challenges_completed']}/5")
        if progress['last_update']:
            print(f"  Last Update: {progress['last_update']}")
        print()
        
        if progress['errors']:
            print("⚠️  RECENT ERRORS")
            for error in progress['errors'][-3:]:  # Show last 3 errors
                print(f"  {error[:75]}...")
            print()
    
    # Check output file
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                output = json.load(f)
            
            if output.get('models_processed'):
                print("✅ RESULTS")
                for model_name, result in output.get('results', {}).items():
                    print(f"  Model: {model_name}")
                    if 'stages' in result:
                        for stage, info in result['stages'].items():
                            if info.get('success'):
                                print(f"    {stage}: ✓ ({info.get('time', 0):.1f}s)")
                print()
        except:
            pass
    
    print("=" * 80)
    print("Press Ctrl+C to exit monitoring")

def main():
    log_file = "outputs/gpt2_behavioral_fingerprint.log"
    output_file = "outputs/gpt2_behavioral_fingerprint.json"
    
    print("Starting GPT-2 monitoring...")
    print(f"Log file: {log_file}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        while True:
            display_dashboard(log_file, output_file)
            time.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()