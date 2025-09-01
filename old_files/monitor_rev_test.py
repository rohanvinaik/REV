#!/usr/bin/env python3
"""
Monitor REV test progress for LLaMA 3.3 70B
"""

import os
import time
import psutil
from datetime import datetime
from pathlib import Path

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def monitor_process():
    print("=" * 60)
    print("REV LLaMA 3.3 70B Test Monitor")
    print("=" * 60)
    
    # Find the process
    target_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'run_rev_complete.py' in ' '.join(cmdline) and 'llama-3.3-70b' in ' '.join(cmdline):
                target_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not target_pid:
        print("REV test process not found. It may have completed.")
        # Check for output file
        output_file = Path("llama3.3_70b_rev_final.json")
        if output_file.exists():
            print(f"✅ Output file found: {output_file}")
            print(f"   Size: {format_bytes(output_file.stat().st_size)}")
            print(f"   Modified: {datetime.fromtimestamp(output_file.stat().st_mtime)}")
        return
    
    print(f"Found process: PID {target_pid}")
    proc = psutil.Process(target_pid)
    
    # Monitor
    start_time = time.time()
    while proc.is_running():
        try:
            # Get process stats
            cpu_percent = proc.cpu_percent(interval=1)
            memory_info = proc.memory_info()
            
            # Calculate runtime
            runtime = time.time() - start_time
            runtime_str = f"{int(runtime//60)}m {int(runtime%60)}s"
            
            # Display status
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update")
            print(f"  CPU Usage: {cpu_percent:.1f}%")
            print(f"  Memory: {format_bytes(memory_info.rss)}")
            print(f"  Runtime: {runtime_str}")
            
            # Check for output file
            output_file = Path("llama3.3_70b_rev_final.json")
            if output_file.exists():
                print(f"  Output: {format_bytes(output_file.stat().st_size)}")
            
            time.sleep(10)
            
        except psutil.NoSuchProcess:
            print("\n✅ Process completed!")
            break
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
    
    # Final check
    output_file = Path("llama3.3_70b_rev_final.json")
    if output_file.exists():
        print(f"\n✅ Test completed successfully!")
        print(f"   Output: {output_file}")
        print(f"   Size: {format_bytes(output_file.stat().st_size)}")

if __name__ == "__main__":
    monitor_process()