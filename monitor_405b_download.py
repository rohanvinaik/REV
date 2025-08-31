#!/usr/bin/env python3
"""
Real-time download progress monitor for LLaMA 3.1 405B FP8
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

def get_directory_size(path):
    """Get total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total

def count_files(path):
    """Count total files in directory."""
    count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            count += len(filenames)
    except (OSError, FileNotFoundError):
        pass
    return count

def format_bytes(bytes_val):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def format_time(seconds):
    """Format seconds to readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def monitor_download():
    """Monitor download progress in real-time."""
    download_path = Path("/Users/rohanvinaik/LLM_models/llama-3.1-405b-fp8")
    
    # Estimated total size for LLaMA 3.1 405B FP8
    # 405B parameters × 1 byte (FP8) ≈ 405GB
    estimated_total_bytes = 405 * 1024 * 1024 * 1024  # 405GB
    total_files_expected = 109  # model shards
    
    print("=" * 80)
    print("LLaMA 3.1 405B FP8 Download Monitor")
    print("=" * 80)
    print(f"Target directory: {download_path}")
    print(f"Estimated total size: {format_bytes(estimated_total_bytes)}")
    print(f"Expected model shards: {total_files_expected}")
    print("=" * 80)
    
    start_time = time.time()
    last_size = 0
    last_time = start_time
    speeds = []
    
    while True:
        current_time = time.time()
        current_size = get_directory_size(download_path)
        file_count = count_files(download_path)
        
        # Count safetensors files specifically
        safetensor_files = 0
        if download_path.exists():
            safetensor_files = len(list(download_path.glob("*.safetensors")))
        
        if current_size == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for download to start...")
            time.sleep(5)
            continue
        
        # Calculate progress
        percentage = min((current_size / estimated_total_bytes) * 100, 100)
        
        # Calculate speed
        time_diff = current_time - last_time
        if time_diff > 0 and current_size > last_size:
            current_speed = (current_size - last_size) / time_diff
            speeds.append(current_speed)
            if len(speeds) > 10:
                speeds.pop(0)
            avg_speed = sum(speeds) / len(speeds)
        else:
            avg_speed = 0
        
        # Calculate ETA
        if avg_speed > 0:
            remaining_bytes = estimated_total_bytes - current_size
            eta_seconds = remaining_bytes / avg_speed
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Calculate elapsed time
        elapsed_seconds = current_time - start_time
        elapsed_str = format_time(elapsed_seconds)
        
        # Display progress
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Update")
        print(f"  Downloaded: {format_bytes(current_size)} / {format_bytes(estimated_total_bytes)} ({percentage:.1f}%)")
        print(f"  Model shards: {safetensor_files}/{total_files_expected}")
        print(f"  Total files: {file_count}")
        print(f"  Speed: {format_bytes(avg_speed)}/s")
        print(f"  Elapsed: {elapsed_str}")
        print(f"  ETA: {eta_str}")
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * percentage / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}] {percentage:.1f}%")
        
        # Check if download is complete
        if safetensor_files >= total_files_expected or percentage >= 99.9:
            print("\n🎉 Download appears to be complete!")
            print(f"Final size: {format_bytes(current_size)}")
            print(f"Model shards downloaded: {safetensor_files}")
            print("\nReady to test with REV framework:")
            print("python run_rev_complete.py /Users/rohanvinaik/LLM_models/llama-3.1-405b-fp8 \\")
            print("  --device auto --challenges 1 --max-tokens 5")
            break
        
        last_size = current_size
        last_time = current_time
        
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    try:
        monitor_download()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        download_path = Path("/Users/rohanvinaik/LLM_models/llama-3.1-405b-fp8")
        current_size = get_directory_size(download_path)
        estimated_total_bytes = 405 * 1024 * 1024 * 1024
        percentage = (current_size / estimated_total_bytes) * 100
        print(f"Final size: {format_bytes(current_size)} ({percentage:.1f}%)")