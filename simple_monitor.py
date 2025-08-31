#!/usr/bin/env python3
"""
Simple download progress monitor - no screen clearing
"""

import os
import time
from pathlib import Path
from datetime import datetime

def get_directory_size(path):
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

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def main():
    download_path = Path("/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
    estimated_total = 140 * 1024 * 1024 * 1024  # 140GB
    
    print("LLaMA 3.3 70B-Instruct Download Monitor")
    print("=" * 50)
    
    last_size = 0
    last_time = time.time()
    
    while True:
        current_time = time.time()
        current_size = get_directory_size(download_path)
        
        if current_size == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')} - Waiting for download to start...")
            time.sleep(10)
            continue
        
        percentage = (current_size / estimated_total) * 100
        
        # Calculate speed
        time_diff = current_time - last_time
        if time_diff > 0 and current_size > last_size:
            speed = (current_size - last_size) / time_diff
            speed_str = f"{format_bytes(speed)}/s"
        else:
            speed_str = "0 B/s"
        
        print(f"{datetime.now().strftime('%H:%M:%S')} - {format_bytes(current_size)} / {format_bytes(estimated_total)} ({percentage:.1f}%) - {speed_str}")
        
        if percentage >= 99:
            print("🎉 Download complete!")
            break
        
        last_size = current_size
        last_time = current_time
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    main()