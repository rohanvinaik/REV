#!/usr/bin/env python3
"""
Real-time download progress monitor for LLaMA 3.3 70B-Instruct
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

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

def format_bytes(bytes_val):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def format_speed(bytes_per_sec):
    """Format download speed."""
    return f"{format_bytes(bytes_per_sec)}/s"

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

def draw_progress_bar(percentage, width=50):
    """Draw a progress bar."""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def monitor_download():
    """Monitor download progress in real-time."""
    download_path = Path("/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct")
    
    # Estimated total size for LLaMA 3.3 70B-Instruct (in bytes)
    # 70B parameters × 2 bytes (float16) ≈ 140GB
    estimated_total_bytes = 140 * 1024 * 1024 * 1024  # 140GB
    
    print("=" * 80)
    print("LLaMA 3.3 70B-Instruct Download Monitor")
    print("=" * 80)
    print(f"Target directory: {download_path}")
    print(f"Estimated total size: {format_bytes(estimated_total_bytes)}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    
    start_time = time.time()
    last_size = 0
    last_time = start_time
    sizes_history = []
    
    try:
        while True:
            current_time = time.time()
            current_size = get_directory_size(download_path)
            
            if current_size == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for download to start...")
                time.sleep(5)
                continue
            
            # Calculate progress
            percentage = min((current_size / estimated_total_bytes) * 100, 100)
            
            # Calculate speed (average over last few measurements)
            time_diff = current_time - last_time
            if time_diff > 0 and current_size > last_size:
                current_speed = (current_size - last_size) / time_diff
                sizes_history.append(current_speed)
                
                # Keep only last 10 measurements for smoother average
                if len(sizes_history) > 10:
                    sizes_history.pop(0)
                
                avg_speed = sum(sizes_history) / len(sizes_history)
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
            
            # Clear screen and display progress
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("=" * 80)
            print("LLaMA 3.3 70B-Instruct Download Monitor")
            print("=" * 80)
            print(f"Downloaded: {format_bytes(current_size)} / {format_bytes(estimated_total_bytes)}")
            print(f"Progress: {draw_progress_bar(percentage)}")
            print(f"Speed: {format_speed(avg_speed)}")
            print(f"Elapsed: {elapsed_str}")
            print(f"ETA: {eta_str}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # File count
            try:
                file_count = len(list(download_path.rglob('*'))) if download_path.exists() else 0
                print(f"Files: {file_count}")
            except:
                pass
            
            print("=" * 80)
            print("Press Ctrl+C to stop monitoring")
            
            # Check if download is complete
            if percentage >= 99.9:
                print("\n🎉 Download appears to be complete!")
                print("\nReady to test with REV framework:")
                print("python run_rev_complete.py /Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct \\")
                print("  --quantize 4bit --challenges 2 --max-tokens 50")
                break
            
            last_size = current_size
            last_time = current_time
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")
        print(f"Final size: {format_bytes(current_size)}")
        print(f"Progress: {percentage:.1f}%")
        
        if percentage > 50:  # If more than 50% downloaded
            print(f"\nYou can test partial download with REV (it will use available weights):")
            print("python run_rev_complete.py /Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct \\")
            print("  --quantize 4bit --challenges 1 --max-tokens 20")

def main():
    """Main function."""
    try:
        monitor_download()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())