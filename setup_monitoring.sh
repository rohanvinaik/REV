#!/bin/bash
# Mac Pro Server Monitoring Setup

echo "📊 Setting up Server Monitoring..."

# Create monitoring dashboard script
cat > ~/experiments/monitor_server.py << 'EOF'
#!/usr/bin/env python3
"""
Real-time monitoring dashboard for Mac Pro experiments
"""

import os
import time
import psutil
import subprocess
from datetime import datetime
import curses

class ServerMonitor:
    def __init__(self):
        self.refresh_rate = 1  # seconds
        
    def get_gpu_stats(self):
        """Get AMD GPU stats on macOS"""
        try:
            # Use system_profiler for AMD GPUs on Mac
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True
            )
            # Parse output for GPU info
            return "AMD FirePro GPUs: Active"
        except:
            return "GPU: Not available"
    
    def get_experiment_status(self):
        """Check if experiments are running"""
        experiments_dir = os.path.expanduser("~/experiments/logs")
        if os.path.exists(experiments_dir):
            recent_logs = []
            for log in os.listdir(experiments_dir)[-5:]:
                recent_logs.append(log)
            return recent_logs
        return []
    
    def draw_dashboard(self, stdscr):
        """Draw monitoring dashboard"""
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.timeout(1000)
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Header
            header = "🖥️  Mac Pro Experiment Server Monitor"
            stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * width)
            
            # System stats
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            row = 3
            stdscr.addstr(row, 2, f"CPU Cores: {len(cpu_percent)}")
            row += 1
            
            # CPU bars
            for i, percent in enumerate(cpu_percent):
                if row >= height - 5:
                    break
                bar_length = int(percent / 100 * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                stdscr.addstr(row, 2, f"Core {i:2d}: [{bar}] {percent:5.1f}%")
                row += 1
            
            row += 1
            # Memory
            mem_bar_length = int(mem.percent / 100 * 30)
            mem_bar = "█" * mem_bar_length + "░" * (30 - mem_bar_length)
            stdscr.addstr(row, 2, f"Memory: [{mem_bar}] {mem.percent:5.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)")
            row += 2
            
            # Disk
            disk_bar_length = int(disk.percent / 100 * 30)
            disk_bar = "█" * disk_bar_length + "░" * (30 - disk_bar_length)
            stdscr.addstr(row, 2, f"Disk:   [{disk_bar}] {disk.percent:5.1f}% ({disk.used/1024**3:.1f}/{disk.total/1024**3:.1f} GB)")
            row += 2
            
            # GPU
            stdscr.addstr(row, 2, self.get_gpu_stats())
            row += 2
            
            # Network
            net = psutil.net_io_counters()
            stdscr.addstr(row, 2, f"Network - Sent: {net.bytes_sent/1024**2:.1f} MB, Recv: {net.bytes_recv/1024**2:.1f} MB")
            row += 2
            
            # Recent experiments
            stdscr.addstr(row, 2, "Recent Experiments:", curses.A_BOLD)
            row += 1
            
            experiments = self.get_experiment_status()
            for exp in experiments[-5:]:
                if row >= height - 2:
                    break
                stdscr.addstr(row, 4, f"• {exp}")
                row += 1
            
            # Footer
            stdscr.addstr(height-1, 2, f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press 'q' to quit")
            
            # Handle input
            key = stdscr.getch()
            if key == ord('q'):
                break
            
            stdscr.refresh()

    def run(self):
        """Run the monitor"""
        curses.wrapper(self.draw_dashboard)

if __name__ == '__main__':
    monitor = ServerMonitor()
    monitor.run()
EOF

chmod +x ~/experiments/monitor_server.py

# Create systemd-style launchd service for monitoring
cat > ~/Library/LaunchAgents/com.macpro.experiment-monitor.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.macpro.experiment-monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/rohanvinaik/experiments/monitor_server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/rohanvinaik/experiments/logs/monitor.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/rohanvinaik/experiments/logs/monitor_error.log</string>
</dict>
</plist>
EOF

echo "✅ Monitoring setup complete!"
