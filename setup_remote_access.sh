#!/bin/bash
# Setup remote access for Mac Pro experiment server

echo "🌐 Configuring Remote Access..."

# 1. Enable SSH with secure settings
sudo systemsetup -setremotelogin on
sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist

# 2. Configure SSH for security
cat > ~/.ssh/config << 'EOF'
# Mac Pro Experiment Server SSH Config
Host macpro-local
    HostName localhost
    User rohanvinaik
    Port 22
    ForwardX11 no
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Enable connection multiplexing for faster subsequent connections
Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
EOF

mkdir -p ~/.ssh/sockets
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config

# 3. Create experiment server API
cat > ~/experiments/server_api.py << 'EOF'
#!/usr/bin/env python3
"""
Simple REST API for Mac Pro Experiment Server
Allows remote experiment submission and monitoring
"""

from flask import Flask, request, jsonify
import subprocess
import json
import os
from datetime import datetime
import threading
import queue

app = Flask(__name__)

# Experiment queue
experiment_queue = queue.Queue()
results_cache = {}

@app.route('/api/experiments/submit', methods=['POST'])
def submit_experiment():
    """Submit a new experiment to the queue"""
    data = request.json
    experiment_id = f"{data['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment = {
        'id': experiment_id,
        'type': data['type'],
        'config': data.get('config', {}),
        'status': 'queued',
        'submitted_at': datetime.now().isoformat()
    }
    
    experiment_queue.put(experiment)
    
    return jsonify({
        'experiment_id': experiment_id,
        'status': 'queued',
        'queue_position': experiment_queue.qsize()
    })

@app.route('/api/experiments/<experiment_id>/status', methods=['GET'])
def get_experiment_status(experiment_id):
    """Get status of a specific experiment"""
    if experiment_id in results_cache:
        return jsonify(results_cache[experiment_id])
    
    # Check if still in queue
    queue_items = list(experiment_queue.queue)
    for i, exp in enumerate(queue_items):
        if exp['id'] == experiment_id:
            return jsonify({
                'id': experiment_id,
                'status': 'queued',
                'queue_position': i + 1
            })
    
    return jsonify({'error': 'Experiment not found'}), 404

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    import psutil
    
    return jsonify({
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'experiments_queued': experiment_queue.qsize(),
        'experiments_completed': len(results_cache)
    })

@app.route('/api/experiments/list', methods=['GET'])
def list_experiments():
    """List all experiments"""
    experiments = []
    
    # Add completed experiments
    experiments.extend(results_cache.values())
    
    # Add queued experiments
    queue_items = list(experiment_queue.queue)
    experiments.extend(queue_items)
    
    return jsonify(experiments)

def experiment_worker():
    """Background worker to process experiments"""
    while True:
        try:
            experiment = experiment_queue.get(timeout=5)
            
            # Update status
            experiment['status'] = 'running'
            experiment['started_at'] = datetime.now().isoformat()
            
            # Run the experiment
            if experiment['type'] == 'hd_computing':
                cmd = ['python3', '~/experiments/genomevault_hd_experiment.py']
            elif experiment['type'] == 'efficiency':
                cmd = ['python3', '~/experiments/run_experiment.py', 'efficiency']
            else:
                cmd = ['python3', '~/experiments/run_experiment.py', experiment['type']]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Store results
            experiment['status'] = 'completed' if result.returncode == 0 else 'failed'
            experiment['completed_at'] = datetime.now().isoformat()
            experiment['output'] = result.stdout
            experiment['error'] = result.stderr
            
            results_cache[experiment['id']] = experiment
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")

# Start background worker
worker_thread = threading.Thread(target=experiment_worker, daemon=True)
worker_thread.start()

if __name__ == '__main__':
    # Run on all interfaces for network access
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# 4. Create Jupyter Lab service for remote notebooks
cat > ~/Library/LaunchAgents/com.macpro.jupyter.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.macpro.jupyter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>jupyter</string>
        <string>lab</string>
        <string>--no-browser</string>
        <string>--port=8888</string>
        <string>--ip=0.0.0.0</string>
        <string>--notebook-dir=/Users/rohanvinaik/experiments</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/rohanvinaik/experiments</string>
    <key>StandardOutPath</key>
    <string>/Users/rohanvinaik/experiments/logs/jupyter.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/rohanvinaik/experiments/logs/jupyter_error.log</string>
</dict>
</plist>
EOF

# 5. Create convenient access script
cat > ~/experiments/connect.sh << 'EOF'
#!/bin/bash
# Connect to Mac Pro Experiment Server

echo "🖥️  Mac Pro Experiment Server Connection Options:"
echo "================================================"
echo
echo "1. SSH Access:"
echo "   ssh rohanvinaik@[YOUR_MAC_IP]"
echo
echo "2. Jupyter Lab:"
echo "   http://[YOUR_MAC_IP]:8888"
echo "   Token: Check ~/experiments/logs/jupyter.log"
echo
echo "3. Experiment API:"
echo "   http://[YOUR_MAC_IP]:5000/api/system/status"
echo
echo "4. Submit experiment via API:"
echo "   curl -X POST http://[YOUR_MAC_IP]:5000/api/experiments/submit \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"type\": \"hd_computing\", \"config\": {}}'"
echo
echo "5. Monitor server:"
echo "   ssh rohanvinaik@[YOUR_MAC_IP] 'python3 ~/experiments/monitor_server.py'"
echo

# Get current IP
IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
echo "Your Mac Pro IP: $IP"
EOF

chmod +x ~/experiments/connect.sh

echo "✅ Remote access configured!"
echo "   Run '~/experiments/connect.sh' for connection info"
