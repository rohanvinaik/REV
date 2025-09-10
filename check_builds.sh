#!/bin/bash

echo "================================================================================================"
echo "REV REFERENCE BUILD STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================================"
echo

# Count running builds
running=$(ps aux | grep "run_rev.py.*--build-reference" | grep -v grep | wc -l)
echo "Active Reference Builds: $running"
echo

echo "Model                  PID      CPU%    MEM%    Runtime    Status"
echo "--------------------------------------------------------------------------------"

# Parse running processes
ps aux | grep "run_rev.py.*--build-reference" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    start_time=$(echo $line | awk '{print $9}')
    
    # Extract model name from command
    model="unknown"
    if [[ $line == *"pythia-70m"* ]]; then
        model="pythia-70m (6L)"
        total_probes=$((6 * 259))
    elif [[ $line == *"distilgpt2"* ]]; then
        model="distilgpt2 (6L)"
        total_probes=$((6 * 257))
    elif [[ $line == *"gpt-neo-125m"* ]]; then
        model="gpt-neo-125m (12L)"
        total_probes=$((12 * 257))
    elif [[ $line == *"llama-2-7b"* ]]; then
        model="llama-2-7b (32L)"
        total_probes=$((32 * 301))
    elif [[ $line == *"mistral"* ]]; then
        model="mistral (32L)"
        total_probes=$((32 * 301))
    elif [[ $line == *"falcon-7b"* ]]; then
        model="falcon-7b (32L)"
        total_probes=$((32 * 301))
    elif [[ $line == *"phi-2"* ]]; then
        model="phi-2 (32L)"
        total_probes=$((32 * 301))
    elif [[ $line == *"pythia-160m"* ]]; then
        model="pythia-160m (12L)"
        total_probes=$((12 * 260))
    elif [[ $line == *"gpt2"* ]]; then
        model="gpt2 (12L)"
        total_probes=$((12 * 260))
    fi
    
    printf "%-22s %-8s %-7s %-7s %-10s Running\n" "$model" "$pid" "${cpu}%" "${mem}%" "$start_time"
done

echo
echo "Estimated Completion Times (based on ~50-100ms per probe):"
echo "--------------------------------------------------------------------------------"
echo "Small models (6-12 layers):  1-3 hours"
echo "Large models (32 layers):    4-8 hours"
echo
echo "Notes:"
echo "• Each probe is applied to EVERY layer"
echo "• Total evaluations = Layers × Probes"
echo "• pythia-70m: 6×259 = 1,554 evaluations"
echo "• distilgpt2: 6×257 = 1,542 evaluations"
echo "• gpt-neo-125m: 12×257 = 3,084 evaluations"
echo "• llama/mistral/falcon/phi: 32×301 = 9,632 evaluations each"
