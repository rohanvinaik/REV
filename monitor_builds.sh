#!/bin/bash

echo "========================================="
echo "Reference Library Build Progress Monitor"
echo "========================================="
echo

while true; do
    clear
    echo "Reference Build Status - $(date '+%H:%M:%S')"
    echo "========================================="
    
    for log in pythia_70m_reference.log distilgpt2_reference.log gpt_neo_125m_reference.log; do
        if [ -f "$log" ]; then
            model=$(echo $log | cut -d_ -f1-2 | tr '_' '-')
            layers=$(grep -oE "across [0-9]+ layers" "$log" 2>/dev/null | head -1 | grep -oE "[0-9]+")
            challenges=$(grep -oE "Using [0-9]+ PoT challenges" "$log" 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1)
            probes_done=$(grep -c "PROBE SUCCESS" "$log" 2>/dev/null || echo 0)
            
            if [ -n "$layers" ] && [ -n "$challenges" ]; then
                total_expected=$((layers * challenges))
                percent=$((probes_done * 100 / total_expected))
                current_layer=$((probes_done / challenges))
                
                echo "Model: $model"
                echo "  Layers: $layers | Challenges: $challenges"
                echo "  Progress: $probes_done / $total_expected ($percent%)"
                echo "  Current Layer: $current_layer / $layers"
                
                # Check if completed
                if grep -q "Successfully processed" "$log" 2>/dev/null; then
                    echo "  Status: ✅ COMPLETE"
                elif ps aux | grep -q "[p]ython run_rev.py.*$(echo $log | cut -d_ -f1)"; then
                    echo "  Status: 🔄 RUNNING"
                else
                    echo "  Status: ❌ STOPPED"
                fi
            else
                echo "Model: $model - Initializing..."
            fi
            echo
        fi
    done
    
    # Check if all builds are complete
    if ! ps aux | grep -q "[p]ython run_rev.py.*build-reference"; then
        echo "All builds completed!"
        break
    fi
    
    sleep 10
done

echo
echo "Final Status:"
echo "============="
ls -lh fingerprint_library/reference_library.json 2>/dev/null || echo "Reference library not found"