#!/usr/bin/env python3
"""
Export behavioral topology from REV profiling results.
Analyzes divergence patterns to identify restriction sites and stable regions.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

class TopologyExtractor:
    def __init__(self, log_file: str, output_json: str = None):
        self.log_file = log_file
        self.output_json = output_json
        self.layer_divergences = {}
        self.topology = {
            'model': None,
            'timestamp': datetime.now().isoformat(),
            'total_layers': 80,
            'restriction_sites': [],
            'stable_regions': [],
            'phase_boundaries': [],
            'behavioral_phases': [],
            'layer_profiles': {},
            'optimization_hints': {}
        }
    
    def parse_log(self):
        """Parse log file to extract divergence scores."""
        model_path = None
        
        with open(self.log_file, 'r') as f:
            for line in f:
                # Extract model path
                if 'model_path' in line or 'Model path:' in line:
                    if '/' in line:
                        parts = line.split('/')
                        model_path = '/'.join(parts[-3:]).strip()
                
                # Extract divergence scores
                if 'PROBE SUCCESS' in line:
                    match = re.search(r'Layer\s+(\d+)\s+\|\s+Divergence:\s+([\d.]+)', line)
                    if match:
                        layer = int(match.group(1))
                        divergence = float(match.group(2))
                        if layer not in self.layer_divergences:
                            self.layer_divergences[layer] = []
                        self.layer_divergences[layer].append(divergence)
        
        # Store model info
        if model_path:
            self.topology['model'] = model_path
    
    def identify_restriction_sites(self):
        """Identify layers with significant behavioral changes."""
        layer_avgs = {k: statistics.mean(v) for k, v in self.layer_divergences.items()}
        
        # Find restriction sites (>2% divergence change)
        for i in sorted(layer_avgs.keys()):
            if i > 0 and (i-1) in layer_avgs:
                prev_div = layer_avgs[i-1]
                curr_div = layer_avgs[i]
                change = abs(curr_div - prev_div)
                pct_change = (change / prev_div * 100) if prev_div > 0 else 0
                
                if change > 0.015 or pct_change > 3:  # Significant change
                    self.topology['restriction_sites'].append({
                        'layer': i,
                        'divergence_delta': round(change, 4),
                        'percent_change': round(pct_change, 2),
                        'before': round(prev_div, 4),
                        'after': round(curr_div, 4)
                    })
    
    def identify_stable_regions(self):
        """Identify consecutive layers with minimal variance."""
        layer_avgs = {k: statistics.mean(v) for k, v in self.layer_divergences.items()}
        
        # Find stable regions (consecutive layers with <1% variance)
        stable_threshold = 0.01
        current_region = []
        
        for i in sorted(layer_avgs.keys()):
            if not current_region:
                current_region = [i]
            else:
                # Check if this layer continues the stable region
                region_values = [layer_avgs[l] for l in current_region + [i]]
                if statistics.stdev(region_values) < stable_threshold:
                    current_region.append(i)
                else:
                    # End of stable region
                    if len(current_region) >= 3:  # At least 3 consecutive layers
                        self.topology['stable_regions'].append({
                            'start': current_region[0],
                            'end': current_region[-1],
                            'layers': len(current_region),
                            'avg_divergence': round(statistics.mean([layer_avgs[l] for l in current_region]), 4),
                            'std_dev': round(statistics.stdev([layer_avgs[l] for l in current_region]), 5)
                        })
                    current_region = [i]
        
        # Don't forget the last region
        if len(current_region) >= 3:
            self.topology['stable_regions'].append({
                'start': current_region[0],
                'end': current_region[-1],
                'layers': len(current_region),
                'avg_divergence': round(statistics.mean([layer_avgs[l] for l in current_region]), 4),
                'std_dev': round(statistics.stdev([layer_avgs[l] for l in current_region]), 5)
            })
    
    def identify_behavioral_phases(self):
        """Group layers into behavioral phases based on divergence patterns."""
        layer_avgs = {k: statistics.mean(v) for k, v in self.layer_divergences.items()}
        
        phases = []
        
        # Phase 1: Embedding (layer 0)
        if 0 in layer_avgs:
            phases.append({
                'phase': 'embedding',
                'layers': [0],
                'avg_divergence': round(layer_avgs[0], 4),
                'description': 'Input embedding and tokenization'
            })
        
        # Phase 2: Early processing (rapid divergence increase)
        early_layers = [l for l in range(1, 6) if l in layer_avgs]
        if early_layers:
            phases.append({
                'phase': 'early_processing',
                'layers': early_layers,
                'avg_divergence': round(statistics.mean([layer_avgs[l] for l in early_layers]), 4),
                'description': 'Rapid feature extraction and initial processing'
            })
        
        # Phase 3: Mid-level (stabilizing)
        mid_layers = [l for l in range(6, 20) if l in layer_avgs]
        if mid_layers:
            phases.append({
                'phase': 'mid_level',
                'layers': mid_layers,
                'avg_divergence': round(statistics.mean([layer_avgs[l] for l in mid_layers]), 4),
                'description': 'Stable mid-level representation building'
            })
        
        # Phase 4: Deep processing (if we have data)
        deep_layers = [l for l in range(20, 60) if l in layer_avgs]
        if deep_layers:
            phases.append({
                'phase': 'deep_processing',
                'layers': deep_layers,
                'avg_divergence': round(statistics.mean([layer_avgs[l] for l in deep_layers]), 4),
                'description': 'Complex semantic and reasoning patterns'
            })
        
        # Phase 5: Output preparation (final layers)
        final_layers = [l for l in range(60, 80) if l in layer_avgs]
        if final_layers:
            phases.append({
                'phase': 'output_preparation',
                'layers': final_layers,
                'avg_divergence': round(statistics.mean([layer_avgs[l] for l in final_layers]), 4),
                'description': 'Final representation and output formatting'
            })
        
        self.topology['behavioral_phases'] = phases
    
    def generate_optimization_hints(self):
        """Generate optimization hints for parallel execution."""
        hints = {
            'parallel_safe_regions': [],
            'sequential_required': [],
            'recommended_batch_size': 1,
            'estimated_memory_per_layer_gb': 2.1
        }
        
        # Parallel safe regions are stable regions
        for region in self.topology['stable_regions']:
            hints['parallel_safe_regions'].append({
                'layers': list(range(region['start'], region['end'] + 1)),
                'recommended_workers': min(11, region['layers']),  # Up to 11 parallel
                'reason': f"Stable region with std_dev={region['std_dev']}"
            })
        
        # Sequential required at restriction sites
        for site in self.topology['restriction_sites']:
            hints['sequential_required'].append({
                'layer': site['layer'],
                'reason': f"Restriction site with {site['percent_change']:.1f}% change"
            })
        
        # Calculate optimal batch size based on 36GB memory limit
        hints['recommended_batch_size'] = min(11, int(36 / hints['estimated_memory_per_layer_gb']))
        
        self.topology['optimization_hints'] = hints
    
    def add_layer_profiles(self):
        """Add detailed profile for each analyzed layer."""
        for layer, divergences in self.layer_divergences.items():
            if divergences:
                self.topology['layer_profiles'][layer] = {
                    'mean': round(statistics.mean(divergences), 4),
                    'std': round(statistics.stdev(divergences), 5) if len(divergences) > 1 else 0,
                    'min': round(min(divergences), 4),
                    'max': round(max(divergences), 4),
                    'samples': len(divergences)
                }
    
    def export_topology(self):
        """Export topology to JSON file."""
        # Parse and analyze
        self.parse_log()
        self.identify_restriction_sites()
        self.identify_stable_regions()
        self.identify_behavioral_phases()
        self.generate_optimization_hints()
        self.add_layer_profiles()
        
        # Determine output file
        if not self.output_json:
            base_name = Path(self.log_file).stem
            self.output_json = f"{base_name}_topology.json"
        
        # Save topology
        with open(self.output_json, 'w') as f:
            json.dump(self.topology, f, indent=2)
        
        print(f"✅ Topology exported to: {self.output_json}")
        
        # Print summary
        self.print_summary()
        
        return self.topology
    
    def print_summary(self):
        """Print topology summary."""
        print("\n📊 BEHAVIORAL TOPOLOGY SUMMARY")
        print("=" * 50)
        
        print(f"\nModel: {self.topology.get('model', 'Unknown')}")
        print(f"Layers analyzed: {len(self.topology['layer_profiles'])}/{self.topology['total_layers']}")
        
        print(f"\n🔍 Restriction Sites: {len(self.topology['restriction_sites'])}")
        for site in self.topology['restriction_sites'][:5]:  # Show first 5
            print(f"  Layer {site['layer']}: {site['percent_change']:.1f}% change")
        
        print(f"\n🏔️ Stable Regions: {len(self.topology['stable_regions'])}")
        for region in self.topology['stable_regions']:
            print(f"  Layers {region['start']}-{region['end']}: "
                  f"{region['layers']} layers, σ={region['std_dev']:.5f}")
        
        print(f"\n🎯 Behavioral Phases: {len(self.topology['behavioral_phases'])}")
        for phase in self.topology['behavioral_phases']:
            if phase['layers']:  # Only show phases with data
                print(f"  {phase['phase'].replace('_', ' ').title()}: "
                      f"Layers {phase['layers'][0]}-{phase['layers'][-1]}")
        
        print(f"\n⚡ Optimization Potential:")
        total_parallel = sum(len(r['layers']) for r in self.topology['optimization_hints']['parallel_safe_regions'])
        print(f"  Parallel-safe layers: {total_parallel}")
        print(f"  Recommended batch size: {self.topology['optimization_hints']['recommended_batch_size']}")
        
        if total_parallel > 0:
            speedup = min(self.topology['optimization_hints']['recommended_batch_size'], 11)
            print(f"  Potential speedup: {speedup}x for parallel regions")


def main():
    parser = argparse.ArgumentParser(description='Extract behavioral topology from REV logs')
    parser.add_argument('log_file', help='Path to REV log file')
    parser.add_argument('--output', '-o', help='Output JSON file (default: auto-generated)')
    parser.add_argument('--json', help='Optional JSON results file to merge')
    
    args = parser.parse_args()
    
    extractor = TopologyExtractor(args.log_file, args.output)
    topology = extractor.export_topology()
    
    # If JSON results provided, merge them
    if args.json and Path(args.json).exists():
        with open(args.json, 'r') as f:
            results = json.load(f)
            results['topology'] = topology
        
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Topology merged into: {args.json}")


if __name__ == "__main__":
    main()