#!/usr/bin/env python3
"""
REV Reference Library Enhancement
=================================

This script implements the missing architectural deep analysis for small models
to build complete reference libraries with restriction sites, behavioral phases,
and optimization hints for precision targeting of large models.

MISSING FUNCTIONALITY DISCOVERED:
1. Small models (7B-34B) need FULL deep behavioral analysis
2. Reference library should store architectural topology, not just fingerprints  
3. Large models should use topology for precision targeting (1-2h vs 37h)

SOLUTION ARCHITECTURE:
1. Deep Analysis Pipeline: Extract restriction sites from small models
2. Enhanced Reference Library: Store topological insights per family
3. Precision Targeting: Use topology to analyze only critical layers in large models
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ReferenceLibraryEnhancer:
    def __init__(self, library_path: str = "fingerprint_library"):
        self.library_path = Path(library_path)
        self.active_library_path = self.library_path / "active_library.json"
        self.reference_library_path = self.library_path / "reference_library.json"
        self.topology_library_path = self.library_path / "topology_library.json"
        
    def analyze_current_gaps(self) -> Dict[str, Any]:
        """Analyze what's missing from current reference library"""
        
        with open(self.active_library_path, 'r') as f:
            active_lib = json.load(f)
        
        gaps = {
            "models_with_topology": [],
            "models_missing_topology": [],
            "families_covered": set(),
            "families_needing_deep_analysis": set()
        }
        
        for fingerprint_id, data in active_lib['fingerprints'].items():
            family = data.get('model_family')
            if family:
                gaps['families_covered'].add(family)
                
            # Check if has deep architectural data
            has_topology = all(key in data for key in [
                'vulnerable_layers', 'stable_layers', 'restriction_sites'
            ])
            
            if has_topology:
                gaps['models_with_topology'].append({
                    'id': fingerprint_id,
                    'family': family,
                    'model_path': data.get('model_path', 'unknown')
                })
            else:
                gaps['models_missing_topology'].append({
                    'id': fingerprint_id, 
                    'family': family,
                    'model_path': data.get('model_path', 'unknown')
                })
                if family:
                    gaps['families_needing_deep_analysis'].add(family)
        
        return gaps
    
    def generate_deep_analysis_plan(self, gaps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan for deep analysis of small models to build topology library"""
        
        plan = []
        
        # Priority order: smallest models first for each family
        family_priorities = {
            'gpt': ['gpt2', 'distilgpt2', 'gpt-neo-125m'],
            'llama': ['llama-2-7b-hf', 'llama-2-7b-chat-hf'],  
            'mistral': ['mistral-7b'],
            'yi': ['yi-34b'],  # Larger but manageable
            'dialogpt': ['microsoft--DialoGPT-small'],
            'pythia': ['EleutherAI--pythia-160m']
        }
        
        for family in gaps['families_needing_deep_analysis']:
            # Find smallest model in this family that needs deep analysis
            family_models = [
                model for model in gaps['models_missing_topology'] 
                if model['family'] == family
            ]
            
            if family_models:
                # Pick the first (assumed smallest) model for deep analysis
                target_model = family_models[0]
                
                plan.append({
                    'family': family,
                    'model_id': target_model['id'],
                    'model_path': target_model['model_path'],
                    'analysis_type': 'deep_behavioral_profiling',
                    'expected_duration_hours': 24 if '7b' in target_model['model_path'].lower() else 48,
                    'priority': 'high' if family in ['llama', 'gpt'] else 'medium',
                    'command': self._generate_deep_analysis_command(target_model)
                })
        
        # Sort by priority and model size
        plan.sort(key=lambda x: (x['priority'] != 'high', x['expected_duration_hours']))
        
        return plan
    
    def _generate_deep_analysis_command(self, model_info: Dict[str, Any]) -> str:
        """Generate the command needed for deep behavioral analysis"""
        
        model_path = model_info['model_path']
        family = model_info['family']
        
        # Use local deep analysis (not API mode) for topology extraction
        cmd = f"""python run_rev.py "{model_path}" \\
    --local \\
    --memory-limit 20 \\
    --challenges 50 \\
    --deep-behavioral-analysis \\
    --extract-topology \\
    --export-topology outputs/{family}_topology.json \\
    --debug \\
    --output outputs/{family}_deep_analysis.json"""
        
        return cmd
    
    def create_topology_library_schema(self) -> Dict[str, Any]:
        """Create the enhanced topology library schema"""
        
        schema = {
            "topology_library": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "description": "Deep architectural topology for model families",
                "families": {}
            }
        }
        
        # Add sample schema based on 70B insights
        sample_topology = {
            "family": "llama",
            "reference_model": "llama-2-7b-hf", 
            "analysis_duration_hours": 24,
            "total_layers": 32,
            "restriction_sites": [
                {
                    "layer": 1,
                    "divergence_delta": 0.1038,
                    "percent_change": 32.75,
                    "significance": "major_behavioral_shift",
                    "attack_vector_risk": "high"
                }
            ],
            "stable_regions": [
                {
                    "start": 4,
                    "end": 16, 
                    "layers": 13,
                    "std_dev": 0.0063,
                    "parallel_safe": True,
                    "recommended_workers": 11
                }
            ],
            "behavioral_phases": [
                {
                    "phase": "embedding",
                    "layers": [0],
                    "avg_divergence": 0.3167,
                    "description": "Input tokenization and embedding"
                }
            ],
            "optimization_hints": {
                "parallel_speedup_potential": "11x",
                "memory_per_layer_gb": 2.1,
                "critical_layers_only": [1, 2, 3, 4],
                "skip_stable_region": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            },
            "precision_targeting": {
                "large_model_strategy": "target_restriction_sites_only",
                "expected_speedup": "15-20x",
                "accuracy_retention": "95%+"
            }
        }
        
        schema["topology_library"]["sample_schema"] = sample_topology
        return schema
    
    def print_enhancement_report(self):
        """Print comprehensive report of what needs to be done"""
        
        print("🔬 REV REFERENCE LIBRARY ENHANCEMENT ANALYSIS")
        print("=" * 60)
        
        gaps = self.analyze_current_gaps()
        plan = self.generate_deep_analysis_plan(gaps)
        
        print(f"\n📊 CURRENT STATUS:")
        print(f"   Families covered: {len(gaps['families_covered'])}")
        print(f"   Models with topology: {len(gaps['models_with_topology'])}")
        print(f"   Models missing topology: {len(gaps['models_missing_topology'])}")
        
        print(f"\n❌ FAMILIES NEEDING DEEP ANALYSIS:")
        for family in gaps['families_needing_deep_analysis']:
            family_models = [m for m in gaps['models_missing_topology'] if m['family'] == family]
            print(f"   {family}: {len(family_models)} models need topology extraction")
        
        print(f"\n🎯 DEEP ANALYSIS EXECUTION PLAN:")
        total_hours = sum(task['expected_duration_hours'] for task in plan)
        print(f"   Total models to analyze: {len(plan)}")
        print(f"   Estimated total time: {total_hours} hours")
        print(f"   High priority families: {len([p for p in plan if p['priority'] == 'high'])}")
        
        print(f"\n📋 EXECUTION SEQUENCE:")
        for i, task in enumerate(plan, 1):
            print(f"   {i}. {task['family'].upper()}: {task['expected_duration_hours']}h")
            print(f"      Model: {task['model_path']}")
            print(f"      Priority: {task['priority']}")
            print()
        
        print(f"🚀 EXPECTED OUTCOME:")
        print(f"   After completion: 405B models analyzed in 1-2h (vs current 37h)")
        print(f"   Precision targeting: 95%+ accuracy with 15-20x speedup")
        print(f"   Complete architectural topology for {len(gaps['families_needing_deep_analysis'])} families")
        
        return plan

def main():
    enhancer = ReferenceLibraryEnhancer()
    plan = enhancer.print_enhancement_report()
    
    # Create topology library schema
    schema = enhancer.create_topology_library_schema()
    
    # Save the enhancement plan
    with open("outputs/reference_library_enhancement_plan.json", 'w') as f:
        json.dump({
            "enhancement_plan": plan,
            "topology_schema": schema,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Enhancement plan saved to: outputs/reference_library_enhancement_plan.json")
    
    # Suggest immediate actions
    print(f"\n⚡ IMMEDIATE ACTIONS:")
    if plan:
        first_task = plan[0]
        print(f"1. Start with {first_task['family']} family:")
        print(f"   {first_task['command']}")
        print()
        print(f"2. While that runs, prepare topology library structure")
        print(f"3. Once complete, integrate topology into reference library")
        print(f"4. Update large model analysis to use precision targeting")

if __name__ == "__main__":
    main()