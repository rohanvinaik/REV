#!/usr/bin/env python3
"""
Unified Reference Library Builder
================================

Ensures ALL model families use the IDENTICAL comprehensive pipeline for reference building.

STANDARDIZED PIPELINE:
- Same orchestration systems (all 7)
- Same comprehensive analysis flags
- Same memory management (36GB)
- Same challenge count per family
- Same behavioral analysis depth
- Same topology extraction method
"""

import subprocess
import sys
import time
from pathlib import Path
import json
from datetime import datetime

# Model family definitions with standardized paths
MODEL_FAMILIES = {
    "pythia": {
        "path": "/Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "family": "pythia",
        "challenges": 100,
        "description": "EleutherAI Pythia 70M - Smallest representative"
    },
    "gpt": {
        "path": "/Users/rohanvinaik/LLM_models/distilgpt2",
        "family": "gpt", 
        "challenges": 100,
        "description": "DistilGPT-2 - OpenAI GPT family representative"
    },
    "llama": {
        "path": "/Users/rohanvinaik/LLM_models/llama-2-7b-hf",
        "family": "llama",
        "challenges": 150,
        "description": "Meta Llama-2-7B - Llama family representative"
    },
    "mistral": {
        "path": "/Users/rohanvinaik/LLM_models/mistral_for_colab",
        "family": "mistral",
        "challenges": 120,
        "description": "Mistral 7B - Mistral AI representative"
    },
    "falcon": {
        "path": "/Users/rohanvinaik/LLM_models/falcon-40b",
        "family": "falcon",
        "challenges": 150,
        "description": "Falcon 40B - Technology Innovation Institute"
    },
    "yi": {
        "path": "/Users/rohanvinaik/LLM_models/yi-34b",
        "family": "yi",
        "challenges": 150,
        "description": "Yi-34B - 01.AI representative"
    },
    "phi": {
        "path": "/Users/rohanvinaik/LLM_models/phi-2",
        "family": "phi",
        "challenges": 100,
        "description": "Microsoft Phi-2 - Small but capable"
    }
}

# STANDARDIZED PIPELINE ARGUMENTS - IDENTICAL FOR ALL FAMILIES
UNIFIED_PIPELINE_ARGS = [
    "--enable-prompt-orchestration",  # All 7 orchestration systems
    "--build-reference",              # Build reference library entry
    "--add-to-library",              # Add to active library
    
    # All orchestration systems enabled
    "--enable-pot",                   # Proof of Thought challenges
    "--enable-kdf",                   # Key Derivation Function adversarial
    "--enable-evolutionary",          # Genetic optimization
    "--enable-dynamic",               # Dynamic synthesis
    "--enable-hierarchical",          # Hierarchical prompt system
    
    # Comprehensive analysis
    "--comprehensive-analysis",       # Deep behavioral analysis
    "--save-analysis-report",         # Save detailed analysis report
    
    # Memory and performance
    "--parallel",                     # Enable parallel processing
    "--parallel-memory-limit", "36.0", # 36GB memory limit
    
    # Debug and logging
    "--debug",                        # Enable debug output
]

def validate_model_path(path: str) -> bool:
    """Validate that model path exists and contains required files."""
    model_path = Path(path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {path}")
        return False
    
    # Check for config.json
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"❌ No config.json found in: {path}")
        return False
    
    print(f"✅ Validated model path: {path}")
    return True

def build_family_reference(family_name: str, config: dict) -> bool:
    """
    Build reference for a single family using the UNIFIED PIPELINE.
    
    This ensures every family gets identical treatment.
    """
    print("=" * 80)
    print(f"BUILDING UNIFIED REFERENCE: {family_name.upper()}")
    print("=" * 80)
    print(f"Model: {config['description']}")
    print(f"Path: {config['path']}")
    print(f"Family: {config['family']}")
    print(f"Challenges: {config['challenges']}")
    print("Pipeline: UNIFIED COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Validate model path
    if not validate_model_path(config['path']):
        return False
    
    # Build standardized command
    cmd = [
        "python", "run_rev.py",
        config['path']  # Model path
    ]
    
    # Add all standardized pipeline arguments
    cmd.extend(UNIFIED_PIPELINE_ARGS)
    
    # Add family-specific arguments
    cmd.extend([
        "--claimed-family", config['family'],
        "--challenges", str(config['challenges']),
        "--output", f"outputs/{family_name}_unified_reference.json"
    ])
    
    # Create log filename
    log_file = f"{family_name}_unified_build.log"
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("Starting build...")
    
    try:
        # Start the process
        start_time = time.time()
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log
            for line in process.stdout:
                print(line.rstrip())
                log.write(line)
                log.flush()
            
            # Wait for completion
            return_code = process.wait()
            
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"✅ SUCCESS: {family_name} reference built in {duration:.1f}s")
            return True
        else:
            print(f"❌ FAILED: {family_name} reference build failed (code: {return_code})")
            return False
            
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted: {family_name} build cancelled")
        return False
    except Exception as e:
        print(f"❌ ERROR: {family_name} build failed: {e}")
        return False

def build_all_references(selected_families=None):
    """
    Build references for all families using the UNIFIED PIPELINE.
    
    Args:
        selected_families: List of family names to build, or None for all
    """
    print("🚀" * 40)
    print("UNIFIED REFERENCE LIBRARY BUILDER")
    print("🚀" * 40)
    print("Ensures ALL families use IDENTICAL comprehensive pipeline")
    print(f"Total families available: {len(MODEL_FAMILIES)}")
    
    if selected_families:
        families_to_build = {k: v for k, v in MODEL_FAMILIES.items() if k in selected_families}
        print(f"Building selected families: {selected_families}")
    else:
        families_to_build = MODEL_FAMILIES
        print("Building ALL families")
    
    print("=" * 80)
    print("UNIFIED PIPELINE CONFIGURATION:")
    print("- Prompt Orchestration: ALL 7 SYSTEMS")
    print("- Analysis: COMPREHENSIVE + BEHAVIORAL")
    print("- Memory Limit: 36GB PARALLEL") 
    print("- Injection Types: ALL 25+ TYPES")
    print("- Topology: FULL EXTRACTION")
    print("- Report: DETAILED ANALYSIS")
    print("=" * 80)
    
    # Track results
    results = {}
    total_start = time.time()
    
    # Build each family
    for family_name, config in families_to_build.items():
        print(f"\n📊 Starting {family_name} family ({len(results)+1}/{len(families_to_build)})")
        
        success = build_family_reference(family_name, config)
        results[family_name] = {
            "success": success,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            print(f"🎉 {family_name} COMPLETED")
        else:
            print(f"💥 {family_name} FAILED")
        
        print("-" * 40)
    
    total_duration = time.time() - total_start
    
    # Summary report
    print("🎯" * 40)
    print("UNIFIED REFERENCE BUILD SUMMARY")
    print("🎯" * 40)
    
    successful = [f for f, r in results.items() if r['success']]
    failed = [f for f, r in results.items() if not r['success']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    for family in successful:
        print(f"   - {family}: {results[family]['config']['description']}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        for family in failed:
            print(f"   - {family}: {results[family]['config']['description']}")
    
    print(f"⏱️ Total time: {total_duration:.1f}s")
    print(f"📝 Log files: {[f'{f}_unified_build.log' for f in results.keys()]}")
    print(f"📊 Output files: {[f'outputs/{f}_unified_reference.json' for f in successful]}")
    
    # Save results
    results_file = f"unified_build_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📋 Results saved: {results_file}")
    
    print("🎯" * 40)
    
    return results

def main():
    """Main entry point with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build unified reference library")
    parser.add_argument("families", nargs="*", 
                       help="Specific families to build (default: all)")
    parser.add_argument("--list-families", action="store_true",
                       help="List available families and exit")
    
    args = parser.parse_args()
    
    if args.list_families:
        print("Available model families:")
        for name, config in MODEL_FAMILIES.items():
            print(f"  {name}: {config['description']}")
            print(f"    Path: {config['path']}")
            print(f"    Challenges: {config['challenges']}")
            print()
        return
    
    # Validate requested families
    if args.families:
        invalid = [f for f in args.families if f not in MODEL_FAMILIES]
        if invalid:
            print(f"❌ Invalid families: {invalid}")
            print(f"Available families: {list(MODEL_FAMILIES.keys())}")
            sys.exit(1)
    
    # Build references
    selected_families = args.families if args.families else None
    results = build_all_references(selected_families)
    
    # Exit with error if any builds failed
    failed_count = sum(1 for r in results.values() if not r['success'])
    sys.exit(failed_count)

if __name__ == "__main__":
    main()