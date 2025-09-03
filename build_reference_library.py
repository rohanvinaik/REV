#!/usr/bin/env python3
"""
Build Reference Library for REV System

This script builds deep behavioral references for model families, enabling
much faster verification of larger models within the same family.

Strategy:
1. Use smallest model in each family for reference generation
2. Run with full prompt orchestration (all 7 systems)
3. Generate deep behavioral topology with restriction sites
4. Save to reference library for future use
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Model families and their representatives
MODEL_FAMILIES = {
    "pythia": {
        "reference_model": "/Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-160m/snapshots/50f5173d932e8e61f858120bcb800b97af589f46"
        ],
        "challenges": 50,
        "priority": 1
    },
    "gpt": {
        "reference_model": "/Users/rohanvinaik/LLM_models/distilgpt2",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/gpt2",
            "/Users/rohanvinaik/LLM_models/gpt2-medium",
            "/Users/rohanvinaik/LLM_models/gpt-neo-125m",
            "/Users/rohanvinaik/LLM_models/gpt-neo-1.3b"
        ],
        "challenges": 50,
        "priority": 2
    },
    "llama": {
        "reference_model": "/Users/rohanvinaik/LLM_models/llama-2-7b-hf",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/llama-2-7b-chat-hf",
            "/Users/rohanvinaik/LLM_models/vicuna-7b-v1.5",
            "/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct",
            "/Users/rohanvinaik/LLM_models/llama-3.1-405b-fp8"
        ],
        "challenges": 100,
        "priority": 3
    },
    "falcon": {
        "reference_model": "/Users/rohanvinaik/LLM_models/falcon-40b",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/falcon-40b-instruct"
        ],
        "challenges": 75,
        "priority": 4
    },
    "yi": {
        "reference_model": "/Users/rohanvinaik/LLM_models/yi-34b",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/yi-34b-chat"
        ],
        "challenges": 75,
        "priority": 5
    },
    "mistral": {
        "reference_model": "/Users/rohanvinaik/LLM_models/mistral_for_colab",
        "family_models": [
            "/Users/rohanvinaik/LLM_models/zephyr-7b-beta-final"
        ],
        "challenges": 50,
        "priority": 6
    },
    "phi": {
        "reference_model": "/Users/rohanvinaik/LLM_models/phi-2",
        "family_models": [],
        "challenges": 40,
        "priority": 7
    }
}

class ReferenceLibraryBuilder:
    """Build and manage reference library for model families."""
    
    def __init__(self, output_dir: str = "fingerprint_library"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.reference_file = self.output_dir / "reference_library.json"
        self.backup_file = self.output_dir / f"reference_library_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.results = []
        self.start_time = None
        
    def load_existing_library(self) -> Dict:
        """Load existing reference library if it exists."""
        if self.reference_file.exists():
            with open(self.reference_file, 'r') as f:
                return json.load(f)
        return {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "description": "Deep behavioral reference library for model families",
            "families": {},
            "fingerprints": {},
            "metadata": {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def backup_library(self):
        """Backup existing library before modifications."""
        if self.reference_file.exists():
            import shutil
            shutil.copy2(self.reference_file, self.backup_file)
            print(f"✅ Backed up existing library to: {self.backup_file}")
    
    def check_model_exists(self, model_path: str) -> bool:
        """Check if model directory exists and contains config.json."""
        path = Path(model_path)
        if not path.exists():
            return False
        
        config_file = path / "config.json"
        return config_file.exists()
    
    def estimate_time(self, family: str, config: Dict) -> str:
        """Estimate processing time for a model family."""
        base_time = 0.5  # 30 minutes base
        
        # Adjust based on model size and challenges
        if "7b" in config["reference_model"].lower():
            base_time *= 2
        elif "34b" in config["reference_model"].lower():
            base_time *= 4
        elif "40b" in config["reference_model"].lower():
            base_time *= 5
        elif "70b" in config["reference_model"].lower():
            base_time *= 8
        elif "405b" in config["reference_model"].lower():
            base_time *= 20
        
        # Adjust for number of challenges
        challenge_factor = config["challenges"] / 50
        total_hours = base_time * challenge_factor
        
        if total_hours < 1:
            return f"{int(total_hours * 60)} minutes"
        else:
            return f"{total_hours:.1f} hours"
    
    def build_reference_for_family(self, family: str, config: Dict, force: bool = False) -> bool:
        """Build reference for a single model family."""
        print(f"\n{'='*80}")
        print(f"Building reference for {family.upper()} family")
        print(f"{'='*80}")
        
        reference_model = config["reference_model"]
        
        # Check if model exists
        if not self.check_model_exists(reference_model):
            print(f"❌ Reference model not found: {reference_model}")
            return False
        
        print(f"Reference model: {Path(reference_model).name}")
        print(f"Challenges: {config['challenges']}")
        print(f"Estimated time: {self.estimate_time(family, config)}")
        
        # Build command
        cmd = [
            "python", "run_rev.py",
            reference_model,
            "--enable-prompt-orchestration",
            "--challenges", str(config["challenges"]),
            "--build-reference",
            "--claimed-family", family,
            "--add-to-library",
            "--debug"
        ]
        
        # Add all orchestration systems
        cmd.extend([
            "--enable-pot",
            "--enable-kdf", 
            "--enable-evolutionary",
            "--enable-dynamic",
            "--enable-hierarchical"
        ])
        
        # Add parallel processing for faster execution
        if config["challenges"] > 50:
            cmd.extend([
                "--parallel",
                "--parallel-memory-limit", "36.0",
                "--parallel-batch-size", "5"
            ])
        
        print(f"\nCommand: {' '.join(cmd)}")
        
        # Run the command
        start_time = time.time()
        try:
            print("\n🚀 Starting reference generation...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            print(f"✅ Reference built successfully in {duration/60:.1f} minutes")
            
            # Save result
            self.results.append({
                "family": family,
                "model": reference_model,
                "success": True,
                "duration": duration,
                "challenges": config["challenges"]
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"❌ Failed to build reference: {e}")
            if e.stdout:
                print("Output:", e.stdout[-500:])  # Last 500 chars
            if e.stderr:
                print("Error:", e.stderr[-500:])
            
            self.results.append({
                "family": family,
                "model": reference_model,
                "success": False,
                "duration": duration,
                "error": str(e)
            })
            
            return False
    
    def validate_references(self) -> Dict:
        """Validate that references were properly created."""
        library = self.load_existing_library()
        validation = {
            "families_with_references": [],
            "families_missing": [],
            "total_fingerprints": len(library.get("fingerprints", {})),
            "validation_passed": True
        }
        
        for family in MODEL_FAMILIES.keys():
            family_data = library.get("families", {}).get(family)
            if family_data and family_data.get("reference_fingerprint"):
                validation["families_with_references"].append(family)
            else:
                # Check if any fingerprints exist for this family
                family_fingerprints = [
                    fp for fp_id, fp in library.get("fingerprints", {}).items()
                    if fp.get("model_family") == family
                ]
                if family_fingerprints:
                    validation["families_with_references"].append(family)
                else:
                    validation["families_missing"].append(family)
                    validation["validation_passed"] = False
        
        return validation
    
    def quick_reference_build(self, families: List[str] = None):
        """Quick build for specific families or smallest models only."""
        if families is None:
            # Use only smallest, fastest models
            families = ["pythia", "gpt", "mistral", "phi"]
        
        print("\n🚀 QUICK REFERENCE BUILD MODE")
        print(f"Building references for: {', '.join(families)}")
        
        for family in families:
            if family in MODEL_FAMILIES:
                config = MODEL_FAMILIES[family].copy()
                # Reduce challenges for quick build
                config["challenges"] = min(20, config["challenges"])
                self.build_reference_for_family(family, config)
    
    def full_reference_build(self, skip_existing: bool = True):
        """Build references for all model families."""
        print("\n🚀 FULL REFERENCE LIBRARY BUILD")
        print(f"Families to process: {len(MODEL_FAMILIES)}")
        
        # Sort by priority
        sorted_families = sorted(
            MODEL_FAMILIES.items(),
            key=lambda x: x[1]["priority"]
        )
        
        # Calculate total estimated time
        total_time = sum(
            float(self.estimate_time(family, config).split()[0])
            for family, config in sorted_families
        )
        
        print(f"Estimated total time: {total_time:.1f} hours")
        
        if skip_existing:
            library = self.load_existing_library()
            existing = [
                family for family, _ in sorted_families
                if family in library.get("families", {})
            ]
            if existing:
                print(f"Skipping existing families: {', '.join(existing)}")
                sorted_families = [
                    (f, c) for f, c in sorted_families
                    if f not in existing
                ]
        
        # Build references
        for family, config in sorted_families:
            success = self.build_reference_for_family(family, config)
            if not success:
                print(f"⚠️  Failed to build reference for {family}, continuing...")
    
    def generate_report(self):
        """Generate summary report of reference building."""
        report_file = self.output_dir / f"reference_build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        validation = self.validate_references()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time if self.start_time else 0,
            "results": self.results,
            "validation": validation,
            "summary": {
                "total_families": len(MODEL_FAMILIES),
                "successful_builds": len([r for r in self.results if r["success"]]),
                "failed_builds": len([r for r in self.results if not r["success"]]),
                "families_with_references": len(validation["families_with_references"]),
                "families_missing": validation["families_missing"]
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("REFERENCE LIBRARY BUILD SUMMARY")
        print("="*80)
        print(f"Total families: {report['summary']['total_families']}")
        print(f"Successful builds: {report['summary']['successful_builds']}")
        print(f"Failed builds: {report['summary']['failed_builds']}")
        print(f"Families with references: {report['summary']['families_with_references']}")
        
        if validation["families_missing"]:
            print(f"\n⚠️  Missing families: {', '.join(validation['families_missing'])}")
        
        if validation["validation_passed"]:
            print("\n✅ Reference library validation PASSED")
        else:
            print("\n❌ Reference library validation FAILED")
        
        print(f"\nReport saved to: {report_file}")
        
        return report


def main():
    """Main function for building reference library."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build reference library for REV system"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "specific"],
        default="full",
        help="Build mode: quick (fast models only), full (all families), specific (specify families)"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        help="Specific families to build (for specific mode)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip families that already have references"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if references exist"
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = ReferenceLibraryBuilder()
    builder.start_time = time.time()
    
    # Backup existing library
    builder.backup_library()
    
    # Print available families
    print("\n📚 Available Model Families:")
    for family, config in MODEL_FAMILIES.items():
        model_name = Path(config["reference_model"]).name
        exists = "✅" if builder.check_model_exists(config["reference_model"]) else "❌"
        print(f"  {exists} {family:10} - {model_name:30} ({config['challenges']} challenges)")
    
    # Run based on mode
    if args.mode == "quick":
        builder.quick_reference_build()
    elif args.mode == "specific":
        if not args.families:
            print("❌ Please specify families with --families flag")
            return
        builder.quick_reference_build(args.families)
    else:  # full
        builder.full_reference_build(skip_existing=not args.force)
    
    # Generate report
    builder.generate_report()
    
    print("\n🎉 Reference library build complete!")
    print("\nNext steps:")
    print("1. Review the reference library at: fingerprint_library/reference_library.json")
    print("2. Test with large models that can now use these references:")
    print("   python run_rev.py /path/to/large_model --challenges 100")
    print("3. Monitor speedup - should see 15-20x improvement for same-family models")


if __name__ == "__main__":
    main()