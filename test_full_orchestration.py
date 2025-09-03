#!/usr/bin/env python3
"""
Test Full Prompt Orchestration Integration
===========================================

This script tests that all prompt systems work together when enabled.
"""

import subprocess
import sys
import time

def test_orchestration():
    """Test the full orchestration with all systems enabled."""
    
    print("=" * 80)
    print("TESTING FULL PROMPT ORCHESTRATION")
    print("=" * 80)
    
    # Test command with all systems enabled
    cmd = [
        "python", "run_rev.py",
        "gpt-2",  # Use a simple model for testing
        "--enable-prompt-orchestration",  # Enable ALL systems
        "--enable-kdf",
        "--enable-evolutionary", 
        "--enable-dynamic",
        "--enable-hierarchical",
        "--prompt-analytics",
        "--challenges", "20",  # Generate 20 prompts
        "--debug",
        "--output", "test_orchestration.json"
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("\nThis will:")
    print("1. Initialize ALL prompt generation systems")
    print("2. Use the unified orchestrator to coordinate them")
    print("3. Generate prompts using:")
    print("   - PoT challenges (behavioral analysis)")
    print("   - KDF adversarial prompts (security)")
    print("   - Evolutionary optimization (discrimination)")
    print("   - Dynamic synthesis (adaptation)")
    print("   - Hierarchical taxonomy (structure)")
    print("4. Apply reference library guidance")
    print("5. Track analytics for effectiveness")
    
    print("\n" + "=" * 80)
    print("Starting test...")
    print("=" * 80 + "\n")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Check output
        if result.returncode == 0:
            print("✅ Command executed successfully!")
            
            # Check for key indicators in output
            output = result.stdout + result.stderr
            
            indicators = {
                "Unified Prompt Orchestrator": "Orchestrator initialized",
                "Generated orchestrated challenges": "Orchestrated generation working",
                "PoT Challenge Generator": "PoT system active",
                "KDF Adversarial Generator": "KDF system active",
                "Genetic Prompt Optimizer": "Evolutionary system active",
                "Dynamic Synthesis System": "Dynamic synthesis active",
                "Hierarchical Prompt System": "Hierarchical system active",
                "systems": "Multiple systems initialized"
            }
            
            print("\n📊 System Status:")
            for indicator, description in indicators.items():
                if indicator in output:
                    print(f"   ✅ {description}")
                else:
                    print(f"   ⚠️  {description} (not detected)")
            
            # Count initialized systems
            if "Initialized Unified Prompt Orchestrator with" in output:
                # Extract the number
                import re
                match = re.search(r"with (\d+) systems", output)
                if match:
                    num_systems = int(match.group(1))
                    print(f"\n🎯 Total Systems Initialized: {num_systems}/7")
                    if num_systems >= 5:
                        print("   ✅ Most systems are working!")
                    elif num_systems >= 3:
                        print("   ⚠️  Some systems may have missing dependencies")
                    else:
                        print("   ❌ Too few systems initialized")
        else:
            print(f"❌ Command failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr[:500]}")
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Error running command: {e}")
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 80)
    
    print("\n📋 SUMMARY:")
    print("The unified prompt orchestration system coordinates multiple generators:")
    print("• Each system contributes specialized prompts")
    print("• Reference library guides targeting")
    print("• Deep analysis enables precision")
    print("• Result: Comprehensive behavioral fingerprinting")

if __name__ == "__main__":
    print("🔬 REV Full Orchestration Test")
    print("Testing all prompt generation systems working together")
    print()
    
    test_orchestration()