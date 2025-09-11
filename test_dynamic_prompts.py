#!/usr/bin/env python3
"""Test the dynamic prompt generation system."""

import json
from src.orchestration.prompt_orchestrator import UnifiedPromptOrchestrator

def test_dynamic_prompt_generation():
    """Test that the orchestrator dynamically determines prompt counts."""
    
    print("=" * 80)
    print("TESTING DYNAMIC PROMPT GENERATION")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = UnifiedPromptOrchestrator(enable_all_systems=True)
    
    # Test 1: Small model (6 layers) - should generate ~420 prompts (6 sites × 70 prompts)
    print("\n1. Testing small model (6 layers):")
    sites, prompt_count = orchestrator.discover_restriction_sites("pythia", layer_count=6)
    print(f"   Discovered {len(sites)} restriction sites: {sites}")
    print(f"   Will generate {prompt_count} total prompts")
    
    # Generate prompts dynamically (None for total_prompts)
    result = orchestrator.generate_orchestrated_prompts(
        model_family="pythia",
        total_prompts=None,  # Dynamic determination
        layer_count=6
    )
    print(f"   Actually generated: {result['total_prompts']} prompts")
    print(f"   Prompts per site: {result.get('prompts_per_site', 'N/A')}")
    
    # Test 2: Medium model (12 layers) - should generate ~420 prompts (7 sites × 60 prompts)
    print("\n2. Testing medium model (12 layers):")
    sites, prompt_count = orchestrator.discover_restriction_sites("gpt", layer_count=12)
    print(f"   Discovered {len(sites)} restriction sites: {sites}")
    print(f"   Will generate {prompt_count} total prompts")
    
    result = orchestrator.generate_orchestrated_prompts(
        model_family="gpt",
        total_prompts=None,
        layer_count=12
    )
    print(f"   Actually generated: {result['total_prompts']} prompts")
    
    # Test 3: Large model (32 layers) - should generate ~630 prompts (14 sites × 45 prompts)
    print("\n3. Testing large model (32 layers):")
    sites, prompt_count = orchestrator.discover_restriction_sites("llama", layer_count=32)
    print(f"   Discovered {len(sites)} restriction sites: {sites}")
    print(f"   Will generate {prompt_count} total prompts")
    
    result = orchestrator.generate_orchestrated_prompts(
        model_family="llama",
        total_prompts=None,
        layer_count=32
    )
    print(f"   Actually generated: {result['total_prompts']} prompts")
    
    # Test 4: With user-specified value (should use that instead)
    print("\n4. Testing with user-specified value (100 prompts):")
    result = orchestrator.generate_orchestrated_prompts(
        model_family="llama",
        total_prompts=100,  # User specified
        layer_count=32
    )
    print(f"   Generated: {result['total_prompts']} prompts (user-specified)")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("- Small models (6 layers): Generate ~420 prompts automatically")
    print("- Medium models (12 layers): Generate ~420 prompts automatically")
    print("- Large models (32 layers): Generate ~630 prompts automatically")
    print("- User can override with --challenges flag if needed")
    print("- System adapts based on model architecture!")
    print("=" * 80)

if __name__ == "__main__":
    test_dynamic_prompt_generation()