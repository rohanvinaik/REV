#!/usr/bin/env python3
"""Test script to diagnose and fix reference library building."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.prompt_orchestrator import UnifiedPromptOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_orchestrator():
    """Test the orchestrator prompt generation."""
    orchestrator = UnifiedPromptOrchestrator(enable_all_systems=True)
    
    # Generate prompts for pythia model
    result = orchestrator.generate_orchestrated_prompts(
        model_family="pythia",
        layer_count=6,
        total_prompts=400
    )
    
    logger.info(f"Orchestrator result keys: {result.keys()}")
    logger.info(f"Total prompts generated: {result.get('total_prompts', 0)}")
    
    # Check prompts by type
    prompts_by_type = result.get('prompts_by_type', {})
    for category, prompts in prompts_by_type.items():
        logger.info(f"Category {category}: {len(prompts)} prompts")
        
        # Check format of first prompt
        if prompts and len(prompts) > 0:
            first = prompts[0]
            logger.info(f"  First prompt type: {type(first)}")
            if isinstance(first, dict):
                logger.info(f"  Dict keys: {first.keys()}")
                if 'prompt' in first:
                    prompt_val = first['prompt']
                    if isinstance(prompt_val, str):
                        logger.info(f"  Prompt text: {prompt_val[:50]}...")
                    else:
                        logger.info(f"  Prompt value is not string: {type(prompt_val)}")
    
    # Count total actual prompts
    total_actual = 0
    all_prompts = []
    for category, prompts in prompts_by_type.items():
        for p in prompts:
            if isinstance(p, dict) and 'prompt' in p:
                all_prompts.append(p['prompt'])
                total_actual += 1
            elif isinstance(p, str):
                all_prompts.append(p)
                total_actual += 1
    
    logger.info(f"Total extractable prompts: {total_actual}")
    return all_prompts

if __name__ == "__main__":
    prompts = test_orchestrator()
    print(f"\n✅ Successfully extracted {len(prompts)} prompts")
    if prompts:
        print(f"Sample prompts:")
        for i, p in enumerate(prompts[:5]):
            print(f"  {i+1}. {p[:80]}...")