#!/usr/bin/env python3
"""
Test script to verify layer execution fixes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
from src.models.true_segment_execution import LayerSegmentExecutor, SegmentExecutionConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_layer_execution():
    """Test that layer execution with fallback produces divergence"""
    
    logger.info("="*60)
    logger.info("TESTING LAYER EXECUTION WITH FIXES")
    logger.info("="*60)
    
    # Create config
    config = SegmentExecutionConfig(
        model_path="/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct",
        use_half_precision=True
    )
    
    # Initialize executor
    logger.info("Initializing segment executor...")
    executor = LayerSegmentExecutor(config)
    
    # Test probes
    test_probes = [
        "Consider a recursive function that computes factorial.",
        "Transform the recursive algorithm to iterative form.",
    ]
    
    logger.info(f"\nTesting with {len(test_probes)} probes")
    
    # Test layer 0
    logger.info("\n" + "="*40)
    logger.info("Testing Layer 0")
    logger.info("="*40)
    
    layer_0_responses = []
    for probe in test_probes:
        logger.info(f"\nProbe: {probe[:50]}...")
        response = executor.execute_behavioral_probe(probe, up_to_layer=0)
        
        if response and response.statistical_signature:
            divergence = response.statistical_signature.get('overall_divergence', 0.0)
            logger.info(f"✓ Layer 0 divergence: {divergence:.3f}")
            layer_0_responses.append(response)
        else:
            logger.error("✗ Layer 0 probe failed")
    
    # Test layer 1 
    logger.info("\n" + "="*40)
    logger.info("Testing Layer 1 (with CPU fallback)")
    logger.info("="*40)
    
    layer_1_responses = []
    for i, probe in enumerate(test_probes):
        logger.info(f"\nProbe: {probe[:50]}...")
        response = executor.execute_behavioral_probe(probe, up_to_layer=1)
        
        if response and response.statistical_signature:
            divergence = response.statistical_signature.get('overall_divergence', 0.0)
            logger.info(f"✓ Layer 1 divergence: {divergence:.3f}")
            
            # Check if divergence increased from layer 0
            if i < len(layer_0_responses):
                layer_0_div = layer_0_responses[i].statistical_signature.get('overall_divergence', 0.0)
                if divergence > layer_0_div:
                    logger.info(f"  ✓ Divergence increased from {layer_0_div:.3f} to {divergence:.3f}")
                else:
                    logger.warning(f"  ⚠️ Divergence did not increase ({layer_0_div:.3f} -> {divergence:.3f})")
            
            layer_1_responses.append(response)
        else:
            logger.error("✗ Layer 1 probe failed")
    
    # Check device consistency
    logger.info("\n" + "="*40)
    logger.info("Checking Device Consistency")
    logger.info("="*40)
    
    for i, resp in enumerate(layer_1_responses):
        if resp and resp.hidden_states is not None:
            device = resp.hidden_states.device
            dtype = resp.hidden_states.dtype
            logger.info(f"Response {i}: device={device}, dtype={dtype}")
            
            if device != executor.device:
                logger.error(f"  ✗ Device mismatch: expected {executor.device}, got {device}")
            else:
                logger.info(f"  ✓ Device correct: {device}")
    
    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        test_layer_execution()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()