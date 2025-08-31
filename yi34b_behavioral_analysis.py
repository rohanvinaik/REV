#!/usr/bin/env python3
"""
Behavioral analysis-based segmentation for Yi-34B.
Determines optimal layer cuts through prompt injection and similarity analysis.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

sys.path.insert(0, 'src')
from src.yi34b_efficient_loader import Yi34BEfficientLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Advanced behavioral probes for different model capabilities
BEHAVIORAL_PROBES = {
    "factual": [
        "Paris is the capital of",
        "Water freezes at temperature",
        "The speed of light is approximately"
    ],
    "syntactic": [
        "The cat sat on the",
        "She quickly ran to the",
        "Complete the pattern: A, B, C,"
    ],
    "semantic": [
        "The opposite of hot is",
        "A synonym for happy is",
        "If rain makes things wet, then fire makes things"
    ],
    "reasoning": [
        "If A > B and B > C, then A is",
        "2 + 2 * 3 equals",
        "All birds can fly. Penguins are birds. Therefore,"
    ],
    "creative": [
        "In a world without gravity,",
        "Imagine a color beyond the visible spectrum called",
        "If time flowed backwards,"
    ],
    "code": [
        "def factorial(n): return",
        "for i in range(10):",
        "class Person: def __init__(self,"
    ]
}


def analyze_behavioral_sites(model_path: str = "/Users/rohanvinaik/LLM_models/yi-34b"):
    """Analyze Yi-34B to find behavioral similarity sites for optimal segmentation."""
    
    loader = Yi34BEfficientLoader(model_path=model_path, use_mmap=True)
    loader.load_config_and_tokenizer()
    loader.map_model_shards()
    
    num_layers = loader.config.num_hidden_layers
    logger.info(f"Analyzing {num_layers} layers for behavioral patterns...")
    
    # Sample layers to analyze (every 5th layer for efficiency)
    sample_layers = list(range(0, num_layers, 5))
    
    layer_behaviors = {}
    
    for layer_idx in sample_layers:
        logger.info(f"Analyzing layer {layer_idx}")
        
        behaviors = []
        for probe_type, probes in BEHAVIORAL_PROBES.items():
            for probe in probes[:2]:  # Use first 2 probes per type
                # Tokenize probe
                inputs = loader.tokenizer(probe, return_tensors="pt")
                
                # Load layer weights
                layer_weights = loader.load_layer_weights(layer_idx)
                
                # Simple behavioral signature: weight statistics
                signature = []
                for key, weight in layer_weights.items():
                    if weight is not None:
                        stats = [
                            float(weight.mean()),
                            float(weight.std()),
                            float(weight.abs().max())
                        ]
                        signature.extend(stats)
                        
                behaviors.append(signature[:100])  # Limit signature size
                
                # Offload to save memory
                loader.offload_layer(layer_idx)
                
        if behaviors:
            layer_behaviors[layer_idx] = np.mean(behaviors, axis=0)
            
    # Find similarity groups
    logger.info("Computing layer similarity matrix...")
    
    layers = sorted(layer_behaviors.keys())
    similarity_matrix = np.zeros((len(layers), len(layers)))
    
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            if i != j:
                similarity = np.corrcoef(
                    layer_behaviors[l1], 
                    layer_behaviors[l2]
                )[0, 1]
                similarity_matrix[i, j] = similarity
                
    # Identify behavioral shifts (low similarity between adjacent samples)
    shifts = []
    for i in range(len(layers) - 1):
        if similarity_matrix[i, i+1] < 0.85:  # Threshold for behavioral shift
            shifts.append(layers[i+1])
            logger.info(f"Behavioral shift detected at layer {layers[i+1]}")
            
    # Create optimal segments
    segments = []
    prev = 0
    for shift in shifts + [num_layers]:
        if shift - prev >= 3:  # Minimum segment size
            segments.append((prev, shift))
            prev = shift
            
    logger.info(f"\nOptimal segmentation for Yi-34B:")
    for i, (start, end) in enumerate(segments):
        logger.info(f"  Segment {i+1}: Layers {start}-{end} ({end-start} layers)")
        
    # Describe behavioral regions
    logger.info("\nBehavioral regions identified:")
    if len(segments) >= 4:
        logger.info(f"  Early layers (0-{segments[0][1]}): Token/syntactic processing")
        logger.info(f"  Early-mid layers ({segments[1][0]}-{segments[1][1]}): Semantic understanding")
        logger.info(f"  Mid-late layers ({segments[2][0]}-{segments[2][1]}): Abstract reasoning")
        logger.info(f"  Late layers ({segments[3][0]}-{num_layers}): Output generation")
        
    loader.cleanup()
    return segments


def run_with_behavioral_segments(text: str):
    """Process text using behaviorally-determined segments."""
    
    # Get optimal segments
    segments = analyze_behavioral_sites()
    
    logger.info(f"\nProcessing text with {len(segments)} behavioral segments")
    
    loader = Yi34BEfficientLoader(
        model_path="/Users/rohanvinaik/LLM_models/yi-34b",
        use_mmap=True
    )
    loader.load_config_and_tokenizer()
    loader.map_model_shards()
    
    # Process with behavioral segments
    inputs = loader.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    for i, (start, end) in enumerate(segments):
        logger.info(f"Processing segment {i+1}: layers {start}-{end}")
        
        # Process layers in this behavioral segment
        layers_per_batch = min(3, end - start)  # Adaptive batch size
        
        for layer_idx in range(start, end, layers_per_batch):
            batch_end = min(layer_idx + layers_per_batch, end)
            logger.info(f"  Processing layers {layer_idx}-{batch_end}")
            
            # Load and process layers
            for l in range(layer_idx, batch_end):
                loader.load_layer_weights(l)
                
            # Process (simplified)
            # ... actual processing would happen here
            
            # Offload
            for l in range(layer_idx, batch_end):
                loader.offload_layer(l)
                
    logger.info("Processing complete!")
    loader.cleanup()


if __name__ == "__main__":
    # Example: Analyze behavioral structure
    print("="*60)
    print("Yi-34B Behavioral Analysis")
    print("="*60)
    
    # Complex prompt for testing
    test_text = """
    Consider a distributed system with eventual consistency guarantees.
    How would you design a conflict resolution mechanism that maintains
    both data integrity and system availability during network partitions?
    """
    
    run_with_behavioral_segments(test_text)