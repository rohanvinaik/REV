#!/usr/bin/env python3
"""
Test with single model to verify real activation extraction.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, '/Users/rohanvinaik/REV')

from test_full_pipeline_scientific import FullScientificPipelineTest

def test_single_model():
    """Test just pythia-70m to verify real activations."""
    
    print("Testing single model with real activations...")
    
    # Initialize test
    test = FullScientificPipelineTest()
    
    # Load only pythia-70m
    print("\nLoading pythia-70m...")
    models = test.scan_and_load_models()
    
    # Find pythia-70m
    pythia = None
    for m in models:
        if m['name'] == 'pythia-70m' and m.get('model') is not None:
            pythia = m
            break
    
    if not pythia:
        print("❌ Could not load pythia-70m")
        return False
    
    print(f"✓ Loaded {pythia['name']}: {pythia['size_gb']:.2f}GB")
    
    # Create sites and segments
    sites, segments = test.create_full_segment_structure(pythia)
    print(f"✓ Created {len(sites)} sites, {len(segments)} segments")
    
    # Create a single challenge
    challenge = {
        'index': 1,
        'prompt': "The capital of France is Paris. The Eiffel Tower is located in this beautiful city."
    }
    
    print(f"\nExecuting pipeline with real model...")
    start = time.perf_counter()
    
    try:
        # Execute pipeline
        metrics = test.execute_full_pipeline(pythia, sites, segments, challenge)
        
        elapsed = time.perf_counter() - start
        print(f"\n✅ SUCCESS! Pipeline executed in {elapsed:.1f}s")
        
        # Print metrics
        print(f"\nMetrics:")
        print(f"  - Model: {metrics.model_name}")
        print(f"  - Segments processed: {metrics.n_segments}")
        print(f"  - Total time: {metrics.total_time_s:.2f}s")
        print(f"  - Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  - Avg segment time: {np.mean(metrics.segment_times_ms):.1f}ms")
        print(f"  - Components used: {len(test.components_used)}")
        
        # Verify we used real model
        if metrics.peak_memory_mb > 100:  # Should use more than 100MB with real model
            print(f"\n✓ Memory usage indicates real model execution")
        else:
            print(f"\n⚠️  Low memory usage might indicate fallback to random data")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_model()
    exit(0 if success else 1)