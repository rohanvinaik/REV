#!/usr/bin/env python3
"""
Memory profiling script for REV system components.

This script profiles memory usage of critical functions.
"""

import sys
from pathlib import Path
import numpy as np
from memory_profiler import profile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hdc.encoder import HDCEncoder
from src.features.taxonomy import HierarchicalFeatureTaxonomy
from src.hypervector.hamming import HammingDistanceOptimized
from src.core.sequential import SequentialTest


@profile
def profile_hdc_encoding():
    """Profile memory usage of HDC encoding."""
    encoder = HDCEncoder(dimension=100000, sparsity=0.01)
    
    # Generate features
    features = np.random.randn(56)
    
    # Encode multiple times
    hypervectors = []
    for _ in range(10):
        hv = encoder.encode_vector(features)
        hypervectors.append(hv)
    
    return hypervectors


@profile
def profile_feature_extraction():
    """Profile memory usage of feature extraction."""
    taxonomy = HierarchicalFeatureTaxonomy()
    
    # Test with various text sizes
    texts = [
        "Short text" * 10,
        "Medium length text that contains more content" * 50,
        "Very long text with lots of repetition" * 200
    ]
    
    all_features = []
    for text in texts:
        features = taxonomy.extract_all_features(text, prompt="test")
        concat = taxonomy.get_concatenated_features(features)
        all_features.append(concat)
    
    return all_features


@profile
def profile_hamming_distance():
    """Profile memory usage of Hamming distance computation."""
    hamming = HammingDistanceOptimized()
    
    # Test with different vector sizes
    sizes = [1000, 10000, 100000]
    distances = []
    
    for size in sizes:
        v1 = np.random.randint(0, 2, size, dtype=np.uint8)
        v2 = np.random.randint(0, 2, size, dtype=np.uint8)
        
        # Compute distance multiple times
        for _ in range(100):
            dist = hamming.distance(v1, v2)
            distances.append(dist)
    
    return distances


@profile  
def profile_sequential_test():
    """Profile memory usage of SPRT."""
    test = SequentialTest()
    
    # Add many samples
    for _ in range(10000):
        test.add_sample(np.random.random())
    
    return test


@profile
def profile_combined_pipeline():
    """Profile memory usage of complete pipeline."""
    # Feature extraction
    taxonomy = HierarchicalFeatureTaxonomy()
    features = taxonomy.extract_all_features("Test text", prompt="test")
    concat = taxonomy.get_concatenated_features(features)
    
    # HDC encoding
    encoder = HDCEncoder(dimension=10000, sparsity=0.01)
    hypervector = encoder.encode_vector(concat)
    
    # Similarity computation
    hamming = HammingDistanceOptimized()
    v2 = np.random.randint(0, 2, 10000, dtype=np.uint8)
    distance = hamming.distance(hypervector, v2)
    
    # Statistical testing
    test = SequentialTest()
    for _ in range(100):
        test.add_sample(np.random.random())
    
    return distance, test


def main():
    """Run all memory profiling."""
    print("=" * 60)
    print("REV SYSTEM MEMORY PROFILING")
    print("=" * 60)
    
    print("\n1. Profiling HDC Encoding...")
    profile_hdc_encoding()
    
    print("\n2. Profiling Feature Extraction...")
    profile_feature_extraction()
    
    print("\n3. Profiling Hamming Distance...")
    profile_hamming_distance()
    
    print("\n4. Profiling Sequential Test...")
    profile_sequential_test()
    
    print("\n5. Profiling Combined Pipeline...")
    profile_combined_pipeline()
    
    print("\n" + "=" * 60)
    print("Memory profiling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()