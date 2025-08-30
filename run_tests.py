#!/usr/bin/env python3
"""Quick test runner to identify main issues."""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports."""
    try:
        print("Testing imports...")
        from src.rev_pipeline import REVPipeline
        print("✓ REVPipeline imported")
        
        from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
        print("✓ HDC encoder imported")
        
        from src.core.sequential import SequentialState
        print("✓ Sequential state imported")
        
        from src.crypto.merkle import build_merkle_tree
        print("✓ Merkle crypto imported")
        
        from src.verifier.streaming_consensus import StreamingConsensusVerifier
        print("✓ Streaming consensus imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_pipeline():
    """Test basic pipeline creation."""
    try:
        print("\nTesting basic pipeline...")
        from src.rev_pipeline import REVPipeline
        
        pipeline = REVPipeline(
            segment_size=512,
            buffer_size=4
        )
        print("✓ Pipeline created")
        
        # Test segment generation
        tokens = list(range(1000))
        segments = list(pipeline.segment_tokens(tokens, use_overlap=False))
        print(f"✓ Generated {len(segments)} segments")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hdc_encoder():
    """Test HDC encoder."""
    try:
        print("\nTesting HDC encoder...")
        from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
        import numpy as np
        
        config = HypervectorConfig(dimension=10000, sparsity=0.01)
        encoder = HypervectorEncoder(config)
        print("✓ HDC encoder created")
        
        # Test encoding
        data = np.random.randn(100)
        encoded = encoder.encode(data)
        print(f"✓ Encoded data shape: {encoded.shape}")
        
        return True
    except Exception as e:
        print(f"✗ HDC encoder error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests."""
    print("=" * 60)
    print("REV System Quick Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Basic Pipeline", test_basic_pipeline()))
    results.append(("HDC Encoder", test_hdc_encoder()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("-" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
