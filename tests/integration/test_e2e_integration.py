#!/usr/bin/env python3
"""
Test E2E integration of all new features in REV v3.0
Verifies cassettes, profiler, and advanced prompts work together.
"""

import subprocess
import json
import sys
from pathlib import Path


def test_cli_help():
    """Test that all CLI flags are available."""
    print("Testing CLI help...")
    result = subprocess.run(
        ["python", "run_rev.py", "--help"],
        capture_output=True,
        text=True
    )
    
    required_flags = [
        "--cassettes",
        "--cassette-topology",
        "--cassette-types",
        "--profiler",
        "--challenges"
    ]
    
    for flag in required_flags:
        if flag not in result.stdout:
            print(f"  ❌ Missing flag: {flag}")
            return False
    
    print("  ✅ All CLI flags present")
    return True


def test_import_modules():
    """Test that all new modules can be imported."""
    print("\nTesting module imports...")
    
    modules_to_test = [
        ("Cassette Executor", "src.challenges.cassette_executor"),
        ("Advanced Cassettes", "src.challenges.advanced_probe_cassettes"),
        ("KDF Prompts", "src.challenges.kdf_prompts"),
        ("Evolutionary Prompts", "src.challenges.evolutionary_prompts"),
        ("Behavior Profiler", "src.analysis.behavior_profiler")
    ]
    
    all_passed = True
    for name, module in modules_to_test:
        try:
            exec(f"import {module}")
            print(f"  ✅ {name}: Imported successfully")
        except ImportError as e:
            print(f"  ❌ {name}: Import failed - {e}")
            all_passed = False
    
    return all_passed


def test_cassette_library():
    """Test cassette library functionality."""
    print("\nTesting cassette library...")
    
    try:
        from src.challenges.advanced_probe_cassettes import CassetteLibrary, ProbeType
        
        library = CassetteLibrary()
        
        # Test that library has cassettes
        if hasattr(library, 'cassettes'):
            print(f"  ✅ Library has {len(library.cassettes)} cassettes")
        
        # Test probe types exist
        probe_types = list(ProbeType)
        print(f"  ✅ {len(probe_types)} probe types available")
        
        # Test that we can generate a probe schedule (main method)
        dummy_topology = {"layers": list(range(10))}
        schedule = library.generate_probe_schedule(dummy_topology)
        print(f"  ✅ Generated schedule for {len(schedule)} layers")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cassette library test failed: {e}")
        return False


def test_behavior_profiler():
    """Test behavior profiler functionality."""
    print("\nTesting behavior profiler...")
    
    try:
        from src.analysis.behavior_profiler import BehaviorProfiler, BehavioralSignature
        import numpy as np
        
        profiler = BehaviorProfiler()
        
        # Test signature creation
        sig = BehavioralSignature()
        sig.response_variability = 0.5
        sig.semantic_coherence = 0.8
        
        # Test vector conversion
        vec = sig.to_vector()
        print(f"  ✅ Created {len(vec)}-dimensional signature")
        
        # Test gradient extraction
        test_activations = {
            0: np.random.randn(512, 4096),
            1: np.random.randn(512, 4096)
        }
        
        gradient_sig = profiler.feature_extractor.extract_gradient_signature(test_activations)
        print(f"  ✅ Extracted gradient signature: {len(gradient_sig)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Behavior profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kdf_prompts():
    """Test KDF prompt generation."""
    print("\nTesting KDF prompts...")
    
    try:
        from src.challenges.kdf_prompts import KDFPromptGenerator
        import os
        
        # Create with required key
        generator = KDFPromptGenerator(prf_key=os.urandom(32))
        
        # Test non-traditional injection
        prompts = generator.generate_non_traditional_injection_prompts(n=5)
        print(f"  ✅ Generated {len(prompts)} injection prompts")
        
        # Test TensorGuard probes
        probes = generator.get_tensorguard_probes()
        print(f"  ✅ Loaded {len(probes)} TensorGuard probes")
        
        # Test challenge export
        challenges = generator.export_for_integration(n=3)
        print(f"  ✅ Exported {len(challenges)} challenges for integration")
        
        return True
        
    except Exception as e:
        print(f"  ❌ KDF prompts test failed: {e}")
        return False


def test_evolutionary_prompts():
    """Test evolutionary prompt generation."""
    print("\nTesting evolutionary prompts...")
    
    try:
        from src.challenges.evolutionary_prompts import GeneticPromptOptimizer
        
        # Mock fitness evaluator
        def mock_evaluator(prompts):
            # Return mock fitness scores
            return {prompt: len(prompt) / 100.0 for prompt in prompts}
        
        optimizer = GeneticPromptOptimizer(
            fitness_evaluator=mock_evaluator,
            population_size=10
        )
        
        # Initialize and evolve
        optimizer.initialize_population()
        optimizer.evolve(generations=2)
        
        best = optimizer.get_best_prompts(n=3)
        print(f"  ✅ Evolved {len(best)} prompts over 2 generations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evolutionary prompts test failed: {e}")
        return False


def test_integration_pipeline():
    """Test that all components integrate properly."""
    print("\nTesting integration pipeline...")
    
    try:
        # Import main pipeline
        from run_rev import REVUnified
        
        # Create pipeline with all features
        pipeline = REVUnified(
            debug=False,
            enable_behavioral_analysis=True,
            enable_pot_challenges=True,
            enable_cassettes=True,
            enable_profiler=True
        )
        
        print("  ✅ Pipeline initialized with all features")
        
        # Check components
        if pipeline.cassette_executor is None:
            print("  ⚠️  Cassette executor needs topology file")
        
        if pipeline.behavior_profiler is not None:
            print("  ✅ Behavior profiler integrated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("REV v3.0 E2E INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("Module Imports", test_import_modules),
        ("Cassette Library", test_cassette_library),
        ("Behavior Profiler", test_behavior_profiler),
        ("KDF Prompts", test_kdf_prompts),
        ("Evolutionary Prompts", test_evolutionary_prompts),
        ("Integration Pipeline", test_integration_pipeline)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nREV v3.0 E2E integration successful:")
        print("  • Cassette analysis integrated ✓")
        print("  • Behavior profiler working ✓")
        print("  • Advanced prompts operational ✓")
        print("  • CLI fully functional ✓")
        return 0
    else:
        print("\n⚠️ Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())