#!/usr/bin/env python3
"""
Simple test for Dynamic Prompt Synthesis System integration.
"""

import os

def test_basic_integration():
    """Test basic KDF integration with dynamic synthesis."""
    print("Testing KDF Integration with Dynamic Synthesis...")
    
    try:
        from src.challenges.kdf_prompts import KDFPromptGenerator
        
        # Create generator
        generator = KDFPromptGenerator(prf_key=os.urandom(32))
        
        print(f"✅ KDF Generator initialized")
        print(f"   Dynamic synthesis available: {generator.dynamic_synthesizer is not None}")
        
        # Generate small challenge set
        challenges = generator.generate_challenge_set(
            n_challenges=5,
            use_dynamic_synthesis=True,
            dynamic_synthesis_ratio=0.4  # 40% dynamic
        )
        
        print(f"✅ Generated {len(challenges)} challenges")
        
        # Count types
        types = [c.get("type", "unknown") for c in challenges]
        type_counts = {t: types.count(t) for t in set(types)}
        
        print("Challenge types:")
        for ctype, count in type_counts.items():
            print(f"   {ctype}: {count}")
        
        # Show sample dynamic challenges
        dynamic = [c for c in challenges if c.get("type") == "dynamic_synthesis"]
        if dynamic:
            print(f"\nSample dynamic synthesis prompt:")
            print(f"   {dynamic[0]['prompt'][:100]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_synthesis_components():
    """Test individual synthesis components."""
    print("\nTesting Synthesis Components...")
    
    try:
        from src.challenges.dynamic_synthesis import DynamicSynthesisSystem
        
        system = DynamicSynthesisSystem()
        print("✅ DynamicSynthesisSystem initialized")
        
        # Test simple prompt generation
        prompt = system.generate_prompt(complexity=1.0)
        print(f"✅ Generated prompt: {prompt[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE DYNAMIC SYNTHESIS TEST")
    print("=" * 50)
    
    results = []
    results.append(("Synthesis Components", test_synthesis_components()))
    results.append(("KDF Integration", test_basic_integration()))
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 Basic integration works!")
    else:
        print("\n⚠️ Some tests failed")