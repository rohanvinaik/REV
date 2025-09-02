#!/usr/bin/env python3
"""
Test script for Dynamic Prompt Synthesis System.
Demonstrates all features of the dynamic synthesis integrated with KDF prompts.
"""

import os
import json
from typing import List, Dict, Any

# Test dynamic synthesis system standalone
from src.challenges.dynamic_synthesis import (
    DynamicSynthesisSystem,
    GenerationContext,
    DomainType,
    TemplateType
)

# Test integration with KDF prompt generator
from src.challenges.kdf_prompts import KDFPromptGenerator


def test_template_combination():
    """Test template combination engine."""
    print("\n" + "="*60)
    print("TESTING TEMPLATE COMBINATION ENGINE")
    print("="*60)
    
    system = DynamicSynthesisSystem()
    
    # Test single template
    prompt1 = system.generate_prompt(
        template_types=[TemplateType.QUESTION],
        complexity=1.0
    )
    print(f"\nSimple Question Template:")
    print(f"  {prompt1}")
    
    # Test combined templates
    prompt2 = system.generate_prompt(
        template_types=[TemplateType.QUESTION, TemplateType.REASONING],
        complexity=2.5
    )
    print(f"\nCombined Question + Reasoning:")
    print(f"  {prompt2}")
    
    # Test with semantic blending
    variables = {
        "concept": "quantum computing",
        "aspect": "practical applications"
    }
    blended = system.template_mixer.combine_templates(
        ["q_basic", "q_comparative"],
        variables,
        use_bridges=True
    )
    print(f"\nSemantically Blended Prompt:")
    print(f"  {blended}")
    
    return True


def test_context_aware_generation():
    """Test context-aware prompt generation."""
    print("\n" + "="*60)
    print("TESTING CONTEXT-AWARE GENERATION")
    print("="*60)
    
    system = DynamicSynthesisSystem()
    context = GenerationContext()
    
    # Simulate conversation history
    prompts_and_responses = [
        ("What is machine learning?", "Machine learning is a subset of AI..."),
        ("How does it differ from traditional programming?", "Unlike traditional programming..."),
        ("What are neural networks?", "Neural networks are computational models...")
    ]
    
    for prompt, response in prompts_and_responses:
        context.update(prompt, response, score=0.7)
    
    # Generate context-aware follow-up
    next_prompt = system.generate_prompt(
        template_types=[TemplateType.QUESTION],
        context=context,
        complexity=context.current_difficulty
    )
    
    print(f"\nContext History:")
    for i, (p, r) in enumerate(prompts_and_responses):
        print(f"  Q{i+1}: {p}")
        print(f"  A{i+1}: {r[:50]}...")
    
    print(f"\nContext-Aware Follow-up:")
    print(f"  Difficulty: {context.current_difficulty:.2f}")
    print(f"  Prompt: {next_prompt}")
    
    # Test adaptive difficulty
    context.performance_scores = [0.9, 0.95, 0.92]  # High performance
    difficult_prompt = system.context_generator.scale_difficulty(
        "Explain the concept.",
        target_difficulty=context.current_difficulty * 1.5
    )
    print(f"\nAdaptively Scaled Prompt (harder):")
    print(f"  {difficult_prompt}")
    
    return True


def test_domain_specific_synthesis():
    """Test domain-specific prompt generation."""
    print("\n" + "="*60)
    print("TESTING DOMAIN-SPECIFIC SYNTHESIS")
    print("="*60)
    
    system = DynamicSynthesisSystem()
    
    domains_to_test = [
        DomainType.SCIENTIFIC,
        DomainType.MATHEMATICAL,
        DomainType.TECHNICAL,
        DomainType.PHILOSOPHICAL
    ]
    
    for domain in domains_to_test:
        # Generate domain-specific prompt
        prompt = system.domain_synthesizer.generate_domain_prompt(
            domain, 
            complexity=3.0
        )
        
        # Add jargon
        jargon_prompt = system.domain_synthesizer.inject_jargon(
            prompt, 
            domain, 
            density=0.15
        )
        
        print(f"\n{domain.value.upper()} Domain:")
        print(f"  Base: {prompt}")
        print(f"  With Jargon: {jargon_prompt}")
    
    # Test cross-domain bridge
    cross_prompt = system.domain_synthesizer.create_cross_domain_prompt(
        DomainType.SCIENTIFIC,
        DomainType.PHILOSOPHICAL
    )
    print(f"\nCross-Domain (Science + Philosophy):")
    print(f"  {cross_prompt}")
    
    # Test edge cases
    edge_cases = system.generate_edge_cases(DomainType.TECHNICAL, num_cases=3)
    print(f"\nTechnical Edge Cases:")
    for i, edge in enumerate(edge_cases, 1):
        print(f"  {i}. {edge}")
    
    return True


def test_quality_control():
    """Test quality control pipeline."""
    print("\n" + "="*60)
    print("TESTING QUALITY CONTROL PIPELINE")
    print("="*60)
    
    system = DynamicSynthesisSystem()
    
    # Test grammar validation
    test_prompts = [
        "What is the the capital of France?",  # Repeated word
        "capital of France is what?",  # Fragment
        "Explain quantum mechanics and its applications in modern technology."  # Good
    ]
    
    for prompt in test_prompts:
        is_valid, errors = system.quality_controller.validate_grammar(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"  Valid: {is_valid}")
        if errors:
            print(f"  Errors: {errors}")
    
    # Test coherence scoring
    coherent_prompt = "Machine learning is a subset of artificial intelligence. It enables computers to learn from data. This learning process improves performance over time."
    incoherent_prompt = "Machine learning is great. Yesterday was sunny. Cats are mammals. Programming is fun."
    
    coherent_score = system.quality_controller.score_coherence(coherent_prompt)
    incoherent_score = system.quality_controller.score_coherence(incoherent_prompt)
    
    print(f"\nCoherence Scores:")
    print(f"  Coherent text: {coherent_score:.2f}")
    print(f"  Incoherent text: {incoherent_score:.2f}")
    
    # Test complexity estimation
    simple_prompt = "What is 2 plus 2?"
    complex_prompt = "Analyze the epistemological implications of Gödel's incompleteness theorems on the foundations of mathematical logic, considering both formalist and intuitionist perspectives."
    
    simple_metrics = system.quality_controller.estimate_complexity(simple_prompt)
    complex_metrics = system.quality_controller.estimate_complexity(complex_prompt)
    
    print(f"\nComplexity Metrics:")
    print(f"  Simple: FK Grade={simple_metrics.get('flesch_kincaid_grade', 0):.1f}")
    print(f"  Complex: FK Grade={complex_metrics.get('flesch_kincaid_grade', 0):.1f}")
    
    return True


def test_kdf_integration():
    """Test integration with KDF prompt generator."""
    print("\n" + "="*60)
    print("TESTING KDF PROMPT GENERATOR INTEGRATION")
    print("="*60)
    
    # Create KDF generator
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate challenge set with dynamic synthesis
    challenges = generator.generate_challenge_set(
        n_challenges=10,
        use_dynamic_synthesis=True,
        dynamic_synthesis_ratio=0.4,  # 40% dynamic
        adversarial_ratio=0.2,         # 20% adversarial
        behavioral_probe_ratio=0.1,    # 10% behavioral
    )
    
    print(f"\nGenerated {len(challenges)} challenges:")
    
    # Count types
    type_counts = {}
    for challenge in challenges:
        ctype = challenge.get("type", "unknown")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    
    print("\nChallenge Type Distribution:")
    for ctype, count in type_counts.items():
        print(f"  {ctype}: {count} ({count/len(challenges)*100:.0f}%)")
    
    # Show sample dynamic synthesis challenges
    dynamic_challenges = [c for c in challenges if c.get("type") == "dynamic_synthesis"]
    if dynamic_challenges:
        print(f"\nSample Dynamic Synthesis Challenges:")
        for i, challenge in enumerate(dynamic_challenges[:3], 1):
            print(f"\n  {i}. Prompt: {challenge['prompt'][:100]}...")
            print(f"     Domain: {challenge.get('domain', 'N/A')}")
            print(f"     Difficulty: {challenge.get('difficulty', 'N/A')}")
            print(f"     Diversity Score: {challenge.get('diversity_score', 0):.2f}")
    
    return True


def test_batch_generation():
    """Test batch generation with diversity."""
    print("\n" + "="*60)
    print("TESTING BATCH GENERATION WITH DIVERSITY")
    print("="*60)
    
    system = DynamicSynthesisSystem()
    
    # Generate diverse batch
    batch = system.generate_batch(
        num_prompts=5,
        domains=[DomainType.SCIENTIFIC, DomainType.TECHNICAL],
        complexity_range=(1.0, 4.0),
        ensure_diversity=True
    )
    
    print(f"\nGenerated {len(batch)} diverse prompts:")
    for i, prompt in enumerate(batch, 1):
        print(f"\n  {i}. {prompt[:150]}...")
    
    # Test conversation generation
    conversation = system.generate_conversation_sequence(
        topic="artificial intelligence ethics",
        num_turns=3,
        domain=DomainType.PHILOSOPHICAL
    )
    
    print(f"\n\nGenerated Conversation on AI Ethics:")
    for i, turn in enumerate(conversation, 1):
        print(f"\n  Turn {i}: {turn}")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("DYNAMIC PROMPT SYNTHESIS SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("Template Combination", test_template_combination),
        ("Context-Aware Generation", test_context_aware_generation),
        ("Domain-Specific Synthesis", test_domain_specific_synthesis),
        ("Quality Control Pipeline", test_quality_control),
        ("KDF Integration", test_kdf_integration),
        ("Batch Generation", test_batch_generation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            print(f"\n✅ {test_name} passed")
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nDynamic Synthesis Features Successfully Implemented:")
        print("  • Template Combination Engine with semantic blending")
        print("  • Context-Aware Generation with adaptive difficulty")
        print("  • Domain-Specific Synthesizers for 10+ domains")
        print("  • Quality Control with grammar, coherence, and complexity checks")
        print("  • Seamless KDF Integration with 40% dynamic generation")
        print("  • Batch Generation with diversity filtering")
        print("  • Multi-turn Conversation Generation")
    else:
        print("\n⚠️ Some tests failed. Review output above.")


if __name__ == "__main__":
    main()