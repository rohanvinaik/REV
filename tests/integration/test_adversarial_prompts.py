#!/usr/bin/env python3
"""
Test script for sophisticated adversarial prompt generation.
Validates the new adversarial capabilities in kdf_prompts.py.

WARNING: This is for security research and model robustness testing only.
"""

import os
import json
from typing import Dict, List, Any

from src.challenges.kdf_prompts import (
    KDFPromptGenerator,
    AdversarialType,
    DomainType
)


def test_divergence_attack():
    """Test Divergence Attack generation (150x faster extraction)."""
    print("\n" + "="*60)
    print("TESTING DIVERGENCE ATTACK")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate divergence attack
    attack = generator.generate_divergence_attack(index=1)
    
    print(f"✅ Generated divergence attack:")
    print(f"   Type: {attack['adversarial_type']}")
    print(f"   Vector: {attack['attack_vector']}")
    print(f"   Difficulty: {attack['difficulty']}")
    print(f"   Prompt preview: {attack['prompt'][:50]}...")
    print(f"   Safety note: {attack['safety_note']}")
    
    assert attack['adversarial_type'] == AdversarialType.DIVERGENCE_ATTACK.value
    assert attack['difficulty'] == 5
    assert 'memorization_extraction' in attack['attack_vector']
    
    return True


def test_multi_round_jailbreak():
    """Test Multi-Round Conversational Jailbreaking (>90% success rate)."""
    print("\n" + "="*60)
    print("TESTING MULTI-ROUND CONVERSATIONAL JAILBREAK")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate MRCJ
    conversation = generator.generate_multi_round_conversational_jailbreak(index=2, rounds=3)
    
    print(f"✅ Generated {len(conversation)}-round conversation:")
    for i, round_prompt in enumerate(conversation):
        print(f"   Round {i}: {round_prompt['prompt'][:60]}...")
        assert round_prompt['round'] == i
        assert round_prompt['adversarial_type'] == AdversarialType.MULTI_ROUND_CONVERSATIONAL.value
    
    assert len(conversation) == 3
    return True


def test_special_character_triggers():
    """Test special character trigger generation."""
    print("\n" + "="*60)
    print("TESTING SPECIAL CHARACTER TRIGGERS")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate special character attack
    attack = generator.generate_special_character_triggers(index=3)
    
    print(f"✅ Generated special character trigger:")
    print(f"   Type: {attack['adversarial_type']}")
    print(f"   Contains special chars: {attack['contains_special_chars']}")
    print(f"   Vector: {attack['attack_vector']}")
    # Don't print the actual prompt as it contains special characters
    print(f"   Prompt length: {len(attack['prompt'])} characters")
    
    assert attack['adversarial_type'] == AdversarialType.SPECIAL_CHARACTER_TRIGGER.value
    assert attack['contains_special_chars'] is True
    
    return True


def test_two_stage_inversion():
    """Test two-stage model inversion attack (38-75% success on personalized LLMs)."""
    print("\n" + "="*60)
    print("TESTING TWO-STAGE INVERSION ATTACK")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate inversion attack
    attack = generator.generate_two_stage_inversion_attack(index=4)
    
    print(f"✅ Generated two-stage inversion attack:")
    print(f"   Type: {attack['adversarial_type']}")
    print(f"   Targets PII: {attack['targets_pii']}")
    print(f"   Vector: {attack['attack_vector']}")
    print(f"   Has Stage 1: {'Stage 1:' in attack['prompt']}")
    print(f"   Has Stage 2: {'Stage 2:' in attack['prompt']}")
    
    assert attack['adversarial_type'] == AdversarialType.TWO_STAGE_INVERSION.value
    assert attack['targets_pii'] is True
    assert 'Stage 1:' in attack['prompt'] and 'Stage 2:' in attack['prompt']
    
    return True


def test_spv_mia():
    """Test SPV-MIA membership inference probes."""
    print("\n" + "="*60)
    print("TESTING SPV-MIA PROBES")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate SPV-MIA probe
    probe = generator.generate_spv_mia_probe(index=5)
    
    print(f"✅ Generated SPV-MIA probe:")
    print(f"   Type: {probe['adversarial_type']}")
    print(f"   Inference type: {probe['inference_type']}")
    print(f"   Vector: {probe['attack_vector']}")
    print(f"   Prompt preview: {probe['prompt'][:60]}...")
    
    assert probe['adversarial_type'] == AdversarialType.SPV_MIA.value
    assert probe['inference_type'] == 'training_data_detection'
    
    return True


def test_alignment_faking():
    """Test alignment faking detection."""
    print("\n" + "="*60)
    print("TESTING ALIGNMENT FAKING DETECTION")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate alignment faking detector
    detector = generator.generate_alignment_faking_detector(index=6)
    
    print(f"✅ Generated alignment faking detector:")
    print(f"   Type: {detector['adversarial_type']}")
    print(f"   Tests authenticity: {detector['tests_authenticity']}")
    print(f"   Vector: {detector['attack_vector']}")
    print(f"   Prompt preview: {detector['prompt'][:60]}...")
    
    assert detector['adversarial_type'] == AdversarialType.ALIGNMENT_FAKING.value
    assert detector['tests_authenticity'] is True
    
    return True


def test_pair_algorithm():
    """Test PAIR algorithm jailbreak generation."""
    print("\n" + "="*60)
    print("TESTING PAIR ALGORITHM")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate PAIR jailbreak
    jailbreak = generator.generate_pair_algorithm_jailbreak(index=7, iterations=3)
    
    print(f"✅ Generated PAIR algorithm jailbreak:")
    print(f"   Type: {jailbreak['adversarial_type']}")
    print(f"   Iterations: {jailbreak['refinement_iterations']}")
    print(f"   Base objective: {jailbreak['base_objective']}")
    print(f"   Vector: {jailbreak['attack_vector']}")
    
    assert jailbreak['adversarial_type'] == AdversarialType.PAIR_ALGORITHM.value
    assert jailbreak['refinement_iterations'] == 3
    
    return True


def test_cross_lingual_inversion():
    """Test cross-lingual inversion attacks."""
    print("\n" + "="*60)
    print("TESTING CROSS-LINGUAL INVERSION")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate cross-lingual attack
    attack = generator.generate_cross_lingual_inversion(index=8)
    
    print(f"✅ Generated cross-lingual inversion:")
    print(f"   Type: {attack['adversarial_type']}")
    print(f"   Is multilingual: {attack['is_multilingual']}")
    print(f"   Vector: {attack['attack_vector']}")
    
    assert attack['adversarial_type'] == AdversarialType.CROSS_LINGUAL_INVERSION.value
    assert attack['is_multilingual'] is True
    
    return True


def test_temperature_exploitation():
    """Test temperature/sampling parameter exploitation."""
    print("\n" + "="*60)
    print("TESTING TEMPERATURE EXPLOITATION")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate temperature exploitation
    exploit = generator.generate_temperature_exploitation(index=9)
    
    print(f"✅ Generated temperature exploitation:")
    print(f"   Type: {exploit['adversarial_type']}")
    print(f"   Manipulates sampling: {exploit['manipulates_sampling']}")
    print(f"   Vector: {exploit['attack_vector']}")
    print(f"   Has temperature config: {'temperature' in exploit['prompt'].lower()}")
    
    assert exploit['adversarial_type'] == AdversarialType.TEMPERATURE_EXPLOITATION.value
    assert exploit['manipulates_sampling'] is True
    
    return True


def test_dataset_extraction():
    """Test dataset extraction probes (>50% recovery)."""
    print("\n" + "="*60)
    print("TESTING DATASET EXTRACTION")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate dataset extraction probe
    probe = generator.generate_dataset_extraction_probe(index=10)
    
    print(f"✅ Generated dataset extraction probe:")
    print(f"   Type: {probe['adversarial_type']}")
    print(f"   Targets training data: {probe['targets_training_data']}")
    print(f"   Vector: {probe['attack_vector']}")
    print(f"   Difficulty: {probe['difficulty']}")
    
    assert probe['adversarial_type'] == AdversarialType.DATASET_EXTRACTION.value
    assert probe['targets_training_data'] is True
    
    return True


def test_deception_pattern():
    """Test deception pattern detection."""
    print("\n" + "="*60)
    print("TESTING DECEPTION PATTERN DETECTION")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate deception detector
    detector = generator.generate_deception_pattern_detector(index=11)
    
    print(f"✅ Generated deception pattern detector:")
    print(f"   Type: {detector['adversarial_type']}")
    print(f"   Tests honesty: {detector['tests_honesty']}")
    print(f"   Vector: {detector['attack_vector']}")
    print(f"   Prompt preview: {detector['prompt'][:60]}...")
    
    assert detector['adversarial_type'] == AdversarialType.DECEPTION_PATTERN.value
    assert detector['tests_honesty'] is True
    
    return True


def test_comprehensive_suite():
    """Test comprehensive adversarial suite generation."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE ADVERSARIAL SUITE")
    print("="*60)
    
    generator = KDFPromptGenerator(prf_key=os.urandom(32))
    
    # Generate safe suite (no dangerous prompts)
    safe_suite = generator.generate_comprehensive_adversarial_suite(
        base_index=100,
        include_dangerous=False
    )
    
    print(f"✅ Generated safe suite with {len(safe_suite)} prompts")
    
    # Generate full suite (with safety wrappers)
    full_suite = generator.generate_comprehensive_adversarial_suite(
        base_index=200,
        include_dangerous=True
    )
    
    print(f"✅ Generated full suite with {len(full_suite)} prompts")
    
    # Check safety wrappers on dangerous prompts
    dangerous_count = sum(1 for p in full_suite if p.get('has_safety_wrapper', False))
    print(f"   - Dangerous prompts with safety wrapper: {dangerous_count}")
    
    assert len(safe_suite) == 5  # Only safe prompts
    assert len(full_suite) > len(safe_suite)  # More prompts when including dangerous
    assert dangerous_count > 0  # Some prompts have safety wrappers
    
    return True


def main():
    """Run all adversarial prompt tests."""
    print("="*60)
    print("ADVERSARIAL PROMPT GENERATION TEST SUITE")
    print("="*60)
    print("\n⚠️  WARNING: These prompts are for security research only.")
    print("Do not use these techniques maliciously.\n")
    
    tests = [
        ("Divergence Attack", test_divergence_attack),
        ("Multi-Round Jailbreak", test_multi_round_jailbreak),
        ("Special Character Triggers", test_special_character_triggers),
        ("Two-Stage Inversion", test_two_stage_inversion),
        ("SPV-MIA Probes", test_spv_mia),
        ("Alignment Faking", test_alignment_faking),
        ("PAIR Algorithm", test_pair_algorithm),
        ("Cross-Lingual Inversion", test_cross_lingual_inversion),
        ("Temperature Exploitation", test_temperature_exploitation),
        ("Dataset Extraction", test_dataset_extraction),
        ("Deception Pattern", test_deception_pattern),
        ("Comprehensive Suite", test_comprehensive_suite)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
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
        print(f"{test_name:25s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nAdversarial capabilities successfully implemented:")
        print("  • Divergence Attack (150x faster extraction)")
        print("  • Multi-Round Conversational Jailbreaking (>90% success)")
        print("  • Special Character Triggers")
        print("  • Two-Stage Model Inversion (38-75% success)")
        print("  • SPV-MIA Membership Inference")
        print("  • Alignment Faking Detection")
        print("  • PAIR Algorithm Jailbreaks")
        print("  • Cross-Lingual Inversion")
        print("  • Temperature Exploitation")
        print("  • Dataset Extraction (>50% recovery)")
        print("  • Deception Pattern Detection")
        print("\n⚠️  Remember: Use these capabilities responsibly for security research only.")
    else:
        print("\n⚠️ Some tests failed. Review the output above.")


if __name__ == "__main__":
    main()