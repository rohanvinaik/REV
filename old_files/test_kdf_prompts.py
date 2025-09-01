#!/usr/bin/env python3
"""
Test enhanced KDF-based challenge generation from Sections 4.2 and 5.2.
"""

import hashlib
import hmac
import json
import os
from typing import List, Dict, Any

from src.challenges.kdf_prompts import (
    EnhancedKDFPromptGenerator,
    DomainType,
    PublicTranscript,
    KDFPromptGenerator,
    make_prompt_generator,
    integrate_with_deterministic_generator
)
from src.challenges.prompt_generator import DeterministicPromptGenerator


def test_hmac_seed_generation():
    """Test HMAC-based seed generation as per Section 4.2."""
    print("Testing HMAC-based Seed Generation (Section 4.2)")
    print("=" * 60)
    
    # Create generator
    master_key = b"test_master_key_12345"
    run_id = "test_run_001"
    gen = EnhancedKDFPromptGenerator(master_key, run_id)
    
    # Test seed generation follows spec: seed_i = HMAC(key, f"{run_id}:{i}")
    for i in range(5):
        seed = gen._generate_seed(i)
        
        # Verify seed is deterministic
        expected_message = f"{run_id}:{i}".encode('utf-8')
        expected_seed = hmac.new(master_key, expected_message, hashlib.sha256).digest()
        
        assert seed == expected_seed, f"Seed mismatch at index {i}"
        print(f"  Index {i}: seed = {seed.hex()[:16]}... (deterministic ✓)")
    
    # Test different run_ids produce different seeds
    gen2 = EnhancedKDFPromptGenerator(master_key, "different_run")
    seed1 = gen._generate_seed(0)
    seed2 = gen2._generate_seed(0)
    assert seed1 != seed2, "Different run_ids should produce different seeds"
    print("  Different run_ids produce different seeds ✓")
    
    # Test different keys produce different seeds
    gen3 = EnhancedKDFPromptGenerator(b"different_key", run_id)
    seed3 = gen3._generate_seed(0)
    assert seed1 != seed3, "Different keys should produce different seeds"
    print("  Different keys produce different seeds ✓")
    
    print("\n✓ HMAC-based seed generation tests passed")


def test_template_synthesis():
    """Test template-based prompt synthesis with domains."""
    print("\nTesting Template-based Prompt Synthesis (Section 5.2)")
    print("=" * 60)
    
    gen = make_prompt_generator(b"test_key", "test_namespace")
    
    # Test all domains are represented
    domains_seen = set()
    challenges = gen.generate_challenge_set(100)
    
    for challenge in challenges:
        domains_seen.add(challenge["domain"])
    
    print(f"  Domains generated: {sorted(domains_seen)}")
    assert len(domains_seen) >= 3, "Should generate multiple domains"
    
    # Test domain-specific generation
    for domain in [DomainType.MATH, DomainType.REASONING, DomainType.CODING]:
        challenge = gen.generate_challenge(domain=domain)
        assert challenge["domain"] == domain.value, f"Domain mismatch for {domain}"
        print(f"  {domain.value}: {challenge['prompt'][:60]}...")
    
    # Test difficulty ranges
    easy_challenges = []
    hard_challenges = []
    
    for i in range(20):
        easy = gen.generate_challenge(index=100+i, difficulty_range=(1, 2))
        hard = gen.generate_challenge(index=200+i, difficulty_range=(4, 5))
        easy_challenges.append(easy["difficulty"])
        hard_challenges.append(hard["difficulty"])
    
    avg_easy = sum(easy_challenges) / len(easy_challenges)
    avg_hard = sum(hard_challenges) / len(hard_challenges)
    
    print(f"\n  Average easy difficulty: {avg_easy:.1f}")
    print(f"  Average hard difficulty: {avg_hard:.1f}")
    assert avg_easy <= 2.0, "Easy challenges should have low difficulty"
    assert avg_hard >= 4.0, "Hard challenges should have high difficulty"
    
    print("\n✓ Template synthesis tests passed")


def test_adversarial_variants():
    """Test adversarial variant generation."""
    print("\nTesting Adversarial Variant Generation")
    print("=" * 60)
    
    gen = make_prompt_generator(b"test_key", "adversarial_test")
    
    # Generate mix with adversarial variants
    challenges = gen.generate_challenge_set(
        n_challenges=50,
        adversarial_ratio=0.3
    )
    
    adversarial_count = sum(1 for c in challenges if c.get("is_adversarial", False))
    ratio = adversarial_count / len(challenges)
    
    print(f"  Generated {adversarial_count}/{len(challenges)} adversarial ({ratio:.1%})")
    # More tolerant range due to randomness
    assert 0.05 <= ratio <= 0.5, f"Adversarial ratio {ratio} out of expected range"
    
    # Test explicit adversarial generation
    adv_challenge = gen.generate_challenge(
        domain=DomainType.MATH,
        use_adversarial=True
    )
    
    if adv_challenge["is_adversarial"]:
        print(f"  Adversarial prompt: {adv_challenge['prompt'][:80]}...")
        assert "error" in adv_challenge["prompt"].lower() or \
               "subtle" in adv_challenge["prompt"].lower(), \
               "Adversarial variant should mention errors"
    
    # Test adversarial domain
    adv_domain_challenge = gen.generate_challenge(
        domain=DomainType.ADVERSARIAL
    )
    assert adv_domain_challenge["domain"] == "adversarial"
    print(f"  Adversarial domain: {adv_domain_challenge['prompt'][:80]}...")
    
    print("\n✓ Adversarial variant tests passed")


def test_public_transcript():
    """Test public transcript generation with Merkle tree."""
    print("\nTesting Public Transcript Generation (Section 4.2)")
    print("=" * 60)
    
    master_key = b"transcript_test_key"
    gen = EnhancedKDFPromptGenerator(master_key, "transcript_run", "1.0.0")
    
    # Generate challenge set
    challenges = gen.generate_challenge_set(
        n_challenges=20,
        adversarial_ratio=0.1,
        difficulty_range=(1, 5)
    )
    
    # Create public transcript
    decoding_policy = {
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0
    }
    
    transcript = gen.create_public_transcript(challenges, decoding_policy)
    
    # Verify transcript contents
    print(f"  Run ID: {transcript.run_id}")
    print(f"  Challenge count: {transcript.challenge_count}")
    print(f"  Domains: {transcript.domains}")
    print(f"  Difficulty range: {transcript.difficulty_range}")
    print(f"  Merkle root: {transcript.merkle_root[:32]}...")
    print(f"  Key commitment: {transcript.key_commitment[:32]}...")
    
    assert transcript.run_id == "transcript_run"
    assert transcript.challenge_count == 20
    assert transcript.version == "1.0.0"
    assert len(transcript.merkle_root) == 64  # SHA256 hex
    assert len(transcript.key_commitment) == 64
    
    # Verify key commitment
    expected_commitment = hashlib.sha256(master_key).hexdigest()
    assert transcript.key_commitment == expected_commitment, "Key commitment mismatch"
    
    # Test Merkle root computation
    merkle_root = gen.compute_merkle_root(challenges)
    assert merkle_root == transcript.merkle_root, "Merkle root mismatch"
    
    # Test challenge verification
    for challenge in challenges[:5]:
        is_valid = gen.verify_challenge(challenge, transcript)
        assert is_valid, f"Challenge {challenge['index']} should be valid"
        print(f"  Challenge {challenge['index']}: verified ✓")
    
    # Test invalid challenge detection
    fake_challenge = challenges[0].copy()
    fake_challenge["seed_hex"] = "0" * 64
    is_valid = gen.verify_challenge(fake_challenge, transcript)
    assert not is_valid, "Fake challenge should be invalid"
    print("  Fake challenge: rejected ✓")
    
    print("\n✓ Public transcript tests passed")


def test_canonicalization():
    """Test canonicalization for reproducibility."""
    print("\nTesting Canonicalization for Reproducibility")
    print("=" * 60)
    
    gen = make_prompt_generator(b"canon_key", "canon_run")
    
    # Generate challenges
    challenges = []
    for i in range(10):
        challenge = gen.generate_challenge(index=i)
        challenges.append(challenge)
    
    # Verify canonical forms are deterministic
    for i, challenge in enumerate(challenges):
        canonical = challenge["canonical_form"]
        
        # Parse canonical form
        canonical_dict = json.loads(canonical)
        
        # Check required fields
        assert "version" in canonical_dict
        assert "template_id" in canonical_dict
        assert "domain" in canonical_dict
        assert "slots" in canonical_dict
        
        # Verify ordering is stable
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        assert canonical == canonical_str, "Canonical form should be stable"
        
        print(f"  Challenge {i}: canonical length = {len(canonical)} chars")
    
    # Test version tracking
    gen_v2 = EnhancedKDFPromptGenerator(b"canon_key", "canon_run", "2.0.0")
    challenge_v2 = gen_v2.generate_challenge(index=0)
    canonical_v2 = json.loads(challenge_v2["canonical_form"])
    assert canonical_v2["version"] == "2.0.0", "Version should be tracked"
    print(f"  Version tracking: v1={challenges[0]['canonical_form'][:20]}...")
    print(f"  Version tracking: v2={challenge_v2['canonical_form'][:20]}...")
    
    print("\n✓ Canonicalization tests passed")


def test_determinism():
    """Test determinism properties of challenge generation."""
    print("\nTesting Determinism Properties")
    print("=" * 60)
    
    key = b"determinism_test_key"
    namespace = "determinism_test"
    
    # Create two generators with same key and namespace
    gen1 = make_prompt_generator(key, namespace)
    gen2 = make_prompt_generator(key, namespace)
    
    # Generate challenges from both
    challenges1 = gen1.generate_challenge_set(20)
    challenges2 = gen2.generate_challenge_set(20)
    
    # Verify exact match
    for i in range(20):
        c1, c2 = challenges1[i], challenges2[i]
        assert c1["prompt"] == c2["prompt"], f"Prompt mismatch at index {i}"
        assert c1["domain"] == c2["domain"], f"Domain mismatch at index {i}"
        assert c1["canonical_form"] == c2["canonical_form"], f"Canonical mismatch at index {i}"
    
    print("  Same key + namespace = identical challenges ✓")
    
    # Test with different namespace
    gen3 = make_prompt_generator(key, "different_namespace")
    challenges3 = gen3.generate_challenge_set(20)
    
    different_count = sum(
        1 for i in range(20) 
        if challenges1[i]["prompt"] != challenges3[i]["prompt"]
    )
    
    assert different_count >= 18, "Different namespace should produce different challenges"
    print(f"  Different namespace: {different_count}/20 challenges differ ✓")
    
    # Test unpredictability
    prompts_seen = set()
    for challenge in challenges1:
        prompts_seen.add(challenge["prompt"])
    
    # Allow some duplicates due to limited template options
    assert len(prompts_seen) >= 15, f"Most prompts should be unique, got {len(prompts_seen)}/20"
    print(f"  Unpredictability: {len(prompts_seen)}/20 unique prompts ✓")
    
    print("\n✓ Determinism tests passed")


def test_integration_with_deterministic_generator():
    """Test integration with existing DeterministicPromptGenerator."""
    print("\nTesting Integration with DeterministicPromptGenerator")
    print("=" * 60)
    
    master_key = b"integration_test_key"
    
    # Create enhanced generator
    enhanced_gen = make_prompt_generator(master_key, "integration_test")
    
    # Create integrated generator
    integrated = integrate_with_deterministic_generator(master_key, enhanced_gen)
    
    # Test it has the expected interface
    assert hasattr(integrated, "generate_challenges")
    assert isinstance(integrated, DeterministicPromptGenerator)
    
    # Generate challenges using integrated interface
    challenges = integrated.generate_challenges(
        ref_model_id="gpt-4",
        cand_model_id="claude-3",
        n=10,
        namespace="test_namespace",
        seed=42
    )
    
    print(f"  Generated {len(challenges)} challenges via integrated interface")
    
    # Verify challenge format
    for i, challenge in enumerate(challenges[:3]):
        assert "id" in challenge
        assert "type" in challenge
        assert "content" in challenge
        assert "metadata" in challenge
        
        print(f"  Challenge {i}: {challenge['id']}")
        print(f"    Content: {challenge['content'][:60]}...")
        print(f"    Metadata keys: {list(challenge['metadata'].keys())}")
    
    # Test backward compatibility
    old_gen = KDFPromptGenerator(master_key, "backward_compat_test")
    
    # Should work as callable
    prompt1 = old_gen()
    prompt2 = old_gen.generate_prompt()
    
    assert isinstance(prompt1, str)
    assert isinstance(prompt2, str)
    assert prompt1 != prompt2  # Different indices
    
    print(f"\n  Backward compatible prompt 1: {prompt1[:60]}...")
    print(f"  Backward compatible prompt 2: {prompt2[:60]}...")
    
    print("\n✓ Integration tests passed")


def test_domain_distribution():
    """Test domain distribution in challenge sets."""
    print("\nTesting Domain Distribution Control")
    print("=" * 60)
    
    gen = make_prompt_generator(b"dist_test_key", "dist_test")
    
    # Test custom distribution
    distribution = {
        DomainType.MATH: 0.4,
        DomainType.CODING: 0.3,
        DomainType.REASONING: 0.2,
        DomainType.KNOWLEDGE: 0.1
    }
    
    challenges = gen.generate_challenge_set(
        n_challenges=100,
        domain_distribution=distribution
    )
    
    # Count domains
    domain_counts = {}
    for challenge in challenges:
        domain = challenge["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("  Domain distribution (100 challenges):")
    for domain, count in sorted(domain_counts.items()):
        expected = distribution.get(DomainType(domain), 0) * 100
        print(f"    {domain}: {count} (expected ~{expected:.0f})")
    
    # Verify distribution is approximately correct (with some tolerance)
    math_ratio = domain_counts.get("math", 0) / 100
    assert 0.3 <= math_ratio <= 0.5, "Math distribution should be ~40%"
    
    print("\n✓ Domain distribution tests passed")


def test_merkle_tree_properties():
    """Test Merkle tree construction and properties."""
    print("\nTesting Merkle Tree Properties")
    print("=" * 60)
    
    gen = make_prompt_generator(b"merkle_key", "merkle_test")
    
    # Test empty set
    empty_root = gen.compute_merkle_root([])
    assert empty_root == hashlib.sha256(b"empty").hexdigest()
    print(f"  Empty set root: {empty_root[:32]}...")
    
    # Test single challenge
    single = gen.generate_challenge_set(1)
    single_root = gen.compute_merkle_root(single)
    assert len(single_root) == 64
    print(f"  Single challenge root: {single_root[:32]}...")
    
    # Test power of 2
    power2 = gen.generate_challenge_set(8)
    power2_root = gen.compute_merkle_root(power2)
    print(f"  8 challenges root: {power2_root[:32]}...")
    
    # Test odd number
    odd = gen.generate_challenge_set(7)
    odd_root = gen.compute_merkle_root(odd)
    print(f"  7 challenges root: {odd_root[:32]}...")
    
    # Verify different sets have different roots
    assert single_root != power2_root != odd_root
    print("  Different sets produce different roots ✓")
    
    # Test root changes if challenge changes
    modified = power2.copy()
    modified[0] = modified[0].copy()
    modified[0]["canonical_form"] = '{"modified": true}'
    modified_root = gen.compute_merkle_root(modified)
    
    assert modified_root != power2_root
    print("  Modified challenge changes root ✓")
    
    print("\n✓ Merkle tree tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Enhanced KDF Challenge Generation (Sections 4.2, 5.2)")
    print("=" * 70)
    
    test_hmac_seed_generation()
    test_template_synthesis()
    test_adversarial_variants()
    test_public_transcript()
    test_canonicalization()
    test_determinism()
    test_integration_with_deterministic_generator()
    test_domain_distribution()
    test_merkle_tree_properties()
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()