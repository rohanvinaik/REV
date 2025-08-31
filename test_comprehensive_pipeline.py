#!/usr/bin/env python3
"""
Test the comprehensive REV pipeline with PoT challenge generation and behavioral analysis.
This demonstrates the fully integrated main pipeline, not specific to any single model.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')

from src.rev_pipeline import REVPipeline, ExecutionPolicy
from src.hdc.encoder import HypervectorConfig
from src.challenges.pot_challenge_generator import (
    PoTChallengeGenerator, 
    ChallengeComplexity,
    ChallengeCategory
)


def test_pot_challenge_generation():
    """Test PoT-style challenge generation."""
    print("="*80)
    print("TEST 1: PoT Challenge Generation")
    print("="*80)
    
    generator = PoTChallengeGenerator(
        enable_info_selection=True,
        min_complexity=ChallengeComplexity.MODERATE
    )
    
    # Generate challenges with different focuses
    print("\n1. Coverage-focused challenges (broad testing):")
    coverage_challenges = generator.generate_verification_challenges(n=5, focus="coverage")
    for i, challenge in enumerate(coverage_challenges[:3]):
        print(f"\n  Challenge {i+1}:")
        print(f"    Category: {challenge.category.value}")
        print(f"    Complexity: {challenge.complexity.name}")
        print(f"    Expected Divergence: {challenge.expected_divergence:.2f}")
        print(f"    Prompt: {challenge.prompt[:100]}...")
    
    print("\n2. Separation-focused challenges (high discrimination):")
    separation_challenges = generator.generate_verification_challenges(n=5, focus="separation")
    for i, challenge in enumerate(separation_challenges[:3]):
        print(f"\n  Challenge {i+1}:")
        print(f"    Category: {challenge.category.value}")
        print(f"    Complexity: {challenge.complexity.name}")
        print(f"    Expected Divergence: {challenge.expected_divergence:.2f}")
        print(f"    Prompt: {challenge.prompt[:100]}...")
    
    print("\n3. Balanced challenges (coverage + separation):")
    balanced_challenges = generator.generate_verification_challenges(n=5, focus="balanced")
    
    # Test information-theoretic selection
    print("\n4. Information-theoretic selection:")
    if generator.selector:
        selected = generator.selector.select_challenges(
            balanced_challenges, 
            n=3, 
            coverage_weight=0.5
        )
        print(f"  Selected {len(selected)} challenges from {len(balanced_challenges)} candidates")
    
    print("\n✓ PoT challenge generation test complete")
    return generator


def test_comprehensive_pipeline():
    """Test the comprehensive REV pipeline with all features."""
    print("\n" + "="*80)
    print("TEST 2: Comprehensive REV Pipeline")
    print("="*80)
    
    # Initialize pipeline with all features enabled
    hdc_config = HypervectorConfig(
        dimension=10000,
        sparsity=0.1,  # Using proper 10% sparsity
        encoding_mode="rev"
    )
    
    pipeline = REVPipeline(
        segment_size=512,
        buffer_size=4,
        hdc_config=hdc_config,
        enable_pot_challenges=True,
        enable_behavioral_analysis=True,
        experiment_name="comprehensive_test"
    )
    
    print("\nPipeline Configuration:")
    print(f"  HDC Dimension: {hdc_config.dimension}")
    print(f"  HDC Sparsity: {hdc_config.sparsity:.1%}")
    print(f"  Segment Size: {pipeline.segment_size}")
    print(f"  Memory Limit: {pipeline.memory_limit_mb} MB")
    print(f"  PoT Challenges: Enabled")
    print(f"  Behavioral Analysis: Enabled")
    
    # Test challenge generation through pipeline
    print("\nGenerating PoT challenges through pipeline...")
    challenges = pipeline.generate_pot_challenges(n=10, focus="balanced")
    print(f"  Generated {len(challenges)} challenges")
    
    # Display challenge statistics
    complexity_dist = {}
    category_dist = {}
    for c in challenges:
        complexity_dist[c.complexity.name] = complexity_dist.get(c.complexity.name, 0) + 1
        category_dist[c.category.value] = category_dist.get(c.category.value, 0) + 1
    
    print("\n  Complexity distribution:")
    for comp, count in complexity_dist.items():
        print(f"    {comp}: {count}")
    
    print("\n  Category distribution:")
    for cat, count in category_dist.items():
        print(f"    {cat}: {count}")
    
    # Test telemetry
    print("\nTesting telemetry tracking...")
    from src.rev_pipeline import SegmentTelemetry
    telemetry = SegmentTelemetry(
        segment_id=1,
        alloc_mb=100,
        peak_mb=150,
        t_ms=250,
        tokens_processed=512
    )
    pipeline.telemetry_records.append(telemetry)
    
    summary = pipeline.get_telemetry_summary()
    print(f"  Telemetry summary: {summary}")
    
    # Test experiment tracking
    print("\nExperiment tracking:")
    print(f"  Experiment ID: {pipeline.experiment_results['experiment_id']}")
    print(f"  Experiment Name: {pipeline.experiment_name}")
    print(f"  Configuration tracked: {len(pipeline.experiment_results['configuration'])} parameters")
    
    print("\n✓ Comprehensive pipeline test complete")
    return pipeline


def test_behavioral_probes():
    """Test behavioral probe generation for segmentation."""
    print("\n" + "="*80)
    print("TEST 3: Behavioral Probe Generation")
    print("="*80)
    
    generator = PoTChallengeGenerator()
    probes = generator.generate_behavioral_probes()
    
    print("\nBehavioral probe categories:")
    for category, probe_list in probes.items():
        print(f"\n  {category.upper()} ({len(probe_list)} probes):")
        for probe in probe_list[:2]:
            print(f"    - {probe}")
    
    total_probes = sum(len(p) for p in probes.values())
    print(f"\nTotal behavioral probes: {total_probes}")
    
    print("\n✓ Behavioral probe generation test complete")
    return probes


def test_challenge_export():
    """Test challenge export functionality."""
    print("\n" + "="*80)
    print("TEST 4: Challenge Export")
    print("="*80)
    
    generator = PoTChallengeGenerator()
    challenges = generator.generate_challenges(n=5)
    
    # Export as JSON
    json_export = generator.export_challenges(challenges, format="json")
    print("\nJSON export (first 500 chars):")
    print(json_export[:500])
    
    # Export as dict
    dict_export = generator.export_challenges(challenges, format="dict")
    print(f"\nDict export: {len(dict_export)} challenges")
    
    print("\n✓ Challenge export test complete")


def test_information_theoretic_selection():
    """Test the information-theoretic challenge selection."""
    print("\n" + "="*80)
    print("TEST 5: Information-Theoretic Selection")
    print("="*80)
    
    generator = PoTChallengeGenerator(enable_info_selection=True)
    
    # Generate many candidates
    candidates = generator.generate_challenges(n=20)
    
    print(f"\nGenerated {len(candidates)} candidate challenges")
    
    # Test selection with different coverage weights
    for coverage_weight in [0.2, 0.5, 0.8]:
        selected = generator.selector.select_challenges(
            candidates, 
            n=5, 
            coverage_weight=coverage_weight
        )
        
        print(f"\nCoverage weight = {coverage_weight}:")
        print(f"  Selected {len(selected)} challenges")
        
        # Analyze selection
        categories = set(c.category for c in selected)
        complexities = set(c.complexity for c in selected)
        avg_divergence = np.mean([c.expected_divergence for c in selected])
        
        print(f"  Unique categories: {len(categories)}")
        print(f"  Unique complexities: {len(complexities)}")
        print(f"  Avg expected divergence: {avg_divergence:.3f}")
    
    # Test score updates
    print("\nTesting score updates...")
    for challenge in selected[:3]:
        observed_divergence = np.random.uniform(0.5, 0.9)
        generator.selector.update_scores(challenge, observed_divergence)
    
    print(f"  Updated scores for {len(generator.selector.category_scores)} categories")
    
    print("\n✓ Information-theoretic selection test complete")


def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE REV PIPELINE TEST SUITE")
    print("="*80)
    print("\nThis demonstrates the fully integrated main pipeline with:")
    print("- PoT-style challenge generation")
    print("- Behavioral analysis integration")
    print("- Information-theoretic selection")
    print("- Comprehensive experiment tracking")
    
    # Run tests
    generator = test_pot_challenge_generation()
    pipeline = test_comprehensive_pipeline()
    probes = test_behavioral_probes()
    test_challenge_export()
    test_information_theoretic_selection()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\n✓ All tests completed successfully!")
    print("\nKey Features Demonstrated:")
    print("1. PoT challenge generation with multiple complexity levels")
    print("2. Coverage-separation trade-off optimization")
    print("3. Information-theoretic challenge selection")
    print("4. Behavioral probe generation for segmentation")
    print("5. Comprehensive pipeline integration")
    print("6. Experiment tracking and telemetry")
    
    print("\nThe main REV pipeline is now enhanced with:")
    print("- Sophisticated challenge generation (not just simple prompts)")
    print("- Behavioral segmentation analysis")
    print("- Proper sparse encoding (fixed from 100% dense bug)")
    print("- Comprehensive statistical reporting")
    print("- Memory-bounded execution with telemetry")


if __name__ == "__main__":
    main()