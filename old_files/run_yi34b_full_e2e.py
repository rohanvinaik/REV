#!/usr/bin/env python3
"""
Full E2E Pipeline execution for Yi-34B model.
This runs the complete REV framework with all features integrated.
"""

import torch
import json
import logging
from datetime import datetime
from pathlib import Path

from src.rev_pipeline import REVPipeline
from src.yi34b_efficient_loader import Yi34BLoader
from src.challenges.pot_challenge_generator import PoTChallengeGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_full_e2e_pipeline():
    """Run the complete E2E pipeline on Yi-34B."""
    
    print("="*80)
    print("YI-34B FULL E2E PIPELINE EXECUTION")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    # Step 1: Load Yi-34B model
    print("Step 1: Loading Yi-34B model...")
    loader = Yi34BLoader(
        model_path="/Users/rohanvinaik/LLM_models/yi-34b",
        memory_limit_gb=8,
        device="cpu"
    )
    
    model, tokenizer = loader.load_model()
    print(f"✓ Model loaded: {loader.get_model_info()['total_params'] / 1e9:.1f}B parameters")
    print(f"✓ Memory usage: {loader.get_memory_usage()['used_gb']:.1f}GB\n")
    
    # Step 2: Initialize REV Pipeline with all features
    print("Step 2: Initializing REV Pipeline...")
    pipeline = REVPipeline(
        segment_size=512,
        buffer_size=4,
        dimension=10000,
        sparsity=0.15,  # 15% for good semantic fingerprinting
        enable_pot_challenges=True,
        enable_behavioral_analysis=True,
        experiment_name="yi34b_full_e2e"
    )
    print("✓ Pipeline initialized with:")
    print("  - PoT challenge generation: ENABLED")
    print("  - Behavioral analysis: ENABLED")
    print("  - Adaptive encoding: ENABLED")
    print("  - Sparsity: 15% (dynamic adjustment enabled)\n")
    
    # Step 3: Generate sophisticated PoT challenges
    print("Step 3: Generating PoT challenges...")
    challenges = pipeline.generate_pot_challenges(n=10, focus="balanced")
    print(f"✓ Generated {len(challenges)} sophisticated challenges")
    
    # Sample challenges
    print("\nSample challenges:")
    for i, challenge in enumerate(challenges[:3], 1):
        print(f"\n{i}. [{challenge.category}] {challenge.complexity}")
        print(f"   {challenge.prompt[:100]}...")
        print(f"   Expected divergence: {challenge.expected_divergence:.2f}")
    
    # Step 4: Run behavioral analysis
    print("\nStep 4: Running behavioral analysis...")
    behavioral_results = pipeline.run_behavioral_analysis(model, tokenizer)
    
    if behavioral_results:
        print("✓ Behavioral analysis complete:")
        print(f"  - Segments discovered: {len(behavioral_results.get('segments', []))}")
        print(f"  - Processing patterns identified: {len(behavioral_results.get('patterns', []))}")
        if 'variance_map' in behavioral_results:
            print(f"  - Variance map generated: {len(behavioral_results['variance_map'])} regions")
    
    # Step 5: Process challenges through pipeline
    print("\nStep 5: Processing challenges through pipeline...")
    
    results = []
    total_challenges = min(5, len(challenges))  # Process first 5 for efficiency
    
    for i, challenge in enumerate(challenges[:total_challenges], 1):
        print(f"\nProcessing challenge {i}/{total_challenges}:")
        print(f"  Category: {challenge.category}")
        print(f"  Complexity: {challenge.complexity}")
        
        try:
            # Create segments for the challenge
            segments = pipeline.create_segments(
                prompt=challenge.prompt,
                model=model,
                tokenizer=tokenizer
            )
            
            # Process segments
            segment_results = []
            for j, segment in enumerate(segments):
                print(f"  Processing segment {j+1}/{len(segments)}...", end='')
                
                # Run segment with telemetry
                result = pipeline.run_segment(
                    segment=segment,
                    model=model,
                    enable_telemetry=True
                )
                
                segment_results.append(result)
                print(" ✓")
            
            # Aggregate results
            challenge_result = {
                'challenge_id': i,
                'category': challenge.category,
                'complexity': challenge.complexity,
                'segments_processed': len(segment_results),
                'total_tokens': sum(r.get('tokens_processed', 0) for r in segment_results),
                'avg_time_ms': sum(r.get('time_ms', 0) for r in segment_results) / len(segment_results),
                'hypervector_generated': all('hypervector' in r for r in segment_results)
            }
            
            results.append(challenge_result)
            
            # Check memory
            memory_usage = loader.get_memory_usage()
            print(f"  Memory: {memory_usage['used_gb']:.1f}GB / {memory_usage['limit_gb']:.1f}GB")
            
            if memory_usage['used_gb'] > memory_usage['limit_gb'] * 0.9:
                print("  ⚠️ Memory usage high, clearing cache...")
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing challenge {i}: {e}")
            results.append({
                'challenge_id': i,
                'error': str(e)
            })
    
    # Step 6: Generate final report
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    # Calculate statistics
    successful_runs = [r for r in results if 'error' not in r]
    failed_runs = [r for r in results if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total challenges processed: {len(results)}")
    print(f"  Successful: {len(successful_runs)}")
    print(f"  Failed: {len(failed_runs)}")
    
    if successful_runs:
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_runs)
        total_time = sum(r.get('avg_time_ms', 0) * r.get('segments_processed', 1) for r in successful_runs)
        
        print(f"\nPerformance:")
        print(f"  Total tokens processed: {total_tokens:,}")
        print(f"  Average throughput: {total_tokens / (total_time / 1000) if total_time > 0 else 0:.1f} tokens/s")
        print(f"  Hypervector generation: {sum(1 for r in successful_runs if r.get('hypervector_generated'))}/{len(successful_runs)}")
    
    # Save results
    output_file = f"yi34b_e2e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'Yi-34B',
            'pipeline_config': {
                'segment_size': 512,
                'buffer_size': 4,
                'dimension': 10000,
                'sparsity': 0.15,
                'pot_challenges': True,
                'behavioral_analysis': True
            },
            'results': results,
            'behavioral_analysis': behavioral_results if behavioral_results else None
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_full_e2e_pipeline()
        print("\n🎉 Full E2E pipeline execution successful!")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline execution failed: {e}")
        import sys
        sys.exit(1)