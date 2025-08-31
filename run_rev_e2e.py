#!/usr/bin/env python3
"""
Complete E2E REV pipeline runner for any large model.
Properly integrates all REV components including adaptive sparsity.
"""

import argparse
import json
import time
import torch
import logging
from datetime import datetime
from pathlib import Path

# REV components
from src.models.large_model_inference import LargeModelInference, LargeModelConfig
from src.rev_pipeline import REVPipeline
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy
from src.challenges.pot_challenge_generator import PoTChallengeGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run complete REV E2E pipeline")
    parser.add_argument("model", help="Model path or HuggingFace ID")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--challenges", type=int, default=3, help="Number of challenges")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print("="*80)
    print("REV FRAMEWORK - COMPLETE E2E PIPELINE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Quantization: {args.quantize}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "config": vars(args),
        "stages": {}
    }
    
    # Stage 1: Load Model
    print("\n[Stage 1/4] Loading Model...")
    start = time.time()
    
    config = LargeModelConfig(
        model_path=args.model,
        device=args.device,
        load_in_8bit=(args.quantize == "8bit"),
        load_in_4bit=(args.quantize == "4bit"),
        low_cpu_mem_usage=True,
        max_new_tokens=100,
        do_sample=False  # Deterministic for verification
    )
    
    inference = LargeModelInference(args.model, config)
    success, message = inference.load_model()
    
    if not success:
        print(f"❌ Failed to load model: {message}")
        return 1
    
    print(f"✅ {message}")
    results["stages"]["model_loading"] = {
        "success": True,
        "time": time.time() - start,
        "message": message,
        "model_info": inference.model_info
    }
    
    # Stage 2: Initialize REV Pipeline
    print("\n[Stage 2/4] Initializing REV Pipeline...")
    start = time.time()
    
    # Configure hyperdimensional computing
    hdc_config = HypervectorConfig(
        dimension=10000,
        sparsity=0.15,  # 15% base sparsity
        encoding_mode="rev"
    )
    
    # Create adaptive encoder
    adaptive_encoder = AdaptiveSparsityEncoder(
        dimension=10000,
        initial_sparsity=0.01,
        min_sparsity=0.005,
        max_sparsity=0.2,
        adjustment_strategy=AdjustmentStrategy.ADAPTIVE
    )
    
    # Initialize pipeline
    pipeline = REVPipeline(
        segment_size=512,
        buffer_size=4,
        hdc_config=hdc_config,
        enable_pot_challenges=True,
        enable_behavioral_analysis=True,
        experiment_name=f"e2e_{Path(args.model).name}"
    )
    
    print("✅ Pipeline initialized with:")
    print(f"   - HDC dimension: 10000")
    print(f"   - Adaptive sparsity: 0.5% - 20%")
    print(f"   - PoT challenges: ENABLED")
    print(f"   - Behavioral analysis: ENABLED")
    
    results["stages"]["pipeline_init"] = {
        "success": True,
        "time": time.time() - start,
        "hdc_dimension": 10000,
        "sparsity_range": "0.5% - 20%"
    }
    
    # Stage 3: Generate PoT Challenges
    print("\n[Stage 3/4] Generating PoT Challenges...")
    start = time.time()
    
    challenges = pipeline.generate_pot_challenges(n=args.challenges, focus="balanced")
    print(f"✅ Generated {len(challenges)} sophisticated challenges")
    
    for i, challenge in enumerate(challenges[:2], 1):
        print(f"\nChallenge {i}:")
        print(f"  Category: {challenge.category}")
        print(f"  Complexity: {challenge.complexity}")
        print(f"  Prompt preview: {challenge.prompt[:80]}...")
    
    results["stages"]["challenge_generation"] = {
        "success": True,
        "time": time.time() - start,
        "challenges_generated": len(challenges)
    }
    
    # Stage 4: Process Challenges
    print(f"\n[Stage 4/4] Processing {len(challenges)} Challenges...")
    start = time.time()
    
    challenge_results = []
    for i, challenge in enumerate(challenges, 1):
        print(f"\nProcessing challenge {i}/{len(challenges)}...")
        
        try:
            # Process with REV
            rev_result = inference.process_for_rev(
                prompt=challenge.prompt,
                extract_activations=True,
                hdc_encoder=pipeline.encoder,
                adaptive_encoder=adaptive_encoder
            )
            
            # Store results
            challenge_result = {
                "challenge_id": i,
                "category": challenge.category,
                "complexity": str(challenge.complexity),
                "success": rev_result["success"],
                "response_preview": rev_result.get("response", "")[:100] if rev_result.get("success") else None,
                "hypervector_generated": "hypervector" in rev_result or "adaptive_hypervector" in rev_result
            }
            
            if "hypervector_sparsity" in rev_result:
                challenge_result["hypervector_sparsity"] = rev_result["hypervector_sparsity"]
            
            if "adaptive_stats" in rev_result:
                challenge_result["adaptive_stats"] = rev_result["adaptive_stats"]
            
            challenge_results.append(challenge_result)
            
            if rev_result["success"]:
                print(f"  ✅ Success - Response: {rev_result['response'][:50]}...")
                if "adaptive_stats" in rev_result:
                    stats = rev_result["adaptive_stats"]
                    print(f"  Sparsity: {stats['final_sparsity']:.1%} (density: {stats['actual_density']:.3f})")
            else:
                print(f"  ❌ Failed: {rev_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing challenge {i}: {e}")
            challenge_results.append({
                "challenge_id": i,
                "success": False,
                "error": str(e)
            })
    
    results["stages"]["challenge_processing"] = {
        "success": True,
        "time": time.time() - start,
        "challenges_processed": len(challenge_results),
        "successful": sum(1 for r in challenge_results if r.get("success")),
        "results": challenge_results
    }
    
    # Summary
    print("\n" + "="*80)
    print("E2E PIPELINE COMPLETE")
    print("="*80)
    
    total_time = sum(stage.get("time", 0) for stage in results["stages"].values())
    successful = sum(1 for r in challenge_results if r.get("success"))
    
    print(f"Total time: {total_time:.1f}s")
    print(f"Challenges processed: {successful}/{len(challenges)}")
    print(f"Model parameters: {results['stages']['model_loading']['model_info']['parameters']/1e9:.1f}B")
    
    # Save results
    output_file = args.output or f"rev_e2e_{Path(args.model).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Cleanup
    inference.cleanup()
    print("✅ Model memory freed")
    
    return 0


if __name__ == "__main__":
    exit(main())