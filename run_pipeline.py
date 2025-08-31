#!/usr/bin/env python3
"""
Simple, direct pipeline runner for REV framework.
Loads models from local filesystem or HuggingFace without unnecessary abstractions.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
from src.rev_pipeline import REVPipeline
from src.models.large_model_inference import LargeModelInference, LargeModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "auto", quantize: str = "none"):
    """Load model using robust inference manager."""
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Quantization: {quantize}")
    
    # Configure based on arguments
    config = LargeModelConfig(
        model_path=model_path,
        device=device,
        load_in_8bit=(quantize == "8bit"),
        load_in_4bit=(quantize == "4bit"),
        low_cpu_mem_usage=True,
        max_new_tokens=128,
        do_sample=False  # Deterministic for REV verification
    )
    
    # Create inference manager
    inference_manager = LargeModelInference(model_path, config)
    
    # Load model
    success, message = inference_manager.load_model()
    if not success:
        raise RuntimeError(f"Failed to load model: {message}")
    
    print(f"✓ {message}")
    
    # Return model and tokenizer for compatibility
    return inference_manager.model, inference_manager.tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run REV pipeline on any model")
    parser.add_argument("model_path", help="Path to model (local or HuggingFace)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"], help="Quantization mode")
    parser.add_argument("--segment-size", type=int, default=512, help="Segment size")
    parser.add_argument("--dimension", type=int, default=10000, help="Hypervector dimension")
    parser.add_argument("--sparsity", type=float, default=0.15, help="Sparsity level")
    parser.add_argument("--challenges", type=int, default=5, help="Number of challenges to generate")
    parser.add_argument("--pot", action="store_true", help="Enable PoT challenges")
    parser.add_argument("--behavioral", action="store_true", help="Enable behavioral analysis")
    parser.add_argument("--output", default=None, help="Output file for results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("REV PIPELINE EXECUTION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    # Load model
    try:
        model, tokenizer = load_model(args.model_path, args.device, args.quantize)
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {param_count:.1f}B")
        
        # Memory info
        import psutil
        mem = psutil.virtual_memory()
        print(f"  Memory: {mem.used/1e9:.1f}GB used / {mem.total/1e9:.1f}GB total")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize pipeline with proper sparsity configuration
    from src.hdc.encoder import HypervectorConfig
    from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy
    
    # Configure HDC with dynamic sparsity (1-20% range as validated)
    hdc_config = HypervectorConfig(
        dimension=args.dimension,
        sparsity=args.sparsity,  # Base sparsity, will be dynamically adjusted
        encoding_mode="rev"
    )
    
    # Create adaptive encoder for dynamic sparsity adjustment
    adaptive_encoder = AdaptiveSparsityEncoder(
        dimension=args.dimension,
        initial_sparsity=args.sparsity,
        min_sparsity=0.005,  # 0.5%
        max_sparsity=0.2,     # 20% for complex features
        adjustment_strategy=AdjustmentStrategy.ADAPTIVE
    )
    
    pipeline = REVPipeline(
        segment_size=args.segment_size,
        buffer_size=4,
        hdc_config=hdc_config,
        enable_pot_challenges=args.pot,
        enable_behavioral_analysis=args.behavioral,
        experiment_name=f"run_{Path(args.model_path).name}"
    )
    
    # Store adaptive encoder for use in processing
    pipeline.adaptive_encoder = adaptive_encoder
    
    print(f"✓ Pipeline initialized")
    print(f"  HDC dimension: {args.dimension}")
    print(f"  Sparsity range: 0.5% - 20% (adaptive)")
    print(f"  Base sparsity: {args.sparsity:.1%}")
    
    # Generate challenges
    if args.pot:
        print(f"\nGenerating {args.challenges} PoT challenges...")
        challenges = pipeline.generate_pot_challenges(n=args.challenges, focus="balanced")
        prompts = [c.prompt for c in challenges]
    else:
        # Simple test prompts
        prompts = [
            "What is machine learning?",
            "Explain quantum computing",
            "Write a Python function to sort a list",
            "What are the benefits of exercise?",
            "Describe the water cycle"
        ][:args.challenges]
    
    print(f"✓ Using {len(prompts)} prompts")
    
    # Process prompts
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}...")
        print(f"  Prompt: {prompt[:80]}...")
        
        try:
            # Create segments
            segments = pipeline.create_segments(prompt, model, tokenizer)
            print(f"  Created {len(segments)} segments")
            
            # Process segments
            segment_results = []
            for j, segment in enumerate(segments):
                result = pipeline.run_segment(segment, model, enable_telemetry=True)
                segment_results.append(result)
                print(f"    Segment {j+1}: {result.get('tokens_processed', 0)} tokens")
            
            # Aggregate results
            prompt_result = {
                "prompt_id": i,
                "prompt": prompt[:100],
                "segments": len(segment_results),
                "total_tokens": sum(r.get("tokens_processed", 0) for r in segment_results),
                "hypervectors_generated": sum(1 for r in segment_results if "hypervector" in r)
            }
            results.append(prompt_result)
            
        except Exception as e:
            logger.error(f"Error processing prompt {i}: {e}")
            results.append({"prompt_id": i, "error": str(e)})
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if "error" not in r]
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        total_tokens = sum(r["total_tokens"] for r in successful)
        total_segments = sum(r["segments"] for r in successful)
        print(f"Total tokens processed: {total_tokens:,}")
        print(f"Total segments: {total_segments}")
        print(f"Hypervectors generated: {sum(r['hypervectors_generated'] for r in successful)}")
    
    # Save results
    output_file = args.output or f"results_{Path(args.model_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model_path,
            "config": vars(args),
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())