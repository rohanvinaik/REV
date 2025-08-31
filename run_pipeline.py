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
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from src.rev_pipeline import REVPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cpu"):
    """Load model from local path or HuggingFace."""
    print(f"Loading model from: {model_path}")
    
    # Check if it's a local path
    if Path(model_path).exists():
        print("Loading from local filesystem...")
        # Use int8 quantization for large models on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Don't use auto device map on CPU
            low_cpu_mem_usage=True,
            load_in_8bit=False,  # Disable 8bit for now
            offload_folder="offload",  # Offload to disk if needed
            offload_state_dict=True
        )
        print(f"Model loaded into memory")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        print(f"Tokenizer loaded")
    else:
        # Try loading from HuggingFace
        print("Loading from HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if device == "cpu":
        print("Moving model to CPU...")
        model = model.to("cpu")
        print("Model on CPU")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad token")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run REV pipeline on any model")
    parser.add_argument("model_path", help="Path to model (local or HuggingFace)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
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
        model, tokenizer = load_model(args.model_path, args.device)
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize pipeline
    pipeline = REVPipeline(
        segment_size=args.segment_size,
        buffer_size=4,
        dimension=args.dimension,
        sparsity=args.sparsity,
        enable_pot_challenges=args.pot,
        enable_behavioral_analysis=args.behavioral,
        experiment_name=f"run_{Path(args.model_path).name}"
    )
    print(f"✓ Pipeline initialized")
    
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