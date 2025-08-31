#!/usr/bin/env python3
"""
REV Framework - Complete E2E Pipeline with Model Comparison
Combines all components: loading, verification, hypervector generation, and comparison.
"""

import argparse
import json
import time
import torch
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# REV components
from src.models.large_model_inference import LargeModelInference, LargeModelConfig
from src.rev_pipeline import REVPipeline
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy
from src.challenges.pot_challenge_generator import PoTChallengeGenerator
from src.hypervector.hamming import HammingDistanceOptimized
from src.core.sequential import SequentialState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class REVComplete:
    """Complete REV pipeline with all capabilities integrated."""
    
    def __init__(self):
        """Initialize REV complete pipeline."""
        self.results = {}
        self.hypervectors = {}
        self.models_processed = []
        
        # Initialize HDC components
        self.hdc_config = HypervectorConfig(
            dimension=10000,
            sparsity=0.15,
            encoding_mode="rev"
        )
        self.encoder = HypervectorEncoder(self.hdc_config)
        
        # Initialize adaptive encoder
        self.adaptive_encoder = AdaptiveSparsityEncoder(
            dimension=10000,
            initial_sparsity=0.01,
            min_sparsity=0.005,
            max_sparsity=0.2,
            adjustment_strategy=AdjustmentStrategy.ADAPTIVE
        )
        
        # Initialize pipeline
        self.pipeline = REVPipeline(
            segment_size=512,
            buffer_size=4,
            hdc_config=self.hdc_config,
            enable_pot_challenges=True,
            enable_behavioral_analysis=True
        )
        
        # Store inference managers for cleanup
        self.inference_managers = {}
    
    def process_model(self, 
                     model_path: str,
                     device: str = "auto",
                     quantize: str = "none",
                     challenges: int = 2,
                     max_new_tokens: int = 50) -> Dict[str, Any]:
        """
        Process a single model through the complete pipeline.
        
        Args:
            model_path: Path to model or HuggingFace ID
            device: Device to use (auto/cpu/cuda)
            quantize: Quantization mode (none/8bit/4bit)
            challenges: Number of challenges to generate
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with processing results
        """
        model_name = Path(model_path).name if Path(model_path).exists() else model_path
        print(f"\n{'='*80}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*80}")
        
        result = {
            "model": model_path,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Load Model
        print(f"\n[Stage 1/5] Loading {model_name}...")
        start = time.time()
        
        config = LargeModelConfig(
            model_path=model_path,
            device=device,
            load_in_8bit=(quantize == "8bit"),
            load_in_4bit=(quantize == "4bit"),
            low_cpu_mem_usage=True,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        inference = LargeModelInference(model_path, config)
        success, message = inference.load_model()
        
        if not success:
            print(f"❌ Failed to load model: {message}")
            result["error"] = message
            return result
        
        self.inference_managers[model_name] = inference
        
        print(f"✅ {message}")
        result["stages"]["loading"] = {
            "success": True,
            "time": time.time() - start,
            "model_info": inference.model_info
        }
        
        # Stage 2: Generate PoT Challenges
        print(f"\n[Stage 2/5] Generating PoT Challenges...")
        start = time.time()
        
        challenges_list = self.pipeline.generate_pot_challenges(n=challenges, focus="balanced")
        print(f"✅ Generated {len(challenges_list)} challenges")
        
        result["stages"]["challenges"] = {
            "count": len(challenges_list),
            "time": time.time() - start
        }
        
        # Stage 3: Process Challenges and Generate Hypervectors
        print(f"\n[Stage 3/5] Processing Challenges & Generating Hypervectors...")
        start = time.time()
        
        hypervectors = []
        responses = []
        
        for i, challenge in enumerate(challenges_list, 1):
            print(f"  Processing challenge {i}/{len(challenges_list)}...", end='')
            
            # Process with REV
            rev_result = inference.process_for_rev(
                prompt=challenge.prompt,
                extract_activations=True,
                hdc_encoder=self.encoder,
                adaptive_encoder=self.adaptive_encoder
            )
            
            if rev_result["success"]:
                responses.append(rev_result.get("response", ""))
                if "hypervector" in rev_result:
                    hypervectors.append(rev_result["hypervector"])
                print(" ✓")
            else:
                print(" ✗")
        
        # Combine hypervectors
        if hypervectors:
            # Average pool hypervectors for model fingerprint
            combined_hv = np.mean(hypervectors, axis=0)
            self.hypervectors[model_name] = combined_hv
            
            sparsity = np.count_nonzero(combined_hv) / len(combined_hv)
            print(f"✅ Generated model fingerprint (sparsity: {sparsity:.1%})")
        else:
            print("❌ No hypervectors generated")
        
        result["stages"]["hypervector_generation"] = {
            "success": len(hypervectors) > 0,
            "count": len(hypervectors),
            "time": time.time() - start,
            "sparsity": sparsity if hypervectors else None
        }
        
        # Stage 4: Extract Behavioral Characteristics
        print(f"\n[Stage 4/5] Extracting Behavioral Characteristics...")
        start = time.time()
        
        if responses:
            # Analyze response patterns
            avg_length = np.mean([len(r.split()) for r in responses])
            unique_tokens = len(set(" ".join(responses).split()))
            
            result["stages"]["behavioral"] = {
                "avg_response_length": avg_length,
                "unique_tokens": unique_tokens,
                "time": time.time() - start
            }
            print(f"✅ Behavioral analysis complete")
        
        # Stage 5: Calculate Metrics
        print(f"\n[Stage 5/5] Calculating Verification Metrics...")
        start = time.time()
        
        if model_name in self.hypervectors:
            hv = self.hypervectors[model_name]
            
            # Self-similarity (should be 1.0)
            self_sim = np.dot(hv, hv) / (np.linalg.norm(hv) ** 2) if np.linalg.norm(hv) > 0 else 0
            
            # Information content
            entropy = -np.sum(np.abs(hv) * np.log(np.abs(hv) + 1e-10))
            
            result["stages"]["metrics"] = {
                "self_similarity": float(self_sim),
                "entropy": float(entropy),
                "dimension": len(hv),
                "time": time.time() - start
            }
            print(f"✅ Metrics calculated")
        
        self.models_processed.append(model_name)
        self.results[model_name] = result
        
        return result
    
    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Compare two processed models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            
        Returns:
            Comparison results
        """
        print(f"\n{'='*80}")
        print(f"Comparing Models: {model1} vs {model2}")
        print(f"{'='*80}")
        
        if model1 not in self.hypervectors or model2 not in self.hypervectors:
            print("❌ Models must be processed first")
            return {"error": "Models not processed"}
        
        hv1 = self.hypervectors[model1]
        hv2 = self.hypervectors[model2]
        
        # Calculate similarities
        comparison = {}
        
        # Cosine similarity
        dot_product = np.dot(hv1, hv2)
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
        else:
            cosine_sim = 0
        
        comparison["cosine_similarity"] = float(cosine_sim)
        print(f"Cosine similarity: {cosine_sim:.4f}")
        
        # Hamming distance
        hamming_calc = HammingDistanceOptimized()
        binary_hv1 = (hv1 != 0).astype(np.uint8)
        binary_hv2 = (hv2 != 0).astype(np.uint8)
        hamming_dist = hamming_calc.distance(binary_hv1, binary_hv2)
        normalized_hamming = hamming_dist / len(binary_hv1)
        
        comparison["hamming_distance"] = int(hamming_dist)
        comparison["normalized_hamming"] = float(normalized_hamming)
        print(f"Hamming distance: {hamming_dist} ({normalized_hamming:.1%} different)")
        
        # Jaccard similarity
        active1 = set(np.where(hv1 != 0)[0])
        active2 = set(np.where(hv2 != 0)[0])
        overlap = len(active1 & active2)
        union = len(active1 | active2)
        jaccard = overlap / union if union > 0 else 0
        
        comparison["jaccard_similarity"] = float(jaccard)
        print(f"Jaccard similarity: {jaccard:.3f}")
        
        # Decision
        threshold = 0.7
        if cosine_sim > threshold:
            decision = "SAME/SIMILAR"
            confidence = (cosine_sim - threshold) / (1 - threshold)
        else:
            decision = "DIFFERENT"
            confidence = (threshold - cosine_sim) / threshold
        
        comparison["decision"] = decision
        comparison["confidence"] = float(confidence)
        
        print(f"\nDecision: {decision} (confidence: {confidence:.1%})")
        
        # Model info comparison
        info1 = self.results[model1]["stages"]["loading"]["model_info"]
        info2 = self.results[model2]["stages"]["loading"]["model_info"]
        
        param_diff = abs(info1["parameters"] - info2["parameters"])
        comparison["parameter_difference"] = param_diff
        
        print(f"Parameter difference: {param_diff/1e9:.1f}B")
        
        return comparison
    
    def generate_report(self, output_file: str):
        """Generate comprehensive report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "framework": "REV Complete E2E Pipeline",
            "models_processed": self.models_processed,
            "results": self.results,
            "comparisons": {}
        }
        
        # Compare all model pairs
        if len(self.models_processed) > 1:
            for i, model1 in enumerate(self.models_processed):
                for model2 in self.models_processed[i+1:]:
                    key = f"{model1}_vs_{model2}"
                    report["comparisons"][key] = self.compare_models(model1, model2)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Report saved to: {output_file}")
        
        return report
    
    def cleanup(self):
        """Clean up all models and free memory."""
        for name, inference in self.inference_managers.items():
            print(f"Cleaning up {name}...")
            inference.cleanup()
        print("✅ All models cleaned up")


def main():
    parser = argparse.ArgumentParser(description="REV Complete E2E Pipeline")
    parser.add_argument("models", nargs="+", help="Model paths or HuggingFace IDs")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--challenges", type=int, default=2, help="Number of challenges")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output", help="Output report file")
    
    args = parser.parse_args()
    
    print("="*80)
    print("REV FRAMEWORK - COMPLETE E2E PIPELINE")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Device: {args.device}")
    print(f"Quantization: {args.quantize}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    # Initialize pipeline
    rev = REVComplete()
    
    # Process each model
    for model_path in args.models:
        try:
            result = rev.process_model(
                model_path=model_path,
                device=args.device,
                quantize=args.quantize,
                challenges=args.challenges,
                max_new_tokens=args.max_tokens
            )
            
            if "error" not in result:
                print(f"\n✅ Successfully processed {result['model_name']}")
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
    
    # Generate report
    output_file = args.output or f"rev_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = rev.generate_report(output_file)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Models processed: {len(rev.models_processed)}")
    
    if rev.models_processed:
        print("\nModels:")
        for model in rev.models_processed:
            info = rev.results[model]["stages"]["loading"]["model_info"]
            print(f"  • {model}: {info['parameters']/1e9:.1f}B parameters")
    
    if report.get("comparisons"):
        print("\nComparisons:")
        for comp_name, comp_result in report["comparisons"].items():
            print(f"  • {comp_name}: {comp_result['decision']} ({comp_result['confidence']:.0%} confidence)")
    
    print(f"\n✅ Complete report saved to: {output_file}")
    
    # Cleanup
    rev.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())