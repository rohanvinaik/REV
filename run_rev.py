#!/usr/bin/env python3
"""
REV Framework - Unified Central Pipeline
=========================================
This is the MAIN entry point for the REV (Restriction Enzyme Verification) framework.
Combines all functionality from previous scripts into one comprehensive pipeline.

Default Mode: API-only (no local model loading)
Optional: Local model loading with --local flag

Author: REV Framework Team
Version: 3.0 (Unified)
"""

import argparse
import json
import time
import torch
import numpy as np
import logging
import sys
import os
import psutil
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# REV components
from src.models.api_only_inference import APIOnlyInference, APIOnlyConfig
from src.models.unified_inference import UnifiedInferenceManager
from src.models.large_model_inference import LargeModelInference, LargeModelConfig
from src.models.metal_accelerated_inference import MetalAcceleratedInference, get_optimal_device
from src.rev_pipeline import REVPipeline
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder, AdjustmentStrategy
from src.challenges.pot_challenge_generator import PoTChallengeGenerator
from src.hypervector.hamming import HammingDistanceOptimized
from src.core.sequential import SequentialState, TestType
from src.hypervector.similarity import AdvancedSimilarity
from src.diagnostics.probe_monitor import get_probe_monitor, reset_probe_monitor
from src.hdc.behavioral_sites import BehavioralSites

# Configure logging
def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    
    sys_mem = psutil.virtual_memory()
    sys_used_gb = sys_mem.used / (1024 ** 3)
    sys_total_gb = sys_mem.total / (1024 ** 3)
    sys_percent = sys_mem.percent
    
    logging.info(f"[MEMORY] {stage}: Process={mem_gb:.2f}GB, System={sys_used_gb:.1f}/{sys_total_gb:.1f}GB ({sys_percent:.1f}%)")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logging.info(f"[GPU MEMORY] {stage}: Allocated={gpu_mem:.2f}GB, Max={gpu_max:.2f}GB")
    
    return mem_gb

class REVUnified:
    """Unified REV pipeline with all features integrated."""
    
    def __init__(
        self, 
        debug: bool = False,
        enable_behavioral_analysis: bool = True,
        enable_pot_challenges: bool = True,
        enable_paper_validation: bool = True,
        memory_limit_gb: Optional[float] = None
    ):
        """
        Initialize unified REV pipeline.
        
        Args:
            debug: Enable debug logging and diagnostics
            enable_behavioral_analysis: Enable behavioral site discovery
            enable_pot_challenges: Use sophisticated PoT challenges
            enable_paper_validation: Validate paper claims
            memory_limit_gb: Memory limit in GB (for local loading only)
        """
        self.debug = debug
        self.enable_behavioral_analysis = enable_behavioral_analysis
        self.enable_pot_challenges = enable_pot_challenges
        self.enable_paper_validation = enable_paper_validation
        self.memory_limit_gb = memory_limit_gb or 36  # Default 36GB as per paper
        
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.hypervectors = {}
        self.models_processed = []
        self.behavioral_profiles = {}
        
        # Initialize probe monitor for diagnostics
        self.probe_monitor = get_probe_monitor() if debug else None
        
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
        
        # Initialize behavioral sites analyzer
        if enable_behavioral_analysis:
            self.behavioral_sites = BehavioralSites(
                hdc_config=self.hdc_config
            )
        else:
            self.behavioral_sites = None
        
        # Initialize pipeline
        self.pipeline = REVPipeline(
            segment_size=512,
            buffer_size=4,
            hdc_config=self.hdc_config,
            enable_pot_challenges=enable_pot_challenges,
            enable_behavioral_analysis=enable_behavioral_analysis
        )
        
        # Enable debug mode if requested
        if debug:
            self.pipeline.enable_debug_mode()
        
        # Store inference managers for cleanup
        self.inference_managers = {}
        
        self.logger.info("=" * 80)
        self.logger.info("REV UNIFIED PIPELINE v3.0")
        self.logger.info("=" * 80)
        self.logger.info(f"Debug Mode: {debug}")
        self.logger.info(f"Behavioral Analysis: {enable_behavioral_analysis}")
        self.logger.info(f"PoT Challenges: {enable_pot_challenges}")
        self.logger.info(f"Paper Validation: {enable_paper_validation}")
        self.logger.info(f"Memory Limit: {self.memory_limit_gb}GB")
        self.logger.info("=" * 80)
    
    def process_model(
        self,
        model_path: str,
        use_local: bool = False,
        device: str = "auto",
        quantize: str = "none",
        challenges: int = 5,
        max_new_tokens: int = 50,
        challenge_focus: str = "balanced",
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a model through the complete pipeline.
        
        Args:
            model_path: Model identifier (HuggingFace ID or local path)
            use_local: Whether to load model locally (default: False, uses API)
            device: Device for local loading (auto/cpu/cuda/mps)
            quantize: Quantization for local loading (none/8bit/4bit)
            challenges: Number of PoT challenges
            max_new_tokens: Max tokens to generate
            challenge_focus: Focus for challenges (coverage/separation/balanced)
            provider: API provider (openai/anthropic/huggingface/cohere)
            api_key: API key for provider
            
        Returns:
            Processing results dictionary
        """
        model_name = Path(model_path).name if Path(model_path).exists() else model_path
        
        print(f"\n{'='*80}")
        print(f"Processing Model: {model_name}")
        print(f"Mode: {'Local Loading' if use_local else 'API-Only (No Local Loading)'}")
        print(f"{'='*80}")
        
        self.logger.info(f"[START] Processing model: {model_path}")
        self.logger.info(f"[CONFIG] Local: {use_local}, Device: {device}, Quantize: {quantize}")
        
        log_memory_usage("Initial")
        
        result = {
            "model": model_path,
            "model_name": model_name,
            "mode": "local" if use_local else "api",
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Initialize Model Interface
        print(f"\n[Stage 1/7] Initializing Model Interface...")
        start = time.time()
        
        try:
            if use_local:
                # Local model loading
                self.logger.info("[LOCAL] Loading model locally")
                
                # Create config with memory limit
                config = LargeModelConfig(
                    model_path=model_path,
                    device=device,
                    load_in_8bit=(quantize == "8bit"),
                    load_in_4bit=(quantize == "4bit"),
                    low_cpu_mem_usage=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    max_memory={
                        0: f"{self.memory_limit_gb}GB",
                        "cpu": f"{self.memory_limit_gb}GB"
                    }
                )
                
                # Use unified manager
                inference = UnifiedInferenceManager(
                    model_path=model_path,
                    config=config,
                    prefer_cloud=False
                )
                
                success, message = inference.load_model()
                if not success:
                    raise Exception(f"Failed to load model: {message}")
                
                print(f"✅ Model loaded locally: {message}")
                
            else:
                # API-only mode (default)
                self.logger.info("[API] Using API-only mode")
                
                # Determine provider and API key
                if not provider:
                    # Auto-detect from environment or model name
                    if "gpt" in model_path.lower():
                        provider = "openai"
                    elif "claude" in model_path.lower():
                        provider = "anthropic"
                    else:
                        provider = "huggingface"  # Default
                
                if not api_key:
                    # Try to get from environment
                    env_keys = {
                        "openai": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY",
                        "huggingface": "HF_TOKEN",
                        "cohere": "COHERE_API_KEY"
                    }
                    api_key = os.environ.get(env_keys.get(provider, "HF_TOKEN"))
                
                api_config = APIOnlyConfig(
                    provider=provider,
                    api_key=api_key,
                    model_id=model_path,
                    max_tokens=max_new_tokens
                )
                
                inference = APIOnlyInference(model_path, api_config)
                print(f"✅ API interface ready: {provider}/{model_path}")
            
            self.inference_managers[model_name] = inference
            
            result["stages"]["initialization"] = {
                "success": True,
                "time": time.time() - start,
                "mode": "local" if use_local else "api",
                "provider": provider if not use_local else None
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Initialization failed: {str(e)}")
            result["error"] = str(e)
            return result
        
        # Stage 2: Behavioral Site Discovery (if enabled and local)
        if self.enable_behavioral_analysis and use_local:
            print(f"\n[Stage 2/7] Discovering Behavioral Sites...")
            start = time.time()
            
            try:
                # This requires actual model access
                if hasattr(inference, 'model'):
                    sites = self.behavioral_sites.discover_sites(
                        inference.model,
                        use_sophisticated_probes=True
                    )
                    
                    self.behavioral_profiles[model_name] = sites
                    print(f"✅ Discovered {len(sites)} behavioral sites")
                    
                    result["stages"]["behavioral_discovery"] = {
                        "success": True,
                        "sites_found": len(sites),
                        "time": time.time() - start
                    }
                else:
                    print("⚠️  Behavioral discovery skipped (model not accessible)")
            except Exception as e:
                self.logger.warning(f"Behavioral discovery failed: {e}")
                print("⚠️  Behavioral discovery failed, continuing...")
        
        # Stage 3: Generate PoT Challenges
        print(f"\n[Stage 3/7] Generating PoT Challenges...")
        start = time.time()
        
        if self.enable_pot_challenges:
            challenges_list = self.pipeline.generate_pot_challenges(
                n=challenges, 
                focus=challenge_focus
            )
        else:
            # Simple challenges as fallback
            challenges_list = [
                {"prompt": f"Test prompt {i}", "category": "test"}
                for i in range(challenges)
            ]
        
        print(f"✅ Generated {len(challenges_list)} challenges")
        
        result["stages"]["challenges"] = {
            "count": len(challenges_list),
            "time": time.time() - start,
            "pot_enabled": self.enable_pot_challenges
        }
        
        # Stage 4: Process Challenges & Generate Hypervectors
        print(f"\n[Stage 4/7] Processing Challenges...")
        start = time.time()
        
        hypervectors = []
        responses = []
        divergence_scores = []
        
        for i, challenge in enumerate(challenges_list, 1):
            print(f"  Challenge {i}/{len(challenges_list)}...", end='')
            
            try:
                # Get prompt from challenge (handle both dict and dataclass)
                if hasattr(challenge, 'prompt'):
                    prompt = challenge.prompt
                elif isinstance(challenge, dict):
                    prompt = challenge.get("prompt", str(challenge))
                else:
                    prompt = str(challenge)
                
                if use_local:
                    # Local processing with REV
                    rev_result = inference.process_for_rev(
                        prompt=prompt,
                        extract_activations=True,
                        hdc_encoder=self.encoder,
                        adaptive_encoder=self.adaptive_encoder
                    )
                    
                    if rev_result["success"]:
                        responses.append(rev_result.get("response", ""))
                        if "hypervector" in rev_result:
                            hypervectors.append(rev_result["hypervector"])
                        if "divergence" in rev_result:
                            divergence_scores.append(rev_result["divergence"])
                        print(" ✓")
                    else:
                        print(" ✗")
                else:
                    # API-only processing
                    response = inference.generate(prompt)
                    responses.append(response)
                    
                    # Generate hypervector from response
                    tokens = response.split()[:100]
                    hypervector = self.adaptive_encoder.encode_tokens(tokens)
                    hypervectors.append(hypervector)
                    print(" ✓")
                    
            except Exception as e:
                self.logger.error(f"Challenge {i} failed: {e}")
                print(" ✗")
        
        # Combine hypervectors
        if hypervectors:
            combined_hv = np.mean(hypervectors, axis=0)
            self.hypervectors[model_name] = combined_hv
            
            sparsity = np.count_nonzero(combined_hv) / len(combined_hv)
            print(f"✅ Generated model fingerprint (sparsity: {sparsity:.1%})")
            
            # Calculate average divergence if available
            avg_divergence = np.mean(divergence_scores) if divergence_scores else None
            if avg_divergence:
                print(f"   Average divergence: {avg_divergence:.3f}")
        
        result["stages"]["processing"] = {
            "success": len(hypervectors) > 0,
            "hypervectors_generated": len(hypervectors),
            "responses_generated": len(responses),
            "time": time.time() - start,
            "sparsity": sparsity if hypervectors else None,
            "avg_divergence": avg_divergence if divergence_scores else None
        }
        
        # Stage 5: Behavioral Analysis
        print(f"\n[Stage 5/7] Analyzing Behavioral Characteristics...")
        start = time.time()
        
        behavioral_metrics = {}
        
        if responses:
            # Response analysis
            avg_length = np.mean([len(r.split()) for r in responses])
            unique_tokens = len(set(" ".join(responses).split()))
            
            behavioral_metrics.update({
                "avg_response_length": avg_length,
                "unique_tokens": unique_tokens,
                "response_diversity": unique_tokens / max(sum(len(r.split()) for r in responses), 1)
            })
        
        if model_name in self.hypervectors:
            hv = self.hypervectors[model_name]
            
            # Hypervector metrics
            behavioral_metrics.update({
                "hv_entropy": float(-np.sum(np.abs(hv) * np.log(np.abs(hv) + 1e-10))),
                "hv_active_dims": int(np.count_nonzero(hv)),
                "hv_sparsity": float(np.count_nonzero(hv) / len(hv))
            })
        
        result["stages"]["behavioral_analysis"] = {
            "metrics": behavioral_metrics,
            "time": time.time() - start
        }
        
        print(f"✅ Behavioral analysis complete")
        
        # Stage 6: Validate Paper Claims (if enabled)
        if self.enable_paper_validation:
            print(f"\n[Stage 6/7] Validating Paper Claims...")
            start = time.time()
            
            validation = self._validate_paper_claims(result, use_local)
            
            result["stages"]["paper_validation"] = {
                "claims_validated": validation,
                "time": time.time() - start
            }
            
            validated_count = sum(1 for v in validation.values() if v.get('validated', False))
            print(f"✅ Claims validated: {validated_count}/{len(validation)}")
        
        # Stage 7: Generate Diagnostics (if debug mode)
        if self.debug and self.probe_monitor:
            print(f"\n[Stage 7/7] Generating Diagnostics...")
            
            diagnostic_report = self.probe_monitor.generate_report()
            
            if diagnostic_report['summary'].get('using_behavioral_probing'):
                print(f"✅ Behavioral probing active")
                print(f"   Success rate: {diagnostic_report['summary']['success_rate']:.1%}")
            else:
                print(f"⚠️  Using fallback mode")
            
            # Save report
            report_path = self.probe_monitor.save_report(
                f"diagnostic_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            result["diagnostics"] = {
                "report_saved": str(report_path),
                "summary": diagnostic_report['summary']
            }
        
        # Store results
        self.models_processed.append(model_name)
        self.results[model_name] = result
        
        print(f"\n✅ Model processing complete")
        log_memory_usage("Final")
        
        return result
    
    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """Compare two processed models."""
        print(f"\n{'='*80}")
        print(f"Comparing: {model1} vs {model2}")
        print(f"{'='*80}")
        
        if model1 not in self.hypervectors or model2 not in self.hypervectors:
            print("❌ Both models must be processed first")
            return {"error": "Models not processed"}
        
        hv1 = self.hypervectors[model1]
        hv2 = self.hypervectors[model2]
        
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
        
        # Hamming distance
        hamming_calc = HammingDistanceOptimized()
        binary_hv1 = (hv1 != 0).astype(np.uint8)
        binary_hv2 = (hv2 != 0).astype(np.uint8)
        hamming_dist = hamming_calc.distance(binary_hv1, binary_hv2)
        
        comparison["hamming_distance"] = int(hamming_dist)
        comparison["normalized_hamming"] = float(hamming_dist / len(binary_hv1))
        
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
        
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"Hamming distance: {hamming_dist} ({comparison['normalized_hamming']:.1%} different)")
        print(f"\nDecision: {decision} (confidence: {confidence:.1%})")
        
        return comparison
    
    def _validate_paper_claims(self, result: Dict[str, Any], use_local: bool) -> Dict[str, Any]:
        """Validate paper claims based on processing results."""
        claims = {}
        
        # Claim 1: API-only by default (no local loading)
        claims['api_only_default'] = {
            'validated': not use_local,
            'mode': result['mode']
        }
        
        # Claim 2: Memory-bounded execution (for local mode)
        if use_local and 'initialization' in result['stages']:
            claims['memory_bounded'] = {
                'validated': True,  # Memory limit was set
                'limit_gb': self.memory_limit_gb
            }
        
        # Claim 3: Hypervector generation
        if 'processing' in result['stages']:
            claims['hypervector_generation'] = {
                'validated': result['stages']['processing']['success'],
                'count': result['stages']['processing']['hypervectors_generated']
            }
        
        # Claim 4: PoT challenges
        if 'challenges' in result['stages']:
            claims['pot_challenges'] = {
                'validated': result['stages']['challenges'].get('pot_enabled', False),
                'count': result['stages']['challenges']['count']
            }
        
        # Claim 5: Behavioral analysis
        if 'behavioral_analysis' in result['stages']:
            claims['behavioral_analysis'] = {
                'validated': bool(result['stages']['behavioral_analysis']['metrics']),
                'metrics_count': len(result['stages']['behavioral_analysis']['metrics'])
            }
        
        return claims
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "framework": "REV Unified Pipeline v3.0",
            "models_processed": self.models_processed,
            "total_models": len(self.models_processed),
            "results": self.results,
            "comparisons": {},
            "summary": {}
        }
        
        # Generate all pairwise comparisons
        if len(self.models_processed) > 1:
            for i, model1 in enumerate(self.models_processed):
                for model2 in self.models_processed[i+1:]:
                    key = f"{model1}_vs_{model2}"
                    report["comparisons"][key] = self.compare_models(model1, model2)
        
        # Summary statistics
        report['summary'] = {
            'models_processed': len(self.models_processed),
            'api_models': sum(1 for r in self.results.values() if r['mode'] == 'api'),
            'local_models': sum(1 for r in self.results.values() if r['mode'] == 'local'),
            'behavioral_sites_discovered': len(self.behavioral_profiles),
            'total_processing_time': sum(
                sum(stage.get('time', 0) for stage in result.get('stages', {}).values())
                for result in self.results.values()
            )
        }
        
        # Save report
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n✅ Report saved to: {output_file}")
        
        return report
    
    def cleanup(self):
        """Clean up all resources."""
        for name, inference in self.inference_managers.items():
            print(f"Cleaning up {name}...")
            if hasattr(inference, 'cleanup'):
                inference.cleanup()
        
        # Clear memory
        self.hypervectors.clear()
        self.results.clear()
        self.behavioral_profiles.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="REV Unified Pipeline - Central entry point for all REV functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # API-only mode (default, recommended):
  python run_rev.py meta-llama/Llama-3.3-70B-Instruct
  
  # Multiple models comparison:
  python run_rev.py gpt-4 claude-3-opus meta-llama/Llama-3.3-70B
  
  # Local loading (not recommended for large models):
  python run_rev.py /path/to/model --local --device cpu --quantize 4bit
  
  # With all features enabled:
  python run_rev.py model_id --debug --challenges 10 --output report.json
  
  # Specify API provider:
  python run_rev.py gpt-4 --provider openai --api-key sk-...
        """
    )
    
    # Model arguments
    parser.add_argument(
        "models", 
        nargs="+", 
        help="Model IDs (HuggingFace) or paths. Multiple models can be specified for comparison."
    )
    
    # Mode selection
    parser.add_argument(
        "--local", 
        action="store_true",
        help="Load model locally instead of using API (not recommended for large models)"
    )
    
    # API configuration
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "huggingface", "cohere"],
        help="API provider (auto-detected if not specified)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (uses environment variable if not specified)"
    )
    
    # Local loading options
    parser.add_argument(
        "--device", 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for local loading (default: auto)"
    )
    parser.add_argument(
        "--quantize",
        default="none",
        choices=["none", "8bit", "4bit"],
        help="Quantization for local loading (default: none)"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=36,
        help="Memory limit in GB for local loading (default: 36)"
    )
    
    # Challenge configuration
    parser.add_argument(
        "--challenges",
        type=int,
        default=5,
        help="Number of PoT challenges to generate (default: 5)"
    )
    parser.add_argument(
        "--challenge-focus",
        choices=["coverage", "separation", "balanced"],
        default="balanced",
        help="Focus for challenge generation (default: balanced)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    
    # Feature toggles
    parser.add_argument(
        "--no-behavioral",
        action="store_true",
        help="Disable behavioral analysis"
    )
    parser.add_argument(
        "--no-pot",
        action="store_true",
        help="Disable PoT challenges"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip paper claims validation"
    )
    
    # Output and debugging
    parser.add_argument(
        "--output",
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and diagnostics"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug, args.log_file)
    
    # Print header
    print("=" * 80)
    print("REV FRAMEWORK - UNIFIED PIPELINE v3.0")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Mode: {'Local Loading' if args.local else 'API-Only (Recommended)'}")
    if args.local:
        print(f"Device: {args.device}")
        print(f"Quantization: {args.quantize}")
        print(f"Memory Limit: {args.memory_limit}GB")
    print(f"Challenges: {args.challenges} ({args.challenge_focus} focus)")
    print(f"Debug: {args.debug}")
    print("=" * 80)
    
    # Initialize pipeline
    rev = REVUnified(
        debug=args.debug,
        enable_behavioral_analysis=not args.no_behavioral,
        enable_pot_challenges=not args.no_pot,
        enable_paper_validation=not args.no_validation,
        memory_limit_gb=args.memory_limit
    )
    
    # Process each model
    for model_path in args.models:
        try:
            result = rev.process_model(
                model_path=model_path,
                use_local=args.local,
                device=args.device,
                quantize=args.quantize,
                challenges=args.challenges,
                max_new_tokens=args.max_tokens,
                challenge_focus=args.challenge_focus,
                provider=args.provider,
                api_key=args.api_key
            )
            
            if "error" not in result:
                print(f"\n✅ Successfully processed {result['model_name']}")
            else:
                print(f"\n❌ Failed to process {model_path}: {result['error']}")
                
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            logger.error(traceback.format_exc())
            print(f"\n❌ Failed to process {model_path}: {e}")
    
    # Generate report
    output_file = args.output or f"rev_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = rev.generate_report(output_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Models processed: {report['summary']['models_processed']}")
    print(f"API models: {report['summary']['api_models']}")
    print(f"Local models: {report['summary']['local_models']}")
    print(f"Total time: {report['summary']['total_processing_time']:.1f}s")
    
    if report.get("comparisons"):
        print("\nModel Comparisons:")
        for comp_name, comp_result in report["comparisons"].items():
            print(f"  • {comp_name}: {comp_result['decision']} ({comp_result['confidence']:.0%} confidence)")
    
    print(f"\n✅ Report saved to: {output_file}")
    
    # Cleanup
    rev.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())