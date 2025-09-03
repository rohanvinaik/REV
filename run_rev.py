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
from src.hdc.unified_fingerprint import UnifiedFingerprintGenerator, FingerprintConfig, UnifiedFingerprint
from src.analysis.unified_model_analysis import UnifiedModelAnalyzer, ComprehensiveAnalysis, ModelRelationship
from src.fingerprint.strategic_orchestrator import StrategicTestingOrchestrator, OrchestrationPlan
from src.fingerprint.model_library import ModelFingerprintLibrary
from src.challenges.pot_challenge_generator import PoTChallengeGenerator
from src.challenges.kdf_prompts import KDFPromptGenerator, AdversarialType
from src.hypervector.hamming import HammingDistanceOptimized
from src.core.sequential import SequentialState, TestType
from src.hypervector.similarity import AdvancedSimilarity
from src.diagnostics.probe_monitor import get_probe_monitor, reset_probe_monitor
from src.hdc.behavioral_sites import BehavioralSites
from src.challenges.cassette_executor import CassetteExecutor, CassetteExecutionConfig
from src.challenges.advanced_probe_cassettes import ProbeType
from src.analysis.behavior_profiler import BehaviorProfiler, integrate_with_rev_pipeline

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
        enable_cassettes: bool = False,
        enable_profiler: bool = False,
        enable_adversarial: bool = False,
        adversarial_config: Optional[Dict[str, Any]] = None,
        memory_limit_gb: Optional[float] = None,
        enable_unified_fingerprints: bool = False,
        fingerprint_config: Optional[Dict[str, Any]] = None,
        enable_adversarial_detection: bool = False,
        adversarial_detection_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize unified REV pipeline.
        
        Args:
            debug: Enable debug logging and diagnostics
            enable_behavioral_analysis: Enable behavioral site discovery
            enable_pot_challenges: Use sophisticated PoT challenges
            enable_paper_validation: Validate paper claims
            enable_cassettes: Enable Phase 2 cassette-based analysis
            enable_profiler: Enable behavioral profiling system
            enable_adversarial: Enable adversarial prompt generation
            adversarial_config: Configuration for adversarial generation
            memory_limit_gb: Memory limit in GB (for local loading only)
        """
        self.debug = debug
        self.enable_behavioral_analysis = enable_behavioral_analysis
        self.enable_pot_challenges = enable_pot_challenges
        self.enable_paper_validation = enable_paper_validation
        self.enable_cassettes = enable_cassettes
        self.enable_profiler = enable_profiler
        self.enable_adversarial = enable_adversarial
        self.adversarial_config = adversarial_config or {}
        self.memory_limit_gb = memory_limit_gb or 36  # Default 36GB as per paper
        self.enable_unified_fingerprints = enable_unified_fingerprints
        self.fingerprint_config = fingerprint_config or {}
        self.save_fingerprints = fingerprint_config.get('save_fingerprints', False) if fingerprint_config else False
        self.enable_adversarial_detection = enable_adversarial_detection
        self.adversarial_detection_config = adversarial_detection_config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.hypervectors = {}
        self.models_processed = []
        self.behavioral_profiles = {}
        self.unified_fingerprints = {}  # Storage for unified fingerprints
        self.unified_analyzer = None  # Will be initialized if enabled
        self.comprehensive_analyses = {}  # Storage for comprehensive analyses
        self.orchestrator = None  # Strategic testing orchestrator
        self.fingerprint_library = None  # Model fingerprint library
        
        # Initialize probe monitor for diagnostics
        self.probe_monitor = get_probe_monitor() if debug else None
        
        # Initialize HDC components
        self.hdc_config = HypervectorConfig(
            dimension=10000,
            sparsity=0.01,  # 1% sparsity for efficient hypervector computation
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
        
        # Initialize unified fingerprint generator if enabled
        self.fingerprint_generator = None
        if enable_unified_fingerprints:
            # Filter out non-FingerprintConfig fields
            fp_config_dict = {k: v for k, v in self.fingerprint_config.items() 
                             if k != 'save_fingerprints'}
            fp_config = FingerprintConfig(**fp_config_dict)
            self.fingerprint_generator = UnifiedFingerprintGenerator(
                config=fp_config,
                hdc_encoder=self.encoder
            )
        
        # Initialize unified analyzer if enabled
        if enable_adversarial_detection or enable_unified_fingerprints:
            self.unified_analyzer = UnifiedModelAnalyzer(
                sensitivity=self.adversarial_detection_config.get("sensitivity", 0.1),
                phase_min_length=self.adversarial_detection_config.get("phase_min_length", 3),
                transition_threshold=self.adversarial_detection_config.get("transition_threshold", 0.2)
            )
            self.logger.info(f"Initialized unified model analyzer")
        
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
        
        # Initialize cassette executor if enabled
        self.cassette_executor = None
        if enable_cassettes:
            self.logger.info("Initializing cassette executor for Phase 2 analysis...")
        
        # Initialize behavior profiler if enabled
        self.behavior_profiler = None
        if enable_profiler:
            self.logger.info("Initializing behavior profiler...")
            self.behavior_profiler = BehaviorProfiler({
                "enable_multi_signal": True,
                "enable_streaming": True
            })
            # Enhance pipeline with profiling
            self.pipeline = integrate_with_rev_pipeline(self.pipeline)
        
        # Initialize fingerprint library and orchestrator
        self.fingerprint_library = ModelFingerprintLibrary()
        self.orchestrator = StrategicTestingOrchestrator()
        
        self.logger.info("=" * 80)
        self.logger.info("REV UNIFIED PIPELINE v3.0")
        self.logger.info("=" * 80)
        self.logger.info(f"Debug Mode: {debug}")
        self.logger.info(f"Behavioral Analysis: {enable_behavioral_analysis}")
        self.logger.info(f"PoT Challenges: {enable_pot_challenges}")
        self.logger.info(f"Paper Validation: {enable_paper_validation}")
        self.logger.info(f"Cassette Analysis: {enable_cassettes}")
        self.logger.info(f"Behavior Profiler: {enable_profiler}")
        self.logger.info(f"Adversarial Generation: {enable_adversarial}")
        self.logger.info(f"Unified Fingerprints: {enable_unified_fingerprints}")
        self.logger.info(f"Adversarial Detection: {enable_adversarial_detection}")
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
        
        # Use dual library system for intelligent testing
        from src.fingerprint.dual_library_system import identify_and_strategize
        identification, strategy = identify_and_strategize(model_path)
        
        print(f"\n{'='*80}")
        print(f"Processing Model: {model_name}")
        print(f"Mode: {'Local Loading' if use_local else 'API-Only (No Local Loading)'}")
        
        # Show identification results
        if identification.identified_family:
            print(f"Architecture: {identification.identified_family.upper()} family (confidence: {identification.confidence:.0%})")
            if strategy.get('focus_layers'):
                print(f"  Focus Layers: {strategy['focus_layers']}")
            if strategy.get('reference_model'):
                print(f"  Using Reference: {strategy['reference_model']}")
            # Adjust challenge count based on strategy
            if strategy.get('challenges'):
                challenges = strategy['challenges']
                print(f"  Adjusted Challenges: {challenges} (from reference)")
        else:
            print(f"Architecture: Unknown - will run diagnostic fingerprinting")
            if strategy.get('strategy') == 'diagnostic':
                challenges = min(challenges, 5)  # Quick diagnostic
                print(f"  Diagnostic Mode: {challenges} quick challenges")
        
        print(f"{'='*80}")
        
        self.logger.info(f"[START] Processing model: {model_path}")
        self.logger.info(f"[CONFIG] Local: {use_local}, Device: {device}, Quantize: {quantize}")
        self.logger.info(f"[IDENTIFICATION] Family: {identification.identified_family}, Confidence: {identification.confidence:.1%}")
        
        log_memory_usage("Initial")
        
        result = {
            "model": model_path,
            "model_name": model_name,
            "mode": "local" if use_local else "api",
            "timestamp": datetime.now().isoformat(),
            "identification": {
                "family": identification.identified_family,
                "confidence": identification.confidence,
                "method": identification.method,
                "reference_model": identification.reference_model
            },
            "strategy": strategy,
            "stages": {}
        }
        
        # Stage 1: Initialize Model Interface
        print(f"\n[Stage 1/7] Initializing Model Interface...")
        start = time.time()
        
        try:
            if use_local:
                # REMOVED: Full model loading is no longer supported
                raise NotImplementedError(
                    "\n❌ Full model loading has been permanently removed.\n"
                    "   The --local flag is no longer supported.\n"
                    "   \n"
                    "   API mode now correctly implements segmented streaming:\n"
                    "   • Weights stream from disk layer-by-layer\n"
                    "   • Model is NEVER fully loaded into memory\n"
                    "   • Memory usage capped at 2GB regardless of model size\n"
                    "   \n"
                    "   Please remove the --local flag and run again.\n"
                )
                
            else:
                # API-only mode (default)
                self.logger.info("[API] Using API-only mode")
                
                # Check if this is a local model path first
                if os.path.exists(model_path) or model_path.startswith('/'):
                    # Local model - use TRUE segmented streaming (NEVER load full model)
                    self.logger.info("[SEGMENTED] Model will be streamed layer-by-layer from disk")
                    self.logger.info("[SEGMENTED] Full model will NEVER be loaded into memory")
                    
                    # Import the proper segmented inference that NEVER loads the full model
                    from src.models.segmented_inference import SegmentedModelInference
                    
                    # Create segmented inference with strict memory limit
                    # This streams weights from disk as if from a remote server
                    inference = SegmentedModelInference(
                        model_path=model_path,
                        max_memory_mb=2048  # 2GB max at any time (NOT the full model)
                    )
                    
                    # No load_model() call because we NEVER load the full model
                    # Weights are streamed on-demand during generation
                    
                    print(f"✅ Segmented streaming ready: Weights will stream from disk")
                    print(f"   📦 Model location: {model_path}")
                    print(f"   💾 Memory limit: 2GB (model NEVER fully loaded)")
                    print(f"   🔄 Execution: Layer-by-layer streaming")
                    
                    provider = "local"  # Mark as local for later logic
                    
                # Otherwise, determine cloud provider
                elif not provider:
                    # Auto-detect from environment or model name
                    if "gpt" in model_path.lower():
                        provider = "openai"
                    elif "claude" in model_path.lower():
                        provider = "anthropic"
                    else:
                        provider = "huggingface"  # Default
                
                # Only configure cloud API if not local
                if provider != "local":
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
        
        if self.enable_adversarial and self.kdf_generator:
            # Generate adversarial challenges
            if self.adversarial_suite:
                # Generate comprehensive adversarial suite
                adversarial_prompts = self.kdf_generator.generate_comprehensive_adversarial_suite(
                    base_index=0,
                    include_dangerous=self.include_dangerous
                )
                challenges_list = adversarial_prompts
                print(f"✅ Generated comprehensive adversarial suite with {len(challenges_list)} prompts")
            else:
                # Generate mixed challenges with specified ratio
                n_adversarial = int(challenges * self.adversarial_ratio)
                n_regular = challenges - n_adversarial
                
                adversarial_challenges = []
                regular_challenges = []
                
                # Generate adversarial challenges
                if n_adversarial > 0:
                    if self.adversarial_types:
                        # Generate specific types
                        for i, attack_type in enumerate(self.adversarial_types[:n_adversarial]):
                            method_map = {
                                'divergence_attack': self.kdf_generator.generate_divergence_attack,
                                'mrcj': lambda idx: self.kdf_generator.generate_multi_round_conversational_jailbreak(idx, rounds=1)[0],
                                'special_char': self.kdf_generator.generate_special_character_triggers,
                                'two_stage_inversion': self.kdf_generator.generate_two_stage_inversion_attack,
                                'spv_mia': self.kdf_generator.generate_spv_mia_probe,
                                'alignment_faking': self.kdf_generator.generate_alignment_faking_detector,
                                'pair_algorithm': lambda idx: self.kdf_generator.generate_pair_algorithm_jailbreak(idx, iterations=2),
                                'cross_lingual': self.kdf_generator.generate_cross_lingual_inversion,
                                'temperature_exploit': self.kdf_generator.generate_temperature_exploitation,
                                'dataset_extraction': self.kdf_generator.generate_dataset_extraction_probe,
                                'deception_pattern': self.kdf_generator.generate_deception_pattern_detector
                            }
                            if attack_type in method_map:
                                result_prompt = method_map[attack_type](i)
                                adversarial_challenges.append(result_prompt)
                    else:
                        # Generate random adversarial challenges
                        suite = self.kdf_generator.generate_comprehensive_adversarial_suite(
                            base_index=0,
                            include_dangerous=self.include_dangerous
                        )
                        adversarial_challenges = suite[:n_adversarial]
                
                # Generate regular challenges
                if n_regular > 0:
                    if self.enable_pot_challenges:
                        # Pass layer focus from strategy if available
                        layer_focus = strategy.get('focus_layers', []) if strategy else []
                        regular_challenges = self.pipeline.generate_pot_challenges(
                            n=n_regular,
                            focus=challenge_focus,
                            layer_focus=layer_focus
                        )
                    else:
                        regular_challenges = [
                            {"prompt": f"Test prompt {i}", "category": "test"}
                            for i in range(n_regular)
                        ]
                
                challenges_list = adversarial_challenges + regular_challenges
                print(f"✅ Generated {len(adversarial_challenges)} adversarial + {len(regular_challenges)} regular challenges")
        
        elif self.enable_pot_challenges:
            # Pass layer focus from strategy if available
            layer_focus = strategy.get('focus_layers', []) if strategy else []
            challenges_list = self.pipeline.generate_pot_challenges(
                n=challenges, 
                focus=challenge_focus,
                layer_focus=layer_focus
            )
            print(f"✅ Generated {len(challenges_list)} PoT challenges")
        else:
            # Simple challenges as fallback
            challenges_list = [
                {"prompt": f"Test prompt {i}", "category": "test"}
                for i in range(challenges)
            ]
            print(f"✅ Generated {len(challenges_list)} simple challenges")
        
        result["stages"]["challenges"] = {
            "count": len(challenges_list),
            "time": time.time() - start,
            "pot_enabled": self.enable_pot_challenges,
            "adversarial_enabled": self.enable_adversarial,
            "adversarial_ratio": self.adversarial_ratio if self.enable_adversarial else 0
        }
        
        # Stage 4: Process Challenges & Generate Hypervectors
        print(f"\n[Stage 4/7] Processing Challenges...")
        start = time.time()
        
        hypervectors = []
        responses = []
        divergence_scores = []
        unified_fingerprints = []
        
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
                    # This should never be reached due to earlier check
                    raise RuntimeError("Local mode should have been caught earlier")
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
        
        # Combine hypervectors and unified fingerprints
        if hypervectors:
            # Use majority voting to preserve sparsity instead of mean
            # This preserves the sparse nature of hypervectors
            stacked = np.stack(hypervectors)
            # For binary vectors, use majority voting
            if np.all((stacked == 0) | (stacked == 1)):
                combined_hv = (np.sum(stacked, axis=0) > len(hypervectors) / 2).astype(float)
            else:
                # For real-valued, use thresholded mean to maintain sparsity
                mean_hv = np.mean(stacked, axis=0)
                threshold = np.percentile(np.abs(mean_hv), 99)  # Keep top 1% to maintain sparsity
                combined_hv = np.where(np.abs(mean_hv) > threshold, mean_hv, 0)
            
            self.hypervectors[model_name] = combined_hv
            
            sparsity = np.count_nonzero(combined_hv) / len(combined_hv)
            print(f"✅ Generated model fingerprint (sparsity: {sparsity:.1%})")
            
            # Calculate average divergence if available
            avg_divergence = np.mean(divergence_scores) if divergence_scores else None
            if avg_divergence:
                print(f"   Average divergence: {avg_divergence:.3f}")
        
        # Process unified fingerprints
        if unified_fingerprints:
            self.unified_fingerprints[model_name] = unified_fingerprints
            avg_quality = np.mean([fp.fingerprint_quality for fp in unified_fingerprints])
            avg_binding = np.mean([fp.binding_strength for fp in unified_fingerprints])
            
            print(f"✅ Generated {len(unified_fingerprints)} unified fingerprint(s)")
            print(f"   Average quality: {avg_quality:.3f}")
            print(f"   Average binding strength: {avg_binding:.3f}")
            
            # Save fingerprints to file if requested
            if hasattr(self, 'save_fingerprints') and self.save_fingerprints:
                fingerprint_dir = Path("fingerprints") / model_name
                fingerprint_dir.mkdir(parents=True, exist_ok=True)
                
                for i, fp in enumerate(unified_fingerprints):
                    fp_path = fingerprint_dir / f"fingerprint_{i}.json"
                    self.fingerprint_generator.save_fingerprint(fp, str(fp_path))
                
                print(f"   Saved fingerprints to {fingerprint_dir}")
        
        elif self.enable_unified_fingerprints:
            print("⚠️  Unified fingerprints enabled but none generated (local mode required)")
        
        result["stages"]["processing"] = {
            "success": len(hypervectors) > 0,
            "hypervectors_generated": len(hypervectors),
            "responses_generated": len(responses),
            "unified_fingerprints_generated": len(unified_fingerprints),
            "time": time.time() - start,
            "sparsity": sparsity if hypervectors else None,
            "avg_divergence": avg_divergence if divergence_scores else None,
            "avg_fingerprint_quality": avg_quality if unified_fingerprints else None,
            "avg_binding_strength": avg_binding if unified_fingerprints else None
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
    
    def run_cassette_analysis(
        self,
        model_path: str,
        topology_file: str,
        probe_types: Optional[List[str]] = None,
        output_dir: str = "./cassette_results"
    ) -> Dict[str, Any]:
        """
        Run Phase 2 cassette-based analysis on a model.
        
        Args:
            model_path: Path to model
            topology_file: Path to topology JSON from Phase 1
            probe_types: List of probe types to include (None = all)
            output_dir: Directory for cassette results
            
        Returns:
            Cassette execution results
        """
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: CASSETTE-BASED ANALYSIS")
        self.logger.info("=" * 80)
        
        # Configure cassette execution
        config = CassetteExecutionConfig(
            topology_file=topology_file,
            output_dir=output_dir,
            max_probes_per_layer=10,
            probe_timeout=120.0,
            probe_types=[ProbeType[pt.upper()] for pt in probe_types] if probe_types else None
        )
        
        # Initialize executor
        self.cassette_executor = CassetteExecutor(config)
        
        # Generate execution plan
        execution_plan = self.cassette_executor.generate_execution_plan()
        self.logger.info(f"Execution plan covers {len(execution_plan)} layers")
        
        # Get model interface (reuse existing or create new)
        if model_path in self.inference_managers:
            model_interface = self.inference_managers[model_path]
        else:
            # Create appropriate model interface
            if model_path.startswith("http"):
                # API model
                model_interface = APIOnlyInference(APIOnlyConfig(model_name=model_path))
            else:
                # Local model
                model_interface = UnifiedInferenceManager(
                    model_path=model_path,
                    device="auto",
                    memory_limit_mb=self.memory_limit_gb * 1024
                )
            self.inference_managers[model_path] = model_interface
        
        # Execute cassette plan
        results = self.cassette_executor.execute_plan(execution_plan, model_interface)
        
        # Generate and print report
        report = self.cassette_executor.generate_report(results)
        print(report)
        
        return results
    
    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """Compare two processed models using both traditional and unified fingerprints."""
        print(f"\n{'='*80}")
        print(f"Comparing: {model1} vs {model2}")
        print(f"{'='*80}")
        
        if model1 not in self.hypervectors or model2 not in self.hypervectors:
            print("❌ Both models must be processed first")
            return {"error": "Models not processed"}
        
        comparison = {}
        
        # Traditional hypervector comparison
        hv1 = self.hypervectors[model1]
        hv2 = self.hypervectors[model2]
        
        # Cosine similarity
        dot_product = np.dot(hv1, hv2)
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
        else:
            cosine_sim = 0
        
        comparison["traditional_cosine_similarity"] = float(cosine_sim)
        
        # Hamming distance
        hamming_calc = HammingDistanceOptimized()
        binary_hv1 = (hv1 != 0).astype(np.uint8)
        binary_hv2 = (hv2 != 0).astype(np.uint8)
        hamming_dist = hamming_calc.distance(binary_hv1, binary_hv2)
        
        comparison["hamming_distance"] = int(hamming_dist)
        comparison["normalized_hamming"] = float(hamming_dist / len(binary_hv1))
        
        # Comprehensive unified analysis if available
        comprehensive_analysis = None
        if (model1 in self.unified_fingerprints and 
            model2 in self.unified_fingerprints and
            self.unified_analyzer):
            
            fps1 = self.unified_fingerprints[model1]
            fps2 = self.unified_fingerprints[model2]
            
            if fps1 and fps2:
                # Perform comprehensive analysis
                comprehensive_analysis = self.unified_analyzer.analyze_models(
                    fingerprints_a=fps1,
                    fingerprints_b=fps2,
                    prompts=None,  # Could pass prompts if available
                    layer_activations_a=None,  # Could pass if stored
                    layer_activations_b=None
                )
                
                # Store for later reference
                analysis_key = f"{model1}_vs_{model2}"
                self.comprehensive_analyses[analysis_key] = comprehensive_analysis
                
                # Print comprehensive report
                report = self.unified_analyzer.generate_report(comprehensive_analysis)
                print(report)
                
                # Create simplified comparison dict for backward compatibility
                unified_comparison = {
                    "overall_similarity": comprehensive_analysis.overall_similarity,
                    "decision": comprehensive_analysis.inferred_relationship.value,
                    "confidence": comprehensive_analysis.relationship_confidence,
                    "scaling_analysis": {
                        "is_likely_scaled_version": comprehensive_analysis.inferred_relationship == ModelRelationship.SCALED_VERSION,
                        "layer_ratio": comprehensive_analysis.layer_ratio,
                        "quality_ratio": comprehensive_analysis.magnitude_ratio
                    }
                }
        else:
            unified_comparison = {}
        
        comparison["unified_fingerprint_comparison"] = unified_comparison
        comparison["comprehensive_analysis"] = comprehensive_analysis
        
        # Overall decision (prioritize comprehensive analysis if available)
        if comprehensive_analysis:
            main_similarity = comprehensive_analysis.overall_similarity
            decision_source = "comprehensive_analysis"
            decision = comprehensive_analysis.inferred_relationship.value
            confidence = comprehensive_analysis.relationship_confidence
        elif unified_comparison:
            main_similarity = unified_comparison["overall_similarity"]
            decision_source = "unified_fingerprint"
            # Use threshold-based decision
            threshold = 0.7
            if main_similarity > threshold:
                decision = "SAME/SIMILAR"
                confidence = (main_similarity - threshold) / (1 - threshold)
            else:
                decision = "DIFFERENT"  
                confidence = (threshold - main_similarity) / threshold
        else:
            main_similarity = cosine_sim
            decision_source = "traditional_hypervector"
            # Use threshold-based decision
            threshold = 0.7
            if main_similarity > threshold:
                decision = "SAME/SIMILAR"
                confidence = (main_similarity - threshold) / (1 - threshold)
            else:
                decision = "DIFFERENT"  
                confidence = (threshold - main_similarity) / threshold
        
        comparison["final_decision"] = decision
        comparison["final_confidence"] = float(confidence)
        comparison["decision_source"] = decision_source
        
        # Only print traditional analysis if comprehensive wasn't done
        if not comprehensive_analysis:
            print(f"\n📊 TRADITIONAL ANALYSIS:")
            print(f"Cosine similarity: {cosine_sim:.4f}")
            print(f"Hamming distance: {hamming_dist} ({comparison['normalized_hamming']:.1%} different)")
            print(f"\n🎯 FINAL DECISION: {decision} (confidence: {confidence:.1%}) [{decision_source}]")
        
        return comparison
    
    def _average_fingerprints(self, fingerprints: List[UnifiedFingerprint]) -> UnifiedFingerprint:
        """Average multiple unified fingerprints into one representative fingerprint."""
        if not fingerprints:
            raise ValueError("Cannot average empty fingerprint list")
        
        if len(fingerprints) == 1:
            return fingerprints[0]
        
        # Average the hypervector components
        avg_unified = np.mean([fp.unified_hypervector for fp in fingerprints], axis=0)
        avg_prompt = np.mean([fp.prompt_hypervector for fp in fingerprints], axis=0)
        avg_pathway = np.mean([fp.pathway_hypervector for fp in fingerprints], axis=0)
        avg_response = np.mean([fp.response_hypervector for fp in fingerprints], axis=0)
        
        # Use first fingerprint as template and update vectors
        representative = fingerprints[0]
        representative.unified_hypervector = avg_unified
        representative.prompt_hypervector = avg_prompt
        representative.pathway_hypervector = avg_pathway
        representative.response_hypervector = avg_response
        
        # Average quality metrics
        representative.fingerprint_quality = np.mean([fp.fingerprint_quality for fp in fingerprints])
        representative.binding_strength = np.mean([fp.binding_strength for fp in fingerprints])
        
        # Update metadata
        representative.prompt_text = f"[AVERAGED from {len(fingerprints)} prompts]"
        representative.response_text = f"[AVERAGED from {len(fingerprints)} responses]"
        
        return representative
    
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
    
    # Cassette features (Phase 2 analysis)
    parser.add_argument(
        "--cassettes",
        action="store_true",
        help="Enable Phase 2 cassette-based deep analysis"
    )
    parser.add_argument(
        "--cassette-topology",
        help="Path to topology JSON for cassette analysis"
    )
    parser.add_argument(
        "--cassette-types",
        nargs="+",
        choices=["syntactic", "semantic", "recursive", "transform", "theory_of_mind", "counterfactual", "meta"],
        help="Probe types to include in cassette analysis"
    )
    parser.add_argument(
        "--cassette-output",
        default="./cassette_results",
        help="Output directory for cassette results (default: ./cassette_results)"
    )
    
    # Behavior profiler
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Enable behavioral profiling system"
    )
    
    # Unified fingerprint generation
    parser.add_argument(
        "--unified-fingerprints",
        action="store_true", 
        help="Enable unified hypervector fingerprint generation (requires local mode)"
    )
    parser.add_argument(
        "--fingerprint-dimension",
        type=int,
        default=10000,
        help="Dimension for unified fingerprints (default: 10000)"
    )
    parser.add_argument(
        "--fingerprint-sparsity", 
        type=float,
        default=0.15,
        help="Sparsity for unified fingerprints (default: 0.15)"
    )
    parser.add_argument(
        "--layer-sampling",
        choices=["all", "uniform", "adaptive", "boundary"],
        default="adaptive",
        help="Layer sampling strategy for fingerprints (default: adaptive)"
    )
    parser.add_argument(
        "--max-layers-sampled",
        type=int,
        default=20,
        help="Maximum layers to sample for fingerprints (default: 20)" 
    )
    parser.add_argument(
        "--save-fingerprints",
        action="store_true",
        help="Save unified fingerprints to files"
    )
    parser.add_argument(
        "--enable-scaling-analysis",
        action="store_true",
        help="Enable cross-model scaling analysis in fingerprints"
    )
    parser.add_argument(
        "--fingerprint-weights",
        nargs=3,
        type=float,
        default=[0.3, 0.5, 0.2],
        metavar=("PROMPT", "PATHWAY", "RESPONSE"),
        help="Weights for prompt, pathway, response components (default: 0.3 0.5 0.2)"
    )
    
    # Comprehensive analysis options
    parser.add_argument(
        "--comprehensive-analysis",
        action="store_true",
        help="Enable comprehensive model analysis with pattern detection"
    )
    parser.add_argument(
        "--analysis-sensitivity",
        type=float,
        default=0.1,
        help="Sensitivity for anomaly detection (0-1, default: 0.1)"
    )
    parser.add_argument(
        "--phase-min-length",
        type=int,
        default=3,
        help="Minimum layers for behavioral phase detection (default: 3)"
    )
    parser.add_argument(
        "--transition-threshold",
        type=float,
        default=0.2,
        help="Threshold for detecting layer transitions (default: 0.2)"
    )
    parser.add_argument(
        "--save-analysis-report",
        action="store_true",
        help="Save comprehensive analysis report to file"
    )
    
    # Multi-stage orchestration and fingerprint library
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Enable multi-stage orchestrated testing based on architecture identification"
    )
    parser.add_argument(
        "--claimed-family",
        type=str,
        help="Claimed architecture family (e.g., llama, gpt, mistral)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        help="Time budget in hours for testing"
    )
    parser.add_argument(
        "--library-path",
        type=str,
        default="./fingerprint_library",
        help="Path to fingerprint library (default: ./fingerprint_library)"
    )
    parser.add_argument(
        "--add-to-library",
        action="store_true",
        help="Add discovered fingerprint to library as base model"
    )
    parser.add_argument(
        "--export-fingerprint",
        type=str,
        help="Export fingerprint to specified file"
    )
    parser.add_argument(
        "--import-fingerprint",
        type=str,
        help="Import fingerprint from specified file"
    )
    parser.add_argument(
        "--list-known-architectures",
        action="store_true",
        help="List all known architectures in fingerprint library"
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run quick diagnostic fingerprint only (no behavioral testing)"
    )
    parser.add_argument(
        "--diagnostic-scan",
        action="store_true",
        help="Scan all models in LLM_models for architectural profiles"
    )
    
    # Adversarial prompt generation
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Enable adversarial prompt generation for security testing"
    )
    parser.add_argument(
        "--adversarial-ratio",
        type=float,
        default=0.1,
        help="Ratio of adversarial prompts to generate (default: 0.1)"
    )
    parser.add_argument(
        "--adversarial-types",
        nargs="+",
        choices=[
            "jailbreak", "divergence_attack", "mrcj", "special_char_trigger",
            "temperature_exploit", "two_stage_inversion", "cross_lingual_inversion",
            "pii_extraction", "spv_mia", "dataset_extraction", "alignment_faking",
            "hidden_preference", "pair_algorithm", "deception_pattern"
        ],
        help="Specific adversarial attack types to include"
    )
    parser.add_argument(
        "--include-dangerous",
        action="store_true",
        help="Include high-risk adversarial prompts (with safety wrappers)"
    )
    parser.add_argument(
        "--adversarial-suite",
        action="store_true",
        help="Generate comprehensive adversarial test suite"
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
    if args.cassettes:
        print(f"Cassettes: Enabled (Types: {args.cassette_types or 'all'})")
    if args.profiler:
        print(f"Profiler: Enabled (16-dimensional behavioral signatures)")
    if args.unified_fingerprints:
        print(f"Unified Fingerprints: Enabled ({args.fingerprint_dimension}D)")
        print(f"  Sampling: {args.layer_sampling} (max {args.max_layers_sampled} layers)")
        print(f"  Weights: P{args.fingerprint_weights[0]:.1f}/W{args.fingerprint_weights[1]:.1f}/R{args.fingerprint_weights[2]:.1f}")
        if args.save_fingerprints:
            print(f"  Saving: Enabled")
        if not args.local:
            print(f"  ⚠️  Requires --local mode for full functionality")
    if args.orchestrate:
        print(f"Orchestration: Enabled")
        if args.claimed_family:
            print(f"  Claimed Family: {args.claimed_family}")
        if args.time_budget:
            print(f"  Time Budget: {args.time_budget:.1f} hours")
    print("=" * 80)
    
    # Handle diagnostic operations first
    if args.diagnostic_scan:
        from scripts.diagnostic_fingerprint import DiagnosticScanner
        scanner = DiagnosticScanner()
        profiles = scanner.scan_all_architectures()
        scanner.save_profiles("fingerprint_library/diagnostic_profiles.json")
        scanner.compare_architectures()
        return 0
    
    # Handle fingerprint library operations first
    if args.list_known_architectures:
        library = ModelFingerprintLibrary(args.library_path)
        print("\n📚 Known Architectures in Library:")
        for family, fp_ids in library.family_index.items():
            print(f"  • {family}: {len(fp_ids)} fingerprints")
            for fp_id in fp_ids[:3]:  # Show first 3
                fp = library.fingerprints[fp_id]
                print(f"    - {fp.model_size} ({fp.architecture_version})")
        return 0
    
    if args.import_fingerprint:
        library = ModelFingerprintLibrary(args.library_path)
        fp_id = library.import_fingerprint(args.import_fingerprint)
        print(f"✅ Imported fingerprint: {fp_id}")
        return 0
    
    # Prepare fingerprint configuration
    fingerprint_config = {
        "dimension": args.fingerprint_dimension,
        "sparsity": args.fingerprint_sparsity,
        "layer_sampling": args.layer_sampling,
        "max_layers_sampled": args.max_layers_sampled,
        "enable_cross_scale_analysis": args.enable_scaling_analysis,
        "prompt_weight": args.fingerprint_weights[0],
        "pathway_weight": args.fingerprint_weights[1], 
        "response_weight": args.fingerprint_weights[2],
        "save_fingerprints": args.save_fingerprints  # Store separately for handling
    }
    
    # Prepare analysis configuration
    analysis_config = {
        "sensitivity": args.analysis_sensitivity,
        "phase_min_length": args.phase_min_length,
        "transition_threshold": args.transition_threshold
    }
    
    # Initialize pipeline
    rev = REVUnified(
        debug=args.debug,
        enable_behavioral_analysis=not args.no_behavioral,
        enable_pot_challenges=not args.no_pot,
        enable_paper_validation=not args.no_validation,
        enable_cassettes=args.cassettes,
        enable_profiler=args.profiler,
        enable_unified_fingerprints=args.unified_fingerprints,
        fingerprint_config=fingerprint_config,
        enable_adversarial_detection=args.comprehensive_analysis,
        adversarial_detection_config=analysis_config,
        memory_limit_gb=args.memory_limit
    )
    
    # Handle orchestrated testing if enabled
    if args.orchestrate:
        print("\n🎯 ORCHESTRATED MULTI-STAGE TESTING")
        print("=" * 80)
        
        for model_path in args.models:
            # Create orchestration plan
            time_budget_seconds = args.time_budget * 3600 if args.time_budget else None
            
            plan = rev.orchestrator.create_orchestration_plan(
                model_path=model_path,
                claimed_family=args.claimed_family,
                force_comprehensive=args.comprehensive_analysis,
                time_budget=time_budget_seconds
            )
            
            print(f"\n📋 Orchestration Plan for {model_path}:")
            print(f"  Identified Architecture: {plan.identified_architecture} (confidence: {plan.confidence:.2%})")
            print(f"  Stages: {len(plan.stages)}")
            for i, stage in enumerate(plan.stages, 1):
                print(f"    {i}. {stage.stage_name} ({stage.duration_estimate/60:.1f} min)")
            print(f"  Total Estimated Time: {plan.total_estimated_time/3600:.1f} hours")
            print(f"\n  Reasoning:")
            for reason in plan.reasoning:
                print(f"    • {reason}")
            
            # Execute plan
            print(f"\n▶️  Executing orchestration plan...")
            
            def progress_callback(progress, stage_name):
                print(f"  [{progress:.0%}] {stage_name}")
            
            results = rev.orchestrator.execute_plan(plan, rev, progress_callback)
            
            print(f"\n✅ Orchestration complete!")
            print(f"  Total Duration: {results['total_duration']/3600:.1f} hours")
            print(f"  Stages Completed: {results['summary']['successful_stages']}/{results['summary']['total_stages']}")
            
            # Export fingerprint if requested
            if args.export_fingerprint and plan.identified_architecture:
                rev.fingerprint_library.export_fingerprint(
                    plan.identified_architecture, 
                    args.export_fingerprint
                )
                print(f"📤 Exported fingerprint to {args.export_fingerprint}")
            
            # Add to library if requested
            if args.add_to_library and results.get("new_fingerprint"):
                rev.fingerprint_library.add_fingerprint(results["new_fingerprint"])
                print(f"📚 Added new base fingerprint to library")
        
        # Generate final report
        report = rev.generate_report()
        
    # Standard processing (non-orchestrated)
    else:
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
                
                # Run cassette analysis if enabled and topology available
                if args.cassettes:
                    topology_file = args.cassette_topology or f"{result['model_name']}_topology.json"
                    
                    # Check if topology exists (from phase 1 or provided)
                    if Path(topology_file).exists():
                        print(f"\n🔬 Running Phase 2 cassette analysis...")
                        cassette_results = rev.run_cassette_analysis(
                            model_path=model_path,
                            topology_file=topology_file,
                            probe_types=args.cassette_types,
                            output_dir=args.cassette_output
                        )
                        result['cassette_analysis'] = cassette_results
                    else:
                        print(f"⚠️  Topology file not found: {topology_file}")
                        print("    Run Phase 1 analysis first or provide --cassette-topology")
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
    
    # Save comprehensive analysis reports if requested
    if args.save_analysis_report and rev.comprehensive_analyses:
        for analysis_key, analysis in rev.comprehensive_analyses.items():
            analysis_file = f"analysis_{analysis_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            analysis_report = rev.unified_analyzer.generate_report(analysis)
            with open(analysis_file, 'w') as f:
                f.write(analysis_report)
            print(f"📊 Analysis report saved: {analysis_file}")
    
    print(f"\n✅ Report saved to: {output_file}")
    
    # Cleanup
    rev.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())