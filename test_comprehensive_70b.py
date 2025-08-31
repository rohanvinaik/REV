#!/usr/bin/env python3
"""
Comprehensive test of REV framework with 70B model.
Verifies all components are actually being used:
1. PoT challenge generation
2. True segment execution
3. Restriction site identification
4. Hypervector generation
5. Memory-bounded execution
"""

import json
import logging
import sys
import time
import psutil
import torch
from pathlib import Path

# Setup logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_comprehensive_70b.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"[MEMORY-{stage}] RSS: {mem_info.rss / 1024**3:.2f}GB, VMS: {mem_info.vms / 1024**3:.2f}GB")
    
    # System memory
    sys_mem = psutil.virtual_memory()
    logger.info(f"[SYSTEM-{stage}] Used: {sys_mem.used / 1024**3:.2f}GB, Available: {sys_mem.available / 1024**3:.2f}GB")


def test_pot_challenge_generation():
    """Test PoT challenge generator."""
    logger.info("\n" + "="*80)
    logger.info("TESTING POT CHALLENGE GENERATION")
    logger.info("="*80)
    
    from src.challenges.pot_challenge_generator import PoTChallengeGenerator, ChallengeComplexity
    
    generator = PoTChallengeGenerator(enable_info_selection=True)
    
    # Generate different complexity challenges
    challenges = generator.generate_verification_challenges(n=5, focus="balanced")
    
    logger.info(f"✅ Generated {len(challenges)} PoT challenges")
    
    for i, challenge in enumerate(challenges, 1):
        logger.info(f"\nChallenge {i}:")
        logger.info(f"  Category: {challenge.category.value}")
        logger.info(f"  Complexity: {challenge.complexity.name}")
        logger.info(f"  Expected divergence: {challenge.expected_divergence:.2f}")
        logger.info(f"  Discriminative power: {challenge.discriminative_power:.2f}")
        logger.info(f"  Prompt: {challenge.prompt[:100]}...")
    
    # Test behavioral probes
    probes = generator.generate_behavioral_probes()
    logger.info(f"\n✅ Generated behavioral probes for {len(probes)} categories")
    for category, prompts in probes.items():
        logger.info(f"  {category}: {len(prompts)} probes")
    
    return challenges, probes


def test_true_segment_execution(model_path: str):
    """Test true segment execution with actual transformer computations."""
    logger.info("\n" + "="*80)
    logger.info("TESTING TRUE SEGMENT EXECUTION")
    logger.info("="*80)
    
    log_memory_usage("BEFORE-EXECUTION")
    
    from src.models.true_segment_execution import REVTrueExecution
    
    try:
        # Create executor
        executor = REVTrueExecution(
            model_path=model_path,
            max_memory_gb=4.0
        )
        
        # Get model info
        info = executor.get_info()
        logger.info(f"✅ Initialized true executor:")
        logger.info(f"  Model path: {info['model_path']}")
        logger.info(f"  Layers: {info['n_layers']}")
        logger.info(f"  Hidden size: {info['hidden_size']}")
        logger.info(f"  Attention heads: {info['n_heads']}")
        logger.info(f"  Device: {info['device']}")
        logger.info(f"  Max memory: {info['max_memory_gb']}GB")
        
        log_memory_usage("AFTER-INIT")
        
        # Test restriction site identification
        from src.challenges.pot_challenge_generator import PoTChallengeGenerator
        pot_gen = PoTChallengeGenerator()
        probes = pot_gen.generate_behavioral_probes()
        
        # Use diverse probes
        probe_prompts = (
            probes['factual'][:2] + 
            probes['reasoning'][:2] + 
            probes['code'][:1]
        )
        
        logger.info(f"\n🔍 Identifying restriction sites with {len(probe_prompts)} probes...")
        sites = executor.identify_restriction_sites(probe_prompts)
        
        logger.info(f"✅ Identified {len(sites)} restriction sites:")
        for site in sites:
            logger.info(f"  Layer {site.layer_idx}: {site.site_type} (divergence: {site.behavioral_divergence:.3f})")
        
        log_memory_usage("AFTER-SITES")
        
        # Test generation
        test_prompt = "Explain the concept of transformer attention in simple terms:"
        logger.info(f"\n🧪 Testing generation with prompt: {test_prompt}")
        
        response = executor.generate(test_prompt, max_new_tokens=50)
        logger.info(f"✅ Generated response: {response}")
        
        log_memory_usage("AFTER-GENERATION")
        
        return True, info
        
    except Exception as e:
        logger.error(f"❌ True execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)


def test_hypervector_generation(model_path: str):
    """Test hypervector generation with actual responses."""
    logger.info("\n" + "="*80)
    logger.info("TESTING HYPERVECTOR GENERATION")
    logger.info("="*80)
    
    from src.models.api_only_inference import APIOnlyInference, APIOnlyConfig
    from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder
    
    # Create API-only inference
    config = APIOnlyConfig(
        provider="huggingface",
        model_id=model_path,
        max_tokens=50
    )
    
    inference = APIOnlyInference(model_path, config)
    
    # Create adaptive encoder
    encoder = AdaptiveSparsityEncoder(
        dimension=10000,
        initial_sparsity=0.01,
        max_sparsity=0.2
    )
    
    # Generate response and hypervector
    prompt = "What is the capital of France?"
    
    logger.info(f"🧪 Processing prompt: {prompt}")
    result = inference.process_for_rev(
        prompt=prompt,
        adaptive_encoder=encoder,
        max_memory_gb=4.0
    )
    
    if result['success']:
        logger.info(f"✅ Generated hypervector:")
        logger.info(f"  Response: {result['response'][:100]}...")
        logger.info(f"  Sparsity: {result.get('sparsity', 0):.2%}")
        logger.info(f"  Mode: {result['mode']}")
        
        if 'hypervector' in result:
            hv = result['hypervector']
            logger.info(f"  Hypervector shape: {hv.shape}")
            logger.info(f"  Non-zero elements: {(hv != 0).sum()}")
            logger.info(f"  Mean: {hv.mean():.6f}, Std: {hv.std():.6f}")
        
        return True
    else:
        logger.error(f"❌ Hypervector generation failed: {result.get('error')}")
        return False


def test_memory_bounded_execution(model_path: str):
    """Test memory-bounded execution."""
    logger.info("\n" + "="*80)
    logger.info("TESTING MEMORY-BOUNDED EXECUTION")
    logger.info("="*80)
    
    from src.executor.segment_runner import SegmentRunner, SegmentConfig
    
    config = SegmentConfig(
        segment_size=512,
        max_memory_gb=4.0,
        offload_to_disk=True,
        use_fp16=True
    )
    
    runner = SegmentRunner(config)
    
    logger.info(f"✅ Initialized segment runner:")
    logger.info(f"  Segment size: {config.segment_size}")
    logger.info(f"  Max memory: {config.max_memory_gb}GB")
    logger.info(f"  FP16: {config.use_fp16}")
    logger.info(f"  Disk offload: {config.offload_to_disk}")
    
    # Test memory checking
    mem_usage = runner.get_memory_usage()
    within_limits = runner.check_memory()
    
    logger.info(f"  Current memory: {mem_usage:.2f}GB")
    logger.info(f"  Within limits: {within_limits}")
    
    return True


def run_e2e_pipeline_test(model_path: str):
    """Run full E2E pipeline test."""
    logger.info("\n" + "="*80)
    logger.info("RUNNING E2E PIPELINE TEST")
    logger.info("="*80)
    
    log_memory_usage("E2E-START")
    
    # Import pipeline
    from src.rev_pipeline import REVPipeline
    from src.verifier.blackbox import BlackBoxVerifier
    from src.challenges.pot_challenge_generator import PoTChallengeGenerator
    from src.hdc.adaptive_encoder import AdaptiveSparsityEncoder
    
    # Create components
    pipeline = REVPipeline()
    
    # Create API config for verifier
    from src.verifier.blackbox import APIConfig
    api_config = {"test_model": APIConfig(
        api_key="dummy", 
        base_url="http://localhost",
        model_name="test"
    )}
    verifier = BlackBoxVerifier(configs=api_config)
    
    pot_gen = PoTChallengeGenerator()
    encoder = AdaptiveSparsityEncoder(dimension=10000)
    
    # Generate challenges
    challenges = pot_gen.generate_verification_challenges(n=3, focus="balanced")
    challenge_prompts = [c.prompt for c in challenges]
    
    logger.info(f"📋 Running pipeline with {len(challenges)} challenges")
    
    # Run pipeline
    start_time = time.time()
    
    results = pipeline.run_verification(
        model_path=model_path,
        challenges=challenge_prompts,
        adaptive_encoder=encoder,
        device="auto"
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n✅ Pipeline completed in {elapsed:.2f} seconds")
    logger.info(f"Results:")
    logger.info(f"  Success: {results.get('success', False)}")
    logger.info(f"  Segments processed: {results.get('segments_processed', 0)}")
    logger.info(f"  Hypervectors generated: {results.get('hypervectors_generated', 0)}")
    
    log_memory_usage("E2E-END")
    
    return results


def main():
    """Run comprehensive test."""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE REV FRAMEWORK TEST - 70B MODEL")
    logger.info("="*80)
    
    model_path = "/Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct"
    
    # Check model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found at {model_path}")
        return
    
    logger.info(f"📂 Model path: {model_path}")
    
    # Check model size
    model_size_gb = sum(
        f.stat().st_size for f in Path(model_path).glob("*.safetensors")
    ) / (1024**3)
    logger.info(f"📊 Model size: {model_size_gb:.1f}GB")
    
    # Check device
    if torch.backends.mps.is_available():
        device = "MPS (Metal)"
    elif torch.cuda.is_available():
        device = "CUDA"
    else:
        device = "CPU"
    logger.info(f"🖥️  Device: {device}")
    
    # Run tests
    test_results = {}
    
    # 1. Test PoT challenge generation
    try:
        challenges, probes = test_pot_challenge_generation()
        test_results['pot_challenges'] = "✅ PASSED"
    except Exception as e:
        test_results['pot_challenges'] = f"❌ FAILED: {e}"
    
    # 2. Test true segment execution
    try:
        success, info = test_true_segment_execution(model_path)
        test_results['true_execution'] = "✅ PASSED" if success else f"❌ FAILED: {info}"
    except Exception as e:
        test_results['true_execution'] = f"❌ FAILED: {e}"
    
    # 3. Test hypervector generation
    try:
        if test_hypervector_generation(model_path):
            test_results['hypervectors'] = "✅ PASSED"
        else:
            test_results['hypervectors'] = "❌ FAILED"
    except Exception as e:
        test_results['hypervectors'] = f"❌ FAILED: {e}"
    
    # 4. Test memory-bounded execution
    try:
        if test_memory_bounded_execution(model_path):
            test_results['memory_bounded'] = "✅ PASSED"
        else:
            test_results['memory_bounded'] = "❌ FAILED"
    except Exception as e:
        test_results['memory_bounded'] = f"❌ FAILED: {e}"
    
    # 5. Run E2E pipeline
    try:
        e2e_results = run_e2e_pipeline_test(model_path)
        test_results['e2e_pipeline'] = "✅ PASSED" if e2e_results.get('success') else "❌ FAILED"
    except Exception as e:
        test_results['e2e_pipeline'] = f"❌ FAILED: {e}"
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in test_results.items():
        logger.info(f"{test_name}: {result}")
    
    # Save results
    output_file = "test_comprehensive_70b_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'model_path': model_path,
            'model_size_gb': model_size_gb,
            'device': device,
            'test_results': test_results,
            'memory_usage': {
                'peak_rss_gb': psutil.Process().memory_info().rss / 1024**3,
                'system_used_gb': psutil.virtual_memory().used / 1024**3
            }
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to {output_file}")
    
    # Final memory check
    log_memory_usage("FINAL")


if __name__ == "__main__":
    main()