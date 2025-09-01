#!/usr/bin/env python3
"""
Script to run Yi-34B model through the REV pipeline.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import psutil
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.yi34b_integration import Yi34BREVIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'yi34b_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def get_system_info():
    """Get system information for logging."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
    return info


def run_single_example(integration, text, output_file=None):
    """Run a single example through the pipeline."""
    logger.info("="*60)
    logger.info("Running single example")
    logger.info(f"Text preview: {text[:100]}...")
    
    # Process text
    result = integration.process_text(
        text=text,
        return_hypervectors=True,
        return_behavioral_features=True
    )
    
    # Log results
    logger.info(f"Status: {result['status']}")
    if result['status'] == 'success':
        logger.info(f"Tokens processed: {result.get('num_tokens', 'N/A')}")
        if 'hypervectors' in result:
            logger.info(f"Hypervector shape: {result['hypervectors']['shape']}")
            logger.info(f"Hypervector sparsity: {result['hypervectors']['sparsity']:.2%}")
        if 'pipeline_output' in result:
            output = result['pipeline_output']
            if 'merkle_root' in output:
                logger.info(f"Merkle root: {output['merkle_root'][:16]}...")
            if 'verification_score' in output:
                logger.info(f"Verification score: {output['verification_score']:.4f}")
    else:
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
    # Save result if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Result saved to {output_file}")
        
    return result


def run_benchmark_suite(integration):
    """Run a comprehensive benchmark suite."""
    logger.info("="*60)
    logger.info("Running benchmark suite")
    
    # Define test texts
    test_texts = [
        # Short text
        "The quick brown fox jumps over the lazy dog.",
        
        # Medium text
        """Artificial intelligence has transformed the landscape of modern computing.
        Machine learning models can now understand and generate human-like text,
        recognize images with superhuman accuracy, and solve complex problems
        that were once thought to be the exclusive domain of human intelligence.""",
        
        # Long text
        """The development of large language models represents a significant milestone
        in artificial intelligence research. These models, trained on vast amounts of
        text data, have demonstrated remarkable capabilities in understanding context,
        generating coherent responses, and even exhibiting forms of reasoning. However,
        they also raise important questions about bias, safety, and the nature of
        intelligence itself. As we continue to push the boundaries of what these models
        can do, we must also carefully consider their implications for society, ensuring
        that their development and deployment are guided by ethical principles and a
        commitment to beneficial outcomes for all of humanity. The challenge ahead is
        not just technical but also philosophical, requiring us to think deeply about
        what we want from artificial intelligence and how we can ensure it serves the
        common good.""",
        
        # Code example
        """def fibonacci(n):
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            elif n == 2:
                return [0, 1]
            else:
                fib = [0, 1]
                for i in range(2, n):
                    fib.append(fib[i-1] + fib[i-2])
                return fib""",
                
        # Scientific text
        """Quantum computing leverages the principles of quantum mechanics to process
        information in fundamentally new ways. Unlike classical bits that exist in
        either 0 or 1 states, quantum bits or qubits can exist in superposition,
        representing both states simultaneously. This property, combined with
        entanglement and quantum interference, enables quantum computers to explore
        multiple solution paths in parallel, potentially solving certain problems
        exponentially faster than classical computers."""
    ]
    
    # Run benchmark
    results = integration.run_benchmark(
        test_texts=test_texts,
        save_results=True,
        output_dir="benchmarks"
    )
    
    # Log summary
    logger.info("="*60)
    logger.info("Benchmark Summary:")
    stats = results['statistics']
    logger.info(f"Success rate: {stats['success_rate']:.1%}")
    logger.info(f"Average tokens: {stats['avg_tokens']:.1f}")
    logger.info(f"Total texts: {stats['total_texts']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Yi-34B through REV pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/rohanvinaik/LLM_models/yi-34b",
        help="Path to Yi-34B model"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=16.0,
        help="Memory limit in GB"
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=512,
        help="Segment size for processing"
    )
    parser.add_argument(
        "--hypervector-dim",
        type=int,
        default=10000,
        help="Dimension of hypervectors"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable model quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu, auto)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to process"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Log system info
    logger.info("="*60)
    logger.info("System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        logger.info(f"{key}: {value}")
    
    # Initialize integration
    logger.info("="*60)
    logger.info("Initializing Yi-34B REV Integration")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Memory limit: {args.memory_limit} GB")
    logger.info(f"Segment size: {args.segment_size}")
    logger.info(f"Hypervector dimension: {args.hypervector_dim}")
    logger.info(f"Use quantization: {not args.no_quantization}")
    
    integration = Yi34BREVIntegration(
        model_path=args.model_path,
        memory_limit_gb=args.memory_limit,
        segment_size=args.segment_size,
        hypervector_dim=args.hypervector_dim,
        use_quantization=not args.no_quantization,
        device=args.device
    )
    
    try:
        # Initialize model and pipeline
        integration.initialize_model()
        integration.initialize_pipeline()
        
        # Run requested operations
        if args.benchmark:
            results = run_benchmark_suite(integration)
        elif args.text:
            results = run_single_example(
                integration,
                args.text,
                args.output
            )
        else:
            # Run default example
            default_text = """The development of artificial intelligence represents one of the most
            significant technological advances of our time. As these systems become more sophisticated,
            they challenge our understanding of intelligence, creativity, and consciousness itself."""
            
            results = run_single_example(
                integration,
                default_text,
                args.output
            )
            
        logger.info("="*60)
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        integration.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    logger.info("Done!")


if __name__ == "__main__":
    main()