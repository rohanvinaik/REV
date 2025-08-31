#!/usr/bin/env python3
"""
Efficient Yi-34B pipeline runner using memory-mapped loading and segment processing.
Designed to handle the 68GB model with limited RAM.
"""

import os
import sys
import torch
import numpy as np
import json
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.yi34b_efficient_loader import Yi34BEfficientLoader
from src.hdc.encoder import HypervectorEncoder
from src.hdc.behavioral_sites import BehavioralSites
from src.hdc.binding_operations import BindingOperations
from src.hdc.error_correction import ErrorCorrection
from src.hypervector.hamming import HammingDistanceOptimized
from src.crypto.merkle import IncrementalMerkleTree as MerkleTree
from src.verifier.decision_aggregator import DecisionAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Yi34BEfficientPipeline:
    """
    Efficient pipeline for processing Yi-34B model with REV framework.
    Uses segment-wise processing and memory-mapped loading.
    """
    
    def __init__(
        self,
        model_path: str = "/Users/rohanvinaik/LLM_models/yi-34b",
        segment_size: int = 256,
        layers_per_segment: int = 5,
        hypervector_dim: int = 8192,
        offload_folder: str = "/tmp/yi34b_offload"
    ):
        """
        Initialize efficient pipeline.
        
        Args:
            model_path: Path to Yi-34B model
            segment_size: Token segment size
            layers_per_segment: Layers to process at once
            hypervector_dim: Dimension of hypervectors
            offload_folder: Folder for weight offloading
        """
        self.model_path = Path(model_path)
        self.segment_size = segment_size
        self.layers_per_segment = layers_per_segment
        self.hypervector_dim = hypervector_dim
        self.offload_folder = offload_folder
        
        # Initialize components
        self.loader = None
        self.encoder = None
        self.behavioral_sites = None
        self.binding_ops = None
        self.error_correction = None
        self.hamming = None
        self.aggregator = None
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "yi-34b",
            "config": {
                "segment_size": segment_size,
                "layers_per_segment": layers_per_segment,
                "hypervector_dim": hypervector_dim
            },
            "segments": [],
            "memory_usage": [],
            "performance": {}
        }
        
    def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Initialize efficient loader
        self.loader = Yi34BEfficientLoader(
            model_path=str(self.model_path),
            load_in_segments=True,
            offload_folder=self.offload_folder,
            use_mmap=True
        )
        
        # Load config and tokenizer
        self.loader.load_config_and_tokenizer()
        self.loader.map_model_shards()
        
        # Initialize HDC components
        from src.hdc.encoder import HypervectorConfig
        encoder_config = HypervectorConfig(
            dimension=self.hypervector_dim,
            sparsity=0.01,
            encoding_mode="rev"
        )
        self.encoder = HypervectorEncoder(config=encoder_config)
        
        self.behavioral_sites = BehavioralSites(
            hdc_config=encoder_config
        )
        
        self.binding_ops = BindingOperations(
            dimension=self.hypervector_dim
        )
        
        from src.hdc.error_correction import ErrorCorrectionConfig
        error_config = ErrorCorrectionConfig(
            dimension=self.hypervector_dim,
            parity_overhead=0.25
        )
        self.error_correction = ErrorCorrection(
            config=error_config
        )
        
        self.hamming = HammingDistanceOptimized()
        
        self.aggregator = DecisionAggregator()
        
        logger.info("Components initialized successfully")
        
    def process_text_segments(self, text: str) -> Dict[str, Any]:
        """
        Process text through Yi-34B in segments.
        
        Args:
            text: Input text
            
        Returns:
            Processing results
        """
        logger.info(f"Processing text of length {len(text)}")
        
        # Tokenize text
        inputs = self.loader.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Yi-34B max context
        )
        
        input_ids = inputs["input_ids"]
        num_tokens = input_ids.shape[1]
        
        logger.info(f"Tokenized to {num_tokens} tokens")
        
        # Split into segments
        segments = []
        for i in range(0, num_tokens, self.segment_size):
            end_idx = min(i + self.segment_size, num_tokens)
            segment_ids = input_ids[:, i:end_idx]
            segments.append(segment_ids)
            
        logger.info(f"Split into {len(segments)} segments")
        
        segment_results = []
        hypervectors = []
        
        # Process each segment
        for idx, segment_ids in enumerate(segments):
            logger.info(f"Processing segment {idx+1}/{len(segments)}")
            
            start_time = time.time()
            
            # Get memory usage before
            memory_before = psutil.Process().memory_info().rss / 1024**3
            
            # Process through model layers
            segment_output = self.loader.process_segments(
                segment_ids,
                segment_size=self.segment_size,
                layers_per_segment=self.layers_per_segment
            )
            
            # Get memory usage after
            memory_after = psutil.Process().memory_info().rss / 1024**3
            memory_delta = memory_after - memory_before
            
            process_time = time.time() - start_time
            
            # Generate hypervector from segment output
            if "final_output" in segment_output:
                # Encode to hypervector
                hv = self.encoder.encode(segment_output["final_output"].flatten())
                hypervectors.append(hv)
                
                # Extract behavioral sites
                sites = self.behavioral_sites.extract_sites(hv)
                
                segment_results.append({
                    "segment_id": idx,
                    "tokens": segment_ids.shape[1],
                    "process_time": process_time,
                    "memory_delta_gb": memory_delta,
                    "num_sites": len(sites),
                    "hypervector_sparsity": np.mean(hv == 0)
                })
                
            self.results["memory_usage"].append(memory_after)
            
        # Combine hypervectors
        if hypervectors:
            combined_hv = self.binding_ops.bind_sequence(hypervectors)
            
            # Apply error correction
            protected_hv = self.error_correction.encode(combined_hv)
            
            # Create Merkle tree from segments
            merkle_leaves = [hv.tobytes() for hv in hypervectors]
            merkle_tree = MerkleTree(merkle_leaves)
            
            self.results["segments"] = segment_results
            self.results["merkle_root"] = merkle_tree.root.hex() if merkle_tree.root else None
            self.results["combined_hypervector"] = {
                "shape": combined_hv.shape,
                "sparsity": float(np.mean(combined_hv == 0)),
                "protected_size": len(protected_hv)
            }
            
        return self.results
        
    def verify_consistency(self, reference_hypervectors: List[np.ndarray]) -> Dict[str, Any]:
        """
        Verify consistency between different runs.
        
        Args:
            reference_hypervectors: Reference hypervectors to compare against
            
        Returns:
            Verification results
        """
        logger.info("Verifying consistency...")
        
        verification_results = {
            "scores": [],
            "distances": [],
            "decision": None
        }
        
        if not reference_hypervectors:
            return verification_results
            
        # Compare with reference
        for ref_hv in reference_hypervectors:
            distance = self.hamming.compute(
                self.results.get("combined_hypervector", {}).get("data", np.zeros(self.hypervector_dim)),
                ref_hv
            )
            
            similarity = 1.0 - (distance / self.hypervector_dim)
            verification_results["scores"].append(similarity)
            verification_results["distances"].append(distance)
            
        # Aggregate decision
        if verification_results["scores"]:
            decision = self.aggregator.aggregate(verification_results["scores"])
            verification_results["decision"] = decision
            
        return verification_results
        
    def run_full_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Run full pipeline on text.
        
        Args:
            text: Input text
            
        Returns:
            Complete results
        """
        logger.info("="*60)
        logger.info("Starting Yi-34B Efficient Pipeline")
        logger.info("="*60)
        
        # Record start time
        pipeline_start = time.time()
        
        # Initialize components if needed
        if self.loader is None:
            self.initialize_components()
            
        # Process text segments
        results = self.process_text_segments(text)
        
        # Calculate performance metrics
        pipeline_time = time.time() - pipeline_start
        
        self.results["performance"] = {
            "total_time": pipeline_time,
            "avg_segment_time": np.mean([s["process_time"] for s in self.results["segments"]]) if self.results["segments"] else 0,
            "peak_memory_gb": max(self.results["memory_usage"]) if self.results["memory_usage"] else 0,
            "avg_memory_gb": np.mean(self.results["memory_usage"]) if self.results["memory_usage"] else 0
        }
        
        logger.info("="*60)
        logger.info("Pipeline Complete")
        logger.info(f"Total time: {pipeline_time:.2f}s")
        logger.info(f"Peak memory: {self.results['performance']['peak_memory_gb']:.2f} GB")
        logger.info(f"Segments processed: {len(self.results['segments'])}")
        
        return self.results
        
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.loader:
            self.loader.cleanup()
            
        # Clear components
        self.loader = None
        self.encoder = None
        self.behavioral_sites = None
        self.binding_ops = None
        self.error_correction = None
        self.hamming = None
        self.aggregator = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Yi-34B through efficient REV pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/rohanvinaik/LLM_models/yi-34b",
        help="Path to Yi-34B model"
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=256,
        help="Token segment size"
    )
    parser.add_argument(
        "--layers-per-segment",
        type=int,
        default=5,
        help="Number of layers to process at once"
    )
    parser.add_argument(
        "--hypervector-dim",
        type=int,
        default=8192,
        help="Dimension of hypervectors"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Default text if none provided
    if args.text is None:
        args.text = """The development of large language models represents a significant 
        milestone in artificial intelligence. These models, trained on vast amounts of text,
        demonstrate remarkable capabilities in understanding and generating human-like text.
        However, verifying their behavior and ensuring consistency remains a challenge."""
        
    # Log system info
    logger.info("System Information:")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    
    # Initialize pipeline
    pipeline = Yi34BEfficientPipeline(
        model_path=args.model_path,
        segment_size=args.segment_size,
        layers_per_segment=args.layers_per_segment,
        hypervector_dim=args.hypervector_dim
    )
    
    try:
        # Run pipeline
        results = pipeline.run_full_pipeline(args.text)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
            
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Model: Yi-34B")
        print(f"Segments processed: {len(results['segments'])}")
        print(f"Total time: {results['performance']['total_time']:.2f}s")
        print(f"Peak memory: {results['performance']['peak_memory_gb']:.2f} GB")
        print(f"Avg memory: {results['performance']['avg_memory_gb']:.2f} GB")
        if results.get('merkle_root'):
            print(f"Merkle root: {results['merkle_root'][:32]}...")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        pipeline.cleanup()
        
    logger.info("Done!")


if __name__ == "__main__":
    main()