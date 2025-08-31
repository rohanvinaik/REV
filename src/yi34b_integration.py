"""
Integration module for Yi-34B model with REV pipeline.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from src.model_loader import Yi34BLoader
from src.rev_pipeline import REVPipeline
from src.executor.segment_runner import SegmentRunner
from src.hdc.encoder import HypervectorEncoder
from src.hdc.behavioral_sites import BehavioralSites
from src.verifier.blackbox import BlackBoxVerifier
from src.verifier.decision_aggregator import DecisionAggregator
from src.core.sequential import DualSequentialTest

logger = logging.getLogger(__name__)


class Yi34BREVIntegration:
    """Integration class for running Yi-34B through REV pipeline."""
    
    def __init__(
        self,
        model_path: str = "/Users/rohanvinaik/LLM_models/yi-34b",
        memory_limit_gb: float = 16.0,
        segment_size: int = 512,
        hypervector_dim: int = 10000,
        use_quantization: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize Yi-34B REV integration.
        
        Args:
            model_path: Path to Yi-34B model
            memory_limit_gb: Memory limit in GB
            segment_size: Size of segments for processing
            hypervector_dim: Dimension of hypervectors
            use_quantization: Whether to use model quantization
            device: Device to run on (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.memory_limit_gb = memory_limit_gb
        self.segment_size = segment_size
        self.hypervector_dim = hypervector_dim
        self.use_quantization = use_quantization
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model_loader = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.encoder = None
        self.behavioral_sites = None
        self.segment_runner = None
        
    def initialize_model(self):
        """Initialize Yi-34B model with memory-efficient loading."""
        logger.info("Initializing Yi-34B model...")
        
        # Determine quantization based on memory and settings
        load_in_8bit = False
        load_in_4bit = False
        
        # Disable quantization on Mac as bitsandbytes requires CUDA
        if self.use_quantization and torch.cuda.is_available():
            model_size_gb = 68  # Yi-34B is ~68GB in bfloat16
            if self.memory_limit_gb < model_size_gb / 4:
                load_in_4bit = True
                logger.info("Using 4-bit quantization for memory efficiency")
            elif self.memory_limit_gb < model_size_gb / 2:
                load_in_8bit = True
                logger.info("Using 8-bit quantization for memory efficiency")
        else:
            logger.info("Quantization disabled (not available on this platform)")
                
        # Initialize loader
        self.model_loader = Yi34BLoader(
            model_path=str(self.model_path),
            device=self.device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None
        )
        
        # Load model
        self.model, self.tokenizer = self.model_loader.load_model()
        logger.info("Model loaded successfully")
        
    def initialize_pipeline(self):
        """Initialize REV pipeline components."""
        logger.info("Initializing REV pipeline...")
        
        # Initialize hypervector encoder
        self.encoder = HypervectorEncoder(
            dimension=self.hypervector_dim,
            sparse=True,
            density=0.01
        )
        
        # Initialize behavioral sites
        self.behavioral_sites = BehavioralSites(
            dimension=self.hypervector_dim,
            num_sites=100,
            zoom_levels=3
        )
        
        # Initialize segment runner
        self.segment_runner = SegmentRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            segment_size=self.segment_size,
            memory_limit=int(self.memory_limit_gb * 1024 * 1024 * 1024),
            device=self.device
        )
        
        # Initialize REV pipeline
        self.pipeline = REVPipeline(
            segment_runner=self.segment_runner,
            encoder=self.encoder,
            behavioral_sites=self.behavioral_sites,
            decision_aggregator=DecisionAggregator(),
            memory_limit=int(self.memory_limit_gb * 1024 * 1024 * 1024)
        )
        
        logger.info("Pipeline initialized successfully")
        
    def process_text(
        self,
        text: str,
        comparison_model: Optional[str] = None,
        return_hypervectors: bool = True,
        return_behavioral_features: bool = True
    ) -> Dict[str, Any]:
        """
        Process text through Yi-34B and REV pipeline.
        
        Args:
            text: Input text to process
            comparison_model: Optional model to compare against
            return_hypervectors: Whether to return hypervectors
            return_behavioral_features: Whether to return behavioral features
            
        Returns:
            Dictionary with processing results
        """
        if self.model is None:
            self.initialize_model()
        if self.pipeline is None:
            self.initialize_pipeline()
            
        logger.info(f"Processing text of length {len(text)}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "yi-34b",
            "text_length": len(text),
            "segment_size": self.segment_size,
            "device": self.device
        }
        
        try:
            # Tokenize text
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
            num_tokens = tokens["input_ids"].shape[1]
            results["num_tokens"] = num_tokens
            
            # Process through pipeline
            pipeline_output = self.pipeline.process_document(
                text=text,
                model_name="yi-34b"
            )
            
            results["pipeline_output"] = pipeline_output
            
            # Extract hypervectors if requested
            if return_hypervectors and "hypervectors" in pipeline_output:
                results["hypervectors"] = {
                    "shape": pipeline_output["hypervectors"].shape,
                    "dtype": str(pipeline_output["hypervectors"].dtype),
                    "sparsity": np.mean(pipeline_output["hypervectors"] == 0)
                }
                
            # Extract behavioral features if requested
            if return_behavioral_features and "behavioral_features" in pipeline_output:
                results["behavioral_features"] = pipeline_output["behavioral_features"]
                
            # Compare with another model if specified
            if comparison_model:
                logger.info(f"Comparing with {comparison_model}")
                verifier = BlackBoxVerifier(
                    api_key=None,  # Set if needed
                    model_name=comparison_model
                )
                comparison_results = verifier.verify(
                    text=text,
                    reference_output=pipeline_output.get("output", ""),
                    threshold=0.9
                )
                results["comparison"] = comparison_results
                
            results["status"] = "success"
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
        
    def run_benchmark(
        self,
        test_texts: List[str],
        save_results: bool = True,
        output_dir: str = "benchmarks"
    ) -> Dict[str, Any]:
        """
        Run benchmark on multiple texts.
        
        Args:
            test_texts: List of texts to process
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark on {len(test_texts)} texts")
        
        benchmark_results = {
            "model": "yi-34b",
            "device": self.device,
            "memory_limit_gb": self.memory_limit_gb,
            "segment_size": self.segment_size,
            "hypervector_dim": self.hypervector_dim,
            "num_texts": len(test_texts),
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for i, text in enumerate(test_texts):
            logger.info(f"Processing text {i+1}/{len(test_texts)}")
            result = self.process_text(
                text,
                return_hypervectors=True,
                return_behavioral_features=True
            )
            benchmark_results["results"].append(result)
            
        # Calculate statistics
        successful = sum(1 for r in benchmark_results["results"] if r["status"] == "success")
        benchmark_results["statistics"] = {
            "success_rate": successful / len(test_texts) if test_texts else 0,
            "avg_tokens": np.mean([r.get("num_tokens", 0) for r in benchmark_results["results"]]),
            "total_texts": len(test_texts),
            "successful": successful,
            "failed": len(test_texts) - successful
        }
        
        # Save results if requested
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_path / f"yi34b_benchmark_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
                
            logger.info(f"Results saved to {filename}")
            benchmark_results["output_file"] = str(filename)
            
        return benchmark_results
        
    def cleanup(self):
        """Clean up resources."""
        if self.model_loader:
            self.model_loader.unload_model()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        logger.info("Resources cleaned up")