#!/usr/bin/env python3
"""
E2E Pipeline for Yi-34B with intelligent behavioral segmentation.
Uses prompt injection to determine structural similarity sites for optimal cuts.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

sys.path.insert(0, 'src')

from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.behavioral_sites import BehavioralSites
from src.hdc.binding_operations import BindingOperations
from src.crypto.merkle import IncrementalMerkleTree
from src.verifier.decision_aggregator import DecisionAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class BehavioralPrompts:
    """Complex determinative prompts for structural analysis."""
    
    # Prompts designed to elicit different behavioral patterns across layers
    STRUCTURAL_PROBES = [
        # Factual recall - early layers
        {"prompt": "The capital of France is", "layer_affinity": "early", "type": "factual"},
        
        # Syntactic understanding - early-mid layers  
        {"prompt": "Complete: The quick brown fox", "layer_affinity": "early_mid", "type": "syntactic"},
        
        # Semantic reasoning - mid layers
        {"prompt": "If all roses are flowers and all flowers need water, then roses", 
         "layer_affinity": "mid", "type": "semantic"},
        
        # Abstract reasoning - mid-late layers
        {"prompt": "The relationship between cause and effect is similar to the relationship between",
         "layer_affinity": "mid_late", "type": "abstract"},
        
        # Creative generation - late layers
        {"prompt": "Imagine a world where gravity works backwards. Describe",
         "layer_affinity": "late", "type": "creative"},
         
        # Multi-hop reasoning
        {"prompt": "John is taller than Mary. Mary is taller than Sue. Who is shortest?",
         "layer_affinity": "distributed", "type": "reasoning"},
         
        # Code understanding
        {"prompt": "def fibonacci(n): if n <= 1: return n; else:",
         "layer_affinity": "specialized", "type": "code"},
         
        # Mathematical reasoning
        {"prompt": "Solve for x: 2x + 5 = 13. x equals",
         "layer_affinity": "specialized", "type": "math"}
    ]
    
    @staticmethod
    def get_probe_set() -> List[Dict]:
        """Get comprehensive probe set for behavioral analysis."""
        return BehavioralPrompts.STRUCTURAL_PROBES


class IntelligentSegmentation:
    """Determine optimal layer segmentation through behavioral analysis."""
    
    def __init__(self, model, tokenizer, hypervector_dim=8192):
        self.model = model
        self.tokenizer = tokenizer
        self.hypervector_dim = hypervector_dim
        self.layer_similarities = {}
        self.cut_points = []
        
    def analyze_layer_behavior(self, prompts: List[Dict]) -> Dict[int, np.ndarray]:
        """Analyze behavioral patterns across layers using probe prompts."""
        logger.info("Analyzing layer behaviors with probe prompts...")
        
        layer_signatures = {}
        num_layers = self.model.config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            signatures = []
            
            for probe in prompts:
                # Get activations for this probe at this layer
                inputs = self.tokenizer(probe["prompt"], return_tensors="pt")
                
                # Hook to capture layer output
                activation = None
                def hook(module, input, output):
                    nonlocal activation
                    activation = output[0].detach()
                
                # Register hook for specific layer
                layer = self.model.model.layers[layer_idx]
                handle = layer.register_forward_hook(hook)
                
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                handle.remove()
                
                # Convert activation to signature
                if activation is not None:
                    # Use statistical moments as lightweight signature
                    # Flatten to ensure we have 1D tensors
                    mean_feat = activation.mean(dim=(0,1)).flatten()
                    std_feat = activation.std(dim=(0,1)).flatten()
                    max_feat = activation.abs().max(dim=1)[0].mean(dim=0).flatten()
                    
                    sig = torch.cat([mean_feat, std_feat, max_feat]).cpu().numpy()
                    signatures.append(sig)
                    
            # Combine signatures for this layer
            if signatures:
                layer_signatures[layer_idx] = np.concatenate(signatures)
                
        return layer_signatures
        
    def find_similarity_cuts(self, layer_signatures: Dict[int, np.ndarray], 
                            threshold: float = 0.15) -> List[int]:
        """Find optimal cut points based on behavioral similarity changes."""
        logger.info("Finding optimal segmentation points...")
        
        cuts = [0]  # Always start at layer 0
        prev_signature = None
        
        for layer_idx in sorted(layer_signatures.keys()):
            curr_signature = layer_signatures[layer_idx]
            
            if prev_signature is not None:
                # Compute similarity between consecutive layers
                similarity = np.corrcoef(prev_signature, curr_signature)[0, 1]
                
                # Detect significant behavioral shift
                if similarity < (1.0 - threshold):
                    cuts.append(layer_idx)
                    logger.info(f"Behavioral shift detected at layer {layer_idx} (similarity: {similarity:.3f})")
                    
            prev_signature = curr_signature
            
        cuts.append(self.model.config.num_hidden_layers)
        
        # Ensure reasonable segment sizes (min 2, max 10 layers)
        refined_cuts = [cuts[0]]
        for cut in cuts[1:]:
            if cut - refined_cuts[-1] >= 2:
                refined_cuts.append(cut)
        
        return refined_cuts


class Yi34BE2EPipeline:
    """Streamlined E2E pipeline for Yi-34B with intelligent segmentation."""
    
    def __init__(self, model_path="/Users/rohanvinaik/LLM_models/yi-34b"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.segmentation = None
        self.segments = []
        
    def load_model(self):
        """Load Yi-34B model and tokenizer with memory-efficient strategy."""
        logger.info(f"Loading Yi-34B from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Use CPU offloading for large model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU to avoid memory issues
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            offload_folder="/tmp/yi34b_offload"
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def analyze_structure(self):
        """Analyze model structure using behavioral probes."""
        if self.model is None:
            self.load_model()
            
        # Get probe prompts
        prompts = BehavioralPrompts.get_probe_set()
        
        # Analyze behavioral patterns
        self.segmentation = IntelligentSegmentation(
            self.model, 
            self.tokenizer
        )
        
        signatures = self.segmentation.analyze_layer_behavior(prompts)
        cuts = self.segmentation.find_similarity_cuts(signatures)
        
        # Create segments based on behavioral analysis
        self.segments = []
        for i in range(len(cuts) - 1):
            self.segments.append({
                "start": cuts[i],
                "end": cuts[i + 1],
                "layers": cuts[i + 1] - cuts[i]
            })
            
        logger.info(f"Created {len(self.segments)} segments based on behavioral analysis")
        for i, seg in enumerate(self.segments):
            logger.info(f"  Segment {i}: layers {seg['start']}-{seg['end']} ({seg['layers']} layers)")
            
        return self.segments
        
    def process_with_rev(self, text: str) -> Dict[str, Any]:
        """Process text through REV pipeline with intelligent segmentation."""
        
        if not self.segments:
            self.analyze_structure()
            
        logger.info(f"Processing text: {text[:100]}...")
        
        # Initialize REV components
        config = HypervectorConfig(dimension=8192, encoding_mode="rev")
        encoder = HypervectorEncoder(config=config)
        behavioral_sites = BehavioralSites(hdc_config=config)
        binding_ops = BindingOperations(dimension=8192)
        merkle_tree = IncrementalMerkleTree()
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
        
        segment_hypervectors = []
        segment_results = []
        
        # Process through segments
        for i, segment in enumerate(self.segments):
            logger.info(f"Processing segment {i+1}/{len(self.segments)}")
            
            # Extract features from segment layers
            with torch.no_grad():
                # Get hidden states from specific layers
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
                # Aggregate hidden states from segment layers
                segment_states = []
                for layer_idx in range(segment['start'], segment['end']):
                    if layer_idx < len(outputs.hidden_states):
                        segment_states.append(outputs.hidden_states[layer_idx])
                        
                if segment_states:
                    # Combine segment states
                    combined = torch.stack(segment_states).mean(dim=0)
                    
                    # Convert to hypervector
                    features = combined.mean(dim=1).float().cpu().numpy().flatten()
                    hypervector = encoder.encode(features)
                    segment_hypervectors.append(hypervector)
                    
                    # Extract behavioral sites
                    sites = behavioral_sites.extract_sites(hypervector)
                    
                    # Add to Merkle tree
                    merkle_tree.add_leaf(hypervector.tobytes())
                    
                    segment_results.append({
                        "segment": i,
                        "layers": f"{segment['start']}-{segment['end']}",
                        "sites_found": len(sites),
                        "hv_sparsity": float(np.mean(hypervector == 0))
                    })
                    
        # Combine all segments
        if segment_hypervectors:
            final_hypervector = binding_ops.bind_sequence(segment_hypervectors)
            
            return {
                "status": "success",
                "segments": segment_results,
                "merkle_root": merkle_tree.root.hex() if merkle_tree.root else None,
                "final_hypervector_stats": {
                    "dimension": len(final_hypervector),
                    "sparsity": float(np.mean(final_hypervector == 0))
                },
                "text_length": len(text),
                "num_tokens": inputs['input_ids'].shape[1]
            }
            
        return {"status": "error", "message": "No segments processed"}
        
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    """Run E2E pipeline with intelligent segmentation."""
    
    # Complex determinative prompt for testing
    test_prompt = """
    Analyze the following scenario and provide reasoning:
    
    A quantum computer can solve certain problems exponentially faster than classical computers.
    However, quantum states are fragile and susceptible to decoherence. Given these constraints,
    explain how error correction codes enable practical quantum computation, and describe the 
    relationship between logical qubits and physical qubits in a fault-tolerant quantum computer.
    
    Consider the threshold theorem and its implications for scalability.
    """
    
    pipeline = Yi34BE2EPipeline()
    
    try:
        # Run full pipeline
        results = pipeline.process_with_rev(test_prompt)
        
        # Display results
        print("\n" + "="*60)
        print("Yi-34B E2E Pipeline Results")
        print("="*60)
        
        if results["status"] == "success":
            print(f"✓ Processed {results['num_tokens']} tokens")
            print(f"✓ Created {len(results['segments'])} behavioral segments")
            
            for seg in results['segments']:
                print(f"  - Segment {seg['segment']}: layers {seg['layers']}, "
                      f"{seg['sites_found']} sites, {seg['hv_sparsity']:.1%} sparse")
                      
            print(f"✓ Merkle root: {results['merkle_root'][:32]}...")
            print(f"✓ Final HV dimension: {results['final_hypervector_stats']['dimension']}")
            print(f"✓ Final HV sparsity: {results['final_hypervector_stats']['sparsity']:.1%}")
        else:
            print(f"✗ Error: {results.get('message', 'Unknown error')}")
            
    finally:
        pipeline.cleanup()
        
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()