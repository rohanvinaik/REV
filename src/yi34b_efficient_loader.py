"""
Efficient loader for Yi-34B model using memory mapping and lazy loading.
"""

import os
import torch
import gc
import mmap
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama import LlamaForCausalLM
import logging

logger = logging.getLogger(__name__)


class Yi34BEfficientLoader:
    """
    Efficient loader for Yi-34B that uses memory mapping and segment-wise loading.
    Designed to handle 68GB model on systems with limited RAM.
    """
    
    def __init__(
        self,
        model_path: str = "/Users/rohanvinaik/LLM_models/yi-34b",
        load_in_segments: bool = True,
        max_shard_size: str = "5GB",
        offload_folder: Optional[str] = "/tmp/yi34b_offload",
        use_mmap: bool = True
    ):
        """
        Initialize efficient Yi-34B loader.
        
        Args:
            model_path: Path to Yi-34B model
            load_in_segments: Whether to load model in segments
            max_shard_size: Maximum size per shard for loading
            offload_folder: Folder for offloading weights
            use_mmap: Whether to use memory mapping
        """
        self.model_path = Path(model_path)
        self.load_in_segments = load_in_segments
        self.max_shard_size = max_shard_size
        self.offload_folder = Path(offload_folder) if offload_folder else None
        self.use_mmap = use_mmap
        
        # Model components
        self.config = None
        self.tokenizer = None
        self.model_shards = {}
        self.loaded_layers = {}
        self.layer_to_shard_map = {}
        
        # Create offload folder if needed
        if self.offload_folder:
            self.offload_folder.mkdir(parents=True, exist_ok=True)
            
    def load_config_and_tokenizer(self):
        """Load model configuration and tokenizer."""
        logger.info("Loading configuration and tokenizer...")
        
        # Load config
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Config loaded: {self.config.num_hidden_layers} layers, "
                   f"hidden size {self.config.hidden_size}")
        
    def map_model_shards(self):
        """Map model shards to layers for efficient loading."""
        logger.info("Mapping model shards...")
        
        # Read the index file
        index_file = self.model_path / "model.safetensors.index.json"
        if not index_file.exists():
            index_file = self.model_path / "pytorch_model.bin.index.json"
            
        with open(index_file, 'r') as f:
            index = json.load(f)
            
        # Map weights to files
        weight_map = index.get("weight_map", {})
        
        # Group by shard file
        for weight_name, shard_file in weight_map.items():
            if shard_file not in self.model_shards:
                self.model_shards[shard_file] = []
            self.model_shards[shard_file].append(weight_name)
            
            # Extract layer number if present
            if "layers." in weight_name:
                layer_num = int(weight_name.split("layers.")[1].split(".")[0])
                if layer_num not in self.layer_to_shard_map:
                    self.layer_to_shard_map[layer_num] = set()
                self.layer_to_shard_map[layer_num].add(shard_file)
                
        logger.info(f"Found {len(self.model_shards)} shard files")
        logger.info(f"Mapped {len(self.layer_to_shard_map)} layers")
        
    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load weights for a specific layer using memory mapping.
        
        Args:
            layer_idx: Layer index to load
            
        Returns:
            Dictionary of layer weights
        """
        if layer_idx in self.loaded_layers:
            return self.loaded_layers[layer_idx]
            
        logger.debug(f"Loading layer {layer_idx}")
        
        layer_weights = {}
        
        # Get shard files for this layer
        shard_files = self.layer_to_shard_map.get(layer_idx, set())
        
        for shard_file in shard_files:
            shard_path = self.model_path / shard_file
            
            # Use safetensors for efficient loading
            if shard_path.suffix == ".safetensors":
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if f"layers.{layer_idx}." in key:
                            if self.use_mmap:
                                # Memory-mapped loading
                                layer_weights[key] = f.get_tensor(key)
                            else:
                                # Regular loading
                                layer_weights[key] = f.get_tensor(key).clone()
            else:
                # Fallback to PyTorch loading
                weights = torch.load(shard_path, map_location="cpu", mmap=self.use_mmap)
                for key, value in weights.items():
                    if f"layers.{layer_idx}." in key:
                        layer_weights[key] = value
                        
        # Cache loaded layer
        self.loaded_layers[layer_idx] = layer_weights
        
        return layer_weights
        
    def offload_layer(self, layer_idx: int):
        """
        Offload a layer to disk to free memory.
        
        Args:
            layer_idx: Layer index to offload
        """
        if layer_idx not in self.loaded_layers:
            return
            
        if self.offload_folder:
            offload_path = self.offload_folder / f"layer_{layer_idx}.pt"
            torch.save(self.loaded_layers[layer_idx], offload_path)
            logger.debug(f"Offloaded layer {layer_idx} to {offload_path}")
            
        del self.loaded_layers[layer_idx]
        gc.collect()
        
    def reload_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Reload an offloaded layer from disk.
        
        Args:
            layer_idx: Layer index to reload
            
        Returns:
            Layer weights
        """
        if layer_idx in self.loaded_layers:
            return self.loaded_layers[layer_idx]
            
        if self.offload_folder:
            offload_path = self.offload_folder / f"layer_{layer_idx}.pt"
            if offload_path.exists():
                logger.debug(f"Reloading layer {layer_idx} from {offload_path}")
                self.loaded_layers[layer_idx] = torch.load(offload_path, map_location="cpu")
                return self.loaded_layers[layer_idx]
                
        # If not offloaded, load from original shards
        return self.load_layer_weights(layer_idx)
        
    def process_segments(
        self,
        input_ids: torch.Tensor,
        segment_size: int = 512,
        layers_per_segment: int = 5
    ) -> Dict[str, Any]:
        """
        Process input through model in segments to minimize memory usage.
        
        Args:
            input_ids: Input token IDs
            segment_size: Size of each segment
            layers_per_segment: Number of layers to keep in memory at once
            
        Returns:
            Processing results
        """
        if self.config is None:
            self.load_config_and_tokenizer()
            self.map_model_shards()
            
        num_layers = self.config.num_hidden_layers
        seq_length = input_ids.shape[1]
        
        results = {
            "hidden_states": [],
            "layer_outputs": {},
            "memory_usage": []
        }
        
        # Initialize embeddings
        logger.info("Processing embeddings...")
        embeddings = self._get_embeddings(input_ids)
        hidden_states = embeddings
        
        # Process through layers in groups
        for layer_group_start in range(0, num_layers, layers_per_segment):
            layer_group_end = min(layer_group_start + layers_per_segment, num_layers)
            
            logger.info(f"Processing layers {layer_group_start}-{layer_group_end}")
            
            # Load layers in this group
            for layer_idx in range(layer_group_start, layer_group_end):
                self.load_layer_weights(layer_idx)
                
            # Process through loaded layers
            for layer_idx in range(layer_group_start, layer_group_end):
                hidden_states = self._forward_layer(hidden_states, layer_idx)
                
                # Store intermediate results if needed
                if layer_idx % 10 == 0:
                    # Convert to float32 before numpy conversion (handle bfloat16)
                    results["layer_outputs"][layer_idx] = hidden_states.mean(dim=1).float().cpu().numpy()
                    
            # Offload processed layers to free memory
            for layer_idx in range(layer_group_start, layer_group_end):
                self.offload_layer(layer_idx)
                
            # Record memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
            else:
                import psutil
                memory_used = psutil.Process().memory_info().rss / 1024**3
            results["memory_usage"].append(memory_used)
            
        results["hidden_states"] = hidden_states
        results["final_output"] = hidden_states.mean(dim=1).float().cpu().numpy()
        
        return results
        
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings."""
        # Load embedding weights
        embed_weights = None
        for shard_file in self.model_shards:
            shard_path = self.model_path / shard_file
            if shard_path.suffix == ".safetensors":
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    if "model.embed_tokens.weight" in f.keys():
                        embed_weights = f.get_tensor("model.embed_tokens.weight")
                        break
                        
        if embed_weights is None:
            raise ValueError("Could not find embedding weights")
            
        return torch.nn.functional.embedding(input_ids, embed_weights)
        
    def _forward_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Forward pass through a single layer.
        
        Args:
            hidden_states: Input hidden states
            layer_idx: Layer index
            
        Returns:
            Output hidden states
        """
        layer_weights = self.loaded_layers[layer_idx]
        
        # Simplified forward pass (actual implementation would be more complex)
        # This is a placeholder - real implementation would use proper Llama layer logic
        
        # Layer norm
        if f"model.layers.{layer_idx}.input_layernorm.weight" in layer_weights:
            ln_weight = layer_weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                normalized_shape=ln_weight.shape,
                weight=ln_weight,
                eps=self.config.rms_norm_eps
            )
            
        # Self-attention (simplified)
        # ... (would implement full attention mechanism here)
        
        # MLP (simplified)
        # ... (would implement full MLP here)
        
        return hidden_states
        
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        segment_size: int = 512
    ) -> str:
        """
        Generate text with streaming processing to minimize memory.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            segment_size: Segment size for processing
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            self.load_config_and_tokenizer()
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        
        # Process through model segments
        results = self.process_segments(
            input_ids,
            segment_size=segment_size
        )
        
        # Simple generation (placeholder - real implementation would be more complex)
        generated_text = prompt + " [Generated content would appear here]"
        
        return generated_text
        
    def cleanup(self):
        """Clean up loaded weights and free memory."""
        logger.info("Cleaning up model weights...")
        
        # Clear loaded layers
        self.loaded_layers.clear()
        
        # Clear model shards
        self.model_shards.clear()
        
        # Remove offload folder if exists
        if self.offload_folder and self.offload_folder.exists():
            import shutil
            shutil.rmtree(self.offload_folder)
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleanup complete")