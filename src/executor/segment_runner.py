"""
Segment Execution Framework for memory-bounded model execution.

This module implements efficient segment-wise model execution with parameter
loading/offloading, KV cache management, and activation extraction.
"""

import gc
import os
import psutil
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass, field
import numpy as np
import torch
from collections import deque
import warnings
from contextlib import contextmanager


@dataclass
class SegmentConfig:
    """Configuration for segment execution."""
    
    segment_size: int = 512
    overlap_size: int = 64
    max_memory_gb: float = 4.0
    offload_to_disk: bool = True
    cache_dir: str = "/tmp/rev_cache"
    extraction_sites: List[str] = field(default_factory=lambda: [
        "embeddings",
        "attention.0", "attention.6", "attention.11",
        "mlp.0", "mlp.6", "mlp.11",
        "layer_norm.final"
    ])
    use_fp16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class KVCache:
    """Key-Value cache for attention layers."""
    
    keys: Dict[int, torch.Tensor] = field(default_factory=dict)
    values: Dict[int, torch.Tensor] = field(default_factory=dict)
    max_seq_len: int = 2048
    current_pos: int = 0
    
    def update(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor):
        """Update cache for a specific layer."""
        batch_size, seq_len, hidden_dim = new_keys.shape
        
        if layer_idx not in self.keys:
            # Initialize cache for this layer
            self.keys[layer_idx] = torch.zeros(
                batch_size, self.max_seq_len, hidden_dim,
                dtype=new_keys.dtype, device=new_keys.device
            )
            self.values[layer_idx] = torch.zeros(
                batch_size, self.max_seq_len, hidden_dim,
                dtype=new_values.dtype, device=new_values.device
            )
        
        # Update cache with new keys/values
        end_pos = min(self.current_pos + seq_len, self.max_seq_len)
        actual_len = end_pos - self.current_pos
        
        self.keys[layer_idx][:, self.current_pos:end_pos] = new_keys[:, :actual_len]
        self.values[layer_idx][:, self.current_pos:end_pos] = new_values[:, :actual_len]
        
        self.current_pos = end_pos
    
    def get(self, layer_idx: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values up to seq_len."""
        if layer_idx not in self.keys:
            return None, None
        
        return (
            self.keys[layer_idx][:, :seq_len],
            self.values[layer_idx][:, :seq_len]
        )
    
    def clear(self):
        """Clear all cached keys/values."""
        self.keys.clear()
        self.values.clear()
        self.current_pos = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SegmentRunner:
    """
    Memory-bounded segment execution for large language models.
    
    Implements efficient execution with parameter loading/offloading,
    KV cache management, and activation extraction at restriction sites.
    """
    
    def __init__(self, config: Optional[SegmentConfig] = None):
        """
        Initialize segment runner.
        
        Args:
            config: Segment execution configuration
        """
        self.config = config or SegmentConfig()
        
        # Memory management
        self.memory_limit = self.config.max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
        
        # KV cache for attention
        self.kv_cache = KVCache(max_seq_len=self.config.segment_size * 4)
        
        # Parameter offloading
        self.offloaded_params = {}
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Activation storage
        self.activations = {}
        
        # Overlapping window buffer
        self.overlap_buffer = deque(maxlen=self.config.overlap_size)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024 * 1024)
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        return self.get_memory_usage() < self.config.max_memory_gb
    
    @contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient execution."""
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Set to evaluation mode to save memory
        original_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        
        try:
            yield
        finally:
            # Restore original mode
            torch.set_grad_enabled(original_mode)
            
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def offload_parameters(self, model: torch.nn.Module, layer_indices: List[int]):
        """
        Offload model parameters to disk/CPU.
        
        Args:
            model: Model to offload parameters from
            layer_indices: Indices of layers to offload
        """
        if not self.config.offload_to_disk:
            return
        
        for idx in layer_indices:
            layer_name = f"layer_{idx}"
            
            # Get layer (assuming transformer architecture)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layer = model.transformer.h[idx]
            elif hasattr(model, 'layers'):
                layer = model.layers[idx]
            else:
                continue
            
            # Save parameters to disk
            param_path = os.path.join(self.config.cache_dir, f"{layer_name}.pt")
            param_dict = {name: param.cpu() for name, param in layer.named_parameters()}
            torch.save(param_dict, param_path)
            
            # Clear from GPU
            for param in layer.parameters():
                param.data = torch.empty(0)
            
            self.offloaded_params[layer_name] = param_path
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def load_parameters(self, model: torch.nn.Module, layer_indices: List[int]):
        """
        Load offloaded parameters back to model.
        
        Args:
            model: Model to load parameters into
            layer_indices: Indices of layers to load
        """
        for idx in layer_indices:
            layer_name = f"layer_{idx}"
            
            if layer_name not in self.offloaded_params:
                continue
            
            # Load parameters from disk
            param_path = self.offloaded_params[layer_name]
            param_dict = torch.load(param_path, map_location='cpu')
            
            # Get layer
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layer = model.transformer.h[idx]
            elif hasattr(model, 'layers'):
                layer = model.layers[idx]
            else:
                continue
            
            # Restore parameters
            for name, param in param_dict.items():
                if hasattr(layer, name.split('.')[0]):
                    target = layer
                    for attr in name.split('.'):
                        target = getattr(target, attr) if hasattr(target, attr) else target
                    if isinstance(target, torch.nn.Parameter):
                        target.data = param.cuda() if torch.cuda.is_available() else param
    
    def extract_activations(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        extraction_sites: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations at restriction sites.
        
        Args:
            model: Model to extract from
            input_ids: Input token IDs
            extraction_sites: Sites to extract activations from
            
        Returns:
            Dictionary mapping site names to activation tensors
        """
        extraction_sites = extraction_sites or self.config.extraction_sites
        activations = {}
        hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                # Store activation
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks
        for site in extraction_sites:
            parts = site.split('.')
            
            if parts[0] == 'embeddings':
                if hasattr(model, 'embeddings'):
                    hook = model.embeddings.register_forward_hook(create_hook(site))
                    hooks.append(hook)
                    
            elif parts[0] == 'attention':
                layer_idx = int(parts[1]) if len(parts) > 1 else 0
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    if layer_idx < len(model.transformer.h):
                        layer = model.transformer.h[layer_idx]
                        if hasattr(layer, 'attn'):
                            hook = layer.attn.register_forward_hook(create_hook(site))
                            hooks.append(hook)
                            
            elif parts[0] == 'mlp':
                layer_idx = int(parts[1]) if len(parts) > 1 else 0
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    if layer_idx < len(model.transformer.h):
                        layer = model.transformer.h[layer_idx]
                        if hasattr(layer, 'mlp'):
                            hook = layer.mlp.register_forward_hook(create_hook(site))
                            hooks.append(hook)
                            
            elif parts[0] == 'layer_norm':
                if parts[1] == 'final' and hasattr(model, 'ln_f'):
                    hook = model.ln_f.register_forward_hook(create_hook(site))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def process_segment(
        self,
        model: torch.nn.Module,
        segment_tokens: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process a single segment with memory-bounded execution.
        
        Args:
            model: Language model
            segment_tokens: Token IDs for segment
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple of (output logits, extracted activations)
        """
        with self.memory_efficient_mode():
            # Check memory before processing
            if not self.check_memory():
                warnings.warn(f"Memory usage high: {self.get_memory_usage():.2f} GB")
                self.kv_cache.clear()
                gc.collect()
            
            # Convert to appropriate precision
            if self.config.use_fp16:
                model = model.half()
            
            # Extract activations at restriction sites
            activations = self.extract_activations(model, segment_tokens)
            
            # Get model output
            with torch.no_grad():
                if use_cache and hasattr(model, 'forward_with_cache'):
                    # Use cached forward if available
                    outputs = model.forward_with_cache(
                        segment_tokens,
                        past_key_values=self.kv_cache
                    )
                else:
                    outputs = model(segment_tokens)
            
            # Extract logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            return logits.detach().cpu(), activations
    
    def run_with_overlapping_windows(
        self,
        model: torch.nn.Module,
        tokens: List[int]
    ) -> Generator[Tuple[torch.Tensor, Dict[str, torch.Tensor]], None, None]:
        """
        Run model on tokens with overlapping windows.
        
        Args:
            model: Language model
            tokens: List of token IDs
            
        Yields:
            Tuples of (logits, activations) for each segment
        """
        total_tokens = len(tokens)
        stride = self.config.segment_size - self.config.overlap_size
        
        for start_idx in range(0, total_tokens, stride):
            end_idx = min(start_idx + self.config.segment_size, total_tokens)
            
            # Get segment with overlap from buffer
            if self.overlap_buffer:
                segment_tokens = list(self.overlap_buffer) + tokens[start_idx:end_idx]
            else:
                segment_tokens = tokens[start_idx:end_idx]
            
            # Convert to tensor
            segment_tensor = torch.tensor([segment_tokens], dtype=torch.long)
            if torch.cuda.is_available():
                segment_tensor = segment_tensor.cuda()
            
            # Process segment
            logits, activations = self.process_segment(model, segment_tensor)
            
            # Update overlap buffer
            if end_idx - start_idx >= self.config.overlap_size:
                self.overlap_buffer.extend(
                    tokens[end_idx - self.config.overlap_size:end_idx]
                )
            
            yield logits, activations
            
            # Check if we need to offload parameters
            if not self.check_memory():
                # Offload earlier layers
                layers_to_offload = list(range(0, 6))  # Offload first 6 layers
                self.offload_parameters(model, layers_to_offload)
    
    def execute_with_memory_limit(
        self,
        model: torch.nn.Module,
        tokens: List[int],
        memory_limit_gb: Optional[float] = None
    ) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Execute model on tokens with strict memory limit.
        
        Args:
            model: Language model
            tokens: List of token IDs
            memory_limit_gb: Memory limit in GB
            
        Returns:
            List of (logits, activations) for each segment
        """
        if memory_limit_gb:
            self.config.max_memory_gb = memory_limit_gb
        
        results = []
        
        # Process in segments
        for logits, activations in self.run_with_overlapping_windows(model, tokens):
            results.append((logits, activations))
            
            # Periodic memory cleanup
            if len(results) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        # Clear KV cache
        self.kv_cache.clear()
        
        # Remove offloaded parameters
        for param_path in self.offloaded_params.values():
            if os.path.exists(param_path):
                os.remove(param_path)
        self.offloaded_params.clear()
        
        # Clear activations
        self.activations.clear()
        
        # Clear overlap buffer
        self.overlap_buffer.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()