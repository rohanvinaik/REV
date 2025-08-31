"""
Model architecture adapters for REV activation extraction.

This module provides adapters for different model architectures to extract
activations from the correct layers. Supports GPT-2, GPT-NeoX, LLaMA, Mistral,
BERT, and T5 families.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureHooks:
    """Hook configuration for a specific architecture."""
    
    attention_pattern: str  # Regex pattern for attention layers
    mlp_pattern: str  # Regex pattern for MLP/FFN layers
    layernorm_pattern: str  # Regex pattern for layer norm
    embedding_pattern: str  # Regex pattern for embeddings
    num_layers_fn: Callable[[nn.Module], int]  # Function to get number of layers
    

class ModelArchitectureAdapter:
    """
    Adapter for extracting activations from different model architectures.
    
    Automatically detects model architecture and registers appropriate hooks.
    """
    
    # Architecture patterns for different model families
    ARCHITECTURES = {
        'gpt2': ArchitectureHooks(
            attention_pattern=r'transformer\.h\.(\d+)\.attn$',
            mlp_pattern=r'transformer\.h\.(\d+)\.mlp$',
            layernorm_pattern=r'transformer\.h\.(\d+)\.ln_\d+$',
            embedding_pattern=r'transformer\.wte$',
            num_layers_fn=lambda m: len(m.transformer.h) if hasattr(m, 'transformer') else 0
        ),
        'gpt_neox': ArchitectureHooks(
            attention_pattern=r'gpt_neox\.layers\.(\d+)\.attention$',
            mlp_pattern=r'gpt_neox\.layers\.(\d+)\.mlp$',
            layernorm_pattern=r'gpt_neox\.layers\.(\d+)\.(?:input|post_attention)_layernorm$',
            embedding_pattern=r'gpt_neox\.embed_in$',
            num_layers_fn=lambda m: len(m.gpt_neox.layers) if hasattr(m, 'gpt_neox') else 0
        ),
        'llama': ArchitectureHooks(
            attention_pattern=r'model\.layers\.(\d+)\.self_attn$',
            mlp_pattern=r'model\.layers\.(\d+)\.mlp$',
            layernorm_pattern=r'model\.layers\.(\d+)\.(?:input|post_attention)_layernorm$',
            embedding_pattern=r'model\.embed_tokens$',
            num_layers_fn=lambda m: len(m.model.layers) if hasattr(m, 'model') else 0
        ),
        'mistral': ArchitectureHooks(
            attention_pattern=r'model\.layers\.(\d+)\.self_attn$',
            mlp_pattern=r'model\.layers\.(\d+)\.mlp$',
            layernorm_pattern=r'model\.layers\.(\d+)\.(?:input|post_attention)_layernorm$',
            embedding_pattern=r'model\.embed_tokens$',
            num_layers_fn=lambda m: len(m.model.layers) if hasattr(m, 'model') else 0
        ),
        'bert': ArchitectureHooks(
            attention_pattern=r'bert\.encoder\.layer\.(\d+)\.attention$',
            mlp_pattern=r'bert\.encoder\.layer\.(\d+)\.intermediate$',
            layernorm_pattern=r'bert\.encoder\.layer\.(\d+)\..*LayerNorm$',
            embedding_pattern=r'bert\.embeddings$',
            num_layers_fn=lambda m: len(m.bert.encoder.layer) if hasattr(m, 'bert') else 0
        ),
        't5': ArchitectureHooks(
            attention_pattern=r'(?:encoder|decoder)\.block\.(\d+)\.layer\.0\.SelfAttention$',
            mlp_pattern=r'(?:encoder|decoder)\.block\.(\d+)\.layer\.1\.DenseReluDense$',
            layernorm_pattern=r'(?:encoder|decoder)\.block\.(\d+)\.layer\.\d+\.layer_norm$',
            embedding_pattern=r'shared$',  # T5 uses shared embeddings
            num_layers_fn=lambda m: len(m.encoder.block) if hasattr(m, 'encoder') else 0
        ),
    }
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None):
        """
        Initialize adapter for a model.
        
        Args:
            model: PyTorch model
            model_name: Optional model name for architecture detection
        """
        self.model = model
        self.model_name = model_name
        self.architecture = self._detect_architecture()
        self.hooks = self.ARCHITECTURES.get(self.architecture)
        self.activations = {}
        self.handles = []
        
        if self.hooks is None:
            logger.warning(f"Unknown architecture: {self.architecture}, using generic hooks")
            self.hooks = self._create_generic_hooks()
    
    def _detect_architecture(self) -> str:
        """
        Detect model architecture from model structure.
        
        Returns:
            Architecture name (e.g., 'gpt2', 'llama', 'bert')
        """
        # Check model config if available
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Check model_type in config
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                
                # Map model types to our architecture keys
                type_mapping = {
                    'gpt2': 'gpt2',
                    'gpt_neo': 'gpt_neox',
                    'gpt_neox': 'gpt_neox',
                    'gptj': 'gpt_neox',  # GPT-J uses similar structure
                    'llama': 'llama',
                    'mistral': 'mistral',
                    'bert': 'bert',
                    'roberta': 'bert',  # RoBERTa uses BERT structure
                    't5': 't5',
                    'mt5': 't5',  # mT5 uses T5 structure
                }
                
                if model_type in type_mapping:
                    return type_mapping[model_type]
            
            # Check architectures list in config
            if hasattr(config, 'architectures'):
                arch_str = str(config.architectures[0]).lower() if config.architectures else ""
                
                for key in self.ARCHITECTURES.keys():
                    if key in arch_str:
                        return key
        
        # Fallback: Check module names
        module_names = [name for name, _ in self.model.named_modules()]
        module_str = ' '.join(module_names).lower()
        
        if 'transformer.h.0' in module_str:
            return 'gpt2'
        elif 'gpt_neox.layers' in module_str:
            return 'gpt_neox'
        elif 'model.layers.0.self_attn' in module_str:
            # Could be LLaMA or Mistral
            if 'mistral' in self.model_name.lower() if self.model_name else False:
                return 'mistral'
            return 'llama'
        elif 'bert.encoder.layer' in module_str:
            return 'bert'
        elif 'encoder.block' in module_str or 'decoder.block' in module_str:
            return 't5'
        
        logger.warning("Could not detect architecture, defaulting to GPT-2")
        return 'gpt2'
    
    def _create_generic_hooks(self) -> ArchitectureHooks:
        """Create generic hooks for unknown architectures."""
        return ArchitectureHooks(
            attention_pattern=r'.*attention.*',
            mlp_pattern=r'.*(?:mlp|ffn|feed_forward).*',
            layernorm_pattern=r'.*(?:layernorm|layer_norm).*',
            embedding_pattern=r'.*(?:embed|embedding).*',
            num_layers_fn=lambda m: 12  # Default assumption
        )
    
    def register_hooks(
        self,
        layers: Optional[List[int]] = None,
        sites: Optional[List[str]] = None
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register hooks to extract activations.
        
        Args:
            layers: List of layer indices to hook (None for all)
            sites: List of site types ('attention', 'mlp', 'layernorm', 'embedding')
        
        Returns:
            List of hook handles
        """
        if sites is None:
            sites = ['attention', 'mlp']  # Default sites
        
        # Clear previous hooks
        self.clear_hooks()
        
        # Get all named modules
        for name, module in self.model.named_modules():
            # Check each site type
            for site in sites:
                pattern = getattr(self.hooks, f'{site}_pattern', None)
                if pattern and re.match(pattern, name):
                    # Extract layer index if present
                    match = re.match(pattern, name)
                    if match and match.groups():
                        layer_idx = int(match.group(1))
                        
                        # Check if this layer should be hooked
                        if layers is not None and layer_idx not in layers:
                            continue
                    
                    # Register hook
                    handle = module.register_forward_hook(
                        self._create_hook_fn(name, site)
                    )
                    self.handles.append(handle)
                    logger.debug(f"Registered hook for {site} at {name}")
        
        if not self.handles:
            logger.warning(f"No hooks registered for architecture {self.architecture}")
        else:
            logger.info(f"Registered {len(self.handles)} hooks for {self.architecture}")
        
        return self.handles
    
    def _create_hook_fn(self, module_name: str, site_type: str):
        """Create a hook function for a specific module."""
        def hook_fn(module, input, output):
            # Store activation
            key = f"{site_type}_{module_name}"
            
            # Handle different output types
            if isinstance(output, torch.Tensor):
                self.activations[key] = output.detach()
            elif isinstance(output, tuple):
                # For attention, we typically want the first output
                self.activations[key] = output[0].detach()
            elif hasattr(output, 'last_hidden_state'):
                # For some transformers outputs
                self.activations[key] = output.last_hidden_state.detach()
            else:
                logger.warning(f"Unknown output type for {module_name}: {type(output)}")
        
        return hook_fn
    
    def extract_activations(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from model forward pass.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for model forward
        
        Returns:
            Dictionary of activations
        """
        # Clear previous activations
        self.activations.clear()
        
        # Forward pass (hooks will capture activations)
        with torch.no_grad():
            _ = self.model(input_ids, **kwargs)
        
        return self.activations.copy()
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
    
    def get_layer_names(self) -> List[str]:
        """Get list of available layer names for this architecture."""
        layer_names = []
        
        for name, _ in self.model.named_modules():
            for site in ['attention', 'mlp', 'layernorm', 'embedding']:
                pattern = getattr(self.hooks, f'{site}_pattern', None)
                if pattern and re.match(pattern, name):
                    layer_names.append(name)
        
        return layer_names
    
    def __del__(self):
        """Clean up hooks when adapter is deleted."""
        self.clear_hooks()


def create_adapter(model: nn.Module, model_name: Optional[str] = None) -> ModelArchitectureAdapter:
    """
    Create an architecture adapter for a model.
    
    Args:
        model: PyTorch model
        model_name: Optional model name for architecture detection
    
    Returns:
        ModelArchitectureAdapter instance
    """
    return ModelArchitectureAdapter(model, model_name)