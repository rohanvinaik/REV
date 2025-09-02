#!/usr/bin/env python3
"""
Unified Hypervector Fingerprint System for REV Pipeline

This module creates comprehensive hypervector representations that link:
- Input prompts (semantic encoding)
- Layer-wise activations (processing pathway)
- Final model responses (output semantics)

The resulting "semantic DNA" enables direct cross-model comparison and
scaling analysis (larger models as scaled versions of smaller ones).
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# REV components
from .encoder import HypervectorEncoder, HypervectorConfig, DEFAULT_DIMENSION
from .behavioral_sites import BehavioralSites
from .binding_operations import BindingOperations, BindingType
from ..crypto.merkle import build_merkle_tree, generate_merkle_proof
from ..hypervector.similarity import AdvancedSimilarity
from ..analysis.pattern_recognition import PatternRecognizer

logger = logging.getLogger(__name__)


@dataclass
class FingerprintConfig:
    """Configuration for unified fingerprint generation"""
    # Core hypervector settings
    dimension: int = DEFAULT_DIMENSION
    sparsity: float = 0.15
    
    # Fingerprint composition weights
    prompt_weight: float = 0.3      # Input prompt contribution
    pathway_weight: float = 0.5     # Layer activation pathway
    response_weight: float = 0.2    # Final response contribution
    
    # Layer sampling strategy
    layer_sampling: str = "adaptive"  # "all", "uniform", "adaptive", "boundary"
    max_layers_sampled: int = 20      # Maximum layers to include
    boundary_sensitivity: float = 0.1 # For boundary detection
    
    # Binding configuration
    binding_type: BindingType = BindingType.CIRCULAR_CONVOLUTION
    enable_temporal_binding: bool = True
    enable_hierarchical_binding: bool = True
    
    # Quality and validation
    min_divergence_threshold: float = 0.01
    enable_merkle_validation: bool = True
    enable_cross_scale_analysis: bool = True
    
    # Storage and caching
    cache_intermediate: bool = True
    compress_storage: bool = True
    enable_delta_encoding: bool = True


@dataclass
class LayerActivationSample:
    """Sample of layer activations for fingerprint generation"""
    layer_idx: int
    activation_vector: np.ndarray
    divergence_score: float
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UnifiedFingerprint:
    """Complete unified hypervector fingerprint"""
    # Core fingerprint
    unified_hypervector: np.ndarray
    
    # Component vectors
    prompt_hypervector: np.ndarray
    pathway_hypervector: np.ndarray  # Layer processing pathway
    response_hypervector: np.ndarray
    
    # Metadata
    model_id: str
    prompt_text: str
    response_text: str
    layer_count: int
    layers_sampled: List[int]
    
    # Metrics
    fingerprint_quality: float
    divergence_stats: Dict[str, float]
    binding_strength: float
    
    # Validation
    merkle_root: Optional[str] = None
    verification_hash: Optional[str] = None
    
    # Temporal
    generation_time: datetime = field(default_factory=datetime.now)
    processing_duration: float = 0.0
    
    # Additional analysis
    scaling_signature: Optional[np.ndarray] = None
    functional_embedding: Optional[np.ndarray] = None


class UnifiedFingerprintGenerator:
    """
    Generates unified hypervector fingerprints linking prompts, 
    layer responses, and final outputs for comprehensive model comparison.
    """
    
    def __init__(self, 
                 config: Optional[FingerprintConfig] = None,
                 hdc_encoder: Optional[HypervectorEncoder] = None):
        """
        Initialize unified fingerprint generator.
        
        Args:
            config: Fingerprint generation configuration
            hdc_encoder: Optional existing HDC encoder
        """
        self.config = config or FingerprintConfig()
        
        # Initialize HDC encoder
        if hdc_encoder:
            self.encoder = hdc_encoder
        else:
            hdc_config = HypervectorConfig(
                dimension=self.config.dimension,
                sparsity=self.config.sparsity,
                encoding_mode="rev",
                normalize=True
            )
            self.encoder = HypervectorEncoder(hdc_config)
        
        # Initialize binding operations
        self.binder = BindingOperations(
            dimension=self.config.dimension,
            binding_type=self.config.binding_type
        )
        
        # Initialize behavioral sites analyzer
        self.behavioral_sites = BehavioralSites()
        
        # Initialize pattern recognizer for scaling analysis
        self.pattern_recognizer = PatternRecognizer()
        
        # Storage for intermediate results
        self.intermediate_cache = {} if self.config.cache_intermediate else None
        
        # Similarity calculator
        self.similarity = AdvancedSimilarity()
        
        logger.info(f"Initialized UnifiedFingerprintGenerator with {self.config.dimension}D vectors")
    
    def generate_unified_fingerprint(self,
                                   model_interface,
                                   prompt: str,
                                   layer_activations: Dict[int, torch.Tensor],
                                   response: str,
                                   model_id: str) -> UnifiedFingerprint:
        """
        Generate complete unified fingerprint linking prompt, layers, and response.
        
        Args:
            model_interface: Model interface for additional queries if needed
            prompt: Input prompt text
            layer_activations: Dict mapping layer indices to activation tensors
            response: Final model response
            model_id: Unique model identifier
            
        Returns:
            UnifiedFingerprint containing all components and metadata
        """
        start_time = time.time()
        
        logger.info(f"Generating unified fingerprint for model {model_id}")
        logger.info(f"  Prompt: {prompt[:50]}...")
        logger.info(f"  Response: {response[:50]}...")
        logger.info(f"  Layer activations: {len(layer_activations)} layers")
        
        # 1. Generate component hypervectors
        prompt_hv = self._encode_prompt(prompt)
        pathway_hv = self._encode_layer_pathway(layer_activations, model_id)
        response_hv = self._encode_response(response)
        
        # 2. Create unified fingerprint through weighted binding
        unified_hv = self._bind_components(prompt_hv, pathway_hv, response_hv)
        
        # 3. Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            prompt_hv, pathway_hv, response_hv, unified_hv, layer_activations
        )
        
        # 4. Generate validation data
        validation_data = self._generate_validation_data(
            unified_hv, prompt, response, layer_activations
        ) if self.config.enable_merkle_validation else {}
        
        # 5. Scaling analysis
        scaling_signature = self._compute_scaling_signature(
            pathway_hv, layer_activations
        ) if self.config.enable_cross_scale_analysis else None
        
        # 6. Functional embedding for semantic analysis
        functional_embedding = self._compute_functional_embedding(
            prompt_hv, pathway_hv, response_hv
        )
        
        processing_time = time.time() - start_time
        
        # Create fingerprint object
        fingerprint = UnifiedFingerprint(
            unified_hypervector=unified_hv,
            prompt_hypervector=prompt_hv,
            pathway_hypervector=pathway_hv,
            response_hypervector=response_hv,
            model_id=model_id,
            prompt_text=prompt,
            response_text=response,
            layer_count=len(layer_activations),
            layers_sampled=list(layer_activations.keys()),
            fingerprint_quality=quality_metrics['overall_quality'],
            divergence_stats=quality_metrics['divergence_stats'],
            binding_strength=quality_metrics['binding_strength'],
            merkle_root=validation_data.get('merkle_root'),
            verification_hash=validation_data.get('verification_hash'),
            processing_duration=processing_time,
            scaling_signature=scaling_signature,
            functional_embedding=functional_embedding
        )
        
        logger.info(f"✅ Generated unified fingerprint in {processing_time:.3f}s")
        logger.info(f"   Quality: {fingerprint.fingerprint_quality:.3f}")
        logger.info(f"   Binding strength: {fingerprint.binding_strength:.3f}")
        
        return fingerprint
    
    def _encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode input prompt as hypervector."""
        # Tokenize and encode prompt
        tokens = prompt.split()[:100]  # Limit for efficiency
        
        # Create semantic hypervector from tokens
        token_hvs = []
        for token in tokens:
            token_hv = self.encoder.encode_text(token)
            token_hvs.append(token_hv)
        
        if not token_hvs:
            return np.zeros(self.config.dimension)
        
        # Bind tokens with positional information
        prompt_hv = np.zeros(self.config.dimension)
        for i, token_hv in enumerate(token_hvs):
            # Create position vector
            position_hv = self.encoder.encode_integer(i)
            
            # Bind token with position
            bound_hv = self.binder.bind(token_hv, position_hv)
            
            # Add to prompt hypervector
            prompt_hv += bound_hv
        
        return self.encoder._normalize(prompt_hv)
    
    def _encode_layer_pathway(self, 
                            layer_activations: Dict[int, torch.Tensor],
                            model_id: str) -> np.ndarray:
        """
        Encode the processing pathway through model layers.
        This captures the "DNA" of how the model processes information.
        """
        if not layer_activations:
            logger.warning("No layer activations provided")
            return np.zeros(self.config.dimension)
        
        # Sample layers according to strategy
        sampled_layers = self._sample_layers(layer_activations)
        
        logger.info(f"Encoding pathway through {len(sampled_layers)} layers")
        
        # Create pathway hypervector
        pathway_hv = np.zeros(self.config.dimension)
        
        for i, layer_idx in enumerate(sampled_layers):
            activation = layer_activations[layer_idx]
            
            # Encode layer activation as hypervector
            layer_hv = self._encode_activation_tensor(activation, layer_idx)
            
            # Create layer position vector
            layer_pos_hv = self.encoder.encode_integer(layer_idx)
            
            # Bind layer content with its position in the model
            positioned_layer_hv = self.binder.bind(layer_hv, layer_pos_hv)
            
            # Add temporal binding if enabled
            if self.config.enable_temporal_binding and i > 0:
                # Bind with previous layer for temporal coherence
                prev_layer_hv = self.encoder.encode_integer(sampled_layers[i-1])
                temporal_hv = self.binder.bind(positioned_layer_hv, prev_layer_hv)
                pathway_hv += temporal_hv
            else:
                pathway_hv += positioned_layer_hv
        
        # Hierarchical binding if enabled
        if self.config.enable_hierarchical_binding:
            # Group layers into functional segments and bind hierarchically
            pathway_hv = self._apply_hierarchical_binding(
                pathway_hv, sampled_layers, layer_activations
            )
        
        return self.encoder._normalize(pathway_hv)
    
    def _encode_activation_tensor(self, 
                                activation: torch.Tensor, 
                                layer_idx: int) -> np.ndarray:
        """Encode a layer activation tensor as a hypervector."""
        # Convert to numpy and flatten
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()
        
        activation_flat = activation.flatten()
        
        # Sample activation values if tensor is very large
        if len(activation_flat) > 10000:
            # Use systematic sampling to preserve patterns
            step = len(activation_flat) // 10000
            activation_flat = activation_flat[::step]
        
        # Statistical encoding of activation patterns
        stats = [
            float(np.mean(activation_flat)),
            float(np.std(activation_flat)),
            float(np.max(activation_flat)),
            float(np.min(activation_flat)),
            float(np.median(activation_flat)),
            float(np.sum(activation_flat > 0) / len(activation_flat)),  # Activation ratio
            float(np.sum(np.abs(activation_flat) < 0.01) / len(activation_flat)),  # Sparsity
        ]
        
        # Encode statistics as hypervector
        stats_hv = np.zeros(self.config.dimension)
        for i, stat in enumerate(stats):
            stat_encoded = self.encoder.encode_float(stat)
            stat_pos = self.encoder.encode_integer(i)
            stats_hv += self.binder.bind(stat_encoded, stat_pos)
        
        # Also encode a sample of actual activation values
        sample_size = min(1000, len(activation_flat))
        sampled_activations = np.random.choice(activation_flat, sample_size, replace=False)
        
        activation_hv = np.zeros(self.config.dimension)
        for i, val in enumerate(sampled_activations[:100]):  # Limit for efficiency
            val_encoded = self.encoder.encode_float(float(val))
            pos_encoded = self.encoder.encode_integer(i)
            activation_hv += self.binder.bind(val_encoded, pos_encoded)
        
        # Combine statistical and activation encodings
        combined_hv = stats_hv * 0.7 + activation_hv * 0.3
        
        return self.encoder._normalize(combined_hv)
    
    def _encode_response(self, response: str) -> np.ndarray:
        """Encode final model response as hypervector."""
        # Similar to prompt encoding but with semantic focus
        tokens = response.split()[:200]  # Allow longer responses
        
        if not tokens:
            return np.zeros(self.config.dimension)
        
        # Create semantic hypervector from response tokens
        token_hvs = []
        for token in tokens:
            token_hv = self.encoder.encode_text(token)
            token_hvs.append(token_hv)
        
        # Bind tokens with enhanced semantic weighting
        response_hv = np.zeros(self.config.dimension)
        for i, token_hv in enumerate(token_hvs):
            # Weight later tokens more heavily (they often contain key information)
            weight = 1.0 + 0.5 * (i / len(token_hvs))
            
            # Create position vector
            position_hv = self.encoder.encode_integer(i)
            
            # Bind token with position
            bound_hv = self.binder.bind(token_hv, position_hv)
            
            # Add with semantic weighting
            response_hv += bound_hv * weight
        
        return self.encoder._normalize(response_hv)
    
    def _bind_components(self, 
                        prompt_hv: np.ndarray,
                        pathway_hv: np.ndarray, 
                        response_hv: np.ndarray) -> np.ndarray:
        """Bind prompt, pathway, and response into unified fingerprint."""
        
        # Apply weights from configuration
        weighted_prompt = prompt_hv * self.config.prompt_weight
        weighted_pathway = pathway_hv * self.config.pathway_weight
        weighted_response = response_hv * self.config.response_weight
        
        # Create unified fingerprint through binding operations
        # First bind prompt with pathway (input-processing relationship)
        input_processing = self.binder.bind(weighted_prompt, weighted_pathway)
        
        # Then bind result with response (processing-output relationship) 
        unified_hv = self.binder.bind(input_processing, weighted_response)
        
        # Add direct connections for completeness
        unified_hv += weighted_prompt + weighted_pathway + weighted_response
        
        return self.encoder._normalize(unified_hv)
    
    def _sample_layers(self, layer_activations: Dict[int, torch.Tensor]) -> List[int]:
        """Sample layers according to configured strategy."""
        all_layers = sorted(layer_activations.keys())
        
        if self.config.layer_sampling == "all":
            return all_layers[:self.config.max_layers_sampled]
        
        elif self.config.layer_sampling == "uniform":
            if len(all_layers) <= self.config.max_layers_sampled:
                return all_layers
            step = len(all_layers) / self.config.max_layers_sampled
            return [all_layers[int(i * step)] for i in range(self.config.max_layers_sampled)]
        
        elif self.config.layer_sampling == "adaptive":
            # Sample more densely where activations change significantly
            return self._adaptive_layer_sampling(layer_activations, all_layers)
        
        elif self.config.layer_sampling == "boundary":
            # Focus on behavioral boundaries
            return self._boundary_layer_sampling(layer_activations, all_layers)
        
        else:
            logger.warning(f"Unknown sampling strategy: {self.config.layer_sampling}")
            return all_layers[:self.config.max_layers_sampled]
    
    def _adaptive_layer_sampling(self, 
                                layer_activations: Dict[int, torch.Tensor],
                                all_layers: List[int]) -> List[int]:
        """Adaptively sample layers based on activation changes."""
        if len(all_layers) <= self.config.max_layers_sampled:
            return all_layers
        
        # Compute activation differences between consecutive layers
        differences = []
        for i in range(len(all_layers) - 1):
            layer1_idx = all_layers[i]
            layer2_idx = all_layers[i + 1]
            
            act1 = layer_activations[layer1_idx].detach().cpu().numpy().flatten()
            act2 = layer_activations[layer2_idx].detach().cpu().numpy().flatten()
            
            # Compute normalized difference
            diff = np.mean(np.abs(act1 - act2)) / (np.std(act1) + np.std(act2) + 1e-8)
            differences.append((layer2_idx, diff))
        
        # Sort by difference magnitude and select top changing layers
        differences.sort(key=lambda x: x[1], reverse=True)
        
        # Always include first and last layers
        sampled = [all_layers[0], all_layers[-1]]
        
        # Add high-difference layers
        remaining_slots = self.config.max_layers_sampled - 2
        for layer_idx, _ in differences[:remaining_slots]:
            if layer_idx not in sampled:
                sampled.append(layer_idx)
        
        return sorted(sampled)
    
    def _boundary_layer_sampling(self,
                                layer_activations: Dict[int, torch.Tensor],
                                all_layers: List[int]) -> List[int]:
        """Sample layers at behavioral boundaries."""
        # Use behavioral sites analyzer to find boundaries
        try:
            # This would integrate with the behavioral analysis system
            # For now, implement a simple statistical boundary detection
            return self._statistical_boundary_sampling(layer_activations, all_layers)
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}, falling back to uniform")
            return self._sample_layers_uniform(all_layers)
    
    def _statistical_boundary_sampling(self,
                                     layer_activations: Dict[int, torch.Tensor],
                                     all_layers: List[int]) -> List[int]:
        """Detect boundaries using statistical methods."""
        boundaries = [all_layers[0]]  # Always include first layer
        
        # Sliding window analysis for boundary detection
        window_size = 3
        for i in range(window_size, len(all_layers) - window_size):
            layer_idx = all_layers[i]
            
            # Get activations for window
            window_activations = []
            for j in range(-window_size, window_size + 1):
                act = layer_activations[all_layers[i + j]]
                window_activations.append(act.detach().cpu().numpy().flatten())
            
            # Compute variance in window
            variances = [np.var(act) for act in window_activations]
            
            # Detect if current layer is a local maximum in variance
            if (variances[window_size] > np.mean(variances) + 
                self.config.boundary_sensitivity * np.std(variances)):
                boundaries.append(layer_idx)
        
        boundaries.append(all_layers[-1])  # Always include last layer
        
        # Ensure we don't exceed max layers
        if len(boundaries) > self.config.max_layers_sampled:
            # Keep most important boundaries
            boundaries = boundaries[::len(boundaries)//self.config.max_layers_sampled]
        
        return sorted(boundaries)
    
    def _apply_hierarchical_binding(self,
                                  pathway_hv: np.ndarray,
                                  sampled_layers: List[int],
                                  layer_activations: Dict[int, torch.Tensor]) -> np.ndarray:
        """Apply hierarchical binding to capture functional segments."""
        
        # Group layers into functional segments (early, middle, late)
        num_layers = len(sampled_layers)
        
        # Create three functional groups
        early_layers = sampled_layers[:num_layers//3]
        middle_layers = sampled_layers[num_layers//3:2*num_layers//3]
        late_layers = sampled_layers[2*num_layers//3:]
        
        # Create hypervectors for each functional segment
        segment_hvs = []
        for segment_layers, segment_name in [(early_layers, "early"), 
                                           (middle_layers, "middle"), 
                                           (late_layers, "late")]:
            if segment_layers:
                segment_hv = np.zeros(self.config.dimension)
                for layer_idx in segment_layers:
                    layer_hv = self._encode_activation_tensor(
                        layer_activations[layer_idx], layer_idx
                    )
                    segment_hv += layer_hv
                
                # Bind with segment identifier
                segment_id_hv = self.encoder.encode_text(segment_name)
                segment_hvs.append(self.binder.bind(segment_hv, segment_id_hv))
        
        # Hierarchically bind segments
        if len(segment_hvs) >= 2:
            hierarchical_hv = segment_hvs[0]
            for segment_hv in segment_hvs[1:]:
                hierarchical_hv = self.binder.bind(hierarchical_hv, segment_hv)
            
            # Combine with original pathway
            return pathway_hv * 0.7 + hierarchical_hv * 0.3
        
        return pathway_hv
    
    def _compute_quality_metrics(self,
                               prompt_hv: np.ndarray,
                               pathway_hv: np.ndarray,
                               response_hv: np.ndarray,
                               unified_hv: np.ndarray,
                               layer_activations: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Compute quality metrics for the fingerprint."""
        
        # Component similarity analysis
        prompt_pathway_sim = 1 - cosine(prompt_hv, pathway_hv)
        pathway_response_sim = 1 - cosine(pathway_hv, response_hv)
        prompt_response_sim = 1 - cosine(prompt_hv, response_hv)
        
        # Binding strength (how well unified vector represents components)
        unified_prompt_sim = 1 - cosine(unified_hv, prompt_hv)
        unified_pathway_sim = 1 - cosine(unified_hv, pathway_hv)
        unified_response_sim = 1 - cosine(unified_hv, response_hv)
        
        binding_strength = np.mean([
            unified_prompt_sim, unified_pathway_sim, unified_response_sim
        ])
        
        # Divergence statistics from layer activations
        divergences = []
        for layer_idx, activation in layer_activations.items():
            act_flat = activation.detach().cpu().numpy().flatten()
            # Simple divergence measure: coefficient of variation
            if np.std(act_flat) > 0:
                divergence = np.std(act_flat) / (np.abs(np.mean(act_flat)) + 1e-8)
                divergences.append(divergence)
        
        divergence_stats = {
            "mean_divergence": np.mean(divergences) if divergences else 0.0,
            "max_divergence": np.max(divergences) if divergences else 0.0,
            "min_divergence": np.min(divergences) if divergences else 0.0,
            "divergence_variance": np.var(divergences) if divergences else 0.0
        }
        
        # Overall quality score
        overall_quality = np.mean([
            binding_strength,
            min(1.0, divergence_stats["mean_divergence"]),  # Cap at 1.0
            0.5 + 0.5 * prompt_pathway_sim,  # Bonus for coherent prompt-pathway
            0.5 + 0.5 * pathway_response_sim  # Bonus for coherent pathway-response
        ])
        
        return {
            "overall_quality": overall_quality,
            "binding_strength": binding_strength,
            "divergence_stats": divergence_stats,
            "component_similarities": {
                "prompt_pathway": prompt_pathway_sim,
                "pathway_response": pathway_response_sim,
                "prompt_response": prompt_response_sim
            },
            "unified_similarities": {
                "unified_prompt": unified_prompt_sim,
                "unified_pathway": unified_pathway_sim,
                "unified_response": unified_response_sim
            }
        }
    
    def _generate_validation_data(self,
                                unified_hv: np.ndarray,
                                prompt: str,
                                response: str,
                                layer_activations: Dict[int, torch.Tensor]) -> Dict[str, str]:
        """Generate Merkle validation data for fingerprint integrity."""
        
        # Create data for Merkle tree
        data_items = [
            f"prompt:{prompt}".encode(),
            f"response:{response}".encode(),
            unified_hv.tobytes()
        ]
        
        # Add layer activation hashes
        for layer_idx in sorted(layer_activations.keys()):
            act_hash = hashlib.sha256(layer_activations[layer_idx].detach().cpu().numpy().tobytes()).hexdigest()
            data_items.append(f"layer_{layer_idx}:{act_hash}".encode())
        
        # Build Merkle tree
        try:
            merkle_tree = build_merkle_tree(data_items)
            merkle_root = merkle_tree.hex() if merkle_tree else None
        except Exception as e:
            logger.warning(f"Failed to build Merkle tree: {e}")
            merkle_root = None
        
        # Create verification hash
        verification_data = f"{prompt}|{response}|{len(layer_activations)}"
        verification_hash = hashlib.sha256(verification_data.encode()).hexdigest()
        
        return {
            "merkle_root": merkle_root,
            "verification_hash": verification_hash
        }
    
    def _compute_scaling_signature(self,
                                 pathway_hv: np.ndarray,
                                 layer_activations: Dict[int, torch.Tensor]) -> np.ndarray:
        """Compute scaling signature for cross-model size analysis."""
        
        # Create a signature that captures model scale characteristics
        scale_features = []
        
        # Layer count signature
        num_layers = len(layer_activations)
        scale_features.extend([
            float(num_layers),
            float(np.log(num_layers + 1)),
            float(num_layers / 100.0)  # Normalized layer count
        ])
        
        # Activation magnitude signatures
        all_activations = []
        for activation in layer_activations.values():
            act_flat = activation.detach().cpu().numpy().flatten()
            all_activations.extend(act_flat[:1000])  # Sample for efficiency
        
        if all_activations:
            all_activations = np.array(all_activations)
            scale_features.extend([
                float(np.mean(np.abs(all_activations))),
                float(np.std(all_activations)),
                float(np.max(np.abs(all_activations))),
                float(np.percentile(np.abs(all_activations), 95))
            ])
        
        # Pathway complexity signature
        pathway_complexity = np.sum(np.abs(pathway_hv))
        pathway_sparsity = np.count_nonzero(pathway_hv) / len(pathway_hv)
        
        scale_features.extend([
            float(pathway_complexity),
            float(pathway_sparsity),
            float(np.entropy(np.abs(pathway_hv) + 1e-10))  # Information content
        ])
        
        # Encode as hypervector
        scaling_hv = np.zeros(self.config.dimension // 4)  # Smaller dimension
        for i, feature in enumerate(scale_features):
            if i < len(scaling_hv):
                feature_encoded = self.encoder.encode_float(feature)[:len(scaling_hv)]
                scaling_hv += feature_encoded * (i + 1) / len(scale_features)
        
        return self.encoder._normalize(scaling_hv)
    
    def _compute_functional_embedding(self,
                                    prompt_hv: np.ndarray,
                                    pathway_hv: np.ndarray,
                                    response_hv: np.ndarray) -> np.ndarray:
        """Compute functional embedding for semantic analysis."""
        
        # Create a reduced-dimension embedding capturing functional relationships
        components = np.stack([prompt_hv, pathway_hv, response_hv])
        
        # Apply PCA for dimensionality reduction
        try:
            pca = PCA(n_components=min(64, components.shape[1]))
            functional_embedding = pca.fit_transform(components.T)[:, 0]  # First component
            
            # Normalize
            if np.linalg.norm(functional_embedding) > 0:
                functional_embedding = functional_embedding / np.linalg.norm(functional_embedding)
            
            return functional_embedding
            
        except Exception as e:
            logger.warning(f"PCA failed for functional embedding: {e}")
            # Fallback: simple average
            return np.mean(components, axis=0)[:64]
    
    def compare_fingerprints(self,
                           fingerprint1: UnifiedFingerprint,
                           fingerprint2: UnifiedFingerprint) -> Dict[str, Any]:
        """
        Compare two unified fingerprints for model similarity analysis.
        
        This is where the key insight comes in: larger models should show
        as scaled versions of smaller models with similar architectures.
        """
        
        logger.info(f"Comparing fingerprints: {fingerprint1.model_id} vs {fingerprint2.model_id}")
        
        # Overall unified similarity
        unified_similarity = 1 - cosine(
            fingerprint1.unified_hypervector,
            fingerprint2.unified_hypervector
        )
        
        # Component-wise similarities
        prompt_similarity = 1 - cosine(
            fingerprint1.prompt_hypervector,
            fingerprint2.prompt_hypervector
        )
        
        pathway_similarity = 1 - cosine(
            fingerprint1.pathway_hypervector,
            fingerprint2.pathway_hypervector
        )
        
        response_similarity = 1 - cosine(
            fingerprint1.response_hypervector,
            fingerprint2.response_hypervector
        )
        
        # Scaling analysis
        scaling_analysis = {}
        if (fingerprint1.scaling_signature is not None and 
            fingerprint2.scaling_signature is not None):
            
            scaling_similarity = 1 - cosine(
                fingerprint1.scaling_signature,
                fingerprint2.scaling_signature
            )
            
            scaling_analysis = {
                "scaling_similarity": scaling_similarity,
                "layer_ratio": fingerprint2.layer_count / fingerprint1.layer_count,
                "quality_ratio": fingerprint2.fingerprint_quality / fingerprint1.fingerprint_quality,
                "is_likely_scaled_version": (
                    scaling_similarity > 0.7 and 
                    pathway_similarity > 0.6 and
                    abs(fingerprint2.layer_count / fingerprint1.layer_count - 
                        round(fingerprint2.layer_count / fingerprint1.layer_count)) < 0.1
                )
            }
        
        # Functional similarity
        functional_similarity = 0.0
        if (fingerprint1.functional_embedding is not None and 
            fingerprint2.functional_embedding is not None):
            
            # Ensure same length for comparison
            min_len = min(len(fingerprint1.functional_embedding), 
                         len(fingerprint2.functional_embedding))
            
            functional_similarity = 1 - cosine(
                fingerprint1.functional_embedding[:min_len],
                fingerprint2.functional_embedding[:min_len]
            )
        
        # Overall assessment
        similarity_scores = [unified_similarity, pathway_similarity, functional_similarity]
        overall_similarity = np.mean([s for s in similarity_scores if not np.isnan(s)])
        
        # Decision logic
        if overall_similarity > 0.8:
            decision = "SAME/VERY_SIMILAR"
            confidence = (overall_similarity - 0.8) / 0.2
        elif overall_similarity > 0.6:
            decision = "SIMILAR"
            confidence = (overall_similarity - 0.6) / 0.2
        elif overall_similarity > 0.4:
            decision = "SOMEWHAT_SIMILAR"
            confidence = (overall_similarity - 0.4) / 0.2
        else:
            decision = "DIFFERENT"
            confidence = (0.4 - overall_similarity) / 0.4
        
        return {
            "overall_similarity": overall_similarity,
            "decision": decision,
            "confidence": min(1.0, confidence),
            "component_similarities": {
                "unified": unified_similarity,
                "prompt": prompt_similarity,
                "pathway": pathway_similarity,
                "response": response_similarity,
                "functional": functional_similarity
            },
            "scaling_analysis": scaling_analysis,
            "quality_comparison": {
                "fingerprint1_quality": fingerprint1.fingerprint_quality,
                "fingerprint2_quality": fingerprint2.fingerprint_quality,
                "quality_difference": abs(fingerprint1.fingerprint_quality - 
                                        fingerprint2.fingerprint_quality)
            }
        }
    
    def save_fingerprint(self, 
                        fingerprint: UnifiedFingerprint, 
                        filepath: str) -> str:
        """Save fingerprint to file."""
        
        # Convert numpy arrays to lists for JSON serialization
        fingerprint_dict = {
            "unified_hypervector": fingerprint.unified_hypervector.tolist(),
            "prompt_hypervector": fingerprint.prompt_hypervector.tolist(),
            "pathway_hypervector": fingerprint.pathway_hypervector.tolist(),
            "response_hypervector": fingerprint.response_hypervector.tolist(),
            "model_id": fingerprint.model_id,
            "prompt_text": fingerprint.prompt_text,
            "response_text": fingerprint.response_text,
            "layer_count": fingerprint.layer_count,
            "layers_sampled": fingerprint.layers_sampled,
            "fingerprint_quality": fingerprint.fingerprint_quality,
            "divergence_stats": fingerprint.divergence_stats,
            "binding_strength": fingerprint.binding_strength,
            "merkle_root": fingerprint.merkle_root,
            "verification_hash": fingerprint.verification_hash,
            "generation_time": fingerprint.generation_time.isoformat(),
            "processing_duration": fingerprint.processing_duration,
            "scaling_signature": fingerprint.scaling_signature.tolist() if fingerprint.scaling_signature is not None else None,
            "functional_embedding": fingerprint.functional_embedding.tolist() if fingerprint.functional_embedding is not None else None,
            "config": {
                "dimension": self.config.dimension,
                "sparsity": self.config.sparsity,
                "weights": {
                    "prompt": self.config.prompt_weight,
                    "pathway": self.config.pathway_weight,
                    "response": self.config.response_weight
                }
            }
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with compression if enabled
        if self.config.compress_storage:
            import gzip
            with gzip.open(f"{filepath}.gz", 'wt') as f:
                json.dump(fingerprint_dict, f, indent=2)
            saved_path = f"{filepath}.gz"
        else:
            with open(filepath, 'w') as f:
                json.dump(fingerprint_dict, f, indent=2)
            saved_path = filepath
        
        logger.info(f"✅ Saved unified fingerprint to {saved_path}")
        return saved_path
    
    def load_fingerprint(self, filepath: str) -> UnifiedFingerprint:
        """Load fingerprint from file."""
        
        # Handle compressed files
        if filepath.endswith('.gz'):
            import gzip
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        # Convert lists back to numpy arrays
        fingerprint = UnifiedFingerprint(
            unified_hypervector=np.array(data["unified_hypervector"]),
            prompt_hypervector=np.array(data["prompt_hypervector"]),
            pathway_hypervector=np.array(data["pathway_hypervector"]),
            response_hypervector=np.array(data["response_hypervector"]),
            model_id=data["model_id"],
            prompt_text=data["prompt_text"],
            response_text=data["response_text"],
            layer_count=data["layer_count"],
            layers_sampled=data["layers_sampled"],
            fingerprint_quality=data["fingerprint_quality"],
            divergence_stats=data["divergence_stats"],
            binding_strength=data["binding_strength"],
            merkle_root=data.get("merkle_root"),
            verification_hash=data.get("verification_hash"),
            generation_time=datetime.fromisoformat(data["generation_time"]),
            processing_duration=data["processing_duration"],
            scaling_signature=np.array(data["scaling_signature"]) if data.get("scaling_signature") else None,
            functional_embedding=np.array(data["functional_embedding"]) if data.get("functional_embedding") else None
        )
        
        logger.info(f"✅ Loaded unified fingerprint from {filepath}")
        return fingerprint


# Integration helper functions
def create_unified_fingerprint_generator(config: Optional[Dict[str, Any]] = None) -> UnifiedFingerprintGenerator:
    """Create fingerprint generator with optional configuration."""
    if config:
        fp_config = FingerprintConfig(**config)
    else:
        fp_config = FingerprintConfig()
    
    return UnifiedFingerprintGenerator(fp_config)


def generate_model_fingerprint(model_interface,
                             prompt: str,
                             model_id: str,
                             config: Optional[Dict[str, Any]] = None) -> UnifiedFingerprint:
    """
    High-level function to generate unified fingerprint for a model.
    
    This function integrates with the REV pipeline to:
    1. Execute the model with layer activation capture
    2. Generate unified hypervector fingerprint
    3. Return complete fingerprint object
    """
    
    generator = create_unified_fingerprint_generator(config)
    
    # Process model with activation capture
    # This would integrate with the existing REV pipeline execution
    try:
        # Get model response and layer activations
        if hasattr(model_interface, 'process_for_rev'):
            # Use REV-specific processing
            rev_result = model_interface.process_for_rev(
                prompt=prompt,
                extract_activations=True,
                capture_all_layers=True
            )
            
            response = rev_result.get("response", "")
            layer_activations = rev_result.get("layer_activations", {})
            
        else:
            # Fallback to basic inference
            response = model_interface.generate(prompt)
            # Would need to implement activation capture for this case
            layer_activations = {}
        
        # Generate fingerprint
        fingerprint = generator.generate_unified_fingerprint(
            model_interface=model_interface,
            prompt=prompt,
            layer_activations=layer_activations,
            response=response,
            model_id=model_id
        )
        
        return fingerprint
        
    except Exception as e:
        logger.error(f"Failed to generate fingerprint for {model_id}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    config = FingerprintConfig(
        dimension=10000,
        layer_sampling="adaptive",
        enable_cross_scale_analysis=True
    )
    
    generator = UnifiedFingerprintGenerator(config)
    print("Unified Fingerprint Generator initialized")
    print(f"Configuration: {config}")