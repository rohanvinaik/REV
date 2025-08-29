"""Hyperdimensional Computing Encoder for REV verification."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union, List, Tuple, Literal
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Constants for REV and HBT HDC implementation
DEFAULT_DIMENSION = 10000  # 10K dimensional vectors as mentioned in REV paper
HBT_DIMENSION = 16384      # 16K dimensional vectors for HBT standard
MAX_DIMENSION = 100000     # Up to 100K as mentioned in REV paper

TensorLike = Union[np.ndarray, torch.Tensor]
EncodingMode = Literal["rev", "hbt", "hybrid"]
ZoomLevel = Literal["corpus", "prompt", "span", "token_window"]


class ProjectionType(Enum):
    """Projection types for hypervector encoding"""
    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"  
    ORTHOGONAL = "orthogonal"


@dataclass
class HypervectorConfig:
    """Configuration for REV/HBT hypervector encoding"""
    dimension: int = DEFAULT_DIMENSION
    encoding_mode: EncodingMode = "rev"
    projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM
    sparsity: float = 0.1
    seed: Optional[int] = None
    normalize: bool = True
    quantize: bool = False
    quantization_bits: int = 8
    similarity_threshold: float = 0.85
    variance_threshold: float = 0.01  # For variance-aware encoding
    multi_scale: bool = False  # Enable multi-scale zoom levels


class HypervectorEncoder:
    """
    Hyperdimensional Computing encoder for REV/HBT model verification.
    
    This implements the complete HDC encoding pipeline for both:
    - REV: 8K-100K dimensional vectors for model activation/logit sketches
    - HBT: 16K dimensional vectors with variance-aware encoding
    - Hybrid: Concatenation of both encoding modes
    """

    def __init__(self, config: Optional[HypervectorConfig] = None) -> None:
        """
        Initialize HDC encoder for REV/HBT.
        
        Args:
            config: Configuration for hypervector encoding
        """
        self.config = config or HypervectorConfig()
        
        # Adjust dimension based on encoding mode
        if self.config.encoding_mode == "hbt":
            self.config.dimension = HBT_DIMENSION
        elif self.config.encoding_mode == "hybrid":
            # Hybrid uses both REV and HBT dimensions
            self.rev_dimension = self.config.dimension
            self.hbt_dimension = HBT_DIMENSION
            self.config.dimension = self.rev_dimension + self.hbt_dimension
        
        # Ensure reproducibility for REV/HBT verification
        if self.config.seed is None:
            self.config.seed = 42
            logger.info("Using default seed=42 for REV/HBT reproducibility")
        
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        self._projection_cache: Dict[str, torch.Tensor] = {}
        self._variance_cache: Dict[str, torch.Tensor] = {}
        self._zoom_levels: Dict[ZoomLevel, Dict] = {}
        self._initialize_projections()

    def _initialize_projections(self) -> None:
        """Initialize projection matrices for encoding"""
        if self.config.projection_type == ProjectionType.SPARSE_RANDOM:
            self._create_sparse_projection()
        elif self.config.projection_type == ProjectionType.RANDOM_GAUSSIAN:
            self._create_gaussian_projection()
        elif self.config.projection_type == ProjectionType.ORTHOGONAL:
            self._create_orthogonal_projection()

    def _create_sparse_projection(self) -> None:
        """Create sparse random projection matrix"""
        # Create sparse projection for efficiency
        n_nonzero = int(self.config.dimension * self.config.sparsity)
        projection = torch.zeros(self.config.dimension)
        
        # Randomly select positions for non-zero elements
        indices = torch.randperm(self.config.dimension)[:n_nonzero]
        values = torch.randn(n_nonzero)
        projection[indices] = values
        
        if self.config.normalize:
            projection = projection / torch.norm(projection)
            
        self._projection_cache["base"] = projection

    def _create_gaussian_projection(self) -> None:
        """Create Gaussian random projection matrix"""
        projection = torch.randn(self.config.dimension)
        
        if self.config.normalize:
            projection = projection / torch.norm(projection)
            
        self._projection_cache["base"] = projection

    def _create_orthogonal_projection(self) -> None:
        """Create orthogonal projection matrix"""
        # Create orthogonal matrix using QR decomposition
        A = torch.randn(self.config.dimension, self.config.dimension)
        Q, _ = torch.qr(A)
        
        self._projection_cache["base"] = Q[0]  # Use first row as base projection

    def encode_sequence(self, sequence: List[float], sequence_id: str = "default") -> torch.Tensor:
        """
        Encode a sequence of values into a hypervector for REV.
        
        This is suitable for encoding model activations or logit sequences
        for comparison in the REV verification protocol.
        
        Args:
            sequence: List of numerical values to encode
            sequence_id: Identifier for the sequence (for caching)
            
        Returns:
            Hyperdimensional vector representation
        """
        if not sequence:
            return torch.zeros(self.config.dimension)
        
        # Get or create projection for this sequence type
        projection_key = f"{sequence_id}_projection"
        if projection_key not in self._projection_cache:
            self._projection_cache[projection_key] = self._projection_cache["base"].clone()
        
        projection = self._projection_cache[projection_key]
        
        # Encode sequence by binding position and value hypervectors
        result = torch.zeros(self.config.dimension)
        
        for i, value in enumerate(sequence):
            # Create position hypervector
            position_hv = self._create_position_hypervector(i)
            
            # Create value hypervector
            value_hv = self._create_value_hypervector(value)
            
            # Bind position and value (element-wise multiplication)
            bound_hv = position_hv * value_hv
            
            # Bundle into result (addition)
            result += bound_hv
        
        if self.config.normalize:
            result = result / torch.norm(result)
        
        if self.config.quantize:
            result = self._quantize_vector(result)
            
        return result

    def _create_position_hypervector(self, position: int) -> torch.Tensor:
        """Create hypervector for a position"""
        cache_key = f"pos_{position}"
        if cache_key not in self._projection_cache:
            # Use circular shift of base vector for position encoding
            base = self._projection_cache["base"]
            shifted = torch.roll(base, shifts=position % self.config.dimension)
            self._projection_cache[cache_key] = shifted
        
        return self._projection_cache[cache_key]

    def _create_value_hypervector(self, value: float) -> torch.Tensor:
        """Create hypervector for a value"""
        # Quantize value for discrete representation
        quantized_value = int(value * 1000) % 10000  # Simple quantization
        
        cache_key = f"val_{quantized_value}"
        if cache_key not in self._projection_cache:
            # Create deterministic hypervector for this value
            torch.manual_seed(self.config.seed + quantized_value)
            hv = torch.randn(self.config.dimension)
            if self.config.normalize:
                hv = hv / torch.norm(hv)
            self._projection_cache[cache_key] = hv
            
            # Restore original seed state
            torch.manual_seed(self.config.seed)
        
        return self._projection_cache[cache_key]

    def _quantize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Quantize vector to specified bit precision"""
        if self.config.quantization_bits >= 32:
            return vector
        
        # Quantize to specified bits
        max_val = 2 ** (self.config.quantization_bits - 1) - 1
        min_val = -max_val
        
        # Normalize to [-1, 1] range first
        normalized = vector / torch.max(torch.abs(vector))
        
        # Quantize
        quantized = torch.round(normalized * max_val)
        quantized = torch.clamp(quantized, min_val, max_val)
        
        # Convert back to float
        return quantized / max_val

    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """
        Compute similarity between two hypervectors.
        
        Uses cosine similarity for continuous vectors or Hamming distance
        for binary vectors, as mentioned in REV paper.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Similarity score in [0, 1]
        """
        # Cosine similarity
        cos_sim = torch.cosine_similarity(hv1.unsqueeze(0), hv2.unsqueeze(0))
        
        # Convert to [0, 1] range  
        similarity = (cos_sim + 1) / 2
        
        return float(similarity)

    def hamming_distance(self, hv1: torch.Tensor, hv2: torch.Tensor, use_lut: bool = True) -> int:
        """
        Compute Hamming distance between binary hypervectors.
        
        This is the core distance measure mentioned in REV for efficient
        comparison of model sketches. Compatible with src/hypervector/hamming.py.
        
        Args:
            hv1: First binary hypervector
            hv2: Second binary hypervector
            use_lut: Whether to use lookup table optimization
            
        Returns:
            Hamming distance (number of differing bits)
        """
        # Binarize vectors
        bin_hv1 = (hv1 > 0).long()
        bin_hv2 = (hv2 > 0).long()
        
        if use_lut:
            # Import Hamming LUT for optimized computation
            try:
                from ..hypervector.hamming import hamming_distance_cpu, pack_binary_vector
                
                # Pack binary vectors for efficient LUT-based computation
                packed_hv1 = pack_binary_vector(bin_hv1.numpy())
                packed_hv2 = pack_binary_vector(bin_hv2.numpy())
                
                return hamming_distance_cpu(packed_hv1, packed_hv2)
            except ImportError:
                # Fallback to standard computation
                pass
        
        # Standard Hamming distance computation
        return int(torch.sum(bin_hv1 != bin_hv2))

    def encode_model_signature(self, activations: List[List[float]], model_id: str) -> torch.Tensor:
        """
        Encode model activations/logits into a signature hypervector for REV.
        
        This creates the model "sketch" that can be efficiently compared
        using Hamming distance for REV verification.
        
        Args:
            activations: List of activation sequences from model layers
            model_id: Identifier for the model
            
        Returns:
            Signature hypervector for the model
        """
        if not activations:
            return torch.zeros(self.config.dimension)
        
        # Bundle all layer activations
        layer_hvs = []
        for i, layer_activations in enumerate(activations):
            layer_id = f"{model_id}_layer_{i}"
            layer_hv = self.encode_sequence(layer_activations, layer_id)
            layer_hvs.append(layer_hv)
        
        # Bundle all layers into final signature
        if not layer_hvs:
            return torch.zeros(self.config.dimension)
        
        signature = torch.stack(layer_hvs).sum(dim=0)
        
        if self.config.normalize:
            signature = signature / torch.norm(signature)
        
        if self.config.quantize:
            signature = self._quantize_vector(signature)
            
        return signature

    def clear_cache(self) -> None:
        """Clear the projection cache to free memory"""
        self._projection_cache.clear()
        self._initialize_projections()

    def get_cache_size(self) -> int:
        """Get the current size of the projection cache"""
        return len(self._projection_cache)
    
    def variance_aware_encoding(self, 
                              features: Union[torch.Tensor, np.ndarray],
                              extract_variance: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform variance-aware encoding for HBT mode.
        
        Args:
            features: Input features to encode
            extract_variance: Whether to extract variance tensors
            
        Returns:
            Tuple of (encoded hypervector, variance tensor if requested)
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Ensure features is 2D [batch_size, feature_dim]
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        batch_size, feature_dim = features.shape
        
        # Create or retrieve variance-aware projection
        if "variance_projection" not in self._projection_cache:
            # Initialize variance-aware projection for HBT
            proj_dim = HBT_DIMENSION if self.config.encoding_mode in ["hbt", "hybrid"] else self.config.dimension
            projection = torch.randn(feature_dim, proj_dim)
            
            # Apply variance scaling
            feature_variance = torch.var(features, dim=0, keepdim=True)
            feature_variance = torch.clamp(feature_variance, min=self.config.variance_threshold)
            projection = projection * torch.sqrt(feature_variance).T
            
            if self.config.normalize:
                projection = F.normalize(projection, dim=1)
            
            self._projection_cache["variance_projection"] = projection
        
        projection = self._projection_cache["variance_projection"]
        
        # Encode features
        encoded = torch.matmul(features, projection)
        
        # Extract variance tensor if requested
        variance_tensor = None
        if extract_variance:
            # Compute variance across batch dimension
            variance_tensor = torch.var(encoded, dim=0)
            self._variance_cache["last_variance"] = variance_tensor
        
        # Apply normalization
        if self.config.normalize:
            encoded = F.normalize(encoded, dim=-1)
        
        # Apply quantization if needed
        if self.config.quantize:
            encoded = self._quantize_vector(encoded)
        
        return encoded.squeeze(0) if batch_size == 1 else encoded, variance_tensor
    
    def encode(self, 
               features: Union[torch.Tensor, np.ndarray, List[float]],
               zoom_level: Optional[ZoomLevel] = None,
               behavioral_site: Optional[str] = None) -> torch.Tensor:
        """
        Enhanced encode method supporting multi-scale zoom levels and behavioral sites.
        
        Args:
            features: Input features to encode
            zoom_level: Optional zoom level for multi-scale encoding
            behavioral_site: Optional behavioral site identifier
            
        Returns:
            Encoded hypervector
        """
        # Convert to tensor if needed
        if isinstance(features, list):
            features = torch.tensor(features, dtype=torch.float32)
        elif isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Apply zoom level processing if specified
        if zoom_level and self.config.multi_scale:
            features = self._apply_zoom_level(features, zoom_level)
        
        # Route to appropriate encoding method based on mode
        if self.config.encoding_mode == "rev":
            # Original REV encoding
            if features.dim() == 1:
                return self.encode_sequence(features.tolist(), behavioral_site or "default")
            else:
                return self.encode_model_signature([f.tolist() for f in features], behavioral_site or "model")
        
        elif self.config.encoding_mode == "hbt":
            # HBT variance-aware encoding
            encoded, _ = self.variance_aware_encoding(features, extract_variance=True)
            return encoded
        
        elif self.config.encoding_mode == "hybrid":
            # Concatenate both encodings
            # REV encoding
            rev_config = HypervectorConfig(
                dimension=self.rev_dimension,
                encoding_mode="rev",
                projection_type=self.config.projection_type,
                seed=self.config.seed
            )
            rev_encoder = HypervectorEncoder(rev_config)
            rev_encoded = rev_encoder.encode(features, zoom_level, behavioral_site)
            
            # HBT encoding
            hbt_encoded, _ = self.variance_aware_encoding(features, extract_variance=True)
            
            # Concatenate
            return torch.cat([rev_encoded, hbt_encoded], dim=-1)
        
        else:
            raise ValueError(f"Unknown encoding mode: {self.config.encoding_mode}")
    
    def _apply_zoom_level(self, features: torch.Tensor, zoom_level: ZoomLevel) -> torch.Tensor:
        """
        Apply zoom level transformations to features.
        
        Args:
            features: Input features
            zoom_level: Zoom level to apply
            
        Returns:
            Transformed features
        """
        if zoom_level == "corpus":
            # Corpus-level: global statistics
            return torch.cat([
                features.mean(dim=0, keepdim=True),
                features.std(dim=0, keepdim=True),
                features.min(dim=0, keepdim=True)[0],
                features.max(dim=0, keepdim=True)[0]
            ], dim=0).flatten()
        
        elif zoom_level == "prompt":
            # Prompt-level: full features
            return features.flatten()
        
        elif zoom_level == "span":
            # Span-level: sliding window with stride
            window_size = min(32, features.shape[0] // 4)
            stride = window_size // 2
            spans = []
            for i in range(0, features.shape[0] - window_size + 1, stride):
                span = features[i:i+window_size]
                spans.append(span.mean(dim=0))
            return torch.stack(spans).flatten() if spans else features.flatten()
        
        elif zoom_level == "token_window":
            # Token window: local context
            window_size = min(8, features.shape[0])
            if features.shape[0] > window_size:
                # Take first, middle, and last windows
                indices = [0, features.shape[0] // 2, -window_size]
                windows = []
                for idx in indices:
                    if idx < 0:
                        windows.append(features[idx:])
                    else:
                        windows.append(features[idx:idx+window_size])
                return torch.cat([w.flatten() for w in windows])
            return features.flatten()
        
        else:
            return features.flatten()


class UnifiedHDCEncoder(HypervectorEncoder):
    """
    Unified HDC encoder with adaptive mode switching for REV/HBT.
    
    This encoder provides intelligent mode selection based on context:
    - Streaming mode: Memory-bounded REV encoding
    - Consensus mode: Variance-aware HBT 16K encoding
    - Hybrid mode: Concatenated encoding for maximum information
    """
    
    def __init__(self, 
                 config: Optional[HypervectorConfig] = None,
                 auto_mode: bool = True) -> None:
        """
        Initialize unified HDC encoder.
        
        Args:
            config: Base configuration for encoding
            auto_mode: Enable automatic mode selection based on context
        """
        super().__init__(config)
        self.auto_mode = auto_mode
        
        # Context tracking for adaptive mode selection
        self.context_history: List[str] = []
        self.mode_stats: Dict[str, int] = {
            "streaming": 0,
            "consensus": 0,
            "hybrid": 0
        }
        
        # Mode-specific encoders
        self._init_mode_encoders()
    
    def _init_mode_encoders(self) -> None:
        """Initialize separate encoders for each mode."""
        # REV encoder for streaming
        self.rev_encoder = HypervectorEncoder(
            HypervectorConfig(
                dimension=DEFAULT_DIMENSION,
                encoding_mode="rev",
                projection_type=self.config.projection_type,
                seed=self.config.seed,
                normalize=self.config.normalize,
                quantize=self.config.quantize
            )
        )
        
        # HBT encoder for consensus
        self.hbt_encoder = HypervectorEncoder(
            HypervectorConfig(
                dimension=HBT_DIMENSION,
                encoding_mode="hbt",
                projection_type=self.config.projection_type,
                seed=self.config.seed,
                normalize=self.config.normalize,
                quantize=self.config.quantize,
                variance_threshold=self.config.variance_threshold
            )
        )
        
        # Hybrid encoder
        self.hybrid_encoder = HypervectorEncoder(
            HypervectorConfig(
                dimension=DEFAULT_DIMENSION,
                encoding_mode="hybrid",
                projection_type=self.config.projection_type,
                seed=self.config.seed,
                normalize=self.config.normalize,
                quantize=self.config.quantize
            )
        )
    
    def encode_adaptive(self,
                       features: Union[torch.Tensor, np.ndarray, List[float]],
                       context: Optional[str] = None,
                       behavioral_site: Optional[str] = None,
                       zoom_level: Optional[ZoomLevel] = None) -> torch.Tensor:
        """
        Adaptively encode features based on context.
        
        Args:
            features: Input features to encode
            context: Context hint ("streaming", "consensus", "hybrid", or None for auto)
            behavioral_site: Optional behavioral site identifier
            zoom_level: Optional zoom level for multi-scale encoding
            
        Returns:
            Encoded hypervector using appropriate mode
        """
        # Determine encoding mode
        if context:
            mode = self._map_context_to_mode(context)
        elif self.auto_mode:
            mode = self._infer_mode_from_features(features)
        else:
            mode = self.config.encoding_mode
        
        # Track mode usage
        self.context_history.append(mode)
        if mode == "rev":
            self.mode_stats["streaming"] += 1
        elif mode == "hbt":
            self.mode_stats["consensus"] += 1
        else:
            self.mode_stats["hybrid"] += 1
        
        # Route to appropriate encoder
        if mode == "rev" or context == "streaming":
            # Use REV encoder for streaming/memory-bounded scenarios
            return self.rev_encoder.encode(features, zoom_level, behavioral_site)
        
        elif mode == "hbt" or context == "consensus":
            # Use HBT encoder for consensus/variance-aware scenarios
            if isinstance(features, list):
                features = torch.tensor(features, dtype=torch.float32)
            elif isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            
            # Apply zoom level if specified
            if zoom_level:
                features = self.hbt_encoder._apply_zoom_level(features, zoom_level)
            
            encoded, variance = self.hbt_encoder.variance_aware_encoding(
                features, 
                extract_variance=True
            )
            
            # Store variance for behavioral site analysis
            if behavioral_site and variance is not None:
                self._variance_cache[f"{behavioral_site}_variance"] = variance
            
            return encoded
        
        else:  # hybrid mode
            # Use hybrid encoder for maximum information
            return self.hybrid_encoder.encode(features, zoom_level, behavioral_site)
    
    def _map_context_to_mode(self, context: str) -> str:
        """Map context string to encoding mode."""
        context_map = {
            "streaming": "rev",
            "consensus": "hbt",
            "hybrid": "hybrid",
            "memory_bounded": "rev",
            "variance_aware": "hbt",
            "full": "hybrid"
        }
        return context_map.get(context.lower(), self.config.encoding_mode)
    
    def _infer_mode_from_features(self, 
                                 features: Union[torch.Tensor, np.ndarray, List[float]]) -> str:
        """
        Automatically infer best encoding mode from features.
        
        Args:
            features: Input features
            
        Returns:
            Inferred encoding mode
        """
        # Convert to tensor for analysis
        if isinstance(features, list):
            features = torch.tensor(features, dtype=torch.float32)
        elif isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Analyze feature characteristics
        feature_size = features.numel()
        feature_dim = features.shape[-1] if features.dim() > 1 else len(features)
        
        # Compute variance if 2D
        if features.dim() > 1:
            variance = torch.var(features).item()
        else:
            variance = torch.var(features).item() if len(features) > 1 else 0
        
        # Decision logic
        if feature_size < 1000 and variance < self.config.variance_threshold:
            # Small, low-variance features -> REV for efficiency
            return "rev"
        elif feature_dim == HBT_DIMENSION or variance > 0.1:
            # HBT-sized or high-variance features -> HBT for variance awareness
            return "hbt"
        elif feature_size > 10000:
            # Large features -> Hybrid for comprehensive encoding
            return "hybrid"
        else:
            # Default to configured mode
            return self.config.encoding_mode
    
    def get_variance_for_site(self, behavioral_site: str) -> Optional[torch.Tensor]:
        """
        Get stored variance tensor for a behavioral site.
        
        Args:
            behavioral_site: Site identifier
            
        Returns:
            Variance tensor if available
        """
        key = f"{behavioral_site}_variance"
        return self._variance_cache.get(key)
    
    def get_mode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage.
        
        Returns:
            Dictionary with mode usage statistics
        """
        total = sum(self.mode_stats.values())
        if total == 0:
            return {
                "total_encodings": 0,
                "mode_distribution": {},
                "last_mode": None
            }
        
        return {
            "total_encodings": total,
            "mode_distribution": {
                k: v / total for k, v in self.mode_stats.items()
            },
            "last_mode": self.context_history[-1] if self.context_history else None,
            "mode_counts": self.mode_stats.copy()
        }
    
    def reset_statistics(self) -> None:
        """Reset mode usage statistics."""
        self.context_history.clear()
        self.mode_stats = {
            "streaming": 0,
            "consensus": 0,
            "hybrid": 0
        }
        self._variance_cache.clear()
    
    def encode_behavioral_site(self,
                              site_features: Dict[str, Union[torch.Tensor, np.ndarray]],
                              site_name: str,
                              zoom_levels: Optional[List[ZoomLevel]] = None) -> Dict[str, torch.Tensor]:
        """
        Encode features from a behavioral site with multi-scale zoom.
        
        Integrates with src/hdc/behavioral_sites.py for comprehensive site analysis.
        
        Args:
            site_features: Dictionary of features from the behavioral site
            site_name: Name of the behavioral site
            zoom_levels: Optional list of zoom levels to apply
            
        Returns:
            Dictionary of encoded hypervectors at different scales
        """
        if zoom_levels is None:
            zoom_levels = ["corpus", "prompt", "span", "token_window"]
        
        encoded_features = {}
        
        for feature_name, features in site_features.items():
            # Encode at each zoom level
            for zoom in zoom_levels:
                key = f"{site_name}_{feature_name}_{zoom}"
                
                # Determine context based on zoom level
                if zoom in ["corpus", "prompt"]:
                    # Global views benefit from variance awareness
                    context = "consensus"
                elif zoom in ["span", "token_window"]:
                    # Local views benefit from streaming efficiency
                    context = "streaming"
                else:
                    context = None  # Let auto-mode decide
                
                # Encode with adaptive mode
                encoded = self.encode_adaptive(
                    features,
                    context=context,
                    behavioral_site=site_name,
                    zoom_level=zoom
                )
                
                encoded_features[key] = encoded
        
        return encoded_features
    
    def compute_hamming_similarity(self,
                                  hv1: torch.Tensor,
                                  hv2: torch.Tensor,
                                  normalize: bool = True) -> float:
        """
        Compute Hamming-based similarity between hypervectors.
        
        Compatible with src/hypervector/hamming.py operations.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            normalize: Whether to normalize to [0, 1] range
            
        Returns:
            Similarity score
        """
        # Ensure compatible dimensions
        if hv1.shape != hv2.shape:
            raise ValueError(f"Shape mismatch: {hv1.shape} vs {hv2.shape}")
        
        # Compute Hamming distance
        distance = self.hamming_distance(hv1, hv2, use_lut=True)
        
        if normalize:
            # Normalize to [0, 1] where 1 is identical
            max_distance = hv1.numel()
            similarity = 1.0 - (distance / max_distance)
            return similarity
        
        return float(distance)