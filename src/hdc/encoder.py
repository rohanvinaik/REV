"""Hyperdimensional Computing Encoder for REV verification."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union, List
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Constants for REV HDC implementation
DEFAULT_DIMENSION = 10000  # 10K dimensional vectors as mentioned in REV paper
MAX_DIMENSION = 100000     # Up to 100K as mentioned in REV paper

TensorLike = Union[np.ndarray, torch.Tensor]


class ProjectionType(Enum):
    """Projection types for hypervector encoding"""
    RANDOM_GAUSSIAN = "random_gaussian"
    SPARSE_RANDOM = "sparse_random"  
    ORTHOGONAL = "orthogonal"


@dataclass
class HypervectorConfig:
    """Configuration for REV hypervector encoding"""
    dimension: int = DEFAULT_DIMENSION
    projection_type: ProjectionType = ProjectionType.SPARSE_RANDOM
    sparsity: float = 0.1
    seed: Optional[int] = None
    normalize: bool = True
    quantize: bool = False
    quantization_bits: int = 8
    similarity_threshold: float = 0.85


class HypervectorEncoder:
    """
    Hyperdimensional Computing encoder for REV model verification.
    
    This implements the complete HDC encoding pipeline mentioned in the REV paper:
    - 8K-100K dimensional vectors for model activation/logit sketches
    - Binding operations (XOR, permutation, circular) for compositional encoding
    - Efficient similarity computation using Hamming distance
    """

    def __init__(self, config: Optional[HypervectorConfig] = None) -> None:
        """
        Initialize HDC encoder for REV.
        
        Args:
            config: Configuration for hypervector encoding
        """
        self.config = config or HypervectorConfig()
        
        # Ensure reproducibility for REV verification
        if self.config.seed is None:
            self.config.seed = 42
            logger.info("Using default seed=42 for REV reproducibility")
        
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        self._projection_cache: Dict[str, torch.Tensor] = {}
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

    def hamming_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> int:
        """
        Compute Hamming distance between binary hypervectors.
        
        This is the core distance measure mentioned in REV for efficient
        comparison of model sketches.
        
        Args:
            hv1: First binary hypervector
            hv2: Second binary hypervector
            
        Returns:
            Hamming distance (number of differing bits)
        """
        # Binarize vectors
        bin_hv1 = (hv1 > 0).long()
        bin_hv2 = (hv2 > 0).long()
        
        # Compute Hamming distance
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