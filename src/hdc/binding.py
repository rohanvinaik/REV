"""Binding operations for Hyperdimensional Computing in REV."""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
import torch


class BindingOperation(Enum):
    """Types of binding operations in HDC"""
    BIND = "bind"        # Bind multiple concepts together
    SUPERPOSE = "superpose"  # Superpose (bundle) multiple vectors


class BindingType(Enum):
    """Specific binding implementation types"""
    MULTIPLY = "multiply"    # Element-wise multiplication (XOR for binary)
    CIRCULAR = "circular"    # Circular convolution
    FOURIER = "fourier"     # Fourier-based binding
    PERMUTATION = "permutation"  # Permutation-based binding


class HypervectorBinder:
    """
    Hypervector binding operations for REV verification.
    
    Implements the binding operations mentioned in the REV paper:
    - XOR binding for binary vectors
    - Circular shift and permutation operations
    - Efficient bundling for vector superposition
    """

    def __init__(self, dimension: int = 10000, seed: Optional[int] = None) -> None:
        """
        Initialize hypervector binder.
        
        Args:
            dimension: Dimensionality of hypervectors
            seed: Random seed for deterministic operations
        """
        self.dimension = dimension
        if seed is not None:
            torch.manual_seed(seed)
        
        # Cache for permutation matrices
        self._permutation_cache: dict[int, torch.Tensor] = {}

    def bind(
        self, 
        vectors: List[torch.Tensor], 
        binding_type: BindingType = BindingType.MULTIPLY
    ) -> torch.Tensor:
        """
        Bind multiple hypervectors together.
        
        This creates a compositional representation that combines
        multiple concepts while preserving their relationships.
        
        Args:
            vectors: List of hypervectors to bind
            binding_type: Type of binding operation to use
            
        Returns:
            Bound hypervector
            
        Raises:
            ValueError: If no vectors provided
        """
        if not vectors:
            raise ValueError("No vectors provided for binding")
        
        if len(vectors) == 1:
            return vectors[0].clone()
        
        result = vectors[0].clone()
        
        for vector in vectors[1:]:
            if binding_type == BindingType.MULTIPLY:
                result = self._multiply_bind(result, vector)
            elif binding_type == BindingType.CIRCULAR:
                result = self._circular_bind(result, vector)
            elif binding_type == BindingType.PERMUTATION:
                result = self._permutation_bind(result, vector)
            elif binding_type == BindingType.FOURIER:
                result = self._fourier_bind(result, vector)
            else:
                raise ValueError(f"Unknown binding type: {binding_type}")
        
        return result

    def _multiply_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication binding (XOR for binary)"""
        return a * b

    def _circular_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circular convolution binding"""
        # Use FFT for efficient circular convolution
        a_fft = torch.fft.fft(a)
        b_fft = torch.fft.fft(b)
        result_fft = a_fft * b_fft
        return torch.fft.ifft(result_fft).real

    def _permutation_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Permutation-based binding"""
        # Create deterministic permutation based on vector b
        perm_idx = self._get_permutation_from_vector(b)
        return a[perm_idx]

    def _fourier_bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fourier-domain binding"""
        # Transform to frequency domain
        a_fft = torch.fft.fft(a)
        b_fft = torch.fft.fft(b)
        
        # Bind in frequency domain
        result_fft = a_fft * torch.conj(b_fft)
        
        # Transform back
        return torch.fft.ifft(result_fft).real

    def _get_permutation_from_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Generate deterministic permutation from a vector"""
        # Use vector values to create deterministic permutation
        _, perm_indices = torch.sort(vector)
        return perm_indices

    def unbind(
        self,
        bound_vector: torch.Tensor,
        known_vectors: List[torch.Tensor],
        binding_type: BindingType = BindingType.MULTIPLY,
    ) -> torch.Tensor:
        """
        Unbind a vector from a bound representation.
        
        This is the inverse operation that extracts one vector
        when the others are known.
        
        Args:
            bound_vector: The bound hypervector
            known_vectors: List of known vectors that were bound
            binding_type: Type of binding that was used
            
        Returns:
            Unbound hypervector
        """
        result = bound_vector.clone()
        
        for vector in known_vectors:
            if binding_type == BindingType.MULTIPLY:
                # For multiplication binding, unbind by dividing
                result = result / (vector + 1e-8)  # Avoid division by zero
            elif binding_type == BindingType.CIRCULAR:
                # For circular binding, unbind using circular correlation
                result = self._circular_unbind(result, vector)
            elif binding_type == BindingType.PERMUTATION:
                # For permutation binding, apply inverse permutation
                result = self._permutation_unbind(result, vector)
            else:
                raise ValueError(f"Unbinding not implemented for {binding_type}")
        
        return result

    def _circular_unbind(self, bound: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Circular correlation for unbinding"""
        # Correlation is convolution with time-reversed signal
        known_reversed = torch.flip(known, [0])
        known_reversed = torch.roll(known_reversed, 1)  # Circular shift
        
        # Use FFT for efficient correlation
        bound_fft = torch.fft.fft(bound)
        known_fft = torch.fft.fft(known_reversed)
        result_fft = bound_fft * known_fft
        
        return torch.fft.ifft(result_fft).real

    def _permutation_unbind(self, bound: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Inverse permutation for unbinding"""
        # Get the permutation that was applied
        perm_idx = self._get_permutation_from_vector(known)
        
        # Create inverse permutation
        inv_perm = torch.argsort(perm_idx)
        
        # Apply inverse permutation
        return bound[inv_perm]

    def bundle(
        self, 
        vectors: List[torch.Tensor], 
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Bundle (superpose) multiple hypervectors.
        
        This creates a distributed representation that contains
        all input vectors. Used for creating sets and collections.
        
        Args:
            vectors: List of hypervectors to bundle
            normalize: Whether to normalize the result
            
        Returns:
            Bundled hypervector
            
        Raises:
            ValueError: If no vectors provided
        """
        if not vectors:
            raise ValueError("No vectors provided for bundling")
        
        # Simple addition bundling
        result = torch.stack(vectors).sum(dim=0)
        
        if normalize:
            norm = torch.norm(result)
            if norm > 0:
                result = result / norm
        
        return result

    def majority(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Create majority vector from binary hypervectors.
        
        Each dimension takes the majority value across all input vectors.
        Useful for cleaning up noisy representations.
        
        Args:
            vectors: List of binary hypervectors
            
        Returns:
            Majority hypervector
        """
        if not vectors:
            raise ValueError("No vectors provided")
        
        # Stack and compute majority along each dimension
        stacked = torch.stack(vectors)
        
        # For binary vectors, majority is sign of sum
        sums = stacked.sum(dim=0)
        majority = torch.sign(sums)
        
        # Handle zeros (map to +1 by default)
        majority[majority == 0] = 1
        
        return majority

    def cleanup(
        self, 
        noisy_vector: torch.Tensor, 
        memory_vectors: List[torch.Tensor],
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Clean up a noisy hypervector using associative memory.
        
        Finds the most similar vector in memory and returns it,
        effectively denoising the input.
        
        Args:
            noisy_vector: Noisy input hypervector
            memory_vectors: Clean reference vectors
            threshold: Minimum similarity threshold
            
        Returns:
            Cleaned hypervector (best match from memory)
        """
        if not memory_vectors:
            return noisy_vector
        
        best_similarity = -1.0
        best_match = noisy_vector
        
        for memory_vector in memory_vectors:
            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                noisy_vector.unsqueeze(0), 
                memory_vector.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = memory_vector
        
        return best_match.clone()


def bind_vectors(
    vectors: List[torch.Tensor], 
    binding_type: BindingType = BindingType.MULTIPLY
) -> torch.Tensor:
    """
    Convenience function to bind multiple vectors.
    
    Args:
        vectors: List of hypervectors to bind
        binding_type: Type of binding operation
        
    Returns:
        Bound hypervector
    """
    binder = HypervectorBinder(dimension=vectors[0].shape[0] if vectors else 10000)
    return binder.bind(vectors, binding_type)


def bundle_vectors(vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    """
    Convenience function to bundle multiple vectors.
    
    Args:
        vectors: List of hypervectors to bundle
        normalize: Whether to normalize the result
        
    Returns:
        Bundled hypervector
    """
    binder = HypervectorBinder(dimension=vectors[0].shape[0] if vectors else 10000)
    return binder.bundle(vectors, normalize)


def circular_shift(vector: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Apply circular shift to a hypervector.
    
    This is a fundamental operation in HDC for creating
    position-dependent representations.
    
    Args:
        vector: Input hypervector
        shift: Number of positions to shift (can be negative)
        
    Returns:
        Circularly shifted hypervector
    """
    return torch.roll(vector, shifts=shift)