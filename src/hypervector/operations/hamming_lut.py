"""
Optimized Hamming distance computation for REV verification.

This module provides 16-bit lookup tables and SIMD-accelerated operations
for efficient Hamming distance computation between hypervector signatures,
as mentioned in the REV paper for 10-20× speedup.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    jit = None
    NUMBA_AVAILABLE = False

from ..types import VectorUInt64

# Global lookup table cache
_POPCOUNT_LUT_16: Optional[NDArray[np.uint8]] = None


def generate_popcount_lut() -> NDArray[np.uint8]:
    """
    Generate a 16-bit population-count lookup table for REV.
    
    This creates the optimized LUT mentioned in the REV paper that provides
    significant speedup for Hamming distance computation.
    
    Returns:
        16-bit population count lookup table
    """
    global _POPCOUNT_LUT_16
    if _POPCOUNT_LUT_16 is None:
        # Generate LUT for all 16-bit values (65536 entries)
        _POPCOUNT_LUT_16 = np.array(
            [bin(i).count("1") for i in range(1 << 16)], 
            dtype=np.uint8
        )
    return _POPCOUNT_LUT_16


def popcount_u64(arr: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """
    Compute population count for uint64 array using 16-bit LUT.
    
    This implements the bit-packed operations mentioned in REV for
    efficient popcount computation.
    
    Args:
        arr: Array of uint64 values
        
    Returns:
        Population count for each element
    """
    lut = generate_popcount_lut()
    result = np.zeros_like(arr, dtype=np.uint64)
    
    for index, val in np.ndenumerate(arr):
        # Process 64-bit value in 16-bit chunks using LUT
        result[index] = (
            lut[(val >> 0) & 0xFFFF] +   # Bits 0-15
            lut[(val >> 16) & 0xFFFF] +  # Bits 16-31  
            lut[(val >> 32) & 0xFFFF] +  # Bits 32-47
            lut[(val >> 48) & 0xFFFF]    # Bits 48-63
        )
    return result


# Numba-accelerated version for additional speedup
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _popcount_u64_fast(
        arr: NDArray[np.uint64], 
        lut: NDArray[np.uint8]
    ) -> NDArray[np.uint64]:
        """Numba-accelerated popcount for maximum performance."""
        result = np.zeros(arr.shape, dtype=np.uint64)
        for i in range(arr.size):
            val = arr.flat[i]
            result.flat[i] = (
                lut[(val >> 0) & 0xFFFF] +
                lut[(val >> 16) & 0xFFFF] +
                lut[(val >> 32) & 0xFFFF] +
                lut[(val >> 48) & 0xFFFF]
            )
        return result
else:
    def _popcount_u64_fast(
        arr: NDArray[np.uint64], 
        lut: NDArray[np.uint8]
    ) -> NDArray[np.uint64]:
        """Fallback to standard popcount when Numba unavailable."""
        return popcount_u64(arr)


def hamming_distance_cpu(
    vec1: VectorUInt64, 
    vec2: VectorUInt64, 
    lut: Optional[NDArray[np.uint8]] = None
) -> int:
    """
    Compute Hamming distance between two packed binary hypervectors.
    
    This implements the optimized Hamming distance computation mentioned
    in the REV paper, using 16-bit LUTs for 10-20× speedup over naive
    bit counting methods.
    
    Args:
        vec1: First packed binary hypervector
        vec2: Second packed binary hypervector  
        lut: Optional precomputed lookup table
        
    Returns:
        Hamming distance (number of differing bits)
    """
    if lut is None:
        lut = generate_popcount_lut()
    
    # XOR to find differing bits
    xor_result = np.bitwise_xor(vec1, vec2)
    
    # Count set bits using optimized popcount
    return int(_popcount_u64_fast(xor_result, lut).sum())


class HammingLUT:
    """
    Hamming distance computer using cached lookup table for REV.
    
    This class implements the optimized Hamming distance computation
    infrastructure mentioned in the REV paper, providing:
    - 16-bit lookup tables for fast bit counting
    - Batch processing for multiple comparisons
    - SIMD acceleration when available
    """

    def __init__(self) -> None:
        """Initialize with precomputed lookup table."""
        self.lut = generate_popcount_lut()
        
    def distance(self, vec1: VectorUInt64, vec2: VectorUInt64) -> int:
        """
        Compute Hamming distance between two vectors.
        
        Args:
            vec1: First packed binary hypervector
            vec2: Second packed binary hypervector
            
        Returns:
            Hamming distance
        """
        return hamming_distance_cpu(vec1, vec2, self.lut)
    
    def distance_batch(
        self, 
        vecs1: NDArray[np.uint64], 
        vecs2: NDArray[np.uint64]
    ) -> NDArray[np.int32]:
        """
        Compute pairwise Hamming distances for batches of vectors.
        
        This is useful for REV verification when comparing multiple
        model signatures efficiently.
        
        Args:
            vecs1: First batch of vectors (N × D)
            vecs2: Second batch of vectors (M × D)  
            
        Returns:
            Distance matrix (N × M)
        """
        n_vecs1, n_vecs2 = vecs1.shape[0], vecs2.shape[0]
        result = np.zeros((n_vecs1, n_vecs2), dtype=np.int32)
        
        for i, v1 in enumerate(vecs1):
            for j, v2 in enumerate(vecs2):
                result[i, j] = hamming_distance_cpu(v1, v2, self.lut)
                
        return result
    
    def similarity_batch(
        self, 
        vecs1: NDArray[np.uint64], 
        vecs2: NDArray[np.uint64],
        max_distance: int
    ) -> NDArray[np.float32]:
        """
        Compute similarity scores from Hamming distances.
        
        Converts Hamming distances to similarities in [0,1] range
        for REV threshold-based decisions.
        
        Args:
            vecs1: First batch of vectors
            vecs2: Second batch of vectors
            max_distance: Maximum possible Hamming distance
            
        Returns:
            Similarity matrix with values in [0,1]
        """
        distances = self.distance_batch(vecs1, vecs2)
        similarities = 1.0 - (distances.astype(np.float32) / max_distance)
        return np.clip(similarities, 0.0, 1.0)
    
    def closest_match(
        self, 
        query: VectorUInt64, 
        candidates: NDArray[np.uint64]
    ) -> tuple[int, int]:
        """
        Find closest matching vector in candidate set.
        
        Useful for REV cleanup operations and nearest neighbor search
        in hypervector memory.
        
        Args:
            query: Query vector
            candidates: Set of candidate vectors
            
        Returns:
            Tuple of (best_index, min_distance)
        """
        min_distance = float('inf')
        best_index = -1
        
        for i, candidate in enumerate(candidates):
            distance = hamming_distance_cpu(query, candidate, self.lut)
            if distance < min_distance:
                min_distance = distance
                best_index = i
                
        return best_index, int(min_distance)


def pack_binary_vector(binary_vec: NDArray[np.bool_]) -> VectorUInt64:
    """
    Pack binary hypervector into uint64 array for efficient storage.
    
    This implements the bit-packing mentioned in REV for memory
    efficiency and faster operations.
    
    Args:
        binary_vec: Binary hypervector (bool array)
        
    Returns:
        Packed representation as uint64 array
    """
    # Pad to multiple of 64 if necessary
    padded_length = ((len(binary_vec) + 63) // 64) * 64
    padded_vec = np.zeros(padded_length, dtype=np.bool_)
    padded_vec[:len(binary_vec)] = binary_vec
    
    # Pack into uint64 chunks
    packed = np.packbits(padded_vec.view(np.uint8)).view(np.uint64)
    
    return packed


def unpack_binary_vector(
    packed_vec: VectorUInt64, 
    original_length: int
) -> NDArray[np.bool_]:
    """
    Unpack uint64 array back to binary hypervector.
    
    Args:
        packed_vec: Packed uint64 representation
        original_length: Original vector length
        
    Returns:
        Unpacked binary hypervector
    """
    # Convert to bytes and unpack bits
    unpacked = np.unpackbits(packed_vec.view(np.uint8)).view(np.bool_)
    
    # Trim to original length
    return unpacked[:original_length]


def export_platform_implementations() -> Dict[str, str]:
    """
    Report available accelerator implementations for REV.
    
    Returns:
        Dictionary mapping platform to implementation type
    """
    impls = {"cpu": "16-bit-lut"}
    
    if NUMBA_AVAILABLE:
        impls["numba"] = "jit-accelerated"
        
    return impls


# Export the main components
__all__ = [
    "HammingLUT",
    "hamming_distance_cpu",
    "generate_popcount_lut", 
    "popcount_u64",
    "pack_binary_vector",
    "unpack_binary_vector",
    "export_platform_implementations"
]