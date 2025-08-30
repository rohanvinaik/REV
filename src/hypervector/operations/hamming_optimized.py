"""
Enhanced Hamming distance operations with multiple acceleration strategies.

This module extends the basic Hamming distance implementation with:
- Memory-mapped file support for large datasets
- Adaptive algorithm selection based on input characteristics
- Advanced similarity metrics beyond Hamming distance
- Optimized distance matrix computation with blocking
- Conversion utilities for different vector formats
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, List, Any, Callable
import numpy as np
from numpy.typing import NDArray
import time
import warnings
import mmap
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import json

try:
    from numba import jit, prange, njit, cuda, typed
    from numba.core import types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    jit = None
    njit = None
    prange = range
    cuda = None
    typed = None
    types = None
    NumbaDict = dict
    NUMBA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from .hamming_lut import (
    HammingLUT,
    hamming_distance_cpu,
    hamming_distance_gpu,
    pack_binary_vector_simd,
    unpack_binary_vector,
    _numba_hamming_batch
)
from ..types import VectorUInt64


class AlgorithmType(Enum):
    """Available Hamming distance algorithms."""
    NAIVE = "naive"
    LUT_16BIT = "lut_16bit"
    SIMD_NUMPY = "simd_numpy"
    NUMBA_PARALLEL = "numba_parallel"
    GPU_TORCH = "gpu_torch"
    GPU_CUPY = "gpu_cupy"
    GPU_NUMBA = "gpu_numba"
    BIT_PARALLEL = "bit_parallel"


@dataclass
class PerformanceProfile:
    """Performance characteristics for algorithm selection."""
    vector_size: int
    batch_size: int
    available_memory: int
    has_gpu: bool
    has_numba: bool
    cache_size: int = 8 * 1024 * 1024  # 8MB L3 cache default


class HammingDistanceOptimized:
    """
    Highly optimized Hamming distance computer with adaptive algorithm selection.
    
    This class provides:
    - Multiple algorithm implementations (CPU, SIMD, GPU)
    - Adaptive selection based on input characteristics
    - Memory-mapped file support for large datasets
    - Advanced similarity metrics
    - Optimized batch processing
    """
    
    def __init__(
        self,
        enable_gpu: bool = True,
        enable_simd: bool = True,
        cache_size: int = 100,
        adaptive: bool = True
    ):
        """
        Initialize optimized Hamming distance computer.
        
        Args:
            enable_gpu: Whether to use GPU acceleration if available
            enable_simd: Whether to use SIMD operations
            cache_size: Number of distance computations to cache
            adaptive: Whether to adaptively select algorithms
        """
        self.enable_gpu = enable_gpu and (TORCH_AVAILABLE or CUPY_AVAILABLE)
        self.enable_simd = enable_simd
        self.adaptive = adaptive
        
        # Initialize base LUT computer
        self.lut_computer = HammingLUT(enable_simd=enable_simd)
        
        # Initialize caches
        self.distance_cache: Dict[Tuple[int, int], int] = {}
        self.cache_size = cache_size
        
        # Performance profiling
        self.algorithm_times: Dict[AlgorithmType, List[float]] = {
            algo: [] for algo in AlgorithmType
        }
        
        # Memory-mapped file handles
        self.mmap_handles: Dict[str, mmap.mmap] = {}
        
    def distance(
        self,
        vec1: Union[VectorUInt64, NDArray[np.bool_], str],
        vec2: Union[VectorUInt64, NDArray[np.bool_], str],
        algorithm: Optional[AlgorithmType] = None
    ) -> int:
        """
        Compute Hamming distance with adaptive algorithm selection.
        
        Args:
            vec1: First vector or path to memory-mapped file
            vec2: Second vector or path to memory-mapped file
            algorithm: Force specific algorithm (None for adaptive)
            
        Returns:
            Hamming distance
        """
        # Handle memory-mapped files
        if isinstance(vec1, str):
            vec1 = self._load_mmap_vector(vec1)
        if isinstance(vec2, str):
            vec2 = self._load_mmap_vector(vec2)
        
        # Convert to packed format if needed
        if isinstance(vec1, np.ndarray) and vec1.dtype == np.bool_:
            vec1 = pack_binary_vector_simd(vec1)
        if isinstance(vec2, np.ndarray) and vec2.dtype == np.bool_:
            vec2 = pack_binary_vector_simd(vec2)
        
        # Check cache
        cache_key = (id(vec1), id(vec2))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Select algorithm
        if algorithm is None and self.adaptive:
            algorithm = self._select_algorithm(vec1, vec2)
        elif algorithm is None:
            algorithm = AlgorithmType.LUT_16BIT
        
        # Compute distance
        start_time = time.perf_counter()
        
        if algorithm == AlgorithmType.NAIVE:
            distance = self._distance_naive(vec1, vec2)
        elif algorithm == AlgorithmType.LUT_16BIT:
            distance = self.lut_computer.distance(vec1, vec2)
        elif algorithm == AlgorithmType.SIMD_NUMPY:
            distance = self.lut_computer.compute_distance_simd(vec1, vec2)
        elif algorithm == AlgorithmType.NUMBA_PARALLEL and NUMBA_AVAILABLE:
            distance = self._distance_numba_parallel(vec1, vec2)
        elif algorithm in [AlgorithmType.GPU_TORCH, AlgorithmType.GPU_CUPY]:
            distance = hamming_distance_gpu(vec1, vec2)
        elif algorithm == AlgorithmType.BIT_PARALLEL:
            distance = self._distance_bit_parallel(vec1, vec2)
        else:
            # Fallback
            distance = self.lut_computer.distance(vec1, vec2)
        
        # Update performance profile
        elapsed = time.perf_counter() - start_time
        self.algorithm_times[algorithm].append(elapsed)
        
        # Update cache
        if len(self.distance_cache) >= self.cache_size:
            # Remove oldest entry
            self.distance_cache.pop(next(iter(self.distance_cache)))
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _select_algorithm(
        self,
        vec1: VectorUInt64,
        vec2: VectorUInt64
    ) -> AlgorithmType:
        """
        Select optimal algorithm based on input characteristics.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Selected algorithm type
        """
        vector_size = len(vec1) * 64  # bits
        
        # GPU for large vectors
        if self.enable_gpu and vector_size > 100000:
            if TORCH_AVAILABLE:
                return AlgorithmType.GPU_TORCH
            elif CUPY_AVAILABLE:
                return AlgorithmType.GPU_CUPY
        
        # Numba parallel for medium vectors
        if NUMBA_AVAILABLE and vector_size > 10000:
            return AlgorithmType.NUMBA_PARALLEL
        
        # SIMD for small-medium vectors
        if self.enable_simd and vector_size > 1000:
            return AlgorithmType.SIMD_NUMPY
        
        # LUT for small vectors
        return AlgorithmType.LUT_16BIT
    
    def _distance_naive(
        self,
        vec1: VectorUInt64,
        vec2: VectorUInt64
    ) -> int:
        """Naive Hamming distance implementation for comparison."""
        xor_result = np.bitwise_xor(vec1, vec2)
        distance = 0
        for val in xor_result:
            # Count bits the slow way
            val = int(val)
            while val:
                distance += val & 1
                val >>= 1
        return distance
    
    def _distance_numba_parallel(
        self,
        vec1: VectorUInt64,
        vec2: VectorUInt64
    ) -> int:
        """Numba-parallel Hamming distance."""
        if not NUMBA_AVAILABLE:
            return self.lut_computer.distance(vec1, vec2)
        
        @njit(parallel=True)
        def parallel_hamming(v1, v2):
            xor_result = np.bitwise_xor(v1, v2)
            total = 0
            for i in prange(len(xor_result)):
                val = xor_result[i]
                count = 0
                for j in range(64):
                    count += (val >> j) & 1
                total += count
            return total
        
        return parallel_hamming(vec1, vec2)
    
    def _distance_bit_parallel(
        self,
        vec1: VectorUInt64,
        vec2: VectorUInt64
    ) -> int:
        """
        Bit-parallel Hamming distance using packed operations.
        
        Uses the parallel bit counting algorithm for efficiency.
        """
        xor_result = np.bitwise_xor(vec1, vec2)
        
        # Parallel bit counting algorithm (Brian Kernighan's algorithm variant)
        total = 0
        for val in xor_result:
            val = int(val)
            # Use built-in popcount for correctness
            # The parallel algorithm needs unsigned arithmetic which is tricky in Python
            count = 0
            while val:
                count += 1
                val &= val - 1  # Clear the least significant bit
            total += count
        
        return total
    
    def _load_mmap_vector(self, filepath: str) -> VectorUInt64:
        """
        Load vector from memory-mapped file.
        
        Args:
            filepath: Path to binary file
            
        Returns:
            Memory-mapped vector
        """
        if filepath not in self.mmap_handles:
            with open(filepath, 'r+b') as f:
                self.mmap_handles[filepath] = mmap.mmap(
                    f.fileno(), 0, access=mmap.ACCESS_READ
                )
        
        mm = self.mmap_handles[filepath]
        # Read as uint64 array
        data = np.frombuffer(mm, dtype=np.uint64)
        return data
    
    def distance_matrix(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]] = None,
        block_size: int = 64,
        symmetric: bool = False
    ) -> NDArray[np.float32]:
        """
        Compute distance matrix with cache-friendly blocking.
        
        Args:
            vecs1: First set of vectors (N × D)
            vecs2: Second set of vectors (M × D), or None for self-distances
            block_size: Block size for cache efficiency
            symmetric: Whether to exploit symmetry (when vecs2 is None)
            
        Returns:
            Distance matrix (N × M or N × N)
        """
        if vecs2 is None:
            vecs2 = vecs1
            symmetric = True
        
        n1, d = vecs1.shape
        n2 = vecs2.shape[0]
        distances = np.zeros((n1, n2), dtype=np.int32)
        
        # Process in blocks for cache efficiency
        for i in range(0, n1, block_size):
            i_end = min(i + block_size, n1)
            
            for j in range(0, n2, block_size):
                j_end = min(j + block_size, n2)
                
                # Skip redundant computations if symmetric
                if symmetric and j > i + block_size:
                    # Already computed in transpose
                    continue
                
                # Compute block
                block1 = vecs1[i:i_end]
                block2 = vecs2[j:j_end]
                
                if NUMBA_AVAILABLE and self.enable_simd:
                    block_distances = _numba_hamming_batch(
                        block1, block2, self.lut_computer.lut_16bit
                    )
                else:
                    block_distances = self.lut_computer.distance_batch(
                        block1, block2
                    )
                
                distances[i:i_end, j:j_end] = block_distances
                
                # Copy to transpose if symmetric and different blocks
                if symmetric and i != j:
                    distances[j:j_end, i:i_end] = block_distances.T
        
        return distances
    
    def similarity_matrix(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]] = None,
        metric: str = "hamming",
        **kwargs
    ) -> NDArray[np.float32]:
        """
        Compute similarity matrix using various metrics.
        
        Args:
            vecs1: First set of vectors
            vecs2: Second set of vectors
            metric: Similarity metric ("hamming", "jaccard", "cosine", "dice")
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Similarity matrix with values in [0, 1]
        """
        if metric == "hamming":
            return self._similarity_hamming(vecs1, vecs2, **kwargs)
        elif metric == "jaccard":
            return self._similarity_jaccard(vecs1, vecs2)
        elif metric == "cosine":
            return self._similarity_cosine(vecs1, vecs2)
        elif metric == "dice":
            return self._similarity_dice(vecs1, vecs2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _similarity_hamming(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]],
        max_distance: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Hamming similarity (1 - normalized distance)."""
        distances = self.distance_matrix(vecs1, vecs2)
        
        if max_distance is None:
            # Maximum possible distance is vector length in bits
            max_distance = vecs1.shape[1] * 64
        
        similarities = 1.0 - (distances.astype(np.float32) / max_distance)
        return np.clip(similarities, 0.0, 1.0)
    
    def _similarity_jaccard(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]]
    ) -> NDArray[np.float32]:
        """
        Jaccard similarity for binary vectors.
        
        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        if vecs2 is None:
            vecs2 = vecs1
        
        n1, n2 = vecs1.shape[0], vecs2.shape[0]
        similarities = np.zeros((n1, n2), dtype=np.float32)
        
        for i in range(n1):
            for j in range(n2):
                # Compute intersection and union
                and_bits = np.bitwise_and(vecs1[i], vecs2[j])
                or_bits = np.bitwise_or(vecs1[i], vecs2[j])
                
                # Count bits
                intersection = sum(bin(int(x)).count('1') for x in and_bits)
                union = sum(bin(int(x)).count('1') for x in or_bits)
                
                if union > 0:
                    similarities[i, j] = intersection / union
                else:
                    similarities[i, j] = 1.0  # Both vectors are zero
        
        return similarities
    
    def _similarity_cosine(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]]
    ) -> NDArray[np.float32]:
        """
        Cosine similarity for binary vectors.
        
        cos(A, B) = |A ∩ B| / sqrt(|A| * |B|)
        """
        if vecs2 is None:
            vecs2 = vecs1
        
        n1, n2 = vecs1.shape[0], vecs2.shape[0]
        similarities = np.zeros((n1, n2), dtype=np.float32)
        
        # Precompute magnitudes
        mag1 = np.array([
            sum(bin(int(x)).count('1') for x in vec)
            for vec in vecs1
        ])
        mag2 = np.array([
            sum(bin(int(x)).count('1') for x in vec)
            for vec in vecs2
        ])
        
        for i in range(n1):
            for j in range(n2):
                # Compute dot product (intersection for binary)
                and_bits = np.bitwise_and(vecs1[i], vecs2[j])
                dot_product = sum(bin(int(x)).count('1') for x in and_bits)
                
                denominator = np.sqrt(mag1[i] * mag2[j])
                if denominator > 0:
                    similarities[i, j] = dot_product / denominator
                else:
                    similarities[i, j] = 0.0
        
        return similarities
    
    def _similarity_dice(
        self,
        vecs1: NDArray[np.uint64],
        vecs2: Optional[NDArray[np.uint64]]
    ) -> NDArray[np.float32]:
        """
        Dice coefficient for binary vectors.
        
        D(A, B) = 2 * |A ∩ B| / (|A| + |B|)
        """
        if vecs2 is None:
            vecs2 = vecs1
        
        n1, n2 = vecs1.shape[0], vecs2.shape[0]
        similarities = np.zeros((n1, n2), dtype=np.float32)
        
        # Precompute cardinalities
        card1 = np.array([
            sum(bin(int(x)).count('1') for x in vec)
            for vec in vecs1
        ])
        card2 = np.array([
            sum(bin(int(x)).count('1') for x in vec)
            for vec in vecs2
        ])
        
        for i in range(n1):
            for j in range(n2):
                # Compute intersection
                and_bits = np.bitwise_and(vecs1[i], vecs2[j])
                intersection = sum(bin(int(x)).count('1') for x in and_bits)
                
                denominator = card1[i] + card2[j]
                if denominator > 0:
                    similarities[i, j] = 2 * intersection / denominator
                else:
                    similarities[i, j] = 1.0  # Both vectors are zero
        
        return similarities
    
    def convert_format(
        self,
        vector: Union[NDArray[np.bool_], NDArray[np.uint8], NDArray[np.uint64], List[int]],
        target_format: str = "uint64"
    ) -> Union[VectorUInt64, NDArray[np.bool_], NDArray[np.uint8]]:
        """
        Convert between different vector formats.
        
        Args:
            vector: Input vector in any supported format
            target_format: Target format ("uint64", "bool", "uint8", "list")
            
        Returns:
            Converted vector
        """
        # Convert to numpy array if needed
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.uint8)
        
        # Determine source format
        if vector.dtype == np.bool_:
            source_format = "bool"
        elif vector.dtype == np.uint8:
            source_format = "uint8"
        elif vector.dtype == np.uint64:
            source_format = "uint64"
        else:
            raise ValueError(f"Unsupported source format: {vector.dtype}")
        
        # Convert as needed
        if source_format == target_format:
            return vector
        
        if target_format == "uint64":
            if source_format in ["bool", "uint8"]:
                return pack_binary_vector_simd(vector.astype(np.bool_))
            else:
                return vector
        
        elif target_format == "bool":
            if source_format == "uint64":
                # Assume original length is len(vector) * 64
                return unpack_binary_vector(vector, len(vector) * 64)
            else:
                return vector.astype(np.bool_)
        
        elif target_format == "uint8":
            if source_format == "uint64":
                unpacked = unpack_binary_vector(vector, len(vector) * 64)
                return unpacked.astype(np.uint8)
            else:
                return vector.astype(np.uint8)
        
        elif target_format == "list":
            if source_format == "uint64":
                unpacked = unpack_binary_vector(vector, len(vector) * 64)
                return unpacked.astype(np.uint8).tolist()
            else:
                return vector.astype(np.uint8).tolist()
        
        else:
            raise ValueError(f"Unknown target format: {target_format}")
    
    def save_vectors(
        self,
        vectors: NDArray[np.uint64],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save vectors to file with optional metadata.
        
        Args:
            vectors: Vectors to save
            filepath: Output file path
            metadata: Optional metadata to store
        """
        filepath = Path(filepath)
        
        # Save vectors as binary
        vectors.tofile(filepath)
        
        # Save metadata if provided
        if metadata:
            meta_path = filepath.with_suffix('.meta.json')
            metadata['shape'] = vectors.shape
            metadata['dtype'] = str(vectors.dtype)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_vectors(
        self,
        filepath: str,
        mmap_mode: Optional[str] = None
    ) -> Tuple[NDArray[np.uint64], Optional[Dict[str, Any]]]:
        """
        Load vectors from file.
        
        Args:
            filepath: Input file path
            mmap_mode: Memory-map mode ('r', 'r+', None)
            
        Returns:
            Tuple of (vectors, metadata)
        """
        filepath = Path(filepath)
        
        # Load metadata if available
        metadata = None
        meta_path = filepath.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        # Load vectors
        if mmap_mode:
            vectors = np.memmap(
                filepath, dtype=np.uint64, mode=mmap_mode
            )
            if metadata and 'shape' in metadata:
                vectors = vectors.reshape(metadata['shape'])
        else:
            vectors = np.fromfile(filepath, dtype=np.uint64)
            if metadata and 'shape' in metadata:
                vectors = vectors.reshape(metadata['shape'])
        
        return vectors, metadata
    
    def benchmark(
        self,
        dimensions: List[int] = [1000, 10000, 100000],
        algorithms: Optional[List[AlgorithmType]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Benchmark different algorithms across dimensions.
        
        Args:
            dimensions: Vector dimensions to test
            algorithms: Algorithms to benchmark (None for all available)
            
        Returns:
            Benchmark results as nested dictionary
        """
        if algorithms is None:
            algorithms = [
                AlgorithmType.NAIVE,
                AlgorithmType.LUT_16BIT,
                AlgorithmType.SIMD_NUMPY
            ]
            if NUMBA_AVAILABLE:
                algorithms.append(AlgorithmType.NUMBA_PARALLEL)
            if self.enable_gpu:
                if TORCH_AVAILABLE:
                    algorithms.append(AlgorithmType.GPU_TORCH)
                elif CUPY_AVAILABLE:
                    algorithms.append(AlgorithmType.GPU_CUPY)
        
        results = {algo.value: {} for algo in algorithms}
        
        for dim in dimensions:
            print(f"\nBenchmarking dimension {dim}...")
            
            # Generate test vectors
            vec1 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            vec2 = np.random.randint(0, 2, size=dim, dtype=np.bool_)
            
            # Pack for efficient operations
            packed1 = pack_binary_vector_simd(vec1)
            packed2 = pack_binary_vector_simd(vec2)
            
            # Benchmark each algorithm
            for algo in algorithms:
                # Warmup
                try:
                    _ = self.distance(packed1, packed2, algorithm=algo)
                    
                    # Time multiple runs
                    n_trials = 10
                    start = time.perf_counter()
                    for _ in range(n_trials):
                        _ = self.distance(packed1, packed2, algorithm=algo)
                    elapsed = (time.perf_counter() - start) / n_trials
                    
                    results[algo.value][dim] = elapsed * 1000  # Convert to ms
                    print(f"  {algo.value}: {elapsed*1000:.3f} ms")
                except Exception as e:
                    print(f"  {algo.value}: Failed - {e}")
                    results[algo.value][dim] = float('inf')
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance statistics for all algorithms.
        
        Returns:
            Performance report with statistics
        """
        report = {}
        
        for algo, times in self.algorithm_times.items():
            if times:
                report[algo.value] = {
                    'n_calls': len(times),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        # Add cache statistics
        report['cache'] = {
            'size': len(self.distance_cache),
            'max_size': self.cache_size,
            'hit_rate': 0.0  # Would need to track hits/misses
        }
        
        # Add system info
        report['system'] = {
            'gpu_available': self.enable_gpu,
            'numba_available': NUMBA_AVAILABLE,
            'simd_enabled': self.enable_simd
        }
        
        return report
    
    def cleanup(self) -> None:
        """Clean up resources (memory-mapped files, GPU memory, etc.)."""
        # Close memory-mapped files
        for mm in self.mmap_handles.values():
            mm.close()
        self.mmap_handles.clear()
        
        # Clear caches
        self.distance_cache.clear()
        
        # Clear GPU memory if applicable
        if TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
        if CUPY_AVAILABLE and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


def create_optimized_computer(
    profile: Optional[PerformanceProfile] = None
) -> HammingDistanceOptimized:
    """
    Factory function to create optimized Hamming computer with best settings.
    
    Args:
        profile: Optional performance profile for configuration
        
    Returns:
        Configured HammingDistanceOptimized instance
    """
    if profile is None:
        # Auto-detect system capabilities
        has_gpu = TORCH_AVAILABLE or CUPY_AVAILABLE
        has_numba = NUMBA_AVAILABLE
        
        profile = PerformanceProfile(
            vector_size=10000,  # Default assumption
            batch_size=100,
            available_memory=8 * 1024 * 1024 * 1024,  # 8GB
            has_gpu=has_gpu,
            has_numba=has_numba
        )
    
    # Configure based on profile
    enable_gpu = profile.has_gpu and profile.vector_size > 50000
    enable_simd = True  # Always beneficial
    cache_size = min(1000, profile.available_memory // (profile.vector_size * 8))
    
    return HammingDistanceOptimized(
        enable_gpu=enable_gpu,
        enable_simd=enable_simd,
        cache_size=cache_size,
        adaptive=True
    )


# Export main components
__all__ = [
    "HammingDistanceOptimized",
    "AlgorithmType",
    "PerformanceProfile",
    "create_optimized_computer"
]