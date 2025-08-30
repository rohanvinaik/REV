"""Hamming distance operations for REV hypervector verification."""

# Re-export from operations module for convenience
from .operations.hamming_lut import (
    HammingLUT,
    hamming_distance_cpu,
    hamming_distance_gpu,
    generate_popcount_lut,
    pack_binary_vector,
    pack_binary_vector_simd,
    unpack_binary_vector,
    export_platform_implementations,
    benchmark_hamming_implementations
)

# Import optimized implementations
from .operations.hamming_optimized import (
    HammingDistanceOptimized,
    AlgorithmType,
    PerformanceProfile,
    create_optimized_computer
)

__all__ = [
    # Basic operations
    "HammingLUT",
    "hamming_distance_cpu",
    "hamming_distance_gpu",
    "generate_popcount_lut",
    "pack_binary_vector",
    "pack_binary_vector_simd",
    "unpack_binary_vector",
    "export_platform_implementations",
    "benchmark_hamming_implementations",
    # Optimized operations
    "HammingDistanceOptimized",
    "AlgorithmType",
    "PerformanceProfile",
    "create_optimized_computer"
]