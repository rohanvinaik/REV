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

__all__ = [
    "HammingLUT",
    "hamming_distance_cpu",
    "hamming_distance_gpu",
    "generate_popcount_lut",
    "pack_binary_vector",
    "pack_binary_vector_simd",
    "unpack_binary_vector",
    "export_platform_implementations",
    "benchmark_hamming_implementations"
]