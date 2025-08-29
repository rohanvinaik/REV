"""
Type aliases for REV hypervector operations.

This module provides consistent type aliases for numpy arrays used throughout
the REV hypervector module, improving type safety and code readability.
"""

from typing import TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

# Basic vector types
VectorF32: TypeAlias = NDArray[np.float32]
VectorF64: TypeAlias = NDArray[np.float64]
VectorBool: TypeAlias = NDArray[np.bool_]
VectorInt32: TypeAlias = NDArray[np.int32]
VectorUInt64: TypeAlias = NDArray[np.uint64]

# Matrix types
MatrixF32: TypeAlias = NDArray[np.float32]
MatrixF64: TypeAlias = NDArray[np.float64]
MatrixBool: TypeAlias = NDArray[np.bool_]

# Union types for flexibility
Vector: TypeAlias = Union[VectorF32, VectorF64, VectorBool]
Matrix: TypeAlias = Union[MatrixF32, MatrixF64, MatrixBool]

# REV-specific types
HypervectorF32: TypeAlias = VectorF32  # Standard hypervector
HypervectorBinary: TypeAlias = VectorBool  # Binary hypervector for REV
PackedBinary: TypeAlias = VectorUInt64  # Packed binary representation for efficiency
DistanceMatrix: TypeAlias = MatrixF32  # Distance/similarity matrix

# REV verification types
ModelSignature: TypeAlias = HypervectorF32  # Model activation/logit signature
ComparisonResult: TypeAlias = VectorF32  # Batch comparison results