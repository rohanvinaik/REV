"""Hyperdimensional Computing Core for REV."""

from .encoder import (
    HypervectorEncoder, 
    HypervectorConfig, 
    ProjectionType,
    UnifiedHDCEncoder
)
from .binding import HypervectorBinder, BindingOperation, BindingType
from .operations import bind_vectors, bundle_vectors, circular_shift

__all__ = [
    "HypervectorEncoder",
    "UnifiedHDCEncoder",
    "HypervectorConfig", 
    "ProjectionType",
    "HypervectorBinder",
    "BindingOperation",
    "BindingType",
    "bind_vectors",
    "bundle_vectors", 
    "circular_shift"
]