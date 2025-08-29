"""
Enhanced Binding Operations for Hyperdimensional Computing.

This module implements advanced binding operations including XOR, permutation,
circular convolution, Fourier-domain binding, and weighted probability binding.
"""

import numpy as np
import torch
from typing import List, Optional, Union, Tuple
from scipy.fft import fft, ifft, fft2, ifft2
from scipy import signal


class BindingOperations:
    """
    Advanced binding operations for hyperdimensional computing in REV.
    
    Implements multiple binding strategies for different use cases:
    - XOR binding for logical relationships
    - Permutation binding for positional encoding
    - Circular convolution for sequences
    - Fourier domain binding for frequency analysis
    - Weighted binding for probability distributions
    """
    
    def __init__(self, dimension: int = 10000, seed: Optional[int] = None):
        """
        Initialize binding operations.
        
        Args:
            dimension: Dimensionality of hypervectors
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Cache for permutation matrices
        self._perm_cache = {}
        
        # Precompute common permutation patterns
        self._init_permutations()
    
    def _init_permutations(self):
        """Initialize common permutation patterns."""
        # Standard cyclic shifts
        for shift in [1, 2, 4, 8, 16, 32]:
            self._perm_cache[f'shift_{shift}'] = np.roll(np.arange(self.dimension), shift)
        
        # Random permutation for mixing
        self._perm_cache['random'] = np.random.permutation(self.dimension)
        
        # Bit-reversal permutation (useful for FFT-like operations)
        n_bits = int(np.log2(self.dimension)) if self.dimension & (self.dimension - 1) == 0 else None
        if n_bits:
            self._perm_cache['bit_reverse'] = self._bit_reversal_permutation(self.dimension)
    
    def _bit_reversal_permutation(self, n: int) -> np.ndarray:
        """Generate bit-reversal permutation."""
        n_bits = int(np.log2(n))
        indices = np.arange(n)
        reversed_indices = np.zeros(n, dtype=int)
        
        for i in range(n):
            reversed_indices[i] = int(bin(i)[2:].zfill(n_bits)[::-1], 2)
        
        return reversed_indices
    
    def xor_bind(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        XOR binding for logical relationships.
        
        For binary vectors, performs XOR. For continuous vectors,
        uses sign-based XOR with magnitude preservation.
        
        Args:
            a: First hypervector
            b: Second hypervector
            
        Returns:
            XOR-bound hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        # Check if binary
        is_binary_a = np.all(np.logical_or(a == 0, a == 1) | np.logical_or(a == -1, a == 1))
        is_binary_b = np.all(np.logical_or(b == 0, b == 1) | np.logical_or(b == -1, b == 1))
        
        if is_binary_a and is_binary_b:
            # True XOR for binary
            return np.logical_xor(a > 0, b > 0).astype(np.float32) * 2 - 1
        else:
            # Sign-based XOR for continuous
            sign_a = np.sign(a)
            sign_b = np.sign(b)
            magnitude = np.sqrt(np.abs(a * b))
            return sign_a * sign_b * magnitude
    
    def permutation_bind(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
        perm_type: str = 'position'
    ) -> np.ndarray:
        """
        Permutation binding for positional encoding.
        
        Args:
            a: First hypervector
            b: Second hypervector (used to determine permutation)
            perm_type: Type of permutation ('position', 'value', 'hash')
            
        Returns:
            Permutation-bound hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        if perm_type == 'position':
            # Use position-based permutation
            shift = int(np.sum(b) % self.dimension)
            return np.roll(a, shift)
        
        elif perm_type == 'value':
            # Use value-based permutation
            perm_indices = np.argsort(b)
            return a[perm_indices]
        
        elif perm_type == 'hash':
            # Use hash-based deterministic permutation
            hash_val = hash(b.tobytes())
            np.random.seed(hash_val % (2**32))
            perm = np.random.permutation(self.dimension)
            np.random.seed(None)  # Reset seed
            return a[perm]
        
        else:
            raise ValueError(f"Unknown permutation type: {perm_type}")
    
    def circular_convolve(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
        mode: str = 'fft'
    ) -> np.ndarray:
        """
        Circular convolution for sequence binding.
        
        Args:
            a: First hypervector
            b: Second hypervector
            mode: Convolution mode ('fft', 'direct', 'overlap-add')
            
        Returns:
            Circularly convolved hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        if mode == 'fft':
            # FFT-based circular convolution (most efficient)
            a_fft = fft(a)
            b_fft = fft(b)
            result_fft = a_fft * b_fft
            result = np.real(ifft(result_fft))
            
        elif mode == 'direct':
            # Direct circular convolution
            result = signal.convolve(a, b, mode='same', method='direct')
            
        elif mode == 'overlap-add':
            # Overlap-add method for long sequences
            result = signal.oaconvolve(a, b, mode='same')
            
        else:
            raise ValueError(f"Unknown convolution mode: {mode}")
        
        return result.astype(np.float32)
    
    def fourier_bind(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
        domain: str = 'frequency'
    ) -> np.ndarray:
        """
        Fourier domain binding for frequency analysis.
        
        Args:
            a: First hypervector
            b: Second hypervector
            domain: Binding domain ('frequency', 'phase', 'magnitude')
            
        Returns:
            Fourier-bound hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        # Transform to frequency domain
        a_fft = fft(a)
        b_fft = fft(b)
        
        if domain == 'frequency':
            # Standard frequency domain multiplication
            result_fft = a_fft * np.conj(b_fft)
            
        elif domain == 'phase':
            # Phase-only binding
            phase_a = np.angle(a_fft)
            phase_b = np.angle(b_fft)
            magnitude = np.abs(a_fft) * np.abs(b_fft)
            result_fft = magnitude * np.exp(1j * (phase_a + phase_b))
            
        elif domain == 'magnitude':
            # Magnitude-only binding
            magnitude_a = np.abs(a_fft)
            magnitude_b = np.abs(b_fft)
            phase = np.angle(a_fft)
            result_fft = (magnitude_a * magnitude_b) * np.exp(1j * phase)
            
        else:
            raise ValueError(f"Unknown Fourier domain: {domain}")
        
        # Transform back to time domain
        result = np.real(ifft(result_fft))
        return result.astype(np.float32)
    
    def weighted_bind(
        self,
        vectors: List[Union[np.ndarray, torch.Tensor]],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Weighted binding for probability distributions.
        
        Args:
            vectors: List of hypervectors to bind
            weights: Weights for each vector (probabilities)
            normalize: Whether to normalize the result
            
        Returns:
            Weighted bound hypervector
        """
        if not vectors:
            raise ValueError("No vectors provided for weighted binding")
        
        # Convert to numpy
        np_vectors = []
        for v in vectors:
            if isinstance(v, torch.Tensor):
                np_vectors.append(v.detach().cpu().numpy())
            else:
                np_vectors.append(v)
        
        # Default equal weights
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        else:
            # Normalize weights to sum to 1
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
        
        # Weighted combination
        result = np.zeros(self.dimension, dtype=np.float32)
        for vec, weight in zip(np_vectors, weights):
            result += weight * vec
        
        # Optional normalization
        if normalize:
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
        
        return result
    
    def protect_bind(
        self,
        data: Union[np.ndarray, torch.Tensor],
        key: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Protected binding using key-based scrambling.
        
        Args:
            data: Data hypervector to protect
            key: Key hypervector for protection
            
        Returns:
            Protected hypervector
        """
        # Convert to numpy
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if isinstance(key, torch.Tensor):
            key = key.detach().cpu().numpy()
        
        # Multi-layer protection
        # Layer 1: XOR binding
        protected = self.xor_bind(data, key)
        
        # Layer 2: Permutation based on key
        protected = self.permutation_bind(protected, key, perm_type='hash')
        
        # Layer 3: Fourier phase scrambling
        key_fft = fft(key)
        phase_scramble = np.exp(1j * np.angle(key_fft))
        protected_fft = fft(protected) * phase_scramble
        protected = np.real(ifft(protected_fft))
        
        return protected.astype(np.float32)
    
    def unprotect_bind(
        self,
        protected: Union[np.ndarray, torch.Tensor],
        key: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Unprotect a protected hypervector using the key.
        
        Args:
            protected: Protected hypervector
            key: Key hypervector used for protection
            
        Returns:
            Unprotected data hypervector
        """
        # Convert to numpy
        if isinstance(protected, torch.Tensor):
            protected = protected.detach().cpu().numpy()
        if isinstance(key, torch.Tensor):
            key = key.detach().cpu().numpy()
        
        # Reverse Layer 3: Fourier phase unscrambling
        key_fft = fft(key)
        phase_unscramble = np.exp(-1j * np.angle(key_fft))
        unprotected_fft = fft(protected) * phase_unscramble
        unprotected = np.real(ifft(unprotected_fft))
        
        # Reverse Layer 2: Inverse permutation
        # Generate same permutation as in protect
        hash_val = hash(key.tobytes())
        np.random.seed(hash_val % (2**32))
        perm = np.random.permutation(self.dimension)
        np.random.seed(None)
        
        # Create inverse permutation
        inv_perm = np.argsort(perm)
        unprotected = unprotected[inv_perm]
        
        # Reverse Layer 1: XOR unbinding (XOR is its own inverse)
        unprotected = self.xor_bind(unprotected, key)
        
        return unprotected.astype(np.float32)
    
    def similarity(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two hypervectors.
        
        Args:
            a: First hypervector
            b: Second hypervector
            metric: Similarity metric ('cosine', 'hamming', 'euclidean')
            
        Returns:
            Similarity score
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        if metric == 'cosine':
            # Cosine similarity
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        elif metric == 'hamming':
            # Hamming distance (for binary)
            binary_a = a > 0
            binary_b = b > 0
            return 1.0 - np.mean(binary_a != binary_b)
        
        elif metric == 'euclidean':
            # Normalized Euclidean distance
            dist = np.linalg.norm(a - b)
            max_dist = np.linalg.norm(a) + np.linalg.norm(b)
            return 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def composite_bind(
        self,
        vectors: List[Union[np.ndarray, torch.Tensor]],
        operations: List[str]
    ) -> np.ndarray:
        """
        Apply composite binding operations in sequence.
        
        Args:
            vectors: List of hypervectors
            operations: List of operations to apply in sequence
                       e.g., ['xor', 'permute', 'convolve']
            
        Returns:
            Composite-bound hypervector
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors for composite binding")
        
        result = vectors[0]
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        
        for i, (vec, op) in enumerate(zip(vectors[1:], operations)):
            if op == 'xor':
                result = self.xor_bind(result, vec)
            elif op == 'permute':
                result = self.permutation_bind(result, vec)
            elif op == 'convolve':
                result = self.circular_convolve(result, vec)
            elif op == 'fourier':
                result = self.fourier_bind(result, vec)
            elif op == 'protect':
                result = self.protect_bind(result, vec)
            else:
                raise ValueError(f"Unknown operation: {op}")
        
        return result