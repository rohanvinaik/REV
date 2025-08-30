"""
Enhanced Binding Operations for Hyperdimensional Computing.

This module implements advanced binding operations including XOR, permutation,
circular convolution, Fourier-domain binding, and weighted probability binding.
"""

import numpy as np
import torch
from typing import List, Optional, Union, Tuple, Dict, Any
from scipy.fft import fft, ifft, fft2, ifft2
from scipy import signal
import hashlib
import struct
from functools import lru_cache


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
        
        # Multi-resolution permutations for hierarchical binding
        for level in [64, 256, 1024, 4096]:
            if level < self.dimension:
                self._perm_cache[f'hierarchical_{level}'] = self._hierarchical_permutation(level)
    
    def _bit_reversal_permutation(self, n: int) -> np.ndarray:
        """Generate bit-reversal permutation."""
        n_bits = int(np.log2(n))
        indices = np.arange(n)
        reversed_indices = np.zeros(n, dtype=int)
        
        for i in range(n):
            reversed_indices[i] = int(bin(i)[2:].zfill(n_bits)[::-1], 2)
        
        return reversed_indices
    
    def _hierarchical_permutation(self, block_size: int) -> np.ndarray:
        """Generate hierarchical block permutation."""
        perm = np.arange(self.dimension)
        n_blocks = self.dimension // block_size
        
        for i in range(n_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, self.dimension)
            perm[start:end] = np.random.permutation(perm[start:end])
        
        return perm
    
    def _init_lut(self):
        """Initialize lookup tables for fast operations."""
        # 16-bit XOR lookup table
        self._lut_table = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                self._lut_table[i, j] = i ^ j
    
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
    
    def blake2b_bind(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor],
        stable: bool = True
    ) -> np.ndarray:
        """
        BLAKE2b-based stable binding for deterministic operations.
        
        Args:
            a: First hypervector
            b: Second hypervector
            stable: Use stable deterministic binding
            
        Returns:
            BLAKE2b-bound hypervector
        """
        # Convert to numpy
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        
        if stable and self.use_blake2b:
            # Create deterministic binding using BLAKE2b
            combined = np.concatenate([a, b])
            hash_input = combined.tobytes()
            
            # Generate dimension-sized output
            result = np.zeros(self.dimension)
            n_chunks = (self.dimension * 4 + 63) // 64  # 4 bytes per float
            
            for i in range(n_chunks):
                chunk_hash = hashlib.blake2b(
                    hash_input + struct.pack('I', i),
                    digest_size=64
                ).digest()
                
                # Convert hash to floats
                for j in range(0, len(chunk_hash), 4):
                    if i * 16 + j // 4 < self.dimension:
                        value = struct.unpack('f', chunk_hash[j:j+4])[0]
                        result[i * 16 + j // 4] = value
            
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            
            return result
        else:
            # Fallback to XOR binding
            return self.xor_bind(a, b)
    
    @lru_cache(maxsize=256)
    def cached_bind(
        self,
        key_a: str,
        key_b: str,
        operation: str = 'xor'
    ) -> np.ndarray:
        """
        Cached binding operation for frequently used combinations.
        
        Args:
            key_a: Cache key for first vector
            key_b: Cache key for second vector
            operation: Binding operation to use
            
        Returns:
            Cached bound vector
        """
        # Generate vectors from keys using BLAKE2b
        a = self._generate_from_key(key_a)
        b = self._generate_from_key(key_b)
        
        if operation == 'xor':
            return self.xor_bind(a, b)
        elif operation == 'permute':
            return self.permutation_bind(a, b)
        elif operation == 'convolve':
            return self.circular_convolution(a, b)
        elif operation == 'blake2b':
            return self.blake2b_bind(a, b)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _generate_from_key(self, key: str) -> np.ndarray:
        """Generate vector from cache key using BLAKE2b."""
        vector = np.zeros(self.dimension)
        key_bytes = key.encode('utf-8')
        
        # Use BLAKE2b to generate deterministic vector
        n_active = int(self.dimension * 0.1)  # 10% sparsity
        for i in range(n_active):
            pos_hash = hashlib.blake2b(key_bytes + struct.pack('I', i), digest_size=4).digest()
            position = struct.unpack('I', pos_hash)[0] % self.dimension
            
            val_hash = hashlib.blake2b(key_bytes + struct.pack('I', i + n_active), digest_size=4).digest()
            value = struct.unpack('f', val_hash)[0]
            
            vector[position] = value
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def hierarchical_bind(
        self,
        vectors: List[np.ndarray],
        zoom_levels: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Hierarchical binding at multiple zoom levels.
        
        Args:
            vectors: List of vectors to bind
            zoom_levels: Zoom levels to generate
            
        Returns:
            Dictionary of bound vectors at each level
        """
        results = {}
        
        for level in zoom_levels:
            if level == 'corpus':
                # Coarse binding - mean pooling
                results[level] = np.mean(vectors, axis=0)
            elif level == 'prompt':
                # Medium binding - weighted sum
                weights = np.linspace(0.5, 1.0, len(vectors))
                results[level] = np.average(vectors, axis=0, weights=weights)
            elif level == 'span':
                # Fine binding - sequential convolution
                result = vectors[0]
                for v in vectors[1:]:
                    result = self.circular_convolution(result, v)
                results[level] = result
            elif level == 'token_window':
                # Ultra-fine binding - composite operations
                result = vectors[0]
                for i, v in enumerate(vectors[1:]):
                    if i % 2 == 0:
                        result = self.xor_bind(result, v)
                    else:
                        result = self.permutation_bind(result, v)
                results[level] = result
        
        return results
    
    def simd_batch_bind(
        self,
        batch_a: np.ndarray,
        batch_b: np.ndarray,
        operation: str = 'xor'
    ) -> np.ndarray:
        """
        SIMD-optimized batch binding operations.
        
        Args:
            batch_a: Batch of first vectors (N x D)
            batch_b: Batch of second vectors (N x D)
            operation: Binding operation
            
        Returns:
            Batch of bound vectors
        """
        if not self.enable_simd:
            # Fallback to sequential
            results = []
            for a, b in zip(batch_a, batch_b):
                if operation == 'xor':
                    results.append(self.xor_bind(a, b))
                elif operation == 'permute':
                    results.append(self.permutation_bind(a, b))
                else:
                    results.append(self.circular_convolution(a, b))
            return np.array(results)
        
        # SIMD-optimized operations
        if operation == 'xor':
            # Vectorized XOR for continuous values
            sign_a = np.sign(batch_a)
            sign_b = np.sign(batch_b)
            magnitude = np.sqrt(np.abs(batch_a * batch_b))
            return sign_a * sign_b * magnitude
        
        elif operation == 'permute':
            # Vectorized permutation
            perm = self._perm_cache.get('random', np.arange(self.dimension))
            return batch_a * batch_b[:, perm]
        
        elif operation == 'convolve':
            # FFT-based batch convolution
            fft_a = np.fft.fft(batch_a, axis=1)
            fft_b = np.fft.fft(batch_b, axis=1)
            return np.real(np.fft.ifft(fft_a * fft_b, axis=1))
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def privacy_preserving_bind(
        self,
        a: np.ndarray,
        b: np.ndarray,
        noise_scale: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Privacy-preserving binding with differential privacy.
        
        Args:
            a: First hypervector
            b: Second hypervector
            noise_scale: Scale of privacy noise
            
        Returns:
            Bound vector and noise vector
        """
        # Perform binding
        bound = self.xor_bind(a, b)
        
        # Add differential privacy noise
        noise = np.random.laplace(0, noise_scale, size=bound.shape)
        private_bound = bound + noise
        
        # Normalize
        norm = np.linalg.norm(private_bound)
        if norm > 0:
            private_bound = private_bound / norm
        
        return private_bound, noise
    
    def homomorphic_bind(
        self,
        a_encrypted: np.ndarray,
        b_encrypted: np.ndarray,
        modulus: int = 2**16
    ) -> np.ndarray:
        """
        Homomorphic-friendly binding operation.
        
        Args:
            a_encrypted: First encrypted vector
            b_encrypted: Second encrypted vector
            modulus: Modulus for ring operations
            
        Returns:
            Homomorphically bound vector
        """
        # Perform binding in encrypted domain
        # Using addition and multiplication that preserve homomorphic properties
        result = (a_encrypted + b_encrypted) % modulus
        result = (result * result) % modulus  # Square for non-linearity
        
        return result