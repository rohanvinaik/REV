"""
Error Correction and Robustness mechanisms for HDC vectors.

This module implements XOR parity blocks, single-block flip correction,
and noise tolerance mechanisms for robust hyperdimensional computing.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import hashlib
from scipy.special import comb


@dataclass
class ErrorCorrectionConfig:
    """Configuration for error correction."""
    
    dimension: int = 10000
    parity_overhead: float = 0.25  # 25% overhead for parity
    block_size: int = 64  # Size of each block for parity
    max_corrections: int = 2  # Maximum number of corrections per block
    noise_threshold: float = 0.1  # Noise tolerance threshold
    use_hamming_code: bool = True  # Use Hamming codes for stronger correction
    use_interleaving: bool = True  # Use interleaving for burst error protection


class ErrorCorrection:
    """
    Error correction and robustness for hyperdimensional vectors.
    
    Implements:
    - XOR parity blocks with 25% overhead
    - Single and multi-block flip correction
    - Noise tolerance mechanisms
    - Hamming codes for stronger error correction
    - Interleaving for burst error protection
    """
    
    def __init__(self, config: Optional[ErrorCorrectionConfig] = None):
        """
        Initialize error correction system.
        
        Args:
            config: Error correction configuration
        """
        self.config = config or ErrorCorrectionConfig()
        
        # Calculate parity dimensions
        self.data_dimension = self.config.dimension
        self.parity_dimension = int(self.data_dimension * self.config.parity_overhead)
        self.total_dimension = self.data_dimension + self.parity_dimension
        
        # Calculate block parameters
        self.num_blocks = self.data_dimension // self.config.block_size
        self.parity_blocks = self.parity_dimension // self.config.block_size
        
        # Precompute Hamming matrices if needed
        if self.config.use_hamming_code:
            self._init_hamming_matrices()
        
        # Initialize interleaving permutation
        if self.config.use_interleaving:
            self._init_interleaving()
    
    def _init_hamming_matrices(self):
        """Initialize Hamming code matrices."""
        # For Hamming(7,4) code as example
        # Can be extended to other Hamming codes
        self.hamming_g = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        self.hamming_h = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ], dtype=np.uint8)
        
        # Syndrome lookup table for single-bit error correction
        self.syndrome_table = {
            0: -1,  # No error
            1: 6,   # Error in position 6
            2: 5,   # Error in position 5
            3: 3,   # Error in position 3
            4: 4,   # Error in position 4
            5: 1,   # Error in position 1
            6: 2,   # Error in position 2
            7: 0    # Error in position 0
        }
    
    def _init_interleaving(self):
        """Initialize interleaving permutation for burst error protection."""
        # Create pseudo-random but deterministic permutation
        np.random.seed(42)
        self.interleave_perm = np.random.permutation(self.data_dimension)
        self.deinterleave_perm = np.argsort(self.interleave_perm)
        np.random.seed(None)
    
    def encode_with_parity(
        self,
        data: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """
        Encode data with XOR parity blocks.
        
        Args:
            data: Input hypervector
            
        Returns:
            Encoded vector with parity
        """
        if isinstance(data, list):
            data = np.array(data)
        
        # Ensure correct dimension
        if len(data) != self.data_dimension:
            raise ValueError(f"Data must have dimension {self.data_dimension}")
        
        # Apply interleaving if configured
        if self.config.use_interleaving:
            data = data[self.interleave_perm]
        
        # Convert to binary if needed
        is_binary = np.all(np.logical_or(data == 0, data == 1) | 
                          np.logical_or(data == -1, data == 1))
        
        if is_binary:
            binary_data = (data > 0).astype(np.uint8)
        else:
            # Quantize continuous values to binary
            binary_data = (data > np.median(data)).astype(np.uint8)
        
        # Calculate XOR parity blocks
        parity_bits = []
        
        for i in range(self.num_blocks):
            start_idx = i * self.config.block_size
            end_idx = start_idx + self.config.block_size
            block = binary_data[start_idx:end_idx]
            
            if self.config.use_hamming_code and self.config.block_size >= 4:
                # Use Hamming code for this block
                parity = self._hamming_encode_block(block)
            else:
                # Simple XOR parity
                parity = self._xor_parity_block(block)
            
            parity_bits.extend(parity)
        
        # Pad parity to match parity dimension
        parity_array = np.array(parity_bits[:self.parity_dimension])
        if len(parity_array) < self.parity_dimension:
            padding = np.zeros(self.parity_dimension - len(parity_array))
            parity_array = np.concatenate([parity_array, padding])
        
        # Combine data and parity
        if is_binary:
            # Convert back to bipolar if original was bipolar
            encoded_data = binary_data * 2 - 1 if np.any(data == -1) else binary_data
            encoded_parity = parity_array * 2 - 1 if np.any(data == -1) else parity_array
        else:
            # Keep original continuous values for data part
            encoded_data = data[self.deinterleave_perm] if self.config.use_interleaving else data
            encoded_parity = parity_array * 2 - 1  # Bipolar parity
        
        encoded = np.concatenate([encoded_data, encoded_parity])
        
        return encoded
    
    def _xor_parity_block(self, block: np.ndarray) -> List[int]:
        """Calculate XOR parity for a block."""
        # Simple XOR parity
        parity = np.bitwise_xor.reduce(block)
        
        # Additional parity bits for error correction
        # Use different combinations of bits
        parity_bits = [
            parity,
            np.bitwise_xor.reduce(block[::2]),  # Even positions
            np.bitwise_xor.reduce(block[1::2]),  # Odd positions
            np.bitwise_xor.reduce(block[:len(block)//2])  # First half
        ]
        
        return parity_bits
    
    def _hamming_encode_block(self, block: np.ndarray) -> List[int]:
        """Encode block using Hamming code."""
        parity_bits = []
        
        # Process block in chunks suitable for Hamming(7,4)
        for i in range(0, len(block), 4):
            if i + 4 <= len(block):
                data_bits = block[i:i+4]
                # Encode using generator matrix
                codeword = np.dot(data_bits, self.hamming_g) % 2
                # Extract parity bits (last 3 bits of codeword)
                parity_bits.extend(codeword[4:])
            else:
                # Handle remaining bits with simple parity
                remaining = block[i:]
                parity_bits.append(np.bitwise_xor.reduce(remaining))
        
        return parity_bits
    
    def decode_with_correction(
        self,
        encoded: np.ndarray,
        correct_errors: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Decode vector and correct errors using parity.
        
        Args:
            encoded: Encoded vector with parity
            correct_errors: Whether to attempt error correction
            
        Returns:
            Tuple of (corrected data, number of corrections made)
        """
        if len(encoded) != self.total_dimension:
            raise ValueError(f"Encoded vector must have dimension {self.total_dimension}")
        
        # Split data and parity
        data = encoded[:self.data_dimension].copy()
        parity = encoded[self.data_dimension:]
        
        # Convert to binary for correction
        is_binary = np.all(np.logical_or(data == 0, data == 1) | 
                          np.logical_or(data == -1, data == 1))
        
        if is_binary:
            binary_data = (data > 0).astype(np.uint8)
        else:
            binary_data = (data > np.median(data)).astype(np.uint8)
        
        # Apply deinterleaving for error detection
        if self.config.use_interleaving:
            binary_data_interleaved = binary_data[self.interleave_perm]
        else:
            binary_data_interleaved = binary_data
        
        corrections_made = 0
        
        if correct_errors:
            # Check and correct each block
            parity_idx = 0
            
            for i in range(self.num_blocks):
                start_idx = i * self.config.block_size
                end_idx = start_idx + self.config.block_size
                block = binary_data_interleaved[start_idx:end_idx]
                
                if self.config.use_hamming_code and self.config.block_size >= 4:
                    # Use Hamming code correction
                    corrected_block, num_corrections = self._hamming_correct_block(
                        block, parity[parity_idx:parity_idx+3]
                    )
                    parity_idx += 3
                else:
                    # Use XOR parity correction
                    parity_bits = parity[parity_idx:parity_idx+4]
                    corrected_block, num_corrections = self._xor_correct_block(
                        block, parity_bits
                    )
                    parity_idx += 4
                
                if num_corrections > 0:
                    binary_data_interleaved[start_idx:end_idx] = corrected_block
                    corrections_made += num_corrections
        
        # Apply deinterleaving
        if self.config.use_interleaving:
            binary_data = binary_data_interleaved[self.deinterleave_perm]
        else:
            binary_data = binary_data_interleaved
        
        # Convert back to original format
        if is_binary:
            if np.any(encoded < 0):
                # Bipolar
                corrected_data = binary_data * 2.0 - 1.0
            else:
                # Binary
                corrected_data = binary_data.astype(np.float32)
        else:
            # For continuous, just return the data part
            corrected_data = data
        
        return corrected_data, corrections_made
    
    def _xor_correct_block(
        self,
        block: np.ndarray,
        parity_bits: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Correct errors in block using XOR parity."""
        # Calculate expected parity
        expected_parity = [
            np.bitwise_xor.reduce(block),
            np.bitwise_xor.reduce(block[::2]),
            np.bitwise_xor.reduce(block[1::2]),
            np.bitwise_xor.reduce(block[:len(block)//2])
        ]
        
        # Check for errors
        parity_check = [
            int(expected_parity[i]) != int(parity_bits[i] > 0.5)
            for i in range(min(len(expected_parity), len(parity_bits)))
        ]
        
        corrections = 0
        
        if any(parity_check):
            # Simple single-bit error correction
            # This is a simplified version - can be enhanced
            error_pattern = sum(2**i for i, err in enumerate(parity_check) if err)
            
            if error_pattern > 0 and error_pattern <= len(block):
                # Flip the bit at the error position
                error_pos = (error_pattern - 1) % len(block)
                block[error_pos] = 1 - block[error_pos]
                corrections = 1
        
        return block, corrections
    
    def _hamming_correct_block(
        self,
        block: np.ndarray,
        parity_bits: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Correct errors using Hamming code."""
        corrections = 0
        
        # Process in chunks of 4 data bits
        for i in range(0, len(block), 4):
            if i + 4 <= len(block) and len(parity_bits) >= 3:
                data_bits = block[i:i+4]
                
                # Reconstruct codeword
                codeword = np.concatenate([data_bits, parity_bits[:3]])
                
                # Calculate syndrome
                syndrome = np.dot(self.hamming_h, codeword) % 2
                syndrome_val = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2]
                
                # Correct if needed
                if syndrome_val in self.syndrome_table:
                    error_pos = self.syndrome_table[syndrome_val]
                    if error_pos >= 0 and error_pos < 4:
                        # Error in data bits
                        block[i + error_pos] = 1 - block[i + error_pos]
                        corrections += 1
        
        return block, corrections
    
    def add_noise(
        self,
        vector: np.ndarray,
        noise_level: float = 0.1,
        noise_type: str = 'gaussian'
    ) -> np.ndarray:
        """
        Add controlled noise to vector for testing.
        
        Args:
            vector: Input vector
            noise_level: Noise intensity (0 to 1)
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'burst')
            
        Returns:
            Noisy vector
        """
        noisy = vector.copy()
        
        if noise_type == 'gaussian':
            # Additive Gaussian noise
            noise = np.random.randn(len(vector)) * noise_level
            noisy = vector + noise
            
        elif noise_type == 'salt_pepper':
            # Random bit flips
            num_flips = int(len(vector) * noise_level)
            flip_positions = np.random.choice(len(vector), num_flips, replace=False)
            
            if np.all(np.logical_or(vector == 0, vector == 1) | 
                     np.logical_or(vector == -1, vector == 1)):
                # Binary/bipolar vectors
                noisy[flip_positions] = -noisy[flip_positions]
            else:
                # Continuous vectors
                noisy[flip_positions] = np.random.randn(num_flips)
                
        elif noise_type == 'burst':
            # Burst errors (consecutive errors)
            burst_length = max(1, int(len(vector) * noise_level))
            burst_start = np.random.randint(0, len(vector) - burst_length)
            
            if np.all(np.logical_or(vector == 0, vector == 1) | 
                     np.logical_or(vector == -1, vector == 1)):
                # Binary/bipolar vectors
                noisy[burst_start:burst_start + burst_length] = \
                    -noisy[burst_start:burst_start + burst_length]
            else:
                # Continuous vectors
                noisy[burst_start:burst_start + burst_length] = \
                    np.random.randn(burst_length)
        
        return noisy
    
    def measure_robustness(
        self,
        original: np.ndarray,
        noisy: np.ndarray,
        corrected: np.ndarray
    ) -> dict:
        """
        Measure robustness metrics.
        
        Args:
            original: Original vector
            noisy: Noisy vector
            corrected: Corrected vector
            
        Returns:
            Dictionary of robustness metrics
        """
        # Convert to binary for bit error rate
        orig_binary = (original > 0).astype(int)
        noisy_binary = (noisy > 0).astype(int)
        corrected_binary = (corrected > 0).astype(int)
        
        # Bit error rates
        ber_noisy = np.mean(orig_binary != noisy_binary)
        ber_corrected = np.mean(orig_binary != corrected_binary)
        
        # Cosine similarities
        cos_sim_noisy = np.dot(original, noisy) / (
            np.linalg.norm(original) * np.linalg.norm(noisy) + 1e-8
        )
        cos_sim_corrected = np.dot(original, corrected) / (
            np.linalg.norm(original) * np.linalg.norm(corrected) + 1e-8
        )
        
        # Improvement metrics
        ber_improvement = (ber_noisy - ber_corrected) / (ber_noisy + 1e-8)
        sim_improvement = (cos_sim_corrected - cos_sim_noisy) / (1 - cos_sim_noisy + 1e-8)
        
        return {
            'ber_noisy': float(ber_noisy),
            'ber_corrected': float(ber_corrected),
            'cosine_sim_noisy': float(cos_sim_noisy),
            'cosine_sim_corrected': float(cos_sim_corrected),
            'ber_improvement': float(ber_improvement),
            'similarity_improvement': float(sim_improvement),
            'correction_success': float(ber_corrected < ber_noisy)
        }
    
    def adaptive_correction(
        self,
        encoded: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> Tuple[np.ndarray, float]:
        """
        Adaptive error correction based on confidence.
        
        Args:
            encoded: Encoded vector with parity
            confidence_threshold: Minimum confidence for correction
            
        Returns:
            Tuple of (corrected vector, confidence score)
        """
        # Try correction
        corrected, num_corrections = self.decode_with_correction(encoded, True)
        
        # Calculate confidence based on number of corrections
        max_corrections = self.num_blocks * self.config.max_corrections
        confidence = 1.0 - (num_corrections / max_corrections)
        
        if confidence < confidence_threshold:
            # Low confidence - return original without correction
            return encoded[:self.data_dimension], confidence
        
        return corrected, confidence