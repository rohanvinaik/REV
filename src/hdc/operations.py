"""Core HDC operations for REV verification."""

from typing import List
import torch


def bind_vectors(vectors: List[torch.Tensor], method: str = "multiply") -> torch.Tensor:
    """
    Bind multiple hypervectors using specified method.
    
    Args:
        vectors: List of hypervectors to bind
        method: Binding method ("multiply", "circular", "xor")
        
    Returns:
        Bound hypervector
    """
    if not vectors:
        raise ValueError("No vectors provided")
    
    result = vectors[0].clone()
    
    for vector in vectors[1:]:
        if method == "multiply":
            result = result * vector
        elif method == "xor":
            # XOR binding for binary vectors
            result = ((result > 0) ^ (vector > 0)).float() * 2 - 1
        elif method == "circular":
            # Circular convolution via FFT
            result_fft = torch.fft.fft(result)
            vector_fft = torch.fft.fft(vector)
            result = torch.fft.ifft(result_fft * vector_fft).real
        else:
            raise ValueError(f"Unknown binding method: {method}")
    
    return result


def bundle_vectors(vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    """
    Bundle (superpose) multiple hypervectors.
    
    Args:
        vectors: List of hypervectors to bundle
        normalize: Whether to normalize the result
        
    Returns:
        Bundled hypervector
    """
    if not vectors:
        raise ValueError("No vectors provided")
    
    result = torch.stack(vectors).sum(dim=0)
    
    if normalize:
        norm = torch.norm(result)
        if norm > 0:
            result = result / norm
    
    return result


def circular_shift(vector: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Apply circular shift to hypervector.
    
    Args:
        vector: Input hypervector
        shift: Number of positions to shift
        
    Returns:
        Circularly shifted hypervector
    """
    return torch.roll(vector, shifts=shift)


def hamming_distance(v1: torch.Tensor, v2: torch.Tensor) -> int:
    """
    Compute Hamming distance between binary hypervectors.
    
    Args:
        v1: First binary hypervector
        v2: Second binary hypervector
        
    Returns:
        Hamming distance
    """
    # Binarize if not already binary
    bin_v1 = (v1 > 0).long()
    bin_v2 = (v2 > 0).long()
    
    return int(torch.sum(bin_v1 != bin_v2))


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Compute cosine similarity between hypervectors.
    
    Args:
        v1: First hypervector
        v2: Second hypervector
        
    Returns:
        Cosine similarity [-1, 1]
    """
    return float(torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))


def permute_vector(vector: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Apply permutation to hypervector.
    
    Args:
        vector: Input hypervector
        permutation: Permutation indices
        
    Returns:
        Permuted hypervector
    """
    return vector[permutation]


def generate_random_hypervector(
    dimension: int, 
    seed: int = 42, 
    binary: bool = False
) -> torch.Tensor:
    """
    Generate random hypervector for REV.
    
    Args:
        dimension: Vector dimensionality
        seed: Random seed for reproducibility
        binary: Whether to generate binary vector
        
    Returns:
        Random hypervector
    """
    torch.manual_seed(seed)
    
    if binary:
        return torch.randint(0, 2, (dimension,)).float() * 2 - 1
    else:
        vector = torch.randn(dimension)
        return vector / torch.norm(vector)  # Normalize