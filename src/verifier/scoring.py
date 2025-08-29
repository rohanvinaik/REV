import difflib
import re
from typing import List


def normalize(text: str) -> str:
    """
    Normalize text for comparison in REV verification.
    
    Simple normalization that can be extended for domain-specific needs.
    """
    return " ".join(text.strip().lower().split())


def bounded_difference(a: str, b: str) -> float:
    """
    Return a score in [0,1] where 0 ~ identical, 1 ~ very different.
    
    Uses 1 - difflib ratio as a baseline distance metric. This is suitable
    for REV verification but can be replaced with more sophisticated
    measures like token-level or semantic similarity.
    
    Args:
        a: First text (reference model output)
        b: Second text (candidate model output)
        
    Returns:
        Distance score in [0,1] where 0 is identical, 1 is maximally different
    """
    na, nb = normalize(a), normalize(b)
    ratio = difflib.SequenceMatcher(a=na, b=nb).ratio()
    return max(0.0, min(1.0, 1.0 - ratio))


def token_jaccard_distance(a: str, b: str) -> float:
    """
    Token-level Jaccard distance for REV verification.
    
    Computes 1 - Jaccard similarity of token sets.
    Useful for cases where word order matters less than content overlap.
    
    Args:
        a: First text
        b: Second text
        
    Returns:
        Jaccard distance in [0,1]
    """
    # Simple tokenization - can be enhanced with proper tokenizers
    tokens_a = set(normalize(a).split())
    tokens_b = set(normalize(b).split())
    
    if not tokens_a and not tokens_b:
        return 0.0
    
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    
    if union == 0:
        return 0.0
    
    jaccard_sim = intersection / union
    return 1.0 - jaccard_sim


def length_normalized_edit_distance(a: str, b: str) -> float:
    """
    Length-normalized edit distance for REV verification.
    
    Computes edit distance normalized by maximum length to get [0,1] score.
    
    Args:
        a: First text
        b: Second text
        
    Returns:
        Normalized edit distance in [0,1]
    """
    na, nb = normalize(a), normalize(b)
    
    if not na and not nb:
        return 0.0
    
    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 0.0
    
    # Simple character-level edit distance
    # In practice, you might want to use more efficient algorithms
    edit_dist = _compute_edit_distance(na, nb)
    
    return min(1.0, edit_dist / max_len)


def _compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute edit distance using dynamic programming.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    if len(s1) == 0:
        return len(s2)
    
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def semantic_similarity_distance(a: str, b: str) -> float:
    """
    Placeholder for semantic similarity-based distance for REV.
    
    This would integrate with embedding models or semantic similarity APIs
    for more sophisticated comparison suitable for LLM output verification.
    
    Args:
        a: First text
        b: Second text
        
    Returns:
        Semantic distance in [0,1] (currently falls back to bounded_difference)
    """
    # Placeholder - would integrate with embedding models
    # For now, fall back to syntactic similarity
    return bounded_difference(a, b)