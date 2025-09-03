#!/usr/bin/env python3
"""
Fingerprint Comparison Example

Demonstrates how to generate, save, load, and compare model fingerprints.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from run_rev import REVUnified
from src.hdc.unified_fingerprint import UnifiedFingerprintGenerator
from src.hypervector.hamming import HammingDistanceOptimized
from src.hypervector.similarity import AdvancedSimilarity


def generate_fingerprint(model_path: str, save_path: str = None) -> Dict:
    """
    Generate and optionally save a model fingerprint.
    
    Args:
        model_path: Path to model directory
        save_path: Optional path to save fingerprint JSON
    
    Returns:
        Fingerprint dictionary
    """
    print(f"\n🔬 Generating fingerprint for: {model_path}")
    print("-" * 50)
    
    # Initialize REV with unified fingerprints
    rev = REVUnified(
        enable_principled_features=True,
        unified_fingerprints=True,
        fingerprint_dimension=10000,
        fingerprint_sparsity=0.01
    )
    
    try:
        # Generate fingerprint
        result = rev.process_model(model_path, challenges=10)
        
        fingerprint = result.get('fingerprint', {})
        
        print(f"✅ Fingerprint generated successfully")
        print(f"   Dimension: {fingerprint.get('dimension', 'Unknown')}")
        print(f"   Sparsity: {fingerprint.get('sparsity', 'Unknown'):.1%}")
        print(f"   Pathways: {', '.join(fingerprint.get('pathways', []))}")
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(fingerprint, f, indent=2)
            print(f"   Saved to: {save_path}")
        
        return fingerprint
        
    except Exception as e:
        print(f"❌ Failed to generate fingerprint: {e}")
        return None
    
    finally:
        rev.cleanup()


def compare_fingerprints(fp1: Dict, fp2: Dict) -> Dict:
    """
    Compare two fingerprints using multiple metrics.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
    
    Returns:
        Comparison results
    """
    print("\n📊 Comparing Fingerprints")
    print("-" * 50)
    
    # Initialize comparison tools
    hamming = HammingDistanceOptimized()
    similarity = AdvancedSimilarity()
    
    results = {}
    
    # Convert to numpy arrays if needed
    if 'vector' in fp1 and 'vector' in fp2:
        v1 = np.array(fp1['vector'], dtype=np.uint8)
        v2 = np.array(fp2['vector'], dtype=np.uint8)
        
        # Hamming distance
        h_dist = hamming.distance(v1, v2)
        h_sim = 1.0 - (h_dist / len(v1))
        results['hamming_distance'] = h_dist
        results['hamming_similarity'] = h_sim
        
        print(f"  Hamming Distance: {h_dist}/{len(v1)}")
        print(f"  Hamming Similarity: {h_sim:.2%}")
        
        # Jaccard similarity
        j_sim = similarity.jaccard_binary(v1, v2)
        results['jaccard_similarity'] = j_sim
        print(f"  Jaccard Similarity: {j_sim:.2%}")
        
        # Cosine similarity
        c_sim = similarity.cosine_binary(v1, v2)
        results['cosine_similarity'] = c_sim
        print(f"  Cosine Similarity: {c_sim:.2%}")
    
    # Compare pathways
    if 'pathways' in fp1 and 'pathways' in fp2:
        pathways1 = set(fp1['pathways'])
        pathways2 = set(fp2['pathways'])
        common = pathways1 & pathways2
        
        results['common_pathways'] = list(common)
        results['pathway_overlap'] = len(common) / max(len(pathways1), len(pathways2))
        
        print(f"\n  Common Pathways: {', '.join(common)}")
        print(f"  Pathway Overlap: {results['pathway_overlap']:.1%}")
    
    # Overall verdict
    avg_similarity = np.mean([
        results.get('hamming_similarity', 0),
        results.get('jaccard_similarity', 0),
        results.get('cosine_similarity', 0)
    ])
    
    results['average_similarity'] = avg_similarity
    
    print(f"\n  📈 Average Similarity: {avg_similarity:.2%}")
    
    if avg_similarity > 0.85:
        print("  ✅ Models are VERY SIMILAR (likely same family)")
    elif avg_similarity > 0.60:
        print("  ⚠️  Models are SOMEWHAT SIMILAR (possibly related)")
    else:
        print("  ❌ Models are DIFFERENT (different families)")
    
    return results


def load_fingerprint(path: str) -> Dict:
    """Load fingerprint from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def visualize_fingerprint(fingerprint: Dict):
    """
    Visualize fingerprint properties.
    
    Args:
        fingerprint: Fingerprint dictionary
    """
    print("\n🎨 Fingerprint Visualization")
    print("-" * 50)
    
    if 'vector' in fingerprint:
        vector = np.array(fingerprint['vector'], dtype=np.uint8)
        
        # Basic statistics
        print(f"  Vector Statistics:")
        print(f"    Length: {len(vector)}")
        print(f"    Ones: {np.sum(vector)} ({np.mean(vector)*100:.2f}%)")
        print(f"    Zeros: {len(vector) - np.sum(vector)} ({(1-np.mean(vector))*100:.2f}%)")
        
        # Visualize first 500 bits as ASCII art
        print(f"\n  First 500 bits pattern:")
        for i in range(0, min(500, len(vector)), 50):
            row = vector[i:i+50]
            visual = ''.join(['█' if bit else '·' for bit in row])
            print(f"    {i:4d}: {visual}")
    
    # Show pathway contributions if available
    if 'pathway_contributions' in fingerprint:
        print(f"\n  Pathway Contributions:")
        for pathway, weight in fingerprint['pathway_contributions'].items():
            bar = '=' * int(weight * 40)
            print(f"    {pathway:12}: [{bar:<40}] {weight:.1%}")


def batch_comparison_example(model_paths: List[str]):
    """
    Compare multiple models in a batch.
    
    Args:
        model_paths: List of model paths
    """
    print("\n" + "=" * 70)
    print("BATCH FINGERPRINT COMPARISON")
    print("=" * 70)
    
    fingerprints = []
    names = []
    
    # Generate fingerprints for all models
    for path in model_paths:
        name = Path(path).name
        names.append(name)
        
        fp = generate_fingerprint(path)
        if fp:
            fingerprints.append(fp)
        else:
            print(f"⚠️  Skipping {name} due to error")
    
    if len(fingerprints) < 2:
        print("❌ Need at least 2 fingerprints for comparison")
        return
    
    # Create similarity matrix
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    print("\n📊 Pairwise Similarity Matrix:")
    print("-" * 50)
    
    hamming = HammingDistanceOptimized()
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                v1 = np.array(fingerprints[i]['vector'], dtype=np.uint8)
                v2 = np.array(fingerprints[j]['vector'], dtype=np.uint8)
                dist = hamming.distance(v1, v2)
                sim = 1.0 - (dist / len(v1))
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Print matrix
    print("\n" + " " * 15 + "  ".join([f"{name[:12]:>12}" for name in names]))
    for i, name in enumerate(names):
        row = [f"{similarity_matrix[i, j]:.2f}" for j in range(n)]
        print(f"{name[:12]:>12}: " + "        ".join(row))
    
    # Find most/least similar pairs
    print("\n📈 Analysis:")
    
    # Most similar (excluding diagonal)
    mask = ~np.eye(n, dtype=bool)
    max_sim = np.max(similarity_matrix[mask])
    max_idx = np.unravel_index(np.argmax(similarity_matrix * mask), (n, n))
    
    print(f"  Most similar: {names[max_idx[0]]} ↔ {names[max_idx[1]]} "
          f"({max_sim:.2%})")
    
    # Least similar
    min_sim = np.min(similarity_matrix[mask])
    min_idx = np.unravel_index(np.argmin(similarity_matrix + np.eye(n)), (n, n))
    
    print(f"  Least similar: {names[min_idx[0]]} ↔ {names[min_idx[1]]} "
          f"({min_sim:.2%})")
    
    # Average similarity
    avg_sim = np.mean(similarity_matrix[mask])
    print(f"  Average similarity: {avg_sim:.2%}")


def main():
    """Main function with example usage."""
    
    print("=" * 70)
    print("FINGERPRINT COMPARISON EXAMPLES")
    print("=" * 70)
    
    # Example paths (update with your models)
    model1_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    model2_path = "/Users/rohanvinaik/LLM_models/pythia-160m"
    
    # Check if models exist
    if not Path(model1_path).exists():
        print(f"⚠️  Model not found: {model1_path}")
        print("Please update the paths in this script to your model locations")
        
        # Try to find some models
        search_paths = [
            Path.home() / "LLM_models",
            Path.home() / ".cache" / "huggingface" / "hub"
        ]
        
        found_models = []
        for search_path in search_paths:
            if search_path.exists():
                configs = list(search_path.glob("*/config.json")) + \
                         list(search_path.glob("*/snapshots/*/config.json"))
                found_models.extend([str(c.parent) for c in configs[:3]])
        
        if found_models:
            print(f"\n✅ Found {len(found_models)} models to use for demo:")
            for model in found_models:
                print(f"   {model}")
            
            if len(found_models) >= 2:
                model1_path = found_models[0]
                model2_path = found_models[1]
            else:
                print("\n❌ Need at least 2 models for comparison")
                return
        else:
            print("\n❌ No models found for demonstration")
            return
    
    # Example 1: Generate and save fingerprints
    print("\n" + "=" * 70)
    print("Example 1: Generate and Save Fingerprints")
    print("=" * 70)
    
    fp1 = generate_fingerprint(
        model1_path,
        save_path="fingerprints/model1.json"
    )
    
    fp2 = generate_fingerprint(
        model2_path,
        save_path="fingerprints/model2.json"
    )
    
    # Example 2: Compare fingerprints
    if fp1 and fp2:
        print("\n" + "=" * 70)
        print("Example 2: Compare Two Fingerprints")
        print("=" * 70)
        
        comparison = compare_fingerprints(fp1, fp2)
        
        # Save comparison results
        with open("fingerprints/comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        print("\n💾 Comparison saved to fingerprints/comparison.json")
    
    # Example 3: Visualize fingerprint
    if fp1:
        print("\n" + "=" * 70)
        print("Example 3: Visualize Fingerprint")
        print("=" * 70)
        
        visualize_fingerprint(fp1)
    
    # Example 4: Batch comparison (if more models available)
    if len(found_models) >= 3:
        batch_comparison_example(found_models[:4])  # Compare up to 4 models
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()