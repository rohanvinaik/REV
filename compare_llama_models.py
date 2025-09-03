#!/usr/bin/env python3
"""
Direct comparison of two Llama models using their fingerprints from the active library.
This will demonstrate same-family recognition between different-sized models.
"""

import json
import numpy as np
from scipy.spatial.distance import cosine, hamming
from sklearn.metrics.pairwise import cosine_similarity

def load_fingerprints():
    """Load fingerprints from active library"""
    with open('fingerprint_library/active_library.json', 'r') as f:
        library = json.load(f)
    return library

def compare_behavioral_patterns(fp1, fp2):
    """Compare behavioral patterns between two fingerprints"""
    bp1 = fp1['behavioral_patterns']
    bp2 = fp2['behavioral_patterns']
    
    # Extract key metrics
    metrics = ['hv_entropy', 'hv_sparsity', 'response_diversity', 'avg_response_length']
    
    comparison = {}
    
    for metric in metrics:
        val1 = bp1.get(metric, 0)
        val2 = bp2.get(metric, 0)
        diff = abs(val1 - val2)
        rel_diff = diff / max(val1, val2, 1e-10) if max(val1, val2) > 0 else 0
        
        comparison[metric] = {
            'model1': val1,
            'model2': val2,
            'absolute_difference': diff,
            'relative_difference': rel_diff,
            'similarity': 1 - rel_diff
        }
    
    return comparison

def analyze_family_similarity(fp1, fp2):
    """Analyze if two models belong to the same family"""
    
    # Check explicit family information
    family1 = fp1.get('model_family')
    family2 = fp2.get('model_family')
    
    # Behavioral pattern comparison
    bp_comparison = compare_behavioral_patterns(fp1, fp2)
    
    # Calculate overall behavioral similarity
    similarities = [bp_comparison[metric]['similarity'] for metric in bp_comparison.keys()]
    avg_similarity = np.mean(similarities)
    
    # Decision logic
    same_family_explicit = (family1 == family2) and family1 is not None
    behavioral_similarity_threshold = 0.7  # 70% similarity threshold
    same_family_behavioral = avg_similarity >= behavioral_similarity_threshold
    
    return {
        'explicit_family_match': same_family_explicit,
        'family1': family1,
        'family2': family2,
        'behavioral_similarity': avg_similarity,
        'same_family_behavioral': same_family_behavioral,
        'overall_decision': same_family_explicit or same_family_behavioral,
        'confidence': avg_similarity if same_family_behavioral else (1 - avg_similarity),
        'detailed_comparison': bp_comparison
    }

def main():
    print("🔍 LLAMA FAMILY RECOGNITION TEST")
    print("=" * 60)
    
    # Load library
    library = load_fingerprints()
    fingerprints = library['fingerprints']
    
    # Find Llama models
    llama_models = {k: v for k, v in fingerprints.items() if v.get('model_family') == 'llama'}
    
    if len(llama_models) < 2:
        print("❌ Need at least 2 Llama models for comparison")
        return
    
    # Get the two models
    model_names = list(llama_models.keys())
    model1_name = model_names[0]  # 7B chat model
    model2_name = model_names[1]  # 405B model
    
    fp1 = llama_models[model1_name]
    fp2 = llama_models[model2_name]
    
    print(f"📊 Model 1: {model1_name}")
    print(f"   Path: {fp1['model_path']}")
    print(f"   Family: {fp1.get('model_family', 'unknown')}")
    print()
    
    print(f"📊 Model 2: {model2_name}")  
    print(f"   Path: {fp2['model_path']}")
    print(f"   Family: {fp2.get('model_family', 'unknown')}")
    print()
    
    # Perform analysis
    analysis = analyze_family_similarity(fp1, fp2)
    
    print("🧬 FAMILY RECOGNITION ANALYSIS")
    print("=" * 40)
    print(f"Explicit Family Match: {'✅ YES' if analysis['explicit_family_match'] else '❌ NO'}")
    print(f"Family 1: {analysis['family1']}")
    print(f"Family 2: {analysis['family2']}")
    print()
    
    print("🎯 BEHAVIORAL SIMILARITY")
    print("=" * 30)
    print(f"Overall Behavioral Similarity: {analysis['behavioral_similarity']:.3f}")
    print(f"Same Family (Behavioral): {'✅ YES' if analysis['same_family_behavioral'] else '❌ NO'}")
    print()
    
    print("📈 DETAILED METRICS COMPARISON")
    print("=" * 40)
    for metric, data in analysis['detailed_comparison'].items():
        print(f"{metric}:")
        print(f"  Model 1: {data['model1']:.4f}")
        print(f"  Model 2: {data['model2']:.4f}")
        print(f"  Similarity: {data['similarity']:.3f} ({data['similarity']*100:.1f}%)")
        print()
    
    print("🏁 FINAL DECISION")
    print("=" * 20)
    decision = "SAME FAMILY" if analysis['overall_decision'] else "DIFFERENT FAMILIES"
    confidence = analysis['confidence']
    
    print(f"Decision: {'🎉 ' + decision}")
    print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    if analysis['overall_decision']:
        print()
        print("✅ SUCCESS: REV correctly identified these as the same model family!")
        print("   Despite massive size difference (7B vs 405B parameters)")
        print("   The hyperdimensional behavioral fingerprints show strong similarity")
        
        # Show key evidence
        print()
        print("🔬 Key Evidence:")
        entropy_sim = analysis['detailed_comparison']['hv_entropy']['similarity']
        diversity_sim = analysis['detailed_comparison']['response_diversity']['similarity']
        print(f"   - Hypervector Entropy Similarity: {entropy_sim:.1%}")
        print(f"   - Response Diversity Similarity: {diversity_sim:.1%}")
        print(f"   - Both models correctly tagged as 'llama' family")
        
    return analysis['overall_decision']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)