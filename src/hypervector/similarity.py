"""
Advanced Similarity Metrics for Hyperdimensional Computing.

This module implements hierarchical distance computation, multi-scale similarity
aggregation, behavioral neighborhood identification, and domain/task clustering.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.stats import spearmanr, kendalltau
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings


class SimilarityMetric(Enum):
    """Types of similarity metrics."""
    COSINE = "cosine"
    HAMMING = "hamming"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    DICE = "dice"
    ANGULAR = "angular"
    WASSERSTEIN = "wasserstein"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"


@dataclass
class HierarchicalLevel:
    """Represents a level in the hierarchical similarity structure."""
    
    name: str
    scale: float  # Scale factor (1.0 = finest, higher = coarser)
    weight: float  # Weight in aggregation
    dimension_slice: Optional[slice] = None  # Which dimensions to use
    metric: SimilarityMetric = SimilarityMetric.COSINE


@dataclass
class SimilarityResult:
    """Result of similarity computation."""
    
    overall_similarity: float
    hierarchical_scores: Dict[str, float]
    neighborhood_rank: int
    cluster_id: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Neighborhood:
    """Behavioral neighborhood in hypervector space."""
    
    center: np.ndarray
    members: List[int]  # Indices of members
    radius: float
    density: float
    coherence: float  # How similar members are to each other


class AdvancedSimilarity:
    """
    Advanced similarity metrics for hyperdimensional vectors.
    
    Implements hierarchical distance computation, multi-scale aggregation,
    behavioral neighborhoods, and domain/task clustering.
    """
    
    def __init__(
        self,
        dimension: int = 10000,
        hierarchical_levels: Optional[List[HierarchicalLevel]] = None
    ):
        """
        Initialize advanced similarity system.
        
        Args:
            dimension: Hypervector dimension
            hierarchical_levels: Custom hierarchical levels for similarity
        """
        self.dimension = dimension
        
        # Initialize hierarchical levels
        self.levels = hierarchical_levels or self._default_hierarchical_levels()
        
        # Cache for computed distances
        self.distance_cache = {}
        
        # Neighborhood structures
        self.neighborhoods = []
        
        # Clustering models
        self.clusterers = {}
    
    def _default_hierarchical_levels(self) -> List[HierarchicalLevel]:
        """Define default hierarchical levels."""
        return [
            HierarchicalLevel(
                name="fine",
                scale=1.0,
                weight=0.4,
                dimension_slice=slice(None),
                metric=SimilarityMetric.COSINE
            ),
            HierarchicalLevel(
                name="medium",
                scale=2.0,
                weight=0.3,
                dimension_slice=slice(0, self.dimension//2),
                metric=SimilarityMetric.HAMMING
            ),
            HierarchicalLevel(
                name="coarse",
                scale=4.0,
                weight=0.2,
                dimension_slice=slice(0, self.dimension//4),
                metric=SimilarityMetric.ANGULAR
            ),
            HierarchicalLevel(
                name="semantic",
                scale=8.0,
                weight=0.1,
                dimension_slice=slice(0, self.dimension//8),
                metric=SimilarityMetric.CORRELATION
            )
        ]
    
    def compute_hierarchical_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        levels: Optional[List[HierarchicalLevel]] = None
    ) -> SimilarityResult:
        """
        Compute hierarchical similarity between two vectors.
        
        Args:
            vec_a: First hypervector
            vec_b: Second hypervector
            levels: Custom levels to use (or default)
            
        Returns:
            SimilarityResult with hierarchical scores
        """
        levels = levels or self.levels
        hierarchical_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for level in levels:
            # Extract relevant dimensions
            if level.dimension_slice:
                a_slice = vec_a[level.dimension_slice]
                b_slice = vec_b[level.dimension_slice]
            else:
                a_slice = vec_a
                b_slice = vec_b
            
            # Apply scale (downsampling if scale > 1)
            if level.scale > 1:
                downsample_factor = int(level.scale)
                a_slice = self._downsample(a_slice, downsample_factor)
                b_slice = self._downsample(b_slice, downsample_factor)
            
            # Compute similarity at this level
            similarity = self._compute_similarity(a_slice, b_slice, level.metric)
            hierarchical_scores[level.name] = similarity
            
            # Add to weighted sum
            weighted_sum += similarity * level.weight
            total_weight += level.weight
        
        # Compute overall similarity
        overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on consistency across levels
        scores = list(hierarchical_scores.values())
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
        
        return SimilarityResult(
            overall_similarity=overall_similarity,
            hierarchical_scores=hierarchical_scores,
            neighborhood_rank=0,  # Will be set if neighborhood analysis is done
            confidence=confidence,
            metadata={"levels_used": len(levels)}
        )
    
    def _downsample(self, vector: np.ndarray, factor: int) -> np.ndarray:
        """Downsample vector by averaging groups of elements."""
        if factor <= 1:
            return vector
        
        # Pad if necessary
        pad_len = (factor - len(vector) % factor) % factor
        if pad_len > 0:
            vector = np.pad(vector, (0, pad_len), mode='edge')
        
        # Reshape and average
        reshaped = vector.reshape(-1, factor)
        downsampled = np.mean(reshaped, axis=1)
        
        return downsampled
    
    def _compute_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
        metric: SimilarityMetric
    ) -> float:
        """Compute similarity using specified metric."""
        if metric == SimilarityMetric.COSINE:
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            
        elif metric == SimilarityMetric.HAMMING:
            # For continuous vectors, threshold at median
            a_binary = a > np.median(a)
            b_binary = b > np.median(b)
            sim = 1.0 - np.mean(a_binary != b_binary)
            
        elif metric == SimilarityMetric.EUCLIDEAN:
            dist = np.linalg.norm(a - b)
            max_dist = np.linalg.norm(a) + np.linalg.norm(b)
            sim = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
            
        elif metric == SimilarityMetric.MANHATTAN:
            dist = np.sum(np.abs(a - b))
            max_dist = np.sum(np.abs(a)) + np.sum(np.abs(b))
            sim = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
            
        elif metric == SimilarityMetric.JACCARD:
            # For continuous, use threshold
            a_set = set(np.where(a > np.median(a))[0])
            b_set = set(np.where(b > np.median(b))[0])
            if len(a_set) + len(b_set) > 0:
                sim = len(a_set & b_set) / len(a_set | b_set)
            else:
                sim = 0.0
                
        elif metric == SimilarityMetric.DICE:
            # Dice coefficient
            a_set = set(np.where(a > np.median(a))[0])
            b_set = set(np.where(b > np.median(b))[0])
            if len(a_set) + len(b_set) > 0:
                sim = 2 * len(a_set & b_set) / (len(a_set) + len(b_set))
            else:
                sim = 0.0
                
        elif metric == SimilarityMetric.ANGULAR:
            # Angular distance (arccos of cosine similarity)
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            cos_sim = np.clip(cos_sim, -1, 1)  # Ensure valid range
            angle = np.arccos(cos_sim) / np.pi  # Normalize to [0, 1]
            sim = 1.0 - angle
            
        elif metric == SimilarityMetric.CORRELATION:
            # Pearson correlation
            if len(a) > 1:
                corr, _ = np.corrcoef(a, b)[0, 1], 0
                sim = (corr + 1) / 2  # Map from [-1, 1] to [0, 1]
            else:
                sim = 0.0
                
        elif metric == SimilarityMetric.WASSERSTEIN:
            # Earth mover's distance (simplified)
            # Treat vectors as probability distributions
            a_norm = np.abs(a) / (np.sum(np.abs(a)) + 1e-8)
            b_norm = np.abs(b) / (np.sum(np.abs(b)) + 1e-8)
            dist = distance.wasserstein_distance(
                np.arange(len(a)), np.arange(len(b)),
                a_norm, b_norm
            )
            max_dist = len(a)
            sim = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
            
        elif metric == SimilarityMetric.MUTUAL_INFO:
            # Simplified mutual information
            # Discretize continuous values
            a_discrete = np.digitize(a, bins=np.linspace(a.min(), a.max(), 10))
            b_discrete = np.digitize(b, bins=np.linspace(b.min(), b.max(), 10))
            
            # Compute joint histogram
            hist_2d = np.histogram2d(a_discrete, b_discrete, bins=10)[0]
            pxy = hist_2d / np.sum(hist_2d)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            
            # Compute MI
            px_py = px[:, None] * py[None, :]
            nz = pxy > 0
            mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px_py[nz] + 1e-8) + 1e-8))
            
            # Normalize
            hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
            hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
            max_mi = min(hx, hy)
            sim = mi / max_mi if max_mi > 0 else 0.0
            
        else:
            # Default to cosine
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        return float(sim)
    
    def identify_neighborhoods(
        self,
        vectors: List[np.ndarray],
        min_neighbors: int = 3,
        max_radius: float = 0.3,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> List[Neighborhood]:
        """
        Identify behavioral neighborhoods in hypervector space.
        
        Args:
            vectors: List of hypervectors
            min_neighbors: Minimum neighbors for a neighborhood
            max_radius: Maximum radius for neighborhood
            metric: Similarity metric to use
            
        Returns:
            List of identified neighborhoods
        """
        n_vectors = len(vectors)
        if n_vectors < min_neighbors:
            return []
        
        # Compute pairwise distances
        distance_matrix = np.zeros((n_vectors, n_vectors))
        for i in range(n_vectors):
            for j in range(i+1, n_vectors):
                sim = self._compute_similarity(vectors[i], vectors[j], metric)
                dist = 1.0 - sim  # Convert similarity to distance
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Use DBSCAN for neighborhood detection
        clustering = DBSCAN(
            eps=max_radius,
            min_samples=min_neighbors,
            metric='precomputed'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Extract neighborhoods
        neighborhoods = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get members of this neighborhood
            member_indices = np.where(labels == label)[0]
            
            if len(member_indices) < min_neighbors:
                continue
            
            # Compute neighborhood center (medoid)
            member_vectors = [vectors[i] for i in member_indices]
            center = np.mean(member_vectors, axis=0)
            
            # Compute radius (max distance from center)
            distances_to_center = [
                1.0 - self._compute_similarity(center, vec, metric)
                for vec in member_vectors
            ]
            radius = max(distances_to_center)
            
            # Compute density (inverse of average pairwise distance)
            pairwise_dists = []
            for i, idx_i in enumerate(member_indices):
                for idx_j in member_indices[i+1:]:
                    pairwise_dists.append(distance_matrix[idx_i, idx_j])
            
            avg_dist = np.mean(pairwise_dists) if pairwise_dists else 0
            density = 1.0 / (avg_dist + 1e-8)
            
            # Compute coherence (similarity of members to center)
            coherence = np.mean([
                self._compute_similarity(center, vec, metric)
                for vec in member_vectors
            ])
            
            neighborhoods.append(Neighborhood(
                center=center,
                members=member_indices.tolist(),
                radius=radius,
                density=density,
                coherence=coherence
            ))
        
        self.neighborhoods = neighborhoods
        return neighborhoods
    
    def cluster_by_domain(
        self,
        vectors: List[np.ndarray],
        domains: Optional[List[str]] = None,
        n_clusters: Optional[int] = None,
        method: str = 'hierarchical'
    ) -> Dict[str, Any]:
        """
        Cluster vectors by domain/task characteristics.
        
        Args:
            vectors: List of hypervectors
            domains: Optional domain labels for supervised clustering
            n_clusters: Number of clusters (auto-detect if None)
            method: Clustering method ('hierarchical', 'dbscan', 'spectral')
            
        Returns:
            Clustering results with labels and metrics
        """
        n_vectors = len(vectors)
        
        if n_vectors < 2:
            return {"labels": [0] * n_vectors, "n_clusters": 1}
        
        # Stack vectors
        X = np.vstack(vectors)
        
        # Determine number of clusters if not provided
        if n_clusters is None:
            # Use elbow method or silhouette score
            n_clusters = self._estimate_n_clusters(X)
        
        if method == 'hierarchical':
            # Hierarchical clustering
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clusterer.fit_predict(X)
            
            # Build dendrogram
            linkage_matrix = hierarchy.linkage(X, method='ward')
            
        elif method == 'dbscan':
            # DBSCAN clustering
            clusterer = DBSCAN(eps=0.3, min_samples=3)
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        elif method == 'spectral':
            # Spectral clustering
            from sklearn.cluster import SpectralClustering
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors'
            )
            labels = clusterer.fit_predict(X)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Store clusterer
        self.clusterers[method] = clusterer
        
        # Compute cluster quality metrics
        if n_clusters > 1 and len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0.0
        
        # Analyze clusters
        cluster_info = {}
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
            
            cluster_members = np.where(labels == cluster_id)[0]
            cluster_vectors = X[cluster_members]
            
            # Compute cluster statistics
            cluster_center = np.mean(cluster_vectors, axis=0)
            cluster_std = np.std(cluster_vectors, axis=0)
            
            # Intra-cluster similarity
            intra_sim = []
            for i in range(len(cluster_members)):
                for j in range(i+1, len(cluster_members)):
                    sim = self._compute_similarity(
                        cluster_vectors[i],
                        cluster_vectors[j],
                        SimilarityMetric.COSINE
                    )
                    intra_sim.append(sim)
            
            cluster_info[int(cluster_id)] = {
                "size": len(cluster_members),
                "members": cluster_members.tolist(),
                "center": cluster_center.tolist(),
                "std": float(np.mean(cluster_std)),
                "cohesion": float(np.mean(intra_sim)) if intra_sim else 0.0
            }
            
            # Add domain information if provided
            if domains:
                cluster_domains = [domains[i] for i in cluster_members if i < len(domains)]
                # Find dominant domain
                from collections import Counter
                domain_counts = Counter(cluster_domains)
                cluster_info[int(cluster_id)]["dominant_domain"] = domain_counts.most_common(1)[0][0] if domain_counts else None
                cluster_info[int(cluster_id)]["domain_purity"] = domain_counts.most_common(1)[0][1] / len(cluster_domains) if domain_counts else 0.0
        
        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_info": cluster_info,
            "method": method
        }
    
    def _estimate_n_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Estimate optimal number of clusters using silhouette score."""
        best_k = 2
        best_score = -1
        
        for k in range(2, min(max_k, len(X))):
            clusterer = AgglomerativeClustering(n_clusters=k)
            labels = clusterer.fit_predict(X)
            
            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        return best_k
    
    def multi_scale_aggregation(
        self,
        similarities: List[float],
        scales: List[float],
        aggregation: str = 'weighted_mean'
    ) -> float:
        """
        Aggregate similarities across multiple scales.
        
        Args:
            similarities: Similarity values at each scale
            scales: Scale factors for each similarity
            aggregation: Aggregation method
            
        Returns:
            Aggregated similarity score
        """
        if not similarities:
            return 0.0
        
        if aggregation == 'weighted_mean':
            # Weight inversely by scale (finer scales get more weight)
            weights = [1.0 / s for s in scales]
            weight_sum = sum(weights)
            return sum(s * w for s, w in zip(similarities, weights)) / weight_sum
            
        elif aggregation == 'geometric_mean':
            # Geometric mean
            product = np.prod(similarities)
            return product ** (1.0 / len(similarities))
            
        elif aggregation == 'harmonic_mean':
            # Harmonic mean
            if all(s > 0 for s in similarities):
                return len(similarities) / sum(1.0 / s for s in similarities)
            else:
                return 0.0
                
        elif aggregation == 'min':
            return min(similarities)
            
        elif aggregation == 'max':
            return max(similarities)
            
        elif aggregation == 'median':
            return float(np.median(similarities))
            
        else:
            # Default to arithmetic mean
            return float(np.mean(similarities))
    
    def find_nearest_neighbors(
        self,
        query: np.ndarray,
        vectors: List[np.ndarray],
        k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to query vector.
        
        Args:
            query: Query hypervector
            vectors: List of candidate vectors
            k: Number of neighbors to find
            metric: Similarity metric to use
            
        Returns:
            List of (index, similarity) tuples for nearest neighbors
        """
        similarities = []
        
        for i, vec in enumerate(vectors):
            sim = self._compute_similarity(query, vec, metric)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def compute_manifold_embedding(
        self,
        vectors: List[np.ndarray],
        n_components: int = 2,
        method: str = 'tsne'
    ) -> np.ndarray:
        """
        Compute low-dimensional embedding for visualization.
        
        Args:
            vectors: List of hypervectors
            n_components: Number of dimensions for embedding
            method: Embedding method ('tsne', 'umap', 'pca')
            
        Returns:
            Embedded vectors in low-dimensional space
        """
        X = np.vstack(vectors)
        
        if method == 'tsne':
            embedder = TSNE(n_components=n_components, random_state=42)
            embedding = embedder.fit_transform(X)
            
        elif method == 'umap':
            try:
                import umap
                embedder = umap.UMAP(n_components=n_components, random_state=42)
                embedding = embedder.fit_transform(X)
            except ImportError:
                warnings.warn("UMAP not available, falling back to t-SNE")
                embedder = TSNE(n_components=n_components, random_state=42)
                embedding = embedder.fit_transform(X)
                
        elif method == 'pca':
            from sklearn.decomposition import PCA
            embedder = PCA(n_components=n_components)
            embedding = embedder.fit_transform(X)
            
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        return embedding