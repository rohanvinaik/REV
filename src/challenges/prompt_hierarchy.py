#!/usr/bin/env python3
"""
Hierarchical Prompt Organization System for REV Pipeline

This module provides a sophisticated tree-based organization system for managing
and selecting prompts with taxonomic structure, intelligent navigation, and
adaptive reorganization capabilities.
"""

import os
import json
import time
import uuid
import heapq
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from abc import ABC, abstractmethod
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class NavigationMode(Enum):
    """Navigation modes for hierarchy traversal"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    SIMILARITY_GUIDED = "similarity_guided"
    EFFECTIVENESS_ORDERED = "effectiveness_ordered"
    RANDOM_WALK = "random_walk"


class OrganizationStrategy(Enum):
    """Strategies for hierarchy reorganization"""
    USAGE_BASED = "usage_based"
    EFFECTIVENESS_BASED = "effectiveness_based"
    SIMILARITY_CLUSTERING = "similarity_clustering"
    BALANCED_TREE = "balanced_tree"
    HYBRID = "hybrid"


class QueryOperator(Enum):
    """Query operators for compound queries"""
    AND = "and"
    OR = "or"
    NOT = "not"
    NEAR = "near"
    FUZZY = "fuzzy"


@dataclass
class PromptNode:
    """Node in the hierarchical prompt organization tree"""
    node_id: str
    name: str
    description: str
    node_type: str  # 'category', 'subcategory', 'template', 'prompt'
    
    # Hierarchical relationships
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    
    # Content and metadata
    content: Optional[str] = None
    template_data: Dict[str, Any] = field(default_factory=dict)
    
    # Classification facets
    domain: Optional[str] = None
    difficulty: int = 1
    purpose: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    # Cross-cutting concerns (multiple inheritance)
    concerns: Set[str] = field(default_factory=set)  # e.g., 'security', 'ethics', 'performance'
    
    # Usage and effectiveness metrics
    usage_count: int = 0
    effectiveness_score: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    
    # Navigation aids
    similarity_cache: Dict[str, float] = field(default_factory=dict)
    feature_vector: Optional[np.ndarray] = None
    
    # Adaptive properties
    promotion_score: float = 0.0
    split_threshold: int = 20  # Max children before considering split
    merge_threshold: int = 2   # Min children before considering merge
    
    def add_child(self, child_id: str) -> None:
        """Add child node ID"""
        self.children_ids.add(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """Remove child node ID"""
        self.children_ids.discard(child_id)
    
    def update_effectiveness(self, score: float, alpha: float = 0.1) -> None:
        """Update effectiveness score with exponential moving average"""
        self.effectiveness_score = (1 - alpha) * self.effectiveness_score + alpha * score
    
    def record_access(self) -> None:
        """Record access for usage tracking"""
        self.usage_count += 1
        self.last_accessed = time.time()
    
    def calculate_promotion_score(self) -> float:
        """Calculate score for promotion/demotion decisions"""
        usage_factor = min(self.usage_count / 100.0, 1.0)  # Normalize usage
        effectiveness_factor = self.effectiveness_score
        recency_factor = 1.0 / (1.0 + (time.time() - self.last_accessed) / 86400)  # Days
        
        self.promotion_score = (
            usage_factor * 0.4 +
            effectiveness_factor * 0.4 +
            recency_factor * 0.2
        )
        return self.promotion_score
    
    def needs_split(self) -> bool:
        """Check if node needs to be split"""
        return len(self.children_ids) > self.split_threshold and self.node_type in ['category', 'subcategory']
    
    def needs_merge(self) -> bool:
        """Check if node needs to be merged"""
        return len(self.children_ids) < self.merge_threshold and self.parent_id is not None


@dataclass 
class QueryCriterion:
    """Single criterion in a compound query"""
    field: str  # 'content', 'domain', 'difficulty', 'tags', etc.
    operator: str  # '=', '!=', '>', '<', 'contains', 'fuzzy', etc.
    value: Any
    weight: float = 1.0
    fuzzy_threshold: float = 0.8  # For fuzzy matching


@dataclass
class CompoundQuery:
    """Complex query with multiple criteria and operators"""
    criteria: List[QueryCriterion]
    logical_operators: List[QueryOperator]  # Between criteria
    result_limit: int = 100
    sort_by: str = 'relevance'  # 'relevance', 'effectiveness', 'usage', 'recent'
    include_children: bool = False
    max_depth: int = -1  # -1 for unlimited


@dataclass
class NavigationPath:
    """Path through the hierarchy with context"""
    nodes: List[str]  # Node IDs in path
    total_distance: float
    path_type: NavigationMode
    branch_points: List[int]  # Indices where branching occurred
    dead_ends: List[str]  # Node IDs of dead ends encountered
    effectiveness_sum: float = 0.0


class PromptTaxonomy:
    """Tree-based hierarchical organization of prompts with multiple inheritance"""
    
    def __init__(self, name: str = "REV Prompt Taxonomy"):
        self.name = name
        self.root_id = str(uuid.uuid4())
        
        # Core data structures
        self.nodes: Dict[str, PromptNode] = {}
        
        # Cross-cutting concern mappings (multiple inheritance)
        self.concern_mappings: Dict[str, Set[str]] = defaultdict(set)  # concern -> node_ids
        
        # Faceted classification indices
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.difficulty_index: Dict[int, Set[str]] = defaultdict(set)
        self.purpose_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Feature extraction and similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_vectorized = False
        
        # Adaptive reorganization
        self.reorganization_strategy = OrganizationStrategy.HYBRID
        self.last_reorganization = time.time()
        self.reorganization_interval = 3600  # 1 hour
        
        # Performance caches
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.path_cache: Dict[Tuple[str, str, NavigationMode], NavigationPath] = {}
        
        # Statistics
        self.access_statistics: Dict[str, int] = defaultdict(int)
        self.query_statistics: Dict[str, int] = defaultdict(int)
        
        # Create root node
        self._create_root()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _create_root(self) -> None:
        """Create root node of the taxonomy"""
        root = PromptNode(
            node_id=self.root_id,
            name="Root",
            description="Root of the prompt taxonomy",
            node_type="root",
            domain="universal",
            purpose="organization"
        )
        self.nodes[self.root_id] = root
    
    def add_node(self, node: PromptNode, parent_id: Optional[str] = None) -> str:
        """Add node to taxonomy with automatic indexing"""
        with self._lock:
            if parent_id is None:
                parent_id = self.root_id
            
            # Validate parent exists
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node {parent_id} not found")
            
            # Add to nodes
            self.nodes[node.node_id] = node
            node.parent_id = parent_id
            
            # Update parent
            parent = self.nodes[parent_id]
            parent.add_child(node.node_id)
            
            # Update indices
            self._update_indices(node)
            
            # Update concern mappings
            for concern in node.concerns:
                self.concern_mappings[concern].add(node.node_id)
            
            # Invalidate relevant caches
            self._invalidate_caches_for_node(node.node_id)
            
            return node.node_id
    
    def remove_node(self, node_id: str, cascade: bool = False) -> bool:
        """Remove node from taxonomy"""
        with self._lock:
            if node_id not in self.nodes or node_id == self.root_id:
                return False
            
            node = self.nodes[node_id]
            
            # Handle children
            if node.children_ids and not cascade:
                raise ValueError(f"Node {node_id} has children. Use cascade=True to remove recursively.")
            
            if cascade:
                # Remove all children recursively
                for child_id in list(node.children_ids):
                    self.remove_node(child_id, cascade=True)
            
            # Remove from parent
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                parent.remove_child(node_id)
            
            # Remove from indices
            self._remove_from_indices(node)
            
            # Remove from concern mappings
            for concern in node.concerns:
                self.concern_mappings[concern].discard(node_id)
            
            # Remove from nodes
            del self.nodes[node_id]
            
            # Invalidate caches
            self._invalidate_caches_for_node(node_id)
            
            return True
    
    def move_node(self, node_id: str, new_parent_id: str) -> bool:
        """Move node to new parent"""
        with self._lock:
            if node_id not in self.nodes or new_parent_id not in self.nodes:
                return False
            
            if node_id == self.root_id:
                raise ValueError("Cannot move root node")
            
            # Prevent cycles
            if self._would_create_cycle(node_id, new_parent_id):
                raise ValueError("Move would create cycle")
            
            node = self.nodes[node_id]
            
            # Remove from old parent
            if node.parent_id and node.parent_id in self.nodes:
                old_parent = self.nodes[node.parent_id]
                old_parent.remove_child(node_id)
            
            # Add to new parent
            new_parent = self.nodes[new_parent_id]
            new_parent.add_child(node_id)
            node.parent_id = new_parent_id
            
            # Invalidate caches
            self._invalidate_caches_for_node(node_id)
            
            return True
    
    def add_concern_mapping(self, concern: str, node_ids: List[str]) -> None:
        """Add cross-cutting concern mapping (multiple inheritance)"""
        with self._lock:
            for node_id in node_ids:
                if node_id in self.nodes:
                    self.nodes[node_id].concerns.add(concern)
                    self.concern_mappings[concern].add(node_id)
    
    def get_nodes_by_concern(self, concern: str) -> List[PromptNode]:
        """Get all nodes with specific concern"""
        node_ids = self.concern_mappings.get(concern, set())
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def navigate(self, start_id: str, mode: NavigationMode, 
                target_id: Optional[str] = None, max_steps: int = 100) -> NavigationPath:
        """Navigate through hierarchy using specified mode"""
        
        # Check cache first
        cache_key = (start_id, target_id or "", mode)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if start_id not in self.nodes:
            raise ValueError(f"Start node {start_id} not found")
        
        path = None
        
        if mode == NavigationMode.BREADTH_FIRST:
            path = self._breadth_first_navigation(start_id, target_id, max_steps)
        elif mode == NavigationMode.DEPTH_FIRST:
            path = self._depth_first_navigation(start_id, target_id, max_steps)
        elif mode == NavigationMode.SIMILARITY_GUIDED:
            path = self._similarity_guided_navigation(start_id, target_id, max_steps)
        elif mode == NavigationMode.EFFECTIVENESS_ORDERED:
            path = self._effectiveness_ordered_navigation(start_id, target_id, max_steps)
        elif mode == NavigationMode.RANDOM_WALK:
            path = self._random_walk_navigation(start_id, max_steps)
        else:
            raise ValueError(f"Unknown navigation mode: {mode}")
        
        # Cache result
        if len(self.path_cache) < 1000:  # Limit cache size
            self.path_cache[cache_key] = path
        
        return path
    
    def _breadth_first_navigation(self, start_id: str, target_id: Optional[str], max_steps: int) -> NavigationPath:
        """Breadth-first traversal of the hierarchy"""
        queue = deque([(start_id, 0, [start_id])])
        visited = {start_id}
        branch_points = []
        dead_ends = []
        
        while queue and len(visited) < max_steps:
            current_id, depth, path = queue.popleft()
            current = self.nodes[current_id]
            current.record_access()
            
            if target_id and current_id == target_id:
                return NavigationPath(
                    nodes=path,
                    total_distance=depth,
                    path_type=NavigationMode.BREADTH_FIRST,
                    branch_points=branch_points,
                    dead_ends=dead_ends
                )
            
            # Add children to queue
            children_added = 0
            for child_id in current.children_ids:
                if child_id not in visited and child_id in self.nodes:
                    queue.append((child_id, depth + 1, path + [child_id]))
                    visited.add(child_id)
                    children_added += 1
            
            # Track branch points and dead ends
            if children_added > 1:
                branch_points.append(len(path) - 1)
            elif children_added == 0 and current.node_type not in ['prompt', 'template']:
                dead_ends.append(current_id)
        
        # Return path through all visited nodes
        visited_list = list(visited)
        return NavigationPath(
            nodes=visited_list,
            total_distance=len(visited_list),
            path_type=NavigationMode.BREADTH_FIRST,
            branch_points=branch_points,
            dead_ends=dead_ends
        )
    
    def _depth_first_navigation(self, start_id: str, target_id: Optional[str], max_steps: int) -> NavigationPath:
        """Depth-first traversal of the hierarchy"""
        stack = [(start_id, 0, [start_id])]
        visited = {start_id}
        branch_points = []
        dead_ends = []
        
        while stack and len(visited) < max_steps:
            current_id, depth, path = stack.pop()
            current = self.nodes[current_id]
            current.record_access()
            
            if target_id and current_id == target_id:
                return NavigationPath(
                    nodes=path,
                    total_distance=depth,
                    path_type=NavigationMode.DEPTH_FIRST,
                    branch_points=branch_points,
                    dead_ends=dead_ends
                )
            
            # Add children to stack (reverse order for proper DFS)
            children = [child_id for child_id in current.children_ids 
                       if child_id not in visited and child_id in self.nodes]
            
            if len(children) > 1:
                branch_points.append(len(path) - 1)
            elif len(children) == 0 and current.node_type not in ['prompt', 'template']:
                dead_ends.append(current_id)
            
            for child_id in reversed(children):
                stack.append((child_id, depth + 1, path + [child_id]))
                visited.add(child_id)
        
        visited_list = list(visited)
        return NavigationPath(
            nodes=visited_list,
            total_distance=len(visited_list),
            path_type=NavigationMode.DEPTH_FIRST,
            branch_points=branch_points,
            dead_ends=dead_ends
        )
    
    def _similarity_guided_navigation(self, start_id: str, target_id: Optional[str], max_steps: int) -> NavigationPath:
        """Navigate using similarity between nodes"""
        if not self.is_vectorized:
            self._vectorize_nodes()
        
        current_id = start_id
        path = [start_id]
        visited = {start_id}
        total_distance = 0.0
        
        target_vector = None
        if target_id and target_id in self.nodes:
            target_vector = self.nodes[target_id].feature_vector
        
        while len(path) < max_steps:
            current = self.nodes[current_id]
            current.record_access()
            
            if target_id and current_id == target_id:
                break
            
            # Find most similar unvisited neighbor
            best_child = None
            best_similarity = -1.0
            
            # Consider children
            for child_id in current.children_ids:
                if child_id not in visited and child_id in self.nodes:
                    child = self.nodes[child_id]
                    
                    if target_vector is not None and child.feature_vector is not None:
                        similarity = 1 - cosine(target_vector, child.feature_vector)
                    else:
                        # Use content similarity
                        similarity = self._calculate_content_similarity(current_id, child_id)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_child = child_id
            
            # Consider siblings and parent if no good children
            if best_similarity < 0.3 and current.parent_id:
                parent = self.nodes[current.parent_id]
                for sibling_id in parent.children_ids:
                    if sibling_id not in visited and sibling_id in self.nodes:
                        similarity = self._calculate_content_similarity(current_id, sibling_id)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_child = sibling_id
            
            if best_child is None:
                break  # No more nodes to visit
            
            path.append(best_child)
            visited.add(best_child)
            total_distance += (1 - best_similarity)
            current_id = best_child
        
        return NavigationPath(
            nodes=path,
            total_distance=total_distance,
            path_type=NavigationMode.SIMILARITY_GUIDED,
            branch_points=[],
            dead_ends=[]
        )
    
    def _effectiveness_ordered_navigation(self, start_id: str, target_id: Optional[str], max_steps: int) -> NavigationPath:
        """Navigate prioritizing most effective nodes"""
        current_id = start_id
        path = [start_id]
        visited = {start_id}
        effectiveness_sum = 0.0
        
        while len(path) < max_steps:
            current = self.nodes[current_id]
            current.record_access()
            effectiveness_sum += current.effectiveness_score
            
            if target_id and current_id == target_id:
                break
            
            # Find most effective unvisited child
            candidates = []
            for child_id in current.children_ids:
                if child_id not in visited and child_id in self.nodes:
                    child = self.nodes[child_id]
                    candidates.append((child_id, child.effectiveness_score))
            
            if not candidates:
                # Try to move up and explore siblings
                if current.parent_id and current.parent_id not in visited:
                    parent = self.nodes[current.parent_id]
                    for sibling_id in parent.children_ids:
                        if sibling_id not in visited and sibling_id in self.nodes:
                            sibling = self.nodes[sibling_id]
                            candidates.append((sibling_id, sibling.effectiveness_score))
            
            if not candidates:
                break
            
            # Select most effective candidate
            best_id = max(candidates, key=lambda x: x[1])[0]
            path.append(best_id)
            visited.add(best_id)
            current_id = best_id
        
        return NavigationPath(
            nodes=path,
            total_distance=len(path),
            path_type=NavigationMode.EFFECTIVENESS_ORDERED,
            branch_points=[],
            dead_ends=[],
            effectiveness_sum=effectiveness_sum
        )
    
    def _random_walk_navigation(self, start_id: str, max_steps: int) -> NavigationPath:
        """Random walk through the hierarchy"""
        import random
        
        current_id = start_id
        path = [start_id]
        visited = {start_id}
        
        while len(path) < max_steps:
            current = self.nodes[current_id]
            current.record_access()
            
            # Get all possible next nodes (children, siblings, parent)
            candidates = []
            
            # Children
            for child_id in current.children_ids:
                if child_id in self.nodes:
                    candidates.append(child_id)
            
            # Siblings
            if current.parent_id and current.parent_id in self.nodes:
                parent = self.nodes[current.parent_id]
                for sibling_id in parent.children_ids:
                    if sibling_id != current_id and sibling_id in self.nodes:
                        candidates.append(sibling_id)
                
                # Parent itself
                candidates.append(current.parent_id)
            
            if not candidates:
                break
            
            # Randomly select next node
            next_id = random.choice(candidates)
            path.append(next_id)
            visited.add(next_id)
            current_id = next_id
        
        return NavigationPath(
            nodes=path,
            total_distance=len(path),
            path_type=NavigationMode.RANDOM_WALK,
            branch_points=[],
            dead_ends=[]
        )
    
    def jump_to_similar(self, source_id: str, target_content: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Jump to similar nodes across hierarchy branches"""
        if source_id not in self.nodes:
            return []
        
        if not self.is_vectorized:
            self._vectorize_nodes()
        
        # Vectorize target content
        target_vector = self.vectorizer.transform([target_content]).toarray()[0]
        
        # Find similar nodes
        similarities = []
        for node_id, node in self.nodes.items():
            if node_id != source_id and node.feature_vector is not None:
                similarity = 1 - cosine(target_vector, node.feature_vector)
                if similarity >= threshold:
                    similarities.append((node_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def query(self, query: CompoundQuery) -> List[Tuple[str, float]]:
        """Execute compound query with ranking"""
        with self._lock:
            results = []
            
            for node_id, node in self.nodes.items():
                score = self._evaluate_query(node, query)
                if score > 0:
                    results.append((node_id, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            if query.result_limit > 0:
                results = results[:query.result_limit]
            
            # Update query statistics
            query_key = f"{len(query.criteria)}_{query.sort_by}"
            self.query_statistics[query_key] += 1
            
            return results
    
    def fuzzy_search(self, text: str, fields: List[str] = None, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Fuzzy text search across specified fields"""
        if fields is None:
            fields = ['name', 'description', 'content']
        
        results = []
        
        for node_id, node in self.nodes.items():
            max_similarity = 0.0
            
            for field in fields:
                field_value = getattr(node, field, '')
                if field_value:
                    similarity = self._fuzzy_match(text.lower(), field_value.lower())
                    max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                results.append((node_id, max_similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def reorganize(self, strategy: OrganizationStrategy = None) -> Dict[str, Any]:
        """Reorganize hierarchy using specified strategy"""
        with self._lock:
            if strategy is None:
                strategy = self.reorganization_strategy
            
            stats = {'nodes_moved': 0, 'nodes_split': 0, 'nodes_merged': 0}
            
            if strategy == OrganizationStrategy.USAGE_BASED:
                stats = self._reorganize_by_usage()
            elif strategy == OrganizationStrategy.EFFECTIVENESS_BASED:
                stats = self._reorganize_by_effectiveness()
            elif strategy == OrganizationStrategy.SIMILARITY_CLUSTERING:
                stats = self._reorganize_by_similarity()
            elif strategy == OrganizationStrategy.BALANCED_TREE:
                stats = self._reorganize_for_balance()
            elif strategy == OrganizationStrategy.HYBRID:
                stats = self._reorganize_hybrid()
            
            self.last_reorganization = time.time()
            
            # Clear caches after reorganization
            self.path_cache.clear()
            self.similarity_cache.clear()
            
            return stats
    
    def auto_split_node(self, node_id: str) -> List[str]:
        """Automatically split node based on child clustering"""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        if not node.needs_split():
            return []
        
        # Get child nodes
        children = [self.nodes[child_id] for child_id in node.children_ids 
                   if child_id in self.nodes]
        
        if len(children) < 4:  # Need minimum children for clustering
            return []
        
        # Vectorize children for clustering
        if not self.is_vectorized:
            self._vectorize_nodes()
        
        # Extract features
        features = []
        valid_children = []
        
        for child in children:
            if child.feature_vector is not None:
                features.append(child.feature_vector)
                valid_children.append(child)
        
        if len(features) < 4:
            return []
        
        # Cluster children
        n_clusters = min(3, len(features) // 3)  # At least 3 children per cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create new subcategory nodes
        new_node_ids = []
        clusters = defaultdict(list)
        
        for child, label in zip(valid_children, cluster_labels):
            clusters[label].append(child)
        
        for cluster_id, cluster_children in clusters.items():
            if len(cluster_children) >= 2:  # Only create subcategory if it has enough children
                # Create subcategory node
                subcategory_id = str(uuid.uuid4())
                subcategory_name = f"{node.name}_Cluster_{cluster_id}"
                
                subcategory = PromptNode(
                    node_id=subcategory_id,
                    name=subcategory_name,
                    description=f"Auto-generated subcategory from {node.name}",
                    node_type="subcategory",
                    domain=node.domain,
                    purpose=node.purpose
                )
                
                # Add subcategory to taxonomy
                self.add_node(subcategory, parent_id=node_id)
                
                # Move cluster children to subcategory
                for child in cluster_children:
                    self.move_node(child.node_id, subcategory_id)
                
                new_node_ids.append(subcategory_id)
        
        return new_node_ids
    
    def auto_merge_node(self, node_id: str) -> bool:
        """Automatically merge node with parent if appropriate"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if not node.needs_merge() or not node.parent_id:
            return False
        
        parent = self.nodes[node.parent_id]
        
        # Move all children to parent
        for child_id in list(node.children_ids):
            self.move_node(child_id, parent.node_id)
        
        # Remove the node
        self.remove_node(node_id)
        return True
    
    def promote_node(self, node_id: str) -> bool:
        """Promote node up one level in hierarchy"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if not node.parent_id:
            return False  # Already at root level
        
        parent = self.nodes[node.parent_id]
        if not parent.parent_id:
            return False  # Parent is root, can't promote further
        
        # Move node to grandparent
        return self.move_node(node_id, parent.parent_id)
    
    def demote_node(self, node_id: str, new_parent_id: str) -> bool:
        """Demote node down one level in hierarchy"""
        if node_id not in self.nodes or new_parent_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        new_parent = self.nodes[new_parent_id]
        
        # Ensure new parent is a child of current parent (demotion)
        if node.parent_id != new_parent.parent_id:
            return False
        
        return self.move_node(node_id, new_parent_id)
    
    def rebalance(self) -> Dict[str, int]:
        """Rebalance tree for optimal navigation"""
        stats = {'promotions': 0, 'demotions': 0, 'splits': 0, 'merges': 0}
        
        # Calculate promotion scores
        for node in self.nodes.values():
            node.calculate_promotion_score()
        
        # Handle splits and merges
        nodes_to_split = [node_id for node_id, node in self.nodes.items() if node.needs_split()]
        for node_id in nodes_to_split:
            splits = self.auto_split_node(node_id)
            stats['splits'] += len(splits)
        
        nodes_to_merge = [node_id for node_id, node in self.nodes.items() if node.needs_merge()]
        for node_id in nodes_to_merge:
            if self.auto_merge_node(node_id):
                stats['merges'] += 1
        
        # Handle promotions/demotions based on scores
        promotion_candidates = [(node_id, node.promotion_score) 
                               for node_id, node in self.nodes.items() 
                               if node.promotion_score > 0.8 and node.parent_id]
        promotion_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for node_id, score in promotion_candidates[:5]:  # Limit promotions
            if self.promote_node(node_id):
                stats['promotions'] += 1
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive taxonomy statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'depth': self._calculate_max_depth(),
            'branching_factor': self._calculate_avg_branching_factor(),
            'node_types': Counter(node.node_type for node in self.nodes.values()),
            'domains': Counter(node.domain for node in self.nodes.values() if node.domain),
            'concerns': dict(self.concern_mappings),
            'most_accessed': self._get_most_accessed_nodes(10),
            'most_effective': self._get_most_effective_nodes(10),
            'query_patterns': dict(self.query_statistics),
            'cache_stats': {
                'similarity_cache_size': len(self.similarity_cache),
                'path_cache_size': len(self.path_cache),
                'cache_hit_rate': self._calculate_cache_hit_rate()
            }
        }
        
        return stats
    
    def export_to_json(self, filepath: str) -> bool:
        """Export taxonomy to JSON file"""
        try:
            export_data = {
                'name': self.name,
                'root_id': self.root_id,
                'nodes': {},
                'concern_mappings': {k: list(v) for k, v in self.concern_mappings.items()},
                'metadata': {
                    'created': time.time(),
                    'version': '1.0',
                    'node_count': len(self.nodes)
                }
            }
            
            # Export nodes (excluding feature vectors)
            for node_id, node in self.nodes.items():
                export_data['nodes'][node_id] = {
                    'node_id': node.node_id,
                    'name': node.name,
                    'description': node.description,
                    'node_type': node.node_type,
                    'parent_id': node.parent_id,
                    'children_ids': list(node.children_ids),
                    'content': node.content,
                    'template_data': node.template_data,
                    'domain': node.domain,
                    'difficulty': node.difficulty,
                    'purpose': node.purpose,
                    'tags': list(node.tags),
                    'concerns': list(node.concerns),
                    'usage_count': node.usage_count,
                    'effectiveness_score': node.effectiveness_score,
                    'last_accessed': node.last_accessed,
                    'creation_time': node.creation_time,
                    'promotion_score': node.promotion_score
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """Import taxonomy from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Clear current taxonomy
            self.nodes.clear()
            self.concern_mappings.clear()
            self._clear_indices()
            
            # Import basic info
            self.name = import_data['name']
            self.root_id = import_data['root_id']
            
            # Import nodes
            for node_id, node_data in import_data['nodes'].items():
                node = PromptNode(
                    node_id=node_data['node_id'],
                    name=node_data['name'],
                    description=node_data['description'],
                    node_type=node_data['node_type'],
                    parent_id=node_data.get('parent_id'),
                    content=node_data.get('content'),
                    template_data=node_data.get('template_data', {}),
                    domain=node_data.get('domain'),
                    difficulty=node_data.get('difficulty', 1),
                    purpose=node_data.get('purpose'),
                    tags=set(node_data.get('tags', [])),
                    concerns=set(node_data.get('concerns', [])),
                    usage_count=node_data.get('usage_count', 0),
                    effectiveness_score=node_data.get('effectiveness_score', 0.0),
                    last_accessed=node_data.get('last_accessed', time.time()),
                    creation_time=node_data.get('creation_time', time.time()),
                    promotion_score=node_data.get('promotion_score', 0.0)
                )
                
                node.children_ids = set(node_data.get('children_ids', []))
                self.nodes[node_id] = node
            
            # Rebuild indices
            self._rebuild_indices()
            
            # Import concern mappings
            if 'concern_mappings' in import_data:
                for concern, node_ids in import_data['concern_mappings'].items():
                    self.concern_mappings[concern] = set(node_ids)
            
            # Clear caches
            self.similarity_cache.clear()
            self.path_cache.clear()
            
            return True
            
        except Exception as e:
            print(f"Import failed: {e}")
            return False
    
    # Helper methods
    
    def _update_indices(self, node: PromptNode) -> None:
        """Update faceted classification indices"""
        if node.domain:
            self.domain_index[node.domain].add(node.node_id)
        
        self.difficulty_index[node.difficulty].add(node.node_id)
        
        if node.purpose:
            self.purpose_index[node.purpose].add(node.node_id)
        
        for tag in node.tags:
            self.tag_index[tag].add(node.node_id)
    
    def _remove_from_indices(self, node: PromptNode) -> None:
        """Remove node from faceted classification indices"""
        if node.domain:
            self.domain_index[node.domain].discard(node.node_id)
        
        self.difficulty_index[node.difficulty].discard(node.node_id)
        
        if node.purpose:
            self.purpose_index[node.purpose].discard(node.node_id)
        
        for tag in node.tags:
            self.tag_index[tag].discard(node.node_id)
    
    def _clear_indices(self) -> None:
        """Clear all indices"""
        self.domain_index.clear()
        self.difficulty_index.clear()
        self.purpose_index.clear()
        self.tag_index.clear()
    
    def _rebuild_indices(self) -> None:
        """Rebuild all indices from current nodes"""
        self._clear_indices()
        for node in self.nodes.values():
            self._update_indices(node)
    
    def _would_create_cycle(self, node_id: str, new_parent_id: str) -> bool:
        """Check if move would create a cycle"""
        current = new_parent_id
        while current:
            if current == node_id:
                return True
            current = self.nodes.get(current, PromptNode('', '', '', '')).parent_id
        return False
    
    def _invalidate_caches_for_node(self, node_id: str) -> None:
        """Invalidate relevant caches when node changes"""
        # Remove similarity cache entries
        keys_to_remove = []
        for key in self.similarity_cache:
            if node_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.similarity_cache[key]
        
        # Remove path cache entries
        path_keys_to_remove = []
        for key in self.path_cache:
            if node_id in key[:2]:  # start_id or target_id
                path_keys_to_remove.append(key)
        
        for key in path_keys_to_remove:
            del self.path_cache[key]
    
    def _vectorize_nodes(self) -> None:
        """Create feature vectors for all nodes with content"""
        texts = []
        node_ids = []
        
        for node_id, node in self.nodes.items():
            if node.content:
                texts.append(node.content)
                node_ids.append(node_id)
        
        if texts:
            vectors = self.vectorizer.fit_transform(texts).toarray()
            for node_id, vector in zip(node_ids, vectors):
                self.nodes[node_id].feature_vector = vector
        
        self.is_vectorized = True
    
    def _calculate_content_similarity(self, node1_id: str, node2_id: str) -> float:
        """Calculate content similarity between two nodes"""
        cache_key = tuple(sorted([node1_id, node2_id]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Use feature vectors if available
        if node1.feature_vector is not None and node2.feature_vector is not None:
            similarity = 1 - cosine(node1.feature_vector, node2.feature_vector)
        else:
            # Fall back to simple text similarity
            text1 = (node1.content or '') + ' ' + (node1.description or '')
            text2 = (node2.content or '') + ' ' + (node2.description or '')
            similarity = self._fuzzy_match(text1.lower(), text2.lower())
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Calculate fuzzy match score between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _evaluate_query(self, node: PromptNode, query: CompoundQuery) -> float:
        """Evaluate how well a node matches a compound query"""
        if not query.criteria:
            return 0.0
        
        scores = []
        
        for criterion in query.criteria:
            score = self._evaluate_criterion(node, criterion)
            scores.append(score * criterion.weight)
        
        # Combine scores based on logical operators
        if len(scores) == 1:
            return scores[0]
        
        final_score = scores[0]
        for i, operator in enumerate(query.logical_operators):
            if i + 1 < len(scores):
                if operator == QueryOperator.AND:
                    final_score = min(final_score, scores[i + 1])
                elif operator == QueryOperator.OR:
                    final_score = max(final_score, scores[i + 1])
                elif operator == QueryOperator.NOT:
                    final_score = final_score * (1 - scores[i + 1])
        
        return max(0.0, min(1.0, final_score))
    
    def _evaluate_criterion(self, node: PromptNode, criterion: QueryCriterion) -> float:
        """Evaluate single criterion against node"""
        field_value = getattr(node, criterion.field, None)
        
        if field_value is None:
            return 0.0
        
        if criterion.operator == '=':
            return 1.0 if field_value == criterion.value else 0.0
        elif criterion.operator == '!=':
            return 0.0 if field_value == criterion.value else 1.0
        elif criterion.operator == '>':
            return 1.0 if field_value > criterion.value else 0.0
        elif criterion.operator == '<':
            return 1.0 if field_value < criterion.value else 0.0
        elif criterion.operator == 'contains':
            if isinstance(field_value, str):
                return 1.0 if criterion.value.lower() in field_value.lower() else 0.0
            elif isinstance(field_value, (set, list)):
                return 1.0 if criterion.value in field_value else 0.0
        elif criterion.operator == 'fuzzy':
            if isinstance(field_value, str):
                similarity = self._fuzzy_match(str(criterion.value).lower(), field_value.lower())
                return 1.0 if similarity >= criterion.fuzzy_threshold else 0.0
        
        return 0.0
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the taxonomy tree"""
        max_depth = 0
        
        def dfs_depth(node_id: str, depth: int) -> int:
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            node = self.nodes.get(node_id)
            if node:
                for child_id in node.children_ids:
                    dfs_depth(child_id, depth + 1)
            
            return max_depth
        
        return dfs_depth(self.root_id, 0)
    
    def _calculate_avg_branching_factor(self) -> float:
        """Calculate average branching factor"""
        total_children = 0
        internal_nodes = 0
        
        for node in self.nodes.values():
            if node.children_ids:
                total_children += len(node.children_ids)
                internal_nodes += 1
        
        return total_children / internal_nodes if internal_nodes > 0 else 0.0
    
    def _get_most_accessed_nodes(self, limit: int) -> List[Tuple[str, int]]:
        """Get most accessed nodes"""
        nodes_usage = [(node_id, node.usage_count) 
                      for node_id, node in self.nodes.items()]
        nodes_usage.sort(key=lambda x: x[1], reverse=True)
        return nodes_usage[:limit]
    
    def _get_most_effective_nodes(self, limit: int) -> List[Tuple[str, float]]:
        """Get most effective nodes"""
        nodes_effectiveness = [(node_id, node.effectiveness_score) 
                              for node_id, node in self.nodes.items()]
        nodes_effectiveness.sort(key=lambda x: x[1], reverse=True)
        return nodes_effectiveness[:limit]
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified estimate)"""
        total_operations = sum(self.access_statistics.values()) + sum(self.query_statistics.values())
        cache_size = len(self.similarity_cache) + len(self.path_cache)
        
        if total_operations == 0:
            return 0.0
        
        # Estimate hit rate based on cache size relative to operations
        estimated_hits = min(cache_size * 2, total_operations)  # Rough estimate
        return estimated_hits / total_operations
    
    # Reorganization strategies
    
    def _reorganize_by_usage(self) -> Dict[str, int]:
        """Reorganize based on usage patterns"""
        stats = {'nodes_moved': 0}
        
        # Promote frequently used nodes
        usage_threshold = np.percentile([node.usage_count for node in self.nodes.values()], 75)
        
        for node in self.nodes.values():
            if (node.usage_count > usage_threshold and 
                node.parent_id and node.parent_id != self.root_id):
                if self.promote_node(node.node_id):
                    stats['nodes_moved'] += 1
        
        return stats
    
    def _reorganize_by_effectiveness(self) -> Dict[str, int]:
        """Reorganize based on effectiveness scores"""
        stats = {'nodes_moved': 0}
        
        # Promote highly effective nodes
        effectiveness_threshold = np.percentile([node.effectiveness_score for node in self.nodes.values()], 80)
        
        for node in self.nodes.values():
            if (node.effectiveness_score > effectiveness_threshold and 
                node.parent_id and node.parent_id != self.root_id):
                if self.promote_node(node.node_id):
                    stats['nodes_moved'] += 1
        
        return stats
    
    def _reorganize_by_similarity(self) -> Dict[str, int]:
        """Reorganize based on content similarity clustering"""
        if not self.is_vectorized:
            self._vectorize_nodes()
        
        stats = {'nodes_moved': 0}
        
        # Find nodes that might be better grouped together
        # This is a simplified implementation
        nodes_with_vectors = [node for node in self.nodes.values() 
                             if node.feature_vector is not None]
        
        if len(nodes_with_vectors) < 4:
            return stats
        
        # Cluster nodes
        vectors = np.array([node.feature_vector for node in nodes_with_vectors])
        n_clusters = min(5, len(vectors) // 3)
        
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Group nodes by cluster
            clusters = defaultdict(list)
            for node, label in zip(nodes_with_vectors, cluster_labels):
                clusters[label].append(node)
            
            # For each cluster, try to group nodes under common parent
            for cluster_nodes in clusters.values():
                if len(cluster_nodes) >= 3:
                    # Find most common parent
                    parents = [node.parent_id for node in cluster_nodes if node.parent_id]
                    if parents:
                        common_parent = max(set(parents), key=parents.count)
                        
                        # Move nodes to common parent if they're not already there
                        for node in cluster_nodes:
                            if (node.parent_id != common_parent and 
                                common_parent in self.nodes):
                                if self.move_node(node.node_id, common_parent):
                                    stats['nodes_moved'] += 1
        
        return stats
    
    def _reorganize_for_balance(self) -> Dict[str, int]:
        """Reorganize for balanced tree structure"""
        return self.rebalance()
    
    def _reorganize_hybrid(self) -> Dict[str, int]:
        """Hybrid reorganization strategy"""
        stats = {'nodes_moved': 0, 'nodes_split': 0, 'nodes_merged': 0}
        
        # Apply multiple strategies with weights
        usage_stats = self._reorganize_by_usage()
        effectiveness_stats = self._reorganize_by_effectiveness()
        balance_stats = self._reorganize_for_balance()
        
        # Combine stats
        stats['nodes_moved'] += usage_stats.get('nodes_moved', 0)
        stats['nodes_moved'] += effectiveness_stats.get('nodes_moved', 0)
        stats['nodes_moved'] += balance_stats.get('promotions', 0)
        stats['nodes_moved'] += balance_stats.get('demotions', 0)
        stats['nodes_split'] += balance_stats.get('splits', 0)
        stats['nodes_merged'] += balance_stats.get('merges', 0)
        
        return stats


class HierarchicalQueryBuilder:
    """Builder for constructing complex hierarchical queries"""
    
    def __init__(self):
        self.criteria: List[QueryCriterion] = []
        self.operators: List[QueryOperator] = []
        self.result_limit = 100
        self.sort_by = 'relevance'
    
    def where(self, field: str, operator: str, value: Any, weight: float = 1.0) -> 'HierarchicalQueryBuilder':
        """Add WHERE criterion"""
        criterion = QueryCriterion(field=field, operator=operator, value=value, weight=weight)
        self.criteria.append(criterion)
        return self
    
    def fuzzy_match(self, field: str, value: str, threshold: float = 0.8, weight: float = 1.0) -> 'HierarchicalQueryBuilder':
        """Add fuzzy matching criterion"""
        criterion = QueryCriterion(
            field=field, 
            operator='fuzzy', 
            value=value, 
            weight=weight,
            fuzzy_threshold=threshold
        )
        self.criteria.append(criterion)
        return self
    
    def and_where(self, field: str, operator: str, value: Any, weight: float = 1.0) -> 'HierarchicalQueryBuilder':
        """Add AND criterion"""
        if self.criteria:
            self.operators.append(QueryOperator.AND)
        return self.where(field, operator, value, weight)
    
    def or_where(self, field: str, operator: str, value: Any, weight: float = 1.0) -> 'HierarchicalQueryBuilder':
        """Add OR criterion"""
        if self.criteria:
            self.operators.append(QueryOperator.OR)
        return self.where(field, operator, value, weight)
    
    def not_where(self, field: str, operator: str, value: Any, weight: float = 1.0) -> 'HierarchicalQueryBuilder':
        """Add NOT criterion"""
        if self.criteria:
            self.operators.append(QueryOperator.NOT)
        return self.where(field, operator, value, weight)
    
    def limit(self, count: int) -> 'HierarchicalQueryBuilder':
        """Set result limit"""
        self.result_limit = count
        return self
    
    def order_by(self, field: str) -> 'HierarchicalQueryBuilder':
        """Set ordering field"""
        self.sort_by = field
        return self
    
    def build(self) -> CompoundQuery:
        """Build the compound query"""
        return CompoundQuery(
            criteria=self.criteria,
            logical_operators=self.operators,
            result_limit=self.result_limit,
            sort_by=self.sort_by
        )


# Integration with EnhancedKDFPromptGenerator
class HierarchicalTemplateSelector:
    """Selector that integrates hierarchy with EnhancedKDFPromptGenerator"""
    
    def __init__(self, taxonomy: PromptTaxonomy):
        self.taxonomy = taxonomy
        self.selection_history: List[str] = []
        self.effectiveness_feedback: Dict[str, List[float]] = defaultdict(list)
        
    def select_templates_hierarchically(self, 
                                      requirements: Dict[str, Any],
                                      count: int = 10,
                                      navigation_mode: NavigationMode = NavigationMode.EFFECTIVENESS_ORDERED,
                                      diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
        """Select templates using hierarchical navigation"""
        
        # Build query from requirements
        query_builder = HierarchicalQueryBuilder()
        
        if 'domain' in requirements:
            query_builder.where('domain', '=', requirements['domain'])
        
        if 'difficulty_min' in requirements:
            query_builder.and_where('difficulty', '>', requirements['difficulty_min'])
        
        if 'difficulty_max' in requirements:
            query_builder.and_where('difficulty', '<', requirements['difficulty_max'])
        
        if 'purpose' in requirements:
            query_builder.and_where('purpose', '=', requirements['purpose'])
        
        if 'tags' in requirements:
            for tag in requirements['tags']:
                query_builder.and_where('tags', 'contains', tag)
        
        query = query_builder.limit(count * 2).build()  # Get more candidates for diversity
        
        # Execute query
        candidates = self.taxonomy.query(query)
        
        # Select diverse subset using navigation
        selected_templates = []
        used_branches = set()
        
        for node_id, score in candidates:
            if len(selected_templates) >= count:
                break
            
            node = self.taxonomy.nodes.get(node_id)
            if not node or node.node_type not in ['template', 'prompt']:
                continue
            
            # Check diversity - avoid too many from same branch
            branch_path = self._get_branch_path(node_id)
            branch_signature = tuple(branch_path[:3])  # Use top 3 levels for diversity
            
            if diversity_factor > 0 and branch_signature in used_branches:
                # Apply diversity penalty
                if len([b for b in used_branches if b == branch_signature]) >= max(1, int(count * diversity_factor)):
                    continue
            
            # Convert to template format
            template_data = {
                'node_id': node_id,
                'name': node.name,
                'content': node.content,
                'template_data': node.template_data,
                'domain': node.domain,
                'difficulty': node.difficulty,
                'purpose': node.purpose,
                'tags': list(node.tags),
                'effectiveness_score': node.effectiveness_score,
                'hierarchy_score': score
            }
            
            selected_templates.append(template_data)
            used_branches.add(branch_signature)
        
        # Record selection
        self.selection_history.extend([t['node_id'] for t in selected_templates])
        
        return selected_templates
    
    def update_template_effectiveness(self, node_id: str, effectiveness_score: float) -> None:
        """Update effectiveness feedback for template"""
        if node_id in self.taxonomy.nodes:
            node = self.taxonomy.nodes[node_id]
            node.update_effectiveness(effectiveness_score)
            self.effectiveness_feedback[node_id].append(effectiveness_score)
    
    def get_navigation_suggestions(self, current_node_id: str, target_requirements: Dict[str, Any]) -> List[str]:
        """Get navigation suggestions based on requirements"""
        if current_node_id not in self.taxonomy.nodes:
            return []
        
        # Use similarity-guided navigation toward requirements
        target_text = ' '.join([
            str(v) for v in target_requirements.values() 
            if isinstance(v, (str, int, float))
        ])
        
        similar_nodes = self.taxonomy.jump_to_similar(current_node_id, target_text)
        return [node_id for node_id, _ in similar_nodes[:10]]
    
    def _get_branch_path(self, node_id: str) -> List[str]:
        """Get path from root to node"""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.taxonomy.nodes:
            path.append(current_id)
            current_id = self.taxonomy.nodes[current_id].parent_id
        
        return list(reversed(path))


if __name__ == "__main__":
    # Example usage
    taxonomy = PromptTaxonomy("REV Prompt Hierarchy")
    
    # Create sample hierarchy
    categories = ['NLP', 'Computer Vision', 'ML Theory', 'Ethics']
    
    for category in categories:
        cat_node = PromptNode(
            node_id=str(uuid.uuid4()),
            name=category,
            description=f"Category for {category} prompts",
            node_type="category",
            domain=category.lower().replace(' ', '_'),
            purpose="organization"
        )
        taxonomy.add_node(cat_node)
        
        # Add subcategories
        for i in range(2):
            subcat_node = PromptNode(
                node_id=str(uuid.uuid4()),
                name=f"{category} Subcategory {i+1}",
                description=f"Subcategory {i+1} for {category}",
                node_type="subcategory",
                domain=category.lower().replace(' ', '_'),
                difficulty=i+2,
                purpose="specialization"
            )
            taxonomy.add_node(subcat_node, parent_id=cat_node.node_id)
    
    # Test navigation
    root_id = taxonomy.root_id
    path = taxonomy.navigate(root_id, NavigationMode.BREADTH_FIRST, max_steps=10)
    print(f"Breadth-first navigation: {len(path.nodes)} nodes visited")
    
    # Test query
    query_builder = HierarchicalQueryBuilder()
    query = query_builder.where('domain', '=', 'nlp').and_where('difficulty', '>', 1).build()
    results = taxonomy.query(query)
    print(f"Query results: {len(results)} matching nodes")
    
    # Test hierarchical template selection
    selector = HierarchicalTemplateSelector(taxonomy)
    templates = selector.select_templates_hierarchically(
        {'domain': 'nlp', 'difficulty_min': 1}, count=5
    )
    print(f"Selected templates: {len(templates)}")
    
    # Print statistics
    stats = taxonomy.get_statistics()
    print(f"Taxonomy statistics: {stats['total_nodes']} nodes, depth {stats['depth']}")