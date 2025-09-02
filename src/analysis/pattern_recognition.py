#!/usr/bin/env python3
"""
Pattern Recognition Components for Response Prediction System

This module provides advanced pattern recognition, clustering, and anomaly detection
for prompt-response analysis in the REV pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import time
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import zscore
import networkx as nx


@dataclass
class PromptPattern:
    """Represents a discovered prompt pattern"""
    pattern_id: str
    pattern_type: str  # 'structural', 'semantic', 'behavioral'
    description: str
    template_regex: Optional[str]
    feature_signature: Dict[str, float]
    example_prompts: List[str]
    response_characteristics: Dict[str, float]
    frequency: int
    confidence: float
    

@dataclass
class ClusterInfo:
    """Information about a prompt cluster"""
    cluster_id: int
    centroid: np.ndarray
    size: int
    density: float
    coherence_score: float
    representative_prompts: List[str]
    common_features: Dict[str, float]
    response_patterns: Dict[str, Any]


class TemplateResponseMapper:
    """Maps templates to response patterns using historical data"""
    
    def __init__(self):
        self.template_patterns: Dict[str, PromptPattern] = {}
        self.response_signatures: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.template_similarity_graph = nx.Graph()
        
        # Pattern matching components
        self.structural_patterns = [
            (r'(?i)what is .+\?', 'definition_question'),
            (r'(?i)how (?:to|do|does) .+\?', 'process_question'),
            (r'(?i)why .+\?', 'explanation_question'),
            (r'(?i)explain .+', 'explanation_request'),
            (r'(?i)compare .+ (?:and|with|to) .+', 'comparison_request'),
            (r'(?i)list .+', 'enumeration_request'),
            (r'(?i)analyze .+', 'analysis_request'),
            (r'(?i)describe .+', 'description_request')
        ]
        
        # Response pattern templates
        self.response_templates = {
            'definition_question': {
                'expected_length_ratio': 2.5,
                'structure': 'definition + examples + context',
                'coherence_weight': 0.8
            },
            'process_question': {
                'expected_length_ratio': 3.2,
                'structure': 'steps + explanations',
                'coherence_weight': 0.9
            },
            'comparison_request': {
                'expected_length_ratio': 4.0,
                'structure': 'similarities + differences + conclusion',
                'coherence_weight': 0.85
            }
        }
    
    def add_template_response_pair(self, template_id: str, prompt: str, 
                                  response: str, features: Dict[str, Any]) -> None:
        """Add a template-response pair to the mapping database"""
        # Extract response characteristics
        response_chars = self._extract_response_characteristics(response)
        
        # Update template pattern
        if template_id not in self.template_patterns:
            pattern_type = self._classify_pattern_type(prompt, features)
            self.template_patterns[template_id] = PromptPattern(
                pattern_id=template_id,
                pattern_type=pattern_type,
                description=self._generate_pattern_description(prompt, features),
                template_regex=self._extract_template_regex(prompt),
                feature_signature=self._create_feature_signature(features),
                example_prompts=[prompt],
                response_characteristics=response_chars,
                frequency=1,
                confidence=0.5
            )
        else:
            # Update existing pattern
            pattern = self.template_patterns[template_id]
            pattern.example_prompts.append(prompt)
            pattern.frequency += 1
            
            # Update response characteristics with exponential moving average
            alpha = 0.3
            for key, value in response_chars.items():
                if key in pattern.response_characteristics:
                    pattern.response_characteristics[key] = (
                        alpha * value + (1 - alpha) * pattern.response_characteristics[key]
                    )
                else:
                    pattern.response_characteristics[key] = value
            
            # Update confidence based on consistency
            pattern.confidence = min(0.95, pattern.confidence + 0.1 / np.sqrt(pattern.frequency))
        
        # Update response signatures
        self.response_signatures[template_id].update(response_chars)
        
        # Update similarity graph
        self._update_similarity_graph(template_id, features)
    
    def predict_response_from_template(self, template_id: str, 
                                     prompt_features: Dict[str, Any]) -> Dict[str, float]:
        """Predict response characteristics based on template mapping"""
        if template_id not in self.template_patterns:
            return self._default_response_prediction(prompt_features)
        
        pattern = self.template_patterns[template_id]
        
        # Base prediction from template pattern
        base_prediction = pattern.response_characteristics.copy()
        
        # Adjust based on prompt features
        feature_adjustment = self._calculate_feature_adjustment(
            prompt_features, pattern.feature_signature
        )
        
        # Apply adjustments
        adjusted_prediction = {}
        for key, value in base_prediction.items():
            adjustment_factor = feature_adjustment.get(key, 1.0)
            adjusted_prediction[key] = value * adjustment_factor
        
        # Add confidence information
        adjusted_prediction['prediction_confidence'] = pattern.confidence
        adjusted_prediction['template_frequency'] = pattern.frequency
        
        return adjusted_prediction
    
    def find_similar_templates(self, template_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar templates using the similarity graph"""
        if template_id not in self.template_similarity_graph:
            return []
        
        # Get neighbors with similarity scores
        neighbors = []
        for neighbor in self.template_similarity_graph.neighbors(template_id):
            edge_data = self.template_similarity_graph[template_id][neighbor]
            similarity = edge_data.get('similarity', 0.0)
            neighbors.append((neighbor, similarity))
        
        # Sort by similarity and return top-k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    def _extract_response_characteristics(self, response: str) -> Dict[str, float]:
        """Extract characteristics from response text"""
        words = response.split()
        sentences = response.split('.')
        
        return {
            'length': len(response),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'complexity_score': self._estimate_response_complexity(response),
            'structure_score': self._analyze_response_structure(response),
            'informativeness': self._estimate_informativeness(response)
        }
    
    def _classify_pattern_type(self, prompt: str, features: Dict[str, Any]) -> str:
        """Classify the type of prompt pattern"""
        # Check structural patterns
        for regex, pattern_type in self.structural_patterns:
            import re
            if re.search(regex, prompt):
                return f"structural_{pattern_type}"
        
        # Check semantic patterns based on features
        if features.get('technical_terms', 0) > 2:
            return 'semantic_technical'
        elif features.get('question_markers', 0) > 0:
            return 'semantic_interrogative'
        elif features.get('instruction_markers', 0) > 0:
            return 'semantic_imperative'
        else:
            return 'semantic_general'
    
    def _generate_pattern_description(self, prompt: str, features: Dict[str, Any]) -> str:
        """Generate a description for the pattern"""
        desc_parts = []
        
        if features.get('question_markers', 0) > 0:
            desc_parts.append("question-based")
        if features.get('instruction_markers', 0) > 0:
            desc_parts.append("instruction-based")
        if features.get('difficulty_level', 1) > 3:
            desc_parts.append("high-complexity")
        if features.get('technical_terms', 0) > 1:
            desc_parts.append("technical")
        
        if not desc_parts:
            desc_parts.append("general")
        
        return f"{', '.join(desc_parts)} prompt pattern"
    
    def _extract_template_regex(self, prompt: str) -> Optional[str]:
        """Extract a regex pattern from the prompt (simplified)"""
        # This would be more sophisticated in practice
        import re
        
        # Replace specific entities with placeholders
        pattern = re.sub(r'\b[A-Z][a-z]+\b', '[ENTITY]', prompt)  # Named entities
        pattern = re.sub(r'\b\d+\b', '[NUMBER]', pattern)  # Numbers
        pattern = re.escape(pattern)  # Escape special characters
        pattern = pattern.replace(r'\[ENTITY\]', r'\w+')  # Restore placeholders
        pattern = pattern.replace(r'\[NUMBER\]', r'\d+')
        
        return pattern
    
    def _create_feature_signature(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Create a numeric signature from features"""
        signature = {}
        
        # Convert relevant features to numeric signature
        numeric_features = [
            'length', 'word_count', 'sentence_count', 'avg_word_length',
            'flesch_kincaid_grade', 'syntactic_complexity', 'vocabulary_diversity',
            'question_markers', 'instruction_markers', 'technical_terms',
            'difficulty_level'
        ]
        
        for feature in numeric_features:
            if feature in features:
                signature[feature] = float(features[feature])
        
        # Boolean features
        boolean_features = ['has_examples', 'requires_reasoning', 'has_constraints', 'multi_step']
        for feature in boolean_features:
            if feature in features:
                signature[feature] = 1.0 if features[feature] else 0.0
        
        return signature
    
    def _calculate_feature_adjustment(self, current_features: Dict[str, Any], 
                                    signature: Dict[str, float]) -> Dict[str, float]:
        """Calculate adjustment factors based on feature differences"""
        adjustments = {}
        current_sig = self._create_feature_signature(current_features)
        
        # Calculate relative differences
        for key in signature:
            if key in current_sig:
                if signature[key] != 0:
                    ratio = current_sig[key] / signature[key]
                    # Length-based adjustments
                    if key in ['length', 'word_count']:
                        adjustments['length'] = ratio
                        adjustments['word_count'] = ratio
                    # Complexity adjustments
                    elif key in ['difficulty_level', 'flesch_kincaid_grade']:
                        adjustments['complexity_score'] = ratio
        
        return adjustments
    
    def _estimate_response_complexity(self, response: str) -> float:
        """Estimate complexity of response text"""
        # Simplified complexity estimation
        sentences = response.split('.')
        words = response.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words)
        
        # Punctuation complexity
        complex_punct = response.count(';') + response.count(':') + response.count('(')
        punct_density = complex_punct / len(words)
        
        # Combine metrics
        complexity = (
            min(avg_sentence_length / 20.0, 1.0) * 0.4 +
            vocabulary_diversity * 0.4 +
            min(punct_density * 10, 1.0) * 0.2
        )
        
        return complexity
    
    def _analyze_response_structure(self, response: str) -> float:
        """Analyze structural coherence of response"""
        # Check for list structures
        list_markers = response.count('1.') + response.count('2.') + response.count('•')
        
        # Check for paragraph structure
        paragraphs = response.split('\n\n')
        
        # Check for logical connectors
        connectors = ['however', 'therefore', 'furthermore', 'moreover', 'consequently']
        connector_count = sum(response.lower().count(c) for c in connectors)
        
        # Combine structure indicators
        structure_score = (
            min(list_markers / 5.0, 0.3) +
            min(len(paragraphs) / 5.0, 0.4) +
            min(connector_count / 3.0, 0.3)
        )
        
        return min(structure_score, 1.0)
    
    def _estimate_informativeness(self, response: str) -> float:
        """Estimate informativeness of response"""
        words = response.split()
        
        # Information-bearing word ratio
        info_words = ['explain', 'because', 'reason', 'example', 'specifically', 'detail']
        info_ratio = sum(1 for word in words if word.lower() in info_words) / max(len(words), 1)
        
        # Entity density (simplified)
        import re
        entities = len(re.findall(r'\b[A-Z][a-z]+\b', response))
        entity_density = entities / max(len(words), 1)
        
        # Question answering indicators
        qa_indicators = ['answer', 'solution', 'result', 'conclusion']
        qa_ratio = sum(1 for word in words if word.lower() in qa_indicators) / max(len(words), 1)
        
        informativeness = (
            info_ratio * 0.4 +
            min(entity_density * 5, 1.0) * 0.3 +
            qa_ratio * 0.3
        )
        
        return min(informativeness, 1.0)
    
    def _update_similarity_graph(self, template_id: str, features: Dict[str, Any]) -> None:
        """Update the template similarity graph"""
        current_signature = self._create_feature_signature(features)
        
        # Add node if not exists
        if not self.template_similarity_graph.has_node(template_id):
            self.template_similarity_graph.add_node(template_id, signature=current_signature)
            return
        
        # Calculate similarities with existing templates
        for other_template in self.template_similarity_graph.nodes():
            if other_template != template_id:
                other_signature = self.template_similarity_graph.nodes[other_template].get('signature', {})
                
                if other_signature:
                    similarity = self._calculate_signature_similarity(current_signature, other_signature)
                    
                    # Add edge if similarity is above threshold
                    if similarity > 0.7:
                        self.template_similarity_graph.add_edge(
                            template_id, other_template, 
                            similarity=similarity
                        )
    
    def _calculate_signature_similarity(self, sig1: Dict[str, float], 
                                      sig2: Dict[str, float]) -> float:
        """Calculate similarity between two feature signatures"""
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        if not common_keys:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = np.array([sig1[key] for key in common_keys])
        vec2 = np.array([sig2[key] for key in common_keys])
        
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _default_response_prediction(self, prompt_features: Dict[str, Any]) -> Dict[str, float]:
        """Default response prediction when no template mapping exists"""
        length = prompt_features.get('length', 100)
        
        return {
            'length': length * 2.5,
            'word_count': prompt_features.get('word_count', 20) * 3.0,
            'complexity_score': prompt_features.get('flesch_kincaid_grade', 8.0) / 15.0,
            'structure_score': 0.6,
            'informativeness': 0.7,
            'prediction_confidence': 0.3,
            'template_frequency': 0
        }


class PromptClusterAnalyzer:
    """Analyzes prompt clusters for pattern discovery"""
    
    def __init__(self, n_clusters: int = 20):
        self.n_clusters = n_clusters
        self.clusterer = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)  # Dimensionality reduction
        
        self.cluster_info: Dict[int, ClusterInfo] = {}
        self.prompt_to_cluster: Dict[str, int] = {}
        
    def fit_clusters(self, prompts: List[str], features_list: List[Dict[str, Any]]) -> Dict[int, ClusterInfo]:
        """Fit clustering model and analyze clusters"""
        if len(prompts) < self.n_clusters:
            self.n_clusters = max(2, len(prompts) // 2)
        
        # Create feature matrix
        text_features = self.vectorizer.fit_transform(prompts).toarray()
        
        # Extract numeric features
        numeric_features = []
        for features in features_list:
            feature_vector = [
                features.get('length', 0),
                features.get('word_count', 0),
                features.get('sentence_count', 0),
                features.get('difficulty_level', 1),
                features.get('question_markers', 0),
                features.get('instruction_markers', 0),
                features.get('technical_terms', 0),
                float(features.get('has_examples', False)),
                float(features.get('requires_reasoning', False)),
                float(features.get('multi_step', False))
            ]
            numeric_features.append(feature_vector)
        
        numeric_features = np.array(numeric_features)
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        # Combine text and numeric features
        combined_features = np.hstack([text_features, numeric_features])
        
        # Reduce dimensionality
        if combined_features.shape[1] > 50:
            combined_features = self.pca.fit_transform(combined_features)
        
        # Fit clustering
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.clusterer.fit_predict(combined_features)
        
        # Analyze clusters
        for cluster_id in range(self.n_clusters):
            cluster_prompts = [prompts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_features = [features_list[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if cluster_prompts:
                cluster_info = self._analyze_cluster(
                    cluster_id, cluster_prompts, cluster_features, combined_features[cluster_labels == cluster_id]
                )
                self.cluster_info[cluster_id] = cluster_info
                
                # Update prompt-to-cluster mapping
                for prompt in cluster_prompts:
                    self.prompt_to_cluster[prompt] = cluster_id
        
        return self.cluster_info
    
    def predict_cluster(self, prompt: str, features: Dict[str, Any]) -> Tuple[int, float]:
        """Predict cluster for new prompt"""
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted")
        
        # Transform prompt
        text_features = self.vectorizer.transform([prompt]).toarray()
        
        # Extract numeric features
        feature_vector = [
            features.get('length', 0),
            features.get('word_count', 0),
            features.get('sentence_count', 0),
            features.get('difficulty_level', 1),
            features.get('question_markers', 0),
            features.get('instruction_markers', 0),
            features.get('technical_terms', 0),
            float(features.get('has_examples', False)),
            float(features.get('requires_reasoning', False)),
            float(features.get('multi_step', False))
        ]
        
        numeric_features = np.array(feature_vector).reshape(1, -1)
        numeric_features = self.scaler.transform(numeric_features)
        
        # Combine features
        combined_features = np.hstack([text_features, numeric_features])
        
        # Apply PCA if needed
        if hasattr(self.pca, 'components_'):
            combined_features = self.pca.transform(combined_features)
        
        # Predict cluster
        cluster_id = self.clusterer.predict(combined_features)[0]
        
        # Calculate confidence based on distance to centroid
        cluster_center = self.clusterer.cluster_centers_[cluster_id]
        distance = np.linalg.norm(combined_features[0] - cluster_center)
        
        # Convert distance to confidence (higher distance = lower confidence)
        max_distance = np.max([np.linalg.norm(center - cluster_center) 
                              for center in self.clusterer.cluster_centers_])
        confidence = 1.0 - (distance / max(max_distance, 1e-6))
        
        return cluster_id, confidence
    
    def get_cluster_representative_prompts(self, cluster_id: int, n: int = 3) -> List[str]:
        """Get representative prompts for a cluster"""
        if cluster_id in self.cluster_info:
            return self.cluster_info[cluster_id].representative_prompts[:n]
        return []
    
    def _analyze_cluster(self, cluster_id: int, prompts: List[str], 
                        features_list: List[Dict[str, Any]], 
                        cluster_vectors: np.ndarray) -> ClusterInfo:
        """Analyze a specific cluster"""
        # Calculate centroid
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Calculate cluster density (average distance to centroid)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        density = 1.0 / (np.mean(distances) + 1e-6)  # Inverse of average distance
        
        # Calculate coherence score (silhouette-like metric)
        if len(cluster_vectors) > 1:
            intra_distances = []
            for i, vec1 in enumerate(cluster_vectors):
                for j, vec2 in enumerate(cluster_vectors):
                    if i != j:
                        intra_distances.append(np.linalg.norm(vec1 - vec2))
            coherence_score = 1.0 / (np.mean(intra_distances) + 1e-6)
        else:
            coherence_score = 1.0
        
        # Find representative prompts (closest to centroid)
        prompt_distances = [(i, dist) for i, dist in enumerate(distances)]
        prompt_distances.sort(key=lambda x: x[1])
        representative_indices = [i for i, _ in prompt_distances[:3]]
        representative_prompts = [prompts[i] for i in representative_indices]
        
        # Analyze common features
        common_features = self._extract_common_features(features_list)
        
        # Analyze response patterns (would need historical response data)
        response_patterns = {
            'avg_expected_length': common_features.get('avg_length', 0) * 2.5,
            'complexity_level': common_features.get('avg_difficulty', 2),
            'structure_type': self._infer_structure_type(prompts)
        }
        
        return ClusterInfo(
            cluster_id=cluster_id,
            centroid=centroid,
            size=len(prompts),
            density=density,
            coherence_score=coherence_score,
            representative_prompts=representative_prompts,
            common_features=common_features,
            response_patterns=response_patterns
        )
    
    def _extract_common_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract common features across cluster"""
        if not features_list:
            return {}
        
        numeric_features = ['length', 'word_count', 'difficulty_level', 'question_markers', 'instruction_markers']
        boolean_features = ['has_examples', 'requires_reasoning', 'multi_step']
        
        common = {}
        
        # Average numeric features
        for feature in numeric_features:
            values = [f.get(feature, 0) for f in features_list]
            if values:
                common[f'avg_{feature}'] = np.mean(values)
                common[f'std_{feature}'] = np.std(values)
        
        # Proportion of boolean features
        for feature in boolean_features:
            values = [f.get(feature, False) for f in features_list]
            common[f'prop_{feature}'] = sum(values) / len(values)
        
        return common
    
    def _infer_structure_type(self, prompts: List[str]) -> str:
        """Infer common structure type from prompts"""
        question_count = sum(1 for p in prompts if '?' in p)
        explanation_count = sum(1 for p in prompts if 'explain' in p.lower())
        compare_count = sum(1 for p in prompts if 'compare' in p.lower())
        list_count = sum(1 for p in prompts if 'list' in p.lower())
        
        if question_count > len(prompts) * 0.7:
            return 'question_based'
        elif explanation_count > len(prompts) * 0.5:
            return 'explanation_based'
        elif compare_count > len(prompts) * 0.3:
            return 'comparison_based'
        elif list_count > len(prompts) * 0.3:
            return 'enumeration_based'
        else:
            return 'mixed'


class AnomalyDetector:
    """Detects unusual prompts that may require special handling"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.lof_detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        
        self.statistical_thresholds = {}
        self.is_fitted = False
        
    def fit(self, prompts: List[str], features_list: List[Dict[str, Any]]) -> None:
        """Fit anomaly detection models"""
        # Create feature matrix
        feature_matrix = []
        feature_names = ['length', 'word_count', 'sentence_count', 'difficulty_level',
                        'question_markers', 'instruction_markers', 'technical_terms']
        
        for features in features_list:
            feature_vector = [features.get(name, 0) for name in feature_names]
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix)
        
        # Fit anomaly detectors
        self.isolation_forest.fit(feature_matrix)
        # LOF doesn't have a separate fit method, it's fit during predict
        
        # Calculate statistical thresholds
        for i, name in enumerate(feature_names):
            values = feature_matrix[:, i]
            self.statistical_thresholds[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25)
            }
        
        self.is_fitted = True
    
    def detect_anomalies(self, prompt: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if prompt is anomalous and classify anomaly type"""
        if not self.is_fitted:
            raise ValueError("Anomaly detector not fitted")
        
        # Create feature vector
        feature_names = ['length', 'word_count', 'sentence_count', 'difficulty_level',
                        'question_markers', 'instruction_markers', 'technical_terms']
        feature_vector = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
        
        # Isolation Forest detection
        isolation_score = self.isolation_forest.decision_function(feature_vector)[0]
        is_isolation_anomaly = self.isolation_forest.predict(feature_vector)[0] == -1
        
        # Statistical anomaly detection
        statistical_anomalies = []
        z_scores = {}
        
        for i, name in enumerate(feature_names):
            value = feature_vector[0, i]
            thresholds = self.statistical_thresholds[name]
            
            # Z-score anomaly
            if thresholds['std'] > 0:
                z_score = abs(value - thresholds['mean']) / thresholds['std']
                z_scores[name] = z_score
                if z_score > 3:  # 3-sigma rule
                    statistical_anomalies.append(f"{name}_zscore")
            
            # IQR anomaly
            iqr_lower = thresholds['q1'] - 1.5 * thresholds['iqr']
            iqr_upper = thresholds['q3'] + 1.5 * thresholds['iqr']
            if value < iqr_lower or value > iqr_upper:
                statistical_anomalies.append(f"{name}_iqr")
        
        # Content-based anomaly detection
        content_anomalies = self._detect_content_anomalies(prompt, features)
        
        # Combine results
        is_anomaly = is_isolation_anomaly or len(statistical_anomalies) > 0 or len(content_anomalies) > 0
        
        anomaly_score = abs(isolation_score)  # Convert to positive score
        if statistical_anomalies:
            anomaly_score += len(statistical_anomalies) * 0.2
        if content_anomalies:
            anomaly_score += len(content_anomalies) * 0.3
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'isolation_score': isolation_score,
            'statistical_anomalies': statistical_anomalies,
            'content_anomalies': content_anomalies,
            'z_scores': z_scores,
            'anomaly_types': self._classify_anomaly_types(statistical_anomalies, content_anomalies),
            'handling_recommendation': self._recommend_handling(is_anomaly, anomaly_score, 
                                                                statistical_anomalies, content_anomalies)
        }
    
    def _detect_content_anomalies(self, prompt: str, features: Dict[str, Any]) -> List[str]:
        """Detect content-based anomalies"""
        anomalies = []
        
        # Extremely long prompts
        if features.get('length', 0) > 2000:
            anomalies.append('extremely_long')
        
        # Extremely short prompts
        if features.get('length', 0) < 10:
            anomalies.append('extremely_short')
        
        # No question or instruction markers
        if features.get('question_markers', 0) == 0 and features.get('instruction_markers', 0) == 0:
            anomalies.append('unclear_intent')
        
        # Excessive repetition
        words = prompt.lower().split()
        if len(words) > 0:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
            if most_common_count > len(words) * 0.3:  # More than 30% repetition
                anomalies.append('excessive_repetition')
        
        # Unusual character patterns
        if len(set(prompt)) < len(prompt) * 0.3:  # Low character diversity
            anomalies.append('low_character_diversity')
        
        # Contains excessive special characters
        special_char_ratio = sum(1 for c in prompt if not c.isalnum() and c != ' ') / max(len(prompt), 1)
        if special_char_ratio > 0.3:
            anomalies.append('excessive_special_chars')
        
        return anomalies
    
    def _classify_anomaly_types(self, statistical_anomalies: List[str], 
                               content_anomalies: List[str]) -> List[str]:
        """Classify types of anomalies detected"""
        types = []
        
        # Length-based anomalies
        if any('length' in anomaly for anomaly in statistical_anomalies):
            types.append('length_anomaly')
        
        if 'extremely_long' in content_anomalies or 'extremely_short' in content_anomalies:
            types.append('extreme_length')
        
        # Complexity anomalies
        if any('difficulty' in anomaly for anomaly in statistical_anomalies):
            types.append('complexity_anomaly')
        
        # Structure anomalies
        if 'unclear_intent' in content_anomalies:
            types.append('structure_anomaly')
        
        # Content quality anomalies
        if any(anomaly in content_anomalies for anomaly in 
               ['excessive_repetition', 'low_character_diversity', 'excessive_special_chars']):
            types.append('quality_anomaly')
        
        return types if types else ['general_anomaly']
    
    def _recommend_handling(self, is_anomaly: bool, anomaly_score: float,
                           statistical_anomalies: List[str], content_anomalies: List[str]) -> str:
        """Recommend how to handle the anomalous prompt"""
        if not is_anomaly:
            return 'normal_processing'
        
        if anomaly_score > 2.0:
            return 'manual_review'
        elif 'extremely_short' in content_anomalies:
            return 'request_clarification'
        elif 'extremely_long' in content_anomalies:
            return 'truncate_or_summarize'
        elif 'unclear_intent' in content_anomalies:
            return 'intent_clarification'
        elif len(statistical_anomalies) > 2:
            return 'careful_processing'
        else:
            return 'standard_processing_with_monitoring'


if __name__ == "__main__":
    # Example usage
    mapper = TemplateResponseMapper()
    clusterer = PromptClusterAnalyzer()
    detector = AnomalyDetector()
    
    # Example data
    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "How does blockchain technology work?",
        "Compare supervised and unsupervised learning."
    ]
    
    features_list = [
        {'length': 25, 'word_count': 4, 'difficulty_level': 2, 'question_markers': 1, 'instruction_markers': 0},
        {'length': 38, 'word_count': 6, 'difficulty_level': 3, 'question_markers': 0, 'instruction_markers': 1},
        {'length': 36, 'word_count': 6, 'difficulty_level': 3, 'question_markers': 1, 'instruction_markers': 0},
        {'length': 42, 'word_count': 6, 'difficulty_level': 4, 'question_markers': 0, 'instruction_markers': 1}
    ]
    
    # Fit models
    clusterer.fit_clusters(prompts, features_list)
    detector.fit(prompts, features_list)
    
    # Test new prompt
    test_prompt = "What are the applications of artificial intelligence in healthcare?"
    test_features = {'length': 65, 'word_count': 10, 'difficulty_level': 3, 'question_markers': 1, 'instruction_markers': 0}
    
    # Predict cluster
    cluster_id, confidence = clusterer.predict_cluster(test_prompt, test_features)
    print(f"Predicted cluster: {cluster_id}, Confidence: {confidence:.2f}")
    
    # Detect anomalies
    anomaly_result = detector.detect_anomalies(test_prompt, test_features)
    print(f"Is anomaly: {anomaly_result['is_anomaly']}, Score: {anomaly_result['anomaly_score']:.2f}")