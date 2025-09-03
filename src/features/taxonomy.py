"""
Hierarchical Feature Taxonomy for Model Fingerprinting
Provides structured feature extraction across multiple abstraction levels
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy import stats
from scipy.spatial import distance
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureDescriptor:
    """Descriptor for a single feature"""
    name: str
    category: str
    subcategory: str
    dimension: int
    importance: float = 0.0
    interpretability: float = 1.0
    compute_cost: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    @abstractmethod
    def extract(self, model_output: Any, **kwargs) -> np.ndarray:
        """Extract features from model output"""
        pass
    
    @abstractmethod
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """Get feature descriptors"""
        pass


class SyntacticFeatures(FeatureExtractor):
    """Syntactic features: token distributions, n-gram patterns, vocabulary usage"""
    
    def __init__(self, vocab_size: int = 50000, max_ngram: int = 3):
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.token_counter = Counter()
        self.ngram_counters = {n: Counter() for n in range(1, max_ngram + 1)}
        
    def extract(self, model_output: Any, **kwargs) -> np.ndarray:
        """Extract syntactic features from tokenized output"""
        features = []
        
        # Token-level features
        if isinstance(model_output, str):
            tokens = model_output.split()
        elif hasattr(model_output, 'tokens'):
            tokens = model_output.tokens
        else:
            tokens = []
            
        if tokens:
            # Token frequency distribution
            token_freq = Counter(tokens)
            self.token_counter.update(token_freq)
            
            # Vocabulary diversity metrics
            vocab_size = len(set(tokens))
            token_count = len(tokens)
            type_token_ratio = vocab_size / max(token_count, 1)
            
            # Zipf distribution parameters
            if len(token_freq) > 1:
                frequencies = sorted(token_freq.values(), reverse=True)
                ranks = np.arange(1, len(frequencies) + 1)
                if len(frequencies) > 10:
                    slope, intercept = np.polyfit(np.log(ranks[:100]), 
                                                 np.log(frequencies[:100]), 1)
                    zipf_alpha = -slope
                else:
                    zipf_alpha = 1.0
            else:
                zipf_alpha = 1.0
                
            # N-gram patterns
            ngram_features = []
            for n in range(1, min(self.max_ngram + 1, len(tokens))):
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                ngram_freq = Counter(ngrams)
                self.ngram_counters[n].update(ngram_freq)
                
                # N-gram entropy
                total_ngrams = len(ngrams)
                if total_ngrams > 0:
                    probs = np.array(list(ngram_freq.values())) / total_ngrams
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    ngram_features.append(entropy)
                else:
                    ngram_features.append(0.0)
                    
            # Lexical complexity metrics
            avg_token_length = np.mean([len(t) for t in tokens])
            token_length_std = np.std([len(t) for t in tokens])
            
            # Sentence-level features (if punctuation available)
            sentences = ' '.join(tokens).split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s])
            
            features.extend([
                type_token_ratio,
                zipf_alpha,
                avg_token_length,
                token_length_std,
                avg_sentence_length,
                vocab_size / self.vocab_size,  # Normalized vocab usage
                *ngram_features
            ])
        else:
            # Return zero features if no tokens
            features = [0.0] * (6 + self.max_ngram)
            
        return np.array(features, dtype=np.float32)
    
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """Get syntactic feature descriptors"""
        descriptors = [
            FeatureDescriptor("type_token_ratio", "syntactic", "diversity", 1, 
                            importance=0.8, interpretability=0.9),
            FeatureDescriptor("zipf_alpha", "syntactic", "distribution", 1, 
                            importance=0.7, interpretability=0.8),
            FeatureDescriptor("avg_token_length", "syntactic", "lexical", 1, 
                            importance=0.5, interpretability=1.0),
            FeatureDescriptor("token_length_std", "syntactic", "lexical", 1, 
                            importance=0.4, interpretability=1.0),
            FeatureDescriptor("avg_sentence_length", "syntactic", "structure", 1, 
                            importance=0.6, interpretability=1.0),
            FeatureDescriptor("vocab_usage", "syntactic", "diversity", 1, 
                            importance=0.7, interpretability=0.9),
        ]
        
        for n in range(1, self.max_ngram + 1):
            descriptors.append(
                FeatureDescriptor(f"{n}gram_entropy", "syntactic", "ngram", 1,
                                importance=0.6 - n*0.1, interpretability=0.8)
            )
            
        return descriptors


class SemanticFeatures(FeatureExtractor):
    """Semantic features: embedding space characteristics, attention patterns"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embedding_cache = []
        self.attention_patterns = []
        
    def extract(self, model_output: Any, **kwargs) -> np.ndarray:
        """Extract semantic features from model embeddings and attention"""
        features = []
        
        # Extract embeddings if available
        embeddings = kwargs.get('embeddings', None)
        if embeddings is not None:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
                
            # Embedding space statistics
            embedding_mean = np.mean(embeddings, axis=0)
            embedding_std = np.std(embeddings, axis=0)
            embedding_skew = stats.skew(embeddings, axis=0)
            embedding_kurtosis = stats.kurtosis(embeddings, axis=0)
            
            # Cosine similarity distribution
            if len(embeddings) > 1:
                similarities = []
                for i in range(min(len(embeddings) - 1, 100)):
                    sim = 1 - distance.cosine(embeddings[i], embeddings[i+1])
                    similarities.append(sim)
                avg_similarity = np.mean(similarities)
                similarity_std = np.std(similarities)
            else:
                avg_similarity = 0.0
                similarity_std = 0.0
                
            # Principal component analysis
            if len(embeddings) > 10:
                cov_matrix = np.cov(embeddings.T)
                eigenvalues = np.linalg.eigvalsh(cov_matrix)[-10:]
                explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            else:
                explained_variance_ratio = np.zeros(10)
                
            features.extend([
                np.mean(embedding_mean),
                np.mean(embedding_std),
                np.mean(np.abs(embedding_skew)),
                np.mean(np.abs(embedding_kurtosis)),
                avg_similarity,
                similarity_std,
                *explained_variance_ratio
            ])
            
        # Extract attention patterns if available
        attention_weights = kwargs.get('attention_weights', None)
        if attention_weights is not None:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
                
            # Attention entropy
            if len(attention_weights.shape) >= 2:
                attention_probs = attention_weights.reshape(-1, attention_weights.shape[-1])
                attention_entropy = []
                for row in attention_probs:
                    if np.sum(row) > 0:
                        row = row / np.sum(row)
                        entropy = -np.sum(row * np.log2(row + 1e-10))
                        attention_entropy.append(entropy)
                        
                if attention_entropy:
                    avg_attention_entropy = np.mean(attention_entropy)
                    std_attention_entropy = np.std(attention_entropy)
                else:
                    avg_attention_entropy = 0.0
                    std_attention_entropy = 0.0
                    
                # Attention focus (max attention weight)
                max_attention = np.max(attention_weights)
                mean_max_attention = np.mean(np.max(attention_weights, axis=-1))
                
                features.extend([
                    avg_attention_entropy,
                    std_attention_entropy,
                    max_attention,
                    mean_max_attention
                ])
            else:
                features.extend([0.0] * 4)
        else:
            # Add zeros if no attention weights
            features.extend([0.0] * 4)
            
        # Ensure we have features even if inputs are missing
        if not features:
            features = [0.0] * 20
            
        return np.array(features[:20], dtype=np.float32)  # Fixed size output
    
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """Get semantic feature descriptors"""
        descriptors = [
            FeatureDescriptor("embedding_mean", "semantic", "embedding", 1,
                            importance=0.7, interpretability=0.6),
            FeatureDescriptor("embedding_std", "semantic", "embedding", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("embedding_skew", "semantic", "embedding", 1,
                            importance=0.5, interpretability=0.5),
            FeatureDescriptor("embedding_kurtosis", "semantic", "embedding", 1,
                            importance=0.5, interpretability=0.5),
            FeatureDescriptor("avg_cosine_similarity", "semantic", "similarity", 1,
                            importance=0.8, interpretability=0.8),
            FeatureDescriptor("similarity_std", "semantic", "similarity", 1,
                            importance=0.6, interpretability=0.7),
        ]
        
        for i in range(10):
            descriptors.append(
                FeatureDescriptor(f"pc{i+1}_variance", "semantic", "pca", 1,
                                importance=0.7 - i*0.05, interpretability=0.4)
            )
            
        descriptors.extend([
            FeatureDescriptor("avg_attention_entropy", "semantic", "attention", 1,
                            importance=0.8, interpretability=0.7),
            FeatureDescriptor("std_attention_entropy", "semantic", "attention", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("max_attention", "semantic", "attention", 1,
                            importance=0.7, interpretability=0.8),
            FeatureDescriptor("mean_max_attention", "semantic", "attention", 1,
                            importance=0.7, interpretability=0.7),
        ])
        
        return descriptors


class BehavioralFeatures(FeatureExtractor):
    """Behavioral features: response consistency, uncertainty, refusal patterns"""
    
    def __init__(self):
        self.response_history = []
        self.uncertainty_scores = []
        self.refusal_patterns = Counter()
        
    def extract(self, model_output: Any, **kwargs) -> np.ndarray:
        """Extract behavioral features from model responses"""
        features = []
        
        # Response consistency metrics
        responses = kwargs.get('response_variations', [])
        if responses and len(responses) > 1:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(responses) - 1):
                for j in range(i + 1, len(responses)):
                    if isinstance(responses[i], str) and isinstance(responses[j], str):
                        # Simple Jaccard similarity
                        set1 = set(responses[i].lower().split())
                        set2 = set(responses[j].lower().split())
                        if set1 or set2:
                            jaccard = len(set1 & set2) / len(set1 | set2)
                            similarities.append(jaccard)
                            
            if similarities:
                consistency_mean = np.mean(similarities)
                consistency_std = np.std(similarities)
            else:
                consistency_mean = 1.0
                consistency_std = 0.0
        else:
            consistency_mean = 1.0
            consistency_std = 0.0
            
        # Uncertainty quantification
        logprobs = kwargs.get('logprobs', None)
        if logprobs is not None:
            if isinstance(logprobs, torch.Tensor):
                logprobs = logprobs.detach().cpu().numpy()
                
            # Convert to probabilities
            probs = np.exp(logprobs)
            
            # Entropy as uncertainty measure
            entropy = -np.sum(probs * logprobs)
            
            # Confidence (max probability)
            max_prob = np.max(probs)
            
            # Prediction variance
            pred_variance = np.var(probs)
            
            features.extend([entropy, max_prob, pred_variance])
        else:
            features.extend([0.0, 1.0, 0.0])
            
        # Refusal behavior analysis
        response_text = kwargs.get('response_text', '')
        if response_text:
            refusal_keywords = [
                'cannot', 'unable', 'sorry', 'apologize', 'inappropriate',
                'harmful', 'unethical', 'illegal', 'refuse', 'decline'
            ]
            
            refusal_score = 0.0
            for keyword in refusal_keywords:
                if keyword in response_text.lower():
                    refusal_score += 1.0
                    self.refusal_patterns[keyword] += 1
                    
            refusal_score /= len(refusal_keywords)
            
            # Response length as behavioral indicator
            response_length = len(response_text.split())
            normalized_length = min(response_length / 500.0, 1.0)  # Normalize to [0,1]
            
            # Sentiment indicators (simple heuristic)
            positive_words = ['yes', 'sure', 'certainly', 'happy', 'glad']
            negative_words = ['no', 'not', 'never', 'unfortunately', 'however']
            
            positive_score = sum(1 for word in positive_words if word in response_text.lower())
            negative_score = sum(1 for word in negative_words if word in response_text.lower())
            
            sentiment_ratio = positive_score / max(positive_score + negative_score, 1)
            
            features.extend([
                refusal_score,
                normalized_length,
                sentiment_ratio
            ])
        else:
            features.extend([0.0, 0.0, 0.5])
            
        # Temperature-like behavior (response diversity)
        temperature_estimate = kwargs.get('temperature_estimate', 1.0)
        features.append(temperature_estimate)
        
        # Add consistency metrics
        features.extend([consistency_mean, consistency_std])
        
        return np.array(features, dtype=np.float32)
    
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """Get behavioral feature descriptors"""
        return [
            FeatureDescriptor("response_entropy", "behavioral", "uncertainty", 1,
                            importance=0.8, interpretability=0.7),
            FeatureDescriptor("max_probability", "behavioral", "uncertainty", 1,
                            importance=0.7, interpretability=0.9),
            FeatureDescriptor("prediction_variance", "behavioral", "uncertainty", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("refusal_score", "behavioral", "safety", 1,
                            importance=0.9, interpretability=1.0),
            FeatureDescriptor("response_length", "behavioral", "style", 1,
                            importance=0.4, interpretability=1.0),
            FeatureDescriptor("sentiment_ratio", "behavioral", "style", 1,
                            importance=0.5, interpretability=0.8),
            FeatureDescriptor("temperature_estimate", "behavioral", "diversity", 1,
                            importance=0.7, interpretability=0.8),
            FeatureDescriptor("consistency_mean", "behavioral", "consistency", 1,
                            importance=0.9, interpretability=0.9),
            FeatureDescriptor("consistency_std", "behavioral", "consistency", 1,
                            importance=0.7, interpretability=0.8),
        ]


class ArchitecturalFeatures(FeatureExtractor):
    """Architectural features: layer-wise statistics, gradient patterns"""
    
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.layer_stats_history = []
        
    def extract(self, model_output: Any, **kwargs) -> np.ndarray:
        """Extract architectural features from model internals"""
        features = []
        
        # Layer-wise activation statistics
        layer_activations = kwargs.get('layer_activations', None)
        if layer_activations is not None:
            layer_means = []
            layer_stds = []
            layer_sparsity = []
            
            for layer_act in layer_activations:
                if isinstance(layer_act, torch.Tensor):
                    layer_act = layer_act.detach().cpu().numpy()
                    
                # Statistics per layer
                layer_means.append(np.mean(np.abs(layer_act)))
                layer_stds.append(np.std(layer_act))
                
                # Sparsity (percentage of near-zero activations)
                sparsity = np.sum(np.abs(layer_act) < 0.01) / layer_act.size
                layer_sparsity.append(sparsity)
                
            # Aggregate statistics across layers
            features.extend([
                np.mean(layer_means),
                np.std(layer_means),
                np.mean(layer_stds),
                np.std(layer_stds),
                np.mean(layer_sparsity),
                np.std(layer_sparsity)
            ])
            
            # Layer-wise trends (linear regression slope)
            if len(layer_means) > 1:
                x = np.arange(len(layer_means))
                mean_trend = np.polyfit(x, layer_means, 1)[0]
                std_trend = np.polyfit(x, layer_stds, 1)[0]
                sparsity_trend = np.polyfit(x, layer_sparsity, 1)[0]
                features.extend([mean_trend, std_trend, sparsity_trend])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 9)
            
        # Gradient flow patterns
        gradients = kwargs.get('gradients', None)
        if gradients is not None:
            grad_norms = []
            for grad in gradients:
                if isinstance(grad, torch.Tensor):
                    grad = grad.detach().cpu().numpy()
                grad_norm = np.linalg.norm(grad.flatten())
                grad_norms.append(grad_norm)
                
            if grad_norms:
                features.extend([
                    np.mean(grad_norms),
                    np.std(grad_norms),
                    np.max(grad_norms),
                    np.min(grad_norms)
                ])
                
                # Gradient vanishing/exploding indicators
                vanishing_score = np.sum(np.array(grad_norms) < 1e-5) / len(grad_norms)
                exploding_score = np.sum(np.array(grad_norms) > 10) / len(grad_norms)
                features.extend([vanishing_score, exploding_score])
            else:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 6)
            
        # Model capacity indicators
        param_count = kwargs.get('param_count', 0)
        if param_count > 0:
            # Normalized by typical model sizes
            size_indicator = np.log10(param_count + 1) / 12  # Normalize by 1T params
            features.append(size_indicator)
        else:
            features.append(0.0)
            
        # Architectural type indicators (transformer-specific)
        num_heads = kwargs.get('num_attention_heads', 0)
        hidden_dim = kwargs.get('hidden_dim', 0)
        
        if num_heads > 0 and hidden_dim > 0:
            head_dim = hidden_dim / num_heads
            features.extend([
                num_heads / 128,  # Normalize by typical max
                head_dim / 128    # Normalize by typical head dimension
            ])
        else:
            features.extend([0.0, 0.0])
            
        return np.array(features, dtype=np.float32)
    
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """Get architectural feature descriptors"""
        return [
            FeatureDescriptor("mean_activation", "architectural", "activation", 1,
                            importance=0.7, interpretability=0.7),
            FeatureDescriptor("std_activation_across_layers", "architectural", "activation", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("mean_layer_std", "architectural", "activation", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("std_layer_std", "architectural", "activation", 1,
                            importance=0.5, interpretability=0.5),
            FeatureDescriptor("mean_sparsity", "architectural", "sparsity", 1,
                            importance=0.8, interpretability=0.8),
            FeatureDescriptor("std_sparsity", "architectural", "sparsity", 1,
                            importance=0.6, interpretability=0.7),
            FeatureDescriptor("activation_trend", "architectural", "trends", 1,
                            importance=0.7, interpretability=0.6),
            FeatureDescriptor("std_trend", "architectural", "trends", 1,
                            importance=0.6, interpretability=0.6),
            FeatureDescriptor("sparsity_trend", "architectural", "trends", 1,
                            importance=0.7, interpretability=0.7),
            FeatureDescriptor("mean_grad_norm", "architectural", "gradient", 1,
                            importance=0.8, interpretability=0.7),
            FeatureDescriptor("std_grad_norm", "architectural", "gradient", 1,
                            importance=0.7, interpretability=0.6),
            FeatureDescriptor("max_grad_norm", "architectural", "gradient", 1,
                            importance=0.6, interpretability=0.8),
            FeatureDescriptor("min_grad_norm", "architectural", "gradient", 1,
                            importance=0.6, interpretability=0.8),
            FeatureDescriptor("vanishing_gradient", "architectural", "gradient", 1,
                            importance=0.9, interpretability=0.9),
            FeatureDescriptor("exploding_gradient", "architectural", "gradient", 1,
                            importance=0.9, interpretability=0.9),
            FeatureDescriptor("model_size_indicator", "architectural", "capacity", 1,
                            importance=0.8, interpretability=1.0),
            FeatureDescriptor("num_heads_normalized", "architectural", "transformer", 1,
                            importance=0.7, interpretability=1.0),
            FeatureDescriptor("head_dimension", "architectural", "transformer", 1,
                            importance=0.6, interpretability=1.0),
        ]


class HierarchicalFeatureTaxonomy:
    """Main taxonomy orchestrator combining all feature categories"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize feature extractors
        self.syntactic = SyntacticFeatures(
            vocab_size=self.config.get('vocab_size', 50000),
            max_ngram=self.config.get('max_ngram', 3)
        )
        self.semantic = SemanticFeatures(
            embedding_dim=self.config.get('embedding_dim', 768)
        )
        self.behavioral = BehavioralFeatures()
        self.architectural = ArchitecturalFeatures(
            num_layers=self.config.get('num_layers', 32)
        )
        
        self.extractors = {
            'syntactic': self.syntactic,
            'semantic': self.semantic,
            'behavioral': self.behavioral,
            'architectural': self.architectural
        }
        
        self.feature_cache = []
        self.importance_scores = {}
        
    def extract_all_features(self, model_output: Any, **kwargs) -> Dict[str, np.ndarray]:
        """Extract all feature categories"""
        all_features = {}
        
        for category, extractor in self.extractors.items():
            try:
                features = extractor.extract(model_output, **kwargs)
                all_features[category] = features
                logger.debug(f"Extracted {len(features)} {category} features")
            except Exception as e:
                logger.warning(f"Failed to extract {category} features: {e}")
                # Return zero features on failure
                descriptors = extractor.get_descriptors()
                all_features[category] = np.zeros(len(descriptors), dtype=np.float32)
                
        return all_features
    
    def get_concatenated_features(self, model_output: Any, **kwargs) -> np.ndarray:
        """Get all features concatenated into a single vector"""
        all_features = self.extract_all_features(model_output, **kwargs)
        concatenated = np.concatenate(list(all_features.values()))
        return concatenated
    
    def get_feature_groups(self) -> Dict[str, List[int]]:
        """Get indices for each feature group in concatenated vector"""
        groups = {}
        offset = 0
        
        for category, extractor in self.extractors.items():
            descriptors = extractor.get_descriptors()
            num_features = len(descriptors)
            groups[category] = list(range(offset, offset + num_features))
            offset += num_features
            
        return groups
    
    def get_all_descriptors(self) -> List[FeatureDescriptor]:
        """Get all feature descriptors"""
        all_descriptors = []
        for extractor in self.extractors.values():
            all_descriptors.extend(extractor.get_descriptors())
        return all_descriptors
    
    def update_importance_scores(self, feature_importance: np.ndarray):
        """Update importance scores based on analysis"""
        descriptors = self.get_all_descriptors()
        assert len(feature_importance) == len(descriptors), \
            f"Importance vector size {len(feature_importance)} doesn't match features {len(descriptors)}"
            
        for i, descriptor in enumerate(descriptors):
            descriptor.importance = float(feature_importance[i])
            self.importance_scores[descriptor.name] = descriptor.importance
            
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top n most important features"""
        descriptors = self.get_all_descriptors()
        sorted_features = sorted(descriptors, key=lambda x: x.importance, reverse=True)
        return [(f.name, f.importance) for f in sorted_features[:n]]
    
    def save_taxonomy(self, filepath: str):
        """Save taxonomy configuration and importance scores"""
        taxonomy_data = {
            'config': self.config,
            'feature_groups': self.get_feature_groups(),
            'descriptors': [
                {
                    'name': d.name,
                    'category': d.category,
                    'subcategory': d.subcategory,
                    'importance': d.importance,
                    'interpretability': d.interpretability,
                    'compute_cost': d.compute_cost
                }
                for d in self.get_all_descriptors()
            ],
            'importance_scores': self.importance_scores
        }
        
        with open(filepath, 'w') as f:
            json.dump(taxonomy_data, f, indent=2)
            
    @classmethod
    def load_taxonomy(cls, filepath: str) -> 'HierarchicalFeatureTaxonomy':
        """Load taxonomy from file"""
        with open(filepath, 'r') as f:
            taxonomy_data = json.load(f)
            
        taxonomy = cls(config=taxonomy_data.get('config', {}))
        
        # Restore importance scores
        if 'importance_scores' in taxonomy_data:
            descriptors = taxonomy.get_all_descriptors()
            for descriptor in descriptors:
                if descriptor.name in taxonomy_data['importance_scores']:
                    descriptor.importance = taxonomy_data['importance_scores'][descriptor.name]
                    
        return taxonomy
    
    def compute_feature_correlations(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation matrix between features"""
        if len(feature_matrix.shape) == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
            
        # Use Spearman correlation for robustness
        correlations = np.corrcoef(feature_matrix.T)
        return correlations
    
    def identify_redundant_features(self, feature_matrix: np.ndarray, 
                                   threshold: float = 0.95) -> List[int]:
        """Identify highly correlated redundant features"""
        correlations = self.compute_feature_correlations(feature_matrix)
        redundant = []
        
        for i in range(correlations.shape[0]):
            for j in range(i + 1, correlations.shape[1]):
                if abs(correlations[i, j]) > threshold:
                    # Keep feature with higher importance
                    descriptors = self.get_all_descriptors()
                    if descriptors[i].importance < descriptors[j].importance:
                        redundant.append(i)
                    else:
                        redundant.append(j)
                        
        return list(set(redundant))