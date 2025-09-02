#!/usr/bin/env python3
"""
Response Prediction System for REV Pipeline

This module provides lightweight prediction models to estimate model outputs without execution,
enabling intelligent prompt pre-filtering and optimization.
"""

import os
import json
import time
import pickle
import hashlib
import statistics
from typing import Dict, List, Optional, Tuple, Any, NamedTuple, Callable, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine, euclidean, hamming
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Core prediction structures
@dataclass
class PromptFeatures:
    """Extracted features from prompt for prediction"""
    # Basic characteristics
    length: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    
    # Complexity metrics
    flesch_kincaid_grade: float
    syntactic_complexity: float
    vocabulary_diversity: float
    
    # Content characteristics
    question_markers: int
    instruction_markers: int
    technical_terms: int
    domain_indicators: List[str]
    
    # Template characteristics
    template_id: Optional[str]
    template_category: Optional[str]
    difficulty_level: int
    
    # Contextual features
    has_examples: bool
    requires_reasoning: bool
    has_constraints: bool
    multi_step: bool
    
    # Semantic features
    tfidf_vector: Optional[np.ndarray] = None
    embedding_vector: Optional[np.ndarray] = None


@dataclass
class ResponsePrediction:
    """Predicted response characteristics"""
    # Response properties
    estimated_length: float
    estimated_word_count: float
    estimated_complexity: float
    
    # Execution properties
    estimated_tokens: int
    estimated_computation_time: float
    estimated_memory_usage: float
    
    # Quality metrics
    predicted_coherence: float
    predicted_informativeness: float
    predicted_diversity: float
    
    # Confidence and reliability
    confidence_interval: Tuple[float, float]
    prediction_confidence: float
    uncertainty_score: float
    
    # Response categories
    response_type: str  # 'informative', 'creative', 'analytical', 'factual'
    expected_structure: str  # 'list', 'paragraph', 'steps', 'comparison'
    
    # Metadata
    prediction_method: str
    similar_prompts_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class HistoricalResponse:
    """Historical response data for learning"""
    prompt: str
    response: str
    features: PromptFeatures
    
    # Actual measured properties
    actual_length: int
    actual_word_count: int
    actual_tokens: int
    execution_time: float
    memory_usage: float
    
    # Quality metrics
    coherence_score: float
    informativeness_score: float
    diversity_score: float
    
    # Metadata
    model_id: str
    timestamp: float
    template_id: Optional[str] = None


class FeatureExtractor:
    """Extracts comprehensive features from prompts for prediction"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
        # Domain indicators
        self.domain_keywords = {
            'technical': ['algorithm', 'system', 'implementation', 'code', 'programming'],
            'scientific': ['experiment', 'hypothesis', 'analysis', 'research', 'study'],
            'mathematical': ['equation', 'formula', 'calculate', 'proof', 'theorem'],
            'creative': ['story', 'imagine', 'creative', 'design', 'artistic'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'examine'],
            'philosophical': ['ethics', 'moral', 'philosophy', 'belief', 'meaning']
        }
        
        # Question and instruction markers
        self.question_markers = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
        self.instruction_markers = ['explain', 'describe', 'analyze', 'compare', 'list', 'create']
    
    def fit(self, prompts: List[str]) -> 'FeatureExtractor':
        """Fit feature extractors on prompt corpus"""
        self.tfidf_vectorizer.fit(prompts)
        self.is_fitted = True
        return self
    
    def extract_features(self, prompt: str, template_id: Optional[str] = None) -> PromptFeatures:
        """Extract comprehensive features from a prompt"""
        # Basic text statistics
        words = prompt.lower().split()
        sentences = prompt.split('.')
        
        length = len(prompt)
        word_count = len(words)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Complexity metrics
        flesch_kincaid = self._calculate_flesch_kincaid(prompt)
        syntactic_complexity = self._calculate_syntactic_complexity(prompt)
        vocabulary_diversity = len(set(words)) / max(len(words), 1)
        
        # Content analysis
        question_markers = sum(1 for marker in self.question_markers if marker in prompt.lower())
        instruction_markers = sum(1 for marker in self.instruction_markers if marker in prompt.lower())
        technical_terms = self._count_technical_terms(words)
        domain_indicators = self._identify_domains(words)
        
        # Template characteristics
        difficulty_level = self._estimate_difficulty(prompt, question_markers, instruction_markers)
        
        # Contextual features
        has_examples = 'example' in prompt.lower() or 'for instance' in prompt.lower()
        requires_reasoning = any(word in prompt.lower() for word in ['because', 'therefore', 'thus', 'reason'])
        has_constraints = any(word in prompt.lower() for word in ['must', 'should', 'cannot', 'only'])
        multi_step = any(word in prompt.lower() for word in ['first', 'then', 'next', 'finally', 'step'])
        
        # TF-IDF vector
        tfidf_vector = None
        if self.is_fitted:
            tfidf_vector = self.tfidf_vectorizer.transform([prompt]).toarray()[0]
        
        return PromptFeatures(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            flesch_kincaid_grade=flesch_kincaid,
            syntactic_complexity=syntactic_complexity,
            vocabulary_diversity=vocabulary_diversity,
            question_markers=question_markers,
            instruction_markers=instruction_markers,
            technical_terms=technical_terms,
            domain_indicators=domain_indicators,
            template_id=template_id,
            template_category=self._categorize_template(prompt),
            difficulty_level=difficulty_level,
            has_examples=has_examples,
            requires_reasoning=requires_reasoning,
            has_constraints=has_constraints,
            multi_step=multi_step,
            tfidf_vector=tfidf_vector
        )
    
    def _calculate_flesch_kincaid(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        sentences = text.split('.')
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower().strip('.,!?";')
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Estimate syntactic complexity based on sentence structure"""
        # Simplified complexity based on punctuation and conjunctions
        complex_markers = [',', ';', ':', 'however', 'although', 'because', 'therefore']
        complexity_score = sum(text.lower().count(marker) for marker in complex_markers)
        return min(complexity_score / max(len(text.split()), 1), 1.0)
    
    def _count_technical_terms(self, words: List[str]) -> int:
        """Count technical terms in word list"""
        technical_indicators = [
            'algorithm', 'implementation', 'optimization', 'framework', 'architecture',
            'protocol', 'interface', 'methodology', 'paradigm', 'heuristic'
        ]
        return sum(1 for word in words if word in technical_indicators)
    
    def _identify_domains(self, words: List[str]) -> List[str]:
        """Identify domain indicators in text"""
        identified_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in words for keyword in keywords):
                identified_domains.append(domain)
        return identified_domains
    
    def _estimate_difficulty(self, prompt: str, question_markers: int, instruction_markers: int) -> int:
        """Estimate prompt difficulty level (1-5)"""
        base_difficulty = 1
        
        # Length complexity
        if len(prompt) > 200:
            base_difficulty += 1
        if len(prompt) > 500:
            base_difficulty += 1
        
        # Question complexity
        if question_markers > 2:
            base_difficulty += 1
        
        # Instruction complexity
        if instruction_markers > 1:
            base_difficulty += 1
        
        # Multi-part questions
        if 'and' in prompt.lower() and ('?' in prompt or any(marker in prompt.lower() for marker in self.instruction_markers)):
            base_difficulty += 1
        
        return min(base_difficulty, 5)
    
    def _categorize_template(self, prompt: str) -> str:
        """Categorize the prompt template type"""
        prompt_lower = prompt.lower()
        
        if any(marker in prompt_lower for marker in ['what', 'why', 'how', 'when']):
            return 'question'
        elif any(marker in prompt_lower for marker in ['explain', 'describe', 'analyze']):
            return 'explanation'
        elif any(marker in prompt_lower for marker in ['compare', 'contrast', 'difference']):
            return 'comparison'
        elif any(marker in prompt_lower for marker in ['create', 'generate', 'design']):
            return 'creative'
        elif any(marker in prompt_lower for marker in ['list', 'enumerate', 'identify']):
            return 'enumeration'
        else:
            return 'general'


class LightweightPredictionModels:
    """Collection of lightweight models for response prediction"""
    
    def __init__(self):
        # Response length predictors
        self.length_predictor = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.word_count_predictor = LinearRegression()
        self.token_predictor = Ridge(alpha=1.0)
        
        # Complexity predictors
        self.complexity_predictor = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
        self.computation_time_predictor = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
        
        # Quality predictors
        self.coherence_predictor = LinearRegression()
        self.informativeness_predictor = Ridge(alpha=0.5)
        
        # Feature scaling
        self.feature_scaler = StandardScaler()
        
        # Model metadata
        self.is_trained = False
        self.feature_names = []
        self.training_size = 0
        
    def prepare_features(self, features: PromptFeatures) -> np.ndarray:
        """Convert PromptFeatures to numpy array for model input"""
        feature_vector = [
            features.length,
            features.word_count,
            features.sentence_count,
            features.avg_word_length,
            features.flesch_kincaid_grade,
            features.syntactic_complexity,
            features.vocabulary_diversity,
            features.question_markers,
            features.instruction_markers,
            features.technical_terms,
            len(features.domain_indicators),
            features.difficulty_level,
            float(features.has_examples),
            float(features.requires_reasoning),
            float(features.has_constraints),
            float(features.multi_step),
        ]
        
        # Add TF-IDF features if available
        if features.tfidf_vector is not None:
            feature_vector.extend(features.tfidf_vector[:50])  # Use top 50 TF-IDF features
        else:
            feature_vector.extend([0.0] * 50)  # Pad with zeros if not available
        
        return np.array(feature_vector).reshape(1, -1)
    
    def train(self, historical_data: List[HistoricalResponse]) -> Dict[str, float]:
        """Train all prediction models on historical data"""
        if not historical_data:
            raise ValueError("No historical data provided for training")
        
        # Prepare training data
        X = []
        y_length = []
        y_word_count = []
        y_tokens = []
        y_complexity = []
        y_computation_time = []
        y_coherence = []
        y_informativeness = []
        
        for response in historical_data:
            features_vector = self.prepare_features(response.features)[0]
            X.append(features_vector)
            
            y_length.append(response.actual_length)
            y_word_count.append(response.actual_word_count)
            y_tokens.append(response.actual_tokens)
            y_complexity.append(response.features.flesch_kincaid_grade)
            y_computation_time.append(response.execution_time)
            y_coherence.append(response.coherence_score)
            y_informativeness.append(response.informativeness_score)
        
        X = np.array(X)
        self.feature_scaler.fit(X)
        X_scaled = self.feature_scaler.transform(X)
        
        # Train models
        models_performance = {}
        
        # Length prediction
        self.length_predictor.fit(X_scaled, y_length)
        length_score = cross_val_score(self.length_predictor, X_scaled, y_length, cv=min(5, len(X))).mean()
        models_performance['length_r2'] = length_score
        
        # Word count prediction
        self.word_count_predictor.fit(X_scaled, y_word_count)
        word_count_score = cross_val_score(self.word_count_predictor, X_scaled, y_word_count, cv=min(5, len(X))).mean()
        models_performance['word_count_r2'] = word_count_score
        
        # Token prediction
        self.token_predictor.fit(X_scaled, y_tokens)
        token_score = cross_val_score(self.token_predictor, X_scaled, y_tokens, cv=min(5, len(X))).mean()
        models_performance['token_r2'] = token_score
        
        # Complexity prediction
        self.complexity_predictor.fit(X_scaled, y_complexity)
        complexity_score = cross_val_score(self.complexity_predictor, X_scaled, y_complexity, cv=min(5, len(X))).mean()
        models_performance['complexity_r2'] = complexity_score
        
        # Computation time prediction
        self.computation_time_predictor.fit(X_scaled, y_computation_time)
        time_score = cross_val_score(self.computation_time_predictor, X_scaled, y_computation_time, cv=min(5, len(X))).mean()
        models_performance['computation_time_r2'] = time_score
        
        # Quality predictors
        self.coherence_predictor.fit(X_scaled, y_coherence)
        coherence_score = cross_val_score(self.coherence_predictor, X_scaled, y_coherence, cv=min(5, len(X))).mean()
        models_performance['coherence_r2'] = coherence_score
        
        self.informativeness_predictor.fit(X_scaled, y_informativeness)
        informativeness_score = cross_val_score(self.informativeness_predictor, X_scaled, y_informativeness, cv=min(5, len(X))).mean()
        models_performance['informativeness_r2'] = informativeness_score
        
        self.is_trained = True
        self.training_size = len(historical_data)
        
        return models_performance
    
    def predict(self, features: PromptFeatures) -> ResponsePrediction:
        """Make comprehensive response prediction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(features)
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        estimated_length = max(0, self.length_predictor.predict(X_scaled)[0])
        estimated_word_count = max(0, self.word_count_predictor.predict(X_scaled)[0])
        estimated_tokens = max(0, int(self.token_predictor.predict(X_scaled)[0]))
        estimated_complexity = self.complexity_predictor.predict(X_scaled)[0]
        estimated_computation_time = max(0, self.computation_time_predictor.predict(X_scaled)[0])
        
        # Quality predictions
        predicted_coherence = np.clip(self.coherence_predictor.predict(X_scaled)[0], 0, 1)
        predicted_informativeness = np.clip(self.informativeness_predictor.predict(X_scaled)[0], 0, 1)
        
        # Estimate memory usage based on tokens and complexity
        estimated_memory_usage = estimated_tokens * 0.004 + estimated_complexity * 100  # MB
        
        # Calculate prediction confidence based on training data similarity
        prediction_confidence = self._calculate_confidence(features)
        
        # Estimate confidence intervals (simplified)
        length_std = estimated_length * 0.2  # Assume 20% standard deviation
        confidence_interval = (
            max(0, estimated_length - 1.96 * length_std),
            estimated_length + 1.96 * length_std
        )
        
        # Predict response characteristics
        response_type = self._predict_response_type(features)
        expected_structure = self._predict_response_structure(features)
        
        # Estimate diversity based on features
        predicted_diversity = self._estimate_diversity(features)
        
        # Calculate uncertainty score
        uncertainty_score = 1.0 - prediction_confidence
        
        return ResponsePrediction(
            estimated_length=estimated_length,
            estimated_word_count=estimated_word_count,
            estimated_complexity=estimated_complexity,
            estimated_tokens=estimated_tokens,
            estimated_computation_time=estimated_computation_time,
            estimated_memory_usage=estimated_memory_usage,
            predicted_coherence=predicted_coherence,
            predicted_informativeness=predicted_informativeness,
            predicted_diversity=predicted_diversity,
            confidence_interval=confidence_interval,
            prediction_confidence=prediction_confidence,
            uncertainty_score=uncertainty_score,
            response_type=response_type,
            expected_structure=expected_structure,
            prediction_method="ensemble_ml",
            similar_prompts_count=self.training_size
        )
    
    def _calculate_confidence(self, features: PromptFeatures) -> float:
        """Calculate prediction confidence based on feature similarity to training data"""
        # Simplified confidence calculation
        base_confidence = 0.7
        
        # Reduce confidence for extreme values
        if features.length > 1000 or features.length < 10:
            base_confidence -= 0.2
        
        if features.difficulty_level > 4:
            base_confidence -= 0.1
        
        if len(features.domain_indicators) == 0:
            base_confidence -= 0.1
        
        return max(0.1, base_confidence)
    
    def _predict_response_type(self, features: PromptFeatures) -> str:
        """Predict response type based on prompt features"""
        if 'creative' in features.domain_indicators:
            return 'creative'
        elif features.technical_terms > 2:
            return 'analytical'
        elif features.question_markers > 0:
            return 'informative'
        else:
            return 'factual'
    
    def _predict_response_structure(self, features: PromptFeatures) -> str:
        """Predict expected response structure"""
        if 'list' in features.template_category or features.instruction_markers > 0:
            return 'list'
        elif features.multi_step:
            return 'steps'
        elif 'compare' in features.template_category:
            return 'comparison'
        else:
            return 'paragraph'
    
    def _estimate_diversity(self, features: PromptFeatures) -> float:
        """Estimate response diversity based on prompt characteristics"""
        diversity_score = 0.5  # Base diversity
        
        # Higher diversity for creative prompts
        if 'creative' in features.domain_indicators:
            diversity_score += 0.2
        
        # Higher diversity for open-ended questions
        if features.question_markers > 0 and not features.has_constraints:
            diversity_score += 0.1
        
        # Lower diversity for technical prompts
        if features.technical_terms > 2:
            diversity_score -= 0.1
        
        return np.clip(diversity_score, 0, 1)


class ResponsePredictor:
    """Main response prediction system integrating all components"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/response_predictor")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.feature_extractor = FeatureExtractor()
        self.prediction_models = LightweightPredictionModels()
        
        # Historical data storage
        self.historical_responses: List[HistoricalResponse] = []
        self.template_response_mapping: Dict[str, List[HistoricalResponse]] = defaultdict(list)
        
        # Pattern recognition components
        self.prompt_clusters: Dict[int, List[str]] = {}
        self.cluster_model: Optional[KMeans] = None
        self.similarity_threshold = 0.8
        
        # Optimization integration
        self.cost_benefit_analyzer = CostBenefitAnalyzer()
        self.priority_queue = PriorityQueue()
        
        # Accuracy improvement
        self.online_learner = OnlineLearner()
        self.ensemble = EnsemblePredictor()
        
        # Performance tracking
        self.prediction_history: deque = deque(maxlen=1000)
        self.accuracy_metrics = {
            'length_mae': 0.0,
            'time_mae': 0.0,
            'coherence_mae': 0.0
        }
        
        # Load cached data
        self._load_cache()
    
    def add_historical_response(self, historical_response: HistoricalResponse) -> None:
        """Add historical response data for learning"""
        self.historical_responses.append(historical_response)
        
        # Update template mapping
        if historical_response.template_id:
            self.template_response_mapping[historical_response.template_id].append(historical_response)
        
        # Update online learner
        self.online_learner.update(historical_response)
        
        # Retrain if we have enough new data
        if len(self.historical_responses) % 50 == 0:
            self._retrain_models()
    
    def predict_response(self, prompt: str, template_id: Optional[str] = None, 
                        use_ensemble: bool = True) -> ResponsePrediction:
        """Predict response characteristics for a given prompt"""
        # Extract features
        features = self.feature_extractor.extract_features(prompt, template_id)
        
        # Try template-based prediction first
        if template_id and template_id in self.template_response_mapping:
            template_prediction = self._predict_from_template(features, template_id)
            if template_prediction.prediction_confidence > 0.8:
                return template_prediction
        
        # Try similarity-based prediction
        similarity_prediction = self._predict_from_similarity(features, prompt)
        
        # Use ML models
        if self.prediction_models.is_trained:
            ml_prediction = self.prediction_models.predict(features)
        else:
            ml_prediction = self._fallback_prediction(features)
        
        # Ensemble prediction
        if use_ensemble and self.prediction_models.is_trained:
            final_prediction = self.ensemble.combine_predictions([
                similarity_prediction,
                ml_prediction
            ], features)
        else:
            final_prediction = ml_prediction
        
        # Add to prediction history
        self.prediction_history.append({
            'prompt': prompt,
            'prediction': final_prediction,
            'timestamp': time.time()
        })
        
        return final_prediction
    
    def _predict_from_template(self, features: PromptFeatures, template_id: str) -> ResponsePrediction:
        """Predict response based on template history"""
        template_responses = self.template_response_mapping[template_id]
        
        if not template_responses:
            return self._fallback_prediction(features)
        
        # Calculate statistics from template history
        lengths = [r.actual_length for r in template_responses]
        word_counts = [r.actual_word_count for r in template_responses]
        tokens = [r.actual_tokens for r in template_responses]
        times = [r.execution_time for r in template_responses]
        coherence_scores = [r.coherence_score for r in template_responses]
        
        estimated_length = statistics.mean(lengths)
        estimated_word_count = statistics.mean(word_counts)
        estimated_tokens = int(statistics.mean(tokens))
        estimated_computation_time = statistics.mean(times)
        predicted_coherence = statistics.mean(coherence_scores)
        
        # Calculate confidence based on template history size and variance
        confidence = min(0.9, len(template_responses) / 20.0)
        if len(template_responses) > 1:
            length_std = statistics.stdev(lengths)
            confidence *= max(0.5, 1.0 - (length_std / estimated_length))
        
        return ResponsePrediction(
            estimated_length=estimated_length,
            estimated_word_count=estimated_word_count,
            estimated_complexity=features.flesch_kincaid_grade,
            estimated_tokens=estimated_tokens,
            estimated_computation_time=estimated_computation_time,
            estimated_memory_usage=estimated_tokens * 0.004,
            predicted_coherence=predicted_coherence,
            predicted_informativeness=0.7,  # Default
            predicted_diversity=0.6,  # Default
            confidence_interval=(estimated_length * 0.8, estimated_length * 1.2),
            prediction_confidence=confidence,
            uncertainty_score=1.0 - confidence,
            response_type='factual',
            expected_structure='paragraph',
            prediction_method="template_history",
            similar_prompts_count=len(template_responses)
        )
    
    def _predict_from_similarity(self, features: PromptFeatures, prompt: str) -> ResponsePrediction:
        """Predict response based on similar historical prompts"""
        if not self.historical_responses or not features.tfidf_vector is not None:
            return self._fallback_prediction(features)
        
        # Find similar prompts
        similarities = []
        for response in self.historical_responses:
            if response.features.tfidf_vector is not None:
                similarity = 1 - cosine(features.tfidf_vector, response.features.tfidf_vector)
                similarities.append((similarity, response))
        
        if not similarities:
            return self._fallback_prediction(features)
        
        # Get top similar responses
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similar = [resp for sim, resp in similarities[:5] if sim > self.similarity_threshold]
        
        if not top_similar:
            return self._fallback_prediction(features)
        
        # Calculate weighted averages
        total_weight = 0
        weighted_length = 0
        weighted_word_count = 0
        weighted_tokens = 0
        weighted_time = 0
        weighted_coherence = 0
        
        for similarity, response in similarities[:5]:
            if similarity > self.similarity_threshold:
                weight = similarity
                total_weight += weight
                weighted_length += response.actual_length * weight
                weighted_word_count += response.actual_word_count * weight
                weighted_tokens += response.actual_tokens * weight
                weighted_time += response.execution_time * weight
                weighted_coherence += response.coherence_score * weight
        
        if total_weight == 0:
            return self._fallback_prediction(features)
        
        estimated_length = weighted_length / total_weight
        estimated_word_count = weighted_word_count / total_weight
        estimated_tokens = int(weighted_tokens / total_weight)
        estimated_computation_time = weighted_time / total_weight
        predicted_coherence = weighted_coherence / total_weight
        
        confidence = min(0.85, total_weight / len(similarities[:5]))
        
        return ResponsePrediction(
            estimated_length=estimated_length,
            estimated_word_count=estimated_word_count,
            estimated_complexity=features.flesch_kincaid_grade,
            estimated_tokens=estimated_tokens,
            estimated_computation_time=estimated_computation_time,
            estimated_memory_usage=estimated_tokens * 0.004,
            predicted_coherence=predicted_coherence,
            predicted_informativeness=0.7,
            predicted_diversity=0.6,
            confidence_interval=(estimated_length * 0.85, estimated_length * 1.15),
            prediction_confidence=confidence,
            uncertainty_score=1.0 - confidence,
            response_type='factual',
            expected_structure='paragraph',
            prediction_method="similarity_based",
            similar_prompts_count=len(top_similar)
        )
    
    def _fallback_prediction(self, features: PromptFeatures) -> ResponsePrediction:
        """Fallback prediction based on heuristics"""
        # Simple heuristic-based prediction
        base_length = features.length * 2.5  # Assume response is 2.5x prompt length
        estimated_word_count = features.word_count * 3.0
        estimated_tokens = int(estimated_word_count * 1.3)  # Approximate tokens
        
        # Time estimation based on complexity
        base_time = 0.5  # Base 0.5 seconds
        complexity_multiplier = 1 + (features.difficulty_level - 1) * 0.3
        estimated_computation_time = base_time * complexity_multiplier
        
        return ResponsePrediction(
            estimated_length=base_length,
            estimated_word_count=estimated_word_count,
            estimated_complexity=features.flesch_kincaid_grade,
            estimated_tokens=estimated_tokens,
            estimated_computation_time=estimated_computation_time,
            estimated_memory_usage=estimated_tokens * 0.004,
            predicted_coherence=0.7,
            predicted_informativeness=0.6,
            predicted_diversity=0.5,
            confidence_interval=(base_length * 0.7, base_length * 1.3),
            prediction_confidence=0.4,  # Low confidence for heuristic
            uncertainty_score=0.6,
            response_type='factual',
            expected_structure='paragraph',
            prediction_method="heuristic_fallback",
            similar_prompts_count=0
        )
    
    def train_models(self, force_retrain: bool = False) -> Dict[str, float]:
        """Train or retrain prediction models"""
        if not self.historical_responses:
            raise ValueError("No historical data available for training")
        
        if self.prediction_models.is_trained and not force_retrain:
            return self.accuracy_metrics
        
        # Fit feature extractor
        prompts = [response.prompt for response in self.historical_responses]
        self.feature_extractor.fit(prompts)
        
        # Train prediction models
        performance = self.prediction_models.train(self.historical_responses)
        
        # Update accuracy metrics
        self.accuracy_metrics.update(performance)
        
        # Train clustering model for pattern recognition
        self._train_clustering()
        
        # Save cache
        self._save_cache()
        
        return performance
    
    def _retrain_models(self) -> None:
        """Retrain models with new data"""
        try:
            self.train_models(force_retrain=True)
        except Exception as e:
            print(f"Failed to retrain models: {e}")
    
    def _train_clustering(self) -> None:
        """Train clustering model for pattern recognition"""
        if len(self.historical_responses) < 10:
            return
        
        # Create feature matrix for clustering
        features_matrix = []
        for response in self.historical_responses:
            if response.features.tfidf_vector is not None:
                features_matrix.append(response.features.tfidf_vector)
        
        if len(features_matrix) < 10:
            return
        
        # Train clustering model
        n_clusters = min(10, len(features_matrix) // 3)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        
        cluster_labels = self.cluster_model.fit_predict(features_matrix)
        
        # Group responses by cluster
        self.prompt_clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if i < len(self.historical_responses):
                self.prompt_clusters[label].append(self.historical_responses[i].prompt)
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get current prediction accuracy metrics"""
        return self.accuracy_metrics.copy()
    
    def _save_cache(self) -> None:
        """Save models and data to cache"""
        try:
            cache_data = {
                'historical_responses': self.historical_responses[-1000:],  # Keep last 1000
                'template_mapping': dict(self.template_response_mapping),
                'accuracy_metrics': self.accuracy_metrics
            }
            
            with open(self.cache_dir / 'response_predictor_cache.pkl', 'wb') as f:
                pickle.dump(cache_data, f)
                
            # Save models
            if self.prediction_models.is_trained:
                with open(self.cache_dir / 'prediction_models.pkl', 'wb') as f:
                    pickle.dump(self.prediction_models, f)
                    
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load models and data from cache"""
        try:
            cache_file = self.cache_dir / 'response_predictor_cache.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.historical_responses = cache_data.get('historical_responses', [])
                self.template_response_mapping = defaultdict(list, cache_data.get('template_mapping', {}))
                self.accuracy_metrics = cache_data.get('accuracy_metrics', self.accuracy_metrics)
            
            # Load models
            models_file = self.cache_dir / 'prediction_models.pkl'
            if models_file.exists():
                with open(models_file, 'rb') as f:
                    self.prediction_models = pickle.load(f)
                    
        except Exception as e:
            print(f"Failed to load cache: {e}")


class CostBenefitAnalyzer:
    """Analyzes cost-benefit for prompt execution decisions"""
    
    def __init__(self):
        self.execution_costs = {
            'computation_time': 1.0,  # Cost per second
            'memory_usage': 0.1,      # Cost per MB
            'api_calls': 0.01         # Cost per API call
        }
        
        self.information_value_weights = {
            'novelty': 0.3,
            'diversity': 0.25,
            'complexity': 0.2,
            'coverage': 0.25
        }
    
    def analyze(self, prediction: ResponsePrediction, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze cost-benefit for executing a prompt"""
        # Calculate costs
        execution_cost = (
            prediction.estimated_computation_time * self.execution_costs['computation_time'] +
            prediction.estimated_memory_usage * self.execution_costs['memory_usage'] +
            self.execution_costs['api_calls']
        )
        
        # Calculate benefits (information value)
        novelty_score = 1.0 - prediction.prediction_confidence  # More novel if less predictable
        diversity_score = prediction.predicted_diversity
        complexity_score = min(1.0, prediction.estimated_complexity / 15.0)  # Normalized complexity
        coverage_score = context.get('coverage_gap', 0.5)  # How much this fills coverage gaps
        
        information_value = (
            novelty_score * self.information_value_weights['novelty'] +
            diversity_score * self.information_value_weights['diversity'] +
            complexity_score * self.information_value_weights['complexity'] +
            coverage_score * self.information_value_weights['coverage']
        )
        
        # Calculate cost-benefit ratio
        cost_benefit_ratio = information_value / max(execution_cost, 0.01)
        
        return {
            'execution_cost': execution_cost,
            'information_value': information_value,
            'cost_benefit_ratio': cost_benefit_ratio,
            'novelty_score': novelty_score,
            'should_execute': cost_benefit_ratio > 1.0
        }


class PriorityQueue:
    """Priority queue for prompt execution based on predicted informativeness"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Tuple[float, str, Dict[str, Any]]] = []
        self.prompt_hashes: Set[str] = set()
    
    def add_prompt(self, prompt: str, prediction: ResponsePrediction, 
                   cost_benefit: Dict[str, float], metadata: Dict[str, Any] = None) -> None:
        """Add prompt to priority queue"""
        # Avoid duplicates
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        if prompt_hash in self.prompt_hashes:
            return
        
        # Calculate priority (higher is better)
        priority = cost_benefit['cost_benefit_ratio']
        
        # Add uncertainty bonus for diverse exploration
        priority += prediction.uncertainty_score * 0.2
        
        # Add to queue
        import heapq
        heapq.heappush(self.queue, (-priority, prompt, {
            'prediction': prediction,
            'cost_benefit': cost_benefit,
            'metadata': metadata or {},
            'timestamp': time.time()
        }))
        
        self.prompt_hashes.add(prompt_hash)
        
        # Maintain max size
        while len(self.queue) > self.max_size:
            _, old_prompt, _ = heapq.heappop(self.queue)
            old_hash = hashlib.md5(old_prompt.encode()).hexdigest()
            self.prompt_hashes.discard(old_hash)
    
    def get_next_prompts(self, n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Get next n highest priority prompts"""
        import heapq
        results = []
        temp_queue = []
        
        for _ in range(min(n, len(self.queue))):
            if self.queue:
                priority, prompt, data = heapq.heappop(self.queue)
                results.append((prompt, data))
                temp_queue.append((priority, prompt, data))
        
        # Restore items to queue
        for item in temp_queue:
            heapq.heappush(self.queue, item)
        
        return results
    
    def size(self) -> int:
        return len(self.queue)


class OnlineLearner:
    """Online learning system for continuous accuracy improvement"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.prediction_errors: deque = deque(maxlen=1000)
        self.feature_importance: Dict[str, float] = defaultdict(float)
        self.adaptation_rate = 0.95  # Exponential decay for old predictions
        
        # Error tracking
        self.length_errors: deque = deque(maxlen=100)
        self.time_errors: deque = deque(maxlen=100)
        self.coherence_errors: deque = deque(maxlen=100)
    
    def update(self, historical_response: HistoricalResponse) -> None:
        """Update learning from new historical response"""
        # This would typically update model weights online
        # For now, we track errors for model retraining triggers
        
        # Calculate errors if we have predictions for this response
        # (This would require storing predictions keyed by prompt hash)
        pass
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained based on error trends"""
        if len(self.prediction_errors) < 20:
            return False
        
        # Check if error is increasing
        recent_errors = list(self.prediction_errors)[-10:]
        older_errors = list(self.prediction_errors)[-20:-10]
        
        recent_avg = statistics.mean(recent_errors)
        older_avg = statistics.mean(older_errors)
        
        return recent_avg > older_avg * 1.2  # 20% increase in error


class EnsemblePredictor:
    """Ensemble methods combining multiple predictors"""
    
    def __init__(self):
        self.method_weights = {
            'template_history': 0.4,
            'similarity_based': 0.35,
            'ensemble_ml': 0.25
        }
        self.confidence_threshold = 0.7
    
    def combine_predictions(self, predictions: List[ResponsePrediction], 
                           features: PromptFeatures) -> ResponsePrediction:
        """Combine multiple predictions using weighted ensemble"""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        if len(predictions) == 1:
            return predictions[0]
        
        # Weight predictions by confidence and method
        total_weight = 0
        weighted_length = 0
        weighted_word_count = 0
        weighted_tokens = 0
        weighted_time = 0
        weighted_coherence = 0
        weighted_informativeness = 0
        
        combined_confidence = 0
        
        for pred in predictions:
            method_weight = self.method_weights.get(pred.prediction_method, 0.1)
            confidence_weight = pred.prediction_confidence
            final_weight = method_weight * confidence_weight
            
            total_weight += final_weight
            weighted_length += pred.estimated_length * final_weight
            weighted_word_count += pred.estimated_word_count * final_weight
            weighted_tokens += pred.estimated_tokens * final_weight
            weighted_time += pred.estimated_computation_time * final_weight
            weighted_coherence += pred.predicted_coherence * final_weight
            weighted_informativeness += pred.predicted_informativeness * final_weight
            
            combined_confidence += pred.prediction_confidence * final_weight
        
        if total_weight == 0:
            return predictions[0]  # Fallback to first prediction
        
        # Calculate ensemble averages
        ensemble_length = weighted_length / total_weight
        ensemble_word_count = weighted_word_count / total_weight
        ensemble_tokens = int(weighted_tokens / total_weight)
        ensemble_time = weighted_time / total_weight
        ensemble_coherence = weighted_coherence / total_weight
        ensemble_informativeness = weighted_informativeness / total_weight
        ensemble_confidence = combined_confidence / total_weight
        
        # Use the prediction with highest confidence for categorical fields
        best_pred = max(predictions, key=lambda p: p.prediction_confidence)
        
        return ResponsePrediction(
            estimated_length=ensemble_length,
            estimated_word_count=ensemble_word_count,
            estimated_complexity=best_pred.estimated_complexity,
            estimated_tokens=ensemble_tokens,
            estimated_computation_time=ensemble_time,
            estimated_memory_usage=ensemble_tokens * 0.004,
            predicted_coherence=ensemble_coherence,
            predicted_informativeness=ensemble_informativeness,
            predicted_diversity=best_pred.predicted_diversity,
            confidence_interval=(ensemble_length * 0.8, ensemble_length * 1.2),
            prediction_confidence=ensemble_confidence,
            uncertainty_score=1.0 - ensemble_confidence,
            response_type=best_pred.response_type,
            expected_structure=best_pred.expected_structure,
            prediction_method="ensemble_weighted",
            similar_prompts_count=sum(p.similar_prompts_count for p in predictions)
        )


# Example usage and integration functions
def integrate_with_rev_pipeline(predictor: ResponsePredictor, 
                               prompts: List[str], 
                               execution_budget: Dict[str, float]) -> List[str]:
    """Integrate predictor with REV pipeline for prompt pre-filtering"""
    cost_benefit_analyzer = CostBenefitAnalyzer()
    priority_queue = PriorityQueue()
    
    # Analyze all prompts
    for prompt in prompts:
        prediction = predictor.predict_response(prompt)
        cost_benefit = cost_benefit_analyzer.analyze(prediction, {'coverage_gap': 0.5})
        
        if cost_benefit['should_execute']:
            priority_queue.add_prompt(prompt, prediction, cost_benefit)
    
    # Select prompts within budget
    selected_prompts = []
    total_cost = 0
    
    candidates = priority_queue.get_next_prompts(len(prompts))
    for prompt, data in candidates:
        prompt_cost = data['cost_benefit']['execution_cost']
        
        if total_cost + prompt_cost <= execution_budget.get('max_cost', float('inf')):
            selected_prompts.append(prompt)
            total_cost += prompt_cost
        
        if len(selected_prompts) >= execution_budget.get('max_prompts', 100):
            break
    
    return selected_prompts


if __name__ == "__main__":
    # Example usage
    predictor = ResponsePredictor()
    
    # Example prompt
    prompt = "Explain the concept of machine learning and its applications in modern technology."
    
    # Make prediction
    prediction = predictor.predict_response(prompt)
    print(f"Predicted response length: {prediction.estimated_length:.0f} characters")
    print(f"Estimated computation time: {prediction.estimated_computation_time:.2f} seconds")
    print(f"Prediction confidence: {prediction.prediction_confidence:.2f}")