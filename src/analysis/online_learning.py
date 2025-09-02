#!/usr/bin/env python3
"""
Online Learning and Accuracy Improvement for Response Prediction System

This module implements online learning algorithms, ensemble methods, and confidence
estimation to continuously improve prediction accuracy in the REV pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.base import clone
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats
import scipy.stats as scipy_stats

from .response_predictor import ResponsePrediction, HistoricalResponse, PromptFeatures


@dataclass
class PredictionAccuracy:
    """Tracks prediction accuracy over time"""
    timestamp: float
    actual_value: float
    predicted_value: float
    prediction_method: str
    confidence: float
    error: float
    relative_error: float
    
    def __post_init__(self):
        self.error = abs(self.actual_value - self.predicted_value)
        if self.actual_value != 0:
            self.relative_error = self.error / abs(self.actual_value)
        else:
            self.relative_error = 0.0


@dataclass
class ModelPerformance:
    """Model performance metrics over time"""
    model_name: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2_score: float
    confidence_calibration: float  # How well calibrated the confidence is
    prediction_count: int
    last_updated: float
    
    # Trend analysis
    error_trend: float = 0.0  # Positive means errors increasing
    confidence_trend: float = 0.0  # Trend in confidence scores
    performance_stability: float = 1.0  # How stable the performance is


class OnlineLearningEngine:
    """Online learning engine with multiple adaptive algorithms"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # Online learning models
        self.online_models = {
            'sgd_length': SGDRegressor(learning_rate='adaptive', eta0=learning_rate, random_state=42),
            'sgd_time': SGDRegressor(learning_rate='adaptive', eta0=learning_rate, random_state=42),
            'pa_coherence': PassiveAggressiveRegressor(C=1.0, random_state=42),
            'pa_informativeness': PassiveAggressiveRegressor(C=1.0, random_state=42)
        }
        
        # Feature scaling for online models
        self.online_scalers = {
            name: StandardScaler() for name in self.online_models.keys()
        }
        
        # Model initialization states
        self.models_initialized = {name: False for name in self.online_models.keys()}
        
        # Prediction accuracy tracking
        self.accuracy_history: deque = deque(maxlen=1000)
        self.model_performances: Dict[str, ModelPerformance] = {}
        
        # Online statistics
        self.feature_stats = OnlineStatistics()
        self.prediction_stats = OnlineStatistics()
        
        # Adaptive learning parameters
        self.adaptation_window = 50  # Window for adaptation decisions
        self.min_samples_for_learning = 10
        
    def update_online(self, features: PromptFeatures, actual_response: Dict[str, float],
                     predicted_response: ResponsePrediction) -> Dict[str, float]:
        """Update online models with new observation"""
        feature_vector = self._extract_feature_vector(features)
        
        # Update feature statistics
        self.feature_stats.update(feature_vector)
        
        # Prepare targets
        targets = {
            'sgd_length': actual_response.get('length', 0),
            'sgd_time': actual_response.get('execution_time', 0),
            'pa_coherence': actual_response.get('coherence_score', 0),
            'pa_informativeness': actual_response.get('informativeness_score', 0)
        }
        
        # Update each online model
        learning_feedback = {}
        
        for model_name, model in self.online_models.items():
            if model_name in targets:
                target_value = targets[model_name]
                
                # Reshape feature vector
                X = feature_vector.reshape(1, -1)
                y = np.array([target_value])
                
                try:
                    # Initialize model if needed
                    if not self.models_initialized[model_name]:
                        # Scale initial feature vector
                        self.online_scalers[model_name].fit(X)
                        X_scaled = self.online_scalers[model_name].transform(X)
                        
                        # Partial fit
                        model.partial_fit(X_scaled, y)
                        self.models_initialized[model_name] = True
                    else:
                        # Update scaler incrementally
                        X_scaled = self.online_scalers[model_name].transform(X)
                        
                        # Partial fit (online learning)
                        model.partial_fit(X_scaled, y)
                    
                    # Calculate learning feedback
                    predicted_value = getattr(predicted_response, self._get_prediction_attr(model_name), 0)
                    error = abs(target_value - predicted_value)
                    learning_feedback[model_name] = {
                        'error': error,
                        'target': target_value,
                        'predicted': predicted_value,
                        'improvement_potential': self._calculate_improvement_potential(error, model_name)
                    }
                    
                    # Track prediction accuracy
                    accuracy = PredictionAccuracy(
                        timestamp=time.time(),
                        actual_value=target_value,
                        predicted_value=predicted_value,
                        prediction_method=model_name,
                        confidence=predicted_response.prediction_confidence,
                        error=error,
                        relative_error=error / max(abs(target_value), 1e-6)
                    )
                    self.accuracy_history.append(accuracy)
                    
                except Exception as e:
                    learning_feedback[model_name] = {'error': f"Learning failed: {e}"}
        
        # Update model performance metrics
        self._update_model_performance_metrics()
        
        return learning_feedback
    
    def predict_online(self, features: PromptFeatures) -> Dict[str, float]:
        """Make predictions using online models"""
        feature_vector = self._extract_feature_vector(features)
        X = feature_vector.reshape(1, -1)
        
        predictions = {}
        
        for model_name, model in self.online_models.items():
            if self.models_initialized[model_name]:
                try:
                    X_scaled = self.online_scalers[model_name].transform(X)
                    prediction = model.predict(X_scaled)[0]
                    predictions[model_name] = max(0, prediction)  # Ensure non-negative
                except Exception as e:
                    predictions[model_name] = 0.0  # Fallback
            else:
                predictions[model_name] = 0.0  # Model not initialized
        
        return predictions
    
    def get_learning_rate_adaptation(self) -> Dict[str, float]:
        """Adapt learning rates based on recent performance"""
        if len(self.accuracy_history) < self.min_samples_for_learning:
            return {name: self.learning_rate for name in self.online_models.keys()}
        
        adapted_rates = {}
        recent_window = list(self.accuracy_history)[-self.adaptation_window:]
        
        for model_name in self.online_models.keys():
            # Get recent errors for this model
            model_errors = [acc.relative_error for acc in recent_window 
                          if acc.prediction_method == model_name]
            
            if len(model_errors) >= 5:
                # Calculate error trend
                recent_errors = model_errors[-10:]
                older_errors = model_errors[-20:-10] if len(model_errors) >= 20 else model_errors[:-10]
                
                if older_errors:
                    recent_avg = np.mean(recent_errors)
                    older_avg = np.mean(older_errors)
                    error_trend = (recent_avg - older_avg) / max(older_avg, 1e-6)
                    
                    # Adapt learning rate
                    if error_trend > 0.1:  # Errors increasing
                        adapted_rates[model_name] = self.learning_rate * 0.8  # Reduce learning rate
                    elif error_trend < -0.1:  # Errors decreasing
                        adapted_rates[model_name] = self.learning_rate * 1.2  # Increase learning rate
                    else:
                        adapted_rates[model_name] = self.learning_rate  # Keep current rate
                else:
                    adapted_rates[model_name] = self.learning_rate
            else:
                adapted_rates[model_name] = self.learning_rate
        
        return adapted_rates
    
    def should_retrain_base_models(self) -> bool:
        """Determine if base models should be retrained"""
        if len(self.accuracy_history) < 50:
            return False
        
        # Check if online models are consistently outperforming
        recent_accuracy = list(self.accuracy_history)[-20:]
        
        # Calculate average performance by method
        method_performance = defaultdict(list)
        for acc in recent_accuracy:
            method_performance[acc.prediction_method].append(acc.relative_error)
        
        # Check if any online method is significantly better
        online_methods = [name for name in self.online_models.keys()]
        base_methods = ['ensemble_ml', 'template_history', 'similarity_based']
        
        online_avg_errors = []
        base_avg_errors = []
        
        for method, errors in method_performance.items():
            avg_error = np.mean(errors)
            if method in online_methods:
                online_avg_errors.append(avg_error)
            elif method in base_methods:
                base_avg_errors.append(avg_error)
        
        if online_avg_errors and base_avg_errors:
            online_avg = np.mean(online_avg_errors)
            base_avg = np.mean(base_avg_errors)
            
            # If online methods are significantly better, suggest retraining
            improvement = (base_avg - online_avg) / max(base_avg, 1e-6)
            return improvement > 0.15  # 15% improvement threshold
        
        return False
    
    def _extract_feature_vector(self, features: PromptFeatures) -> np.ndarray:
        """Extract numeric feature vector from PromptFeatures"""
        feature_list = [
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
            float(features.multi_step)
        ]
        
        return np.array(feature_list, dtype=np.float64)
    
    def _get_prediction_attr(self, model_name: str) -> str:
        """Map model name to prediction attribute"""
        mapping = {
            'sgd_length': 'estimated_length',
            'sgd_time': 'estimated_computation_time',
            'pa_coherence': 'predicted_coherence',
            'pa_informativeness': 'predicted_informativeness'
        }
        return mapping.get(model_name, 'estimated_length')
    
    def _calculate_improvement_potential(self, error: float, model_name: str) -> float:
        """Calculate improvement potential for a model"""
        if model_name not in self.model_performances:
            return 1.0  # High potential if no history
        
        performance = self.model_performances[model_name]
        if performance.mae == 0:
            return 0.0
        
        # Improvement potential based on current error vs historical average
        improvement_potential = max(0.0, min(1.0, (performance.mae - error) / performance.mae))
        return improvement_potential
    
    def _update_model_performance_metrics(self) -> None:
        """Update performance metrics for all models"""
        if len(self.accuracy_history) < 10:
            return
        
        recent_window = list(self.accuracy_history)[-50:]  # Last 50 predictions
        
        # Group by prediction method
        method_accuracies = defaultdict(list)
        for acc in recent_window:
            method_accuracies[acc.prediction_method].append(acc)
        
        # Update performance for each method
        for method_name, accuracies in method_accuracies.items():
            if len(accuracies) >= 5:
                errors = [acc.error for acc in accuracies]
                relative_errors = [acc.relative_error for acc in accuracies]
                confidences = [acc.confidence for acc in accuracies]
                
                mae = np.mean(errors)
                rmse = np.sqrt(np.mean([e**2 for e in errors]))
                
                # R-squared approximation (simplified)
                actual_values = [acc.actual_value for acc in accuracies]
                predicted_values = [acc.predicted_value for acc in accuracies]
                
                if len(set(actual_values)) > 1:
                    r2 = max(0.0, 1 - np.var(errors) / np.var(actual_values))
                else:
                    r2 = 0.0
                
                # Confidence calibration (how well confidence matches actual accuracy)
                confidence_calibration = 1.0 - np.mean([abs(conf - (1 - rel_err)) 
                                                       for conf, rel_err in zip(confidences, relative_errors)])
                confidence_calibration = max(0.0, min(1.0, confidence_calibration))
                
                # Update or create performance record
                if method_name in self.model_performances:
                    old_perf = self.model_performances[method_name]
                    
                    # Calculate trends
                    error_trend = (mae - old_perf.mae) / max(old_perf.mae, 1e-6)
                    confidence_trend = (np.mean(confidences) - 
                                      (old_perf.confidence_calibration if hasattr(old_perf, 'last_confidence') else np.mean(confidences)))
                    
                    # Performance stability (lower variance = higher stability)
                    performance_stability = 1.0 / (1.0 + np.std(relative_errors))
                else:
                    error_trend = 0.0
                    confidence_trend = 0.0
                    performance_stability = 1.0
                
                self.model_performances[method_name] = ModelPerformance(
                    model_name=method_name,
                    mae=mae,
                    rmse=rmse,
                    r2_score=r2,
                    confidence_calibration=confidence_calibration,
                    prediction_count=len(accuracies),
                    last_updated=time.time(),
                    error_trend=error_trend,
                    confidence_trend=confidence_trend,
                    performance_stability=performance_stability
                )


class OnlineStatistics:
    """Incremental statistics calculation"""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance calculation
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update(self, value: Union[float, np.ndarray]) -> None:
        """Update statistics with new value(s)"""
        if isinstance(value, np.ndarray):
            for v in value.flatten():
                self._update_single(float(v))
        else:
            self._update_single(float(value))
    
    def _update_single(self, value: float) -> None:
        """Update statistics with single value using Welford's algorithm"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def variance(self) -> float:
        """Calculate variance"""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
    def std(self) -> float:
        """Calculate standard deviation"""
        return np.sqrt(self.variance())
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics"""
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.variance(),
            'std': self.std(),
            'min': self.min_val if self.min_val != float('inf') else 0.0,
            'max': self.max_val if self.max_val != float('-inf') else 0.0
        }


class EnsemblePredictor:
    """Advanced ensemble predictor with multiple combination strategies"""
    
    def __init__(self):
        # Base predictors
        self.base_predictors = {}
        
        # Ensemble methods
        self.ensemble_methods = {
            'weighted_average': self._weighted_average_ensemble,
            'stacking': self._stacking_ensemble,
            'adaptive_weighting': self._adaptive_weighting_ensemble,
            'confidence_weighted': self._confidence_weighted_ensemble,
            'performance_weighted': self._performance_weighted_ensemble
        }
        
        # Ensemble weights (learned over time)
        self.method_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.method_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Stacking meta-learner
        self.meta_learner = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.meta_learner_trained = False
        
        # Feature importance for ensemble selection
        self.feature_importance: Dict[str, float] = defaultdict(float)
        
    def add_predictor(self, name: str, predictor: Callable) -> None:
        """Add a base predictor to the ensemble"""
        self.base_predictors[name] = predictor
    
    def combine_predictions(self, predictions: Dict[str, ResponsePrediction], 
                          features: PromptFeatures,
                          method: str = 'adaptive_weighting') -> ResponsePrediction:
        """Combine predictions using specified ensemble method"""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        if method not in self.ensemble_methods:
            method = 'weighted_average'  # Fallback
        
        ensemble_func = self.ensemble_methods[method]
        combined_prediction = ensemble_func(predictions, features)
        
        # Update method performance tracking
        self._update_method_performance(method, combined_prediction, predictions)
        
        return combined_prediction
    
    def _weighted_average_ensemble(self, predictions: Dict[str, ResponsePrediction],
                                  features: PromptFeatures) -> ResponsePrediction:
        """Weighted average ensemble based on prediction confidence"""
        total_weight = 0.0
        weighted_values = defaultdict(float)
        
        # Calculate weights and weighted sums
        for pred_name, prediction in predictions.items():
            weight = prediction.prediction_confidence * self.method_weights[pred_name]
            total_weight += weight
            
            weighted_values['estimated_length'] += prediction.estimated_length * weight
            weighted_values['estimated_word_count'] += prediction.estimated_word_count * weight
            weighted_values['estimated_tokens'] += prediction.estimated_tokens * weight
            weighted_values['estimated_computation_time'] += prediction.estimated_computation_time * weight
            weighted_values['predicted_coherence'] += prediction.predicted_coherence * weight
            weighted_values['predicted_informativeness'] += prediction.predicted_informativeness * weight
        
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
        
        # Normalize weighted sums
        for key in weighted_values:
            weighted_values[key] /= total_weight
        
        # Use the prediction with highest confidence for categorical fields
        best_prediction = max(predictions.values(), key=lambda p: p.prediction_confidence)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean([p.prediction_confidence for p in predictions.values()])
        
        return ResponsePrediction(
            estimated_length=weighted_values['estimated_length'],
            estimated_word_count=weighted_values['estimated_word_count'],
            estimated_complexity=best_prediction.estimated_complexity,
            estimated_tokens=int(weighted_values['estimated_tokens']),
            estimated_computation_time=weighted_values['estimated_computation_time'],
            estimated_memory_usage=int(weighted_values['estimated_tokens']) * 0.004,
            predicted_coherence=weighted_values['predicted_coherence'],
            predicted_informativeness=weighted_values['predicted_informativeness'],
            predicted_diversity=best_prediction.predicted_diversity,
            confidence_interval=(
                weighted_values['estimated_length'] * 0.8,
                weighted_values['estimated_length'] * 1.2
            ),
            prediction_confidence=ensemble_confidence,
            uncertainty_score=1.0 - ensemble_confidence,
            response_type=best_prediction.response_type,
            expected_structure=best_prediction.expected_structure,
            prediction_method="weighted_average_ensemble",
            similar_prompts_count=sum(p.similar_prompts_count for p in predictions.values())
        )
    
    def _stacking_ensemble(self, predictions: Dict[str, ResponsePrediction],
                          features: PromptFeatures) -> ResponsePrediction:
        """Stacking ensemble using meta-learner"""
        if not self.meta_learner_trained:
            # Fallback to weighted average if meta-learner not trained
            return self._weighted_average_ensemble(predictions, features)
        
        # Prepare features for meta-learner
        meta_features = []
        
        # Add base predictions as features
        for pred_name in sorted(predictions.keys()):
            prediction = predictions[pred_name]
            meta_features.extend([
                prediction.estimated_length,
                prediction.estimated_word_count,
                prediction.predicted_coherence,
                prediction.prediction_confidence
            ])
        
        # Add original features
        feature_vector = [
            features.length,
            features.word_count,
            features.difficulty_level,
            features.question_markers + features.instruction_markers
        ]
        meta_features.extend(feature_vector)
        
        # Predict with meta-learner
        try:
            meta_input = np.array(meta_features).reshape(1, -1)
            ensemble_length = self.meta_learner.predict(meta_input)[0]
            
            # Use weighted average for other predictions
            base_ensemble = self._weighted_average_ensemble(predictions, features)
            
            # Override length prediction with meta-learner result
            base_ensemble.estimated_length = max(0, ensemble_length)
            base_ensemble.prediction_method = "stacking_ensemble"
            
            return base_ensemble
            
        except Exception:
            # Fallback to weighted average if meta-learner fails
            return self._weighted_average_ensemble(predictions, features)
    
    def _adaptive_weighting_ensemble(self, predictions: Dict[str, ResponsePrediction],
                                   features: PromptFeatures) -> ResponsePrediction:
        """Adaptive weighting based on recent performance"""
        # Calculate adaptive weights based on recent performance
        adaptive_weights = {}
        total_adaptive_weight = 0.0
        
        for pred_name in predictions.keys():
            if pred_name in self.method_performance_history:
                recent_errors = list(self.method_performance_history[pred_name])[-10:]
                if recent_errors:
                    # Weight inversely proportional to recent average error
                    avg_error = np.mean(recent_errors)
                    adaptive_weights[pred_name] = 1.0 / (1.0 + avg_error)
                else:
                    adaptive_weights[pred_name] = 1.0
            else:
                adaptive_weights[pred_name] = 1.0
            
            total_adaptive_weight += adaptive_weights[pred_name]
        
        # Normalize weights
        if total_adaptive_weight > 0:
            for pred_name in adaptive_weights:
                adaptive_weights[pred_name] /= total_adaptive_weight
        
        # Apply adaptive weights
        weighted_values = defaultdict(float)
        total_confidence = 0.0
        
        for pred_name, prediction in predictions.items():
            weight = adaptive_weights[pred_name]
            
            weighted_values['estimated_length'] += prediction.estimated_length * weight
            weighted_values['estimated_word_count'] += prediction.estimated_word_count * weight
            weighted_values['estimated_tokens'] += prediction.estimated_tokens * weight
            weighted_values['estimated_computation_time'] += prediction.estimated_computation_time * weight
            weighted_values['predicted_coherence'] += prediction.predicted_coherence * weight
            weighted_values['predicted_informativeness'] += prediction.predicted_informativeness * weight
            total_confidence += prediction.prediction_confidence * weight
        
        # Best prediction for categorical fields
        best_prediction = max(predictions.values(), key=lambda p: p.prediction_confidence)
        
        return ResponsePrediction(
            estimated_length=weighted_values['estimated_length'],
            estimated_word_count=weighted_values['estimated_word_count'],
            estimated_complexity=best_prediction.estimated_complexity,
            estimated_tokens=int(weighted_values['estimated_tokens']),
            estimated_computation_time=weighted_values['estimated_computation_time'],
            estimated_memory_usage=int(weighted_values['estimated_tokens']) * 0.004,
            predicted_coherence=weighted_values['predicted_coherence'],
            predicted_informativeness=weighted_values['predicted_informativeness'],
            predicted_diversity=best_prediction.predicted_diversity,
            confidence_interval=(
                weighted_values['estimated_length'] * 0.85,
                weighted_values['estimated_length'] * 1.15
            ),
            prediction_confidence=total_confidence,
            uncertainty_score=1.0 - total_confidence,
            response_type=best_prediction.response_type,
            expected_structure=best_prediction.expected_structure,
            prediction_method="adaptive_weighting_ensemble",
            similar_prompts_count=sum(p.similar_prompts_count for p in predictions.values())
        )
    
    def _confidence_weighted_ensemble(self, predictions: Dict[str, ResponsePrediction],
                                     features: PromptFeatures) -> ResponsePrediction:
        """Ensemble weighted by prediction confidence and uncertainty"""
        # Calculate confidence-based weights
        confidence_weights = {}
        total_weight = 0.0
        
        for pred_name, prediction in predictions.items():
            # Weight based on confidence and inverse uncertainty
            confidence_component = prediction.prediction_confidence
            uncertainty_component = 1.0 - prediction.uncertainty_score
            weight = (confidence_component + uncertainty_component) / 2.0
            
            confidence_weights[pred_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for pred_name in confidence_weights:
                confidence_weights[pred_name] /= total_weight
        
        # Apply weights
        weighted_values = defaultdict(float)
        ensemble_confidence = 0.0
        
        for pred_name, prediction in predictions.items():
            weight = confidence_weights[pred_name]
            
            weighted_values['estimated_length'] += prediction.estimated_length * weight
            weighted_values['estimated_word_count'] += prediction.estimated_word_count * weight
            weighted_values['estimated_tokens'] += prediction.estimated_tokens * weight
            weighted_values['estimated_computation_time'] += prediction.estimated_computation_time * weight
            weighted_values['predicted_coherence'] += prediction.predicted_coherence * weight
            weighted_values['predicted_informativeness'] += prediction.predicted_informativeness * weight
            ensemble_confidence += prediction.prediction_confidence * weight
        
        best_prediction = max(predictions.values(), key=lambda p: p.prediction_confidence)
        
        return ResponsePrediction(
            estimated_length=weighted_values['estimated_length'],
            estimated_word_count=weighted_values['estimated_word_count'],
            estimated_complexity=best_prediction.estimated_complexity,
            estimated_tokens=int(weighted_values['estimated_tokens']),
            estimated_computation_time=weighted_values['estimated_computation_time'],
            estimated_memory_usage=int(weighted_values['estimated_tokens']) * 0.004,
            predicted_coherence=weighted_values['predicted_coherence'],
            predicted_informativeness=weighted_values['predicted_informativeness'],
            predicted_diversity=best_prediction.predicted_diversity,
            confidence_interval=(
                weighted_values['estimated_length'] * 0.9,
                weighted_values['estimated_length'] * 1.1
            ),
            prediction_confidence=ensemble_confidence,
            uncertainty_score=1.0 - ensemble_confidence,
            response_type=best_prediction.response_type,
            expected_structure=best_prediction.expected_structure,
            prediction_method="confidence_weighted_ensemble",
            similar_prompts_count=sum(p.similar_prompts_count for p in predictions.values())
        )
    
    def _performance_weighted_ensemble(self, predictions: Dict[str, ResponsePrediction],
                                     features: PromptFeatures) -> ResponsePrediction:
        """Ensemble weighted by historical performance metrics"""
        performance_weights = {}
        total_weight = 0.0
        
        for pred_name, prediction in predictions.items():
            # Get historical performance
            if pred_name in self.method_performance_history:
                recent_errors = list(self.method_performance_history[pred_name])[-20:]
                if recent_errors:
                    # Weight inversely proportional to error, with recency bias
                    weights_by_recency = np.linspace(0.5, 1.0, len(recent_errors))
                    weighted_errors = np.average(recent_errors, weights=weights_by_recency)
                    performance_weight = 1.0 / (1.0 + weighted_errors)
                else:
                    performance_weight = 0.5  # Default for no history
            else:
                performance_weight = 0.5
            
            # Combine with current confidence
            final_weight = (performance_weight + prediction.prediction_confidence) / 2.0
            
            performance_weights[pred_name] = final_weight
            total_weight += final_weight
        
        # Normalize weights
        if total_weight > 0:
            for pred_name in performance_weights:
                performance_weights[pred_name] /= total_weight
        
        # Apply performance weights
        weighted_values = defaultdict(float)
        ensemble_confidence = 0.0
        
        for pred_name, prediction in predictions.items():
            weight = performance_weights[pred_name]
            
            weighted_values['estimated_length'] += prediction.estimated_length * weight
            weighted_values['estimated_word_count'] += prediction.estimated_word_count * weight
            weighted_values['estimated_tokens'] += prediction.estimated_tokens * weight
            weighted_values['estimated_computation_time'] += prediction.estimated_computation_time * weight
            weighted_values['predicted_coherence'] += prediction.predicted_coherence * weight
            weighted_values['predicted_informativeness'] += prediction.predicted_informativeness * weight
            ensemble_confidence += prediction.prediction_confidence * weight
        
        best_prediction = max(predictions.values(), key=lambda p: p.prediction_confidence)
        
        return ResponsePrediction(
            estimated_length=weighted_values['estimated_length'],
            estimated_word_count=weighted_values['estimated_word_count'],
            estimated_complexity=best_prediction.estimated_complexity,
            estimated_tokens=int(weighted_values['estimated_tokens']),
            estimated_computation_time=weighted_values['estimated_computation_time'],
            estimated_memory_usage=int(weighted_values['estimated_tokens']) * 0.004,
            predicted_coherence=weighted_values['predicted_coherence'],
            predicted_informativeness=weighted_values['predicted_informativeness'],
            predicted_diversity=best_prediction.predicted_diversity,
            confidence_interval=(
                weighted_values['estimated_length'] * 0.88,
                weighted_values['estimated_length'] * 1.12
            ),
            prediction_confidence=ensemble_confidence,
            uncertainty_score=1.0 - ensemble_confidence,
            response_type=best_prediction.response_type,
            expected_structure=best_prediction.expected_structure,
            prediction_method="performance_weighted_ensemble",
            similar_prompts_count=sum(p.similar_prompts_count for p in predictions.values())
        )
    
    def train_meta_learner(self, training_data: List[Tuple[Dict[str, ResponsePrediction], 
                                                          PromptFeatures, float]]) -> float:
        """Train the meta-learner for stacking ensemble"""
        if len(training_data) < 10:
            return 0.0  # Not enough data
        
        X_meta = []
        y_meta = []
        
        for predictions, features, actual_length in training_data:
            # Prepare meta-features
            meta_features = []
            
            # Add base predictions
            for pred_name in sorted(predictions.keys()):
                prediction = predictions[pred_name]
                meta_features.extend([
                    prediction.estimated_length,
                    prediction.estimated_word_count,
                    prediction.predicted_coherence,
                    prediction.prediction_confidence
                ])
            
            # Add original features
            feature_vector = [
                features.length,
                features.word_count,
                features.difficulty_level,
                features.question_markers + features.instruction_markers
            ]
            meta_features.extend(feature_vector)
            
            X_meta.append(meta_features)
            y_meta.append(actual_length)
        
        # Train meta-learner
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        try:
            self.meta_learner.fit(X_meta, y_meta)
            self.meta_learner_trained = True
            
            # Calculate cross-validation score
            cv_score = cross_val_score(self.meta_learner, X_meta, y_meta, cv=3).mean()
            return cv_score
            
        except Exception as e:
            self.meta_learner_trained = False
            return 0.0
    
    def _update_method_performance(self, method: str, combined_prediction: ResponsePrediction,
                                  base_predictions: Dict[str, ResponsePrediction]) -> None:
        """Update performance tracking for ensemble method"""
        # This would be called after getting actual results
        # For now, we estimate performance based on prediction agreement
        
        # Calculate agreement between ensemble and base predictions
        agreements = []
        for pred_name, prediction in base_predictions.items():
            length_agreement = 1.0 - abs(prediction.estimated_length - combined_prediction.estimated_length) / max(combined_prediction.estimated_length, 1.0)
            agreements.append(max(0.0, length_agreement))
        
        avg_agreement = np.mean(agreements) if agreements else 0.5
        
        # Use agreement as a proxy for performance (higher agreement might indicate better ensemble)
        self.method_performance_history[method].append(1.0 - avg_agreement)  # Lower is better
    
    def get_ensemble_performance_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance"""
        summary = {}
        
        for method, error_history in self.method_performance_history.items():
            if error_history:
                errors = list(error_history)
                summary[method] = {
                    'avg_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'recent_trend': np.mean(errors[-5:]) - np.mean(errors[-10:-5]) if len(errors) >= 10 else 0.0,
                    'prediction_count': len(errors),
                    'weight': self.method_weights[method]
                }
        
        return summary


class ConfidenceIntervalEstimator:
    """Estimates confidence intervals for predictions using various methods"""
    
    def __init__(self):
        self.historical_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_distributions: Dict[str, scipy_stats.rv_continuous] = {}
        
    def update_error_history(self, prediction_method: str, predicted_value: float,
                           actual_value: float) -> None:
        """Update error history for a prediction method"""
        error = actual_value - predicted_value
        self.historical_errors[prediction_method].append(error)
        
        # Update error distribution if we have enough data
        if len(self.historical_errors[prediction_method]) >= 20:
            self._fit_error_distribution(prediction_method)
    
    def estimate_confidence_interval(self, prediction: ResponsePrediction,
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """Estimate confidence interval for a prediction"""
        method = prediction.prediction_method
        predicted_value = prediction.estimated_length
        
        if method in self.error_distributions:
            # Use fitted distribution
            distribution = self.error_distributions[method]
            alpha = 1 - confidence_level
            
            try:
                lower_error = distribution.ppf(alpha / 2)
                upper_error = distribution.ppf(1 - alpha / 2)
                
                lower_bound = predicted_value + lower_error
                upper_bound = predicted_value + upper_error
                
                return (max(0, lower_bound), upper_bound)
                
            except Exception:
                # Fallback to empirical method
                return self._empirical_confidence_interval(method, predicted_value, confidence_level)
        
        elif method in self.historical_errors and len(self.historical_errors[method]) >= 5:
            # Use empirical method
            return self._empirical_confidence_interval(method, predicted_value, confidence_level)
        
        else:
            # Default confidence interval based on prediction confidence
            uncertainty_factor = 1.0 - prediction.prediction_confidence
            margin = predicted_value * uncertainty_factor * 0.5
            
            return (max(0, predicted_value - margin), predicted_value + margin)
    
    def _empirical_confidence_interval(self, method: str, predicted_value: float,
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate empirical confidence interval from historical errors"""
        errors = list(self.historical_errors[method])
        
        if len(errors) < 5:
            # Not enough data, use default
            margin = predicted_value * 0.2
            return (max(0, predicted_value - margin), predicted_value + margin)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_error = np.percentile(errors, lower_percentile)
        upper_error = np.percentile(errors, upper_percentile)
        
        lower_bound = predicted_value + lower_error
        upper_bound = predicted_value + upper_error
        
        return (max(0, lower_bound), upper_bound)
    
    def _fit_error_distribution(self, method: str) -> None:
        """Fit a distribution to the error history"""
        errors = list(self.historical_errors[method])
        
        if len(errors) < 10:
            return
        
        # Try different distributions
        distributions = [
            scipy_stats.norm,
            scipy_stats.t,
            scipy_stats.laplace,
            scipy_stats.logistic
        ]
        
        best_fit = None
        best_score = float('inf')
        
        for dist in distributions:
            try:
                params = dist.fit(errors)
                fitted_dist = dist(*params)
                
                # Calculate goodness of fit (Kolmogorov-Smirnov test)
                ks_stat, p_value = scipy_stats.kstest(errors, fitted_dist.cdf)
                
                if ks_stat < best_score:
                    best_score = ks_stat
                    best_fit = fitted_dist
                    
            except Exception:
                continue  # Skip if fitting fails
        
        if best_fit is not None:
            self.error_distributions[method] = best_fit
    
    def get_calibration_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get calibration metrics for each prediction method"""
        calibration = {}
        
        for method, errors in self.historical_errors.items():
            if len(errors) >= 10:
                error_list = list(errors)
                
                calibration[method] = {
                    'mean_error': np.mean(error_list),
                    'std_error': np.std(error_list),
                    'median_error': np.median(error_list),
                    'mae': np.mean(np.abs(error_list)),
                    'rmse': np.sqrt(np.mean(np.square(error_list))),
                    'error_range': np.max(error_list) - np.min(error_list),
                    'has_distribution_fit': method in self.error_distributions
                }
        
        return calibration


if __name__ == "__main__":
    # Example usage
    from .response_predictor import PromptFeatures
    
    # Initialize components
    online_engine = OnlineLearningEngine()
    ensemble = EnsemblePredictor()
    confidence_estimator = ConfidenceIntervalEstimator()
    
    # Example feature
    features = PromptFeatures(
        length=50,
        word_count=10,
        sentence_count=2,
        avg_word_length=5.0,
        flesch_kincaid_grade=8.0,
        syntactic_complexity=0.3,
        vocabulary_diversity=0.8,
        question_markers=1,
        instruction_markers=0,
        technical_terms=1,
        domain_indicators=['technical'],
        template_id='tech_question',
        template_category='question',
        difficulty_level=3,
        has_examples=False,
        requires_reasoning=True,
        has_constraints=False,
        multi_step=False
    )
    
    # Simulate online learning update
    actual_response = {
        'length': 150,
        'execution_time': 1.2,
        'coherence_score': 0.8,
        'informativeness_score': 0.7
    }
    
    # Mock prediction
    from .response_predictor import ResponsePrediction
    mock_prediction = ResponsePrediction(
        estimated_length=140.0,
        estimated_word_count=30.0,
        estimated_complexity=8.0,
        estimated_tokens=35,
        estimated_computation_time=1.1,
        estimated_memory_usage=140.0,
        predicted_coherence=0.75,
        predicted_informativeness=0.65,
        predicted_diversity=0.6,
        confidence_interval=(120.0, 160.0),
        prediction_confidence=0.8,
        uncertainty_score=0.2,
        response_type='informative',
        expected_structure='paragraph',
        prediction_method='test_method',
        similar_prompts_count=5
    )
    
    # Update online learning
    feedback = online_engine.update_online(features, actual_response, mock_prediction)
    print(f"Learning feedback: {feedback}")
    
    # Get online predictions
    online_predictions = online_engine.predict_online(features)
    print(f"Online predictions: {online_predictions}")
    
    # Update confidence interval estimator
    confidence_estimator.update_error_history('test_method', 140.0, 150.0)
    ci = confidence_estimator.estimate_confidence_interval(mock_prediction)
    print(f"Confidence interval: {ci}")