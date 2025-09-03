"""
Automatic Feature Discovery and Selection System
Uses mutual information, LASSO, elastic net, and dimensionality reduction
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    SelectKBest, f_classif, chi2
)
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
from umap import UMAP
import shap
from scipy import stats
from scipy.sparse import csr_matrix
import warnings
import logging
from dataclasses import dataclass
import joblib
import json

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection process"""
    selected_indices: List[int]
    importance_scores: np.ndarray
    method: str
    parameters: Dict[str, Any]
    performance_score: float = 0.0


class AutomaticFeaturizer:
    """Automatic feature discovery and selection system"""
    
    def __init__(self, 
                 n_features_to_select: int = 100,
                 selection_method: str = 'mutual_info',
                 reduction_method: str = 'umap'):
        """
        Initialize automatic featurizer
        
        Args:
            n_features_to_select: Number of features to select
            selection_method: Feature selection method
            reduction_method: Dimensionality reduction method
        """
        self.n_features_to_select = n_features_to_select
        self.selection_method = selection_method
        self.reduction_method = reduction_method
        
        # Feature selection models
        self.lasso_model = None
        self.elastic_net_model = None
        self.mutual_info_scores = None
        self.shap_values = None
        
        # Dimensionality reduction models
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None
        
        # Feature statistics
        self.feature_stats = {}
        self.selected_features = []
        self.feature_importance = None
        
    def discover_features_mutual_info(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     task_type: str = 'classification') -> FeatureSelectionResult:
        """
        Discover features using mutual information
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values
            task_type: 'classification' or 'regression'
        """
        logger.info(f"Discovering features using mutual information for {task_type}")
        
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
        self.mutual_info_scores = mi_scores
        
        # Select top features
        top_indices = np.argsort(mi_scores)[-self.n_features_to_select:][::-1]
        
        return FeatureSelectionResult(
            selected_indices=top_indices.tolist(),
            importance_scores=mi_scores,
            method='mutual_information',
            parameters={'task_type': task_type},
            performance_score=np.mean(mi_scores[top_indices])
        )
    
    def select_features_lasso(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             alpha: Optional[float] = None,
                             cv: int = 5) -> FeatureSelectionResult:
        """
        Select features using LASSO regularization
        
        Args:
            X: Feature matrix
            y: Target values
            alpha: Regularization parameter (auto if None)
            cv: Cross-validation folds
        """
        logger.info("Selecting features using LASSO")
        
        if alpha is None:
            # Use cross-validation to find optimal alpha
            lasso_cv = LassoCV(cv=cv, random_state=42, max_iter=10000)
            lasso_cv.fit(X, y)
            alpha = lasso_cv.alpha_
            logger.info(f"Selected alpha: {alpha}")
            
        self.lasso_model = Lasso(alpha=alpha, max_iter=10000)
        self.lasso_model.fit(X, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(self.lasso_model.coef_)
        
        # Select non-zero features
        selected_mask = importance > 0
        selected_indices = np.where(selected_mask)[0]
        
        # If too many features, select top ones
        if len(selected_indices) > self.n_features_to_select:
            top_indices = np.argsort(importance)[-self.n_features_to_select:][::-1]
            selected_indices = top_indices
            
        return FeatureSelectionResult(
            selected_indices=selected_indices.tolist(),
            importance_scores=importance,
            method='lasso',
            parameters={'alpha': alpha, 'cv': cv},
            performance_score=self.lasso_model.score(X, y)
        )
    
    def select_features_elastic_net(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   alpha: Optional[float] = None,
                                   l1_ratio: float = 0.5,
                                   cv: int = 5) -> FeatureSelectionResult:
        """
        Select features using Elastic Net (combines L1 and L2 regularization)
        
        Args:
            X: Feature matrix
            y: Target values  
            alpha: Regularization parameter
            l1_ratio: Balance between L1 and L2 (0=L2, 1=L1)
            cv: Cross-validation folds
        """
        logger.info("Selecting features using Elastic Net")
        
        if alpha is None:
            # Use cross-validation
            elastic_cv = ElasticNetCV(
                l1_ratio=l1_ratio, 
                cv=cv, 
                random_state=42,
                max_iter=10000
            )
            elastic_cv.fit(X, y)
            alpha = elastic_cv.alpha_
            logger.info(f"Selected alpha: {alpha}")
            
        self.elastic_net_model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000
        )
        self.elastic_net_model.fit(X, y)
        
        # Get feature importance
        importance = np.abs(self.elastic_net_model.coef_)
        
        # Select top features
        top_indices = np.argsort(importance)[-self.n_features_to_select:][::-1]
        
        return FeatureSelectionResult(
            selected_indices=top_indices.tolist(),
            importance_scores=importance,
            method='elastic_net',
            parameters={'alpha': alpha, 'l1_ratio': l1_ratio, 'cv': cv},
            performance_score=self.elastic_net_model.score(X, y)
        )
    
    def reduce_dimensions_tsne(self,
                              X: np.ndarray,
                              n_components: int = 2,
                              perplexity: float = 30.0) -> np.ndarray:
        """
        Reduce dimensions using t-SNE
        
        Args:
            X: Feature matrix
            n_components: Number of dimensions
            perplexity: t-SNE perplexity parameter
        """
        logger.info(f"Reducing to {n_components}D using t-SNE")
        
        self.tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
        
        X_reduced = self.tsne_model.fit_transform(X)
        return X_reduced
    
    def reduce_dimensions_umap(self,
                              X: np.ndarray,
                              n_components: int = 50,
                              n_neighbors: int = 15,
                              min_dist: float = 0.1) -> np.ndarray:
        """
        Reduce dimensions using UMAP
        
        Args:
            X: Feature matrix
            n_components: Number of dimensions
            n_neighbors: UMAP neighbors parameter
            min_dist: UMAP minimum distance parameter
        """
        logger.info(f"Reducing to {n_components}D using UMAP")
        
        self.umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        X_reduced = self.umap_model.fit_transform(X)
        return X_reduced
    
    def reduce_dimensions_pca(self,
                             X: np.ndarray,
                             n_components: Optional[int] = None,
                             variance_threshold: float = 0.95) -> np.ndarray:
        """
        Reduce dimensions using PCA
        
        Args:
            X: Feature matrix
            n_components: Number of components (auto if None)
            variance_threshold: Cumulative variance to preserve
        """
        if n_components is None:
            # Determine components to preserve variance
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            
        logger.info(f"Reducing to {n_components}D using PCA")
        
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_reduced = self.pca_model.fit_transform(X)
        
        logger.info(f"Explained variance: {np.sum(self.pca_model.explained_variance_ratio_):.3f}")
        
        return X_reduced
    
    def compute_shap_values(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          model: Any = None,
                          sample_size: int = 100) -> np.ndarray:
        """
        Compute SHAP values for feature importance
        
        Args:
            X: Feature matrix
            y: Target values
            model: Model to explain (uses elastic net if None)
            sample_size: Number of samples for SHAP
        """
        logger.info("Computing SHAP values for feature importance")
        
        if model is None:
            # Train a simple model if none provided
            if self.elastic_net_model is None:
                self.select_features_elastic_net(X, y)
            model = self.elastic_net_model
            
        # Sample data for SHAP (for efficiency)
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        # Create SHAP explainer
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            # Average absolute SHAP values across samples
            if len(shap_values.shape) > 1:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            else:
                feature_importance = np.abs(shap_values)
                
            self.shap_values = shap_values
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}, using model coefficients")
            # Fallback to model coefficients
            if hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_)
            else:
                feature_importance = np.ones(X.shape[1]) / X.shape[1]
                
        return feature_importance
    
    def ensemble_feature_selection(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  methods: List[str] = None) -> FeatureSelectionResult:
        """
        Ensemble multiple feature selection methods
        
        Args:
            X: Feature matrix
            y: Target values
            methods: List of methods to ensemble
        """
        if methods is None:
            methods = ['mutual_info', 'lasso', 'elastic_net']
            
        logger.info(f"Ensemble feature selection using {methods}")
        
        all_scores = []
        all_results = []
        
        for method in methods:
            if method == 'mutual_info':
                result = self.discover_features_mutual_info(X, y)
            elif method == 'lasso':
                result = self.select_features_lasso(X, y)
            elif method == 'elastic_net':
                result = self.select_features_elastic_net(X, y)
            else:
                logger.warning(f"Unknown method {method}, skipping")
                continue
                
            all_results.append(result)
            
            # Normalize scores to [0, 1]
            scores = result.importance_scores
            if np.max(scores) > 0:
                scores = scores / np.max(scores)
            all_scores.append(scores)
            
        # Average normalized scores
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Select top features based on ensemble
        top_indices = np.argsort(ensemble_scores)[-self.n_features_to_select:][::-1]
        
        return FeatureSelectionResult(
            selected_indices=top_indices.tolist(),
            importance_scores=ensemble_scores,
            method='ensemble',
            parameters={'methods': methods},
            performance_score=np.mean([r.performance_score for r in all_results])
        )
    
    def fit_transform(self,
                     X: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     selection_method: Optional[str] = None,
                     reduction_method: Optional[str] = None) -> np.ndarray:
        """
        Full pipeline: feature selection + dimensionality reduction
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            selection_method: Override default selection method
            reduction_method: Override default reduction method
        """
        selection_method = selection_method or self.selection_method
        reduction_method = reduction_method or self.reduction_method
        
        # Feature selection
        if y is not None:
            if selection_method == 'mutual_info':
                result = self.discover_features_mutual_info(X, y)
            elif selection_method == 'lasso':
                result = self.select_features_lasso(X, y)
            elif selection_method == 'elastic_net':
                result = self.select_features_elastic_net(X, y)
            elif selection_method == 'ensemble':
                result = self.ensemble_feature_selection(X, y)
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
                
            self.selected_features = result.selected_indices
            self.feature_importance = result.importance_scores
            
            # Select features
            X_selected = X[:, self.selected_features]
        else:
            X_selected = X
            
        # Dimensionality reduction
        if reduction_method == 'pca':
            X_reduced = self.reduce_dimensions_pca(X_selected)
        elif reduction_method == 'tsne':
            X_reduced = self.reduce_dimensions_tsne(X_selected)
        elif reduction_method == 'umap':
            X_reduced = self.reduce_dimensions_umap(X_selected)
        elif reduction_method == 'none':
            X_reduced = X_selected
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")
            
        return X_reduced
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted models
        
        Args:
            X: Feature matrix
        """
        # Apply feature selection
        if self.selected_features:
            X = X[:, self.selected_features]
            
        # Apply dimensionality reduction
        if self.pca_model is not None:
            X = self.pca_model.transform(X)
        elif self.umap_model is not None:
            X = self.umap_model.transform(X)
            
        return X
    
    def analyze_feature_stability(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 n_iterations: int = 10,
                                 subsample_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Analyze feature selection stability across subsamples
        
        Args:
            X: Feature matrix
            y: Target values
            n_iterations: Number of bootstrap iterations
            subsample_ratio: Ratio of samples to use
        """
        logger.info(f"Analyzing feature stability over {n_iterations} iterations")
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        subsample_size = int(n_samples * subsample_ratio)
        
        feature_counts = np.zeros(n_features)
        importance_accumulator = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, subsample_size, replace=True)
            X_sub = X[indices]
            y_sub = y[indices]
            
            # Run feature selection
            result = self.ensemble_feature_selection(X_sub, y_sub)
            
            # Track selected features
            for idx in result.selected_indices:
                feature_counts[idx] += 1
                
            importance_accumulator.append(result.importance_scores)
            
        # Compute stability metrics
        selection_frequency = feature_counts / n_iterations
        importance_mean = np.mean(importance_accumulator, axis=0)
        importance_std = np.std(importance_accumulator, axis=0)
        
        # Identify stable features (selected >50% of the time)
        stable_features = np.where(selection_frequency > 0.5)[0]
        
        return {
            'selection_frequency': selection_frequency,
            'importance_mean': importance_mean,
            'importance_std': importance_std,
            'stable_features': stable_features.tolist(),
            'stability_score': np.mean(selection_frequency[stable_features]) if len(stable_features) > 0 else 0
        }
    
    def save_model(self, filepath: str):
        """Save fitted featurizer models"""
        model_data = {
            'n_features_to_select': self.n_features_to_select,
            'selection_method': self.selection_method,
            'reduction_method': self.reduction_method,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'feature_stats': self.feature_stats
        }
        
        # Save main config
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        # Save sklearn models
        if self.lasso_model:
            joblib.dump(self.lasso_model, filepath.replace('.json', '_lasso.pkl'))
        if self.elastic_net_model:
            joblib.dump(self.elastic_net_model, filepath.replace('.json', '_elastic.pkl'))
        if self.pca_model:
            joblib.dump(self.pca_model, filepath.replace('.json', '_pca.pkl'))
        if self.umap_model:
            joblib.dump(self.umap_model, filepath.replace('.json', '_umap.pkl'))
            
        logger.info(f"Saved featurizer models to {filepath}")
        
    @classmethod
    def load_model(cls, filepath: str) -> 'AutomaticFeaturizer':
        """Load fitted featurizer models"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            
        featurizer = cls(
            n_features_to_select=model_data['n_features_to_select'],
            selection_method=model_data['selection_method'],
            reduction_method=model_data['reduction_method']
        )
        
        featurizer.selected_features = model_data.get('selected_features', [])
        if model_data.get('feature_importance'):
            featurizer.feature_importance = np.array(model_data['feature_importance'])
        featurizer.feature_stats = model_data.get('feature_stats', {})
        
        # Load sklearn models if they exist
        import os
        base_path = filepath.replace('.json', '')
        
        if os.path.exists(f"{base_path}_lasso.pkl"):
            featurizer.lasso_model = joblib.load(f"{base_path}_lasso.pkl")
        if os.path.exists(f"{base_path}_elastic.pkl"):
            featurizer.elastic_net_model = joblib.load(f"{base_path}_elastic.pkl")
        if os.path.exists(f"{base_path}_pca.pkl"):
            featurizer.pca_model = joblib.load(f"{base_path}_pca.pkl")
        if os.path.exists(f"{base_path}_umap.pkl"):
            featurizer.umap_model = joblib.load(f"{base_path}_umap.pkl")
            
        logger.info(f"Loaded featurizer models from {filepath}")
        return featurizer