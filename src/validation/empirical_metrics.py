"""
Empirical validation metrics for REV model fingerprinting.
Includes ROC curve generation, AUC calculation, and classification metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification performance metrics."""
    
    family: str
    precision: float
    recall: float
    f1_score: float
    support: int
    auc_score: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    confusion_matrix: np.ndarray


class EmpiricalValidator:
    """Compute empirical validation metrics for model family classification."""
    
    def __init__(self, reference_library_path: str = "fingerprint_library/reference_library.json"):
        """
        Initialize validator with reference library.
        
        Args:
            reference_library_path: Path to reference fingerprint library
        """
        self.reference_library_path = Path(reference_library_path)
        self.reference_fingerprints = self._load_reference_library()
        self.model_families = list(self.reference_fingerprints.keys())
        
    def _load_reference_library(self) -> Dict[str, Any]:
        """Load reference fingerprint library."""
        if not self.reference_library_path.exists():
            logger.warning(f"Reference library not found at {self.reference_library_path}")
            return {}
            
        with open(self.reference_library_path, 'r') as f:
            library = json.load(f)
            
        # Extract model families from library
        families = {}
        for model_id, data in library.items():
            family = data.get('family', 'unknown')
            if family not in families:
                families[family] = []
            families[family].append(data)
            
        return families
    
    def compute_similarity_scores(
        self,
        test_fingerprint: np.ndarray,
        reference_fingerprints: Dict[str, List[np.ndarray]]
    ) -> Dict[str, float]:
        """
        Compute similarity scores between test and reference fingerprints.
        
        Args:
            test_fingerprint: Test model fingerprint vector
            reference_fingerprints: Reference fingerprints by family
            
        Returns:
            Dictionary of similarity scores by family
        """
        scores = {}
        
        for family, fingerprints in reference_fingerprints.items():
            # Compute average similarity to family
            similarities = []
            for ref_fp in fingerprints:
                # Handle different fingerprint types
                if isinstance(ref_fp, dict):
                    ref_vector = np.array(ref_fp.get('hypervector', []))
                else:
                    ref_vector = np.array(ref_fp)
                
                if len(ref_vector) > 0 and len(test_fingerprint) > 0:
                    # Ensure same dimensions
                    min_len = min(len(test_fingerprint), len(ref_vector))
                    test_vec = test_fingerprint[:min_len]
                    ref_vec = ref_vector[:min_len]
                    
                    # Compute cosine similarity
                    similarity = np.dot(test_vec, ref_vec) / (
                        np.linalg.norm(test_vec) * np.linalg.norm(ref_vec) + 1e-10
                    )
                    similarities.append(similarity)
            
            scores[family] = np.mean(similarities) if similarities else 0.0
            
        return scores
    
    def generate_roc_curves(
        self,
        test_results: List[Dict[str, Any]],
        target_family: Optional[str] = None
    ) -> Dict[str, ClassificationMetrics]:
        """
        Generate ROC curves for model family classification.
        
        Args:
            test_results: List of test results with true and predicted families
            target_family: Specific family to analyze (None for all)
            
        Returns:
            Dictionary of ClassificationMetrics by family
        """
        # Prepare data for ROC analysis
        y_true = []
        y_scores = {}
        
        for result in test_results:
            true_family = result['true_family']
            scores = result['similarity_scores']
            
            y_true.append(true_family)
            for family in self.model_families:
                if family not in y_scores:
                    y_scores[family] = []
                y_scores[family].append(scores.get(family, 0.0))
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(
            y_true, 
            classes=self.model_families
        )
        
        metrics = {}
        
        for i, family in enumerate(self.model_families):
            if target_family and family != target_family:
                continue
                
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(
                y_true_bin[:, i],
                y_scores[family]
            )
            
            # Compute AUC
            auc_score = auc(fpr, tpr)
            
            # Compute optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Generate predictions at optimal threshold
            y_pred = [1 if score >= optimal_threshold else 0 
                     for score in y_scores[family]]
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true_bin[:, i], y_pred)
            
            # Compute precision, recall, F1
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[family] = ClassificationMetrics(
                family=family,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=np.sum(y_true_bin[:, i]),
                auc_score=auc_score,
                fpr=fpr,
                tpr=tpr,
                thresholds=thresholds,
                confusion_matrix=cm
            )
            
        return metrics
    
    def compute_multiclass_metrics(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute multi-class classification metrics.
        
        Args:
            test_results: List of test results
            
        Returns:
            Dictionary containing classification report and confusion matrix
        """
        y_true = []
        y_pred = []
        
        for result in test_results:
            true_family = result['true_family']
            scores = result['similarity_scores']
            
            # Predict family with highest score
            predicted_family = max(scores.items(), key=lambda x: x[1])[0]
            
            y_true.append(true_family)
            y_pred.append(predicted_family)
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.model_families,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.model_families)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def analyze_decision_boundaries(
        self,
        test_results: List[Dict[str, Any]],
        num_thresholds: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Analyze decision boundaries for different threshold values.
        
        Args:
            test_results: List of test results
            num_thresholds: Number of threshold values to evaluate
            
        Returns:
            Dictionary containing threshold analysis results
        """
        thresholds = np.linspace(0, 1, num_thresholds)
        results = {}
        
        for family in self.model_families:
            accuracies = []
            precisions = []
            recalls = []
            
            for threshold in thresholds:
                tp = fp = tn = fn = 0
                
                for result in test_results:
                    true_family = result['true_family']
                    score = result['similarity_scores'].get(family, 0.0)
                    
                    if score >= threshold:
                        if true_family == family:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if true_family == family:
                            fn += 1
                        else:
                            tn += 1
                
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
            
            results[family] = {
                'thresholds': thresholds,
                'accuracies': np.array(accuracies),
                'precisions': np.array(precisions),
                'recalls': np.array(recalls)
            }
        
        return results
    
    def compute_false_positive_rates(
        self,
        test_results: List[Dict[str, Any]],
        confidence_levels: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ) -> Dict[str, Dict[float, float]]:
        """
        Compute false positive rates at different confidence levels.
        
        Args:
            test_results: List of test results
            confidence_levels: List of confidence thresholds
            
        Returns:
            Dictionary of FPR by family and confidence level
        """
        fpr_results = {}
        
        for family in self.model_families:
            fpr_by_confidence = {}
            
            for confidence in confidence_levels:
                fp = 0
                tn = 0
                
                for result in test_results:
                    true_family = result['true_family']
                    score = result['similarity_scores'].get(family, 0.0)
                    
                    if true_family != family:  # Negative case
                        if score >= confidence:
                            fp += 1
                        else:
                            tn += 1
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fpr_by_confidence[confidence] = fpr
            
            fpr_results[family] = fpr_by_confidence
        
        return fpr_results
    
    def export_metrics(
        self,
        metrics: Dict[str, ClassificationMetrics],
        output_path: str = "experiments/results/validation_metrics.json"
    ):
        """Export metrics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        export_data = {}
        for family, metric in metrics.items():
            export_data[family] = {
                'precision': metric.precision,
                'recall': metric.recall,
                'f1_score': metric.f1_score,
                'support': metric.support,
                'auc_score': metric.auc_score,
                'confusion_matrix': metric.confusion_matrix.tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported validation metrics to {output_path}")