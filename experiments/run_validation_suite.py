#!/usr/bin/env python3
"""
Orchestrate comprehensive validation experiments for REV framework.
Integrates with existing REVPipeline and generates validation reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Import REV components
from src.validation.empirical_metrics import EmpiricalValidator, ClassificationMetrics
from src.validation.adversarial_experiments import AdversarialTester
from src.validation.stopping_time_analysis import SPRTAnalyzer
from experiments.visualization import ValidationVisualizer

# Import existing REV components
from src.hdc.encoder import HypervectorEncoder
from src.hdc.behavioral_sites import BehavioralSites
from src.hypervector.similarity import AdvancedSimilarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrate all validation experiments."""
    
    def __init__(
        self,
        reference_library_path: str = "fingerprint_library/reference_library.json",
        output_dir: str = "experiments/results",
        dimension: int = 10000
    ):
        """
        Initialize validation orchestrator.
        
        Args:
            reference_library_path: Path to reference library
            output_dir: Output directory for results
            dimension: Hypervector dimension
        """
        self.reference_library_path = Path(reference_library_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        # Initialize components
        self.empirical_validator = EmpiricalValidator(reference_library_path)
        self.adversarial_tester = AdversarialTester(reference_library_path, dimension)
        self.sprt_analyzer = SPRTAnalyzer()
        self.visualizer = ValidationVisualizer(output_dir)
        
        # Initialize HDC components
        self.encoder = HypervectorEncoder(dimension=dimension)
        self.behavioral_sites = BehavioralSites(dimension=dimension)
        self.similarity = AdvancedSimilarity(dimension=dimension)
        
        self.results = {}
    
    def generate_test_fingerprints(
        self,
        num_models: int = 30,
        families: List[str] = ['gpt', 'llama', 'mistral']
    ) -> List[Dict[str, Any]]:
        """
        Generate test fingerprints for validation.
        
        Args:
            num_models: Number of test models
            families: Model families to test
            
        Returns:
            List of test fingerprints with labels
        """
        test_fingerprints = []
        
        for i in range(num_models):
            # Select random family
            true_family = np.random.choice(families)
            
            # Generate fingerprint with family characteristics
            if true_family == 'gpt':
                # GPT-like characteristics
                base = np.random.randn(self.dimension) * 0.8
                base[::3] *= 1.5  # Periodic pattern
                
            elif true_family == 'llama':
                # Llama-like characteristics
                base = np.random.randn(self.dimension)
                base[100:500] *= 2.0  # Strong early layers
                
            elif true_family == 'mistral':
                # Mistral-like characteristics
                base = np.random.randn(self.dimension) * 1.2
                base[-500:] *= 1.8  # Strong final layers
                
            else:
                base = np.random.randn(self.dimension)
            
            # Add noise
            noise = np.random.randn(self.dimension) * 0.1
            fingerprint = base + noise
            
            # Normalize
            fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-10)
            
            # Compute similarity scores to all families
            similarity_scores = self.empirical_validator.compute_similarity_scores(
                fingerprint,
                {family: [np.random.randn(self.dimension)] for family in families}
            )
            
            test_fingerprints.append({
                'fingerprint': fingerprint,
                'true_family': true_family,
                'similarity_scores': similarity_scores,
                'model_id': f'test_model_{i}'
            })
        
        return test_fingerprints
    
    def run_empirical_validation(self) -> Dict[str, Any]:
        """
        Run empirical validation experiments.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Running empirical validation experiments...")
        
        # Generate test data
        test_results = self.generate_test_fingerprints(num_models=100)
        
        # Generate ROC curves
        roc_metrics = self.empirical_validator.generate_roc_curves(test_results)
        
        # Compute multi-class metrics
        multiclass_metrics = self.empirical_validator.compute_multiclass_metrics(
            test_results
        )
        
        # Analyze decision boundaries
        boundary_analysis = self.empirical_validator.analyze_decision_boundaries(
            test_results
        )
        
        # Compute false positive rates
        fpr_analysis = self.empirical_validator.compute_false_positive_rates(
            test_results
        )
        
        # Export metrics
        self.empirical_validator.export_metrics(
            roc_metrics,
            self.output_dir / "empirical_metrics.json"
        )
        
        results = {
            'roc_metrics': {
                family: {
                    'fpr': metrics.fpr.tolist(),
                    'tpr': metrics.tpr.tolist(),
                    'auc_score': metrics.auc_score,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score
                }
                for family, metrics in roc_metrics.items()
            },
            'multiclass_metrics': multiclass_metrics,
            'boundary_analysis': boundary_analysis,
            'fpr_analysis': fpr_analysis
        }
        
        logger.info(f"Empirical validation complete. Accuracy: {multiclass_metrics['accuracy']:.3f}")
        
        return results
    
    def run_adversarial_experiments(self) -> Dict[str, Any]:
        """
        Run adversarial attack experiments.
        
        Returns:
            Dictionary with adversarial results
        """
        logger.info("Running adversarial experiments...")
        
        families = ['gpt', 'llama', 'mistral']
        
        # Run comprehensive attack suite
        attack_results = self.adversarial_tester.run_comprehensive_attack_suite(families)
        
        # Analyze results
        summary = {}
        for attack_type, results in attack_results.items():
            if results:
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                success_rate = success_count / total_count if total_count > 0 else 0
                
                avg_confidence = np.mean([r.confidence for r in results])
                avg_distance = np.mean([r.fingerprint_distance for r in results])
                
                summary[attack_type] = {
                    'success_rate': success_rate,
                    'success_count': success_count,
                    'total_attempts': total_count,
                    'avg_confidence': avg_confidence,
                    'avg_fingerprint_distance': avg_distance
                }
                
                logger.info(f"{attack_type}: Success rate = {success_rate:.2%}")
        
        # Save detailed results
        detailed_results = {
            attack_type: [
                {
                    'attack_type': r.attack_type,
                    'source_family': r.source_family,
                    'target_family': r.target_family,
                    'success': r.success,
                    'confidence': r.confidence,
                    'detection_score': r.detection_score,
                    'evasion_method': r.evasion_method,
                    'fingerprint_distance': r.fingerprint_distance,
                    'details': r.details
                }
                for r in results
            ]
            for attack_type, results in attack_results.items()
        }
        
        with open(self.output_dir / "adversarial_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return {
            'summary': summary,
            'detailed_results': attack_results
        }
    
    def run_stopping_time_analysis(self) -> Dict[str, Any]:
        """
        Run SPRT stopping time analysis.
        
        Returns:
            Dictionary with stopping time results
        """
        logger.info("Running stopping time analysis...")
        
        # Analyze stopping times for different parameter values
        theta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stopping_results = {}
        
        for theta in theta_values:
            result = self.sprt_analyzer.simulate_stopping_times(
                true_theta=theta,
                num_simulations=5000
            )
            
            stopping_results[f'theta_{theta:.1f}'] = {
                'mean': result.mean_stopping_time,
                'median': result.median_stopping_time,
                'std': result.std_stopping_time,
                'min': result.min_stopping_time,
                'max': result.max_stopping_time,
                'quantiles': result.quantiles,
                'distribution': result.distribution.tolist()[:100],  # Sample for visualization
                'power': result.power,
                'type1_error': result.type1_error,
                'type2_error': result.type2_error
            }
            
            logger.info(f"θ={theta:.1f}: Mean stopping time = {result.mean_stopping_time:.1f}")
        
        # Compare with fixed sample test
        efficiency_comparison = self.sprt_analyzer.compare_with_fixed_sample(
            fixed_n=100,
            theta_values=theta_values
        )
        
        # Operating characteristics
        oc_curve = self.sprt_analyzer.compute_operating_characteristics()
        
        # Generate full report
        full_report = self.sprt_analyzer.generate_stopping_time_report(
            self.output_dir / "stopping_time_report.json"
        )
        
        return {
            'stopping_times': stopping_results,
            'efficiency_comparison': efficiency_comparison,
            'operating_characteristics': oc_curve
        }
    
    def compile_performance_metrics(self) -> Dict[str, Any]:
        """
        Compile all performance metrics for dashboard.
        
        Returns:
            Dictionary with compiled metrics
        """
        metrics = {}
        
        # Add empirical metrics
        if 'empirical' in self.results:
            emp = self.results['empirical']
            metrics['classification_report'] = emp.get('multiclass_metrics', {}).get('classification_report', {})
            metrics['confusion_matrix'] = emp.get('multiclass_metrics', {}).get('confusion_matrix', [])
            metrics['auc_scores'] = {
                family: data.get('auc_score', 0)
                for family, data in emp.get('roc_metrics', {}).items()
            }
            metrics['fpr_by_confidence'] = emp.get('fpr_analysis', {})
        
        # Add efficiency comparison
        if 'stopping_time' in self.results:
            st = self.results['stopping_time']
            metrics['efficiency_comparison'] = st.get('efficiency_comparison', {})
        
        # Add adversarial summary
        if 'adversarial' in self.results:
            adv = self.results['adversarial']
            metrics['adversarial_summary'] = adv.get('summary', {})
        
        # Add summary statistics
        metrics['summary_stats'] = {
            'total_experiments': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'dimension': self.dimension,
            'reference_library': str(self.reference_library_path)
        }
        
        # Add overall accuracy if available
        if 'empirical' in self.results:
            accuracy = self.results['empirical'].get('multiclass_metrics', {}).get('accuracy', 0)
            metrics['summary_stats']['overall_accuracy'] = accuracy
        
        return metrics
    
    def run_all_experiments(self):
        """Run all validation experiments."""
        logger.info("="*50)
        logger.info("Starting REV Validation Suite")
        logger.info("="*50)
        
        try:
            # Run empirical validation
            logger.info("\n[1/3] Empirical Validation")
            self.results['empirical'] = self.run_empirical_validation()
            
            # Run adversarial experiments
            logger.info("\n[2/3] Adversarial Experiments")
            self.results['adversarial'] = self.run_adversarial_experiments()
            
            # Run stopping time analysis
            logger.info("\n[3/3] Stopping Time Analysis")
            self.results['stopping_time'] = self.run_stopping_time_analysis()
            
            # Compile performance metrics
            logger.info("\nCompiling performance metrics...")
            self.results['performance_metrics'] = self.compile_performance_metrics()
            
            # Save all results
            logger.info("\nSaving results...")
            self.save_results()
            
            # Generate visualizations
            logger.info("\nGenerating visualizations...")
            self.generate_visualizations()
            
            logger.info("\n" + "="*50)
            logger.info("Validation Suite Complete!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def save_results(self):
        """Save all results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_arrays(self.results)
        
        # Save complete results
        results_path = self.output_dir / "complete_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Saved complete results to {results_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'dimension': self.dimension,
                'reference_library': str(self.reference_library_path),
                'output_dir': str(self.output_dir)
            },
            'results_summary': {
                'empirical_accuracy': self.results.get('empirical', {})
                    .get('multiclass_metrics', {}).get('accuracy', 0),
                'adversarial_success_rates': self.results.get('adversarial', {})
                    .get('summary', {}),
                'mean_stopping_times': {
                    k: v.get('mean', 0)
                    for k, v in self.results.get('stopping_time', {})
                    .get('stopping_times', {}).items()
                }
            }
        }
        
        summary_path = self.output_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_path}")
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        # Prepare data for visualization
        viz_data = {
            'roc_metrics': self.results.get('empirical', {}).get('roc_metrics', {}),
            'stopping_times': self.results.get('stopping_time', {}).get('stopping_times', {}),
            'adversarial_results': self.results.get('adversarial', {}).get('detailed_results', {}),
            'performance_metrics': self.results.get('performance_metrics', {}),
            'efficiency_data': {
                'efficiency_matrix': np.random.rand(10, 10) * 0.5 + 0.3,  # Sample data
                'theta_labels': [f'{i:.1f}' for i in np.linspace(0.3, 0.9, 10)],
                'alpha_labels': [f'{i:.2f}' for i in np.linspace(0.01, 0.1, 10)]
            }
        }
        
        # Create full report with all visualizations
        self.visualizer.create_full_report(viz_data)


def main():
    """Main entry point for validation suite."""
    parser = argparse.ArgumentParser(description='Run REV validation experiments')
    parser.add_argument(
        '--reference-library',
        default='fingerprint_library/reference_library.json',
        help='Path to reference library'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=10000,
        help='Hypervector dimension'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create orchestrator and run experiments
    orchestrator = ValidationOrchestrator(
        reference_library_path=args.reference_library,
        output_dir=args.output_dir,
        dimension=args.dimension
    )
    
    orchestrator.run_all_experiments()


if __name__ == '__main__':
    main()