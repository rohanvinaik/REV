#!/usr/bin/env python3
"""
Response Prediction and Analysis Module for REV Pipeline

This module provides comprehensive response prediction capabilities to optimize
prompt selection and execution in the REV framework through intelligent
pre-filtering and resource optimization.
"""

from .response_predictor import (
    ResponsePredictor,
    FeatureExtractor, 
    LightweightPredictionModels,
    ResponsePrediction,
    HistoricalResponse,
    PromptFeatures
)

from .pattern_recognition import (
    TemplateResponseMapper,
    PromptClusterAnalyzer,
    AnomalyDetector,
    PromptPattern,
    ClusterInfo
)

from .optimization_integration import (
    PromptSelectionOptimizer,
    REVPipelineIntegration,
    ExecutionBudget,
    ExecutionPlan,
    InformationGainEstimator,
    PromptExecutionItem,
    CostBenefitAnalyzer,
    PriorityQueue
)

from .online_learning import (
    OnlineLearningEngine,
    EnsemblePredictor,
    ConfidenceIntervalEstimator,
    PredictionAccuracy,
    ModelPerformance,
    OnlineStatistics
)

__version__ = "1.0.0"
__author__ = "REV Development Team"

__all__ = [
    # Core prediction classes
    'ResponsePredictor',
    'FeatureExtractor',
    'LightweightPredictionModels',
    'ResponsePrediction',
    'HistoricalResponse',
    'PromptFeatures',
    
    # Pattern recognition
    'TemplateResponseMapper',
    'PromptClusterAnalyzer', 
    'AnomalyDetector',
    'PromptPattern',
    'ClusterInfo',
    
    # Optimization integration
    'PromptSelectionOptimizer',
    'REVPipelineIntegration',
    'ExecutionBudget',
    'ExecutionPlan',
    'InformationGainEstimator',
    'PromptExecutionItem',
    'CostBenefitAnalyzer',
    'PriorityQueue',
    
    # Online learning
    'OnlineLearningEngine',
    'EnsemblePredictor',
    'ConfidenceIntervalEstimator',
    'PredictionAccuracy',
    'ModelPerformance',
    'OnlineStatistics'
]

# Module-level convenience functions
def create_response_prediction_system(cache_dir: str = None) -> ResponsePredictor:
    """Create a complete response prediction system with default configuration."""
    return ResponsePredictor(cache_dir=cache_dir)


def create_optimization_pipeline(predictor: ResponsePredictor = None) -> REVPipelineIntegration:
    """Create an optimization pipeline integrated with REV framework."""
    if predictor is None:
        predictor = create_response_prediction_system()
    
    optimizer = PromptSelectionOptimizer(predictor)
    return REVPipelineIntegration(predictor, optimizer)


def optimize_prompts_for_rev(prompts: list, budget_params: dict = None) -> dict:
    """
    High-level function to optimize prompts for REV pipeline execution.
    
    Args:
        prompts: List of candidate prompts
        budget_params: Budget parameters dict (cost, time, memory limits)
    
    Returns:
        Dictionary containing optimized execution plan and results
    """
    # Create system components
    predictor = create_response_prediction_system()
    integration = create_optimization_pipeline(predictor)
    
    # Set default budget if not provided
    if budget_params is None:
        budget_params = {
            'max_total_cost': 50.0,
            'max_execution_time': 300.0,
            'max_memory_usage': 4096.0,
            'max_prompts': 10
        }
    
    budget = ExecutionBudget(**budget_params)
    
    # Optimize and execute
    results = integration.optimize_and_execute(prompts, budget)
    
    return {
        'selected_prompts': results['execution_plan'].selected_prompts,
        'execution_order': results['execution_plan'].execution_order,
        'estimated_cost': results['execution_plan'].estimated_total_cost,
        'estimated_time': results['execution_plan'].estimated_total_time,
        'expected_information_gain': results['execution_plan'].expected_information_gain,
        'optimization_method': results['execution_plan'].optimization_method,
        'performance_metrics': results['performance_metrics'],
        'recommendation': results['optimization_summary']['recommendation']
    }


# System information
SYSTEM_INFO = {
    'name': 'REV Response Prediction System',
    'version': __version__,
    'description': 'Intelligent response prediction and optimization for REV pipeline',
    'features': [
        'Lightweight prediction models with <100ms response time',
        'Pattern recognition and template mapping',
        'Multi-objective optimization (greedy, genetic, simulated annealing)',
        'Online learning with adaptive model weights',
        'Ensemble methods with confidence intervals',
        'Anomaly detection for unusual prompts',
        'Cost-benefit analysis for execution planning',
        'Real-time adaptation and early stopping'
    ],
    'performance_targets': {
        'prediction_time': '<100ms per prompt',
        'memory_usage': '<50MB for full system',
        'accuracy': '>85% length prediction accuracy',
        'optimization_time': '<5s for 100 prompts'
    }
}


def get_system_info() -> dict:
    """Get system information and capabilities."""
    return SYSTEM_INFO.copy()


def print_system_info():
    """Print system information to console."""
    info = get_system_info()
    print(f"🚀 {info['name']} v{info['version']}")
    print(f"📋 {info['description']}")
    print("\n✨ Features:")
    for feature in info['features']:
        print(f"  • {feature}")
    print("\n🎯 Performance Targets:")
    for metric, target in info['performance_targets'].items():
        print(f"  • {metric}: {target}")


# Module initialization
def _check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'numpy', 'pandas', 'scikit-learn', 'scipy', 'networkx'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        import warnings
        warnings.warn(
            f"Missing dependencies: {', '.join(missing_modules)}. "
            f"Some features may not work correctly. "
            f"Install with: pip install {' '.join(missing_modules)}",
            UserWarning
        )
        return False
    
    return True


# Check dependencies on import
_DEPENDENCIES_OK = _check_dependencies()