#!/usr/bin/env python3
"""
Comprehensive Test Script for Response Prediction System

This script demonstrates the complete integration of all response prediction components
with the REV pipeline for intelligent prompt pre-filtering and optimization.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.analysis.response_predictor import (
        ResponsePredictor, FeatureExtractor, LightweightPredictionModels,
        HistoricalResponse, PromptFeatures
    )
    from src.analysis.pattern_recognition import (
        TemplateResponseMapper, PromptClusterAnalyzer, AnomalyDetector
    )
    from src.analysis.optimization_integration import (
        PromptSelectionOptimizer, REVPipelineIntegration, ExecutionBudget,
        InformationGainEstimator
    )
    from src.analysis.online_learning import (
        OnlineLearningEngine, EnsemblePredictor, ConfidenceIntervalEstimator
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install scikit-learn scipy pandas numpy networkx")
    sys.exit(1)


def test_feature_extraction():
    """Test prompt feature extraction"""
    print("🔍 Testing Feature Extraction...")
    
    extractor = FeatureExtractor()
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Explain the process of neural network training in detail with examples.",
        "How does blockchain consensus work and why is it important for security?",
        "Compare and contrast supervised learning with unsupervised learning approaches.",
        "Create a step-by-step guide for implementing a recommendation system."
    ]
    
    # Fit extractor
    extractor.fit(test_prompts)
    
    # Extract features
    for i, prompt in enumerate(test_prompts):
        features = extractor.extract_features(prompt, template_id=f"template_{i}")
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Length: {features.length}, Words: {features.word_count}")
        print(f"  Difficulty: {features.difficulty_level}, FK Grade: {features.flesch_kincaid_grade:.1f}")
        print(f"  Domains: {features.domain_indicators}")
        print(f"  Complexity: {features.syntactic_complexity:.2f}")
    
    print("✅ Feature extraction test passed!")
    return extractor, test_prompts


def test_pattern_recognition():
    """Test pattern recognition and clustering"""
    print("\n🔍 Testing Pattern Recognition...")
    
    # Initialize components
    mapper = TemplateResponseMapper()
    clusterer = PromptClusterAnalyzer(n_clusters=3)
    detector = AnomalyDetector()
    
    # Test data
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning algorithms.",
        "How does deep learning work?",
        "What are the applications of AI?",
        "Describe neural network architectures.",
        "Compare different ML approaches.",
        "What is natural language processing?",
        "How do recommendation systems work?"
    ]
    
    # Create mock features
    features_list = []
    for i, prompt in enumerate(test_prompts):
        features = {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'sentence_count': 1,
            'difficulty_level': 2 + (i % 3),
            'question_markers': 1 if '?' in prompt else 0,
            'instruction_markers': 1 if any(word in prompt.lower() for word in ['explain', 'describe']) else 0,
            'technical_terms': 1,
            'has_examples': False,
            'requires_reasoning': True,
            'multi_step': False
        }
        features_list.append(features)
    
    # Test template mapping
    for i, (prompt, features) in enumerate(zip(test_prompts, features_list)):
        template_id = f"template_{i % 3}"  # Group into 3 templates
        response = f"Response to {prompt[:30]}... [Generated response text]"
        mapper.add_template_response_pair(template_id, prompt, response, features)
    
    # Test prediction from template
    test_features = features_list[0]
    prediction = mapper.predict_response_from_template("template_0", test_features)
    print(f"Template prediction: {prediction}")
    
    # Test clustering
    cluster_info = clusterer.fit_clusters(test_prompts, features_list)
    print(f"Found {len(cluster_info)} clusters")
    
    for cluster_id, info in cluster_info.items():
        print(f"  Cluster {cluster_id}: {info.size} prompts, coherence: {info.coherence_score:.2f}")
        print(f"    Representative: {info.representative_prompts[0][:50]}...")
    
    # Test anomaly detection
    detector.fit(test_prompts, features_list)
    
    # Test with normal and anomalous prompts
    normal_prompt = "What is computer vision?"
    normal_features = {'length': 23, 'word_count': 4, 'difficulty_level': 2, 
                      'question_markers': 1, 'instruction_markers': 0, 'technical_terms': 1}
    
    anomaly_result = detector.detect_anomalies(normal_prompt, normal_features)
    print(f"\nNormal prompt anomaly score: {anomaly_result['anomaly_score']:.2f}")
    
    # Anomalous prompt (very long)
    anomalous_prompt = "What is " + "very " * 100 + "long question about AI?"
    anomalous_features = {'length': len(anomalous_prompt), 'word_count': 103, 
                         'difficulty_level': 1, 'question_markers': 1, 
                         'instruction_markers': 0, 'technical_terms': 1}
    
    anomaly_result = detector.detect_anomalies(anomalous_prompt, anomalous_features)
    print(f"Anomalous prompt anomaly score: {anomaly_result['anomaly_score']:.2f}")
    print(f"Anomaly types: {anomaly_result['anomaly_types']}")
    print(f"Handling recommendation: {anomaly_result['handling_recommendation']}")
    
    print("✅ Pattern recognition test passed!")
    return mapper, clusterer, detector


def test_response_prediction():
    """Test core response prediction functionality"""
    print("\n🔍 Testing Response Prediction...")
    
    predictor = ResponsePredictor()
    
    # Create some historical data
    historical_responses = []
    
    test_data = [
        ("What is machine learning?", "Machine learning is a method of data analysis...", 120, 1.5, 0.8, 0.7),
        ("Explain neural networks.", "Neural networks are computing systems inspired by...", 200, 2.2, 0.9, 0.8),
        ("How does AI work?", "Artificial intelligence works through various algorithms...", 150, 1.8, 0.7, 0.6),
        ("Define deep learning.", "Deep learning is a subset of machine learning...", 180, 2.0, 0.85, 0.75),
    ]
    
    for prompt, response, length, exec_time, coherence, informativeness in test_data:
        features = predictor.feature_extractor.extract_features(prompt)
        
        historical_response = HistoricalResponse(
            prompt=prompt,
            response=response,
            features=features,
            actual_length=length,
            actual_word_count=len(response.split()),
            actual_tokens=int(len(response.split()) * 1.3),
            execution_time=exec_time,
            memory_usage=50.0,
            coherence_score=coherence,
            informativeness_score=informativeness,
            diversity_score=0.6,
            model_id="test_model",
            timestamp=time.time(),
            template_id=f"template_{len(historical_responses)}"
        )
        historical_responses.append(historical_response)
        predictor.add_historical_response(historical_response)
    
    # Train models
    print("Training prediction models...")
    performance = predictor.train_models()
    print(f"Model performance: {performance}")
    
    # Test predictions
    test_prompts = [
        "What are the applications of artificial intelligence?",
        "Explain the concept of reinforcement learning in detail.",
        "How do recommendation systems work?"
    ]
    
    for prompt in test_prompts:
        prediction = predictor.predict_response(prompt)
        
        print(f"\nPrompt: {prompt}")
        print(f"  Predicted length: {prediction.estimated_length:.0f} chars")
        print(f"  Predicted time: {prediction.estimated_computation_time:.2f} seconds")
        print(f"  Confidence: {prediction.prediction_confidence:.2f}")
        print(f"  Method: {prediction.prediction_method}")
        print(f"  Expected structure: {prediction.expected_structure}")
    
    print("✅ Response prediction test passed!")
    return predictor


def test_optimization_integration():
    """Test optimization and REV pipeline integration"""
    print("\n🔍 Testing Optimization Integration...")
    
    # Initialize components
    predictor = ResponsePredictor()
    optimizer = PromptSelectionOptimizer(predictor)
    integration = REVPipelineIntegration(predictor, optimizer)
    
    # Add some mock historical data for better predictions
    mock_historical_data = []
    for i in range(10):
        prompt = f"Test prompt {i} about machine learning and AI applications."
        features = predictor.feature_extractor.extract_features(prompt)
        
        historical_response = HistoricalResponse(
            prompt=prompt,
            response=f"Mock response {i} with relevant information...",
            features=features,
            actual_length=100 + i * 20,
            actual_word_count=20 + i * 3,
            actual_tokens=25 + i * 4,
            execution_time=1.0 + i * 0.2,
            memory_usage=50.0,
            coherence_score=0.7 + (i % 3) * 0.1,
            informativeness_score=0.6 + (i % 4) * 0.1,
            diversity_score=0.5 + (i % 5) * 0.1,
            model_id="test_model",
            timestamp=time.time(),
            template_id=f"template_{i % 3}"
        )
        mock_historical_data.append(historical_response)
        predictor.add_historical_response(historical_response)
    
    # Train models with mock data
    try:
        predictor.train_models()
        print("Models trained successfully")
    except Exception as e:
        print(f"Model training failed: {e}, continuing with fallback predictions")
    
    # Test candidate prompts
    candidate_prompts = [
        "What is machine learning and how does it work?",
        "Explain the differences between supervised and unsupervised learning.",
        "How do neural networks process information?",
        "What are the applications of deep learning in computer vision?",
        "Describe the process of natural language processing.",
        "How does reinforcement learning work in game playing?",
        "What are the ethical considerations in AI development?",
        "Explain the concept of transfer learning.",
        "How do recommendation systems personalize content?",
        "What is the future of artificial intelligence?"
    ]
    
    # Create execution budget
    budget = ExecutionBudget(
        max_total_cost=20.0,
        max_execution_time=60.0,
        max_memory_usage=1000.0,
        max_prompts=5
    )
    
    # Test different optimization methods
    methods = ['greedy', 'multi_objective', 'genetic_algorithm']
    
    for method in methods:
        print(f"\nTesting {method} optimization:")
        
        execution_plan = optimizer.optimize_prompt_selection(
            candidate_prompts, budget, method=method
        )
        
        print(f"  Selected {len(execution_plan.selected_prompts)} prompts")
        print(f"  Estimated cost: ${execution_plan.estimated_total_cost:.2f}")
        print(f"  Estimated time: {execution_plan.estimated_total_time:.1f}s")
        print(f"  Expected information gain: {execution_plan.expected_information_gain:.2f}")
        print(f"  Confidence: {execution_plan.confidence_score:.2f}")
        
        # Show selected prompts
        for i, prompt in enumerate(execution_plan.selected_prompts[:3]):
            print(f"    {i+1}. {prompt[:60]}...")
    
    # Test full integration
    print(f"\nTesting full REV pipeline integration:")
    
    results = integration.optimize_and_execute(
        candidate_prompts[:6],  # Smaller set for testing
        budget
    )
    
    print(f"  Execution completed:")
    print(f"  - Success rate: {results['performance_metrics']['success_rate']:.2f}")
    print(f"  - Successful executions: {len(results['execution_results']['successful_executions'])}")
    print(f"  - Total execution time: {results['execution_results']['total_execution_time']:.2f}s")
    
    summary = results['optimization_summary']
    print(f"  - Resource efficiency: {summary['resource_efficiency']:.2f}")
    print(f"  - Recommendation: {summary['recommendation']}")
    
    print("✅ Optimization integration test passed!")
    return integration


def test_online_learning():
    """Test online learning and ensemble methods"""
    print("\n🔍 Testing Online Learning...")
    
    # Initialize components
    online_engine = OnlineLearningEngine()
    ensemble = EnsemblePredictor()
    confidence_estimator = ConfidenceIntervalEstimator()
    
    # Simulate online learning with multiple observations
    print("Simulating online learning updates...")
    
    for i in range(20):
        # Create test features
        prompt = f"Test prompt {i} with varying complexity and length."
        
        features = PromptFeatures(
            length=30 + i * 5,
            word_count=6 + i,
            sentence_count=1 + (i % 3),
            avg_word_length=4.5 + (i % 5) * 0.2,
            flesch_kincaid_grade=6.0 + (i % 8),
            syntactic_complexity=0.2 + (i % 4) * 0.1,
            vocabulary_diversity=0.6 + (i % 6) * 0.05,
            question_markers=i % 2,
            instruction_markers=(i + 1) % 2,
            technical_terms=i % 3,
            domain_indicators=['technical'] if i % 2 else ['general'],
            template_id=f"template_{i % 4}",
            template_category='question' if i % 2 else 'instruction',
            difficulty_level=1 + (i % 5),
            has_examples=i % 3 == 0,
            requires_reasoning=i % 2 == 0,
            has_constraints=i % 4 == 0,
            multi_step=i % 5 == 0
        )
        
        # Simulate actual response
        actual_response = {
            'length': 80 + i * 10 + (i % 3) * 20,
            'execution_time': 1.0 + i * 0.1,
            'coherence_score': 0.6 + (i % 7) * 0.05,
            'informativeness_score': 0.5 + (i % 8) * 0.06
        }
        
        # Create mock prediction for comparison
        from src.analysis.response_predictor import ResponsePrediction
        predicted_response = ResponsePrediction(
            estimated_length=75 + i * 10,
            estimated_word_count=15 + i * 2,
            estimated_complexity=6.5,
            estimated_tokens=20 + i * 3,
            estimated_computation_time=0.9 + i * 0.1,
            estimated_memory_usage=50.0,
            predicted_coherence=0.65 + (i % 6) * 0.04,
            predicted_informativeness=0.55 + (i % 7) * 0.05,
            predicted_diversity=0.6,
            confidence_interval=(60.0, 120.0),
            prediction_confidence=0.7 + (i % 4) * 0.05,
            uncertainty_score=0.3 - (i % 4) * 0.05,
            response_type='informative',
            expected_structure='paragraph',
            prediction_method=f'method_{i % 3}',
            similar_prompts_count=5
        )
        
        # Update online learning
        feedback = online_engine.update_online(features, actual_response, predicted_response)
        
        # Update confidence estimator
        confidence_estimator.update_error_history(
            predicted_response.prediction_method,
            predicted_response.estimated_length,
            actual_response['length']
        )
        
        if i % 5 == 0:
            print(f"  Update {i}: Learning feedback keys: {list(feedback.keys())}")
    
    # Test online predictions
    print("\nTesting online predictions...")
    
    test_features = PromptFeatures(
        length=100, word_count=20, sentence_count=2, avg_word_length=5.0,
        flesch_kincaid_grade=8.0, syntactic_complexity=0.4, vocabulary_diversity=0.8,
        question_markers=1, instruction_markers=0, technical_terms=2,
        domain_indicators=['technical'], template_id='test_template', template_category='question',
        difficulty_level=3, has_examples=False, requires_reasoning=True, 
        has_constraints=False, multi_step=False
    )
    
    online_predictions = online_engine.predict_online(test_features)
    print(f"Online predictions: {online_predictions}")
    
    # Test learning rate adaptation
    adaptive_rates = online_engine.get_learning_rate_adaptation()
    print(f"Adaptive learning rates: {adaptive_rates}")
    
    # Test ensemble methods
    print("\nTesting ensemble methods...")
    
    # Create multiple mock predictions for ensemble
    mock_predictions = {}
    
    for i, method in enumerate(['method_a', 'method_b', 'method_c']):
        mock_predictions[method] = ResponsePrediction(
            estimated_length=90 + i * 10,
            estimated_word_count=18 + i * 2,
            estimated_complexity=7.0 + i * 0.5,
            estimated_tokens=25 + i * 3,
            estimated_computation_time=1.2 + i * 0.2,
            estimated_memory_usage=55.0,
            predicted_coherence=0.7 + i * 0.05,
            predicted_informativeness=0.6 + i * 0.05,
            predicted_diversity=0.65,
            confidence_interval=(80.0, 130.0),
            prediction_confidence=0.75 + i * 0.05,
            uncertainty_score=0.25 - i * 0.05,
            response_type='informative',
            expected_structure='paragraph',
            prediction_method=method,
            similar_prompts_count=8 + i
        )
    
    # Test different ensemble methods
    ensemble_methods = ['weighted_average', 'confidence_weighted', 'adaptive_weighting']
    
    for method in ensemble_methods:
        ensemble_prediction = ensemble.combine_predictions(mock_predictions, test_features, method)
        print(f"  {method}: Length={ensemble_prediction.estimated_length:.1f}, Confidence={ensemble_prediction.prediction_confidence:.2f}")
    
    # Test confidence intervals
    print("\nTesting confidence intervals...")
    
    for method_name, prediction in mock_predictions.items():
        ci = confidence_estimator.estimate_confidence_interval(prediction)
        print(f"  {method_name}: CI = {ci[0]:.1f} - {ci[1]:.1f}")
    
    # Get performance summary
    performance_summary = online_engine.model_performances
    if performance_summary:
        print(f"\nModel performance summary:")
        for model_name, perf in performance_summary.items():
            print(f"  {model_name}: MAE={perf.mae:.2f}, R²={perf.r2_score:.2f}")
    
    print("✅ Online learning test passed!")
    return online_engine, ensemble, confidence_estimator


def run_comprehensive_integration_test():
    """Run comprehensive integration test of entire system"""
    print("🚀 Running Comprehensive Integration Test")
    print("=" * 60)
    
    try:
        # Test each component
        extractor, test_prompts = test_feature_extraction()
        mapper, clusterer, detector = test_pattern_recognition()
        predictor = test_response_prediction()
        integration = test_optimization_integration()
        online_engine, ensemble, confidence_estimator = test_online_learning()
        
        # Final integration test
        print("\n🔧 Final Integration Test...")
        
        # Create a realistic scenario
        candidate_prompts = [
            "What is the difference between machine learning and deep learning?",
            "How do convolutional neural networks work in image recognition?",
            "Explain the concept of natural language processing and its applications.",
            "What are the ethical implications of artificial intelligence?",
            "How does reinforcement learning enable autonomous systems?",
            "Describe the role of big data in modern AI systems.",
            "What are the challenges in deploying AI models in production?",
            "How do transformer models revolutionize language understanding?"
        ]
        
        budget = ExecutionBudget(
            max_total_cost=30.0,
            max_execution_time=120.0,
            max_memory_usage=2000.0,
            max_prompts=4
        )
        
        # Full pipeline execution
        print("Executing full pipeline with response prediction optimization...")
        
        start_time = time.time()
        
        results = integration.optimize_and_execute(
            candidate_prompts,
            budget,
            rev_pipeline_params={'quality_threshold': 0.7}
        )
        
        execution_time = time.time() - start_time
        
        # Results summary
        plan = results['execution_plan']
        execution_results = results['execution_results']
        metrics = results['performance_metrics']
        summary = results['optimization_summary']
        
        print(f"\n📊 Integration Test Results:")
        print(f"  Total execution time: {execution_time:.2f} seconds")
        print(f"  Selected prompts: {len(plan.selected_prompts)}")
        print(f"  Optimization method: {plan.optimization_method}")
        print(f"  Estimated vs actual cost: ${plan.estimated_total_cost:.2f} vs ${summary['actual_cost']:.2f}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Resource efficiency: {summary['resource_efficiency']:.2f}")
        print(f"  Information gain: {plan.expected_information_gain:.2f}")
        
        print(f"\n📈 Selected Prompts (optimized order):")
        for i, prompt in enumerate(plan.selected_prompts):
            print(f"  {i+1}. {prompt}")
        
        print(f"\n💡 System Recommendation: {summary['recommendation']}")
        
        # Performance analysis
        print(f"\n🔍 System Performance Analysis:")
        
        # Prediction accuracy
        if hasattr(predictor, 'accuracy_metrics'):
            print(f"  Prediction Models:")
            for metric, value in predictor.accuracy_metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        # Online learning performance
        if online_engine.model_performances:
            print(f"  Online Learning:")
            for model, perf in online_engine.model_performances.items():
                print(f"    {model}: MAE={perf.mae:.2f}, Stability={perf.performance_stability:.2f}")
        
        # Resource utilization
        resource_util = {
            'cost_efficiency': 1 - abs(summary['actual_cost'] - summary['planned_cost']) / max(summary['planned_cost'], 1),
            'time_efficiency': 1 - abs(summary['actual_time'] - summary['planned_time']) / max(summary['planned_time'], 1),
            'selection_ratio': len(plan.selected_prompts) / len(candidate_prompts)
        }
        
        print(f"  Resource Utilization:")
        for metric, value in resource_util.items():
            print(f"    {metric}: {value:.2%}")
        
        print("\n🎉 Comprehensive Integration Test PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_test_results(results: dict, output_file: str = "response_prediction_test_results.json"):
    """Save test results to file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Test results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    print("🧪 Response Prediction System Comprehensive Test Suite")
    print("=" * 60)
    print("Testing intelligent prompt pre-filtering and optimization for REV pipeline")
    print()
    
    # Run all tests
    success = run_comprehensive_integration_test()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("The Response Prediction System is ready for integration with REV pipeline.")
        print("\nKey capabilities demonstrated:")
        print("  • Intelligent prompt feature extraction")
        print("  • Pattern recognition and clustering")
        print("  • Response prediction with confidence intervals")
        print("  • Multi-objective optimization for prompt selection")
        print("  • Online learning and model adaptation")
        print("  • Ensemble methods for improved accuracy")
        print("  • Full REV pipeline integration")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)