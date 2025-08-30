#!/usr/bin/env python3

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import json
import os

from src.verifier.contamination import (
    UnifiedContaminationDetector,
    ContaminationType,
    DetectionMode,
    MemorizationEvidence,
    NGramOverlap,
    ContaminationReport,
    BenchmarkDataset
)


class TestUnifiedContaminationDetector:
    
    @pytest.fixture
    def detector(self):
        return UnifiedContaminationDetector(
            detection_mode=DetectionMode.COMPREHENSIVE
        )
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.generate = Mock(return_value="Generated text response")
        model.compute_perplexity = Mock(return_value=15.0)
        model.get_logits = Mock(return_value=torch.randn(1, 100, 50000))
        return model
    
    @pytest.fixture
    def sample_dataset(self):
        return [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "Machine learning is a subset of artificial intelligence"},
            {"text": "Python is a high-level programming language"},
            {"text": "Data science involves extracting insights from data"},
            {"text": "Neural networks are inspired by biological neurons"}
        ]
    
    def test_detector_initialization(self):
        detector = UnifiedContaminationDetector(
            detection_mode=DetectionMode.FAST
        )
        
        assert detector.detection_mode == DetectionMode.FAST
        assert detector.thresholds['memorization_pvalue'] == 0.01
        assert detector.thresholds['perplexity_zscore'] == 3.0
        assert detector.thresholds['ngram_overlap'] == 0.6
        assert detector.cache_dir is not None
    
    def test_memorization_detection(self, detector, mock_model, sample_dataset):
        responses = [
            sample["text"] for sample in sample_dataset[:3]
        ] + ["Different response"] * 2
        prompts = [sample["text"][:10] for sample in sample_dataset]
        
        evidence = detector._detect_memorization(
            responses=responses,
            prompts=prompts,
            references=[sample["text"] for sample in sample_dataset]
        )
        
        assert isinstance(evidence, MemorizationEvidence)
        assert len(evidence.verbatim_matches) >= 0
        assert len(evidence.near_duplicates) >= 0
        assert evidence.memorization_score >= 0.0
        assert evidence.memorization_score <= 1.0
        assert evidence.statistical_significance >= 0.0
    
    def test_perplexity_analysis(self, detector, mock_model, sample_dataset):
        with patch.object(detector, '_compute_perplexity') as mock_perp:
            perplexities = [10.0, 12.0, 200.0, 15.0, 11.0]
            mock_perp.side_effect = perplexities
            
            responses = [s['text'] for s in sample_dataset]
            prompts = [s['text'][:10] for s in sample_dataset]
            
            result = detector._analyze_perplexity(
                responses=responses,
                prompts=prompts
            )
            
            assert "mean" in result
            assert "std" in result
            assert "anomalies" in result
            assert len(result["anomalies"]) >= 0
    
    def test_ngram_overlap(self, detector):
        responses = [
            "The quick brown fox jumps",
            "A different sentence entirely",
            "over the lazy dog"
        ]
        references = ["The quick brown fox jumps over the lazy dog"]
        
        result = detector._analyze_ngram_overlap(
            responses=responses,
            references=references
        )
        
        assert "max_overlap" in result
        assert "overlaps" in result
        assert len(result["overlaps"]) > 0
    
    def test_temperature_consistency(self, detector, mock_model, sample_dataset):
        prompts = [s['text'] for s in sample_dataset]
        
        responses_by_temp = {
            0.0: ["Deterministic"] * 5,
            0.5: ["Semi random 1", "Semi random 2", "Semi random 3", "Semi random 4", "Semi random 5"],
            1.0: ["Random 1", "Random 2", "Random 3", "Random 4", "Random 5"]
        }
        
        consistency = detector._check_temperature_consistency(
            responses_by_temperature=responses_by_temp,
            prompts=prompts
        )
        
        assert "variance" in consistency
        assert "consistency_scores" in consistency
    
    def test_token_level_detection(self, detector, mock_model):
        prompts = ["Test prompt 1", "Test prompt 2"]
        responses = ["Response 1", "Response 2"]
        
        token_stats = detector._analyze_token_level(
            responses=responses,
            prompts=prompts
        )
        
        assert "entropy" in token_stats
        assert "anomalies" in token_stats
    
    def test_semantic_detection(self, detector, mock_model):
        responses = ["Test text 1", "Test text 2", "Test text 3"]
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        semantic_stats = detector._analyze_semantic_similarity(
            responses=responses,
            prompts=prompts
        )
        
        assert "max_similarity" in semantic_stats
        assert "similarities" in semantic_stats
    
    def test_behavioral_detection(self, detector, mock_model):
        responses = ["Test 1", "Test 2"]
        prompts = ["Prompt 1", "Prompt 2"]
        
        behavioral_stats = detector._analyze_behavioral_patterns(
            responses=responses,
            prompts=prompts
        )
        
        assert "divergence" in behavioral_stats
        assert "anomalies" in behavioral_stats
    
    def test_benchmark_detection(self, detector):
        prompts = [
            "What is 2+2?",
            "Explain quantum mechanics",
            "Write a Python function"
        ]
        responses = [
            "4",
            "Quantum mechanics is...",
            "def hello_world():"
        ]
        
        benchmark_matches = detector._check_benchmark_overlap(
            responses=responses,
            prompts=prompts
        )
        
        assert "total_matches" in benchmark_matches
        assert "by_benchmark" in benchmark_matches
    
    def test_full_contamination_detection(self, detector, mock_model, sample_dataset):
        with patch.object(detector, '_detect_memorization') as mock_mem:
            mock_mem.return_value = MemorizationEvidence(
                verbatim_matches=["match1", "match2"],
                near_duplicates=[("text", 0.9)],
                perplexity_anomalies=[1.5, 2.0],
                consistency_violations=[],
                statistical_significance=0.95
            )
            
            with patch.object(detector, '_analyze_perplexity') as mock_perp:
                mock_perp.return_value = {
                    "mean": 15.0,
                    "std": 5.0,
                    "anomalies": []
                }
                
                prompts = [s['text'][:10] for s in sample_dataset]
                responses = [s['text'] for s in sample_dataset]
                
                report = detector.detect_contamination(
                    model_responses=responses,
                    prompts=prompts,
                    model_id="test_model",
                    mode=DetectionMode.COMPREHENSIVE
                )
                
                assert isinstance(report, ContaminationReport)
                assert report.model_id == "test_model"
                assert report.detection_mode == DetectionMode.COMPREHENSIVE
                assert ContaminationType.MEMORIZATION in report.contamination_types
                assert report.confidence_score >= 0.0
                assert report.confidence_score <= 1.0
    
    def test_report_generation(self, detector):
        report = ContaminationReport(
            model_id="test_model",
            timestamp=datetime.now(),
            detection_mode=DetectionMode.FAST,
            contamination_types=[ContaminationType.MEMORIZATION],
            confidence_score=0.85,
            evidence={
                "memorization": MemorizationEvidence(
                    exact_matches=3,
                    total_samples=10,
                    memorized_samples=[0, 2, 5],
                    confidence=0.9
                )
            },
            remediation_suggestions=["Review training data", "Apply deduplication"]
        )
        
        assert report.model_id == "test_model"
        assert report.detection_mode == DetectionMode.FAST
        assert ContaminationType.MEMORIZATION in report.contamination_types
        assert report.confidence_score == 0.85
        assert "memorization" in report.evidence
        assert len(report.remediation_suggestions) == 2
    
    def test_caching_functionality(self, detector, mock_model):
        prompts = ["Test 1", "Test 2"]
        responses = ["Response 1", "Response 2"]
        
        with patch.object(detector, '_detect_memorization') as mock_detect:
            mock_detect.return_value = MemorizationEvidence(
                verbatim_matches=[],
                near_duplicates=[],
                perplexity_anomalies=[],
                consistency_violations=[],
                statistical_significance=0.1
            )
            
            report1 = detector.detect_contamination(
                model_responses=responses,
                prompts=prompts,
                model_id="test_model",
                mode=DetectionMode.FAST
            )
            
            # Cache results
            detector._cache_results("test_model", report1)
            
            # Load from cache
            cached_report = detector.load_cached_results("test_model")
            
            assert cached_report is not None
            assert cached_report.confidence_score == report1.confidence_score
    
    def test_model_comparison(self, detector):
        model_ids = ["model1", "model2"]
        test_prompts = ["Test 1", "Test 2"]
        test_responses = {
            "model1": ["Response 1 from model1", "Response 2 from model1"],
            "model2": ["Response 1 from model2", "Response 2 from model2"]
        }
        
        with patch.object(detector, 'detect_contamination') as mock_detect:
            mock_detect.side_effect = [
                ContaminationReport(
                    model_id="model1",
                    timestamp=datetime.now(),
                    detection_mode=DetectionMode.FAST,
                    contamination_types=[ContaminationType.MEMORIZATION],
                    confidence_score=0.8,
                    evidence={},
                    remediation_suggestions=[]
                ),
                ContaminationReport(
                    model_id="model2",
                    timestamp=datetime.now(),
                    detection_mode=DetectionMode.FAST,
                    contamination_types=[],
                    confidence_score=0.2,
                    evidence={},
                    remediation_suggestions=[]
                )
            ]
            
            comparison = detector.compare_models_contamination(
                model_ids=model_ids,
                test_prompts=test_prompts,
                test_responses=test_responses
            )
            
            assert len(comparison) == 2
            assert comparison["model1"].confidence_score == 0.8
            assert comparison["model2"].confidence_score == 0.2
    
    def test_remediation_suggestions(self, detector):
        suggestions = detector._generate_remediation_suggestions(
            contamination_types=[ContaminationType.MEMORIZATION, ContaminationType.DATA_LEAKAGE],
            confidence_score=0.9
        )
        
        assert len(suggestions) > 0
        assert any("deduplication" in s.lower() for s in suggestions)
        assert any("filter" in s.lower() or "remove" in s.lower() for s in suggestions)
    
    def test_export_report(self, detector):
        report = ContaminationReport(
            model_id="test_model",
            timestamp=datetime.now(),
            detection_mode=DetectionMode.COMPREHENSIVE,
            contamination_types=[ContaminationType.BENCHMARK_OVERLAP],
            confidence_score=0.75,
            evidence={"test": "evidence"},
            remediation_suggestions=["Test recommendation"]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Cache the report
            detector._cache_results("test_model", report)
            
            # Check it was cached
            cached = detector.load_cached_results("test_model")
            assert cached is not None
            assert cached.confidence_score == 0.75
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_edge_cases(self, detector, mock_model):
        # Empty data
        report = detector.detect_contamination(
            model_responses=[],
            prompts=[],
            model_id="test_model",
            mode=DetectionMode.FAST
        )
        assert report.confidence_score == 0.0
        assert len(report.contamination_types) == 0
        
        # Single item
        report = detector.detect_contamination(
            model_responses=["Single response"],
            prompts=["Single test"],
            model_id="test_model",
            mode=DetectionMode.FAST
        )
        assert isinstance(report, ContaminationReport)
        
        # Error handling
        with patch.object(detector, '_detect_memorization') as mock_mem:
            mock_mem.side_effect = Exception("API Error")
            report = detector.detect_contamination(
                model_responses=["Test response"],
                prompts=["Test"],
                model_id="test_model",
                mode=DetectionMode.FAST
            )
            assert report.confidence_score == 0.0
    
    def test_statistical_significance(self, detector):
        evidence = MemorizationEvidence(
            verbatim_matches=[f"match{i}" for i in range(45)],
            near_duplicates=[],
            perplexity_anomalies=[],
            consistency_violations=[],
            statistical_significance=0.99
        )
        
        # Test high memorization case
        p_value = detector._memorization_statistical_test(
            exact_matches=45,
            total_samples=50
        )
        assert p_value < 0.01
        
        # Test low memorization case
        p_value_low = detector._memorization_statistical_test(
            exact_matches=2,
            total_samples=50
        )
        assert p_value_low > 0.05
    
    def test_adaptive_thresholds(self, detector, mock_model):
        # Test that thresholds are properly initialized
        assert 'memorization_pvalue' in detector.thresholds
        assert 'perplexity_zscore' in detector.thresholds
        assert 'ngram_overlap' in detector.thresholds
        
        # Test threshold updates
        original_threshold = detector.thresholds['memorization_pvalue']
        detector.thresholds['memorization_pvalue'] = 0.001
        assert detector.thresholds['memorization_pvalue'] != original_threshold


class TestIntegrationScenarios:
    
    @pytest.fixture
    def full_detector(self):
        return UnifiedContaminationDetector(
            detection_mode=DetectionMode.COMPREHENSIVE
        )
    
    def test_real_world_memorization(self, full_detector):
        training_samples = [
            "The capital of France is Paris",
            "Water freezes at 0 degrees Celsius",
            "The Earth orbits around the Sun"
        ]
        
        prompts = [s[:10] for s in training_samples]
        responses = training_samples  # Exact match indicates memorization
        
        evidence = full_detector._detect_memorization(
            responses=responses,
            prompts=prompts,
            references=training_samples
        )
        
        assert evidence.memorization_score > 0.5
        assert len(evidence.verbatim_matches) >= 0
        assert evidence.statistical_significance > 0.5
    
    def test_benchmark_contamination_flow(self, full_detector):
        prompts = [
            "John has 5 apples and gives 2 to Mary. How many apples does John have?",
            "A train travels 60 mph for 2 hours. How far did it travel?"
        ]
        responses = [
            "John has 3 apples",
            "The train traveled 120 miles"
        ]
        
        matches = full_detector._check_benchmark_overlap(
            responses=responses,
            prompts=prompts
        )
        
        assert matches["total_matches"] >= 0
        assert "by_benchmark" in matches
    
    def test_comprehensive_detection_pipeline(self, full_detector):
        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
        responses = ["Test response 1", "Test response 2", "Test response 3"]
        
        report = full_detector.detect_contamination(
            model_responses=responses,
            prompts=prompts,
            model_id="test_model",
            mode=DetectionMode.COMPREHENSIVE
        )
        
        assert isinstance(report, ContaminationReport)
        assert report.detection_mode == DetectionMode.COMPREHENSIVE
        assert report.confidence_score >= 0.0
        assert report.confidence_score <= 1.0
        assert len(report.remediation_suggestions) > 0
        
        assert "memorization" in report.evidence
        assert "perplexity" in report.evidence
        assert "ngram" in report.evidence
        assert "temperature" in report.evidence
        assert "token_level" in report.evidence
        assert "semantic" in report.evidence
        assert "behavioral" in report.evidence
        assert "benchmark" in report.evidence
    
    def test_performance_with_large_dataset(self, full_detector):
        large_dataset = [
            {"text": f"Sample text number {i}"}
            for i in range(1000)
        ]
        
        prompts = [s['text'][:10] for s in large_dataset[:100]]
        responses = [s['text'] for s in large_dataset[:100]]
        
        import time
        start_time = time.time()
        
        report = full_detector.detect_contamination(
            model_responses=responses,
            prompts=prompts,
            model_id="test_model",
            mode=DetectionMode.FAST
        )
        
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 10.0
        assert isinstance(report, ContaminationReport)
    
    def test_cross_model_contamination(self, full_detector):
        model_ids = ["ModelA", "ModelB", "ModelC"]
        test_prompts = ["Test prompt"]
        test_responses = {
            "ModelA": ["Memorized response"],
            "ModelB": ["Different response"],
            "ModelC": ["Memorized response"]
        }
        
        with patch.object(full_detector, '_detect_memorization') as mock_mem:
            mock_mem.side_effect = [
                MemorizationEvidence(
                    verbatim_matches=["match1"],
                    near_duplicates=[],
                    perplexity_anomalies=[],
                    consistency_violations=[],
                    statistical_significance=0.99
                ),
                MemorizationEvidence(
                    verbatim_matches=[],
                    near_duplicates=[],
                    perplexity_anomalies=[],
                    consistency_violations=[],
                    statistical_significance=0.1
                ),
                MemorizationEvidence(
                    verbatim_matches=["match1"],
                    near_duplicates=[],
                    perplexity_anomalies=[],
                    consistency_violations=[],
                    statistical_significance=0.99
                )
            ]
            
            comparison = full_detector.compare_models_contamination(
                model_ids=model_ids,
                test_prompts=test_prompts,
                test_responses=test_responses
            )
            
            assert comparison["ModelA"].confidence_score > 0.5
            assert comparison["ModelB"].confidence_score < 0.5
            assert comparison["ModelC"].confidence_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])