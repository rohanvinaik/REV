#!/usr/bin/env python3
"""
Analysis Module for REV Pipeline

This module provides response prediction and behavioral analysis capabilities
for the REV framework.
"""

from .response_predictor import (
    ResponsePredictor,
    FeatureExtractor,
    LightweightPredictionModels,
    ResponsePrediction,
    HistoricalResponse,
    PromptFeatures
)

from .behavior_profiler import BehaviorProfiler

from .unified_model_analysis import UnifiedModelAnalyzer

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

    # Behavioral analysis
    'BehaviorProfiler',

    # Unified analysis
    'UnifiedModelAnalyzer',
]
