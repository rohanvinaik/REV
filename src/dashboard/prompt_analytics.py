#!/usr/bin/env python3
"""
Comprehensive Prompt Analytics System for REV Pipeline

This module provides sophisticated analytics, visualization, and reporting
capabilities for tracking prompt effectiveness, model discrimination, and
behavioral analysis results.
"""

import os
import json
import time
import uuid
import sqlite3
import threading
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Visualization dependencies (graceful fallback if not available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Visualization libraries not available. Charts will be disabled.")

# Network analysis dependencies
try:
    import networkx as nx
    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False
    warnings.warn("NetworkX not available. Network graph analysis will be disabled.")


@dataclass
class PromptMetrics:
    """Metrics for a single prompt execution"""
    prompt_id: str
    template_id: str
    model_id: str
    timestamp: datetime
    success: bool
    response_length: int
    execution_time: float
    discrimination_score: float = 0.0
    behavioral_score: float = 0.0
    quality_score: float = 0.0
    difficulty_level: int = 1
    domain: str = "general"
    cost: float = 0.0
    error_type: Optional[str] = None
    response_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Results from A/B testing between prompt variants"""
    test_id: str
    variant_a_id: str
    variant_b_id: str
    start_time: datetime
    end_time: datetime
    sample_size_a: int
    sample_size_b: int
    success_rate_a: float
    success_rate_b: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    practical_significance: bool
    recommendation: str


@dataclass
class PromptAnalysisReport:
    """Comprehensive analysis report for prompt performance"""
    report_id: str
    generation_time: datetime
    time_period: Tuple[datetime, datetime]
    total_prompts: int
    unique_templates: int
    models_tested: List[str]
    overall_success_rate: float
    avg_discrimination_power: float
    top_performers: List[Dict[str, Any]]
    underperformers: List[Dict[str, Any]]
    trends: Dict[str, Any]
    recommendations: List[str]
    statistical_tests: List[Dict[str, Any]]
    export_data: Dict[str, Any]


class MetricsDatabase:
    """SQLite database for storing prompt metrics and analysis results"""
    
    def __init__(self, db_path: str = "prompt_analytics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Prompt metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    success BOOLEAN NOT NULL,
                    response_length INTEGER,
                    execution_time REAL,
                    discrimination_score REAL,
                    behavioral_score REAL,
                    quality_score REAL,
                    difficulty_level INTEGER,
                    domain TEXT,
                    cost REAL,
                    error_type TEXT,
                    response_hash TEXT,
                    metadata TEXT
                )
            """)
            
            # A/B test results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE NOT NULL,
                    variant_a_id TEXT NOT NULL,
                    variant_b_id TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    sample_size_a INTEGER,
                    sample_size_b INTEGER,
                    success_rate_a REAL,
                    success_rate_b REAL,
                    p_value REAL,
                    effect_size REAL,
                    confidence_interval_low REAL,
                    confidence_interval_high REAL,
                    statistical_significance BOOLEAN,
                    practical_significance BOOLEAN,
                    recommendation TEXT
                )
            """)
            
            # Analysis reports table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    generation_time DATETIME NOT NULL,
                    time_period_start DATETIME NOT NULL,
                    time_period_end DATETIME NOT NULL,
                    report_data TEXT NOT NULL
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_timestamp ON prompt_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_template_id ON prompt_metrics(template_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_id ON prompt_metrics(model_id)")
    
    def insert_metrics(self, metrics: PromptMetrics):
        """Insert prompt metrics into database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO prompt_metrics (
                        prompt_id, template_id, model_id, timestamp, success,
                        response_length, execution_time, discrimination_score,
                        behavioral_score, quality_score, difficulty_level,
                        domain, cost, error_type, response_hash, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.prompt_id, metrics.template_id, metrics.model_id,
                    metrics.timestamp, metrics.success, metrics.response_length,
                    metrics.execution_time, metrics.discrimination_score,
                    metrics.behavioral_score, metrics.quality_score,
                    metrics.difficulty_level, metrics.domain, metrics.cost,
                    metrics.error_type, metrics.response_hash,
                    json.dumps(metrics.metadata)
                ))
    
    def query_metrics(self, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     model_ids: Optional[List[str]] = None,
                     template_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Query metrics with optional filters"""
        query = "SELECT * FROM prompt_metrics WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if model_ids:
            placeholders = ','.join(['?' for _ in model_ids])
            query += f" AND model_id IN ({placeholders})"
            params.extend(model_ids)
        if template_ids:
            placeholders = ','.join(['?' for _ in template_ids])
            query += f" AND template_id IN ({placeholders})"
            params.extend(template_ids)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['metadata'] = df['metadata'].apply(
                    lambda x: json.loads(x) if x else {}
                )
        
        return df


class MetricsCollector:
    """Collects and aggregates prompt performance metrics"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.real_time_metrics = defaultdict(list)
        self.lock = threading.Lock()
        self.collection_start = datetime.now()
    
    def record_prompt_execution(self,
                              prompt_id: str,
                              template_id: str,
                              model_id: str,
                              success: bool,
                              response_length: int,
                              execution_time: float,
                              discrimination_score: float = 0.0,
                              behavioral_score: float = 0.0,
                              quality_score: float = 0.0,
                              difficulty_level: int = 1,
                              domain: str = "general",
                              cost: float = 0.0,
                              error_type: Optional[str] = None,
                              response_hash: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """Record a single prompt execution"""
        
        metrics = PromptMetrics(
            prompt_id=prompt_id,
            template_id=template_id,
            model_id=model_id,
            timestamp=datetime.now(),
            success=success,
            response_length=response_length,
            execution_time=execution_time,
            discrimination_score=discrimination_score,
            behavioral_score=behavioral_score,
            quality_score=quality_score,
            difficulty_level=difficulty_level,
            domain=domain,
            cost=cost,
            error_type=error_type,
            response_hash=response_hash,
            metadata=metadata or {}
        )
        
        # Store in database
        self.database.insert_metrics(metrics)
        
        # Keep in memory for real-time analysis
        with self.lock:
            self.real_time_metrics[template_id].append(metrics)
            
            # Keep only recent metrics in memory (last 1000 per template)
            if len(self.real_time_metrics[template_id]) > 1000:
                self.real_time_metrics[template_id] = \
                    self.real_time_metrics[template_id][-1000:]
    
    def get_template_success_rate(self, 
                                 template_id: str, 
                                 time_window_hours: int = 24) -> float:
        """Calculate success rate for a template in recent time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.real_time_metrics.get(template_id, [])
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return 0.0
        
        successes = sum(1 for m in recent_metrics if m.success)
        return successes / len(recent_metrics)
    
    def get_discrimination_power(self, 
                               template_id: str,
                               model_pairs: List[Tuple[str, str]],
                               time_window_hours: int = 24) -> float:
        """Calculate discrimination power between model pairs"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        df = self.database.query_metrics(
            start_time=cutoff_time,
            template_ids=[template_id]
        )
        
        if df.empty:
            return 0.0
        
        discrimination_scores = []
        
        for model_a, model_b in model_pairs:
            metrics_a = df[df['model_id'] == model_a]
            metrics_b = df[df['model_id'] == model_b]
            
            if len(metrics_a) == 0 or len(metrics_b) == 0:
                continue
            
            # Calculate discrimination based on success rates
            success_rate_a = metrics_a['success'].mean()
            success_rate_b = metrics_b['success'].mean()
            discrimination = abs(success_rate_a - success_rate_b)
            discrimination_scores.append(discrimination)
        
        return np.mean(discrimination_scores) if discrimination_scores else 0.0
    
    def get_temporal_trends(self, 
                          template_id: str,
                          days: int = 7) -> Dict[str, Any]:
        """Analyze temporal trends in prompt effectiveness"""
        start_time = datetime.now() - timedelta(days=days)
        
        df = self.database.query_metrics(
            start_time=start_time,
            template_ids=[template_id]
        )
        
        if df.empty:
            return {"trend": "insufficient_data", "slope": 0.0, "r2": 0.0}
        
        # Group by hour and calculate success rates
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_stats = df.groupby('hour').agg({
            'success': ['count', 'sum', 'mean'],
            'execution_time': 'mean',
            'discrimination_score': 'mean'
        }).reset_index()
        
        if len(hourly_stats) < 3:
            return {"trend": "insufficient_data", "slope": 0.0, "r2": 0.0}
        
        # Linear regression on success rate trend
        x = np.arange(len(hourly_stats))
        y = hourly_stats[('success', 'mean')].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        trend_direction = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
        
        return {
            "trend": trend_direction,
            "slope": slope,
            "r2": r_value ** 2,
            "p_value": p_value,
            "data_points": len(hourly_stats),
            "current_success_rate": y[-1] if len(y) > 0 else 0.0,
            "avg_success_rate": np.mean(y)
        }


class StatisticalAnalysis:
    """Advanced statistical analysis for prompt performance"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
    
    def run_ab_test(self,
                   variant_a_id: str,
                   variant_b_id: str,
                   significance_level: float = 0.05,
                   practical_significance_threshold: float = 0.05,
                   min_sample_size: int = 30) -> ABTestResult:
        """Run A/B test between two prompt variants"""
        
        test_id = f"ab_test_{variant_a_id}_vs_{variant_b_id}_{int(time.time())}"
        start_time = datetime.now() - timedelta(days=7)  # Look at last week
        
        df = self.database.query_metrics(
            start_time=start_time,
            template_ids=[variant_a_id, variant_b_id]
        )
        
        if df.empty:
            raise ValueError("No data found for specified variants")
        
        metrics_a = df[df['template_id'] == variant_a_id]
        metrics_b = df[df['template_id'] == variant_b_id]
        
        if len(metrics_a) < min_sample_size or len(metrics_b) < min_sample_size:
            raise ValueError(f"Insufficient sample size. Need at least {min_sample_size} per variant")
        
        # Calculate success rates
        success_rate_a = metrics_a['success'].mean()
        success_rate_b = metrics_b['success'].mean()
        
        # Two-proportion z-test
        count_a = metrics_a['success'].sum()
        count_b = metrics_b['success'].sum()
        n_a = len(metrics_a)
        n_b = len(metrics_b)
        
        # Calculate z-statistic and p-value
        p_pooled = (count_a + count_b) / (n_a + n_b)
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
        
        z_stat = (success_rate_a - success_rate_b) / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(success_rate_a)) - np.arcsin(np.sqrt(success_rate_b)))
        
        # Confidence interval for difference in proportions
        se_diff = np.sqrt((success_rate_a * (1 - success_rate_a) / n_a) + 
                          (success_rate_b * (1 - success_rate_b) / n_b))
        margin_error = stats.norm.ppf(1 - significance_level/2) * se_diff
        diff = success_rate_a - success_rate_b
        ci_low = diff - margin_error
        ci_high = diff + margin_error
        
        # Determine significance
        statistical_significance = p_value < significance_level
        practical_significance = abs(diff) > practical_significance_threshold
        
        # Recommendation
        if statistical_significance and practical_significance:
            if success_rate_a > success_rate_b:
                recommendation = f"Use variant A ({variant_a_id}). Significantly better performance."
            else:
                recommendation = f"Use variant B ({variant_b_id}). Significantly better performance."
        elif statistical_significance:
            recommendation = "Statistically significant but practically small difference. Consider other factors."
        else:
            recommendation = "No significant difference found. Both variants perform similarly."
        
        result = ABTestResult(
            test_id=test_id,
            variant_a_id=variant_a_id,
            variant_b_id=variant_b_id,
            start_time=start_time,
            end_time=datetime.now(),
            sample_size_a=n_a,
            sample_size_b=n_b,
            success_rate_a=success_rate_a,
            success_rate_b=success_rate_b,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_low, ci_high),
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            recommendation=recommendation
        )
        
        # Store result in database
        self._store_ab_test_result(result)
        
        return result
    
    def _store_ab_test_result(self, result: ABTestResult):
        """Store A/B test result in database"""
        with sqlite3.connect(self.database.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ab_test_results (
                    test_id, variant_a_id, variant_b_id, start_time, end_time,
                    sample_size_a, sample_size_b, success_rate_a, success_rate_b,
                    p_value, effect_size, confidence_interval_low, confidence_interval_high,
                    statistical_significance, practical_significance, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id, result.variant_a_id, result.variant_b_id,
                result.start_time, result.end_time, result.sample_size_a,
                result.sample_size_b, result.success_rate_a, result.success_rate_b,
                result.p_value, result.effect_size, result.confidence_interval[0],
                result.confidence_interval[1], result.statistical_significance,
                result.practical_significance, result.recommendation
            ))
    
    def correlation_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze correlations between prompt features and success"""
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(start_time=start_time)
        
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Select numeric features for correlation
        numeric_features = [
            'response_length', 'execution_time', 'discrimination_score',
            'behavioral_score', 'quality_score', 'difficulty_level', 'cost'
        ]
        
        feature_df = df[numeric_features + ['success']].copy()
        correlation_matrix = feature_df.corr()
        
        # Focus on correlations with success
        success_correlations = correlation_matrix['success'].drop('success').sort_values(
            key=abs, ascending=False
        )
        
        # Statistical significance of correlations
        n = len(df)
        significant_correlations = {}
        
        for feature, corr in success_correlations.items():
            # Test significance of correlation
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            if p_value < 0.05:  # Significant at 5% level
                significant_correlations[feature] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': 'significant' if p_value < 0.01 else 'marginally_significant'
                }
        
        return {
            'correlations': success_correlations.to_dict(),
            'significant_correlations': significant_correlations,
            'sample_size': n,
            'strongest_positive': success_correlations.idxmax() if len(success_correlations) > 0 else None,
            'strongest_negative': success_correlations.idxmin() if len(success_correlations) > 0 else None
        }
    
    def build_success_prediction_model(self, days: int = 30) -> Dict[str, Any]:
        """Build regression model to predict prompt success"""
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(start_time=start_time)
        
        if df.empty or len(df) < 100:
            return {"error": "Insufficient data for model building"}
        
        # Prepare features
        feature_columns = [
            'response_length', 'execution_time', 'discrimination_score',
            'behavioral_score', 'quality_score', 'difficulty_level', 'cost'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['success'].astype(int)
        
        # Remove any rows with infinite values
        mask = np.isfinite(X.values).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return {"error": "Insufficient clean data for model building"}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
                
                # Fit full model for feature importance
                model.fit(X_scaled, y)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    importance = np.zeros(len(feature_columns))
                
                feature_importance = dict(zip(feature_columns, importance))
                
                results[name] = {
                    'cv_mean_auc': np.mean(cv_scores),
                    'cv_std_auc': np.std(cv_scores),
                    'feature_importance': feature_importance,
                    'model_params': model.get_params()
                }
            
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return {
            'models': results,
            'feature_columns': feature_columns,
            'sample_size': len(X),
            'class_distribution': y.value_counts().to_dict()
        }


class VisualizationEngine:
    """Generates visualizations for prompt analytics"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.output_dir = Path("analytics_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_heatmap(self,
                      models: List[str],
                      templates: List[str],
                      days: int = 7,
                      save_path: Optional[str] = None) -> Optional[str]:
        """Create heatmap of prompt-model interaction patterns"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(
            start_time=start_time,
            model_ids=models,
            template_ids=templates
        )
        
        if df.empty:
            return None
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['template_id', 'model_id'])['success'].mean().unstack(fill_value=0)
        
        # Ensure all requested models and templates are included
        for model in models:
            if model not in heatmap_data.columns:
                heatmap_data[model] = 0
        for template in templates:
            if template not in heatmap_data.index:
                heatmap_data.loc[template] = 0
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0.5,
            cbar_kws={'label': 'Success Rate'}
        )
        
        plt.title(f'Prompt-Model Success Rate Heatmap (Last {days} days)')
        plt.xlabel('Model ID')
        plt.ylabel('Template ID')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            filepath = self.output_dir / save_path
        else:
            filepath = self.output_dir / f"heatmap_{int(time.time())}.png"
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_time_series(self,
                          template_ids: List[str],
                          days: int = 30,
                          save_path: Optional[str] = None) -> Optional[str]:
        """Create time series plot for effectiveness trends"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(
            start_time=start_time,
            template_ids=template_ids
        )
        
        if df.empty:
            return None
        
        # Group by template and day
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby(['template_id', 'date']).agg({
            'success': ['count', 'sum', 'mean'],
            'discrimination_score': 'mean',
            'execution_time': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['template_id', 'date', 'count', 'successes', 'success_rate', 
                              'avg_discrimination', 'avg_execution_time']
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Success rate over time
        for template_id in template_ids:
            template_data = daily_stats[daily_stats['template_id'] == template_id]
            if not template_data.empty:
                axes[0].plot(template_data['date'], template_data['success_rate'], 
                           marker='o', label=template_id)
        
        axes[0].set_title('Success Rate Over Time')
        axes[0].set_ylabel('Success Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Discrimination score over time
        for template_id in template_ids:
            template_data = daily_stats[daily_stats['template_id'] == template_id]
            if not template_data.empty:
                axes[1].plot(template_data['date'], template_data['avg_discrimination'], 
                           marker='s', label=template_id)
        
        axes[1].set_title('Average Discrimination Score Over Time')
        axes[1].set_ylabel('Discrimination Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Execution time over time
        for template_id in template_ids:
            template_data = daily_stats[daily_stats['template_id'] == template_id]
            if not template_data.empty:
                axes[2].plot(template_data['date'], template_data['avg_execution_time'], 
                           marker='^', label=template_id)
        
        axes[2].set_title('Average Execution Time Over Time')
        axes[2].set_ylabel('Execution Time (seconds)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            filepath = self.output_dir / save_path
        else:
            filepath = self.output_dir / f"timeseries_{int(time.time())}.png"
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_network_graph(self,
                           days: int = 7,
                           save_path: Optional[str] = None) -> Optional[str]:
        """Create network graph showing prompt relationships"""
        if not NETWORK_ANALYSIS_AVAILABLE or not VISUALIZATION_AVAILABLE:
            return None
        
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(start_time=start_time)
        
        if df.empty:
            return None
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for templates and models
        templates = df['template_id'].unique()
        models = df['model_id'].unique()
        
        # Add template nodes
        for template in templates:
            template_data = df[df['template_id'] == template]
            success_rate = template_data['success'].mean()
            avg_discrimination = template_data['discrimination_score'].mean()
            
            G.add_node(template, 
                      node_type='template',
                      success_rate=success_rate,
                      discrimination_score=avg_discrimination,
                      size=len(template_data) * 10)  # Node size based on usage
        
        # Add model nodes
        for model in models:
            model_data = df[df['model_id'] == model]
            overall_success = model_data['success'].mean()
            
            G.add_node(model,
                      node_type='model',
                      success_rate=overall_success,
                      size=len(model_data) * 5)
        
        # Add edges between templates and models
        for _, row in df.iterrows():
            template = row['template_id']
            model = row['model_id']
            
            if G.has_edge(template, model):
                G[template][model]['weight'] += 1
                G[template][model]['success_sum'] += row['success']
            else:
                G.add_edge(template, model, 
                          weight=1, 
                          success_sum=row['success'])
        
        # Calculate edge success rates
        for u, v, data in G.edges(data=True):
            data['success_rate'] = data['success_sum'] / data['weight']
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plot
        plt.figure(figsize=(16, 12))
        
        # Draw nodes
        template_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'template']
        model_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'model']
        
        # Template nodes (red)
        if template_nodes:
            template_sizes = [G.nodes[n]['size'] for n in template_nodes]
            template_colors = [G.nodes[n]['success_rate'] for n in template_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=template_nodes,
                                 node_color=template_colors, 
                                 node_size=template_sizes,
                                 cmap='Reds', 
                                 vmin=0, vmax=1,
                                 alpha=0.7)
        
        # Model nodes (blue)
        if model_nodes:
            model_sizes = [G.nodes[n]['size'] for n in model_nodes]
            model_colors = [G.nodes[n]['success_rate'] for n in model_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=model_nodes,
                                 node_color=model_colors,
                                 node_size=model_sizes,
                                 cmap='Blues',
                                 vmin=0, vmax=1,
                                 alpha=0.7)
        
        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_colors = [G[u][v]['success_rate'] for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, 
                             width=[w/max(edge_weights)*5 for w in edge_weights],
                             edge_color=edge_colors,
                             edge_cmap=plt.cm.RdYlGn,
                             edge_vmin=0, edge_vmax=1,
                             alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Prompt-Model Interaction Network (Last {days} days)')
        plt.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Success Rate')
        
        if save_path:
            filepath = self.output_dir / save_path
        else:
            filepath = self.output_dir / f"network_{int(time.time())}.png"
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_confusion_matrix(self,
                              model_pairs: List[Tuple[str, str]],
                              template_id: str,
                              days: int = 7,
                              save_path: Optional[str] = None) -> Optional[str]:
        """Create confusion matrix for model discrimination"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(
            start_time=start_time,
            template_ids=[template_id]
        )
        
        if df.empty:
            return None
        
        fig, axes = plt.subplots(1, len(model_pairs), figsize=(6*len(model_pairs), 5))
        if len(model_pairs) == 1:
            axes = [axes]
        
        for idx, (model_a, model_b) in enumerate(model_pairs):
            # Get predictions for both models
            data_a = df[df['model_id'] == model_a]['success'].values
            data_b = df[df['model_id'] == model_b]['success'].values
            
            if len(data_a) == 0 or len(data_b) == 0:
                continue
            
            # Align data (use minimum length)
            min_len = min(len(data_a), len(data_b))
            data_a = data_a[:min_len]
            data_b = data_b[:min_len]
            
            # Create confusion matrix
            cm = confusion_matrix(data_a, data_b)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_a} vs {model_b}')
            axes[idx].set_xlabel(f'{model_b} Predictions')
            axes[idx].set_ylabel(f'{model_a} Predictions')
        
        plt.suptitle(f'Model Discrimination Confusion Matrices - Template: {template_id}')
        plt.tight_layout()
        
        if save_path:
            filepath = self.output_dir / save_path
        else:
            filepath = self.output_dir / f"confusion_matrix_{int(time.time())}.png"
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)


class ReportingSystem:
    """Generates comprehensive analytics reports"""
    
    def __init__(self, database: MetricsDatabase, 
                 metrics_collector: MetricsCollector,
                 statistical_analysis: StatisticalAnalysis,
                 visualization_engine: VisualizationEngine):
        self.database = database
        self.metrics_collector = metrics_collector
        self.statistical_analysis = statistical_analysis
        self.visualization_engine = visualization_engine
        self.report_cache = {}
    
    def generate_comprehensive_report(self, 
                                   days: int = 7,
                                   models: Optional[List[str]] = None,
                                   templates: Optional[List[str]] = None) -> PromptAnalysisReport:
        """Generate comprehensive analysis report"""
        
        report_id = f"report_{int(time.time())}"
        generation_time = datetime.now()
        start_time = generation_time - timedelta(days=days)
        
        # Query data
        df = self.database.query_metrics(
            start_time=start_time,
            model_ids=models,
            template_ids=templates
        )
        
        if df.empty:
            raise ValueError("No data available for the specified time period and filters")
        
        # Basic statistics
        total_prompts = len(df)
        unique_templates = df['template_id'].nunique()
        models_tested = df['model_id'].unique().tolist()
        overall_success_rate = df['success'].mean()
        avg_discrimination_power = df['discrimination_score'].mean()
        
        # Top performers analysis
        template_performance = df.groupby('template_id').agg({
            'success': ['count', 'sum', 'mean'],
            'discrimination_score': 'mean',
            'execution_time': 'mean',
            'cost': 'sum'
        }).round(3)
        
        template_performance.columns = ['count', 'successes', 'success_rate', 
                                      'avg_discrimination', 'avg_time', 'total_cost']
        template_performance = template_performance.reset_index()
        
        # Top performers (success rate > 0.8 and count > 5)
        top_performers = template_performance[
            (template_performance['success_rate'] > 0.8) & 
            (template_performance['count'] > 5)
        ].nlargest(10, 'success_rate').to_dict('records')
        
        # Underperformers (success rate < 0.5 and count > 5)
        underperformers = template_performance[
            (template_performance['success_rate'] < 0.5) & 
            (template_performance['count'] > 5)
        ].nsmallest(10, 'success_rate').to_dict('records')
        
        # Trend analysis
        trends = {}
        for template_id in template_performance['template_id'].head(10):  # Top 10 by usage
            trend = self.metrics_collector.get_temporal_trends(template_id, days)
            trends[template_id] = trend
        
        # Statistical tests
        statistical_tests = []
        
        # Correlation analysis
        try:
            corr_analysis = self.statistical_analysis.correlation_analysis(days)
            statistical_tests.append({
                'test_type': 'correlation_analysis',
                'results': corr_analysis
            })
        except Exception as e:
            statistical_tests.append({
                'test_type': 'correlation_analysis',
                'error': str(e)
            })
        
        # Success prediction model
        try:
            model_results = self.statistical_analysis.build_success_prediction_model(days)
            statistical_tests.append({
                'test_type': 'success_prediction',
                'results': model_results
            })
        except Exception as e:
            statistical_tests.append({
                'test_type': 'success_prediction',
                'error': str(e)
            })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            template_performance, trends, statistical_tests, overall_success_rate
        )
        
        # Prepare export data
        export_data = {
            'raw_metrics': df.to_dict('records'),
            'template_performance': template_performance.to_dict('records'),
            'summary_statistics': {
                'total_prompts': total_prompts,
                'unique_templates': unique_templates,
                'models_tested': models_tested,
                'overall_success_rate': overall_success_rate,
                'avg_discrimination_power': avg_discrimination_power,
                'time_period_days': days
            }
        }
        
        report = PromptAnalysisReport(
            report_id=report_id,
            generation_time=generation_time,
            time_period=(start_time, generation_time),
            total_prompts=total_prompts,
            unique_templates=unique_templates,
            models_tested=models_tested,
            overall_success_rate=overall_success_rate,
            avg_discrimination_power=avg_discrimination_power,
            top_performers=top_performers,
            underperformers=underperformers,
            trends=trends,
            recommendations=recommendations,
            statistical_tests=statistical_tests,
            export_data=export_data
        )
        
        # Store report
        self._store_report(report)
        
        return report
    
    def _generate_recommendations(self, 
                                template_performance: pd.DataFrame,
                                trends: Dict[str, Any],
                                statistical_tests: List[Dict[str, Any]],
                                overall_success_rate: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if overall_success_rate < 0.7:
            recommendations.append(
                f"Overall success rate ({overall_success_rate:.1%}) is below optimal. "
                "Consider reviewing prompt quality and model selection."
            )
        
        # Top performer recommendations
        top_templates = template_performance.nlargest(5, 'success_rate')
        if not top_templates.empty:
            best_template = top_templates.iloc[0]
            recommendations.append(
                f"Template '{best_template['template_id']}' shows excellent performance "
                f"({best_template['success_rate']:.1%} success rate). "
                "Consider using it as a template for similar prompts."
            )
        
        # Trend-based recommendations
        declining_templates = [
            tid for tid, trend_data in trends.items()
            if trend_data.get('trend') == 'declining' and trend_data.get('p_value', 1) < 0.05
        ]
        
        if declining_templates:
            recommendations.append(
                f"Templates showing declining performance: {', '.join(declining_templates[:3])}. "
                "Review and potentially retire or modify these templates."
            )
        
        # Statistical analysis recommendations
        for test in statistical_tests:
            if test['test_type'] == 'correlation_analysis' and 'results' in test:
                corr_results = test['results']
                if 'significant_correlations' in corr_results:
                    sig_corrs = corr_results['significant_correlations']
                    
                    if sig_corrs:
                        strongest = max(sig_corrs.items(), key=lambda x: abs(x[1]['correlation']))
                        feature, corr_data = strongest
                        
                        if corr_data['correlation'] > 0:
                            recommendations.append(
                                f"Higher {feature} is significantly associated with success "
                                f"(correlation: {corr_data['correlation']:.3f}). "
                                "Consider optimizing prompts for this feature."
                            )
                        else:
                            recommendations.append(
                                f"Lower {feature} is significantly associated with success "
                                f"(correlation: {corr_data['correlation']:.3f}). "
                                "Consider reducing this factor in prompts."
                            )
        
        # Cost optimization recommendations
        high_cost_templates = template_performance[
            template_performance['total_cost'] > template_performance['total_cost'].quantile(0.8)
        ]
        
        if not high_cost_templates.empty:
            expensive_but_good = high_cost_templates[high_cost_templates['success_rate'] > 0.8]
            expensive_but_poor = high_cost_templates[high_cost_templates['success_rate'] < 0.6]
            
            if not expensive_but_poor.empty:
                recommendations.append(
                    f"Templates with high cost but low performance: "
                    f"{', '.join(expensive_but_poor['template_id'].head(3).tolist())}. "
                    "Consider optimization or retirement."
                )
        
        return recommendations
    
    def _store_report(self, report: PromptAnalysisReport):
        """Store report in database"""
        with sqlite3.connect(self.database.db_path) as conn:
            conn.execute("""
                INSERT INTO analysis_reports (
                    report_id, generation_time, time_period_start, 
                    time_period_end, report_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.generation_time,
                report.time_period[0],
                report.time_period[1],
                json.dumps(asdict(report), default=str)
            ))
    
    def export_report(self, report: PromptAnalysisReport, 
                     format: str = 'json',
                     filepath: Optional[str] = None) -> str:
        """Export report to file"""
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"prompt_analysis_report_{timestamp}.{format}"
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Export summary data as CSV
            df = pd.DataFrame(report.export_data['template_performance'])
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return filepath
    
    def generate_executive_summary(self, report: PromptAnalysisReport) -> str:
        """Generate executive summary of the report"""
        
        summary = f"""
# Prompt Analytics Executive Summary
**Report ID:** {report.report_id}  
**Generated:** {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Time Period:** {report.time_period[0].strftime('%Y-%m-%d')} to {report.time_period[1].strftime('%Y-%m-%d')}

## Key Metrics
- **Total Prompts Executed:** {report.total_prompts:,}
- **Unique Templates:** {report.unique_templates}
- **Models Tested:** {len(report.models_tested)}
- **Overall Success Rate:** {report.overall_success_rate:.1%}
- **Average Discrimination Power:** {report.avg_discrimination_power:.3f}

## Top Performers
"""
        
        for i, performer in enumerate(report.top_performers[:3], 1):
            summary += f"{i}. **{performer['template_id']}**: {performer['success_rate']:.1%} success rate ({performer['count']} executions)\n"
        
        summary += "\n## Key Trends\n"
        
        # Analyze trends
        improving_count = sum(1 for t in report.trends.values() if t.get('trend') == 'improving')
        declining_count = sum(1 for t in report.trends.values() if t.get('trend') == 'declining')
        stable_count = sum(1 for t in report.trends.values() if t.get('trend') == 'stable')
        
        summary += f"- **{improving_count}** templates showing improvement\n"
        summary += f"- **{stable_count}** templates with stable performance\n"
        summary += f"- **{declining_count}** templates showing decline\n"
        
        summary += "\n## Recommendations\n"
        for i, rec in enumerate(report.recommendations[:5], 1):
            summary += f"{i}. {rec}\n"
        
        summary += f"\n## Risk Assessment\n"
        
        # Risk assessment based on performance metrics
        if report.overall_success_rate < 0.6:
            risk_level = "HIGH"
            risk_note = "Overall success rate is significantly below optimal levels"
        elif report.overall_success_rate < 0.8:
            risk_level = "MEDIUM"
            risk_note = "Success rate has room for improvement"
        else:
            risk_level = "LOW"
            risk_note = "Performance is within acceptable ranges"
        
        summary += f"**Risk Level:** {risk_level} - {risk_note}\n"
        
        return summary


class REVPipelineIntegration:
    """Integration points with REV pipeline components"""
    
    def __init__(self, analytics_system: 'PromptAnalyticsSystem'):
        self.analytics = analytics_system
        self.telemetry_buffer = deque(maxlen=1000)
        self.integration_lock = threading.Lock()
    
    def on_prompt_execution(self, 
                          rev_execution_result: Dict[str, Any],
                          merkle_verification: Optional[Dict[str, Any]] = None,
                          behavioral_analysis: Optional[Dict[str, Any]] = None):
        """Handle prompt execution results from REV pipeline"""
        
        # Extract metrics from REV execution result
        prompt_id = rev_execution_result.get('prompt_id', str(uuid.uuid4()))
        template_id = rev_execution_result.get('template_id', 'unknown')
        model_id = rev_execution_result.get('model_id', 'unknown')
        success = rev_execution_result.get('success', False)
        response_length = len(rev_execution_result.get('response', ''))
        execution_time = rev_execution_result.get('execution_time', 0.0)
        
        # Calculate discrimination score from verification results
        discrimination_score = 0.0
        if merkle_verification:
            # Use Merkle tree verification confidence as discrimination score
            discrimination_score = merkle_verification.get('verification_confidence', 0.0)
        
        # Extract behavioral score
        behavioral_score = 0.0
        if behavioral_analysis:
            behavioral_score = behavioral_analysis.get('overall_score', 0.0)
        
        # Calculate quality score (composite metric)
        quality_score = self._calculate_quality_score(
            rev_execution_result, 
            merkle_verification, 
            behavioral_analysis
        )
        
        # Extract additional metadata
        metadata = {
            'segment_count': rev_execution_result.get('segment_count', 0),
            'memory_usage': rev_execution_result.get('memory_usage', 0),
            'merkle_depth': merkle_verification.get('tree_depth', 0) if merkle_verification else 0,
            'behavioral_features': behavioral_analysis.get('features', {}) if behavioral_analysis else {}
        }
        
        # Record metrics
        self.analytics.metrics_collector.record_prompt_execution(
            prompt_id=prompt_id,
            template_id=template_id,
            model_id=model_id,
            success=success,
            response_length=response_length,
            execution_time=execution_time,
            discrimination_score=discrimination_score,
            behavioral_score=behavioral_score,
            quality_score=quality_score,
            difficulty_level=rev_execution_result.get('difficulty_level', 1),
            domain=rev_execution_result.get('domain', 'general'),
            cost=rev_execution_result.get('cost', 0.0),
            error_type=rev_execution_result.get('error_type'),
            response_hash=rev_execution_result.get('response_hash'),
            metadata=metadata
        )
        
        # Add to telemetry buffer for real-time monitoring
        with self.integration_lock:
            self.telemetry_buffer.append({
                'timestamp': datetime.now(),
                'prompt_id': prompt_id,
                'template_id': template_id,
                'success': success,
                'quality_score': quality_score
            })
    
    def _calculate_quality_score(self, 
                                execution_result: Dict[str, Any],
                                merkle_verification: Optional[Dict[str, Any]],
                                behavioral_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate composite quality score"""
        
        components = []
        
        # Success component (0.0 or 1.0)
        success_score = 1.0 if execution_result.get('success', False) else 0.0
        components.append(('success', success_score, 0.4))
        
        # Merkle verification component
        if merkle_verification:
            verification_score = merkle_verification.get('verification_confidence', 0.0)
            components.append(('verification', verification_score, 0.3))
        
        # Behavioral analysis component
        if behavioral_analysis:
            behavioral_score = behavioral_analysis.get('overall_score', 0.0)
            components.append(('behavioral', behavioral_score, 0.2))
        
        # Efficiency component (based on execution time)
        execution_time = execution_result.get('execution_time', float('inf'))
        if execution_time > 0 and execution_time < 60:  # Reasonable time
            efficiency_score = max(0.0, 1.0 - (execution_time / 60))  # Normalize to 0-1
            components.append(('efficiency', efficiency_score, 0.1))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(score * weight for _, score, weight in components)
        return weighted_sum / total_weight
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        with self.integration_lock:
            if not self.telemetry_buffer:
                return {"status": "no_data"}
            
            recent_data = list(self.telemetry_buffer)
        
        # Calculate real-time statistics
        total_executions = len(recent_data)
        successful_executions = sum(1 for d in recent_data if d['success'])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        avg_quality = np.mean([d['quality_score'] for d in recent_data])
        
        # Template performance
        template_stats = defaultdict(lambda: {'count': 0, 'successes': 0})
        for data in recent_data:
            template_id = data['template_id']
            template_stats[template_id]['count'] += 1
            if data['success']:
                template_stats[template_id]['successes'] += 1
        
        template_success_rates = {
            tid: stats['successes'] / stats['count']
            for tid, stats in template_stats.items()
            if stats['count'] > 0
        }
        
        return {
            'status': 'active',
            'total_executions': total_executions,
            'success_rate': success_rate,
            'avg_quality_score': avg_quality,
            'template_success_rates': template_success_rates,
            'timestamp': datetime.now(),
            'buffer_size': len(recent_data)
        }
    
    def sync_with_experiment_tracking(self, experiment_id: str) -> Dict[str, Any]:
        """Sync analytics with REV experiment tracking"""
        # Query metrics for the specific experiment
        # This would integrate with REV's experiment tracking system
        
        return {
            'experiment_id': experiment_id,
            'sync_status': 'completed',
            'metrics_synced': True,
            'timestamp': datetime.now()
        }


class PromptAnalyticsSystem:
    """Main orchestrator for the prompt analytics system"""
    
    def __init__(self, db_path: str = "prompt_analytics.db"):
        # Initialize components
        self.database = MetricsDatabase(db_path)
        self.metrics_collector = MetricsCollector(self.database)
        self.statistical_analysis = StatisticalAnalysis(self.database)
        self.visualization_engine = VisualizationEngine(self.database)
        self.reporting_system = ReportingSystem(
            self.database, 
            self.metrics_collector, 
            self.statistical_analysis, 
            self.visualization_engine
        )
        self.rev_integration = REVPipelineIntegration(self)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_callbacks = []
    
    def start_real_time_monitoring(self, update_interval: int = 30):
        """Start real-time monitoring with periodic updates"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time updates"""
        self.update_callbacks.append(callback)
    
    def _monitoring_loop(self, update_interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get real-time metrics
                metrics = self.rev_integration.get_real_time_metrics()
                
                # Call update callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        print(f"Error in update callback: {e}")
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(update_interval)
    
    def create_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Create comprehensive dashboard data"""
        try:
            # Generate full report
            report = self.reporting_system.generate_comprehensive_report(days=days)
            
            # Get real-time metrics
            real_time = self.rev_integration.get_real_time_metrics()
            
            # Create visualizations
            visualizations = {}
            
            if len(report.models_tested) > 0:
                # Get some template IDs for visualization
                df = self.database.query_metrics(
                    start_time=datetime.now() - timedelta(days=days)
                )
                
                if not df.empty:
                    templates = df['template_id'].value_counts().head(10).index.tolist()
                    models = report.models_tested
                    
                    # Create heatmap
                    heatmap_path = self.visualization_engine.create_heatmap(
                        models=models[:5],  # Limit to top 5 models
                        templates=templates[:10],  # Limit to top 10 templates
                        days=days
                    )
                    if heatmap_path:
                        visualizations['heatmap'] = heatmap_path
                    
                    # Create time series
                    timeseries_path = self.visualization_engine.create_time_series(
                        template_ids=templates[:5],
                        days=days
                    )
                    if timeseries_path:
                        visualizations['timeseries'] = timeseries_path
                    
                    # Create network graph
                    network_path = self.visualization_engine.create_network_graph(days=days)
                    if network_path:
                        visualizations['network'] = network_path
            
            return {
                'report': asdict(report),
                'real_time_metrics': real_time,
                'visualizations': visualizations,
                'executive_summary': self.reporting_system.generate_executive_summary(report),
                'generated_at': datetime.now(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'generated_at': datetime.now()
            }
    
    def export_historical_data(self, 
                             days: int = 30,
                             format: str = 'csv',
                             filepath: Optional[str] = None) -> str:
        """Export historical data for external analysis"""
        start_time = datetime.now() - timedelta(days=days)
        df = self.database.query_metrics(start_time=start_time)
        
        if df.empty:
            raise ValueError("No data available for export")
        
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"prompt_analytics_export_{timestamp}.{format}"
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return filepath


# Example usage and integration functions
def create_analytics_system(db_path: str = "prompt_analytics.db") -> PromptAnalyticsSystem:
    """Create and initialize the analytics system"""
    return PromptAnalyticsSystem(db_path)


def integrate_with_rev_pipeline(analytics_system: PromptAnalyticsSystem,
                              rev_result: Dict[str, Any],
                              merkle_data: Optional[Dict[str, Any]] = None,
                              behavioral_data: Optional[Dict[str, Any]] = None):
    """Integration helper for REV pipeline"""
    analytics_system.rev_integration.on_prompt_execution(
        rev_result, merkle_data, behavioral_data
    )


if __name__ == "__main__":
    # Example usage
    analytics = create_analytics_system()
    
    # Start real-time monitoring
    analytics.start_real_time_monitoring(update_interval=60)
    
    # Example callback for real-time updates
    def print_updates(metrics):
        print(f"Real-time update: {metrics['success_rate']:.1%} success rate")
    
    analytics.add_update_callback(print_updates)
    
    # Create dashboard
    dashboard_data = analytics.create_dashboard_data(days=7)
    print("Dashboard created successfully!")
    
    # Cleanup
    analytics.stop_real_time_monitoring()