#!/usr/bin/env python3
"""
Unified Model Analysis System

This module provides comprehensive analytical output from model fingerprints
to detect patterns that indicate various model relationships:
- Adversarial modifications (prompt-specific divergences)
- Scaling relationships (same family, different sizes)
- Family differences (similar behavior, different architectures)
- Fine-tuning variations
- Quantization effects
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import find_peaks
import logging

from ..hdc.unified_fingerprint import UnifiedFingerprint
from ..hypervector.similarity import AdvancedSimilarity

logger = logging.getLogger(__name__)


class ModelRelationship(Enum):
    """Types of relationships between models"""
    IDENTICAL = "identical"
    SCALED_VERSION = "scaled_version"  # Same family, different size
    FINE_TUNED = "fine_tuned"  # Same base, different training
    QUANTIZED = "quantized"  # Same model, reduced precision
    ADVERSARIAL = "adversarial"  # Maliciously modified
    SAME_FAMILY = "same_family"  # Same architecture family
    DIFFERENT_FAMILY = "different_family"  # Different architecture
    UNRELATED = "unrelated"  # No clear relationship


@dataclass
class LayerTransition:
    """Represents a significant transition point in model processing"""
    layer_idx: int
    transition_strength: float
    transition_type: str  # "attention_shift", "representation_change", "semantic_boundary"
    affected_dimensions: List[int]


@dataclass
class BehavioralPhase:
    """Represents a phase of model behavior across layers"""
    start_layer: int
    end_layer: int
    phase_type: str  # "encoding", "reasoning", "decoding", "attention_focus"
    characteristic_vector: np.ndarray
    stability: float  # How stable this phase is


@dataclass
class DivergencePoint:
    """Point where models diverge in behavior"""
    prompt_idx: int
    layer_idx: int
    divergence_magnitude: float
    divergence_type: str  # "sudden", "gradual", "oscillating"
    affected_tokens: List[str]
    statistical_significance: float


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis output for model comparison"""
    # Core similarity metrics
    overall_similarity: float
    layer_wise_similarity: np.ndarray
    prompt_wise_similarity: np.ndarray
    
    # Behavioral patterns
    behavioral_phases_a: List[BehavioralPhase]
    behavioral_phases_b: List[BehavioralPhase]
    phase_alignment: float  # How well phases align between models
    
    # Transitions and boundaries
    transitions_a: List[LayerTransition]
    transitions_b: List[LayerTransition]
    transition_correlation: float
    
    # Divergence analysis
    divergence_points: List[DivergencePoint]
    max_divergence: float
    divergence_pattern: str  # "consistent", "prompt_specific", "layer_specific", "random"
    
    # Scaling analysis
    layer_ratio: float  # Ratio of layer counts/depths
    magnitude_ratio: float  # Ratio of activation magnitudes
    complexity_ratio: float  # Ratio of behavioral complexity
    scaling_consistency: float  # How consistent scaling is across layers
    
    # Architecture insights
    attention_pattern_similarity: float
    feedforward_pattern_similarity: float
    residual_pattern_similarity: float
    architectural_signature_distance: float
    
    # Relationship inference
    inferred_relationship: ModelRelationship
    relationship_confidence: float
    relationship_evidence: Dict[str, Any]
    
    # Anomaly detection
    anomalous_behaviors: List[Dict[str, Any]]
    anomaly_score: float
    potential_threats: List[str]
    
    # Metadata
    model_a_id: str
    model_b_id: str
    analysis_timestamp: str
    analysis_parameters: Dict[str, Any]


class UnifiedModelAnalyzer:
    """
    Performs comprehensive model analysis to detect patterns and infer relationships.
    """
    
    def __init__(self, 
                 sensitivity: float = 0.1,
                 phase_min_length: int = 3,
                 transition_threshold: float = 0.2):
        """
        Initialize the unified analyzer.
        
        Args:
            sensitivity: Detection sensitivity (0-1)
            phase_min_length: Minimum layers for a behavioral phase
            transition_threshold: Threshold for detecting transitions
        """
        self.sensitivity = sensitivity
        self.phase_min_length = phase_min_length
        self.transition_threshold = transition_threshold
        self.similarity_calculator = AdvancedSimilarity()
        
    def analyze_models(self,
                      fingerprints_a: List[UnifiedFingerprint],
                      fingerprints_b: List[UnifiedFingerprint],
                      prompts: Optional[List[str]] = None,
                      layer_activations_a: Optional[Dict[int, np.ndarray]] = None,
                      layer_activations_b: Optional[Dict[int, np.ndarray]] = None) -> ComprehensiveAnalysis:
        """
        Perform comprehensive analysis of two models.
        
        Args:
            fingerprints_a: Fingerprints from model A
            fingerprints_b: Fingerprints from model B
            prompts: Original prompts used
            layer_activations_a: Optional layer activations from model A
            layer_activations_b: Optional layer activations from model B
            
        Returns:
            Comprehensive analysis with patterns and inferred relationships
        """
        logger.info("Starting comprehensive model analysis...")
        
        # Compute basic similarities
        overall_sim = self._compute_overall_similarity(fingerprints_a, fingerprints_b)
        layer_wise_sim = self._compute_layer_wise_similarity(fingerprints_a, fingerprints_b)
        prompt_wise_sim = self._compute_prompt_wise_similarity(fingerprints_a, fingerprints_b)
        
        # Detect behavioral phases
        phases_a = self._detect_behavioral_phases(fingerprints_a, layer_activations_a)
        phases_b = self._detect_behavioral_phases(fingerprints_b, layer_activations_b)
        phase_alignment = self._compute_phase_alignment(phases_a, phases_b)
        
        # Detect transitions
        transitions_a = self._detect_transitions(fingerprints_a, layer_activations_a)
        transitions_b = self._detect_transitions(fingerprints_b, layer_activations_b)
        transition_corr = self._compute_transition_correlation(transitions_a, transitions_b)
        
        # Analyze divergences
        divergence_points = self._find_divergence_points(
            fingerprints_a, fingerprints_b, prompts
        )
        max_divergence = max([d.divergence_magnitude for d in divergence_points]) if divergence_points else 0
        divergence_pattern = self._classify_divergence_pattern(divergence_points)
        
        # Scaling analysis
        scaling_metrics = self._analyze_scaling_relationship(
            fingerprints_a, fingerprints_b, phases_a, phases_b
        )
        
        # Architecture analysis
        arch_metrics = self._analyze_architectural_patterns(
            fingerprints_a, fingerprints_b, layer_activations_a, layer_activations_b
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalous_behaviors(
            fingerprints_a, fingerprints_b, divergence_points
        )
        
        # Infer relationship
        relationship, confidence, evidence = self._infer_model_relationship(
            overall_sim, layer_wise_sim, prompt_wise_sim,
            phase_alignment, transition_corr,
            divergence_pattern, scaling_metrics, arch_metrics,
            anomalies
        )
        
        # Identify potential threats
        threats = self._identify_potential_threats(
            divergence_points, anomalies, relationship
        )
        
        return ComprehensiveAnalysis(
            overall_similarity=overall_sim,
            layer_wise_similarity=layer_wise_sim,
            prompt_wise_similarity=prompt_wise_sim,
            behavioral_phases_a=phases_a,
            behavioral_phases_b=phases_b,
            phase_alignment=phase_alignment,
            transitions_a=transitions_a,
            transitions_b=transitions_b,
            transition_correlation=transition_corr,
            divergence_points=divergence_points,
            max_divergence=max_divergence,
            divergence_pattern=divergence_pattern,
            layer_ratio=scaling_metrics["layer_ratio"],
            magnitude_ratio=scaling_metrics["magnitude_ratio"],
            complexity_ratio=scaling_metrics["complexity_ratio"],
            scaling_consistency=scaling_metrics["consistency"],
            attention_pattern_similarity=arch_metrics["attention_similarity"],
            feedforward_pattern_similarity=arch_metrics["feedforward_similarity"],
            residual_pattern_similarity=arch_metrics["residual_similarity"],
            architectural_signature_distance=arch_metrics["signature_distance"],
            inferred_relationship=relationship,
            relationship_confidence=confidence,
            relationship_evidence=evidence,
            anomalous_behaviors=anomalies,
            anomaly_score=len(anomalies) / 10.0,  # Normalized score
            potential_threats=threats,
            model_a_id=fingerprints_a[0].model_id if fingerprints_a else "unknown",
            model_b_id=fingerprints_b[0].model_id if fingerprints_b else "unknown",
            analysis_timestamp=str(np.datetime64('now')),
            analysis_parameters={
                "sensitivity": self.sensitivity,
                "phase_min_length": self.phase_min_length,
                "transition_threshold": self.transition_threshold
            }
        )
    
    def _compute_overall_similarity(self, 
                                   fps_a: List[UnifiedFingerprint],
                                   fps_b: List[UnifiedFingerprint]) -> float:
        """Compute overall similarity between fingerprint sets."""
        if not fps_a or not fps_b:
            return 0.0
        
        # Average unified hypervectors
        avg_a = np.mean([fp.unified_hypervector for fp in fps_a], axis=0)
        avg_b = np.mean([fp.unified_hypervector for fp in fps_b], axis=0)
        
        return 1 - cosine(avg_a, avg_b)
    
    def _compute_layer_wise_similarity(self,
                                      fps_a: List[UnifiedFingerprint],
                                      fps_b: List[UnifiedFingerprint]) -> np.ndarray:
        """Compute similarity at each layer."""
        if not fps_a or not fps_b:
            return np.array([])
        
        # Extract pathway hypervectors (contain layer information)
        pathways_a = [fp.pathway_hypervector for fp in fps_a]
        pathways_b = [fp.pathway_hypervector for fp in fps_b]
        
        # Compute similarities across layers
        n_layers = min(len(pathways_a[0]) // 100, len(pathways_b[0]) // 100)
        layer_sims = []
        
        for layer_idx in range(n_layers):
            # Extract layer-specific portions
            layer_vecs_a = [p[layer_idx*100:(layer_idx+1)*100] for p in pathways_a]
            layer_vecs_b = [p[layer_idx*100:(layer_idx+1)*100] for p in pathways_b]
            
            # Average and compare
            avg_layer_a = np.mean(layer_vecs_a, axis=0)
            avg_layer_b = np.mean(layer_vecs_b, axis=0)
            
            sim = 1 - cosine(avg_layer_a, avg_layer_b) if len(avg_layer_a) > 0 else 0
            layer_sims.append(sim)
        
        return np.array(layer_sims)
    
    def _compute_prompt_wise_similarity(self,
                                       fps_a: List[UnifiedFingerprint],
                                       fps_b: List[UnifiedFingerprint]) -> np.ndarray:
        """Compute similarity for each prompt."""
        n_prompts = min(len(fps_a), len(fps_b))
        prompt_sims = []
        
        for i in range(n_prompts):
            sim = 1 - cosine(fps_a[i].unified_hypervector, fps_b[i].unified_hypervector)
            prompt_sims.append(sim)
        
        return np.array(prompt_sims)
    
    def _detect_behavioral_phases(self,
                                 fingerprints: List[UnifiedFingerprint],
                                 layer_activations: Optional[Dict[int, np.ndarray]]) -> List[BehavioralPhase]:
        """Detect distinct behavioral phases across layers."""
        phases = []
        
        if not fingerprints:
            return phases
        
        # Use pathway hypervectors to detect phases
        pathway_matrix = np.array([fp.pathway_hypervector for fp in fingerprints])
        
        # Sliding window to detect stable regions
        window_size = self.phase_min_length
        n_windows = len(pathway_matrix[0]) // 100 - window_size + 1
        
        phase_boundaries = [0]
        current_phase_vec = None
        
        for i in range(n_windows):
            window_vecs = pathway_matrix[:, i*100:(i+window_size)*100]
            window_mean = np.mean(window_vecs, axis=0)
            
            if current_phase_vec is None:
                current_phase_vec = window_mean
            else:
                # Check if this is significantly different from current phase
                diff = np.linalg.norm(window_mean - current_phase_vec)
                if diff > self.transition_threshold:
                    phase_boundaries.append(i)
                    current_phase_vec = window_mean
        
        phase_boundaries.append(n_windows + window_size - 1)
        
        # Create phase objects
        for i in range(len(phase_boundaries) - 1):
            start = phase_boundaries[i]
            end = phase_boundaries[i + 1]
            
            # Determine phase type based on patterns
            phase_vec = np.mean(pathway_matrix[:, start*100:end*100], axis=0)
            phase_type = self._classify_phase_type(phase_vec, start, end)
            
            phases.append(BehavioralPhase(
                start_layer=start,
                end_layer=end,
                phase_type=phase_type,
                characteristic_vector=phase_vec,
                stability=self._compute_phase_stability(pathway_matrix, start, end)
            ))
        
        return phases
    
    def _classify_phase_type(self, phase_vec: np.ndarray, start: int, end: int) -> str:
        """Classify the type of behavioral phase."""
        # Simple heuristic based on position and characteristics
        total_layers = 80  # Approximate
        position = (start + end) / 2 / total_layers
        
        if position < 0.2:
            return "encoding"
        elif position < 0.4:
            return "attention_focus"
        elif position < 0.7:
            return "reasoning"
        else:
            return "decoding"
    
    def _compute_phase_stability(self, pathway_matrix: np.ndarray, start: int, end: int) -> float:
        """Compute how stable a phase is."""
        phase_vecs = pathway_matrix[:, start*100:end*100]
        
        # Compute variance across the phase
        if phase_vecs.size > 0:
            variance = np.var(phase_vecs)
            # Convert to stability score (lower variance = higher stability)
            stability = 1.0 / (1.0 + variance)
        else:
            stability = 0.0
        
        return stability
    
    def _compute_phase_alignment(self,
                                phases_a: List[BehavioralPhase],
                                phases_b: List[BehavioralPhase]) -> float:
        """Compute how well behavioral phases align between models."""
        if not phases_a or not phases_b:
            return 0.0
        
        # Compare phase boundaries and types
        alignment_scores = []
        
        for phase_a in phases_a:
            best_match = 0.0
            for phase_b in phases_b:
                # Check boundary overlap
                overlap_start = max(phase_a.start_layer, phase_b.start_layer)
                overlap_end = min(phase_a.end_layer, phase_b.end_layer)
                
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / (
                        phase_a.end_layer - phase_a.start_layer
                    )
                    
                    # Check type match
                    type_match = 1.0 if phase_a.phase_type == phase_b.phase_type else 0.5
                    
                    # Check vector similarity
                    vec_sim = 1 - cosine(
                        phase_a.characteristic_vector,
                        phase_b.characteristic_vector
                    )
                    
                    match_score = overlap_ratio * type_match * vec_sim
                    best_match = max(best_match, match_score)
            
            alignment_scores.append(best_match)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _detect_transitions(self,
                           fingerprints: List[UnifiedFingerprint],
                           layer_activations: Optional[Dict[int, np.ndarray]]) -> List[LayerTransition]:
        """Detect significant transitions in model processing."""
        transitions = []
        
        if not fingerprints:
            return transitions
        
        # Analyze pathway hypervectors for transitions
        pathway_matrix = np.array([fp.pathway_hypervector for fp in fingerprints])
        
        # Compute differences between consecutive layers
        n_layers = len(pathway_matrix[0]) // 100
        
        for i in range(1, n_layers):
            prev_layer = pathway_matrix[:, (i-1)*100:i*100]
            curr_layer = pathway_matrix[:, i*100:(i+1)*100]
            
            # Compute transition strength
            diff = np.mean(np.abs(curr_layer - prev_layer))
            
            if diff > self.transition_threshold:
                # Identify affected dimensions
                dim_diffs = np.abs(np.mean(curr_layer, axis=0) - np.mean(prev_layer, axis=0))
                affected_dims = np.where(dim_diffs > np.percentile(dim_diffs, 90))[0].tolist()
                
                # Classify transition type
                trans_type = self._classify_transition_type(
                    prev_layer, curr_layer, i, n_layers
                )
                
                transitions.append(LayerTransition(
                    layer_idx=i,
                    transition_strength=float(diff),
                    transition_type=trans_type,
                    affected_dimensions=affected_dims
                ))
        
        return transitions
    
    def _classify_transition_type(self,
                                 prev_layer: np.ndarray,
                                 curr_layer: np.ndarray,
                                 layer_idx: int,
                                 total_layers: int) -> str:
        """Classify the type of transition."""
        # Simple heuristics based on patterns
        mean_change = np.mean(curr_layer) - np.mean(prev_layer)
        std_change = np.std(curr_layer) - np.std(prev_layer)
        
        if abs(mean_change) > 0.1:
            return "representation_change"
        elif abs(std_change) > 0.05:
            return "attention_shift"
        else:
            return "semantic_boundary"
    
    def _compute_transition_correlation(self,
                                       trans_a: List[LayerTransition],
                                       trans_b: List[LayerTransition]) -> float:
        """Compute correlation between transition patterns."""
        if not trans_a or not trans_b:
            return 0.0
        
        # Create transition vectors
        max_layer = max(
            max([t.layer_idx for t in trans_a]) if trans_a else 0,
            max([t.layer_idx for t in trans_b]) if trans_b else 0
        )
        
        trans_vec_a = np.zeros(max_layer + 1)
        trans_vec_b = np.zeros(max_layer + 1)
        
        for t in trans_a:
            trans_vec_a[t.layer_idx] = t.transition_strength
        
        for t in trans_b:
            trans_vec_b[t.layer_idx] = t.transition_strength
        
        # Compute correlation
        if np.std(trans_vec_a) > 0 and np.std(trans_vec_b) > 0:
            correlation = np.corrcoef(trans_vec_a, trans_vec_b)[0, 1]
        else:
            correlation = 0.0
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _find_divergence_points(self,
                               fps_a: List[UnifiedFingerprint],
                               fps_b: List[UnifiedFingerprint],
                               prompts: Optional[List[str]]) -> List[DivergencePoint]:
        """Find points where models diverge significantly."""
        divergences = []
        
        n_prompts = min(len(fps_a), len(fps_b))
        
        for prompt_idx in range(n_prompts):
            fp_a = fps_a[prompt_idx]
            fp_b = fps_b[prompt_idx]
            
            # Compare pathway vectors layer by layer
            pathway_a = fp_a.pathway_hypervector
            pathway_b = fp_b.pathway_hypervector
            
            n_layers = min(len(pathway_a), len(pathway_b)) // 100
            
            for layer_idx in range(n_layers):
                layer_a = pathway_a[layer_idx*100:(layer_idx+1)*100]
                layer_b = pathway_b[layer_idx*100:(layer_idx+1)*100]
                
                # Compute divergence
                divergence = np.linalg.norm(layer_a - layer_b)
                
                if divergence > self.sensitivity:
                    # Analyze divergence pattern
                    div_type = self._classify_divergence_type(
                        layer_a, layer_b, layer_idx, n_layers
                    )
                    
                    # Statistical significance
                    if len(layer_a) > 0 and len(layer_b) > 0:
                        _, p_value = stats.ks_2samp(layer_a, layer_b)
                        significance = 1 - p_value
                    else:
                        significance = 0.0
                    
                    divergences.append(DivergencePoint(
                        prompt_idx=prompt_idx,
                        layer_idx=layer_idx,
                        divergence_magnitude=float(divergence),
                        divergence_type=div_type,
                        affected_tokens=[],  # Would need token info
                        statistical_significance=significance
                    ))
        
        return divergences
    
    def _classify_divergence_type(self,
                                 vec_a: np.ndarray,
                                 vec_b: np.ndarray,
                                 layer_idx: int,
                                 total_layers: int) -> str:
        """Classify the type of divergence."""
        diff = vec_a - vec_b
        
        # Check if divergence is sudden or gradual
        if np.max(np.abs(diff)) > 2 * np.mean(np.abs(diff)):
            return "sudden"
        elif np.std(diff) < 0.1:
            return "gradual"
        else:
            return "oscillating"
    
    def _classify_divergence_pattern(self, divergences: List[DivergencePoint]) -> str:
        """Classify overall divergence pattern."""
        if not divergences:
            return "none"
        
        # Analyze distribution of divergences
        prompt_counts = {}
        layer_counts = {}
        
        for div in divergences:
            prompt_counts[div.prompt_idx] = prompt_counts.get(div.prompt_idx, 0) + 1
            layer_counts[div.layer_idx] = layer_counts.get(div.layer_idx, 0) + 1
        
        # Check if divergences are concentrated
        max_prompt_count = max(prompt_counts.values())
        max_layer_count = max(layer_counts.values())
        total_divs = len(divergences)
        
        if max_prompt_count > 0.7 * total_divs:
            return "prompt_specific"
        elif max_layer_count > 0.7 * total_divs:
            return "layer_specific"
        elif len(divergences) > 10 and np.std(list(prompt_counts.values())) < 2:
            return "consistent"
        else:
            return "random"
    
    def _analyze_scaling_relationship(self,
                                     fps_a: List[UnifiedFingerprint],
                                     fps_b: List[UnifiedFingerprint],
                                     phases_a: List[BehavioralPhase],
                                     phases_b: List[BehavioralPhase]) -> Dict[str, float]:
        """Analyze if models have a scaling relationship."""
        metrics = {}
        
        # Layer ratio
        n_phases_a = len(phases_a)
        n_phases_b = len(phases_b)
        metrics["layer_ratio"] = n_phases_a / n_phases_b if n_phases_b > 0 else 1.0
        
        # Magnitude ratio
        mag_a = np.mean([np.linalg.norm(fp.unified_hypervector) for fp in fps_a])
        mag_b = np.mean([np.linalg.norm(fp.unified_hypervector) for fp in fps_b])
        metrics["magnitude_ratio"] = mag_a / mag_b if mag_b > 0 else 1.0
        
        # Complexity ratio (entropy-based)
        entropy_a = stats.entropy(np.abs(np.mean([fp.unified_hypervector for fp in fps_a], axis=0)) + 1e-10)
        entropy_b = stats.entropy(np.abs(np.mean([fp.unified_hypervector for fp in fps_b], axis=0)) + 1e-10)
        metrics["complexity_ratio"] = entropy_a / entropy_b if entropy_b > 0 else 1.0
        
        # Scaling consistency
        # Check if ratios are consistent (would indicate scaling)
        ratios = [metrics["layer_ratio"], metrics["magnitude_ratio"], metrics["complexity_ratio"]]
        metrics["consistency"] = 1.0 - np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 0.0
        
        return metrics
    
    def _analyze_architectural_patterns(self,
                                       fps_a: List[UnifiedFingerprint],
                                       fps_b: List[UnifiedFingerprint],
                                       layer_acts_a: Optional[Dict[int, np.ndarray]],
                                       layer_acts_b: Optional[Dict[int, np.ndarray]]) -> Dict[str, float]:
        """Analyze architectural pattern similarities."""
        metrics = {}
        
        # Extract patterns from pathway hypervectors
        pathways_a = np.array([fp.pathway_hypervector for fp in fps_a])
        pathways_b = np.array([fp.pathway_hypervector for fp in fps_b])
        
        # Attention pattern similarity (using FFT to detect periodic patterns)
        fft_a = np.abs(np.fft.fft(np.mean(pathways_a, axis=0)))
        fft_b = np.abs(np.fft.fft(np.mean(pathways_b, axis=0)))
        metrics["attention_similarity"] = 1 - cosine(fft_a[:100], fft_b[:100])
        
        # Feedforward pattern (using autocorrelation)
        auto_a = np.correlate(np.mean(pathways_a, axis=0), np.mean(pathways_a, axis=0), mode='same')
        auto_b = np.correlate(np.mean(pathways_b, axis=0), np.mean(pathways_b, axis=0), mode='same')
        metrics["feedforward_similarity"] = 1 - cosine(auto_a[:100], auto_b[:100])
        
        # Residual pattern (using differences)
        if pathways_a.shape[1] > 200 and pathways_b.shape[1] > 200:
            residual_a = pathways_a[:, 100:200] - pathways_a[:, :100]
            residual_b = pathways_b[:, 100:200] - pathways_b[:, :100]
            metrics["residual_similarity"] = 1 - cosine(
                np.mean(residual_a, axis=0),
                np.mean(residual_b, axis=0)
            )
        else:
            metrics["residual_similarity"] = 0.5
        
        # Overall architectural signature
        sig_a = np.concatenate([fft_a[:50], auto_a[:50]])
        sig_b = np.concatenate([fft_b[:50], auto_b[:50]])
        metrics["signature_distance"] = euclidean(sig_a, sig_b)
        
        return metrics
    
    def _detect_anomalous_behaviors(self,
                                   fps_a: List[UnifiedFingerprint],
                                   fps_b: List[UnifiedFingerprint],
                                   divergences: List[DivergencePoint]) -> List[Dict[str, Any]]:
        """Detect anomalous behaviors that might indicate tampering."""
        anomalies = []
        
        # Check for sudden quality drops
        qualities_a = [fp.fingerprint_quality for fp in fps_a]
        qualities_b = [fp.fingerprint_quality for fp in fps_b]
        
        if qualities_a and np.min(qualities_a) < 0.5 * np.mean(qualities_a):
            anomalies.append({
                "type": "quality_drop",
                "model": "a",
                "severity": 1 - np.min(qualities_a) / np.mean(qualities_a),
                "description": "Sudden quality drop detected in model A"
            })
        
        if qualities_b and np.min(qualities_b) < 0.5 * np.mean(qualities_b):
            anomalies.append({
                "type": "quality_drop",
                "model": "b",
                "severity": 1 - np.min(qualities_b) / np.mean(qualities_b),
                "description": "Sudden quality drop detected in model B"
            })
        
        # Check for inconsistent binding
        bindings_a = [fp.binding_strength for fp in fps_a]
        bindings_b = [fp.binding_strength for fp in fps_b]
        
        if bindings_a and np.std(bindings_a) > 0.3:
            anomalies.append({
                "type": "inconsistent_binding",
                "model": "a",
                "severity": np.std(bindings_a),
                "description": "Inconsistent binding patterns in model A"
            })
        
        # Check for concentrated divergences (potential backdoor)
        if divergences:
            prompt_specific_divs = {}
            for div in divergences:
                if div.divergence_magnitude > 0.5:
                    prompt_specific_divs[div.prompt_idx] = prompt_specific_divs.get(div.prompt_idx, 0) + 1
            
            for prompt_idx, count in prompt_specific_divs.items():
                if count > 5:  # Many divergences on same prompt
                    anomalies.append({
                        "type": "prompt_specific_divergence",
                        "prompt_idx": prompt_idx,
                        "severity": count / 10.0,
                        "description": f"Suspicious prompt-specific behavior on prompt {prompt_idx}"
                    })
        
        return anomalies
    
    def _infer_model_relationship(self,
                                 overall_sim: float,
                                 layer_wise_sim: np.ndarray,
                                 prompt_wise_sim: np.ndarray,
                                 phase_alignment: float,
                                 transition_corr: float,
                                 divergence_pattern: str,
                                 scaling_metrics: Dict[str, float],
                                 arch_metrics: Dict[str, float],
                                 anomalies: List[Dict[str, Any]]) -> Tuple[ModelRelationship, float, Dict[str, Any]]:
        """Infer the relationship between models based on all metrics."""
        evidence = {}
        
        # Check for identical models
        if overall_sim > 0.99 and phase_alignment > 0.95:
            evidence["high_similarity"] = overall_sim
            evidence["phase_match"] = phase_alignment
            return ModelRelationship.IDENTICAL, 0.95, evidence
        
        # Check for adversarial modification
        if (divergence_pattern == "prompt_specific" and 
            len(anomalies) > 0 and 
            overall_sim > 0.85):
            evidence["prompt_specific_divergence"] = divergence_pattern
            evidence["anomalies"] = len(anomalies)
            evidence["base_similarity"] = overall_sim
            return ModelRelationship.ADVERSARIAL, 0.8, evidence
        
        # Check for scaled version
        if (scaling_metrics["consistency"] > 0.7 and
            phase_alignment > 0.7 and
            transition_corr > 0.6):
            evidence["scaling_consistency"] = scaling_metrics["consistency"]
            evidence["phase_alignment"] = phase_alignment
            evidence["layer_ratio"] = scaling_metrics["layer_ratio"]
            return ModelRelationship.SCALED_VERSION, 0.75, evidence
        
        # Check for fine-tuned version
        if (overall_sim > 0.8 and
            arch_metrics["signature_distance"] < 10 and
            divergence_pattern == "consistent"):
            evidence["base_similarity"] = overall_sim
            evidence["architecture_match"] = 1 - arch_metrics["signature_distance"] / 100
            return ModelRelationship.FINE_TUNED, 0.7, evidence
        
        # Check for quantized version
        if (overall_sim > 0.9 and
            scaling_metrics["magnitude_ratio"] < 0.8 and
            phase_alignment > 0.85):
            evidence["similarity"] = overall_sim
            evidence["magnitude_reduction"] = 1 - scaling_metrics["magnitude_ratio"]
            return ModelRelationship.QUANTIZED, 0.65, evidence
        
        # Check for same family
        if (arch_metrics["attention_similarity"] > 0.7 and
            arch_metrics["feedforward_similarity"] > 0.7 and
            phase_alignment > 0.5):
            evidence["architecture_similarity"] = (
                arch_metrics["attention_similarity"] + 
                arch_metrics["feedforward_similarity"]
            ) / 2
            return ModelRelationship.SAME_FAMILY, 0.6, evidence
        
        # Check for different family
        if overall_sim > 0.5:
            evidence["moderate_similarity"] = overall_sim
            return ModelRelationship.DIFFERENT_FAMILY, 0.5, evidence
        
        # Otherwise unrelated
        evidence["low_similarity"] = overall_sim
        return ModelRelationship.UNRELATED, 0.7, evidence
    
    def _identify_potential_threats(self,
                                   divergences: List[DivergencePoint],
                                   anomalies: List[Dict[str, Any]],
                                   relationship: ModelRelationship) -> List[str]:
        """Identify potential security threats."""
        threats = []
        
        # Check for backdoor indicators
        prompt_specific_count = sum(1 for d in divergences if d.statistical_significance > 0.9)
        if prompt_specific_count > 3:
            threats.append("Potential backdoor: Multiple prompt-specific divergences with high significance")
        
        # Check for model poisoning
        quality_anomalies = [a for a in anomalies if a["type"] == "quality_drop"]
        if quality_anomalies and relationship == ModelRelationship.ADVERSARIAL:
            threats.append("Potential model poisoning: Quality drops with adversarial relationship")
        
        # Check for extraction attack
        if relationship == ModelRelationship.FINE_TUNED and len(anomalies) > 5:
            threats.append("Potential extraction attack: Fine-tuned model with multiple anomalies")
        
        # Check for tampering
        binding_anomalies = [a for a in anomalies if a["type"] == "inconsistent_binding"]
        if binding_anomalies:
            threats.append("Potential tampering: Inconsistent binding patterns detected")
        
        return threats
    
    def generate_report(self, analysis: ComprehensiveAnalysis) -> str:
        """Generate a human-readable report from the analysis."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Models Compared: {analysis.model_a_id} vs {analysis.model_b_id}")
        report.append(f"Overall Similarity: {analysis.overall_similarity:.2%}")
        report.append(f"Inferred Relationship: {analysis.inferred_relationship.value}")
        report.append(f"Confidence: {analysis.relationship_confidence:.2%}")
        report.append("")
        
        # Relationship Evidence
        report.append("RELATIONSHIP EVIDENCE")
        report.append("-" * 40)
        for key, value in analysis.relationship_evidence.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.3f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
        
        # Behavioral Analysis
        report.append("BEHAVIORAL ANALYSIS")
        report.append("-" * 40)
        report.append(f"Phase Alignment: {analysis.phase_alignment:.2%}")
        report.append(f"Transition Correlation: {analysis.transition_correlation:.3f}")
        report.append(f"Model A Phases: {len(analysis.behavioral_phases_a)}")
        for phase in analysis.behavioral_phases_a[:3]:  # Show first 3
            report.append(f"  - {phase.phase_type}: layers {phase.start_layer}-{phase.end_layer} (stability: {phase.stability:.2f})")
        report.append(f"Model B Phases: {len(analysis.behavioral_phases_b)}")
        for phase in analysis.behavioral_phases_b[:3]:  # Show first 3
            report.append(f"  - {phase.phase_type}: layers {phase.start_layer}-{phase.end_layer} (stability: {phase.stability:.2f})")
        report.append("")
        
        # Divergence Analysis
        if analysis.divergence_points:
            report.append("DIVERGENCE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Divergence Pattern: {analysis.divergence_pattern}")
            report.append(f"Max Divergence: {analysis.max_divergence:.3f}")
            report.append(f"Total Divergence Points: {len(analysis.divergence_points)}")
            
            # Show top divergences
            top_divs = sorted(analysis.divergence_points, 
                            key=lambda x: x.divergence_magnitude, 
                            reverse=True)[:3]
            report.append("Top Divergences:")
            for div in top_divs:
                report.append(f"  - Prompt {div.prompt_idx}, Layer {div.layer_idx}: {div.divergence_magnitude:.3f} ({div.divergence_type})")
            report.append("")
        
        # Scaling Analysis
        report.append("SCALING ANALYSIS")
        report.append("-" * 40)
        report.append(f"Layer Ratio: {analysis.layer_ratio:.2f}")
        report.append(f"Magnitude Ratio: {analysis.magnitude_ratio:.2f}")
        report.append(f"Complexity Ratio: {analysis.complexity_ratio:.2f}")
        report.append(f"Scaling Consistency: {analysis.scaling_consistency:.2%}")
        report.append("")
        
        # Architecture Analysis
        report.append("ARCHITECTURE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Attention Pattern Similarity: {analysis.attention_pattern_similarity:.2%}")
        report.append(f"Feedforward Pattern Similarity: {analysis.feedforward_pattern_similarity:.2%}")
        report.append(f"Residual Pattern Similarity: {analysis.residual_pattern_similarity:.2%}")
        report.append(f"Architecture Signature Distance: {analysis.architectural_signature_distance:.3f}")
        report.append("")
        
        # Security Analysis
        if analysis.potential_threats or analysis.anomalous_behaviors:
            report.append("SECURITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"Anomaly Score: {analysis.anomaly_score:.2f}")
            
            if analysis.potential_threats:
                report.append("Potential Threats:")
                for threat in analysis.potential_threats:
                    report.append(f"  ⚠️  {threat}")
            
            if analysis.anomalous_behaviors:
                report.append("Anomalous Behaviors:")
                for anomaly in analysis.anomalous_behaviors[:3]:  # Show first 3
                    report.append(f"  - {anomaly['type']}: {anomaly['description']} (severity: {anomaly['severity']:.2f})")
            report.append("")
        
        # Interpretation
        report.append("INTERPRETATION")
        report.append("-" * 40)
        
        if analysis.inferred_relationship == ModelRelationship.IDENTICAL:
            report.append("✅ Models appear to be identical or near-identical versions.")
        elif analysis.inferred_relationship == ModelRelationship.ADVERSARIAL:
            report.append("⚠️  WARNING: Potential adversarial modification detected!")
            report.append("   One model may have been tampered with or backdoored.")
        elif analysis.inferred_relationship == ModelRelationship.SCALED_VERSION:
            report.append("📏 Models appear to be scaled versions of the same architecture.")
            report.append(f"   Scaling factor approximately {analysis.layer_ratio:.1f}x")
        elif analysis.inferred_relationship == ModelRelationship.FINE_TUNED:
            report.append("🎯 Models appear to share a base but one is fine-tuned.")
        elif analysis.inferred_relationship == ModelRelationship.QUANTIZED:
            report.append("🗜️  One model appears to be a quantized version of the other.")
        elif analysis.inferred_relationship == ModelRelationship.SAME_FAMILY:
            report.append("👨‍👩‍👧‍👦 Models belong to the same architecture family.")
        elif analysis.inferred_relationship == ModelRelationship.DIFFERENT_FAMILY:
            report.append("🔄 Models are from different architecture families but show some similarity.")
        else:
            report.append("❓ Models appear to be unrelated.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)