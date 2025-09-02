#!/usr/bin/env python3
"""
Adversarial Detection System using Unified Fingerprints

This module uses behavioral fingerprints to detect adversarially modified,
backdoored, or tampered models by comparing against known-good reference
fingerprints.
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import warnings

# REV components
from ..hdc.unified_fingerprint import UnifiedFingerprint, UnifiedFingerprintGenerator
from ..hypervector.similarity import AdvancedSimilarity
from ..crypto.merkle import build_merkle_tree, verify_merkle_proof

logger = logging.getLogger(__name__)


@dataclass
class AdversarialIndicator:
    """Indicator of potential adversarial modification"""
    indicator_type: str  # "pathway_deviation", "response_anomaly", "binding_weakness", etc.
    severity: float  # 0-1 scale
    description: str
    affected_layers: List[int] = field(default_factory=list)
    statistical_evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class IntegrityVerificationResult:
    """Result of model integrity verification"""
    model_id: str
    reference_id: str
    is_authentic: bool
    confidence: float
    
    # Detailed scores
    pathway_similarity: float
    response_similarity: float
    binding_integrity: float
    statistical_similarity: float
    
    # Adversarial indicators
    adversarial_indicators: List[AdversarialIndicator] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    
    # Specific threats detected
    detected_threats: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    verification_time: datetime = field(default_factory=datetime.now)
    verification_duration: float = 0.0


@dataclass
class ReferenceFingerprint:
    """Reference fingerprint for a known-good model"""
    fingerprint: UnifiedFingerprint
    model_id: str
    version: str
    creation_time: datetime
    merkle_root: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical baselines
    layer_statistics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    response_patterns: Dict[str, Any] = field(default_factory=dict)
    behavioral_boundaries: List[int] = field(default_factory=list)


class AdversarialDetector:
    """
    Detects adversarially modified models by comparing behavioral fingerprints
    against known-good reference models.
    """
    
    def __init__(self, 
                 reference_dir: str = "./reference_fingerprints",
                 sensitivity_level: str = "medium"):
        """
        Initialize adversarial detection system.
        
        Args:
            reference_dir: Directory containing reference fingerprints
            sensitivity_level: Detection sensitivity ("low", "medium", "high", "paranoid")
        """
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
        self.sensitivity_level = sensitivity_level
        self.sensitivity_thresholds = self._get_sensitivity_thresholds(sensitivity_level)
        
        # Load reference fingerprints
        self.reference_fingerprints = {}
        self._load_reference_fingerprints()
        
        # Initialize similarity calculator
        self.similarity = AdvancedSimilarity()
        
        # Detection statistics
        self.detection_stats = {
            "models_verified": 0,
            "threats_detected": 0,
            "false_positives": 0  # Estimated
        }
        
        logger.info(f"Initialized AdversarialDetector with {len(self.reference_fingerprints)} references")
    
    def _get_sensitivity_thresholds(self, level: str) -> Dict[str, float]:
        """Get detection thresholds based on sensitivity level."""
        thresholds = {
            "low": {
                "pathway_threshold": 0.6,
                "response_threshold": 0.5,
                "binding_threshold": 0.5,
                "statistical_threshold": 0.4,
                "anomaly_threshold": 3.0,  # Standard deviations
                "min_confidence": 0.7
            },
            "medium": {
                "pathway_threshold": 0.75,
                "response_threshold": 0.65,
                "binding_threshold": 0.6,
                "statistical_threshold": 0.5,
                "anomaly_threshold": 2.5,
                "min_confidence": 0.8
            },
            "high": {
                "pathway_threshold": 0.85,
                "response_threshold": 0.75,
                "binding_threshold": 0.7,
                "statistical_threshold": 0.6,
                "anomaly_threshold": 2.0,
                "min_confidence": 0.85
            },
            "paranoid": {
                "pathway_threshold": 0.95,
                "response_threshold": 0.85,
                "binding_threshold": 0.8,
                "statistical_threshold": 0.7,
                "anomaly_threshold": 1.5,
                "min_confidence": 0.9
            }
        }
        
        return thresholds.get(level, thresholds["medium"])
    
    def verify_model_integrity(self,
                              model_fingerprint: UnifiedFingerprint,
                              reference_model_id: Optional[str] = None) -> IntegrityVerificationResult:
        """
        Verify model integrity by comparing against reference fingerprint.
        
        Args:
            model_fingerprint: Fingerprint of model to verify
            reference_model_id: ID of reference model (auto-detect if None)
            
        Returns:
            IntegrityVerificationResult with detailed analysis
        """
        start_time = time.time()
        
        # Find appropriate reference fingerprint
        if reference_model_id:
            if reference_model_id not in self.reference_fingerprints:
                raise ValueError(f"Reference model {reference_model_id} not found")
            reference = self.reference_fingerprints[reference_model_id]
        else:
            reference = self._find_best_reference(model_fingerprint)
            if not reference:
                raise ValueError("No suitable reference model found")
        
        logger.info(f"Verifying model {model_fingerprint.model_id} against reference {reference.model_id}")
        
        # 1. Pathway Analysis - Most sensitive to adversarial modifications
        pathway_similarity = self._analyze_pathway_integrity(
            model_fingerprint.pathway_hypervector,
            reference.fingerprint.pathway_hypervector,
            model_fingerprint.layers_sampled,
            reference.fingerprint.layers_sampled
        )
        
        # 2. Response Pattern Analysis
        response_similarity = self._analyze_response_patterns(
            model_fingerprint.response_hypervector,
            reference.fingerprint.response_hypervector,
            model_fingerprint.response_text,
            reference.fingerprint.response_text
        )
        
        # 3. Binding Integrity Check
        binding_integrity = self._check_binding_integrity(
            model_fingerprint,
            reference.fingerprint
        )
        
        # 4. Statistical Similarity Analysis
        statistical_similarity = self._statistical_analysis(
            model_fingerprint,
            reference.fingerprint
        )
        
        # 5. Detect specific adversarial indicators
        adversarial_indicators = self._detect_adversarial_indicators(
            model_fingerprint,
            reference.fingerprint,
            pathway_similarity,
            response_similarity,
            binding_integrity
        )
        
        # 6. Identify specific threats
        detected_threats = self._identify_threats(
            adversarial_indicators,
            pathway_similarity,
            response_similarity,
            statistical_similarity
        )
        
        # Calculate overall authenticity
        weighted_score = (
            pathway_similarity * 0.4 +
            response_similarity * 0.2 +
            binding_integrity * 0.2 +
            statistical_similarity * 0.2
        )
        
        thresholds = self.sensitivity_thresholds
        is_authentic = (
            pathway_similarity >= thresholds["pathway_threshold"] and
            response_similarity >= thresholds["response_threshold"] and
            binding_integrity >= thresholds["binding_threshold"] and
            weighted_score >= thresholds["min_confidence"]
        )
        
        # Determine risk level
        risk_level = self._calculate_risk_level(
            adversarial_indicators,
            detected_threats,
            weighted_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_authentic,
            risk_level,
            adversarial_indicators,
            detected_threats
        )
        
        verification_duration = time.time() - start_time
        
        result = IntegrityVerificationResult(
            model_id=model_fingerprint.model_id,
            reference_id=reference.model_id,
            is_authentic=is_authentic,
            confidence=weighted_score,
            pathway_similarity=pathway_similarity,
            response_similarity=response_similarity,
            binding_integrity=binding_integrity,
            statistical_similarity=statistical_similarity,
            adversarial_indicators=adversarial_indicators,
            risk_level=risk_level,
            detected_threats=detected_threats,
            recommendations=recommendations,
            verification_duration=verification_duration
        )
        
        # Update statistics
        self.detection_stats["models_verified"] += 1
        if detected_threats:
            self.detection_stats["threats_detected"] += 1
        
        logger.info(f"Verification complete: {'AUTHENTIC' if is_authentic else 'SUSPICIOUS'} "
                   f"(confidence: {weighted_score:.3f}, risk: {risk_level})")
        
        return result
    
    def _analyze_pathway_integrity(self,
                                  test_pathway: np.ndarray,
                                  ref_pathway: np.ndarray,
                                  test_layers: List[int],
                                  ref_layers: List[int]) -> float:
        """
        Analyze processing pathway integrity - most sensitive to adversarial changes.
        
        Adversarial modifications often create subtle but detectable changes in
        how the model processes information through its layers.
        """
        # Direct cosine similarity
        pathway_sim = 1 - cosine(test_pathway, ref_pathway)
        
        # Layer coverage similarity
        layer_overlap = len(set(test_layers) & set(ref_layers)) / max(len(test_layers), len(ref_layers))
        
        # Activation pattern analysis
        test_pattern = self._extract_activation_pattern(test_pathway)
        ref_pattern = self._extract_activation_pattern(ref_pathway)
        pattern_sim = 1 - cosine(test_pattern, ref_pattern)
        
        # Detect unusual spikes or drops (potential backdoor triggers)
        spike_anomaly = self._detect_pathway_anomalies(test_pathway, ref_pathway)
        
        # Combined pathway integrity score
        integrity = (
            pathway_sim * 0.5 +
            layer_overlap * 0.2 +
            pattern_sim * 0.2 +
            (1 - spike_anomaly) * 0.1
        )
        
        return integrity
    
    def _analyze_response_patterns(self,
                                  test_response: np.ndarray,
                                  ref_response: np.ndarray,
                                  test_text: str,
                                  ref_text: str) -> float:
        """Analyze response generation patterns for anomalies."""
        # Vector similarity
        response_sim = 1 - cosine(test_response, ref_response)
        
        # Statistical distribution comparison
        test_dist = np.abs(test_response)
        ref_dist = np.abs(ref_response)
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pval = ks_2samp(test_dist, ref_dist)
        dist_similarity = ks_pval  # Higher p-value means more similar
        
        # Entropy comparison (adversarial outputs often have different entropy)
        test_entropy = -np.sum(test_dist * np.log(test_dist + 1e-10))
        ref_entropy = -np.sum(ref_dist * np.log(ref_dist + 1e-10))
        entropy_ratio = min(test_entropy, ref_entropy) / max(test_entropy, ref_entropy)
        
        # Combined score
        return response_sim * 0.6 + dist_similarity * 0.2 + entropy_ratio * 0.2
    
    def _check_binding_integrity(self,
                                test_fp: UnifiedFingerprint,
                                ref_fp: UnifiedFingerprint) -> float:
        """
        Check binding integrity between components.
        
        Adversarial modifications often weaken the binding between components
        as the model's internal consistency is disrupted.
        """
        # Check if unified vector properly represents components
        test_binding = self._measure_binding_strength(test_fp)
        ref_binding = self._measure_binding_strength(ref_fp)
        
        # Binding consistency
        binding_ratio = min(test_binding, ref_binding) / max(test_binding, ref_binding)
        
        # Component coherence
        test_coherence = self._measure_component_coherence(test_fp)
        ref_coherence = self._measure_component_coherence(ref_fp)
        coherence_ratio = min(test_coherence, ref_coherence) / max(test_coherence, ref_coherence)
        
        return binding_ratio * 0.6 + coherence_ratio * 0.4
    
    def _measure_binding_strength(self, fingerprint: UnifiedFingerprint) -> float:
        """Measure how well unified vector represents individual components."""
        unified = fingerprint.unified_hypervector
        
        # Check correlation with each component
        prompt_corr = np.corrcoef(unified, fingerprint.prompt_hypervector)[0, 1]
        pathway_corr = np.corrcoef(unified, fingerprint.pathway_hypervector)[0, 1]
        response_corr = np.corrcoef(unified, fingerprint.response_hypervector)[0, 1]
        
        # Average correlation (should be high for good binding)
        return np.mean([abs(prompt_corr), abs(pathway_corr), abs(response_corr)])
    
    def _measure_component_coherence(self, fingerprint: UnifiedFingerprint) -> float:
        """Measure coherence between fingerprint components."""
        # Prompt-pathway coherence
        pp_coherence = 1 - cosine(
            fingerprint.prompt_hypervector,
            fingerprint.pathway_hypervector
        )
        
        # Pathway-response coherence
        pr_coherence = 1 - cosine(
            fingerprint.pathway_hypervector,
            fingerprint.response_hypervector
        )
        
        return (pp_coherence + pr_coherence) / 2
    
    def _statistical_analysis(self,
                            test_fp: UnifiedFingerprint,
                            ref_fp: UnifiedFingerprint) -> float:
        """Statistical analysis of fingerprint distributions."""
        # Compare divergence statistics
        if test_fp.divergence_stats and ref_fp.divergence_stats:
            test_divs = list(test_fp.divergence_stats.values())
            ref_divs = list(ref_fp.divergence_stats.values())
            
            # Normalize to same length
            min_len = min(len(test_divs), len(ref_divs))
            test_divs = test_divs[:min_len]
            ref_divs = ref_divs[:min_len]
            
            # Wasserstein distance (Earth Mover's Distance)
            if len(test_divs) > 0:
                emd = wasserstein_distance(test_divs, ref_divs)
                emd_similarity = 1 / (1 + emd)  # Convert to similarity
            else:
                emd_similarity = 0.5
        else:
            emd_similarity = 0.5
        
        # Compare quality metrics
        quality_ratio = min(test_fp.fingerprint_quality, ref_fp.fingerprint_quality) / \
                       max(test_fp.fingerprint_quality, ref_fp.fingerprint_quality)
        
        return emd_similarity * 0.6 + quality_ratio * 0.4
    
    def _detect_adversarial_indicators(self,
                                      test_fp: UnifiedFingerprint,
                                      ref_fp: UnifiedFingerprint,
                                      pathway_sim: float,
                                      response_sim: float,
                                      binding_integrity: float) -> List[AdversarialIndicator]:
        """Detect specific indicators of adversarial modification."""
        indicators = []
        thresholds = self.sensitivity_thresholds
        
        # 1. Pathway Deviation Indicator
        if pathway_sim < thresholds["pathway_threshold"]:
            severity = 1 - pathway_sim
            indicators.append(AdversarialIndicator(
                indicator_type="pathway_deviation",
                severity=severity,
                description=f"Processing pathway deviates significantly from reference (similarity: {pathway_sim:.3f})",
                affected_layers=self._identify_affected_layers(test_fp, ref_fp),
                statistical_evidence={"pathway_similarity": pathway_sim}
            ))
        
        # 2. Response Anomaly Indicator
        if response_sim < thresholds["response_threshold"]:
            severity = 1 - response_sim
            indicators.append(AdversarialIndicator(
                indicator_type="response_anomaly",
                severity=severity,
                description=f"Response patterns show anomalies (similarity: {response_sim:.3f})",
                statistical_evidence={"response_similarity": response_sim}
            ))
        
        # 3. Binding Weakness Indicator
        if binding_integrity < thresholds["binding_threshold"]:
            severity = 1 - binding_integrity
            indicators.append(AdversarialIndicator(
                indicator_type="binding_weakness",
                severity=severity,
                description=f"Component binding is weakened (integrity: {binding_integrity:.3f})",
                statistical_evidence={"binding_integrity": binding_integrity}
            ))
        
        # 4. Activation Spike Indicator
        spike_locations = self._detect_activation_spikes(test_fp, ref_fp)
        if spike_locations:
            indicators.append(AdversarialIndicator(
                indicator_type="activation_spike",
                severity=0.7,
                description=f"Unusual activation spikes detected at layers {spike_locations}",
                affected_layers=spike_locations,
                statistical_evidence={"spike_count": len(spike_locations)}
            ))
        
        # 5. Entropy Anomaly Indicator
        entropy_anomaly = self._detect_entropy_anomaly(test_fp, ref_fp)
        if entropy_anomaly > thresholds["anomaly_threshold"]:
            indicators.append(AdversarialIndicator(
                indicator_type="entropy_anomaly",
                severity=min(1.0, entropy_anomaly / 5),
                description=f"Information entropy deviates by {entropy_anomaly:.1f} standard deviations",
                statistical_evidence={"entropy_deviation": entropy_anomaly}
            ))
        
        return indicators
    
    def _identify_threats(self,
                        indicators: List[AdversarialIndicator],
                        pathway_sim: float,
                        response_sim: float,
                        statistical_sim: float) -> List[str]:
        """Identify specific threat types based on indicators."""
        threats = []
        
        # Check indicator patterns for known threat signatures
        indicator_types = {ind.indicator_type for ind in indicators}
        high_severity_count = sum(1 for ind in indicators if ind.severity > 0.7)
        
        # Backdoor Detection
        if ("activation_spike" in indicator_types and 
            "pathway_deviation" in indicator_types):
            threats.append("BACKDOOR: Potential backdoor trigger detected")
        
        # Model Poisoning
        if (pathway_sim < 0.6 and response_sim < 0.5):
            threats.append("POISONING: Model shows signs of data poisoning")
        
        # Fine-tuning Attack
        if ("binding_weakness" in indicator_types and
            statistical_sim < 0.5):
            threats.append("FINE-TUNING: Unauthorized fine-tuning detected")
        
        # Adversarial Patching
        if high_severity_count >= 3:
            threats.append("ADVERSARIAL_PATCH: Multiple adversarial modifications detected")
        
        # Trojan Detection
        if ("activation_spike" in indicator_types and
            "response_anomaly" in indicator_types):
            threats.append("TROJAN: Potential trojan behavior detected")
        
        # Model Replacement
        if (pathway_sim < 0.4 and response_sim < 0.4 and statistical_sim < 0.4):
            threats.append("REPLACEMENT: Model may have been replaced entirely")
        
        return threats
    
    def _calculate_risk_level(self,
                            indicators: List[AdversarialIndicator],
                            threats: List[str],
                            confidence: float) -> str:
        """Calculate overall risk level."""
        if not indicators and not threats:
            return "low"
        
        max_severity = max([ind.severity for ind in indicators], default=0)
        threat_count = len(threats)
        
        if threat_count >= 3 or max_severity > 0.8:
            return "critical"
        elif threat_count >= 2 or max_severity > 0.6:
            return "high"
        elif threat_count >= 1 or max_severity > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self,
                                is_authentic: bool,
                                risk_level: str,
                                indicators: List[AdversarialIndicator],
                                threats: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not is_authentic:
            if risk_level == "critical":
                recommendations.append("IMMEDIATE ACTION: Quarantine model and conduct forensic analysis")
                recommendations.append("DO NOT DEPLOY: Model shows critical security violations")
                recommendations.append("ALERT: Notify security team immediately")
            elif risk_level == "high":
                recommendations.append("CAUTION: Do not use in production without further verification")
                recommendations.append("REVIEW: Conduct detailed security audit")
                recommendations.append("TEST: Run comprehensive adversarial testing suite")
            elif risk_level == "medium":
                recommendations.append("VERIFY: Confirm model provenance and training data")
                recommendations.append("MONITOR: Deploy with enhanced monitoring if necessary")
                recommendations.append("COMPARE: Test against multiple reference models")
            
            # Specific threat mitigations
            if "BACKDOOR" in str(threats):
                recommendations.append("SCAN: Run backdoor detection tools")
                recommendations.append("TEST: Check for trigger-based behaviors")
            
            if "POISONING" in str(threats):
                recommendations.append("RETRAIN: Consider retraining from clean data")
                recommendations.append("FILTER: Apply output filtering and sanitization")
            
            if "FINE-TUNING" in str(threats):
                recommendations.append("REVERT: Load original pre-trained weights")
                recommendations.append("AUDIT: Review fine-tuning procedures")
        else:
            if risk_level != "low":
                recommendations.append("MONITOR: Model appears authentic but shows minor anomalies")
                recommendations.append("DOCUMENT: Record verification results for audit trail")
        
        return recommendations
    
    def _extract_activation_pattern(self, pathway: np.ndarray) -> np.ndarray:
        """Extract activation pattern features from pathway."""
        # Compute statistical features
        features = []
        
        # Basic statistics
        features.append(np.mean(pathway))
        features.append(np.std(pathway))
        features.append(np.max(np.abs(pathway)))
        features.append(np.min(np.abs(pathway)))
        
        # Percentiles
        for p in [25, 50, 75, 90, 95]:
            features.append(np.percentile(np.abs(pathway), p))
        
        # Activation characteristics
        features.append(np.sum(pathway > 0) / len(pathway))  # Positive ratio
        features.append(np.sum(np.abs(pathway) < 0.01) / len(pathway))  # Near-zero ratio
        
        return np.array(features)
    
    def _detect_pathway_anomalies(self, test: np.ndarray, ref: np.ndarray) -> float:
        """Detect anomalous spikes or patterns in pathway."""
        # Compute difference
        diff = np.abs(test - ref)
        
        # Find anomalous regions (> 3 std dev from mean difference)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        anomalies = diff > (mean_diff + 3 * std_diff)
        
        return np.sum(anomalies) / len(anomalies)
    
    def _identify_affected_layers(self,
                                test_fp: UnifiedFingerprint,
                                ref_fp: UnifiedFingerprint) -> List[int]:
        """Identify which layers show the most deviation."""
        affected = []
        
        # Compare sampled layers
        test_layers = set(test_fp.layers_sampled)
        ref_layers = set(ref_fp.layers_sampled)
        
        # Layers unique to test model (potential modification points)
        unique_to_test = test_layers - ref_layers
        affected.extend(list(unique_to_test))
        
        return sorted(affected)
    
    def _detect_activation_spikes(self,
                                test_fp: UnifiedFingerprint,
                                ref_fp: UnifiedFingerprint) -> List[int]:
        """Detect unusual activation spikes that might indicate triggers."""
        spikes = []
        
        # Analyze pathway vector for sudden changes
        test_pathway = test_fp.pathway_hypervector
        
        # Compute local variance
        window_size = len(test_pathway) // 20  # 5% window
        for i in range(0, len(test_pathway) - window_size, window_size):
            window = test_pathway[i:i+window_size]
            local_var = np.var(window)
            
            # Check if variance is unusually high
            if local_var > np.mean(test_pathway) + 3 * np.std(test_pathway):
                # Map back to layer index (approximate)
                layer_idx = int(i / len(test_pathway) * max(test_fp.layers_sampled))
                if layer_idx not in spikes:
                    spikes.append(layer_idx)
        
        return spikes
    
    def _detect_entropy_anomaly(self,
                              test_fp: UnifiedFingerprint,
                              ref_fp: UnifiedFingerprint) -> float:
        """Detect entropy anomalies in fingerprint distributions."""
        # Calculate entropy for each fingerprint
        test_entropy = -np.sum(np.abs(test_fp.unified_hypervector) * 
                               np.log(np.abs(test_fp.unified_hypervector) + 1e-10))
        ref_entropy = -np.sum(np.abs(ref_fp.unified_hypervector) * 
                             np.log(np.abs(ref_fp.unified_hypervector) + 1e-10))
        
        # Compute deviation in standard deviations
        entropy_diff = abs(test_entropy - ref_entropy)
        
        # Estimate standard deviation (simplified)
        estimated_std = ref_entropy * 0.1  # Assume 10% variation is normal
        
        return entropy_diff / estimated_std if estimated_std > 0 else 0
    
    def _find_best_reference(self, test_fp: UnifiedFingerprint) -> Optional[ReferenceFingerprint]:
        """Find the best matching reference fingerprint."""
        if not self.reference_fingerprints:
            return None
        
        best_match = None
        best_similarity = 0
        
        for ref_id, ref in self.reference_fingerprints.items():
            # Quick similarity check
            similarity = 1 - cosine(
                test_fp.unified_hypervector,
                ref.fingerprint.unified_hypervector
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = ref
        
        # Require minimum similarity to use as reference
        if best_similarity < 0.3:  # Too different to be same model family
            return None
        
        return best_match
    
    def create_reference_fingerprint(self,
                                   fingerprint: UnifiedFingerprint,
                                   model_id: str,
                                   version: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> ReferenceFingerprint:
        """Create a reference fingerprint from a known-good model."""
        
        # Build Merkle tree for integrity
        merkle_data = [
            fingerprint.unified_hypervector.tobytes(),
            fingerprint.prompt_hypervector.tobytes(),
            fingerprint.pathway_hypervector.tobytes(),
            fingerprint.response_hypervector.tobytes()
        ]
        merkle_root = build_merkle_tree(merkle_data).hex()
        
        # Extract layer statistics
        layer_statistics = {}
        for layer in fingerprint.layers_sampled:
            layer_statistics[layer] = {
                "index": layer,
                "quality": fingerprint.fingerprint_quality,
                "binding": fingerprint.binding_strength
            }
        
        # Create reference
        reference = ReferenceFingerprint(
            fingerprint=fingerprint,
            model_id=model_id,
            version=version,
            creation_time=datetime.now(),
            merkle_root=merkle_root,
            metadata=metadata or {},
            layer_statistics=layer_statistics
        )
        
        # Save to reference directory
        self.save_reference_fingerprint(reference)
        
        # Add to loaded references
        self.reference_fingerprints[model_id] = reference
        
        logger.info(f"Created reference fingerprint for {model_id} v{version}")
        
        return reference
    
    def save_reference_fingerprint(self, reference: ReferenceFingerprint):
        """Save reference fingerprint to disk."""
        filename = f"{reference.model_id}_{reference.version}.json"
        filepath = self.reference_dir / filename
        
        # Convert to serializable format
        data = {
            "model_id": reference.model_id,
            "version": reference.version,
            "creation_time": reference.creation_time.isoformat(),
            "merkle_root": reference.merkle_root,
            "metadata": reference.metadata,
            "layer_statistics": reference.layer_statistics,
            "fingerprint": {
                "unified": reference.fingerprint.unified_hypervector.tolist(),
                "prompt": reference.fingerprint.prompt_hypervector.tolist(),
                "pathway": reference.fingerprint.pathway_hypervector.tolist(),
                "response": reference.fingerprint.response_hypervector.tolist(),
                "model_id": reference.fingerprint.model_id,
                "layer_count": reference.fingerprint.layer_count,
                "layers_sampled": reference.fingerprint.layers_sampled,
                "quality": reference.fingerprint.fingerprint_quality,
                "binding": reference.fingerprint.binding_strength
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved reference fingerprint to {filepath}")
    
    def _load_reference_fingerprints(self):
        """Load all reference fingerprints from disk."""
        if not self.reference_dir.exists():
            return
        
        for filepath in self.reference_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct fingerprint
                fp_data = data["fingerprint"]
                fingerprint = UnifiedFingerprint(
                    unified_hypervector=np.array(fp_data["unified"]),
                    prompt_hypervector=np.array(fp_data["prompt"]),
                    pathway_hypervector=np.array(fp_data["pathway"]),
                    response_hypervector=np.array(fp_data["response"]),
                    model_id=fp_data["model_id"],
                    prompt_text="[REFERENCE]",
                    response_text="[REFERENCE]",
                    layer_count=fp_data["layer_count"],
                    layers_sampled=fp_data["layers_sampled"],
                    fingerprint_quality=fp_data["quality"],
                    divergence_stats={},
                    binding_strength=fp_data["binding"]
                )
                
                reference = ReferenceFingerprint(
                    fingerprint=fingerprint,
                    model_id=data["model_id"],
                    version=data["version"],
                    creation_time=datetime.fromisoformat(data["creation_time"]),
                    merkle_root=data["merkle_root"],
                    metadata=data.get("metadata", {}),
                    layer_statistics=data.get("layer_statistics", {})
                )
                
                self.reference_fingerprints[data["model_id"]] = reference
                
            except Exception as e:
                logger.warning(f"Failed to load reference from {filepath}: {e}")
    
    def generate_security_report(self, result: IntegrityVerificationResult) -> str:
        """Generate a comprehensive security report."""
        
        report = f"""
# Model Security Verification Report
**Generated:** {datetime.now().isoformat()}
**Model:** {result.model_id}
**Reference:** {result.reference_id}

## Verification Summary
- **Status:** {'✅ AUTHENTIC' if result.is_authentic else '❌ SUSPICIOUS'}
- **Confidence:** {result.confidence:.1%}
- **Risk Level:** {result.risk_level.upper()}

## Detailed Analysis

### Component Similarities
- **Processing Pathway:** {result.pathway_similarity:.3f}
- **Response Patterns:** {result.response_similarity:.3f}
- **Binding Integrity:** {result.binding_integrity:.3f}
- **Statistical Match:** {result.statistical_similarity:.3f}

### Detected Threats
"""
        
        if result.detected_threats:
            for threat in result.detected_threats:
                report += f"- ⚠️  {threat}\n"
        else:
            report += "- ✅ No specific threats detected\n"
        
        report += "\n### Adversarial Indicators\n"
        
        if result.adversarial_indicators:
            for indicator in sorted(result.adversarial_indicators, key=lambda x: x.severity, reverse=True):
                severity_icon = "🔴" if indicator.severity > 0.7 else "🟡" if indicator.severity > 0.4 else "🟢"
                report += f"\n{severity_icon} **{indicator.indicator_type.replace('_', ' ').title()}**\n"
                report += f"   - Severity: {indicator.severity:.2f}\n"
                report += f"   - Description: {indicator.description}\n"
                if indicator.affected_layers:
                    report += f"   - Affected Layers: {indicator.affected_layers}\n"
        else:
            report += "- ✅ No adversarial indicators detected\n"
        
        report += "\n## Recommendations\n"
        
        if result.recommendations:
            for rec in result.recommendations:
                report += f"- {rec}\n"
        else:
            report += "- ✅ Model appears safe for deployment\n"
        
        report += f"\n---\n*Verification completed in {result.verification_duration:.2f} seconds*\n"
        
        return report


# Helper functions for integration
def verify_model_integrity(model_fingerprint: UnifiedFingerprint,
                          reference_model_id: Optional[str] = None,
                          sensitivity: str = "medium") -> IntegrityVerificationResult:
    """
    Quick function to verify model integrity.
    
    Args:
        model_fingerprint: Fingerprint of model to verify
        reference_model_id: ID of reference model (auto-detect if None)
        sensitivity: Detection sensitivity level
        
    Returns:
        IntegrityVerificationResult
    """
    detector = AdversarialDetector(sensitivity_level=sensitivity)
    return detector.verify_model_integrity(model_fingerprint, reference_model_id)


def create_model_reference(fingerprint: UnifiedFingerprint,
                         model_id: str,
                         version: str = "1.0") -> ReferenceFingerprint:
    """Create a reference fingerprint for a trusted model."""
    detector = AdversarialDetector()
    return detector.create_reference_fingerprint(fingerprint, model_id, version)


if __name__ == "__main__":
    # Example usage
    print("Adversarial Detection System initialized")
    print("Use verify_model_integrity() to check models against references")
    print("Use create_model_reference() to create trusted references")