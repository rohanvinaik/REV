#!/usr/bin/env python3
"""
Model Fingerprint Library

This module maintains a registry of base model fingerprints for different
architectures, enabling intelligent model identification and adaptive testing.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import hashlib
import logging

from ..hdc.unified_fingerprint import UnifiedFingerprint
from ..analysis.unified_model_analysis import ModelRelationship

logger = logging.getLogger(__name__)


@dataclass
class BaseModelFingerprint:
    """Fingerprint of a known base model architecture"""
    model_family: str  # "llama", "gpt", "mistral", "qwen", "yi", etc.
    model_size: str  # "7B", "13B", "70B", "175B", etc.
    architecture_version: str  # "llama-2", "llama-3", "gpt-3.5", etc.
    
    # Core fingerprint data
    unified_fingerprint: UnifiedFingerprint
    
    # Architectural characteristics
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    
    # Behavioral signatures
    layer_transitions: List[int]  # Layers where major transitions occur
    behavioral_phases: List[Tuple[int, int, str]]  # (start, end, phase_type)
    attention_patterns: np.ndarray  # FFT of attention patterns
    processing_signature: np.ndarray  # Overall processing pattern
    
    # Performance characteristics
    memory_footprint_gb: float
    optimal_segment_size: int
    optimal_batch_size: int
    
    # Testing recommendations
    recommended_cassettes: List[str]
    vulnerable_layers: List[int]  # Layers sensitive to adversarial attacks
    stable_layers: List[int]  # Layers good for baseline comparison
    
    # Metadata
    fingerprint_id: str = field(default_factory=lambda: hashlib.sha256(
        str(datetime.now()).encode()).hexdigest()[:16])
    creation_date: datetime = field(default_factory=datetime.now)
    validation_score: float = 1.0  # How well validated this fingerprint is
    source: str = "empirical"  # "empirical", "synthetic", "hybrid"


@dataclass
class ModelIdentificationResult:
    """Result of model architecture identification"""
    identified_family: Optional[str]
    confidence: float
    closest_matches: List[Tuple[str, float]]  # (model_id, similarity)
    is_novel: bool
    architectural_features: Dict[str, Any]
    recommended_strategy: str
    reasoning: List[str]


class ModelFingerprintLibrary:
    """
    Registry of base model fingerprints for architecture identification
    and adaptive testing strategy selection.
    """
    
    def __init__(self, library_path: str = "./fingerprint_library"):
        """
        Initialize the fingerprint library.
        
        Args:
            library_path: Directory to store/load fingerprint data
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        
        self.fingerprints: Dict[str, BaseModelFingerprint] = {}
        self.family_index: Dict[str, List[str]] = {}  # family -> [fingerprint_ids]
        
        # Similarity thresholds
        self.identification_threshold = 0.85  # Min similarity for family match
        self.novel_threshold = 0.60  # Below this, consider novel architecture
        
        # Load existing fingerprints
        self._load_library()
        
        # Initialize default fingerprints if empty
        if not self.fingerprints:
            self._initialize_default_fingerprints()
    
    def _initialize_default_fingerprints(self):
        """Initialize library with known base model fingerprints."""
        logger.info("Initializing default model fingerprints...")
        
        # These would be populated from actual model analysis
        # For now, creating templates
        
        # Llama family
        self._add_template_fingerprint(
            family="llama",
            size="70B",
            version="llama-3",
            num_layers=80,
            hidden_size=8192,
            num_heads=64,
            vocab_size=128256,
            transitions=[0, 20, 40, 60],
            phases=[(0, 20, "encoding"), (20, 60, "reasoning"), (60, 80, "decoding")],
            cassettes=["recursive", "theory_of_mind", "counterfactual"],
            vulnerable=[15, 35, 55],
            stable=[5, 25, 45, 65]
        )
        
        self._add_template_fingerprint(
            family="llama",
            size="7B",
            version="llama-3",
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            vocab_size=128256,
            transitions=[0, 8, 16, 24],
            phases=[(0, 8, "encoding"), (8, 24, "reasoning"), (24, 32, "decoding")],
            cassettes=["syntactic", "semantic", "transform"],
            vulnerable=[6, 14, 22],
            stable=[2, 10, 18, 26]
        )
        
        # GPT family
        self._add_template_fingerprint(
            family="gpt",
            size="175B",
            version="gpt-3",
            num_layers=96,
            hidden_size=12288,
            num_heads=96,
            vocab_size=50257,
            transitions=[0, 24, 48, 72],
            phases=[(0, 24, "encoding"), (24, 72, "reasoning"), (72, 96, "decoding")],
            cassettes=["recursive", "meta", "counterfactual"],
            vulnerable=[20, 44, 68],
            stable=[8, 32, 56, 80]
        )
        
        # Mistral family
        self._add_template_fingerprint(
            family="mistral",
            size="7B",
            version="mistral-v0.1",
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            vocab_size=32000,
            transitions=[0, 10, 20, 28],
            phases=[(0, 10, "encoding"), (10, 28, "reasoning"), (28, 32, "decoding")],
            cassettes=["semantic", "transform", "theory_of_mind"],
            vulnerable=[8, 16, 24],
            stable=[4, 12, 20, 28]
        )
        
        # Yi family
        self._add_template_fingerprint(
            family="yi",
            size="34B",
            version="yi-1.5",
            num_layers=60,
            hidden_size=7168,
            num_heads=56,
            vocab_size=64000,
            transitions=[0, 15, 30, 45],
            phases=[(0, 15, "encoding"), (15, 45, "reasoning"), (45, 60, "decoding")],
            cassettes=["recursive", "transform", "meta"],
            vulnerable=[12, 28, 44],
            stable=[5, 20, 35, 50]
        )
        
        # Qwen family
        self._add_template_fingerprint(
            family="qwen",
            size="72B",
            version="qwen-2",
            num_layers=80,
            hidden_size=8192,
            num_heads=64,
            vocab_size=152064,
            transitions=[0, 18, 40, 62],
            phases=[(0, 18, "encoding"), (18, 62, "reasoning"), (62, 80, "decoding")],
            cassettes=["syntactic", "recursive", "counterfactual"],
            vulnerable=[14, 36, 58],
            stable=[7, 28, 49, 70]
        )
        
        self._save_library()
    
    def _add_template_fingerprint(self, **kwargs):
        """Add a template fingerprint to the library."""
        # Create mock unified fingerprint
        mock_fp = UnifiedFingerprint(
            unified_hypervector=np.random.randn(10000),
            prompt_hypervector=np.random.randn(10000),
            pathway_hypervector=np.random.randn(10000),
            response_hypervector=np.random.randn(10000),
            model_id=f"{kwargs['family']}-{kwargs['size']}",
            prompt_text="Template prompt for fingerprint generation",
            response_text="Template response",
            layer_count=kwargs['num_layers'],
            layers_sampled=list(range(0, kwargs['num_layers'], max(1, kwargs['num_layers']//10))),
            fingerprint_quality=0.95,
            divergence_stats={'mean': 0.5, 'std': 0.1, 'max': 0.7, 'min': 0.3},
            binding_strength=0.9
        )
        
        base_fp = BaseModelFingerprint(
            model_family=kwargs['family'],
            model_size=kwargs['size'],
            architecture_version=kwargs['version'],
            unified_fingerprint=mock_fp,
            num_layers=kwargs['num_layers'],
            hidden_size=kwargs['hidden_size'],
            num_attention_heads=kwargs['num_heads'],
            vocab_size=kwargs['vocab_size'],
            layer_transitions=kwargs['transitions'],
            behavioral_phases=kwargs['phases'],
            attention_patterns=np.random.randn(100),  # Mock FFT
            processing_signature=np.random.randn(200),  # Mock signature
            memory_footprint_gb=kwargs['hidden_size'] * kwargs['num_layers'] / 500,
            optimal_segment_size=512,
            optimal_batch_size=4,
            recommended_cassettes=kwargs['cassettes'],
            vulnerable_layers=kwargs['vulnerable'],
            stable_layers=kwargs['stable']
        )
        
        self.add_fingerprint(base_fp)
    
    def add_fingerprint(self, fingerprint: BaseModelFingerprint):
        """Add a new base model fingerprint to the library."""
        fp_id = fingerprint.fingerprint_id
        self.fingerprints[fp_id] = fingerprint
        
        # Update family index
        if fingerprint.model_family not in self.family_index:
            self.family_index[fingerprint.model_family] = []
        self.family_index[fingerprint.model_family].append(fp_id)
        
        logger.info(f"Added fingerprint for {fingerprint.model_family}-{fingerprint.model_size}")
    
    def identify_model(self, 
                      test_fingerprint: UnifiedFingerprint,
                      layer_count: Optional[int] = None,
                      vocab_size: Optional[int] = None) -> ModelIdentificationResult:
        """
        Identify the architecture family of a model based on its fingerprint.
        
        Args:
            test_fingerprint: Fingerprint of the model to identify
            layer_count: Optional layer count hint
            vocab_size: Optional vocabulary size hint
            
        Returns:
            Identification result with family, confidence, and recommendations
        """
        logger.info("Identifying model architecture...")
        
        matches = []
        architectural_features = {}
        
        # Compare against all known fingerprints
        for fp_id, base_fp in self.fingerprints.items():
            similarity = self._compute_fingerprint_similarity(
                test_fingerprint, base_fp.unified_fingerprint
            )
            
            # Bonus for matching architectural hints
            if layer_count and abs(base_fp.num_layers - layer_count) < 5:
                similarity += 0.05
            if vocab_size and abs(base_fp.vocab_size - vocab_size) < 1000:
                similarity += 0.05
            
            similarity = min(similarity, 1.0)
            matches.append((fp_id, similarity, base_fp))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Determine if we have a family match
        identified_family = None
        confidence = 0.0
        is_novel = False
        recommended_strategy = "comprehensive"
        reasoning = []
        
        if matches and matches[0][1] >= self.identification_threshold:
            # Strong match found
            best_match = matches[0][2]
            identified_family = best_match.model_family
            confidence = matches[0][1]
            
            architectural_features = {
                "num_layers": best_match.num_layers,
                "hidden_size": best_match.hidden_size,
                "attention_heads": best_match.num_attention_heads,
                "layer_transitions": best_match.layer_transitions,
                "behavioral_phases": best_match.behavioral_phases
            }
            
            recommended_strategy = "targeted"
            reasoning.append(f"Strong match with {best_match.model_family}-{best_match.model_size} (confidence: {confidence:.2%})")
            reasoning.append(f"Recommended cassettes: {', '.join(best_match.recommended_cassettes)}")
            reasoning.append(f"Focus on layers: {best_match.vulnerable_layers}")
            
        elif matches and matches[0][1] >= self.novel_threshold:
            # Weak match - possibly variant or fine-tuned
            best_match = matches[0][2]
            identified_family = f"{best_match.model_family}-variant"
            confidence = matches[0][1]
            
            architectural_features = {
                "closest_family": best_match.model_family,
                "estimated_layers": layer_count or best_match.num_layers,
                "similarity_to_base": confidence
            }
            
            recommended_strategy = "adaptive"
            reasoning.append(f"Possible variant of {best_match.model_family} (confidence: {confidence:.2%})")
            reasoning.append("Using adaptive strategy with expanded testing")
            
        else:
            # Novel architecture
            is_novel = True
            confidence = 0.0
            
            architectural_features = {
                "layer_count": layer_count,
                "vocab_size": vocab_size,
                "fingerprint_hash": hashlib.sha256(
                    test_fingerprint.unified_hypervector.tobytes()
                ).hexdigest()[:16]
            }
            
            recommended_strategy = "exploratory"
            reasoning.append("Novel architecture detected")
            reasoning.append("Full exploratory analysis recommended")
            reasoning.append("Will create new base fingerprint after analysis")
        
        # Format closest matches
        closest = [(m[0], m[1]) for m in matches[:5]]
        
        return ModelIdentificationResult(
            identified_family=identified_family,
            confidence=confidence,
            closest_matches=closest,
            is_novel=is_novel,
            architectural_features=architectural_features,
            recommended_strategy=recommended_strategy,
            reasoning=reasoning
        )
    
    def get_testing_strategy(self, 
                            identification: ModelIdentificationResult) -> Dict[str, Any]:
        """
        Get adaptive testing strategy based on model identification.
        
        Args:
            identification: Result from identify_model
            
        Returns:
            Testing strategy configuration
        """
        # Determine strategy approach based on confidence
        if identification.confidence >= 0.85:
            approach = "targeted"
        elif identification.confidence >= 0.5:
            approach = "adaptive"
        else:
            approach = "exploratory"
            
        strategy = {
            "approach": approach,
            "cassettes": [],
            "focus_layers": [],
            "baseline_layers": [],
            "adversarial_config": {},
            "optimization": {}
        }
        
        if approach == "targeted":
            # Use known vulnerabilities and optimizations
            if hasattr(identification, 'closest_matches') and identification.closest_matches:
                base_fp = self.fingerprints[identification.closest_matches[0][0]]
                strategy["cassettes"] = base_fp.recommended_cassettes
                strategy["focus_layers"] = base_fp.vulnerable_layers
                strategy["baseline_layers"] = base_fp.stable_layers
            elif identification.reference_model:
                # Use reference model if available
                strategy["cassettes"] = ["syntactic", "semantic", "recursive"]
                strategy["focus_layers"] = list(range(0, 80, 15))  # Every 15th layer
                strategy["baseline_layers"] = [0, 10, 20, 30]
            else:
                # Default targeted strategy
                strategy["cassettes"] = ["syntactic", "semantic"]
                strategy["focus_layers"] = list(range(0, 50, 10))
                strategy["baseline_layers"] = [0, 25, 49]
            
            if hasattr(identification, 'closest_matches') and identification.closest_matches:
                base_fp = self.fingerprints[identification.closest_matches[0][0]]
                strategy["adversarial_config"] = {
                    "sensitivity": "high",
                    "target_layers": base_fp.vulnerable_layers,
                    "attack_types": ["divergence", "extraction", "inversion"]
                }
                strategy["optimization"] = {
                    "segment_size": base_fp.optimal_segment_size,
                    "batch_size": base_fp.optimal_batch_size,
                    "skip_stable": True
                }
                
        elif approach == "adaptive":
            # Moderate testing with some assumptions
            strategy["cassettes"] = ["syntactic", "semantic", "transform", "theory_of_mind"]
            strategy["focus_layers"] = list(range(0, 80, 10))  # Sample every 10th layer
            strategy["baseline_layers"] = [0, 20, 40, 60]
            strategy["adversarial_config"] = {
                "sensitivity": "medium",
                "exploration_mode": True
            }
            strategy["optimization"] = {
                "segment_size": 512,
                "batch_size": 4,
                "adaptive_sampling": True
            }
            
        else:  # exploratory
            # Comprehensive analysis for novel architecture
            strategy["cassettes"] = ["syntactic", "semantic", "recursive", "transform", 
                                   "theory_of_mind", "counterfactual", "meta"]
            strategy["focus_layers"] = list(range(0, min(100, identification.architectural_features.get("layer_count", 100)), 5))
            strategy["baseline_layers"] = list(range(0, min(100, identification.architectural_features.get("layer_count", 100)), 20))
            strategy["adversarial_config"] = {
                "sensitivity": "low",
                "full_spectrum": True,
                "discovery_mode": True
            }
            strategy["optimization"] = {
                "segment_size": 256,  # Smaller for discovery
                "batch_size": 2,
                "profile_all": True
            }
        
        return strategy
    
    def _compute_fingerprint_similarity(self, fp1: UnifiedFingerprint, fp2: UnifiedFingerprint) -> float:
        """Compute similarity between two fingerprints."""
        # Cosine similarity of unified hypervectors
        from scipy.spatial.distance import cosine
        
        sim = 1 - cosine(fp1.unified_hypervector, fp2.unified_hypervector)
        
        # Weight by quality
        sim *= (fp1.fingerprint_quality * fp2.fingerprint_quality)
        
        return float(sim)
    
    def _save_library(self):
        """Save fingerprint library to disk."""
        library_file = self.library_path / "fingerprint_library.json"
        
        # Convert to serializable format
        library_data = {
            "fingerprints": {},
            "family_index": self.family_index,
            "metadata": {
                "version": "1.0",
                "last_updated": str(datetime.now()),
                "num_fingerprints": len(self.fingerprints)
            }
        }
        
        for fp_id, fp in self.fingerprints.items():
            # Serialize fingerprint (skip numpy arrays for now)
            fp_data = {
                "model_family": fp.model_family,
                "model_size": fp.model_size,
                "architecture_version": fp.architecture_version,
                "num_layers": fp.num_layers,
                "hidden_size": fp.hidden_size,
                "num_attention_heads": fp.num_attention_heads,
                "vocab_size": fp.vocab_size,
                "layer_transitions": fp.layer_transitions,
                "behavioral_phases": fp.behavioral_phases,
                "recommended_cassettes": fp.recommended_cassettes,
                "vulnerable_layers": fp.vulnerable_layers,
                "stable_layers": fp.stable_layers,
                "memory_footprint_gb": fp.memory_footprint_gb,
                "optimal_segment_size": fp.optimal_segment_size,
                "optimal_batch_size": fp.optimal_batch_size,
                "fingerprint_id": fp.fingerprint_id,
                "creation_date": str(fp.creation_date),
                "validation_score": fp.validation_score,
                "source": fp.source
            }
            library_data["fingerprints"][fp_id] = fp_data
        
        with open(library_file, 'w') as f:
            json.dump(library_data, f, indent=2)
        
        logger.info(f"Saved {len(self.fingerprints)} fingerprints to library")
    
    def _load_library(self):
        """Load fingerprint library from disk."""
        library_file = self.library_path / "fingerprint_library.json"
        
        if not library_file.exists():
            logger.info("No existing library found, starting fresh")
            return
        
        try:
            with open(library_file, 'r') as f:
                library_data = json.load(f)
            
            self.family_index = library_data.get("family_index", {})
            
            # Reconstruct fingerprints (with mock arrays for now)
            for fp_id, fp_data in library_data.get("fingerprints", {}).items():
                # Create mock unified fingerprint
                mock_fp = UnifiedFingerprint(
                    unified_hypervector=np.random.randn(10000),
                    prompt_hypervector=np.random.randn(10000),
                    pathway_hypervector=np.random.randn(10000),
                    response_hypervector=np.random.randn(10000),
                    model_id=f"{fp_data['model_family']}-{fp_data['model_size']}",
                    fingerprint_quality=0.95,
                    binding_strength=0.9
                )
                
                base_fp = BaseModelFingerprint(
                    model_family=fp_data["model_family"],
                    model_size=fp_data["model_size"],
                    architecture_version=fp_data["architecture_version"],
                    unified_fingerprint=mock_fp,
                    num_layers=fp_data["num_layers"],
                    hidden_size=fp_data["hidden_size"],
                    num_attention_heads=fp_data["num_attention_heads"],
                    vocab_size=fp_data["vocab_size"],
                    layer_transitions=fp_data["layer_transitions"],
                    behavioral_phases=fp_data["behavioral_phases"],
                    attention_patterns=np.random.randn(100),
                    processing_signature=np.random.randn(200),
                    memory_footprint_gb=fp_data["memory_footprint_gb"],
                    optimal_segment_size=fp_data["optimal_segment_size"],
                    optimal_batch_size=fp_data["optimal_batch_size"],
                    recommended_cassettes=fp_data["recommended_cassettes"],
                    vulnerable_layers=fp_data["vulnerable_layers"],
                    stable_layers=fp_data["stable_layers"],
                    fingerprint_id=fp_data["fingerprint_id"],
                    validation_score=fp_data["validation_score"],
                    source=fp_data["source"]
                )
                
                self.fingerprints[fp_id] = base_fp
            
            logger.info(f"Loaded {len(self.fingerprints)} fingerprints from library")
            
        except Exception as e:
            logger.error(f"Error loading library: {e}")
            logger.info("Starting with fresh library")
    
    def export_fingerprint(self, fp_id: str, output_path: str):
        """Export a specific fingerprint for sharing."""
        if fp_id not in self.fingerprints:
            raise ValueError(f"Fingerprint {fp_id} not found")
        
        fp = self.fingerprints[fp_id]
        
        # Create comprehensive export
        export_data = {
            "fingerprint": asdict(fp),
            "metadata": {
                "export_date": str(datetime.now()),
                "library_version": "1.0",
                "fingerprint_id": fp_id
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported fingerprint to {output_path}")
    
    def import_fingerprint(self, import_path: str) -> str:
        """Import a fingerprint from file."""
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        fp_data = import_data["fingerprint"]
        
        # Reconstruct fingerprint
        # (Similar to _load_library but for single fingerprint)
        
        logger.info(f"Imported fingerprint from {import_path}")
        return fp_data["fingerprint_id"]