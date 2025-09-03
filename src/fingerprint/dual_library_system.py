"""
Dual Library System for REV
- Reference Library: Base fingerprints from smallest models of each family
- Active Library: Continuously updated with new runs
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelIdentification:
    """Result of model identification"""
    identified_family: Optional[str]
    confidence: float
    method: str  # "name_match", "fingerprint_match", "unknown"
    reference_model: Optional[str]  # e.g., "gpt2", "llama-7b"
    notes: str = ""


class DualLibrarySystem:
    """
    Manages both Reference and Active libraries for model fingerprinting.
    
    Reference Library: Contains base fingerprints from smallest/simplest models
    Active Library: Updated with every run, contains all accumulated knowledge
    """
    
    # Known model family patterns
    FAMILY_PATTERNS = {
        "gpt": ["gpt", "gpt2", "gpt-neo", "gpt-j", "distilgpt"],
        "llama": ["llama", "alpaca", "vicuna", "guanaco"],
        "mistral": ["mistral", "mixtral"],
        "qwen": ["qwen", "qwen2"],
        "yi": ["yi-"],
        "falcon": ["falcon"],
        "bloom": ["bloom", "bloomz"],
        "opt": ["opt-"],
        "pythia": ["pythia"],
        "dolly": ["dolly"],
        "stablelm": ["stablelm", "stable-lm"],
    }
    
    # Reference models (smallest of each family)
    REFERENCE_MODELS = {
        "gpt": "gpt2",  # 124M
        "llama": "llama-2-7b",  # 7B
        "mistral": "mistral-7b",  # 7B
        "qwen": "qwen-1.8b",  # 1.8B
        "yi": "yi-6b",  # 6B
        "falcon": "falcon-1b",  # 1B
        "bloom": "bloom-560m",  # 560M
        "opt": "opt-125m",  # 125M
        "pythia": "pythia-70m",  # 70M
    }
    
    def __init__(self, 
                 reference_path: str = "fingerprint_library/reference_library.json",
                 active_path: str = "fingerprint_library/active_library.json"):
        """Initialize dual library system."""
        self.reference_path = Path(reference_path)
        self.active_path = Path(active_path)
        
        # Load libraries
        self.reference_library = self._load_library(self.reference_path)
        self.active_library = self._load_library(self.active_path)
        
        # Ensure required structure
        self._ensure_library_structure(self.reference_library)
        self._ensure_library_structure(self.active_library)
    
    def _load_library(self, path: Path) -> Dict:
        """Load a library from disk."""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load library {path}: {e}")
        
        return {}
    
    def _ensure_library_structure(self, library: Dict):
        """Ensure library has required structure."""
        if "fingerprints" not in library:
            library["fingerprints"] = {}
        if "metadata" not in library:
            library["metadata"] = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
    
    def identify_model(self, model_path: str) -> ModelIdentification:
        """
        Identify a model using the dual library system.
        
        Step 1: Try name-based identification
        Step 2: If no match, return "needs_diagnostic" 
        """
        model_name = Path(model_path).name.lower()
        
        # Step 1: Try name-based identification
        for family, patterns in self.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_name:
                    # Found a family match!
                    reference_model = self.REFERENCE_MODELS.get(family)
                    
                    # Check if we have this reference in our library
                    if self._has_reference_fingerprint(family):
                        return ModelIdentification(
                            identified_family=family,
                            confidence=0.95,  # High confidence from name match
                            method="name_match",
                            reference_model=reference_model,
                            notes=f"Matched pattern '{pattern}' in model name"
                        )
                    else:
                        return ModelIdentification(
                            identified_family=family,
                            confidence=0.85,
                            method="name_match_no_reference",
                            reference_model=reference_model,
                            notes=f"Matched pattern '{pattern}' but no reference fingerprint available"
                        )
        
        # No name match - need diagnostic fingerprinting
        return ModelIdentification(
            identified_family=None,
            confidence=0.0,
            method="unknown",
            reference_model=None,
            notes="No family pattern matched - diagnostic fingerprinting required"
        )
    
    def _has_reference_fingerprint(self, family: str) -> bool:
        """Check if we have a reference fingerprint for this family."""
        for fp_id, fp_data in self.reference_library.get("fingerprints", {}).items():
            if fp_data.get("model_family") == family:
                return True
        return False
    
    def get_reference_fingerprint(self, family: str) -> Optional[Dict]:
        """Get the reference fingerprint for a model family."""
        for fp_id, fp_data in self.reference_library.get("fingerprints", {}).items():
            if fp_data.get("model_family") == family:
                return fp_data
        return None
    
    def add_to_active_library(self, fingerprint_data: Dict, model_info: Dict):
        """Add a new fingerprint to the active library."""
        # Generate ID
        fp_id = f"{model_info['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add to active library
        self.active_library["fingerprints"][fp_id] = {
            **fingerprint_data,
            **model_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metadata
        self.active_library["metadata"]["last_updated"] = datetime.now().isoformat()
        self.active_library["metadata"]["num_fingerprints"] = len(self.active_library["fingerprints"])
        
        # Save
        self._save_library(self.active_library, self.active_path)
        logger.info(f"Added fingerprint {fp_id} to active library")
    
    def add_reference_fingerprint(self, family: str, fingerprint_data: Dict):
        """Add a reference fingerprint (should be rare - only for new families)."""
        if self._has_reference_fingerprint(family):
            logger.warning(f"Reference fingerprint for {family} already exists")
            return
        
        fp_id = f"{family}_reference"
        self.reference_library["fingerprints"][fp_id] = {
            **fingerprint_data,
            "model_family": family,
            "is_reference": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metadata
        self.reference_library["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save
        self._save_library(self.reference_library, self.reference_path)
        logger.info(f"Added reference fingerprint for {family}")
    
    def _save_library(self, library: Dict, path: Path):
        """Save library to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(library, f, indent=2)
    
    def get_testing_strategy(self, identification: ModelIdentification) -> Dict:
        """
        Get testing strategy based on identification.
        
        Returns:
            Dict with testing configuration
        """
        if identification.method == "name_match":
            # We know the family - use targeted testing
            reference_fp = self.get_reference_fingerprint(identification.identified_family)
            if reference_fp:
                return {
                    "strategy": "targeted",
                    "reference_model": identification.reference_model,
                    "focus_layers": reference_fp.get("vulnerable_layers", []),
                    "skip_layers": reference_fp.get("stable_layers", []),
                    "cassettes": reference_fp.get("recommended_cassettes", ["syntactic", "semantic"]),
                    "expected_patterns": reference_fp.get("behavioral_patterns", {}),
                    "challenges": 10,  # Standard amount
                    "notes": f"Using reference from {identification.reference_model}"
                }
            else:
                return {
                    "strategy": "standard",
                    "challenges": 15,
                    "notes": "Family identified but no reference available"
                }
        
        elif identification.method == "unknown":
            # Need diagnostic first
            return {
                "strategy": "diagnostic",
                "challenges": 5,  # Quick diagnostic
                "sample_layers": list(range(0, 100, 10)),  # Sample every 10th layer
                "notes": "Unknown model - running diagnostic fingerprinting first"
            }
        
        else:
            # Default fallback
            return {
                "strategy": "exploratory",
                "challenges": 20,
                "notes": "Full exploratory analysis"
            }


def create_dual_library() -> DualLibrarySystem:
    """Create and initialize the dual library system."""
    return DualLibrarySystem()


# Integration helper
def identify_and_strategize(model_path: str) -> Tuple[ModelIdentification, Dict]:
    """
    Identify a model and get its testing strategy.
    
    Returns:
        (identification, strategy)
    """
    library = create_dual_library()
    identification = library.identify_model(model_path)
    strategy = library.get_testing_strategy(identification)
    
    return identification, strategy