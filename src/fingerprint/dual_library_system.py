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
        Identify a model through LIGHT behavioral analysis.
        
        This does a QUICK probe (not deep analysis) to identify the model family
        so we can use reference libraries for 15-20x speedup.
        
        Process:
        1. Quick probe (1 prompt per layer sample)
        2. Build light topological signature
        3. Match against reference library
        4. Return family identification for targeted testing
        """
        model_name = Path(model_path).name.lower()
        
        # SECURITY: We do behavioral verification, but LIGHTWEIGHT
        # This is just for family identification, not full fingerprinting
        logger.info(f"Starting LIGHT behavioral probe for {model_name}")
        logger.info("Quick topology scan for reference matching...")
        
        # For now, mark as requiring light probe
        # The actual light probe will be done by the pipeline
        # Then it will call identify_from_behavioral_analysis with the results
        return ModelIdentification(
            identified_family=None,
            confidence=0.0,
            method="needs_light_probe",  # Signal for light analysis
            reference_model=None,
            notes="Light behavioral probe needed for family identification"
        )
    
    def identify_from_behavioral_analysis(self, fingerprint_data: Dict) -> ModelIdentification:
        """
        Identify a model based on behavioral analysis results.
        This is the REAL identification based on variance profiles and topology.
        
        Args:
            fingerprint_data: The behavioral fingerprint data from analysis
            
        Returns:
            ModelIdentification with actual confidence based on behavior
        """
        # Extract behavioral topology from fingerprint
        restriction_sites = fingerprint_data.get("restriction_sites", [])
        variance_profile = fingerprint_data.get("variance_profile", [])
        layer_divergences = fingerprint_data.get("layer_divergences", {})
        
        if not restriction_sites and not variance_profile:
            return ModelIdentification(
                identified_family=None,
                confidence=0.0,
                method="behavioral_incomplete",
                reference_model=None,
                notes="Insufficient behavioral data for identification"
            )
        
        # Compare topology against reference library
        best_match = None
        best_similarity = 0.0
        
        for fp_id, ref_data in self.reference_library.get("fingerprints", {}).items():
            ref_sites = ref_data.get("restriction_sites", [])
            ref_profile = ref_data.get("variance_profile", [])
            
            # Compute topological similarity
            similarity = self._compute_topology_similarity(
                restriction_sites, ref_sites,
                variance_profile, ref_profile
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = ref_data.get("model_family")
        
        # Determine confidence from similarity score
        if best_similarity > 0.85:
            confidence = best_similarity
            method = "behavioral_match"
            notes = f"Strong behavioral match (similarity: {best_similarity:.2%})"
        elif best_similarity > 0.60:
            confidence = best_similarity
            method = "behavioral_partial"
            notes = f"Partial behavioral match (similarity: {best_similarity:.2%})"
        else:
            confidence = 0.0
            method = "behavioral_unknown"
            notes = f"No behavioral match found (max similarity: {best_similarity:.2%})"
            best_match = None
        
        reference_model = None
        if best_match:
            reference_model = self.REFERENCE_MODELS.get(best_match)
        
        return ModelIdentification(
            identified_family=best_match,
            confidence=confidence,
            method=method,
            reference_model=reference_model,
            notes=notes
        )
    
    def _compute_topology_similarity(self, sites1, sites2, profile1, profile2):
        """Compute similarity between two behavioral topologies.
        
        Enhanced to focus on variance delta patterns rather than layer counts.
        """
        if not sites1 or not sites2:
            return 0.0
        
        # Extract variance delta patterns (the actual behavioral changes)
        def extract_delta_pattern(sites):
            """Extract pattern of variance changes."""
            pattern = []
            for site in sites:
                if isinstance(site, dict):
                    # Get the delta magnitude and direction
                    delta = site.get("divergence_delta", 0.0)
                    percent = site.get("percent_change", 0.0)
                    pattern.append({
                        "delta": abs(delta),  # Magnitude
                        "direction": 1 if delta > 0 else -1 if delta < 0 else 0,
                        "percent": abs(percent),
                        "relative_pos": site.get("layer", 0)  # Will normalize later
                    })
            return pattern
        
        pattern1 = extract_delta_pattern(sites1)
        pattern2 = extract_delta_pattern(sites2)
        
        # Normalize positions relative to model depth (don't care about absolute layer count)
        if pattern1 and pattern2:
            max_pos1 = max([p["relative_pos"] for p in pattern1], default=1)
            max_pos2 = max([p["relative_pos"] for p in pattern2], default=1)
            
            for p in pattern1:
                p["relative_pos"] = p["relative_pos"] / max_pos1 if max_pos1 > 0 else 0
            for p in pattern2:
                p["relative_pos"] = p["relative_pos"] / max_pos2 if max_pos2 > 0 else 0
        
        # Compare delta patterns (behavior) - PRIMARY FACTOR
        delta_similarity = 0.0
        if pattern1 and pattern2:
            # Match patterns based on similar deltas and directions
            matched = 0
            total_comparisons = 0
            
            for p1 in pattern1:
                # Skip near-zero changes (noise)
                if p1["delta"] < 0.05:
                    continue
                    
                best_match = 0.0
                for p2 in pattern2:
                    if p2["delta"] < 0.05:
                        continue
                    
                    # Compare delta magnitudes (normalized)
                    delta_sim = 1.0 - min(abs(p1["delta"] - p2["delta"]) / max(p1["delta"], p2["delta"], 0.01), 1.0)
                    
                    # Compare directions (1.0 if same, 0.0 if opposite)
                    dir_sim = 1.0 if p1["direction"] == p2["direction"] else 0.0
                    
                    # Compare relative positions (less important)
                    pos_sim = 1.0 - abs(p1["relative_pos"] - p2["relative_pos"])
                    
                    # Weighted combination: behavior matters more than position
                    match_score = 0.5 * delta_sim + 0.3 * dir_sim + 0.2 * pos_sim
                    best_match = max(best_match, match_score)
                
                matched += best_match
                total_comparisons += 1
            
            if total_comparisons > 0:
                delta_similarity = matched / total_comparisons
        
        # Compare site positions (SECONDARY FACTOR - much less weight)
        position_similarity = 0.0
        if sites1 and sites2:
            # Just check if restriction sites occur at similar relative positions
            max1 = max([s.get("layer", s) if isinstance(s, dict) else s for s in sites1], default=1)
            max2 = max([s.get("layer", s) if isinstance(s, dict) else s for s in sites2], default=1)
            
            norm1 = set([(s.get("layer", s) if isinstance(s, dict) else s) / max1 for s in sites1])
            norm2 = set([(s.get("layer", s) if isinstance(s, dict) else s) / max2 for s in sites2])
            
            # Count positions that are close
            matches = 0
            for n1 in norm1:
                for n2 in norm2:
                    if abs(n1 - n2) < 0.15:  # Within 15% relative position
                        matches += 1
                        break
            
            position_similarity = matches / max(len(norm1), len(norm2)) if max(len(norm1), len(norm2)) > 0 else 0
        
        # Compare variance profiles if available (TERTIARY FACTOR)
        profile_similarity = 0.0
        if profile1 and profile2:
            # Simple correlation of variance patterns
            min_len = min(len(profile1), len(profile2))
            if min_len > 0:
                corr = sum([p1 * p2 for p1, p2 in zip(profile1[:min_len], profile2[:min_len])])
                norm = (sum([p**2 for p in profile1[:min_len]]) * sum([p**2 for p in profile2[:min_len]]))**0.5
                if norm > 0:
                    profile_similarity = corr / norm
        
        # NEW WEIGHTS: Variance patterns matter most, positions matter least
        # 60% delta patterns, 25% relative positions, 15% profiles
        return 0.6 * delta_similarity + 0.25 * position_similarity + 0.15 * profile_similarity
    
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
        """
        Add a reference fingerprint (should be rare - only for new families).
        
        VALIDATION: Ensures reference has adequate quality before adding.
        """
        # Check if reference already exists
        if self._has_reference_fingerprint(family):
            logger.warning(f"Reference fingerprint for {family} already exists")
            return
        
        # VALIDATION: Ensure reference quality
        MIN_CHALLENGES = 250  # Minimum required for a valid reference
        challenges = fingerprint_data.get("challenges_processed", 0)
        
        if challenges < MIN_CHALLENGES:
            logger.error(
                f"REJECTED: Reference for {family} has only {challenges} challenges "
                f"(minimum: {MIN_CHALLENGES}). Run with --enable-prompt-orchestration!"
            )
            raise ValueError(
                f"Reference fingerprint rejected: insufficient challenges ({challenges} < {MIN_CHALLENGES}). "
                f"Rebuild with: --build-reference --enable-prompt-orchestration"
            )
        
        # Validate restriction sites exist
        restriction_sites = fingerprint_data.get("restriction_sites", [])
        if len(restriction_sites) < 3:
            logger.warning(
                f"Reference for {family} has only {len(restriction_sites)} restriction sites. "
                f"Consider rebuilding with deeper analysis."
            )
        
        # Validate behavioral metrics exist
        if "behavioral_metrics" not in fingerprint_data:
            logger.warning(f"Reference for {family} missing behavioral metrics")
        
        # Add reference with validation metadata
        fp_id = f"{family}_reference"
        self.reference_library["fingerprints"][fp_id] = {
            **fingerprint_data,
            "model_family": family,
            "is_reference": True,
            "validation": {
                "challenges": challenges,
                "restriction_sites": len(restriction_sites),
                "passed_validation": True,
                "validation_timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metadata
        self.reference_library["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Save
        self._save_library(self.reference_library, self.reference_path)
        logger.info(
            f"✅ Added VALIDATED reference for {family}: "
            f"{challenges} challenges, {len(restriction_sites)} restriction sites"
        )
    
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
        # If we have a behavioral match with good confidence, use reference
        if identification.method == "behavioral_match" and identification.confidence > 0.7:
            # We have a family identification - check for reference
            reference_fp = self.get_reference_fingerprint(identification.identified_family)
            if reference_fp:
                # Extract restriction sites from reference
                restriction_sites = reference_fp.get("restriction_sites", [])
                focus_layers = []
                if restriction_sites:
                    # Use restriction sites as focus layers
                    for site in restriction_sites:
                        if isinstance(site, dict) and "layer" in site:
                            focus_layers.append(site["layer"])
                        elif isinstance(site, int):
                            focus_layers.append(site)
                
                return {
                    "strategy": "targeted",
                    "reference_model": identification.reference_model,
                    "focus_layers": focus_layers[:10],  # Top 10 restriction sites
                    "restriction_sites": restriction_sites,
                    "skip_layers": reference_fp.get("stable_layers", []),
                    "cassettes": reference_fp.get("recommended_cassettes", ["syntactic", "semantic"]),
                    "expected_patterns": reference_fp.get("behavioral_patterns", {}),
                    "challenges": reference_fp.get("challenges_processed", 10),  # Use same as reference
                    "notes": f"Using reference from {identification.reference_model} with {len(focus_layers)} restriction sites"
                }
            else:
                # Family identified but no reference available
                return {
                    "strategy": "standard",
                    "challenges": 15,
                    "notes": f"Family {identification.identified_family} identified but no reference available"
                }
        
        elif identification.method == "needs_light_probe":
            # Do LIGHT probe first (not deep analysis)
            return {
                "strategy": "light_probe",
                "challenges": 5,  # Just 5 quick probes
                "sample_layers": "adaptive",  # Sample based on layer count
                "notes": "Light probe for family identification (enables 15-20x speedup)"
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