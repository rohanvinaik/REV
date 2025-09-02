#!/usr/bin/env python3
"""
Advanced Probe Cassette System for REV Framework
Implements cassette-style probe groups for second-phase analysis after base topology discovery.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
import hashlib
import json
from abc import ABC, abstractmethod

class ProbeType(Enum):
    """Categories of cognitive challenges for layer-specific analysis."""
    SYNTACTIC = "syntactic"           # Token/structure manipulation
    SEMANTIC = "semantic"              # Meaning and relationships
    RECURSIVE = "recursive"            # Self-referential reasoning
    ADVERSARIAL = "adversarial"       # Robustness testing
    THEORY_OF_MIND = "theory_of_mind" # Belief and intention modeling
    COMPUTATIONAL = "computational"    # Algorithmic reasoning
    ANALOGICAL = "analogical"         # Cross-domain mapping
    COUNTERFACTUAL = "counterfactual" # Alternative reasoning
    META_COGNITIVE = "meta_cognitive"  # Reasoning about reasoning
    MULTIMODAL = "multimodal"         # Cross-linguistic/format

@dataclass
class Probe:
    """Individual probe with metadata."""
    text: str
    type: ProbeType
    complexity: int  # 1-10 scale
    expected_difficulty_by_layer: Dict[str, float]  # Layer range -> expected difficulty
    tags: List[str]
    id: str = None
    
    def __post_init__(self):
        if not self.id:
            # Generate stable ID from content
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:8]

@dataclass
class ProbeCassette:
    """Collection of related probes for systematic analysis."""
    name: str
    description: str
    probe_type: ProbeType
    probes: List[Probe]
    layer_ranges: List[tuple]  # [(start, end), ...] for recommended layers
    required_baseline: bool = True  # Whether baseline topology needed first
    
    def get_probes_for_layer(self, layer: int) -> List[Probe]:
        """Get relevant probes for specific layer."""
        relevant = []
        for start, end in self.layer_ranges:
            if start <= layer <= end:
                # Filter by expected effectiveness at this layer
                for probe in self.probes:
                    for range_key, difficulty in probe.expected_difficulty_by_layer.items():
                        range_start, range_end = map(int, range_key.split('-'))
                        if range_start <= layer <= range_end:
                            relevant.append(probe)
                            break
        return relevant

class ProbeFactory:
    """Factory for creating specialized probe cassettes."""
    
    @staticmethod
    def create_syntactic_cassette() -> ProbeCassette:
        """Probes for testing syntactic processing and token manipulation."""
        return ProbeCassette(
            name="syntactic_adversarial",
            description="Test model's robustness to syntactic variations and corruptions",
            probe_type=ProbeType.SYNTACTIC,
            layer_ranges=[(0, 20), (70, 79)],  # Early and late layers
            probes=[
                Probe(
                    text="Wh@t h@pp3n5 t0 y0ur und3r5t@nd1ng wh3n t0k3n5 @r3 c0rrupt3d?",
                    type=ProbeType.SYNTACTIC,
                    complexity=6,
                    expected_difficulty_by_layer={"0-10": 0.9, "11-30": 0.5, "31-79": 0.3},
                    tags=["corruption", "tokenization", "robustness"]
                ),
                Probe(
                    text="Parse this: The horse raced past the barn fell and understood why.",
                    type=ProbeType.SYNTACTIC,
                    complexity=7,
                    expected_difficulty_by_layer={"0-15": 0.8, "16-40": 0.4, "41-79": 0.2},
                    tags=["garden_path", "parsing", "ambiguity"]
                ),
                Probe(
                    text="ThE   qUiCk BrOwN   FOX     jUmPs    OvEr ThE    lAzY   DOG",
                    type=ProbeType.SYNTACTIC,
                    complexity=3,
                    expected_difficulty_by_layer={"0-5": 0.7, "6-20": 0.2, "21-79": 0.1},
                    tags=["spacing", "capitalization", "normalization"]
                ),
                Probe(
                    text=".)sentence backwards a is This(",
                    type=ProbeType.SYNTACTIC,
                    complexity=4,
                    expected_difficulty_by_layer={"0-10": 0.8, "11-30": 0.3, "31-79": 0.1},
                    tags=["reversal", "structure", "reconstruction"]
                ),
            ]
        )
    
    @staticmethod
    def create_recursive_cassette() -> ProbeCassette:
        """Probes for testing recursive and self-referential reasoning."""
        return ProbeCassette(
            name="recursive_depth",
            description="Test model's capacity for recursive and self-referential reasoning",
            probe_type=ProbeType.RECURSIVE,
            layer_ranges=[(15, 50)],  # Mid-to-deep layers
            probes=[
                Probe(
                    text="If I asked you to explain why explaining explanations is recursive, how many levels of recursion would your explanation contain?",
                    type=ProbeType.RECURSIVE,
                    complexity=8,
                    expected_difficulty_by_layer={"0-20": 0.9, "21-40": 0.6, "41-79": 0.4},
                    tags=["meta_recursion", "depth", "self_reference"]
                ),
                Probe(
                    text="Define f(n) = f(f(n-1)) + f(n-f(n-1)) for n>2, f(1)=f(2)=1. What is f(5)?",
                    type=ProbeType.RECURSIVE,
                    complexity=9,
                    expected_difficulty_by_layer={"0-30": 0.95, "31-50": 0.7, "51-79": 0.5},
                    tags=["computation", "nested_calls", "memoization"]
                ),
                Probe(
                    text="This sentence contains exactly __ words, where __ is the number that makes this sentence true.",
                    type=ProbeType.RECURSIVE,
                    complexity=7,
                    expected_difficulty_by_layer={"0-25": 0.8, "26-50": 0.5, "51-79": 0.3},
                    tags=["self_reference", "fixed_point", "counting"]
                ),
                Probe(
                    text="Complete truthfully: 'This statement is false because...'",
                    type=ProbeType.RECURSIVE,
                    complexity=6,
                    expected_difficulty_by_layer={"0-20": 0.7, "21-45": 0.5, "46-79": 0.4},
                    tags=["paradox", "liar", "consistency"]
                ),
            ]
        )
    
    @staticmethod
    def create_theory_of_mind_cassette() -> ProbeCassette:
        """Probes for testing belief modeling and intentionality understanding."""
        return ProbeCassette(
            name="theory_of_mind_cascade",
            description="Test model's ability to track nested beliefs and intentions",
            probe_type=ProbeType.THEORY_OF_MIND,
            layer_ranges=[(25, 60)],  # Deep semantic layers
            probes=[
                Probe(
                    text="Alice thinks Bob knows that Carol believes David suspects Alice is lying. Who needs convincing first?",
                    type=ProbeType.THEORY_OF_MIND,
                    complexity=8,
                    expected_difficulty_by_layer={"0-30": 0.9, "31-50": 0.6, "51-79": 0.4},
                    tags=["nested_beliefs", "chain", "resolution"]
                ),
                Probe(
                    text="I want you to understand that I need you to help me make them realize what we're trying to accomplish. Count the intentionality levels.",
                    type=ProbeType.THEORY_OF_MIND,
                    complexity=7,
                    expected_difficulty_by_layer={"0-35": 0.85, "36-55": 0.5, "56-79": 0.3},
                    tags=["intentionality", "levels", "counting"]
                ),
                Probe(
                    text="If everyone wrongly believes everyone else knows a secret, but nobody actually knows it, what happens when someone asks?",
                    type=ProbeType.THEORY_OF_MIND,
                    complexity=9,
                    expected_difficulty_by_layer={"0-40": 0.95, "41-60": 0.7, "61-79": 0.5},
                    tags=["false_belief", "common_knowledge", "paradox"]
                ),
                Probe(
                    text="Model what you think I think you're thinking about my model of your thought process.",
                    type=ProbeType.THEORY_OF_MIND,
                    complexity=10,
                    expected_difficulty_by_layer={"0-45": 0.98, "46-65": 0.8, "66-79": 0.6},
                    tags=["recursive_modeling", "meta_cognition", "depth"]
                ),
            ]
        )
    
    @staticmethod
    def create_computational_cassette() -> ProbeCassette:
        """Probes for testing algorithmic and computational reasoning."""
        return ProbeCassette(
            name="computational_complexity",
            description="Test model's ability to perform algorithmic reasoning",
            probe_type=ProbeType.COMPUTATIONAL,
            layer_ranges=[(20, 55)],
            probes=[
                Probe(
                    text="Find a Hamiltonian path: A→B, B→C, C→A, A→D, D→B. Start from A.",
                    type=ProbeType.COMPUTATIONAL,
                    complexity=8,
                    expected_difficulty_by_layer={"0-25": 0.9, "26-45": 0.6, "46-79": 0.4},
                    tags=["graph", "NP_complete", "search"]
                ),
                Probe(
                    text="Color nodes A,B,C,D with R,G,B where edges A-B, B-C, C-D, D-A, A-C require different colors.",
                    type=ProbeType.COMPUTATIONAL,
                    complexity=7,
                    expected_difficulty_by_layer={"0-30": 0.85, "31-50": 0.5, "51-79": 0.3},
                    tags=["constraint_satisfaction", "coloring", "search"]
                ),
                Probe(
                    text="If SHA256('x') = y and SHA256(y) = 'x', what property must SHA256 have?",
                    type=ProbeType.COMPUTATIONAL,
                    complexity=9,
                    expected_difficulty_by_layer={"0-35": 0.95, "36-55": 0.7, "56-79": 0.5},
                    tags=["cryptography", "hash", "impossibility"]
                ),
                Probe(
                    text="Sort [3,1,4,1,5,9,2,6] using only comparisons. Show each step.",
                    type=ProbeType.COMPUTATIONAL,
                    complexity=5,
                    expected_difficulty_by_layer={"0-20": 0.6, "21-40": 0.3, "41-79": 0.1},
                    tags=["sorting", "algorithm", "steps"]
                ),
            ]
        )
    
    @staticmethod
    def create_analogical_cassette() -> ProbeCassette:
        """Probes for testing cross-domain analogical reasoning."""
        return ProbeCassette(
            name="analogical_mapping",
            description="Test model's ability to map structures across domains",
            probe_type=ProbeType.ANALOGICAL,
            layer_ranges=[(30, 70)],  # Higher-level reasoning layers
            probes=[
                Probe(
                    text="If DNA:RNA:Protein :: Idea:?:Action, what is '?'",
                    type=ProbeType.ANALOGICAL,
                    complexity=7,
                    expected_difficulty_by_layer={"0-35": 0.8, "36-55": 0.5, "56-79": 0.3},
                    tags=["structure_mapping", "biology", "cognition"]
                ),
                Probe(
                    text="Map solar system to neural network: planets=___, orbits=___, gravity=___",
                    type=ProbeType.ANALOGICAL,
                    complexity=8,
                    expected_difficulty_by_layer={"0-40": 0.85, "41-60": 0.6, "61-79": 0.4},
                    tags=["system_mapping", "physics", "ML"]
                ),
                Probe(
                    text="Apply Fibonacci pattern to emotions: happy, sad, happy+sad, ?",
                    type=ProbeType.ANALOGICAL,
                    complexity=6,
                    expected_difficulty_by_layer={"0-30": 0.7, "31-50": 0.4, "51-79": 0.2},
                    tags=["pattern_transfer", "sequence", "emotion"]
                ),
                Probe(
                    text="Explain 4D space using only 2D metaphors for a 1D being.",
                    type=ProbeType.ANALOGICAL,
                    complexity=10,
                    expected_difficulty_by_layer={"0-45": 0.95, "46-65": 0.8, "66-79": 0.6},
                    tags=["dimensional_reduction", "explanation", "constraints"]
                ),
            ]
        )
    
    @staticmethod
    def create_counterfactual_cassette() -> ProbeCassette:
        """Probes for testing counterfactual and alternative reasoning."""
        return ProbeCassette(
            name="counterfactual_worlds",
            description="Test model's ability to reason about alternative realities",
            probe_type=ProbeType.COUNTERFACTUAL,
            layer_ranges=[(35, 75)],
            probes=[
                Probe(
                    text="If gravity repelled instead of attracted, describe the evolution of intelligence.",
                    type=ProbeType.COUNTERFACTUAL,
                    complexity=9,
                    expected_difficulty_by_layer={"0-40": 0.9, "41-60": 0.7, "61-79": 0.5},
                    tags=["physics", "evolution", "speculation"]
                ),
                Probe(
                    text="In a world where effect precedes cause, how would science work?",
                    type=ProbeType.COUNTERFACTUAL,
                    complexity=10,
                    expected_difficulty_by_layer={"0-45": 0.95, "46-65": 0.8, "66-79": 0.6},
                    tags=["causality", "temporal", "methodology"]
                ),
                Probe(
                    text="If mathematics were base-7 from the start, what would be different today?",
                    type=ProbeType.COUNTERFACTUAL,
                    complexity=7,
                    expected_difficulty_by_layer={"0-35": 0.75, "36-55": 0.5, "56-79": 0.3},
                    tags=["mathematics", "history", "systems"]
                ),
                Probe(
                    text="Describe communication if humans could only speak in questions.",
                    type=ProbeType.COUNTERFACTUAL,
                    complexity=6,
                    expected_difficulty_by_layer={"0-30": 0.7, "31-50": 0.4, "51-79": 0.2},
                    tags=["language", "constraints", "adaptation"]
                ),
            ]
        )
    
    @staticmethod
    def create_multimodal_cassette() -> ProbeCassette:
        """Probes mixing languages, formats, and modalities."""
        return ProbeCassette(
            name="multimodal_fusion",
            description="Test model's ability to handle mixed formats and languages",
            probe_type=ProbeType.MULTIMODAL,
            layer_ranges=[(10, 40), (60, 79)],
            probes=[
                Probe(
                    text="Explain pourquoi 機械学習 ist важно für die Zukunft של AI.",
                    type=ProbeType.MULTIMODAL,
                    complexity=7,
                    expected_difficulty_by_layer={"0-20": 0.8, "21-40": 0.5, "41-79": 0.3},
                    tags=["multilingual", "code_switching", "coherence"]
                ),
                Probe(
                    text="01010111 01101000 01100001 01110100 means 什么 in español?",
                    type=ProbeType.MULTIMODAL,
                    complexity=6,
                    expected_difficulty_by_layer={"0-15": 0.7, "16-35": 0.4, "36-79": 0.2},
                    tags=["binary", "translation", "formats"]
                ),
                Probe(
                    text="🔢➕🔢🟰🔢🔢 translates to which mathematical equation?",
                    type=ProbeType.MULTIMODAL,
                    complexity=5,
                    expected_difficulty_by_layer={"0-25": 0.6, "26-45": 0.3, "46-79": 0.1},
                    tags=["emoji", "math", "visual"]
                ),
                Probe(
                    text="ROT13(Base64('Hello')) in 1337speak =?",
                    type=ProbeType.MULTIMODAL,
                    complexity=8,
                    expected_difficulty_by_layer={"0-30": 0.85, "31-50": 0.6, "51-79": 0.4},
                    tags=["encoding", "transformation", "leetspeak"]
                ),
            ]
        )

class CassetteLibrary:
    """Library managing all probe cassettes for second-phase analysis."""
    
    def __init__(self):
        self.cassettes: Dict[str, ProbeCassette] = {}
        self.factory = ProbeFactory()
        self._initialize_library()
    
    def _initialize_library(self):
        """Load all available cassettes."""
        self.cassettes["syntactic"] = self.factory.create_syntactic_cassette()
        self.cassettes["recursive"] = self.factory.create_recursive_cassette()
        self.cassettes["theory_of_mind"] = self.factory.create_theory_of_mind_cassette()
        self.cassettes["computational"] = self.factory.create_computational_cassette()
        self.cassettes["analogical"] = self.factory.create_analogical_cassette()
        self.cassettes["counterfactual"] = self.factory.create_counterfactual_cassette()
        self.cassettes["multimodal"] = self.factory.create_multimodal_cassette()
    
    def get_cassette(self, name: str) -> Optional[ProbeCassette]:
        """Retrieve a specific cassette by name."""
        return self.cassettes.get(name)
    
    def get_cassettes_for_layer(self, layer: int) -> List[ProbeCassette]:
        """Get all cassettes relevant for a specific layer."""
        relevant = []
        for cassette in self.cassettes.values():
            for start, end in cassette.layer_ranges:
                if start <= layer <= end:
                    relevant.append(cassette)
                    break
        return relevant
    
    def get_cassettes_by_type(self, probe_type: ProbeType) -> List[ProbeCassette]:
        """Get all cassettes of a specific type."""
        return [c for c in self.cassettes.values() if c.probe_type == probe_type]
    
    def generate_probe_schedule(self, topology: Dict[str, Any]) -> Dict[int, List[Probe]]:
        """
        Generate an optimal probe schedule based on discovered topology.
        
        Args:
            topology: Discovered behavioral topology from phase 1
            
        Returns:
            Mapping of layer index to probes to run
        """
        schedule = {}
        
        # Parse topology for key transition points
        restriction_sites = topology.get("restriction_sites", [])
        stable_regions = topology.get("stable_regions", [])
        phase_boundaries = topology.get("phase_boundaries", [])
        
        # Schedule intensive probing at boundaries
        for site in restriction_sites:
            layer = site.get("layer", 0)
            # Get all relevant cassettes for this boundary
            cassettes = self.get_cassettes_for_layer(layer)
            schedule[layer] = []
            for cassette in cassettes:
                # Add high-complexity probes at boundaries
                probes = [p for p in cassette.get_probes_for_layer(layer) 
                         if p.complexity >= 7]
                schedule[layer].extend(probes[:2])  # Limit to 2 per cassette
        
        # Schedule lighter probing in stable regions
        for region in stable_regions:
            start = region.get("start", 0)
            end = region.get("end", 0)
            # Sample 3 layers from each stable region
            sample_layers = [start, (start + end) // 2, end]
            
            for layer in sample_layers:
                cassettes = self.get_cassettes_for_layer(layer)
                schedule[layer] = schedule.get(layer, [])
                for cassette in cassettes:
                    # Add medium-complexity probes in stable regions
                    probes = [p for p in cassette.get_probes_for_layer(layer) 
                             if 4 <= p.complexity <= 6]
                    schedule[layer].extend(probes[:1])  # Just 1 per cassette
        
        return schedule
    
    def export_schedule(self, schedule: Dict[int, List[Probe]], filepath: str):
        """Export probe schedule to JSON for execution."""
        export_data = {
            "version": "1.0",
            "schedule": {}
        }
        
        for layer, probes in schedule.items():
            export_data["schedule"][str(layer)] = [
                {
                    "id": probe.id,
                    "text": probe.text,
                    "type": probe.type.value,
                    "complexity": probe.complexity,
                    "tags": probe.tags
                }
                for probe in probes
            ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def analyze_results(self, results: Dict[int, Dict[str, float]], 
                       schedule: Dict[int, List[Probe]]) -> Dict[str, Any]:
        """
        Analyze probe results to identify layer capabilities.
        
        Args:
            results: Mapping of layer -> probe_id -> divergence score
            schedule: The probe schedule that was executed
            
        Returns:
            Analysis of layer capabilities and specializations
        """
        analysis = {
            "layer_specializations": {},
            "probe_type_effectiveness": {},
            "complexity_thresholds": {},
            "anomalies": []
        }
        
        # Analyze by layer
        for layer, probe_results in results.items():
            if layer not in schedule:
                continue
                
            layer_probes = {p.id: p for p in schedule[layer]}
            
            # Group by probe type
            type_scores = {}
            for probe_id, score in probe_results.items():
                if probe_id in layer_probes:
                    probe = layer_probes[probe_id]
                    if probe.type not in type_scores:
                        type_scores[probe.type] = []
                    type_scores[probe.type].append(score)
            
            # Identify specialization
            if type_scores:
                best_type = min(type_scores.keys(), 
                              key=lambda t: sum(type_scores[t])/len(type_scores[t]))
                worst_type = max(type_scores.keys(), 
                               key=lambda t: sum(type_scores[t])/len(type_scores[t]))
                
                analysis["layer_specializations"][layer] = {
                    "best_capability": best_type.value,
                    "worst_capability": worst_type.value,
                    "score_spread": max([sum(s)/len(s) for s in type_scores.values()]) - 
                                   min([sum(s)/len(s) for s in type_scores.values()])
                }
        
        # Analyze probe type effectiveness across layers
        for probe_type in ProbeType:
            type_results = []
            for layer, probe_results in results.items():
                if layer in schedule:
                    layer_probes = {p.id: p for p in schedule[layer]}
                    for probe_id, score in probe_results.items():
                        if probe_id in layer_probes and layer_probes[probe_id].type == probe_type:
                            type_results.append((layer, score))
            
            if type_results:
                analysis["probe_type_effectiveness"][probe_type.value] = {
                    "mean_score": sum(s for _, s in type_results) / len(type_results),
                    "best_layer": min(type_results, key=lambda x: x[1])[0],
                    "worst_layer": max(type_results, key=lambda x: x[1])[0]
                }
        
        # Identify complexity thresholds
        for complexity in range(1, 11):
            complexity_results = []
            for layer, probe_results in results.items():
                if layer in schedule:
                    layer_probes = {p.id: p for p in schedule[layer]}
                    for probe_id, score in probe_results.items():
                        if probe_id in layer_probes and layer_probes[probe_id].complexity == complexity:
                            complexity_results.append((layer, score))
            
            if complexity_results:
                analysis["complexity_thresholds"][complexity] = {
                    "viable_layers": [l for l, s in complexity_results if s < 0.5],
                    "struggle_layers": [l for l, s in complexity_results if s > 0.7]
                }
        
        # Detect anomalies (unexpected good/bad performance)
        for layer, probe_results in results.items():
            if layer in schedule:
                layer_probes = {p.id: p for p in schedule[layer]}
                for probe_id, score in probe_results.items():
                    if probe_id in layer_probes:
                        probe = layer_probes[probe_id]
                        # Check against expected difficulty
                        for range_key, expected_diff in probe.expected_difficulty_by_layer.items():
                            range_start, range_end = map(int, range_key.split('-'))
                            if range_start <= layer <= range_end:
                                if abs(score - expected_diff) > 0.3:  # Large deviation
                                    analysis["anomalies"].append({
                                        "layer": layer,
                                        "probe_id": probe_id,
                                        "expected": expected_diff,
                                        "actual": score,
                                        "deviation": score - expected_diff
                                    })
                                break
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Initialize library
    library = CassetteLibrary()
    
    # Simulate topology from phase 1
    sample_topology = {
        "restriction_sites": [
            {"layer": 4, "divergence_delta": 0.05},
            {"layer": 20, "divergence_delta": 0.03},
            {"layer": 36, "divergence_delta": 0.02}
        ],
        "stable_regions": [
            {"start": 5, "end": 19},
            {"start": 21, "end": 35},
            {"start": 37, "end": 50}
        ],
        "phase_boundaries": [4, 20, 36, 52]
    }
    
    # Generate probe schedule based on topology
    schedule = library.generate_probe_schedule(sample_topology)
    
    # Export for execution
    library.export_schedule(schedule, "probe_schedule.json")
    
    print(f"Generated probe schedule for {len(schedule)} layers")
    for layer, probes in sorted(schedule.items())[:5]:
        print(f"  Layer {layer}: {len(probes)} probes")
        for probe in probes[:2]:
            print(f"    - {probe.type.value}: {probe.text[:50]}...")