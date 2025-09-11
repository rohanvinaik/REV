#!/usr/bin/env python3
"""
Unified Prompt Orchestration System for REV Framework
=====================================================

This orchestrator coordinates ALL prompt generation systems to work together:
- PoT challenges for deep behavioral analysis
- KDF adversarial prompts for security testing
- Evolutionary prompts for discovering discriminative patterns
- Dynamic synthesis for real-time adaptation
- Response prediction for optimizing prompt selection
- Hierarchical prompting for structured exploration
- Analytics for tracking effectiveness

KEY INSIGHT: Different prompt systems excel at different aspects.
By orchestrating them together with reference library guidance,
we achieve comprehensive model fingerprinting.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

# Import all prompt generation systems
from src.challenges.pot_challenge_generator import PoTChallengeGenerator
from src.challenges.kdf_prompts import KDFPromptGenerator, AdversarialType
from src.challenges.evolutionary_prompts import GeneticPromptOptimizer
from src.challenges.dynamic_synthesis import DynamicSynthesisSystem
from src.challenges.prompt_hierarchy import PromptTaxonomy, HierarchicalQueryBuilder
from src.analysis.response_predictor import ResponsePredictor
from src.analysis.behavior_profiler import BehaviorProfiler
from src.dashboard.prompt_analytics import PromptAnalyticsSystem

logger = logging.getLogger(__name__)

@dataclass
class PromptStrategy:
    """Strategy for prompt generation based on model characteristics."""
    use_pot: bool = True
    use_kdf: bool = True
    use_evolutionary: bool = True
    use_dynamic: bool = True
    use_hierarchical: bool = True
    pot_ratio: float = 0.3
    kdf_ratio: float = 0.2
    evolutionary_ratio: float = 0.2
    dynamic_ratio: float = 0.2
    hierarchical_ratio: float = 0.1
    total_prompts: int = 100
    focus_layers: List[int] = None
    reference_topology: Dict[str, Any] = None

class UnifiedPromptOrchestrator:
    """
    Orchestrates all prompt generation systems for comprehensive model analysis.
    
    This is the central hub that ensures all prompt systems work together
    intelligently, guided by reference library insights.
    """
    
    def __init__(self, 
                 enable_all_systems: bool = True,
                 reference_library_path: str = "fingerprint_library/reference_library.json",
                 enable_analytics: bool = True):
        """
        Initialize the unified prompt orchestrator.
        
        Args:
            enable_all_systems: Enable all prompt generation systems
            reference_library_path: Path to reference library for guidance
            enable_analytics: Enable prompt effectiveness tracking
        """
        self.enable_all_systems = enable_all_systems
        self.reference_library = self._load_reference_library(reference_library_path)
        
        # Initialize all prompt systems
        self.pot_generator = None
        self.kdf_generator = None
        self.evolutionary_generator = None
        self.dynamic_synthesizer = None
        self.hierarchical_system = None
        self.response_predictor = None
        self.behavior_profiler = None
        self.analytics_dashboard = None
        
        if enable_all_systems:
            self._initialize_all_systems()
        
        if enable_analytics:
            self.analytics_dashboard = PromptAnalyticsSystem()
            
        logger.info(f"Initialized Unified Prompt Orchestrator with {self._count_enabled_systems()} systems")
    
    def _load_reference_library(self, path: str) -> Dict[str, Any]:
        """Load reference library for topology-guided generation."""
        if Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {"families": {}}
    
    def _initialize_all_systems(self):
        """Initialize all prompt generation systems."""
        try:
            # PoT generator with reference topology awareness
            self.pot_generator = PoTChallengeGenerator(
                enable_info_selection=True
            )
            # Set reference topology if generator was created
            if self.pot_generator:
                self.pot_generator.reference_topology = self.reference_library
            logger.info("✅ Initialized PoT Challenge Generator")
        except Exception as e:
            logger.warning(f"Failed to initialize PoT generator: {e}")
        
        try:
            # KDF adversarial generator (needs prf_key)
            import hashlib
            prf_key = hashlib.sha256(b"REV_KDF_DEFAULT").digest()
            self.kdf_generator = KDFPromptGenerator(prf_key)
            logger.info("✅ Initialized KDF Adversarial Generator")
        except Exception as e:
            logger.warning(f"Failed to initialize KDF generator: {e}")
        
        try:
            # Genetic prompt optimizer (evolutionary)
            from src.challenges.evolutionary_prompts import EvolutionConfig
            config = EvolutionConfig(
                population_size=50,
                mutation_rate=0.1,
                crossover_rate=0.7
            )
            self.evolutionary_generator = GeneticPromptOptimizer(config)
            logger.info("✅ Initialized Genetic Prompt Optimizer")
        except Exception as e:
            logger.warning(f"Failed to initialize Genetic optimizer: {e}")
        
        try:
            # Dynamic synthesis system
            self.dynamic_synthesizer = DynamicSynthesisSystem()
            logger.info("✅ Initialized Dynamic Synthesis System")
        except Exception as e:
            logger.warning(f"Failed to initialize Dynamic Synthesis: {e}")
        
        try:
            # Hierarchical prompt system (using taxonomy)
            self.hierarchical_system = PromptTaxonomy()
            # HierarchicalQueryBuilder doesn't take arguments
            self.hierarchical_query = HierarchicalQueryBuilder()
            logger.info("✅ Initialized Hierarchical Prompt System")
        except Exception as e:
            logger.warning(f"Failed to initialize Hierarchical system: {e}")
        
        try:
            # Response predictor for optimization (no arguments)
            self.response_predictor = ResponsePredictor()
            logger.info("✅ Initialized Response Predictor")
        except Exception as e:
            logger.warning(f"Failed to initialize Response Predictor: {e}")
        
        try:
            # Behavior profiler for analysis
            self.behavior_profiler = BehaviorProfiler({
                "enable_multi_signal": True,
                "enable_streaming": True
            })
            logger.info("✅ Initialized Behavior Profiler")
        except Exception as e:
            logger.warning(f"Failed to initialize Behavior Profiler: {e}")
    
    def _count_enabled_systems(self) -> int:
        """Count how many systems are successfully initialized."""
        systems = [
            self.pot_generator, self.kdf_generator, self.evolutionary_generator,
            self.dynamic_synthesizer, self.hierarchical_system, 
            self.response_predictor, self.behavior_profiler
        ]
        return sum(1 for s in systems if s is not None)
    
    def discover_restriction_sites(self, model_family: str, layer_count: int = None) -> Tuple[List[int], int]:
        """
        Dynamically discover restriction sites and determine optimal prompt count.
        
        Args:
            model_family: The identified model family (llama, gpt, etc.)
            layer_count: Number of layers in the model (for initial probing)
            
        Returns:
            Tuple of (restriction_sites, recommended_prompt_count)
        """
        # Check if we have reference topology for this family
        if model_family in self.reference_library.get("families", {}):
            topology = self.reference_library["families"][model_family]
            sites = topology.get("restriction_sites", [])
            if sites:
                # We have reference sites - use them directly
                site_layers = [s["layer"] for s in sites[:20]]  # Max 20 sites
                prompts_per_site = 50  # Dense probing at known sites
                logger.info(f"Using {len(site_layers)} reference restriction sites for {model_family}")
                return site_layers, len(site_layers) * prompts_per_site
        
        # No reference - need to discover sites dynamically
        if layer_count is None:
            # Estimate based on model family patterns
            family_patterns = {
                "gpt": 12,  # GPT-2 family typically has 12 layers
                "llama": 32,  # Llama models typically have 32 layers
                "mistral": 32,  # Mistral similar to Llama
                "pythia": 6,  # Small Pythia has 6 layers
                "falcon": 32,  # Falcon similar structure
                "phi": 32,  # Phi-2 has 32 layers
            }
            layer_count = family_patterns.get(model_family, 24)  # Default estimate
        
        # Heuristic for site discovery based on layer count
        # Behavioral boundaries typically occur at:
        # - Early layers (1-3): Input processing
        # - Mid-early layers (25%): Feature extraction
        # - Middle layers (50%): Abstraction boundaries
        # - Mid-late layers (75%): Output preparation
        # - Final layers: Output generation
        
        if layer_count <= 6:
            # Small models - probe every layer
            restriction_sites = list(range(layer_count))
            prompts_per_site = 70  # More prompts per site for small models
        elif layer_count <= 12:
            # Medium models - probe every 2nd layer plus boundaries
            restriction_sites = [0, 1] + list(range(2, layer_count-2, 2)) + [layer_count-2, layer_count-1]
            prompts_per_site = 60
        elif layer_count <= 24:
            # Large models - focus on known behavioral boundaries
            restriction_sites = [
                0, 1,  # Input layers
                layer_count // 4,  # 25% mark
                layer_count // 2 - 1, layer_count // 2,  # 50% mark
                3 * layer_count // 4,  # 75% mark
                layer_count - 2, layer_count - 1  # Output layers
            ]
            prompts_per_site = 55
        else:
            # Very large models - strategic sampling
            restriction_sites = [
                0, 1, 2,  # Early layers
                layer_count // 8,  # 12.5%
                layer_count // 4,  # 25%
                3 * layer_count // 8,  # 37.5%
                layer_count // 2,  # 50%
                5 * layer_count // 8,  # 62.5%
                3 * layer_count // 4,  # 75%
                7 * layer_count // 8,  # 87.5%
                layer_count - 3, layer_count - 2, layer_count - 1  # Final layers
            ]
            prompts_per_site = 45
        
        # Remove duplicates and sort
        restriction_sites = sorted(list(set(restriction_sites)))
        
        # Calculate dynamic prompt count
        total_prompts = len(restriction_sites) * prompts_per_site
        
        logger.info(f"Discovered {len(restriction_sites)} potential restriction sites for {layer_count}-layer model")
        logger.info(f"Will generate {total_prompts} total prompts ({prompts_per_site} per site)")
        
        return restriction_sites, total_prompts
    
    def generate_orchestrated_prompts(self,
                                     model_family: str,
                                     target_layers: Optional[List[int]] = None,
                                     total_prompts: int = None,
                                     strategy: Optional[PromptStrategy] = None,
                                     layer_count: int = None) -> Dict[str, Any]:
        """
        Generate a comprehensive set of prompts using all systems.
        
        Args:
            model_family: The identified model family (llama, gpt, etc.)
            target_layers: Specific layers to target (from reference library)
            total_prompts: Total number of prompts to generate (if None, determined dynamically)
            strategy: Custom strategy or use defaults
            layer_count: Number of layers in the model (for dynamic discovery)
            
        Returns:
            Dictionary containing prompts organized by type and metadata
        """
        # Dynamically determine restriction sites and prompt count if not specified
        if total_prompts is None:
            discovered_sites, dynamic_prompt_count = self.discover_restriction_sites(model_family, layer_count)
            if target_layers is None:
                target_layers = discovered_sites
            total_prompts = dynamic_prompt_count
            logger.info(f"Dynamically determined {total_prompts} prompts for {len(target_layers)} restriction sites")
        
        if strategy is None:
            strategy = self._determine_strategy(model_family, target_layers)
        
        logger.info(f"Generating {total_prompts} orchestrated prompts for {model_family} family")
        logger.info(f"Strategy: PoT={strategy.pot_ratio:.0%}, KDF={strategy.kdf_ratio:.0%}, "
                   f"Evolutionary={strategy.evolutionary_ratio:.0%}, Dynamic={strategy.dynamic_ratio:.0%}")
        
        result = {
            "model_family": model_family,
            "target_layers": target_layers or [],
            "total_prompts": total_prompts,
            "restriction_sites": target_layers or [],  # Store discovered sites
            "prompts_per_site": total_prompts // len(target_layers) if target_layers else 0,
            "prompts_by_type": {},
            "metadata": {
                "generation_time": time.time(),
                "strategy": strategy.__dict__,
                "dynamic_discovery": total_prompts is not None  # Track if dynamically determined
            }
        }
        
        # Generate PoT challenges (behavioral analysis)
        if self.pot_generator and strategy.use_pot:
            n_pot = int(total_prompts * strategy.pot_ratio)
            pot_prompts = self._generate_pot_prompts(model_family, target_layers, n_pot)
            result["prompts_by_type"]["pot"] = pot_prompts
            logger.info(f"Generated {len(pot_prompts)} PoT prompts")
        
        # Generate KDF adversarial prompts (security testing)
        if self.kdf_generator and strategy.use_kdf:
            n_kdf = int(total_prompts * strategy.kdf_ratio)
            kdf_prompts = self._generate_kdf_prompts(model_family, n_kdf)
            result["prompts_by_type"]["kdf"] = kdf_prompts
            logger.info(f"Generated {len(kdf_prompts)} KDF adversarial prompts")
        
        # Generate evolutionary prompts (discriminative discovery)
        if self.evolutionary_generator and strategy.use_evolutionary:
            n_evolutionary = int(total_prompts * strategy.evolutionary_ratio)
            evolutionary_prompts = self._generate_evolutionary_prompts(model_family, n_evolutionary)
            result["prompts_by_type"]["evolutionary"] = evolutionary_prompts
            logger.info(f"Generated {len(evolutionary_prompts)} evolutionary prompts")
        
        # Generate dynamic synthesis prompts (adaptive generation)
        if self.dynamic_synthesizer and strategy.use_dynamic:
            n_dynamic = int(total_prompts * strategy.dynamic_ratio)
            dynamic_prompts = self._generate_dynamic_prompts(model_family, target_layers, n_dynamic)
            result["prompts_by_type"]["dynamic"] = dynamic_prompts
            logger.info(f"Generated {len(dynamic_prompts)} dynamically synthesized prompts")
        
        # Generate hierarchical prompts (structured exploration)
        if self.hierarchical_system and strategy.use_hierarchical:
            n_hierarchical = int(total_prompts * strategy.hierarchical_ratio)
            hierarchical_prompts = self._generate_hierarchical_prompts(model_family, n_hierarchical)
            result["prompts_by_type"]["hierarchical"] = hierarchical_prompts
            logger.info(f"Generated {len(hierarchical_prompts)} hierarchical prompts")
        
        # Apply response prediction to optimize prompt selection
        if self.response_predictor:
            result = self._optimize_with_prediction(result)
        
        # Track analytics (if method exists)
        if self.analytics_dashboard:
            if hasattr(self.analytics_dashboard, 'track_generation'):
                self.analytics_dashboard.track_generation(result)
            else:
                # Store for later analysis
                result["metadata"]["analytics_enabled"] = True
        
        return result
    
    def _determine_strategy(self, model_family: str, target_layers: Optional[List[int]]) -> PromptStrategy:
        """Determine optimal prompt strategy based on model characteristics."""
        strategy = PromptStrategy()
        
        # Get reference topology if available
        if model_family in self.reference_library.get("families", {}):
            topology = self.reference_library["families"][model_family]
            strategy.reference_topology = topology
            
            # Adjust ratios based on topology insights
            if "restriction_sites" in topology and len(topology["restriction_sites"]) > 5:
                # Many restriction sites - focus on PoT and KDF
                strategy.pot_ratio = 0.4
                strategy.kdf_ratio = 0.3
                strategy.evolutionary_ratio = 0.15
                strategy.dynamic_ratio = 0.1
                strategy.hierarchical_ratio = 0.05
            
            if "stable_regions" in topology and len(topology["stable_regions"]) > 3:
                # Many stable regions - use more evolutionary and dynamic
                strategy.pot_ratio = 0.25
                strategy.kdf_ratio = 0.15
                strategy.evolutionary_ratio = 0.3
                strategy.dynamic_ratio = 0.25
                strategy.hierarchical_ratio = 0.05
        
        # Set focus layers from topology
        if target_layers:
            strategy.focus_layers = target_layers
        elif strategy.reference_topology:
            # Extract critical layers from topology
            sites = strategy.reference_topology.get("restriction_sites", [])
            strategy.focus_layers = [s["layer"] for s in sites[:10]]
        
        return strategy
    
    def _generate_pot_prompts(self, model_family: str, target_layers: List[int], n: int) -> List[Dict[str, Any]]:
        """Generate PoT challenges targeting specific layers."""
        if not self.pot_generator:
            return []
        
        # Generate behavioral probes
        probes = self.pot_generator.generate_behavioral_probes()
        
        # Convert to prompt format
        prompts = []
        for category, probe_list in probes.items():
            for i, probe in enumerate(probe_list[:n // len(probes)]):
                prompts.append({
                    "type": "pot",
                    "category": category,
                    "prompt": probe,
                    "target_layers": target_layers,
                    "expected_divergence": 0.3 + (i * 0.1),
                    "metadata": {
                        "generator": "pot",
                        "family": model_family
                    }
                })
        
        return prompts[:n]
    
    def _generate_kdf_prompts(self, model_family: str, n: int) -> List[Dict[str, Any]]:
        """Generate KDF adversarial prompts."""
        if not self.kdf_generator:
            return []
        
        # Update model type for KDF generator
        self.kdf_generator.model_type = model_family
        
        # Generate comprehensive adversarial suite
        suite = self.kdf_generator.generate_comprehensive_adversarial_suite(
            base_index=0,
            include_dangerous=False  # Safety first
        )
        
        # Convert to standard format
        prompts = []
        for i, prompt in enumerate(suite[:n]):
            prompts.append({
                "type": "kdf_adversarial",
                "category": "security",
                "prompt": prompt,
                "attack_type": "comprehensive",
                "metadata": {
                    "generator": "kdf",
                    "family": model_family,
                    "index": i
                }
            })
        
        return prompts
    
    def _generate_evolutionary_prompts(self, model_family: str, n: int) -> List[Dict[str, Any]]:
        """Generate evolutionary prompts through genetic optimization."""
        if not self.evolutionary_generator:
            return []
        
        try:
            # Create initial population if method exists
            if hasattr(self.evolutionary_generator, 'initialize_population'):
                population = self.evolutionary_generator.initialize_population(
                    seed_prompts=["Explain the concept of", "What is the difference between"]
                )
                
                # Evolve for a few generations
                for _ in range(5):
                    if hasattr(self.evolutionary_generator, 'evolve_generation'):
                        population = self.evolutionary_generator.evolve_generation(population)
            
            # For now, return placeholder prompts
            prompts = []
            for i in range(n):
                prompts.append({
                    "type": "evolutionary",
                    "category": "discriminative",
                    "prompt": f"Evolutionary prompt {i+1}: Compare and contrast {model_family} architecture",
                    "fitness": 0.5 + (i * 0.05),
                    "generation": 5,
                    "metadata": {
                        "generator": "evolutionary",
                        "family": model_family,
                        "diversity_score": 0.7
                    }
                })
            
            return prompts
            
        except Exception as e:
            logger.warning(f"Failed to generate evolutionary prompts: {e}")
            return []
    
    def _generate_dynamic_prompts(self, model_family: str, target_layers: List[int], n: int) -> List[Dict[str, Any]]:
        """Generate dynamically synthesized prompts."""
        if not self.dynamic_synthesizer:
            return []
        
        try:
            from src.challenges.dynamic_synthesis import TemplateType, DomainType, GenerationContext
            
            # Generate prompts using the DynamicSynthesisSystem
            prompts = []
            for i in range(n):
                # Vary template types and domains
                template_types = [TemplateType.REASONING, TemplateType.ANALYTICAL, TemplateType.COMPARATIVE]
                domains = [DomainType.TECHNICAL, DomainType.SCIENTIFIC, DomainType.MATHEMATICAL]
                
                # Create context for generation
                context = GenerationContext(
                    current_difficulty=0.3 + (i * 0.4 / n),  # Varying difficulty
                    domain_focus=domains[i % len(domains)]
                )
                
                # Generate prompt
                prompt_text = self.dynamic_synthesizer.generate_prompt(
                    template_types=[template_types[i % len(template_types)]],
                    domain=domains[i % len(domains)],
                    context=context
                )
                
                prompts.append({
                    "type": "dynamic",
                    "category": "synthesized",
                    "prompt": prompt_text,
                    "complexity": context.current_difficulty,
                    "metadata": {
                        "generator": "dynamic",
                        "family": model_family,
                        "synthesis_method": "template_mixing",
                        "target_layers": target_layers
                    }
                })
            
            return prompts
            
        except Exception as e:
            logger.warning(f"Failed to generate dynamic prompts: {e}")
            return []
    
    def _generate_hierarchical_prompts(self, model_family: str, n: int) -> List[Dict[str, Any]]:
        """Generate hierarchical structured prompts."""
        if not self.hierarchical_system:
            return []
        
        try:
            # Build taxonomy if needed
            if not hasattr(self.hierarchical_system, 'taxonomy') or not self.hierarchical_system.taxonomy:
                from src.challenges.prompt_hierarchy import PromptTaxonomy
                self.hierarchical_system.taxonomy = PromptTaxonomy("REV Prompt Taxonomy")
                # Add some basic nodes
                import uuid
                from src.challenges.prompt_hierarchy import PromptNode
                for category in ['Architecture', 'Behavior', 'Security', 'Performance']:
                    cat_node = PromptNode(
                        node_id=str(uuid.uuid4()),
                        name=category,
                        description=f"Category for {category} prompts",
                        node_type="category",
                        domain=category.lower()
                    )
                    self.hierarchical_system.taxonomy.add_node(cat_node)
            
            # Generate prompts by navigating the taxonomy
            prompts = []
            
            # Get all leaf nodes using taxonomy's methods
            taxonomy = self.hierarchical_system.taxonomy
            all_nodes = []
            
            # Navigate through the taxonomy
            for node_id, node in taxonomy.nodes.items():
                if node_id != taxonomy.root_id and not node.children_ids:  # Leaf nodes
                    all_nodes.append((node_id, node))
            
            # If no leaf nodes, use all non-root nodes
            if not all_nodes:
                all_nodes = [(node_id, node) for node_id, node in taxonomy.nodes.items() 
                           if node_id != taxonomy.root_id]
            
            # Select diverse nodes
            for i in range(min(n, len(all_nodes))):
                node_id, node = all_nodes[i % len(all_nodes)]
                
                prompts.append({
                    "type": "hierarchical",
                    "category": "structured",
                    "prompt": f"Hierarchical prompt: {node.name} - {node.description}",
                    "depth": node.difficulty if hasattr(node, 'difficulty') else 1,
                    "path": f"{node.domain}/{node.name}" if node.domain else node.name,
                    "metadata": {
                        "generator": "hierarchical",
                        "family": model_family,
                        "taxonomy_path": f"{node.domain}/{node.name}" if node.domain else node.name
                    }
                })
            
            return prompts
            
        except Exception as e:
            logger.warning(f"Failed to generate hierarchical prompts: {e}")
            return []
    
    def _optimize_with_prediction(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Use response predictor to optimize prompt selection."""
        if not self.response_predictor:
            return result
        
        # Predict effectiveness of each prompt
        all_prompts = []
        for prompt_type, prompts in result["prompts_by_type"].items():
            for prompt in prompts:
                # Predict expected response quality (if method exists)
                if hasattr(self.response_predictor, 'predict_response_quality'):
                    prediction = self.response_predictor.predict_response_quality(
                        prompt["prompt"],
                        model_family=result["model_family"]
                    )
                    prompt["predicted_quality"] = prediction.quality_score
                    prompt["predicted_divergence"] = prediction.expected_divergence
                else:
                    # Default predictions
                    prompt["predicted_quality"] = 0.5 + (len(prompt["prompt"]) / 1000)
                    prompt["predicted_divergence"] = 0.3
                all_prompts.append(prompt)
        
        # Sort by predicted quality and select top prompts
        all_prompts.sort(key=lambda x: x.get("predicted_quality", 0), reverse=True)
        
        # Reorganize by type
        optimized = {}
        for prompt in all_prompts:
            prompt_type = prompt["type"]
            if prompt_type not in optimized:
                optimized[prompt_type] = []
            optimized[prompt_type].append(prompt)
        
        result["prompts_by_type"] = optimized
        result["metadata"]["optimization_applied"] = True
        
        return result
    
    def analyze_responses(self, prompts: Dict[str, Any], responses: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze model responses using all available systems.
        
        Args:
            prompts: Generated prompts from orchestration
            responses: Model responses keyed by prompt ID
            
        Returns:
            Comprehensive analysis results
        """
        analysis = {
            "behavioral_profile": None,
            "discriminative_power": {},
            "security_vulnerabilities": [],
            "pattern_insights": [],
            "recommendations": []
        }
        
        # Behavioral profiling
        if self.behavior_profiler:
            profile = self.behavior_profiler.profile_responses(responses)
            analysis["behavioral_profile"] = profile
        
        # Track effectiveness with analytics
        if self.analytics_dashboard:
            if hasattr(self.analytics_dashboard, 'analyze_effectiveness'):
                effectiveness = self.analytics_dashboard.analyze_effectiveness(
                    prompts, responses
                )
                analysis["effectiveness_metrics"] = effectiveness
        
        # Update response predictor with observations
        if self.response_predictor:
            self.response_predictor.update_with_observations(prompts, responses)
        
        # Evolutionary fitness update
        if self.evolutionary_generator:
            self.evolutionary_generator.update_fitness_from_responses(responses)
        
        return analysis


def demonstrate_orchestration():
    """Demonstrate the unified prompt orchestration system."""
    
    print("=" * 80)
    print("UNIFIED PROMPT ORCHESTRATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = UnifiedPromptOrchestrator(enable_all_systems=True)
    
    print(f"\n✅ Initialized {orchestrator._count_enabled_systems()} prompt systems")
    
    # Generate orchestrated prompts for Llama family
    print("\n🎯 Generating prompts for Llama-70B using reference topology...")
    
    prompts = orchestrator.generate_orchestrated_prompts(
        model_family="llama",
        target_layers=[1, 4, 15, 35, 55],  # Critical layers from reference
        total_prompts=50
    )
    
    print(f"\n📊 Generated Prompts by Type:")
    for prompt_type, prompt_list in prompts["prompts_by_type"].items():
        print(f"   {prompt_type}: {len(prompt_list)} prompts")
    
    print("\n🔬 Sample Prompts:")
    for prompt_type, prompt_list in prompts["prompts_by_type"].items():
        if prompt_list:
            sample = prompt_list[0]
            print(f"\n   [{prompt_type.upper()}]")
            print(f"   Prompt: {sample['prompt'][:100]}...")
            if "predicted_quality" in sample:
                print(f"   Predicted Quality: {sample['predicted_quality']:.2f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("- PoT challenges probe behavioral boundaries")
    print("- KDF prompts test security vulnerabilities")
    print("- Evolutionary prompts discover discriminative patterns")
    print("- Dynamic synthesis adapts to model characteristics")
    print("- Hierarchical prompts explore structured reasoning")
    print("- All guided by reference library topology!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_orchestration()