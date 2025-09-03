#!/usr/bin/env python3
"""
Advanced Prompt Orchestration Example

Demonstrates using all 7 prompt generation systems for comprehensive testing.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from run_rev import REVUnified
from src.orchestration.prompt_orchestrator import PromptOrchestrator
from src.orchestration.prompt_analytics import PromptAnalytics


def demonstrate_orchestration(model_path: str):
    """
    Demonstrate advanced prompt orchestration features.
    
    Args:
        model_path: Path to model directory
    """
    print("=" * 70)
    print("ADVANCED PROMPT ORCHESTRATION DEMONSTRATION")
    print("=" * 70)
    
    # Initialize REV with all features enabled
    rev = REVUnified(
        memory_limit_gb=4.0,
        enable_prompt_orchestration=True,
        enable_principled_features=True,
        debug=True
    )
    
    # 1. Show available prompt systems
    print("\n📚 Available Prompt Generation Systems:")
    print("-" * 40)
    systems = [
        ("PoT", "Proof of Thought", "Behavioral boundary probes", "30%"),
        ("KDF", "Key Derivation", "Security/adversarial testing", "20%"),
        ("Evolutionary", "Genetic Optimization", "Evolving effective prompts", "20%"),
        ("Dynamic", "Template Synthesis", "Pattern-based generation", "20%"),
        ("Hierarchical", "Taxonomical", "Category exploration", "10%"),
        ("Predictor", "Response Predictor", "Effectiveness scoring", "N/A"),
        ("Profiler", "Behavior Profiler", "Pattern analysis", "N/A")
    ]
    
    for name, full_name, purpose, weight in systems:
        print(f"  {name:15} - {full_name:20} | {purpose:30} | Weight: {weight}")
    
    # 2. Test different strategies
    print("\n🎯 Testing Different Strategies:")
    print("-" * 40)
    
    strategies = ["balanced", "adversarial", "behavioral", "comprehensive"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        
        # Generate small batch of prompts for demonstration
        orchestrator = PromptOrchestrator()
        prompts = orchestrator.generate_prompts(n=5, strategy=strategy)
        
        print(f"  Sample prompts:")
        for i, prompt in enumerate(prompts[:3], 1):
            # Truncate long prompts for display
            display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"    {i}. {display_prompt}")
    
    # 3. Run full orchestrated analysis
    print("\n🔬 Running Full Orchestrated Analysis:")
    print("-" * 40)
    
    try:
        result = rev.process_model(
            model_path,
            challenges=20,  # Use 20 for demo
            orchestration_config={
                "strategy": "balanced",
                "enable_analytics": True,
                "systems": {
                    "pot": {"weight": 0.3, "enabled": True},
                    "kdf": {"weight": 0.2, "enabled": True},
                    "evolutionary": {"weight": 0.2, "enabled": True},
                    "dynamic": {"weight": 0.2, "enabled": True},
                    "hierarchical": {"weight": 0.1, "enabled": True}
                }
            }
        )
        
        print(f"✅ Analysis completed successfully")
        print(f"   Model Family: {result['model_family']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Show prompt analytics if available
        if 'prompt_analytics' in result:
            analytics = result['prompt_analytics']
            print(f"\n📊 Prompt Analytics:")
            print(f"   Total Prompts: {analytics.get('total', 0)}")
            print(f"   Effective Prompts: {analytics.get('effective', 0)}")
            print(f"   Average Response Time: {analytics.get('avg_response_time', 0):.2f}s")
            
            if 'system_breakdown' in analytics:
                print(f"\n   System Performance:")
                for system, stats in analytics['system_breakdown'].items():
                    print(f"     {system}: {stats['count']} prompts, "
                          f"{stats['effectiveness']:.1%} effective")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    
    finally:
        rev.cleanup()


def adaptive_orchestration_example(model_path: str):
    """
    Demonstrate adaptive prompt orchestration based on model responses.
    """
    print("\n" + "=" * 70)
    print("ADAPTIVE ORCHESTRATION EXAMPLE")
    print("=" * 70)
    
    print("\nThis example shows how the system adapts its prompting strategy")
    print("based on model responses in real-time.\n")
    
    # Initialize with adaptive configuration
    rev = REVUnified(
        enable_prompt_orchestration=True,
        enable_principled_features=True
    )
    
    # Custom configuration for adaptive behavior
    adaptive_config = {
        "initial_strategy": "balanced",
        "adaptation_enabled": True,
        "adaptation_threshold": 0.7,  # Switch strategy if effectiveness < 70%
        "fallback_strategy": "comprehensive",
        "min_samples_before_adaptation": 5
    }
    
    print("Configuration:")
    for key, value in adaptive_config.items():
        print(f"  {key}: {value}")
    
    try:
        print("\n🔄 Starting adaptive analysis...")
        
        # Process with adaptive configuration
        result = rev.process_model(
            model_path,
            challenges=30,
            orchestration_config=adaptive_config
        )
        
        print("\n📈 Adaptation Results:")
        if 'adaptations' in result:
            for adaptation in result['adaptations']:
                print(f"  Step {adaptation['step']}: "
                      f"Switched from {adaptation['from']} to {adaptation['to']} "
                      f"(effectiveness was {adaptation['trigger']:.1%})")
        
        print(f"\nFinal Confidence: {result['confidence']:.2%}")
        
    except Exception as e:
        print(f"❌ Adaptive analysis failed: {e}")
    
    finally:
        rev.cleanup()


def evolutionary_prompt_example():
    """
    Demonstrate evolutionary prompt optimization.
    """
    print("\n" + "=" * 70)
    print("EVOLUTIONARY PROMPT OPTIMIZATION")
    print("=" * 70)
    
    from src.orchestration.evolutionary_prompts import EvolutionaryPromptGenerator
    
    generator = EvolutionaryPromptGenerator(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        num_generations=5
    )
    
    print("\nEvolutionary Parameters:")
    print(f"  Population Size: {generator.population_size}")
    print(f"  Generations: {generator.num_generations}")
    print(f"  Mutation Rate: {generator.mutation_rate}")
    print(f"  Crossover Rate: {generator.crossover_rate}")
    
    print("\n🧬 Evolving prompts through generations...")
    
    # Initial population
    print("\nGeneration 0 (Initial):")
    initial_prompts = generator.generate(n=3)
    for i, prompt in enumerate(initial_prompts, 1):
        print(f"  {i}. {prompt['text'][:80]}...")
    
    # Simulate evolution with mock fitness scores
    for gen in range(1, 4):
        print(f"\nGeneration {gen}:")
        
        # Mock fitness evaluation
        fitness_scores = [0.5 + i * 0.1 for i in range(len(generator.population))]
        generator.update_fitness(fitness_scores)
        
        # Evolve
        generator.evolve()
        
        # Show best prompts
        evolved = generator.generate(n=2)
        for i, prompt in enumerate(evolved, 1):
            print(f"  {i}. {prompt['text'][:80]}...")
            print(f"     Fitness: {prompt.get('expected_fitness', 0):.2f}")


def main():
    """Main function to run all examples."""
    
    # Check for model path
    model_path = "/Users/rohanvinaik/LLM_models/pythia-70m"
    
    if not Path(model_path).exists():
        print("⚠️  Model not found at default path.")
        print(f"   Looking for: {model_path}")
        print("\n   Please update the model_path variable with your model location.")
        
        # Try to find alternative
        alt_paths = [
            Path.home() / "LLM_models",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("/opt/models"),
            Path("/mnt/models")
        ]
        
        for alt in alt_paths:
            if alt.exists():
                configs = list(alt.glob("*/config.json")) + \
                         list(alt.glob("*/snapshots/*/config.json"))
                if configs:
                    model_path = str(configs[0].parent)
                    print(f"\n✅ Found model at: {model_path}")
                    break
        else:
            print("\n❌ No models found. Please specify a valid model path.")
            return
    
    # Run demonstrations
    demonstrate_orchestration(model_path)
    adaptive_orchestration_example(model_path)
    evolutionary_prompt_example()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()