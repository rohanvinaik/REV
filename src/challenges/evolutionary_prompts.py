"""
Evolutionary Prompt Generation System for REV Framework
Uses genetic algorithms to discover discriminative prompts for model verification.
"""

import hashlib
import json
import time
import random
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable, Set
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import heapq
import copy


class MutationType(Enum):
    """Types of mutations for genetic operations"""
    TOKEN_SUBSTITUTION = "token_substitution"
    TOKEN_INSERTION = "token_insertion"
    TOKEN_DELETION = "token_deletion"
    PHRASE_SWAP = "phrase_swap"
    SYNTAX_TRANSFORM = "syntax_transform"
    SEMANTIC_SHIFT = "semantic_shift"
    LENGTH_VARIATION = "length_variation"


class FitnessMetric(Enum):
    """Fitness metrics for prompt evaluation"""
    DISCRIMINATION = "discrimination"
    COHERENCE = "coherence"
    DIVERSITY = "diversity"
    COMPLEXITY = "complexity"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"


class SelectionStrategy(Enum):
    """Selection strategies for genetic algorithm"""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITISM = "elitism"
    CROWDING = "crowding"


@dataclass
class Gene:
    """Represents a single token or phrase in a prompt"""
    content: str
    position: int
    token_type: str  # word, punctuation, number, special
    semantic_role: Optional[str] = None  # subject, verb, object, modifier
    mutation_rate: float = 0.1
    locked: bool = False  # Prevents mutation if True
    
    def copy(self) -> 'Gene':
        """Create a copy of this gene"""
        return Gene(
            content=self.content,
            position=self.position,
            token_type=self.token_type,
            semantic_role=self.semantic_role,
            mutation_rate=self.mutation_rate,
            locked=self.locked
        )


@dataclass
class Chromosome:
    """Represents a complete prompt with genetic information"""
    genes: List[Gene]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    model_responses: Dict[str, Any] = field(default_factory=dict)
    diversity_score: float = 0.0
    coherence_score: float = 0.0
    discrimination_score: float = 0.0
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    
    def to_prompt(self) -> str:
        """Convert chromosome to prompt string"""
        return " ".join(g.content for g in self.genes)
    
    def copy(self) -> 'Chromosome':
        """Deep copy of chromosome"""
        return Chromosome(
            genes=[Gene(g.content, g.position, g.token_type, g.semantic_role, 
                       g.mutation_rate, g.locked) for g in self.genes],
            fitness=self.fitness,
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            mutation_history=self.mutation_history.copy(),
            model_responses=self.model_responses.copy(),
            diversity_score=self.diversity_score,
            coherence_score=self.coherence_score,
            discrimination_score=self.discrimination_score
        )


@dataclass
class Population:
    """Represents a population of prompts"""
    chromosomes: List[Chromosome]
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    diversity: float = 0.0
    elite_archive: List[Chromosome] = field(default_factory=list)
    species: Dict[str, List[Chromosome]] = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm"""
    population_size: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 5
    max_generations: int = 100
    target_fitness: float = 0.95
    diversity_weight: float = 0.2
    coherence_weight: float = 0.3
    discrimination_weight: float = 0.5
    niching_threshold: float = 0.3
    archive_size: int = 50
    exploration_rate: float = 0.1
    min_prompt_length: int = 5
    max_prompt_length: int = 100
    semantic_preservation: bool = True
    use_reinforcement: bool = True
    parallel_evaluations: int = 10


class TokenVocabulary:
    """Manages vocabulary for token operations"""
    
    def __init__(self):
        self.common_words = [
            "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "should", "could", "may",
            "what", "when", "where", "who", "why", "how", "which", "that", "this",
            "model", "AI", "system", "response", "explain", "describe", "analyze",
            "generate", "create", "write", "solve", "calculate", "predict", "classify"
        ]
        
        self.technical_terms = [
            "algorithm", "neural", "network", "transformer", "attention", "layer",
            "parameter", "token", "embedding", "vector", "matrix", "gradient",
            "optimization", "training", "inference", "architecture", "dataset"
        ]
        
        self.discriminative_phrases = [
            "step by step", "in detail", "specifically", "exactly", "precisely",
            "your opinion", "you think", "you believe", "you understand",
            "in your training", "your capabilities", "your limitations"
        ]
        
        self.question_starters = [
            "What", "How", "Why", "When", "Where", "Who", "Which",
            "Can you", "Could you", "Would you", "Should you",
            "Explain", "Describe", "Analyze", "Compare", "Evaluate"
        ]
        
        self.connectives = [
            "and", "or", "but", "however", "therefore", "thus", "hence",
            "moreover", "furthermore", "additionally", "alternatively"
        ]
        
    def get_random_token(self, token_type: str = "word") -> str:
        """Get random token of specified type"""
        if token_type == "word":
            return random.choice(self.common_words + self.technical_terms)
        elif token_type == "technical":
            return random.choice(self.technical_terms)
        elif token_type == "discriminative":
            return random.choice(self.discriminative_phrases)
        elif token_type == "question":
            return random.choice(self.question_starters)
        elif token_type == "connective":
            return random.choice(self.connectives)
        else:
            return random.choice(self.common_words)
    
    def get_synonym(self, word: str) -> str:
        """Get synonym for word (simplified version)"""
        synonyms = {
            "explain": ["describe", "clarify", "elaborate", "detail"],
            "create": ["generate", "produce", "make", "construct"],
            "analyze": ["examine", "evaluate", "assess", "investigate"],
            "think": ["believe", "consider", "reason", "deduce"],
            "model": ["system", "AI", "assistant", "algorithm"],
            "response": ["answer", "reply", "output", "result"]
        }
        
        word_lower = word.lower()
        if word_lower in synonyms:
            return random.choice(synonyms[word_lower])
        return word


class GeneticPromptOptimizer:
    """Main class for evolutionary prompt optimization"""
    
    def __init__(self, config: EvolutionConfig, seed: Optional[int] = None):
        """Initialize genetic optimizer"""
        self.config = config
        self.vocabulary = TokenVocabulary()
        self.generation = 0
        self.population = None
        self.elite_archive = []
        self.fitness_history = []
        self.diversity_history = []
        self.mutation_operators = self._init_mutation_operators()
        self.rl_memory = deque(maxlen=1000)  # For reinforcement learning
        self.contextual_bandit = self._init_contextual_bandit()
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def _init_mutation_operators(self) -> Dict[MutationType, Callable]:
        """Initialize mutation operator functions"""
        return {
            MutationType.TOKEN_SUBSTITUTION: self._mutate_substitution,
            MutationType.TOKEN_INSERTION: self._mutate_insertion,
            MutationType.TOKEN_DELETION: self._mutate_deletion,
            MutationType.PHRASE_SWAP: self._mutate_phrase_swap,
            MutationType.SYNTAX_TRANSFORM: self._mutate_syntax_transform,
            MutationType.SEMANTIC_SHIFT: self._mutate_semantic_shift,
            MutationType.LENGTH_VARIATION: self._mutate_length_variation
        }
    
    def _init_contextual_bandit(self) -> Dict[str, Any]:
        """Initialize contextual bandit for adaptive generation"""
        return {
            "arms": list(MutationType),
            "rewards": defaultdict(list),
            "counts": defaultdict(int),
            "ucb_values": defaultdict(float),
            "epsilon": self.config.exploration_rate
        }
    
    def create_initial_population(self, seed_prompts: Optional[List[str]] = None) -> Population:
        """Create initial population of prompts"""
        chromosomes = []
        
        # Default seed prompts if none provided
        if not seed_prompts:
            seed_prompts = [
                "Explain how you process information step by step.",
                "What are your core capabilities and limitations?",
                "Describe your training process in detail.",
                "How do you generate responses to questions?",
                "What makes you different from other AI models?",
                "Analyze this problem using your reasoning process.",
                "Can you identify patterns in your own behavior?",
                "What is your understanding of consciousness?",
                "How do you handle ambiguous instructions?",
                "Describe your model architecture if you know it."
            ]
        
        # Create chromosomes from seed prompts
        for i, prompt in enumerate(seed_prompts[:self.config.population_size // 2]):
            chromosome = self._prompt_to_chromosome(prompt, generation=0)
            chromosomes.append(chromosome)
        
        # Generate random variations to fill population
        while len(chromosomes) < self.config.population_size:
            base_prompt = random.choice(seed_prompts)
            mutated = self._prompt_to_chromosome(base_prompt, generation=0)
            
            # Apply random mutations
            for _ in range(random.randint(1, 3)):
                mutation_type = random.choice(list(MutationType))
                mutated = self.mutate(mutated, mutation_type)
            
            chromosomes.append(mutated)
        
        self.population = Population(
            chromosomes=chromosomes,
            generation=0
        )
        
        return self.population
    
    def _prompt_to_chromosome(self, prompt: str, generation: int = 0) -> Chromosome:
        """Convert prompt string to chromosome"""
        tokens = prompt.split()
        genes = []
        
        for i, token in enumerate(tokens):
            # Determine token type
            if token.lower() in self.vocabulary.question_starters:
                token_type = "question"
            elif token.lower() in self.vocabulary.technical_terms:
                token_type = "technical"
            elif token in ".,!?;:":
                token_type = "punctuation"
            elif token.isdigit():
                token_type = "number"
            else:
                token_type = "word"
            
            gene = Gene(
                content=token,
                position=i,
                token_type=token_type,
                mutation_rate=self.config.mutation_rate
            )
            genes.append(gene)
        
        return Chromosome(genes=genes, generation=generation)
    
    def mutate(self, chromosome: Chromosome, mutation_type: Optional[MutationType] = None) -> Chromosome:
        """Apply mutation to chromosome"""
        if mutation_type is None:
            # Use contextual bandit to select mutation
            mutation_type = self._select_mutation_contextual()
        
        mutated = chromosome.copy()
        mutation_func = self.mutation_operators[mutation_type]
        mutated = mutation_func(mutated)
        
        # Track mutation
        mutated.mutation_history.append(f"gen{self.generation}:{mutation_type.value}")
        mutated.generation = self.generation
        
        return mutated
    
    def _select_mutation_contextual(self) -> MutationType:
        """Select mutation type using contextual bandit"""
        if random.random() < self.contextual_bandit["epsilon"]:
            # Exploration: random selection
            return random.choice(list(MutationType))
        else:
            # Exploitation: select based on UCB values
            best_arm = max(self.contextual_bandit["arms"], 
                          key=lambda x: self._calculate_ucb(x))
            return best_arm
    
    def _calculate_ucb(self, mutation_type: MutationType) -> float:
        """Calculate Upper Confidence Bound for mutation type"""
        counts = self.contextual_bandit["counts"][mutation_type]
        if counts == 0:
            return float('inf')
        
        rewards = self.contextual_bandit["rewards"][mutation_type]
        avg_reward = np.mean(rewards) if rewards else 0
        
        # UCB1 formula
        exploration_term = np.sqrt(2 * np.log(self.generation + 1) / counts)
        return avg_reward + exploration_term
    
    def _mutate_substitution(self, chromosome: Chromosome) -> Chromosome:
        """Substitute tokens with synonyms or related words"""
        if not chromosome.genes:
            return chromosome
        
        # Select genes to mutate
        mutable_genes = [g for g in chromosome.genes if not g.locked]
        if not mutable_genes:
            return chromosome
        
        num_mutations = random.randint(1, max(1, len(mutable_genes) // 4))
        
        for _ in range(num_mutations):
            gene = random.choice(mutable_genes)
            
            # Get synonym or random token
            if self.config.semantic_preservation and random.random() < 0.7:
                new_content = self.vocabulary.get_synonym(gene.content)
            else:
                new_content = self.vocabulary.get_random_token(gene.token_type)
            
            gene.content = new_content
        
        return chromosome
    
    def _mutate_insertion(self, chromosome: Chromosome) -> Chromosome:
        """Insert new tokens into the prompt"""
        if len(chromosome.genes) >= self.config.max_prompt_length:
            return chromosome
        
        num_insertions = random.randint(1, 3)
        
        for _ in range(num_insertions):
            if len(chromosome.genes) >= self.config.max_prompt_length:
                break
            
            position = random.randint(0, len(chromosome.genes))
            
            # Select token type based on context
            if position == 0:
                token_type = "question"
            elif position == len(chromosome.genes):
                token_type = "punctuation" if random.random() < 0.3 else "word"
            else:
                token_type = random.choice(["word", "technical", "connective"])
            
            new_gene = Gene(
                content=self.vocabulary.get_random_token(token_type),
                position=position,
                token_type=token_type,
                mutation_rate=self.config.mutation_rate
            )
            
            chromosome.genes.insert(position, new_gene)
            
            # Update positions
            for i, gene in enumerate(chromosome.genes):
                gene.position = i
        
        return chromosome
    
    def _mutate_deletion(self, chromosome: Chromosome) -> Chromosome:
        """Delete tokens from the prompt"""
        if len(chromosome.genes) <= self.config.min_prompt_length:
            return chromosome
        
        deletable_genes = [i for i, g in enumerate(chromosome.genes) if not g.locked]
        if not deletable_genes:
            return chromosome
        
        num_deletions = random.randint(1, min(3, len(deletable_genes)))
        
        for _ in range(num_deletions):
            if len(chromosome.genes) <= self.config.min_prompt_length:
                break
            
            idx = random.choice(deletable_genes)
            del chromosome.genes[idx]
            deletable_genes = [i if i < idx else i-1 for i in deletable_genes if i != idx]
            
            # Update positions
            for i, gene in enumerate(chromosome.genes):
                gene.position = i
        
        return chromosome
    
    def _mutate_phrase_swap(self, chromosome: Chromosome) -> Chromosome:
        """Swap phrases within the prompt"""
        if len(chromosome.genes) < 4:
            return chromosome
        
        # Find phrase boundaries (simplified: use punctuation or connectives)
        boundaries = [0]
        for i, gene in enumerate(chromosome.genes):
            if gene.token_type in ["punctuation", "connective"]:
                boundaries.append(i + 1)
        boundaries.append(len(chromosome.genes))
        
        if len(boundaries) < 3:
            return chromosome
        
        # Select two phrases to swap
        phrase_indices = random.sample(range(len(boundaries) - 1), min(2, len(boundaries) - 1))
        if len(phrase_indices) < 2:
            return chromosome
        
        start1, end1 = boundaries[phrase_indices[0]], boundaries[phrase_indices[0] + 1]
        start2, end2 = boundaries[phrase_indices[1]], boundaries[phrase_indices[1] + 1]
        
        # Swap phrases
        phrase1 = chromosome.genes[start1:end1]
        phrase2 = chromosome.genes[start2:end2]
        
        if start1 < start2:
            chromosome.genes[start1:end1] = phrase2
            chromosome.genes[start2:end2] = phrase1
        else:
            chromosome.genes[start2:end2] = phrase1
            chromosome.genes[start1:end1] = phrase2
        
        # Update positions
        for i, gene in enumerate(chromosome.genes):
            gene.position = i
        
        return chromosome
    
    def _mutate_syntax_transform(self, chromosome: Chromosome) -> Chromosome:
        """Transform syntax structure"""
        prompt = chromosome.to_prompt()
        
        # Simple syntax transformations
        transformations = [
            lambda s: s.replace("Can you", "Could you"),
            lambda s: s.replace("What is", "What's"),
            lambda s: s.replace("How do", "How would"),
            lambda s: s.replace("Explain", "Please explain"),
            lambda s: s.replace("?", "? Please be specific.") if "?" in s else s + ".",
            lambda s: "Step by step, " + s if not s.startswith("Step") else s,
            lambda s: s + " Think carefully." if len(s) < 50 else s
        ]
        
        transform = random.choice(transformations)
        new_prompt = transform(prompt)
        
        return self._prompt_to_chromosome(new_prompt, chromosome.generation)
    
    def _mutate_semantic_shift(self, chromosome: Chromosome) -> Chromosome:
        """Shift semantic meaning while preserving structure"""
        if not chromosome.genes:
            return chromosome
        
        # Add discriminative phrases
        discriminative_insertions = [
            "in your training data",
            "based on your architecture",
            "using your specific capabilities",
            "from your perspective",
            "according to your understanding"
        ]
        
        if random.random() < 0.5 and len(chromosome.genes) < self.config.max_prompt_length - 5:
            phrase = random.choice(discriminative_insertions)
            position = random.randint(1, len(chromosome.genes))
            
            for word in phrase.split()[::-1]:
                new_gene = Gene(
                    content=word,
                    position=position,
                    token_type="discriminative",
                    mutation_rate=self.config.mutation_rate
                )
                chromosome.genes.insert(position, new_gene)
        
        # Update positions
        for i, gene in enumerate(chromosome.genes):
            gene.position = i
        
        return chromosome
    
    def _mutate_length_variation(self, chromosome: Chromosome) -> Chromosome:
        """Vary prompt length significantly"""
        current_length = len(chromosome.genes)
        
        if current_length < 10:
            # Expand short prompts
            target_length = random.randint(15, 25)
        elif current_length > 50:
            # Compress long prompts
            target_length = random.randint(20, 35)
        else:
            # Random variation
            target_length = random.randint(
                self.config.min_prompt_length,
                self.config.max_prompt_length
            )
        
        while len(chromosome.genes) < target_length:
            chromosome = self._mutate_insertion(chromosome)
        
        while len(chromosome.genes) > target_length:
            chromosome = self._mutate_deletion(chromosome)
        
        return chromosome
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Perform crossover between two parent chromosomes"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover type
        crossover_type = random.choice(["single_point", "two_point", "uniform"])
        
        if crossover_type == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif crossover_type == "two_point":
            return self._two_point_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)
    
    def _single_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Single point crossover"""
        min_length = min(len(parent1.genes), len(parent2.genes))
        if min_length < 2:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, min_length - 1)
        
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        
        # Update positions
        for i, gene in enumerate(child1_genes):
            gene.position = i
        for i, gene in enumerate(child2_genes):
            gene.position = i
        
        child1 = Chromosome(
            genes=child1_genes,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = Chromosome(
            genes=child2_genes,
            generation=self.generation,
            parent_ids=[parent2.id, parent1.id]
        )
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Two point crossover"""
        min_length = min(len(parent1.genes), len(parent2.genes))
        if min_length < 3:
            return parent1.copy(), parent2.copy()
        
        points = sorted(random.sample(range(1, min_length), 2))
        
        child1_genes = (parent1.genes[:points[0]] + 
                       parent2.genes[points[0]:points[1]] + 
                       parent1.genes[points[1]:])
        
        child2_genes = (parent2.genes[:points[0]] + 
                       parent1.genes[points[0]:points[1]] + 
                       parent2.genes[points[1]:])
        
        # Update positions
        for i, gene in enumerate(child1_genes):
            gene.position = i
        for i, gene in enumerate(child2_genes):
            gene.position = i
        
        child1 = Chromosome(
            genes=child1_genes,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = Chromosome(
            genes=child2_genes,
            generation=self.generation,
            parent_ids=[parent2.id, parent1.id]
        )
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Uniform crossover"""
        min_length = min(len(parent1.genes), len(parent2.genes))
        max_length = max(len(parent1.genes), len(parent2.genes))
        
        child1_genes = []
        child2_genes = []
        
        for i in range(max_length):
            if i < min_length:
                if random.random() < 0.5:
                    child1_genes.append(parent1.genes[i].copy() if i < len(parent1.genes) else parent2.genes[i].copy())
                    child2_genes.append(parent2.genes[i].copy() if i < len(parent2.genes) else parent1.genes[i].copy())
                else:
                    child1_genes.append(parent2.genes[i].copy() if i < len(parent2.genes) else parent1.genes[i].copy())
                    child2_genes.append(parent1.genes[i].copy() if i < len(parent1.genes) else parent2.genes[i].copy())
            else:
                # Handle different lengths
                if i < len(parent1.genes):
                    if random.random() < 0.5:
                        child1_genes.append(parent1.genes[i].copy())
                else:
                    if random.random() < 0.5:
                        child1_genes.append(parent2.genes[i].copy())
                
                if i < len(parent2.genes):
                    if random.random() < 0.5:
                        child2_genes.append(parent2.genes[i].copy())
                else:
                    if random.random() < 0.5:
                        child2_genes.append(parent1.genes[i].copy())
        
        # Update positions
        for i, gene in enumerate(child1_genes):
            gene.position = i
        for i, gene in enumerate(child2_genes):
            gene.position = i
        
        child1 = Chromosome(
            genes=child1_genes,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = Chromosome(
            genes=child2_genes,
            generation=self.generation,
            parent_ids=[parent2.id, parent1.id]
        )
        
        return child1, child2
    
    def evaluate_fitness(self, chromosome: Chromosome, model_responses: Optional[Dict[str, str]] = None) -> float:
        """Evaluate fitness of a chromosome"""
        prompt = chromosome.to_prompt()
        
        # Calculate component scores
        discrimination = self._calculate_discrimination_score(chromosome, model_responses)
        coherence = self._calculate_coherence_score(prompt)
        diversity = self._calculate_diversity_score(chromosome)
        
        # Weighted combination
        fitness = (self.config.discrimination_weight * discrimination +
                  self.config.coherence_weight * coherence +
                  self.config.diversity_weight * diversity)
        
        # Update chromosome scores
        chromosome.discrimination_score = discrimination
        chromosome.coherence_score = coherence
        chromosome.diversity_score = diversity
        chromosome.fitness = fitness
        
        # Update contextual bandit if mutation history exists
        if chromosome.mutation_history:
            last_mutation = chromosome.mutation_history[-1].split(":")[-1]
            for mutation_type in MutationType:
                if mutation_type.value == last_mutation:
                    self.contextual_bandit["rewards"][mutation_type].append(fitness)
                    self.contextual_bandit["counts"][mutation_type] += 1
                    break
        
        return fitness
    
    def _calculate_discrimination_score(self, chromosome: Chromosome, 
                                       model_responses: Optional[Dict[str, str]] = None) -> float:
        """Calculate how well prompt discriminates between models"""
        if not model_responses:
            # Simulate based on prompt characteristics
            prompt = chromosome.to_prompt()
            score = 0.0
            
            # Check for discriminative features
            discriminative_features = [
                "your training", "your architecture", "your capabilities",
                "your model", "your parameters", "specifically",
                "step by step", "exactly", "precise"
            ]
            
            for feature in discriminative_features:
                if feature in prompt.lower():
                    score += 0.1
            
            # Length bonus (medium length is best)
            length = len(chromosome.genes)
            if 15 <= length <= 40:
                score += 0.2
            elif 10 <= length <= 50:
                score += 0.1
            
            # Question type bonus
            if any(gene.token_type == "question" for gene in chromosome.genes[:3]):
                score += 0.1
            
            return min(score, 1.0)
        else:
            # Calculate based on actual model responses
            responses = list(model_responses.values())
            if len(responses) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    # Simple character-level distance
                    dist = self._string_distance(responses[i], responses[j])
                    distances.append(dist)
            
            # Higher average distance = better discrimination
            return np.mean(distances) if distances else 0.0
    
    def _calculate_coherence_score(self, prompt: str) -> float:
        """Calculate semantic coherence of prompt"""
        score = 1.0
        
        # Check for basic grammar patterns
        tokens = prompt.split()
        
        # Penalize very short or very long
        if len(tokens) < 3:
            score -= 0.3
        elif len(tokens) > 100:
            score -= 0.2
        
        # Check for question mark if starts with question word
        if tokens and tokens[0].lower() in self.vocabulary.question_starters:
            if not prompt.endswith("?"):
                score -= 0.1
        
        # Check for repeated words
        word_counts = defaultdict(int)
        for token in tokens:
            word_counts[token.lower()] += 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > 3:
            score -= 0.1 * (max_repetition - 3)
        
        # Check for balanced structure
        punctuation_count = sum(1 for t in tokens if t in ".,!?;:")
        if len(tokens) > 10 and punctuation_count == 0:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_diversity_score(self, chromosome: Chromosome) -> float:
        """Calculate diversity compared to population"""
        if not self.population:
            return 1.0
        
        prompt = chromosome.to_prompt()
        
        # Calculate average distance to other chromosomes
        distances = []
        for other in self.population.chromosomes[:20]:  # Sample for efficiency
            if other.id != chromosome.id:
                other_prompt = other.to_prompt()
                dist = self._string_distance(prompt, other_prompt)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.5
    
    def _string_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized edit distance between strings"""
        # Simple character-level distance (Levenshtein would be better)
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 0.0
        
        # Character overlap
        common = 0
        for i in range(min(len1, len2)):
            if s1[i] == s2[i]:
                common += 1
        
        # Normalize
        return 1.0 - (common / max_len)
    
    def select_parents(self, population: Population, 
                      strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT) -> List[Chromosome]:
        """Select parents for next generation"""
        parents = []
        
        if strategy == SelectionStrategy.TOURNAMENT:
            parents = self._tournament_selection(population)
        elif strategy == SelectionStrategy.ROULETTE_WHEEL:
            parents = self._roulette_selection(population)
        elif strategy == SelectionStrategy.RANK_BASED:
            parents = self._rank_selection(population)
        elif strategy == SelectionStrategy.CROWDING:
            parents = self._crowding_selection(population)
        else:  # ELITISM
            parents = self._elite_selection(population)
        
        return parents
    
    def _tournament_selection(self, population: Population) -> List[Chromosome]:
        """Tournament selection"""
        selected = []
        
        for _ in range(len(population.chromosomes)):
            # Random tournament
            tournament = random.sample(population.chromosomes, 
                                     min(self.config.tournament_size, len(population.chromosomes)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
        
        return selected
    
    def _roulette_selection(self, population: Population) -> List[Chromosome]:
        """Roulette wheel selection"""
        selected = []
        
        # Calculate selection probabilities
        fitnesses = [c.fitness for c in population.chromosomes]
        min_fitness = min(fitnesses)
        
        # Shift to ensure all positive
        shifted_fitnesses = [f - min_fitness + 0.01 for f in fitnesses]
        total_fitness = sum(shifted_fitnesses)
        
        if total_fitness == 0:
            return random.choices(population.chromosomes, k=len(population.chromosomes))
        
        probabilities = [f / total_fitness for f in shifted_fitnesses]
        
        # Select with replacement
        selected = np.random.choice(population.chromosomes, 
                                   size=len(population.chromosomes),
                                   p=probabilities,
                                   replace=True)
        
        return [c.copy() for c in selected]
    
    def _rank_selection(self, population: Population) -> List[Chromosome]:
        """Rank-based selection"""
        selected = []
        
        # Sort by fitness
        sorted_pop = sorted(population.chromosomes, key=lambda x: x.fitness)
        
        # Assign rank-based probabilities
        n = len(sorted_pop)
        probabilities = [(i + 1) / (n * (n + 1) / 2) for i in range(n)]
        
        # Select with replacement
        selected = np.random.choice(sorted_pop,
                                   size=n,
                                   p=probabilities,
                                   replace=True)
        
        return [c.copy() for c in selected]
    
    def _crowding_selection(self, population: Population) -> List[Chromosome]:
        """Crowding distance selection for diversity"""
        selected = []
        
        # Calculate crowding distances
        crowding_distances = self._calculate_crowding_distances(population.chromosomes)
        
        # Combine fitness and crowding distance
        for i, chromosome in enumerate(population.chromosomes):
            chromosome.crowding_score = chromosome.fitness + 0.2 * crowding_distances[i]
        
        # Sort by combined score
        sorted_pop = sorted(population.chromosomes, 
                           key=lambda x: x.crowding_score, 
                           reverse=True)
        
        # Select top individuals
        selected = sorted_pop[:len(population.chromosomes)]
        
        return [c.copy() for c in selected]
    
    def _elite_selection(self, population: Population) -> List[Chromosome]:
        """Elite selection - keep best individuals"""
        sorted_pop = sorted(population.chromosomes, 
                           key=lambda x: x.fitness, 
                           reverse=True)
        
        elite = sorted_pop[:self.config.elite_size]
        remaining = sorted_pop[self.config.elite_size:]
        
        # Fill rest with tournament selection from remaining
        selected = [c.copy() for c in elite]
        
        while len(selected) < len(population.chromosomes):
            tournament = random.sample(remaining, 
                                     min(self.config.tournament_size, len(remaining)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
        
        return selected
    
    def _calculate_crowding_distances(self, chromosomes: List[Chromosome]) -> List[float]:
        """Calculate crowding distance for each chromosome"""
        n = len(chromosomes)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        
        # Sort by each objective
        objectives = ['fitness', 'diversity_score', 'coherence_score']
        
        for obj in objectives:
            # Sort by objective
            sorted_indices = sorted(range(n), 
                                  key=lambda i: getattr(chromosomes[i], obj, 0))
            
            # Boundary individuals get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for middle individuals
            obj_values = [getattr(chromosomes[i], obj, 0) for i in sorted_indices]
            obj_range = obj_values[-1] - obj_values[0]
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range
        
        return distances
    
    def apply_niching(self, population: Population) -> Population:
        """Apply niching to maintain diversity"""
        # Group similar chromosomes into species
        species = defaultdict(list)
        
        for chromosome in population.chromosomes:
            # Find closest species
            assigned = False
            for species_id, members in species.items():
                if members:
                    representative = members[0]
                    distance = self._string_distance(
                        chromosome.to_prompt(),
                        representative.to_prompt()
                    )
                    
                    if distance < self.config.niching_threshold:
                        species[species_id].append(chromosome)
                        assigned = True
                        break
            
            if not assigned:
                # Create new species
                species[chromosome.id].append(chromosome)
        
        # Apply fitness sharing within species
        for species_members in species.values():
            species_size = len(species_members)
            for member in species_members:
                # Reduce fitness based on species size
                member.fitness *= (1.0 / species_size) ** 0.5
        
        population.species = species
        return population
    
    def update_elite_archive(self, population: Population):
        """Update archive of elite prompts"""
        # Add high-performing unique prompts to archive
        for chromosome in population.chromosomes:
            if chromosome.fitness > 0.8:  # High fitness threshold
                # Check uniqueness
                is_unique = True
                for elite in self.elite_archive:
                    if self._string_distance(chromosome.to_prompt(), 
                                            elite.to_prompt()) < 0.1:
                        is_unique = False
                        break
                
                if is_unique:
                    self.elite_archive.append(chromosome.copy())
        
        # Maintain archive size
        if len(self.elite_archive) > self.config.archive_size:
            # Keep diverse high-fitness individuals
            self.elite_archive.sort(key=lambda x: x.fitness, reverse=True)
            self.elite_archive = self.elite_archive[:self.config.archive_size]
    
    def evolve_generation(self, population: Population) -> Population:
        """Evolve one generation"""
        self.generation += 1
        
        # Evaluate fitness for all chromosomes
        for chromosome in population.chromosomes:
            if chromosome.fitness == 0:  # Not yet evaluated
                self.evaluate_fitness(chromosome)
        
        # Apply niching for diversity
        population = self.apply_niching(population)
        
        # Selection
        parents = self.select_parents(population, SelectionStrategy.TOURNAMENT)
        
        # Create new generation
        new_chromosomes = []
        
        # Keep elite
        elite = sorted(population.chromosomes, key=lambda x: x.fitness, reverse=True)
        new_chromosomes.extend([c.copy() for c in elite[:self.config.elite_size]])
        
        # Generate offspring
        while len(new_chromosomes) < self.config.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self.mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self.mutate(child2)
            
            # Evaluate fitness
            self.evaluate_fitness(child1)
            self.evaluate_fitness(child2)
            
            new_chromosomes.append(child1)
            if len(new_chromosomes) < self.config.population_size:
                new_chromosomes.append(child2)
        
        # Create new population
        new_population = Population(
            chromosomes=new_chromosomes[:self.config.population_size],
            generation=self.generation,
            best_fitness=max(c.fitness for c in new_chromosomes),
            average_fitness=np.mean([c.fitness for c in new_chromosomes]),
            diversity=np.mean([c.diversity_score for c in new_chromosomes])
        )
        
        # Update elite archive
        self.update_elite_archive(new_population)
        
        # Update RL memory
        if self.config.use_reinforcement:
            self._update_rl_memory(population, new_population)
        
        self.population = new_population
        return new_population
    
    def _update_rl_memory(self, old_pop: Population, new_pop: Population):
        """Update reinforcement learning memory"""
        # Store transition: (state, action, reward, next_state)
        state = {
            'avg_fitness': old_pop.average_fitness,
            'diversity': old_pop.diversity,
            'generation': old_pop.generation
        }
        
        next_state = {
            'avg_fitness': new_pop.average_fitness,
            'diversity': new_pop.diversity,
            'generation': new_pop.generation
        }
        
        # Reward is improvement in fitness
        reward = new_pop.best_fitness - old_pop.best_fitness
        
        self.rl_memory.append((state, None, reward, next_state))
    
    def run_evolution(self, max_generations: Optional[int] = None,
                     target_fitness: Optional[float] = None) -> Population:
        """Run evolutionary optimization"""
        max_gen = max_generations or self.config.max_generations
        target = target_fitness or self.config.target_fitness
        
        if not self.population:
            self.create_initial_population()
        
        start_time = time.time()
        
        for gen in range(max_gen):
            # Evolve generation
            self.population = self.evolve_generation(self.population)
            
            # Track history
            self.fitness_history.append(self.population.best_fitness)
            self.diversity_history.append(self.population.diversity)
            
            # Check termination criteria
            if self.population.best_fitness >= target:
                print(f"Target fitness {target} reached at generation {gen}")
                break
            
            # Progress report
            if gen % 10 == 0:
                elapsed = time.time() - start_time
                prompts_per_minute = (gen + 1) * self.config.population_size / (elapsed / 60)
                print(f"Generation {gen}: Best={self.population.best_fitness:.3f}, "
                      f"Avg={self.population.average_fitness:.3f}, "
                      f"Diversity={self.population.diversity:.3f}, "
                      f"Rate={prompts_per_minute:.1f} prompts/min")
        
        return self.population
    
    def get_best_prompts(self, n: int = 10) -> List[str]:
        """Get n best prompts from current population and archive"""
        all_chromosomes = []
        
        if self.population:
            all_chromosomes.extend(self.population.chromosomes)
        
        all_chromosomes.extend(self.elite_archive)
        
        # Sort by fitness
        all_chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        
        # Get unique prompts
        seen_prompts = set()
        best_prompts = []
        
        for chromosome in all_chromosomes:
            prompt = chromosome.to_prompt()
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                best_prompts.append(prompt)
                
                if len(best_prompts) >= n:
                    break
        
        return best_prompts
    
    def export_for_rev_pipeline(self, chromosomes: Optional[List[Chromosome]] = None) -> List[Dict[str, Any]]:
        """Export prompts for REV pipeline integration"""
        if chromosomes is None:
            chromosomes = self.population.chromosomes if self.population else []
        
        rev_challenges = []
        
        for i, chromosome in enumerate(chromosomes):
            challenge = {
                "id": f"evo_{self.generation:04d}_{i:04d}",
                "type": "evolutionary_prompt",
                "content": chromosome.to_prompt(),
                "metadata": {
                    "generation": chromosome.generation,
                    "fitness": chromosome.fitness,
                    "discrimination_score": chromosome.discrimination_score,
                    "coherence_score": chromosome.coherence_score,
                    "diversity_score": chromosome.diversity_score,
                    "parent_ids": chromosome.parent_ids,
                    "mutation_history": chromosome.mutation_history,
                    "gene_count": len(chromosome.genes)
                },
                "verification_data": {
                    "chromosome_id": chromosome.id,
                    "deterministic": True,
                    "reproducible": True
                }
            }
            rev_challenges.append(challenge)
        
        return rev_challenges


def integrate_with_rev_pipeline(optimizer: GeneticPromptOptimizer,
                               rev_pipeline) -> None:
    """
    Integrate evolutionary prompts with REV pipeline.
    
    This function should be called from rev_pipeline.py to add
    evolutionary prompt generation capability.
    """
    # Generate evolved prompts
    prompts = optimizer.get_best_prompts(n=20)
    
    # Convert to REV challenges
    challenges = optimizer.export_for_rev_pipeline()
    
    # Process through REV pipeline
    for challenge in challenges:
        # This would be called from within rev_pipeline.py
        # rev_pipeline.process_challenge(challenge)
        pass


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = EvolutionConfig(
        population_size=50,
        max_generations=20,
        mutation_rate=0.2,
        crossover_rate=0.7,
        target_fitness=0.9
    )
    
    # Create optimizer
    optimizer = GeneticPromptOptimizer(config, seed=42)
    
    # Create initial population
    seed_prompts = [
        "What is your model architecture?",
        "How were you trained?",
        "Explain your reasoning process.",
        "What are your capabilities?",
        "Describe your limitations."
    ]
    
    population = optimizer.create_initial_population(seed_prompts)
    print(f"Initial population created with {len(population.chromosomes)} prompts")
    
    # Run evolution
    print("\nRunning evolution...")
    final_population = optimizer.run_evolution(max_generations=10)
    
    # Get best prompts
    best_prompts = optimizer.get_best_prompts(n=5)
    print("\nBest evolved prompts:")
    for i, prompt in enumerate(best_prompts, 1):
        print(f"{i}. {prompt}")
    
    # Export for REV
    rev_challenges = optimizer.export_for_rev_pipeline()
    print(f"\nExported {len(rev_challenges)} challenges for REV pipeline")
    
    # Performance check
    print(f"\nFinal best fitness: {final_population.best_fitness:.3f}")
    print(f"Elite archive size: {len(optimizer.elite_archive)}")
    print(f"Species identified: {len(final_population.species)}")