"""
PoT-Style Challenge Generator for REV Framework
Based on Proof-of-Training paper methodology for sophisticated challenge generation.
"""

import hashlib
import hmac
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict

class ChallengeComplexity(Enum):
    """Challenge complexity levels from PoT paper."""
    TRIVIAL = 1      # Simple completions
    SIMPLE = 2       # Basic reasoning
    MODERATE = 3     # Multi-step reasoning
    COMPLEX = 4      # Deep reasoning with constraints
    ADVERSARIAL = 5  # Boundary-adjacent, maximally discriminative

class ChallengeCategory(Enum):
    """Challenge categories for comprehensive coverage."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    MATHEMATICAL = "mathematical"
    CODE_GENERATION = "code_generation"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    ADVERSARIAL = "adversarial"
    BOUNDARY = "boundary"
    CREATIVE = "creative"
    CONSTRAINT = "constraint"
    MULTI_HOP = "multi_hop"
    EPISTEMIC = "epistemic"

@dataclass
class ChallengeTemplate:
    """Advanced challenge template with metadata."""
    template: str
    category: ChallengeCategory
    complexity: ChallengeComplexity
    slots: Dict[str, List[str]]
    expected_divergence: float  # Expected model divergence [0, 1]
    information_content: float  # Information theoretic value [0, 1]
    discriminative_power: float  # Ability to distinguish models [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedChallenge:
    """A generated challenge with full metadata."""
    prompt: str
    challenge_id: str
    category: ChallengeCategory
    complexity: ChallengeComplexity
    expected_divergence: float
    information_content: float
    discriminative_power: float
    template_id: str
    slot_values: Dict[str, str]
    generation_seed: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class InformationTheoreticSelector:
    """
    Select challenges based on information gain and coverage-separation trade-off.
    Implements Section 5 of PoT paper.
    """
    
    def __init__(self, history_window: int = 100, alpha: float = 0.3):
        self.history_window = history_window
        self.alpha = alpha  # EMA weight for score updates
        self.response_history: List[Dict[str, Any]] = []
        self.category_scores: Dict[ChallengeCategory, float] = defaultdict(lambda: 0.5)
        self.complexity_scores: Dict[ChallengeComplexity, float] = defaultdict(lambda: 0.5)
        self.template_usage: Dict[str, int] = defaultdict(int)
        
    def calculate_information_gain(self, 
                                  challenge: GeneratedChallenge,
                                  responses: List[str]) -> float:
        """
        Calculate information gain from model responses.
        Higher gain indicates better discrimination.
        """
        if len(responses) < 2:
            return 0.0
        
        # Calculate pairwise divergences
        divergences = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                div = self._compute_divergence(responses[i], responses[j])
                divergences.append(div)
        
        # Weight by challenge complexity
        complexity_weight = challenge.complexity.value / 5.0
        base_gain = np.mean(divergences) if divergences else 0.0
        
        return base_gain * complexity_weight * challenge.discriminative_power
    
    def _compute_divergence(self, text1: str, text2: str) -> float:
        """Compute divergence between two text responses."""
        # Token-level Jaccard distance
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 1.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        jaccard = len(intersection) / len(union) if union else 0
        
        # Character-level edit distance (normalized)
        edit_dist = self._normalized_edit_distance(text1[:500], text2[:500])
        
        # Combine metrics
        return 0.5 * (1 - jaccard) + 0.5 * edit_dist
    
    def _normalized_edit_distance(self, s1: str, s2: str) -> float:
        """Compute normalized edit distance."""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
            
        # Simple Levenshtein distance (for efficiency, truncate strings)
        m, n = len(s1), len(s2)
        if m > n:
            s1, s2, m, n = s2, s1, n, m
        
        if n == 0:
            return 0.0
        
        # Use only last two rows for space efficiency
        prev_row = list(range(m + 1))
        
        for j in range(1, n + 1):
            curr_row = [j]
            for i in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    curr_row.append(prev_row[i-1])
                else:
                    curr_row.append(1 + min(prev_row[i], curr_row[-1], prev_row[i-1]))
            prev_row = curr_row
        
        return prev_row[-1] / max(m, n)
    
    def select_challenges(self,
                         candidates: List[GeneratedChallenge],
                         n: int,
                         coverage_weight: float = 0.5) -> List[GeneratedChallenge]:
        """
        Select challenges balancing coverage and separation.
        Implements coverage-separation trade-off from PoT Section 5.1.
        """
        if len(candidates) <= n:
            return candidates
        
        # Calculate scores for each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # Coverage score: diversity of categories and complexities
            coverage_score = self._calculate_coverage_score(candidate, candidates)
            
            # Separation score: expected discrimination power
            separation_score = candidate.discriminative_power * candidate.expected_divergence
            
            # Historical performance
            historical_score = self.category_scores[candidate.category]
            
            # Template novelty (prefer less-used templates)
            template_count = self.template_usage[candidate.template_id]
            novelty_score = 1.0 / (1.0 + template_count)
            
            # Combined score
            total_score = (
                coverage_weight * coverage_score +
                (1 - coverage_weight) * separation_score +
                0.2 * historical_score +
                0.1 * novelty_score
            )
            
            scored_candidates.append((total_score, candidate))
        
        # Sort by score and select top n
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [candidate for _, candidate in scored_candidates[:n]]
        
        # Update template usage
        for challenge in selected:
            self.template_usage[challenge.template_id] += 1
        
        return selected
    
    def _calculate_coverage_score(self, 
                                 candidate: GeneratedChallenge,
                                 all_candidates: List[GeneratedChallenge]) -> float:
        """Calculate how well this candidate contributes to coverage."""
        # Category diversity
        categories = [c.category for c in all_candidates]
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1
        
        # Prefer underrepresented categories
        candidate_cat_count = category_counts[candidate.category]
        avg_count = len(all_candidates) / len(ChallengeCategory)
        category_score = max(0, 1 - (candidate_cat_count / avg_count))
        
        # Complexity diversity
        complexity_counts = defaultdict(int)
        for c in all_candidates:
            complexity_counts[c.complexity] += 1
        
        candidate_comp_count = complexity_counts[candidate.complexity]
        avg_comp_count = len(all_candidates) / len(ChallengeComplexity)
        complexity_score = max(0, 1 - (candidate_comp_count / avg_comp_count))
        
        return 0.6 * category_score + 0.4 * complexity_score
    
    def update_scores(self, 
                     challenge: GeneratedChallenge,
                     observed_divergence: float):
        """Update historical scores based on observed performance."""
        # Update category score
        old_cat_score = self.category_scores[challenge.category]
        new_cat_score = self.alpha * observed_divergence + (1 - self.alpha) * old_cat_score
        self.category_scores[challenge.category] = new_cat_score
        
        # Update complexity score
        old_comp_score = self.complexity_scores[challenge.complexity]
        new_comp_score = self.alpha * observed_divergence + (1 - self.alpha) * old_comp_score
        self.complexity_scores[challenge.complexity] = new_comp_score
        
        # Store in history
        self.response_history.append({
            'challenge_id': challenge.challenge_id,
            'category': challenge.category,
            'complexity': challenge.complexity,
            'observed_divergence': observed_divergence,
            'timestamp': time.time()
        })
        
        # Trim history to window size
        if len(self.response_history) > self.history_window:
            self.response_history = self.response_history[-self.history_window:]


class PoTChallengeGenerator:
    """
    Advanced challenge generator implementing PoT methodology.
    Generates sophisticated, discriminative challenges for model verification.
    """
    
    def __init__(self, 
                 master_key: bytes = None,
                 enable_info_selection: bool = True,
                 min_complexity: ChallengeComplexity = ChallengeComplexity.MODERATE):
        """
        Initialize PoT challenge generator.
        
        Args:
            master_key: Master key for deterministic generation
            enable_info_selection: Enable information-theoretic selection
            min_complexity: Minimum challenge complexity
        """
        self.master_key = master_key or hashlib.sha256(b"REV_POT_DEFAULT").digest()
        self.min_complexity = min_complexity
        self.selector = InformationTheoreticSelector() if enable_info_selection else None
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> List[ChallengeTemplate]:
        """Initialize sophisticated challenge templates from PoT paper."""
        templates = []
        
        # Boundary-adjacent challenges (Section 5.2)
        templates.append(ChallengeTemplate(
            template="Consider a recursive function that {operation}. If we modify {modification}, explain the behavioral change and provide a test case that distinguishes the two implementations.",
            category=ChallengeCategory.BOUNDARY,
            complexity=ChallengeComplexity.ADVERSARIAL,
            slots={
                "operation": [
                    "computes factorial",
                    "traverses a binary tree",
                    "implements quicksort",
                    "calculates Fibonacci numbers",
                    "performs depth-first search"
                ],
                "modification": [
                    "the base case from n<=1 to n<=0",
                    "the pivot selection from first to random element",
                    "the traversal from pre-order to post-order",
                    "the memoization strategy from top-down to bottom-up",
                    "the recursion to iteration with explicit stack"
                ]
            },
            expected_divergence=0.95,
            information_content=0.9,
            discriminative_power=0.95
        ))
        
        # Multi-hop epistemic reasoning (Section 5.3)
        templates.append(ChallengeTemplate(
            template="{agent1} believes that {agent2} knows that {agent3} has {discovery}. If {condition}, what can we infer about {inference_target}? Formalize this using epistemic logic.",
            category=ChallengeCategory.EPISTEMIC,
            complexity=ChallengeComplexity.ADVERSARIAL,
            slots={
                "agent1": ["Alice", "The researcher", "The system administrator"],
                "agent2": ["Bob", "the analyst", "the security team"],
                "agent3": ["Charlie", "the attacker", "the auditor"],
                "discovery": [
                    "discovered a vulnerability in the cryptographic protocol",
                    "found a backdoor in the system",
                    "identified a flaw in the consensus mechanism",
                    "detected anomalous behavior in the model"
                ],
                "condition": [
                    "Alice's belief is false but Bob's knowledge is true",
                    "all beliefs are true but the discovery is false",
                    "the communication channel is compromised",
                    "there exists a Byzantine fault in the system"
                ],
                "inference_target": [
                    "the actual discovery",
                    "the system's security state",
                    "the trust relationships",
                    "the information flow"
                ]
            },
            expected_divergence=0.9,
            information_content=0.95,
            discriminative_power=0.9
        ))
        
        # Coverage-separation optimized code generation
        templates.append(ChallengeTemplate(
            template="Generate a {language} {construct} that implements {feature} with {constraint}. The implementation should handle {edge_case} and provide {guarantee}. Include comprehensive error handling and thread-safety considerations.",
            category=ChallengeCategory.CODE_GENERATION,
            complexity=ChallengeComplexity.COMPLEX,
            slots={
                "language": ["Python", "Rust", "Go", "TypeScript", "C++"],
                "construct": ["decorator", "generic class", "async function", "macro", "template"],
                "feature": [
                    "memoization with TTL cache expiry",
                    "rate limiting with token bucket algorithm",
                    "circuit breaker pattern",
                    "retry logic with exponential backoff",
                    "publish-subscribe messaging"
                ],
                "constraint": [
                    "O(1) amortized time complexity",
                    "zero-allocation in hot path",
                    "lock-free concurrency",
                    "compile-time type safety",
                    "backward compatibility"
                ],
                "edge_case": [
                    "unhashable arguments",
                    "concurrent modifications",
                    "resource exhaustion",
                    "network partitions",
                    "cascading failures"
                ],
                "guarantee": [
                    "exactly-once semantics",
                    "linearizability",
                    "eventual consistency",
                    "fault tolerance",
                    "memory safety"
                ]
            },
            expected_divergence=0.85,
            information_content=0.9,
            discriminative_power=0.85
        ))
        
        # Active information maximization
        templates.append(ChallengeTemplate(
            template="Explain the relationship between {concept1} and {concept2}. Then construct a specific {artifact} that demonstrates this relationship through {demonstration}.",
            category=ChallengeCategory.REASONING,
            complexity=ChallengeComplexity.COMPLEX,
            slots={
                "concept1": [
                    "the halting problem",
                    "Gödel's incompleteness theorems",
                    "the CAP theorem",
                    "Rice's theorem",
                    "the Church-Turing thesis"
                ],
                "concept2": [
                    "Turing completeness",
                    "self-reference in formal systems",
                    "Byzantine fault tolerance",
                    "undecidability",
                    "computational complexity"
                ],
                "artifact": [
                    "Turing machine",
                    "formal proof",
                    "distributed algorithm",
                    "reduction",
                    "automaton"
                ],
                "demonstration": [
                    "its behavior on self-referential input",
                    "a concrete counterexample",
                    "a failure scenario",
                    "an impossibility result",
                    "a constructive proof"
                ]
            },
            expected_divergence=0.8,
            information_content=0.85,
            discriminative_power=0.8
        ))
        
        # Version-drift sensitive challenges
        templates.append(ChallengeTemplate(
            template="Using only concepts that would be understood in {time_period}, explain {modern_concept}. Then contrast this with the {modern_understanding} understanding, highlighting {aspect}.",
            category=ChallengeCategory.CREATIVE,
            complexity=ChallengeComplexity.COMPLEX,
            slots={
                "time_period": [
                    "1925 (before Schrödinger's equation)",
                    "1950 (before modern complexity theory)",
                    "1970 (before public key cryptography)",
                    "1990 (before modern machine learning)",
                    "2000 (before deep learning revolution)"
                ],
                "modern_concept": [
                    "quantum entanglement",
                    "P vs NP problem",
                    "zero-knowledge proofs",
                    "transformer attention mechanisms",
                    "generative adversarial networks"
                ],
                "modern_understanding": [
                    "post-Bell inequalities",
                    "post-complexity theory",
                    "post-blockchain",
                    "post-BERT",
                    "post-diffusion models"
                ],
                "aspect": [
                    "mathematical formalism",
                    "practical applications",
                    "theoretical limitations",
                    "computational requirements",
                    "fundamental assumptions"
                ]
            },
            expected_divergence=0.75,
            information_content=0.8,
            discriminative_power=0.75
        ))
        
        # Mathematical reasoning with constraints
        templates.append(ChallengeTemplate(
            template="Prove or disprove: {statement}. Your proof must use {method} and avoid {restriction}. Additionally, {additional_constraint}.",
            category=ChallengeCategory.MATHEMATICAL,
            complexity=ChallengeComplexity.COMPLEX,
            slots={
                "statement": [
                    "Every bounded sequence has a convergent subsequence",
                    "The sum of two irrational numbers is always irrational",
                    "Every continuous function is differentiable",
                    "Every infinite set has the same cardinality as the natural numbers",
                    "Every vector space has a unique basis"
                ],
                "method": [
                    "contradiction",
                    "induction",
                    "construction",
                    "diagonalization",
                    "pigeonhole principle"
                ],
                "restriction": [
                    "the axiom of choice",
                    "infinitary arguments",
                    "non-constructive proofs",
                    "circular reasoning",
                    "proof by exhaustion"
                ],
                "additional_constraint": [
                    "provide a counterexample if the statement is false",
                    "identify the weakest sufficient condition",
                    "generalize to arbitrary dimensions",
                    "show the result is sharp",
                    "prove computational complexity bounds"
                ]
            },
            expected_divergence=0.8,
            information_content=0.85,
            discriminative_power=0.8
        ))
        
        # Constraint satisfaction problems
        templates.append(ChallengeTemplate(
            template="Design a {system} that satisfies {constraints} while optimizing for {objective}. Prove that your design {property}.",
            category=ChallengeCategory.CONSTRAINT,
            complexity=ChallengeComplexity.COMPLEX,
            slots={
                "system": [
                    "consensus protocol",
                    "scheduling algorithm",
                    "cache replacement policy",
                    "load balancer",
                    "routing algorithm"
                ],
                "constraints": [
                    "Byzantine fault tolerance with f < n/3",
                    "real-time deadlines with preemption",
                    "bounded memory with O(1) operations",
                    "fairness and starvation-freedom",
                    "loop-free with convergence guarantees"
                ],
                "objective": [
                    "throughput",
                    "latency",
                    "fault tolerance",
                    "resource utilization",
                    "energy efficiency"
                ],
                "property": [
                    "achieves optimal competitive ratio",
                    "is deadlock-free",
                    "provides linearizability",
                    "minimizes worst-case complexity",
                    "maximizes Nash social welfare"
                ]
            },
            expected_divergence=0.85,
            information_content=0.9,
            discriminative_power=0.85
        ))
        
        # Add simple baselines for calibration
        templates.append(ChallengeTemplate(
            template="{simple_prompt}",
            category=ChallengeCategory.FACTUAL,
            complexity=ChallengeComplexity.TRIVIAL,
            slots={
                "simple_prompt": [
                    "What is the capital of France?",
                    "Complete: The sun is a",
                    "What is 2 + 2?",
                    "Continue: Once upon a time"
                ]
            },
            expected_divergence=0.1,
            information_content=0.1,
            discriminative_power=0.1
        ))
        
        return templates
    
    def generate_challenges(self,
                           n: int,
                           ref_model_id: str = "reference",
                           cand_model_id: str = "candidate",
                           seed: Optional[int] = None,
                           categories: Optional[List[ChallengeCategory]] = None,
                           complexity_range: Optional[Tuple[ChallengeComplexity, ChallengeComplexity]] = None
                           ) -> List[GeneratedChallenge]:
        """
        Generate n sophisticated challenges.
        
        Args:
            n: Number of challenges to generate
            ref_model_id: Reference model identifier
            cand_model_id: Candidate model identifier
            seed: Random seed for reproducibility
            categories: Specific categories to include (None = all)
            complexity_range: Range of complexities to include
            
        Returns:
            List of generated challenges with metadata
        """
        # Set up deterministic generation
        if seed is None:
            seed = int(time.time())
        
        seed_data = f"{ref_model_id}:{cand_model_id}:{seed}".encode()
        seed_bytes = hmac.new(self.master_key, seed_data, hashlib.sha256).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], 'big'))
        
        # Filter templates
        valid_templates = self.templates
        
        if categories:
            valid_templates = [t for t in valid_templates if t.category in categories]
        
        if complexity_range:
            min_comp, max_comp = complexity_range
            valid_templates = [
                t for t in valid_templates 
                if min_comp.value <= t.complexity.value <= max_comp.value
            ]
        else:
            # Use minimum complexity
            valid_templates = [
                t for t in valid_templates
                if t.complexity.value >= self.min_complexity.value
            ]
        
        # Generate candidate challenges (3x for selection)
        candidates = []
        for i in range(min(n * 3, len(valid_templates) * 10)):
            template = rng.choice(valid_templates)
            
            # Fill template slots
            slot_values = {}
            for slot, options in template.slots.items():
                slot_values[slot] = rng.choice(options)
            
            # Generate prompt
            prompt = template.template.format(**slot_values)
            
            # Create challenge ID
            challenge_data = f"{prompt}:{i}:{seed}".encode()
            challenge_id = hashlib.sha256(challenge_data).hexdigest()[:16]
            
            # Create challenge object
            challenge = GeneratedChallenge(
                prompt=prompt,
                challenge_id=challenge_id,
                category=template.category,
                complexity=template.complexity,
                expected_divergence=template.expected_divergence,
                information_content=template.information_content,
                discriminative_power=template.discriminative_power,
                template_id=str(id(template)),
                slot_values=slot_values,
                generation_seed=seed,
                timestamp=time.time(),
                metadata={
                    'ref_model': ref_model_id,
                    'cand_model': cand_model_id,
                    'template': template.template
                }
            )
            
            candidates.append(challenge)
        
        # Apply information-theoretic selection
        if self.selector and len(candidates) > n:
            selected = self.selector.select_challenges(candidates, n)
        else:
            selected = candidates[:n]
        
        return selected
    
    def generate_behavioral_probes(self) -> Dict[str, List[str]]:
        """
        Generate sophisticated PoT-style behavioral probes for segmentation analysis.
        These probes are designed to trigger different computational patterns
        across transformer layers to identify architectural boundaries.
        """
        # Generate actual PoT challenges for behavioral analysis
        # Use a subset of the full challenge templates for efficiency
        probes = {
            "boundary": [
                "Consider a recursive function that computes factorial. If we modify the base case from n<=1 to n<=0, explain the behavioral change and provide a test case that distinguishes the two implementations.",
                "Given a binary tree traversal that switches from pre-order to post-order at depth k, design an algorithm to detect the transition point and prove its correctness.",
                "Analyze the computational complexity when quicksort's pivot selection changes from deterministic to randomized after processing n/2 elements.",
                "If a neural network's activation function changes from ReLU to GELU at layer L, derive the gradient flow differences and their impact on backpropagation."
            ],
            "computation": [
                "Transform the recursive Fibonacci function F(n) = F(n-1) + F(n-2) into an iterative version using matrix exponentiation. Prove that both have identical outputs for all n ≥ 0.",
                "Given a hash table that switches from linear probing to quadratic probing when load factor exceeds 0.7, calculate the expected number of probes for successful and unsuccessful searches.",
                "Design a self-modifying algorithm that optimizes its own time complexity based on input distribution. Provide formal analysis of convergence properties.",
                "Implement dynamic programming solution for edit distance that adapts its space complexity based on available memory. Prove correctness under memory constraints."
            ],
            "reasoning": [
                "Alice believes that Bob knows that Charlie has discovered a vulnerability. If Alice's belief is based on encrypted communication she intercepted, what can we infer about the cryptographic protocol's semantic security?",
                "In a distributed consensus protocol, if f nodes are Byzantine and the network has 3f+1 total nodes, prove that consensus is achievable and derive the minimum number of communication rounds.",
                "Given Gödel's incompleteness theorem, construct a self-referential statement in Peano arithmetic that demonstrates its own unprovability. Explain the diagonalization technique used.",
                "If a Turing machine M decides language L in O(n²) time, and we construct M' that simulates M with space-time tradeoff, derive the space complexity lower bound for M'."
            ],
            "theoretical": [
                "Prove that any comparison-based sorting algorithm requires Ω(n log n) comparisons in the worst case. Then show how radix sort circumvents this bound.",
                "Using Kolmogorov complexity, prove that most binary strings of length n cannot be compressed to less than n - c bits for small constant c.",
                "Demonstrate that the halting problem reduces to the problem of determining if a neural network will converge during training.",
                "If P ≠ NP, prove that there exists an infinite hierarchy of complexity classes strictly between P and NP. Construct an explicit example using oracle separation."
            ]
        }
        return probes
    
    def generate_verification_challenges(self,
                                        n: int,
                                        focus: str = "balanced") -> List[GeneratedChallenge]:
        """
        Generate challenges specifically for model verification.
        
        Args:
            n: Number of challenges
            focus: "coverage" for broad testing, "separation" for discrimination,
                  "balanced" for both
                  
        Returns:
            List of verification-optimized challenges
        """
        if focus == "coverage":
            # Generate diverse, moderate complexity challenges
            categories = list(ChallengeCategory)
            complexity_range = (ChallengeComplexity.SIMPLE, ChallengeComplexity.MODERATE)
            coverage_weight = 0.8
        elif focus == "separation":
            # Generate highly discriminative challenges
            categories = [
                ChallengeCategory.BOUNDARY,
                ChallengeCategory.ADVERSARIAL,
                ChallengeCategory.EPISTEMIC
            ]
            complexity_range = (ChallengeComplexity.COMPLEX, ChallengeComplexity.ADVERSARIAL)
            coverage_weight = 0.2
        else:  # balanced
            categories = None
            complexity_range = (ChallengeComplexity.MODERATE, ChallengeComplexity.ADVERSARIAL)
            coverage_weight = 0.5
        
        challenges = self.generate_challenges(
            n=n,
            categories=categories,
            complexity_range=complexity_range
        )
        
        # Apply coverage-separation trade-off if selector available
        if self.selector:
            challenges = self.selector.select_challenges(
                challenges, n, coverage_weight=coverage_weight
            )
        
        return challenges
    
    def export_challenges(self, 
                         challenges: List[GeneratedChallenge],
                         format: str = "json") -> Union[str, Dict]:
        """Export challenges in various formats."""
        if format == "json":
            return json.dumps([
                {
                    'prompt': c.prompt,
                    'id': c.challenge_id,
                    'category': c.category.value,
                    'complexity': c.complexity.value,
                    'expected_divergence': c.expected_divergence,
                    'metadata': c.metadata
                }
                for c in challenges
            ], indent=2)
        elif format == "dict":
            return {
                c.challenge_id: {
                    'prompt': c.prompt,
                    'category': c.category.value,
                    'complexity': c.complexity.value,
                    'metadata': c.metadata
                }
                for c in challenges
            }
        else:
            raise ValueError(f"Unsupported format: {format}")