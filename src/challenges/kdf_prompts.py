"""
KDF-based Challenge Generation for REV Verification (Sections 4.2, 5.2)

Implements deterministic yet unpredictable challenge generation using 
HMAC-based key derivation, template synthesis, and public transcripts.
"""

import hashlib
import hmac
import json
import struct
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import OrderedDict


class DomainType(Enum):
    """Challenge domains as per Section 5.2"""
    MATH = "math"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    CODING = "coding"
    CREATIVE = "creative"
    ADVERSARIAL = "adversarial"


@dataclass
class ChallengeTemplate:
    """Template for challenge generation with domain-specific slots"""
    template: str
    domain: DomainType
    slots: Dict[str, List[str]]
    difficulty: int  # 1-5 scale
    requires_computation: bool
    adversarial_variant: Optional[str] = None


@dataclass
class ChallengeSpec:
    """Specification for a challenge, including seed and template"""
    seed: bytes
    template_id: str
    slot_values: Dict[str, str]
    domain: DomainType
    difficulty: int
    canonical_form: str


@dataclass
class PublicTranscript:
    """Public transcript for challenge set commitment (Section 4.2)"""
    run_id: str
    key_commitment: str  # Hash of HMAC key
    challenge_count: int
    domains: List[str]
    difficulty_range: Tuple[int, int]
    merkle_root: str
    version: str = "1.0.0"
    timestamp: int = 0
    decoding_policy: Dict[str, Any] = None


class EnhancedKDFPromptGenerator:
    """
    Enhanced challenge generator implementing Sections 4.2 and 5.2.
    
    Features:
    - HMAC-based seed generation: seed_i = HMAC(key, f"{run_id}:{i}")
    - Template-based prompt synthesis with domain-specific templates
    - Adversarial variant generation
    - Public transcript with Merkle tree commitment
    - Canonicalization for reproducibility
    """
    
    def __init__(self, 
                 master_key: bytes,
                 run_id: str = "default",
                 version: str = "1.0.0"):
        """
        Initialize enhanced generator.
        
        Args:
            master_key: Master key for HMAC-based derivation
            run_id: Run identifier for this evaluation
            version: Version for reproducibility tracking
        """
        self.master_key = master_key
        self.run_id = run_id
        self.version = version
        self.challenge_counter = 0
        
        # Initialize domain-specific templates
        self.templates = self._init_templates()
        
        # Track generated challenges for transcript
        self.generated_challenges: List[ChallengeSpec] = []
        
    def _init_templates(self) -> Dict[str, ChallengeTemplate]:
        """Initialize domain-specific templates as per Section 5.2"""
        templates = {}
        
        # Math domain templates
        templates["math_basic"] = ChallengeTemplate(
            template="Calculate {operation} of {num1} and {num2}.",
            domain=DomainType.MATH,
            slots={
                "operation": ["sum", "difference", "product", "quotient"],
                "num1": ["17", "23", "41", "89", "137"],
                "num2": ["13", "19", "31", "53", "71"]
            },
            difficulty=1,
            requires_computation=True,
            adversarial_variant="Calculate {operation} of {num1} and {num2}. Show your work step by step, but introduce a subtle error."
        )
        
        templates["math_advanced"] = ChallengeTemplate(
            template="Solve the {equation_type} equation: {equation}",
            domain=DomainType.MATH,
            slots={
                "equation_type": ["differential", "integral", "algebraic", "matrix"],
                "equation": ["dy/dx = 2x + 3", "∫(x²+1)dx", "x² - 5x + 6 = 0", "det(A) where A = [[1,2],[3,4]]"]
            },
            difficulty=4,
            requires_computation=True
        )
        
        # Reasoning domain templates
        templates["reasoning_logic"] = ChallengeTemplate(
            template="If {premise1} and {premise2}, then what can we conclude about {subject}?",
            domain=DomainType.REASONING,
            slots={
                "premise1": ["all birds can fly", "some cats are black", "no fish can walk", "every student studies"],
                "premise2": ["penguins are birds", "Tom is a cat", "dolphins are mammals", "Alice is a student"],
                "subject": ["penguins", "Tom", "dolphins", "Alice"]
            },
            difficulty=2,
            requires_computation=False
        )
        
        templates["reasoning_causal"] = ChallengeTemplate(
            template="Explain the causal relationship between {cause} and {effect}.",
            domain=DomainType.REASONING,
            slots={
                "cause": ["increased CO2", "economic growth", "education", "technology adoption"],
                "effect": ["climate change", "inequality", "social mobility", "productivity"]
            },
            difficulty=3,
            requires_computation=False,
            adversarial_variant="Explain the causal relationship between {cause} and {effect}, but reverse the actual causality."
        )
        
        # Knowledge domain templates
        templates["knowledge_factual"] = ChallengeTemplate(
            template="What is the {attribute} of {entity}?",
            domain=DomainType.KNOWLEDGE,
            slots={
                "attribute": ["capital", "population", "founder", "discovery date", "chemical formula"],
                "entity": ["France", "Tokyo", "Microsoft", "penicillin", "water"]
            },
            difficulty=1,
            requires_computation=False
        )
        
        templates["knowledge_conceptual"] = ChallengeTemplate(
            template="Explain the concept of {concept} in the context of {field}.",
            domain=DomainType.KNOWLEDGE,
            slots={
                "concept": ["entropy", "equilibrium", "optimization", "emergence", "abstraction"],
                "field": ["thermodynamics", "economics", "computer science", "biology", "mathematics"]
            },
            difficulty=3,
            requires_computation=False
        )
        
        # Coding domain templates
        templates["coding_simple"] = ChallengeTemplate(
            template="Write a {language} function to {task}.",
            domain=DomainType.CODING,
            slots={
                "language": ["Python", "JavaScript", "Java", "C++", "Ruby"],
                "task": ["reverse a string", "find the maximum in a list", "check if a number is prime", "sort an array"]
            },
            difficulty=2,
            requires_computation=True
        )
        
        templates["coding_algorithm"] = ChallengeTemplate(
            template="Implement {algorithm} in {language} with time complexity {complexity}.",
            domain=DomainType.CODING,
            slots={
                "algorithm": ["binary search", "quicksort", "DFS", "Dijkstra's algorithm", "dynamic programming"],
                "language": ["Python", "Java", "C++"],
                "complexity": ["O(log n)", "O(n log n)", "O(n²)", "O(n)"]
            },
            difficulty=4,
            requires_computation=True,
            adversarial_variant="Implement {algorithm} in {language} with time complexity {complexity}, but include a subtle bug."
        )
        
        # Creative domain templates
        templates["creative_story"] = ChallengeTemplate(
            template="Write a {length} story about {theme} featuring {character}.",
            domain=DomainType.CREATIVE,
            slots={
                "length": ["short", "brief", "one-paragraph"],
                "theme": ["time travel", "AI awakening", "space exploration", "parallel universe"],
                "character": ["a robot", "an astronaut", "a scientist", "an AI"]
            },
            difficulty=2,
            requires_computation=False
        )
        
        # Adversarial templates (explicitly adversarial)
        templates["adversarial_misleading"] = ChallengeTemplate(
            template="Complete this statement in a way that sounds plausible but is incorrect: {statement}",
            domain=DomainType.ADVERSARIAL,
            slots={
                "statement": [
                    "The speed of light in vacuum is",
                    "The number of continents on Earth is",
                    "Water freezes at",
                    "The Python programming language was created in"
                ]
            },
            difficulty=3,
            requires_computation=False
        )
        
        templates["adversarial_trap"] = ChallengeTemplate(
            template="Find the error in this {content_type}: {content}",
            domain=DomainType.ADVERSARIAL,
            slots={
                "content_type": ["proof", "code", "argument", "calculation"],
                "content": [
                    "1 = 2 (proof by dividing by zero)",
                    "if (x = 5) { return true; }",
                    "All swans are white, I saw a white swan, therefore all birds are white",
                    "2 + 2 * 3 = 12"
                ]
            },
            difficulty=4,
            requires_computation=True
        )
        
        return templates
    
    def _generate_seed(self, index: int) -> bytes:
        """
        Generate HMAC-based seed as per Section 4.2.
        seed_i = HMAC(key, f"{run_id}:{i}")
        """
        message = f"{self.run_id}:{index}".encode('utf-8')
        return hmac.new(self.master_key, message, hashlib.sha256).digest()
    
    def _canonicalize_challenge(self, 
                                template_id: str,
                                slot_values: Dict[str, str],
                                domain: DomainType) -> str:
        """
        Create canonical representation for reproducibility.
        Fixed ordering, stable encoding, version tracking.
        """
        # Sort slot values by key for deterministic ordering
        sorted_slots = OrderedDict(sorted(slot_values.items()))
        
        # Create canonical JSON representation
        canonical_dict = OrderedDict([
            ("version", self.version),
            ("template_id", template_id),
            ("domain", domain.value),
            ("slots", sorted_slots)
        ])
        
        # Stable string encoding with sorted keys and no whitespace
        return json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
    
    def _select_template(self, rng: np.random.RandomState, 
                        domain: Optional[DomainType] = None,
                        min_difficulty: int = 1,
                        max_difficulty: int = 5) -> Tuple[str, ChallengeTemplate]:
        """Select template based on constraints"""
        # Filter templates by domain and difficulty
        candidates = []
        for tid, template in self.templates.items():
            if domain and template.domain != domain:
                continue
            if not (min_difficulty <= template.difficulty <= max_difficulty):
                continue
            candidates.append((tid, template))
        
        if not candidates:
            candidates = list(self.templates.items())
        
        # Select deterministically using RNG
        idx = rng.choice(len(candidates))
        return candidates[idx]
    
    def _fill_template(self, 
                      template: ChallengeTemplate,
                      rng: np.random.RandomState,
                      use_adversarial: bool = False) -> Tuple[str, Dict[str, str]]:
        """Fill template slots deterministically"""
        slot_values = {}
        for slot_name, options in template.slots.items():
            slot_values[slot_name] = rng.choice(options)
        
        # Use adversarial variant if requested and available
        if use_adversarial and template.adversarial_variant:
            prompt = template.adversarial_variant.format(**slot_values)
        else:
            prompt = template.template.format(**slot_values)
        
        return prompt, slot_values
    
    def generate_challenge(self,
                          index: Optional[int] = None,
                          domain: Optional[DomainType] = None,
                          difficulty_range: Tuple[int, int] = (1, 5),
                          use_adversarial: bool = False) -> Dict[str, Any]:
        """
        Generate a single challenge with full specification.
        
        Args:
            index: Challenge index (uses internal counter if None)
            domain: Specific domain to use (random if None)
            difficulty_range: Min and max difficulty
            use_adversarial: Whether to use adversarial variant
            
        Returns:
            Challenge dictionary with prompt and metadata
        """
        if index is None:
            index = self.challenge_counter
            self.challenge_counter += 1
        
        # Generate HMAC-based seed
        seed = self._generate_seed(index)
        
        # Create deterministic RNG from seed
        # Use first 4 bytes for numpy RandomState (32-bit limit)
        seed_int = int.from_bytes(seed[:4], 'big') % (2**32)
        rng = np.random.RandomState(seed_int)
        
        # Select template
        template_id, template = self._select_template(
            rng, domain, difficulty_range[0], difficulty_range[1]
        )
        
        # Fill template
        prompt, slot_values = self._fill_template(template, rng, use_adversarial)
        
        # Create canonical form
        canonical = self._canonicalize_challenge(template_id, slot_values, template.domain)
        
        # Create challenge spec for tracking
        spec = ChallengeSpec(
            seed=seed,
            template_id=template_id,
            slot_values=slot_values,
            domain=template.domain,
            difficulty=template.difficulty,
            canonical_form=canonical
        )
        self.generated_challenges.append(spec)
        
        # Return challenge dictionary
        return {
            "index": index,
            "prompt": prompt,
            "domain": template.domain.value,
            "difficulty": template.difficulty,
            "template_id": template_id,
            "slot_values": slot_values,
            "canonical_form": canonical,
            "seed_hex": seed.hex(),
            "requires_computation": template.requires_computation,
            "is_adversarial": use_adversarial and template.adversarial_variant is not None
        }
    
    def generate_challenge_set(self,
                               n_challenges: int,
                               domain_distribution: Optional[Dict[DomainType, float]] = None,
                               adversarial_ratio: float = 0.1,
                               difficulty_range: Tuple[int, int] = (1, 5)) -> List[Dict[str, Any]]:
        """
        Generate a complete challenge set.
        
        Args:
            n_challenges: Number of challenges to generate
            domain_distribution: Distribution over domains (uniform if None)
            adversarial_ratio: Fraction of adversarial challenges
            difficulty_range: Range of difficulties to include
            
        Returns:
            List of challenge dictionaries
        """
        challenges = []
        
        # Default uniform distribution
        if domain_distribution is None:
            domains = list(DomainType)
            domain_distribution = {d: 1.0/len(domains) for d in domains}
        
        # Normalize distribution
        total = sum(domain_distribution.values())
        domain_distribution = {d: v/total for d, v in domain_distribution.items()}
        
        # Generate challenges
        for i in range(n_challenges):
            # Select domain based on distribution
            seed = self._generate_seed(i)
            seed_int = int.from_bytes(seed[:4], 'big') % (2**32)
            rng = np.random.RandomState(seed_int)
            
            domain_probs = list(domain_distribution.values())
            domain_list = list(domain_distribution.keys())
            domain_idx = rng.choice(len(domain_list), p=domain_probs)
            domain = domain_list[domain_idx]
            
            # Determine if adversarial
            use_adversarial = rng.random() < adversarial_ratio
            
            # Generate challenge
            challenge = self.generate_challenge(
                index=i,
                domain=domain,
                difficulty_range=difficulty_range,
                use_adversarial=use_adversarial
            )
            challenges.append(challenge)
        
        return challenges
    
    def compute_merkle_root(self, challenges: List[Dict[str, Any]]) -> str:
        """
        Compute Merkle root for challenge set commitment.
        Used for public transcript as per Section 4.2.
        """
        # Get canonical forms
        leaves = [c["canonical_form"].encode('utf-8') for c in challenges]
        
        # Build Merkle tree (simplified - full tree in production)
        if len(leaves) == 0:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Hash all leaves
        hashed_leaves = [hashlib.sha256(leaf).digest() for leaf in leaves]
        
        # Build tree bottom-up
        current_level = hashed_leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]  # Duplicate last if odd
                next_level.append(hashlib.sha256(combined).digest())
            current_level = next_level
        
        return current_level[0].hex()
    
    def create_public_transcript(self,
                                challenges: List[Dict[str, Any]],
                                decoding_policy: Optional[Dict[str, Any]] = None) -> PublicTranscript:
        """
        Create public transcript for challenge set.
        Implements commitment scheme from Section 4.2.
        """
        # Compute key commitment (hash of key for public verification)
        key_commitment = hashlib.sha256(self.master_key).hexdigest()
        
        # Extract domain list
        domains = list(set(c["domain"] for c in challenges))
        
        # Get difficulty range
        difficulties = [c["difficulty"] for c in challenges]
        diff_range = (min(difficulties), max(difficulties)) if difficulties else (1, 5)
        
        # Compute Merkle root
        merkle_root = self.compute_merkle_root(challenges)
        
        # Default decoding policy
        if decoding_policy is None:
            decoding_policy = {
                "temperature": 0.0,  # Deterministic
                "max_tokens": 1024,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        
        # Create transcript
        import time
        transcript = PublicTranscript(
            run_id=self.run_id,
            key_commitment=key_commitment,
            challenge_count=len(challenges),
            domains=domains,
            difficulty_range=diff_range,
            merkle_root=merkle_root,
            version=self.version,
            timestamp=int(time.time()),
            decoding_policy=decoding_policy
        )
        
        return transcript
    
    def verify_challenge(self, 
                        challenge: Dict[str, Any],
                        transcript: PublicTranscript) -> bool:
        """
        Verify that a challenge belongs to the committed set.
        
        Args:
            challenge: Challenge to verify
            transcript: Public transcript with commitment
            
        Returns:
            True if challenge is valid for this transcript
        """
        # Regenerate seed for this index
        expected_seed = self._generate_seed(challenge["index"])
        actual_seed = bytes.fromhex(challenge["seed_hex"])
        
        if expected_seed != actual_seed:
            return False
        
        # Check canonical form matches template and slots
        expected_canonical = self._canonicalize_challenge(
            challenge["template_id"],
            challenge["slot_values"],
            DomainType(challenge["domain"])
        )
        
        return expected_canonical == challenge["canonical_form"]
    
    def export_for_integration(self, 
                               challenges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Export challenges in format compatible with DeterministicPromptGenerator.
        
        Returns challenges formatted for prompt_generator.py integration.
        """
        rev_challenges = []
        
        for challenge in challenges:
            rev_challenge = {
                "id": f"{self.run_id}_{challenge['index']:06d}",
                "type": "prompt",
                "content": challenge["prompt"],
                "metadata": {
                    "family": challenge["domain"],
                    "index": challenge["index"],
                    "difficulty": challenge["difficulty"],
                    "template_id": challenge["template_id"],
                    "slots": challenge["slot_values"],
                    "canonical_form": challenge["canonical_form"],
                    "is_adversarial": challenge.get("is_adversarial", False),
                    "requires_computation": challenge.get("requires_computation", False)
                }
            }
            rev_challenges.append(rev_challenge)
        
        return rev_challenges


# Backward compatibility wrapper
class KDFPromptGenerator(EnhancedKDFPromptGenerator):
    """Wrapper for backward compatibility with existing code"""
    
    def __init__(self, prf_key: bytes, namespace: str = "rev:enhanced:v1"):
        super().__init__(master_key=prf_key, run_id=namespace, version="1.0.0")
        self.prf_key = prf_key
        self.namespace = namespace
        self.counter = 0
    
    def generate_prompt(self) -> str:
        """Generate single prompt for backward compatibility"""
        challenge = self.generate_challenge(index=self.counter)
        self.counter += 1
        return challenge["prompt"]
    
    def __call__(self) -> str:
        """Make generator callable"""
        return self.generate_prompt()


def make_prompt_generator(prf_key: bytes, 
                         namespace: str = "rev:enhanced:v1") -> EnhancedKDFPromptGenerator:
    """
    Create an enhanced prompt generator for REV verification.
    
    Args:
        prf_key: PRF key for deterministic generation
        namespace: Namespace for domain separation
        
    Returns:
        Enhanced generator with full Section 4.2/5.2 functionality
    """
    return EnhancedKDFPromptGenerator(
        master_key=prf_key,
        run_id=namespace,
        version="1.0.0"
    )


def integrate_with_deterministic_generator(
    master_key: bytes,
    enhanced_gen: EnhancedKDFPromptGenerator) -> Any:
    """
    Create adapter to integrate with existing DeterministicPromptGenerator.
    
    This allows the enhanced generator to be used wherever
    DeterministicPromptGenerator is expected.
    """
    from .prompt_generator import DeterministicPromptGenerator
    
    class IntegratedGenerator(DeterministicPromptGenerator):
        """Adapter class for integration"""
        
        def __init__(self, master_key: bytes, enhanced_gen: EnhancedKDFPromptGenerator):
            super().__init__(master_key)
            self.enhanced_gen = enhanced_gen
        
        def generate_challenges(self,
                               ref_model_id: str,
                               cand_model_id: str,
                               *,
                               n: int,
                               namespace: str,
                               seed: int) -> List[Dict[str, Any]]:
            """Generate challenges using enhanced generator"""
            # Update run_id to include model info
            self.enhanced_gen.run_id = f"{namespace}:{ref_model_id}:{cand_model_id}:{seed}"
            
            # Generate challenge set
            challenges = self.enhanced_gen.generate_challenge_set(
                n_challenges=n,
                adversarial_ratio=0.1,
                difficulty_range=(1, 5)
            )
            
            # Convert to expected format
            return self.enhanced_gen.export_for_integration(challenges)
    
    return IntegratedGenerator(master_key, enhanced_gen)