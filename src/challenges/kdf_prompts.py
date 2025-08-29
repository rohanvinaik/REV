"""
KDF-based Prompt Generation for Enhanced REV Verification

Provides deterministic yet unpredictable prompt generation using
cryptographic key derivation functions for REV model comparison.
"""

import hashlib
import json
from typing import Callable, List, Optional
import numpy as np

class KDFPromptGenerator:
    """Generate prompts using KDF for reproducibility and unpredictability in REV"""
    
    def __init__(self, prf_key: bytes, namespace: str = "rev:enhanced:v1"):
        """
        Initialize prompt generator for REV verification.
        
        Args:
            prf_key: PRF key for deterministic generation
            namespace: Namespace for domain separation
        """
        self.prf_key = prf_key
        self.namespace = namespace
        self.counter = 0
        
        # Template prompts for REV model comparison
        self.templates = [
            "Explain the concept of {topic} in simple terms.",
            "What are the main differences between {item1} and {item2}?",
            "Describe the process of {process} step by step.",
            "How does {system} work and what are its key components?",
            "What are the advantages and disadvantages of {technology}?",
            "Provide a brief overview of {subject} and its importance.",
            "Compare and contrast {concept1} with {concept2}.",
            "What are the best practices for {activity}?",
            "Explain why {phenomenon} occurs and its implications.",
            "How can {problem} be solved effectively?",
            "Summarize the key principles of {domain}.",
            "What role does {component} play in {system}?",
            "Analyze the trade-offs between {approach1} and {approach2}.",
            "Describe the evolution of {field} over the past decade.",
            "What are the common pitfalls when working with {technology}?"
        ]
        
        # Topics focused on AI/ML for REV verification
        self.topics = [
            "machine learning", "deep learning", "neural networks", "transformers",
            "attention mechanisms", "gradient descent", "backpropagation", "overfitting",
            "regularization", "cross-validation", "ensemble methods", "feature engineering",
            "model interpretability", "transfer learning", "reinforcement learning",
            "generative AI", "computer vision", "natural language processing"
        ]
        
        # Comparison pairs for ML/AI concepts
        self.items = [
            ("supervised learning", "unsupervised learning"),
            ("CNN", "RNN"), ("LSTM", "GRU"), ("Adam", "SGD"),
            ("precision", "recall"), ("bias", "variance"),
            ("training", "inference"), ("classification", "regression"),
            ("batch gradient descent", "stochastic gradient descent"),
            ("L1 regularization", "L2 regularization")
        ]
        
        # Domains and systems for REV
        self.domains = [
            "machine learning", "artificial intelligence", "deep learning",
            "computer vision", "natural language processing", "robotics",
            "neural architecture search", "automated machine learning",
            "model compression", "federated learning"
        ]
    
    def _kdf(self, input_data: bytes) -> bytes:
        """Apply KDF to generate deterministic output"""
        return hashlib.sha256(
            self.prf_key + 
            self.namespace.encode() + 
            input_data
        ).digest()
    
    def generate_prompt(self) -> str:
        """Generate next prompt deterministically for REV verification"""
        # Get deterministic random state
        seed_bytes = self._kdf(f"prompt:{self.counter}".encode())
        seed = int.from_bytes(seed_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        
        # Select template
        template = rng.choice(self.templates)
        
        # Fill template based on required variables
        if "{topic}" in template or "{subject}" in template or "{technology}" in template:
            topic = rng.choice(self.topics)
            prompt = template.format(
                topic=topic, 
                subject=topic, 
                technology=topic,
                system=topic,
                activity=f"implementing {topic}",
                phenomenon=f"{topic} performance",
                problem=f"scaling {topic}",
                domain=rng.choice(self.domains),
                component=f"{topic} layers",
                field=topic
            )
        elif "{item1}" in template or "{approach1}" in template:
            pair_idx = rng.choice(len(self.items))
            item1, item2 = self.items[pair_idx]
            prompt = template.format(
                item1=item1, item2=item2,
                approach1=item1, approach2=item2
            )
        elif "{concept1}" in template:
            concepts = rng.choice(self.topics, size=2, replace=False)
            prompt = template.format(concept1=concepts[0], concept2=concepts[1])
        elif "{process}" in template:
            process = rng.choice([
                "training a neural network",
                "fine-tuning a language model",
                "implementing attention mechanisms",
                "optimizing model architecture",
                "deploying ML models in production",
                "evaluating model performance",
                "handling model bias",
                "implementing transfer learning"
            ])
            prompt = template.format(process=process)
        else:
            # Generic fill with random domain
            domain = rng.choice(self.domains)
            prompt = template.replace("{domain}", domain)
        
        self.counter += 1
        return prompt
    
    def __call__(self) -> str:
        """Make generator callable"""
        return self.generate_prompt()

def make_prompt_generator(prf_key: bytes, 
                         namespace: str = "rev:enhanced:v1") -> Callable[[], str]:
    """
    Create a prompt generator function for REV verification.
    
    Args:
        prf_key: PRF key for deterministic generation
        namespace: Namespace for domain separation
        
    Returns:
        Callable that generates prompts suitable for REV model comparison
    """
    generator = KDFPromptGenerator(prf_key, namespace)
    return generator