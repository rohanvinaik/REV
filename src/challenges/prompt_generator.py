import hmac
import hashlib
import random
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str
    category: str
    slots: Dict[str, List[str]]

class DeterministicPromptGenerator:
    """Prompt generator with unified API for REV verification"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.templates = self._init_templates()
    
    def _init_templates(self) -> List[PromptTemplate]:
        """Initialize prompt templates for REV model comparison"""
        return [
            PromptTemplate(
                "Explain {topic} in {style}.",
                "factual",
                {
                    "topic": ["machine learning", "neural networks", "deep learning", "AI safety", "transformers"],
                    "style": ["simple terms", "one sentence", "technical detail", "layman terms"]
                }
            ),
            PromptTemplate(
                "What is {concept}?",
                "knowledge",
                {
                    "concept": ["attention mechanism", "backpropagation", "gradient descent", "overfitting", "regularization"]
                }
            ),
            PromptTemplate(
                "Compare {item1} and {item2}.",
                "comparison",
                {
                    "item1": ["supervised learning", "CNN", "RNN", "LSTM"],
                    "item2": ["unsupervised learning", "transformer", "GAN", "autoencoder"]
                }
            ),
            PromptTemplate(
                "Write a {length} explanation of {topic}.",
                "explanation",
                {
                    "length": ["brief", "detailed", "comprehensive"],
                    "topic": ["model training", "inference", "evaluation metrics", "cross-validation"]
                }
            ),
            PromptTemplate(
                "List {count} examples of {category}.",
                "enumeration",
                {
                    "count": ["three", "five", "ten"],
                    "category": ["ML algorithms", "neural architectures", "activation functions", "loss functions"]
                }
            )
        ]
    
    def _sample_slots(self, rng: random.Random, template: PromptTemplate) -> Dict[str, str]:
        """Sample slot values deterministically"""
        result = {}
        for slot, options in template.slots.items():
            result[slot] = rng.choice(options)
        return result
    
    def generate_challenges(self,
                           ref_model_id: str,
                           cand_model_id: str,
                           *,
                           n: int,
                           namespace: str,
                           seed: int) -> List[Dict[str, Any]]:
        """Generate challenges with unified signature for REV"""
        
        # Create deterministic RNG using HMAC-based key derivation
        seed_data = f"{namespace}:{seed}:{ref_model_id}:{cand_model_id}".encode()
        seed_bytes = hmac.new(self.master_key, seed_data, hashlib.sha256).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], 'big'))
        
        challenges = []
        
        for i in range(n):
            # Select template deterministically
            template = rng.choice(self.templates)
            
            # Fill slots deterministically
            slots = self._sample_slots(rng, template)
            prompt = template.template.format(**slots)
            
            challenges.append({
                "prompt": prompt,
                "family": template.category,
                "idx": i,
                "ref_model": ref_model_id,
                "cand_model": cand_model_id,
                "namespace": namespace,
                "template_id": id(template),
                "slots": slots
            })
        
        return challenges
    
    # Backward compatibility wrapper
    def __call__(self) -> str:
        """Generate single prompt for backward compatibility"""
        challenges = self.generate_challenges(
            "default", "default",
            n=1, namespace="default", seed=random.randint(0, 2**32)
        )
        return challenges[0]["prompt"] if challenges else ""


# Factory function for backward compatibility
def make_prompt_generator(master_key: bytes, namespace: str = "default"):
    """Create prompt generator for REV verification"""
    gen = DeterministicPromptGenerator(master_key)
    
    # Return a callable that generates single prompts
    def prompt_fn():
        return gen()
    
    # Attach the full generator for access to all methods
    prompt_fn.generator = gen
    
    return prompt_fn


def create_prompt_challenges(master_key: bytes, 
                            family: str,
                            params: Dict[str, Any],
                            n_challenges: int) -> List[Dict[str, Any]]:
    """Create deterministic prompt challenges for REV verification"""
    
    generator = DeterministicPromptGenerator(master_key)
    namespace = f"{family}:{json.dumps(params, sort_keys=True)}"
    
    # Extract model IDs from params if available
    ref_model = params.get("ref_model", "default")
    cand_model = params.get("cand_model", "default") 
    seed = params.get("seed", 42)
    
    challenges = generator.generate_challenges(
        ref_model, cand_model,
        n=n_challenges,
        namespace=namespace,
        seed=seed
    )
    
    # Convert to REV format
    rev_challenges = []
    for challenge in challenges:
        rev_challenge = {
            "id": f"{family}_{challenge['idx']:06d}",
            "type": "prompt",
            "content": challenge["prompt"],
            "metadata": {
                "family": challenge["family"],
                "index": challenge["idx"],
                "namespace": challenge["namespace"],
                "template_id": challenge.get("template_id"),
                "slots": challenge.get("slots", {})
            }
        }
        rev_challenges.append(rev_challenge)
    
    return rev_challenges