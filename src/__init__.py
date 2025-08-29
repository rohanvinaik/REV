"""
REV - Restriction Enzyme Verification

A framework for memory-bounded, black-box LLM comparison using 
restriction enzyme verification techniques combined with:

- Sequential Testing Framework with anytime-valid statistical tests
- Challenge Generation with seeded, deterministic prompts  
- Model Verification Pipeline with streaming execution
- Hyperdimensional Computing Core with 8K-100K dimensional vectors
- Hamming Distance LUTs with 16-bit lookup tables for 10-20× speedup
- Privacy-Preserving Infrastructure with zero-knowledge proofs

Components adapted from PoT_Experiments and GenomeVault for REV verification.
"""

from .core.sequential import sequential_verify, SPRTResult, SequentialState
from .core.boundaries import SequentialTest, eb_radius
from .challenges.prompt_generator import DeterministicPromptGenerator
from .verifier.decision import EnhancedSequentialTester, Verdict
from .hdc.encoder import HypervectorEncoder, HypervectorConfig
from .hypervector.hamming import HammingLUT, hamming_distance_cpu
from .crypto.zk_proofs import ZKProofSystem, generate_zk_proof
from .privacy.differential_privacy import DifferentialPrivacyMechanism, PrivacyLevel

__version__ = "0.1.0"

__all__ = [
    # Sequential testing
    "sequential_verify", "SPRTResult", "SequentialState",
    "SequentialTest", "eb_radius",
    
    # Challenge generation
    "DeterministicPromptGenerator",
    
    # Model verification
    "EnhancedSequentialTester", "Verdict",
    
    # Hyperdimensional computing
    "HypervectorEncoder", "HypervectorConfig", 
    "HammingLUT", "hamming_distance_cpu",
    
    # Privacy and cryptography
    "ZKProofSystem", "generate_zk_proof",
    "DifferentialPrivacyMechanism", "PrivacyLevel"
]