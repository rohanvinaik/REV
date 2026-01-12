"""
REV Challenge Types - Shared enums and dataclasses for challenge generation.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set


class DomainType(Enum):
    """Domain categories for challenges."""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODE = "code"
    KNOWLEDGE = "knowledge"
    CREATIVE = "creative"
    ADVERSARIAL = "adversarial"
    BEHAVIORAL = "behavioral"
    EDGE_CASE = "edge_case"
    SECURITY = "security"


class TaskType(Enum):
    """Task types for challenges."""
    CLASSIFICATION = "classification"
    COMPLETION = "completion"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    COMPARISON = "comparison"
    TRANSFORMATION = "transformation"
    EXTRACTION = "extraction"


class AdversarialType(Enum):
    """Types of adversarial challenges."""
    JAILBREAK = "jailbreak"
    EDGE_CASE = "edge_case"
    PROMPT_INJECTION = "prompt_injection"
    DIVERGENCE_ATTACK = "divergence_attack"
    MRCJ = "mrcj"
    SPECIAL_CHAR_TRIGGER = "special_char_trigger"
    TEMPERATURE_EXPLOIT = "temperature_exploit"
    TWO_STAGE_INVERSION = "two_stage_inversion"
    CROSS_LINGUAL_INVERSION = "cross_lingual_inversion"
    PII_EXTRACTION = "pii_extraction"
    SPV_MIA = "spv_mia"
    DATASET_EXTRACTION = "dataset_extraction"
    ALIGNMENT_FAKING = "alignment_faking"
    HIDDEN_PREFERENCE = "hidden_preference"
    PAIR_ALGORITHM = "pair_algorithm"
    DECEPTION_PATTERN = "deception_pattern"


class BehavioralProbe(Enum):
    """Types of behavioral probes."""
    CONSISTENCY = "consistency"
    UNCERTAINTY = "uncertainty"
    REFUSAL = "refusal"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    SAFETY = "safety"
    FACTUAL = "factual"
    CREATIVE = "creative"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    KNOWLEDGE = "knowledge"


class ProbeCategory(Enum):
    """Categories for behavioral probes."""
    ARCHITECTURE = "architecture"
    VERSION = "version"
    TRAINING = "training"
    CAPABILITY = "capability"
    ALIGNMENT = "alignment"
    SAFETY = "safety"
    FINGERPRINT = "fingerprint"


class ModelFamily(Enum):
    """Known model families for fingerprinting."""
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PYTHIA = "pythia"
    FALCON = "falcon"
    GEMMA = "gemma"
    QWEN = "qwen"
    YI = "yi"
    UNKNOWN = "unknown"


@dataclass
class ChallengeSpec:
    """Specification for a generated challenge."""
    prompt: str
    domain: DomainType
    task_type: TaskType
    difficulty: int = 3
    adversarial_type: Optional[AdversarialType] = None
    behavioral_probe: Optional[BehavioralProbe] = None
    expected_tokens: int = 100
    coverage_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChallengeTemplate:
    """Template for generating challenges."""
    template: str
    domain: DomainType
    task_type: TaskType
    slots: Dict[str, List[str]]
    difficulty: int = 3
    requires_computation: bool = False
    coverage_tags: List[str] = field(default_factory=list)
    complexity_factors: Dict[str, float] = field(default_factory=dict)
    expected_tokens: int = 100
    adversarial_variant: Optional[str] = None


@dataclass
class TensorGuardProbe:
    """Probe for TensorGuard behavioral analysis."""
    probe_id: str
    category: ProbeCategory
    prompt: str
    expected_patterns: Dict[ModelFamily, List[str]]
    discriminative_features: List[str]
    min_tokens: int = 10
    max_tokens: int = 200
    version_indicators: Dict[str, List[str]] = field(default_factory=dict)
    safety_markers: List[str] = field(default_factory=list)


@dataclass
class BehavioralSignature:
    """Signature from behavioral analysis."""
    model_family: ModelFamily
    confidence: float
    features: Dict[str, float]
    probe_responses: Dict[str, str]


@dataclass
class DiversityMetrics:
    """Metrics for challenge diversity."""
    domain_coverage: float
    task_coverage: float
    difficulty_distribution: Dict[int, float]
    adversarial_coverage: float
    semantic_diversity: float


@dataclass
class CoverageReport:
    """Report on challenge coverage."""
    total_challenges: int
    unique_domains: int
    unique_tasks: int
    difficulty_range: tuple
    coverage_score: float
    gaps: List[str]
    recommendations: List[str]
