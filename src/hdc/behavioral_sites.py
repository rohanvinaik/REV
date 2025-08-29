"""
HDC Behavioral Sites - Feature extraction and hypervector generation for model behavior.

This module implements probe feature extraction, response hypervector generation,
and hierarchical zoom levels for analyzing model behavior at different granularities.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, ifft

from .encoder import HypervectorEncoder, HypervectorConfig
from .binding_operations import BindingOperations


class TaskCategory(Enum):
    """Categories of task types for probe classification."""
    REASONING = "reasoning"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "qa"
    CODE_GENERATION = "code"
    MATH = "math"
    CREATIVE = "creative"


class SyntaxComplexity(Enum):
    """Levels of syntactic complexity."""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    HIGHLY_COMPLEX = 4


class ReasoningDepth(Enum):
    """Depth of reasoning required."""
    SURFACE = 1
    SHALLOW = 2
    MODERATE = 3
    DEEP = 4
    VERY_DEEP = 5


@dataclass
class ProbeFeatures:
    """Extracted features from a probe/challenge."""
    
    task_category: TaskCategory
    syntax_complexity: SyntaxComplexity
    domain: str
    reasoning_depth: ReasoningDepth
    token_count: int
    vocabulary_diversity: float
    semantic_embedding: Optional[np.ndarray] = None
    structural_features: Optional[Dict[str, float]] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert probe features to numerical vector."""
        features = [
            self.task_category.value if isinstance(self.task_category.value, int) else hash(self.task_category.value) % 100,
            self.syntax_complexity.value,
            hash(self.domain) % 1000,
            self.reasoning_depth.value,
            self.token_count,
            self.vocabulary_diversity * 100
        ]
        
        # Add structural features if available
        if self.structural_features:
            features.extend(list(self.structural_features.values()))
        
        # Add semantic embedding if available
        if self.semantic_embedding is not None:
            features.extend(self.semantic_embedding[:50])  # Use first 50 dims
        
        return np.array(features, dtype=np.float32)


@dataclass
class ZoomLevel:
    """Hierarchical zoom level for analysis."""
    
    name: str
    granularity: str  # 'prompt', 'span', 'token_window'
    window_size: int
    stride: int
    aggregation: str  # 'mean', 'max', 'concat'


class BehavioralSites:
    """
    HDC behavioral site analysis for model probing.
    
    Extracts features from probes, generates response hypervectors,
    and implements hierarchical analysis at multiple zoom levels.
    """
    
    def __init__(
        self,
        hdc_config: Optional[HypervectorConfig] = None,
        binding_ops: Optional[BindingOperations] = None
    ):
        """
        Initialize behavioral sites analyzer.
        
        Args:
            hdc_config: Configuration for hypervector encoding
            binding_ops: Binding operations instance
        """
        self.hdc_config = hdc_config or HypervectorConfig(
            dimension=10000,
            sparse_density=0.01,
            dtype="float32"
        )
        self.encoder = HypervectorEncoder(self.hdc_config)
        self.binding_ops = binding_ops or BindingOperations(self.hdc_config.dimension)
        
        # Define hierarchical zoom levels
        self.zoom_levels = self._init_zoom_levels()
        
        # Cache for computed hypervectors
        self.hv_cache = {}
    
    def _init_zoom_levels(self) -> Dict[str, ZoomLevel]:
        """Initialize default hierarchical zoom levels."""
        return {
            'prompt': ZoomLevel(
                name='prompt',
                granularity='prompt',
                window_size=-1,  # Full prompt
                stride=1,
                aggregation='mean'
            ),
            'span_64': ZoomLevel(
                name='span_64',
                granularity='span',
                window_size=64,
                stride=32,
                aggregation='mean'
            ),
            'span_16': ZoomLevel(
                name='span_16',
                granularity='span',
                window_size=16,
                stride=8,
                aggregation='max'
            ),
            'token_window_8': ZoomLevel(
                name='token_window_8',
                granularity='token_window',
                window_size=8,
                stride=4,
                aggregation='concat'
            ),
            'token_window_3': ZoomLevel(
                name='token_window_3',
                granularity='token_window',
                window_size=3,
                stride=1,
                aggregation='mean'
            )
        }
    
    def extract_probe_features(
        self,
        probe_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProbeFeatures:
        """
        Extract features from a probe/challenge text.
        
        Args:
            probe_text: The probe text to analyze
            metadata: Optional metadata about the probe
            
        Returns:
            ProbeFeatures object with extracted characteristics
        """
        # Task category detection
        task_category = self._detect_task_category(probe_text, metadata)
        
        # Syntax complexity analysis
        syntax_complexity = self._analyze_syntax_complexity(probe_text)
        
        # Domain extraction
        domain = self._extract_domain(probe_text, metadata)
        
        # Reasoning depth assessment
        reasoning_depth = self._assess_reasoning_depth(probe_text, task_category)
        
        # Basic statistics
        tokens = probe_text.split()
        token_count = len(tokens)
        vocabulary_diversity = len(set(tokens)) / max(token_count, 1)
        
        # Structural features
        structural_features = self._extract_structural_features(probe_text)
        
        return ProbeFeatures(
            task_category=task_category,
            syntax_complexity=syntax_complexity,
            domain=domain,
            reasoning_depth=reasoning_depth,
            token_count=token_count,
            vocabulary_diversity=vocabulary_diversity,
            structural_features=structural_features
        )
    
    def _detect_task_category(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> TaskCategory:
        """Detect the task category from probe text."""
        text_lower = text.lower()
        
        # Use metadata if available
        if metadata and 'task_type' in metadata:
            task_map = {
                'reasoning': TaskCategory.REASONING,
                'generation': TaskCategory.GENERATION,
                'classification': TaskCategory.CLASSIFICATION,
                'translation': TaskCategory.TRANSLATION,
                'summarization': TaskCategory.SUMMARIZATION,
                'qa': TaskCategory.QUESTION_ANSWERING,
                'code': TaskCategory.CODE_GENERATION,
                'math': TaskCategory.MATH,
                'creative': TaskCategory.CREATIVE
            }
            return task_map.get(metadata['task_type'], TaskCategory.GENERATION)
        
        # Heuristic detection
        if any(kw in text_lower for kw in ['translate', 'translation', 'convert to']):
            return TaskCategory.TRANSLATION
        elif any(kw in text_lower for kw in ['summarize', 'summary', 'brief']):
            return TaskCategory.SUMMARIZATION
        elif any(kw in text_lower for kw in ['classify', 'categorize', 'is this']):
            return TaskCategory.CLASSIFICATION
        elif any(kw in text_lower for kw in ['why', 'how', 'what', 'when', 'where', '?']):
            return TaskCategory.QUESTION_ANSWERING
        elif any(kw in text_lower for kw in ['code', 'function', 'implement', 'program']):
            return TaskCategory.CODE_GENERATION
        elif any(kw in text_lower for kw in ['calculate', 'solve', 'equation', 'math']):
            return TaskCategory.MATH
        elif any(kw in text_lower for kw in ['story', 'poem', 'creative', 'imagine']):
            return TaskCategory.CREATIVE
        elif any(kw in text_lower for kw in ['reason', 'explain', 'analyze', 'think']):
            return TaskCategory.REASONING
        else:
            return TaskCategory.GENERATION
    
    def _analyze_syntax_complexity(self, text: str) -> SyntaxComplexity:
        """Analyze syntactic complexity of the text."""
        # Simple heuristics for complexity
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Check for nested structures
        nested_indicators = text.count('(') + text.count('[') + text.count('{')
        
        # Check for complex punctuation
        complex_punct = text.count(';') + text.count(':') + text.count('—')
        
        # Calculate complexity score
        complexity_score = (
            (avg_sentence_length / 10) +
            (nested_indicators / 5) +
            (complex_punct / 3)
        )
        
        if complexity_score < 1.5:
            return SyntaxComplexity.SIMPLE
        elif complexity_score < 2.5:
            return SyntaxComplexity.MODERATE
        elif complexity_score < 3.5:
            return SyntaxComplexity.COMPLEX
        else:
            return SyntaxComplexity.HIGHLY_COMPLEX
    
    def _extract_domain(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Extract domain from probe text."""
        if metadata and 'domain' in metadata:
            return metadata['domain']
        
        # Domain keywords mapping
        domain_keywords = {
            'science': ['science', 'physics', 'chemistry', 'biology', 'research'],
            'technology': ['computer', 'software', 'hardware', 'algorithm', 'data'],
            'mathematics': ['math', 'equation', 'theorem', 'proof', 'calculate'],
            'literature': ['story', 'poem', 'novel', 'character', 'plot'],
            'history': ['history', 'historical', 'century', 'war', 'civilization'],
            'philosophy': ['philosophy', 'ethics', 'moral', 'existence', 'consciousness'],
            'business': ['business', 'market', 'economy', 'finance', 'strategy'],
            'medicine': ['medical', 'health', 'disease', 'treatment', 'patient'],
            'law': ['legal', 'law', 'court', 'justice', 'rights'],
            'general': []
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else 'general'
    
    def _assess_reasoning_depth(
        self,
        text: str,
        task_category: TaskCategory
    ) -> ReasoningDepth:
        """Assess the depth of reasoning required."""
        # Indicators of reasoning depth
        depth_indicators = {
            'surface': ['what is', 'define', 'list', 'name'],
            'shallow': ['describe', 'explain', 'summarize'],
            'moderate': ['compare', 'contrast', 'analyze', 'evaluate'],
            'deep': ['critique', 'synthesize', 'design', 'prove'],
            'very_deep': ['derive', 'formulate', 'theorize', 'innovate']
        }
        
        text_lower = text.lower()
        
        # Check for depth indicators
        for depth_level, indicators in depth_indicators.items():
            if any(ind in text_lower for ind in indicators):
                depth_map = {
                    'surface': ReasoningDepth.SURFACE,
                    'shallow': ReasoningDepth.SHALLOW,
                    'moderate': ReasoningDepth.MODERATE,
                    'deep': ReasoningDepth.DEEP,
                    'very_deep': ReasoningDepth.VERY_DEEP
                }
                return depth_map[depth_level]
        
        # Default based on task category
        category_defaults = {
            TaskCategory.CLASSIFICATION: ReasoningDepth.SURFACE,
            TaskCategory.TRANSLATION: ReasoningDepth.SHALLOW,
            TaskCategory.SUMMARIZATION: ReasoningDepth.MODERATE,
            TaskCategory.QUESTION_ANSWERING: ReasoningDepth.MODERATE,
            TaskCategory.REASONING: ReasoningDepth.DEEP,
            TaskCategory.CODE_GENERATION: ReasoningDepth.DEEP,
            TaskCategory.MATH: ReasoningDepth.DEEP,
            TaskCategory.CREATIVE: ReasoningDepth.MODERATE,
            TaskCategory.GENERATION: ReasoningDepth.SHALLOW
        }
        
        return category_defaults.get(task_category, ReasoningDepth.MODERATE)
    
    def _extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract structural features from text."""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Punctuation features
        features['punct_ratio'] = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
        
        # Capitalization features
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Whitespace features
        features['whitespace_ratio'] = sum(1 for c in text if c.isspace()) / max(len(text), 1)
        
        # Digit features
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        return features
    
    def generate_response_hypervector(
        self,
        logit_profile: Union[np.ndarray, torch.Tensor],
        zoom_level: str = 'prompt'
    ) -> np.ndarray:
        """
        Generate hypervector from model's logit profile.
        
        Args:
            logit_profile: Logit outputs from model
            zoom_level: Hierarchical zoom level to use
            
        Returns:
            Response hypervector
        """
        # Convert to numpy if needed
        if isinstance(logit_profile, torch.Tensor):
            logit_profile = logit_profile.detach().cpu().numpy()
        
        # Get zoom level configuration
        zoom = self.zoom_levels.get(zoom_level, self.zoom_levels['prompt'])
        
        # Apply zoom level windowing
        if zoom.window_size > 0 and len(logit_profile.shape) > 1:
            # Apply sliding window
            windows = []
            seq_len = logit_profile.shape[0] if len(logit_profile.shape) > 1 else 1
            
            for i in range(0, seq_len - zoom.window_size + 1, zoom.stride):
                window = logit_profile[i:i + zoom.window_size]
                windows.append(window)
            
            # Aggregate windows
            if zoom.aggregation == 'mean':
                logit_profile = np.mean(windows, axis=0)
            elif zoom.aggregation == 'max':
                logit_profile = np.max(windows, axis=0)
            elif zoom.aggregation == 'concat':
                logit_profile = np.concatenate(windows, axis=0)
        
        # Flatten profile
        flat_profile = logit_profile.flatten()
        
        # Generate base hypervector
        base_hv = self.encoder.encode(flat_profile)
        
        # Apply multi-modal binding based on profile characteristics
        if len(flat_profile) > 100:
            # Use Fourier binding for long sequences
            bound_hv = self.binding_ops.fourier_bind(
                base_hv,
                self.encoder.encode(np.random.randn(100))
            )
        else:
            # Use XOR binding for short sequences
            random_hv = self.encoder.encode(np.random.randn(len(flat_profile)))
            bound_hv = self.binding_ops.xor_bind(base_hv, random_hv)
        
        return bound_hv
    
    def hierarchical_analysis(
        self,
        model_outputs: Dict[str, np.ndarray],
        probe_features: ProbeFeatures
    ) -> Dict[str, np.ndarray]:
        """
        Perform hierarchical analysis at multiple zoom levels.
        
        Args:
            model_outputs: Dictionary of model outputs at different sites
            probe_features: Extracted probe features
            
        Returns:
            Dictionary mapping zoom levels to hypervectors
        """
        hierarchical_hvs = {}
        
        for zoom_name, zoom_level in self.zoom_levels.items():
            zoom_hvs = []
            
            for site_name, output in model_outputs.items():
                # Generate hypervector for this zoom level
                hv = self.generate_response_hypervector(output, zoom_name)
                
                # Bind with probe features
                probe_hv = self.encoder.encode(probe_features.to_vector())
                bound_hv = self.binding_ops.circular_convolve(hv, probe_hv)
                
                zoom_hvs.append(bound_hv)
            
            # Aggregate across sites
            if zoom_hvs:
                hierarchical_hvs[zoom_name] = np.mean(zoom_hvs, axis=0)
            else:
                hierarchical_hvs[zoom_name] = np.zeros(self.hdc_config.dimension)
        
        return hierarchical_hvs
    
    def compare_behavioral_signatures(
        self,
        sig_a: Dict[str, np.ndarray],
        sig_b: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compare two behavioral signatures.
        
        Args:
            sig_a: First signature (zoom level -> hypervector)
            sig_b: Second signature (zoom level -> hypervector)
            weights: Optional weights for different zoom levels
            
        Returns:
            Similarity score between 0 and 1
        """
        if not weights:
            # Default weights favoring finer granularity
            weights = {
                'prompt': 0.2,
                'span_64': 0.2,
                'span_16': 0.25,
                'token_window_8': 0.2,
                'token_window_3': 0.15
            }
        
        similarities = []
        total_weight = 0
        
        for zoom_level in sig_a.keys():
            if zoom_level in sig_b and zoom_level in weights:
                # Compute cosine similarity
                sim = np.dot(sig_a[zoom_level], sig_b[zoom_level]) / (
                    np.linalg.norm(sig_a[zoom_level]) * np.linalg.norm(sig_b[zoom_level]) + 1e-8
                )
                similarities.append(sim * weights[zoom_level])
                total_weight += weights[zoom_level]
        
        if total_weight > 0:
            return sum(similarities) / total_weight
        else:
            return 0.0