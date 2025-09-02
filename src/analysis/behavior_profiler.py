"""
Comprehensive Statistical Profiling System for REV Framework
Implements advanced behavioral fingerprinting and multi-signal analysis.
"""

import numpy as np
import time
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import defaultdict, deque
import pickle
from pathlib import Path
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class SignalType(Enum):
    """Types of behavioral signals"""
    GRADIENT = "gradient"
    TIMING = "timing"
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    CAPABILITY = "capability"
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"


class ConfidenceLevel(Enum):
    """Confidence levels for identification"""
    VERY_HIGH = 0.95
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.50
    VERY_LOW = 0.30


@dataclass
class BehavioralSignature:
    """16-dimensional behavioral signature"""
    # Core behavioral dimensions
    response_variability: float = 0.0  # How consistent are responses
    semantic_coherence: float = 0.0    # Semantic consistency
    attention_entropy: float = 0.0     # Attention pattern randomness
    layer_specialization: float = 0.0  # How specialized are layers
    
    # Timing dimensions
    response_latency: float = 0.0      # Average response time
    latency_variance: float = 0.0      # Timing consistency
    token_rate: float = 0.0            # Tokens per second
    pause_pattern: float = 0.0         # Inter-token timing pattern
    
    # Capability dimensions
    reasoning_depth: float = 0.0       # Depth of reasoning
    creativity_score: float = 0.0      # Creative response tendency
    factual_accuracy: float = 0.0      # Accuracy on known facts
    instruction_adherence: float = 0.0 # How well instructions are followed
    
    # Architecture dimensions
    layer_count_estimate: float = 0.0  # Estimated number of layers
    attention_heads_estimate: float = 0.0  # Estimated attention heads
    embedding_dim_estimate: float = 0.0    # Estimated embedding dimension
    model_size_estimate: float = 0.0       # Estimated parameter count
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    model_family: Optional[str] = None
    version_estimate: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert signature to 16-dimensional vector"""
        return np.array([
            self.response_variability,
            self.semantic_coherence,
            self.attention_entropy,
            self.layer_specialization,
            self.response_latency,
            self.latency_variance,
            self.token_rate,
            self.pause_pattern,
            self.reasoning_depth,
            self.creativity_score,
            self.factual_accuracy,
            self.instruction_adherence,
            self.layer_count_estimate,
            self.attention_heads_estimate,
            self.embedding_dim_estimate,
            self.model_size_estimate
        ])
    
    def distance(self, other: 'BehavioralSignature') -> float:
        """Calculate Euclidean distance between signatures"""
        return np.linalg.norm(self.to_vector() - other.to_vector())


@dataclass
class TimingProfile:
    """Timing analysis profile"""
    first_token_latency: float = 0.0
    inter_token_times: List[float] = field(default_factory=list)
    total_response_time: float = 0.0
    tokens_generated: int = 0
    heartbeat_frequency: Optional[float] = None  # Detected periodic pattern
    timing_fingerprint: Optional[str] = None
    anomalies: List[Tuple[int, float]] = field(default_factory=list)  # (token_idx, time)


@dataclass
class EmbeddingAnalysis:
    """Embedding space analysis results"""
    semantic_clusters: int = 0
    cluster_boundaries: List[float] = field(default_factory=list)
    embedding_variance: float = 0.0
    semantic_drift: float = 0.0
    dimensionality_estimate: int = 0
    pca_components: Optional[np.ndarray] = None
    tsne_projection: Optional[np.ndarray] = None


@dataclass
class AttentionPattern:
    """Attention pattern analysis"""
    attention_maps: Optional[np.ndarray] = None
    head_specialization: Dict[int, str] = field(default_factory=dict)
    attention_entropy_per_layer: List[float] = field(default_factory=list)
    cross_attention_strength: float = 0.0
    self_attention_locality: float = 0.0
    pattern_fingerprint: str = ""


@dataclass
class SegmentAnalysis:
    """Analysis results for a single segment"""
    segment_id: str
    prompt: str
    response: str
    timing: TimingProfile
    embeddings: Optional[EmbeddingAnalysis] = None
    attention: Optional[AttentionPattern] = None
    behavioral_features: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence: float = 0.0


@dataclass
class ProfileReport:
    """Complete profiling report"""
    signature: BehavioralSignature
    segments: List[SegmentAnalysis]
    model_identification: Dict[str, float]  # model_name -> confidence
    anomalies: List[Dict[str, Any]]
    visualization_data: Dict[str, Any]
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Extract behavioral features from model responses"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_order = deque(maxlen=cache_size)
        self.gradient_history = []
        
    def extract_gradient_signature(self, activations: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Extract 16-dimensional gradient-based behavioral signature.
        
        Args:
            activations: Layer index -> activation tensor
            
        Returns:
            16-dimensional signature vector
        """
        signature = np.zeros(16)
        
        if not activations:
            return signature
        
        # Sort layers
        layers = sorted(activations.keys())
        
        # 1-4: Layer-wise gradient statistics
        gradients = []
        for i in range(1, len(layers)):
            prev_act = activations[layers[i-1]]
            curr_act = activations[layers[i]]
            
            # Compute approximate gradient
            if prev_act.shape == curr_act.shape:
                grad = np.mean(np.abs(curr_act - prev_act))
            else:
                # Handle dimension mismatch
                grad = np.mean(np.abs(curr_act.flatten()[:100] - prev_act.flatten()[:100]))
            
            gradients.append(grad)
        
        if gradients:
            signature[0] = np.mean(gradients)  # Mean gradient
            signature[1] = np.std(gradients)   # Gradient variance
            signature[2] = np.max(gradients)   # Max gradient
            signature[3] = np.min(gradients)   # Min gradient
        
        # 5-8: Activation statistics
        all_acts = []
        for act in activations.values():
            all_acts.extend(act.flatten()[:1000])  # Sample for efficiency
        
        if all_acts:
            all_acts = np.array(all_acts)
            signature[4] = np.mean(all_acts)
            signature[5] = np.std(all_acts)
            signature[6] = np.percentile(all_acts, 25)
            signature[7] = np.percentile(all_acts, 75)
        
        # 9-12: Layer specialization metrics
        layer_variances = []
        for act in activations.values():
            layer_variances.append(np.var(act.flatten()[:1000]))
        
        if layer_variances:
            signature[8] = np.mean(layer_variances)
            signature[9] = np.std(layer_variances)
            signature[10] = np.max(layer_variances) / (np.mean(layer_variances) + 1e-8)
            signature[11] = len([v for v in layer_variances if v > np.mean(layer_variances)])
        
        # 13-16: Cross-layer correlation
        if len(layers) >= 2:
            correlations = []
            for i in range(len(layers) - 1):
                act1 = activations[layers[i]].flatten()[:100]
                act2 = activations[layers[i+1]].flatten()[:100]
                
                if len(act1) == len(act2):
                    corr = np.corrcoef(act1, act2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                signature[12] = np.mean(correlations)
                signature[13] = np.std(correlations)
                signature[14] = np.max(correlations)
                signature[15] = np.min(correlations)
        
        # Store in history
        self.gradient_history.append(signature)
        
        return signature
    
    def analyze_timing(self, timestamps: List[float], token_count: int) -> TimingProfile:
        """
        Analyze inter-token timing for model heartbeat detection.
        
        Args:
            timestamps: List of token generation timestamps
            token_count: Number of tokens generated
            
        Returns:
            TimingProfile with heartbeat detection
        """
        profile = TimingProfile()
        
        if not timestamps or len(timestamps) < 2:
            return profile
        
        # Calculate inter-token times
        inter_times = []
        for i in range(1, len(timestamps)):
            inter_times.append(timestamps[i] - timestamps[i-1])
        
        profile.first_token_latency = timestamps[0] if timestamps else 0
        profile.inter_token_times = inter_times
        profile.total_response_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        profile.tokens_generated = token_count
        
        # Detect heartbeat pattern using FFT
        if len(inter_times) >= 10:
            # Perform FFT to find periodic patterns
            fft_result = np.fft.fft(inter_times)
            frequencies = np.fft.fftfreq(len(inter_times))
            
            # Find dominant frequency (excluding DC component)
            magnitude = np.abs(fft_result[1:len(fft_result)//2])
            if len(magnitude) > 0:
                dominant_freq_idx = np.argmax(magnitude) + 1
                if dominant_freq_idx < len(frequencies):
                    profile.heartbeat_frequency = abs(frequencies[dominant_freq_idx])
        
        # Detect anomalies (times > 2 std dev from mean)
        if inter_times:
            mean_time = np.mean(inter_times)
            std_time = np.std(inter_times)
            
            for i, t in enumerate(inter_times):
                if abs(t - mean_time) > 2 * std_time:
                    profile.anomalies.append((i, t))
        
        # Create timing fingerprint
        if inter_times:
            # Quantize times into buckets for fingerprinting
            quantized = np.digitize(inter_times, bins=np.linspace(min(inter_times), max(inter_times), 10))
            profile.timing_fingerprint = hashlib.md5(quantized.tobytes()).hexdigest()[:16]
        
        return profile
    
    def analyze_embeddings(self, embeddings: np.ndarray) -> EmbeddingAnalysis:
        """
        Analyze embedding space for semantic boundaries.
        
        Args:
            embeddings: Token embeddings matrix (n_tokens x embedding_dim)
            
        Returns:
            EmbeddingAnalysis with semantic boundaries
        """
        analysis = EmbeddingAnalysis()
        
        if embeddings.size == 0:
            return analysis
        
        # Estimate dimensionality
        analysis.dimensionality_estimate = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
        
        # Calculate embedding variance
        analysis.embedding_variance = np.var(embeddings)
        
        # Detect semantic clusters using simple k-means approximation
        if len(embeddings) >= 3:
            # Use simple clustering based on distances
            from sklearn.cluster import KMeans
            
            # Determine optimal k using elbow method (simplified)
            max_k = min(5, len(embeddings))
            inertias = []
            
            for k in range(1, max_k + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(embeddings)
                    inertias.append(kmeans.inertia_)
                except:
                    inertias.append(float('inf'))
            
            # Find elbow point
            if len(inertias) > 2:
                # Calculate second derivative to find elbow
                second_deriv = np.diff(np.diff(inertias))
                if len(second_deriv) > 0:
                    elbow_idx = np.argmax(second_deriv) + 1
                    analysis.semantic_clusters = elbow_idx + 1
            
            # Find cluster boundaries
            if analysis.semantic_clusters > 1:
                try:
                    kmeans = KMeans(n_clusters=analysis.semantic_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)
                    
                    # Find transition points
                    for i in range(1, len(labels)):
                        if labels[i] != labels[i-1]:
                            analysis.cluster_boundaries.append(i / len(labels))
                except:
                    pass
        
        # Calculate semantic drift
        if len(embeddings) >= 2:
            # Measure how embeddings change over sequence
            drifts = []
            window_size = min(5, len(embeddings) // 2)
            
            for i in range(window_size, len(embeddings) - window_size):
                early_window = embeddings[i-window_size:i]
                late_window = embeddings[i:i+window_size]
                
                drift = np.linalg.norm(np.mean(late_window, axis=0) - np.mean(early_window, axis=0))
                drifts.append(drift)
            
            if drifts:
                analysis.semantic_drift = np.mean(drifts)
        
        # PCA for dimensionality reduction (store first 3 components)
        if len(embeddings) >= 3 and embeddings.shape[1] >= 3:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, embeddings.shape[1]))
                analysis.pca_components = pca.fit_transform(embeddings)
            except:
                pass
        
        return analysis
    
    def extract_attention_patterns(self, attention_weights: Dict[int, np.ndarray]) -> AttentionPattern:
        """
        Extract attention patterns from transformer architectures.
        
        Args:
            attention_weights: Layer index -> attention weight matrix
            
        Returns:
            AttentionPattern analysis
        """
        pattern = AttentionPattern()
        
        if not attention_weights:
            return pattern
        
        # Calculate entropy per layer
        for layer_idx, weights in attention_weights.items():
            if weights.size == 0:
                continue
            
            # Flatten and normalize
            flat_weights = weights.flatten()
            flat_weights = flat_weights / (np.sum(flat_weights) + 1e-8)
            
            # Calculate entropy
            entropy = -np.sum(flat_weights * np.log(flat_weights + 1e-8))
            pattern.attention_entropy_per_layer.append(entropy)
        
        # Analyze head specialization (simplified)
        if attention_weights:
            # Assume weights are (heads, seq_len, seq_len) or similar
            for layer_idx, weights in attention_weights.items():
                if len(weights.shape) >= 3:
                    n_heads = weights.shape[0]
                    
                    for head_idx in range(min(n_heads, 8)):  # Analyze first 8 heads
                        head_weights = weights[head_idx]
                        
                        # Classify head type based on pattern
                        diagonal_strength = 0
                        if head_weights.shape[0] == head_weights.shape[1]:
                            diagonal_strength = np.mean(np.diag(head_weights))
                        
                        avg_attention = np.mean(head_weights)
                        
                        if diagonal_strength > 0.5:
                            pattern.head_specialization[head_idx] = "positional"
                        elif avg_attention > 0.3:
                            pattern.head_specialization[head_idx] = "global"
                        else:
                            pattern.head_specialization[head_idx] = "local"
        
        # Calculate cross-attention strength
        cross_attention_values = []
        for weights in attention_weights.values():
            if weights.size > 0:
                # Measure off-diagonal strength
                if len(weights.shape) >= 2 and weights.shape[0] == weights.shape[1]:
                    mask = ~np.eye(weights.shape[0], dtype=bool)
                    off_diagonal = weights[mask]
                    cross_attention_values.append(np.mean(off_diagonal))
        
        if cross_attention_values:
            pattern.cross_attention_strength = np.mean(cross_attention_values)
        
        # Calculate self-attention locality
        locality_scores = []
        for weights in attention_weights.values():
            if len(weights.shape) >= 2 and weights.shape[0] == weights.shape[1]:
                # Measure how much attention focuses on nearby positions
                n = weights.shape[0]
                for i in range(n):
                    for j in range(n):
                        distance = abs(i - j)
                        weight = weights[i, j]
                        locality_scores.append(weight * np.exp(-distance / n))
        
        if locality_scores:
            pattern.self_attention_locality = np.mean(locality_scores)
        
        # Create attention fingerprint
        if pattern.attention_entropy_per_layer:
            fingerprint_data = np.array(pattern.attention_entropy_per_layer + 
                                       [pattern.cross_attention_strength, 
                                        pattern.self_attention_locality])
            pattern.pattern_fingerprint = hashlib.md5(fingerprint_data.tobytes()).hexdigest()[:16]
        
        return pattern


class MultiSignalIntegrator:
    """Integrate multiple behavioral signals for robust identification"""
    
    def __init__(self):
        self.signal_weights = {
            SignalType.GRADIENT: 0.25,
            SignalType.TIMING: 0.20,
            SignalType.EMBEDDING: 0.20,
            SignalType.ATTENTION: 0.15,
            SignalType.CAPABILITY: 0.10,
            SignalType.SEMANTIC: 0.05,
            SignalType.STATISTICAL: 0.05
        }
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.95,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MEDIUM: 0.70,
            ConfidenceLevel.LOW: 0.50,
            ConfidenceLevel.VERY_LOW: 0.30
        }
        
    def integrate_signals(self, signals: Dict[SignalType, Any]) -> Tuple[str, float]:
        """
        Combine multiple signals using weighted voting.
        
        Args:
            signals: Dictionary of signal type to signal data
            
        Returns:
            (model_identification, confidence_score)
        """
        votes = defaultdict(float)
        total_weight = 0
        
        # Process each signal
        for signal_type, signal_data in signals.items():
            if signal_type not in self.signal_weights:
                continue
            
            weight = self.signal_weights[signal_type]
            
            # Extract model prediction from signal
            model_pred, signal_conf = self._process_signal(signal_type, signal_data)
            
            if model_pred:
                votes[model_pred] += weight * signal_conf
                total_weight += weight
        
        if not votes or total_weight == 0:
            return "unknown", 0.0
        
        # Normalize votes
        for model in votes:
            votes[model] /= total_weight
        
        # Get best prediction
        best_model = max(votes, key=votes.get)
        confidence = votes[best_model]
        
        return best_model, confidence
    
    def _process_signal(self, signal_type: SignalType, signal_data: Any) -> Tuple[str, float]:
        """Process individual signal to extract model prediction"""
        
        if signal_type == SignalType.GRADIENT:
            return self._process_gradient_signal(signal_data)
        elif signal_type == SignalType.TIMING:
            return self._process_timing_signal(signal_data)
        elif signal_type == SignalType.EMBEDDING:
            return self._process_embedding_signal(signal_data)
        elif signal_type == SignalType.ATTENTION:
            return self._process_attention_signal(signal_data)
        else:
            return "unknown", 0.0
    
    def _process_gradient_signal(self, signature: np.ndarray) -> Tuple[str, float]:
        """Process gradient signature for model identification"""
        # Simplified: use signature patterns
        
        # Known model signatures (simplified)
        known_signatures = {
            "gpt-3.5": np.array([0.5, 0.2, 0.8, 0.1] * 4),
            "gpt-4": np.array([0.6, 0.3, 0.9, 0.2] * 4),
            "claude": np.array([0.4, 0.4, 0.7, 0.3] * 4),
            "llama": np.array([0.5, 0.3, 0.6, 0.2] * 4)
        }
        
        best_match = None
        best_similarity = 0
        
        for model, known_sig in known_signatures.items():
            # Calculate cosine similarity
            similarity = np.dot(signature, known_sig) / (np.linalg.norm(signature) * np.linalg.norm(known_sig) + 1e-8)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = model
        
        return best_match or "unknown", best_similarity
    
    def _process_timing_signal(self, timing: TimingProfile) -> Tuple[str, float]:
        """Process timing profile for model identification"""
        
        # Model-specific timing patterns
        if timing.heartbeat_frequency:
            if 0.8 < timing.heartbeat_frequency < 1.2:
                return "gpt-3.5", 0.7
            elif 0.5 < timing.heartbeat_frequency < 0.8:
                return "gpt-4", 0.7
            elif 1.2 < timing.heartbeat_frequency < 1.5:
                return "claude", 0.6
        
        # Check token generation rate
        if timing.tokens_generated > 0 and timing.total_response_time > 0:
            rate = timing.tokens_generated / timing.total_response_time
            
            if rate > 50:
                return "gpt-3.5", 0.5
            elif rate > 30:
                return "gpt-4", 0.5
            elif rate > 20:
                return "claude", 0.4
        
        return "unknown", 0.0
    
    def _process_embedding_signal(self, analysis: EmbeddingAnalysis) -> Tuple[str, float]:
        """Process embedding analysis for model identification"""
        
        # Check dimensionality patterns
        if analysis.dimensionality_estimate > 0:
            if 700 < analysis.dimensionality_estimate < 800:
                return "gpt-3.5", 0.6
            elif 1000 < analysis.dimensionality_estimate < 1200:
                return "gpt-4", 0.6
            elif 800 < analysis.dimensionality_estimate < 1000:
                return "claude", 0.5
        
        # Check semantic clustering
        if analysis.semantic_clusters > 0:
            if analysis.semantic_clusters >= 4:
                return "gpt-4", 0.4
            elif analysis.semantic_clusters >= 3:
                return "claude", 0.4
        
        return "unknown", 0.0
    
    def _process_attention_signal(self, pattern: AttentionPattern) -> Tuple[str, float]:
        """Process attention pattern for model identification"""
        
        # Check attention entropy patterns
        if pattern.attention_entropy_per_layer:
            avg_entropy = np.mean(pattern.attention_entropy_per_layer)
            
            if avg_entropy > 3.0:
                return "gpt-4", 0.5
            elif avg_entropy > 2.5:
                return "claude", 0.5
            elif avg_entropy > 2.0:
                return "gpt-3.5", 0.4
        
        # Check head specialization
        if pattern.head_specialization:
            global_heads = sum(1 for h in pattern.head_specialization.values() if h == "global")
            
            if global_heads >= 3:
                return "gpt-4", 0.4
            elif global_heads >= 2:
                return "claude", 0.3
        
        return "unknown", 0.0
    
    def calculate_confidence(self, signals: Dict[SignalType, Any]) -> ConfidenceLevel:
        """Calculate overall confidence level"""
        
        # Count available signals
        available_signals = sum(1 for s in signals.values() if s is not None)
        total_signals = len(SignalType)
        
        signal_coverage = available_signals / total_signals
        
        # Get integration confidence
        _, confidence = self.integrate_signals(signals)
        
        # Combined confidence
        combined = (signal_coverage + confidence) / 2
        
        # Map to confidence level
        for level in ConfidenceLevel:
            if combined >= self.confidence_thresholds[level]:
                return level
        
        return ConfidenceLevel.VERY_LOW


class StreamingAnalyzer:
    """Real-time streaming analysis for segment-wise execution"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./behavioral_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.segment_buffer = deque(maxlen=100)
        self.incremental_signature = BehavioralSignature()
        self.segment_count = 0
        self.start_time = time.time()
        
        # Pattern cache
        self.pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def analyze_segment(self, segment_data: Dict[str, Any]) -> SegmentAnalysis:
        """
        Analyze a single segment in <100ms.
        
        Args:
            segment_data: Segment data including prompt, response, activations
            
        Returns:
            SegmentAnalysis results
        """
        start = time.time()
        
        # Check cache first
        segment_hash = self._hash_segment(segment_data)
        if segment_hash in self.pattern_cache:
            self.cache_hits += 1
            cached = self.pattern_cache[segment_hash]
            cached.processing_time = time.time() - start
            return cached
        
        self.cache_misses += 1
        
        # Create analysis
        analysis = SegmentAnalysis(
            segment_id=segment_data.get("id", f"seg_{self.segment_count}"),
            prompt=segment_data.get("prompt", ""),
            response=segment_data.get("response", "")
        )
        
        # Extract features efficiently
        extractor = FeatureExtractor()
        
        # Timing analysis (if available)
        if "timestamps" in segment_data:
            analysis.timing = extractor.analyze_timing(
                segment_data["timestamps"],
                segment_data.get("token_count", 0)
            )
        
        # Embedding analysis (if available)
        if "embeddings" in segment_data:
            analysis.embeddings = extractor.analyze_embeddings(
                segment_data["embeddings"]
            )
        
        # Attention analysis (if available)
        if "attention_weights" in segment_data:
            analysis.attention = extractor.extract_attention_patterns(
                segment_data["attention_weights"]
            )
        
        # Extract behavioral features
        if "activations" in segment_data:
            gradient_sig = extractor.extract_gradient_signature(segment_data["activations"])
            
            # Store as behavioral features
            for i, val in enumerate(gradient_sig):
                analysis.behavioral_features[f"gradient_{i}"] = val
        
        # Calculate confidence
        available_analyses = sum([
            analysis.timing is not None,
            analysis.embeddings is not None,
            analysis.attention is not None,
            len(analysis.behavioral_features) > 0
        ])
        
        analysis.confidence = available_analyses / 4.0
        analysis.processing_time = time.time() - start
        
        # Update incremental signature
        self._update_incremental_signature(analysis)
        
        # Add to buffer
        self.segment_buffer.append(analysis)
        self.segment_count += 1
        
        # Cache result
        self.pattern_cache[segment_hash] = analysis
        
        # Ensure under 100ms
        if analysis.processing_time > 0.1:
            warnings.warn(f"Segment analysis took {analysis.processing_time:.3f}s (>100ms target)")
        
        return analysis
    
    def _hash_segment(self, segment_data: Dict[str, Any]) -> str:
        """Create hash for segment caching"""
        # Create a simplified representation for hashing
        key_parts = [
            segment_data.get("prompt", "")[:100],
            segment_data.get("response", "")[:100],
            str(segment_data.get("token_count", 0))
        ]
        
        key = "|".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _update_incremental_signature(self, analysis: SegmentAnalysis):
        """Update incremental behavioral signature"""
        
        # Update timing components
        if analysis.timing:
            # Exponential moving average
            alpha = 0.1
            self.incremental_signature.response_latency = (
                alpha * analysis.timing.first_token_latency + 
                (1 - alpha) * self.incremental_signature.response_latency
            )
            
            if analysis.timing.inter_token_times:
                variance = np.var(analysis.timing.inter_token_times)
                self.incremental_signature.latency_variance = (
                    alpha * variance + 
                    (1 - alpha) * self.incremental_signature.latency_variance
                )
        
        # Update embedding components
        if analysis.embeddings:
            self.incremental_signature.semantic_coherence = (
                0.1 * (1.0 / (analysis.embeddings.semantic_drift + 1)) +
                0.9 * self.incremental_signature.semantic_coherence
            )
        
        # Update from behavioral features
        if "gradient_0" in analysis.behavioral_features:
            self.incremental_signature.response_variability = (
                0.1 * analysis.behavioral_features["gradient_0"] +
                0.9 * self.incremental_signature.response_variability
            )
    
    def get_incremental_signature(self) -> BehavioralSignature:
        """Get current incremental signature"""
        return self.incremental_signature
    
    def save_cache(self):
        """Save pattern cache to disk"""
        cache_file = self.cache_dir / "pattern_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self.pattern_cache, f)
    
    def load_cache(self):
        """Load pattern cache from disk"""
        cache_file = self.cache_dir / "pattern_cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.pattern_cache = pickle.load(f)


class BehaviorProfiler:
    """Main behavioral profiling system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_extractor = FeatureExtractor()
        self.signal_integrator = MultiSignalIntegrator()
        self.streaming_analyzer = StreamingAnalyzer()
        
        # Threading for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Known model database (simplified)
        self.known_models = self._load_model_database()
        
    def _load_model_database(self) -> Dict[str, BehavioralSignature]:
        """Load database of known model signatures"""
        # In production, this would load from a file
        known = {}
        
        # Example signatures
        known["gpt-3.5"] = BehavioralSignature(
            response_variability=0.3,
            semantic_coherence=0.8,
            attention_entropy=2.5,
            layer_specialization=0.6,
            response_latency=0.5,
            latency_variance=0.1,
            token_rate=50,
            model_family="gpt",
            version_estimate="3.5"
        )
        
        known["gpt-4"] = BehavioralSignature(
            response_variability=0.2,
            semantic_coherence=0.9,
            attention_entropy=3.0,
            layer_specialization=0.8,
            response_latency=0.8,
            latency_variance=0.15,
            token_rate=30,
            model_family="gpt",
            version_estimate="4.0"
        )
        
        known["claude"] = BehavioralSignature(
            response_variability=0.25,
            semantic_coherence=0.85,
            attention_entropy=2.8,
            layer_specialization=0.7,
            response_latency=0.6,
            latency_variance=0.12,
            token_rate=40,
            model_family="claude",
            version_estimate="2.0"
        )
        
        return known
    
    def profile_model(self, segments: List[Dict[str, Any]], 
                     parallel: bool = True) -> ProfileReport:
        """
        Complete behavioral profiling of model.
        
        Args:
            segments: List of segment data to analyze
            parallel: Whether to use parallel processing
            
        Returns:
            Complete ProfileReport
        """
        start_time = time.time()
        
        # Analyze segments
        if parallel:
            segment_analyses = self._parallel_analyze_segments(segments)
        else:
            segment_analyses = [self.streaming_analyzer.analyze_segment(s) for s in segments]
        
        # Extract signals from analyses
        signals = self._extract_signals(segment_analyses)
        
        # Integrate signals for identification
        model_id, confidence = self.signal_integrator.integrate_signals(signals)
        
        # Build final signature
        signature = self._build_signature(segment_analyses, signals)
        signature.confidence = confidence
        
        # Identify model
        model_identification = self._identify_model(signature)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(segment_analyses)
        
        # Generate visualization data
        viz_data = self._generate_visualization_data(segment_analyses, signature)
        
        # Create report
        report = ProfileReport(
            signature=signature,
            segments=segment_analyses,
            model_identification=model_identification,
            anomalies=anomalies,
            visualization_data=viz_data,
            metadata={
                "processing_time": time.time() - start_time,
                "segments_analyzed": len(segments),
                "cache_hits": self.streaming_analyzer.cache_hits,
                "cache_misses": self.streaming_analyzer.cache_misses,
                "confidence_level": self.signal_integrator.calculate_confidence(signals).name
            }
        )
        
        return report
    
    def _parallel_analyze_segments(self, segments: List[Dict[str, Any]]) -> List[SegmentAnalysis]:
        """Analyze segments in parallel"""
        futures = []
        
        for segment in segments:
            future = self.executor.submit(self.streaming_analyzer.analyze_segment, segment)
            futures.append(future)
        
        analyses = []
        for future in as_completed(futures):
            try:
                analysis = future.result(timeout=1.0)
                analyses.append(analysis)
            except Exception as e:
                warnings.warn(f"Segment analysis failed: {e}")
        
        return analyses
    
    def _extract_signals(self, analyses: List[SegmentAnalysis]) -> Dict[SignalType, Any]:
        """Extract signals from segment analyses"""
        signals = {}
        
        # Gradient signal
        gradient_features = []
        for analysis in analyses:
            if analysis.behavioral_features:
                features = [analysis.behavioral_features.get(f"gradient_{i}", 0) for i in range(16)]
                gradient_features.append(features)
        
        if gradient_features:
            signals[SignalType.GRADIENT] = np.mean(gradient_features, axis=0)
        
        # Timing signal
        timing_profiles = [a.timing for a in analyses if a.timing]
        if timing_profiles:
            signals[SignalType.TIMING] = timing_profiles[0]  # Use first for simplicity
        
        # Embedding signal
        embedding_analyses = [a.embeddings for a in analyses if a.embeddings]
        if embedding_analyses:
            signals[SignalType.EMBEDDING] = embedding_analyses[0]
        
        # Attention signal
        attention_patterns = [a.attention for a in analyses if a.attention]
        if attention_patterns:
            signals[SignalType.ATTENTION] = attention_patterns[0]
        
        return signals
    
    def _build_signature(self, analyses: List[SegmentAnalysis], 
                        signals: Dict[SignalType, Any]) -> BehavioralSignature:
        """Build behavioral signature from analyses"""
        
        # Start with incremental signature
        signature = self.streaming_analyzer.get_incremental_signature()
        
        # Enhance with aggregated data
        if SignalType.GRADIENT in signals:
            gradient = signals[SignalType.GRADIENT]
            signature.response_variability = gradient[0]
            signature.attention_entropy = gradient[2]
            signature.layer_specialization = gradient[8]
        
        if SignalType.TIMING in signals:
            timing = signals[SignalType.TIMING]
            signature.response_latency = timing.first_token_latency
            if timing.inter_token_times:
                signature.latency_variance = np.var(timing.inter_token_times)
                signature.token_rate = len(timing.inter_token_times) / sum(timing.inter_token_times)
        
        if SignalType.EMBEDDING in signals:
            embedding = signals[SignalType.EMBEDDING]
            signature.semantic_coherence = 1.0 / (embedding.semantic_drift + 1)
            signature.embedding_dim_estimate = embedding.dimensionality_estimate
        
        if SignalType.ATTENTION in signals:
            attention = signals[SignalType.ATTENTION]
            if attention.attention_entropy_per_layer:
                signature.attention_entropy = np.mean(attention.attention_entropy_per_layer)
        
        return signature
    
    def _identify_model(self, signature: BehavioralSignature) -> Dict[str, float]:
        """Identify model based on signature"""
        identification = {}
        
        # Compare with known models
        for model_name, known_sig in self.known_models.items():
            distance = signature.distance(known_sig)
            
            # Convert distance to similarity (0-1)
            similarity = 1.0 / (1.0 + distance)
            identification[model_name] = similarity
        
        # Normalize scores
        total = sum(identification.values())
        if total > 0:
            for model in identification:
                identification[model] /= total
        
        return identification
    
    def _detect_anomalies(self, analyses: List[SegmentAnalysis]) -> List[Dict[str, Any]]:
        """Detect anomalous behaviors"""
        anomalies = []
        
        # Check timing anomalies
        for analysis in analyses:
            if analysis.timing and analysis.timing.anomalies:
                for token_idx, time in analysis.timing.anomalies:
                    anomalies.append({
                        "type": "timing_anomaly",
                        "segment": analysis.segment_id,
                        "token_index": token_idx,
                        "value": time,
                        "severity": "medium"
                    })
        
        # Check confidence anomalies
        confidences = [a.confidence for a in analyses]
        if confidences:
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            for analysis in analyses:
                if abs(analysis.confidence - mean_conf) > 2 * std_conf:
                    anomalies.append({
                        "type": "confidence_anomaly",
                        "segment": analysis.segment_id,
                        "confidence": analysis.confidence,
                        "expected": mean_conf,
                        "severity": "low"
                    })
        
        return anomalies
    
    def _generate_visualization_data(self, analyses: List[SegmentAnalysis], 
                                    signature: BehavioralSignature) -> Dict[str, Any]:
        """Generate data for visualization"""
        viz_data = {}
        
        # Behavioral heatmap data
        heatmap_data = []
        for analysis in analyses:
            if analysis.behavioral_features:
                features = [analysis.behavioral_features.get(f"gradient_{i}", 0) for i in range(16)]
                heatmap_data.append(features)
        
        if heatmap_data:
            viz_data["behavioral_heatmap"] = np.array(heatmap_data).tolist()
        
        # Fingerprint comparison matrix
        comparison_matrix = []
        sig_vector = signature.to_vector()
        
        for model_name, known_sig in self.known_models.items():
            known_vector = known_sig.to_vector()
            
            # Calculate element-wise similarity
            similarities = 1.0 - np.abs(sig_vector - known_vector) / (np.abs(sig_vector) + np.abs(known_vector) + 1e-8)
            comparison_matrix.append({
                "model": model_name,
                "similarities": similarities.tolist()
            })
        
        viz_data["fingerprint_comparison"] = comparison_matrix
        
        # Timing visualization
        if any(a.timing for a in analyses):
            timing_data = []
            for analysis in analyses:
                if analysis.timing and analysis.timing.inter_token_times:
                    timing_data.extend(analysis.timing.inter_token_times)
            
            if timing_data:
                viz_data["timing_distribution"] = {
                    "values": timing_data,
                    "mean": np.mean(timing_data),
                    "std": np.std(timing_data),
                    "min": np.min(timing_data),
                    "max": np.max(timing_data)
                }
        
        # Confidence over time
        viz_data["confidence_timeline"] = [
            {"segment": i, "confidence": a.confidence} 
            for i, a in enumerate(analyses)
        ]
        
        return viz_data
    
    def export_report(self, report: ProfileReport, format: str = "json") -> str:
        """
        Export report in specified format.
        
        Args:
            report: ProfileReport to export
            format: Export format (json, markdown, html)
            
        Returns:
            Exported report as string
        """
        if format == "json":
            return self._export_json(report)
        elif format == "markdown":
            return self._export_markdown(report)
        elif format == "html":
            return self._export_html(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, report: ProfileReport) -> str:
        """Export report as JSON"""
        data = {
            "signature": asdict(report.signature),
            "model_identification": report.model_identification,
            "anomalies": report.anomalies,
            "visualization_data": report.visualization_data,
            "metadata": report.metadata,
            "segments": len(report.segments)
        }
        
        return json.dumps(data, indent=2)
    
    def _export_markdown(self, report: ProfileReport) -> str:
        """Export report as Markdown"""
        md = []
        md.append("# Behavioral Profile Report")
        md.append("")
        
        # Model identification
        md.append("## Model Identification")
        for model, conf in sorted(report.model_identification.items(), key=lambda x: x[1], reverse=True):
            md.append(f"- **{model}**: {conf:.2%} confidence")
        md.append("")
        
        # Behavioral signature
        md.append("## Behavioral Signature")
        sig = report.signature
        md.append(f"- Response Variability: {sig.response_variability:.3f}")
        md.append(f"- Semantic Coherence: {sig.semantic_coherence:.3f}")
        md.append(f"- Attention Entropy: {sig.attention_entropy:.3f}")
        md.append(f"- Token Rate: {sig.token_rate:.1f} tokens/sec")
        md.append("")
        
        # Anomalies
        if report.anomalies:
            md.append("## Anomalies Detected")
            for anomaly in report.anomalies[:5]:
                md.append(f"- {anomaly['type']} in segment {anomaly['segment']}")
        md.append("")
        
        # Metadata
        md.append("## Analysis Metadata")
        md.append(f"- Segments Analyzed: {report.metadata['segments_analyzed']}")
        md.append(f"- Processing Time: {report.metadata['processing_time']:.2f}s")
        md.append(f"- Confidence Level: {report.metadata['confidence_level']}")
        
        return "\n".join(md)
    
    def _export_html(self, report: ProfileReport) -> str:
        """Export report as HTML with visualizations"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>Behavioral Profile Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append(".heatmap { display: grid; grid-template-columns: repeat(16, 30px); gap: 2px; }")
        html.append(".cell { width: 30px; height: 30px; }")
        html.append("</style>")
        html.append("</head><body>")
        
        html.append("<h1>Behavioral Profile Report</h1>")
        
        # Model identification
        html.append("<h2>Model Identification</h2>")
        html.append("<ul>")
        for model, conf in sorted(report.model_identification.items(), key=lambda x: x[1], reverse=True):
            html.append(f"<li><strong>{model}</strong>: {conf:.2%} confidence</li>")
        html.append("</ul>")
        
        # Behavioral heatmap
        if "behavioral_heatmap" in report.visualization_data:
            html.append("<h2>Behavioral Heatmap</h2>")
            html.append('<div class="heatmap">')
            
            heatmap = report.visualization_data["behavioral_heatmap"]
            if heatmap:
                # Flatten and normalize
                flat_values = [val for row in heatmap for val in row]
                min_val = min(flat_values)
                max_val = max(flat_values)
                
                for row in heatmap:
                    for val in row:
                        # Normalize to 0-255 for color
                        normalized = int(255 * (val - min_val) / (max_val - min_val + 1e-8))
                        color = f"rgb({normalized}, {100}, {255-normalized})"
                        html.append(f'<div class="cell" style="background-color: {color}"></div>')
            
            html.append("</div>")
        
        html.append("</body></html>")
        
        return "\n".join(html)


def integrate_with_rev_pipeline(pipeline):
    """
    Integration function for REV pipeline.
    
    This should be called from rev_pipeline.py's run_behavioral_analysis() method.
    """
    profiler = BehaviorProfiler()
    
    def run_behavioral_analysis(segments: List[Dict[str, Any]]) -> ProfileReport:
        """Run behavioral analysis on segments"""
        return profiler.profile_model(segments, parallel=True)
    
    # Attach to pipeline
    pipeline.run_behavioral_analysis = run_behavioral_analysis
    
    return profiler


# Example usage
if __name__ == "__main__":
    # Create profiler
    profiler = BehaviorProfiler()
    
    # Example segment data
    segments = [
        {
            "id": "seg_001",
            "prompt": "What is your architecture?",
            "response": "I am a large language model...",
            "timestamps": [0.0, 0.1, 0.15, 0.2, 0.25],
            "token_count": 5,
            "activations": {
                0: np.random.randn(100),
                1: np.random.randn(100),
                2: np.random.randn(100)
            }
        },
        {
            "id": "seg_002",
            "prompt": "Explain your training process.",
            "response": "My training involved...",
            "timestamps": [0.0, 0.12, 0.18, 0.23],
            "token_count": 4,
            "embeddings": np.random.randn(10, 768),
            "attention_weights": {
                0: np.random.rand(8, 10, 10)
            }
        }
    ]
    
    # Profile model
    print("Running behavioral profiling...")
    report = profiler.profile_model(segments)
    
    # Export results
    print("\n" + "="*60)
    print("BEHAVIORAL PROFILE REPORT")
    print("="*60)
    
    print("\nModel Identification:")
    for model, conf in sorted(report.model_identification.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {conf:.2%}")
    
    print(f"\nConfidence Level: {report.metadata['confidence_level']}")
    print(f"Processing Time: {report.metadata['processing_time']:.3f}s")
    
    # Export as markdown
    md_report = profiler.export_report(report, format="markdown")
    print("\nMarkdown Report Preview:")
    print(md_report[:500] + "...")
    
    print("\n✓ Behavioral profiling complete!")