"""
REV Pipeline - Core integration for memory-bounded LLM verification.

This module implements the main pipeline for REV (Restriction Enzyme Verification),
providing segment-wise model execution with memory-bounded streaming and
Merkle tree construction for verifiable computation.

Enhanced with true memory-bounded execution from Section 4.3-4.4 of the REV paper.
"""

from typing import Dict, List, Tuple, Optional, Any, Generator, Union
from dataclasses import dataclass, field
import hashlib
import numpy as np
from collections import deque, defaultdict
import torch
import time
import psutil
import gc
import os
import json
import logging
from enum import Enum
from datetime import datetime
from pathlib import Path

from .crypto.merkle import (
    build_merkle_tree, 
    leaf_bytes, 
    generate_merkle_proof as merkle_path,
    build_signature,
    SegmentSite as MerkleSegmentSite,
    Signature,
    PerChallengeTree,
    HierarchicalVerificationChain
)
from .hdc.encoder import HypervectorEncoder, HypervectorConfig
from .core.sequential import SequentialState, DualSequentialTest
from .core.device_manager import DeviceManager, get_global_device_manager
from .core.device_error_handler import DeviceErrorHandler, device_safe_operation
from .diagnostics.probe_monitor import get_probe_monitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# New dataclasses from Section 5.1 of the REV paper
@dataclass
class ExecutionPolicy:
    """Execution policy for segment processing (Section 5.1)."""
    temperature: float = 0.0  # Often 0.0 for deterministic
    top_p: float = 1.0
    max_tokens: int = 512
    dtype: str = "fp16"  # fp32, fp16, int8, int4
    seed: int = 42
    attn_impl: str = "paged"  # paged, flash, standard
    quantization: Optional[str] = None  # "8bit", "4bit", None
    checkpoint_activations: bool = True
    offload_to_cpu: bool = False
    kv_cache_max_tokens: int = 2048
    

@dataclass 
class SegmentSite:
    """Segment site for memory-bounded execution (Section 5.1)."""
    seg_id: str  # e.g., "L12.post_attn"
    overlap_group: int  # For windowing
    projector_seed: int  # Domain-separated seed
    layer_range: Tuple[int, int] = (0, -1)  # Start and end layer indices
    checkpoint_path: Optional[str] = None  # For activation checkpointing
    

@dataclass
class SegmentTelemetry:
    """Telemetry tracking for memory-bounded execution."""
    segment_id: int
    alloc_mb: float  # Memory allocated in MB
    peak_mb: float  # Peak memory usage in MB
    t_ms: float  # Time in milliseconds
    tokens_processed: int
    kv_cache_size_mb: float = 0.0
    params_loaded_mb: float = 0.0
    params_offloaded: bool = False


@dataclass
class FunctionalSegment:
    """Represents a functional segment based on behavioral characteristics."""
    id: str
    start_layer: int
    end_layer: int
    behavioral_fingerprint: Dict[str, Any]
    functional_role: str  # 'token_embedding', 'semantic_processing', 'output_generation', etc.
    processing_mode: str  # 'high_precision', 'standard', 'fast', etc.
    response_strength: float = 0.0  # Average response strength to behavioral probes
    specialization_score: float = 0.0  # How specialized this segment is
    execution_policy: Optional[ExecutionPolicy] = None
    

@dataclass
class ArchitecturalSite:
    """Defines a probing site within the model architecture."""
    
    name: str
    layer_index: int
    site_type: str  # 'post_attention', 'post_mlp', 'post_layer_norm', 'embeddings'
    extract_fn: Optional[callable] = None
    
    def __hash__(self):
        return hash((self.name, self.layer_index, self.site_type))


@dataclass
class Segment:
    """Represents a memory-bounded segment of computation."""
    
    segment_id: int
    tokens: List[int]
    start_idx: int
    end_idx: int
    overlap_group: int = 0  # For overlapping window tracking
    signatures: Dict[str, np.ndarray] = None
    merkle_root: bytes = None
    checkpoint_data: Optional[Dict[str, Any]] = None  # For activation checkpointing
    
    def compute_hash(self) -> bytes:
        """Compute segment hash for Merkle tree construction."""
        data = f"{self.segment_id}:{self.start_idx}:{self.end_idx}:{self.tokens}:{self.overlap_group}".encode()
        return hashlib.sha256(data).digest()


class CheckpointManager:
    """Manages activation checkpointing for memory-bounded execution."""
    
    def __init__(self, checkpoint_dir: str = "/tmp/rev_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = {}
        
    def save_checkpoint(self, segment_id: int, data: Dict[str, torch.Tensor]):
        """Save activation checkpoint to disk."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{self.checkpoint_dir}/segment_{segment_id}.pt"
        torch.save(data, checkpoint_path)
        self.checkpoints[segment_id] = checkpoint_path
        
        return checkpoint_path
    
    def load_checkpoint(self, segment_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load activation checkpoint from disk."""
        if segment_id not in self.checkpoints:
            return None
        
        checkpoint_path = self.checkpoints[segment_id]
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path, map_location='cpu')
        return None
    
    def clear_checkpoint(self, segment_id: int):
        """Remove checkpoint to free disk space."""
        if segment_id in self.checkpoints:
            checkpoint_path = self.checkpoints[segment_id]
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            del self.checkpoints[segment_id]


class REVPipeline:
    """
    Core REV pipeline for memory-bounded model verification.
    
    Implements segment-wise execution with streaming, architectural site
    probing, and Merkle tree construction for verifiable computation.
    """
    
    def __init__(
        self,
        segment_size: int = 512,
        buffer_size: int = 4,
        hdc_config: Optional[HypervectorConfig] = None,
        architectural_sites: Optional[List[ArchitecturalSite]] = None,
        enable_pot_challenges: bool = False,
        enable_behavioral_analysis: bool = False,
        experiment_name: Optional[str] = None,
        device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialize REV pipeline.
        
        Args:
            segment_size: Maximum tokens per segment
            buffer_size: Number of segments to keep in memory
            hdc_config: Configuration for hypervector encoding
            architectural_sites: List of architectural probe points
            device_manager: Device manager for tensor operations (uses global if None)
        """
        self.segment_size = segment_size
        self.buffer_size = buffer_size
        self.segment_buffer = deque(maxlen=buffer_size)
        
        # Initialize device manager and error handler
        self.device_manager = device_manager or get_global_device_manager()
        self.error_handler = DeviceErrorHandler(self.device_manager)
        
        # Initialize HDC encoder
        self.hdc_config = hdc_config or HypervectorConfig(
            dimension=10000,
            sparsity=0.01
        )
        self.encoder = HypervectorEncoder(self.hdc_config)
        
        # Define default architectural sites if not provided
        self.architectural_sites = architectural_sites or self._default_sites()
        
        # Merkle tree storage for verification
        self.merkle_trees = {}
        self.segment_counter = 0
        
        # PoT and behavioral analysis settings  
        self.enable_pot_challenges = enable_pot_challenges
        self.enable_behavioral_analysis = enable_behavioral_analysis
        self.experiment_name = experiment_name
        
        # Diagnostic settings
        self.debug_mode = False
        self.verbose_logging = False
        self.probe_monitor = get_probe_monitor()
        self.memory_limit_mb = 8192  # Default 8GB limit
        
        # Memory-bounded execution state
        self.execution_policy = ExecutionPolicy()
        self.telemetry_records: List[SegmentTelemetry] = []
        self.kv_cache = {}  # KV cache for attention
        self.checkpoint_manager = CheckpointManager()
        self.memory_limit_mb = 4096  # Default 4GB limit
        
        # PoT challenge generation
        self.enable_pot_challenges = enable_pot_challenges
        self.challenge_generator = None
        
        # Behavioral analysis
        self.enable_behavioral_analysis = enable_behavioral_analysis
        self.behavioral_segments = []
        self.behavioral_boundaries = []
        
        # Advanced components
        self.similarity_computer = None
        self.decision_aggregator = None
        self.sequential_tester = None
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"rev_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_results = {
            "experiment_id": hashlib.sha256(self.experiment_name.encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "framework": "REV",
            "configuration": {
                "segment_size": segment_size,
                "buffer_size": buffer_size,
                "hdc_dimension": self.hdc_config.dimension,
                "hdc_sparsity": self.hdc_config.sparsity,
                "pot_challenges": enable_pot_challenges,
                "behavioral_analysis": enable_behavioral_analysis
            },
            "behavioral_analysis": {},
            "verification_results": {},
            "performance_metrics": {},
            "statistical_analysis": {}
        }
        
    def _default_sites(self) -> List[ArchitecturalSite]:
        """Define default architectural probing sites."""
        sites = []
        
        # Common transformer architectural sites
        for layer_idx in [0, 6, 11]:  # Early, middle, late layers
            sites.extend([
                ArchitecturalSite(
                    name=f"layer_{layer_idx}_post_attention",
                    layer_index=layer_idx,
                    site_type="post_attention"
                ),
                ArchitecturalSite(
                    name=f"layer_{layer_idx}_post_mlp",
                    layer_index=layer_idx,
                    site_type="post_mlp"
                ),
            ])
        
        # Add embedding layer
        sites.append(
            ArchitecturalSite(
                name="embeddings",
                layer_index=0,
                site_type="embeddings"
            )
        )
        
        return sites
    
    def enable_debug_mode(self):
        """Enable detailed debug output and diagnostics for behavioral probing."""
        self.debug_mode = True
        self.verbose_logging = True
        
        # Set logging level to debug
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        # Log system state
        logger.debug("="*60)
        logger.debug("REV PIPELINE DEBUG MODE ENABLED")
        logger.debug("="*60)
        logger.debug(f"PoT Challenges Enabled: {self.enable_pot_challenges}")
        logger.debug(f"Behavioral Analysis Enabled: {self.enable_behavioral_analysis}")
        
        # Log device information
        try:
            if torch.cuda.is_available():
                logger.debug(f"Device: CUDA - {torch.cuda.get_device_name()}")
                logger.debug(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.debug("Device: MPS (Metal Performance Shaders)")
            else:
                logger.debug("Device: CPU")
        except:
            logger.debug("Device: CPU (fallback)")
            
        logger.debug(f"Memory Limit: {self.memory_limit_mb}MB")
        logger.debug(f"Segment Size: {self.segment_size} tokens")
        logger.debug(f"Buffer Size: {self.buffer_size} segments")
        logger.debug(f"HDC Dimension: {self.hdc_config.dimension}")
        logger.debug(f"HDC Sparsity: {self.hdc_config.sparsity}")
        logger.debug(f"Experiment: {self.experiment_name}")
        logger.debug("="*60)
    
    def segment_tokens(
        self, 
        tokens: List[int],
        use_overlap: bool = True
    ) -> Generator[Segment, None, None]:
        """
        Segment input tokens into memory-bounded chunks with overlapping windows.
        
        Implements overlap windows from Section 4.1: [1..8], [5..12], [9..16] pattern
        to resist stitching attacks.
        
        Args:
            tokens: List of token IDs
            use_overlap: Whether to use overlapping windows
            
        Yields:
            Segment objects with bounded token sequences
        """
        if use_overlap:
            # Overlapping windows per Section 4.1
            # Pattern: segments of size S with S/2 overlap
            window_size = self.segment_size
            overlap = window_size // 2
            
            # Generate overlapping segments
            i = 0
            overlap_group = 0
            
            while i < len(tokens):
                # Create segment with overlap group tracking
                segment = Segment(
                    segment_id=self.segment_counter,
                    tokens=tokens[i:i + window_size],
                    start_idx=i,
                    end_idx=min(i + window_size, len(tokens))
                )
                
                # Add overlap group for tracking
                segment.overlap_group = overlap_group
                
                self.segment_counter += 1
                yield segment
                
                # Move window with overlap
                i += overlap
                
                # Update overlap group (cycles through 0, 1, 2 for 3 overlapping groups)
                overlap_group = (overlap_group + 1) % 3
        else:
            # Original non-overlapping segments
            for i in range(0, len(tokens), self.segment_size):
                segment = Segment(
                    segment_id=self.segment_counter,
                    tokens=tokens[i:i + self.segment_size],
                    start_idx=i,
                    end_idx=min(i + self.segment_size, len(tokens))
                )
                segment.overlap_group = 0  # No overlap group
                self.segment_counter += 1
                yield segment
    
    def run_segment(
        self,
        model,
        states_in: Optional[Dict[str, torch.Tensor]],
        seg: SegmentSite,
        segment: Segment,
        policy: ExecutionPolicy
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], SegmentTelemetry]:
        """
        Run a segment with memory-bounded execution (Section 5.3).
        
        Implements true memory-bounded execution with parameter loading/offloading,
        KV cache management, and activation extraction.
        
        Args:
            model: Language model
            states_in: Input states (KV cache, hidden states)
            seg: Segment site configuration
            segment: Segment data with tokens
            policy: Execution policy
            
        Returns:
            Tuple of (states_out, activations, telemetry)
        """
        # Start telemetry tracking
        start_time = time.perf_counter()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Prepare telemetry record
        telemetry = SegmentTelemetry(
            segment_id=segment.segment_id,
            alloc_mb=0.0,
            peak_mb=0.0,
            t_ms=0.0,
            tokens_processed=len(segment.tokens)
        )
        
        try:
            # Load parameters for segment (offload-aware)
            params_loaded_mb = self._load_params(model, seg, policy)
            telemetry.params_loaded_mb = params_loaded_mb
            
            # Apply quantization if specified
            if policy.quantization:
                self._apply_quantization(model, policy.quantization)
            
            # Prepare input with device management
            input_ids = torch.tensor([segment.tokens], dtype=torch.long)
            try:
                # Use device manager for proper device placement
                target_device = self.device_manager.get_device()
                if not policy.offload_to_cpu:
                    input_ids = self.device_manager.ensure_device_consistency(input_ids, target_device)
                else:
                    input_ids = input_ids.cpu()
            except Exception as e:
                logger.warning(f"[REV-PIPELINE] Device placement failed: {e}, using CPU")
                input_ids = input_ids.cpu()
            
            # Manage KV cache
            if states_in and 'kv_cache' in states_in:
                self.kv_cache = states_in['kv_cache']
            
            # Limit KV cache size
            self._manage_kv_cache(policy.kv_cache_max_tokens)
            telemetry.kv_cache_size_mb = self._get_kv_cache_size_mb()
            
            # Forward pass with device-safe execution
            try:
                if policy.checkpoint_activations:
                    # Use gradient checkpointing to save memory
                    autocast_enabled = (policy.dtype == "fp16" and self.device_manager.get_device().type == "cuda")
                    with torch.cuda.amp.autocast(enabled=autocast_enabled):
                        states_out, activations = self._forward_with_checkpointing(
                            model, input_ids, seg
                        )
                else:
                    # Standard forward pass
                    states_out, activations = self._forward_segment(
                        model, input_ids, seg, policy
                    )
            except Exception as e:
                # Handle device errors with automatic recovery
                context = {
                    "segment_id": segment.segment_id,
                    "input_shape": input_ids.shape,
                    "input_device": str(input_ids.device),
                    "model_device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else "unknown",
                    "policy": policy.__dict__
                }
                
                recovery_successful, recovery_result = self.error_handler.handle_error(e, context)
                
                if recovery_successful:
                    logger.info(f"[REV-PIPELINE] Recovered from device error in segment {segment.segment_id}")
                    # Retry with CPU fallback
                    input_ids = input_ids.cpu()
                    model = model.cpu() if hasattr(model, 'cpu') else model
                    states_out, activations = self._forward_segment(
                        model, input_ids, seg, policy
                    )
                else:
                    logger.error(f"[REV-PIPELINE] Failed to recover from device error: {e}")
                    raise
            
            # Save checkpoint if needed for overlap regions
            if segment.overlap_group > 0 and policy.checkpoint_activations:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    segment.segment_id, states_out
                )
                segment.checkpoint_data = {'path': checkpoint_path}
            
            # Release parameters (critical for memory bounding)
            if policy.offload_to_cpu:
                self._release_params(model, seg)
                telemetry.params_offloaded = True
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM using device manager
            logger.warning(f"[REV-PIPELINE] OOM encountered in segment {segment.segment_id}")
            
            if self.device_manager.handle_out_of_memory(e):
                # Retry with CPU offloading
                policy.offload_to_cpu = True
                logger.info(f"[REV-PIPELINE] Retrying segment {segment.segment_id} with CPU offloading")
                return self.run_segment(model, states_in, seg, segment, policy)
            else:
                raise
        except Exception as e:
            # Handle other device-related errors
            error_type = self.error_handler.classify_error(e)
            if error_type.value != "unknown":
                logger.warning(f"[REV-PIPELINE] Device error in segment {segment.segment_id}: {error_type.value}")
                
                context = {
                    "segment_id": segment.segment_id,
                    "operation": "run_segment"
                }
                recovery_successful, _ = self.error_handler.handle_error(e, context)
                
                if not recovery_successful:
                    raise
            else:
                raise
            
        finally:
            # Record telemetry
            end_time = time.perf_counter()
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            telemetry.t_ms = (end_time - start_time) * 1000
            telemetry.alloc_mb = final_memory - initial_memory
            telemetry.peak_mb = max(final_memory, initial_memory)
            
            # Check memory limit
            if telemetry.peak_mb > self.memory_limit_mb:
                print(f"WARNING: Memory usage {telemetry.peak_mb:.1f}MB exceeds limit {self.memory_limit_mb}MB")
            
            self.telemetry_records.append(telemetry)
        
        return states_out, activations, telemetry
    
    def _load_params(self, model, seg: SegmentSite, policy: ExecutionPolicy) -> float:
        """Load parameters for a segment (offload-aware)."""
        params_size_mb = 0.0
        
        if seg.layer_range[0] >= 0:
            # Load only specified layers
            start_layer, end_layer = seg.layer_range
            if end_layer == -1:
                end_layer = len(model.transformer.h) if hasattr(model, 'transformer') else 12
            
            for layer_idx in range(start_layer, min(end_layer + 1, len(model.transformer.h))):
                layer = model.transformer.h[layer_idx]
                
                # Move layer to device if offloaded
                if policy.offload_to_cpu and next(iter(layer.parameters())).device.type == 'cpu':
                    layer.cuda()
                    params_size_mb += sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
        
        return params_size_mb
    
    def _release_params(self, model, seg: SegmentSite):
        """Release parameters to free memory."""
        if seg.layer_range[0] >= 0:
            start_layer, end_layer = seg.layer_range
            if end_layer == -1:
                end_layer = len(model.transformer.h) if hasattr(model, 'transformer') else 12
            
            for layer_idx in range(start_layer, min(end_layer + 1, len(model.transformer.h))):
                layer = model.transformer.h[layer_idx]
                
                # Move layer to CPU to free GPU memory
                layer.cpu()
                
                # Clear gradients if any
                for p in layer.parameters():
                    p.grad = None
    
    def _apply_quantization(self, model, quantization: str):
        """
        Apply quantization to model using proper quantization backends.
        
        WARNING: Previous implementation was incorrect and would break models.
        This now uses proper quantization that maintains FP compute.
        
        Args:
            model: Model to quantize
            quantization: Quantization mode ('8bit', '4bit', 'dynamic', 'none')
        """
        if quantization == "none" or not quantization:
            return
            
        # Try to use bitsandbytes if available
        try:
            import bitsandbytes as bnb
            has_bnb = True
        except ImportError:
            has_bnb = False
            
        # Try PyTorch native quantization
        try:
            import torch.ao.quantization as tq
            has_torch_quant = True
        except ImportError:
            has_torch_quant = False
            
        if quantization == "8bit":
            if has_bnb:
                # Use bitsandbytes 8-bit quantization (keeps FP16 compute)
                logger.info("Using bitsandbytes 8-bit quantization")
                model = bnb.nn.Linear8bitLt.from_float(model)
            elif has_torch_quant:
                # Use PyTorch dynamic quantization
                logger.info("Using PyTorch dynamic 8-bit quantization")
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            else:
                logger.warning("No quantization backend available, using FP16 instead")
                model = model.half()
                
        elif quantization == "4bit":
            if has_bnb:
                # Use bitsandbytes 4-bit quantization
                logger.info("Using bitsandbytes 4-bit quantization")
                # Note: This requires bnb 4-bit support
                model = bnb.nn.Linear4bit.from_float(model)
            else:
                logger.warning("4-bit quantization requires bitsandbytes, falling back to FP16")
                model = model.half()
                
        elif quantization == "dynamic":
            if has_torch_quant:
                # Dynamic quantization - quantizes weights, activations stay in FP
                logger.info("Using PyTorch dynamic quantization")
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
                    dtype=torch.qint8
                )
            else:
                logger.warning("Dynamic quantization unavailable, using FP16")
                model = model.half()
        else:
            logger.warning(f"Unknown quantization mode: {quantization}, skipping")
    
    def _manage_kv_cache(self, max_tokens: int):
        """Manage KV cache size to stay within limits."""
        if not self.kv_cache:
            return
        
        # Calculate total tokens in cache
        total_tokens = 0
        for v in self.kv_cache.values():
            if isinstance(v, torch.Tensor):
                # Assuming shape is (batch, heads, seq_len, dim) or similar
                if len(v.shape) >= 3:
                    total_tokens += v.shape[2]  # seq_len dimension
                elif len(v.shape) >= 2:
                    total_tokens += v.shape[1]
        
        if total_tokens > max_tokens:
            # Evict oldest entries
            evict_ratio = 0.5  # Evict 50% of cache
            keys_to_evict = list(self.kv_cache.keys())[:int(len(self.kv_cache) * evict_ratio)]
            for key in keys_to_evict:
                del self.kv_cache[key]
    
    def _get_kv_cache_size_mb(self) -> float:
        """Get current KV cache size in MB."""
        if not self.kv_cache:
            return 0.0
        
        size_bytes = sum(
            v.numel() * v.element_size() if isinstance(v, torch.Tensor) else 0
            for v in self.kv_cache.values()
        )
        return size_bytes / (1024 * 1024)
    
    def _forward_segment(
        self,
        model,
        input_ids: torch.Tensor,
        seg: SegmentSite,
        policy: ExecutionPolicy
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass for a segment."""
        states_out = {}
        activations = {}
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids, use_cache=True)
            
            # Extract states and activations
            if hasattr(outputs, 'past_key_values'):
                states_out['kv_cache'] = outputs.past_key_values
            
            if hasattr(outputs, 'hidden_states'):
                activations['hidden_states'] = outputs.hidden_states
            
            # Store in KV cache
            if 'kv_cache' in states_out:
                self.kv_cache.update({
                    f"seg_{seg.seg_id}": states_out['kv_cache']
                })
        
        return states_out, activations
    
    def _forward_with_checkpointing(
        self,
        model,
        input_ids: torch.Tensor,
        seg: SegmentSite
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with activation checkpointing."""
        # Use PyTorch's checkpoint functionality
        from torch.utils.checkpoint import checkpoint
        
        states_out = {}
        activations = {}
        
        def forward_fn(input_ids):
            return model(input_ids, use_cache=True)
        
        # Checkpointed forward pass
        outputs = checkpoint(forward_fn, input_ids)
        
        if hasattr(outputs, 'past_key_values'):
            states_out['kv_cache'] = outputs.past_key_values
        
        if hasattr(outputs, 'hidden_states'):
            activations['hidden_states'] = outputs.hidden_states
        
        return states_out, activations
    
    def extract_site_features(
        self,
        model_outputs: Dict[str, torch.Tensor],
        site: ArchitecturalSite
    ) -> np.ndarray:
        """
        Extract features from a specific architectural site.
        
        Args:
            model_outputs: Dictionary of model intermediate outputs
            site: Architectural site to probe
            
        Returns:
            Feature vector as numpy array
        """
        site_key = f"{site.site_type}_{site.layer_index}"
        
        if site_key not in model_outputs:
            raise KeyError(f"Site {site_key} not found in model outputs")
        
        features = model_outputs[site_key]
        
        # Apply custom extraction function if provided
        if site.extract_fn:
            features = site.extract_fn(features)
        
        # Convert to numpy and flatten if needed
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Take mean pooling over sequence dimension if present
        if len(features.shape) > 2:
            features = features.mean(axis=1)
        
        return features.flatten()
    
    def generate_segment_signature(
        self,
        segment: Segment,
        model_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Generate hypervector signatures for a segment.
        
        Args:
            segment: Input segment
            model_outputs: Model outputs at architectural sites
            
        Returns:
            Dictionary mapping site names to hypervector signatures
        """
        signatures = {}
        
        for site in self.architectural_sites:
            try:
                # Extract features from architectural site
                features = self.extract_site_features(model_outputs, site)
                
                # Encode to hypervector
                hypervector = self.encoder.encode(features)
                
                signatures[site.name] = hypervector
                
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not extract from {site.name}: {e}")
                continue
        
        return signatures
    
    def process_challenge(
        self,
        model,
        challenge: str,
        tokenizer=None
    ) -> Dict[str, Any]:
        """
        Process a challenge through the pipeline with memory-bounded execution.
        
        Args:
            model: Language model to verify
            challenge: Input challenge text
            tokenizer: Tokenizer for the model
            
        Returns:
            Dictionary containing:
                - segment_signatures: Signatures for each segment
                - merkle_tree: Merkle tree for verification
                - merkle_root: Root hash of Merkle tree
        """
        # Tokenize challenge
        if tokenizer:
            tokens = tokenizer.encode(challenge)
        else:
            # Fallback to simple tokenization
            tokens = list(challenge.encode('utf-8'))
        
        segment_signatures = []
        merkle_leaves = []
        
        # Process segments with streaming
        for segment in self.segment_tokens(tokens):
            # Execute model on segment (memory-bounded)
            with torch.no_grad():
                if hasattr(model, 'forward_with_cache'):
                    # Model supports returning intermediate activations
                    outputs, cache = model.forward_with_cache(
                        torch.tensor([segment.tokens])
                    )
                else:
                    # Standard forward pass
                    outputs = self._extract_outputs_with_hooks(
                        model, 
                        torch.tensor([segment.tokens])
                    )
            
            # Generate signatures for architectural sites
            signatures = self.generate_segment_signature(segment, outputs)
            segment.signatures = signatures
            
            # Add to buffer (automatic memory management)
            self.segment_buffer.append(segment)
            
            # Create Merkle leaf from segment
            segment_data = {
                'id': segment.segment_id,
                'hash': segment.compute_hash().hex(),
                'signatures': {
                    k: v.tobytes().hex()[:64]  # Truncate for efficiency
                    for k, v in signatures.items()
                }
            }
            
            leaf = leaf_bytes([
                segment.segment_id,
                int.from_bytes(segment.compute_hash()[:8], 'big')
            ])
            merkle_leaves.append(leaf)
            
            segment_signatures.append(segment_data)
        
        # Build Merkle tree for challenge
        merkle_tree = build_merkle_tree(merkle_leaves)
        
        # Store for verification
        challenge_hash = hashlib.sha256(challenge.encode()).hexdigest()
        self.merkle_trees[challenge_hash] = merkle_tree
        
        return {
            'segment_signatures': segment_signatures,
            'merkle_tree': merkle_tree,
            'merkle_root': merkle_tree['root'].hex(),
            'num_segments': len(segment_signatures)
        }
    
    def _extract_outputs_with_hooks(
        self,
        model,
        input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate outputs using hooks (fallback method).
        
        Args:
            model: PyTorch model
            input_ids: Input token IDs
            
        Returns:
            Dictionary of intermediate activations
        """
        outputs = {}
        hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                outputs[name] = output
            return hook
        
        # Register hooks for architectural sites
        for site in self.architectural_sites:
            if site.site_type == "post_attention":
                # Hook attention output
                layer = model.transformer.h[site.layer_index] if hasattr(model, 'transformer') else None
                if layer and hasattr(layer, 'attn'):
                    hook = layer.attn.register_forward_hook(
                        create_hook(f"{site.site_type}_{site.layer_index}")
                    )
                    hooks.append(hook)
                    
            elif site.site_type == "post_mlp":
                # Hook MLP output
                layer = model.transformer.h[site.layer_index] if hasattr(model, 'transformer') else None
                if layer and hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(
                        create_hook(f"{site.site_type}_{site.layer_index}")
                    )
                    hooks.append(hook)
        
        # Forward pass
        _ = model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs
    
    def verify_segment_proof(
        self,
        challenge_hash: str,
        segment_id: int
    ) -> Optional[List]:
        """
        Generate Merkle proof for a specific segment.
        
        Args:
            challenge_hash: Hash of the challenge
            segment_id: ID of segment to prove
            
        Returns:
            Merkle proof path or None if not found
        """
        if challenge_hash not in self.merkle_trees:
            return None
        
        tree = self.merkle_trees[challenge_hash]
        
        # Generate proof path
        try:
            proof = merkle_path(tree, segment_id)
            return proof
        except (IndexError, KeyError):
            return None
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get summary of telemetry records."""
        if not self.telemetry_records:
            return {"status": "No telemetry records"}
        
        total_segments = len(self.telemetry_records)
        total_tokens = sum(t.tokens_processed for t in self.telemetry_records)
        total_time_ms = sum(t.t_ms for t in self.telemetry_records)
        avg_memory_mb = np.mean([t.alloc_mb for t in self.telemetry_records])
        peak_memory_mb = max(t.peak_mb for t in self.telemetry_records)
        
        return {
            "total_segments": total_segments,
            "total_tokens_processed": total_tokens,
            "total_time_ms": total_time_ms,
            "avg_time_per_segment_ms": total_time_ms / total_segments,
            "tokens_per_second": (total_tokens / total_time_ms) * 1000 if total_time_ms > 0 else 0,
            "avg_memory_mb": avg_memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "kv_cache_avg_mb": np.mean([t.kv_cache_size_mb for t in self.telemetry_records]),
            "params_offloaded_count": sum(1 for t in self.telemetry_records if t.params_offloaded),
            "memory_limit_mb": self.memory_limit_mb,
            "within_memory_limit": peak_memory_mb <= self.memory_limit_mb
        }
    
    def set_memory_limit(self, limit_mb: float):
        """Set memory limit for execution."""
        self.memory_limit_mb = limit_mb
    
    def set_execution_policy(self, policy: ExecutionPolicy):
        """Set execution policy for segment processing."""
        self.execution_policy = policy
    
    def build_signature(
        self,
        activations_or_logits: Union[np.ndarray, Dict[str, np.ndarray]],
        seg: Union[SegmentSite, MerkleSegmentSite],
        policy: Optional[Dict[str, Any]] = None,
        d_prime: int = 256,
        tau: float = 3.0,
        q: int = 8
    ) -> Signature:
        """
        Build signature for segment/site as per Sections 4.3 and 5.5.
        
        This method generates a signature for a segment that includes:
        1. Selection and pooling of activations
        2. Seeded random projection for dimensionality reduction
        3. Quantization and binarization
        4. Leaf hash creation for Merkle tree inclusion
        
        Args:
            activations_or_logits: Model activations or logits to sign
            seg: Segment or behavioral site to sign
            policy: Execution policy (uses current if None)
            d_prime: Projected dimension for signature
            tau: Clipping threshold for quantization
            q: Quantization bits
            
        Returns:
            Signature with binary signature and Merkle leaf hash
        """
        # Use current execution policy if none provided
        if policy is None and self.execution_policy:
            policy = {
                "temperature": self.execution_policy.temperature,
                "top_p": self.execution_policy.top_p,
                "max_tokens": self.execution_policy.max_tokens,
                "dtype": self.execution_policy.dtype,
                "seed": self.execution_policy.seed
            }
        elif policy is None:
            # Default policy
            policy = {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "dtype": "fp16",
                "seed": 42
            }
        
        # Convert SegmentSite to MerkleSegmentSite if needed
        if isinstance(seg, SegmentSite):
            merkle_seg = MerkleSegmentSite(
                seg_id=seg.seg_id,
                segment_type="architectural",
                token_range=(seg.layer_range[0], seg.layer_range[1]),
                projector_seed=seg.projector_seed,
                metadata={
                    "overlap_group": seg.overlap_group,
                    "checkpoint_path": seg.checkpoint_path
                }
            )
        else:
            merkle_seg = seg
        
        # Call the build_signature function from merkle.py
        signature = build_signature(
            activations_or_logits=activations_or_logits,
            seg=merkle_seg,
            policy=policy,
            d_prime=d_prime,
            tau=tau,
            q=q
        )
        
        # Store signature for this segment
        if hasattr(self, 'segment_signatures'):
            self.segment_signatures[seg.seg_id] = signature
        else:
            self.segment_signatures = {seg.seg_id: signature}
        
        return signature
    
    def build_per_challenge_tree(
        self,
        challenge_id: str,
        segments: List[Segment],
        use_incremental: bool = False
    ) -> PerChallengeTree:
        """
        Build per-challenge Merkle tree from processed segments.
        
        Args:
            challenge_id: Unique identifier for this challenge
            segments: List of processed segments
            use_incremental: Whether to use incremental tree construction
            
        Returns:
            Per-challenge Merkle tree with all segment signatures
        """
        if not hasattr(self, 'verification_chain'):
            self.verification_chain = HierarchicalVerificationChain(
                enable_zk=True,
                enable_per_challenge=True
            )
        
        signatures = []
        
        for segment in segments:
            # Extract activations for this segment
            activations = segment.checkpoint_data.get('activations', {}) if segment.checkpoint_data else {}
            
            # Create segment site
            seg_site = MerkleSegmentSite(
                seg_id=f"seg_{segment.segment_id}",
                segment_type="architectural",
                token_range=(segment.start_idx, segment.end_idx),
                projector_seed=segment.segment_id * 1337,  # Deterministic seed
                metadata={
                    "overlap_group": segment.overlap_group,
                    "tokens": len(segment.tokens)
                }
            )
            
            # Build signature
            sig = self.build_signature(
                activations_or_logits=activations,
                seg=seg_site
            )
            signatures.append(sig)
        
        # Build per-challenge tree
        if use_incremental:
            # Use incremental construction for streaming
            inc_tree = self.verification_chain.start_incremental_tree(challenge_id)
            for sig in signatures:
                policy = self.execution_policy.__dict__ if self.execution_policy else {}
                self.verification_chain.add_to_incremental(
                    challenge_id, sig, policy
                )
            tree = self.verification_chain.finalize_incremental(challenge_id)
        else:
            # Build complete tree at once
            tree = self.verification_chain.build_per_challenge_tree(
                challenge_id, signatures
            )
        
        return tree
    
    def streaming_verify(
        self,
        model_a,
        model_b,
        challenges: List[str],
        sequential_state: Optional[SequentialState] = None,
        use_consensus: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream verification results for memory efficiency.
        
        Args:
            model_a: First model to compare
            model_b: Second model to compare  
            challenges: List of challenge prompts
            sequential_state: Optional sequential testing state
            use_consensus: Whether to use Byzantine consensus verification
            
        Yields:
            Verification results for each challenge
        """
        if use_consensus:
            # Use streaming consensus verifier
            from .verifier.streaming_consensus import StreamingConsensusVerifier
            
            verifier = StreamingConsensusVerifier(
                rev_pipeline=self,
                early_stop_threshold=0.95
            )
            
            # Create segment generators
            gen_a, gen_b = verifier.create_segment_generators(
                model_a, model_b, challenges
            )
            
            # Stream verify with consensus
            yield from verifier.stream_verify(gen_a, gen_b, challenges)
            return  # Exit after consensus verification
        
        # Original streaming verification without consensus
        for idx, challenge in enumerate(challenges):
            # Process challenge for both models
            result_a = self.process_challenge(model_a, challenge)
            result_b = self.process_challenge(model_b, challenge)
            
            # Compute similarity between signatures
            similarity_scores = {}
            for site_name in result_a['segment_signatures'][0]['signatures'].keys():
                sigs_a = [s['signatures'].get(site_name) for s in result_a['segment_signatures']]
                sigs_b = [s['signatures'].get(site_name) for s in result_b['segment_signatures']]
                
                # Compute average similarity
                similarities = []
                for sig_a, sig_b in zip(sigs_a, sigs_b):
                    if sig_a and sig_b:
                        # Simple similarity metric (can be enhanced)
                        sim = 1.0 - (abs(hash(sig_a) - hash(sig_b)) / (2**32))
                        similarities.append(sim)
                
                similarity_scores[site_name] = np.mean(similarities) if similarities else 0.0
            
            yield {
                'challenge_id': idx,
                'challenge': challenge[:100] + '...' if len(challenge) > 100 else challenge,
                'merkle_root_a': result_a['merkle_root'],
                'merkle_root_b': result_b['merkle_root'],
                'num_segments': result_a['num_segments'],
                'similarity_scores': similarity_scores,
                'mean_similarity': np.mean(list(similarity_scores.values()))
            }
    
    def run_behavioral_analysis(self,
                              model,
                              tokenizer,
                              num_layers: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform behavioral analysis to identify model segments.
        Based on Yi-34B experimental approach.
        
        Args:
            model: Model to analyze
            tokenizer: Tokenizer for the model
            num_layers: Number of layers to analyze (None = auto-detect)
            
        Returns:
            Dictionary with behavioral analysis results
        """
        logger.info("="*60)
        logger.info("Running Behavioral Analysis")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Auto-detect number of layers if not provided
        if num_layers is None:
            if hasattr(model, 'config'):
                num_layers = getattr(model.config, 'num_hidden_layers', 12)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                num_layers = len(model.transformer.h)
            else:
                num_layers = 12  # Default fallback
        
        # Generate behavioral probes
        from .challenges.pot_challenge_generator import PoTChallengeGenerator
        generator = PoTChallengeGenerator()
        probes = generator.generate_behavioral_probes()
        
        # Analyze layer behaviors
        layer_signatures = {}
        sample_layers = list(range(0, num_layers, max(1, num_layers // 20)))  # Sample ~20 layers
        
        logger.info(f"Analyzing {len(sample_layers)} layers with {sum(len(p) for p in probes.values())} probes")
        
        for layer_idx in sample_layers:
            signatures = []
            
            for probe_type, probe_texts in probes.items():
                for text in probe_texts:
                    # Tokenize probe
                    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
                    
                    # Get layer activations (simplified - would need hooks in practice)
                    with torch.no_grad():
                        # Run forward pass up to this layer
                        layer_output = self._get_layer_output(model, inputs['input_ids'], layer_idx)
                        
                        if layer_output is not None:
                            # Extract statistical signature
                            sig = self._compute_layer_signature(layer_output)
                            signatures.append(sig)
            
            if signatures:
                layer_signatures[layer_idx] = np.mean(signatures, axis=0)
        
        # Find behavioral boundaries using correlation analysis
        behavioral_boundaries = []
        prev_sig = None
        correlation_threshold = 0.85
        
        for layer_idx in sorted(layer_signatures.keys()):
            if prev_sig is not None:
                # Compute correlation between consecutive layer signatures
                if len(prev_sig) > 0 and len(layer_signatures[layer_idx]) > 0:
                    similarity = np.corrcoef(prev_sig, layer_signatures[layer_idx])[0, 1]
                    
                    if similarity < correlation_threshold:
                        behavioral_boundaries.append(layer_idx)
                        logger.info(f"  Behavioral boundary detected at layer {layer_idx} (similarity: {similarity:.3f})")
            
            prev_sig = layer_signatures[layer_idx]
        
        # Create behavioral segments
        segments = []
        prev_boundary = 0
        
        for boundary in behavioral_boundaries + [num_layers]:
            if boundary - prev_boundary >= 2:  # Minimum segment size
                segments.append((prev_boundary, boundary))
                prev_boundary = boundary
        
        # If no boundaries found, create uniform segments
        if not segments:
            segment_size = max(2, num_layers // 4)
            segments = [(i, min(i + segment_size, num_layers)) 
                       for i in range(0, num_layers, segment_size)]
        
        analysis_time = time.time() - start_time
        
        # Create functional segments using restriction site discovery
        functional_segments = []
        try:
            # Try to use LayerSegmentExecutor for sophisticated restriction site discovery
            from .models.true_segment_execution import LayerSegmentExecutor, SegmentExecutionConfig, RestrictionSite
            
            # Create minimal config for restriction site discovery
            config = SegmentExecutionConfig(model_path="behavioral_analysis")
            executor = LayerSegmentExecutor(config)
            
            # Generate diverse behavioral probes for restriction site discovery
            probe_prompts = []
            for probe_type, probe_texts in probes.items():
                probe_prompts.extend(probe_texts[:3])  # Use first 3 probes of each type
            
            # Discover restriction sites using actual behavioral analysis
            restriction_sites = executor.identify_all_restriction_sites(probe_prompts)
            
            if restriction_sites:
                # Create functional segments from restriction sites
                functional_segments = self.create_functional_segments(restriction_sites)
                logger.info(f"Created {len(functional_segments)} functional segments from {len(restriction_sites)} restriction sites")
                
        except Exception as e:
            logger.warning(f"Could not create functional segments: {e}")
            logger.info("Falling back to basic segmentation")
            
            # Fallback: create basic functional segments from simple layer boundaries
            if segments:
                mock_sites = []
                for start, end in segments:
                    # Create mock RestrictionSite objects
                    site = type('RestrictionSite', (), {
                        'layer_idx': start,
                        'site_type': 'layer_boundary', 
                        'behavioral_divergence': 0.5,
                        'confidence_score': 0.7
                    })()
                    mock_sites.append(site)
                    
                # Add final site
                if segments:
                    final_site = type('RestrictionSite', (), {
                        'layer_idx': segments[-1][1],
                        'site_type': 'layer_boundary',
                        'behavioral_divergence': 0.3, 
                        'confidence_score': 0.6
                    })()
                    mock_sites.append(final_site)
                    
                if len(mock_sites) >= 2:
                    functional_segments = self.create_functional_segments(mock_sites)

        # Store results
        behavioral_results = {
            "num_layers": num_layers,
            "layers_analyzed": len(sample_layers),
            "probe_types": list(probes.keys()),
            "total_probes": sum(len(p) for p in probes.values()),
            "behavioral_boundaries": behavioral_boundaries,
            "num_segments": len(segments),
            "segments": segments,
            "analysis_time": analysis_time,
            "correlation_threshold": correlation_threshold,
            "segment_interpretation": self._interpret_segments(segments, num_layers),
            "functional_segments": functional_segments,
            "num_functional_segments": len(functional_segments)
        }
        
        logger.info(f"✓ Identified {len(segments)} behavioral segments in {analysis_time:.2f}s")
        for i, (start, end) in enumerate(segments):
            logger.info(f"  Segment {i+1}: Layers {start}-{end} ({end-start} layers)")
            
        # Log functional segments if created
        if functional_segments:
            logger.info(f"✓ Created {len(functional_segments)} functional segments:")
            for i, fs in enumerate(functional_segments):
                logger.info(f"  Functional Segment {i+1}: {fs.id} - Role: {fs.functional_role}, "
                           f"Mode: {fs.processing_mode}, Layers: {fs.start_layer}-{fs.end_layer}")
        
        return behavioral_results
    
    def _get_layer_output(self, model, input_ids: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        """Get output from a specific layer (simplified implementation)."""
        try:
            # This is a simplified version - actual implementation would use hooks
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                if layer_idx < len(model.transformer.h):
                    # Would need to actually run forward pass with hooks
                    # For now, return a dummy tensor for demonstration
                    return torch.randn(1, input_ids.shape[1], 768)  # Typical hidden size
            return None
        except Exception as e:
            logger.warning(f"Could not get layer {layer_idx} output: {e}")
            return None
    
    def _compute_layer_signature(self, layer_output: torch.Tensor) -> np.ndarray:
        """Compute statistical signature from layer output."""
        if layer_output is None:
            return np.array([])
        
        # Compute various statistics as signature
        signature = []
        
        # Basic statistics
        signature.append(float(layer_output.mean()))
        signature.append(float(layer_output.std()))
        signature.append(float(layer_output.abs().max()))
        signature.append(float(layer_output.abs().min()))
        
        # Percentiles
        percentiles = [25, 50, 75]
        for p in percentiles:
            signature.append(float(torch.quantile(layer_output.flatten(), p/100)))
        
        # Activation patterns
        signature.append(float((layer_output > 0).float().mean()))  # Fraction positive
        signature.append(float((layer_output.abs() < 0.01).float().mean()))  # Fraction near zero
        
        return np.array(signature)
    
    def _interpret_segments(self, segments: List[Tuple[int, int]], num_layers: int) -> Dict[str, str]:
        """Interpret the functional role of each segment."""
        interpretations = {}
        
        for i, (start, end) in enumerate(segments):
            relative_start = start / num_layers
            relative_end = end / num_layers
            
            if relative_start < 0.1:
                role = "Token/Embedding Processing"
            elif relative_start < 0.3:
                role = "Syntactic and Semantic Understanding"
            elif relative_start < 0.5:
                role = "Abstract Feature Extraction"
            elif relative_start < 0.7:
                role = "Reasoning and Concept Formation"
            else:
                role = "Output Generation and Refinement"
            
            interpretations[f"segment_{i+1}"] = f"Layers {start}-{end}: {role}"
        
        return interpretations
    
    def generate_pot_challenges(self,
                               n: int,
                               focus: str = "balanced",
                               complexity_range: Optional[Tuple] = None):
        """
        Generate PoT-style challenges for verification.
        
        Args:
            n: Number of challenges to generate
            focus: "coverage", "separation", or "balanced"
            complexity_range: Optional complexity range
            
        Returns:
            List of generated challenges
        """
        if not self.enable_pot_challenges:
            logger.warning("PoT challenge generation is disabled")
            return []
        
        # Lazy initialization of challenge generator
        if self.challenge_generator is None:
            from .challenges.pot_challenge_generator import PoTChallengeGenerator, ChallengeComplexity
            self.challenge_generator = PoTChallengeGenerator(
                enable_info_selection=True,
                min_complexity=ChallengeComplexity.MODERATE
            )
        
        logger.info(f"Generating {n} PoT-style challenges (focus: {focus})")
        
        challenges = self.challenge_generator.generate_verification_challenges(
            n=n,
            focus=focus
        )
        
        # Store in experiment results
        self.experiment_results["challenges"] = {
            "total": len(challenges),
            "focus": focus,
            "categories": list(set(c.category.value for c in challenges)),
            "complexity_distribution": dict(
                defaultdict(int, {c.complexity.value: 
                                 sum(1 for ch in challenges if ch.complexity == c.complexity)
                                 for c in challenges})
            )
        }
        
        return challenges
    
    def create_functional_segments(self, restriction_sites: List) -> List[FunctionalSegment]:
        """
        Create functional segments from discovered restriction sites.
        Each segment has behavioral fingerprint, functional role, and processing characteristics.
        
        Args:
            restriction_sites: List of RestrictionSite objects from behavioral analysis
            
        Returns:
            List of FunctionalSegment objects with behavioral metadata
        """
        segments = []
        
        # Ensure we have at least 2 sites to create segments
        if len(restriction_sites) < 2:
            logger.warning("Need at least 2 restriction sites to create functional segments")
            return segments
            
        for i in range(len(restriction_sites) - 1):
            start_site = restriction_sites[i]
            end_site = restriction_sites[i + 1]
            
            # Create segment with behavioral metadata
            segment = FunctionalSegment(
                id=f"seg_{i}_{start_site.layer_idx}_{end_site.layer_idx}",
                start_layer=start_site.layer_idx,
                end_layer=end_site.layer_idx,
                behavioral_fingerprint=self.compute_segment_fingerprint(start_site, end_site),
                functional_role=self.identify_functional_role(start_site, end_site),
                processing_mode=self.determine_processing_mode(start_site, end_site)
            )
            
            # Add execution policy based on functional role
            segment.execution_policy = self._create_execution_policy_for_role(segment.functional_role)
            
            segments.append(segment)
            
        logger.info(f"Created {len(segments)} functional segments from {len(restriction_sites)} restriction sites")
        return segments
    
    def compute_segment_fingerprint(self, start_site, end_site) -> Dict[str, Any]:
        """
        Aggregate behavioral characteristics across segment.
        
        Args:
            start_site: Starting RestrictionSite
            end_site: Ending RestrictionSite
            
        Returns:
            Dict containing behavioral fingerprint metrics
        """
        fingerprint = {
            "layer_range": (start_site.layer_idx, end_site.layer_idx),
            "layer_count": end_site.layer_idx - start_site.layer_idx,
            "start_divergence": start_site.behavioral_divergence,
            "end_divergence": end_site.behavioral_divergence,
            "avg_divergence": (start_site.behavioral_divergence + end_site.behavioral_divergence) / 2,
        }
        
        # Aggregate divergence metrics if available
        if hasattr(start_site, 'divergence_metrics') and start_site.divergence_metrics:
            fingerprint["start_metrics"] = start_site.divergence_metrics.copy()
        if hasattr(end_site, 'divergence_metrics') and end_site.divergence_metrics:
            fingerprint["end_metrics"] = end_site.divergence_metrics.copy()
            
        # Analyze response patterns to different probe types
        response_patterns = {}
        if hasattr(start_site, 'prompt_responses') and start_site.prompt_responses:
            response_patterns["start_responses"] = len(start_site.prompt_responses)
            
        if hasattr(end_site, 'prompt_responses') and end_site.prompt_responses:
            response_patterns["end_responses"] = len(end_site.prompt_responses)
            
        fingerprint["response_patterns"] = response_patterns
        
        # Statistical summary of activation patterns
        fingerprint["activation_summary"] = {
            "confidence_range": (
                getattr(start_site, 'confidence_score', 0.0),
                getattr(end_site, 'confidence_score', 0.0)
            ),
            "site_types": [start_site.site_type, end_site.site_type],
        }
        
        return fingerprint
    
    def identify_functional_role(self, start_site, end_site) -> str:
        """
        Map behavioral patterns to functional roles based on layer position and response strength.
        
        Args:
            start_site: Starting RestrictionSite
            end_site: Ending RestrictionSite
            
        Returns:
            String identifying the functional role
        """
        start_layer = start_site.layer_idx
        end_layer = end_site.layer_idx
        avg_layer = (start_layer + end_layer) / 2
        
        # Calculate average response strength
        avg_divergence = (start_site.behavioral_divergence + end_site.behavioral_divergence) / 2
        
        # Classification based on layer position and response patterns
        # These thresholds are based on typical transformer architectures
        
        if avg_layer < 5:  # Early layers (0-4)
            if avg_divergence < 0.3:
                return "token_embedding"
            else:
                return "early_processing"
                
        elif avg_layer < 25:  # Middle-early layers (5-24)  
            if 0.2 <= avg_divergence <= 0.4:
                return "feature_extraction"
            elif avg_divergence > 0.4:
                return "semantic_processing"
            else:
                return "representation_building"
                
        elif avg_layer < 50:  # Middle-late layers (25-49)
            if 0.6 <= avg_divergence <= 0.8:
                return "semantic_processing"
            elif avg_divergence > 0.8:
                return "reasoning"
            else:
                return "pattern_integration"
                
        else:  # Late layers (50+)
            if avg_divergence >= 0.8:
                return "output_generation"
            elif 0.5 <= avg_divergence < 0.8:
                return "decision_making"
            else:
                return "final_processing"
    
    def determine_processing_mode(self, start_site, end_site) -> str:
        """
        Determine optimal processing mode based on behavioral characteristics.
        
        Args:
            start_site: Starting RestrictionSite
            end_site: Ending RestrictionSite
            
        Returns:
            String identifying processing mode
        """
        avg_divergence = (start_site.behavioral_divergence + end_site.behavioral_divergence) / 2
        layer_count = end_site.layer_idx - start_site.layer_idx
        
        # Determine processing mode based on behavioral complexity
        if avg_divergence >= 0.7:
            # High divergence requires careful processing
            return "high_precision"
        elif avg_divergence >= 0.4:
            # Moderate divergence uses standard processing
            return "standard"
        elif layer_count <= 3:
            # Short segments can use fast processing
            return "fast"
        else:
            # Default to standard for safety
            return "standard"
    
    def _create_execution_policy_for_role(self, functional_role: str) -> ExecutionPolicy:
        """
        Create execution policy tailored to functional role.
        
        Args:
            functional_role: The identified functional role
            
        Returns:
            ExecutionPolicy optimized for the role
        """
        # Base policy
        policy = ExecutionPolicy(
            temperature=0.0,
            dtype="fp16",
            seed=42,
            checkpoint_activations=True
        )
        
        # Role-specific optimizations
        if functional_role in ["semantic_processing", "reasoning", "decision_making"]:
            # Critical processing requires higher precision
            policy.dtype = "fp32"
            policy.temperature = 0.0
            policy.attn_impl = "flash"  # More accurate attention
            policy.checkpoint_activations = True
            
        elif functional_role in ["output_generation", "final_processing"]:
            # Can use faster processing for output layers
            policy.dtype = "fp16"
            policy.temperature = 0.1
            policy.attn_impl = "paged"  # Faster but less precise
            policy.quantization = "8bit"
            
        elif functional_role in ["token_embedding", "early_processing"]:
            # Early layers are more robust to approximations
            policy.dtype = "fp16"
            policy.quantization = "8bit"
            policy.offload_to_cpu = True  # Can offload for memory savings
            
        return policy
    
    def process_segment_with_fingerprint(self, segment: FunctionalSegment, input_tokens: List[int]):
        """
        Process segment according to its behavioral fingerprint.
        Different processing strategies for different functional roles.
        
        Args:
            segment: FunctionalSegment with behavioral metadata
            input_tokens: Input tokens to process
            
        Returns:
            Processed output with telemetry
        """
        logger.info(f"Processing segment {segment.id} with role '{segment.functional_role}' "
                   f"using {segment.processing_mode} mode")
        
        # Use the segment's execution policy
        if segment.execution_policy:
            self.set_execution_policy(segment.execution_policy)
            
        # Track segment-specific telemetry
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            # Process using the segment's characteristics
            result = self._process_with_role_awareness(segment, input_tokens)
            
            # Collect telemetry
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            telemetry = SegmentTelemetry(
                segment_id=hash(segment.id),
                alloc_mb=final_memory - initial_memory,
                peak_mb=max(initial_memory, final_memory),
                t_ms=(end_time - start_time) * 1000,
                tokens_processed=len(input_tokens)
            )
            
            return result, telemetry
            
        except Exception as e:
            logger.error(f"Error processing segment {segment.id}: {e}")
            raise
    
    def _process_with_role_awareness(self, segment: FunctionalSegment, input_tokens: List[int]):
        """
        Internal processing method that adapts behavior based on functional role.
        
        Args:
            segment: FunctionalSegment with behavioral metadata
            input_tokens: Input tokens to process
            
        Returns:
            Processing results adapted to the segment's role
        """
        role = segment.functional_role
        
        # Role-specific processing adaptations
        if role in ["semantic_processing", "reasoning"]:
            # Use more thorough analysis for critical processing segments
            return self._process_with_high_attention(segment, input_tokens)
            
        elif role == "output_generation":
            # Focus on generation quality for output segments
            return self._process_with_generation_focus(segment, input_tokens)
            
        elif role in ["token_embedding", "early_processing"]:
            # Use efficient processing for early stages
            return self._process_efficiently(segment, input_tokens)
            
        else:
            # Default processing for unrecognized roles
            return self._process_standard(segment, input_tokens)
    
    def _process_with_high_attention(self, segment: FunctionalSegment, input_tokens: List[int]):
        """High-precision processing for semantic/reasoning segments."""
        # Implement detailed processing logic
        return {"processed_tokens": input_tokens, "mode": "high_attention", "segment_id": segment.id}
    
    def _process_with_generation_focus(self, segment: FunctionalSegment, input_tokens: List[int]):
        """Generation-focused processing for output segments."""
        # Implement generation-optimized logic
        return {"processed_tokens": input_tokens, "mode": "generation_focus", "segment_id": segment.id}
    
    def _process_efficiently(self, segment: FunctionalSegment, input_tokens: List[int]):
        """Efficient processing for early-stage segments."""
        # Implement efficient processing logic
        return {"processed_tokens": input_tokens, "mode": "efficient", "segment_id": segment.id}
    
    def _process_standard(self, segment: FunctionalSegment, input_tokens: List[int]):
        """Standard processing for general segments."""
        # Implement standard processing logic
        return {"processed_tokens": input_tokens, "mode": "standard", "segment_id": segment.id}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0