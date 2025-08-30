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
from collections import deque
import torch
import time
import psutil
import gc
import os
from enum import Enum

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
from .core.sequential import SequentialState


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
        architectural_sites: Optional[List[ArchitecturalSite]] = None
    ):
        """
        Initialize REV pipeline.
        
        Args:
            segment_size: Maximum tokens per segment
            buffer_size: Number of segments to keep in memory
            hdc_config: Configuration for hypervector encoding
            architectural_sites: List of architectural probe points
        """
        self.segment_size = segment_size
        self.buffer_size = buffer_size
        self.segment_buffer = deque(maxlen=buffer_size)
        
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
        
        # Memory-bounded execution state
        self.execution_policy = ExecutionPolicy()
        self.telemetry_records: List[SegmentTelemetry] = []
        self.kv_cache = {}  # KV cache for attention
        self.checkpoint_manager = CheckpointManager()
        self.memory_limit_mb = 4096  # Default 4GB limit
        
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
            
            # Prepare input
            input_ids = torch.tensor([segment.tokens], dtype=torch.long)
            if torch.cuda.is_available() and not policy.offload_to_cpu:
                input_ids = input_ids.cuda()
            
            # Manage KV cache
            if states_in and 'kv_cache' in states_in:
                self.kv_cache = states_in['kv_cache']
            
            # Limit KV cache size
            self._manage_kv_cache(policy.kv_cache_max_tokens)
            telemetry.kv_cache_size_mb = self._get_kv_cache_size_mb()
            
            # Forward pass with activation checkpointing
            if policy.checkpoint_activations:
                # Use gradient checkpointing to save memory
                with torch.cuda.amp.autocast(enabled=(policy.dtype == "fp16")):
                    states_out, activations = self._forward_with_checkpointing(
                        model, input_ids, seg
                    )
            else:
                # Standard forward pass
                states_out, activations = self._forward_segment(
                    model, input_ids, seg, policy
                )
            
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
            # Handle OOM by offloading and retrying
            print(f"OOM encountered, offloading to CPU and retrying segment {segment.segment_id}")
            gc.collect()
            torch.cuda.empty_cache()
            
            # Retry with CPU offloading
            policy.offload_to_cpu = True
            return self.run_segment(model, states_in, seg, segment, policy)
            
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
        """Apply quantization to model parameters."""
        if quantization == "8bit":
            # Apply 8-bit quantization
            for param in model.parameters():
                param.data = param.data.to(torch.int8)
        elif quantization == "4bit":
            # Apply 4-bit quantization (simplified)
            for param in model.parameters():
                # Pack two 4-bit values into one int8
                param.data = (param.data * 7).round().clamp(-8, 7).to(torch.int8)
    
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