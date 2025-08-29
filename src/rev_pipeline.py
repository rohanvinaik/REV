"""
REV Pipeline - Core integration for memory-bounded LLM verification.

This module implements the main pipeline for REV (Restriction Enzyme Verification),
providing segment-wise model execution with memory-bounded streaming and
Merkle tree construction for verifiable computation.
"""

from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass
import hashlib
import numpy as np
from collections import deque
import torch

from .crypto.merkle import build_merkle_tree, leaf_bytes, generate_merkle_proof as merkle_path
from .hdc.encoder import HypervectorEncoder, HypervectorConfig
from .core.sequential import SequentialState


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
    signatures: Dict[str, np.ndarray] = None
    merkle_root: bytes = None
    
    def compute_hash(self) -> bytes:
        """Compute segment hash for Merkle tree construction."""
        data = f"{self.segment_id}:{self.start_idx}:{self.end_idx}:{self.tokens}".encode()
        return hashlib.sha256(data).digest()


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
            sparse_density=0.01,
            dtype="float32"
        )
        self.encoder = HypervectorEncoder(self.hdc_config)
        
        # Define default architectural sites if not provided
        self.architectural_sites = architectural_sites or self._default_sites()
        
        # Merkle tree storage for verification
        self.merkle_trees = {}
        self.segment_counter = 0
        
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
    
    def segment_tokens(self, tokens: List[int]) -> Generator[Segment, None, None]:
        """
        Segment input tokens into memory-bounded chunks.
        
        Args:
            tokens: List of token IDs
            
        Yields:
            Segment objects with bounded token sequences
        """
        for i in range(0, len(tokens), self.segment_size):
            segment = Segment(
                segment_id=self.segment_counter,
                tokens=tokens[i:i + self.segment_size],
                start_idx=i,
                end_idx=min(i + self.segment_size, len(tokens))
            )
            self.segment_counter += 1
            yield segment
    
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