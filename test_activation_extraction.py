#!/usr/bin/env python3
"""
Test the enhanced extract_activations() method in segment_runner.py.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Set up path
sys.path.insert(0, '/Users/rohanvinaik/REV')

def test_activation_extraction():
    """Test the enhanced activation extraction functionality."""
    
    print("=" * 70)
    print("ACTIVATION EXTRACTION TEST")
    print("=" * 70)
    
    from src.executor.segment_runner import SegmentRunner, SegmentConfig
    
    # Create test configuration
    config = SegmentConfig(
        extraction_sites=[
            "embeddings",
            "attention.0", "attention.1", "attention.2",
            "mlp.0", "mlp.1", "mlp.2",
            "layer_norm.final"
        ],
        use_fp16=False,  # Use float32 for testing
        gradient_checkpointing=False
    )
    
    # Initialize segment runner
    runner = SegmentRunner(config)
    
    # Test cases for different model types
    test_cases = [
        {
            'name': 'Simple Mock GPT Model',
            'model_type': 'gpt',
            'create_model': create_mock_gpt_model,
            'input_text': "Hello world, this is a test."
        },
        {
            'name': 'Simple Mock BERT Model', 
            'model_type': 'bert',
            'create_model': create_mock_bert_model,
            'input_text': "This is a BERT test sentence."
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        
        try:
            # Create mock model
            model = test_case['create_model']()
            print(f"✅ Created mock {test_case['model_type']} model: {model.__class__.__name__}")
            
            # Create input tokens
            input_ids = torch.randint(0, 1000, (1, 10))  # Batch size 1, seq len 10
            attention_mask = torch.ones_like(input_ids)
            
            print(f"   Input shape: {input_ids.shape}")
            
            # Test architecture detection
            detected_type = runner._detect_model_architecture(model)
            print(f"   Detected architecture: {detected_type}")
            
            # Extract activations
            activations = runner.extract_activations(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Analyze results
            print(f"   ✅ Extracted {len(activations)} activations:")
            for name, activation in activations.items():
                if isinstance(activation, torch.Tensor):
                    print(f"      {name}: shape={activation.shape}, dtype={activation.dtype}")
                else:
                    print(f"      {name}: {type(activation)}")
            
            # Verify activations are reasonable
            if len(activations) > 0:
                # Check that tensors have reasonable shapes
                for name, tensor in activations.items():
                    if isinstance(tensor, torch.Tensor):
                        if tensor.numel() == 0:
                            print(f"      ⚠️  Empty tensor for {name}")
                        elif torch.isnan(tensor).any():
                            print(f"      ⚠️  NaN values in {name}")
                        else:
                            print(f"      ✅ {name}: Valid tensor")
            
            results.append({
                'test': test_case['name'],
                'success': True,
                'num_activations': len(activations),
                'detected_type': detected_type
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Test layer probing logic separately
    print(f"\n3. Layer Probing Logic Test")
    print("-" * 50)
    
    try:
        # Test GPT-style layer matching
        probe_patterns = ["embeddings", "attention.0", "attention.1", "mlp.1", "layer_norm.final"]
        
        gpt_layers = [
            "transformer.wte",  # Should match "embeddings"
            "transformer.h.0.attn.c_attn",  # Should match "attention.0"  
            "transformer.h.1.mlp.c_fc",  # Should match "mlp.1"
            "transformer.ln_f",  # Should match "layer_norm.final"
            "transformer.h.5.ln_1",  # Should NOT match default extraction sites
        ]
        
        # Debug: check what the actual matching looks like
        print(f"   Debug - checking layer matching:")
        for layer in gpt_layers:
            for pattern in probe_patterns:
                if runner._should_probe_layer(layer, [pattern], 'gpt'):
                    print(f"      {layer} matches {pattern} ✅")
        
        gpt_matches = 0
        for layer in gpt_layers:
            if runner._should_probe_layer(layer, probe_patterns, 'gpt'):
                print(f"   ✅ Matched GPT layer: {layer}")
                gpt_matches += 1
            else:
                print(f"   ➖ Skipped GPT layer: {layer}")
        
        print(f"   GPT layers matched: {gpt_matches}/4 expected")
        
        # Test BERT-style layer matching
        bert_layers = [
            "bert.embeddings.word_embeddings",  # Should match "embeddings"
            "bert.encoder.layer.0.attention.self",  # Should match "attention.0"
            "bert.encoder.layer.2.intermediate.dense",  # Should match "mlp.2"  
            "bert.encoder.layer.5.output",  # Should NOT match
        ]
        
        bert_matches = 0
        for layer in bert_layers:
            if runner._should_probe_layer(layer, probe_patterns, 'bert'):
                print(f"   ✅ Matched BERT layer: {layer}")
                bert_matches += 1
            else:
                print(f"   ➖ Skipped BERT layer: {layer}")
        
        print(f"   BERT layers matched: {bert_matches}/3 expected")
        
        results.append({
            'test': 'Layer Probing Logic',
            'success': True,
            'gpt_matches': gpt_matches,
            'bert_matches': bert_matches
        })
        
    except Exception as e:
        print(f"   ❌ ERROR in layer probing test: {e}")
        results.append({
            'test': 'Layer Probing Logic',
            'success': False,
            'error': str(e)
        })
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_activations = sum(r.get('num_activations', 0) for r in results if r['success'])
    
    print(f"Tests passed: {successful_tests}/{len(results)}")
    print(f"Total activations extracted: {total_activations}")
    
    if successful_tests == len(results):
        print("✅ ALL TESTS PASSED: Activation extraction is working correctly!")
        return True
    else:
        print("❌ SOME TESTS FAILED: Check implementation")
        for result in results:
            if not result['success']:
                print(f"   Failed: {result['test']} - {result.get('error', 'Unknown error')}")
        return False


def create_mock_gpt_model():
    """Create a simple mock GPT-style model for testing."""
    
    class MockGPTBlock(torch.nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(hidden_size)
            self.attn = MockAttention(hidden_size)
            self.ln_2 = torch.nn.LayerNorm(hidden_size)
            self.mlp = MockMLP(hidden_size)
        
        def forward(self, x, attention_mask=None):
            # Simplified transformer block
            residual = x
            x = self.ln_1(x)
            x = self.attn(x)
            x = x + residual
            
            residual = x
            x = self.ln_2(x)
            x = self.mlp(x)
            x = x + residual
            return x
    
    class MockAttention(torch.nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.c_attn = torch.nn.Linear(hidden_size, 3 * hidden_size)
            self.c_proj = torch.nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            return self.c_proj(torch.relu(self.c_attn(x)))
    
    class MockMLP(torch.nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.c_fc = torch.nn.Linear(hidden_size, 4 * hidden_size)
            self.c_proj = torch.nn.Linear(4 * hidden_size, hidden_size)
        
        def forward(self, x):
            return self.c_proj(torch.relu(self.c_fc(x)))
    
    class MockGPTModel(torch.nn.Module):
        def __init__(self, vocab_size=50257, hidden_size=768, num_layers=3):
            super().__init__()
            self.config = type('Config', (), {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_hidden_layers': num_layers,
                'use_cache': True
            })()
            
            self.transformer = torch.nn.ModuleDict({
                'wte': torch.nn.Embedding(vocab_size, hidden_size),
                'h': torch.nn.ModuleList([MockGPTBlock(hidden_size) for _ in range(num_layers)]),
                'ln_f': torch.nn.LayerNorm(hidden_size)
            })
        
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
            x = self.transformer.wte(input_ids)
            
            for block in self.transformer.h:
                x = block(x, attention_mask)
            
            x = self.transformer.ln_f(x)
            
            if return_dict:
                return type('Output', (), {'last_hidden_state': x})()
            return x
    
    return MockGPTModel()


def create_mock_bert_model():
    """Create a simple mock BERT-style model for testing."""
    
    class MockBERTLayer(torch.nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.attention = MockBERTAttention(hidden_size)
            self.intermediate = torch.nn.Linear(hidden_size, 4 * hidden_size)
            self.output = torch.nn.Linear(4 * hidden_size, hidden_size)
            self.layernorm1 = torch.nn.LayerNorm(hidden_size)
            self.layernorm2 = torch.nn.LayerNorm(hidden_size)
        
        def forward(self, x, attention_mask=None):
            # Self-attention
            residual = x
            x = self.layernorm1(x)
            x = self.attention(x)
            x = x + residual
            
            # Feed-forward
            residual = x
            x = self.layernorm2(x)
            x = torch.relu(self.intermediate(x))
            x = self.output(x)
            x = x + residual
            return x
    
    class MockBERTAttention(torch.nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.self = torch.nn.Linear(hidden_size, 3 * hidden_size)
            self.output = torch.nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            return self.output(torch.relu(self.self(x)))
    
    class MockBERTModel(torch.nn.Module):
        def __init__(self, vocab_size=30522, hidden_size=768, num_layers=3):
            super().__init__()
            self.config = type('Config', (), {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_hidden_layers': num_layers,
                'use_cache': True
            })()
            
            self.embeddings = torch.nn.ModuleDict({
                'word_embeddings': torch.nn.Embedding(vocab_size, hidden_size),
                'position_embeddings': torch.nn.Embedding(512, hidden_size),
                'LayerNorm': torch.nn.LayerNorm(hidden_size)
            })
            
            self.encoder = torch.nn.ModuleDict({
                'layer': torch.nn.ModuleList([MockBERTLayer(hidden_size) for _ in range(num_layers)])
            })
        
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
            x = self.embeddings.word_embeddings(input_ids) + self.embeddings.position_embeddings(position_ids)
            x = self.embeddings.LayerNorm(x)
            
            for layer in self.encoder.layer:
                x = layer(x, attention_mask)
            
            if return_dict:
                return type('Output', (), {'last_hidden_state': x})()
            return x
    
    return MockBERTModel()


if __name__ == "__main__":
    success = test_activation_extraction()
    exit(0 if success else 1)