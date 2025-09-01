#!/usr/bin/env python3
"""
Check what divergence metrics are actually returned
"""

import torch
from src.models.true_segment_execution import BehavioralResponse, LayerSegmentExecutor, SegmentExecutionConfig
from unittest.mock import MagicMock, patch

# Create two different behavioral responses
response1 = BehavioralResponse(
    hidden_states=torch.randn(1, 5, 4096),
    statistical_signature={'mean_activation': 0.5, 'activation_entropy': 3.0}
)

response2 = BehavioralResponse(
    hidden_states=torch.randn(1, 5, 4096),
    statistical_signature={'mean_activation': 0.7, 'activation_entropy': 3.5}
)

config = SegmentExecutionConfig(model_path="/fake/path")

with patch.object(LayerSegmentExecutor, '__init__', lambda x, *args, **kwargs: None):
    executor = LayerSegmentExecutor.__new__(LayerSegmentExecutor)
    executor.device_manager = MagicMock()
    executor.device_manager.get_device.return_value = torch.device('cpu')
    
    divergence = executor.compute_behavioral_divergence(response1, response2)
    
    print("Actual divergence metrics returned:")
    for key, value in divergence.items():
        print(f"  {key}: {value}")