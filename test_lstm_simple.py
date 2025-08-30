#!/usr/bin/env python3
"""
Simple test to isolate LSTM unpacking issue.
"""

import torch
import torch.nn as nn

def test_lstm():
    """Test LSTM output unpacking."""
    print("🔍 Testing LSTM output unpacking...")
    
    # Create a simple LSTM
    lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
    
    # Create test input
    x = torch.randn(8, 500, 1)  # batch_size=8, seq_len=500, features=1
    
    print(f"Input shape: {x.shape}")
    
    try:
        # Test LSTM forward pass
        output, (hidden, cell) = lstm(x)
        print(f"✅ LSTM output shape: {output.shape}")
        print(f"✅ Hidden shape: {hidden.shape}")
        print(f"✅ Cell shape: {cell.shape}")
        
        # Test with checkpointing
        print("\n🔍 Testing with checkpointing...")
        output_checkpoint = torch.utils.checkpoint.checkpoint(
            lambda x: lstm(x)[0], x, use_reentrant=False
        )
        print(f"✅ Checkpointed output shape: {output_checkpoint.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm()
