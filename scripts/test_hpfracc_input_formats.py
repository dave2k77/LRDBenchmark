#!/usr/bin/env python3
"""
Test script to explore hpfracc fractional neural network input formats
"""

import numpy as np
import jax.numpy as jnp
from hpfracc.ml import FractionalNeuralNetwork, BackendType
from hpfracc.ml.backends import get_backend_manager

def test_input_formats():
    """Test different input formats for hpfracc models"""
    
    print("🧪 Testing HPFracc Input Formats")
    print("=" * 50)
    
    # Set backend to JAX
    try:
        backend_manager = get_backend_manager()
        backend_manager.switch_backend(BackendType.JAX)
        print("✅ Set backend to JAX")
    except Exception as e:
        print(f"⚠️ Could not set backend: {e}")
    
    # Test different model configurations
    test_configs = [
        # (name, input_size, hidden_sizes, output_size)
        ("Single Feature", 1, [32, 16], 1),
        ("Multiple Features", 5, [32, 16], 1),
        ("Time Series Features", 10, [32, 16], 1),
        ("Large Input", 50, [64, 32], 1),
    ]
    
    for config_name, input_size, hidden_sizes, output_size in test_configs:
        print(f"\n🔧 Testing Configuration: {config_name}")
        print(f"   Input size: {input_size}, Hidden: {hidden_sizes}, Output: {output_size}")
        
        try:
            # Create model
            model = FractionalNeuralNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                fractional_order=0.5
            )
            
            # Fix backend issues
            if hasattr(model, 'backend'):
                model.backend = BackendType.JAX
            if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                model.config.backend = BackendType.JAX
            if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                model.tensor_ops.backend = BackendType.JAX
            
            print("   ✅ Model created successfully")
            
            # Test different input shapes
            test_inputs = [
                # (name, input_data, expected_shape)
                (f"Batch of {input_size}-feature vectors", 
                 np.random.randn(5, input_size), (5, input_size)),
                (f"Single {input_size}-feature vector", 
                 np.random.randn(1, input_size), (1, input_size)),
                (f"Large batch of {input_size}-feature vectors", 
                 np.random.randn(100, input_size), (100, input_size)),
            ]
            
            for input_name, input_data, expected_shape in test_inputs:
                print(f"\n     🔍 Testing: {input_name}")
                print(f"        Input shape: {input_data.shape}, dtype: {input_data.dtype}")
                print(f"        Expected: {expected_shape}")
                
                try:
                    # Try forward pass
                    output = model.forward(input_data, use_fractional=False)
                    print(f"        ✅ Success! Output shape: {output.shape}")
                    
                    # Check if output is JAX array
                    if hasattr(output, 'numpy'):
                        print(f"        📊 Output is JAX array, can convert to numpy")
                        numpy_output = output.numpy()
                        print(f"        📊 Numpy output shape: {numpy_output.shape}")
                    
                except Exception as e:
                    print(f"        ❌ Failed: {e}")
            
        except Exception as e:
            print(f"   ❌ Failed to create model: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Summary of working input formats:")

if __name__ == "__main__":
    test_input_formats()
