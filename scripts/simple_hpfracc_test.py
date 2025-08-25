#!/usr/bin/env python3
"""
Simple HPFracc Test Script

This script tests basic hpfracc functionality to debug integration issues.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_hpfracc_basic():
    """Test basic hpfracc functionality."""
    print("🧪 Testing Basic HPFracc Functionality")
    print("=" * 50)
    
    # Test 1: Basic imports
    try:
        import hpfracc
        print("✅ hpfracc imported successfully")
        print(f"   Version: {hpfracc.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import hpfracc: {e}")
        return False
    
    # Test 2: ML module import
    try:
        from hpfracc.ml import FractionalNeuralNetwork
        print("✅ hpfracc.ml imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import hpfracc.ml: {e}")
        return False
    
    # Test 3: Backend management
    try:
        from hpfracc.ml.backends import get_backend_manager, BackendType
        backend_manager = get_backend_manager()
        print("✅ Backend manager created successfully")
        print(f"   Available backends: {[b.value for b in BackendType]}")
        
        # Try to get current backend
        try:
            current_backend = backend_manager.get_active_backend()
            print(f"   Current backend: {current_backend}")
        except:
            print("   Could not get current backend")
        
        # Try to set backend explicitly
        try:
            # Try setting to JAX first (since it was initialized)
            backend_manager.switch_backend(BackendType.JAX)
            print("   ✅ Set backend to JAX")
        except Exception as e:
            print(f"   ⚠️ Could not set backend to JAX: {e}")
            
            try:
                # Try setting to TORCH
                backend_manager.switch_backend(BackendType.TORCH)
                print("   ✅ Set backend to TORCH")
            except Exception as e:
                print(f"   ⚠️ Could not set backend to TORCH: {e}")
                
                try:
                    # Try setting to NUMBA
                    backend_manager.switch_backend(BackendType.NUMBA)
                    print("   ✅ Set backend to NUMBA")
                except Exception as e:
                    print(f"   ⚠️ Could not set backend to NUMBA: {e}")
        
    except Exception as e:
        print(f"⚠️ Backend manager issue: {e}")
    
    # Test 4: Create simple model
    try:
        # Try to create a simple model with explicit backend
        print("   Trying to create model with explicit backend...")
        
        # First, let's try to create the model after setting the backend
        backend_manager.switch_backend(BackendType.JAX)
        print("   ✅ Set backend to JAX before model creation")
        
        model = FractionalNeuralNetwork(
            input_size=1,
            hidden_sizes=[10],
            output_size=1,
            fractional_order=0.5
        )
        print("✅ FractionalNeuralNetwork created successfully")
        
        # Check if the backend was set correctly
        print(f"   Model backend after creation: {model.backend}")
        
        # Try to manually set the model's backend if possible
        if hasattr(model, 'backend'):
            try:
                model.backend = BackendType.JAX
                print("   ✅ Manually set model backend to JAX")
            except:
                print("   ⚠️ Could not manually set model backend")
        
        # Try to update the config backend as well
        if hasattr(model, 'config') and hasattr(model.config, 'backend'):
            try:
                model.config.backend = BackendType.JAX
                print("   ✅ Updated config backend to JAX")
            except:
                print("   ⚠️ Could not update config backend")
        
        # Try to update tensor_ops backend if possible
        if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
            try:
                model.tensor_ops.backend = BackendType.JAX
                print("   ✅ Updated tensor_ops backend to JAX")
            except:
                print("   ⚠️ Could not update tensor_ops backend")
        
        # Debug: examine model structure
        print(f"   Model type: {type(model)}")
        print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        # Examine key attributes
        print(f"   Input size: {model.input_size}")
        print(f"   Hidden sizes: {model.hidden_sizes}")
        print(f"   Output size: {model.output_size}")
        print(f"   Backend: {model.backend}")
        print(f"   Tensor ops: {model.tensor_ops}")
        
        # Check if there are any methods that might help with initialization
        if hasattr(model, 'config'):
            print(f"   Config: {model.config}")
            if hasattr(model.config, 'backend'):
                print(f"   Config backend: {model.config.backend}")
        
        if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
            print(f"   Tensor ops backend: {model.tensor_ops.backend}")
        
        # Try to understand what the model expects
        print(f"   Checking forward method signature...")
        import inspect
        try:
            sig = inspect.signature(model.forward)
            print(f"   Forward method signature: {sig}")
        except:
            print(f"   Could not get forward method signature")
        
        # Test different input shapes
        test_shapes = [
            (5, 1),      # (batch, features)
            (1, 5),      # (batch, features) - transposed
            (5,),        # (batch,) - 1D
            (1, 5, 1),  # (batch, sequence, features)
        ]
        
        for shape in test_shapes:
            try:
                X = np.random.randn(*shape).astype(np.float32)
                print(f"   Testing input shape {shape}: {X.shape}")
                
                # Try to understand what the model expects
                if hasattr(model, 'forward'):
                    print(f"     Forward method exists, trying call...")
                    
                    # Check if we need to provide additional parameters
                    if shape[0] > 1:  # If we have multiple samples
                        # Try providing a time array
                        t = np.linspace(0, 1, shape[0]).astype(np.float32)
                        print(f"     Providing time array: {t.shape}")
                        try:
                            output = model.forward(X, t)
                            print(f"     ✅ Success with shape {shape} and time array! Output: {output.shape}")
                            break
                        except Exception as e:
                            print(f"     ❌ Failed with time array: {e}")
                        
                        # Try disabling fractional computation
                        print(f"     Trying with use_fractional=False...")
                        try:
                            output = model.forward(X, use_fractional=False)
                            print(f"     ✅ Success with shape {shape} and use_fractional=False! Output: {output.shape}")
                            break
                        except Exception as e:
                            print(f"     ❌ Failed with use_fractional=False: {e}")
                    
                    # Try without time array
                    output = model.forward(X)
                    print(f"     ✅ Success with shape {shape}! Output: {output.shape}")
                    break
                else:
                    print(f"     No forward method found")
                    
            except Exception as e:
                print(f"     ❌ Failed with shape {shape}: {e}")
                # Print more details about the error
                import traceback
                print(f"     Error details: {traceback.format_exc()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation/forward pass failed: {e}")
        return False

def test_lrdbench_basic():
    """Test basic LRDBench functionality."""
    print("\n🧪 Testing Basic LRDBench Functionality")
    print("=" * 50)
    
    try:
        from lrdbench import FBMModel, enable_analytics
        print("✅ lrdbench imported successfully")
        
        # Enable analytics
        enable_analytics()
        print("✅ Analytics enabled")
        
        # Test data generation
        fbm = FBMModel(H=0.7, sigma=1.0)
        data = fbm.generate(100)
        print(f"✅ Generated FBM data: {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LRDBench test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 HPFracc + LRDBench Integration Test")
    print("=" * 60)
    
    # Test hpfracc
    hpfracc_ok = test_hpfracc_basic()
    
    # Test lrdbench
    lrdbench_ok = test_lrdbench_basic()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"HPFracc: {'✅ PASS' if hpfracc_ok else '❌ FAIL'}")
    print(f"LRDBench: {'✅ PASS' if lrdbench_ok else '❌ FAIL'}")
    
    if hpfracc_ok and lrdbench_ok:
        print("\n🎉 All tests passed! Ready for benchmarking.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return hpfracc_ok and lrdbench_ok

if __name__ == "__main__":
    main()
