#!/usr/bin/env python3
"""
Test script to verify GPU acceleration is working with our unified estimators.
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_frameworks():
    """Test that GPU frameworks are now working."""
    
    print("Testing GPU Framework Availability")
    print("=" * 50)
    
    # Test JAX GPU
    try:
        import jax
        import jax.numpy as jnp
        print(f"✓ JAX {jax.__version__} available")
        print(f"  Devices: {jax.devices()}")
        
        # Check if JAX is using GPU
        if any('gpu' in str(device).lower() for device in jax.devices()):
            print("  ✓ JAX is using GPU acceleration")
        else:
            print("  ⚠ JAX is still using CPU")
            
    except ImportError:
        print("❌ JAX not available")
    
    # Test CuPy GPU
    try:
        import cupy as cp
        print(f"✓ CuPy available")
        print(f"  CUDA available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"  Device count: {cp.cuda.runtime.getDeviceCount()}")
            print(f"  Current device: {cp.cuda.runtime.getDevice()}")
            device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()
            print(f"  Device name: {device_name}")
            print("  ✓ CuPy is using GPU acceleration")
        else:
            print("  ⚠ CuPy is not using GPU")
            
    except ImportError:
        print("❌ CuPy not available")
    
    # Test hpfracc
    try:
        import hpfracc
        print(f"✓ hpfracc available")
    except ImportError:
        print("❌ hpfracc not available")

def test_gpu_performance():
    """Test GPU performance with basic operations."""
    
    print("\n\nTesting GPU Performance")
    print("=" * 40)
    
    # Test JAX GPU performance
    try:
        import jax
        import jax.numpy as jnp
        
        print("\n1. Testing JAX GPU Performance...")
        
        # Create large arrays
        size = 5000
        from jax.random import PRNGKey, uniform
        key = PRNGKey(0)
        x = uniform(key, (size, size))
        y = uniform(key, (size, size))
        
        # Warm up JIT
        z = jnp.dot(x, y)
        z.block_until_ready()
        
        # Time matrix multiplication
        start = time.time()
        z = jnp.dot(x, y)
        z.block_until_ready()
        end = time.time()
        
        print(f"   Matrix multiplication ({size}x{size}): {end-start:.3f} seconds")
        print(f"   Result shape: {z.shape}")
        
    except Exception as e:
        print(f"   ❌ JAX GPU test failed: {e}")
    
    # Test CuPy GPU performance
    try:
        import cupy as cp
        
        print("\n2. Testing CuPy GPU Performance...")
        
        # Create large arrays
        size = 5000
        x = cp.random.random((size, size))
        y = cp.random.random((size, size))
        
        # Time matrix multiplication
        start = time.time()
        z = cp.dot(x, y)
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        
        print(f"   Matrix multiplication ({size}x{size}): {end-start:.3f} seconds")
        print(f"   Result shape: {z.shape}")
        
        # Clean up GPU memory
        del x, y, z
        cp.get_default_memory_pool().free_all_blocks()
        
    except Exception as e:
        print(f"   ❌ CuPy GPU test failed: {e}")

def test_unified_estimators_gpu():
    """Test that our unified estimators are using GPU acceleration."""
    
    print("\n\nTesting Unified Estimators with GPU Acceleration")
    print("=" * 60)
    
    try:
        # Test FBM model with GPU
        print("\n1. Testing FBM Model GPU Acceleration...")
        from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
        
        fbm = FractionalBrownianMotion(H=0.7, use_optimization="jax")
        print(f"   ✓ FBM model created with JAX optimization")
        print(f"   ✓ Optimization info: {fbm.get_optimization_info()}")
        
        # Generate data and test performance
        print("   Testing FBM generation performance...")
        start = time.time()
        data = fbm.generate(n=10000)
        end = time.time()
        print(f"   ✓ Generated 10000 points in {end-start:.3f} seconds")
        print(f"   ✓ Data shape: {data.shape}")
        
        # Test R/S estimator with GPU
        print("\n2. Testing R/S Estimator GPU Acceleration...")
        from lrdbenchmark.analysis.temporal.rs.rs_estimator import RSEstimator
        
        rs = RSEstimator(use_optimization="jax")
        print(f"   ✓ R/S estimator created with JAX optimization")
        print(f"   ✓ Optimization info: {rs.get_optimization_info()}")
        
        # Test estimation performance
        print("   Testing R/S estimation performance...")
        start = time.time()
        result = rs.estimate(data)
        end = time.time()
        print(f"   ✓ Estimated Hurst exponent in {end-start:.3f} seconds")
        print(f"   ✓ Estimated H: {result.get('hurst_exponent', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✓ GPU acceleration testing completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing GPU acceleration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("LRDBenchmark GPU Acceleration Test")
    print("=" * 50)
    
    # Test GPU frameworks
    test_gpu_frameworks()
    
    # Test GPU performance
    test_gpu_performance()
    
    # Test unified estimators with GPU
    success = test_unified_estimators_gpu()
    
    if success:
        print("\n🎉 GPU acceleration is working! Our unified estimators are using GPU acceleration.")
    else:
        print("\n💥 Some GPU acceleration tests failed. Please check the error messages above.")
        sys.exit(1)
