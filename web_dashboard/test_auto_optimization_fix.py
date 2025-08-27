#!/usr/bin/env python3
"""
Test script to verify auto-optimization system is working after dependency fixes.
"""

import sys
import os
import time
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_auto_optimization():
    """Test that auto-optimization system can be imported and used."""
    print("🧪 Testing Auto-Optimization System...")
    
    try:
        # Test import
        from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator
        from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
        print("✅ Imports successful")
        
        # Generate test data
        fgn = FractionalGaussianNoise(H=0.7)
        test_data = fgn.generate(1000, seed=42)
        print("✅ Test data generated")
        
        # Test DFA auto-optimization
        print("\n🔍 Testing DFA Auto-Optimization...")
        dfa_estimator = AutoOptimizedEstimator('dfa')
        print(f"✅ DFA estimator created with optimization level: {dfa_estimator.optimization_level}")
        
        start_time = time.time()
        result = dfa_estimator.estimate(test_data)
        execution_time = time.time() - start_time
        
        print(f"✅ DFA estimation successful:")
        print(f"   - Hurst parameter: {result['hurst_parameter']:.6f}")
        print(f"   - Execution time: {execution_time:.4f}s")
        print(f"   - Optimization level: {result['optimization_info']['level']}")
        
        # Test RS auto-optimization
        print("\n🔍 Testing RS Auto-Optimization...")
        rs_estimator = AutoOptimizedEstimator('rs')
        print(f"✅ RS estimator created with optimization level: {rs_estimator.optimization_level}")
        
        start_time = time.time()
        result = rs_estimator.estimate(test_data)
        execution_time = time.time() - start_time
        
        print(f"✅ RS estimation successful:")
        print(f"   - Hurst parameter: {result['hurst_parameter']:.6f}")
        print(f"   - Execution time: {execution_time:.4f}s")
        print(f"   - Optimization level: {result['optimization_info']['level']}")
        
        # Test DMA auto-optimization
        print("\n🔍 Testing DMA Auto-Optimization...")
        dma_estimator = AutoOptimizedEstimator('dma')
        print(f"✅ DMA estimator created with optimization level: {dma_estimator.optimization_level}")
        
        start_time = time.time()
        result = dma_estimator.estimate(test_data)
        execution_time = time.time() - start_time
        
        print(f"✅ DMA estimation successful:")
        print(f"   - Hurst parameter: {result['hurst_parameter']:.6f}")
        print(f"   - Execution time: {execution_time:.4f}s")
        print(f"   - Optimization level: {result['optimization_info']['level']}")
        
        print("\n🎉 All auto-optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Auto-optimization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_auto_optimization()
    if success:
        print("\n✅ Auto-optimization system is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Auto-optimization system has issues!")
        sys.exit(1)
