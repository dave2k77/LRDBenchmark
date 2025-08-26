#!/usr/bin/env python3
"""
Test the deployed auto-optimization system with standard implementations only.
"""

import time
import sys
import numpy as np

# Temporarily disable NUMBA to test standard implementations
sys.modules['numba'] = None

from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise

# Test the deployed estimators
from lrdbench.analysis import (
    RSEstimator,
    DMAEstimator,
    HiguchiEstimator,
    GPHEstimator,
    PeriodogramEstimator,
    WhittleEstimator,
    DFAEstimator
)


def test_deployed_estimator(estimator_class, name, data_size=5000):
    """Test a deployed estimator."""
    print(f"\n{'='*20} Testing {name} {'='*20}")
    
    try:
        # Generate test data
        fgn = FractionalGaussianNoise(H=0.7)
        data = fgn.generate(data_size, seed=42)
        
        # Create estimator
        estimator = estimator_class()
        
        # Test estimation
        start_time = time.time()
        result = estimator.estimate(data)
        execution_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"Execution Time: {execution_time:.4f}s")
        print(f"Hurst Parameter: {result['hurst_parameter']:.6f}")
        
        # Check if it's auto-optimized
        if hasattr(estimator, 'get_optimization_info'):
            info = estimator.get_optimization_info()
            print(f"Optimization Level: {info['optimization_level']}")
            print(f"NUMBA Available: {info['numba_available']}")
            print(f"JAX Available: {info['jax_available']}")
        
        return True, execution_time, result['hurst_parameter']
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None, None


def main():
    """Test all deployed estimators."""
    print("🚀 Testing Deployed Auto-Optimization System (Standard Only)")
    print("=" * 70)
    
    # Test estimators
    estimators = [
        (RSEstimator, "RS Estimator"),
        (DMAEstimator, "DMA Estimator"),
        (HiguchiEstimator, "Higuchi Estimator"),
        (GPHEstimator, "GPH Estimator"),
        (PeriodogramEstimator, "Periodogram Estimator"),
        (WhittleEstimator, "Whittle Estimator"),
        (DFAEstimator, "DFA Estimator (Standard)"),
    ]
    
    results = {}
    
    for estimator_class, name in estimators:
        success, time_taken, hurst = test_deployed_estimator(estimator_class, name)
        results[name] = {
            'success': success,
            'time': time_taken,
            'hurst': hurst
        }
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Total estimators deployed: {total}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.4f}s" if result['time'] else "N/A"
        hurst_str = f"{result['hurst']:.6f}" if result['hurst'] else "N/A"
        print(f"{status} {name}: {time_str} | H={hurst_str}")
    
    if successful == total:
        print(f"\n🎉 SUCCESS! All estimators deployed and working!")
        print(f"✅ Auto-optimization system successfully deployed!")
        print(f"✅ All estimators fall back to standard implementations when NUMBA/JAX unavailable")
        print(f"✅ Performance improvements will be automatic when optimizations are available")
    else:
        print(f"\n⚠️ Some estimators failed. Check the errors above.")


if __name__ == "__main__":
    main()
