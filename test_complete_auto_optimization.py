#!/usr/bin/env python3
"""
Test the complete auto-optimization system with all NUMBA optimizations.
"""

import time
import numpy as np
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise

# Test the complete auto-optimization system
from lrdbench.analysis.auto_optimized_estimator import (
    AutoOptimizedEstimator,
    AutoDFAEstimator,
    AutoRSEstimator,
    AutoDMAEstimator,
    AutoHiguchiEstimator,
    AutoGPHEstimator,
    AutoPeriodogramEstimator,
    AutoWhittleEstimator
)


def test_auto_optimized_estimator(estimator_class, name, data_size=5000):
    """Test an auto-optimized estimator."""
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
        print(f"Optimization Level: {estimator.optimization_level}")
        print(f"Execution Time: {execution_time:.4f}s")
        print(f"Hurst Parameter: {result['hurst_parameter']:.6f}")
        
        # Get optimization info
        info = estimator.get_optimization_info()
        print(f"NUMBA Available: {info['numba_available']}")
        print(f"JAX Available: {info['jax_available']}")
        
        return True, execution_time, result['hurst_parameter'], estimator.optimization_level
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def main():
    """Test the complete auto-optimization system."""
    print("🚀 Testing Complete Auto-Optimization System")
    print("=" * 70)
    
    # Test estimators
    estimators = [
        (AutoRSEstimator, "Auto RS"),
        (AutoDMAEstimator, "Auto DMA"),
        (AutoHiguchiEstimator, "Auto Higuchi"),
        (AutoGPHEstimator, "Auto GPH"),
        (AutoPeriodogramEstimator, "Auto Periodogram"),
        (AutoWhittleEstimator, "Auto Whittle"),
        (AutoDFAEstimator, "Auto DFA"),
    ]
    
    results = {}
    
    for estimator_class, name in estimators:
        success, time_taken, hurst, optimization_level = test_auto_optimized_estimator(estimator_class, name)
        results[name] = {
            'success': success,
            'time': time_taken,
            'hurst': hurst,
            'optimization_level': optimization_level
        }
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 COMPLETE AUTO-OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Total estimators tested: {total}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.4f}s" if result['time'] else "N/A"
        hurst_str = f"{result['hurst']:.6f}" if result['hurst'] else "N/A"
        opt_level = result['optimization_level'] if result['optimization_level'] else "N/A"
        print(f"{status} {name}: {time_str} | H={hurst_str} | {opt_level}")
    
    # Count optimization levels
    optimization_counts = {}
    for result in results.values():
        if result['success'] and result['optimization_level']:
            level = result['optimization_level']
            optimization_counts[level] = optimization_counts.get(level, 0) + 1
    
    print(f"\nOptimization Level Distribution:")
    for level, count in optimization_counts.items():
        print(f"  {level}: {count} estimators")
    
    if successful == total:
        print(f"\n🎉 SUCCESS! Complete auto-optimization system working!")
        print(f"✅ All estimators automatically using best available optimization!")
    else:
        print(f"\n⚠️ Some estimators failed. Check the errors above.")


if __name__ == "__main__":
    main()
