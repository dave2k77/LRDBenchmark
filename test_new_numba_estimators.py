#!/usr/bin/env python3
"""
Test the new NUMBA-optimized estimators.
"""

import time
import numpy as np
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise

# Test the new NUMBA-optimized estimators
from lrdbench.analysis.temporal.higuchi.higuchi_estimator_numba_optimized import NumbaOptimizedHiguchiEstimator
from lrdbench.analysis.spectral.gph.gph_estimator_numba_optimized import NumbaOptimizedGPHEstimator
from lrdbench.analysis.spectral.periodogram.periodogram_estimator_numba_optimized import NumbaOptimizedPeriodogramEstimator
from lrdbench.analysis.spectral.whittle.whittle_estimator_numba_optimized import NumbaOptimizedWhittleEstimator


def test_numba_estimator(estimator_class, name, data_size=5000):
    """Test a NUMBA-optimized estimator."""
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
        
        return True, execution_time, result['hurst_parameter']
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    """Test all new NUMBA-optimized estimators."""
    print("🚀 Testing New NUMBA-Optimized Estimators")
    print("=" * 60)
    
    # Test estimators
    estimators = [
        (NumbaOptimizedHiguchiEstimator, "NUMBA Higuchi"),
        (NumbaOptimizedGPHEstimator, "NUMBA GPH"),
        (NumbaOptimizedPeriodogramEstimator, "NUMBA Periodogram"),
        (NumbaOptimizedWhittleEstimator, "NUMBA Whittle"),
    ]
    
    results = {}
    
    for estimator_class, name in estimators:
        success, time_taken, hurst = test_numba_estimator(estimator_class, name)
        results[name] = {
            'success': success,
            'time': time_taken,
            'hurst': hurst
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
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
        print(f"{status} {name}: {time_str} | H={hurst_str}")
    
    if successful == total:
        print(f"\n🎉 SUCCESS! All new NUMBA estimators working!")
    else:
        print(f"\n⚠️ Some estimators failed. Check the errors above.")


if __name__ == "__main__":
    main()
