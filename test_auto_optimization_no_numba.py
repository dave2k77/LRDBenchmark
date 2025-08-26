#!/usr/bin/env python3
"""
Test auto-optimization without NUMBA to isolate issues.
"""

import time
import sys
import os

# Temporarily disable NUMBA
sys.modules['numba'] = None

from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator


def test_estimator_no_numba(estimator_type: str, data_size: int = 1000):
    """Test estimator without NUMBA."""
    print(f"\n{'='*20} Testing {estimator_type.upper()} (No NUMBA) {'='*20}")
    
    try:
        # Generate test data
        fgn = FractionalGaussianNoise(H=0.7)
        data = fgn.generate(data_size, seed=42)
        
        # Test auto-optimized estimator
        auto_estimator = AutoOptimizedEstimator(estimator_type)
        
        start_time = time.time()
        result = auto_estimator.estimate(data)
        execution_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"Optimization Level: {auto_estimator.optimization_level}")
        print(f"Execution Time: {execution_time:.4f}s")
        print(f"Hurst Parameter: {result['hurst_parameter']:.6f}")
        
        return True, execution_time, result['hurst_parameter']
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None, None


def main():
    """Test all estimators without NUMBA."""
    print("🚀 Auto-Optimization Test (No NUMBA)")
    print("=" * 50)
    
    estimators = [
        'dfa',
        'rs', 
        'dma',
        'higuchi',
        'gph',
        'periodogram',
        'whittle'
    ]
    
    results = {}
    
    for estimator_type in estimators:
        success, time_taken, hurst = test_estimator_no_numba(estimator_type)
        results[estimator_type] = {
            'success': success,
            'time': time_taken,
            'hurst': hurst
        }
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Total estimators: {total}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for estimator, result in results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.4f}s" if result['time'] else "N/A"
        hurst_str = f"{result['hurst']:.6f}" if result['hurst'] else "N/A"
        print(f"{status} {estimator.upper()}: {time_str} | H={hurst_str}")


if __name__ == "__main__":
    main()
