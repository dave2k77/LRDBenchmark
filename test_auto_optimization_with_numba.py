#!/usr/bin/env python3
"""
Test auto-optimization with NUMBA enabled to measure performance improvements.
"""

import time
import numpy as np
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator


def test_estimator_with_benchmark(estimator_type: str, data_size: int = 5000):
    """Test estimator and benchmark all implementations."""
    print(f"\n{'='*20} Testing {estimator_type.upper()} {'='*20}")
    
    try:
        # Generate test data
        fgn = FractionalGaussianNoise(H=0.7)
        data = fgn.generate(data_size, seed=42)
        
        # Test auto-optimized estimator
        auto_estimator = AutoOptimizedEstimator(estimator_type)
        
        start_time = time.time()
        result = auto_estimator.estimate(data)
        execution_time = time.time() - start_time
        
        print(f"✅ Auto-Optimized Success!")
        print(f"Optimization Level: {auto_estimator.optimization_level}")
        print(f"Execution Time: {execution_time:.4f}s")
        print(f"Hurst Parameter: {result['hurst_parameter']:.6f}")
        
        # Benchmark all implementations
        print("\nBenchmark Results:")
        benchmark_results = auto_estimator.benchmark_all_implementations(data)
        
        for impl, res in benchmark_results.items():
            if res['success']:
                speedup = res.get('speedup', 'N/A')
                print(f"  {impl.upper()}: {res['time']:.4f}s (speedup: {speedup})")
            else:
                print(f"  {impl.upper()}: Failed - {res.get('error', 'Unknown error')}")
        
        return True, execution_time, result['hurst_parameter'], benchmark_results
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None, None, None


def main():
    """Test all estimators with NUMBA enabled."""
    print("🚀 Auto-Optimization Test (With NUMBA)")
    print("=" * 60)
    
    # Test estimators that have NUMBA optimizations
    estimators = [
        'dfa',
        'rs', 
        'dma'
    ]
    
    results = {}
    
    for estimator_type in estimators:
        success, time_taken, hurst, benchmark = test_estimator_with_benchmark(estimator_type)
        results[estimator_type] = {
            'success': success,
            'time': time_taken,
            'hurst': hurst,
            'benchmark': benchmark
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Total estimators tested: {total}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nDetailed Performance Results:")
    for estimator, result in results.items():
        if result['success'] and result['benchmark']:
            print(f"\n{estimator.upper()}:")
            benchmark = result['benchmark']
            
            # Find best performance
            best_time = float('inf')
            best_impl = None
            
            for impl, res in benchmark.items():
                if res['success'] and res['time']:
                    if res['time'] < best_time:
                        best_time = res['time']
                        best_impl = impl
            
            if best_impl:
                print(f"  Best: {best_impl.upper()} ({best_time:.4f}s)")
                
                # Show speedups
                if 'standard' in benchmark and benchmark['standard']['success']:
                    standard_time = benchmark['standard']['time']
                    for impl, res in benchmark.items():
                        if impl != 'standard' and res['success'] and res['time']:
                            speedup = standard_time / res['time']
                            print(f"  {impl.upper()} speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
