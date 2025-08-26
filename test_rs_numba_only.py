#!/usr/bin/env python3
"""Test RS NUMBA optimization specifically."""

import time
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator

print("Testing RS NUMBA optimization...")

try:
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7)
    data = fgn.generate(5000, seed=42)
    
    # Test auto-optimized RS estimator
    auto_estimator = AutoOptimizedEstimator('rs')
    
    start_time = time.time()
    result = auto_estimator.estimate(data)
    execution_time = time.time() - start_time
    
    print(f"✅ Success!")
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
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
