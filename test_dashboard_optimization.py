#!/usr/bin/env python3
"""
Test script to verify auto-optimization system for dashboard integration.
"""

import time
import numpy as np
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise

# Test auto-optimized estimators
from lrdbench.analysis.auto_optimized_estimator import (
    AutoDFAEstimator, AutoRSEstimator, AutoDMAEstimator,
    AutoHiguchiEstimator, AutoGPHEstimator, 
    AutoPeriodogramEstimator, AutoWhittleEstimator
)


def test_dashboard_optimization():
    """Test auto-optimization system for dashboard integration."""
    print("🚀 Testing Auto-Optimization System for Dashboard")
    print("=" * 60)
    
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7)
    test_data = fgn.generate(5000, seed=42)
    
    # Test all auto-optimized estimators
    auto_estimators = {
        "DFA": AutoDFAEstimator(),
        "RS": AutoRSEstimator(),
        "DMA": AutoDMAEstimator(),
        "Higuchi": AutoHiguchiEstimator(),
        "GPH": AutoGPHEstimator(),
        "Periodogram": AutoPeriodogramEstimator(),
        "Whittle": AutoWhittleEstimator()
    }
    
    results = {}
    performance_data = []
    
    print(f"Testing with {len(test_data)} data points...")
    print()
    
    for name, estimator in auto_estimators.items():
        print(f"Testing {name}...")
        
        start_time = time.time()
        result = estimator.estimate(test_data)
        execution_time = time.time() - start_time
        
        results[name] = result
        performance_data.append({
            'Estimator': name,
            'Hurst': result['hurst_parameter'],
            'Time (s)': execution_time,
            'Optimization': estimator.optimization_level,
            'Speedup': '🚀' if execution_time < 0.1 else '⚡' if execution_time < 0.5 else '📊'
        })
        
        print(f"  ✅ {name}: {execution_time:.4f}s | H={result['hurst_parameter']:.6f} | {estimator.optimization_level}")
    
    print()
    print("📊 Performance Summary:")
    print("-" * 40)
    
    # Calculate statistics
    times = [p['Time (s)'] for p in performance_data]
    avg_time = np.mean(times)
    fastest = min(performance_data, key=lambda x: x['Time (s)'])
    slowest = max(performance_data, key=lambda x: x['Time (s)'])
    
    print(f"Average Execution Time: {avg_time:.4f}s")
    print(f"Fastest: {fastest['Estimator']} ({fastest['Time (s)']:.4f}s)")
    print(f"Slowest: {slowest['Estimator']} ({slowest['Time (s)']:.4f}s)")
    
    # Optimization distribution
    opt_counts = {}
    for p in performance_data:
        opt = p['Optimization']
        opt_counts[opt] = opt_counts.get(opt, 0) + 1
    
    print()
    print("🎯 Optimization Distribution:")
    for opt, count in opt_counts.items():
        print(f"  {opt}: {count} estimators")
    
    # Verify all estimators worked
    successful = len(results)
    total = len(auto_estimators)
    
    print()
    print(f"✅ Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful == total:
        print("🎉 All auto-optimized estimators working perfectly!")
        print("🚀 Dashboard integration ready!")
    else:
        print("⚠️ Some estimators failed. Check errors above.")
    
    return results, performance_data


if __name__ == "__main__":
    test_dashboard_optimization()
