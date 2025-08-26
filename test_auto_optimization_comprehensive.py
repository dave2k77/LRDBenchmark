#!/usr/bin/env python3
"""
Comprehensive Auto-Optimization Test Suite

This script tests the auto-optimization system across all available estimators
and measures performance improvements.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
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


def test_estimator_performance(estimator_type: str, data_sizes: List[int], 
                              num_trials: int = 3) -> Dict[str, Any]:
    """
    Test performance of an estimator across different data sizes.
    
    Parameters
    ----------
    estimator_type : str
        Type of estimator to test
    data_sizes : List[int]
        List of data sizes to test
    num_trials : int
        Number of trials per data size
        
    Returns
    -------
    dict
        Performance results
    """
    print(f"\n{'='*20} Testing {estimator_type.upper()} {'='*20}")
    
    results = {
        'estimator_type': estimator_type,
        'data_sizes': data_sizes,
        'auto_times': [],
        'auto_hurst': [],
        'auto_optimization_level': [],
        'benchmark_results': [],
        'success': True
    }
    
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7)
    
    for size in data_sizes:
        print(f"\nData size: {size}")
        
        try:
            # Test auto-optimized estimator
            auto_estimator = AutoOptimizedEstimator(estimator_type)
            
            # Run multiple trials
            times = []
            hurst_values = []
            
            for trial in range(num_trials):
                data = fgn.generate(size, seed=42 + trial)
                
                start_time = time.time()
                result = auto_estimator.estimate(data)
                execution_time = time.time() - start_time
                
                times.append(execution_time)
                hurst_values.append(result['hurst_parameter'])
            
            # Calculate average performance
            avg_time = np.mean(times)
            avg_hurst = np.mean(hurst_values)
            
            print(f"Auto-Optimized ({auto_estimator.optimization_level}): {avg_time:.4f}s")
            print(f"Average Hurst: {avg_hurst:.6f}")
            
            # Store results
            results['auto_times'].append(avg_time)
            results['auto_hurst'].append(avg_hurst)
            results['auto_optimization_level'].append(auto_estimator.optimization_level)
            
            # Benchmark all implementations
            data = fgn.generate(size, seed=42)
            benchmark_results = auto_estimator.benchmark_all_implementations(data)
            results['benchmark_results'].append(benchmark_results)
            
            print("Benchmark Results:")
            for impl, res in benchmark_results.items():
                if res['success']:
                    speedup = res.get('speedup', 'N/A')
                    print(f"  {impl.upper()}: {res['time']:.4f}s (speedup: {speedup})")
                else:
                    print(f"  {impl.upper()}: Failed - {res.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            break
    
    return results


def create_performance_report(all_results: List[Dict[str, Any]]) -> str:
    """
    Create a comprehensive performance report.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        Results from all estimator tests
        
    Returns
    -------
    str
        Markdown formatted report
    """
    report = "# Auto-Optimization Comprehensive Test Report\n\n"
    report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Summary table
    report += "## 📊 Performance Summary\n\n"
    report += "| Estimator | Optimization Level | Avg Speedup | Status |\n"
    report += "|-----------|-------------------|-------------|---------|\n"
    
    for result in all_results:
        if result['success'] and result['benchmark_results']:
            # Calculate average speedup
            speedups = []
            for benchmark in result['benchmark_results']:
                if 'numba' in benchmark and benchmark['numba']['success']:
                    speedup = benchmark['numba'].get('speedup', 0)
                    if speedup != 'N/A':
                        speedups.append(speedup)
            
            avg_speedup = np.mean(speedups) if speedups else 0
            optimization_level = result['auto_optimization_level'][0] if result['auto_optimization_level'] else 'Unknown'
            
            status = "✅ Working" if result['success'] else "❌ Failed"
            report += f"| {result['estimator_type'].upper()} | {optimization_level} | {avg_speedup:.1f}x | {status} |\n"
        else:
            report += f"| {result['estimator_type'].upper()} | N/A | N/A | ❌ Failed |\n"
    
    # Detailed results
    report += "\n## 📈 Detailed Results\n\n"
    
    for result in all_results:
        report += f"### {result['estimator_type'].upper()} Estimator\n\n"
        
        if result['success']:
            report += f"**Optimization Level**: {result['auto_optimization_level'][0]}\n\n"
            
            report += "| Data Size | Execution Time | Hurst Parameter |\n"
            report += "|-----------|----------------|-----------------|\n"
            
            for i, size in enumerate(result['data_sizes']):
                if i < len(result['auto_times']):
                    time_val = result['auto_times'][i]
                    hurst_val = result['auto_hurst'][i]
                    report += f"| {size} | {time_val:.4f}s | {hurst_val:.6f} |\n"
            
            # Benchmark comparison
            if result['benchmark_results']:
                report += "\n**Performance Comparison**:\n\n"
                for i, benchmark in enumerate(result['benchmark_results']):
                    report += f"Data Size {result['data_sizes'][i]}:\n"
                    for impl, res in benchmark.items():
                        if res['success']:
                            speedup = res.get('speedup', 'N/A')
                            report += f"- {impl.upper()}: {res['time']:.4f}s (speedup: {speedup})\n"
                        else:
                            report += f"- {impl.upper()}: Failed - {res.get('error', 'Unknown error')}\n"
                    report += "\n"
        else:
            report += f"**Status**: Failed\n"
            report += f"**Error**: {result.get('error', 'Unknown error')}\n\n"
    
    return report


def main():
    """Run comprehensive auto-optimization tests."""
    print("🚀 Comprehensive Auto-Optimization Test Suite")
    print("=" * 60)
    
    # Test parameters
    data_sizes = [1000, 5000, 10000]
    num_trials = 2  # Reduced for faster testing
    
    # Estimators to test
    estimators = [
        'dfa',
        'rs', 
        'dma',
        'higuchi',
        'gph',
        'periodogram',
        'whittle'
    ]
    
    all_results = []
    
    for estimator_type in estimators:
        try:
            result = test_estimator_performance(estimator_type, data_sizes, num_trials)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to test {estimator_type}: {e}")
            all_results.append({
                'estimator_type': estimator_type,
                'success': False,
                'error': str(e)
            })
    
    # Generate report
    report = create_performance_report(all_results)
    
    # Save report
    with open("auto_optimization_comprehensive_report.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("✅ Comprehensive testing completed!")
    print("📄 Report saved to: auto_optimization_comprehensive_report.md")
    
    # Print summary
    successful_tests = sum(1 for r in all_results if r['success'])
    total_tests = len(all_results)
    
    print(f"\n📊 Summary:")
    print(f"- Total estimators tested: {total_tests}")
    print(f"- Successful tests: {successful_tests}")
    print(f"- Success rate: {successful_tests/total_tests*100:.1f}%")


if __name__ == "__main__":
    main()
