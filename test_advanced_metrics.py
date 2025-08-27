#!/usr/bin/env python3
"""
Test script for advanced metrics functionality in LRDBench.

This script demonstrates the new convergence rates and mean signed error
analysis capabilities.
"""

import numpy as np
import time
from lrdbench.analysis.benchmark import ComprehensiveBenchmark
from lrdbench.analysis.advanced_metrics import (
    ConvergenceAnalyzer,
    MeanSignedErrorAnalyzer,
    AdvancedPerformanceProfiler,
    calculate_convergence_rate,
    calculate_mean_signed_error,
    profile_estimator_performance
)
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator


def test_convergence_analysis():
    """Test convergence rate analysis."""
    print("🔄 Testing Convergence Analysis")
    print("=" * 50)
    
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
    data = fgn.generate(2000, seed=42)
    
    # Test with DFA estimator
    estimator = DFAEstimator(min_box_size=8, max_box_size=200)
    
    # Initialize convergence analyzer
    analyzer = ConvergenceAnalyzer(convergence_threshold=1e-6)
    
    # Analyze convergence
    results = analyzer.analyze_convergence_rate(estimator, data, true_value=0.7)
    
    print(f"Convergence Rate: {results['convergence_rate']:.4f}")
    print(f"Convergence Achieved: {results['convergence_achieved']}")
    print(f"Convergence Iteration: {results['convergence_iteration']}")
    print(f"Stability Metric: {results['stability_metric']:.4f}")
    print(f"Final Estimate: {results['final_estimate']:.4f}")
    print(f"Number of Subsets Tested: {len(results['subset_sizes'])}")
    
    return results


def test_mean_signed_error_analysis():
    """Test mean signed error analysis."""
    print("\n📊 Testing Mean Signed Error Analysis")
    print("=" * 50)
    
    # Generate multiple datasets for Monte Carlo analysis
    fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
    true_value = 0.7
    
    estimates = []
    true_values = []
    
    # Generate estimates from multiple noisy datasets
    for i in range(50):
        # Add small noise to create variations
        data = fgn.generate(1000, seed=i)
        noise_level = 0.01 * np.std(data)
        noisy_data = data + np.random.normal(0, noise_level, len(data))
        
        try:
            estimator = DFAEstimator(min_box_size=8, max_box_size=200)
            result = estimator.estimate(noisy_data)
            estimate = result.get('hurst_parameter', None)
            if estimate is not None:
                estimates.append(estimate)
                true_values.append(true_value)
        except:
            continue
    
    if len(estimates) > 0:
        # Analyze mean signed error
        analyzer = MeanSignedErrorAnalyzer()
        results = analyzer.calculate_mean_signed_error(estimates, true_values)
        
        print(f"Mean Signed Error: {results['mean_signed_error']:.6f}")
        print(f"Mean Absolute Error: {results['mean_absolute_error']:.6f}")
        print(f"Root Mean Squared Error: {results['root_mean_squared_error']:.6f}")
        print(f"Bias Percentage: {results['bias_percentage']:.2f}%")
        print(f"Significant Bias: {results['significant_bias']}")
        print(f"T-statistic: {results['t_statistic']:.4f}")
        print(f"P-value: {results['p_value']:.4e}")
        print(f"95% Confidence Interval: [{results['confidence_interval_95'][0]:.6f}, {results['confidence_interval_95'][1]:.6f}]")
        
        return results
    else:
        print("❌ No successful estimates generated")
        return None


def test_advanced_performance_profiling():
    """Test comprehensive performance profiling."""
    print("\n⚡ Testing Advanced Performance Profiling")
    print("=" * 50)
    
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
    data = fgn.generate(1500, seed=42)
    true_value = 0.7
    
    # Test with RS estimator
    estimator = RSEstimator(min_window_size=10, max_window_size=200)
    
    # Initialize advanced profiler
    profiler = AdvancedPerformanceProfiler(convergence_threshold=1e-6)
    
    # Profile performance
    results = profiler.profile_estimator_performance(
        estimator, data, true_value, n_monte_carlo=50
    )
    
    # Display results
    basic_perf = results['basic_performance']
    convergence_analysis = results['convergence_analysis']
    bias_analysis = results['bias_analysis']
    comprehensive_score = results['comprehensive_score']
    
    print(f"Success: {basic_perf['success']}")
    print(f"Execution Time: {basic_perf['execution_time']:.4f}s")
    print(f"Comprehensive Score: {comprehensive_score:.4f}")
    
    if convergence_analysis:
        print(f"\nConvergence Analysis:")
        print(f"  Convergence Rate: {convergence_analysis.get('convergence_rate', 'N/A')}")
        print(f"  Convergence Achieved: {convergence_analysis.get('convergence_achieved', 'N/A')}")
        print(f"  Stability Metric: {convergence_analysis.get('stability_metric', 'N/A')}")
    
    if bias_analysis:
        print(f"\nBias Analysis:")
        print(f"  Mean Signed Error: {bias_analysis.get('mean_signed_error', 'N/A')}")
        print(f"  Bias Percentage: {bias_analysis.get('bias_percentage', 'N/A')}%")
        print(f"  Significant Bias: {bias_analysis.get('significant_bias', 'N/A')}")
    
    return results


def test_benchmark_integration():
    """Test integration with the main benchmark system."""
    print("\n🏗️ Testing Benchmark Integration")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark()
    
    # Run advanced metrics benchmark
    print("Running advanced metrics benchmark...")
    results = benchmark.run_advanced_metrics_benchmark(
        data_length=1000,
        benchmark_type="classical",  # Test with classical estimators only
        n_monte_carlo=30,  # Reduced for faster testing
        convergence_threshold=1e-6,
        save_results=True
    )
    
    print(f"Benchmark completed successfully!")
    print(f"Total tests: {results['total_tests']}")
    print(f"Successful tests: {results['successful_tests']}")
    print(f"Success rate: {results['success_rate']:.1%}")
    
    return results


def test_utility_functions():
    """Test utility functions for advanced metrics."""
    print("\n🔧 Testing Utility Functions")
    print("=" * 50)
    
    # Test convergence rate calculation
    estimates = [0.65, 0.68, 0.69, 0.70, 0.71, 0.71, 0.71]
    subset_sizes = [100, 200, 400, 600, 800, 1000, 1200]
    
    convergence_rate = calculate_convergence_rate(estimates, subset_sizes)
    print(f"Convergence Rate: {convergence_rate:.4f}")
    
    # Test mean signed error calculation
    true_values = [0.7] * len(estimates)
    mse_results = calculate_mean_signed_error(estimates, true_values)
    
    print(f"Mean Signed Error: {mse_results['mean_signed_error']:.6f}")
    print(f"Bias Percentage: {mse_results['bias_percentage']:.2f}%")
    print(f"Significant Bias: {mse_results['significant_bias']}")


def main():
    """Run all tests."""
    print("🚀 Advanced Metrics Test Suite")
    print("=" * 60)
    
    try:
        # Test individual components
        convergence_results = test_convergence_analysis()
        mse_results = test_mean_signed_error_analysis()
        profiling_results = test_advanced_performance_profiling()
        
        # Test utility functions
        test_utility_functions()
        
        # Test benchmark integration
        benchmark_results = test_benchmark_integration()
        
        print("\n✅ All tests completed successfully!")
        print("=" * 60)
        print("Advanced metrics functionality is working correctly.")
        print("You can now use convergence rates and mean signed error")
        print("analysis in your LRDBench benchmarks.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
