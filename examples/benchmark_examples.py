#!/usr/bin/env python3
"""
Example script demonstrating the enhanced LRDBench benchmark features.
"""

from analysis.benchmark import ComprehensiveBenchmark

def main():
    """Demonstrate different benchmark types and contamination options."""
    
    print("🚀 LRDBench Enhanced Benchmark Examples")
    print("=" * 50)
    
    # Initialize benchmark system
    benchmark = ComprehensiveBenchmark(output_dir="example_benchmark_results")
    
    # Example 1: Classical estimators only
    print("\n📊 Example 1: Classical Estimators Only")
    print("-" * 40)
    results_classical = benchmark.run_classical_benchmark(
        data_length=1000,
        save_results=True
    )
    
    # Example 2: ML estimators with contamination
    print("\n📊 Example 2: ML Estimators with Gaussian Noise")
    print("-" * 40)
    results_ml = benchmark.run_ml_benchmark(
        data_length=1000,
        contamination_type='additive_gaussian',
        contamination_level=0.2,
        save_results=True
    )
    
    # Example 3: Neural estimators with outliers
    print("\n📊 Example 3: Neural Estimators with Outliers")
    print("-" * 40)
    results_neural = benchmark.run_neural_benchmark(
        data_length=1000,
        contamination_type='outliers',
        contamination_level=0.1,
        save_results=True
    )
    
    # Example 4: Custom comprehensive benchmark
    print("\n📊 Example 4: Custom Comprehensive Benchmark")
    print("-" * 40)
    results_custom = benchmark.run_comprehensive_benchmark(
        data_length=1000,
        benchmark_type='comprehensive',
        contamination_type='trend',
        contamination_level=0.3,
        save_results=True
    )
    
    print("\n✅ All examples completed!")
    print("\n📁 Results saved to 'example_benchmark_results' directory")
    print("\n💡 Key Features Demonstrated:")
    print("   ✓ Different benchmark types (classical, ML, neural, comprehensive)")
    print("   ✓ Contamination options (noise, outliers, trend, etc.)")
    print("   ✓ Configurable contamination levels")
    print("   ✓ Automatic result saving and reporting")

if __name__ == "__main__":
    main()
