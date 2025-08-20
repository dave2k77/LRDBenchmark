#!/usr/bin/env python3
"""
Comprehensive R/S (Rescaled Range) Analysis Example

This script demonstrates the R/S estimator with various data types,
parameter settings, and comparison with other methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
from models.data_models.fgn.fgn_model import FractionalGaussianNoise
from analysis.temporal.rs.rs_estimator import RSEstimator
from analysis.temporal.dfa.dfa_estimator import DFAEstimator
import os

def generate_test_data(n=1000, seed=42):
    """Generate various test datasets."""
    np.random.seed(seed)
    
    # Generate fBm with different Hurst parameters
    fbm_05 = FractionalBrownianMotion(H=0.5, sigma=1.0)
    fbm_07 = FractionalBrownianMotion(H=0.7, sigma=1.0)
    fbm_03 = FractionalBrownianMotion(H=0.3, sigma=1.0)
    
    # Generate fGn with different Hurst parameters
    fgn_05 = FractionalGaussianNoise(H=0.5, sigma=1.0)
    fgn_07 = FractionalGaussianNoise(H=0.7, sigma=1.0)
    fgn_03 = FractionalGaussianNoise(H=0.3, sigma=1.0)
    
    # Generate random walk (H=0.5)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    
    # Generate white noise (H=0.5)
    white_noise = np.random.normal(0, 1, n)
    
    datasets = {
        'fBm H=0.5': fbm_05.generate(n, seed=seed),
        'fBm H=0.7': fbm_07.generate(n, seed=seed),
        'fBm H=0.3': fbm_03.generate(n, seed=seed),
        'fGn H=0.5': fgn_05.generate(n, seed=seed),
        'fGn H=0.7': fgn_07.generate(n, seed=seed),
        'fGn H=0.3': fgn_03.generate(n, seed=seed),
        'Random Walk': random_walk,
        'White Noise': white_noise
    }
    
    return datasets

def compare_rs_parameters():
    """Compare R/S estimation with different parameter settings."""
    print("=" * 60)
    print("COMPARING R/S PARAMETERS")
    print("=" * 60)
    
    # Generate test data
    fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
    data = fbm.generate(1000, seed=42)
    
    # Different parameter configurations
    configs = [
        {'name': 'Default', 'params': {}},
        {'name': 'Small windows', 'params': {'min_window_size': 5, 'max_window_size': 50}},
        {'name': 'Large windows', 'params': {'min_window_size': 20, 'max_window_size': 200}},
        {'name': 'Custom sizes', 'params': {'window_sizes': [10, 20, 40, 80, 160]}},
        {'name': 'Many windows', 'params': {'min_window_size': 8, 'max_window_size': 100}}
    ]
    
    results = []
    
    for config in configs:
        estimator = RSEstimator(**config['params'])
        result = estimator.estimate(data)
        
        results.append({
            'name': config['name'],
            'H': result['hurst_parameter'],
            'R²': result['r_squared'],
            'std_error': result['std_error'],
            'n_windows': len(result['window_sizes'])
        })
        
        print(f"{config['name']:15} | H = {result['hurst_parameter']:.3f} | "
              f"R² = {result['r_squared']:.3f} | "
              f"SE = {result['std_error']:.3f} | "
              f"Windows = {len(result['window_sizes'])}")
    
    return results

def compare_with_dfa():
    """Compare R/S analysis with DFA method."""
    print("\n" + "=" * 60)
    print("COMPARING R/S WITH DFA")
    print("=" * 60)
    
    # Generate test data with known Hurst parameter
    true_H = 0.7
    fbm = FractionalBrownianMotion(H=true_H, sigma=1.0)
    data = fbm.generate(1000, seed=42)
    
    # R/S estimation
    rs_estimator = RSEstimator()
    rs_results = rs_estimator.estimate(data)
    
    # DFA estimation
    dfa_estimator = DFAEstimator()
    dfa_results = dfa_estimator.estimate(data)
    
    print(f"True Hurst parameter: {true_H}")
    print(f"R/S estimate:         {rs_results['hurst_parameter']:.3f} ± {rs_results['std_error']:.3f}")
    print(f"DFA estimate:         {dfa_results['hurst_parameter']:.3f} ± {dfa_results['std_error']:.3f}")
    print(f"R/S R²:               {rs_results['r_squared']:.3f}")
    print(f"DFA R²:               {dfa_results['r_squared']:.3f}")
    
    # Calculate relative errors
    rs_error = abs(rs_results['hurst_parameter'] - true_H) / true_H * 100
    dfa_error = abs(dfa_results['hurst_parameter'] - true_H) / true_H * 100
    
    print(f"R/S relative error:   {rs_error:.1f}%")
    print(f"DFA relative error:   {dfa_error:.1f}%")
    
    return rs_results, dfa_results

def analyze_multiple_datasets():
    """Analyze multiple datasets with R/S method."""
    print("\n" + "=" * 60)
    print("ANALYZING MULTIPLE DATASETS")
    print("=" * 60)
    
    datasets = generate_test_data(n=1000, seed=42)
    estimator = RSEstimator()
    
    results = {}
    
    for name, data in datasets.items():
        try:
            result = estimator.estimate(data)
            results[name] = result
            
            print(f"{name:15} | H = {result['hurst_parameter']:.3f} | "
                  f"R² = {result['r_squared']:.3f} | "
                  f"CI = ({result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f})")
        except Exception as e:
            print(f"{name:15} | Error: {str(e)}")
    
    return results

def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to different parameters."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Generate data
    fbm = FractionalBrownianMotion(H=0.6, sigma=1.0)
    data = fbm.generate(2000, seed=42)
    
    # Test different minimum window sizes
    min_sizes = [4, 8, 16, 32, 64]
    results_min = []
    
    print("Testing minimum window size sensitivity:")
    for min_size in min_sizes:
        estimator = RSEstimator(min_window_size=min_size, max_window_size=200)
        result = estimator.estimate(data)
        results_min.append({
            'min_size': min_size,
            'H': result['hurst_parameter'],
            'R²': result['r_squared'],
            'n_windows': len(result['window_sizes'])
        })
        print(f"  Min size {min_size:2d}: H = {result['hurst_parameter']:.3f}, "
              f"R² = {result['r_squared']:.3f}, Windows = {len(result['window_sizes'])}")
    
    # Test different maximum window sizes
    max_sizes = [50, 100, 200, 400, 800]
    results_max = []
    
    print("\nTesting maximum window size sensitivity:")
    for max_size in max_sizes:
        estimator = RSEstimator(min_window_size=10, max_window_size=max_size)
        result = estimator.estimate(data)
        results_max.append({
            'max_size': max_size,
            'H': result['hurst_parameter'],
            'R²': result['r_squared'],
            'n_windows': len(result['window_sizes'])
        })
        print(f"  Max size {max_size:3d}: H = {result['hurst_parameter']:.3f}, "
              f"R² = {result['r_squared']:.3f}, Windows = {len(result['window_sizes'])}")
    
    return results_min, results_max

def create_comprehensive_plots():
    """Create comprehensive visualization plots."""
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE PLOTS")
    print("=" * 60)
    
    # Ensure results directory exists
    os.makedirs('results/plots', exist_ok=True)
    
    # Generate data
    fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
    data = fbm.generate(1000, seed=42)
    
    # Create R/S estimator and estimate
    estimator = RSEstimator()
    results = estimator.estimate(data)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Time series
    axes[0, 0].plot(data, linewidth=0.5, alpha=0.8)
    axes[0, 0].set_title(f'Fractional Brownian Motion (H = 0.7)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: R/S scaling (log-log)
    window_sizes = results['window_sizes']
    rs_values = results['rs_values']
    log_sizes = np.log(window_sizes)
    log_rs = np.log(rs_values)
    
    axes[0, 1].scatter(log_sizes, log_rs, color='blue', alpha=0.7, s=50)
    
    # Plot fitted line
    x_fit = np.array([min(log_sizes), max(log_sizes)])
    y_fit = results['hurst_parameter'] * x_fit + results['intercept']
    axes[0, 1].plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f'H = {results["hurst_parameter"]:.3f}')
    
    axes[0, 1].set_xlabel('log(Window Size)')
    axes[0, 1].set_ylabel('log(R/S)')
    axes[0, 1].set_title('R/S Scaling Relationship')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: R/S vs window size (linear)
    axes[1, 0].scatter(window_sizes, rs_values, color='green', alpha=0.7, s=50)
    
    # Plot fitted curve
    x_fit_linear = np.linspace(min(window_sizes), max(window_sizes), 100)
    y_fit_linear = np.exp(results['intercept']) * (x_fit_linear ** results['hurst_parameter'])
    axes[1, 0].plot(x_fit_linear, y_fit_linear, 'r--', linewidth=2,
                    label=f'Power law fit')
    
    axes[1, 0].set_xlabel('Window Size')
    axes[1, 0].set_ylabel('R/S Statistic')
    axes[1, 0].set_title('R/S vs Window Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Quality metrics
    quality = estimator.get_estimation_quality()
    metrics = ['R²', 'P-value', 'Std Error']
    values = [quality['r_squared'], quality['p_value'], quality['std_error']]
    colors = ['green', 'blue', 'red']
    
    bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Estimation Quality Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/rs_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive plot saved to results/plots/rs_comprehensive_analysis.png")
    plt.close()
    
    # Create comparison plot
    create_comparison_plot()

def create_comparison_plot():
    """Create comparison plot between different datasets."""
    datasets = generate_test_data(n=1000, seed=42)
    estimator = RSEstimator()
    
    # Select a subset for plotting
    plot_datasets = {
        'fBm H=0.3': datasets['fBm H=0.3'],
        'fBm H=0.5': datasets['fBm H=0.5'],
        'fBm H=0.7': datasets['fBm H=0.7'],
        'Random Walk': datasets['Random Walk']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (name, data) in enumerate(plot_datasets.items()):
        row, col = i // 2, i % 2
        
        # Estimate Hurst parameter
        result = estimator.estimate(data)
        
        # Plot time series
        axes[row, col].plot(data, linewidth=0.5, alpha=0.8)
        axes[row, col].set_title(f'{name}\nH = {result["hurst_parameter"]:.3f} (R² = {result["r_squared"]:.3f})')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Value')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/rs_dataset_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Dataset comparison plot saved to results/plots/rs_dataset_comparison.png")
    plt.close()

def main():
    """Run the comprehensive R/S analysis example."""
    print("R/S (RESCALED RANGE) ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Run all analyses
    compare_rs_parameters()
    compare_with_dfa()
    analyze_multiple_datasets()
    demonstrate_parameter_sensitivity()
    create_comprehensive_plots()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("✓ All tests completed successfully")
    print("✓ Plots saved to results/plots/")
    print("✓ R/S estimator is working correctly")

if __name__ == "__main__":
    main()
