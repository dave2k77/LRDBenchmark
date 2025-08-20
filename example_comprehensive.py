"""
Comprehensive example demonstrating synthetic data generation and estimation.

This script shows how to:
1. Generate synthetic data using the fBm model
2. Estimate parameters using the DFA estimator
3. Compare theoretical vs estimated parameters
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
from analysis.temporal.dfa.dfa_estimator import DFAEstimator


def main():
    """Main function demonstrating comprehensive analysis."""
    
    print("Comprehensive Synthetic Data Generation and Analysis Example")
    print("=" * 60)
    
    # Set parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    n = 2000
    n_trials = 10
    
    # Results storage
    results = {}
    
    for H_true in hurst_values:
        print(f"\nAnalyzing fBm with true H = {H_true}")
        print("-" * 40)
        
        # Generate multiple realizations
        estimated_H_values = []
        
        for trial in range(n_trials):
            # Generate synthetic data
            fbm = FractionalBrownianMotion(H=H_true, sigma=1.0)
            data = fbm.generate(n, seed=42 + trial)
            
            # Estimate Hurst parameter using DFA
            dfa_estimator = DFAEstimator(min_box_size=8, max_box_size=n//8)
            dfa_results = dfa_estimator.estimate(data)
            
            estimated_H = dfa_results['hurst_parameter']
            estimated_H_values.append(estimated_H)
            
            if trial == 0:  # Save first trial for visualization
                results[H_true] = {
                    'data': data,
                    'dfa_results': dfa_results,
                    'estimator': dfa_estimator
                }
        
        # Calculate statistics
        mean_H = np.mean(estimated_H_values)
        std_H = np.std(estimated_H_values)
        bias = mean_H - H_true
        
        print(f"  True H: {H_true:.3f}")
        print(f"  Estimated H (mean ± std): {mean_H:.3f} ± {std_H:.3f}")
        print(f"  Bias: {bias:.3f}")
        print(f"  Relative error: {abs(bias)/H_true*100:.1f}%")
    
    # Create comprehensive visualization
    create_comprehensive_plot(results, hurst_values)
    
    print("\nAnalysis completed! Check the 'results/plots' directory for saved figures.")


def create_comprehensive_plot(results, hurst_values):
    """Create a comprehensive visualization of the results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time series
    ax1 = fig.add_subplot(gs[0, :])
    for H in hurst_values:
        data = results[H]['data']
        ax1.plot(data[:500], alpha=0.7, linewidth=0.8, label=f'H = {H}')
    ax1.set_title('Generated fBm Time Series (first 500 points)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-5: DFA scaling for each H
    for i, H in enumerate(hurst_values):
        ax = fig.add_subplot(gs[1, i])
        
        dfa_results = results[H]['dfa_results']
        estimator = results[H]['estimator']
        
        # Plot scaling relationship
        ax.scatter(dfa_results['log_sizes'], dfa_results['log_fluctuations'], 
                  alpha=0.7, s=30)
        
        # Plot fitted line
        x_fit = np.array([min(dfa_results['log_sizes']), max(dfa_results['log_sizes'])])
        y_fit = dfa_results['slope'] * x_fit + dfa_results['intercept']
        ax.plot(x_fit, y_fit, 'r-', linewidth=2)
        
        ax.set_title(f'H = {H} (Est: {dfa_results["hurst_parameter"]:.3f})')
        ax.set_xlabel('log(Box Size)')
        ax.set_ylabel('log(Fluctuation)')
        ax.grid(True, alpha=0.3)
        
        # Add R² value
        ax.text(0.05, 0.95, f'R² = {dfa_results["r_squared"]:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 6: Estimation accuracy
    ax6 = fig.add_subplot(gs[2, :2])
    
    true_H = np.array(hurst_values)
    estimated_H = np.array([results[H]['dfa_results']['hurst_parameter'] for H in hurst_values])
    
    ax6.scatter(true_H, estimated_H, s=100, alpha=0.7)
    ax6.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect estimation')
    ax6.set_xlabel('True Hurst Parameter')
    ax6.set_ylabel('Estimated Hurst Parameter')
    ax6.set_title('Estimation Accuracy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add error bars or confidence intervals
    for i, H in enumerate(hurst_values):
        ci = results[H]['estimator'].get_confidence_intervals()
        ci_range = ci['hurst_parameter']
        ax6.vlines(true_H[i], ci_range[0], ci_range[1], alpha=0.5, colors='gray')
    
    # Plot 7: Estimation quality
    ax7 = fig.add_subplot(gs[2, 2:])
    
    r_squared_values = [results[H]['dfa_results']['r_squared'] for H in hurst_values]
    ax7.bar(range(len(hurst_values)), r_squared_values, alpha=0.7)
    ax7.set_xlabel('Hurst Parameter')
    ax7.set_ylabel('R²')
    ax7.set_title('Estimation Quality (R²)')
    ax7.set_xticks(range(len(hurst_values)))
    ax7.set_xticklabels([f'{H:.1f}' for H in hurst_values])
    ax7.grid(True, alpha=0.3)
    
    # Add horizontal line at R² = 0.95
    ax7.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Excellent fit threshold')
    ax7.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to parameter choices."""
    
    print("\nParameter Sensitivity Analysis")
    print("=" * 40)
    
    # Generate data
    H_true = 0.7
    n = 1000
    fbm = FractionalBrownianMotion(H=H_true, sigma=1.0)
    data = fbm.generate(n, seed=42)
    
    # Test different DFA parameters
    min_sizes = [4, 8, 16]
    max_sizes = [n//8, n//4, n//2]
    
    fig, axes = plt.subplots(len(min_sizes), len(max_sizes), figsize=(15, 12))
    
    for i, min_size in enumerate(min_sizes):
        for j, max_size in enumerate(max_sizes):
            ax = axes[i, j]
            
            # Estimate with current parameters
            dfa_estimator = DFAEstimator(min_box_size=min_size, max_box_size=max_size)
            dfa_results = dfa_estimator.estimate(data)
            
            # Plot scaling
            ax.scatter(dfa_results['log_sizes'], dfa_results['log_fluctuations'], 
                      alpha=0.7, s=20)
            
            # Plot fitted line
            x_fit = np.array([min(dfa_results['log_sizes']), max(dfa_results['log_sizes'])])
            y_fit = dfa_results['slope'] * x_fit + dfa_results['intercept']
            ax.plot(x_fit, y_fit, 'r-', linewidth=2)
            
            ax.set_title(f'min={min_size}, max={max_size}\nH={dfa_results["hurst_parameter"]:.3f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
    demonstrate_parameter_sensitivity()
