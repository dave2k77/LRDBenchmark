"""
Example script demonstrating Fractional Brownian Motion generation.

This script shows how to use the fBm model to generate synthetic data
and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))
from models.data_models.fbm.fbm_model import FractionalBrownianMotion


def main():
    """Main function demonstrating fBm generation and analysis."""
    
    print("Fractional Brownian Motion (fBm) Example")
    print("=" * 50)
    
    # Create fBm model with different Hurst parameters
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    n = 1000
    
    # Generate data for each Hurst parameter
    fbm_data = {}
    for H in hurst_values:
        print(f"Generating fBm with H = {H}")
        fbm = FractionalBrownianMotion(H=H, sigma=1.0)
        data = fbm.generate(n, seed=42)
        fbm_data[H] = data
        
        # Print theoretical properties
        properties = fbm.get_theoretical_properties()
        print(f"  - Variance: {properties['variance']:.2f}")
        print(f"  - Long-range dependence: {properties['long_range_dependence']}")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, H in enumerate(hurst_values):
        data = fbm_data[H]
        ax = axes[i]
        
        # Plot the time series
        ax.plot(data, linewidth=0.8, alpha=0.8)
        ax.set_title(f'fBm with H = {H}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        ax.text(0.02, 0.98, f'Std: {np.std(data):.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/plots/fbm_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Demonstrate increments (fGn)
    print("Fractional Gaussian Noise (fGn) - Increments of fBm")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, H in enumerate(hurst_values):
        data = fbm_data[H]
        fbm = FractionalBrownianMotion(H=H, sigma=1.0)
        increments = fbm.get_increments(data)
        
        ax = axes[i]
        ax.plot(increments, linewidth=0.8, alpha=0.8)
        ax.set_title(f'fGn (increments) with H = {H}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Increment')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Std: {np.std(increments):.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/plots/fgn_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare different generation methods
    print("Comparing different generation methods")
    print("=" * 50)
    
    H = 0.7
    methods = ['davies_harte', 'cholesky', 'circulant']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, method in enumerate(methods):
        fbm = FractionalBrownianMotion(H=H, sigma=1.0, method=method)
        data = fbm.generate(n, seed=42)
        
        ax = axes[i]
        ax.plot(data, linewidth=0.8, alpha=0.8)
        ax.set_title(f'fBm with H = {H} using {method} method')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Std: {np.std(data):.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/plots/fbm_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Example completed! Check the 'results/plots' directory for saved figures.")


if __name__ == "__main__":
    main()
