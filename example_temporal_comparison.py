#!/usr/bin/env python3
"""
Comprehensive comparison of temporal estimators.

This script demonstrates all four temporal estimators (DFA, R/S, Higuchi, DMA)
working together on various synthetic data types, providing comparisons
and comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

# Import models
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
from models.data_models.fgn.fgn_model import FractionalGaussianNoise
from models.data_models.mrw.mrw_model import MultifractalRandomWalk
from models.data_models.arfima.arfima_model import ARFIMAModel

# Import estimators
from analysis.temporal.dfa.dfa_estimator import DFAEstimator
from analysis.temporal.rs.rs_estimator import RSEstimator
from analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
from analysis.temporal.dma.dma_estimator import DMAEstimator

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create results directory
os.makedirs("results/plots", exist_ok=True)


def generate_test_datasets() -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Generate various test datasets with known Hurst parameters.
    
    Returns
    -------
    Dict[str, Tuple[np.ndarray, float]]
        Dictionary mapping dataset names to (data, true_hurst) tuples.
    """
    datasets = {}
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. fBm with H = 0.3 (anti-persistent)
    fbm_03 = FractionalBrownianMotion(H=0.3)
    datasets['fBm H=0.3'] = (fbm_03.generate(2000), 0.3)
    
    # 2. fBm with H = 0.5 (random walk)
    fbm_05 = FractionalBrownianMotion(H=0.5)
    datasets['fBm H=0.5'] = (fbm_05.generate(2000), 0.5)
    
    # 3. fBm with H = 0.7 (persistent)
    fbm_07 = FractionalBrownianMotion(H=0.7)
    datasets['fBm H=0.7'] = (fbm_07.generate(2000), 0.7)
    
    # 4. fGn with H = 0.3
    fgn_03 = FractionalGaussianNoise(H=0.3)
    datasets['fGn H=0.3'] = (fgn_03.generate(2000), 0.3)
    
    # 5. fGn with H = 0.7
    fgn_07 = FractionalGaussianNoise(H=0.7)
    datasets['fGn H=0.7'] = (fgn_07.generate(2000), 0.7)
    
    # 6. MRW with H = 0.6
    mrw_06 = MultifractalRandomWalk(H=0.6, lambda_param=0.1)
    datasets['MRW H=0.6'] = (mrw_06.generate(2000), 0.6)
    
    # 7. ARFIMA with d = 0.3 (H = 0.8)
    arfima_03 = ARFIMAModel(d=0.3)
    datasets['ARFIMA d=0.3'] = (arfima_03.generate(2000), 0.8)
    
    return datasets


def create_estimators() -> Dict[str, object]:
    """
    Create instances of all temporal estimators.
    
    Returns
    -------
    Dict[str, object]
        Dictionary mapping estimator names to estimator instances.
    """
    estimators = {
        'DFA': DFAEstimator(min_box_size=10, max_box_size=200),
        'R/S': RSEstimator(min_window_size=10, max_window_size=200),
        'Higuchi': HiguchiEstimator(min_k=2, max_k=50),
        'DMA': DMAEstimator(min_window_size=4, max_window_size=100)
    }
    return estimators


def estimate_hurst_all_methods(data: np.ndarray, estimators: Dict[str, object]) -> Dict[str, Dict]:
    """
    Estimate Hurst parameter using all methods.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    estimators : Dict[str, object]
        Dictionary of estimator instances.
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping estimator names to their results.
    """
    results = {}
    
    for name, estimator in estimators.items():
        try:
            result = estimator.estimate(data)
            results[name] = result
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = None
    
    return results


def compare_estimators(datasets: Dict[str, Tuple[np.ndarray, float]], 
                      estimators: Dict[str, object]) -> Dict[str, Dict]:
    """
    Compare all estimators on all datasets.
    
    Parameters
    ----------
    datasets : Dict[str, Tuple[np.ndarray, float]]
        Dictionary of datasets with true Hurst values.
    estimators : Dict[str, object]
        Dictionary of estimator instances.
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping dataset names to estimator results.
    """
    all_results = {}
    
    for dataset_name, (data, true_hurst) in datasets.items():
        print(f"Processing {dataset_name}...")
        results = estimate_hurst_all_methods(data, estimators)
        all_results[dataset_name] = {
            'true_hurst': true_hurst,
            'estimates': results
        }
    
    return all_results


def create_comparison_plots(all_results: Dict[str, Dict], 
                           estimators: Dict[str, object]) -> None:
    """
    Create comprehensive comparison plots.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results from all estimators on all datasets.
    estimators : Dict[str, object]
        Dictionary of estimator instances.
    """
    dataset_names = list(all_results.keys())
    estimator_names = list(estimators.keys())
    
    # Extract results for plotting
    true_values = [all_results[name]['true_hurst'] for name in dataset_names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Temporal Estimators', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(estimator_names)))
    
    for i, estimator_name in enumerate(estimator_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        estimated_values = []
        valid_datasets = []
        valid_true = []
        
        for j, dataset_name in enumerate(dataset_names):
            result = all_results[dataset_name]['estimates'].get(estimator_name)
            if result is not None and 'hurst_parameter' in result:
                estimated_values.append(result['hurst_parameter'])
                valid_datasets.append(dataset_name)
                valid_true.append(true_values[j])
        
        if estimated_values:
            # Plot estimated vs true
            ax.scatter(valid_true, estimated_values, c=[colors[i]], 
                      s=100, alpha=0.7, label=f'{estimator_name}')
            
            # Add perfect correlation line
            min_val = min(min(valid_true), min(estimated_values))
            max_val = max(max(valid_true), max(estimated_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Add dataset labels
            for j, dataset_name in enumerate(valid_datasets):
                ax.annotate(dataset_name, (valid_true[j], estimated_values[j]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('True Hurst Parameter')
        ax.set_ylabel('Estimated Hurst Parameter')
        ax.set_title(f'{estimator_name} Estimator')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/temporal_estimators_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def create_accuracy_analysis(all_results: Dict[str, Dict], 
                           estimators: Dict[str, object]) -> None:
    """
    Create accuracy analysis plots.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results from all estimators on all datasets.
    estimators : Dict[str, object]
        Dictionary of estimator instances.
    """
    dataset_names = list(all_results.keys())
    estimator_names = list(estimators.keys())
    
    # Calculate accuracy metrics
    metrics = {}
    
    for estimator_name in estimator_names:
        errors = []
        r_squared_values = []
        
        for dataset_name in dataset_names:
            result = all_results[dataset_name]['estimates'].get(estimator_name)
            if result is not None and 'hurst_parameter' in result:
                true_hurst = all_results[dataset_name]['true_hurst']
                estimated_hurst = result['hurst_parameter']
                
                # Absolute error
                error = abs(estimated_hurst - true_hurst)
                errors.append(error)
                
                # R-squared if available
                if 'r_squared' in result:
                    r_squared_values.append(result['r_squared'])
        
        if errors:
            metrics[estimator_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'mean_r_squared': np.mean(r_squared_values) if r_squared_values else None
            }
    
    # Create accuracy comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean absolute error
    estimator_names_plot = list(metrics.keys())
    mean_errors = [metrics[name]['mean_error'] for name in estimator_names_plot]
    std_errors = [metrics[name]['std_error'] for name in estimator_names_plot]
    
    bars1 = ax1.bar(estimator_names_plot, mean_errors, yerr=std_errors, 
                    capsize=5, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Estimation Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars1, mean_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{error:.3f}', ha='center', va='bottom')
    
    # Plot 2: R-squared values
    r_squared_values = [metrics[name]['mean_r_squared'] for name in estimator_names_plot]
    valid_r_squared = [(name, val) for name, val in zip(estimator_names_plot, r_squared_values) 
                      if val is not None]
    
    if valid_r_squared:
        names, values = zip(*valid_r_squared)
        bars2 = ax2.bar(names, values, alpha=0.7, color='lightgreen')
        ax2.set_ylabel('Mean R-squared')
        ax2.set_title('Estimation Quality')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/temporal_estimators_accuracy.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for estimator_name, metric in metrics.items():
        print(f"\n{estimator_name}:")
        print(f"  Mean Absolute Error: {metric['mean_error']:.4f} ± {metric['std_error']:.4f}")
        print(f"  Maximum Error: {metric['max_error']:.4f}")
        if metric['mean_r_squared'] is not None:
            print(f"  Mean R-squared: {metric['mean_r_squared']:.4f}")


def create_individual_scaling_plots(all_results: Dict[str, Dict], 
                                  estimators: Dict[str, object]) -> None:
    """
    Create individual scaling plots for each estimator on a selected dataset.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results from all estimators on all datasets.
    estimators : Dict[str, object]
        Dictionary of estimator instances.
    """
    # Use the fBm H=0.7 dataset for demonstration
    dataset_name = 'fBm H=0.7'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Scaling Plots for {dataset_name}', fontsize=16, fontweight='bold')
    
    for i, (estimator_name, estimator) in enumerate(estimators.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        result = all_results[dataset_name]['estimates'].get(estimator_name)
        if result is not None and 'hurst_parameter' in result:
            # Create the scaling plot directly in the subplot
            try:
                # Set the results in the estimator
                estimator.results = result
                
                # Get the log-log data for plotting
                if 'log_sizes' in result and 'log_fluctuations' in result:
                    # DFA or DMA
                    log_sizes = result['log_sizes']
                    log_fluctuations = result['log_fluctuations']
                    xlabel = 'log(Box Size)' if estimator_name == 'DFA' else 'log(Window Size)'
                    ylabel = 'log(Fluctuation)'
                elif 'log_sizes' in result and 'log_rs' in result:
                    # R/S
                    log_sizes = result['log_sizes']
                    log_fluctuations = result['log_rs']
                    xlabel = 'log(Window Size)'
                    ylabel = 'log(R/S)'
                elif 'log_k' in result and 'log_lengths' in result:
                    # Higuchi
                    log_sizes = result['log_k']
                    log_fluctuations = result['log_lengths']
                    xlabel = 'log(k)'
                    ylabel = 'log(Curve Length)'
                else:
                    raise ValueError(f"No plotting data found for {estimator_name}")
                
                # Plot the data points
                ax.scatter(log_sizes, log_fluctuations, color='blue', alpha=0.7, s=50, label='Data points')
                
                # Plot the fitted line
                hurst = result['hurst_parameter']
                r_squared = result.get('r_squared', 'N/A')
                
                # Calculate fitted line
                if estimator_name == 'Higuchi':
                    # Higuchi: log(L) = -D * log(k) + c, where D = 2 - H
                    D = 2 - hurst
                    slope = -D
                else:
                    # DFA, R/S, DMA: log(y) = H * log(x) + c
                    slope = hurst
                
                # Fit line through the data points
                z = np.polyfit(log_sizes, log_fluctuations, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
                y_fit = p(x_fit)
                
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
                       label=f'Fit: H = {hurst:.3f} (R² = {r_squared:.3f})')
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{estimator_name} Scaling')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Plot failed:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{estimator_name} Scaling')
        else:
            # No valid results available
            ax.text(0.5, 0.5, f'No valid results\nfor {estimator_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{estimator_name} Scaling')
    
    plt.tight_layout()
    plt.savefig('results/plots/temporal_estimators_scaling.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_parameter_sensitivity() -> None:
    """
    Demonstrate the sensitivity of estimators to parameter settings.
    """
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    fbm = FractionalBrownianMotion(H=0.6)
    data = fbm.generate(2000)
    
    # Test different parameter settings for each estimator
    parameter_settings = {
        'DFA': [
            {'min_box_size': 10, 'max_box_size': 100},
            {'min_box_size': 20, 'max_box_size': 200},
            {'min_box_size': 5, 'max_box_size': 50}
        ],
        'R/S': [
            {'min_window_size': 10, 'max_window_size': 100},
            {'min_window_size': 20, 'max_window_size': 200},
            {'min_window_size': 5, 'max_window_size': 50}
        ],
        'Higuchi': [
            {'min_k': 2, 'max_k': 30},
            {'min_k': 5, 'max_k': 50},
            {'min_k': 3, 'max_k': 20}
        ],
        'DMA': [
            {'min_window_size': 4, 'max_window_size': 50},
            {'min_window_size': 8, 'max_window_size': 100},
            {'min_window_size': 3, 'max_window_size': 25}
        ]
    }
    
    for estimator_name, settings_list in parameter_settings.items():
        print(f"\n{estimator_name} Estimator:")
        print("-" * 40)
        
        for i, settings in enumerate(settings_list):
            try:
                if estimator_name == 'DFA':
                    estimator = DFAEstimator(**settings)
                elif estimator_name == 'R/S':
                    estimator = RSEstimator(**settings)
                elif estimator_name == 'Higuchi':
                    estimator = HiguchiEstimator(**settings)
                elif estimator_name == 'DMA':
                    estimator = DMAEstimator(**settings)
                
                result = estimator.estimate(data)
                hurst = result['hurst_parameter']
                r_squared = result.get('r_squared', 'N/A')
                
                print(f"  Settings {i+1}: H = {hurst:.3f}, R² = {r_squared}")
                
            except Exception as e:
                print(f"  Settings {i+1}: Failed - {e}")


def main():
    """Main function to run the comprehensive comparison."""
    print("COMPREHENSIVE TEMPORAL ESTIMATORS COMPARISON")
    print("=" * 60)
    
    # Generate test datasets
    print("Generating test datasets...")
    datasets = generate_test_datasets()
    
    # Create estimators
    print("Creating estimators...")
    estimators = create_estimators()
    
    # Run comparisons
    print("Running comparisons...")
    all_results = compare_estimators(datasets, estimators)
    
    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(all_results, estimators)
    
    print("Creating accuracy analysis...")
    create_accuracy_analysis(all_results, estimators)
    
    print("Creating individual scaling plots...")
    create_individual_scaling_plots(all_results, estimators)
    
    # Demonstrate parameter sensitivity
    demonstrate_parameter_sensitivity()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Plots saved to results/plots/")
    print("- temporal_estimators_comparison.png")
    print("- temporal_estimators_accuracy.png") 
    print("- temporal_estimators_scaling.png")


if __name__ == "__main__":
    main()
