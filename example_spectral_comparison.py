#!/usr/bin/env python3
"""
Comprehensive Spectral Estimators Comparison

This script compares different spectral estimators for Hurst parameter estimation:
- Periodogram (with Welch averaging and multi-taper options)
- Whittle (with local Whittle option)
- GPH (with bias correction)

The script generates synthetic data from various models and evaluates
the performance of each estimator.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

# Import models
from models.data_models.fgn.fgn_model import FractionalGaussianNoise
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
from models.data_models.arfima.arfima_model import ARFIMAModel

# Import estimators
from analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
from analysis.spectral.whittle.whittle_estimator import WhittleEstimator
from analysis.spectral.gph.gph_estimator import GPHEstimator
# Temporal estimators
from analysis.temporal.dfa.dfa_estimator import DFAEstimator
from analysis.temporal.dma.dma_estimator import DMAEstimator
from analysis.temporal.rs.rs_estimator import RSEstimator
from analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator


def generate_test_datasets():
    """Generate test datasets with known Hurst parameters."""
    print("Generating test datasets...")
    
    datasets = {}
    
    # fGn datasets with different H values
    for H in [0.3, 0.5, 0.7]:
        fgn = FractionalGaussianNoise(H=H)
        data = fgn.generate(2048)
        datasets[f'fGn H={H}'] = {'data': data, 'true_H': H}
    
    # fBm increment datasets
    for H in [0.3, 0.7]:
        fbm = FractionalBrownianMotion(H=H)
        fbm_data = fbm.generate(2049)  # Need n+1 for n increments
        data = fbm.get_increments(fbm_data)
        datasets[f'fBm incr H={H}'] = {'data': data, 'true_H': H}
    
    # ARFIMA dataset
    arfima = ARFIMAModel(d=0.3, ar_params=[0.5], ma_params=[0.2])
    data = arfima.generate(2048)
    datasets['ARFIMA d=0.3'] = {'data': data, 'true_H': 0.3 + 0.5}  # H = d + 0.5
    
    return datasets


def create_estimators():
    """Create different spectral estimators for comparison."""
    print("Creating estimators...")
    
    estimators = {}
    
    # Standard estimators
    estimators['Periodogram'] = PeriodogramEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, use_welch=True
    )
    
    estimators['Periodogram (Multi-taper)'] = PeriodogramEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, use_multitaper=True, n_tapers=3
    )
    
    estimators['Whittle (Local)'] = WhittleEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, use_local_whittle=True
    )
    
    estimators['Whittle (Standard)'] = WhittleEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, use_local_whittle=False
    )
    
    estimators['GPH (Bias-corrected)'] = GPHEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, apply_bias_correction=True
    )
    
    estimators['GPH (No bias correction)'] = GPHEstimator(
        min_freq_ratio=0.01, max_freq_ratio=0.1, apply_bias_correction=False
    )

    # Temporal estimators
    estimators['DFA'] = DFAEstimator()
    estimators['DMA'] = DMAEstimator()
    estimators['R/S'] = RSEstimator()
    estimators['Higuchi'] = HiguchiEstimator()
    
    return estimators


def run_comparisons(datasets, estimators):
    """Run all estimators on all datasets."""
    print("Running comparisons...")
    
    all_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"Processing {dataset_name}...")
        data = dataset_info['data']
        true_H = dataset_info['true_H']
        
        all_results[dataset_name] = {
            'true_H': true_H,
            'estimates': {}
        }
        
        for estimator_name, estimator in estimators.items():
            try:
                result = estimator.estimate(data)
                all_results[dataset_name]['estimates'][estimator_name] = result
            except Exception as e:
                print(f"  Error with {estimator_name}: {e}")
                all_results[dataset_name]['estimates'][estimator_name] = None
    
    return all_results


def create_comparison_plots(all_results, save_dir, datasets):
    """Create comprehensive comparison plots."""
    print("Creating comparison plots...")
    
    # Extract results for plotting
    estimator_names = []
    dataset_names = []
    estimated_Hs = []
    true_Hs = []
    r_squared_values = []
    
    for dataset_name, dataset_info in all_results.items():
        true_H = dataset_info['true_H']
        for estimator_name, result in dataset_info['estimates'].items():
            if result is not None and 'hurst_parameter' in result:
                estimator_names.append(estimator_name)
                dataset_names.append(dataset_name)
                estimated_Hs.append(result['hurst_parameter'])
                true_Hs.append(true_H)
                r_squared_values.append(result.get('r_squared', np.nan))
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Estimated vs True Hurst parameters
    plt.subplot(2, 2, 1)
    unique_estimators = list(set(estimator_names))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_estimators)))
    
    for i, estimator in enumerate(unique_estimators):
        mask = [name == estimator for name in estimator_names]
        est_H = [estimated_Hs[j] for j, m in enumerate(mask) if m]
        true_H = [true_Hs[j] for j, m in enumerate(mask) if m]
        plt.scatter(true_H, est_H, label=estimator, color=colors[i], alpha=0.7, s=60)
    
    # Plot diagonal line
    min_H, max_H = min(true_Hs), max(true_Hs)
    plt.plot([min_H, max_H], [min_H, max_H], 'k--', alpha=0.5, label='Perfect')
    plt.xlabel('True Hurst Parameter')
    plt.ylabel('Estimated Hurst Parameter')
    plt.title('Estimated vs True Hurst Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error heatmap (datasets x estimators)
    plt.subplot(2, 2, 2)
    ds_names = list(all_results.keys())
    est_names = sorted(list({ename for ds in all_results.values() for ename in ds['estimates'].keys()}))
    error_matrix = np.full((len(ds_names), len(est_names)), np.nan)

    name_to_col = {name: i for i, name in enumerate(est_names)}
    for r, ds in enumerate(ds_names):
        true_H = all_results[ds]['true_H']
        for est_name, res in all_results[ds]['estimates'].items():
            if res is not None and 'hurst_parameter' in res:
                c = name_to_col[est_name]
                error_matrix[r, c] = abs(res['hurst_parameter'] - true_H)

    im = plt.imshow(error_matrix, cmap='viridis', aspect='auto')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('|H_est − H_true|')
    plt.xticks(range(len(est_names)), est_names, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(ds_names)), ds_names, fontsize=8)
    # Annotate cells
    mean_val = np.nanmean(error_matrix)
    for i in range(error_matrix.shape[0]):
        for j in range(error_matrix.shape[1]):
            val = error_matrix[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                         color='white' if val > mean_val else 'black')
    plt.title('Absolute Error Heatmap', fontsize=10)
    
    # Plot 3: R-squared values
    plt.subplot(2, 2, 3)
    r_squared_by_estimator = {}
    for i, estimator in enumerate(unique_estimators):
        mask = [name == estimator for name in estimator_names]
        r_sq = [r_squared_values[j] for j, m in enumerate(mask) if m and not np.isnan(r_squared_values[j])]
        r_squared_by_estimator[estimator] = r_sq
    
    plt.boxplot(r_squared_by_estimator.values(), labels=r_squared_by_estimator.keys())
    plt.ylabel('R-squared')
    plt.title('R-squared Values by Estimator')
    plt.xticks(rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Mean absolute error by estimator
    plt.subplot(2, 2, 4)
    errors = [est - true for est, true in zip(estimated_Hs, true_Hs)]
    mae_by_estimator = {}
    for i, estimator in enumerate(unique_estimators):
        mask = [name == estimator for name in estimator_names]
        est_errors = [abs(errors[j]) for j, m in enumerate(mask) if m]
        mae_by_estimator[estimator] = np.mean(est_errors)
    
    estimators_sorted = sorted(mae_by_estimator.keys(), key=lambda x: mae_by_estimator[x])
    maes_sorted = [mae_by_estimator[est] for est in estimators_sorted]
    
    bars = plt.bar(range(len(estimators_sorted)), maes_sorted)
    plt.xlabel('Estimator')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error by Estimator')
    plt.xticks(range(len(estimators_sorted)), estimators_sorted, rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if maes_sorted[i] < 0.1:
            bar.set_color('green')
        elif maes_sorted[i] < 0.2:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'spectral_estimators_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_accuracy_analysis(all_results):
    """Create detailed accuracy analysis."""
    print("Creating accuracy analysis...")
    
    accuracy_summary = {}
    
    for dataset_name, dataset_info in all_results.items():
        true_H = dataset_info['true_H']
        
        for estimator_name, result in dataset_info['estimates'].items():
            if result is not None and 'hurst_parameter' in result:
                error = abs(result['hurst_parameter'] - true_H)
                r_squared = result.get('r_squared', np.nan)
                
                if estimator_name not in accuracy_summary:
                    accuracy_summary[estimator_name] = {'errors': [], 'r_squared': []}
                
                accuracy_summary[estimator_name]['errors'].append(error)
                accuracy_summary[estimator_name]['r_squared'].append(r_squared)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    
    for estimator_name, stats in accuracy_summary.items():
        errors = stats['errors']
        r_squared = [r for r in stats['r_squared'] if not np.isnan(r)]
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        mean_r_squared = np.mean(r_squared) if r_squared else np.nan
        
        print(f"\n{estimator_name}:")
        print(f"  Mean Absolute Error: {mean_error:.4f} ± {std_error:.4f}")
        print(f"  Maximum Error: {max_error:.4f}")
        print(f"  Mean R-squared: {mean_r_squared:.4f}")
    
    return accuracy_summary


def create_individual_scaling_plots(all_results, save_dir):
    """Create individual scaling plots for each estimator."""
    print("Creating individual scaling plots...")
    
    dataset_names = list(all_results.keys())
    estimator_names = list(next(iter(all_results.values()))['estimates'].keys())
    
    # Split estimators into categories
    spectral_estimators = [name for name in estimator_names if any(keyword in name.lower() 
                                                                   for keyword in ['periodogram', 'whittle', 'gph'])]
    temporal_estimators = [name for name in estimator_names if any(keyword in name.lower() 
                                                                  for keyword in ['dfa', 'dma', 'rs', 'higuchi'])]
    
    # Create spectral estimators plot
    if spectral_estimators:
        n_datasets = len(all_results)
        n_estimators = len(spectral_estimators)
        
        fig, axes = plt.subplots(n_datasets, n_estimators, 
                                figsize=(4*n_estimators, 4*n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        if n_estimators == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset_name in enumerate(dataset_names):
            for j, estimator_name in enumerate(spectral_estimators):
                ax = axes[i, j]
                
                result = all_results[dataset_name]['estimates'].get(estimator_name)
                if result is not None and 'hurst_parameter' in result:
                    try:
                        # Get the log-log data for plotting
                        if 'log_freq' in result and 'log_psd' in result:
                            # Spectral methods
                            log_sizes = result['log_freq']
                            log_fluctuations = result['log_psd']
                            xlabel = 'log(Frequency)'
                            ylabel = 'log(PSD)'
                        elif 'log_regressor' in result and 'log_periodogram' in result:
                            # GPH
                            log_sizes = result['log_regressor']
                            log_fluctuations = result['log_periodogram']
                            xlabel = 'log(4 sin²(ω/2))'
                            ylabel = 'log(Periodogram)'
                        elif 'log_model' in result and 'log_periodogram' in result:
                            # Whittle
                            log_sizes = result['log_model']
                            log_fluctuations = result['log_periodogram']
                            xlabel = 'log(Model Spectrum)'
                            ylabel = 'log(Periodogram)'
                        else:
                            raise ValueError(f"No plotting data found for {estimator_name}")
                        
                        # Plot the data points
                        ax.scatter(log_sizes, log_fluctuations, color='blue', alpha=0.7, s=30)
                        
                        # Plot the fitted line
                        hurst = result['hurst_parameter']
                        r_squared = result.get('r_squared', 'N/A')
                        
                        # Fit line through the data points
                        z = np.polyfit(log_sizes, log_fluctuations, 1)
                        p = np.poly1d(z)
                        x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
                        y_fit = p(x_fit)
                        
                        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, 
                               label=f'H = {hurst:.3f}')
                        
                        ax.set_xlabel(xlabel, fontsize=6)
                        ax.set_ylabel(ylabel, fontsize=6)
                        ax.set_title(f'{estimator_name}\nH = {hurst:.3f} (R² = {r_squared:.3f})', 
                                   fontsize=7)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=5)
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=6)
                        ax.set_title(f'{estimator_name}\nError', fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=6)
                    ax.set_title(f'{estimator_name}\nNo data', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'spectral_estimators_scaling.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create temporal estimators plot (if any exist)
    if temporal_estimators:
        n_datasets = len(all_results)
        n_estimators = len(temporal_estimators)
        
        fig, axes = plt.subplots(n_datasets, n_estimators, 
                                figsize=(4*n_estimators, 4*n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        if n_estimators == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset_name in enumerate(dataset_names):
            for j, estimator_name in enumerate(temporal_estimators):
                ax = axes[i, j]
                
                result = all_results[dataset_name]['estimates'].get(estimator_name)
                if result is not None and 'hurst_parameter' in result:
                    try:
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
                        ax.scatter(log_sizes, log_fluctuations, color='blue', alpha=0.7, s=30)
                        
                        # Plot the fitted line
                        hurst = result['hurst_parameter']
                        r_squared = result.get('r_squared', 'N/A')
                        
                        # Calculate fitted line
                        if estimator_name == 'Higuchi':
                            # Higuchi: log(L) = -D * log(k) + c, where D = 2 - H
                            D = 2 - hurst
                            slope = -D
                        else:
                            # Most methods: log(y) = H * log(x) + c
                            slope = hurst
                        
                        # Fit line through the data points
                        z = np.polyfit(log_sizes, log_fluctuations, 1)
                        p = np.poly1d(z)
                        x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
                        y_fit = p(x_fit)
                        
                        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, 
                               label=f'H = {hurst:.3f}')
                        
                        ax.set_xlabel(xlabel, fontsize=6)
                        ax.set_ylabel(ylabel, fontsize=6)
                        ax.set_title(f'{estimator_name}\nH = {hurst:.3f} (R² = {r_squared:.3f})', 
                                   fontsize=7)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=5)
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=6)
                        ax.set_title(f'{estimator_name}\nError', fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=6)
                    ax.set_title(f'{estimator_name}\nNo data', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'temporal_estimators_scaling.png', dpi=300, bbox_inches='tight')
        plt.show()


def parameter_sensitivity_analysis():
    """Test parameter sensitivity for each estimator."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7)
    test_data = fgn.generate(1024)
    
    # Test different parameter settings
    test_settings = [
        {'min_freq_ratio': 0.01, 'max_freq_ratio': 0.1},
        {'min_freq_ratio': 0.02, 'max_freq_ratio': 0.15},
        {'min_freq_ratio': 0.005, 'max_freq_ratio': 0.08},
    ]
    
    for estimator_class, name in [
        (PeriodogramEstimator, "Periodogram Estimator"),
        (WhittleEstimator, "Whittle Estimator"),
        (GPHEstimator, "GPH Estimator")
    ]:
        print(f"\n{name}:")
        print("-" * 40)
        
        for i, settings in enumerate(test_settings, 1):
            try:
                estimator = estimator_class(**settings)
                result = estimator.estimate(test_data)
                hurst = result['hurst_parameter']
                r_squared = result.get('r_squared', 'N/A')
                print(f"  Settings {i}: H = {hurst:.3f}, R² = {r_squared}")
            except Exception as e:
                print(f"  Settings {i}: Error - {e}")


def main():
    """Main function to run the comprehensive comparison."""
    print("COMPREHENSIVE SPECTRAL ESTIMATORS COMPARISON")
    print("=" * 60)
    
    # Create results directory
    save_dir = Path("results/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test datasets
    datasets = generate_test_datasets()
    
    # Create estimators
    estimators = create_estimators()
    
    # Run comparisons
    all_results = run_comparisons(datasets, estimators)
    
    # Create comparison plots
    create_comparison_plots(all_results, save_dir, datasets)
    
    # Create accuracy analysis
    create_accuracy_analysis(all_results)
    
    # Create individual scaling plots
    create_individual_scaling_plots(all_results, save_dir)
    
    # Parameter sensitivity analysis
    parameter_sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Plots saved to {save_dir}/")
    print("- spectral_estimators_comparison.png")
    print("- spectral_estimators_accuracy.png")
    print("- spectral_estimators_scaling.png")
    print("- temporal_estimators_scaling.png")


if __name__ == "__main__":
    main()
