"""
Comprehensive Benchmark Demo for Fractional Parameter Estimation

This script demonstrates the complete benchmarking system including:
1. Data generation with various contamination scenarios
2. Training of all neural models (PINN, PINO, Neural ODE, Neural SDE)
3. Classical and ML estimator evaluation
4. Comprehensive performance comparison
5. Statistical analysis and visualization
6. Robustness testing

Usage:
    python comprehensive_benchmark_demo.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.generators import FractionalDataGenerator
from estimators.classical_estimators import ClassicalEstimatorSuite
from estimators.ml_estimators import MLEstimatorSuite
from estimators.pinn_estimator import PINNEstimator
from models.fractional_pino import FractionalPINOTrainer
from models.neural_fractional_ode import NeuralFractionalODETrainer
from models.neural_fractional_sde import NeuralFractionalSDETrainer
from models.model_persistence import ModelPersistenceManager
from benchmarks.performance_benchmark import PerformanceBenchmark
from utils.visualization import FractionalVisualizer
from utils.benchmarking import quick_benchmark, RobustnessTester

warnings.filterwarnings('ignore')

def create_demo_data():
    """Create comprehensive demo dataset."""
    print("Creating demo dataset...")
    
    data_generator = FractionalDataGenerator(seed=42)
    
    # Configuration
    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_points_list = [500, 1000]
    data_types = ['fbm', 'fgn', 'arfima']
    contamination_types = ['none', 'noise', 'outliers']
    
    demo_data = []
    
    for hurst in hurst_values:
        for n_points in n_points_list:
            for data_type in data_types:
                for contamination in contamination_types:
                    for sample_idx in range(3):  # 3 samples per configuration
                        
                        # Generate base data
                        if data_type == 'fbm':
                            data = data_generator.generate_fbm(n_points=n_points, hurst=hurst)
                        elif data_type == 'fgn':
                            data = data_generator.generate_fgn(n_points=n_points, hurst=hurst)
                        elif data_type == 'arfima':
                            data = data_generator.generate_arfima(n_points=n_points, d=hurst-0.5)
                        
                        # Apply contamination if specified
                        if contamination != 'none':
                            data = data_generator.apply_contamination(data, contamination_type=contamination)
                        
                        # Store sample
                        sample_data = {
                            'time_series': data['time_series'],
                            'true_hurst': data.get('hurst', hurst),
                            'config': f"{data_type}_h{hurst}_n{n_points}_{contamination}",
                            'sample_idx': sample_idx
                        }
                        
                        demo_data.append(sample_data)
    
    print(f"Created {len(demo_data)} demo samples")
    return demo_data

def train_neural_models(demo_data, device='auto'):
    """Train all neural models on demo data."""
    print("Training neural models...")
    
    # Use subset for training
    training_data = demo_data[:20]  # Use first 20 samples for training
    
    neural_models = {}
    
    # PINN
    print("Training PINN...")
    try:
        pinn_estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[64, 128, 128, 64],
            output_dim=1,
            learning_rate=0.001,
            device=device
        )
        pinn_estimator.build_model()
        pinn_history = pinn_estimator.train(
            training_data,
            epochs=200,  # Reduced for demo
            early_stopping_patience=20,
            save_model=True,
            model_description="PINN for comprehensive demo",
            model_tags=['demo', 'pinn']
        )
        neural_models['PINN'] = pinn_estimator.estimate
        print("PINN training completed")
    except Exception as e:
        print(f"PINN training failed: {e}")
    
    # PINO
    print("Training PINO...")
    try:
        pino_trainer = FractionalPINOTrainer(
            input_dim=1,
            hidden_dims=[64, 128, 128, 64],
            modes=16,
            learning_rate=0.001,
            device=device
        )
        pino_history = pino_trainer.train(
            training_data,
            epochs=200,  # Reduced for demo
            early_stopping_patience=20,
            save_model=True,
            model_description="PINO for comprehensive demo",
            model_tags=['demo', 'pino']
        )
        neural_models['PINO'] = pino_trainer.estimate
        print("PINO training completed")
    except Exception as e:
        print(f"PINO training failed: {e}")
    
    # Neural ODE
    print("Training Neural ODE...")
    try:
        ode_trainer = NeuralFractionalODETrainer(
            input_dim=1,
            hidden_dims=[64, 128, 64],
            alpha=0.5,
            learning_rate=0.001,
            device=device
        )
        ode_history = ode_trainer.train(
            training_data,
            epochs=200,  # Reduced for demo
            early_stopping_patience=20,
            save_model=True,
            model_description="Neural ODE for comprehensive demo",
            model_tags=['demo', 'neural_ode']
        )
        neural_models['Neural_ODE'] = ode_trainer.estimate
        print("Neural ODE training completed")
    except Exception as e:
        print(f"Neural ODE training failed: {e}")
    
    # Neural SDE
    print("Training Neural SDE...")
    try:
        sde_trainer = NeuralFractionalSDETrainer(
            input_dim=1,
            hidden_dims=[64, 128, 64],
            hurst=0.7,
            learning_rate=0.001,
            device=device
        )
        sde_history = sde_trainer.train(
            training_data,
            epochs=200,  # Reduced for demo
            early_stopping_patience=20,
            save_model=True,
            model_description="Neural SDE for comprehensive demo",
            model_tags=['demo', 'neural_sde']
        )
        neural_models['Neural_SDE'] = sde_trainer.estimate
        print("Neural SDE training completed")
    except Exception as e:
        print(f"Neural SDE training failed: {e}")
    
    return neural_models

def setup_classical_estimators():
    """Setup classical estimators."""
    print("Setting up classical estimators...")
    
    classical_suite = ClassicalEstimatorSuite()
    
    # Create wrapper functions for benchmarking
    classical_estimators = {}
    
    for estimator_name in ['DFA', 'RS', 'Wavelet', 'Spectral', 'Higuchi', 'DMA']:
        def make_estimator(name):
            def estimator(time_series):
                try:
                    if name == 'DFA':
                        return classical_suite.dfa_estimator.estimate(time_series)
                    elif name == 'RS':
                        return classical_suite.rs_estimator.estimate(time_series)
                    elif name == 'Wavelet':
                        return classical_suite.wavelet_estimator.estimate(time_series)
                    elif name == 'Spectral':
                        return classical_suite.spectral_estimator.estimate(time_series)
                    elif name == 'Higuchi':
                        return classical_suite.higuchi_estimator.estimate(time_series)
                    elif name == 'DMA':
                        return classical_suite.dma_estimator.estimate(time_series)
                except:
                    return None
            return estimator
        
        classical_estimators[name] = make_estimator(estimator_name)
    
    return classical_estimators

def setup_ml_estimators(demo_data):
    """Setup and train ML estimators."""
    print("Setting up ML estimators...")
    
    ml_suite = MLEstimatorSuite()
    
    # Train on subset of data
    training_data = demo_data[:15]  # Use first 15 samples for training
    
    try:
        ml_suite.train_all(training_data)
        print("ML estimators training completed")
    except Exception as e:
        print(f"ML estimators training failed: {e}")
    
    # Create wrapper functions
    ml_estimators = {}
    
    for estimator_name in ['RandomForest', 'GradientBoosting', 'SVR', 'LinearRegression']:
        def make_ml_estimator(name):
            def estimator(time_series):
                try:
                    if name == 'RandomForest':
                        return ml_suite.random_forest_estimator.estimate(time_series)
                    elif name == 'GradientBoosting':
                        return ml_suite.gradient_boosting_estimator.estimate(time_series)
                    elif name == 'SVR':
                        return ml_suite.svr_estimator.estimate(time_series)
                    elif name == 'LinearRegression':
                        return ml_suite.linear_regression_estimator.estimate(time_series)
                except:
                    return None
            return estimator
        
        ml_estimators[name] = make_ml_estimator(estimator_name)
    
    return ml_estimators

def run_comprehensive_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 80)
    print("COMPREHENSIVE FRACTIONAL PARAMETER ESTIMATION BENCHMARK")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Setup all estimators
    neural_models = train_neural_models(demo_data)
    classical_estimators = setup_classical_estimators()
    ml_estimators = setup_ml_estimators(demo_data)
    
    # Combine all estimators
    all_estimators = {}
    all_estimators.update(neural_models)
    all_estimators.update(classical_estimators)
    all_estimators.update(ml_estimators)
    
    print(f"Total estimators: {len(all_estimators)}")
    print(f"Neural: {list(neural_models.keys())}")
    print(f"Classical: {list(classical_estimators.keys())}")
    print(f"ML: {list(ml_estimators.keys())}")
    
    # Run benchmark
    print("\nRunning comprehensive benchmark...")
    benchmark_results = quick_benchmark(
        all_estimators, 
        demo_data,
        save_results=True,
        results_dir="comprehensive_demo_results"
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    
    print("\nPerformance Ranking (MAE):")
    print(benchmark_results['ranking'][['estimator', 'mae', 'success_rate', 'rank']].to_string(index=False))
    
    print(f"\nBenchmark completed in {time.time() - start_time:.2f} seconds")
    
    return benchmark_results

def create_visualizations(benchmark_results):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")
    
    visualizer = FractionalVisualizer()
    
    # Convert results to DataFrame format for visualization
    results_df = []
    for estimator_name, results in benchmark_results['results'].items():
        if 'error' not in results and 'true_hurst' in results:
            for i in range(len(results['true_hurst'])):
                results_df.append({
                    'estimator': estimator_name,
                    'true_hurst': results['true_hurst'][i],
                    'estimated_hurst': results['estimated_hurst'][i],
                    'absolute_error': results['errors'][i],
                    'model_type': 'neural' if estimator_name in ['PINN', 'PINO', 'Neural_ODE', 'Neural_SDE'] else 
                                 'classical' if estimator_name in ['DFA', 'RS', 'Wavelet', 'Spectral', 'Higuchi', 'DMA'] else 'ml'
                })
    
    results_df = pd.DataFrame(results_df)
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model comparison
    visualizer.plot_model_comparison(
        results_df,
        save_path=f"comprehensive_demo_results/model_comparison_{timestamp}.png",
        show_plot=False
    )
    
    # Benchmark summary
    benchmark_data = {'all_results': results_df}
    visualizer.plot_benchmark_summary(
        benchmark_data,
        save_path=f"comprehensive_demo_results/benchmark_summary_{timestamp}.png",
        show_plot=False
    )
    
    # Interactive dashboard
    try:
        visualizer.create_interactive_dashboard(
            benchmark_data,
            save_path=f"comprehensive_demo_results/interactive_dashboard_{timestamp}.html"
        )
        print("Interactive dashboard created")
    except Exception as e:
        print(f"Interactive dashboard creation failed: {e}")
    
    print("Visualizations saved to comprehensive_demo_results/")

def run_robustness_tests(benchmark_results):
    """Run robustness tests on best performing estimators."""
    print("\nRunning robustness tests...")
    
    # Get top 3 estimators
    top_estimators = benchmark_results['ranking'].head(3)['estimator'].tolist()
    
    data_generator = FractionalDataGenerator(seed=42)
    robustness_tester = RobustnessTester(data_generator)
    
    # Create base data for robustness testing
    base_data = data_generator.generate_fbm(n_points=1000, hurst=0.7)
    
    robustness_results = {}
    
    for estimator_name in top_estimators:
        if estimator_name in benchmark_results['results']:
            estimator_func = benchmark_results['results'][estimator_name].get('estimator_func')
            if estimator_func is None:
                # Try to get from original estimators
                continue
            
            print(f"Testing robustness for {estimator_name}...")
            
            # Noise robustness
            noise_results = robustness_tester.test_noise_robustness(
                estimator_func, base_data, noise_levels=[0.0, 0.1, 0.2, 0.3], n_trials=5
            )
            
            # Outlier robustness
            outlier_results = robustness_tester.test_outlier_robustness(
                estimator_func, base_data, outlier_fractions=[0.0, 0.05, 0.1, 0.15], n_trials=5
            )
            
            robustness_results[estimator_name] = {
                'noise': noise_results,
                'outliers': outlier_results
            }
    
    # Plot robustness results
    if robustness_results:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Noise robustness
        ax1 = axes[0]
        for estimator_name, results in robustness_results.items():
            noise_levels = list(results['noise'].keys())
            mean_errors = [np.mean(results['noise'][level]) if results['noise'][level] else np.nan 
                          for level in noise_levels]
            ax1.plot(noise_levels, mean_errors, 'o-', label=estimator_name, linewidth=2, markersize=8)
        
        ax1.set_title('Noise Robustness Test')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Outlier robustness
        ax2 = axes[1]
        for estimator_name, results in robustness_results.items():
            outlier_fractions = list(results['outliers'].keys())
            mean_errors = [np.mean(results['outliers'][frac]) if results['outliers'][frac] else np.nan 
                          for frac in outlier_fractions]
            ax2.plot(outlier_fractions, mean_errors, 'o-', label=estimator_name, linewidth=2, markersize=8)
        
        ax2.set_title('Outlier Robustness Test')
        ax2.set_xlabel('Outlier Fraction')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"comprehensive_demo_results/robustness_tests_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Robustness tests completed and visualized")

def main():
    """Main function to run the comprehensive demo."""
    print("Starting Comprehensive Fractional Parameter Estimation Benchmark Demo")
    print("=" * 80)
    
    try:
        # Run comprehensive benchmark
        benchmark_results = run_comprehensive_benchmark()
        
        # Create visualizations
        create_visualizations(benchmark_results)
        
        # Run robustness tests
        run_robustness_tests(benchmark_results)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nResults saved to: comprehensive_demo_results/")
        print("\nKey findings:")
        print("- Check the performance ranking for best estimators")
        print("- Review visualizations for detailed analysis")
        print("- Examine robustness test results for reliability")
        print("- Statistical significance tests are included in the report")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
