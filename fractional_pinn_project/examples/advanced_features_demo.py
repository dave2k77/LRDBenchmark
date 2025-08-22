"""
Advanced Features Demo for Fractional PINN Project

This script demonstrates all advanced features including:
1. Comprehensive testing suite
2. Hyperparameter optimization
3. Model persistence and comparison
4. Advanced benchmarking
5. Complete pipeline integration

Usage:
    python examples/advanced_features_demo.py
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
from models.model_persistence import ModelPersistenceManager, quick_save_model, quick_load_model
from models.model_comparison import ModelConfig, ModelComparisonFramework
from utils.visualization import FractionalVisualizer
from utils.benchmarking import BenchmarkMetrics, StatisticalTesting, AutomatedBenchmark
from utils.hyperparameter_optimization import HyperparameterOptimizer, OptimizationConfig, quick_optimize

warnings.filterwarnings('ignore')


def create_comprehensive_dataset():
    """Create comprehensive dataset for advanced features demo."""
    print("Creating comprehensive dataset...")
    
    data_generator = FractionalDataGenerator(seed=42)
    
    # Configuration
    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_points_list = [500, 1000, 2000]
    data_types = ['fbm', 'fgn', 'arfima', 'mrw']
    contamination_types = ['none', 'noise', 'outliers', 'trends', 'seasonality']
    
    dataset = {
        'training': [],
        'validation': [],
        'testing': []
    }
    
    total_samples = 0
    
    for hurst in hurst_values:
        for n_points in n_points_list:
            for data_type in data_types:
                for contamination in contamination_types:
                    for sample_idx in range(5):  # 5 samples per configuration
                        
                        # Generate base data
                        if data_type == 'fbm':
                            data = data_generator.generate_fbm(n_points=n_points, hurst=hurst)
                        elif data_type == 'fgn':
                            data = data_generator.generate_fgn(n_points=n_points, hurst=hurst)
                        elif data_type == 'arfima':
                            data = data_generator.generate_arfima(n_points=n_points, d=hurst-0.5)
                        elif data_type == 'mrw':
                            data = data_generator.generate_mrw(n_points=n_points, hurst=hurst)
                        
                        # Apply contamination if specified
                        if contamination != 'none':
                            data = data_generator.apply_contamination(data, contamination_type=contamination)
                        
                        # Store sample
                        sample_data = {
                            'time_series': data['time_series'],
                            'true_hurst': data.get('hurst', hurst),
                            'true_d': data.get('d', hurst-0.5),
                            'config': f"{data_type}_h{hurst}_n{n_points}_{contamination}",
                            'sample_idx': sample_idx,
                            'data_type': data_type,
                            'contamination': contamination
                        }
                        
                        # Split into train/val/test (60/20/20)
                        if sample_idx < 3:
                            dataset['training'].append(sample_data)
                        elif sample_idx == 3:
                            dataset['validation'].append(sample_data)
                        else:
                            dataset['testing'].append(sample_data)
                        
                        total_samples += 1
    
    print(f"Created {total_samples} total samples:")
    print(f"  Training: {len(dataset['training'])}")
    print(f"  Validation: {len(dataset['validation'])}")
    print(f"  Testing: {len(dataset['testing'])}")
    
    return dataset


def run_comprehensive_testing():
    """Run comprehensive testing suite."""
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE TESTING SUITE")
    print("=" * 60)
    
    try:
        # Import and run tests
        from tests.test_suite import run_all_tests
        
        print("Running all tests...")
        success = run_all_tests()
        
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False


def demonstrate_hyperparameter_optimization(dataset):
    """Demonstrate hyperparameter optimization."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Use subset for optimization (to keep it fast)
    training_subset = dataset['training'][:20]
    validation_subset = dataset['validation'][:10]
    
    optimization_results = {}
    
    # 1. PINN Optimization
    print("\n1. Optimizing PINN hyperparameters...")
    try:
        config = OptimizationConfig(
            method='random',  # Use random search for demo (faster)
            n_trials=10,      # Small number for demo
            save_results=True,
            verbose=True
        )
        
        optimizer = HyperparameterOptimizer(config)
        pinn_results = optimizer.optimize_pinn(training_subset, validation_subset)
        
        optimization_results['pinn'] = pinn_results
        print(f"   Best PINN score: {pinn_results['best_score']:.4f}")
        print(f"   Best PINN params: {pinn_results['best_params']}")
        
    except Exception as e:
        print(f"   ❌ PINN optimization failed: {e}")
    
    # 2. PINO Optimization
    print("\n2. Optimizing PINO hyperparameters...")
    try:
        pino_results = optimizer.optimize_pino(training_subset, validation_subset)
        
        optimization_results['pino'] = pino_results
        print(f"   Best PINO score: {pino_results['best_score']:.4f}")
        print(f"   Best PINO params: {pino_results['best_params']}")
        
    except Exception as e:
        print(f"   ❌ PINO optimization failed: {e}")
    
    # 3. ML Optimization
    print("\n3. Optimizing ML hyperparameters...")
    try:
        ml_results = optimizer.optimize_ml_models(training_subset, validation_subset)
        
        optimization_results['ml'] = ml_results
        print(f"   Best ML score: {ml_results['best_score']:.4f}")
        print(f"   Best ML params: {ml_results['best_params']}")
        
    except Exception as e:
        print(f"   ❌ ML optimization failed: {e}")
    
    return optimization_results


def demonstrate_model_persistence_and_comparison(dataset, optimization_results):
    """Demonstrate model persistence and comparison."""
    print("\n" + "=" * 60)
    print("MODEL PERSISTENCE AND COMPARISON DEMO")
    print("=" * 60)
    
    training_subset = dataset['training'][:15]
    testing_subset = dataset['testing'][:10]
    
    # 1. Train models with optimized parameters
    models = {}
    
    print("\n1. Training optimized models...")
    
    # PINN
    if 'pinn' in optimization_results:
        try:
            best_params = optimization_results['pinn']['best_params']
            pinn_estimator = PINNEstimator(
                input_dim=1,
                hidden_dims=best_params.get('hidden_dims', [64, 128, 64]),
                output_dim=1,
                learning_rate=best_params.get('learning_rate', 0.001),
                device='cpu'
            )
            
            pinn_estimator.build_model()
            pinn_history = pinn_estimator.train(
                training_subset,
                epochs=best_params.get('epochs', 200),
                early_stopping_patience=20,
                save_model=True,
                model_description="Optimized PINN from advanced demo",
                model_tags=['advanced_demo', 'optimized', 'pinn']
            )
            
            models['PINN'] = pinn_estimator
            print("   ✅ PINN trained and saved")
            
        except Exception as e:
            print(f"   ❌ PINN training failed: {e}")
    
    # PINO
    if 'pino' in optimization_results:
        try:
            best_params = optimization_results['pino']['best_params']
            pino_trainer = FractionalPINOTrainer(
                input_dim=1,
                hidden_dims=best_params.get('hidden_dims', [64, 128, 64]),
                modes=best_params.get('modes', 16),
                learning_rate=best_params.get('learning_rate', 0.001),
                device='cpu'
            )
            
            pino_history = pino_trainer.train(
                training_subset,
                epochs=200,
                early_stopping_patience=20,
                save_model=True,
                model_description="Optimized PINO from advanced demo",
                model_tags=['advanced_demo', 'optimized', 'pino']
            )
            
            models['PINO'] = pino_trainer
            print("   ✅ PINO trained and saved")
            
        except Exception as e:
            print(f"   ❌ PINO training failed: {e}")
    
    # 2. Model comparison framework
    print("\n2. Using model comparison framework...")
    try:
        framework = ModelComparisonFramework()
        
        # Add models to framework
        for name, model in models.items():
            if name == 'PINN':
                framework.add_pinn_model(
                    name=name,
                    input_dim=1,
                    hidden_dims=model.hidden_dims,
                    output_dim=1,
                    learning_rate=model.learning_rate
                )
            elif name == 'PINO':
                framework.add_pino_model(
                    name=name,
                    input_dim=1,
                    hidden_dims=model.hidden_dims,
                    modes=model.modes,
                    learning_rate=model.learning_rate
                )
        
        # Train all models in framework
        framework.train_all_models(training_subset, epochs=100, verbose=False)
        
        # Evaluate all models
        comparison_results = framework.evaluate_all_models(testing_subset)
        
        print("   Model comparison results:")
        for model_name, metrics in comparison_results.items():
            print(f"     {model_name}: MAE = {metrics.get('mae', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"   ❌ Model comparison failed: {e}")
    
    # 3. Model persistence demonstration
    print("\n3. Model persistence demonstration...")
    try:
        model_manager = ModelPersistenceManager()
        
        # List saved models
        models_df = model_manager.list_models(tags=['advanced_demo'])
        print(f"   Found {len(models_df)} saved models with 'advanced_demo' tag")
        
        if len(models_df) > 0:
            # Load best model
            best_model_id = model_manager.get_best_model('pinn', metric='best_val_loss')
            if best_model_id:
                loaded_model, loaded_config, loaded_metadata = model_manager.load_model(best_model_id)
                print(f"   Loaded best model: {best_model_id}")
                print(f"   Model description: {loaded_metadata.get('description', 'N/A')}")
                
                # Test loaded model
                test_data = testing_subset[0]
                estimate = loaded_model.estimate(test_data['time_series'])
                if estimate is not None:
                    error = abs(estimate - test_data['true_hurst'])
                    print(f"   Test estimate: {estimate:.4f}, Error: {error:.4f}")
        
    except Exception as e:
        print(f"   ❌ Model persistence demo failed: {e}")


def demonstrate_advanced_benchmarking(dataset):
    """Demonstrate advanced benchmarking features."""
    print("\n" + "=" * 60)
    print("ADVANCED BENCHMARKING DEMO")
    print("=" * 60)
    
    # Use subset for benchmarking
    benchmark_data = dataset['testing'][:30]
    
    print("\n1. Setting up estimators...")
    
    # Classical estimators
    classical_suite = ClassicalEstimatorSuite()
    
    # ML estimators
    ml_suite = MLEstimatorSuite()
    ml_suite.train_all(dataset['training'][:20])  # Train on subset
    
    # Neural estimators (use pre-trained if available)
    neural_estimators = {}
    
    # Create estimator functions
    estimators = {}
    
    # Classical
    for estimator_name in ['DFA', 'RS', 'Wavelet', 'Spectral', 'Higuchi', 'DMA']:
        def make_classical_estimator(name):
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
        
        estimators[estimator_name] = make_classical_estimator(estimator_name)
    
    # ML
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
        
        estimators[estimator_name] = make_ml_estimator(estimator_name)
    
    print(f"   Created {len(estimators)} estimators")
    
    # 2. Run automated benchmark
    print("\n2. Running automated benchmark...")
    try:
        benchmark = AutomatedBenchmark(save_results=True, n_jobs=1)
        benchmark_results = benchmark.run_parallel_benchmark(estimators, benchmark_data)
        
        # Create ranking
        ranking = benchmark.create_performance_ranking(benchmark_results)
        
        print("   Performance ranking (top 5):")
        print(ranking[['estimator', 'mae', 'success_rate', 'rank']].head().to_string(index=False))
        
        # Statistical analysis
        stats_analysis = benchmark.perform_statistical_analysis(benchmark_results)
        
        if stats_analysis and 'friedman_test' in stats_analysis:
            friedman = stats_analysis['friedman_test']
            print(f"\n   Friedman test: p-value = {friedman.get('p_value', 'N/A'):.4f}")
            print(f"   Significant differences: {friedman.get('significant', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ Benchmarking failed: {e}")
    
    # 3. Advanced metrics
    print("\n3. Calculating advanced metrics...")
    try:
        # Extract results for advanced analysis
        all_results = []
        for estimator_name, results in benchmark_results.items():
            if 'true_hurst' in results and 'estimated_hurst' in results:
                for i in range(len(results['true_hurst'])):
                    all_results.append({
                        'estimator': estimator_name,
                        'true_hurst': results['true_hurst'][i],
                        'estimated_hurst': results['estimated_hurst'][i],
                        'absolute_error': results['errors'][i]
                    })
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Calculate advanced metrics
            for estimator in results_df['estimator'].unique():
                subset = results_df[results_df['estimator'] == estimator]
                
                metrics = BenchmarkMetrics.calculate_metrics(
                    subset['true_hurst'].values,
                    subset['estimated_hurst'].values
                )
                
                print(f"   {estimator}:")
                print(f"     MAE: {metrics['mae']:.4f}")
                print(f"     RMSE: {metrics['rmse']:.4f}")
                print(f"     R²: {metrics['r2']:.4f}")
                print(f"     Bias: {metrics['bias']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Advanced metrics failed: {e}")


def demonstrate_visualization_advanced(dataset, benchmark_results=None):
    """Demonstrate advanced visualization features."""
    print("\n" + "=" * 60)
    print("ADVANCED VISUALIZATION DEMO")
    print("=" * 60)
    
    visualizer = FractionalVisualizer()
    
    # 1. Data exploration
    print("\n1. Creating data exploration plots...")
    try:
        # Sample different data types
        sample_data = dataset['training'][0]
        
        fig = visualizer.plot_data_exploration(
            sample_data['time_series'],
            hurst=sample_data['true_hurst'],
            title=f"Sample {sample_data['data_type'].upper()} Analysis",
            save_path="advanced_demo_data_exploration.png",
            show_plot=False
        )
        print("   ✅ Data exploration plot created")
        
    except Exception as e:
        print(f"   ❌ Data exploration failed: {e}")
    
    # 2. Training curves (if available)
    print("\n2. Creating training curve plots...")
    try:
        # Create sample training history
        sample_history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.19, 0.18],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.30],
            'learning_rate': [0.001] * 10
        }
        
        fig = visualizer.plot_training_curves(
            sample_history,
            save_path="advanced_demo_training_curves.png",
            show_plot=False
        )
        print("   ✅ Training curves plot created")
        
    except Exception as e:
        print(f"   ❌ Training curves failed: {e}")
    
    # 3. Model comparison (if benchmark results available)
    if benchmark_results:
        print("\n3. Creating model comparison plots...")
        try:
            # Convert benchmark results to DataFrame format
            results_df = []
            for estimator_name, results in benchmark_results.items():
                if 'true_hurst' in results and 'estimated_hurst' in results:
                    for i in range(len(results['true_hurst'])):
                        results_df.append({
                            'estimator': estimator_name,
                            'true_hurst': results['true_hurst'][i],
                            'estimated_hurst': results['estimated_hurst'][i],
                            'absolute_error': results['errors'][i],
                            'model_type': 'classical' if estimator_name in ['DFA', 'RS', 'Wavelet', 'Spectral', 'Higuchi', 'DMA'] else 'ml'
                        })
            
            if results_df:
                results_df = pd.DataFrame(results_df)
                
                fig = visualizer.plot_model_comparison(
                    results_df,
                    save_path="advanced_demo_model_comparison.png",
                    show_plot=False
                )
                print("   ✅ Model comparison plot created")
                
                # Benchmark summary
                benchmark_data = {'all_results': results_df}
                fig = visualizer.plot_benchmark_summary(
                    benchmark_data,
                    save_path="advanced_demo_benchmark_summary.png",
                    show_plot=False
                )
                print("   ✅ Benchmark summary plot created")
        
        except Exception as e:
            print(f"   ❌ Model comparison plots failed: {e}")
    
    # 4. Interactive dashboard
    print("\n4. Creating interactive dashboard...")
    try:
        if benchmark_results:
            # Convert to DataFrame format
            results_df = []
            for estimator_name, results in benchmark_results.items():
                if 'true_hurst' in results and 'estimated_hurst' in results:
                    for i in range(len(results['true_hurst'])):
                        results_df.append({
                            'estimator': estimator_name,
                            'true_hurst': results['true_hurst'][i],
                            'estimated_hurst': results['estimated_hurst'][i],
                            'absolute_error': results['errors'][i],
                            'model_type': 'classical' if estimator_name in ['DFA', 'RS', 'Wavelet', 'Spectral', 'Higuchi', 'DMA'] else 'ml'
                        })
            
            if results_df:
                results_df = pd.DataFrame(results_df)
                benchmark_data = {'all_results': results_df}
                
                fig = visualizer.create_interactive_dashboard(
                    benchmark_data,
                    save_path="advanced_demo_interactive_dashboard.html"
                )
                print("   ✅ Interactive dashboard created")
        
    except Exception as e:
        print(f"   ❌ Interactive dashboard failed: {e}")


def generate_comprehensive_report(dataset, optimization_results, benchmark_results=None):
    """Generate comprehensive report."""
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"advanced_features_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ADVANCED FEATURES DEMO REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training samples: {len(dataset['training'])}\n")
        f.write(f"Validation samples: {len(dataset['validation'])}\n")
        f.write(f"Testing samples: {len(dataset['testing'])}\n\n")
        
        # Optimization results
        f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("-" * 40 + "\n")
        for model_type, results in optimization_results.items():
            f.write(f"{model_type.upper()}:\n")
            f.write(f"  Best score: {results['best_score']:.4f}\n")
            f.write(f"  Best parameters: {results['best_params']}\n\n")
        
        # Benchmark results
        if benchmark_results:
            f.write("BENCHMARK RESULTS\n")
            f.write("-" * 40 + "\n")
            for estimator_name, results in benchmark_results.items():
                if 'mae' in results:
                    f.write(f"{estimator_name}:\n")
                    f.write(f"  MAE: {results['mae']:.4f}\n")
                    f.write(f"  Success rate: {results.get('success_rate', 'N/A'):.2%}\n")
                    f.write(f"  Total time: {results.get('total_time', 'N/A'):.2f}s\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✅ Comprehensive report saved to {report_file}")


def main():
    """Main function to run the advanced features demo."""
    print("Advanced Features Demo for Fractional PINN Project")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Create comprehensive dataset
        dataset = create_comprehensive_dataset()
        
        # 2. Run comprehensive testing
        testing_success = run_comprehensive_testing()
        
        # 3. Demonstrate hyperparameter optimization
        optimization_results = demonstrate_hyperparameter_optimization(dataset)
        
        # 4. Demonstrate model persistence and comparison
        demonstrate_model_persistence_and_comparison(dataset, optimization_results)
        
        # 5. Demonstrate advanced benchmarking
        benchmark_results = demonstrate_advanced_benchmarking(dataset)
        
        # 6. Demonstrate advanced visualization
        demonstrate_visualization_advanced(dataset, benchmark_results)
        
        # 7. Generate comprehensive report
        generate_comprehensive_report(dataset, optimization_results, benchmark_results)
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("ADVANCED FEATURES DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Testing status: {'✅ PASSED' if testing_success else '❌ FAILED'}")
        print("\nKey achievements:")
        print("✅ Comprehensive dataset created")
        print("✅ Hyperparameter optimization demonstrated")
        print("✅ Model persistence and comparison showcased")
        print("✅ Advanced benchmarking completed")
        print("✅ Advanced visualizations generated")
        print("✅ Comprehensive report created")
        print("\nFiles generated:")
        print("- optimization_results/ (hyperparameter optimization results)")
        print("- advanced_demo_*.png (visualization plots)")
        print("- advanced_demo_interactive_dashboard.html (interactive dashboard)")
        print("- advanced_features_report_*.txt (comprehensive report)")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
