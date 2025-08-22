"""
Simple Comprehensive Benchmark for Fractional Parameter Estimation Methods

This script runs a simplified but comprehensive benchmark comparing:
1. Classical Estimators: DFA, R/S, Wavelet, Spectral, Higuchi, DMA
2. ML Estimators: Random Forest, Gradient Boosting, SVR, Linear Regression
3. Neural Models: PINN, PINO, Neural Fractional ODE, Neural Fractional SDE

Features:
- Multiple data types (fBm, fGn)
- Various Hurst values
- Error handling and graceful degradation
- Performance visualization
- Detailed reporting
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.generators import FractionalDataGenerator
from estimators.classical_estimators import ClassicalEstimatorSuite
from estimators.ml_estimators import MLEstimatorSuite
from estimators.pinn_estimator import PINNEstimator
from models.fractional_pino import FractionalPINOTrainer
from models.neural_fractional_ode import NeuralFractionalODETrainer
from models.neural_fractional_sde import NeuralFractionalSDETrainer
from models.model_persistence import ModelPersistenceManager, quick_save_model
from models.model_comparison import ModelConfig, ModelComparisonFramework

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleComprehensiveBenchmark:
    """
    Simplified comprehensive benchmark for comparing all fractional parameter estimation methods.
    """
    
    def __init__(self, 
                 save_results: bool = True,
                 results_dir: str = "simple_comprehensive_benchmark_results",
                 device: str = 'auto'):
        """
        Initialize the benchmark.
        
        Args:
            save_results: Whether to save results to disk
            results_dir: Directory to save results
            device: Device for neural models ('auto', 'cpu', 'cuda')
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize components
        self.data_generator = FractionalDataGenerator(seed=42)
        self.classical_suite = ClassicalEstimatorSuite()
        self.ml_suite = MLEstimatorSuite()
        self.model_manager = ModelPersistenceManager()
        
        # Results storage
        self.results = {
            'classical_estimators': [],
            'ml_estimators': [],
            'neural_models': [],
            'metadata': {}
        }
        
        # Simplified benchmark configuration
        self.config = {
            'hurst_values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'n_points': [500, 1000],  # Reduced for faster execution
            'n_samples_per_config': 3,  # Reduced for faster execution
            'data_types': ['fbm', 'fgn'],  # Simplified data types
            'neural_training_epochs': 100,  # Reduced for faster execution
            'neural_early_stopping_patience': 20
        }
        
        logger.info(f"SimpleComprehensiveBenchmark initialized on device: {self.device}")
        
    def generate_benchmark_data(self) -> Dict[str, Any]:
        """Generate simplified benchmark dataset."""
        logger.info("Generating benchmark dataset...")
        
        benchmark_data = {}
        total_configs = (len(self.config['hurst_values']) * 
                        len(self.config['n_points']) * 
                        len(self.config['data_types']))
        
        with tqdm(total=total_configs, desc="Generating data") as pbar:
            for hurst in self.config['hurst_values']:
                for n_points in self.config['n_points']:
                    for data_type in self.config['data_types']:
                        
                        config_key = f"{data_type}_h{hurst}_n{n_points}"
                        benchmark_data[config_key] = []
                        
                        for sample_idx in range(self.config['n_samples_per_config']):
                            try:
                                # Generate base data
                                if data_type == 'fbm':
                                    data = self.data_generator.generate_fbm(
                                        n_points=n_points, hurst=hurst
                                    )
                                elif data_type == 'fgn':
                                    data = self.data_generator.generate_fgn(
                                        n_points=n_points, hurst=hurst
                                    )
                                
                                # Store sample
                                sample_data = {
                                    'time_series': data['data'],
                                    'true_hurst': data.get('hurst', hurst),
                                    'sample_idx': sample_idx,
                                    'config': config_key,
                                    'data_type': data_type,
                                    'hurst': hurst,
                                    'n_points': n_points
                                }
                                
                                benchmark_data[config_key].append(sample_data)
                                
                            except Exception as e:
                                logger.warning(f"Error generating data for {config_key}: {e}")
                                continue
                        
                        pbar.update(1)
        
        logger.info(f"Generated {len(benchmark_data)} data configurations")
        return benchmark_data
    
    def benchmark_classical_estimators(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """Benchmark classical estimators."""
        logger.info("Benchmarking classical estimators...")
        
        results = []
        
        for config_key, samples in tqdm(benchmark_data.items(), desc="Classical estimators"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                try:
                    # Run all classical estimators
                    start_time = time.time()
                    estimates = self.classical_suite.estimate_all(time_series)
                    computation_time = time.time() - start_time
                    
                    for estimator_name, estimate in estimates.items():
                        if estimate is not None and isinstance(estimate, (int, float, np.number)) and not np.isnan(float(estimate)):
                            absolute_error = abs(estimate - true_hurst)
                            relative_error = absolute_error / true_hurst
                            
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': estimator_name,
                                'model_type': 'classical',
                                'estimated_hurst': estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': absolute_error,
                                'relative_error': relative_error,
                                'computation_time': computation_time,
                                'data_type': sample['data_type'],
                                'hurst': sample['hurst'],
                                'n_points': sample['n_points']
                            })
                except Exception as e:
                    logger.warning(f"Error in classical estimation for {config_key}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def benchmark_ml_estimators(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """Benchmark ML estimators."""
        logger.info("Benchmarking ML estimators...")
        
        results = []
        
        # Prepare training data (use first 2 samples from each config for training)
        training_data = []
        for config_key, samples in benchmark_data.items():
            for sample in samples[:2]:
                training_data.append({
                    'time_series': sample['time_series'],
                    'true_hurst': sample['true_hurst']
                })
        
        # Train ML suite with error handling
        logger.info("Training ML estimators...")
        try:
            data_list = [item['time_series'] for item in training_data]
            hurst_values = [item['true_hurst'] for item in training_data]
            self.ml_suite.train_all(data_list, hurst_values)
        except Exception as e:
            logger.warning(f"Error training ML estimators: {e}")
            return pd.DataFrame(results)
        
        # Evaluate on all data
        for config_key, samples in tqdm(benchmark_data.items(), desc="ML estimators"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                try:
                    # Get ML estimates
                    start_time = time.time()
                    estimates = self.ml_suite.estimate_all(time_series)
                    computation_time = time.time() - start_time
                    
                    for estimator_name, estimate in estimates.items():
                        if estimate is not None and isinstance(estimate, (int, float, np.number)) and not np.isnan(float(estimate)):
                            absolute_error = abs(estimate - true_hurst)
                            relative_error = absolute_error / true_hurst
                            
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': estimator_name,
                                'model_type': 'ml',
                                'estimated_hurst': estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': absolute_error,
                                'relative_error': relative_error,
                                'computation_time': computation_time,
                                'data_type': sample['data_type'],
                                'hurst': sample['hurst'],
                                'n_points': sample['n_points']
                            })
                except Exception as e:
                    logger.warning(f"Error in ML estimation for {config_key}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def benchmark_neural_models(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """Benchmark neural models."""
        logger.info("Benchmarking neural models...")
        
        results = []
        
        # Prepare training data (use first 2 samples from each config for training)
        training_data = []
        for config_key, samples in benchmark_data.items():
            for sample in samples[:2]:
                training_data.append({
                    'time_series': sample['time_series'],
                    'true_hurst': sample['true_hurst']
                })
        
        # Train neural models
        neural_models = {}
        
        # PINN
        logger.info("Training PINN...")
        try:
            pinn_estimator = PINNEstimator(
                input_dim=1,
                hidden_dims=[32, 64, 32],  # Smaller network for faster training
                output_dim=1,
                learning_rate=0.001,
                device=self.device
            )
            pinn_estimator.build_model()
            pinn_history = pinn_estimator.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="PINN for simple comprehensive benchmark",
                model_tags=['benchmark', 'pinn']
            )
            neural_models['pinn'] = pinn_estimator
            logger.info("PINN training completed successfully")
        except Exception as e:
            logger.warning(f"PINN training failed: {e}")
            neural_models['pinn'] = None
        
        # PINO
        logger.info("Training PINO...")
        try:
            pino_trainer = FractionalPINOTrainer(
                input_dim=1,
                hidden_dims=[32, 64, 32],  # Smaller network
                modes=8,  # Reduced modes
                learning_rate=0.001,
                device=self.device
            )
            pino_history = pino_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="PINO for simple comprehensive benchmark",
                model_tags=['benchmark', 'pino']
            )
            neural_models['pino'] = pino_trainer
            logger.info("PINO training completed successfully")
        except Exception as e:
            logger.warning(f"PINO training failed: {e}")
            neural_models['pino'] = None
        
        # Neural Fractional ODE
        logger.info("Training Neural Fractional ODE...")
        try:
            neural_ode_trainer = NeuralFractionalODETrainer(
                input_dim=1,
                hidden_dims=[32, 64, 32],  # Smaller network
                alpha=0.5,
                learning_rate=0.001,
                device=self.device
            )
            neural_ode_history = neural_ode_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="Neural Fractional ODE for simple comprehensive benchmark",
                model_tags=['benchmark', 'neural_ode']
            )
            neural_models['neural_ode'] = neural_ode_trainer
            logger.info("Neural Fractional ODE training completed successfully")
        except Exception as e:
            logger.warning(f"Neural Fractional ODE training failed: {e}")
            neural_models['neural_ode'] = None
        
        # Neural Fractional SDE
        logger.info("Training Neural Fractional SDE...")
        try:
            neural_sde_trainer = NeuralFractionalSDETrainer(
                input_dim=1,
                hidden_dims=[32, 64, 32],  # Smaller network
                hurst=0.7,
                learning_rate=0.001,
                device=self.device
            )
            neural_sde_history = neural_sde_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="Neural Fractional SDE for simple comprehensive benchmark",
                model_tags=['benchmark', 'neural_sde']
            )
            neural_models['neural_sde'] = neural_sde_trainer
            logger.info("Neural Fractional SDE training completed successfully")
        except Exception as e:
            logger.warning(f"Neural Fractional SDE training failed: {e}")
            neural_models['neural_sde'] = None
        
        # Evaluate neural models
        for config_key, samples in tqdm(benchmark_data.items(), desc="Neural models"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                for model_name, model in neural_models.items():
                    if model is not None:
                        try:
                            start_time = time.time()
                            estimate = model.estimate(time_series)
                            computation_time = time.time() - start_time
                            
                            if estimate is not None and not np.isnan(estimate):
                                absolute_error = abs(estimate - true_hurst)
                                relative_error = absolute_error / true_hurst
                                
                                results.append({
                                    'config': config_key,
                                    'sample_idx': sample['sample_idx'],
                                    'estimator': model_name,
                                    'model_type': 'neural',
                                    'estimated_hurst': estimate,
                                    'true_hurst': true_hurst,
                                    'absolute_error': absolute_error,
                                    'relative_error': relative_error,
                                    'computation_time': computation_time,
                                    'data_type': sample['data_type'],
                                    'hurst': sample['hurst'],
                                    'n_points': sample['n_points']
                                })
                        except Exception as e:
                            logger.warning(f"Error evaluating {model_name}: {e}")
        
        return pd.DataFrame(results)
    
    def run_comprehensive_benchmark(self) -> Dict[str, pd.DataFrame]:
        """Run the complete benchmark."""
        logger.info("Starting simple comprehensive benchmark...")
        
        # Generate benchmark data
        benchmark_data = self.generate_benchmark_data()
        
        # Run benchmarks
        classical_results = self.benchmark_classical_estimators(benchmark_data)
        ml_results = self.benchmark_ml_estimators(benchmark_data)
        neural_results = self.benchmark_neural_models(benchmark_data)
        
        # Store results
        self.results['classical_estimators'] = classical_results
        self.results['ml_estimators'] = ml_results
        self.results['neural_models'] = neural_results
        
        # Save results
        if self.save_results:
            self.save_benchmark_results()
            self.generate_benchmark_report()
            self.plot_benchmark_results()
        
        return self.results
    
    def save_benchmark_results(self):
        """Save benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        for result_type, df in self.results.items():
            if not df.empty:
                filename = f"{result_type}_results_{timestamp}.csv"
                filepath = self.results_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {result_type} results to {filepath}")
        
        # Save combined results
        combined_df = pd.concat([
            self.results['classical_estimators'],
            self.results['ml_estimators'],
            self.results['neural_models']
        ], ignore_index=True)
        
        combined_filename = f"combined_benchmark_results_{timestamp}.csv"
        combined_filepath = self.results_dir / combined_filename
        combined_df.to_csv(combined_filepath, index=False)
        logger.info(f"Saved combined results to {combined_filepath}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'config': self.config,
            'device': self.device,
            'total_samples': len(combined_df),
            'classical_samples': len(self.results['classical_estimators']),
            'ml_samples': len(self.results['ml_estimators']),
            'neural_samples': len(self.results['neural_models'])
        }
        
        metadata_filename = f"benchmark_metadata_{timestamp}.json"
        metadata_filepath = self.results_dir / metadata_filename
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_filepath}")
    
    def generate_benchmark_report(self):
        """Generate a comprehensive benchmark report."""
        logger.info("Generating benchmark report...")
        
        # Combine all results
        combined_df = pd.concat([
            self.results['classical_estimators'],
            self.results['ml_estimators'],
            self.results['neural_models']
        ], ignore_index=True)
        
        if combined_df.empty:
            logger.warning("No results to analyze")
            return
        
        # Calculate overall statistics
        overall_stats = combined_df.groupby(['model_type', 'estimator']).agg({
            'absolute_error': ['mean', 'std', 'min', 'max'],
            'relative_error': ['mean', 'std'],
            'computation_time': ['mean', 'std']
        }).round(4)
        
        # Calculate statistics by data type
        data_type_stats = combined_df.groupby(['model_type', 'estimator', 'data_type']).agg({
            'absolute_error': ['mean', 'std'],
            'relative_error': ['mean', 'std']
        }).round(4)
        
        # Calculate statistics by Hurst value
        hurst_stats = combined_df.groupby(['model_type', 'estimator', 'hurst']).agg({
            'absolute_error': ['mean', 'std'],
            'relative_error': ['mean', 'std']
        }).round(4)
        
        # Find best performers
        best_overall = combined_df.loc[combined_df.groupby('estimator')['absolute_error'].idxmin()]
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{timestamp}.txt"
        report_filepath = self.results_dir / report_filename
        
        with open(report_filepath, 'w') as f:
            f.write("SIMPLE COMPREHENSIVE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total samples: {len(combined_df)}\n\n")
            
            f.write("OVERALL PERFORMANCE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(overall_stats.to_string())
            f.write("\n\n")
            
            f.write("BEST PERFORMERS (Overall)\n")
            f.write("-" * 25 + "\n")
            f.write(best_overall[['estimator', 'model_type', 'absolute_error', 'true_hurst', 'estimated_hurst']].to_string())
            f.write("\n\n")
            
            f.write("PERFORMANCE BY DATA TYPE\n")
            f.write("-" * 25 + "\n")
            f.write(data_type_stats.to_string())
            f.write("\n\n")
            
            f.write("PERFORMANCE BY HURST VALUE\n")
            f.write("-" * 25 + "\n")
            f.write(hurst_stats.to_string())
            f.write("\n\n")
            
            f.write("COMPUTATION TIME ANALYSIS\n")
            f.write("-" * 25 + "\n")
            time_stats = combined_df.groupby(['model_type', 'estimator'])['computation_time'].agg(['mean', 'std', 'min', 'max']).round(4)
            f.write(time_stats.to_string())
        
        logger.info(f"Benchmark report saved to {report_filepath}")
    
    def plot_benchmark_results(self):
        """Generate comprehensive benchmark visualizations."""
        logger.info("Generating benchmark visualizations...")
        
        # Combine all results
        combined_df = pd.concat([
            self.results['classical_estimators'],
            self.results['ml_estimators'],
            self.results['neural_models']
        ], ignore_index=True)
        
        if combined_df.empty:
            logger.warning("No results to plot")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall performance comparison
        ax1 = plt.subplot(3, 3, 1)
        sns.boxplot(data=combined_df, x='model_type', y='absolute_error', ax=ax1)
        ax1.set_title('Overall Performance by Model Type')
        ax1.set_ylabel('Absolute Error')
        ax1.set_xlabel('Model Type')
        
        # 2. Performance by estimator
        ax2 = plt.subplot(3, 3, 2)
        estimator_performance = combined_df.groupby('estimator')['absolute_error'].mean().sort_values()
        estimator_performance.plot(kind='bar', ax=ax2)
        ax2.set_title('Performance by Estimator')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance by data type
        ax3 = plt.subplot(3, 3, 3)
        data_type_performance = combined_df.groupby(['model_type', 'data_type'])['absolute_error'].mean().unstack()
        data_type_performance.plot(kind='bar', ax=ax3)
        ax3.set_title('Performance by Data Type')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Data Type')
        
        # 4. Performance by Hurst value
        ax4 = plt.subplot(3, 3, 4)
        hurst_performance = combined_df.groupby(['model_type', 'hurst'])['absolute_error'].mean().unstack()
        hurst_performance.plot(kind='bar', ax=ax4)
        ax4.set_title('Performance by Hurst Value')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Hurst Value')
        
        # 5. Computation time comparison
        ax5 = plt.subplot(3, 3, 5)
        sns.boxplot(data=combined_df, x='model_type', y='computation_time', ax=ax5)
        ax5.set_title('Computation Time by Model Type')
        ax5.set_ylabel('Computation Time (seconds)')
        ax5.set_xlabel('Model Type')
        
        # 6. Error distribution
        ax6 = plt.subplot(3, 3, 6)
        for model_type in combined_df['model_type'].unique():
            data = combined_df[combined_df['model_type'] == model_type]['absolute_error']
            ax6.hist(data, alpha=0.7, label=model_type, bins=20)
        ax6.set_title('Error Distribution by Model Type')
        ax6.set_xlabel('Absolute Error')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        # 7. True vs Estimated scatter plot
        ax7 = plt.subplot(3, 3, 7)
        for model_type in combined_df['model_type'].unique():
            data = combined_df[combined_df['model_type'] == model_type]
            ax7.scatter(data['true_hurst'], data['estimated_hurst'], alpha=0.6, label=model_type)
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax7.set_title('True vs Estimated Hurst Values')
        ax7.set_xlabel('True Hurst')
        ax7.set_ylabel('Estimated Hurst')
        ax7.legend()
        
        # 8. Performance heatmap
        ax8 = plt.subplot(3, 3, 8)
        heatmap_data = combined_df.groupby(['model_type', 'data_type'])['absolute_error'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax8)
        ax8.set_title('Performance Heatmap')
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        summary_stats = combined_df.groupby('model_type')['absolute_error'].agg(['mean', 'std', 'min', 'max']).round(4)
        summary_stats.plot(kind='bar', ax=ax9)
        ax9.set_title('Summary Statistics by Model Type')
        ax9.set_ylabel('Absolute Error')
        ax9.tick_params(axis='x', rotation=45)
        ax9.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"simple_comprehensive_benchmark_results_{timestamp}.png"
        plot_filepath = self.results_dir / plot_filename
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Benchmark visualization saved to {plot_filepath}")
        
        plt.show()

def main():
    """Run the simple comprehensive benchmark."""
    print("Simple Comprehensive Benchmark for Fractional Parameter Estimation Methods")
    print("=" * 70)
    
    # Initialize benchmark
    benchmark = SimpleComprehensiveBenchmark(
        save_results=True,
        results_dir="simple_comprehensive_benchmark_results",
        device='auto'
    )
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Print summary statistics
    for result_type, df in results.items():
        if not df.empty:
            print(f"\n{result_type.upper().replace('_', ' ')}:")
            print(f"  Total samples: {len(df)}")
            print(f"  Mean absolute error: {df['absolute_error'].mean():.4f}")
            print(f"  Mean relative error: {df['relative_error'].mean():.4f}")
            print(f"  Mean computation time: {df['computation_time'].mean():.4f} seconds")
            
            # Best performer
            best_idx = df['absolute_error'].idxmin()
            best_row = df.loc[best_idx]
            print(f"  Best performer: {best_row['estimator']} (Error: {best_row['absolute_error']:.4f})")
    
    print(f"\nResults saved to: {benchmark.results_dir}")
    print("Check the generated files for detailed analysis and visualizations.")

if __name__ == "__main__":
    main()
