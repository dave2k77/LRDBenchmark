"""
Performance Benchmark for Fractional Neural Models

This script provides comprehensive benchmarking of:
1. Neural Models: PINN, PINO, Neural Fractional ODE, Neural Fractional SDE
2. Classical Estimators: DFA, R/S, Wavelet, Spectral, Higuchi, DMA
3. ML Estimators: Random Forest, Gradient Boosting, SVR, etc.

Features:
- Multiple data types (fBm, fGn, ARFIMA, MRW)
- Various contamination scenarios
- Statistical significance testing
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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

class PerformanceBenchmark:
    """
    Comprehensive benchmark for comparing fractional parameter estimation methods.
    """
    
    def __init__(self, 
                 save_results: bool = True,
                 results_dir: str = "benchmark_results",
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
            'neural_models': [],
            'classical_estimators': [],
            'ml_estimators': [],
            'metadata': {}
        }
        
        # Benchmark configuration
        self.config = {
            'hurst_values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'n_points': [100, 500, 1000, 2000],
            'n_samples_per_config': 10,
            'contamination_types': [
                'none', 'noise', 'outliers', 'trends', 
                'seasonality', 'missing_data', 'heteroscedasticity'
            ],
            'data_types': ['fbm', 'fgn', 'arfima', 'mrw'],
            'neural_training_epochs': 500,
            'neural_early_stopping_patience': 50
        }
        
        logger.info(f"PerformanceBenchmark initialized on device: {self.device}")
        
    def generate_benchmark_data(self) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark dataset.
        
        Returns:
            Dictionary containing all benchmark data
        """
        logger.info("Generating benchmark dataset...")
        
        benchmark_data = {}
        total_configs = (len(self.config['hurst_values']) * 
                        len(self.config['n_points']) * 
                        len(self.config['data_types']) * 
                        len(self.config['contamination_types']))
        
        with tqdm(total=total_configs, desc="Generating data") as pbar:
            for hurst in self.config['hurst_values']:
                for n_points in self.config['n_points']:
                    for data_type in self.config['data_types']:
                        for contamination in self.config['contamination_types']:
                            
                            config_key = f"{data_type}_h{hurst}_n{n_points}_{contamination}"
                            benchmark_data[config_key] = []
                            
                            for sample_idx in range(self.config['n_samples_per_config']):
                                # Generate base data
                                if data_type == 'fbm':
                                    data = self.data_generator.generate_fbm(
                                        n_points=n_points, hurst=hurst
                                    )
                                elif data_type == 'fgn':
                                    data = self.data_generator.generate_fgn(
                                        n_points=n_points, hurst=hurst
                                    )
                                elif data_type == 'arfima':
                                    data = self.data_generator.generate_arfima(
                                        n_points=n_points, d=hurst-0.5
                                    )
                                elif data_type == 'mrw':
                                    data = self.data_generator.generate_mrw(
                                        n_points=n_points, hurst=hurst
                                    )
                                
                                # Apply contamination if specified
                                if contamination != 'none':
                                    data = self.data_generator.apply_contamination(
                                        data, contamination_type=contamination
                                    )
                                
                                # Store sample
                                sample_data = {
                                    'time_series': data['time_series'],
                                    'true_hurst': data.get('hurst', hurst),
                                    'true_d': data.get('d', hurst-0.5),
                                    'sample_idx': sample_idx,
                                    'config': config_key
                                }
                                
                                benchmark_data[config_key].append(sample_data)
                            
                            pbar.update(1)
        
        logger.info(f"Generated {len(benchmark_data)} data configurations")
        return benchmark_data
    
    def benchmark_classical_estimators(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Benchmark classical estimators.
        
        Args:
            benchmark_data: Generated benchmark data
            
        Returns:
            DataFrame with classical estimator results
        """
        logger.info("Benchmarking classical estimators...")
        
        results = []
        
        for config_key, samples in tqdm(benchmark_data.items(), desc="Classical estimators"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                # Run all classical estimators
                estimates = self.classical_suite.estimate_all(time_series)
                
                for estimator_name, estimate in estimates.items():
                    if estimate is not None and not np.isnan(estimate):
                        results.append({
                            'config': config_key,
                            'sample_idx': sample['sample_idx'],
                            'estimator': estimator_name,
                            'estimated_hurst': estimate,
                            'true_hurst': true_hurst,
                            'absolute_error': abs(estimate - true_hurst),
                            'relative_error': abs(estimate - true_hurst) / true_hurst,
                            'model_type': 'classical'
                        })
        
        return pd.DataFrame(results)
    
    def benchmark_ml_estimators(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Benchmark ML estimators.
        
        Args:
            benchmark_data: Generated benchmark data
            
        Returns:
            DataFrame with ML estimator results
        """
        logger.info("Benchmarking ML estimators...")
        
        results = []
        
        # Train ML models on a subset of data
        training_data = []
        for config_key, samples in benchmark_data.items():
            for sample in samples[:5]:  # Use first 5 samples for training
                training_data.append({
                    'time_series': sample['time_series'],
                    'true_hurst': sample['true_hurst']
                })
        
        # Train ML suite
        logger.info("Training ML estimators...")
        self.ml_suite.train_all(training_data)
        
        # Evaluate on all data
        for config_key, samples in tqdm(benchmark_data.items(), desc="ML estimators"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                # Get ML estimates
                estimates = self.ml_suite.estimate_all(time_series)
                
                for estimator_name, estimate in estimates.items():
                    if estimate is not None and not np.isnan(estimate):
                        results.append({
                            'config': config_key,
                            'sample_idx': sample['sample_idx'],
                            'estimator': estimator_name,
                            'estimated_hurst': estimate,
                            'true_hurst': true_hurst,
                            'absolute_error': abs(estimate - true_hurst),
                            'relative_error': abs(estimate - true_hurst) / true_hurst,
                            'model_type': 'ml'
                        })
        
        return pd.DataFrame(results)
    
    def benchmark_neural_models(self, benchmark_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Benchmark neural models (PINN, PINO, Neural ODE, Neural SDE).
        
        Args:
            benchmark_data: Generated benchmark data
            
        Returns:
            DataFrame with neural model results
        """
        logger.info("Benchmarking neural models...")
        
        results = []
        
        # Train neural models on a subset of data
        training_data = []
        for config_key, samples in benchmark_data.items():
            for sample in samples[:3]:  # Use first 3 samples for training
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
                hidden_dims=[64, 128, 128, 64],
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
                model_description="PINN for performance benchmark",
                model_tags=['benchmark', 'pinn']
            )
            neural_models['pinn'] = pinn_estimator
        except Exception as e:
            logger.warning(f"PINN training failed: {e}")
            neural_models['pinn'] = None
        
        # PINO
        logger.info("Training PINO...")
        try:
            pino_trainer = FractionalPINOTrainer(
                input_dim=1,
                hidden_dims=[64, 128, 128, 64],
                modes=16,
                learning_rate=0.001,
                device=self.device
            )
            pino_history = pino_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="PINO for performance benchmark",
                model_tags=['benchmark', 'pino']
            )
            neural_models['pino'] = pino_trainer
        except Exception as e:
            logger.warning(f"PINO training failed: {e}")
            neural_models['pino'] = None
        
        # Neural ODE
        logger.info("Training Neural ODE...")
        try:
            ode_trainer = NeuralFractionalODETrainer(
                input_dim=1,
                hidden_dims=[64, 128, 64],
                alpha=0.5,
                learning_rate=0.001,
                device=self.device
            )
            ode_history = ode_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="Neural ODE for performance benchmark",
                model_tags=['benchmark', 'neural_ode']
            )
            neural_models['neural_ode'] = ode_trainer
        except Exception as e:
            logger.warning(f"Neural ODE training failed: {e}")
            neural_models['neural_ode'] = None
        
        # Neural SDE
        logger.info("Training Neural SDE...")
        try:
            sde_trainer = NeuralFractionalSDETrainer(
                input_dim=1,
                hidden_dims=[64, 128, 64],
                hurst=0.7,
                learning_rate=0.001,
                device=self.device
            )
            sde_history = sde_trainer.train(
                training_data,
                epochs=self.config['neural_training_epochs'],
                early_stopping_patience=self.config['neural_early_stopping_patience'],
                save_model=True,
                model_description="Neural SDE for performance benchmark",
                model_tags=['benchmark', 'neural_sde']
            )
            neural_models['neural_sde'] = sde_trainer
        except Exception as e:
            logger.warning(f"Neural SDE training failed: {e}")
            neural_models['neural_sde'] = None
        
        # Evaluate neural models on all data
        for config_key, samples in tqdm(benchmark_data.items(), desc="Neural models"):
            for sample in samples:
                time_series = sample['time_series']
                true_hurst = sample['true_hurst']
                
                # PINN
                if neural_models['pinn'] is not None:
                    try:
                        pinn_estimate = neural_models['pinn'].estimate(time_series)
                        if pinn_estimate is not None and not np.isnan(pinn_estimate):
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': 'PINN',
                                'estimated_hurst': pinn_estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': abs(pinn_estimate - true_hurst),
                                'relative_error': abs(pinn_estimate - true_hurst) / true_hurst,
                                'model_type': 'neural'
                            })
                    except Exception as e:
                        logger.warning(f"PINN estimation failed: {e}")
                
                # PINO
                if neural_models['pino'] is not None:
                    try:
                        pino_estimate = neural_models['pino'].estimate(time_series)
                        if pino_estimate is not None and not np.isnan(pino_estimate):
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': 'PINO',
                                'estimated_hurst': pino_estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': abs(pino_estimate - true_hurst),
                                'relative_error': abs(pino_estimate - true_hurst) / true_hurst,
                                'model_type': 'neural'
                            })
                    except Exception as e:
                        logger.warning(f"PINO estimation failed: {e}")
                
                # Neural ODE
                if neural_models['neural_ode'] is not None:
                    try:
                        ode_estimate = neural_models['neural_ode'].estimate(time_series)
                        if ode_estimate is not None and not np.isnan(ode_estimate):
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': 'Neural_ODE',
                                'estimated_hurst': ode_estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': abs(ode_estimate - true_hurst),
                                'relative_error': abs(ode_estimate - true_hurst) / true_hurst,
                                'model_type': 'neural'
                            })
                    except Exception as e:
                        logger.warning(f"Neural ODE estimation failed: {e}")
                
                # Neural SDE
                if neural_models['neural_sde'] is not None:
                    try:
                        sde_estimate = neural_models['neural_sde'].estimate(time_series)
                        if sde_estimate is not None and not np.isnan(sde_estimate):
                            results.append({
                                'config': config_key,
                                'sample_idx': sample['sample_idx'],
                                'estimator': 'Neural_SDE',
                                'estimated_hurst': sde_estimate,
                                'true_hurst': true_hurst,
                                'absolute_error': abs(sde_estimate - true_hurst),
                                'relative_error': abs(sde_estimate - true_hurst) / true_hurst,
                                'model_type': 'neural'
                            })
                    except Exception as e:
                        logger.warning(f"Neural SDE estimation failed: {e}")
        
        return pd.DataFrame(results)
    
    def run_comprehensive_benchmark(self) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive benchmark across all methods.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting comprehensive performance benchmark...")
        start_time = time.time()
        
        # Generate benchmark data
        benchmark_data = self.generate_benchmark_data()
        
        # Run benchmarks
        classical_results = self.benchmark_classical_estimators(benchmark_data)
        ml_results = self.benchmark_ml_estimators(benchmark_data)
        neural_results = self.benchmark_neural_models(benchmark_data)
        
        # Combine results
        all_results = pd.concat([classical_results, ml_results, neural_results], ignore_index=True)
        
        # Calculate additional metrics
        all_results['squared_error'] = (all_results['estimated_hurst'] - all_results['true_hurst'])**2
        
        # Store results
        self.results = {
            'all_results': all_results,
            'classical_results': classical_results,
            'ml_results': ml_results,
            'neural_results': neural_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'config': self.config,
                'total_samples': len(all_results),
                'benchmark_duration': time.time() - start_time
            }
        }
        
        # Save results if requested
        if self.save_results:
            self.save_benchmark_results()
        
        logger.info(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
        return self.results
    
    def save_benchmark_results(self):
        """Save benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = self.results_dir / f"benchmark_results_{timestamp}.csv"
        self.results['all_results'].to_csv(results_file, index=False)
        
        # Save metadata
        metadata_file = self.results_dir / f"benchmark_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.results['metadata'], f, indent=2, default=str)
        
        # Save detailed results by model type
        for model_type in ['classical', 'ml', 'neural']:
            if f'{model_type}_results' in self.results:
                detailed_file = self.results_dir / f"{model_type}_results_{timestamp}.csv"
                self.results[f'{model_type}_results'].to_csv(detailed_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive performance summary.
        
        Returns:
            DataFrame with performance summary
        """
        if 'all_results' not in self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")
        
        df = self.results['all_results']
        
        # Group by estimator and calculate metrics
        summary = df.groupby(['estimator', 'model_type']).agg({
            'absolute_error': ['mean', 'std', 'median'],
            'relative_error': ['mean', 'std', 'median'],
            'squared_error': ['mean', 'std'],
            'estimated_hurst': ['count']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        # Add RMSE
        summary['rmse'] = np.sqrt(summary['squared_error_mean'])
        
        # Sort by absolute error mean
        summary = summary.sort_values('absolute_error_mean')
        
        return summary
    
    def create_performance_visualizations(self, save_plots: bool = True):
        """
        Create comprehensive performance visualizations.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if 'all_results' not in self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")
        
        df = self.results['all_results']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fractional Parameter Estimation Performance Comparison', fontsize=16)
        
        # 1. Overall performance comparison
        ax1 = axes[0, 0]
        performance_summary = df.groupby('estimator')['absolute_error'].mean().sort_values()
        performance_summary.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Mean Absolute Error by Estimator')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance by model type
        ax2 = axes[0, 1]
        model_performance = df.groupby('model_type')['absolute_error'].mean()
        model_performance.plot(kind='bar', ax=ax2, color=['red', 'blue', 'green'])
        ax2.set_title('Performance by Model Type')
        ax2.set_ylabel('Mean Absolute Error')
        
        # 3. Error distribution
        ax3 = axes[0, 2]
        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type]['absolute_error']
            ax3.hist(subset, alpha=0.6, label=model_type, bins=20)
        ax3.set_title('Error Distribution by Model Type')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. True vs Estimated scatter plot
        ax4 = axes[1, 0]
        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type]
            ax4.scatter(subset['true_hurst'], subset['estimated_hurst'], 
                       alpha=0.6, label=model_type, s=20)
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_title('True vs Estimated Hurst Exponent')
        ax4.set_xlabel('True Hurst')
        ax4.set_ylabel('Estimated Hurst')
        ax4.legend()
        
        # 5. Performance by data type
        ax5 = axes[1, 1]
        df['data_type'] = df['config'].str.split('_').str[0]
        data_performance = df.groupby('data_type')['absolute_error'].mean()
        data_performance.plot(kind='bar', ax=ax5, color='orange')
        ax5.set_title('Performance by Data Type')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Performance by contamination
        ax6 = axes[1, 2]
        df['contamination'] = df['config'].str.split('_').str[-1]
        contam_performance = df.groupby('contamination')['absolute_error'].mean()
        contam_performance.plot(kind='bar', ax=ax6, color='purple')
        ax6.set_title('Performance by Contamination Type')
        ax6.set_ylabel('Mean Absolute Error')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.results_dir / f"performance_comparison_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {plot_file}")
        
        plt.show()
    
    def generate_detailed_report(self) -> str:
        """
        Generate detailed benchmark report.
        
        Returns:
            String containing the detailed report
        """
        if 'all_results' not in self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")
        
        df = self.results['all_results']
        metadata = self.results['metadata']
        
        report = []
        report.append("=" * 80)
        report.append("FRACTIONAL PARAMETER ESTIMATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {metadata['timestamp']}")
        report.append(f"Device: {metadata['device']}")
        report.append(f"Total samples: {metadata['total_samples']}")
        report.append(f"Benchmark duration: {metadata['benchmark_duration']:.2f} seconds")
        report.append("")
        
        # Overall performance summary
        report.append("OVERALL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        summary = self.generate_performance_summary()
        report.append(summary.to_string())
        report.append("")
        
        # Best performers
        report.append("TOP 5 BEST PERFORMING ESTIMATORS")
        report.append("-" * 40)
        top_5 = summary.head(5)
        for idx, (estimator, row) in enumerate(top_5.iterrows(), 1):
            report.append(f"{idx}. {estimator[0]} ({estimator[1]})")
            report.append(f"   MAE: {row['absolute_error_mean']:.4f} ± {row['absolute_error_std']:.4f}")
            report.append(f"   RMSE: {row['rmse']:.4f}")
            report.append("")
        
        # Performance by model type
        report.append("PERFORMANCE BY MODEL TYPE")
        report.append("-" * 40)
        model_summary = df.groupby('model_type').agg({
            'absolute_error': ['mean', 'std'],
            'relative_error': ['mean', 'std']
        }).round(4)
        report.append(model_summary.to_string())
        report.append("")
        
        # Performance by data type
        report.append("PERFORMANCE BY DATA TYPE")
        report.append("-" * 40)
        df['data_type'] = df['config'].str.split('_').str[0]
        data_summary = df.groupby('data_type')['absolute_error'].agg(['mean', 'std']).round(4)
        report.append(data_summary.to_string())
        report.append("")
        
        # Performance by contamination
        report.append("PERFORMANCE BY CONTAMINATION TYPE")
        report.append("-" * 40)
        df['contamination'] = df['config'].str.split('_').str[-1]
        contam_summary = df.groupby('contamination')['absolute_error'].agg(['mean', 'std']).round(4)
        report.append(contam_summary.to_string())
        report.append("")
        
        # Statistical significance testing
        report.append("STATISTICAL SIGNIFICANCE TESTING")
        report.append("-" * 40)
        neural_errors = df[df['model_type'] == 'neural']['absolute_error']
        classical_errors = df[df['model_type'] == 'classical']['absolute_error']
        ml_errors = df[df['model_type'] == 'ml']['absolute_error']
        
        # Neural vs Classical
        t_stat, p_value = stats.ttest_ind(neural_errors, classical_errors)
        report.append(f"Neural vs Classical: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Neural vs ML
        t_stat, p_value = stats.ttest_ind(neural_errors, ml_errors)
        report.append(f"Neural vs ML: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Classical vs ML
        t_stat, p_value = stats.ttest_ind(classical_errors, ml_errors)
        report.append(f"Classical vs ML: t={t_stat:.4f}, p={p_value:.4f}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the performance benchmark."""
    print("Fractional Parameter Estimation Performance Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(
        save_results=True,
        results_dir="benchmark_results",
        device='auto'
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate summary
    summary = benchmark.generate_performance_summary()
    print("\nPerformance Summary:")
    print(summary)
    
    # Create visualizations
    benchmark.create_performance_visualizations(save_plots=True)
    
    # Generate detailed report
    report = benchmark.generate_detailed_report()
    print("\n" + report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = benchmark.results_dir / f"benchmark_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nBenchmark completed! Results saved to {benchmark.results_dir}")


if __name__ == "__main__":
    main()
