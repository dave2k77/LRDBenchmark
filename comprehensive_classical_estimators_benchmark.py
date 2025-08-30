#!/usr/bin/env python3
"""
Comprehensive Classical Estimators Benchmark.

This script benchmarks all 13 unified classical estimators across different optimization
frameworks (JAX, Numba, NumPy) and data types to demonstrate the unified system's capabilities.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all unified estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator

from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator

from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
from lrdbenchmark.analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator

from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
from lrdbenchmark.analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import MultifractalWaveletLeadersEstimator

# Import data models
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise


class ComprehensiveEstimatorBenchmark:
    """Comprehensive benchmark for all unified classical estimators."""
    
    def __init__(self):
        """Initialize the benchmark system."""
        self.estimators = {
            # Temporal Estimators
            'R/S': RSEstimator,
            'DFA': DFAEstimator,
            'Higuchi': HiguchiEstimator,
            'DMA': DMAEstimator,
            
            # Spectral Estimators
            'GPH': GPHEstimator,
            'Whittle': WhittleEstimator,
            'Periodogram': PeriodogramEstimator,
            
            # Wavelet Estimators
            'Wavelet Variance': WaveletVarianceEstimator,
            'Wavelet Whittle': WaveletWhittleEstimator,
            'CWT': CWTEstimator,
            'Log Variance': WaveletLogVarianceEstimator,
            
            # Multifractal Estimators
            'MFDFA': MFDFAEstimator,
            'Wavelet Leaders': MultifractalWaveletLeadersEstimator,
        }
        
        self.optimization_frameworks = ['numpy', 'numba', 'jax']
        self.data_sizes = [500, 1000, 2000]
        self.hurst_values = [0.3, 0.5, 0.7, 0.9]
        
        self.results = []
        
    def generate_test_data(self, hurst: float, size: int, data_type: str = 'fbm') -> np.ndarray:
        """Generate test data with specified Hurst parameter."""
        if data_type == 'fbm':
            model = FractionalBrownianMotion(H=hurst)
            return model.generate(size)
        elif data_type == 'fgn':
            model = FractionalGaussianNoise(H=hurst)
            return model.generate(size)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def benchmark_estimator(self, estimator_class, estimator_name: str, data: np.ndarray, 
                           hurst: float, size: int, framework: str) -> Dict[str, Any]:
        """Benchmark a single estimator."""
        try:
            # Create estimator instance
            start_time = time.time()
            estimator = estimator_class(use_optimization=framework)
            creation_time = time.time() - start_time
            
            # Get optimization info
            opt_info = estimator.get_optimization_info()
            
            # Run estimation
            start_time = time.time()
            result = estimator.estimate(data)
            estimation_time = time.time() - start_time
            
            # Extract results
            hurst_estimate = result.get('hurst_parameter', np.nan)
            method_used = result.get('method', 'unknown')
            actual_framework = result.get('optimization_framework', framework)
            
            # Calculate accuracy
            if not np.isnan(hurst_estimate):
                accuracy = abs(hurst_estimate - hurst)
                relative_error = accuracy / hurst
            else:
                accuracy = np.nan
                relative_error = np.nan
            
            return {
                'estimator': estimator_name,
                'framework': framework,
                'actual_framework': actual_framework,
                'data_size': size,
                'true_hurst': hurst,
                'estimated_hurst': hurst_estimate,
                'accuracy': accuracy,
                'relative_error': relative_error,
                'method_used': method_used,
                'creation_time': creation_time,
                'estimation_time': estimation_time,
                'total_time': creation_time + estimation_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'estimator': estimator_name,
                'framework': framework,
                'actual_framework': framework,
                'data_size': size,
                'true_hurst': hurst,
                'estimated_hurst': np.nan,
                'accuracy': np.nan,
                'relative_error': np.nan,
                'method_used': 'failed',
                'creation_time': 0,
                'estimation_time': 0,
                'total_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all estimators and configurations."""
        print("🚀 Starting Comprehensive Classical Estimators Benchmark...")
        print(f"📊 Testing {len(self.estimators)} estimators")
        print(f"⚡ Testing {len(self.optimization_frameworks)} optimization frameworks")
        print(f"📏 Testing {len(self.data_sizes)} data sizes")
        print(f"🎯 Testing {len(self.hurst_values)} Hurst values")
        print("=" * 80)
        
        total_tests = len(self.estimators) * len(self.optimization_frameworks) * len(self.data_sizes) * len(self.hurst_values)
        current_test = 0
        
        for estimator_name, estimator_class in self.estimators.items():
            print(f"\n🔍 Testing {estimator_name} Estimator...")
            
            for framework in self.optimization_frameworks:
                print(f"  ⚡ Framework: {framework}")
                
                for size in self.data_sizes:
                    for hurst in self.hurst_values:
                        current_test += 1
                        
                        # Generate test data
                        data = self.generate_test_data(hurst, size)
                        
                        # Run benchmark
                        result = self.benchmark_estimator(
                            estimator_class, estimator_name, data, hurst, size, framework
                        )
                        
                        self.results.append(result)
                        
                        # Progress indicator
                        progress = (current_test / total_tests) * 100
                        if result['success']:
                            print(f"    ✅ Size={size}, H={hurst}: H_est={result['estimated_hurst']:.3f}, "
                                  f"Time={result['total_time']:.3f}s, Progress={progress:.1f}%")
                        else:
                            print(f"    ❌ Size={size}, H={hurst}: Failed - {result['error']}, "
                                  f"Progress={progress:.1f}%")
        
        print("\n" + "=" * 80)
        print("🎉 Benchmark completed!")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark results."""
        print("\n📊 Analyzing Results...")
        
        # Overall statistics
        total_tests = len(df)
        successful_tests = len(df[df['success'] == True])
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"📈 Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        # Success rate by estimator
        print("\n🏆 Success Rate by Estimator:")
        estimator_success = df.groupby('estimator')['success'].agg(['count', 'sum'])
        estimator_success['success_rate'] = (estimator_success['sum'] / estimator_success['count']) * 100
        for estimator, row in estimator_success.iterrows():
            print(f"  {estimator}: {row['success_rate']:.1f}% ({row['sum']}/{row['count']})")
        
        # Success rate by framework
        print("\n⚡ Success Rate by Framework:")
        framework_success = df.groupby('framework')['success'].agg(['count', 'sum'])
        framework_success['success_rate'] = (framework_success['sum'] / framework_success['count']) * 100
        for framework, row in framework_success.iterrows():
            print(f"  {framework}: {row['success_rate']:.1f}% ({row['sum']}/{row['count']})")
        
        # Performance analysis
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            print(f"\n⏱️ Performance Analysis (Successful Tests):")
            print(f"  Average Creation Time: {successful_df['creation_time'].mean():.4f}s")
            print(f"  Average Estimation Time: {successful_df['estimation_time'].mean():.4f}s")
            print(f"  Average Total Time: {successful_df['total_time'].mean():.4f}s")
            
            # Accuracy analysis
            print(f"\n🎯 Accuracy Analysis:")
            print(f"  Average Absolute Error: {successful_df['accuracy'].mean():.4f}")
            print(f"  Average Relative Error: {successful_df['relative_error'].mean():.4f}")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'estimator_success': estimator_success,
            'framework_success': framework_success
        }
    
    def create_performance_plots(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive performance visualization plots."""
        print("\n📊 Creating Performance Plots...")
        
        # Filter successful tests
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("❌ No successful tests to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Classical Estimators Benchmark Results', fontsize=16)
        
        # Plot 1: Success Rate by Estimator
        ax1 = axes[0, 0]
        estimator_success = successful_df.groupby('estimator').size()
        estimator_success.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
        ax1.set_title('Successful Tests by Estimator')
        ax1.set_ylabel('Number of Successful Tests')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Success Rate by Framework
        ax2 = axes[0, 1]
        framework_success = successful_df.groupby('framework').size()
        framework_success.plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.7)
        ax2.set_title('Successful Tests by Framework')
        ax2.set_ylabel('Number of Successful Tests')
        
        # Plot 3: Performance by Estimator
        ax3 = axes[0, 2]
        estimator_performance = successful_df.groupby('estimator')['total_time'].mean()
        estimator_performance.plot(kind='bar', ax=ax3, color='orange', alpha=0.7)
        ax3.set_title('Average Execution Time by Estimator')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance by Framework
        ax4 = axes[1, 0]
        framework_performance = successful_df.groupby('framework')['total_time'].mean()
        framework_performance.plot(kind='bar', ax=ax4, color='red', alpha=0.7)
        ax4.set_title('Average Execution Time by Framework')
        ax4.set_ylabel('Time (seconds)')
        
        # Plot 5: Accuracy by Estimator
        ax5 = axes[1, 1]
        estimator_accuracy = successful_df.groupby('estimator')['accuracy'].mean()
        estimator_accuracy.plot(kind='bar', ax=ax5, color='purple', alpha=0.7)
        ax5.set_title('Average Accuracy by Estimator')
        ax5.set_ylabel('Absolute Error')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Performance vs Data Size
        ax6 = axes[1, 2]
        size_performance = successful_df.groupby('data_size')['total_time'].mean()
        size_performance.plot(kind='line', ax=ax6, color='brown', marker='o', linewidth=2, markersize=8)
        ax6.set_title('Performance vs Data Size')
        ax6.set_xlabel('Data Size')
        ax6.set_ylabel('Average Time (seconds)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📁 Plots saved to: {save_path}")
        
        plt.show()
    
    def create_heatmap_plots(self, df: pd.DataFrame, save_path: str = None):
        """Create heatmap visualizations for detailed analysis."""
        print("\n🔥 Creating Heatmap Plots...")
        
        # Filter successful tests
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("❌ No successful tests to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Benchmark Analysis Heatmaps', fontsize=16)
        
        # Heatmap 1: Execution Time by Estimator and Framework
        ax1 = axes[0, 0]
        time_pivot = successful_df.pivot_table(
            values='total_time', 
            index='estimator', 
            columns='framework', 
            aggfunc='mean'
        )
        sns.heatmap(time_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Execution Time (seconds)')
        ax1.set_xlabel('Framework')
        ax1.set_ylabel('Estimator')
        
        # Heatmap 2: Accuracy by Estimator and Framework
        ax2 = axes[0, 1]
        accuracy_pivot = successful_df.pivot_table(
            values='accuracy', 
            index='estimator', 
            columns='framework', 
            aggfunc='mean'
        )
        sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Absolute Error')
        ax2.set_xlabel('Framework')
        ax2.set_ylabel('Estimator')
        
        # Heatmap 3: Performance by Estimator and Data Size
        ax3 = axes[1, 0]
        size_time_pivot = successful_df.pivot_table(
            values='total_time', 
            index='estimator', 
            columns='data_size', 
            aggfunc='mean'
        )
        sns.heatmap(size_time_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax3)
        ax3.set_title('Execution Time vs Data Size')
        ax3.set_xlabel('Data Size')
        ax3.set_ylabel('Estimator')
        
        # Heatmap 4: Success Rate by Estimator and Framework
        ax4 = axes[1, 1]
        success_pivot = df.pivot_table(
            values='success', 
            index='estimator', 
            columns='framework', 
            aggfunc='mean'
        )
        sns.heatmap(success_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Success Rate')
        ax4.set_xlabel('Framework')
        ax4.set_ylabel('Estimator')
        
        plt.tight_layout()
        
        if save_path:
            heatmap_path = save_path.replace('.png', '_heatmaps.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            print(f"📁 Heatmap plots saved to: {heatmap_path}")
        
        plt.show()
    
    def save_results(self, df: pd.DataFrame, filename: str = "comprehensive_benchmark_results.csv"):
        """Save benchmark results to CSV."""
        df.to_csv(filename, index=False)
        print(f"📁 Results saved to: {filename}")
        
        # Also save summary statistics
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write("Comprehensive Classical Estimators Benchmark Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Tests: {len(df)}\n")
            f.write(f"Successful Tests: {len(df[df['success'] == True])}\n")
            f.write(f"Success Rate: {(len(df[df['success'] == True]) / len(df)) * 100:.1f}%\n\n")
            
            f.write("Success Rate by Estimator:\n")
            estimator_success = df.groupby('estimator')['success'].agg(['count', 'sum'])
            estimator_success['success_rate'] = (estimator_success['sum'] / estimator_success['count']) * 100
            for estimator, row in estimator_success.iterrows():
                f.write(f"  {estimator}: {row['success_rate']:.1f}% ({row['sum']}/{row['count']})\n")
            
            f.write("\nSuccess Rate by Framework:\n")
            framework_success = df.groupby('framework')['success'].agg(['count', 'sum'])
            framework_success['success_rate'] = (framework_success['sum'] / framework_success['count']) * 100
            for framework, row in framework_success.iterrows():
                f.write(f"  {framework}: {row['success_rate']:.1f}% ({row['sum']}/{row['count']})\n")
        
        print(f"📁 Summary saved to: {summary_filename}")


def main():
    """Run the comprehensive benchmark."""
    print("🚀 LRDBenchmark - Comprehensive Classical Estimators Benchmark")
    print("=" * 80)
    
    # Create benchmark instance
    benchmark = ComprehensiveEstimatorBenchmark()
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    analysis = benchmark.analyze_results(results_df)
    
    # Create visualizations
    benchmark.create_performance_plots(results_df, "comprehensive_benchmark_performance.png")
    benchmark.create_heatmap_plots(results_df, "comprehensive_benchmark_heatmaps.png")
    
    # Save results
    benchmark.save_results(results_df)
    
    print("\n🎉 Benchmark completed successfully!")
    print("📊 Check the generated plots and CSV files for detailed results.")
    
    return results_df, analysis


if __name__ == "__main__":
    results, analysis = main()
