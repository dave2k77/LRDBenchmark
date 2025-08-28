#!/usr/bin/env python3
"""
Final Comprehensive Benchmark for ML and NN Estimators

This script tests all Machine Learning and Neural Network estimators
using the pretrained models that were just created.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Any, Tuple
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import estimators
from lrdbench.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
from lrdbench.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbench.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbench.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
from lrdbench.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator

# Import data models
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MLNNFinalBenchmark:
    """
    Final comprehensive benchmark for ML and NN estimators.
    """
    
    def __init__(self):
        """Initialize the benchmark."""
        self.results = []
        self.estimators = {}
        self.test_data = {}
        
    def generate_test_data(self, n_samples: int = 100, sequence_length: int = 1000) -> Dict[str, List[Tuple[np.ndarray, float]]]:
        """
        Generate comprehensive test data.
        
        Parameters
        ----------
        n_samples : int
            Number of samples per data type
        sequence_length : int
            Length of each time series
            
        Returns
        -------
        Dict[str, List[Tuple[np.ndarray, float]]]
            Test data organized by type
        """
        print("🔧 Generating test data...")
        
        test_data = {
            'fgn': [],
            'fbm': [],
            'arfima': []
        }
        
        # Generate FGN data
        print("  📊 Generating FGN data...")
        for i in range(n_samples):
            h = np.random.uniform(0.1, 0.9)
            model = FractionalGaussianNoise(H=h)
            data = model.generate(sequence_length)
            test_data['fgn'].append((data, h))
            
        # Generate FBM data
        print("  📊 Generating FBM data...")
        for i in range(n_samples):
            h = np.random.uniform(0.1, 0.9)
            model = FractionalBrownianMotion(H=h)
            data = model.generate(sequence_length)
            test_data['fbm'].append((data, h))
            
        # Generate ARFIMA data
        print("  📊 Generating ARFIMA data...")
        for i in range(n_samples):
            h = np.random.uniform(0.1, 0.9)
            d = h - 0.5  # ARFIMA parameter
            model = ARFIMAModel(d=d, ar_params=[0.3], ma_params=[0.2])
            data = model.generate(sequence_length)
            test_data['arfima'].append((data, h))
            
        print(f"✅ Generated {n_samples} samples each for FGN, FBM, and ARFIMA")
        return test_data
    
    def initialize_estimators(self):
        """Initialize all ML and NN estimators."""
        print("🔧 Initializing estimators...")
        
        # Traditional ML estimators
        self.estimators['RandomForest'] = RandomForestEstimator(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.estimators['GradientBoosting'] = GradientBoostingEstimator(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.estimators['SVR'] = SVREstimator(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        )
        
        # Enhanced Neural Network estimators
        self.estimators['EnhancedCNN'] = EnhancedCNNEstimator(
            conv_channels=[32, 64, 128, 256],
            fc_layers=[512, 256, 128, 64],
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,  # Reduced for faster testing
            use_residual=True,
            use_attention=True,
            random_state=42
        )
        
        self.estimators['EnhancedLSTM'] = EnhancedLSTMEstimator(
            hidden_size=128,
            num_layers=3,
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,  # Reduced for faster testing
            bidirectional=True,
            use_attention=True,
            attention_heads=8,
            random_state=42
        )
        
        self.estimators['EnhancedGRU'] = EnhancedGRUEstimator(
            hidden_size=128,
            num_layers=3,
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,  # Reduced for faster testing
            bidirectional=True,
            use_attention=True,
            attention_heads=8,
            random_state=42
        )
        
        self.estimators['EnhancedTransformer'] = EnhancedTransformerEstimator(
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            learning_rate=0.0001,
            batch_size=16,
            epochs=50,  # Reduced for faster testing
            use_layer_norm=True,
            use_residual=True,
            random_state=42
        )
        
        print(f"✅ Initialized {len(self.estimators)} estimators")
    
    def test_estimator(self, estimator_name: str, estimator, data_list: List[Tuple[np.ndarray, float]], data_type: str) -> List[Dict[str, Any]]:
        """
        Test a single estimator on a dataset.
        
        Parameters
        ----------
        estimator_name : str
            Name of the estimator
        estimator : object
            The estimator instance
        data_list : List[Tuple[np.ndarray, float]]
            List of (data, true_h) tuples
        data_type : str
            Type of data (fgn, fbm, arfima)
            
        Returns
        -------
        List[Dict[str, Any]]
            Test results
        """
        results = []
        
        print(f"  🧪 Testing {estimator_name} on {data_type.upper()} data...")
        
        for i, (data, true_h) in enumerate(data_list):
            try:
                start_time = time.time()
                
                # Estimate Hurst parameter
                result = estimator.estimate(data)
                
                end_time = time.time()
                
                # Extract results
                estimated_h = result.get('hurst_parameter', 0.5)
                method = result.get('method', 'Unknown')
                confidence_interval = result.get('confidence_interval', (0.0, 1.0))
                
                # Calculate metrics
                error = abs(estimated_h - true_h)
                mse = (estimated_h - true_h) ** 2
                
                results.append({
                    'estimator': estimator_name,
                    'data_type': data_type,
                    'sample_id': i,
                    'true_h': true_h,
                    'estimated_h': estimated_h,
                    'error': error,
                    'mse': mse,
                    'method': method,
                    'confidence_interval_lower': confidence_interval[0],
                    'confidence_interval_upper': confidence_interval[1],
                    'execution_time': end_time - start_time,
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'estimator': estimator_name,
                    'data_type': data_type,
                    'sample_id': i,
                    'true_h': true_h,
                    'estimated_h': None,
                    'error': None,
                    'mse': None,
                    'method': 'failed',
                    'confidence_interval_lower': None,
                    'confidence_interval_upper': None,
                    'execution_time': None,
                    'status': f'error: {str(e)}'
                })
        
        return results
    
    def run_benchmark(self, n_samples: int = 50, sequence_length: int = 1000):
        """
        Run the comprehensive benchmark.
        
        Parameters
        ----------
        n_samples : int
            Number of test samples per data type
        sequence_length : int
            Length of each time series
        """
        print("🚀 Starting ML/NN Final Comprehensive Benchmark")
        print("=" * 60)
        
        # Generate test data
        self.test_data = self.generate_test_data(n_samples, sequence_length)
        
        # Initialize estimators
        self.initialize_estimators()
        
        # Run tests
        print("\n🧪 Running comprehensive tests...")
        
        for estimator_name, estimator in self.estimators.items():
            print(f"\n📊 Testing {estimator_name}...")
            
            for data_type, data_list in self.test_data.items():
                results = self.test_estimator(estimator_name, estimator, data_list, data_type)
                self.results.extend(results)
        
        # Analyze results
        self.analyze_results()
        
        # Save results
        self.save_results()
        
        print("\n✅ Benchmark completed!")
    
    def analyze_results(self):
        """Analyze and summarize the benchmark results."""
        print("\n📈 Analyzing results...")
        
        df = pd.DataFrame(self.results)
        
        # Filter successful results
        successful = df[df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("❌ No successful results to analyze!")
            return
        
        # Calculate summary statistics
        summary = []
        
        for estimator in successful['estimator'].unique():
            for data_type in successful['data_type'].unique():
                subset = successful[
                    (successful['estimator'] == estimator) & 
                    (successful['data_type'] == data_type)
                ]
                
                if len(subset) > 0:
                    summary.append({
                        'estimator': estimator,
                        'data_type': data_type,
                        'n_samples': len(subset),
                        'mean_error': subset['error'].mean(),
                        'std_error': subset['error'].std(),
                        'mean_mse': subset['mse'].mean(),
                        'std_mse': subset['mse'].std(),
                        'mean_execution_time': subset['execution_time'].mean(),
                        'success_rate': len(subset) / len(df[
                            (df['estimator'] == estimator) & 
                            (df['data_type'] == data_type)
                        ]) * 100
                    })
        
        summary_df = pd.DataFrame(summary)
        
        # Print summary
        print("\n📊 Summary Statistics:")
        print("=" * 80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Find best performers
        print("\n🏆 Best Performers by Data Type:")
        print("=" * 50)
        
        for data_type in summary_df['data_type'].unique():
            data_subset = summary_df[summary_df['data_type'] == data_type]
            best_mse = data_subset.loc[data_subset['mean_mse'].idxmin()]
            best_error = data_subset.loc[data_subset['mean_error'].idxmin()]
            
            print(f"\n{data_type.upper()}:")
            print(f"  Best MSE: {best_mse['estimator']} ({best_mse['mean_mse']:.4f})")
            print(f"  Best Error: {best_error['estimator']} ({best_error['mean_error']:.4f})")
        
        # Overall best performer
        overall_best = summary_df.loc[summary_df['mean_mse'].idxmin()]
        print(f"\n🎯 Overall Best Performer: {overall_best['estimator']} "
              f"(MSE: {overall_best['mean_mse']:.4f}, "
              f"Error: {overall_best['mean_error']:.4f})")
        
        self.summary_df = summary_df
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        results_file = f"ml_nn_final_benchmark_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Save summary
        if hasattr(self, 'summary_df'):
            summary_file = f"ml_nn_final_benchmark_summary_{timestamp}.csv"
            self.summary_df.to_csv(summary_file, index=False)
            print(f"💾 Summary results saved to: {summary_file}")
        
        # Generate report
        self.generate_report(timestamp)
    
    def generate_report(self, timestamp: str):
        """Generate a comprehensive markdown report."""
        report_file = f"ML_NN_FINAL_BENCHMARK_REPORT_{timestamp}.md"
        
        df = pd.DataFrame(self.results)
        successful = df[df['status'] == 'success'].copy()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🚀 ML/NN Final Comprehensive Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Overview\n\n")
            f.write(f"- **Total Tests:** {len(df)}\n")
            f.write(f"- **Successful Tests:** {len(successful)}\n")
            f.write(f"- **Success Rate:** {len(successful)/len(df)*100:.1f}%\n")
            f.write(f"- **Estimators Tested:** {len(df['estimator'].unique())}\n")
            f.write(f"- **Data Types:** {', '.join(df['data_type'].unique())}\n\n")
            
            f.write("## 🧪 Estimators Tested\n\n")
            for estimator in df['estimator'].unique():
                estimator_data = df[df['estimator'] == estimator]
                success_rate = len(estimator_data[estimator_data['status'] == 'success']) / len(estimator_data) * 100
                f.write(f"- **{estimator}**: {success_rate:.1f}% success rate\n")
            f.write("\n")
            
            if hasattr(self, 'summary_df'):
                f.write("## 📈 Performance Summary\n\n")
                f.write("| Estimator | Data Type | Samples | Mean Error | Mean MSE | Success Rate |\n")
                f.write("|-----------|-----------|---------|------------|----------|--------------|\n")
                
                for _, row in self.summary_df.iterrows():
                    f.write(f"| {row['estimator']} | {row['data_type']} | {row['n_samples']} | "
                           f"{row['mean_error']:.4f} | {row['mean_mse']:.4f} | {row['success_rate']:.1f}% |\n")
                f.write("\n")
            
            f.write("## 🏆 Best Performers\n\n")
            if hasattr(self, 'summary_df'):
                for data_type in self.summary_df['data_type'].unique():
                    data_subset = self.summary_df[self.summary_df['data_type'] == data_type]
                    best_mse = data_subset.loc[data_subset['mean_mse'].idxmin()]
                    f.write(f"**{data_type.upper()}**: {best_mse['estimator']} (MSE: {best_mse['mean_mse']:.4f})\n")
                f.write("\n")
            
            f.write("## 📋 Detailed Results\n\n")
            f.write("The complete detailed results are available in the CSV file.\n\n")
            
            f.write("## 🔧 Technical Details\n\n")
            f.write("- **Test Data**: Generated using FGN, FBM, and ARFIMA models\n")
            f.write("- **Hurst Range**: 0.1 to 0.9\n")
            f.write("- **Sequence Length**: 1000 points\n")
            f.write("- **Pretrained Models**: Used for all enhanced neural network estimators\n")
            f.write("- **Traditional ML**: Used scikit-learn implementations\n\n")
            
            f.write("## 📁 Files Generated\n\n")
            f.write(f"- `ml_nn_final_benchmark_results_{timestamp}.csv`: Detailed results\n")
            f.write(f"- `ml_nn_final_benchmark_summary_{timestamp}.csv`: Summary statistics\n")
            f.write(f"- `ML_NN_FINAL_BENCHMARK_REPORT_{timestamp}.md`: This report\n\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by the ML/NN Final Benchmark System*\n")
        
        print(f"📄 Report generated: {report_file}")


def main():
    """Main function to run the benchmark."""
    print("🎯 ML/NN Final Comprehensive Benchmark")
    print("Testing all Machine Learning and Neural Network estimators")
    print("using pretrained models...")
    print()
    
    # Create and run benchmark
    benchmark = MLNNFinalBenchmark()
    benchmark.run_benchmark(n_samples=30, sequence_length=1000)  # Reduced samples for faster execution
    
    print("\n🎉 Benchmark completed successfully!")
    print("Check the generated files for detailed results.")


if __name__ == "__main__":
    main()
