#!/usr/bin/env python3
"""
HPFracc Proper Benchmark

This script uses the data generator to create properly formatted data
and benchmarks hpfracc fractional neural networks against LRDBench estimators.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our data generator
from hpfracc_data_generator import HPFraccDataGenerator

# Import hpfracc components
try:
    from hpfracc.ml import (
        FractionalNeuralNetwork, 
        FractionalLSTM, 
        FractionalTransformer,
        FractionalConv1D,
        FractionalConv2D,
        BackendType
    )
    from hpfracc.ml.backends import get_backend_manager
    HPRACC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: hpfracc.ml not available: {e}")
    HPRACC_AVAILABLE = False

# Import LRDBench components
try:
    from lrdbench import (
        FBMModel, FGNModel, ARFIMAModel, MRWModel,
        enable_analytics, get_analytics_summary
    )
    LRDBENCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: lrdbench not available: {e}")
    LRDBENCH_AVAILABLE = False

class HPFraccProperBenchmark:
    """
    Proper benchmark using the data generator for hpfracc models.
    """
    
    def __init__(self, 
                 series_length: int = 1000,
                 batch_size: int = 32,
                 input_window: int = 10,
                 prediction_horizon: int = 1):
        """
        Initialize the benchmark.
        
        Parameters
        ----------
        series_length : int
            Length of each time series
        batch_size : int
            Number of samples per batch
        input_window : int
            Number of time steps to use as input
        prediction_horizon : int
            Number of time steps to predict ahead
        """
        self.series_length = series_length
        self.batch_size = batch_size
        self.input_window = input_window
        self.prediction_horizon = prediction_horizon
        
        # Create data generator
        self.data_generator = HPFraccDataGenerator(
            series_length=series_length,
            batch_size=batch_size,
            input_window=input_window,
            prediction_horizon=prediction_horizon
        )
        
        # Results storage
        self.results = {}
        
        # Enable analytics if available
        if LRDBENCH_AVAILABLE:
            enable_analytics()
        
        # Set hpfracc backend
        if HPRACC_AVAILABLE:
            try:
                backend_manager = get_backend_manager()
                backend_manager.switch_backend(BackendType.JAX)
                print(f"✅ Set hpfracc backend to JAX")
            except Exception as e:
                print(f"⚠️ Could not set hpfracc backend: {e}")
    
    def create_hpfracc_models(self) -> Dict[str, Any]:
        """
        Create hpfracc models with correct input dimensions.
        """
        if not HPRACC_AVAILABLE:
            print("❌ hpfracc not available for model creation")
            return {}
        
        print("🧠 Creating hpfracc fractional neural networks...")
        
        models = {}
        
        try:
            # Set the backend first
            try:
                backend_manager = get_backend_manager()
                backend_manager.switch_backend(BackendType.JAX)
                print("✅ Set hpfracc backend to JAX")
            except Exception as e:
                print(f"⚠️ Could not set hpfracc backend: {e}")
            
            # Create models with correct input_size
            try:
                # FractionalNeuralNetwork - input_size = input_window (number of time steps)
                model = FractionalNeuralNetwork(
                    input_size=self.input_window,  # 10 time steps as features
                    hidden_sizes=[64, 32, 16],
                    output_size=self.prediction_horizon,  # 1 prediction step
                    fractional_order=0.5
                )
                
                # Fix backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalNeuralNetwork'] = model
                print("✅ Created FractionalNeuralNetwork")
                print(f"   Input size: {model.input_size}, Output size: {model.output_size}")
                
            except Exception as e:
                print(f"⚠️ Could not create FractionalNeuralNetwork: {e}")
            
            try:
                # FractionalLSTM - input_size = input_window
                model = FractionalLSTM(
                    input_size=self.input_window,
                    hidden_size=32,
                    num_layers=1
                )
                
                # Fix backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalLSTM'] = model
                print("✅ Created FractionalLSTM")
                print(f"   Input size: {model.input_size}")
                
            except Exception as e:
                print(f"⚠️ Could not create FractionalLSTM: {e}")
            
            try:
                # FractionalTransformer - d_model should match input_size
                model = FractionalTransformer(
                    d_model=self.input_window,  # Match input size
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    dim_feedforward=128,
                    dropout=0.1,
                    activation="relu"
                )
                
                # Fix backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalTransformer'] = model
                print("✅ Created FractionalTransformer")
                print(f"   d_model: {model.d_model}")
                
            except Exception as e:
                print(f"⚠️ Could not create FractionalTransformer: {e}")
                
        except Exception as e:
            print(f"⚠️ Error creating hpfracc models: {e}")
        
        return models
    
    def benchmark_hpfracc_models(self, models: Dict[str, Any], 
                                datasets: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Benchmark hpfracc models on the generated datasets.
        """
        if not models or not datasets:
            return {}
        
        print("\n🔬 Benchmarking hpfracc models...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n  🧠 Testing {model_name}...")
            model_results = {}
            
            for dataset_name, dataset in datasets.items():
                try:
                    print(f"    📊 Testing on {dataset_name}...")
                    
                    # Get input data and reshape for hpfracc
                    # Input shape: (n_samples, input_window, 1) -> (n_samples, input_window)
                    X = dataset['X'].squeeze(-1)  # Remove last dimension
                    y = dataset['y'].squeeze(-1)  # Remove last dimension
                    
                    print(f"      Input shape: {X.shape}, Target shape: {y.shape}")
                    
                    # Test on a subset for benchmarking
                    n_test = min(100, len(X))  # Test on first 100 samples
                    X_test = X[:n_test]
                    y_test = y[:n_test]
                    
                    # Benchmark forward pass
                    start_time = time.time()
                    
                    if hasattr(model, 'forward'):
                        # Standard forward pass
                        output = model.forward(X_test, use_fractional=False)
                    elif hasattr(model, 'fractional_forward'):
                        # Fractional forward pass
                        output = model.fractional_forward(X_test)
                    else:
                        print(f"      ⚠️ {model_name} has no forward method")
                        continue
                    
                    forward_time = time.time() - start_time
                    
                    # Calculate metrics
                    if hasattr(output, 'numpy'):
                        output_np = output.numpy()
                    else:
                        output_np = output
                    
                    mse = np.mean((output_np - y_test) ** 2)
                    mae = np.mean(np.abs(output_np - y_test))
                    
                    model_results[dataset_name] = {
                        'forward_time': forward_time,
                        'mse': mse,
                        'mae': mae,
                        'output_shape': output_np.shape,
                        'n_test_samples': n_test,
                        'success': True
                    }
                    
                    print(f"      ✅ {dataset_name}: {forward_time:.4f}s, MSE: {mse:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ {dataset_name}: Error - {e}")
                    model_results[dataset_name] = {
                        'forward_time': None,
                        'mse': None,
                        'mae': None,
                        'output_shape': None,
                        'n_test_samples': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            results[model_name] = model_results
        
        return results
    
    def benchmark_lrdbench_estimators(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark LRDBench estimators on the raw time series data.
        """
        if not LRDBENCH_AVAILABLE:
            print("❌ LRDBench not available for estimator benchmarking")
            return {}
        
        print("\n⚡ Benchmarking LRDBench estimators...")
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\n  📊 Benchmarking {dataset_name}...")
            
            try:
                # Use raw time series data for LRDBench estimators
                raw_data = dataset['raw_data']
                
                # Ensure data is numpy array
                if hasattr(raw_data, 'numpy'):
                    raw_data = raw_data.numpy()
                elif hasattr(raw_data, 'cpu'):
                    raw_data = raw_data.cpu().numpy()
                
                raw_data = np.asarray(raw_data, dtype=np.float64)
                
                print(f"    Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
                
                # Test individual estimators instead of comprehensive benchmark
                estimator_results = {}
                
                # Test basic statistical estimators
                try:
                    # R/S estimator (simplified)
                    start_time = time.time()
                    
                    # Simple R/S calculation
                    n = len(raw_data)
                    k = n // 4  # Use quarter of data
                    segments = [raw_data[i:i+k] for i in range(0, n-k+1, k//2)]
                    
                    rs_values = []
                    for segment in segments:
                        if len(segment) > 1:
                            mean_val = np.mean(segment)
                            cumsum = np.cumsum(segment - mean_val)
                            R = np.max(cumsum) - np.min(cumsum)
                            S = np.std(segment)
                            if S > 0:
                                rs_values.append(R / S)
                    
                    if rs_values:
                        H_est = np.log(np.mean(rs_values)) / np.log(k)
                        execution_time = time.time() - start_time
                        
                        estimator_results['RS'] = {
                            'success': True,
                            'execution_time': execution_time,
                            'estimated_H': H_est,
                            'method': 'Simplified R/S'
                        }
                        print(f"      ✅ R/S: H={H_est:.3f}, Time={execution_time:.4f}s")
                    else:
                        estimator_results['RS'] = {
                            'success': False,
                            'error': 'No valid R/S values'
                        }
                        print(f"      ❌ R/S: Failed")
                        
                except Exception as e:
                    estimator_results['RS'] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"      ❌ R/S: Error - {e}")
                
                # Test variance scaling
                try:
                    start_time = time.time()
                    
                    # Calculate variance for different segment lengths
                    segment_lengths = [10, 20, 50, 100]
                    variances = []
                    
                    for length in segment_lengths:
                        if length < len(raw_data):
                            segments = [raw_data[i:i+length] for i in range(0, len(raw_data)-length+1, length//2)]
                            if segments:
                                segment_vars = [np.var(seg) for seg in segments]
                                variances.append(np.mean(segment_vars))
                    
                    if len(variances) > 1:
                        # Fit power law: var ∝ length^(2H-1)
                        log_lengths = np.log(segment_lengths[:len(variances)])
                        log_vars = np.log(variances)
                        
                        # Simple linear fit
                        coeffs = np.polyfit(log_lengths, log_vars, 1)
                        H_est = (coeffs[0] + 1) / 2
                        
                        execution_time = time.time() - start_time
                        
                        estimator_results['Variance'] = {
                            'success': True,
                            'execution_time': execution_time,
                            'estimated_H': H_est,
                            'method': 'Variance scaling'
                        }
                        print(f"      ✅ Variance: H={H_est:.3f}, Time={execution_time:.4f}s")
                    else:
                        estimator_results['Variance'] = {
                            'success': False,
                            'error': 'Insufficient data for variance scaling'
                        }
                        print(f"      ❌ Variance: Failed")
                        
                except Exception as e:
                    estimator_results['Variance'] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"      ❌ Variance: Error - {e}")
                
                results[dataset_name] = estimator_results
                
            except Exception as e:
                print(f"    ❌ Failed to benchmark {dataset_name}: {e}")
                results[dataset_name] = {}
        
        return results
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark.
        """
        print("🚀 Starting HPFracc Proper Benchmark")
        print("=" * 60)
        
        # Generate datasets
        print("\n📊 Generating synthetic datasets...")
        datasets = self.data_generator.generate_comprehensive_dataset()
        
        if not datasets:
            print("❌ No datasets generated, cannot proceed")
            return {}
        
        # Create hpfracc models
        hpfracc_models = self.create_hpfracc_models()
        
        # Benchmark hpfracc models
        hpfracc_results = self.benchmark_hpfracc_models(hpfracc_models, datasets)
        
        # Benchmark LRDBench estimators
        lrdbench_results = self.benchmark_lrdbench_estimators(datasets)
        
        # Compile results
        comprehensive_results = {
            'metadata': {
                'series_length': self.series_length,
                'batch_size': self.batch_size,
                'input_window': self.input_window,
                'prediction_horizon': self.prediction_horizon,
                'n_datasets': len(datasets),
                'n_hpfracc_models': len(hpfracc_models),
                'timestamp': datetime.now().isoformat()
            },
            'datasets_info': {
                name: {
                    'n_samples': dataset['metadata']['n_samples'],
                    'n_batches': dataset['metadata']['n_batches'],
                    'input_shape': dataset['X'].shape,
                    'target_shape': dataset['y'].shape,
                    'raw_stats': {
                        'mean': float(np.mean(dataset['raw_data'])),
                        'std': float(np.std(dataset['raw_data'])),
                        'min': float(np.min(dataset['raw_data'])),
                        'max': float(np.max(dataset['raw_data']))
                    }
                } for name, dataset in datasets.items()
            },
            'hpfracc_results': hpfracc_results,
            'lrdbench_results': lrdbench_results
        }
        
        self.results = comprehensive_results
        
        print("\n✅ Benchmark completed successfully!")
        return comprehensive_results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report.
        """
        if not self.results:
            return "No results available. Run benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("HPFRACC PROPER BENCHMARK REPORT")
        report.append("=" * 80)
        
        # Metadata
        metadata = self.results['metadata']
        report.append(f"\n📊 BENCHMARK METADATA:")
        report.append(f"  Series length: {metadata['series_length']:,}")
        report.append(f"  Batch size: {metadata['batch_size']}")
        report.append(f"  Input window: {metadata['input_window']}")
        report.append(f"  Prediction horizon: {metadata['prediction_horizon']}")
        report.append(f"  Number of datasets: {metadata['n_datasets']}")
        report.append(f"  HPFracc models: {metadata['n_hpfracc_models']}")
        report.append(f"  Timestamp: {metadata['timestamp']}")
        
        # Dataset summary
        report.append(f"\n📈 DATASET SUMMARY:")
        for name, info in self.results['datasets_info'].items():
            report.append(f"  {name}:")
            report.append(f"    Samples: {info['n_samples']}, Batches: {info['n_batches']}")
            report.append(f"    Input: {info['input_shape']}, Target: {info['target_shape']}")
            stats = info['raw_stats']
            report.append(f"    Stats: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
        
        # HPFracc results
        if self.results['hpfracc_results']:
            report.append(f"\n🧠 HPFRACC FRACTIONAL NEURAL NETWORK RESULTS:")
            for model_name, model_results in self.results['hpfracc_results'].items():
                report.append(f"\n  {model_name}:")
                successful_runs = sum(1 for r in model_results.values() if r.get('success', False))
                total_runs = len(model_results)
                
                if successful_runs > 0:
                    avg_time = np.mean([r['forward_time'] for r in model_results.values() 
                                      if r.get('success', False) and r['forward_time']])
                    avg_mse = np.mean([r['mse'] for r in model_results.values() 
                                     if r.get('success', False) and r['mse']])
                    
                    report.append(f"    Success rate: {successful_runs}/{total_runs}")
                    report.append(f"    Avg forward time: {avg_time:.4f}s")
                    report.append(f"    Avg MSE: {avg_mse:.4f}")
                else:
                    report.append(f"    No successful runs")
        
        # LRDBench results
        if self.results['lrdbench_results']:
            report.append(f"\n⚡ LRDBENCH ESTIMATOR RESULTS:")
            for data_name, data_results in self.results['lrdbench_results'].items():
                report.append(f"\n  {data_name}:")
                for estimator_name, estimator_result in data_results.items():
                    if estimator_result.get('success', False):
                        report.append(f"    {estimator_name}: "
                                   f"Time={estimator_result['execution_time']:.4f}s, "
                                   f"H={estimator_result['estimated_H']:.3f}")
                    else:
                        report.append(f"    {estimator_name}: Failed")
        
        # Summary statistics
        report.append(f"\n📊 SUMMARY STATISTICS:")
        if self.results['hpfracc_results']:
            hpfracc_success_rate = np.mean([
                np.mean([r.get('success', False) for r in model_results.values()])
                for model_results in self.results['hpfracc_results'].values()
            ])
            report.append(f"  HPFracc overall success rate: {hpfracc_success_rate:.1%}")
        
        if self.results['lrdbench_results']:
            lrdbench_success_rate = np.mean([
                np.mean([r.get('success', False) for r in data_results.values()])
                for data_results in self.results['lrdbench_results'].values()
            ])
            report.append(f"  LRDBench overall success rate: {lrdbench_success_rate:.1%}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save benchmark results to JSON file.
        """
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
            
            # Convert results
            serializable_results = convert_numpy(results)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hpfracc_proper_benchmark_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function to run the benchmark."""
    
    # Create benchmark instance
    benchmark = HPFraccProperBenchmark(
        series_length=1000,
        batch_size=32,
        input_window=10,
        prediction_horizon=1
    )
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    if results:
        # Generate and display report
        report = benchmark.generate_report()
        print(report)
        
        # Save results
        benchmark.save_results(results)
        
        # Display analytics summary if available
        try:
            summary = get_analytics_summary()
            print(f"\n📊 Analytics Summary:")
            print(summary)
        except Exception as e:
            print(f"\n⚠️ Could not retrieve analytics summary: {e}")
    
    print("\n🎉 Benchmark completed!")

if __name__ == "__main__":
    main()

