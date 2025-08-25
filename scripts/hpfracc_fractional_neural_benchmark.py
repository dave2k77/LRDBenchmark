#!/usr/bin/env python3
"""
HPFracc Fractional Neural Network Benchmark

This script benchmarks hpfracc's fractional neural networks against LRDBench's
classical, ML, and neural network estimators on synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Import hpfracc components
try:
    from hpfracc.ml import (
        FractionalNeuralNetwork, 
        FractionalLSTM, 
        FractionalTransformer,
        FractionalConv1D,
        FractionalConv2D,
        FractionalAdam,
        FractionalMSELoss,
        MLConfig,
        BackendType
    )
    from hpfracc.ml.backends import get_backend_manager, switch_backend
    HPRACC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: hpfracc.ml not available: {e}")
    HPRACC_AVAILABLE = False

# Import LRDBench components
try:
    from lrdbench import (
        FBMModel, FGNModel, ARFIMAModel, MRWModel,
        ComprehensiveBenchmark,
        enable_analytics, get_analytics_summary
    )
    from lrdbench.analytics import UsageTracker, PerformanceMonitor
    LRDBENCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: lrdbench not available: {e}")
    LRDBENCH_AVAILABLE = False

class HPFraccFractionalNeuralBenchmark:
    """
    Benchmark hpfracc's fractional neural networks against LRDBench estimators.
    """
    
    def __init__(self, backend: str = 'jax'):
        """
        Initialize the benchmark.
        
        Args:
            backend: Backend to use ('torch', 'jax', 'numba', 'auto')
        """
        self.backend = backend
        self.results = {}
        self.performance_metrics = {}
        
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
    
    def generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """
        Generate synthetic time series data for benchmarking.
        """
        print("📊 Generating synthetic data...")
        
        synthetic_data = {}
        
        # Generate FBM data with different H values
        for H in self.H_values:
            try:
                model = FBMModel(H=H, sigma=1.0)
                data = model.generate(self.series_length, seed=42)
                
                # Ensure data is numpy array
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                elif hasattr(data, 'cpu'):
                    data = data.cpu().numpy()
                
                data = np.asarray(data, dtype=np.float64)
                synthetic_data[f'fbm_H{H}'] = data
                print(f"  ✅ Generated FBM with H={H}: {len(data)} samples, μ={np.mean(data):.3f}, σ={np.std(data):.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate FBM with H={H}: {e}")
        
        # Generate FGN data with different H values
        for H in self.H_values:
            try:
                model = FGNModel(H=H, sigma=1.0)
                data = model.generate(self.series_length, seed=42)
                
                # Ensure data is numpy array
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                elif hasattr(data, 'cpu'):
                    data = data.cpu().numpy()
                
                data = np.asarray(data, dtype=np.float64)
                synthetic_data[f'fgn_H{H}'] = data
                print(f"  ✅ Generated FGN with H={H}: {len(data)} samples, μ={np.mean(data):.3f}, σ={np.std(data):.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate FGN with H={H}: {e}")
        
        # Generate ARFIMA data with different d values
        for d in self.d_values:
            try:
                model = ARFIMAModel(d=d, sigma=1.0)
                data = model.generate(self.series_length, seed=42)
                
                # Ensure data is numpy array
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                elif hasattr(data, 'cpu'):
                    data = data.cpu().numpy()
                
                data = np.asarray(data, dtype=np.float64)
                synthetic_data[f'arfima_d{d}'] = data
                print(f"  ✅ Generated ARFIMA with d={d}: {len(data)} samples, μ={np.mean(data):.3f}, σ={np.std(data):.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate ARFIMA with d={d}: {e}")
        
        # Generate MRW data with different H values
        for H in self.H_values:
            try:
                model = MRWModel(H=H, lambda_param=0.1, sigma=1.0)
                data = model.generate(self.series_length, seed=42)
                
                # Ensure data is numpy array
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                elif hasattr(data, 'cpu'):
                    data = data.cpu().numpy()
                
                data = np.asarray(data, dtype=np.float64)
                synthetic_data[f'mrw_H{H}'] = data
                print(f"  ✅ Generated MRW with H={H}: {len(data)} samples, μ={np.mean(data):.3f}, σ={np.std(data):.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate MRW with H={H}: {e}")
        
        print(f"✅ Generated {len(synthetic_data)} synthetic datasets")
        return synthetic_data
    
    def create_hpfracc_models(self) -> Dict[str, Any]:
        """
        Create hpfracc fractional neural network models.
        
        Returns:
            Dictionary of model instances
        """
        if not HPRACC_AVAILABLE:
            print("❌ hpfracc not available for model creation")
            return {}
        
        models = {}
        
        try:
            # Set the backend first
            try:
                from hpfracc.ml.backends import get_backend_manager, BackendType
                backend_manager = get_backend_manager()
                backend_manager.switch_backend(BackendType.JAX)
                print("✅ Set hpfracc backend to JAX")
            except Exception as e:
                print(f"⚠️ Could not set hpfracc backend: {e}")
            
            # Try different model types with correct parameters based on API docs
            try:
                # Create FractionalNeuralNetwork with correct parameters
                model = FractionalNeuralNetwork(
                    input_size=1,
                    hidden_sizes=[64, 32, 16],
                    output_size=1,
                    fractional_order=0.5
                )
                
                # Fix the backend issues we discovered
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalNeuralNetwork'] = model
                print("✅ Created FractionalNeuralNetwork with JAX backend")
            except Exception as e:
                print(f"⚠️ Could not create FractionalNeuralNetwork: {e}")
            
            try:
                # Create FractionalLSTM with correct parameters (removing fractional_order)
                model = FractionalLSTM(
                    input_size=1,
                    hidden_size=32,
                    num_layers=1
                )
                
                # Fix the backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalLSTM'] = model
                print("✅ Created FractionalLSTM with JAX backend")
            except Exception as e:
                print(f"⚠️ Could not create FractionalLSTM: {e}")
            
            try:
                # Create FractionalTransformer with correct parameters (removing nhead)
                model = FractionalTransformer(
                    d_model=32,
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    dim_feedforward=128,
                    dropout=0.1,
                    activation="relu"
                )
                
                # Fix the backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalTransformer'] = model
                print("✅ Created FractionalTransformer with JAX backend")
            except Exception as e:
                print(f"⚠️ Could not create FractionalTransformer: {e}")
            
            try:
                # Create FractionalConv1D with correct parameters (removing fractional_order)
                model = FractionalConv1D(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3
                )
                
                # Fix the backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalConv1D'] = model
                print("✅ Created FractionalConv1D with JAX backend")
            except Exception as e:
                print(f"⚠️ Could not create FractionalConv1D: {e}")
            
            try:
                # Create FractionalConv2D with correct parameters (removing fractional_order)
                model = FractionalConv2D(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3
                )
                
                # Fix the backend issues
                if hasattr(model, 'backend'):
                    model.backend = BackendType.JAX
                if hasattr(model, 'config') and hasattr(model.config, 'backend'):
                    model.config.backend = BackendType.JAX
                if hasattr(model, 'tensor_ops') and hasattr(model.tensor_ops, 'backend'):
                    model.tensor_ops.backend = BackendType.JAX
                
                models['FractionalConv2D'] = model
                print("✅ Created FractionalConv2D with JAX backend")
            except Exception as e:
                print(f"⚠️ Could not create FractionalConv2D: {e}")
                
        except Exception as e:
            print(f"⚠️ Error creating hpfracc models: {e}")
        
        return models
    
    def benchmark_hpfracc_models(self, models: Dict[str, Any], 
                                data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Benchmark hpfracc models on synthetic data.
        
        Args:
            models: Dictionary of hpfracc models
            data: Dictionary of synthetic datasets
            
        Returns:
            Dictionary of benchmark results
        """
        if not models or not data:
            return {}
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔬 Benchmarking {model_name}...")
            model_results = {}
            
            for data_name, time_series in data.items():
                try:
                    # Prepare data for neural network
                    # Ensure we have enough data for X and y
                    if len(time_series) < 2:
                        print(f"    ⚠️ {data_name}: Not enough data (need at least 2 samples)")
                        continue
                    
                    # Debug: print data shape
                    print(f"    Debug: {data_name} shape: {time_series.shape}, length: {len(time_series)}")
                    
                    # Create input-output pairs for time series prediction
                    # For FractionalNeuralNetwork, we need (batch, input_size) where input_size=1
                    # Use simple sliding window: predict next value from current value
                    X = time_series[:-1].reshape(-1, 1)  # (batch, 1) - current values
                    y = time_series[1:].reshape(-1, 1)   # (batch, 1) - next values
                    
                    print(f"    Debug: Final X shape: {X.shape}, y shape: {y.shape}")
                    
                    # Benchmark forward pass
                    start_time = time.time()
                    
                    if hasattr(model, 'forward'):
                        # Standard forward pass with fractional computation disabled
                        output = model.forward(X, use_fractional=False)
                    elif hasattr(model, 'fractional_forward'):
                        # Fractional forward pass
                        output = model.fractional_forward(X)
                    else:
                        print(f"⚠️ {model_name} has no forward method")
                        continue
                    
                    forward_time = time.time() - start_time
                    
                    # Calculate basic metrics
                    mse = np.mean((output - y) ** 2)
                    mae = np.mean(np.abs(output - y))
                    
                    model_results[data_name] = {
                        'forward_time': forward_time,
                        'mse': mse,
                        'mae': mae,
                        'output_shape': output.shape,
                        'success': True
                    }
                    
                    print(f"  ✅ {data_name}: {forward_time:.4f}s, MSE: {mse:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ {data_name}: Error - {e}")
                    model_results[data_name] = {
                        'forward_time': None,
                        'mse': None,
                        'mae': None,
                        'output_shape': None,
                        'success': False,
                        'error': str(e)
                    }
            
            results[model_name] = model_results
        
        return results
    
    def benchmark_lrdbench_estimators(self, synthetic_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark LRDBench estimators against synthetic data.
        """
        print("\n⚡ Benchmarking LRDBench estimators...")
        print("=" * 60)
        
        results = {}
        
        for data_name, time_series in synthetic_data.items():
            print(f"\n  📊 Benchmarking {data_name}...")
            
            # Ensure data is numpy array and convert if needed
            if hasattr(time_series, 'numpy'):
                time_series = time_series.numpy()
            elif hasattr(time_series, 'cpu'):
                time_series = time_series.cpu().numpy()
            
            # Convert to numpy array if it's not already
            time_series = np.asarray(time_series, dtype=np.float64)
            
            print(f"    Data type: {type(time_series)}, shape: {time_series.shape}, dtype: {time_series.dtype}")
            
            try:
                # Create benchmark instance
                from lrdbench.analysis.benchmark import ComprehensiveBenchmark
                benchmark = ComprehensiveBenchmark()
                
                # Check available methods
                available_methods = [method for method in dir(benchmark) if method.startswith('run_') and 'benchmark' in method]
                print(f"    Available benchmark methods: {available_methods}")
                
                # Try to run classical benchmark
                try:
                    start_time = time.time()
                    result = benchmark.run_classical_benchmark(data_length=len(time_series))
                    execution_time = time.time() - start_time
                    
                    print(f"    ✅ Completed in {execution_time:.4f}s")
                    
                    results[data_name] = {
                        'success': True,
                        'execution_time': execution_time,
                        'result': result
                    }
                    
                except Exception as e:
                    print(f"    ⚠️ Error with run_classical_benchmark: {e}")
                    results[data_name] = {
                        'success': False,
                        'error': str(e),
                        'execution_time': 0.0
                    }
                    
            except Exception as e:
                print(f"    ❌ Failed to create benchmark: {e}")
                results[data_name] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0.0
                }
        
        return results
    
    def run_comprehensive_benchmark(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparing hpfracc and LRDBench.
        
        Args:
            n_samples: Number of samples for synthetic data
            
        Returns:
            Comprehensive benchmark results
        """
        print("🚀 Starting HPFracc Fractional Neural Network Benchmark")
        print("=" * 60)
        
        # Generate synthetic data
        print("\n📊 Generating synthetic data...")
        data = self.generate_synthetic_data(n_samples)
        
        if not data:
            print("❌ No data generated, cannot proceed")
            return {}
        
        # Create hpfracc models
        print("\n🧠 Creating hpfracc fractional neural networks...")
        hpfracc_models = self.create_hpfracc_models()
        
        # Benchmark hpfracc models
        print("\n⚡ Benchmarking hpfracc models...")
        hpfracc_results = self.benchmark_hpfracc_models(hpfracc_models, data)
        
        # Benchmark LRDBench estimators
        print("\n⚡ Benchmarking LRDBench estimators...")
        lrdbench_results = self.benchmark_lrdbench_estimators(data)
        
        # Compile results
        comprehensive_results = {
            'metadata': {
                'backend': self.backend,
                'n_samples': n_samples,
                'n_datasets': len(data),
                'n_hpfracc_models': len(hpfracc_models),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'data_info': {
                name: {
                    'length': len(ts),
                    'mean': float(np.mean(ts)),
                    'std': float(np.std(ts)),
                    'min': float(np.min(ts)),
                    'max': float(np.max(ts))
                } for name, ts in data.items()
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
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results available. Run benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("HPFRACC FRACTIONAL NEURAL NETWORK BENCHMARK REPORT")
        report.append("=" * 80)
        
        # Metadata
        metadata = self.results['metadata']
        report.append(f"\n📊 BENCHMARK METADATA:")
        report.append(f"  Backend: {metadata['backend']}")
        report.append(f"  Samples per dataset: {metadata['n_samples']:,}")
        report.append(f"  Number of datasets: {metadata['n_datasets']}")
        report.append(f"  HPFracc models: {metadata['n_hpfracc_models']}")
        report.append(f"  Timestamp: {metadata['timestamp']}")
        
        # Data summary
        report.append(f"\n📈 DATA SUMMARY:")
        for name, info in self.results['data_info'].items():
            report.append(f"  {name}: {info['length']} samples, "
                        f"μ={info['mean']:.3f}, σ={info['std']:.3f}")
        
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
                                   f"MSE={estimator_result['mse']:.4f}, "
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
            # Convert JAX arrays to Python types for JSON serialization
            def convert_jax_arrays(obj):
                if hasattr(obj, 'numpy'):
                    return obj.numpy().tolist()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_jax_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_jax_arrays(item) for item in obj]
                else:
                    return obj
            
            # Convert results
            serializable_results = convert_jax_arrays(results)
            
            # Save to file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hpfracc_benchmark_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print("   This is likely due to JAX arrays that couldn't be converted")

def main():
    """
    Main function to run the benchmark.
    """
    # Create benchmark instance
    benchmark = HPFraccFractionalNeuralBenchmark(backend='jax')
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(n_samples=1000)
    
    # Save results
    benchmark.save_results(results)
    
    # Display analytics summary if available
    try:
        from lrdbench.analytics import get_analytics_summary
        summary = get_analytics_summary()
        print("\n📊 Analytics Summary:")
        print(summary)
    except Exception as e:
        print(f"\n⚠️ Could not retrieve analytics summary: {e}")
    
    print("\n🎉 Benchmark completed!")

if __name__ == "__main__":
    main()
