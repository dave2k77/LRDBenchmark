#!/usr/bin/env python3
"""
Debug Estimator Benchmark: Test All Estimators vs Individual Data Generators

This script systematically tests all available estimators against one data generator
at a time to identify and fix issues before running the full comprehensive benchmark.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import psutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import data models
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
from models.data_models.fgn.fgn_model import FractionalGaussianNoise
from models.data_models.arfima.arfima_model import ARFIMAModel
from models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Try to import neural fSDE
try:
    from models.data_models.neural_fsde import create_fsde_net
    NEURAL_FSDE_AVAILABLE = True
except ImportError:
    NEURAL_FSDE_AVAILABLE = False

# Import all available estimators
try:
    from analysis.temporal.dfa.dfa_estimator import DFAEstimator
    DFA_AVAILABLE = True
except ImportError:
    DFA_AVAILABLE = False

try:
    from analysis.temporal.rs.rs_estimator import RSEstimator
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

try:
    from analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
    HIGUCHI_AVAILABLE = True
except ImportError:
    HIGUCHI_AVAILABLE = False

try:
    from analysis.temporal.dma.dma_estimator import DMAEstimator
    DMA_AVAILABLE = True
except ImportError:
    DMA_AVAILABLE = False

try:
    from analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
    PERIODOGRAM_AVAILABLE = True
except ImportError:
    PERIODOGRAM_AVAILABLE = False

try:
    from analysis.spectral.whittle.whittle_estimator import WhittleEstimator
    WHITTLE_AVAILABLE = True
except ImportError:
    WHITTLE_AVAILABLE = False

try:
    from analysis.spectral.gph.gph_estimator import GPHEstimator
    GPH_AVAILABLE = True
except ImportError:
    GPH_AVAILABLE = False

try:
    from analysis.wavelet.log_variance.wavelet_log_variance_estimator import WaveletLogVarianceEstimator
    WAVELET_LOG_VAR_AVAILABLE = True
except ImportError:
    WAVELET_LOG_VAR_AVAILABLE = False

try:
    from analysis.wavelet.variance.wavelet_variance_estimator import WaveletVarianceEstimator
    WAVELET_VAR_AVAILABLE = True
except ImportError:
    WAVELET_VAR_AVAILABLE = False

try:
    from analysis.wavelet.whittle.wavelet_whittle_estimator import WaveletWhittleEstimator
    WAVELET_WHITTLE_AVAILABLE = True
except ImportError:
    WAVELET_WHITTLE_AVAILABLE = False

try:
    from analysis.wavelet.cwt.cwt_estimator import CWTEstimator
    CWT_AVAILABLE = True
except ImportError:
    CWT_AVAILABLE = False

try:
    from analysis.multifractal.mfdfa.mfdfa_estimator import MFDFAEstimator
    MFDFA_AVAILABLE = True
except ImportError:
    MFDFA_AVAILABLE = False

try:
    from analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator import WaveletLeadersEstimator
    WAVELET_LEADERS_AVAILABLE = True
except ImportError:
    WAVELET_LEADERS_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DebugEstimatorBenchmark:
    """Debug benchmark for testing all estimators against individual data generators."""
    
    def __init__(self, data_length: int = 2048, n_trials: int = 3):
        self.data_length = data_length
        self.n_trials = n_trials
        self.results = {}
        
        # Initialize all available estimators with debug-friendly parameters
        self.estimators = self._initialize_estimators()
        
        # Test parameters for different Hurst values
        self.test_hurst_values = [0.3, 0.5, 0.7]
        
        print(f"Initialized {len(self.estimators)} estimators")
        print("Available estimators:")
        for name, info in self.estimators.items():
            print(f"  - {name} ({info['category']})")
        
        # Debug: Check import flags
        print("\nImport flags:")
        print(f"  WAVELET_VAR_AVAILABLE: {WAVELET_VAR_AVAILABLE}")
        print(f"  WAVELET_LOG_VAR_AVAILABLE: {WAVELET_LOG_VAR_AVAILABLE}")
        print(f"  WAVELET_WHITTLE_AVAILABLE: {WAVELET_WHITTLE_AVAILABLE}")
        print(f"  CWT_AVAILABLE: {CWT_AVAILABLE}")
        
    def _initialize_estimators(self) -> Dict[str, Any]:
        """Initialize all available estimators with debug-friendly parameters."""
        estimators = {}
        
        # Temporal estimators
        if DFA_AVAILABLE:
            estimators['DFA'] = {
                'class': DFAEstimator,
                'params': {'min_box_size': 10, 'max_box_size': 100},
                'category': 'temporal'
            }
        
        if RS_AVAILABLE:
            estimators['R/S'] = {
                'class': RSEstimator,
                'params': {'min_window_size': 10, 'max_window_size': 100},
                'category': 'temporal'
            }
        
        if HIGUCHI_AVAILABLE:
            estimators['Higuchi'] = {
                'class': HiguchiEstimator,
                'params': {'min_k': 2, 'max_k': 20},
                'category': 'temporal'
            }
        
        if DMA_AVAILABLE:
            estimators['DMA'] = {
                'class': DMAEstimator,
                'params': {'min_window_size': 4, 'max_window_size': 50},
                'category': 'temporal'
            }
        
        # Spectral estimators
        if PERIODOGRAM_AVAILABLE:
            estimators['Periodogram'] = {
                'class': PeriodogramEstimator,
                'params': {'min_freq_ratio': 0.01, 'max_freq_ratio': 0.1},
                'category': 'spectral'
            }
        
        if WHITTLE_AVAILABLE:
            estimators['Whittle'] = {
                'class': WhittleEstimator,
                'params': {'min_freq_ratio': 0.01, 'max_freq_ratio': 0.1},
                'category': 'spectral'
            }
        
        if GPH_AVAILABLE:
            estimators['GPH'] = {
                'class': GPHEstimator,
                'params': {'min_freq_ratio': 0.01, 'max_freq_ratio': 0.1},
                'category': 'spectral'
            }
        
        # Wavelet estimators - using smaller scales for debugging
        if WAVELET_LOG_VAR_AVAILABLE:
            estimators['Wavelet Log Variance'] = {
                'class': WaveletLogVarianceEstimator,
                'params': {'scales': list(range(2, 9))},  # Much smaller scales
                'category': 'wavelet'
            }
        
        if WAVELET_VAR_AVAILABLE:
            estimators['Wavelet Variance'] = {
                'class': WaveletVarianceEstimator,
                'params': {'scales': list(range(2, 9))},  # Much smaller scales
                'category': 'wavelet'
            }
        
        if WAVELET_WHITTLE_AVAILABLE:
            estimators['Wavelet Whittle'] = {
                'class': WaveletWhittleEstimator,
                'params': {'scales': list(range(2, 9))},  # Much smaller scales
                'category': 'wavelet'
            }
        
        if CWT_AVAILABLE:
            estimators['CWT'] = {
                'class': CWTEstimator,
                'params': {'scales': np.logspace(1, 2, 8)},  # Much smaller scales
                'category': 'wavelet'
            }
        
        # Multifractal estimators
        if MFDFA_AVAILABLE:
            estimators['MFDFA'] = {
                'class': MFDFAEstimator,
                'params': {'min_box_size': 10, 'max_box_size': 100, 'q_values': [-2, -1, 0, 1, 2]},
                'category': 'multifractal'
            }
        
        if WAVELET_LEADERS_AVAILABLE:
            estimators['Wavelet Leaders'] = {
                'class': WaveletLeadersEstimator,
                'params': {'min_scale': 2, 'max_scale': 8},  # Much smaller scales
                'category': 'multifractal'
            }
        
        return estimators
    
    def test_single_estimator(self, estimator_name: str, estimator_info: Dict, 
                             data: np.ndarray, true_hurst: float) -> Dict[str, Any]:
        """Test a single estimator and return detailed results."""
        try:
            print(f"    Testing {estimator_name}...", end=' ')
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create estimator instance
            estimator = estimator_info['class'](**estimator_info['params'])
            
            # Run estimation
            results = estimator.estimate(data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Extract estimated Hurst parameter
            estimated_hurst = results.get('hurst_parameter', None)
            if estimated_hurst is None:
                for key in ['H', 'hurst', 'fractal_dimension']:
                    if key in results:
                        estimated_hurst = results[key]
                        break
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Calculate accuracy metrics
            if estimated_hurst is not None and true_hurst is not None:
                absolute_error = abs(estimated_hurst - true_hurst)
                relative_error = absolute_error / true_hurst if true_hurst != 0 else float('inf')
                squared_error = (estimated_hurst - true_hurst) ** 2
            else:
                absolute_error = relative_error = squared_error = float('inf')
            
            result = {
                'estimator_name': estimator_name,
                'estimated_hurst': estimated_hurst,
                'true_hurst': true_hurst,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'squared_error': squared_error,
                'success': True,
                'raw_results': results
            }
            
            print(f"✓ H_est={estimated_hurst:.3f}, error={absolute_error:.3f}, time={execution_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed: {error_msg}")
            
            # Print full traceback for debugging
            print(f"      Full error: {traceback.format_exc()}")
            
            return {
                'estimator_name': estimator_name,
                'estimated_hurst': None,
                'true_hurst': true_hurst,
                'execution_time': 0,
                'memory_usage': 0,
                'absolute_error': float('inf'),
                'relative_error': float('inf'),
                'squared_error': float('inf'),
                'success': False,
                'error': error_msg,
                'raw_results': {}
            }
    
    def test_data_generator(self, generator_name: str, generator_class, generator_params: Dict) -> None:
        """Test all estimators against a single data generator."""
        print(f"\n{'='*80}")
        print(f"TESTING DATA GENERATOR: {generator_name}")
        print(f"{'='*80}")
        
        generator_results = []
        
        for hurst_value in self.test_hurst_values:
            print(f"\n  Hurst parameter: {hurst_value}")
            
            for trial in range(self.n_trials):
                print(f"    Trial {trial + 1}:")
                
                # Generate test data
                try:
                    np.random.seed(trial)
                    
                    if generator_name == 'fBm':
                        generator = generator_class(H=hurst_value, **generator_params)
                        data = generator.generate(self.data_length, seed=trial)
                        true_hurst = hurst_value
                    elif generator_name == 'fGn':
                        generator = generator_class(H=hurst_value, **generator_params)
                        data = generator.generate(self.data_length, seed=trial)
                        true_hurst = hurst_value
                    elif generator_name == 'ARFIMA':
                        d_value = hurst_value - 0.5
                        generator = generator_class(d=d_value, **generator_params)
                        data = generator.generate(self.data_length, seed=trial)
                        true_hurst = hurst_value
                    elif generator_name == 'MRW':
                        generator = generator_class(H=hurst_value, **generator_params)
                        data = generator.generate(self.data_length, seed=trial)
                        true_hurst = hurst_value
                    elif 'Neural fSDE' in generator_name:
                        params = generator_params.copy()
                        params['hurst_parameter'] = hurst_value
                        generator = generator_class(**params)
                        data = generator.simulate(n_samples=self.data_length, dt=0.01)[:, 0]
                        true_hurst = hurst_value
                    else:
                        generator = generator_class(**generator_params)
                        data = generator.generate(self.data_length, seed=trial)
                        true_hurst = 0.5
                    
                    print(f"      Generated data: length={len(data)}, mean={data.mean():.3f}, std={data.std():.3f}")
                    
                except Exception as e:
                    print(f"      ✗ Failed to generate data: {e}")
                    continue
                
                # Test all estimators
                for estimator_name, estimator_info in self.estimators.items():
                    result = self.test_single_estimator(
                        estimator_name, estimator_info, data, true_hurst
                    )
                    
                    result.update({
                        'generator_name': generator_name,
                        'hurst_value': hurst_value,
                        'trial': trial,
                        'data_length': self.data_length
                    })
                    
                    generator_results.append(result)
        
        # Analyze results for this generator
        self._analyze_generator_results(generator_name, generator_results)
        
        # Store results
        self.results[generator_name] = generator_results
    
    def _analyze_generator_results(self, generator_name: str, results: List[Dict]) -> None:
        """Analyze results for a specific data generator."""
        print(f"\n  📊 RESULTS SUMMARY FOR {generator_name}")
        print(f"  {'-'*60}")
        
        df = pd.DataFrame(results)
        
        # Group by estimator
        for estimator_name in df['estimator_name'].unique():
            estimator_data = df[df['estimator_name'] == estimator_name]
            successful = estimator_data[estimator_data['success'] == True]
            failed = estimator_data[estimator_data['success'] == False]
            
            print(f"    {estimator_name}:")
            
            if len(successful) > 0:
                success_rate = len(successful) / len(estimator_data) * 100
                mean_error = successful['absolute_error'].mean()
                mean_time = successful['execution_time'].mean()
                
                print(f"      ✓ Success: {success_rate:.1f}% ({len(successful)}/{len(estimator_data)})")
                print(f"      ✓ Mean Error: {mean_error:.3f}")
                print(f"      ✓ Mean Time: {mean_time:.3f}s")
            else:
                print(f"      ✗ Failed: 0% ({len(failed)}/{len(estimator_data)})")
                
                # Show error details for failed estimators
                if len(failed) > 0:
                    errors = failed['error'].unique()
                    print(f"      ✗ Errors: {', '.join(errors[:3])}")  # Show first 3 unique errors
    
    def run_debug_benchmark(self) -> None:
        """Run the debug benchmark against all data generators."""
        print("=== DEBUG ESTIMATOR BENCHMARK ===")
        print(f"Data length: {self.data_length}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Estimators: {len(self.estimators)}")
        
        # Test against fBm
        self.test_data_generator('fBm', FractionalBrownianMotion, {'sigma': 1.0})
        
        # Test against fGn
        self.test_data_generator('fGn', FractionalGaussianNoise, {'sigma': 1.0})
        
        # Test against ARFIMA
        self.test_data_generator('ARFIMA', ARFIMAModel, {
            'ar_params': [0.5], 'ma_params': [0.3], 'sigma': 1.0, 'method': 'spectral'
        })
        
        # Test against MRW
        self.test_data_generator('MRW', MultifractalRandomWalk, {
            'lambda_param': 0.02, 'sigma': 1.0, 'method': 'cascade'
        })
        
        # Test against Neural fSDE if available
        if NEURAL_FSDE_AVAILABLE:
            try:
                self.test_data_generator('Neural fSDE (JAX)', create_fsde_net, {
                    'state_dim': 1, 'hidden_dim': 32, 'num_layers': 3, 'framework': 'jax'
                })
            except Exception as e:
                print(f"Neural fSDE (JAX) not available: {e}")
        
        print(f"\n{'='*80}")
        print("DEBUG BENCHMARK COMPLETE")
        print(f"{'='*80}")
        
        # Save results
        self.save_results()
    
    def save_results(self) -> None:
        """Save debug benchmark results."""
        Path('debug_results').mkdir(exist_ok=True)
        
        # Save detailed results
        all_results = []
        for generator_results in self.results.values():
            all_results.extend(generator_results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv('debug_results/debug_benchmark_results.csv', index=False)
            print("Debug results saved to: debug_results/debug_benchmark_results.csv")
            
            # Create summary
            summary = {}
            for estimator_name in df['estimator_name'].unique():
                estimator_data = df[df['estimator_name'] == estimator_name]
                successful = estimator_data[estimator_data['success'] == True]
                
                if len(successful) > 0:
                    summary[estimator_name] = {
                        'total_trials': len(estimator_data),
                        'successful_trials': len(successful),
                        'success_rate': len(successful) / len(estimator_data),
                        'mean_absolute_error': successful['absolute_error'].mean(),
                        'mean_execution_time': successful['execution_time'].mean()
                    }
                else:
                    summary[estimator_name] = {
                        'total_trials': len(estimator_data),
                        'successful_trials': 0,
                        'success_rate': 0.0,
                        'mean_absolute_error': float('inf'),
                        'mean_execution_time': 0.0
                    }
            
            summary_df = pd.DataFrame(summary).T
            summary_df.to_csv('debug_results/debug_summary.csv')
            print("Debug summary saved to: debug_results/debug_summary.csv")
            
            # Show overall summary
            print("\n📊 OVERALL DEBUG SUMMARY:")
            print("-" * 60)
            for estimator_name, stats in summary.items():
                success_rate = stats['success_rate'] * 100
                if stats['success_rate'] > 0:
                    mae = stats['mean_absolute_error']
                    time = stats['mean_execution_time']
                    print(f"  {estimator_name}: {success_rate:.1f}% success, MAE: {mae:.3f}, Time: {time:.3f}s")
                else:
                    print(f"  {estimator_name}: {success_rate:.1f}% success (FAILED)")

def main():
    """Main function to run the debug estimator benchmark."""
    print("=== DEBUG ESTIMATOR BENCHMARK: SYSTEMATIC TESTING ===")
    
    benchmark = DebugEstimatorBenchmark(data_length=2048, n_trials=3)
    
    try:
        benchmark.run_debug_benchmark()
        
    except Exception as e:
        print(f"Error during debug benchmark: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
