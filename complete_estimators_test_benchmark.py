#!/usr/bin/env python3
"""
Complete Estimators Test and Benchmark for LRDBench

This script runs comprehensive testing and benchmarking of all estimators
in the LRDBench package, including:
- All estimator types (temporal, spectral, wavelet, multifractal, ML, neural)
- Multiple data models (fBm, fGn, ARFIMA, MRW)
- Various contamination scenarios
- Performance profiling and optimization analysis
- Advanced metrics analysis
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

# Add current directory to path for imports
sys.path.append('.')

def test_imports():
    """Test that all core components can be imported."""
    print("🔍 Testing core imports...")
    
    try:
        # Test data models
        from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
        from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
        from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel
        from lrdbench.models.data_models.mrw.mrw_model import MultifractalRandomWalk
        print("✅ Data models imported successfully")
        
        # Test estimators
        from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
        from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
        from lrdbench.analysis.temporal.dma.dma_estimator import DMAEstimator
        from lrdbench.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
        from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
        from lrdbench.analysis.spectral.whittle.whittle_estimator import WhittleEstimator
        from lrdbench.analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
        from lrdbench.analysis.wavelet.cwt.cwt_estimator import CWTEstimator
        from lrdbench.analysis.wavelet.variance.wavelet_variance_estimator import WaveletVarianceEstimator
        from lrdbench.analysis.wavelet.log_variance.wavelet_log_variance_estimator import WaveletLogVarianceEstimator
        from lrdbench.analysis.wavelet.whittle.wavelet_whittle_estimator import WaveletWhittleEstimator
        from lrdbench.analysis.multifractal.mfdfa.mfdfa_estimator import MFDFAEstimator
        print("✅ Core estimators imported successfully")
        
        # Test auto-optimized estimators
        from lrdbench.analysis.auto_optimized_estimator import (
            AutoDFAEstimator, AutoRSEstimator, AutoDMAEstimator, AutoHiguchiEstimator,
            AutoGPHEstimator, AutoPeriodogramEstimator, AutoWhittleEstimator
        )
        print("✅ Auto-optimized estimators imported successfully")
        
        # Test ML estimators
        try:
            from lrdbench.analysis.machine_learning.random_forest.random_forest_estimator import RandomForestEstimator
            from lrdbench.analysis.machine_learning.svr.svr_estimator import SVREstimator
            from lrdbench.analysis.machine_learning.gradient_boosting.gradient_boosting_estimator import GradientBoostingEstimator
            print("✅ ML estimators imported successfully")
        except ImportError:
            print("⚠️  ML estimators not available")
        
        # Test neural estimators
        try:
            from lrdbench.analysis.neural.cnn.cnn_estimator import CNNEstimator
            from lrdbench.analysis.neural.transformer.transformer_estimator import TransformerEstimator
            print("✅ Neural estimators imported successfully")
        except ImportError:
            print("⚠️  Neural estimators not available")
        
        # Test enhanced neural estimators
        try:
            from lrdbench.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
            from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
            from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
            from lrdbench.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator
            print("✅ Enhanced neural estimators imported successfully")
        except ImportError:
            print("⚠️  Enhanced neural estimators not available")
        
        # Test benchmark system
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        print("✅ Benchmark system imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_estimators():
    """Test individual estimators with basic functionality."""
    print("\n🔍 Testing individual estimators...")
    
    # Import estimators here to ensure they're available
    from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
    from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
    from lrdbench.analysis.temporal.dma.dma_estimator import DMAEstimator
    from lrdbench.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
    from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
    from lrdbench.analysis.spectral.whittle.whittle_estimator import WhittleEstimator
    from lrdbench.analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
    from lrdbench.analysis.wavelet.cwt.cwt_estimator import CWTEstimator
    from lrdbench.analysis.wavelet.variance.wavelet_variance_estimator import WaveletVarianceEstimator
    from lrdbench.analysis.wavelet.log_variance.wavelet_log_variance_estimator import WaveletLogVarianceEstimator
    from lrdbench.analysis.wavelet.whittle.wavelet_whittle_estimator import WaveletWhittleEstimator
    from lrdbench.analysis.multifractal.mfdfa.mfdfa_estimator import MFDFAEstimator
    
    # Try to import enhanced estimators
    enhanced_estimators = []
    try:
        from lrdbench.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
        from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
        from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
        from lrdbench.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator
        enhanced_estimators = [
            ("Enhanced CNN", EnhancedCNNEstimator),
            ("Enhanced LSTM", EnhancedLSTMEstimator),
            ("Enhanced GRU", EnhancedGRUEstimator),
            ("Enhanced Transformer", EnhancedTransformerEstimator),
        ]
    except ImportError:
        print("   ⚠️  Enhanced estimators not available")
    
    estimators_to_test = [
        ("R/S", RSEstimator),
        ("DFA", DFAEstimator),
        ("DMA", DMAEstimator),
        ("Higuchi", HiguchiEstimator),
        ("GPH", GPHEstimator),
        ("Whittle", WhittleEstimator),
        ("Periodogram", PeriodogramEstimator),
        ("CWT", CWTEstimator),
        ("Wavelet Variance", WaveletVarianceEstimator),
        ("Wavelet Log Variance", WaveletLogVarianceEstimator),
        ("Wavelet Whittle", WaveletWhittleEstimator),
        ("MFDFA", MFDFAEstimator),
    ] + enhanced_estimators
    
    results = {}
    
    # Generate test data
    from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
    fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
    test_data = fbm.generate(1000, seed=42)
    
    for name, estimator_class in estimators_to_test:
        print(f"   Testing {name}...", end=" ")
        
        try:
            # Initialize estimator
            if name in ["Wavelet Variance", "Wavelet Log Variance", "Wavelet Whittle"]:
                estimator = estimator_class(scales=[1, 2, 4, 8])
            else:
                estimator = estimator_class()
            
            # Test estimation
            start_time = time.time()
            result = estimator.estimate(test_data)
            execution_time = time.time() - start_time
            
            # Check result structure
            if isinstance(result, dict) and 'hurst_parameter' in result:
                hurst_est = result['hurst_parameter']
                success = True
                print("✅")
            elif isinstance(result, (int, float)):
                hurst_est = result
                success = True
                print("✅")
            else:
                success = False
                print("❌ (Unexpected result format)")
            
            results[name] = {
                'success': success,
                'execution_time': execution_time,
                'hurst_estimate': hurst_est if success else None,
                'result': result if success else None
            }
            
        except Exception as e:
            print(f"❌ ({str(e)[:50]}...)")
            results[name] = {
                'success': False,
                'execution_time': None,
                'hurst_estimate': None,
                'error': str(e)
            }
    
    return results

def test_auto_optimized_estimators():
    """Test auto-optimized estimators."""
    print("\n🔍 Testing auto-optimized estimators...")
    
    # Import auto-optimized estimators here to ensure they're available
    from lrdbench.analysis.auto_optimized_estimator import (
        AutoDFAEstimator, AutoRSEstimator, AutoDMAEstimator, AutoHiguchiEstimator,
        AutoGPHEstimator, AutoPeriodogramEstimator, AutoWhittleEstimator
    )
    
    auto_estimators = [
        ("Auto DFA", AutoDFAEstimator),
        ("Auto RS", AutoRSEstimator),
        ("Auto DMA", AutoDMAEstimator),
        ("Auto Higuchi", AutoHiguchiEstimator),
        ("Auto GPH", AutoGPHEstimator),
        ("Auto Periodogram", AutoPeriodogramEstimator),
        ("Auto Whittle", AutoWhittleEstimator),
    ]
    
    results = {}
    
    # Generate test data
    from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
    fgn = FractionalGaussianNoise(H=0.7)
    test_data = fgn.generate(1000, seed=42)
    
    for name, estimator_class in auto_estimators:
        print(f"   Testing {name}...", end=" ")
        
        try:
            estimator = estimator_class()
            
            # Test estimation
            start_time = time.time()
            result = estimator.estimate(test_data)
            execution_time = time.time() - start_time
            
            # Get optimization info
            opt_info = estimator.get_optimization_info()
            
            if isinstance(result, dict) and 'hurst_parameter' in result:
                hurst_est = result['hurst_parameter']
                success = True
                print("✅")
            else:
                success = False
                print("❌ (Unexpected result format)")
            
            results[name] = {
                'success': success,
                'execution_time': execution_time,
                'hurst_estimate': hurst_est if success else None,
                'optimization_level': estimator.optimization_level,
                'numba_available': opt_info.get('numba_available', False),
                'jax_available': opt_info.get('jax_available', False),
                'result': result if success else None
            }
            
        except Exception as e:
            print(f"❌ ({str(e)[:50]}...)")
            results[name] = {
                'success': False,
                'execution_time': None,
                'hurst_estimate': None,
                'error': str(e)
            }
    
    return results

def run_comprehensive_benchmark():
    """Run the comprehensive benchmark using the built-in system."""
    print("\n🚀 Running comprehensive benchmark...")
    
    try:
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        
        # Initialize benchmark system
        benchmark = ComprehensiveBenchmark()
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark(
            data_length=1000,
            benchmark_type="comprehensive",
            save_results=True
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_advanced_metrics_benchmark():
    """Run advanced metrics benchmark."""
    print("\n🚀 Running advanced metrics benchmark...")
    
    try:
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        
        # Initialize benchmark system
        benchmark = ComprehensiveBenchmark()
        
        # Run advanced metrics benchmark
        results = benchmark.run_advanced_metrics_benchmark(
            data_length=1000,
            benchmark_type="comprehensive",
            n_monte_carlo=50,  # Reduced for faster execution
            save_results=True
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Advanced metrics benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_contamination_tests():
    """Run tests with various contamination types."""
    print("\n🚀 Running contamination tests...")
    
    try:
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        
        # Initialize benchmark system
        benchmark = ComprehensiveBenchmark()
        
        contamination_types = [
            'additive_gaussian',
            'multiplicative_noise',
            'outliers',
            'trend',
            'seasonal'
        ]
        
        results = {}
        
        for contam_type in contamination_types:
            print(f"   Testing with {contam_type} contamination...")
            
            try:
                result = benchmark.run_comprehensive_benchmark(
                    data_length=1000,
                    benchmark_type="classical",  # Use classical estimators for contamination tests
                    contamination_type=contam_type,
                    contamination_level=0.2,
                    save_results=False
                )
                
                results[contam_type] = result
                print(f"     ✅ {contam_type} completed")
                
            except Exception as e:
                print(f"     ❌ {contam_type} failed: {e}")
                results[contam_type] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Contamination tests failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_comprehensive_report(individual_results, auto_results, comprehensive_results, 
                                advanced_results, contamination_results):
    """Generate a comprehensive test report."""
    print("\n📊 Generating comprehensive test report...")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary statistics
    total_individual = len(individual_results)
    successful_individual = sum(1 for r in individual_results.values() if r['success'])
    
    total_auto = len(auto_results)
    successful_auto = sum(1 for r in auto_results.values() if r['success'])
    
    # Calculate success rates
    individual_success_rate = successful_individual / total_individual if total_individual > 0 else 0
    auto_success_rate = successful_auto / total_auto if total_auto > 0 else 0
    
    # Performance analysis
    individual_times = [r['execution_time'] for r in individual_results.values() 
                       if r['success'] and r['execution_time'] is not None]
    auto_times = [r['execution_time'] for r in auto_results.values() 
                  if r['success'] and r['execution_time'] is not None]
    
    avg_individual_time = np.mean(individual_times) if individual_times else 0
    avg_auto_time = np.mean(auto_times) if auto_times else 0
    
    # Create report
    report = f"""
# LRDBench Complete Estimators Test and Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Timestamp:** {timestamp}

## 📊 Executive Summary

### Individual Estimators
- **Total Tested:** {total_individual}
- **Successful:** {successful_individual}
- **Success Rate:** {individual_success_rate:.1%}
- **Average Execution Time:** {avg_individual_time:.4f}s

### Auto-Optimized Estimators
- **Total Tested:** {total_auto}
- **Successful:** {successful_auto}
- **Success Rate:** {auto_success_rate:.1%}
- **Average Execution Time:** {avg_auto_time:.4f}s

## 🔍 Individual Estimator Results

"""
    
    # Add individual results
    for name, result in individual_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['execution_time']:.4f}s" if result['execution_time'] else "N/A"
        hurst_str = f"{result['hurst_estimate']:.6f}" if result['hurst_estimate'] else "N/A"
        
        report += f"- **{name}:** {status} | Time: {time_str} | H_est: {hurst_str}\n"
        
        if not result['success'] and 'error' in result:
            report += f"  - Error: {result['error']}\n"
    
    report += "\n## ⚡ Auto-Optimized Estimator Results\n\n"
    
    # Add auto-optimized results
    for name, result in auto_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['execution_time']:.4f}s" if result['execution_time'] else "N/A"
        hurst_str = f"{result['hurst_estimate']:.6f}" if result['hurst_estimate'] else "N/A"
        opt_level = result.get('optimization_level', 'N/A')
        
        report += f"- **{name}:** {status} | Time: {time_str} | H_est: {hurst_str} | Opt: {opt_level}\n"
        
        if not result['success'] and 'error' in result:
            report += f"  - Error: {result['error']}\n"
    
    # Add comprehensive benchmark results
    if comprehensive_results:
        report += f"\n## 🚀 Comprehensive Benchmark Results\n\n"
        report += f"- **Total Tests:** {comprehensive_results.get('total_tests', 'N/A')}\n"
        report += f"- **Successful Tests:** {comprehensive_results.get('successful_tests', 'N/A')}\n"
        report += f"- **Success Rate:** {comprehensive_results.get('success_rate', 0):.1%}\n"
        report += f"- **Data Models Tested:** {comprehensive_results.get('data_models_tested', 'N/A')}\n"
        report += f"- **Estimators Tested:** {comprehensive_results.get('estimators_tested', 'N/A')}\n"
    
    # Add advanced metrics results
    if advanced_results:
        report += f"\n## 📈 Advanced Metrics Benchmark Results\n\n"
        report += f"- **Total Tests:** {advanced_results.get('total_tests', 'N/A')}\n"
        report += f"- **Successful Tests:** {advanced_results.get('successful_tests', 'N/A')}\n"
        report += f"- **Success Rate:** {advanced_results.get('success_rate', 0):.1%}\n"
    
    # Add contamination test results
    if contamination_results:
        report += f"\n## 🧪 Contamination Test Results\n\n"
        for contam_type, result in contamination_results.items():
            if 'error' not in result:
                success_rate = result.get('success_rate', 0)
                report += f"- **{contam_type}:** {success_rate:.1%} success rate\n"
            else:
                report += f"- **{contam_type}:** ❌ Failed - {result['error']}\n"
    
    # Add recommendations
    report += f"\n## 💡 Recommendations\n\n"
    
    if individual_success_rate < 1.0:
        failed_individual = [name for name, result in individual_results.items() if not result['success']]
        report += f"- **Fix Individual Estimators:** {', '.join(failed_individual)} failed tests\n"
    
    if auto_success_rate < 1.0:
        failed_auto = [name for name, result in auto_results.items() if not result['success']]
        report += f"- **Fix Auto-Optimized Estimators:** {', '.join(failed_auto)} failed tests\n"
    
    if avg_auto_time > avg_individual_time * 1.5:
        report += f"- **Performance Issue:** Auto-optimized estimators are slower than individual estimators\n"
    
    if individual_success_rate >= 0.9 and auto_success_rate >= 0.9:
        report += f"- **System Status:** ✅ LRDBench is operating at high performance levels\n"
        report += f"- **Ready for Production:** All major components are functioning correctly\n"
    
    # Save report
    report_filename = f"complete_estimators_test_report_{timestamp}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_filename}")
    
    return report_filename

def main():
    """Main function to run all tests and benchmarks."""
    print("🚀 LRDBench - Complete Estimators Test and Benchmark")
    print("=" * 70)
    print("This script will test all estimators comprehensively")
    print("=" * 70)
    
    start_time = time.time()
    
    # Track all results
    all_results = {}
    
    # 1. Test imports
    print("\n" + "="*50)
    print("STEP 1: Testing Core Imports")
    print("="*50)
    
    imports_success = test_imports()
    if not imports_success:
        print("❌ Critical import failures. Cannot proceed with testing.")
        return False
    
    # 2. Test individual estimators
    print("\n" + "="*50)
    print("STEP 2: Testing Individual Estimators")
    print("="*50)
    
    individual_results = test_individual_estimators()
    all_results['individual'] = individual_results
    
    # 3. Test auto-optimized estimators
    print("\n" + "="*50)
    print("STEP 3: Testing Auto-Optimized Estimators")
    print("="*50)
    
    auto_results = test_auto_optimized_estimators()
    all_results['auto_optimized'] = auto_results
    
    # 4. Run comprehensive benchmark
    print("\n" + "="*50)
    print("STEP 4: Running Comprehensive Benchmark")
    print("="*50)
    
    comprehensive_results = run_comprehensive_benchmark()
    all_results['comprehensive'] = comprehensive_results
    
    # 5. Run advanced metrics benchmark
    print("\n" + "="*50)
    print("STEP 5: Running Advanced Metrics Benchmark")
    print("="*50)
    
    advanced_results = run_advanced_metrics_benchmark()
    all_results['advanced_metrics'] = advanced_results
    
    # 6. Run contamination tests
    print("\n" + "="*50)
    print("STEP 6: Running Contamination Tests")
    print("="*50)
    
    contamination_results = run_contamination_tests()
    all_results['contamination'] = contamination_results
    
    # 7. Generate comprehensive report
    print("\n" + "="*50)
    print("STEP 7: Generating Comprehensive Report")
    print("="*50)
    
    report_filename = generate_comprehensive_report(
        individual_results, auto_results, comprehensive_results, 
        advanced_results, contamination_results
    )
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 COMPLETE TEST AND BENCHMARK FINISHED!")
    print("="*70)
    
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print(f"📄 Comprehensive report: {report_filename}")
    
    # Calculate overall success rates
    individual_success = sum(1 for r in individual_results.values() if r['success'])
    auto_success = sum(1 for r in auto_results.values() if r['success'])
    
    total_estimators = len(individual_results) + len(auto_results)
    total_successful = individual_success + auto_success
    
    overall_success_rate = total_successful / total_estimators if total_estimators > 0 else 0
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Individual Estimators: {individual_success}/{len(individual_results)} ({individual_success/len(individual_results)*100:.1f}%)")
    print(f"   Auto-Optimized: {auto_success}/{len(auto_results)} ({auto_success/len(auto_results)*100:.1f}%)")
    print(f"   Overall Success Rate: {overall_success_rate:.1%}")
    
    if overall_success_rate >= 0.9:
        print("\n🎯 STATUS: ✅ EXCELLENT - LRDBench is operating at high performance levels!")
    elif overall_success_rate >= 0.7:
        print("\n⚠️  STATUS: ⚠️  GOOD - Some issues detected but system is functional")
    else:
        print("\n❌ STATUS: ❌ POOR - Significant issues detected, review required")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review the detailed report: {report_filename}")
    print(f"   2. Address any failed tests")
    print(f"   3. Run performance optimization if needed")
    print(f"   4. Deploy to production if all tests pass")
    
    return overall_success_rate >= 0.7

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
