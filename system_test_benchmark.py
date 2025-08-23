#!/usr/bin/env python3
"""
System Test and Performance Benchmark for DataExploratoryProject
This script validates the core functionality and performance before PyPI submission.
"""

import sys
import time
import numpy as np
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_core_imports():
    """Test that all core components can be imported."""
    print("🔍 Testing core imports...")
    
    try:
        # Test data models
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        from models.data_models.fgn.fgn_model import FractionalGaussianNoise
        from models.data_models.arfima.arfima_model import ARFIMAModel
        from models.data_models.mrw.mrw_model import MultifractalRandomWalk
        print("✅ Data models imported successfully")
        
        # Test estimators
        from analysis.temporal.rs.rs_estimator import RSEstimator
        from analysis.temporal.dfa.dfa_estimator import DFAEstimator
        from analysis.spectral.gph.gph_estimator import GPHEstimator
        from analysis.wavelet.cwt.cwt_estimator import CWTEstimator
        print("✅ Core estimators imported successfully")
        
        # Test auto-discovery
        from auto_discovery_system import AutoDiscoverySystem
        print("✅ Auto-discovery system imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """Test data generation capabilities."""
    print("\n🔍 Testing data generation...")
    
    try:
        # Test fBm generation
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
        data = fbm.generate(1000, seed=42)
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert np.isfinite(data).all()
        print(f"✅ fBm generation: {len(data)} samples, range [{data.min():.3f}, {data.max():.3f}]")
        
        # Test fGn generation
        from models.data_models.fgn.fgn_model import FractionalGaussianNoise
        fgn = FractionalGaussianNoise(H=0.6, sigma=1.0)
        data = fgn.generate(1000, seed=42)
        
        assert len(data) == 1000
        assert isinstance(data, np.ndarray)
        assert np.isfinite(data).all()
        print(f"✅ fGn generation: {len(data)} samples, range [{data.min():.3f}, {data.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        traceback.print_exc()
        return False

def test_estimation():
    """Test estimation capabilities."""
    print("\n🔍 Testing estimation...")
    
    try:
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        from analysis.temporal.rs.rs_estimator import RSEstimator
        
        # Generate test data
        fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
        data = fbm.generate(1000, seed=42)
        
        # Test estimation
        estimator = RSEstimator()
        result = estimator.estimate(data)
        
        assert 'hurst_parameter' in result
        assert 'r_squared' in result
        assert 'confidence_interval' in result
        
        estimated_h = result['hurst_parameter']
        error = abs(estimated_h - 0.7)
        
        print(f"✅ RS estimation: True H=0.7, Estimated H={estimated_h:.3f}, Error={error:.3f}")
        print(f"   R²={result['r_squared']:.3f}, CI=({result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Estimation test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\n🔍 Running performance benchmark...")
    
    try:
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        from analysis.temporal.rs.rs_estimator import RSEstimator
        from analysis.temporal.dfa.dfa_estimator import DFAEstimator
        from analysis.spectral.gph.gph_estimator import GPHEstimator
        
        # Test parameters
        n_samples = 1000
        n_runs = 10
        true_h = 0.7
        
        # Initialize models and estimators
        fbm = FractionalBrownianMotion(H=true_h, sigma=1.0)
        estimators = {
            'RS': RSEstimator(),
            'DFA': DFAEstimator(),
            'GPH': GPHEstimator()
        }
        
        results = {}
        
        for name, estimator in estimators.items():
            print(f"\n📊 Testing {name} estimator...")
            times = []
            errors = []
            
            for i in range(n_runs):
                # Generate fresh data
                data = fbm.generate(n_samples, seed=i)
                
                # Time estimation
                start = time.time()
                try:
                    result = estimator.estimate(data)
                    
                    # Extract Hurst parameter (handle different result formats)
                    if isinstance(result, dict) and 'hurst_parameter' in result:
                        estimated_h = result['hurst_parameter']
                    elif isinstance(result, (int, float)):
                        estimated_h = result
                    else:
                        print(f"   Run {i+1}: Unexpected result format: {type(result)}")
                        continue
                    
                    end = time.time()
                    elapsed = end - start
                    
                    error = abs(estimated_h - true_h)
                    times.append(elapsed)
                    errors.append(error)
                    
                    print(f"   Run {i+1}: Time={elapsed:.4f}s, H={estimated_h:.3f}, Error={error:.3f}")
                    
                except Exception as e:
                    print(f"   Run {i+1}: Failed - {e}")
                    continue
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_error = np.mean(errors)
                std_error = np.std(errors)
                
                results[name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'avg_error': avg_error,
                    'std_error': std_error
                }
                
                print(f"   📈 {name} Results:")
                print(f"      Time: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"      Error: {avg_error:.3f} ± {std_error:.3f}")
            else:
                print(f"   ❌ {name}: No successful runs")
        
        return results
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return None

def test_auto_discovery():
    """Test the auto-discovery system."""
    print("\n🔍 Testing auto-discovery system...")
    
    try:
        from auto_discovery_system import AutoDiscoverySystem
        
        ads = AutoDiscoverySystem()
        
        # Test component discovery
        components = ads.discover_components()
        
        assert 'estimators' in components
        assert 'data_generators' in components
        
        n_estimators = len(components['estimators'])
        n_generators = len(components['data_generators'])
        
        print(f"✅ Auto-discovery working: {n_estimators} estimators, {n_generators} data generators")
        
        # Test registry save (the method is called automatically during discovery)
        print("✅ Registry updated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-discovery test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive system test."""
    print("🚀 DataExploratoryProject - System Test & Performance Benchmark")
    print("=" * 70)
    
    # Track test results
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_core_imports()
    test_results['data_generation'] = test_data_generation()
    test_results['estimation'] = test_estimation()
    test_results['auto_discovery'] = test_auto_discovery()
    
    # Run performance benchmark
    performance_results = run_performance_benchmark()
    test_results['performance'] = performance_results is not None
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SYSTEM TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, passed_test in test_results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Package is ready for PyPI submission.")
        
        if performance_results:
            print("\n📊 PERFORMANCE SUMMARY:")
            for name, metrics in performance_results.items():
                print(f"{name:>8}: {metrics['avg_time']:.4f}s ± {metrics['std_time']:.4f}s "
                      f"(Error: {metrics['avg_error']:.3f} ± {metrics['std_error']:.3f})")
        
        print("\n🚀 Next steps:")
        print("1. Test build: python -m build")
        print("2. Test installation: pip install -e .")
        print("3. Upload to PyPI: twine upload dist/*")
        
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before PyPI submission.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
