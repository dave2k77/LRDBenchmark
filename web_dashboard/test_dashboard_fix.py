#!/usr/bin/env python3
"""
Test script to verify dashboard components work correctly
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_estimators():
    """Test all estimators to ensure they work correctly."""
    print("🧪 Testing all estimators...")
    
    # Generate test data
    data = np.random.randn(1000)
    true_H = 0.7
    
    estimators_to_test = {
        "DFA": "lrdbench.analysis.temporal.dfa.dfa_estimator.DFAEstimator",
        "RS": "lrdbench.analysis.temporal.rs.rs_estimator.RSEstimator", 
        "DMA": "lrdbench.analysis.temporal.dma.dma_estimator.DMAEstimator",
        "Higuchi": "lrdbench.analysis.temporal.higuchi.higuchi_estimator.HiguchiEstimator",
        "GPH": "lrdbench.analysis.spectral.gph.gph_estimator.GPHEstimator",
        "Periodogram": "lrdbench.analysis.spectral.periodogram.periodogram_estimator.PeriodogramEstimator",
        "Whittle": "lrdbench.analysis.spectral.whittle.whittle_estimator.WhittleEstimator"
    }
    
    results = {}
    
    for name, import_path in estimators_to_test.items():
        try:
            # Import the estimator
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            estimator_class = getattr(module, class_name)
            
            # Create and run estimator
            estimator = estimator_class()
            result = estimator.estimate(data)
            
            # Check if hurst_parameter is in result
            hurst_est = result.get('hurst_parameter', None)
            if hurst_est is not None:
                error = abs(hurst_est - true_H)
                results[name] = {
                    'success': True,
                    'estimated_hurst': hurst_est,
                    'true_hurst': true_H,
                    'error': error,
                    'result_keys': list(result.keys())
                }
                print(f"✅ {name}: H={hurst_est:.3f}, Error={error:.3f}")
            else:
                results[name] = {
                    'success': False,
                    'error_message': 'No Hurst parameter found',
                    'result_keys': list(result.keys())
                }
                print(f"❌ {name}: No Hurst parameter found")
                
        except Exception as e:
            results[name] = {
                'success': False,
                'error_message': str(e)
            }
            print(f"❌ {name}: {str(e)}")
    
    return results

def test_auto_optimization():
    """Test auto-optimization system."""
    print("\n🧪 Testing auto-optimization system...")
    
    try:
        from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator
        print("✅ AutoOptimizedEstimator imported successfully")
        
        # Test with DFA
        auto_dfa = AutoOptimizedEstimator("dfa")
        data = np.random.randn(1000)
        result = auto_dfa.estimate(data)
        
        hurst_est = result.get('hurst_parameter', None)
        if hurst_est is not None:
            print(f"✅ Auto-optimized DFA: H={hurst_est:.3f}")
            return True
        else:
            print(f"❌ Auto-optimized DFA: No Hurst parameter found")
            return False
            
    except Exception as e:
        print(f"❌ Auto-optimization failed: {str(e)}")
        return False

def test_data_generation():
    """Test data generation models."""
    print("\n🧪 Testing data generation...")
    
    try:
        from lrdbench import FBMModel, FGNModel, ARFIMAModel, MRWModel
        
        # Test FBM
        fbm = FBMModel(H=0.7, sigma=1.0)
        fbm_data = fbm.generate(1000, seed=42)
        print(f"✅ FBM data generated: {len(fbm_data)} points")
        
        # Test FGN
        fgn = FGNModel(H=0.6, sigma=1.0)
        fgn_data = fgn.generate(1000, seed=42)
        print(f"✅ FGN data generated: {len(fgn_data)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 LRDBenchmark Dashboard Component Test")
    print("=" * 50)
    
    # Test data generation
    data_gen_ok = test_data_generation()
    
    # Test estimators
    estimator_results = test_estimators()
    
    # Test auto-optimization
    auto_opt_ok = test_auto_optimization()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    successful_estimators = sum(1 for r in estimator_results.values() if r['success'])
    total_estimators = len(estimator_results)
    
    print(f"Data Generation: {'✅ PASS' if data_gen_ok else '❌ FAIL'}")
    print(f"Estimators: {successful_estimators}/{total_estimators} successful")
    print(f"Auto-Optimization: {'✅ PASS' if auto_opt_ok else '❌ FAIL'}")
    
    if successful_estimators == total_estimators and data_gen_ok and auto_opt_ok:
        print("\n🎉 All tests passed! Dashboard should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")
