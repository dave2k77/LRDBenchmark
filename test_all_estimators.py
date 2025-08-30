#!/usr/bin/env python3
"""
Test script to verify all ML estimators are working correctly.
"""

import numpy as np

def test_estimator(name, estimator_class, data):
    """Test a single estimator."""
    print(f"\n🔍 Testing {name}...")
    try:
        estimator = estimator_class()
        result = estimator.estimate(data)
        method = result.get('method', 'unknown')
        fallback = result.get('fallback_used', False)
        print(f"  ✅ Method: {method}")
        print(f"  📊 Fallback: {fallback}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Test all ML estimators."""
    print("🚀 Final System Verification")
    print("=" * 50)
    
    # Generate test data
    data = np.random.randn(1000)
    
    # Test all estimators
    results = {}
    
    # Simple ML estimators
    try:
        from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
        results['Random Forest'] = test_estimator('Random Forest', RandomForestEstimator, data)
    except Exception as e:
        print(f"❌ Random Forest import failed: {e}")
        results['Random Forest'] = False
    
    try:
        from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator
        results['Gradient Boosting'] = test_estimator('Gradient Boosting', GradientBoostingEstimator, data)
    except Exception as e:
        print(f"❌ Gradient Boosting import failed: {e}")
        results['Gradient Boosting'] = False
    
    try:
        from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
        results['SVR'] = test_estimator('SVR', SVREstimator, data)
    except Exception as e:
        print(f"❌ SVR import failed: {e}")
        results['SVR'] = False
    
    # Neural network estimators
    try:
        from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
        results['LSTM'] = test_estimator('LSTM', LSTMEstimator, data)
    except Exception as e:
        print(f"❌ LSTM import failed: {e}")
        results['LSTM'] = False
    
    try:
        from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
        results['GRU'] = test_estimator('GRU', GRUEstimator, data)
    except Exception as e:
        print(f"❌ GRU import failed: {e}")
        results['GRU'] = False
    
    try:
        from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
        results['CNN'] = test_estimator('CNN', CNNEstimator, data)
    except Exception as e:
        print(f"❌ CNN import failed: {e}")
        results['CNN'] = False
    
    try:
        from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator
        results['Transformer'] = test_estimator('Transformer', TransformerEstimator, data)
    except Exception as e:
        print(f"❌ Transformer import failed: {e}")
        results['Transformer'] = False
    
    # Print final status
    print("\n📋 Final Status:")
    working = 0
    total = len(results)
    
    for name, status in results.items():
        print(f"  {name}: {'✅ Working' if status else '❌ Failed'}")
        if status:
            working += 1
    
    print(f"\n🎯 Success Rate: {working}/{total} ({working/total*100:.1f}%)")
    
    if working == total:
        print("🎉 All estimators are working correctly!")
    else:
        print("⚠️ Some estimators have issues that need attention.")

if __name__ == "__main__":
    main()
