#!/usr/bin/env python3
"""
Test Individual ML Estimators

This script tests each ML estimator individually to identify which ones are failing.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_individual_estimator(estimator_class, estimator_name: str):
    """Test a single ML estimator."""
    print(f"\n🧪 Testing {estimator_name}...")
    
    try:
        # Create estimator
        estimator = estimator_class()
        print(f"   ✅ {estimator_name} created successfully")
        
        # Generate simple test data
        from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(H=0.7)
        test_data = fbm.generate(500)
        
        # Try to estimate
        result = estimator.estimate(test_data)
        hurst_estimate = result.get('hurst_parameter', None)
        
        if hurst_estimate is not None:
            print(f"   ✅ {estimator_name} estimation successful: H = {hurst_estimate:.3f}")
            return True
        else:
            print(f"   ❌ {estimator_name} estimation failed: No Hurst parameter")
            return False
            
    except Exception as e:
        print(f"   ❌ {estimator_name} failed: {e}")
        return False

def main():
    """Test all ML estimators individually."""
    print("🧪 Testing Individual ML Estimators")
    print("=" * 50)
    
    # Import ML estimators
    try:
        from lrdbench.analysis.machine_learning.cnn_estimator import CNNEstimator
        from lrdbench.analysis.machine_learning.lstm_estimator import LSTMEstimator
        from lrdbench.analysis.machine_learning.gru_estimator import GRUEstimator
        from lrdbench.analysis.machine_learning.transformer_estimator import TransformerEstimator
        from lrdbench.analysis.machine_learning.svr_estimator import SVREstimator
        from lrdbench.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
        from lrdbench.analysis.machine_learning.neural_network_estimator import NeuralNetworkEstimator
        from lrdbench.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
        
        print("✅ All ML estimators imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing ML estimators: {e}")
        return
    
    # Test each estimator
    estimators = [
        (RandomForestEstimator, "RandomForest"),
        (SVREstimator, "SVR"),
        (NeuralNetworkEstimator, "NeuralNetwork"),
        (GradientBoostingEstimator, "GradientBoosting"),
        (CNNEstimator, "CNN"),
        (LSTMEstimator, "LSTM"),
        (GRUEstimator, "GRU"),
        (TransformerEstimator, "Transformer"),
    ]
    
    successful = []
    failed = []
    
    for estimator_class, estimator_name in estimators:
        success = test_individual_estimator(estimator_class, estimator_name)
        if success:
            successful.append(estimator_name)
        else:
            failed.append(estimator_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {len(successful)}/{len(estimators)}")
    print(f"❌ Failed: {len(failed)}/{len(estimators)}")
    
    if successful:
        print(f"\n✅ Working estimators:")
        for name in successful:
            print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed estimators:")
        for name in failed:
            print(f"   - {name}")

if __name__ == "__main__":
    main()
