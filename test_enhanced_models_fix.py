#!/usr/bin/env python3
"""
Quick test to verify that enhanced neural network estimators 
are loading their PyTorch models correctly.
"""

import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced estimators
from lrdbench.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
from lrdbench.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator

def test_enhanced_estimators():
    """Test that enhanced estimators load their PyTorch models correctly."""
    
    print("🧪 Testing Enhanced Neural Network Estimators")
    print("=" * 50)
    
    # Create test data
    test_data = np.random.randn(1000)
    
    # Test each enhanced estimator
    estimators = {
        'EnhancedCNN': EnhancedCNNEstimator(),
        'EnhancedLSTM': EnhancedLSTMEstimator(),
        'EnhancedGRU': EnhancedGRUEstimator(),
        'EnhancedTransformer': EnhancedTransformerEstimator()
    }
    
    for name, estimator in estimators.items():
        print(f"\n📊 Testing {name}...")
        
        try:
            # Try to estimate
            result = estimator.estimate(test_data)
            
            # Check if we got a PyTorch model
            if hasattr(estimator.model, 'forward') and callable(getattr(estimator.model, 'forward', None)):
                print(f"  ✅ {name} loaded PyTorch model successfully!")
                print(f"  📈 Estimated H: {result['hurst_parameter']:.4f}")
                print(f"  🔧 Method: {result['method']}")
            else:
                print(f"  ⚠️ {name} loaded scikit-learn model (fallback)")
                print(f"  📈 Estimated H: {result['hurst_parameter']:.4f}")
                print(f"  🔧 Method: {result['method']}")
                
        except Exception as e:
            print(f"  ❌ {name} failed: {str(e)}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_enhanced_estimators()
