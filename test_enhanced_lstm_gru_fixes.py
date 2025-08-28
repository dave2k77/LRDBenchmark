#!/usr/bin/env python3
"""
Test script for enhanced LSTM and GRU estimators with improvements.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion

def test_enhanced_models():
    """Test the enhanced LSTM and GRU models with improvements."""
    
    print("🧪 Testing Enhanced LSTM and GRU Models with Improvements")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating test data...")
    test_h_values = [0.3, 0.5, 0.7]
    sequence_length = 1000
    
    test_data = {}
    for h in test_h_values:
        # Generate FGN data
        fgn_model = FractionalGaussianNoise(H=h)
        fgn_data = fgn_model.generate(sequence_length)
        test_data[f"FGN_H{h}"] = fgn_data
        
        # Generate FBM data
        fbm_model = FractionalBrownianMotion(H=h)
        fbm_data = fbm_model.generate(sequence_length)
        test_data[f"FBM_H{h}"] = fbm_data
    
    # Test Enhanced LSTM
    print("\n🔬 Testing Enhanced LSTM Estimator...")
    lstm_estimator = EnhancedLSTMEstimator()
    
    for data_name, data in test_data.items():
        try:
            result = lstm_estimator.estimate(data)
            print(f"  {data_name}: H = {result['hurst_parameter']:.4f} "
                  f"({result['method']})")
        except Exception as e:
            print(f"  {data_name}: Error - {e}")
    
    # Test Enhanced GRU
    print("\n🔬 Testing Enhanced GRU Estimator...")
    gru_estimator = EnhancedGRUEstimator()
    
    for data_name, data in test_data.items():
        try:
            result = gru_estimator.estimate(data)
            print(f"  {data_name}: H = {result['hurst_parameter']:.4f} "
                  f"({result['method']})")
        except Exception as e:
            print(f"  {data_name}: Error - {e}")
    
    # Test model info
    print("\n📋 Model Information:")
    print("Enhanced LSTM:")
    lstm_info = lstm_estimator.get_model_info()
    for key, value in lstm_info.items():
        if key in ['model_type', 'architecture', 'hidden_size', 'num_layers', 'bidirectional', 'use_attention']:
            print(f"  {key}: {value}")
    
    print("\nEnhanced GRU:")
    gru_info = gru_estimator.get_model_info()
    for key, value in gru_info.items():
        if key in ['model_type', 'architecture', 'hidden_size', 'num_layers', 'bidirectional', 'use_attention']:
            print(f"  {key}: {value}")
    
    print("\n✅ Enhanced LSTM and GRU testing completed!")

if __name__ == "__main__":
    test_enhanced_models()
