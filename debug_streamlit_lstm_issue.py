#!/usr/bin/env python3
"""
Debug script to test LSTM/GRU estimators in a simulated Streamlit environment
"""

import sys
import os
import numpy as np

# Simulate running from web_dashboard directory
os.chdir('web_dashboard')
sys.path.insert(0, '..')

try:
    from lrdbench.analysis.machine_learning.lstm_estimator import LSTMEstimator
    from lrdbench.analysis.machine_learning.gru_estimator import GRUEstimator
    from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
    
    print("✅ Imports successful")
    
    # Test data generation - use 1000 points like in the error
    fgn = FractionalGaussianNoise(H=0.7)
    test_data = fgn.generate(1000, seed=42)
    print(f"✅ Generated test data: {test_data.shape}")
    
    # Test LSTM with detailed debugging
    print(f"\n🔍 Testing LSTM Estimator with detailed debugging:")
    lstm = LSTMEstimator()
    
    # Check initial state
    print(f"Initial is_trained: {lstm.is_trained}")
    print(f"Initial _torch_model: {lstm._torch_model is not None}")
    print(f"Initial scaler: {lstm.scaler}")
    
    # Try to load pretrained model
    success = lstm._try_load_pretrained_model()
    print(f"Pretrained model load success: {success}")
    
    if success:
        print(f"After loading - is_trained: {lstm.is_trained}")
        print(f"After loading - _torch_model: {lstm._torch_model is not None}")
        print(f"After loading - scaler: {lstm.scaler}")
        
        if lstm.scaler:
            print(f"Scaler n_features_in_: {lstm.scaler.n_features_in_}")
            print(f"Scaler feature_names_in_: {getattr(lstm.scaler, 'feature_names_in_', 'Not available')}")
        
        # Test estimation with detailed error handling
        try:
            print(f"\n🔍 Attempting estimation with data shape: {test_data.shape}")
            result = lstm.estimate(test_data)
            print(f"✅ LSTM estimation successful: H = {result['hurst_parameter']:.3f}")
        except Exception as e:
            print(f"❌ LSTM estimation failed: {str(e)}")
            if "StandardScaler is expecting" in str(e):
                print("🔍 This is the StandardScaler dimension mismatch error!")
                print("🔍 Let's check what's happening in _prepare_sequences...")
                
                # Try to debug the _prepare_sequences method
                try:
                    # This should trigger the error
                    X_seq = lstm._prepare_sequences(test_data, fit_scaler=False)
                    print(f"✅ _prepare_sequences worked: {X_seq.shape}")
                except Exception as seq_error:
                    print(f"❌ _prepare_sequences failed: {str(seq_error)}")
                    
            import traceback
            traceback.print_exc()
    else:
        print("❌ LSTM pretrained model not loaded")
    
    # Test GRU with similar debugging
    print(f"\n🔍 Testing GRU Estimator with detailed debugging:")
    gru = GRUEstimator()
    
    # Check initial state
    print(f"Initial is_trained: {gru.is_trained}")
    print(f"Initial _torch_model: {gru._torch_model is not None}")
    print(f"Initial scaler: {gru.scaler}")
    
    # Try to load pretrained model
    success = gru._try_load_pretrained_model()
    print(f"Pretrained model load success: {success}")
    
    if success:
        print(f"After loading - is_trained: {gru.is_trained}")
        print(f"After loading - _torch_model: {gru._torch_model is not None}")
        print(f"After loading - scaler: {gru.scaler}")
        
        if gru.scaler:
            print(f"Scaler n_features_in_: {gru.scaler.n_features_in_}")
            print(f"Scaler feature_names_in_: {getattr(gru.scaler, 'feature_names_in_', 'Not available')}")
        
        # Test estimation with detailed error handling
        try:
            print(f"\n🔍 Attempting estimation with data shape: {test_data.shape}")
            result = gru.estimate(test_data)
            print(f"✅ GRU estimation successful: H = {result['hurst_parameter']:.3f}")
        except Exception as e:
            print(f"❌ GRU estimation failed: {str(e)}")
            if "StandardScaler is expecting" in str(e):
                print("🔍 This is the StandardScaler dimension mismatch error!")
                print("🔍 Let's check what's happening in _prepare_sequences...")
                
                # Try to debug the _prepare_sequences method
                try:
                    # This should trigger the error
                    X_seq = gru._prepare_sequences(test_data, fit_scaler=False)
                    print(f"✅ _prepare_sequences worked: {X_seq.shape}")
                except Exception as seq_error:
                    print(f"❌ _prepare_sequences failed: {str(seq_error)}")
                    
            import traceback
            traceback.print_exc()
    else:
        print("❌ GRU pretrained model not loaded")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
