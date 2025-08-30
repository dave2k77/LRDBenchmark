#!/usr/bin/env python3
"""
Train Machine Learning Estimators for Long-Range Dependence Analysis

This script demonstrates the development vs production workflow:
- Development: Train models with synthetic data
- Production: Use pretrained models for estimation

Features:
- Synthetic data generation with known Hurst parameters
- Training of all ML estimators (Random Forest, Gradient Boosting, SVR, LSTM, GRU, CNN, Transformer)
- Model persistence and loading
- Performance comparison
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# GPU memory management
try:
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        # Set memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
except ImportError:
    print("⚠️ PyTorch not available")

def generate_synthetic_data(
    n_samples: int = 1000,
    min_hurst: float = 0.1,
    max_hurst: float = 0.9,
    data_length: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with known Hurst parameters.
    
    Parameters
    ----------
    n_samples : int
        Number of time series to generate
    min_hurst : float
        Minimum Hurst parameter value
    max_hurst : float
        Maximum Hurst parameter value
    data_length : int
        Length of each time series
        
    Returns
    -------
    tuple
        (X, y) where X is the time series data and y is the Hurst parameters
    """
    print(f"🔧 Generating {n_samples} synthetic time series with length {data_length}...")
    
    # Generate Hurst parameters
    hurst_params = np.random.uniform(min_hurst, max_hurst, n_samples)
    
    # Generate time series data
    X = []
    y = []
    
    for i, hurst in enumerate(hurst_params):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # Generate fractional Brownian motion
        # This is a simplified version - in practice, use proper FBM generation
        t = np.linspace(0, 1, data_length)
        
        # Generate noise with appropriate correlation structure
        if hurst > 0.5:
            # Persistent (long memory)
            noise = np.random.randn(data_length)
            # Apply moving average to create persistence
            window_size = int(data_length * hurst * 0.1)
            if window_size > 1:
                noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
        elif hurst < 0.5:
            # Anti-persistent (short memory)
            noise = np.random.randn(data_length)
            # Apply differencing to create anti-persistence
            noise = np.diff(noise, prepend=noise[0])
        else:
            # No memory (random walk)
            noise = np.random.randn(data_length)
        
        # Scale by Hurst parameter
        scaled_noise = noise * (hurst ** 0.5)
        
        # Create cumulative sum to get time series
        time_series = np.cumsum(scaled_noise)
        
        X.append(time_series)
        y.append(hurst)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Generated {len(X)} time series with Hurst parameters in [{min_hurst:.2f}, {max_hurst:.2f}]")
    return X, y

def train_simple_ml_estimators(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train the simple ML estimators (Random Forest, Gradient Boosting, SVR).
    
    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Target Hurst parameters
        
    Returns
    -------
    dict
        Training results for each estimator
    """
    print("\n🚀 Training Simple ML Estimators...")
    
    results = {}
    
    # Random Forest
    try:
        print("  🌳 Training Random Forest...")
        from lrdbenchmark.analysis.machine_learning.random_forest_estimator_unified import RandomForestEstimator
        
        rf_estimator = RandomForestEstimator(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        start_time = time.time()
        rf_results = rf_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['random_forest'] = {
            'results': rf_results,
            'training_time': training_time,
            'estimator': rf_estimator
        }
        
        print(f"    ✅ Random Forest trained in {training_time:.2f}s")
        print(f"    📊 Test R²: {rf_results.get('test_r2', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"    ❌ Random Forest training failed: {e}")
        results['random_forest'] = {'error': str(e)}
    
    # Gradient Boosting
    try:
        print("  🌱 Training Gradient Boosting...")
        from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator_unified import GradientBoostingEstimator
        
        gb_estimator = GradientBoostingEstimator(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        start_time = time.time()
        gb_results = gb_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['gradient_boosting'] = {
            'results': gb_results,
            'training_time': training_time,
            'estimator': gb_estimator
        }
        
        print(f"    ✅ Gradient Boosting trained in {training_time:.2f}s")
        print(f"    📊 Test R²: {gb_results.get('test_r2', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"    ❌ Gradient Boosting training failed: {e}")
        results['gradient_boosting'] = {'error': str(e)}
    
    # SVR
    try:
        print("  🔧 Training SVR...")
        from lrdbenchmark.analysis.machine_learning.svr_estimator_unified import SVREstimator
        
        svr_estimator = SVREstimator(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            random_state=42
        )
        
        start_time = time.time()
        svr_results = svr_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['svr'] = {
            'results': svr_results,
            'training_time': training_time,
            'estimator': svr_estimator
        }
        
        print(f"    ✅ SVR trained in {training_time:.2f}s")
        print(f"    📊 Test R²: {svr_results.get('test_r2', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"    ❌ SVR training failed: {e}")
        results['svr'] = {'error': str(e)}
    
    return results

def train_neural_network_estimators(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train the neural network estimators (LSTM, GRU, CNN, Transformer).
    
    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Target Hurst parameters
        
    Returns
    -------
    dict
        Training results for each estimator
    """
    print("\n🧠 Training Neural Network Estimators...")
    
    results = {}
    
    # LSTM
    try:
        print("  🔄 Training LSTM...")
        from lrdbenchmark.analysis.machine_learning.lstm_estimator_unified import LSTMEstimator
        
        lstm_estimator = LSTMEstimator(
            hidden_size=32,  # Further reduced for GPU memory
            num_layers=2,
            learning_rate=0.001,
            epochs=50,  # Reduced for demo
            batch_size=8,   # Further reduced for GPU memory
            use_gradient_checkpointing=True
        )
        
        start_time = time.time()
        lstm_results = lstm_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['lstm'] = {
            'results': lstm_results,
            'training_time': training_time,
            'estimator': lstm_estimator
        }
        
        print(f"    ✅ LSTM trained in {training_time:.2f}s")
        
    except Exception as e:
        print(f"    ❌ LSTM training failed: {e}")
        results['lstm'] = {'error': str(e)}
    
    # GRU
    try:
        print("  🔄 Training GRU...")
        from lrdbenchmark.analysis.machine_learning.gru_estimator_unified import GRUEstimator
        
        gru_estimator = GRUEstimator(
            hidden_size=32,  # Further reduced for GPU memory
            num_layers=2,
            learning_rate=0.001,
            epochs=50,  # Reduced for demo
            batch_size=8,   # Further reduced for GPU memory
            use_gradient_checkpointing=True
        )
        
        start_time = time.time()
        gru_results = gru_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['gru'] = {
            'results': gru_results,
            'training_time': training_time,
            'estimator': gru_estimator
        }
        
        print(f"    ✅ GRU trained in {training_time:.2f}s")
        
    except Exception as e:
        print(f"    ❌ GRU training failed: {e}")
        results['gru'] = {'error': str(e)}
    
    # CNN
    try:
        print("  🖼️ Training CNN...")
        from lrdbenchmark.analysis.machine_learning.cnn_estimator_unified import CNNEstimator
        
        cnn_estimator = CNNEstimator(
            conv_channels=[8, 16, 32],  # Further reduced for GPU memory
            fc_layers=[64, 32],         # Further reduced for GPU memory
            learning_rate=0.001,
            epochs=50,  # Reduced for demo
            batch_size=8,   # Further reduced for GPU memory
            use_gradient_checkpointing=True
        )
        
        start_time = time.time()
        cnn_results = cnn_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['cnn'] = {
            'results': cnn_results,
            'training_time': training_time,
            'estimator': cnn_estimator
        }
        
        print(f"    ✅ CNN trained in {training_time:.2f}s")
        
    except Exception as e:
        print(f"    ❌ CNN training failed: {e}")
        results['cnn'] = {'error': str(e)}
    
    # Transformer
    try:
        print("  ⚡ Training Transformer...")
        from lrdbenchmark.analysis.machine_learning.transformer_estimator_unified import TransformerEstimator
        
        transformer_estimator = TransformerEstimator(
            d_model=128,
            nhead=8,
            num_layers=2,
            learning_rate=0.001,
            epochs=50,  # Reduced for demo
            batch_size=32
        )
        
        start_time = time.time()
        transformer_results = transformer_estimator.train(X, y)
        training_time = time.time() - start_time
        
        results['transformer'] = {
            'results': transformer_results,
            'training_time': training_time,
            'estimator': transformer_estimator
        }
        
        print(f"    ✅ Transformer trained in {training_time:.2f}s")
        
    except Exception as e:
        print(f"    ❌ Transformer training failed: {e}")
        results['transformer'] = {'error': str(e)}
    
    return results

def test_pretrained_models(test_data: np.ndarray, training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test the pretrained models to demonstrate the production workflow.
    
    Parameters
    ----------
    test_data : np.ndarray
        Test time series data
    training_results : dict
        Results from training
        
    Returns
    -------
    dict
        Test results for each estimator
    """
    print("\n🧪 Testing Pretrained Models (Production Workflow)...")
    
    test_results = {}
    
    for estimator_name, estimator_info in training_results.items():
        if 'error' in estimator_info:
            continue
            
        try:
            print(f"  🔍 Testing {estimator_name}...")
            estimator = estimator_info['estimator']
            
            # Test estimation
            start_time = time.time()
            result = estimator.estimate(test_data)
            estimation_time = time.time() - start_time
            
            test_results[estimator_name] = {
                'method': result.get('method', 'unknown'),
                'hurst_parameter': result.get('hurst_parameter', 0.5),
                'estimation_time': estimation_time,
                'fallback_used': result.get('fallback_used', False)
            }
            
            print(f"    ✅ Method: {result.get('method', 'unknown')}")
            print(f"    📊 Hurst: {result.get('hurst_parameter', 0.5):.4f}")
            print(f"    ⏱️  Time: {estimation_time:.4f}s")
            
        except Exception as e:
            print(f"    ❌ Testing failed: {e}")
            test_results[estimator_name] = {'error': str(e)}
    
    return test_results

def main():
    """Main training and testing workflow."""
    print("🚀 ML Estimators Training and Testing Workflow")
    print("=" * 60)
    
    # Configuration
    n_samples = 500  # Reduced for demo
    data_length = 500  # Reduced for demo
    
    # Generate synthetic data
    X, y = generate_synthetic_data(
        n_samples=n_samples,
        data_length=data_length
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Samples: {len(X)}")
    print(f"  Data length: {X.shape[1]}")
    print(f"  Hurst range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Mean Hurst: {y.mean():.3f}")
    
    # Split data for training and testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📈 Data Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train simple ML estimators
    simple_results = train_simple_ml_estimators(X_train, y_train)
    
    # Train neural network estimators
    nn_results = train_neural_network_estimators(X_train, y_train)
    
    # Combine results
    all_results = {**simple_results, **nn_results}
    
    # Test pretrained models
    test_results = test_pretrained_models(X_test[0], all_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TRAINING SUMMARY")
    print("=" * 60)
    
    successful_trainings = 0
    total_training_time = 0
    
    for name, info in all_results.items():
        if 'error' not in info:
            successful_trainings += 1
            total_training_time += info.get('training_time', 0)
            print(f"✅ {name.upper()}: {info.get('training_time', 0):.2f}s")
        else:
            print(f"❌ {name.upper()}: {info['error']}")
    
    print(f"\n🎯 Success Rate: {successful_trainings}/{len(all_results)} ({100*successful_trainings/len(all_results):.1f}%)")
    print(f"⏱️  Total Training Time: {total_training_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("🧪 TESTING SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    for name, info in test_results.items():
        if 'error' not in info:
            successful_tests += 1
            print(f"✅ {name.upper()}: {info['method']} (H={info['hurst_parameter']:.4f})")
        else:
            print(f"❌ {name.upper()}: {info['error']}")
    
    print(f"\n🎯 Test Success Rate: {successful_tests}/{len(test_results)} ({100*successful_tests/len(test_results):.1f}%)")
    
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS")
    print("=" * 60)
    print("1. 🏭 Production: Use pretrained models for estimation")
    print("2. 🔄 Retraining: Retrain models with new data when needed")
    print("3. 📊 Evaluation: Compare ML vs classical estimators")
    print("4. 🚀 Deployment: Integrate into production systems")
    
    print("\n🎉 Training workflow completed!")

if __name__ == "__main__":
    main()
