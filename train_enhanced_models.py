#!/usr/bin/env python3
"""
Training Script for Enhanced Neural Network Models

This script demonstrates how to train the enhanced CNN, LSTM, GRU, and Transformer models
with adaptive input sizes and comprehensive training curricula.
"""

import numpy as np
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data generation imports
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel

# Enhanced estimator imports
from lrdbench.analysis.machine_learning.enhanced_cnn_estimator import EnhancedCNNEstimator
from lrdbench.analysis.machine_learning.enhanced_lstm_estimator import EnhancedLSTMEstimator
from lrdbench.analysis.machine_learning.enhanced_gru_estimator import EnhancedGRUEstimator
from lrdbench.analysis.machine_learning.enhanced_transformer_estimator import EnhancedTransformerEstimator


def generate_training_data(
    n_samples: int = 1000,
    sequence_length: int = 1000,
    hurst_range: Tuple[float, float] = (0.1, 0.9),
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate comprehensive training data for Hurst parameter estimation.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    sequence_length : int
        Length of each time series
    hurst_range : Tuple[float, float]
        Range of Hurst parameters to generate
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        Training data and corresponding Hurst parameters
    """
    np.random.seed(random_state)
    
    data_list = []
    labels = []
    
    print(f"Generating {n_samples} training samples...")
    
    # Generate fGn data
    n_fgn = n_samples // 3
    for i in range(n_fgn):
        H = np.random.uniform(hurst_range[0], hurst_range[1])
        fgn = FractionalGaussianNoise(H=H)
        data = fgn.generate(sequence_length, seed=random_state + i)
        data_list.append(data)
        labels.append(H)
    
    # Generate fBm data
    n_fbm = n_samples // 3
    for i in range(n_fbm):
        H = np.random.uniform(hurst_range[0], hurst_range[1])
        fbm = FractionalBrownianMotion(H=H)
        data = fbm.generate(sequence_length, seed=random_state + n_fgn + i)
        data_list.append(data)
        labels.append(H)
    
    # Generate ARFIMA data
    n_arfima = n_samples - n_fgn - n_fbm
    for i in range(n_arfima):
        H = np.random.uniform(hurst_range[0], hurst_range[1])
        arfima = ARFIMAModel(d=H-0.5, ar_params=[0.3], ma_params=[0.2])
        data = arfima.generate(sequence_length, seed=random_state + n_fgn + n_fbm + i)
        data_list.append(data)
        labels.append(H)
    
    print(f"Generated {len(data_list)} samples with Hurst parameters in range {hurst_range}")
    print(f"Data shapes: {[data.shape for data in data_list[:5]]}...")
    print(f"Label range: {min(labels):.3f} to {max(labels):.3f}")
    
    return data_list, labels


def train_enhanced_model(
    estimator_class,
    estimator_name: str,
    data_list: List[np.ndarray],
    labels: List[float],
    **kwargs
) -> dict:
    """
    Train an enhanced neural network model.
    
    Parameters
    ----------
    estimator_class
        The enhanced estimator class to train
    estimator_name : str
        Name of the estimator for logging
    data_list : List[np.ndarray]
        Training data
    labels : List[float]
        Training labels
    **kwargs
        Additional parameters for the estimator
    
    Returns
    -------
    dict
        Training results
    """
    print(f"\n{'='*60}")
    print(f"Training {estimator_name}")
    print(f"{'='*60}")
    
    # Create estimator
    estimator = estimator_class(**kwargs)
    
    # Train model
    results = estimator.train_model(data_list, labels, save_model=True)
    
    print(f"\n{estimator_name} Training Results:")
    print(f"  Final Train Loss: {results['final_train_loss']:.6f}")
    print(f"  Final Val Loss: {results['final_val_loss']:.6f}")
    print(f"  Final Train MAE: {results['final_train_mae']:.6f}")
    print(f"  Final Val MAE: {results['final_val_mae']:.6f}")
    print(f"  Best Val Loss: {results['best_val_loss']:.6f}")
    print(f"  Epochs Trained: {results['epochs_trained']}")
    
    return results


def test_enhanced_model(
    estimator_class,
    estimator_name: str,
    test_data: np.ndarray,
    true_h: float,
    **kwargs
) -> dict:
    """
    Test an enhanced neural network model.
    
    Parameters
    ----------
    estimator_class
        The enhanced estimator class to test
    estimator_name : str
        Name of the estimator for logging
    test_data : np.ndarray
        Test data
    true_h : float
        True Hurst parameter
    **kwargs
        Additional parameters for the estimator
    
    Returns
    -------
    dict
        Test results
    """
    print(f"\nTesting {estimator_name}...")
    
    # Create estimator
    estimator = estimator_class(**kwargs)
    
    # Estimate Hurst parameter
    result = estimator.estimate(test_data)
    
    estimated_h = result['hurst_parameter']
    method = result['method']
    error = abs(estimated_h - true_h)
    
    print(f"  True H: {true_h:.3f}")
    print(f"  Estimated H: {estimated_h:.3f}")
    print(f"  Absolute Error: {error:.3f}")
    print(f"  Method: {method}")
    
    return result


def main():
    """Main training function."""
    print("🚀 Enhanced Neural Network Models Training")
    print("=" * 60)
    
    # Configuration
    n_samples = 500  # Reduced for faster training
    sequence_length = 1000
    hurst_range = (0.1, 0.9)
    random_state = 42
    
    # Generate training data
    print("📊 Generating training data...")
    data_list, labels = generate_training_data(
        n_samples=n_samples,
        sequence_length=sequence_length,
        hurst_range=hurst_range,
        random_state=random_state
    )
    
    # Training configurations for each model
    training_configs = {
        "Enhanced CNN": {
            "class": EnhancedCNNEstimator,
            "params": {
                "conv_channels": [32, 64, 128],
                "fc_layers": [256, 128, 64],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,  # Reduced for demonstration
                "use_residual": True,
                "use_attention": True,
            }
        },
        "Enhanced LSTM": {
            "class": EnhancedLSTMEstimator,
            "params": {
                "hidden_size": 128,
                "num_layers": 3,
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,  # Reduced for demonstration
                "bidirectional": True,
                "use_attention": True,
                "attention_heads": 8,
            }
        },
        "Enhanced GRU": {
            "class": EnhancedGRUEstimator,
            "params": {
                "hidden_size": 128,
                "num_layers": 3,
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,  # Reduced for demonstration
                "bidirectional": True,
                "use_attention": True,
                "attention_heads": 8,
            }
        },
        "Enhanced Transformer": {
            "class": EnhancedTransformerEstimator,
            "params": {
                "d_model": 128,  # Reduced for faster training
                "nhead": 8,
                "num_layers": 4,  # Reduced for faster training
                "dim_feedforward": 512,  # Reduced for faster training
                "dropout": 0.1,
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 50,  # Reduced for demonstration
                "use_layer_norm": True,
                "use_residual": True,
            }
        }
    }
    
    # Train all models
    training_results = {}
    
    for model_name, config in training_configs.items():
        try:
            results = train_enhanced_model(
                config["class"],
                model_name,
                data_list,
                labels,
                **config["params"]
            )
            training_results[model_name] = results
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            training_results[model_name] = {"error": str(e)}
    
    # Generate test data
    print(f"\n{'='*60}")
    print("🧪 Testing Enhanced Models")
    print(f"{'='*60}")
    
    # Create test data with known Hurst parameters
    test_cases = [
        ("fGn (H=0.3)", FractionalGaussianNoise(H=0.3).generate(1000, seed=999), 0.3),
        ("fGn (H=0.7)", FractionalGaussianNoise(H=0.7).generate(1000, seed=998), 0.7),
        ("fBm (H=0.5)", FractionalBrownianMotion(H=0.5).generate(1000, seed=997), 0.5),
    ]
    
    # Test all models
    test_results = {}
    
    for model_name, config in training_configs.items():
        if model_name in training_results and "error" not in training_results[model_name]:
            test_results[model_name] = {}
            
            for test_name, test_data, true_h in test_cases:
                try:
                    result = test_enhanced_model(
                        config["class"],
                        f"{model_name} - {test_name}",
                        test_data,
                        true_h,
                        **config["params"]
                    )
                    test_results[model_name][test_name] = result
                except Exception as e:
                    print(f"❌ Error testing {model_name} on {test_name}: {e}")
                    test_results[model_name][test_name] = {"error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 Training Summary")
    print(f"{'='*60}")
    
    for model_name, results in training_results.items():
        if "error" in results:
            print(f"❌ {model_name}: FAILED - {results['error']}")
        else:
            print(f"✅ {model_name}: SUCCESS")
            print(f"   Best Val Loss: {results['best_val_loss']:.6f}")
            print(f"   Final Val MAE: {results['final_val_mae']:.6f}")
            print(f"   Epochs: {results['epochs_trained']}")
    
    print(f"\n{'='*60}")
    print("🎯 Testing Summary")
    print(f"{'='*60}")
    
    for model_name, results in test_results.items():
        print(f"\n{model_name}:")
        for test_name, result in results.items():
            if "error" in result:
                print(f"  ❌ {test_name}: FAILED - {result['error']}")
            else:
                estimated_h = result['hurst_parameter']
                method = result['method']
                print(f"  ✅ {test_name}: H_est={estimated_h:.3f}, Method={method}")
    
    print(f"\n{'='*60}")
    print("🎉 Training Complete!")
    print(f"{'='*60}")
    print("\nModels have been saved to their respective directories:")
    print("- models/enhanced_cnn/")
    print("- models/enhanced_lstm/")
    print("- models/enhanced_gru/")
    print("- models/enhanced_transformer/")
    print("\nYou can now use these trained models in production!")


if __name__ == "__main__":
    main()
