"""
Machine Learning Estimators for Long-Range Dependence Analysis

This package provides machine learning-based approaches for estimating
Hurst parameters and long-range dependence characteristics from time series data.

The estimators include:
- Neural Network Regression
- Random Forest Regression
- Support Vector Regression
- Gradient Boosting Regression
- Convolutional Neural Networks
- Recurrent Neural Networks (LSTM/GRU)
- Transformer-based approaches
"""

from .base_ml_estimator import BaseMLEstimator
from .neural_network_estimator import NeuralNetworkEstimator
from .random_forest_estimator import RandomForestEstimator
from .svr_estimator import SVREstimator
from .gradient_boosting_estimator import GradientBoostingEstimator
from .cnn_estimator import CNNEstimator
from .lstm_estimator import LSTMEstimator
from .gru_estimator import GRUEstimator
from .transformer_estimator import TransformerEstimator

# Enhanced neural network estimators
from .enhanced_cnn_estimator import EnhancedCNNEstimator
from .enhanced_lstm_estimator import EnhancedLSTMEstimator
from .enhanced_gru_estimator import EnhancedGRUEstimator
from .enhanced_transformer_estimator import EnhancedTransformerEstimator

__all__ = [
    "BaseMLEstimator",
    "NeuralNetworkEstimator",
    "RandomForestEstimator",
    "SVREstimator",
    "GradientBoostingEstimator",
    "CNNEstimator",
    "LSTMEstimator",
    "GRUEstimator",
    "TransformerEstimator",
    # Enhanced estimators
    "EnhancedCNNEstimator",
    "EnhancedLSTMEstimator",
    "EnhancedGRUEstimator",
    "EnhancedTransformerEstimator",
]
