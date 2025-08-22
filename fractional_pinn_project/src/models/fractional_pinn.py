"""
Fractional Physics-Informed Neural Network (PINN)

This module implements the main fractional PINN model for long-range dependence estimation.
It combines neural networks with fractional calculus operators and physics-informed constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .fractional_operators import (
    FractionalOperator, 
    MarchaudDerivative, 
    WeylDerivative, 
    CaputoDerivative,
    HybridFractionalOperator,
    AdaptiveFractionalOperator,
    hurst_to_fractional_order,
    fractional_order_to_hurst
)
from .physics_constraints import PhysicsConstraints


class FractionalPINN(nn.Module):
    """
    Fractional Physics-Informed Neural Network for long-range dependence estimation.
    
    This model combines neural networks with fractional calculus operators
    and physics-informed constraints to estimate Hurst parameters from time series data.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [128, 256, 128, 64],
                 dropout_rate: float = 0.2,
                 use_fractional_operators: bool = True,
                 fractional_operator_type: str = "hybrid",
                 physics_constraints: bool = True,
                 constraint_weights: Optional[Dict[str, float]] = None,
                 hurst_range: Tuple[float, float] = (0.1, 0.9),
                 device: str = "cpu"):
        """
        Initialize the Fractional PINN.
        
        Args:
            input_size: Size of input time series
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_fractional_operators: Whether to use fractional operators
            fractional_operator_type: Type of fractional operator ("marchaud", "weyl", "caputo", "hybrid", "adaptive")
            physics_constraints: Whether to use physics constraints
            constraint_weights: Weights for different physics constraints
            hurst_range: Valid range for Hurst parameter
            device: Device to run the model on
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_fractional_operators = use_fractional_operators
        self.fractional_operator_type = fractional_operator_type
        self.physics_constraints = physics_constraints
        self.hurst_range = hurst_range
        self.device = device
        
        # Build neural network architecture
        self._build_network()
        
        # Initialize fractional operators
        if self.use_fractional_operators:
            self._setup_fractional_operators()
        
        # Initialize physics constraints
        if self.physics_constraints:
            self.physics_constraints = PhysicsConstraints(
                hurst_range=hurst_range,
                constraint_weights=constraint_weights
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer (single value for Hurst parameter)
        layers.append(nn.Linear(self.hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Additional layers for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combination layer for final prediction
        self.combination_layer = nn.Sequential(
            nn.Linear(33, 16),  # 32 features + 1 network output
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def _setup_fractional_operators(self):
        """Setup fractional calculus operators."""
        if self.fractional_operator_type == "marchaud":
            self.fractional_operator = MarchaudDerivative(order=0.5)
        elif self.fractional_operator_type == "weyl":
            self.fractional_operator = WeylDerivative(order=0.5)
        elif self.fractional_operator_type == "caputo":
            self.fractional_operator = CaputoDerivative(order=0.5)
        elif self.fractional_operator_type == "hybrid":
            self.fractional_operator = HybridFractionalOperator(order=0.5)
        elif self.fractional_operator_type == "adaptive":
            self.fractional_operator = AdaptiveFractionalOperator(initial_order=0.5)
        else:
            raise ValueError(f"Unknown fractional operator type: {self.fractional_operator_type}")
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input time series of shape (batch_size, sequence_length)
            
        Returns:
            Predicted Hurst parameters of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Extract features from the time series
        features = self.feature_extractor(x)
        
        # Apply fractional operators if enabled
        if self.use_fractional_operators and hasattr(self, 'fractional_operator'):
            try:
                # Apply fractional derivative
                x_frac = self.fractional_operator(x)
                
                # Compute additional features from fractional derivative
                frac_features = torch.stack([
                    torch.mean(x_frac, dim=1),
                    torch.std(x_frac, dim=1),
                    torch.var(x_frac, dim=1),
                    torch.max(x_frac, dim=1)[0],
                    torch.min(x_frac, dim=1)[0]
                ], dim=1)
                
                # Concatenate with original features
                features = torch.cat([features, frac_features], dim=1)
                
            except Exception as e:
                warnings.warn(f"Fractional operator failed: {e}. Using original features only.")
        
        # Pass through the main network
        network_output = self.network(x)
        
        # Combine features and network output
        combined = torch.cat([features, network_output], dim=1)
        hurst_pred = self.combination_layer(combined)
        
        # Scale to the valid Hurst range
        hurst_min, hurst_max = self.hurst_range
        hurst_pred = hurst_min + (hurst_max - hurst_min) * hurst_pred
        
        return hurst_pred
    
    def compute_loss(self, 
                    x: torch.Tensor, 
                    y_true: Optional[torch.Tensor] = None,
                    lambda_physics: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including physics constraints.
        
        Args:
            x: Input time series
            y_true: True Hurst parameters (optional, for supervised learning)
            lambda_physics: Weight for physics constraint loss
            
        Returns:
            Dictionary containing different loss components
        """
        # Forward pass
        hurst_pred = self.forward(x)
        
        losses = {}
        
        # Supervised loss (if labels provided)
        if y_true is not None:
            losses['supervised'] = F.mse_loss(hurst_pred.squeeze(), y_true)
        else:
            losses['supervised'] = torch.tensor(0.0, device=self.device)
        
        # Physics constraint loss
        if self.physics_constraints and hasattr(self, 'physics_constraints'):
            try:
                if self.use_fractional_operators and hasattr(self, 'fractional_operator'):
                    physics_loss = self.physics_constraints.compute_total_constraint_loss(
                        x, hurst_pred.squeeze(), self.fractional_operator
                    )
                else:
                    physics_loss = self.physics_constraints.compute_total_constraint_loss(
                        x, hurst_pred.squeeze()
                    )
                losses['physics'] = physics_loss
            except Exception as e:
                warnings.warn(f"Physics constraints failed: {e}")
                losses['physics'] = torch.tensor(0.0, device=self.device)
        else:
            losses['physics'] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        losses['total'] = losses['supervised'] + lambda_physics * losses['physics']
        
        return losses
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            x: Input time series
            
        Returns:
            Predicted Hurst parameters as numpy array
        """
        self.eval()
        with torch.no_grad():
            hurst_pred = self.forward(x)
            return hurst_pred.cpu().numpy()
    
    def get_fractional_order(self) -> float:
        """Get the current fractional order if using adaptive operator."""
        if (self.use_fractional_operators and 
            hasattr(self, 'fractional_operator') and 
            hasattr(self.fractional_operator, 'current_order')):
            return self.fractional_operator.current_order
        else:
            return 0.5  # Default order
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'use_fractional_operators': self.use_fractional_operators,
            'fractional_operator_type': self.fractional_operator_type,
            'physics_constraints': self.physics_constraints,
            'hurst_range': self.hurst_range,
            'device': self.device
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = "cpu"):
        """Load a model from a file."""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout_rate=checkpoint['dropout_rate'],
            use_fractional_operators=checkpoint['use_fractional_operators'],
            fractional_operator_type=checkpoint['fractional_operator_type'],
            physics_constraints=checkpoint['physics_constraints'],
            hurst_range=checkpoint['hurst_range'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class EnsembleFractionalPINN(nn.Module):
    """
    Ensemble of Fractional PINNs for improved performance and robustness.
    
    This class combines multiple Fractional PINNs with different configurations
    to improve prediction accuracy and reduce uncertainty.
    """
    
    def __init__(self,
                 input_size: int,
                 n_models: int = 5,
                 hidden_sizes_list: Optional[List[List[int]]] = None,
                 fractional_operator_types: Optional[List[str]] = None,
                 device: str = "cpu"):
        """
        Initialize the ensemble.
        
        Args:
            input_size: Size of input time series
            n_models: Number of models in the ensemble
            hidden_sizes_list: List of hidden sizes for each model
            fractional_operator_types: List of fractional operator types for each model
            device: Device to run the models on
        """
        super().__init__()
        
        self.input_size = input_size
        self.n_models = n_models
        self.device = device
        
        # Default configurations if not provided
        if hidden_sizes_list is None:
            hidden_sizes_list = [
                [128, 256, 128, 64],
                [256, 512, 256, 128],
                [64, 128, 64, 32],
                [512, 1024, 512, 256],
                [96, 192, 96, 48]
            ][:n_models]
        
        if fractional_operator_types is None:
            fractional_operator_types = ["marchaud", "weyl", "caputo", "hybrid", "adaptive"][:n_models]
        
        # Create ensemble models
        self.models = nn.ModuleList([
            FractionalPINN(
                input_size=input_size,
                hidden_sizes=hidden_sizes_list[i],
                fractional_operator_type=fractional_operator_types[i],
                device=device
            ) for i in range(n_models)
        ])
        
        # Learnable weights for ensemble combination
        self.ensemble_weights = nn.Parameter(torch.ones(n_models) / n_models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input time series
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted combination
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = torch.zeros_like(predictions[0])
        
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input time series
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)  # (n_models, batch_size, 1)
            
            mean_pred = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            return mean_pred, uncertainty
