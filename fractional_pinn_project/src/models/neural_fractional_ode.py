"""
Neural Fractional ODE Model

This module implements Neural Fractional Ordinary Differential Equations,
where neural networks learn the right-hand side of fractional differential
equations. This approach combines the expressiveness of neural networks
with the mathematical structure of fractional ODEs.

Key Features:
1. Neural network parameterization of fractional ODE right-hand side
2. Fractional derivative computation using various methods
3. Physics-informed constraints for fractional dynamics
4. Adaptive time stepping for numerical integration
5. Multi-scale feature extraction for complex dynamics

Author: Fractional PINN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from scipy.special import gamma
from scipy.integrate import solve_ivp

# Import our custom modules
from .mellin_transform import FractionalMellinTransform
from .physics_constraints import PhysicsConstraints

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FractionalDerivative:
    """
    Compute fractional derivatives using various numerical methods.
    
    This class provides implementations of different fractional derivative
    approximations including Grünwald-Letnikov, Caputo, and Riemann-Liouville.
    """
    
    def __init__(self, method: str = 'grunwald_letnikov'):
        """
        Initialize fractional derivative calculator.
        
        Args:
            method: Method for computing fractional derivatives
                   ('grunwald_letnikov', 'caputo', 'riemann_liouville')
        """
        self.method = method
    
    def grunwald_letnikov(self, y: torch.Tensor, alpha: float, dt: float = 1.0) -> torch.Tensor:
        """
        Compute fractional derivative using Grünwald-Letnikov method.
        
        Args:
            y: Function values
            alpha: Fractional order (0 < alpha < 1)
            dt: Time step
            
        Returns:
            Fractional derivative
        """
        n = y.shape[-1]
        coeffs = torch.zeros(n, device=y.device)
        
        # Compute binomial coefficients
        coeffs[0] = 1.0
        for k in range(1, n):
            coeffs[k] = coeffs[k-1] * (1 - (alpha + 1) / k)
        
        # Compute fractional derivative
        result = torch.zeros_like(y)
        for i in range(n):
            result[..., i] = torch.sum(coeffs[:i+1] * y[..., :i+1].flip(-1), dim=-1)
        
        return result / (dt ** alpha)
    
    def caputo(self, y: torch.Tensor, alpha: float, dt: float = 1.0) -> torch.Tensor:
        """
        Compute Caputo fractional derivative.
        
        Args:
            y: Function values
            alpha: Fractional order (0 < alpha < 1)
            dt: Time step
            
        Returns:
            Caputo fractional derivative
        """
        n = y.shape[-1]
        result = torch.zeros_like(y)
        
        # Compute Caputo derivative
        for i in range(1, n):
            weights = torch.zeros(i, device=y.device)
            for j in range(i):
                weights[j] = ((i - j) ** alpha - (i - j - 1) ** alpha) / gamma(alpha + 1)
            
            result[..., i] = torch.sum(weights * y[..., 1:i+1], dim=-1)
        
        return result / (dt ** alpha)
    
    def __call__(self, y: torch.Tensor, alpha: float, dt: float = 1.0) -> torch.Tensor:
        """
        Compute fractional derivative using specified method.
        
        Args:
            y: Function values
            alpha: Fractional order
            dt: Time step
            
        Returns:
            Fractional derivative
        """
        if self.method == 'grunwald_letnikov':
            return self.grunwald_letnikov(y, alpha, dt)
        elif self.method == 'caputo':
            return self.caputo(y, alpha, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class NeuralODENetwork(nn.Module):
    """
    Neural network for learning the right-hand side of fractional ODEs.
    
    This network takes the current state and time as input and outputs
    the right-hand side of the fractional differential equation.
    """
    
    def __init__(self, 
                 input_dim: int = 2,  # state + time
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize neural ODE network.
        
        Args:
            input_dim: Input dimension (state + time)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural ODE network.
        
        Args:
            t: Time points
            y: State values
            
        Returns:
            Right-hand side of ODE
        """
        # Concatenate time and state
        inputs = torch.cat([t.unsqueeze(-1), y], dim=-1)
        return self.network(inputs)


class NeuralFractionalODE(nn.Module):
    """
    Neural Fractional Ordinary Differential Equation model.
    
    This model uses neural networks to learn the right-hand side of
    fractional differential equations and solves them numerically.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 alpha: float = 0.5,
                 derivative_method: str = 'grunwald_letnikov',
                 use_mellin_transform: bool = True,
                 use_physics_constraints: bool = True):
        """
        Initialize Neural Fractional ODE.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            alpha: Fractional order
            derivative_method: Method for computing fractional derivatives
            use_mellin_transform: Whether to use Mellin transform
            use_physics_constraints: Whether to use physics constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.alpha = alpha
        self.use_mellin_transform = use_mellin_transform
        self.use_physics_constraints = use_physics_constraints
        
        # Neural network for ODE right-hand side
        self.ode_network = NeuralODENetwork(
            input_dim=input_dim + 1,  # state + time
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Fractional derivative calculator
        self.fractional_derivative = FractionalDerivative(method=derivative_method)
        
        # Physics components
        if use_mellin_transform:
            self.mellin_transform = FractionalMellinTransform()
        
        if use_physics_constraints:
            self.physics_constraints = PhysicsConstraints()
        
        # Multi-scale feature extraction
        self._build_multi_scale_features()
        
        # Hurst exponent estimator
        self.hurst_estimator = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ensure Hurst exponent is between 0 and 1
        )
    
    def _build_multi_scale_features(self):
        """Build multi-scale feature extraction layers."""
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]
        ])
        
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dims[-1],
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, t: torch.Tensor, y0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Neural Fractional ODE.
        
        Args:
            t: Time points
            y0: Initial conditions
            
        Returns:
            Tuple of (solution, hurst_exponent)
        """
        batch_size, seq_len = t.shape[:2]
        
        # Solve the fractional ODE numerically
        solution = self._solve_fractional_ode(t, y0)
        
        # Extract features for Hurst estimation
        features = self._extract_features(solution)
        
        # Estimate Hurst exponent
        hurst_exponent = self.hurst_estimator(features)
        
        return solution, hurst_exponent
    
    def _solve_fractional_ode(self, t: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """
        Solve the fractional ODE using numerical integration.
        
        Args:
            t: Time points
            y0: Initial conditions
            
        Returns:
            Solution trajectory
        """
        batch_size, seq_len = t.shape[:2]
        device = t.device
        
        # Initialize solution
        solution = torch.zeros(batch_size, seq_len, self.output_dim, device=device)
        solution[:, 0] = y0
        
        # Numerical integration using Adams-Bashforth method
        dt = t[:, 1] - t[:, 0]
        
        for i in range(1, seq_len):
            # Compute fractional derivative
            frac_deriv = self.fractional_derivative(solution[:, :i+1], self.alpha, dt[0])
            
            # Compute right-hand side using neural network
            rhs = self.ode_network(t[:, i:i+1], solution[:, i:i+1])
            
            # Update solution using implicit Euler method
            solution[:, i] = solution[:, i-1] + dt[0] * rhs.squeeze(1)
        
        return solution
    
    def _extract_features(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Extract features from solution for Hurst estimation.
        
        Args:
            solution: Solution trajectory
            
        Returns:
            Extracted features
        """
        batch_size, seq_len, _ = solution.shape
        
        # Reshape for convolution
        x = solution.transpose(1, 2)  # (batch, channels, length)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv_layer in self.multi_scale_conv:
            features = conv_layer(x)
            multi_scale_features.append(features)
        
        # Combine multi-scale features
        if len(multi_scale_features) > 1:
            combined_features = torch.stack(multi_scale_features, dim=1)
            combined_features = combined_features.mean(dim=1)
        else:
            combined_features = multi_scale_features[0]
        
        # Global average pooling
        features = F.adaptive_avg_pool1d(combined_features, 1).squeeze(-1)
        
        return features
    
    def compute_physics_loss(self, 
                           t: torch.Tensor, 
                           y: torch.Tensor, 
                           hurst: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            t: Time points
            y: Solution values
            hurst: Predicted Hurst exponents
            
        Returns:
            Physics loss
        """
        total_loss = torch.tensor(0.0, device=t.device)
        
        # Physics constraints loss
        if self.use_physics_constraints and hasattr(self, 'physics_constraints'):
            physics_loss = self.physics_constraints.compute_loss(t, y, hurst)
            total_loss += physics_loss
        
        # Mellin transform loss
        if self.use_mellin_transform and hasattr(self, 'mellin_transform'):
            mellin_loss = self.mellin_transform.compute_loss(t, y, hurst)
            total_loss += mellin_loss
        
        # ODE consistency loss
        ode_loss = self._compute_ode_consistency_loss(t, y)
        total_loss += ode_loss
        
        return total_loss
    
    def _compute_ode_consistency_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute ODE consistency loss.
        
        Args:
            t: Time points
            y: Solution values
            
        Returns:
            ODE consistency loss
        """
        batch_size, seq_len, _ = y.shape
        device = t.device
        
        # Compute fractional derivative
        dt = t[:, 1] - t[:, 0]
        frac_deriv = self.fractional_derivative(y, self.alpha, dt[0])
        
        # Compute right-hand side using neural network
        rhs = self.ode_network(t, y)
        
        # ODE consistency loss
        ode_loss = F.mse_loss(frac_deriv, rhs)
        
        return ode_loss


class NeuralFractionalODETrainer:
    """
    Trainer for Neural Fractional ODE model.
    
    This class handles the training of the neural fractional ODE model
    with physics-informed constraints and numerical integration.
    """
    
    def __init__(self,
                 model: NeuralFractionalODE,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize Neural Fractional ODE trainer.
        
        Args:
            model: Neural Fractional ODE model
            learning_rate: Learning rate for optimization
            device: Device to use
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'ode_loss': [],
            'hurst_loss': []
        }
    
    def train_step(self, 
                   t: torch.Tensor,
                   y_target: torch.Tensor,
                   y0: torch.Tensor,
                   hurst_target: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            t: Time points
            y_target: Target solution values
            y0: Initial conditions
            hurst_target: Target Hurst exponents
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred, hurst_pred = self.model(t, y0)
        
        # Compute losses
        data_loss = self.mse_loss(y_pred, y_target)
        hurst_loss = self.mse_loss(hurst_pred, hurst_target)
        
        # Physics loss
        physics_loss = self.model.compute_physics_loss(t, y_pred, hurst_pred)
        
        # Total loss
        total_loss = data_loss + 0.1 * physics_loss + 0.05 * hurst_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'hurst_loss': hurst_loss.item()
        }
    
    def validate(self, 
                 t: torch.Tensor,
                 y_target: torch.Tensor,
                 y0: torch.Tensor,
                 hurst_target: torch.Tensor) -> Dict[str, float]:
        """
        Perform validation.
        
        Args:
            t: Time points
            y_target: Target solution values
            y0: Initial conditions
            hurst_target: Target Hurst exponents
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            y_pred, hurst_pred = self.model(t, y0)
            
            # Compute validation losses
            data_loss = self.mse_loss(y_pred, y_target)
            hurst_loss = self.mse_loss(hurst_pred, hurst_target)
            physics_loss = self.model.compute_physics_loss(t, y_pred, hurst_pred)
            
            total_loss = data_loss + 0.1 * physics_loss + 0.05 * hurst_loss
        
        return {
            'val_total_loss': total_loss.item(),
            'val_data_loss': data_loss.item(),
            'val_physics_loss': physics_loss.item(),
            'val_hurst_loss': hurst_loss.item()
        }
    
    def train(self,
              train_loader,
              val_loader,
              epochs: int = 1000,
              early_stopping_patience: int = 50,
              verbose: bool = True,
              save_model: bool = True,
              model_description: str = "",
              model_tags: List[str] = None) -> Dict[str, List[float]]:
        """
        Train the Neural Fractional ODE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            save_model: Whether to save the trained model
            model_description: Description for the saved model
            model_tags: Tags for the saved model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            epoch_losses = {'total_loss': 0, 'data_loss': 0, 'physics_loss': 0, 'hurst_loss': 0}
            num_batches = 0
            
            for t, y_target, y0, hurst_target in train_loader:
                t = t.to(self.device)
                y_target = y_target.to(self.device)
                y0 = y0.to(self.device)
                hurst_target = hurst_target.to(self.device)
                
                losses = self.train_step(t, y_target, y0, hurst_target)
                
                for key, value in losses.items():
                    epoch_losses[key] += value
                num_batches += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                self.training_history[key].append(epoch_losses[key])
            
            # Validation
            val_losses = {'val_total_loss': 0, 'val_data_loss': 0, 
                          'val_physics_loss': 0, 'val_hurst_loss': 0}
            val_batches = 0
            
            for t, y_target, y0, hurst_target in val_loader:
                t = t.to(self.device)
                y_target = y_target.to(self.device)
                y0 = y0.to(self.device)
                hurst_target = hurst_target.to(self.device)
                
                val_metrics = self.validate(t, y_target, y0, hurst_target)
                
                for key, value in val_metrics.items():
                    val_losses[key] += value
                val_batches += 1
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= val_batches
            
            # Update scheduler
            self.scheduler.step(val_losses['val_total_loss'])
            
            # Early stopping
            if val_losses['val_total_loss'] < best_val_loss:
                best_val_loss = val_losses['val_total_loss']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {epoch_losses['total_loss']:.6f}, "
                      f"Val Loss = {val_losses['val_total_loss']:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Save the trained model using the persistence system
        if save_model:
            try:
                from .model_persistence import quick_save_model
                from .model_comparison import ModelConfig
                
                # Create model config
                config = ModelConfig(
                    model_type='neural_ode',
                    input_dim=1,
                    hidden_dims=[64, 128, 64],
                    output_dim=1,
                    learning_rate=self.learning_rate,
                    epochs=epochs,
                    alpha=0.5,
                    use_mellin_transform=True,
                    use_physics_constraints=True
                )
                
                # Save model
                training_duration = time.time() - start_time
                model_id = quick_save_model(
                    model=self.model,
                    config=config,
                    training_history=self.training_history,
                    description=model_description or f"Neural Fractional ODE model trained for {epochs} epochs",
                    tags=model_tags or ['neural_ode', 'fractional', 'trained']
                )
                
                if verbose:
                    print(f"Model saved successfully with ID: {model_id}")
                    
            except ImportError:
                # Fallback to simple save if persistence system not available
                torch.save(self.model.state_dict(), 'best_neural_fractional_ode.pth')
                if verbose:
                    print("Model saved as 'best_neural_fractional_ode.pth' (fallback)")
        
        return self.training_history


# Convenience functions
def create_neural_fractional_ode(input_dim: int = 1,
                                hidden_dims: List[int] = [64, 128, 64],
                                output_dim: int = 1,
                                alpha: float = 0.5,
                                derivative_method: str = 'grunwald_letnikov',
                                use_mellin_transform: bool = True,
                                use_physics_constraints: bool = True) -> NeuralFractionalODE:
    """
    Create a Neural Fractional ODE model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        alpha: Fractional order
        derivative_method: Method for computing fractional derivatives
        use_mellin_transform: Whether to use Mellin transform
        use_physics_constraints: Whether to use physics constraints
        
    Returns:
        Neural Fractional ODE model
    """
    return NeuralFractionalODE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        alpha=alpha,
        derivative_method=derivative_method,
        use_mellin_transform=use_mellin_transform,
        use_physics_constraints=use_physics_constraints
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Neural Fractional ODE...")
    
    # Create model
    model = create_neural_fractional_ode(
        input_dim=1,
        hidden_dims=[32, 64, 32],
        output_dim=1,
        alpha=0.5
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    
    t = torch.linspace(0, 10, seq_len).unsqueeze(0).repeat(batch_size, 1)
    y0 = torch.randn(batch_size, 1)
    
    solution, hurst = model(t, y0)
    
    print(f"Time shape: {t.shape}")
    print(f"Initial condition shape: {y0.shape}")
    print(f"Solution shape: {solution.shape}")
    print(f"Hurst exponent shape: {hurst.shape}")
    print(f"Hurst values: {hurst.squeeze()}")
    
    # Test physics loss
    physics_loss = model.compute_physics_loss(t, solution, hurst)
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    print("Neural Fractional ODE test completed successfully!")
