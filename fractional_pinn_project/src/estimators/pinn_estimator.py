"""
PINN Estimator for Fractional Time Series Analysis

This module provides the Physics-Informed Neural Network (PINN) estimator
for fractional time series analysis, integrating the Fractional Mellin Transform
and physics-informed constraints.

The PINN approach combines:
1. Neural network approximation of the solution
2. Physics-informed constraints (fractional operators, Mellin transforms)
3. Data-driven learning from observed time series
4. Multi-scale feature extraction

Author: Fractional PINN Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import time

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fractional_pinn import FractionalPINN
from models.mellin_transform import FractionalMellinTransform
from models.physics_constraints import PhysicsConstraints

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PINNEstimator:
    """
    Physics-Informed Neural Network estimator for fractional time series analysis.
    
    This estimator uses a PINN approach to estimate Hurst exponents and other
    fractional parameters by incorporating physics-informed constraints and
    the Fractional Mellin Transform.
    """
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize the PINN estimator.
        
        Args:
            input_dim: Input dimension (typically 1 for time series)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (typically 1 for Hurst exponent)
            learning_rate: Learning rate for optimization
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.pinn = None
        self.mellin_transform = None
        self.physics_constraints = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'mellin_loss': [],
            'validation_loss': [],
            'hurst_estimates': []
        }
        
        # Results storage
        self.results = {}
        
        print(f"PINN Estimator initialized on device: {self.device}")
    
    def build_model(self, 
                   use_mellin_transform: bool = True,
                   use_physics_constraints: bool = True,
                   constraint_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Build the PINN model with specified components.
        
        Args:
            use_mellin_transform: Whether to use Fractional Mellin Transform
            use_physics_constraints: Whether to use physics-informed constraints
            constraint_weights: Weights for different constraint terms
        """
        # Initialize PINN
        self.pinn = FractionalPINN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        ).to(self.device)
        
        # Initialize Mellin transform if requested
        if use_mellin_transform:
            self.mellin_transform = FractionalMellinTransform().to(self.device)
        
        # Initialize physics constraints if requested
        if use_physics_constraints:
            self.physics_constraints = PhysicsConstraints().to(self.device)
        
        # Set constraint weights
        if constraint_weights is None:
            self.constraint_weights = {
                'data': 1.0,
                'physics': 0.1,
                'mellin': 0.05,
                'regularization': 0.01
            }
        else:
            self.constraint_weights = constraint_weights
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.pinn.parameters(), lr=self.learning_rate)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print("PINN model built successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.pinn.parameters()):,}")
    
    def prepare_data(self, 
                    data: np.ndarray,
                    validation_split: float = 0.2,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            data: Time series data
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Normalize data
        data_mean = np.mean(data)
        data_std = np.std(data)
        normalized_data = (data - data_mean) / data_std
        
        # Create time points
        t = np.linspace(0, 1, len(data)).reshape(-1, 1)
        
        # Split into train and validation
        n_train = int(len(data) * (1 - validation_split))
        
        t_train = t[:n_train]
        y_train = normalized_data[:n_train].reshape(-1, 1)
        
        t_val = t[n_train:]
        y_val = normalized_data[n_train:].reshape(-1, 1)
        
        # Convert to tensors
        t_train_tensor = torch.FloatTensor(t_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        t_val_tensor = torch.FloatTensor(t_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(t_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(t_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Store normalization parameters
        self.data_mean = data_mean
        self.data_std = data_std
        
        print(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        return train_loader, val_loader
    
    def compute_loss(self, 
                    t: torch.Tensor, 
                    y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    hurst_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total loss including all constraint terms.
        
        Args:
            t: Time points
            y_true: True values
            y_pred: Predicted values
            hurst_pred: Predicted Hurst exponent
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Data loss (MSE)
        data_loss = nn.MSELoss()(y_pred, y_true)
        
        # Initialize loss components
        loss_components = {'data': data_loss}
        total_loss = self.constraint_weights['data'] * data_loss
        
        # Physics constraints loss
        if self.physics_constraints is not None:
            physics_loss = self.physics_constraints.compute_loss(
                t, y_pred, hurst_pred
            )
            loss_components['physics'] = physics_loss
            total_loss += self.constraint_weights['physics'] * physics_loss
        
        # Mellin transform loss
        if self.mellin_transform is not None:
            mellin_loss = self.mellin_transform.compute_loss(
                t, y_pred, hurst_pred
            )
            loss_components['mellin'] = mellin_loss
            total_loss += self.constraint_weights['mellin'] * mellin_loss
        
        # Regularization loss
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.pinn.parameters():
            l2_reg += torch.norm(param)
        
        loss_components['regularization'] = l2_reg
        total_loss += self.constraint_weights['regularization'] * l2_reg
        
        return total_loss, loss_components
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        self.pinn.train()
        total_loss = 0.0
        loss_components = {'data': 0.0, 'physics': 0.0, 'mellin': 0.0, 'regularization': 0.0}
        n_batches = 0
        
        for t_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred, hurst_pred = self.pinn(t_batch)
            
            # Compute loss
            loss, components = self.compute_loss(t_batch, y_batch, y_pred, hurst_pred)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value.item()
            n_batches += 1
        
        # Average losses
        total_loss /= n_batches
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return total_loss, loss_components
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.pinn.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for t_batch, y_batch in val_loader:
                # Forward pass
                y_pred, hurst_pred = self.pinn(t_batch)
                
                # Compute loss (only data loss for validation)
                loss = nn.MSELoss()(y_pred, y_batch)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 1000,
              early_stopping_patience: int = 50,
              verbose: bool = True,
              save_model: bool = True,
              model_description: str = "",
              model_tags: List[str] = None) -> Dict[str, List[float]]:
        """
        Train the PINN model.
        
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
        if self.pinn is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training PINN", disable=not verbose):
            # Train
            train_loss, loss_components = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.training_history['total_loss'].append(train_loss)
            self.training_history['data_loss'].append(loss_components['data'])
            self.training_history['physics_loss'].append(loss_components.get('physics', 0.0))
            self.training_history['mellin_loss'].append(loss_components.get('mellin', 0.0))
            self.training_history['validation_loss'].append(val_loss)
            
            # Get current Hurst estimate
            with torch.no_grad():
                t_sample = torch.FloatTensor([[0.5]]).to(self.device)
                _, hurst_pred = self.pinn(t_sample)
                self.training_history['hurst_estimates'].append(hurst_pred.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.pinn.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.pinn.load_state_dict(best_model_state)
        
        # Save the trained model using the persistence system
        if save_model:
            try:
                from ..models.model_persistence import quick_save_model
                from ..models.model_comparison import ModelConfig
                
                # Create model config
                config = ModelConfig(
                    model_type='pinn',
                    input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims,
                    output_dim=self.output_dim,
                    learning_rate=self.learning_rate,
                    epochs=epochs,
                    use_mellin_transform=True,
                    use_physics_constraints=True
                )
                
                # Save model
                training_duration = time.time() - start_time
                model_id = quick_save_model(
                    model=self.pinn,
                    config=config,
                    training_history=self.training_history,
                    description=model_description or f"PINN model trained for {epochs} epochs",
                    tags=model_tags or ['pinn', 'fractional', 'trained']
                )
                
                if verbose:
                    print(f"Model saved successfully with ID: {model_id}")
                    
            except ImportError:
                # Fallback to simple save if persistence system not available
                torch.save(self.pinn.state_dict(), 'best_pinn_model.pth')
                if verbose:
                    print("Model saved as 'best_pinn_model.pth' (fallback)")
        
        return self.training_history
    
    def estimate(self, 
                data: np.ndarray,
                epochs: int = 1000,
                batch_size: int = 32,
                validation_split: float = 0.2,
                use_mellin_transform: bool = True,
                use_physics_constraints: bool = True,
                constraint_weights: Optional[Dict[str, float]] = None,
                verbose: bool = True) -> Dict:
        """
        Estimate Hurst exponent using PINN.
        
        Args:
            data: Time series data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            use_mellin_transform: Whether to use Mellin transform
            use_physics_constraints: Whether to use physics constraints
            constraint_weights: Weights for constraint terms
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing estimation results
        """
        start_time = time.time()
        
        # Build model
        self.build_model(
            use_mellin_transform=use_mellin_transform,
            use_physics_constraints=use_physics_constraints,
            constraint_weights=constraint_weights
        )
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            data, validation_split, batch_size
        )
        
        # Train model
        history = self.train(
            train_loader, val_loader, epochs, verbose=verbose
        )
        
        # Get final estimate
        with torch.no_grad():
            t_full = torch.FloatTensor(np.linspace(0, 1, len(data)).reshape(-1, 1)).to(self.device)
            y_pred, hurst_pred = self.pinn(t_full)
            
            # Denormalize predictions
            y_pred_denorm = y_pred.cpu().numpy() * self.data_std + self.data_mean
            hurst_estimate = hurst_pred.item()
        
        # Calculate confidence interval using bootstrap
        confidence_interval = self._bootstrap_confidence_interval(data, n_bootstrap=100)
        
        # Store results
        self.results = {
            'hurst': hurst_estimate,
            'confidence_interval': confidence_interval,
            'training_history': history,
            'predictions': y_pred_denorm,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'training_time': time.time() - start_time,
            'final_train_loss': history['total_loss'][-1],
            'final_val_loss': history['validation_loss'][-1]
        }
        
        if verbose:
            print(f"\nPINN Estimation Complete:")
            print(f"Hurst Estimate: {hurst_estimate:.4f}")
            print(f"Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
            print(f"Training Time: {self.results['training_time']:.2f} seconds")
        
        return self.results
    
    def _bootstrap_confidence_interval(self, 
                                     data: np.ndarray, 
                                     n_bootstrap: int = 100,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap resampling.
        
        Args:
            data: Original time series data
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data[indices]
            
            # Quick estimate using a subset of epochs
            try:
                # Use a smaller model for bootstrap
                temp_pinn = FractionalPINN(
                    input_dim=1,
                    hidden_dims=[32, 64, 32],
                    output_dim=1
                ).to(self.device)
                
                # Quick training
                temp_optimizer = optim.Adam(temp_pinn.parameters(), lr=1e-2)
                
                # Prepare data
                data_mean = np.mean(bootstrap_data)
                data_std = np.std(bootstrap_data)
                normalized_data = (bootstrap_data - data_mean) / data_std
                
                t = torch.FloatTensor(np.linspace(0, 1, len(bootstrap_data)).reshape(-1, 1)).to(self.device)
                y = torch.FloatTensor(normalized_data.reshape(-1, 1)).to(self.device)
                
                # Quick training (fewer epochs)
                for _ in range(50):
                    temp_optimizer.zero_grad()
                    y_pred, hurst_pred = temp_pinn(t)
                    loss = nn.MSELoss()(y_pred, y)
                    loss.backward()
                    temp_optimizer.step()
                
                # Get estimate
                with torch.no_grad():
                    t_sample = torch.FloatTensor([[0.5]]).to(self.device)
                    _, hurst_est = temp_pinn(t_sample)
                    bootstrap_estimates.append(hurst_est.item())
                    
            except Exception as e:
                # If bootstrap fails, skip this sample
                continue
        
        if len(bootstrap_estimates) < 10:
            # Fallback to simple interval
            return (self.results['hurst'] - 0.05, self.results['hurst'] + 0.05)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
        upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.training_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Component losses
        axes[0, 1].plot(self.training_history['data_loss'], label='Data')
        if 'physics_loss' in self.training_history:
            axes[0, 1].plot(self.training_history['physics_loss'], label='Physics')
        if 'mellin_loss' in self.training_history:
            axes[0, 1].plot(self.training_history['mellin_loss'], label='Mellin')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Validation loss
        axes[1, 0].plot(self.training_history['validation_loss'])
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Hurst estimates
        axes[1, 1].plot(self.training_history['hurst_estimates'])
        axes[1, 1].set_title('Hurst Estimates')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Hurst Exponent')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self, data: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot estimation results."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original data vs predictions
        t = np.linspace(0, 1, len(data))
        axes[0].plot(t, data, 'b-', label='Original Data', alpha=0.7)
        axes[0].plot(t, self.results['predictions'], 'r--', label='PINN Predictions', alpha=0.8)
        axes[0].set_title(f'PINN Estimation Results (H = {self.results["hurst"]:.3f})')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Residuals
        residuals = data - self.results['predictions'].flatten()
        axes[1].plot(t, residuals, 'g-', alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_title('Residuals')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Residual')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Convenience functions
def estimate_hurst_pinn(data: np.ndarray, 
                       epochs: int = 1000,
                       use_mellin_transform: bool = True,
                       verbose: bool = True) -> float:
    """
    Quick function to estimate Hurst exponent using PINN.
    
    Args:
        data: Time series data
        epochs: Number of training epochs
        use_mellin_transform: Whether to use Mellin transform
        verbose: Whether to print progress
        
    Returns:
        Estimated Hurst exponent
    """
    estimator = PINNEstimator()
    results = estimator.estimate(
        data, 
        epochs=epochs,
        use_mellin_transform=use_mellin_transform,
        verbose=verbose
    )
    return results['hurst']


if __name__ == "__main__":
    # Example usage and testing
    print("Testing PINN Estimator...")
    
    # Generate test data
    np.random.seed(42)
    n_points = 1000
    hurst_true = 0.7
    
    # Generate fBm using Davies-Harte method
    freq = np.fft.fftfreq(2 * n_points, 1.0)
    power_spectrum = np.abs(freq) ** (1 - 2 * hurst_true)
    power_spectrum[0] = 0
    
    noise = np.random.normal(0, 1, 2 * n_points) + 1j * np.random.normal(0, 1, 2 * n_points)
    filtered_noise = noise * np.sqrt(power_spectrum)
    test_data = np.real(np.fft.ifft(filtered_noise))[:n_points]
    
    print(f"Generated test data with true Hurst = {hurst_true}")
    
    # Test PINN estimator
    estimator = PINNEstimator()
    results = estimator.estimate(
        test_data, 
        epochs=200,  # Reduced for testing
        use_mellin_transform=True,
        verbose=True
    )
    
    print(f"\nPINN Estimation Results:")
    print(f"Estimated Hurst: {results['hurst']:.4f}")
    print(f"True Hurst: {hurst_true:.4f}")
    print(f"Absolute Error: {abs(results['hurst'] - hurst_true):.4f}")
    print(f"Confidence Interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    
    # Plot results
    estimator.plot_training_history()
    estimator.plot_results(test_data)
    
    print("\nPINN estimator test completed successfully!")
