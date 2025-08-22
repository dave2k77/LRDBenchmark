"""
Fractional PINO (Physics-Informed Neural Operator) Model

This module implements a Physics-Informed Neural Operator for fractional
time series analysis. PINO learns the mapping between function spaces,
making it particularly suitable for learning solution operators for
families of fractional differential equations.

Key Features:
1. Neural operator architecture (Fourier Neural Operator inspired)
2. Physics-informed constraints for fractional operators
3. Mellin transform integration
4. Multi-scale feature extraction
5. Operator learning for function-to-function mapping

Author: Fractional PINN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Import our custom modules
from .mellin_transform import FractionalMellinTransform
from .physics_constraints import PhysicsConstraints

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FourierLayer(nn.Module):
    """
    Fourier layer for neural operator architecture.
    
    This layer performs spectral convolution in the Fourier domain,
    which is particularly effective for learning operators on function spaces.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        """
        Initialize Fourier layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to use
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Learnable weights for Fourier domain
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier layer.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output tensor after Fourier convolution
        """
        batch_size, channels, length = x.shape
        
        # Compute FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply with learnable weights in Fourier domain
        out_ft = torch.zeros(batch_size, self.out_channels, length // 2 + 1, 
                           device=x.device, dtype=torch.cfloat)
        
        # Apply convolution in Fourier domain
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=length)
        
        return x


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution layer for neural operators.
    
    This layer combines Fourier convolution with standard convolution
    for effective operator learning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        """
        Initialize spectral convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier layer
        self.fourier_layer = FourierLayer(in_channels, out_channels, modes)
        
        # Standard convolution layer
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Fourier convolution
        x_fourier = self.fourier_layer(x)
        
        # Standard convolution
        x_conv = self.conv(x)
        
        # Combine both
        return x_fourier + x_conv


class FractionalPINO(nn.Module):
    """
    Fractional Physics-Informed Neural Operator.
    
    This model learns solution operators for fractional differential equations
    by combining neural operator architecture with physics-informed constraints.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dims: List[int] = [64, 128, 128, 64],
                 output_dim: int = 1,
                 modes: int = 16,
                 use_mellin_transform: bool = True,
                 use_physics_constraints: bool = True):
        """
        Initialize Fractional PINO.
        
        Args:
            input_dim: Input dimension (typically 1 for time series)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (typically 1 for Hurst exponent)
            modes: Number of Fourier modes for spectral layers
            use_mellin_transform: Whether to use Mellin transform
            use_physics_constraints: Whether to use physics constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.modes = modes
        self.use_mellin_transform = use_mellin_transform
        self.use_physics_constraints = use_physics_constraints
        
        # Build neural operator architecture
        self._build_operator_layers()
        
        # Initialize physics components
        if use_mellin_transform:
            self.mellin_transform = FractionalMellinTransform()
        
        if use_physics_constraints:
            self.physics_constraints = PhysicsConstraints()
        
        # Multi-scale feature extraction
        self._build_multi_scale_features()
        
        # Output projection for Hurst exponent estimation
        self.hurst_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # Ensure Hurst exponent is between 0 and 1
        )
    
    def _build_operator_layers(self):
        """Build the neural operator layers."""
        layers = []
        
        # Input projection
        layers.append(nn.Conv1d(self.input_dim, self.hidden_dims[0], 1))
        
        # Spectral convolution layers
        for i in range(len(self.hidden_dims) - 1):
            layers.extend([
                SpectralConv1d(self.hidden_dims[i], self.hidden_dims[i + 1], self.modes),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.operator_layers = nn.ModuleList(layers)
    
    def _build_multi_scale_features(self):
        """Build multi-scale feature extraction layers."""
        self.multi_scale_layers = nn.ModuleList([
            nn.Conv1d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]
        ])
        
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dims[-1],
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Fractional PINO.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tuple of (function_output, hurst_exponent)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for 1D convolution: (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # Apply neural operator layers
        for layer in self.operator_layers:
            x = layer(x)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv_layer in self.multi_scale_layers:
            features = conv_layer(x)
            multi_scale_features.append(features)
        
        # Combine multi-scale features
        if len(multi_scale_features) > 1:
            # Use attention mechanism to combine scales
            combined_features = torch.stack(multi_scale_features, dim=1)  # (batch, scales, channels, length)
            combined_features = combined_features.mean(dim=1)  # (batch, channels, length)
        else:
            combined_features = multi_scale_features[0]
        
        # Global average pooling for function representation
        function_representation = F.adaptive_avg_pool1d(combined_features, 1).squeeze(-1)
        
        # Estimate Hurst exponent
        hurst_exponent = self.hurst_projection(function_representation)
        
        # Reshape back to original format
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        return x, hurst_exponent
    
    def compute_physics_loss(self, 
                           t: torch.Tensor, 
                           y: torch.Tensor, 
                           hurst: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            t: Time points
            y: Function values
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
        
        return total_loss
    
    def operator_forward(self, 
                        input_function: torch.Tensor, 
                        target_points: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned operator to map input function to output function.
        
        Args:
            input_function: Input function values
            target_points: Points where to evaluate the output function
            
        Returns:
            Output function values
        """
        # This is a simplified version - in practice, you would implement
        # the full operator mapping here
        batch_size, seq_len, _ = input_function.shape
        
        # Apply the neural operator
        output_function, hurst = self.forward(input_function)
        
        # Interpolate to target points if needed
        if target_points.shape[1] != seq_len:
            # Simple linear interpolation (could be improved)
            output_function = F.interpolate(
                output_function.transpose(1, 2),
                size=target_points.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        return output_function


class FractionalPINOTrainer:
    """
    Trainer for Fractional PINO model.
    
    This class handles the training of the PINO model with physics-informed
    constraints and operator learning objectives.
    """
    
    def __init__(self,
                 model: FractionalPINO,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize PINO trainer.
        
        Args:
            model: Fractional PINO model
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
        self.operator_loss = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'operator_loss': [],
            'hurst_loss': []
        }
    
    def train_step(self, 
                   batch_data: torch.Tensor,
                   batch_targets: torch.Tensor,
                   batch_hurst: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch_data: Input data
            batch_targets: Target function values
            batch_hurst: True Hurst exponents
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output_function, predicted_hurst = self.model(batch_data)
        
        # Compute losses
        data_loss = self.mse_loss(output_function, batch_targets)
        hurst_loss = self.mse_loss(predicted_hurst, batch_hurst)
        
        # Physics loss
        physics_loss = self.model.compute_physics_loss(
            batch_data, output_function, predicted_hurst
        )
        
        # Operator loss (simplified - could be enhanced)
        operator_loss = self.operator_loss(output_function, batch_targets)
        
        # Total loss
        total_loss = data_loss + 0.1 * physics_loss + 0.1 * operator_loss + 0.05 * hurst_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'operator_loss': operator_loss.item(),
            'hurst_loss': hurst_loss.item()
        }
    
    def validate(self, 
                 val_data: torch.Tensor,
                 val_targets: torch.Tensor,
                 val_hurst: torch.Tensor) -> Dict[str, float]:
        """
        Perform validation.
        
        Args:
            val_data: Validation data
            val_targets: Validation targets
            val_hurst: Validation Hurst exponents
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            output_function, predicted_hurst = self.model(val_data)
            
            # Compute validation losses
            data_loss = self.mse_loss(output_function, val_targets)
            hurst_loss = self.mse_loss(predicted_hurst, val_hurst)
            physics_loss = self.model.compute_physics_loss(
                val_data, output_function, predicted_hurst
            )
            
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
        Train the PINO model.
        
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
            epoch_losses = {'total_loss': 0, 'data_loss': 0, 'physics_loss': 0, 
                          'operator_loss': 0, 'hurst_loss': 0}
            num_batches = 0
            
            for batch_data, batch_targets, batch_hurst in train_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_hurst = batch_hurst.to(self.device)
                
                losses = self.train_step(batch_data, batch_targets, batch_hurst)
                
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
            
            for batch_data, batch_targets, batch_hurst in val_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_hurst = batch_hurst.to(self.device)
                
                val_metrics = self.validate(batch_data, batch_targets, batch_hurst)
                
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
                    model_type='pino',
                    input_dim=1,
                    hidden_dims=[64, 128, 128, 64],
                    output_dim=1,
                    learning_rate=self.learning_rate,
                    epochs=epochs,
                    modes=16,
                    use_mellin_transform=True,
                    use_physics_constraints=True
                )
                
                # Save model
                training_duration = time.time() - start_time
                model_id = quick_save_model(
                    model=self.model,
                    config=config,
                    training_history=self.training_history,
                    description=model_description or f"Fractional PINO model trained for {epochs} epochs",
                    tags=model_tags or ['pino', 'fractional', 'trained']
                )
                
                if verbose:
                    print(f"Model saved successfully with ID: {model_id}")
                    
            except ImportError:
                # Fallback to simple save if persistence system not available
                torch.save(self.model.state_dict(), 'best_fractional_pino.pth')
                if verbose:
                    print("Model saved as 'best_fractional_pino.pth' (fallback)")
        
        return self.training_history


# Convenience functions
def create_fractional_pino(input_dim: int = 1,
                          hidden_dims: List[int] = [64, 128, 128, 64],
                          output_dim: int = 1,
                          modes: int = 16,
                          use_mellin_transform: bool = True,
                          use_physics_constraints: bool = True) -> FractionalPINO:
    """
    Create a Fractional PINO model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        modes: Number of Fourier modes
        use_mellin_transform: Whether to use Mellin transform
        use_physics_constraints: Whether to use physics constraints
        
    Returns:
        Fractional PINO model
    """
    return FractionalPINO(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        modes=modes,
        use_mellin_transform=use_mellin_transform,
        use_physics_constraints=use_physics_constraints
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Fractional PINO...")
    
    # Create model
    model = create_fractional_pino(
        input_dim=1,
        hidden_dims=[32, 64, 64, 32],
        output_dim=1,
        modes=8
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    
    x = torch.randn(batch_size, seq_len, 1)
    output_function, hurst = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output function shape: {output_function.shape}")
    print(f"Hurst exponent shape: {hurst.shape}")
    print(f"Hurst values: {hurst.squeeze()}")
    
    # Test physics loss
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    physics_loss = model.compute_physics_loss(t, output_function, hurst)
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    print("Fractional PINO test completed successfully!")
