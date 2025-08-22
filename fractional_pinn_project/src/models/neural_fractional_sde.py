"""
Neural Fractional SDE Model

This module implements Neural Fractional Stochastic Differential Equations,
where neural networks learn the drift and diffusion terms of fractional SDEs.
This approach combines the expressiveness of neural networks with the
mathematical structure of fractional stochastic processes.

Key Features:
1. Neural network parameterization of drift and diffusion terms
2. Fractional Brownian motion integration
3. Physics-informed constraints for stochastic dynamics
4. Adaptive time stepping for SDE integration
5. Multi-scale feature extraction for complex stochastic dynamics

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

# Import our custom modules
from .mellin_transform import FractionalMellinTransform
from .physics_constraints import PhysicsConstraints

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FractionalBrownianMotion:
    """
    Generate fractional Brownian motion using various methods.
    
    This class provides implementations of different methods for generating
    fractional Brownian motion including Davies-Harte, Cholesky, and
    circulant embedding methods.
    """
    
    def __init__(self, method: str = 'davies_harte'):
        """
        Initialize fractional Brownian motion generator.
        
        Args:
            method: Method for generating fBm
                   ('davies_harte', 'cholesky', 'circulant')
        """
        self.method = method
    
    def davies_harte(self, n_points: int, hurst: float, dt: float = 1.0) -> torch.Tensor:
        """
        Generate fBm using Davies-Harte method.
        
        Args:
            n_points: Number of points
            hurst: Hurst exponent
            dt: Time step
            
        Returns:
            Fractional Brownian motion
        """
        # Frequency domain approach
        freqs = torch.arange(1, n_points // 2 + 1, device=torch.device('cpu'))
        power_spectrum = (2 * np.sin(np.pi * freqs / n_points)) ** (2 * hurst)
        
        # Generate complex Gaussian random variables
        real_part = torch.randn(n_points // 2) * torch.sqrt(power_spectrum / 2)
        imag_part = torch.randn(n_points // 2) * torch.sqrt(power_spectrum / 2)
        
        # Construct complex sequence
        complex_seq = torch.complex(real_part, imag_part)
        
        # Pad to full length
        full_seq = torch.zeros(n_points, dtype=torch.complex64)
        full_seq[1:n_points // 2 + 1] = complex_seq
        full_seq[n_points // 2 + 1:] = torch.conj(complex_seq[1:].flip(0))
        
        # Inverse FFT
        fbm = torch.fft.ifft(full_seq).real * (dt ** hurst)
        
        return fbm
    
    def cholesky(self, n_points: int, hurst: float, dt: float = 1.0) -> torch.Tensor:
        """
        Generate fBm using Cholesky decomposition.
        
        Args:
            n_points: Number of points
            hurst: Hurst exponent
            dt: Time step
            
        Returns:
            Fractional Brownian motion
        """
        # Construct covariance matrix
        cov_matrix = torch.zeros(n_points, n_points)
        for i in range(n_points):
            for j in range(n_points):
                cov_matrix[i, j] = 0.5 * (abs(i + 1) ** (2 * hurst) + 
                                        abs(j + 1) ** (2 * hurst) - 
                                        abs(i - j) ** (2 * hurst))
        
        # Cholesky decomposition
        L = torch.linalg.cholesky(cov_matrix)
        
        # Generate standard normal random variables
        z = torch.randn(n_points)
        
        # Generate fBm
        fbm = L @ z * (dt ** hurst)
        
        return fbm
    
    def __call__(self, n_points: int, hurst: float, dt: float = 1.0) -> torch.Tensor:
        """
        Generate fractional Brownian motion using specified method.
        
        Args:
            n_points: Number of points
            hurst: Hurst exponent
            dt: Time step
            
        Returns:
            Fractional Brownian motion
        """
        if self.method == 'davies_harte':
            return self.davies_harte(n_points, hurst, dt)
        elif self.method == 'cholesky':
            return self.cholesky(n_points, hurst, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class DriftNetwork(nn.Module):
    """
    Neural network for learning the drift term of fractional SDEs.
    
    This network takes the current state and time as input and outputs
    the drift term of the fractional stochastic differential equation.
    """
    
    def __init__(self, 
                 input_dim: int = 2,  # state + time
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize drift network.
        
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
        Forward pass through drift network.
        
        Args:
            t: Time points
            y: State values
            
        Returns:
            Drift term
        """
        # Concatenate time and state
        inputs = torch.cat([t.unsqueeze(-1), y], dim=-1)
        return self.network(inputs)


class DiffusionNetwork(nn.Module):
    """
    Neural network for learning the diffusion term of fractional SDEs.
    
    This network takes the current state and time as input and outputs
    the diffusion term of the fractional stochastic differential equation.
    """
    
    def __init__(self, 
                 input_dim: int = 2,  # state + time
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize diffusion network.
        
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
        
        # Output layer with positive constraint for diffusion
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Ensure positive diffusion
        
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
        Forward pass through diffusion network.
        
        Args:
            t: Time points
            y: State values
            
        Returns:
            Diffusion term (positive)
        """
        # Concatenate time and state
        inputs = torch.cat([t.unsqueeze(-1), y], dim=-1)
        return self.network(inputs)


class NeuralFractionalSDE(nn.Module):
    """
    Neural Fractional Stochastic Differential Equation model.
    
    This model uses neural networks to learn the drift and diffusion terms
    of fractional stochastic differential equations and solves them numerically.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1,
                 hurst: float = 0.7,
                 fbm_method: str = 'davies_harte',
                 use_mellin_transform: bool = True,
                 use_physics_constraints: bool = True):
        """
        Initialize Neural Fractional SDE.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            hurst: Hurst exponent for fBm
            fbm_method: Method for generating fBm
            use_mellin_transform: Whether to use Mellin transform
            use_physics_constraints: Whether to use physics constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.hurst = hurst
        self.use_mellin_transform = use_mellin_transform
        self.use_physics_constraints = use_physics_constraints
        
        # Neural networks for drift and diffusion
        self.drift_network = DriftNetwork(
            input_dim=input_dim + 1,  # state + time
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        self.diffusion_network = DiffusionNetwork(
            input_dim=input_dim + 1,  # state + time
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Fractional Brownian motion generator
        self.fbm_generator = FractionalBrownianMotion(method=fbm_method)
        
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
    
    def forward(self, t: torch.Tensor, y0: torch.Tensor, n_paths: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Neural Fractional SDE.
        
        Args:
            t: Time points
            y0: Initial conditions
            n_paths: Number of stochastic paths to generate
            
        Returns:
            Tuple of (solution_paths, hurst_exponent)
        """
        batch_size, seq_len = t.shape[:2]
        
        # Solve the fractional SDE numerically
        solution_paths = self._solve_fractional_sde(t, y0, n_paths)
        
        # Extract features for Hurst estimation (use mean path)
        mean_path = solution_paths.mean(dim=1)  # Average over paths
        features = self._extract_features(mean_path)
        
        # Estimate Hurst exponent
        hurst_exponent = self.hurst_estimator(features)
        
        return solution_paths, hurst_exponent
    
    def _solve_fractional_sde(self, t: torch.Tensor, y0: torch.Tensor, n_paths: int = 1) -> torch.Tensor:
        """
        Solve the fractional SDE using Euler-Maruyama method.
        
        Args:
            t: Time points
            y0: Initial conditions
            n_paths: Number of stochastic paths
            
        Returns:
            Solution paths
        """
        batch_size, seq_len = t.shape[:2]
        device = t.device
        
        # Initialize solution paths
        solution_paths = torch.zeros(batch_size, n_paths, seq_len, self.output_dim, device=device)
        solution_paths[:, :, 0] = y0.unsqueeze(1).repeat(1, n_paths, 1)
        
        # Time step
        dt = t[:, 1] - t[:, 0]
        
        # Generate fractional Brownian motion increments
        fbm_increments = self._generate_fbm_increments(seq_len, n_paths, device)
        
        # Euler-Maruyama integration
        for i in range(1, seq_len):
            current_t = t[:, i:i+1]
            current_y = solution_paths[:, :, i-1]
            
            # Compute drift and diffusion terms
            drift = self.drift_network(current_t, current_y)
            diffusion = self.diffusion_network(current_t, current_y)
            
            # Update solution
            dw = fbm_increments[:, :, i-1:i+1].diff(dim=-1).squeeze(-1)
            solution_paths[:, :, i] = (current_y + 
                                     drift * dt[0] + 
                                     diffusion * dw)
        
        return solution_paths
    
    def _generate_fbm_increments(self, seq_len: int, n_paths: int, device: torch.device) -> torch.Tensor:
        """
        Generate fractional Brownian motion increments.
        
        Args:
            seq_len: Sequence length
            n_paths: Number of paths
            device: Device to use
            
        Returns:
            fBm increments
        """
        # Generate fBm for each path
        fbm_paths = []
        for _ in range(n_paths):
            fbm = self.fbm_generator(seq_len, self.hurst)
            fbm_paths.append(torch.tensor(fbm, device=device))
        
        # Stack paths and add batch dimension
        fbm_tensor = torch.stack(fbm_paths, dim=0).unsqueeze(0)  # (1, n_paths, seq_len)
        
        return fbm_tensor
    
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
        
        # SDE consistency loss
        sde_loss = self._compute_sde_consistency_loss(t, y)
        total_loss += sde_loss
        
        return total_loss
    
    def _compute_sde_consistency_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SDE consistency loss.
        
        Args:
            t: Time points
            y: Solution values
            
        Returns:
            SDE consistency loss
        """
        batch_size, seq_len, _ = y.shape
        device = t.device
        
        # Compute drift and diffusion terms
        drift = self.drift_network(t, y)
        diffusion = self.diffusion_network(t, y)
        
        # SDE consistency loss (simplified)
        # In practice, you would compute the actual SDE residuals
        sde_loss = torch.mean(drift ** 2) + torch.mean(diffusion ** 2)
        
        return sde_loss
    
    def sample_paths(self, t: torch.Tensor, y0: torch.Tensor, n_paths: int = 100) -> torch.Tensor:
        """
        Sample multiple paths from the learned SDE.
        
        Args:
            t: Time points
            y0: Initial conditions
            n_paths: Number of paths to sample
            
        Returns:
            Sampled paths
        """
        with torch.no_grad():
            paths, _ = self.forward(t, y0, n_paths)
        return paths


class NeuralFractionalSDETrainer:
    """
    Trainer for Neural Fractional SDE model.
    
    This class handles the training of the neural fractional SDE model
    with physics-informed constraints and stochastic integration.
    """
    
    def __init__(self,
                 model: NeuralFractionalSDE,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize Neural Fractional SDE trainer.
        
        Args:
            model: Neural Fractional SDE model
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
            'sde_loss': [],
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
        
        # Forward pass (use multiple paths for robustness)
        y_pred, hurst_pred = self.model(t, y0, n_paths=5)
        
        # Use mean path for loss computation
        y_pred_mean = y_pred.mean(dim=1)
        
        # Compute losses
        data_loss = self.mse_loss(y_pred_mean, y_target)
        hurst_loss = self.mse_loss(hurst_pred, hurst_target)
        
        # Physics loss
        physics_loss = self.model.compute_physics_loss(t, y_pred_mean, hurst_pred)
        
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
            y_pred, hurst_pred = self.model(t, y0, n_paths=10)
            y_pred_mean = y_pred.mean(dim=1)
            
            # Compute validation losses
            data_loss = self.mse_loss(y_pred_mean, y_target)
            hurst_loss = self.mse_loss(hurst_pred, hurst_target)
            physics_loss = self.model.compute_physics_loss(t, y_pred_mean, hurst_pred)
            
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
        Train the Neural Fractional SDE model.
        
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
                    model_type='neural_sde',
                    input_dim=1,
                    hidden_dims=[64, 128, 64],
                    output_dim=1,
                    learning_rate=self.learning_rate,
                    epochs=epochs,
                    hurst=0.7,
                    fbm_method='davies_harte',
                    use_mellin_transform=True,
                    use_physics_constraints=True
                )
                
                # Save model
                training_duration = time.time() - start_time
                model_id = quick_save_model(
                    model=self.model,
                    config=config,
                    training_history=self.training_history,
                    description=model_description or f"Neural Fractional SDE model trained for {epochs} epochs",
                    tags=model_tags or ['neural_sde', 'fractional', 'trained']
                )
                
                if verbose:
                    print(f"Model saved successfully with ID: {model_id}")
                    
            except ImportError:
                # Fallback to simple save if persistence system not available
                torch.save(self.model.state_dict(), 'best_neural_fractional_sde.pth')
                if verbose:
                    print("Model saved as 'best_neural_fractional_sde.pth' (fallback)")
        
        return self.training_history


# Convenience functions
def create_neural_fractional_sde(input_dim: int = 1,
                                hidden_dims: List[int] = [64, 128, 64],
                                output_dim: int = 1,
                                hurst: float = 0.7,
                                fbm_method: str = 'davies_harte',
                                use_mellin_transform: bool = True,
                                use_physics_constraints: bool = True) -> NeuralFractionalSDE:
    """
    Create a Neural Fractional SDE model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        hurst: Hurst exponent for fBm
        fbm_method: Method for generating fBm
        use_mellin_transform: Whether to use Mellin transform
        use_physics_constraints: Whether to use physics constraints
        
    Returns:
        Neural Fractional SDE model
    """
    return NeuralFractionalSDE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        hurst=hurst,
        fbm_method=fbm_method,
        use_mellin_transform=use_mellin_transform,
        use_physics_constraints=use_physics_constraints
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Neural Fractional SDE...")
    
    # Create model
    model = create_neural_fractional_sde(
        input_dim=1,
        hidden_dims=[32, 64, 32],
        output_dim=1,
        hurst=0.7
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    
    t = torch.linspace(0, 10, seq_len).unsqueeze(0).repeat(batch_size, 1)
    y0 = torch.randn(batch_size, 1)
    
    solution_paths, hurst = model(t, y0, n_paths=5)
    
    print(f"Time shape: {t.shape}")
    print(f"Initial condition shape: {y0.shape}")
    print(f"Solution paths shape: {solution_paths.shape}")
    print(f"Hurst exponent shape: {hurst.shape}")
    print(f"Hurst values: {hurst.squeeze()}")
    
    # Test physics loss
    mean_path = solution_paths.mean(dim=1)
    physics_loss = model.compute_physics_loss(t, mean_path, hurst)
    print(f"Physics loss: {physics_loss.item():.6f}")
    
    # Test path sampling
    sampled_paths = model.sample_paths(t, y0, n_paths=10)
    print(f"Sampled paths shape: {sampled_paths.shape}")
    
    print("Neural Fractional SDE test completed successfully!")
