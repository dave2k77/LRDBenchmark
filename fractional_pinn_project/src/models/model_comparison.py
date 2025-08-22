"""
Comprehensive Model Comparison Framework

This module provides a unified framework for comparing different neural approaches
for fractional time series analysis, including PINN, PINO, Neural Fractional ODE,
and Neural Fractional SDE models.

Key Features:
1. Unified interface for all neural models
2. Systematic evaluation and comparison
3. Performance metrics computation
4. Visualization and reporting
5. Hyperparameter optimization support

Author: Fractional PINN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import json

# Import our custom modules
from .fractional_pinn import FractionalPINN
from .fractional_pino import FractionalPINO
from .neural_fractional_ode import NeuralFractionalODE
from .neural_fractional_sde import NeuralFractionalSDE
from .mellin_transform import FractionalMellinTransform
from .physics_constraints import PhysicsConstraints

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for neural models."""
    model_type: str  # 'pinn', 'pino', 'neural_ode', 'neural_sde'
    input_dim: int = 1
    hidden_dims: List[int] = None
    output_dim: int = 1
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 32
    early_stopping_patience: int = 50
    use_mellin_transform: bool = True
    use_physics_constraints: bool = True
    
    # Model-specific parameters
    alpha: float = 0.5  # For Neural ODE
    hurst: float = 0.7  # For Neural SDE
    modes: int = 16  # For PINO
    fbm_method: str = 'davies_harte'  # For Neural SDE
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for model comparison."""
    model_name: str
    hurst_mae: float
    hurst_rmse: float
    hurst_r2: float
    training_time: float
    inference_time: float
    memory_usage: float
    convergence_epochs: int
    final_loss: float
    physics_loss: float
    mellin_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'hurst_mae': self.hurst_mae,
            'hurst_rmse': self.hurst_rmse,
            'hurst_r2': self.hurst_r2,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage,
            'convergence_epochs': self.convergence_epochs,
            'final_loss': self.final_loss,
            'physics_loss': self.physics_loss,
            'mellin_loss': self.mellin_loss
        }


class BaseNeuralModel(ABC):
    """Abstract base class for all neural models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the neural model."""
        pass
    
    @abstractmethod
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        pass
    
    @abstractmethod
    def estimate_hurst(self, data: torch.Tensor) -> float:
        """Estimate Hurst exponent."""
        pass


class PINNModel(BaseNeuralModel):
    """PINN model wrapper."""
    
    def build_model(self) -> None:
        """Build PINN model."""
        from ..estimators.pinn_estimator import PINNEstimator
        
        self.model = PINNEstimator(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            learning_rate=self.config.learning_rate,
            device='auto'
        )
        
        self.model.build_model(
            use_mellin_transform=self.config.use_mellin_transform,
            use_physics_constraints=self.config.use_physics_constraints
        )
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Train PINN model."""
        return self.model.train(
            train_loader, val_loader,
            epochs=self.config.epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with PINN."""
        self.model.model.eval()
        with torch.no_grad():
            return self.model.model(data)
    
    def estimate_hurst(self, data: torch.Tensor) -> float:
        """Estimate Hurst exponent with PINN."""
        result = self.model.estimate(data.numpy())
        return result['hurst_exponent']


class PINOModel(BaseNeuralModel):
    """PINO model wrapper."""
    
    def build_model(self) -> None:
        """Build PINO model."""
        from .fractional_pino import create_fractional_pino
        
        self.model = create_fractional_pino(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            modes=self.config.modes,
            use_mellin_transform=self.config.use_mellin_transform,
            use_physics_constraints=self.config.use_physics_constraints
        )
        
        self.trainer = self.model.FractionalPINOTrainer(
            self.model, self.config.learning_rate, 'auto'
        )
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Train PINO model."""
        return self.trainer.train(
            train_loader, val_loader,
            epochs=self.config.epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with PINO."""
        self.model.eval()
        with torch.no_grad():
            output_function, hurst = self.model(data)
            return output_function
    
    def estimate_hurst(self, data: torch.Tensor) -> float:
        """Estimate Hurst exponent with PINO."""
        self.model.eval()
        with torch.no_grad():
            _, hurst = self.model(data)
            return hurst.mean().item()


class NeuralODEModel(BaseNeuralModel):
    """Neural Fractional ODE model wrapper."""
    
    def build_model(self) -> None:
        """Build Neural Fractional ODE model."""
        from .neural_fractional_ode import create_neural_fractional_ode
        
        self.model = create_neural_fractional_ode(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            alpha=self.config.alpha,
            use_mellin_transform=self.config.use_mellin_transform,
            use_physics_constraints=self.config.use_physics_constraints
        )
        
        self.trainer = self.model.NeuralFractionalODETrainer(
            self.model, self.config.learning_rate, 'auto'
        )
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Train Neural Fractional ODE model."""
        return self.trainer.train(
            train_loader, val_loader,
            epochs=self.config.epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with Neural Fractional ODE."""
        self.model.eval()
        with torch.no_grad():
            # For ODE, we need time points and initial conditions
            t = torch.linspace(0, 1, data.shape[1], device=data.device).unsqueeze(0)
            y0 = data[:, 0:1]
            solution, _ = self.model(t, y0)
            return solution
    
    def estimate_hurst(self, data: torch.Tensor) -> float:
        """Estimate Hurst exponent with Neural Fractional ODE."""
        self.model.eval()
        with torch.no_grad():
            t = torch.linspace(0, 1, data.shape[1], device=data.device).unsqueeze(0)
            y0 = data[:, 0:1]
            _, hurst = self.model(t, y0)
            return hurst.mean().item()


class NeuralSDEModel(BaseNeuralModel):
    """Neural Fractional SDE model wrapper."""
    
    def build_model(self) -> None:
        """Build Neural Fractional SDE model."""
        from .neural_fractional_sde import create_neural_fractional_sde
        
        self.model = create_neural_fractional_sde(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            hurst=self.config.hurst,
            fbm_method=self.config.fbm_method,
            use_mellin_transform=self.config.use_mellin_transform,
            use_physics_constraints=self.config.use_physics_constraints
        )
        
        self.trainer = self.model.NeuralFractionalSDETrainer(
            self.model, self.config.learning_rate, 'auto'
        )
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Train Neural Fractional SDE model."""
        return self.trainer.train(
            train_loader, val_loader,
            epochs=self.config.epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with Neural Fractional SDE."""
        self.model.eval()
        with torch.no_grad():
            # For SDE, we need time points and initial conditions
            t = torch.linspace(0, 1, data.shape[1], device=data.device).unsqueeze(0)
            y0 = data[:, 0:1]
            solution_paths, _ = self.model(t, y0, n_paths=1)
            return solution_paths.squeeze(1)  # Remove path dimension
    
    def estimate_hurst(self, data: torch.Tensor) -> float:
        """Estimate Hurst exponent with Neural Fractional SDE."""
        self.model.eval()
        with torch.no_grad():
            t = torch.linspace(0, 1, data.shape[1], device=data.device).unsqueeze(0)
            y0 = data[:, 0:1]
            _, hurst = self.model(t, y0, n_paths=5)
            return hurst.mean().item()


class ModelComparisonFramework:
    """
    Comprehensive framework for comparing neural models for fractional time series analysis.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the comparison framework.
        
        Args:
            device: Device to use for computation
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.models = {}
        self.results = {}
        self.metrics = []
    
    def add_model(self, config: ModelConfig) -> None:
        """
        Add a model to the comparison framework.
        
        Args:
            config: Model configuration
        """
        model_name = f"{config.model_type}_{len(self.models)}"
        
        if config.model_type == 'pinn':
            model = PINNModel(config)
        elif config.model_type == 'pino':
            model = PINOModel(config)
        elif config.model_type == 'neural_ode':
            model = NeuralODEModel(config)
        elif config.model_type == 'neural_sde':
            model = NeuralSDEModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.models[model_name] = model
    
    def train_all_models(self, train_loader, val_loader) -> Dict[str, Dict[str, List[float]]]:
        """
        Train all models in the framework.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary of training histories
        """
        training_histories = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Build model
            model.build_model()
            
            # Train model
            start_time = time.time()
            history = model.train(train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Store results
            training_histories[model_name] = history
            self.results[model_name] = {
                'history': history,
                'training_time': training_time
            }
        
        return training_histories
    
    def evaluate_models(self, test_data: torch.Tensor, true_hurst: np.ndarray) -> List[EvaluationMetrics]:
        """
        Evaluate all models on test data.
        
        Args:
            test_data: Test data
            true_hurst: True Hurst exponents
            
        Returns:
            List of evaluation metrics
        """
        metrics_list = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Inference time
            start_time = time.time()
            predicted_hurst = []
            
            for i in range(len(test_data)):
                hurst_est = model.estimate_hurst(test_data[i:i+1])
                predicted_hurst.append(hurst_est)
            
            inference_time = time.time() - start_time
            
            # Convert to numpy arrays
            predicted_hurst = np.array(predicted_hurst)
            
            # Compute metrics
            hurst_mae = np.mean(np.abs(predicted_hurst - true_hurst))
            hurst_rmse = np.sqrt(np.mean((predicted_hurst - true_hurst) ** 2))
            hurst_r2 = 1 - np.sum((predicted_hurst - true_hurst) ** 2) / np.sum((true_hurst - np.mean(true_hurst)) ** 2)
            
            # Memory usage (approximate)
            if hasattr(model.model, 'parameters'):
                memory_usage = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024 / 1024  # MB
            else:
                memory_usage = 0.0
            
            # Training metrics
            history = self.results[model_name]['history']
            training_time = self.results[model_name]['training_time']
            convergence_epochs = len(history.get('total_loss', []))
            final_loss = history.get('total_loss', [0])[-1] if history.get('total_loss') else 0
            physics_loss = history.get('physics_loss', [0])[-1] if history.get('physics_loss') else 0
            mellin_loss = history.get('mellin_loss', [0])[-1] if history.get('mellin_loss') else 0
            
            # Create metrics object
            metrics = EvaluationMetrics(
                model_name=model_name,
                hurst_mae=hurst_mae,
                hurst_rmse=hurst_rmse,
                hurst_r2=hurst_r2,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_usage,
                convergence_epochs=convergence_epochs,
                final_loss=final_loss,
                physics_loss=physics_loss,
                mellin_loss=mellin_loss
            )
            
            metrics_list.append(metrics)
        
        self.metrics = metrics_list
        return metrics_list
    
    def create_comparison_report(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive comparison report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Comparison report as DataFrame
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate_models first.")
        
        # Create DataFrame
        report_data = [metrics.to_dict() for metrics in self.metrics]
        report_df = pd.DataFrame(report_data)
        
        # Sort by Hurst MAE (lower is better)
        report_df = report_df.sort_values('hurst_mae')
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"Comparison report saved to {save_path}")
        
        return report_df
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create comparison plots.
        
        Args:
            save_path: Path to save the plots
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate_models first.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Neural Model Comparison for Fractional Time Series Analysis', fontsize=16)
        
        # Extract data
        model_names = [m.model_name for m in self.metrics]
        hurst_mae = [m.hurst_mae for m in self.metrics]
        hurst_rmse = [m.hurst_rmse for m in self.metrics]
        hurst_r2 = [m.hurst_r2 for m in self.metrics]
        training_times = [m.training_time for m in self.metrics]
        inference_times = [m.inference_time for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics]
        
        # Plot 1: Hurst MAE
        axes[0, 0].bar(model_names, hurst_mae, color='skyblue')
        axes[0, 0].set_title('Hurst Exponent MAE')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Hurst RMSE
        axes[0, 1].bar(model_names, hurst_rmse, color='lightcoral')
        axes[0, 1].set_title('Hurst Exponent RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Hurst R²
        axes[0, 2].bar(model_names, hurst_r2, color='lightgreen')
        axes[0, 2].set_title('Hurst Exponent R²')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Training Time
        axes[1, 0].bar(model_names, training_times, color='gold')
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Inference Time
        axes[1, 1].bar(model_names, inference_times, color='plum')
        axes[1, 1].set_title('Inference Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Memory Usage
        axes[1, 2].bar(model_names, memory_usage, color='lightblue')
        axes[1, 2].set_title('Memory Usage')
        axes[1, 2].set_ylabel('Memory (MB)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves for all models.
        
        Args:
            save_path: Path to save the plots
        """
        if not self.results:
            raise ValueError("No training results available. Run train_all_models first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Comparison', fontsize=16)
        
        # Plot training curves
        for i, (model_name, result) in enumerate(self.results.items()):
            history = result['history']
            
            # Total loss
            if 'total_loss' in history:
                axes[0, 0].plot(history['total_loss'], label=model_name, alpha=0.8)
            
            # Physics loss
            if 'physics_loss' in history:
                axes[0, 1].plot(history['physics_loss'], label=model_name, alpha=0.8)
            
            # Validation loss
            if 'val_total_loss' in history:
                axes[1, 0].plot(history['val_total_loss'], label=model_name, alpha=0.8)
            
            # Hurst loss
            if 'hurst_loss' in history:
                axes[1, 1].plot(history['hurst_loss'], label=model_name, alpha=0.8)
        
        # Set labels and titles
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].set_title('Physics Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].set_title('Hurst Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


# Convenience functions
def create_comprehensive_comparison() -> ModelComparisonFramework:
    """
    Create a comprehensive comparison framework with all neural models.
    
    Returns:
        Model comparison framework
    """
    framework = ModelComparisonFramework()
    
    # Add PINN model
    pinn_config = ModelConfig(
        model_type='pinn',
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        epochs=1000
    )
    framework.add_model(pinn_config)
    
    # Add PINO model
    pino_config = ModelConfig(
        model_type='pino',
        hidden_dims=[64, 128, 128, 64],
        learning_rate=1e-3,
        epochs=1000,
        modes=16
    )
    framework.add_model(pino_config)
    
    # Add Neural Fractional ODE model
    neural_ode_config = ModelConfig(
        model_type='neural_ode',
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        epochs=1000,
        alpha=0.5
    )
    framework.add_model(neural_ode_config)
    
    # Add Neural Fractional SDE model
    neural_sde_config = ModelConfig(
        model_type='neural_sde',
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        epochs=1000,
        hurst=0.7,
        fbm_method='davies_harte'
    )
    framework.add_model(neural_sde_config)
    
    return framework


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Comparison Framework...")
    
    # Create framework
    framework = create_comprehensive_comparison()
    
    print(f"Created framework with {len(framework.models)} models:")
    for model_name in framework.models.keys():
        print(f"  - {model_name}")
    
    print("Model comparison framework test completed successfully!")
