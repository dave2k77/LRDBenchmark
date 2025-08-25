#!/usr/bin/env python3
"""
HPFracc Data Generator

This script generates synthetic time series data in the exact format required
by hpfracc fractional neural networks for time series prediction tasks.
"""

import numpy as np
import jax.numpy as jnp
from scipy.special import gamma
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class HPFraccDataGenerator:
    """
    Generates synthetic time series data optimized for hpfracc models.
    """
    
    def __init__(self, 
                 series_length: int = 1000,
                 batch_size: int = 32,
                 input_window: int = 10,
                 prediction_horizon: int = 1):
        """
        Initialize the data generator.
        
        Parameters
        ----------
        series_length : int
            Length of each time series
        batch_size : int
            Number of samples per batch
        input_window : int
            Number of time steps to use as input (lookback window)
        prediction_horizon : int
            Number of time steps to predict ahead
        """
        self.series_length = series_length
        self.batch_size = batch_size
        self.input_window = input_window
        self.prediction_horizon = prediction_horizon
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def generate_fbm_data(self, H: float = 0.7, sigma: float = 1.0) -> np.ndarray:
        """
        Generate Fractional Brownian Motion data.
        
        Parameters
        ----------
        H : float
            Hurst parameter (0 < H < 1)
        sigma : float
            Standard deviation of the process
            
        Returns
        -------
        np.ndarray
            Generated time series of shape (series_length,)
        """
        n = self.series_length
        
        # Generate increments
        increments = np.random.normal(0, sigma, n)
        
        # Apply fractional integration
        if H != 0.5:
            # Use power law scaling
            t = np.arange(1, n + 1)
            weights = (t ** (H - 0.5)) / gamma(H + 0.5)
            fbm = np.convolve(increments, weights, mode='full')[:n]
        else:
            # Standard Brownian motion
            fbm = np.cumsum(increments)
        
        return fbm
    
    def generate_fgn_data(self, H: float = 0.7, sigma: float = 1.0) -> np.ndarray:
        """
        Generate Fractional Gaussian Noise data.
        
        Parameters
        ----------
        H : float
            Hurst parameter (0 < H < 1)
        sigma : float
            Standard deviation of the process
            
        Returns
        -------
        np.ndarray
            Generated time series of shape (series_length,)
        """
        n = self.series_length
        
        # Generate FBM first
        fbm = self.generate_fbm_data(H, sigma)
        
        # Take differences to get FGN
        fgn = np.diff(fbm)
        
        # Pad to maintain length
        fgn = np.concatenate([[0], fgn])
        
        return fgn
    
    def generate_arfima_data(self, d: float = 0.3, sigma: float = 1.0) -> np.ndarray:
        """
        Generate ARFIMA data.
        
        Parameters
        ----------
        d : float
            Fractional integration parameter (-0.5 < d < 0.5)
        sigma : float
            Standard deviation of innovations
            
        Returns
        -------
        np.ndarray
            Generated time series of shape (series_length,)
        """
        n = self.series_length
        
        # Generate white noise
        innovations = np.random.normal(0, sigma, n)
        
        # Apply fractional integration
        if abs(d) > 1e-6:
            # Use binomial expansion for fractional differencing
            t = np.arange(n)
            weights = np.zeros(n)
            weights[0] = 1.0
            
            for i in range(1, n):
                weights[i] = weights[i-1] * (d + i - 1) / i
            
            # Apply weights
            arfima = np.convolve(innovations, weights, mode='full')[:n]
        else:
            arfima = innovations
        
        return arfima
    
    def generate_mrw_data(self, H: float = 0.7, lambda_param: float = 0.1, 
                          sigma: float = 1.0) -> np.ndarray:
        """
        Generate Multifractal Random Walk data.
        
        Parameters
        ----------
        H : float
            Hurst parameter (0 < H < 1)
        lambda_param : float
            Multifractality parameter
        sigma : float
            Standard deviation of the process
            
        Returns
        -------
        np.ndarray
            Generated time series of shape (series_length,)
        """
        n = self.series_length
        
        # Generate base process (FBM)
        base_process = self.generate_fbm_data(H, sigma)
        
        # Apply multifractal modulation
        modulation = np.exp(lambda_param * np.random.normal(0, 1, n))
        mrw = base_process * modulation
        
        return mrw
    
    def create_sliding_windows(self, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window input-output pairs for time series prediction.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series of shape (series_length,)
            
        Returns
        -------
        tuple
            (X, y) where:
            - X: Input windows of shape (n_samples, input_window, 1)
            - y: Target values of shape (n_samples, prediction_horizon, 1)
        """
        n = len(time_series)
        
        # Calculate number of samples
        n_samples = n - self.input_window - self.prediction_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Time series too short. Need at least {self.input_window + self.prediction_horizon} samples")
        
        # Initialize arrays
        X = np.zeros((n_samples, self.input_window, 1))
        y = np.zeros((n_samples, self.prediction_horizon, 1))
        
        # Create sliding windows
        for i in range(n_samples):
            # Input window
            X[i, :, 0] = time_series[i:i + self.input_window]
            
            # Target values
            y[i, :, 0] = time_series[i + self.input_window:i + self.input_window + self.prediction_horizon]
        
        return X, y
    
    def create_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create batches from input-output pairs.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_window, 1)
        y : np.ndarray
            Target data of shape (n_samples, prediction_horizon, 1)
            
        Returns
        -------
        list
            List of (X_batch, y_batch) tuples
        """
        n_samples = len(X)
        batches = []
        
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            batches.append((X_batch, y_batch))
        
        return batches
    
    def generate_dataset(self, model_type: str = 'fbm', **kwargs) -> Dict[str, Any]:
        """
        Generate a complete dataset for a specific model type.
        
        Parameters
        ----------
        model_type : str
            Type of model ('fbm', 'fgn', 'arfima', 'mrw')
        **kwargs
            Model-specific parameters
            
        Returns
        -------
        dict
            Dataset containing:
            - 'raw_data': Original time series
            - 'X': Input windows
            - 'y': Target values
            - 'batches': List of batched data
            - 'metadata': Generation parameters
        """
        print(f"📊 Generating {model_type.upper()} dataset...")
        
        # Generate raw time series
        if model_type == 'fbm':
            raw_data = self.generate_fbm_data(**kwargs)
        elif model_type == 'fgn':
            raw_data = self.generate_fgn_data(**kwargs)
        elif model_type == 'arfima':
            raw_data = self.generate_arfima_data(**kwargs)
        elif model_type == 'mrw':
            raw_data = self.generate_mrw_data(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create sliding windows
        X, y = self.create_sliding_windows(raw_data)
        
        # Create batches
        batches = self.create_batches(X, y)
        
        # Compile dataset
        dataset = {
            'raw_data': raw_data,
            'X': X,
            'y': y,
            'batches': batches,
            'metadata': {
                'model_type': model_type,
                'series_length': self.series_length,
                'input_window': self.input_window,
                'prediction_horizon': self.prediction_horizon,
                'batch_size': self.batch_size,
                'n_samples': len(X),
                'n_batches': len(batches),
                **kwargs
            }
        }
        
        print(f"  ✅ Generated {len(X)} samples in {len(batches)} batches")
        print(f"  📏 Input shape: {X.shape}, Target shape: {y.shape}")
        
        return dataset
    
    def generate_comprehensive_dataset(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset with multiple model types.
        
        Returns
        -------
        dict
            Dictionary containing datasets for all model types
        """
        print("🚀 Generating Comprehensive Dataset")
        print("=" * 50)
        
        datasets = {}
        
        # Generate FBM datasets with different H values
        for H in [0.3, 0.5, 0.7, 0.9]:
            datasets[f'fbm_H{H}'] = self.generate_dataset('fbm', H=H, sigma=1.0)
        
        # Generate FGN datasets with different H values
        for H in [0.3, 0.5, 0.7, 0.9]:
            datasets[f'fgn_H{H}'] = self.generate_dataset('fgn', H=H, sigma=1.0)
        
        # Generate ARFIMA datasets with different d values
        for d in [0.1, 0.2, 0.3, 0.4]:
            datasets[f'arfima_d{d}'] = self.generate_dataset('arfima', d=d, sigma=1.0)
        
        # Generate MRW datasets with different H values
        for H in [0.3, 0.5, 0.7, 0.9]:
            datasets[f'mrw_H{H}'] = self.generate_dataset('mrw', H=H, lambda_param=0.1, sigma=1.0)
        
        print(f"\n✅ Generated {len(datasets)} datasets")
        
        return datasets
    
    def visualize_dataset(self, dataset: Dict[str, Any], save_path: str = None):
        """
        Visualize the generated dataset.
        
        Parameters
        ----------
        dataset : dict
            Dataset to visualize
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Dataset: {dataset['metadata']['model_type']}", fontsize=16)
        
        # Raw time series
        axes[0, 0].plot(dataset['raw_data'])
        axes[0, 0].set_title('Raw Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)
        
        # Input windows (first few samples)
        n_show = min(5, len(dataset['X']))
        for i in range(n_show):
            axes[0, 1].plot(dataset['X'][i, :, 0], alpha=0.7, label=f'Sample {i+1}')
        axes[0, 1].set_title(f'Input Windows (First {n_show} Samples)')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Target values distribution
        axes[1, 0].hist(dataset['y'].flatten(), bins=30, alpha=0.7)
        axes[1, 0].set_title('Target Values Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Sample input-output relationship
        sample_idx = 0
        axes[1, 1].plot(dataset['X'][sample_idx, :, 0], 'b-', label='Input', linewidth=2)
        axes[1, 1].axvline(x=len(dataset['X'][sample_idx])-1, color='r', linestyle='--', label='Prediction Point')
        axes[1, 1].scatter(len(dataset['X'][sample_idx])-1, dataset['y'][sample_idx, 0, 0], 
                           color='r', s=100, zorder=5, label='Target')
        axes[1, 1].set_title('Sample Input-Output Relationship')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")
        
        plt.show()

def main():
    """Main function to demonstrate the data generator."""
    
    # Create data generator
    generator = HPFraccDataGenerator(
        series_length=1000,
        batch_size=32,
        input_window=10,
        prediction_horizon=1
    )
    
    # Generate comprehensive dataset
    datasets = generator.generate_comprehensive_dataset()
    
    # Visualize a sample dataset
    sample_key = list(datasets.keys())[0]
    print(f"\n🎨 Visualizing {sample_key}...")
    generator.visualize_dataset(datasets[sample_key])
    
    # Print dataset statistics
    print("\n📊 Dataset Statistics:")
    print("=" * 40)
    for name, dataset in datasets.items():
        metadata = dataset['metadata']
        print(f"{name}:")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Batches: {metadata['n_batches']}")
        print(f"  Input shape: {dataset['X'].shape}")
        print(f"  Target shape: {dataset['y'].shape}")
        print(f"  Raw data stats: μ={np.mean(dataset['raw_data']):.3f}, σ={np.std(dataset['raw_data']):.3f}")
        print()

if __name__ == "__main__":
    main()
