"""
Data Generators for Fractional PINN Project

This module provides comprehensive data generation capabilities for testing
fractional PINN models with Mellin transform integration. It includes:

1. Fractional Brownian Motion (fBm)
2. Fractional Gaussian Noise (fGn)
3. ARFIMA models
4. Multifractal Random Walk (MRW)
5. Contamination models (noise, outliers, trends, seasonality)

Author: Fractional PINN Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.signal import periodogram
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FractionalDataGenerator:
    """
    Comprehensive data generator for fractional time series.
    
    This class provides methods to generate various types of fractional
    time series data with known Hurst exponents and other parameters.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def generate_fbm(self, 
                    n_points: int = 1000,
                    hurst: float = 0.7,
                    dt: float = 1.0,
                    method: str = 'davies_harte') -> Dict[str, np.ndarray]:
        """
        Generate Fractional Brownian Motion (fBm).
        
        Args:
            n_points: Number of points to generate
            hurst: Hurst exponent (0 < H < 1)
            dt: Time step
            method: Generation method ('davies_harte', 'cholesky', 'circulant')
            
        Returns:
            Dictionary containing time series data and metadata
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")
        
        # Generate time array
        t = np.arange(0, n_points * dt, dt)
        
        if method == 'davies_harte':
            fbm = self._generate_fbm_davies_harte(n_points, hurst, dt)
        elif method == 'cholesky':
            fbm = self._generate_fbm_cholesky(n_points, hurst, dt)
        elif method == 'circulant':
            fbm = self._generate_fbm_circulant(n_points, hurst, dt)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'data': fbm,
            'time': t,
            'hurst': hurst,
            'dt': dt,
            'type': 'fbm',
            'method': method,
            'n_points': n_points
        }
    
    def _generate_fbm_davies_harte(self, n_points: int, hurst: float, dt: float) -> np.ndarray:
        """Generate fBm using Davies-Harte method."""
        # Power spectrum
        freq = np.fft.fftfreq(2 * n_points, dt)
        power_spectrum = np.abs(freq) ** (1 - 2 * hurst)
        power_spectrum[0] = 0  # DC component
        
        # Generate complex Gaussian noise
        noise = np.random.normal(0, 1, 2 * n_points) + 1j * np.random.normal(0, 1, 2 * n_points)
        
        # Apply power spectrum
        filtered_noise = noise * np.sqrt(power_spectrum)
        
        # Inverse FFT
        fbm = np.real(np.fft.ifft(filtered_noise))[:n_points]
        
        return fbm
    
    def _generate_fbm_cholesky(self, n_points: int, hurst: float, dt: float) -> np.ndarray:
        """Generate fBm using Cholesky decomposition."""
        # Covariance matrix
        t = np.arange(n_points) * dt
        cov_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                cov_matrix[i, j] = 0.5 * (abs(t[i])**(2*hurst) + abs(t[j])**(2*hurst) - 
                                        abs(t[i] - t[j])**(2*hurst))
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        
        # Generate standard normal noise
        noise = np.random.normal(0, 1, n_points)
        
        # Apply transformation
        fbm = L @ noise
        
        return fbm
    
    def _generate_fbm_circulant(self, n_points: int, hurst: float, dt: float) -> np.ndarray:
        """Generate fBm using circulant embedding."""
        # This is a simplified version - in practice, more sophisticated
        # circulant embedding methods would be used
        return self._generate_fbm_davies_harte(n_points, hurst, dt)
    
    def generate_fgn(self, 
                    n_points: int = 1000,
                    hurst: float = 0.7,
                    dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate Fractional Gaussian Noise (fGn).
        
        Args:
            n_points: Number of points to generate
            hurst: Hurst exponent (0 < H < 1)
            dt: Time step
            
        Returns:
            Dictionary containing time series data and metadata
        """
        # Generate fBm and take differences
        fbm_data = self.generate_fbm(n_points + 1, hurst, dt)
        fbm = fbm_data['data']
        
        # fGn is the increment process of fBm
        fgn = np.diff(fbm)
        
        return {
            'data': fgn,
            'time': fbm_data['time'][:-1],
            'hurst': hurst,
            'dt': dt,
            'type': 'fgn',
            'n_points': n_points
        }
    
    def generate_arfima(self, 
                       n_points: int = 1000,
                       d: float = 0.3,
                       ar_params: Optional[List[float]] = None,
                       ma_params: Optional[List[float]] = None,
                       sigma: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate ARFIMA(p,d,q) process.
        
        Args:
            n_points: Number of points to generate
            d: Fractional differencing parameter (-0.5 < d < 0.5)
            ar_params: AR parameters
            ma_params: MA parameters
            sigma: Standard deviation of innovations
            
        Returns:
            Dictionary containing time series data and metadata
        """
        if not -0.5 < d < 0.5:
            raise ValueError("Fractional differencing parameter must be between -0.5 and 0.5")
        
        # Default parameters
        if ar_params is None:
            ar_params = [0.5]
        if ma_params is None:
            ma_params = [0.3]
        
        # Generate innovations
        innovations = np.random.normal(0, sigma, n_points + 100)  # Extra points for warm-up
        
        # Fractional differencing coefficients
        frac_coeffs = self._get_fractional_coefficients(d, n_points + 100)
        
        # Apply fractional differencing
        frac_diff_series = np.convolve(innovations, frac_coeffs, mode='valid')
        
        # Apply ARMA filter
        arma_series = self._apply_arma_filter(frac_diff_series, ar_params, ma_params)
        
        # Remove warm-up period
        final_series = arma_series[100:]
        
        return {
            'data': final_series,
            'time': np.arange(len(final_series)),
            'd': d,
            'ar_params': ar_params,
            'ma_params': ma_params,
            'sigma': sigma,
            'type': 'arfima',
            'n_points': len(final_series)
        }
    
    def _get_fractional_coefficients(self, d: float, max_lag: int) -> np.ndarray:
        """Calculate fractional differencing coefficients."""
        coeffs = np.zeros(max_lag)
        coeffs[0] = 1.0
        
        for k in range(1, max_lag):
            coeffs[k] = coeffs[k-1] * (d - k + 1) / k
        
        return coeffs
    
    def _apply_arma_filter(self, 
                          series: np.ndarray, 
                          ar_params: List[float], 
                          ma_params: List[float]) -> np.ndarray:
        """Apply ARMA filter to time series."""
        p, q = len(ar_params), len(ma_params)
        n = len(series)
        
        # Initialize output
        output = np.zeros(n)
        
        # Copy input
        output[:] = series[:]
        
        # Apply AR filter
        for i in range(p, n):
            for j, ar_param in enumerate(ar_params):
                output[i] -= ar_param * output[i - j - 1]
        
        # Apply MA filter (simplified)
        if q > 0:
            ma_innovations = np.random.normal(0, 1, n)
            for i in range(q, n):
                for j, ma_param in enumerate(ma_params):
                    output[i] += ma_param * ma_innovations[i - j - 1]
        
        return output
    
    def generate_mrw(self, 
                    n_points: int = 1000,
                    hurst: float = 0.7,
                    multifractal_strength: float = 0.1,
                    dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate Multifractal Random Walk (MRW).
        
        Args:
            n_points: Number of points to generate
            hurst: Base Hurst exponent
            multifractal_strength: Strength of multifractality
            dt: Time step
            
        Returns:
            Dictionary containing time series data and metadata
        """
        # Generate base fBm
        fbm_data = self.generate_fbm(n_points, hurst, dt)
        fbm = fbm_data['data']
        
        # Generate multifractal time
        mf_time = self._generate_multifractal_time(n_points, multifractal_strength)
        
        # Apply multifractal transformation
        mrw = fbm * np.exp(mf_time)
        
        return {
            'data': mrw,
            'time': fbm_data['time'],
            'hurst': hurst,
            'multifractal_strength': multifractal_strength,
            'dt': dt,
            'type': 'mrw',
            'n_points': n_points
        }
    
    def _generate_multifractal_time(self, n_points: int, strength: float) -> np.ndarray:
        """Generate multifractal time series."""
        # Generate log-normal cascade
        cascade = np.random.lognormal(0, strength, n_points)
        
        # Normalize
        cascade = (cascade - np.mean(cascade)) / np.std(cascade)
        
        return cascade * strength
    
    def add_contamination(self, 
                         data: np.ndarray,
                         contamination_type: str,
                         intensity: float = 0.1,
                         **kwargs) -> Dict[str, np.ndarray]:
        """
        Add contamination to time series data.
        
        Args:
            data: Original time series
            contamination_type: Type of contamination
            intensity: Intensity of contamination
            **kwargs: Additional parameters for specific contamination types
            
        Returns:
            Dictionary containing contaminated data and metadata
        """
        n_points = len(data)
        contaminated_data = data.copy()
        
        if contamination_type == 'noise':
            # Add Gaussian noise
            noise_level = intensity * np.std(data)
            noise = np.random.normal(0, noise_level, n_points)
            contaminated_data += noise
            
        elif contamination_type == 'outliers':
            # Add outliers
            outlier_fraction = intensity
            n_outliers = int(outlier_fraction * n_points)
            outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
            
            outlier_magnitude = kwargs.get('outlier_magnitude', 3.0)
            outlier_std = outlier_magnitude * np.std(data)
            
            for idx in outlier_indices:
                contaminated_data[idx] += np.random.normal(0, outlier_std)
                
        elif contamination_type == 'trend':
            # Add linear trend
            trend_slope = intensity * np.std(data) / n_points
            trend = np.linspace(0, trend_slope * n_points, n_points)
            contaminated_data += trend
            
        elif contamination_type == 'seasonality':
            # Add seasonal component
            period = kwargs.get('period', n_points // 4)
            seasonal_amplitude = intensity * np.std(data)
            t = np.arange(n_points)
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)
            contaminated_data += seasonal
            
        elif contamination_type == 'missing_data':
            # Add missing data
            missing_fraction = intensity
            n_missing = int(missing_fraction * n_points)
            missing_indices = np.random.choice(n_points, n_missing, replace=False)
            contaminated_data[missing_indices] = np.nan
            
        elif contamination_type == 'heteroscedasticity':
            # Add heteroscedastic noise
            noise_std = intensity * np.std(data)
            varying_std = noise_std * (1 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_points)))
            noise = np.random.normal(0, 1, n_points) * varying_std
            contaminated_data += noise
            
        else:
            raise ValueError(f"Unknown contamination type: {contamination_type}")
        
        return {
            'data': contaminated_data,
            'original_data': data,
            'contamination_type': contamination_type,
            'intensity': intensity,
            'parameters': kwargs
        }
    
    def generate_comprehensive_dataset(self, 
                                     n_points: int = 1000,
                                     hurst_range: Tuple[float, float] = (0.1, 0.9),
                                     n_samples_per_hurst: int = 10,
                                     include_contamination: bool = True) -> Dict[str, List[Dict]]:
        """
        Generate comprehensive dataset for benchmarking.
        
        Args:
            n_points: Number of points per time series
            hurst_range: Range of Hurst exponents to test
            n_samples_per_hurst: Number of samples per Hurst exponent
            include_contamination: Whether to include contaminated versions
            
        Returns:
            Dictionary containing datasets for different models
        """
        datasets = {
            'fbm': [],
            'fgn': [],
            'arfima': [],
            'mrw': []
        }
        
        hurst_values = np.linspace(hurst_range[0], hurst_range[1], n_samples_per_hurst)
        
        for hurst in hurst_values:
            # Generate fBm
            for _ in range(n_samples_per_hurst):
                fbm_data = self.generate_fbm(n_points, hurst)
                datasets['fbm'].append(fbm_data)
                
                # Generate fGn
                fgn_data = self.generate_fgn(n_points, hurst)
                datasets['fgn'].append(fgn_data)
                
                # Generate ARFIMA
                d = hurst - 0.5  # Convert Hurst to fractional differencing parameter
                if -0.5 < d < 0.5:
                    arfima_data = self.generate_arfima(n_points, d)
                    datasets['arfima'].append(arfima_data)
                
                # Generate MRW
                mrw_data = self.generate_mrw(n_points, hurst)
                datasets['mrw'].append(mrw_data)
        
        # Add contamination if requested
        if include_contamination:
            contamination_types = ['noise', 'outliers', 'trend', 'seasonality', 'missing_data', 'heteroscedasticity']
            
            for model_type, dataset in datasets.items():
                contaminated_datasets = []
                
                for data_dict in dataset:
                    # Add each type of contamination
                    for cont_type in contamination_types:
                        contaminated = self.add_contamination(
                            data_dict['data'], 
                            cont_type, 
                            intensity=0.1
                        )
                        contaminated_datasets.append({
                            **data_dict,
                            'data': contaminated['data'],
                            'contamination': cont_type
                        })
                
                datasets[f'{model_type}_contaminated'] = contaminated_datasets
        
        return datasets
    
    def save_dataset(self, 
                    dataset: Dict[str, List[Dict]], 
                    filename: str,
                    format: str = 'npz') -> None:
        """
        Save generated dataset to file.
        
        Args:
            dataset: Dataset to save
            filename: Output filename
            format: File format ('npz', 'csv', 'json')
        """
        if format == 'npz':
            # Save as compressed numpy arrays
            save_dict = {}
            for model_type, data_list in dataset.items():
                for i, data_dict in enumerate(data_list):
                    for key, value in data_dict.items():
                        if isinstance(value, np.ndarray):
                            save_dict[f'{model_type}_{i}_{key}'] = value
                        else:
                            save_dict[f'{model_type}_{i}_{key}'] = np.array([value])
            
            np.savez_compressed(filename, **save_dict)
            
        elif format == 'csv':
            # Save as CSV (flattened)
            all_data = []
            for model_type, data_list in dataset.items():
                for i, data_dict in enumerate(data_list):
                    row = {
                        'model_type': model_type,
                        'sample_id': i,
                        'data': data_dict['data'].tolist(),
                        'parameters': str(data_dict)
                    }
                    all_data.append(row)
            
            df = pd.DataFrame(all_data)
            df.to_csv(filename, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_dataset(self, 
                    filename: str,
                    format: str = 'npz') -> Dict[str, List[Dict]]:
        """
        Load dataset from file.
        
        Args:
            filename: Input filename
            format: File format
            
        Returns:
            Loaded dataset
        """
        if format == 'npz':
            # Load from compressed numpy arrays
            loaded = np.load(filename)
            dataset = {}
            
            # Reconstruct dataset structure
            current_model = None
            current_sample = None
            current_data = {}
            
            for key in loaded.keys():
                parts = key.split('_')
                model_type = parts[0]
                sample_id = int(parts[1])
                param_name = '_'.join(parts[2:])
                
                if model_type not in dataset:
                    dataset[model_type] = []
                
                if sample_id >= len(dataset[model_type]):
                    dataset[model_type].append({})
                
                value = loaded[key]
                if value.size == 1:
                    value = value.item()
                
                dataset[model_type][sample_id][param_name] = value
            
            return dataset
            
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions for quick data generation
def generate_fbm_series(n_points: int = 1000, 
                       hurst: float = 0.7, 
                       seed: Optional[int] = None) -> np.ndarray:
    """Quick function to generate fBm series."""
    generator = FractionalDataGenerator(seed)
    return generator.generate_fbm(n_points, hurst)['data']


def generate_fgn_series(n_points: int = 1000, 
                       hurst: float = 0.7, 
                       seed: Optional[int] = None) -> np.ndarray:
    """Quick function to generate fGn series."""
    generator = FractionalDataGenerator(seed)
    return generator.generate_fgn(n_points, hurst)['data']


def generate_arfima_series(n_points: int = 1000, 
                          d: float = 0.3, 
                          seed: Optional[int] = None) -> np.ndarray:
    """Quick function to generate ARFIMA series."""
    generator = FractionalDataGenerator(seed)
    return generator.generate_arfima(n_points, d)['data']


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Fractional Data Generator...")
    
    # Initialize generator
    generator = FractionalDataGenerator(seed=42)
    
    # Generate sample data
    fbm_data = generator.generate_fbm(1000, hurst=0.7)
    fgn_data = generator.generate_fgn(1000, hurst=0.7)
    arfima_data = generator.generate_arfima(1000, d=0.3)
    mrw_data = generator.generate_mrw(1000, hurst=0.7)
    
    print(f"Generated fBm with Hurst={fbm_data['hurst']}, length={len(fbm_data['data'])}")
    print(f"Generated fGn with Hurst={fgn_data['hurst']}, length={len(fgn_data['data'])}")
    print(f"Generated ARFIMA with d={arfima_data['d']}, length={len(arfima_data['data'])}")
    print(f"Generated MRW with Hurst={mrw_data['hurst']}, length={len(mrw_data['data'])}")
    
    # Test contamination
    contaminated = generator.add_contamination(fbm_data['data'], 'noise', intensity=0.1)
    print(f"Added noise contamination, new std: {np.std(contaminated['data']):.3f}")
    
    print("Data generation test completed successfully!")
