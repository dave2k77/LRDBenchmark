"""
Classical Estimators for Fractional Time Series Analysis

This module provides comprehensive implementations of classical methods for
estimating Hurst exponents and fractional parameters in time series data.
These estimators serve as benchmarks for the fractional PINN approach.

Methods included:
1. Detrended Fluctuation Analysis (DFA)
2. Rescaled Range Analysis (R/S)
3. Wavelet-based methods (CWT, Variance, Log-Variance)
4. Spectral methods (GPH, Whittle, Periodogram)
5. Higuchi method
6. DMA (Detrending Moving Average)

Author: Fractional PINN Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, signal
from scipy.optimize import minimize
from scipy.signal import periodogram, welch
import pywt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ClassicalEstimator:
    """
    Base class for classical estimators.
    
    Provides common functionality and interface for all classical
    estimation methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the estimator.
        
        Args:
            name: Name of the estimation method
        """
        self.name = name
        self.results = {}
    
    def estimate(self, data: np.ndarray, **kwargs) -> Dict:
        """
        Estimate parameters from time series data.
        
        Args:
            data: Time series data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing estimation results
        """
        raise NotImplementedError("Subclasses must implement estimate method")
    
    def get_hurst_exponent(self) -> Optional[float]:
        """Get the estimated Hurst exponent."""
        return self.results.get('hurst', None)
    
    def get_confidence_interval(self) -> Optional[Tuple[float, float]]:
        """Get confidence interval for the estimate."""
        return self.results.get('confidence_interval', None)
    
    def get_r_squared(self) -> Optional[float]:
        """Get R-squared value of the fit."""
        return self.results.get('r_squared', None)


class DFAEstimator(ClassicalEstimator):
    """
    Detrended Fluctuation Analysis (DFA) estimator.
    
    DFA is a method for determining the statistical self-affinity of a signal.
    It is useful for analyzing time series that appear to be long-memory processes.
    """
    
    def __init__(self):
        super().__init__("DFA")
    
    def estimate(self, 
                data: np.ndarray,
                min_scale: int = 4,
                max_scale: int = None,
                n_scales: int = 20,
                polynomial_order: int = 1) -> Dict:
        """
        Estimate Hurst exponent using DFA.
        
        Args:
            data: Time series data
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
            n_scales: Number of scales to use
            polynomial_order: Order of polynomial for detrending
            
        Returns:
            Dictionary containing DFA results
        """
        if max_scale is None:
            max_scale = len(data) // 4
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales, dtype=int)
        scales = np.unique(scales)
        
        # Calculate fluctuation function
        fluctuations = []
        for scale in scales:
            f = self._calculate_fluctuation(data, scale, polynomial_order)
            fluctuations.append(f)
        
        fluctuations = np.array(fluctuations)
        
        # Fit power law
        log_scales = np.log10(scales)
        log_fluctuations = np.log10(fluctuations)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluctuations)
        hurst = slope
        
        # Calculate confidence interval
        n = len(scales)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_scales**2) / (n * np.sum((log_scales - np.mean(log_scales))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'scales': scales,
            'fluctuations': fluctuations,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results
    
    def _calculate_fluctuation(self, 
                             data: np.ndarray, 
                             scale: int, 
                             polynomial_order: int) -> float:
        """Calculate fluctuation for a given scale."""
        n = len(data)
        n_boxes = n // scale
        
        if n_boxes == 0:
            return np.nan
        
        fluctuations = []
        
        for i in range(n_boxes):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = data[start_idx:end_idx]
            
            # Detrend
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, polynomial_order)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            
            # Calculate RMS
            f = np.sqrt(np.mean(detrended**2))
            fluctuations.append(f)
        
        return np.mean(fluctuations)


class RSEstimator(ClassicalEstimator):
    """
    Rescaled Range Analysis (R/S) estimator.
    
    R/S analysis is one of the oldest methods for estimating the Hurst exponent.
    It measures the rescaled range of the cumulative deviation from the mean.
    """
    
    def __init__(self):
        super().__init__("R/S")
    
    def estimate(self, 
                data: np.ndarray,
                min_scale: int = 10,
                max_scale: int = None,
                n_scales: int = 20) -> Dict:
        """
        Estimate Hurst exponent using R/S analysis.
        
        Args:
            data: Time series data
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
            n_scales: Number of scales to use
            
        Returns:
            Dictionary containing R/S results
        """
        if max_scale is None:
            max_scale = len(data) // 4
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales, dtype=int)
        scales = np.unique(scales)
        
        # Calculate R/S values
        rs_values = []
        for scale in scales:
            rs = self._calculate_rs(data, scale)
            if not np.isnan(rs):
                rs_values.append(rs)
            else:
                scales = scales[:-1]  # Remove this scale
        
        rs_values = np.array(rs_values)
        
        # Fit power law
        log_scales = np.log10(scales)
        log_rs = np.log10(rs_values)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_rs)
        hurst = slope
        
        # Calculate confidence interval
        n = len(scales)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_scales**2) / (n * np.sum((log_scales - np.mean(log_scales))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'scales': scales,
            'rs_values': rs_values,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results
    
    def _calculate_rs(self, data: np.ndarray, scale: int) -> float:
        """Calculate R/S value for a given scale."""
        n = len(data)
        n_boxes = n // scale
        
        if n_boxes == 0:
            return np.nan
        
        rs_values = []
        
        for i in range(n_boxes):
            start_idx = i * scale
            end_idx = start_idx + scale
            segment = data[start_idx:end_idx]
            
            # Calculate mean
            mean_val = np.mean(segment)
            
            # Calculate cumulative deviation
            dev = segment - mean_val
            cum_dev = np.cumsum(dev)
            
            # Calculate range
            r = np.max(cum_dev) - np.min(cum_dev)
            
            # Calculate standard deviation
            s = np.std(segment)
            
            if s > 0:
                rs_values.append(r / s)
        
        return np.mean(rs_values) if rs_values else np.nan


class WaveletEstimator(ClassicalEstimator):
    """
    Wavelet-based estimator using Continuous Wavelet Transform.
    
    This method uses the wavelet transform to analyze the scaling properties
    of the time series at different scales.
    """
    
    def __init__(self, wavelet: str = 'db4'):
        super().__init__("Wavelet")
        self.wavelet = wavelet
    
    def estimate(self, 
                data: np.ndarray,
                min_scale: int = 2,
                max_scale: int = None,
                n_scales: int = 50) -> Dict:
        """
        Estimate Hurst exponent using wavelet analysis.
        
        Args:
            data: Time series data
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
            n_scales: Number of scales to use
            
        Returns:
            Dictionary containing wavelet results
        """
        if max_scale is None:
            max_scale = len(data) // 8
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
        
        # Calculate wavelet coefficients
        coefficients = []
        for scale in scales:
            coef = self._calculate_wavelet_coefficients(data, scale)
            coefficients.append(coef)
        
        coefficients = np.array(coefficients)
        
        # Calculate wavelet variance
        variance = np.var(coefficients, axis=1)
        
        # Fit power law
        log_scales = np.log10(scales)
        log_variance = np.log10(variance)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(log_variance) | np.isinf(log_variance))
        if np.sum(valid_mask) < 3:
            return {'hurst': np.nan, 'error': 'Insufficient valid data points'}
        
        log_scales = log_scales[valid_mask]
        log_variance = log_variance[valid_mask]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_variance)
        hurst = (slope + 1) / 2  # Convert to Hurst exponent
        
        # Calculate confidence interval
        n = len(log_scales)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_scales**2) / (n * np.sum((log_scales - np.mean(log_scales))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'scales': scales,
            'variance': variance,
            'coefficients': coefficients,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results
    
    def _calculate_wavelet_coefficients(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Calculate wavelet coefficients for a given scale."""
        # Use continuous wavelet transform
        widths = np.array([scale])
        coeffs, freqs = pywt.cwt(data, widths, self.wavelet)
        return coeffs.flatten()


class SpectralEstimator(ClassicalEstimator):
    """
    Spectral estimator using periodogram analysis.
    
    This method analyzes the power spectrum of the time series to estimate
    the Hurst exponent from the spectral slope.
    """
    
    def __init__(self, method: str = 'periodogram'):
        super().__init__(f"Spectral_{method}")
        self.method = method
    
    def estimate(self, 
                data: np.ndarray,
                min_freq: float = None,
                max_freq: float = None,
                n_freqs: int = 100) -> Dict:
        """
        Estimate Hurst exponent using spectral analysis.
        
        Args:
            data: Time series data
            min_freq: Minimum frequency for analysis
            max_freq: Maximum frequency for analysis
            n_freqs: Number of frequency points
            
        Returns:
            Dictionary containing spectral results
        """
        if min_freq is None:
            min_freq = 2.0 / len(data)
        if max_freq is None:
            max_freq = 0.5
        
        # Calculate power spectrum
        if self.method == 'periodogram':
            freqs, psd = periodogram(data, fs=1.0, nfft=len(data))
        elif self.method == 'welch':
            freqs, psd = welch(data, fs=1.0, nperseg=len(data)//4)
        else:
            raise ValueError(f"Unknown spectral method: {self.method}")
        
        # Filter frequency range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]
        
        # Remove zero and negative values
        valid_mask = (psd > 0) & ~np.isnan(psd) & ~np.isinf(psd)
        if np.sum(valid_mask) < 3:
            return {'hurst': np.nan, 'error': 'Insufficient valid data points'}
        
        freqs = freqs[valid_mask]
        psd = psd[valid_mask]
        
        # Fit power law
        log_freqs = np.log10(freqs)
        log_psd = np.log10(psd)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_freqs, log_psd)
        hurst = (1 - slope) / 2  # Convert to Hurst exponent
        
        # Calculate confidence interval
        n = len(log_freqs)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_freqs**2) / (n * np.sum((log_freqs - np.mean(log_freqs))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'frequencies': freqs,
            'power_spectrum': psd,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results


class HiguchiEstimator(ClassicalEstimator):
    """
    Higuchi method for estimating fractal dimension and Hurst exponent.
    
    This method calculates the fractal dimension of a time series by
    measuring the length of the curve at different scales.
    """
    
    def __init__(self):
        super().__init__("Higuchi")
    
    def estimate(self, 
                data: np.ndarray,
                max_k: int = None,
                n_k: int = 20) -> Dict:
        """
        Estimate Hurst exponent using Higuchi method.
        
        Args:
            data: Time series data
            max_k: Maximum k value for analysis
            n_k: Number of k values to use
            
        Returns:
            Dictionary containing Higuchi results
        """
        if max_k is None:
            max_k = len(data) // 4
        
        # Generate k values
        k_values = np.logspace(1, np.log10(max_k), n_k, dtype=int)
        k_values = np.unique(k_values)
        
        # Calculate curve lengths
        lengths = []
        for k in k_values:
            length = self._calculate_curve_length(data, k)
            if not np.isnan(length):
                lengths.append(length)
            else:
                k_values = k_values[:-1]  # Remove this k
        
        lengths = np.array(lengths)
        
        # Fit power law
        log_k = np.log10(k_values)
        log_lengths = np.log10(lengths)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_lengths)
        fractal_dim = 2 - slope
        hurst = 2 - fractal_dim  # Convert to Hurst exponent
        
        # Calculate confidence interval
        n = len(k_values)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_k**2) / (n * np.sum((log_k - np.mean(log_k))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'fractal_dimension': fractal_dim,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'k_values': k_values,
            'lengths': lengths,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results
    
    def _calculate_curve_length(self, data: np.ndarray, k: int) -> float:
        """Calculate curve length for a given k value."""
        n = len(data)
        if k >= n:
            return np.nan
        
        lengths = []
        
        for m in range(k):
            # Extract subsequence
            indices = np.arange(m, n, k)
            if len(indices) < 2:
                continue
            
            subsequence = data[indices]
            
            # Calculate length
            length = np.sum(np.abs(np.diff(subsequence)))
            lengths.append(length)
        
        return np.mean(lengths) if lengths else np.nan


class DMAEstimator(ClassicalEstimator):
    """
    Detrending Moving Average (DMA) estimator.
    
    DMA is a method that removes trends using a moving average and then
    analyzes the fluctuations to estimate the Hurst exponent.
    """
    
    def __init__(self):
        super().__init__("DMA")
    
    def estimate(self, 
                data: np.ndarray,
                min_scale: int = 4,
                max_scale: int = None,
                n_scales: int = 20) -> Dict:
        """
        Estimate Hurst exponent using DMA.
        
        Args:
            data: Time series data
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
            n_scales: Number of scales to use
            
        Returns:
            Dictionary containing DMA results
        """
        if max_scale is None:
            max_scale = len(data) // 4
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales, dtype=int)
        scales = np.unique(scales)
        
        # Calculate fluctuations
        fluctuations = []
        for scale in scales:
            f = self._calculate_fluctuation(data, scale)
            if not np.isnan(f):
                fluctuations.append(f)
            else:
                scales = scales[:-1]  # Remove this scale
        
        fluctuations = np.array(fluctuations)
        
        # Fit power law
        log_scales = np.log10(scales)
        log_fluctuations = np.log10(fluctuations)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluctuations)
        hurst = slope
        
        # Calculate confidence interval
        n = len(scales)
        t_critical = stats.t.ppf(0.975, n - 2)  # 95% confidence
        hurst_std = std_err * np.sqrt(np.sum(log_scales**2) / (n * np.sum((log_scales - np.mean(log_scales))**2)))
        ci_lower = hurst - t_critical * hurst_std
        ci_upper = hurst + t_critical * hurst_std
        
        self.results = {
            'hurst': hurst,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': r_value**2,
            'p_value': p_value,
            'scales': scales,
            'fluctuations': fluctuations,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }
        
        return self.results
    
    def _calculate_fluctuation(self, data: np.ndarray, scale: int) -> float:
        """Calculate fluctuation for a given scale using DMA."""
        n = len(data)
        
        # Calculate moving average
        window = np.ones(scale) / scale
        ma = np.convolve(data, window, mode='same')
        
        # Detrend
        detrended = data - ma
        
        # Calculate RMS
        f = np.sqrt(np.mean(detrended**2))
        
        return f


class ClassicalEstimatorSuite:
    """
    Suite of classical estimators for comprehensive benchmarking.
    
    This class provides a unified interface to multiple classical estimation
    methods, allowing for easy comparison and benchmarking.
    """
    
    def __init__(self):
        """Initialize the estimator suite."""
        self.estimators = {
            'dfa': DFAEstimator(),
            'rs': RSEstimator(),
            'wavelet': WaveletEstimator(),
            'spectral': SpectralEstimator(),
            'higuchi': HiguchiEstimator(),
            'dma': DMAEstimator()
        }
        self.results = {}
    
    def estimate_all(self, data: np.ndarray, **kwargs) -> Dict[str, Dict]:
        """
        Run all estimators on the given data.
        
        Args:
            data: Time series data
            **kwargs: Additional parameters for estimators
            
        Returns:
            Dictionary containing results from all estimators
        """
        results = {}
        
        for name, estimator in self.estimators.items():
            try:
                result = estimator.estimate(data, **kwargs)
                results[name] = result
            except Exception as e:
                results[name] = {'error': str(e), 'hurst': np.nan}
        
        self.results = results
        return results
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of all estimation results.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for name, result in self.results.items():
            if 'error' in result:
                summary_data.append({
                    'method': name,
                    'hurst': np.nan,
                    'confidence_lower': np.nan,
                    'confidence_upper': np.nan,
                    'r_squared': np.nan,
                    'p_value': np.nan,
                    'error': result['error']
                })
            else:
                ci = result.get('confidence_interval', (np.nan, np.nan))
                summary_data.append({
                    'method': name,
                    'hurst': result.get('hurst', np.nan),
                    'confidence_lower': ci[0] if ci[0] is not None else np.nan,
                    'confidence_upper': ci[1] if ci[1] is not None else np.nan,
                    'r_squared': result.get('r_squared', np.nan),
                    'p_value': result.get('p_value', np.nan),
                    'error': None
                })
        
        return pd.DataFrame(summary_data)
    
    def get_best_estimate(self, criterion: str = 'r_squared') -> Tuple[str, float]:
        """
        Get the best estimate based on a criterion.
        
        Args:
            criterion: Criterion for selecting best estimate ('r_squared', 'p_value')
            
        Returns:
            Tuple of (method_name, hurst_value)
        """
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and not np.isnan(v.get('hurst', np.nan))}
        
        if not valid_results:
            return None, np.nan
        
        if criterion == 'r_squared':
            best_method = max(valid_results.keys(), 
                            key=lambda x: valid_results[x].get('r_squared', 0))
        elif criterion == 'p_value':
            best_method = min(valid_results.keys(), 
                            key=lambda x: valid_results[x].get('p_value', 1))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return best_method, valid_results[best_method]['hurst']


# Convenience functions
def estimate_hurst_dfa(data: np.ndarray, **kwargs) -> float:
    """Quick function to estimate Hurst exponent using DFA."""
    estimator = DFAEstimator()
    result = estimator.estimate(data, **kwargs)
    return result.get('hurst', np.nan)


def estimate_hurst_rs(data: np.ndarray, **kwargs) -> float:
    """Quick function to estimate Hurst exponent using R/S."""
    estimator = RSEstimator()
    result = estimator.estimate(data, **kwargs)
    return result.get('hurst', np.nan)


def estimate_hurst_wavelet(data: np.ndarray, **kwargs) -> float:
    """Quick function to estimate Hurst exponent using wavelet method."""
    estimator = WaveletEstimator()
    result = estimator.estimate(data, **kwargs)
    return result.get('hurst', np.nan)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Classical Estimators...")
    
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
    
    # Test all estimators
    suite = ClassicalEstimatorSuite()
    results = suite.estimate_all(test_data)
    
    # Print summary
    summary = suite.get_summary()
    print("\nEstimation Results:")
    print(summary[['method', 'hurst', 'r_squared', 'p_value']].to_string(index=False))
    
    # Get best estimate
    best_method, best_hurst = suite.get_best_estimate()
    print(f"\nBest estimate: {best_method} = {best_hurst:.3f}")
    print(f"True Hurst: {hurst_true:.3f}")
    print(f"Absolute error: {abs(best_hurst - hurst_true):.3f}")
    
    print("\nClassical estimators test completed successfully!")
