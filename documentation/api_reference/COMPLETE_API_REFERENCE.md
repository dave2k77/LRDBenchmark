# 📚 **COMPLETE API REFERENCE - LRDBench**

This document provides a comprehensive, detailed reference for all classes, methods, and components in the LRDBench framework.

## 📦 **Package Structure**

```
lrdbench/
├── __init__.py                    # Main package with convenient imports
├── analysis/                      # All estimator implementations
│   ├── benchmark.py              # ComprehensiveBenchmark class
│   ├── temporal/                 # Temporal domain estimators
│   ├── spectral/                 # Spectral domain estimators
│   ├── wavelet/                  # Wavelet domain estimators
│   ├── multifractal/             # Multifractal estimators
│   ├── machine_learning/         # ML estimators
│   └── high_performance/         # JAX and Numba optimized versions
└── models/                       # Data models and utilities
    ├── data_models/              # Synthetic data generators
    ├── contamination/            # Data contamination models
    └── pretrained_models/        # Pre-trained ML and neural models
```

## 🚀 **Quick Import Guide**

```python
# Main package
import lrdbench

# Core components
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Data models
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbench.models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Classical estimators
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
from lrdbench.analysis.temporal.dma.dma_estimator import DMAEstimator
from lrdbench.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator

# Pre-trained models
from lrdbench.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
from lrdbench.models.pretrained_models.transformer_pretrained import TransformerPretrainedModel
```

---

## 🔬 **COMPREHENSIVE BENCHMARK**

### **Class: ComprehensiveBenchmark**

**Location:** `lrdbench.analysis.benchmark.ComprehensiveBenchmark`

**Purpose:** Main entry point for running systematic evaluations of all long-range dependence estimators.

#### **Constructor**
```python
def __init__(self, output_dir: Optional[str] = None)
```

**Parameters:**
- `output_dir` (str, optional): Directory to save benchmark results. Defaults to "benchmark_results".

#### **Methods**

##### **`run_comprehensive_benchmark()`**
```python
def run_comprehensive_benchmark(
    self,
    data_length: int = 1000,
    benchmark_type: str = 'comprehensive',
    contamination_type: Optional[str] = None,
    contamination_level: float = 0.1,
    save_results: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `data_length` (int): Length of test data to generate (default: 1000)
- `benchmark_type` (str): Type of benchmark to run
  - `'comprehensive'`: All estimators (default)
  - `'classical'`: Only classical statistical estimators
  - `'ML'`: Only machine learning estimators (non-neural)
  - `'neural'`: Only neural network estimators
- `contamination_type` (str, optional): Type of contamination to apply
- `contamination_level` (float): Level/intensity of contamination (0.0 to 1.0)
- `save_results` (bool): Whether to save results to file

**Returns:** Dictionary containing comprehensive benchmark results

**Example:**
```python
benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark(
    data_length=2000,
    contamination_type='additive_gaussian',
    contamination_level=0.15
)
```

##### **`run_classical_benchmark()`**
```python
def run_classical_benchmark(
    self,
    data_length: int = 1000,
    contamination_type: Optional[str] = None,
    contamination_level: float = 0.1,
    save_results: bool = True
) -> Dict[str, Any]
```

**Purpose:** Convenience method for running benchmarks with only classical statistical estimators.

##### **`run_ml_benchmark()`**
```python
def run_ml_benchmark(
    self,
    data_length: int = 1000,
    contamination_type: Optional[str] = None,
    contamination_level: float = 0.1,
    save_results: bool = True
) -> Dict[str, Any]
```

**Purpose:** Convenience method for running benchmarks with only machine learning estimators.

##### **`run_neural_benchmark()`**
```python
def run_neural_benchmark(
    self,
    data_length: int = 1000,
    contamination_type: Optional[str] = None,
    contamination_level: float = 0.1,
    save_results: bool = True
) -> Dict[str, Any]
```

**Purpose:** Convenience method for running benchmarks with only neural network estimators.

---

## 📊 **DATA MODELS**

### **Base Model Class**

**Location:** `lrdbench.models.data_models.base_model.BaseModel`

**Purpose:** Abstract base class for all stochastic data models.

#### **Abstract Methods**
```python
def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray
def get_theoretical_properties(self) -> Dict[str, Any]
```

### **1. Fractional Brownian Motion (fBm)**

**Location:** `lrdbench.models.data_models.fbm.fbm_model.FractionalBrownianMotion`

**Purpose:** Generates self-similar Gaussian processes with long-range dependence.

#### **Constructor**
```python
def __init__(self, H: float, sigma: float = 1.0, method: str = 'davies_harte')
```

**Parameters:**
- `H` (float): Hurst parameter ∈ (0, 1)
- `sigma` (float): Standard deviation > 0
- `method` (str): Generation method - 'davies_harte', 'cholesky', or 'circulant'

#### **Methods**

##### **`generate()`**
```python
def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray
```

**Parameters:**
- `n` (int): Number of data points to generate
- `seed` (int, optional): Random seed for reproducibility

**Returns:** np.ndarray of shape (n,) containing fBm data

**Example:**
```python
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)
```

##### **`get_theoretical_properties()`**
```python
def get_theoretical_properties(self) -> Dict[str, Any]
```

**Returns:** Dictionary with theoretical properties including:
- `hurst_parameter`: The Hurst parameter H
- `long_range_dependence`: Boolean indicating if H > 0.5
- `variance`: Theoretical variance
- `autocorrelation`: Theoretical autocorrelation function

### **2. Fractional Gaussian Noise (fGn)**

**Location:** `lrdbench.models.data_models.fgn.fgn_model.FractionalGaussianNoise`

**Purpose:** Generates stationary increments of fractional Brownian motion.

#### **Constructor**
```python
def __init__(self, H: float, sigma: float = 1.0, method: str = 'davies_harte')
```

**Parameters:** Same as fBm

#### **Methods**
Same interface as fBm, but generates fGn data instead.

### **3. ARFIMA Model**

**Location:** `lrdbench.models.data_models.arfima.arfima_model.ARFIMAModel`

**Purpose:** Generates AutoRegressive Fractionally Integrated Moving Average time series.

#### **Constructor**
```python
def __init__(self, d: float, ar_params: Optional[List[float]] = None, 
             ma_params: Optional[List[float]] = None, sigma: float = 1.0)
```

**Parameters:**
- `d` (float): Fractional integration parameter ∈ (-0.5, 0.5)
- `ar_params` (List[float], optional): AR coefficients
- `ma_params` (List[float], optional): MA coefficients
- `sigma` (float): Innovation standard deviation > 0

#### **Methods**
Same interface as fBm, but generates ARFIMA data.

**Note:** For ARFIMA models, the `d` parameter relates to Hurst as H = d + 0.5.

### **4. Multifractal Random Walk (MRW)**

**Location:** `lrdbench.models.data_models.mrw.mrw_model.MultifractalRandomWalk`

**Purpose:** Generates non-Gaussian multifractal processes.

#### **Constructor**
```python
def __init__(self, H: float, lambda_param: float, sigma: float = 1.0)
```

**Parameters:**
- `H` (float): Hurst parameter ∈ (0, 1)
- `lambda_param` (float): Intermittency parameter > 0
- `sigma` (float): Base volatility > 0

#### **Methods**
Same interface as fBm, but generates MRW data.

---

## 🔍 **ESTIMATORS**

### **Base Estimator Class**

**Location:** `lrdbench.analysis.base_estimator.BaseEstimator`

**Purpose:** Abstract base class for all estimators.

#### **Abstract Methods**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
def get_confidence_intervals(self) -> Tuple[float, float]
def get_estimation_quality(self) -> Dict[str, Any]
```

### **Temporal Domain Estimators**

#### **1. R/S Estimator**

**Location:** `lrdbench.analysis.temporal.rs.rs_estimator.RSEstimator`

**Purpose:** Rescaled Range Analysis for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, min_window_size: int = 10, max_window_size: Optional[int] = None, 
             window_sizes: Optional[List[int]] = None, overlap: bool = False)
```

**Parameters:**
- `min_window_size` (int): Minimum window size for analysis
- `max_window_size` (int, optional): Maximum window size
- `window_sizes` (List[int], optional): Specific window sizes to use
- `overlap` (bool): Whether to use overlapping windows

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `data` (np.ndarray): Input time series

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `r_squared` (float): Goodness of fit (R²)
- `confidence_interval` (Tuple[float, float]): 95% confidence interval
- `window_sizes` (List[int]): Analysis window sizes used
- `rs_values` (List[float]): R/S values for each window
- `std_error` (float): Standard error of estimate
- `p_value` (float): P-value of the linear regression
- `intercept` (float): Intercept of the linear fit
- `slope` (float): Slope of the linear fit (equals H)

**Example:**
```python
rs_estimator = RSEstimator()
result = rs_estimator.estimate(data)
hurst = result['hurst_parameter']
```

#### **2. DFA Estimator**

**Location:** `lrdbench.analysis.temporal.dfa.dfa_estimator.DFAEstimator`

**Purpose:** Detrended Fluctuation Analysis for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, min_window_size: int = 10, max_window_size: Optional[int] = None,
             window_sizes: Optional[List[int]] = None, polynomial_order: int = 1)
```

**Parameters:**
- `min_window_size` (int): Minimum window size for analysis
- `max_window_size` (int, optional): Maximum window size
- `window_sizes` (List[int], optional): Specific window sizes to use
- `polynomial_order` (int): Order of polynomial for detrending

#### **Methods**
Same interface as R/S estimator.

#### **3. DMA Estimator**

**Location:** `lrdbench.analysis.temporal.dma.dma_estimator.DMAEstimator`

**Purpose:** Detrending Moving Average for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, min_window_size: int = 10, max_window_size: Optional[int] = None,
             window_sizes: Optional[List[int]] = None, window_type: str = 'box')
```

**Parameters:**
- `min_window_size` (int): Minimum window size for analysis
- `max_window_size` (int, optional): Maximum window size
- `window_sizes` (List[int], optional): Specific window sizes to use
- `window_type` (str): Type of moving average window ('box', 'gaussian', 'exponential')

#### **Methods**
Same interface as R/S estimator.

#### **4. Higuchi Estimator**

**Location:** `lrdbench.analysis.temporal.higuchi.higuchi_estimator.HiguchiEstimator`

**Purpose:** Higuchi method for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, min_window_size: int = 10, max_window_size: Optional[int] = None,
             window_sizes: Optional[List[int]] = None, k_max: Optional[int] = None)
```

**Parameters:**
- `min_window_size` (int): Minimum window size for analysis
- `max_window_size` (int, optional): Maximum window size
- `window_sizes` (List[int], optional): Specific window sizes to use
- `k_max` (int, optional): Maximum k value for Higuchi method

#### **Methods**
Same interface as R/S estimator.

### **Spectral Domain Estimators**

#### **1. GPH Estimator**

**Location:** `lrdbench.analysis.spectral.gph.gph_estimator.GPHEstimator`

**Purpose:** Geweke-Porter-Hudak estimator for long-memory processes.

#### **Constructor**
```python
def __init__(self, n_freq: Optional[int] = None, freq_range: Optional[Tuple[float, float]] = None,
             bandwidth: Optional[float] = None)
```

**Parameters:**
- `n_freq` (int, optional): Number of frequencies to use
- `freq_range` (Tuple[float, float], optional): Frequency range (low, high)
- `bandwidth` (float, optional): Bandwidth for spectral estimation

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `d_parameter` (float): Estimated fractional integration parameter
- `r_squared` (float): Goodness of fit
- `confidence_interval` (Tuple[float, float]): 95% confidence interval
- `frequencies` (np.ndarray): Frequencies used in estimation
- `periodogram` (np.ndarray): Periodogram values
- `std_error` (float): Standard error of estimate

#### **2. Whittle Estimator**

**Location:** `lrdbench.analysis.spectral.whittle.whittle_estimator.WhittleEstimator`

**Purpose:** Whittle likelihood estimator for long-memory processes.

#### **Constructor**
```python
def __init__(self, model_type: str = 'fgn', optimization_method: str = 'l-bfgs-b')
```

**Parameters:**
- `model_type` (str): Type of model ('fgn', 'arfima', 'fbm')
- `optimization_method` (str): Optimization method for likelihood maximization

#### **Methods**
Same interface as GPH estimator.

#### **3. Periodogram Estimator**

**Location:** `lrdbench.analysis.spectral.periodogram.periodogram_estimator.PeriodogramEstimator`

**Purpose:** Periodogram-based estimator for long-memory processes.

#### **Constructor**
```python
def __init__(self, n_freq: Optional[int] = None, freq_range: Optional[Tuple[float, float]] = None,
             window: Optional[str] = None)
```

**Parameters:**
- `n_freq` (int, optional): Number of frequencies to use
- `freq_range` (Tuple[float, float], optional): Frequency range (low, high)
- `window` (str, optional): Window function ('hann', 'hamming', 'blackman', 'boxcar')

#### **Methods**
Same interface as GPH estimator.

### **Wavelet Domain Estimators**

#### **1. CWT Estimator**

**Location:** `lrdbench.analysis.wavelet.cwt.cwt_estimator.CWTEstimator`

**Purpose:** Continuous Wavelet Transform estimator for Hurst parameter.

#### **Constructor**
```python
def __init__(self, wavelet: str = 'db4', scales: Optional[List[float]] = None,
             max_scale: Optional[float] = None, n_scales: int = 8)
```

**Parameters:**
- `wavelet` (str): Wavelet type ('db4', 'haar', 'sym4', etc.)
- `scales` (List[float], optional): Specific scales to use
- `max_scale` (float, optional): Maximum scale value
- `n_scales` (int): Number of scales to use

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `r_squared` (float): Goodness of fit
- `confidence_interval` (Tuple[float, float]): 95% confidence interval
- `scales` (List[float]): Wavelet scales used
- `coefficients` (List[float]): Wavelet coefficient variances
- `std_error` (float): Standard error of estimate

#### **2. Wavelet Variance Estimator**

**Location:** `lrdbench.analysis.wavelet.variance.wavelet_variance_estimator.WaveletVarianceEstimator`

**Purpose:** Wavelet variance method for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, wavelet: str = 'db4', scales: Optional[List[float]] = None,
             max_scale: Optional[float] = None, n_scales: int = 8)
```

**Parameters:** Same as CWT estimator.

#### **Methods**
Same interface as CWT estimator.

#### **3. Wavelet Log Variance Estimator**

**Location:** `lrdbench.analysis.wavelet.log_variance.wavelet_log_variance_estimator.WaveletLogVarianceEstimator`

**Purpose:** Wavelet log variance method for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, wavelet: str = 'db4', scales: Optional[List[float]] = None,
             max_scale: Optional[float] = None, n_scales: int = 8)
```

**Parameters:** Same as CWT estimator.

#### **Methods**
Same interface as CWT estimator.

#### **4. Wavelet Whittle Estimator**

**Location:** `lrdbench.analysis.wavelet.whittle.wavelet_whittle_estimator.WaveletWhittleEstimator`

**Purpose:** Wavelet Whittle likelihood estimator for long-memory processes.

#### **Constructor**
```python
def __init__(self, wavelet: str = 'db4', scales: Optional[List[float]] = None,
             max_scale: Optional[float] = None, n_scales: int = 8,
             optimization_method: str = 'l-bfgs-b')
```

**Parameters:** Same as CWT estimator plus:
- `optimization_method` (str): Optimization method for likelihood maximization

#### **Methods**
Same interface as CWT estimator.

### **Multifractal Estimators**

#### **1. MFDFA Estimator**

**Location:** `lrdbench.analysis.multifractal.mfdfa.mfdfa_estimator.MFDFAEstimator`

**Purpose:** Multifractal Detrended Fluctuation Analysis.

#### **Constructor**
```python
def __init__(self, q_values: Optional[List[float]] = None, min_window_size: int = 10,
             max_window_size: Optional[int] = None, polynomial_order: int = 1)
```

**Parameters:**
- `q_values` (List[float], optional): q values for multifractal analysis
- `min_window_size` (int): Minimum window size for analysis
- `max_window_size` (int, optional): Maximum window size
- `polynomial_order` (int): Order of polynomial for detrending

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent (q=2)
- `multifractal_spectrum` (Dict): Full multifractal spectrum
- `q_values` (List[float]): q values used
- `hq_values` (List[float]): Generalized Hurst exponents
- `tau_values` (List[float]): Mass exponents
- `f_alpha_values` (List[float]): Multifractal spectrum

#### **2. Multifractal Wavelet Leaders Estimator**

**Location:** `lrdbench.analysis.multifractal.wavelet_leaders.multifractal_wavelet_leaders_estimator.MultifractalWaveletLeadersEstimator`

**Purpose:** Multifractal analysis using wavelet leaders.

#### **Constructor**
```python
def __init__(self, wavelet: str = 'db4', scales: Optional[List[float]] = None,
             max_scale: Optional[float] = None, n_scales: int = 8)
```

**Parameters:** Same as CWT estimator.

#### **Methods**
Same interface as MFDFA estimator.

### **Machine Learning Estimators**

#### **Base ML Estimator**

**Location:** `lrdbench.analysis.machine_learning.base_ml_estimator.BaseMLEstimator`

**Purpose:** Base class for all machine learning estimators.

#### **Abstract Methods**
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> None
def predict(self, X: np.ndarray) -> np.ndarray
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

#### **1. Random Forest Estimator**

**Location:** `lrdbench.analysis.machine_learning.random_forest_estimator.RandomForestEstimator`

**Purpose:** Random Forest regression for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
             min_samples_split: int = 2, min_samples_leaf: int = 1,
             random_state: Optional[int] = None)
```

**Parameters:**
- `n_estimators` (int): Number of trees in the forest
- `max_depth` (int, optional): Maximum depth of trees
- `min_samples_split` (int): Minimum samples required to split
- `min_samples_leaf` (int): Minimum samples required at leaf node
- `random_state` (int, optional): Random seed

#### **Methods**

##### **`fit()`**
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> None
```

**Parameters:**
- `X` (np.ndarray): Feature matrix of shape (n_samples, n_features)
- `y` (np.ndarray): Target values of shape (n_samples,)

##### **`predict()`**
```python
def predict(self, X: np.ndarray) -> np.ndarray
```

**Parameters:**
- `X` (np.ndarray): Feature matrix

**Returns:** Predicted values

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `data` (np.ndarray): Input time series

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `confidence_interval` (Tuple[float, float]): Prediction interval
- `feature_importance` (Dict): Feature importance scores

#### **2. Gradient Boosting Estimator**

**Location:** `lrdbench.analysis.machine_learning.gradient_boosting_estimator.GradientBoostingEstimator`

**Purpose:** Gradient Boosting regression for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
             max_depth: int = 3, min_samples_split: int = 2,
             min_samples_leaf: int = 1, random_state: Optional[int] = None)
```

**Parameters:**
- `n_estimators` (int): Number of boosting stages
- `learning_rate` (float): Learning rate
- `max_depth` (int): Maximum depth of trees
- `min_samples_split` (int): Minimum samples required to split
- `min_samples_leaf` (int): Minimum samples required at leaf node
- `random_state` (int, optional): Random seed

#### **Methods**
Same interface as Random Forest estimator.

#### **3. SVR Estimator**

**Location:** `lrdbench.analysis.machine_learning.svr_estimator.SVREstimator`

**Purpose:** Support Vector Regression for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1,
             gamma: Optional[str] = None, random_state: Optional[int] = None)
```

**Parameters:**
- `kernel` (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
- `C` (float): Regularization parameter
- `epsilon` (float): Epsilon in epsilon-SVR model
- `gamma` (str, optional): Kernel coefficient ('scale', 'auto', or float)
- `random_state` (int, optional): Random seed

#### **Methods**
Same interface as Random Forest estimator.

### **Neural Network Estimators**

#### **1. CNN Estimator**

**Location:** `lrdbench.analysis.machine_learning.cnn_estimator.CNNEstimator`

**Purpose:** Convolutional Neural Network for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, input_length: int = 500, n_filters: int = 64,
             kernel_size: int = 3, n_layers: int = 3, dropout_rate: float = 0.2,
             learning_rate: float = 0.001, batch_size: int = 32,
             random_state: Optional[int] = None)
```

**Parameters:**
- `input_length` (int): Length of input time series
- `n_filters` (int): Number of filters in convolutional layers
- `kernel_size` (int): Size of convolutional kernels
- `n_layers` (int): Number of convolutional layers
- `dropout_rate` (float): Dropout rate for regularization
- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `random_state` (int, optional): Random seed

#### **Methods**

##### **`fit()`**
```python
def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
        validation_split: float = 0.2, verbose: bool = True) -> None
```

**Parameters:**
- `X` (np.ndarray): Feature matrix of shape (n_samples, input_length)
- `y` (np.ndarray): Target values of shape (n_samples,)
- `epochs` (int): Number of training epochs
- `validation_split` (float): Fraction of data for validation
- `verbose` (bool): Whether to print training progress

##### **`predict()`**
```python
def predict(self, X: np.ndarray) -> np.ndarray
```

**Parameters:**
- `X` (np.ndarray): Feature matrix

**Returns:** Predicted values

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `data` (np.ndarray): Input time series

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `confidence_interval` (Tuple[float, float]): Prediction interval
- `model_output` (np.ndarray): Raw model output

#### **2. Transformer Estimator**

**Location:** `lrdbench.analysis.machine_learning.transformer_estimator.TransformerEstimator`

**Purpose:** Transformer-based estimator for Hurst parameter estimation.

#### **Constructor**
```python
def __init__(self, input_length: int = 500, d_model: int = 128, n_heads: int = 8,
             n_layers: int = 4, d_ff: int = 512, dropout_rate: float = 0.1,
             learning_rate: float = 0.001, batch_size: int = 32,
             random_state: Optional[int] = None)
```

**Parameters:**
- `input_length` (int): Length of input time series
- `d_model` (int): Dimension of model embeddings
- `n_heads` (int): Number of attention heads
- `n_layers` (int): Number of transformer layers
- `d_ff` (int): Dimension of feed-forward network
- `dropout_rate` (float): Dropout rate for regularization
- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `random_state` (int, optional): Random seed

#### **Methods**
Same interface as CNN estimator.

---

## 🤖 **PRE-TRAINED MODELS**

### **Base Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.base_pretrained_model.BasePretrainedModel`

**Purpose:** Base class for all pre-trained models.

#### **Abstract Methods**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
def load_model(self, model_path: str) -> None
def save_model(self, model_path: str) -> None
```

### **1. CNN Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.cnn_pretrained.CNNPretrainedModel`

**Purpose:** Pre-trained CNN model for immediate use without training.

#### **Constructor**
```python
def __init__(self, input_length: int = 500, model_path: Optional[str] = None)
```

**Parameters:**
- `input_length` (int): Length of input time series
- `model_path` (str, optional): Path to pre-trained model weights

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `data` (np.ndarray): Input time series

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `confidence_interval` (Tuple[float, float]): Prediction interval
- `model_output` (np.ndarray): Raw model output

### **2. Transformer Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.transformer_pretrained.TransformerPretrainedModel`

**Purpose:** Pre-trained Transformer model for immediate use without training.

#### **Constructor**
```python
def __init__(self, input_length: int = 500, model_path: Optional[str] = None)
```

**Parameters:** Same as CNN Pre-trained Model.

#### **Methods**
Same interface as CNN Pre-trained Model.

### **3. ML Pre-trained Models**

#### **Random Forest Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.ml_pretrained.RandomForestPretrainedModel`

**Purpose:** Pre-trained Random Forest model with heuristic-based estimation.

#### **Constructor**
```python
def __init__(self, model_path: Optional[str] = None)
```

**Parameters:**
- `model_path` (str, optional): Path to pre-trained model

#### **Methods**

##### **`estimate()`**
```python
def estimate(self, data: np.ndarray) -> Dict[str, Any]
```

**Parameters:**
- `data` (np.ndarray): Input time series

**Returns:** Dictionary with keys:
- `hurst_parameter` (float): Estimated Hurst exponent
- `estimation_method` (str): Method used ('heuristic')
- `confidence_interval` (Tuple[float, float]): Estimated confidence interval

#### **SVR Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.ml_pretrained.SVREstimatorPretrainedModel`

**Purpose:** Pre-trained SVR model with heuristic-based estimation.

#### **Constructor and Methods**
Same interface as Random Forest Pre-trained Model.

#### **Gradient Boosting Pre-trained Model**

**Location:** `lrdbench.models.pretrained_models.ml_pretrained.GradientBoostingPretrainedModel`

**Purpose:** Pre-trained Gradient Boosting model with heuristic-based estimation.

#### **Constructor and Methods**
Same interface as Random Forest Pre-trained Model.

---

## 🚀 **HIGH-PERFORMANCE ESTIMATORS**

### **JAX Optimized Versions**

**Location:** `lrdbench.analysis.high_performance.jax.*`

**Purpose:** GPU-accelerated versions of estimators using JAX.

**Available Estimators:**
- `cwt_jax.CWTEstimatorJAX`
- `dfa_jax.DFAEstimatorJAX`
- `dma_jax.DMAEstimatorJAX`
- `gph_jax.GPHEstimatorJAX`
- `higuchi_jax.HiguchiEstimatorJAX`
- `mfdfa_jax.MFDFAEstimatorJAX`
- `multifractal_wavelet_leaders_jax.MultifractalWaveletLeadersEstimatorJAX`
- `periodogram_jax.PeriodogramEstimatorJAX`
- `rs_jax.RSEstimatorJAX`
- `wavelet_log_variance_jax.WaveletLogVarianceEstimatorJAX`
- `wavelet_variance_jax.WaveletVarianceEstimatorJAX`
- `wavelet_whittle_jax.WaveletWhittleEstimatorJAX`
- `whittle_jax.WhittleEstimatorJAX`

**Note:** These estimators have the same interface as their CPU counterparts but require JAX installation.

### **Numba Optimized Versions**

**Location:** `lrdbench.analysis.high_performance.numba.*`

**Purpose:** CPU-optimized versions of estimators using Numba JIT compilation.

**Available Estimators:**
- `cwt_numba.CWTEstimatorNumba`
- `dfa_numba.DFAEstimatorNumba`
- `dma_numba.DMAEstimatorNumba`
- `gph_numba.GPHEstimatorNumba`
- `higuchi_numba.HiguchiEstimatorNumba`
- `mfdfa_numba.MFDFAEstimatorNumba`
- `multifractal_wavelet_leaders_numba.MultifractalWaveletLeadersEstimatorNumba`
- `periodogram_numba.PeriodogramEstimatorNumba`
- `rs_numba.RSEstimatorNumba`
- `wavelet_log_variance_numba.WaveletLogVarianceEstimatorNumba`
- `wavelet_variance_numba.WaveletVarianceEstimatorNumba`
- `wavelet_whittle_numba.WaveletWhittleEstimatorNumba`
- `whittle_numba.WhittleEstimatorNumba`

**Note:** These estimators have the same interface as their standard counterparts but are optimized with Numba.

---

## 🔧 **UTILITY FUNCTIONS**

### **Data Contamination Models**

**Location:** `lrdbench.models.contamination.contamination_models.*`

**Purpose:** Models for adding various types of contamination to test data robustness.

**Available Contamination Types:**
- `AdditiveGaussianNoise`: Add Gaussian noise to data
- `MultiplicativeNoise`: Add multiplicative noise to data
- `OutlierContamination`: Add outliers to data
- `TrendContamination`: Add linear or polynomial trends
- `SeasonalContamination`: Add seasonal patterns
- `MissingDataContamination`: Remove data points

**Usage:**
```python
from lrdbench.models.contamination.contamination_models import AdditiveGaussianNoise

contaminator = AdditiveGaussianNoise(noise_level=0.1, std=0.5)
contaminated_data = contaminator.apply(clean_data)
```

---

## 📊 **OUTPUT FORMATS**

### **Standard Estimator Output**

All estimators return a standardized dictionary format:

```python
{
    'hurst_parameter': float,           # Estimated Hurst exponent
    'r_squared': float,                 # Goodness of fit (R²)
    'confidence_interval': Tuple,       # 95% confidence interval
    'std_error': float,                 # Standard error
    'p_value': float,                   # P-value (if applicable)
    'execution_time': float,            # Execution time in seconds
    'success': bool,                    # Whether estimation succeeded
    'error_message': str,               # Error message if failed
    # Additional estimator-specific fields...
}
```

### **Benchmark Output**

The ComprehensiveBenchmark returns:

```python
{
    'timestamp': str,                   # ISO timestamp
    'benchmark_type': str,              # Type of benchmark run
    'contamination_type': str,          # Contamination type (if any)
    'contamination_level': float,       # Contamination intensity
    'total_tests': int,                 # Total number of tests
    'successful_tests': int,            # Number of successful tests
    'success_rate': float,              # Success rate (0.0 to 1.0)
    'data_models_tested': int,          # Number of data models tested
    'estimators_tested': int,           # Number of estimators tested
    'results': Dict,                    # Detailed results by data model
    'execution_time': float,            # Total execution time
    'memory_usage': float               # Memory usage in MB
}
```

---

## 🚨 **ERROR HANDLING**

### **Common Error Types**

1. **ImportError**: Missing dependencies
   - **Solution**: Install required packages (`pip install lrdbench[full]`)

2. **ValueError**: Invalid parameters
   - **Solution**: Check parameter ranges and types

3. **RuntimeError**: Estimation failure
   - **Solution**: Check data quality and length requirements

4. **MemoryError**: Insufficient memory
   - **Solution**: Reduce data length or use chunked processing

### **Error Recovery**

Most estimators include fallback methods:
- **Statistical fallbacks**: Use simple statistical methods if complex estimation fails
- **Parameter validation**: Automatic parameter adjustment within valid ranges
- **Graceful degradation**: Continue execution even if some estimators fail

---

## 💡 **BEST PRACTICES**

### **Data Preparation**

1. **Data Length**: Use ≥500 points for reliable estimation
2. **Data Quality**: Remove trends and seasonal components if possible
3. **Normalization**: Scale data to unit variance for consistent results

### **Estimator Selection**

1. **Start with classical estimators** for baseline performance
2. **Use wavelet methods** for robust, scale-invariant analysis
3. **Apply ML/Neural methods** for complex, non-linear relationships
4. **Consider computational cost** for large datasets

### **Benchmarking**

1. **Test multiple estimators** to compare performance
2. **Use contamination testing** to assess robustness
3. **Vary data lengths** to understand estimator behavior
4. **Save results** for post-analysis and comparison

---

## 🔗 **RELATED DOCUMENTATION**

- [**Main API Reference**](README.md) - Overview and quick start
- [**Benchmark Documentation**](estimators/benchmark.md) - Comprehensive benchmarking system
- [**Project README**](../../README.md) - Project overview and installation
- [**Examples**](../../examples/) - Usage examples and tutorials
- [**Demos**](../../demos/) - Interactive demonstrations

---

**For questions or issues, please refer to the main project documentation or create an issue on GitHub.**
