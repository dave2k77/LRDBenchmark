# 📚 **LRDBench API Reference**

Welcome to the comprehensive API reference for LRDBench, a framework for long-range dependence estimation with enhanced ML and neural network estimators.

## 🚀 **Quick Start**

```python
# Install the package
pip install lrdbench

# Import the main components
import lrdbench
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Run a comprehensive benchmark
benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark()
```

## 📦 **Package Structure**

```
lrdbench/
├── __init__.py                    # Main package with convenient imports
├── analysis/                      # All estimator implementations
│   ├── benchmark.py              # ComprehensiveBenchmark class
│   ├── auto_optimized_estimator.py # Revolutionary auto-optimization system
│   ├── temporal/                 # Temporal domain estimators
│   ├── spectral/                 # Spectral domain estimators
│   ├── wavelet/                  # Wavelet domain estimators
│   ├── multifractal/             # Multifractal estimators
│   ├── machine_learning/         # Enhanced ML and neural network estimators
│   └── high_performance/         # JAX and Numba optimized versions
└── models/                       # Data models and utilities
    ├── data_models/              # Synthetic data generators
    ├── contamination/            # Data contamination models
    └── pretrained_models/        # Pre-trained ML and neural models
```

## 🔧 **Core Components**

### **ComprehensiveBenchmark**
The main entry point for running benchmarks:

```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Run different types of benchmarks
results = benchmark.run_comprehensive_benchmark()  # All 18 estimators
results = benchmark.run_classical_benchmark()      # Classical only
results = benchmark.run_contamination_robustness_test()  # Contamination analysis
```

### **Estimators**
All estimators can be imported directly from the main package:

```python
# Classical estimators
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
from lrdbench.analysis.temporal.dma.dma_estimator import DMAEstimator
from lrdbench.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator

from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
from lrdbench.analysis.spectral.whittle.whittle_estimator import WhittleEstimator
from lrdbench.analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator

from lrdbench.analysis.wavelet.cwt.cwt_estimator import CWTEstimator
from lrdbench.analysis.wavelet.variance.wavelet_variance_estimator import WaveletVarianceEstimator
from lrdbench.analysis.wavelet.log_variance.wavelet_log_variance_estimator import WaveletLogVarianceEstimator
from lrdbench.analysis.wavelet.whittle.wavelet_whittle_estimator import WaveletWhittleEstimator

from lrdbench.analysis.multifractal.mfdfa.mfdfa_estimator import MFDFAEstimator
from lrdbench.analysis.multifractal.wavelet_leaders.multifractal_wavelet_leaders_estimator import MultifractalWaveletLeadersEstimator

# Enhanced ML and Neural Network estimators (NEW!)
from lrdbench import (
    CNNEstimator,           # Enhanced CNN with residual connections
    LSTMEstimator,          # Enhanced LSTM with bidirectional architecture
    GRUEstimator,           # Enhanced GRU with attention mechanisms
    TransformerEstimator,   # Enhanced Transformer with self-attention
    RandomForestEstimator,  # Enhanced Random Forest
    SVREstimator,           # Enhanced SVR
    GradientBoostingEstimator  # Enhanced Gradient Boosting
)
```

### **Data Models**
Synthetic data generators:

```python
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel
from lrdbench.models.data_models.mrw.mrw_model import MultifractalRandomWalk

# Generate synthetic data
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)
```

### **Enhanced Neural Network Models**
Production-ready neural estimators with pre-trained models:

```python
# All enhanced estimators are available directly
from lrdbench import CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator

# Use immediately with pre-trained models
cnn_estimator = CNNEstimator()
result = cnn_estimator.estimate(data)
print(f"Hurst parameter: {result['hurst_parameter']:.3f}")

# LSTM estimator
lstm_estimator = LSTMEstimator()
result = lstm_estimator.estimate(data)

# GRU estimator
gru_estimator = GRUEstimator()
result = gru_estimator.estimate(data)

# Transformer estimator
transformer_estimator = TransformerEstimator()
result = transformer_estimator.estimate(data)
```

## 📊 **Usage Examples**

### **Basic Benchmark**
```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type='comprehensive',
    contamination_type='additive_gaussian',
    contamination_level=0.1
)
```

### **Individual Estimator**
```python
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator

rs_estimator = RSEstimator()
result = rs_estimator.estimate(data)
print(f"Hurst parameter: {result['hurst_parameter']}")
```

### **Enhanced Neural Network Estimators**
```python
# Import enhanced estimators
from lrdbench import CNNEstimator, LSTMEstimator

# Use CNN estimator
cnn_estimator = CNNEstimator()
result = cnn_estimator.estimate(data)
print(f"CNN Hurst: {result['hurst_parameter']:.3f}")

# Use LSTM estimator
lstm_estimator = LSTMEstimator()
result = lstm_estimator.estimate(data)
print(f"LSTM Hurst: {result['hurst_parameter']:.3f}")
```

### **Data Generation**
```python
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion

# Generate fBm data
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)

# Generate ARFIMA data
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel
arfima = ARFIMAModel(d=0.3, ar_params=[0.5], ma_params=[0.2])
data = arfima.generate(1000, seed=42)
```

## 🎯 **Available Estimators**

### **Temporal Domain (4)**
- **R/S**: Rescaled Range Analysis
- **DFA**: Detrended Fluctuation Analysis  
- **DMA**: Detrending Moving Average
- **Higuchi**: Higuchi Method

### **Spectral Domain (3)**
- **GPH**: Geweke-Porter-Hudak
- **Whittle**: Whittle Estimator
- **Periodogram**: Periodogram Method

### **Wavelet Domain (4)**
- **CWT**: Continuous Wavelet Transform
- **Wavelet Variance**: Wavelet Variance Method
- **Wavelet Log Variance**: Wavelet Log Variance Method
- **Wavelet Whittle**: Wavelet Whittle Method

### **Multifractal (2)**
- **MFDFA**: Multifractal Detrended Fluctuation Analysis
- **Wavelet Leaders**: Multifractal Wavelet Leaders

### **🤖 Machine Learning (3)**
- **Random Forest**: Enhanced Random Forest Regression
- **Gradient Boosting**: Enhanced Gradient Boosting Regression
- **SVR**: Enhanced Support Vector Regression

### **🧠 Enhanced Neural Networks (4)**
- **CNN**: Enhanced Convolutional Neural Network with residual connections
- **LSTM**: Enhanced Long Short-Term Memory with bidirectional architecture
- **GRU**: Enhanced Gated Recurrent Unit with attention mechanisms
- **Transformer**: Enhanced Transformer with multi-head self-attention

## 🔍 **Detailed Documentation**

For detailed information about each component, see the specific documentation files:

- [**Estimators**](estimators/) - Detailed API for each estimator type
- [**Models**](models/) - Data model implementations and usage
- [**Benchmark**](estimators/benchmark.md) - Comprehensive benchmarking system
- [**Enhanced Neural Models**](../../ENHANCED_NEURAL_MODELS.md) - Complete guide to ML and neural estimators

## 💡 **Best Practices**

1. **Use the ComprehensiveBenchmark** for systematic evaluation
2. **Enhanced neural estimators** are ready for production use with pre-trained models
3. **Adaptive input handling** automatically adjusts to different sequence lengths
4. **Contamination testing** helps assess robustness across all estimator types
5. **Check success rates** before relying on results
6. **Import enhanced estimators directly** from the main package for simplicity

## 🚨 **Common Issues**

- **Import errors**: Use `from lrdbench import CNNEstimator` for enhanced estimators
- **Data length**: Wavelet estimators require sufficient data length
- **Dependencies**: Install PyTorch for neural estimators, scikit-learn for ML estimators
- **Memory**: Large datasets may require chunked processing
- **Model loading**: Enhanced estimators automatically handle pre-trained model loading

## 🆕 **What's New in v1.6.0**

- **18 Total Estimators**: Increased from 12 to 18 with enhanced ML and neural methods
- **Simplified Imports**: Enhanced estimators available directly from main package
- **Pre-trained Models**: All neural estimators work immediately without training
- **Adaptive Architecture**: Automatic handling of different input sizes
- **Production Ready**: Enhanced estimators designed for real-world applications

---

**For questions or issues, please refer to the main project documentation, enhanced neural models guide, or create an issue on GitHub.**
