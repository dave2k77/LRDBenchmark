# 📚 **LRDBench API Reference**

Welcome to the comprehensive API reference for LRDBench, a framework for long-range dependence estimation.

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
│   └── multifractal/             # Multifractal estimators
└── models/                       # Data models and utilities
    ├── data_models/              # Synthetic data generators
    ├── contamination/            # Data contamination models
    └── pretrained_models/        # Pre-trained models
```

## 🔧 **Core Components**

### **ComprehensiveBenchmark**
The main entry point for running benchmarks:

```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Run different types of benchmarks
results = benchmark.run_comprehensive_benchmark()  # All estimators
results = benchmark.run_classical_benchmark()      # Classical only
results = benchmark.run_contamination_robustness_test()  # Contamination analysis
```

### **Estimators**
All estimators can be imported directly:

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

# Machine Learning estimators
from lrdbench.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
from lrdbench.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbench.analysis.machine_learning.svr_estimator import SVREstimator

# Neural Network estimators
from lrdbench.analysis.machine_learning.cnn_estimator import CNNEstimator
from lrdbench.analysis.machine_learning.transformer_estimator import TransformerEstimator
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

### **Pre-trained Models**
Production-ready models that don't require training:

```python
from lrdbench.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
from lrdbench.models.pretrained_models.transformer_pretrained import TransformerPretrainedModel
from lrdbench.models.pretrained_models.ml_pretrained import (
    RandomForestPretrainedModel, 
    SVREstimatorPretrainedModel, 
    GradientBoostingPretrainedModel
)

# Use pre-trained models immediately
cnn_model = CNNPretrainedModel(input_length=500)
result = cnn_model.estimate(data)
```

## 📊 **Usage Examples**

### **Basic Benchmark**
```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type='classical',
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

### **Machine Learning (3)**
- **Random Forest**: Random Forest Regression
- **Gradient Boosting**: Gradient Boosting Regression
- **SVR**: Support Vector Regression

### **Neural Networks (2)**
- **CNN**: Convolutional Neural Network
- **Transformer**: Transformer Encoder

## 🔍 **Detailed Documentation**

For detailed information about each component, see the specific documentation files:

- [**Estimators**](estimators/) - Detailed API for each estimator type
- [**Models**](models/) - Data model implementations and usage
- [**Benchmark**](estimators/benchmark.md) - Comprehensive benchmarking system

## 💡 **Best Practices**

1. **Use the ComprehensiveBenchmark** for systematic evaluation
2. **Pre-trained models** are ready for production use
3. **Adaptive wavelet scaling** automatically adjusts to data length
4. **Contamination testing** helps assess robustness
5. **Check success rates** before relying on results

## 🚨 **Common Issues**

- **Import errors**: Ensure you're using `from lrdbench.analysis...` not `from analysis...`
- **Data length**: Wavelet estimators require sufficient data length
- **Dependencies**: Install optional dependencies for full functionality
- **Memory**: Large datasets may require chunked processing

---

**For questions or issues, please refer to the main project documentation or create an issue on GitHub.**
