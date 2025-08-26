# Getting Started Guide

Welcome to LRDBench! This guide will help you get up and running quickly with long-range dependence estimation.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Concepts](#basic-concepts)
4. [First Examples](#first-examples)
5. [Next Steps](#next-steps)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install from PyPI

```bash
pip install lrdbench
```

### Step 2: Verify Installation

```python
import lrdbench
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

print("Installation successful!")
```

## Quick Start

### Generate Your First Synthetic Data

```python
# Import the fBm model
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion

# Create a model with Hurst parameter H = 0.7
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)

# Generate 1000 data points
data = fbm.generate(1000, seed=42)

# Plot the data
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(data)
plt.title('Fractional Brownian Motion (H = 0.7)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

### Estimate Parameters from Data

```python
# Import the DFA estimator
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator

# Create estimator
dfa = DFAEstimator(min_box_size=8, max_box_size=200)

# Estimate Hurst parameter
results = dfa.estimate(data)

print(f"Estimated Hurst parameter: {results['hurst_parameter']:.3f}")
print(f"R-squared: {results['r_squared']:.3f}")

# Plot the scaling relationship
dfa.plot_scaling()
```

### Run a Comprehensive Benchmark

```python
# Import the benchmark system
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Initialize benchmark
benchmark = ComprehensiveBenchmark()

# Run comprehensive test
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    contamination_type='additive_gaussian',
    contamination_level=0.1
)

print(f"Success rate: {results['success_rate']:.1%}")
print(f"Total tests: {results['total_tests']}")

## Data Contamination Analysis

LRDBenchmark includes a comprehensive contamination testing system:

```python
from lrdbench.models.contamination.contamination_models import ContaminationModel

# Create contamination model
contamination_model = ContaminationModel()

# Apply different types of contamination
data_with_noise = contamination_model.add_noise_gaussian(data, std=0.1)
data_with_trend = contamination_model.add_trend_linear(data, slope=0.01)
data_with_spikes = contamination_model.add_artifact_spikes(data, probability=0.01)

# Test estimator robustness
results = benchmark.run_contamination_robustness_test()
```
```

## Basic Concepts

### What is Long-Range Dependence?

Long-range dependence (LRD) is a property of time series where observations that are far apart in time are still correlated. This is quantified by the **Hurst parameter (H)**:

- **H > 0.5**: Persistent (positive correlations)
- **H < 0.5**: Anti-persistent (negative correlations)  
- **H = 0.5**: Independent (no long-range correlations)

### Available Data Models

1. **fBm (Fractional Brownian Motion)**: Self-similar Gaussian process
2. **fGn (Fractional Gaussian Noise)**: Stationary increments of fBm
3. **ARFIMA**: AutoRegressive Fractionally Integrated Moving Average
4. **MRW (Multifractal Random Walk)**: Non-Gaussian multifractal process

### Available Estimators

#### Classical Estimators (12 total)
- **Temporal**: R/S, DFA, DMA, Higuchi
- **Spectral**: GPH, Whittle, Periodogram
- **Wavelet**: CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle
- **Multifractal**: MFDFA

#### 🚀 Auto-Optimized Estimators
- All 12 estimators with NUMBA/JAX performance optimizations
- Automatic optimization selection (NUMBA → JAX → Standard)
- Up to 850x speedup with NUMBA optimizations
- Robust error handling and fallback mechanisms

## First Examples

### Example 1: Compare Multiple Estimators

```python
import numpy as np
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
from lrdbench.analysis.wavelet.cwt.cwt_estimator import CWTEstimator

# Generate test data
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)

# Test different estimators
estimators = {
    'R/S': RSEstimator(),
    'DFA': DFAEstimator(),
    'CWT': CWTEstimator()
}

print("Estimator Comparison:")
print("-" * 40)
for name, estimator in estimators.items():
    try:
        result = estimator.estimate(data)
        h_est = result['hurst_parameter']
        error = abs(h_est - 0.7)
        print(f"{name:>8}: H_est={h_est:.4f}, Error={error:.4f}")
    except Exception as e:
        print(f"{name:>8}: Failed - {e}")
```

### Example 2: Test with Contamination

```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Initialize benchmark
benchmark = ComprehensiveBenchmark()

# Test robustness under different contamination types
contamination_types = ['additive_gaussian', 'outliers', 'trend']

for cont_type in contamination_types:
    results = benchmark.run_classical_benchmark(
        data_length=500,
        contamination_type=cont_type,
        contamination_level=0.2
    )
    print(f"{cont_type:>20}: {results['success_rate']:.1%} success rate")
```

### Example 3: Use Pre-trained Models

```python
from lrdbench.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
from lrdbench.models.pretrained_models.transformer_pretrained import TransformerPretrainedModel

# Load pre-trained models
cnn_model = CNNPretrainedModel(input_length=500)
transformer_model = TransformerPretrainedModel(input_length=500)

# Generate test data
fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)

# Use models immediately (no training required)
cnn_result = cnn_model.estimate(data)
transformer_result = transformer_model.estimate(data)

print(f"CNN: H_est={cnn_result['hurst_parameter']:.4f}")
print(f"Transformer: H_est={transformer_result['hurst_parameter']:.4f}")
```

## Next Steps

### 1. Explore the API Reference
- [Complete API Reference](../api_reference/COMPLETE_API_REFERENCE.md)
- [Benchmark Documentation](../api_reference/estimators/benchmark.md)
- [Individual Estimator Docs](../api_reference/estimators/)

### 2. Run the Demos
- [Comprehensive API Demo](../../demos/comprehensive_api_demo.py)
- [CPU-Based Demos](../../demos/cpu_based/)
- [GPU-Based Demos](../../demos/gpu_based/)

### 3. Learn Advanced Features
- **Contamination Testing**: Test estimator robustness
- **Adaptive Wavelet Scaling**: Automatic scale optimization
- **High-Performance Versions**: JAX and Numba optimized estimators
- **Benchmark System**: Systematic performance evaluation

### 4. Contribute and Extend
- Add new estimators
- Implement new data models
- Optimize existing methods
- Improve documentation

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # ❌ Wrong
   from analysis.benchmark import ComprehensiveBenchmark
   
   # ✅ Correct
   from lrdbench.analysis.benchmark import ComprehensiveBenchmark
   ```

2. **Data Length Issues**
   - Wavelet estimators require ≥100 points
   - Recommended: ≥500 points for reliable results
   - Very long data (>10,000 points) may be slow

3. **Memory Issues**
   - Reduce data length
   - Use smaller contamination levels
   - Process data in chunks

### Getting Help

- Check the [API Reference](../api_reference/)
- Run the [demos](../../demos/) for examples
- Review the [project README](../../README.md)
- Create an issue on GitHub for bugs

---

**Welcome to LRDBench! Start exploring long-range dependence estimation today.**
