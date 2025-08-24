# 🔬 **ComprehensiveBenchmark**

The `ComprehensiveBenchmark` class is the main entry point for running systematic evaluations of all long-range dependence estimators in LRDBench.

## 📦 **Import**

```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark
```

## 🚀 **Quick Start**

```python
# Initialize the benchmark system
benchmark = ComprehensiveBenchmark()

# Run a comprehensive benchmark (all estimators)
results = benchmark.run_comprehensive_benchmark()

# Run specific benchmark types
results_classical = benchmark.run_classical_benchmark()
results_ml = benchmark.run_ml_benchmark()
results_neural = benchmark.run_neural_benchmark()
```

## 🏗️ **Class Definition**

```python
class ComprehensiveBenchmark:
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the benchmark system.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save benchmark results (default: "benchmark_results")
        """
```

## 📊 **Available Benchmark Types**

### **1. Comprehensive Benchmark**
Tests all available estimators:
```python
results = benchmark.run_comprehensive_benchmark(
    data_length=1000,
    benchmark_type='comprehensive',
    contamination_type='additive_gaussian',
    contamination_level=0.1,
    save_results=True
)
```

### **2. Classical Benchmark**
Tests only classical statistical estimators:
```python
results = benchmark.run_classical_benchmark(
    data_length=1000,
    contamination_type='outliers',
    contamination_level=0.2
)
```

### **3. ML Benchmark**
Tests only machine learning estimators:
```python
results = benchmark.run_ml_benchmark(
    data_length=1000,
    contamination_type='trend',
    contamination_level=0.15
)
```

### **4. Neural Benchmark**
Tests only neural network estimators:
```python
results = benchmark.run_neural_benchmark(
    data_length=1000,
    contamination_type='seasonal',
    contamination_level=0.1
)
```

## 🎯 **Parameters**

### **Common Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_length` | int | 1000 | Length of synthetic data to generate |
| `contamination_type` | str | None | Type of contamination to apply |
| `contamination_level` | float | 0.1 | Intensity of contamination (0.0 to 1.0) |
| `save_results` | bool | True | Whether to save results to files |

### **Contamination Types**

| Type | Description | Parameters |
|------|-------------|------------|
| `'additive_gaussian'` | Add Gaussian noise | `noise_level`, `std` |
| `'multiplicative_noise'` | Multiplicative noise | `noise_level`, `std` |
| `'outliers'` | Add outliers | `outlier_fraction`, `outlier_magnitude` |
| `'trend'` | Add linear trend | `trend_strength`, `trend_type` |
| `'seasonal'` | Add seasonal pattern | `seasonal_strength`, `period` |
| `'missing_data'` | Remove data points | `missing_fraction`, `missing_pattern` |

## 📈 **Output Structure**

The benchmark returns a comprehensive results dictionary:

```python
{
    'timestamp': '2025-08-24T12:00:00',
    'benchmark_type': 'comprehensive',
    'contamination_type': 'additive_gaussian',
    'contamination_level': 0.1,
    'total_tests': 52,
    'successful_tests': 40,
    'success_rate': 0.769,
    'data_models_tested': 4,
    'estimators_tested': 13,
    'results': {
        'fBm': {
            'data_params': {'H': 0.7, 'sigma': 1.0},
            'estimator_results': [...]
        },
        'fGn': {...},
        'ARFIMAModel': {...},
        'MRW': {...}
    }
}
```

### **Estimator Results**

Each estimator result contains:

```python
{
    'estimator': 'R/S',
    'success': True,
    'execution_time': 0.005,
    'estimated_hurst': 0.6360,
    'true_hurst': 0.7000,
    'error': 0.0640,
    'r_squared': 0.95,
    'p_value': 0.001,
    'intercept': 0.1,
    'slope': 0.5,
    'std_error': 0.02,
    'full_result': {...}
}
```

## 🔍 **Data Models Tested**

The benchmark automatically tests with these synthetic data models:

1. **fBm** (Fractional Brownian Motion)
   - Parameters: `H=0.7`, `sigma=1.0`
   - Generates: Self-similar Gaussian process

2. **fGn** (Fractional Gaussian Noise)
   - Parameters: `H=0.7`, `sigma=1.0`
   - Generates: Stationary increments of fBm

3. **ARFIMAModel** (ARFIMA)
   - Parameters: `d=0.3`, `ar_params=[0.5]`, `ma_params=[0.2]`
   - Generates: Long-memory time series

4. **MRW** (Multifractal Random Walk)
   - Parameters: `H=0.7`, `lambda_param=0.5`, `sigma=1.0`
   - Generates: Non-Gaussian multifractal process

## 🎯 **Estimators Tested**

### **Classical Estimators (13 total)**
- **Temporal**: R/S, DFA, DMA, Higuchi
- **Spectral**: GPH, Whittle, Periodogram
- **Wavelet**: CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle
- **Multifractal**: MFDFA, Wavelet Leaders

### **ML Estimators (3 total)**
- Random Forest, Gradient Boosting, SVR

### **Neural Estimators (2 total)**
- CNN, Transformer

## 💾 **Result Saving**

Results are automatically saved to:

- **JSON**: `comprehensive_benchmark_YYYYMMDD_HHMMSS.json`
- **CSV**: `benchmark_summary_YYYYMMDD_HHMMSS.csv`

### **Custom Output Directory**
```python
benchmark = ComprehensiveBenchmark(output_dir="my_results")
```

## 📊 **Performance Analysis**

### **Top Performers**
The benchmark automatically ranks estimators by:
- Average error across all data models
- Execution time
- Success rate

### **Detailed Breakdown**
- Top 3 estimators for each data model
- Error ranges and execution time statistics
- Contamination impact analysis

## 🔧 **Advanced Usage**

### **Custom Data Lengths**
```python
# Test with different data lengths
for length in [100, 500, 1000, 2000]:
    results = benchmark.run_comprehensive_benchmark(data_length=length)
    print(f"Length {length}: {results['success_rate']:.1%} success rate")
```

### **Contamination Testing**
```python
# Test robustness under different conditions
contamination_types = ['additive_gaussian', 'outliers', 'trend']
for cont_type in contamination_types:
    results = benchmark.run_comprehensive_benchmark(
        contamination_type=cont_type,
        contamination_level=0.2
    )
    print(f"{cont_type}: {results['success_rate']:.1%} success rate")
```

### **Benchmark Comparison**
```python
# Compare different estimator categories
classical_results = benchmark.run_classical_benchmark()
ml_results = benchmark.run_ml_benchmark()
neural_results = benchmark.run_neural_benchmark()

print(f"Classical: {classical_results['success_rate']:.1%}")
print(f"ML: {ml_results['success_rate']:.1%}")
print(f"Neural: {neural_results['success_rate']:.1%}")
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Low Success Rates**
   - Increase `data_length` for wavelet estimators
   - Reduce `contamination_level`
   - Check estimator parameters

2. **Memory Issues**
   - Reduce `data_length`
   - Use smaller contamination levels
   - Process results in chunks

3. **Import Errors**
   - Ensure package is installed: `pip install lrdbench`
   - Use correct import paths: `from lrdbench.analysis.benchmark import ComprehensiveBenchmark`

### **Performance Tips**

- **Wavelet estimators** work best with longer data (>500 points)
- **ML estimators** are fastest for large datasets
- **Neural estimators** provide best accuracy but slower execution
- **Classical estimators** offer good balance of speed and accuracy

## 📚 **Examples**

### **Complete Benchmark Example**
```python
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Initialize
benchmark = ComprehensiveBenchmark(output_dir="my_benchmark_results")

# Run comprehensive test
results = benchmark.run_comprehensive_benchmark(
    data_length=2000,
    contamination_type='additive_gaussian',
    contamination_level=0.15,
    save_results=True
)

# Analyze results
print(f"Overall success rate: {results['success_rate']:.1%}")
print(f"Total tests: {results['total_tests']}")
print(f"Successful: {results['successful_tests']}")

# Access specific results
for model_name, model_data in results['results'].items():
    print(f"\n{model_name}:")
    for est_result in model_data['estimator_results']:
        if est_result['success']:
            print(f"  {est_result['estimator']}: H_est={est_result['estimated_hurst']:.4f}, Error={est_result['error']:.4f}")
```

---

**For more information, see the main [API Reference](../README.md) or the [project documentation](../../../README.md).**
