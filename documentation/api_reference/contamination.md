# Data Contamination System

## Overview

The LRDBenchmark contamination system provides comprehensive tools for testing estimator robustness under various real-world data conditions. This system allows researchers to evaluate how well different long-range dependence estimators perform when faced with common data quality issues.

## Contamination Types

### 1. Trend Contamination

#### Linear Trend
```python
from lrdbench.models.contamination.contamination_models import ContaminationModel

contamination_model = ContaminationModel()
data_with_trend = contamination_model.add_trend_linear(data, slope=0.01)
```

#### Polynomial Trend
```python
data_with_poly_trend = contamination_model.add_trend_polynomial(data, degree=2, coefficient=0.001)
```

#### Exponential Trend
```python
data_with_exp_trend = contamination_model.add_trend_exponential(data, rate=0.01)
```

#### Seasonal Trend
```python
data_with_seasonal = contamination_model.add_trend_seasonal(data, period=100, amplitude=0.5)
```

### 2. Noise Contamination

#### Gaussian Noise
```python
data_with_noise = contamination_model.add_noise_gaussian(data, std=0.1)
```

#### Colored Noise
```python
data_with_colored_noise = contamination_model.add_noise_colored(data, power=1.0)
```

#### Impulsive Noise
```python
data_with_impulsive = contamination_model.add_noise_impulsive(data, probability=0.01)
```

### 3. Artifact Contamination

#### Spikes
```python
data_with_spikes = contamination_model.add_artifact_spikes(data, probability=0.01)
```

#### Level Shifts
```python
data_with_shifts = contamination_model.add_artifact_level_shifts(data, probability=0.005)
```

#### Missing Data
```python
data_with_missing = contamination_model.add_artifact_missing_data(data, probability=0.02)
```

### 4. Sampling Issues

#### Irregular Sampling
```python
data_irregular = contamination_model.add_sampling_irregular(data, probability=0.1)
```

#### Aliasing
```python
data_aliased = contamination_model.add_sampling_aliasing(data, frequency=0.1)
```

### 5. Measurement Errors

#### Systematic Bias
```python
data_with_bias = contamination_model.add_measurement_systematic(data, bias=0.1)
```

#### Random Measurement Error
```python
data_with_error = contamination_model.add_measurement_random(data, std=0.05)
```

## Contamination Configuration

### ContaminationConfig

```python
from lrdbench.models.contamination.contamination_models import ContaminationConfig

config = ContaminationConfig(
    trend_slope=0.01,
    trend_polynomial_degree=2,
    trend_exponential_rate=0.1,
    trend_seasonal_period=100,
    trend_seasonal_amplitude=0.5,
    artifact_spike_probability=0.01,
    artifact_spike_amplitude=3.0,
    artifact_level_shift_probability=0.005,
    artifact_level_shift_amplitude=2.0,
    artifact_missing_probability=0.02,
    noise_gaussian_std=0.1,
    noise_colored_power=1.0,
    noise_impulsive_probability=0.005,
    noise_impulsive_amplitude=5.0,
    sampling_irregular_probability=0.1,
    sampling_aliasing_frequency=0.1,
    measurement_systematic_bias=0.1,
    measurement_random_std=0.05
)

contamination_model = ContaminationModel(config)
```

## Robustness Analysis

### Running Contamination Tests

```python
from lrdbench import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Run contamination robustness analysis
estimators = ['DFA', 'RS', 'GPH', 'CWT', 'Periodogram']

# Enhanced ML and Neural Network estimators (NEW!)
if hasattr(lrdbench, 'CNNEstimator'):
    estimators.extend(['CNN', 'LSTM', 'GRU', 'Transformer', 'RandomForest', 'SVR'])

results = benchmark.run_contamination_robustness_test(
    data=data,
    contamination_types=['gaussian_noise', 'linear_trend', 'spikes'],
    estimators=estimators
)
```

### Interpreting Results

The contamination analysis returns:

- **Robustness Score**: Percentage of performance maintained under contamination
- **Performance Change**: Absolute change in Hurst parameter estimation
- **Ranking**: Estimators ranked by robustness across contamination types

## Web Dashboard Integration

The contamination system is fully integrated into the LRDBenchmark web dashboard:

1. **Data Generation Tab**: Apply contamination during data generation
2. **Contamination Analysis Tab**: Run comprehensive robustness tests
3. **Results Visualization**: Heatmaps and rankings of estimator performance

### Dashboard Features

- **13 Contamination Types**: All contamination methods available
- **18 Total Estimators**: Complete coverage including enhanced ML and neural methods
- **Intensity Controls**: Adjustable contamination strength
- **Real-time Application**: Apply contamination during data generation
- **Robustness Testing**: Automated testing across multiple scenarios
- **Performance Ranking**: Visual comparison of estimator robustness

## Best Practices

### 1. Systematic Testing
```python
# Test multiple contamination levels
contamination_levels = [0.01, 0.05, 0.1, 0.2]
for level in contamination_levels:
    contaminated_data = contamination_model.add_noise_gaussian(data, std=level)
    results = benchmark.run_benchmark(contaminated_data)
```

### 2. Multiple Contamination Types
```python
# Apply multiple contamination types
contamination_types = [
    lambda d: contamination_model.add_noise_gaussian(d, std=0.1),
    lambda d: contamination_model.add_trend_linear(d, slope=0.01),
    lambda d: contamination_model.add_artifact_spikes(d, probability=0.01)
]

for contam_func in contamination_types:
    contaminated_data = contam_func(data.copy())
    # Run analysis
```

### 3. Robustness Comparison
```python
# Compare estimators across contamination scenarios
estimators = ['DFA', 'RS', 'GPH', 'CWT', 'Wavelet Variance', 'Periodogram']
contamination_scenarios = ['clean', 'noise', 'trend', 'spikes']

# Enhanced ML and Neural Network estimators (NEW!)
if hasattr(lrdbench, 'CNNEstimator'):
    estimators.extend(['CNN', 'LSTM', 'GRU', 'Transformer', 'RandomForest', 'SVR'])

robustness_matrix = benchmark.compare_robustness(
    data=data,
    estimators=estimators,
    scenarios=contamination_scenarios
)
```

## Performance Considerations

- **Memory Efficient**: Contamination models use in-place operations
- **Configurable Intensity**: Adjust contamination strength for different scenarios
- **Batch Processing**: Apply multiple contamination types efficiently
- **Validation**: Built-in validation for contamination parameters

## Integration with Auto-Optimization

The contamination system works seamlessly with the auto-optimization system:

```python
# Auto-optimized estimators with contamination testing
from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator

estimator = AutoOptimizedEstimator('dfa')
contaminated_data = contamination_model.add_noise_gaussian(data, std=0.1)
result = estimator.estimate(contaminated_data)
```

This combination provides both performance optimization and robustness testing for comprehensive long-range dependence analysis.
