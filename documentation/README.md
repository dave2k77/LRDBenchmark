# Documentation

This directory contains comprehensive documentation for LRDBench, a framework for long-range dependence estimation.

## Documentation Structure

### 📚 **API Reference**
- **Complete API Reference**: Comprehensive documentation for all components
- **Models**: API documentation for all stochastic data models
- **Estimators**: API documentation for all parameter estimation methods
- **Benchmark**: Comprehensive benchmarking system documentation

### 📖 **User Guides**
- **Getting Started**: Quick start guide and installation instructions
- **Examples**: Working examples and use cases
- **Demos**: Interactive demonstration scripts

### 🔬 **Technical Documentation**
- **Model Theory**: Mathematical foundations and theoretical background
- **Implementation Details**: Algorithm details and computational methods
- **Performance Analysis**: Benchmarking and optimization guides

### 📊 **Research Documentation**
- **Methodology**: Research methodology and validation approaches
- **Validation Studies**: Comprehensive validation of models and estimators
- **Comparison Studies**: Comparative analysis of different methods

## Quick Navigation

- [Getting Started Guide](user_guides/getting_started.md)
- [Complete API Reference](api_reference/COMPLETE_API_REFERENCE.md)
- [API Reference Overview](api_reference/README.md)
- [Model Theory](technical/model_theory.md)
- [Project Instructions](project_instructions.md)

## Package Structure

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

## Available Components

### Data Models (4 total)
- **fBm**: Fractional Brownian Motion
- **fGn**: Fractional Gaussian Noise  
- **ARFIMA**: AutoRegressive Fractionally Integrated Moving Average
- **MRW**: Multifractal Random Walk

### Estimators (18 total)
- **Classical (13)**: R/S, DFA, DMA, Higuchi, GPH, Whittle, Periodogram, CWT, Wavelet Variance, Wavelet Log Variance, Wavelet Whittle, MFDFA, Wavelet Leaders
- **ML (3)**: Random Forest, Gradient Boosting, SVR
- **Neural (2)**: CNN, Transformer

### Core Features
- **Comprehensive Benchmarking**: Systematic evaluation of all estimators
- **Contamination Testing**: Robustness assessment under various conditions
- **Adaptive Wavelet Scaling**: Automatic scale optimization
- **Pre-trained Models**: Production-ready ML and neural models
- **High-Performance Options**: GPU acceleration with JAX

## Installation and Usage

### Quick Start
```bash
pip install lrdbench
```

```python
import lrdbench
from lrdbench.analysis.benchmark import ComprehensiveBenchmark

# Run comprehensive benchmark
benchmark = ComprehensiveBenchmark()
results = benchmark.run_comprehensive_benchmark()
```

## Contributing to Documentation

When adding new features or models, please update the relevant documentation sections:

1. **API Reference**: Add complete docstrings and parameter descriptions
2. **User Guides**: Create tutorials and examples
3. **Technical Docs**: Document mathematical foundations
4. **Research Docs**: Include validation and comparison results

## Documentation Standards

- Use clear, concise language
- Include mathematical formulas where appropriate
- Provide code examples for all major functions
- Include references to relevant literature
- Maintain consistent formatting and structure
- Use the new `lrdbench` package import paths

## Getting Help

- **API Reference**: [Complete API Reference](api_reference/COMPLETE_API_REFERENCE.md)
- **Examples**: [Demo Scripts](../../demos/)
- **Project Overview**: [Main README](../../README.md)
- **Issues**: Create an issue on GitHub for bugs or questions

---

**LRDBench provides comprehensive tools for long-range dependence estimation with production-ready components and extensive documentation.**
