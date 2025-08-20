# Synthetic Data Generation and Analysis Project

A comprehensive repository for exploring synthetic data generation techniques and estimation methods for various stochastic processes.

## Project Overview

This project focuses on implementing and analyzing five key stochastic models:
- **ARFIMA** (AutoRegressive Fractionally Integrated Moving Average)
- **fBm** (Fractional Brownian Motion)
- **fGn** (Fractional Gaussian Noise)
- **MRW** (Multifractal Random Walk)
- **Neural fSDE** (Neural network-based fractional SDEs)

## Project Status

🎉 **PROJECT COMPLETE - 100%** 🎉

All major components have been successfully implemented and tested:

- ✅ **Data Models**: 5/5 models fully implemented and optimized
- ✅ **Estimators**: 13/13 estimators with comprehensive testing
- ✅ **High-Performance**: JAX and Numba optimized versions
- ✅ **Neural fSDE**: Hybrid JAX/PyTorch implementation
- ✅ **Documentation**: Complete API reference and user guides
- ✅ **Testing**: 144 tests passing (100% success rate)
- ✅ **Demos**: 8 comprehensive demonstration scripts
- ✅ **Real-World Confounds**: Contamination models and robustness testing

## Project Structure

```
DataExploratoryProject/
├── models/
│   ├── data_models/          # Model implementations
│   │   ├── arfima/          # ARFIMA model
│   │   ├── fbm/             # Fractional Brownian Motion
│   │   ├── fgn/             # Fractional Gaussian Noise
│   │   ├── mrw/             # Multifractal Random Walk
│   │   └── neural_fsde/     # Neural fSDE models
│   └── estimators/          # Parameter estimation methods
├── analysis/                # Analysis and estimation results
│   ├── temporal/            # Time-domain estimators
│   │   ├── dfa/            # Detrended Fluctuation Analysis
│   │   ├── rs/             # R/S Analysis
│   │   ├── higuchi/        # Higuchi method
│   │   └── dma/            # Detrending Moving Average
│   ├── spectral/           # Frequency-domain estimators
│   │   ├── periodogram/    # Periodogram method
│   │   ├── whittle/        # Whittle estimator
│   │   └── gph/            # Geweke-Porter-Hudak estimator
│   ├── wavelet/            # Wavelet-based estimators
│   │   ├── log_variance/   # Wavelet Log Variance
│   │   ├── variance/       # Wavelet Variance
│   │   ├── whittle/        # Wavelet Whittle
│   │   └── cwt/            # Continuous Wavelet Transform
│   ├── multifractal/       # Multifractal analysis
│   │   ├── mfdfa/          # Multifractal Detrended Fluctuation Analysis
│   │   └── wavelet_leaders/ # Multifractal Wavelet Leaders
│   └── high_performance/   # Optimized implementations
│       ├── jax/            # JAX-optimized versions
│       └── numba/          # Numba-optimized versions
├── tests/                  # Unit tests and validation
├── documentation/          # Comprehensive documentation
├── results/               # Results and outputs
│   └── plots/             # Generated plots and visualizations
└── venv/                  # Virtual environment
```

## Models

### ARFIMA (AutoRegressive Fractionally Integrated Moving Average)
- Long-memory time series model
- Combines ARMA with fractional differencing
- Useful for modeling persistent time series

### fBm (Fractional Brownian Motion)
- Self-similar Gaussian process
- Characterized by Hurst parameter H
- Exhibits long-range dependence

### fGn (Fractional Gaussian Noise)
- Increments of fractional Brownian motion
- Stationary process with long memory
- Related to fBm through differencing

### MRW (Multifractal Random Walk)
- Non-Gaussian multifractal process
- Exhibits scale-invariant properties
- Characterized by multifractal spectrum

### Neural fSDE (Neural Fractional Stochastic Differential Equations)
- Neural network-based fractional SDEs
- Hybrid JAX/PyTorch implementation
- Multiple numerical schemes (Euler, Milstein, Heun)
- GPU acceleration support

## Estimators

### Temporal Estimators
- **DFA**: Detrended Fluctuation Analysis
- **R/S**: Rescaled Range Analysis
- **Higuchi**: Higuchi's fractal dimension method
- **DMA**: Detrending Moving Average

### Spectral Estimators
- **Periodogram**: Power spectral density estimation
- **Whittle**: Maximum likelihood estimation in frequency domain
- **GPH**: Geweke-Porter-Hudak estimator for long memory

### Wavelet Estimators
- **Wavelet Log Variance**: Log-variance of wavelet coefficients
- **Wavelet Variance**: Variance of wavelet coefficients
- **Wavelet Whittle**: Whittle estimation using wavelets
- **CWT**: Continuous Wavelet Transform analysis

### Multifractal Estimators
- **MFDFA**: Multifractal Detrended Fluctuation Analysis
- **Wavelet Leaders**: Multifractal analysis using wavelet leaders

### High-Performance Implementations
- **JAX**: GPU-accelerated implementations
- **Numba**: JIT-compiled implementations

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
import sys
sys.path.insert(0, '.')

# Generate synthetic data
from models.data_models.fbm.fbm_model import FractionalBrownianMotion
fbm = FractionalBrownianMotion(H=0.7)
data = fbm.generate(1000, seed=42)

# Estimate Hurst parameter
from analysis.temporal.dfa.dfa_estimator import DFAEstimator
estimator = DFAEstimator()
result = estimator.estimate(data)
print(f"Estimated H: {result['hurst_parameter']:.3f}")
```

### Neural fSDE Example

```python
from models.data_models.neural_fsde import create_fsde_net

# Create neural fSDE model
model = create_fsde_net(
    state_dim=1,
    hidden_dim=32,
    num_layers=3,
    hurst_parameter=0.7
)

# Simulate time series
trajectory = model.simulate(n_samples=1000, dt=0.01)
```

### Demo Scripts

Run the comprehensive demo scripts:

```bash
# CPU-based demos
python demos/cpu_based/comprehensive_model_demo.py
python demos/cpu_based/parameter_estimation_demo.py

# GPU-based demos (requires JAX)
python demos/gpu_based/jax_performance_demo.py
python demos/gpu_based/high_performance_comparison_demo.py
```

See the [demos/README.md](demos/README.md) for detailed information about all available demos.

## Contributing

[Guidelines for contributing to the project]

## License

[License information]

## References

- Beran, J. (1994). Statistics for Long-Memory Processes.
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature.
- Abry, P., & Veitch, D. (1998). Wavelet analysis of long-range-dependent traffic.
- Muzy, J. F., Bacry, E., & Arneodo, A. (1991). Wavelets and multifractal formalism for singular signals.
- Hayashi, K., & Nakagawa, K. (2022). fSDE-Net: Generating Time Series Data with Long-term Memory.
- Nakagawa, K., & Hayashi, K. (2024). Lf-Net: Generating Fractional Time-Series with Latent Fractional-Net.
