# Model Theory

This document provides the mathematical foundations and theoretical background for all stochastic models implemented in LRDBench.

## Table of Contents

1. [Fractional Brownian Motion (fBm)](#fractional-brownian-motion-fbm)
2. [Fractional Gaussian Noise (fGn)](#fractional-gaussian-noise-fgn)
3. [ARFIMA Models](#arfima-models)
4. [Multifractal Random Walk (MRW)](#multifractal-random-walk-mrw)
5. [Model Relationships](#model-relationships)
6. [Mathematical Properties](#mathematical-properties)
7. [Implementation Details](#implementation-details)

## Fractional Brownian Motion (fBm)

### Definition

Fractional Brownian Motion is a self-similar Gaussian process with stationary increments. For a Hurst parameter H ∈ (0, 1), fBm is defined as a Gaussian process B_H(t) with:

1. **Zero mean**: E[B_H(t)] = 0
2. **Covariance function**: E[B_H(t)B_H(s)] = (σ²/2)(|t|^(2H) + |s|^(2H) - |t-s|^(2H))
3. **Self-similarity**: B_H(at) = a^H B_H(t) for all a > 0
4. **Stationary increments**: B_H(t) - B_H(s) has the same distribution as B_H(t-s)

### Mathematical Properties

#### Self-Similarity

The self-similarity property states that:

B_H(at) = a^H B_H(t)

This means that scaling the time axis by a factor a scales the process by a^H.

#### Variance Scaling

The variance of fBm scales as:

Var(B_H(t)) = σ²|t|^(2H)

#### Long-Range Dependence

For H > 0.5, fBm exhibits long-range dependence:

- **Persistent**: H > 0.5 (positive correlations)
- **Anti-persistent**: H < 0.5 (negative correlations)
- **Independent**: H = 0.5 (standard Brownian motion)

#### Autocorrelation Function

The autocorrelation function of fBm increments (fGn) is:

ρ(k) = (1/2)(|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))

For large k, this behaves as:

ρ(k) ≈ H(2H-1)k^(2H-2)

#### Power Spectral Density

The power spectral density of fBm is:

S(f) = σ²|f|^(-2H-1)

### Generation Methods

#### Davies-Harte Method

The Davies-Harte method uses the spectral representation:

1. **Spectral Density**: S(f) = σ²(2 sin(πf/n))^(1-2H)
2. **Complex Noise**: Generate Z(f) ~ CN(0, 1)
3. **Filtering**: Y(f) = √S(f) Z(f)
4. **Inverse FFT**: B_H(t) = FFT^(-1)(Y(f))

#### Cholesky Decomposition

1. **Covariance Matrix**: C(i,j) = σ²/2(|i|^(2H) + |j|^(2H) - |i-j|^(2H))
2. **Cholesky Decomposition**: C = LL^T
3. **Generation**: B_H = LZ where Z ~ N(0, I)

#### Circulant Embedding

1. **Autocovariance**: γ(k) = σ²/2(|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
2. **Circulant Matrix**: Construct circulant matrix from γ(k)
3. **Eigenvalue Decomposition**: C = QΛQ^T
4. **Generation**: B_H = Q√ΛZ

### Implementation

**Location**: `lrdbench.models.data_models.fbm.fbm_model.FractionalBrownianMotion`

**Usage**:
```python
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion

fbm = FractionalBrownianMotion(H=0.7, sigma=1.0, method='davies_harte')
data = fbm.generate(1000, seed=42)
```

## Fractional Gaussian Noise (fGn)

### Definition

Fractional Gaussian Noise is the increments of fBm:

X(t) = B_H(t+1) - B_H(t)

### Properties

#### Stationarity

fGn is a stationary process with:

- **Mean**: E[X(t)] = 0
- **Variance**: Var(X(t)) = σ²
- **Autocorrelation**: ρ(k) = (1/2)(|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))

#### Long-Range Dependence

For H > 0.5, fGn exhibits long-range dependence with:

- **Persistent**: H > 0.5 (positive correlations)
- **Anti-persistent**: H < 0.5 (negative correlations)
- **Independent**: H = 0.5 (white noise)

#### Power Spectral Density

The power spectral density of fGn is:

S(f) = σ²(2 sin(πf))^(1-2H)

### Implementation

**Location**: `lrdbench.models.data_models.fgn.fgn_model.FractionalGaussianNoise`

**Usage**:
```python
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise

fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
data = fgn.generate(1000, seed=42)
```

## ARFIMA Models

### Definition

ARFIMA (AutoRegressive Fractionally Integrated Moving Average) models are defined as:

φ(B)(1-B)^d X_t = θ(B)ε_t

where:
- φ(B) = 1 - φ₁B - φ₂B² - ... - φ_p B^p (AR polynomial)
- θ(B) = 1 + θ₁B + θ₂B² + ... + θ_q B^q (MA polynomial)
- (1-B)^d is the fractional differencing operator
- ε_t ~ N(0, σ²) is white noise

### Mathematical Properties

#### Fractional Differencing

The fractional differencing operator (1-B)^d is defined as:

(1-B)^d = Σ_{k=0}^∞ (-1)^k (d choose k) B^k

where (d choose k) = Γ(d+1)/(Γ(k+1)Γ(d-k+1)) is the generalized binomial coefficient.

#### Long-Memory Property

For 0 < d < 0.5, the process exhibits long-memory with:

- **Autocorrelation**: ρ(k) ≈ ck^(2d-1) for large k
- **Power Spectrum**: S(f) ≈ c|f|^(-2d) for small f
- **Hurst Parameter**: H = d + 0.5

#### Stationarity and Invertibility

- **Stationary**: for -0.5 < d < 0.5
- **Invertible**: for -0.5 < d < 0.5
- **Long-memory**: for 0 < d < 0.5

### Implementation

**Location**: `lrdbench.models.data_models.arfima.arfima_model.ARFIMAModel`

**Usage**:
```python
from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel

arfima = ARFIMAModel(d=0.3, ar_params=[0.5], ma_params=[0.2], sigma=1.0)
data = arfima.generate(1000, seed=42)
```

**Performance Features**:
- **FFT-based fractional differencing**: O(n log n) complexity
- **Efficient AR/MA filtering**: Using scipy.signal.lfilter
- **Spectral method as default**: Optimal performance for most cases

## Multifractal Random Walk (MRW)

### Definition

Multifractal Random Walk is a non-Gaussian process that exhibits multifractal scaling properties. It is defined as:

X(t) = Σ_{i=1}^t ε_i exp(ω_i)

where:
- ε_i ~ N(0, σ²) are Gaussian innovations
- ω_i is a multifractal process with:
  - E[ω_i] = -λ²/2 (ensures E[X(t)] = 0)
  - Cov(ω_i, ω_j) = λ² log_2(|i-j|+1) for i ≠ j

### Mathematical Properties

#### Multifractal Spectrum

The multifractal spectrum f(α) describes the distribution of local Hölder exponents:

f(α) = 1 - (α - H)²/(2λ²)

where:
- α is the local Hölder exponent
- H is the Hurst parameter
- λ is the intermittency parameter

#### Scaling Properties

The q-th order structure function scales as:

S_q(τ) = E[|X(t+τ) - X(t)|^q] ≈ τ^ζ(q)

where the scaling exponents ζ(q) are:

ζ(q) = qH - λ²q(q-1)/2

#### Non-Gaussian Properties

- **Heavy tails**: Due to exponential multifractal modulation
- **Intermittency**: Clustering of extreme events
- **Scale invariance**: Self-similarity across multiple scales

### Implementation

**Location**: `lrdbench.models.data_models.mrw.mrw_model.MultifractalRandomWalk`

**Usage**:
```python
from lrdbench.models.data_models.mrw.mrw_model import MultifractalRandomWalk

mrw = MultifractalRandomWalk(H=0.7, lambda_param=0.5, sigma=1.0)
data = mrw.generate(1000, seed=42)
```

## Model Relationships

### fBm ↔ fGn

- **fGn to fBm**: B_H(t) = Σ_{i=1}^t X_i where X_i ~ fGn
- **fBm to fGn**: X_t = B_H(t) - B_H(t-1)

### ARFIMA ↔ fGn

- **ARFIMA to fGn**: For d = H - 0.5, ARFIMA(d) ≈ fGn(H)
- **fGn to ARFIMA**: fGn(H) can be approximated by ARFIMA(d = H - 0.5)

### MRW ↔ fBm

- **MRW to fBm**: For λ → 0, MRW → fBm
- **fBm to MRW**: fBm is the Gaussian limit of MRW

## Mathematical Properties

### Self-Similarity

All models exhibit self-similarity:

- **fBm**: B_H(at) = a^H B_H(t)
- **fGn**: X(at) = a^(H-1) X(t)
- **ARFIMA**: X(at) = a^(d-0.5) X(t)
- **MRW**: X(at) = a^H X(t) (in distribution)

### Long-Range Dependence

Long-range dependence is quantified by the Hurst parameter H:

- **fBm/fGn**: Direct parameter H
- **ARFIMA**: H = d + 0.5
- **MRW**: Direct parameter H

### Power Law Scaling

All models exhibit power law scaling:

- **Autocorrelation**: ρ(k) ≈ ck^(2H-2)
- **Power Spectrum**: S(f) ≈ c|f|^(-2H-1)
- **Structure Function**: S_q(τ) ≈ τ^ζ(q)

## Implementation Details

### Performance Optimizations

1. **FFT-based Methods**: Used for spectral generation (Davies-Harte)
2. **Vectorized Operations**: NumPy-based implementations
3. **Memory Management**: Efficient memory usage for large datasets
4. **Parallel Processing**: Support for multiprocessing where applicable

### Numerical Stability

1. **Parameter Validation**: All parameters are validated before use
2. **Error Handling**: Robust error handling for edge cases
3. **Precision Control**: Configurable precision for numerical operations
4. **Fallback Methods**: Multiple generation methods for robustness

### Quality Assurance

1. **Unit Tests**: Comprehensive test coverage
2. **Statistical Validation**: Verification of theoretical properties
3. **Performance Benchmarks**: Regular performance testing
4. **Documentation**: Complete API documentation

### Usage Patterns

```python
# Standard usage
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion

fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
data = fbm.generate(1000, seed=42)

# Get theoretical properties
properties = fbm.get_theoretical_properties()
print(f"Hurst parameter: {properties['hurst_parameter']}")
print(f"Long-range dependence: {properties['long_range_dependence']}")

# Batch generation
data_batch = [fbm.generate(1000, seed=i) for i in range(10)]

# Parameter exploration
for H in [0.3, 0.5, 0.7, 0.9]:
    fbm = FractionalBrownianMotion(H=H, sigma=1.0)
    data = fbm.generate(1000, seed=42)
    # Process data...
```

---

**For more information, see the [Complete API Reference](../api_reference/COMPLETE_API_REFERENCE.md) or the [project documentation](../../README.md).**
