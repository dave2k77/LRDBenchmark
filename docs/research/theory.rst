Theoretical Foundations
======================

This document provides the theoretical foundations for the statistical tests, validation techniques, and mathematical principles underlying LRDBench.

Long-Range Dependence (LRD)
---------------------------

Definition and Properties
~~~~~~~~~~~~~~~~~~~~~~~~

Long-range dependence (LRD), also known as long memory or persistence, is a statistical property of time series where correlations between observations decay slowly with increasing time lag.

For a stationary time series :math:`\{X_t\}` with autocorrelation function :math:`\rho(k)`, LRD is characterized by:

.. math::

   \rho(k) \sim c|k|^{2H-2} \text{ as } k \to \infty

where:
- :math:`H` is the Hurst parameter (:math:`0.5 < H < 1`)
- :math:`c` is a positive constant
- :math:`\sim` denotes asymptotic equivalence

The Hurst parameter :math:`H` quantifies the degree of long-range dependence:
- :math:`H = 0.5`: No long-range dependence (white noise)
- :math:`0.5 < H < 1`: Positive long-range dependence (persistence)
- :math:`0 < H < 0.5`: Negative long-range dependence (anti-persistence)

Power Spectral Density
~~~~~~~~~~~~~~~~~~~~~

For LRD processes, the power spectral density :math:`S(f)` exhibits a power-law behavior at low frequencies:

.. math::

   S(f) \sim c|f|^{1-2H} \text{ as } f \to 0

This relationship forms the basis for spectral-based estimators like the Geweke-Porter-Hudak (GPH) estimator.

Fractional Brownian Motion (FBM)
-------------------------------

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~

Fractional Brownian Motion :math:`B_H(t)` is a continuous-time Gaussian process with stationary increments, defined by:

.. math::

   B_H(t) = \frac{1}{\Gamma(H + \frac{1}{2})} \int_{-\infty}^t (t-s)^{H-\frac{1}{2}} dB(s)

where:
- :math:`H` is the Hurst parameter (:math:`0 < H < 1`)
- :math:`\Gamma(\cdot)` is the gamma function
- :math:`B(s)` is standard Brownian motion

Properties
~~~~~~~~~~

1. **Self-similarity**: :math:`B_H(at) \stackrel{d}{=} a^H B_H(t)` for all :math:`a > 0`
2. **Stationary increments**: :math:`B_H(t+s) - B_H(s) \stackrel{d}{=} B_H(t) - B_H(0)`
3. **Variance scaling**: :math:`\text{Var}[B_H(t)] = t^{2H}`
4. **Covariance function**: :math:`\text{Cov}[B_H(s), B_H(t)] = \frac{1}{2}(|s|^{2H} + |t|^{2H} - |s-t|^{2H})`

Generation Methods
~~~~~~~~~~~~~~~~~

**Method 1: Cholesky Decomposition**
The covariance matrix :math:`\Sigma` is constructed and decomposed as :math:`\Sigma = LL^T`. The FBM is then generated as :math:`B_H = LZ` where :math:`Z` is a vector of independent standard normal random variables.

**Method 2: Circulant Embedding**
The covariance function is embedded in a circulant matrix, which can be diagonalized using the FFT, enabling efficient generation.

**Method 3: Davies-Harte Method**
Uses the spectral representation of FBM to generate samples via FFT.

Fractional Gaussian Noise (FGN)
------------------------------

Definition
~~~~~~~~~~

Fractional Gaussian Noise is the increment process of FBM:

.. math::

   X_t = B_H(t+1) - B_H(t)

Properties
~~~~~~~~~~

1. **Stationarity**: FGN is a stationary process
2. **Autocorrelation**: :math:`\rho(k) = \frac{1}{2}(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})`
3. **Long-range dependence**: For :math:`H > 0.5`, :math:`\rho(k) \sim H(2H-1)k^{2H-2}` as :math:`k \to \infty`

ARFIMA Models
-------------

Definition
~~~~~~~~~~

An ARFIMA(p,d,q) process :math:`\{X_t\}` satisfies:

.. math::

   \Phi(B)(1-B)^d X_t = \Theta(B)\epsilon_t

where:
- :math:`\Phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p` (AR polynomial)
- :math:`\Theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q` (MA polynomial)
- :math:`(1-B)^d` is the fractional differencing operator
- :math:`\epsilon_t` is white noise

Fractional Differencing
~~~~~~~~~~~~~~~~~~~~~~

The fractional differencing operator :math:`(1-B)^d` is defined by the binomial expansion:

.. math::

   (1-B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-B)^k

where :math:`\binom{d}{k} = \frac{d(d-1)\cdots(d-k+1)}{k!}`

For :math:`|d| < 0.5`, the process is stationary and invertible. The relationship between :math:`d` and the Hurst parameter is:

.. math::

   H = d + 0.5

Multifractal Random Walk (MRW)
-----------------------------

Definition
~~~~~~~~~~

The Multifractal Random Walk is defined as:

.. math::

   X(t) = \int_0^t e^{\omega(s)} dB(s)

where:
- :math:`B(s)` is standard Brownian motion
- :math:`\omega(s)` is a stationary Gaussian process with covariance:

.. math::

   \text{Cov}[\omega(s), \omega(t)] = \lambda^2 \log_+ \frac{T}{|s-t| + \ell}

Parameters:
- :math:`\lambda^2`: Intermittency parameter
- :math:`T`: Integral time scale
- :math:`\ell`: Small-scale cutoff

Properties
~~~~~~~~~~

1. **Multifractality**: The process exhibits different scaling exponents at different time scales
2. **Log-normal multipliers**: The process can be constructed using log-normal multipliers
3. **Scaling**: :math:`\langle |X(t+\tau) - X(t)|^q \rangle \sim \tau^{\zeta(q)}` where :math:`\zeta(q)` is the multifractal spectrum

Statistical Estimators
=====================

Temporal Domain Estimators
-------------------------

Detrended Fluctuation Analysis (DFA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. **Integration**: :math:`y(i) = \sum_{k=1}^i (x_k - \bar{x})`
2. **Segmentation**: Divide into :math:`N_s = N/s` non-overlapping segments
3. **Detrending**: Fit polynomial :math:`y_n(i)` to each segment
4. **Fluctuation**: Calculate :math:`F^2(s) = \frac{1}{s} \sum_{i=1}^s [y(i) - y_n(i)]^2`
5. **Scaling**: :math:`F(s) \sim s^H`

**Theoretical Foundation**:
DFA measures the scaling of fluctuations around local trends, making it robust to non-stationarities.

**Mathematical Formulation**:
For a time series of length :math:`N`, the DFA fluctuation function is:

.. math::

   F(s) = \sqrt{\frac{1}{N_s} \sum_{v=1}^{N_s} F^2(v,s)}

where :math:`F^2(v,s)` is the mean squared fluctuation in segment :math:`v` of size :math:`s`.

R/S Analysis
~~~~~~~~~~~

**Algorithm**:

1. **Segmentation**: Divide data into segments of length :math:`k`
2. **Rescaled Range**: For each segment, calculate:
   - :math:`R = \max_{1 \leq i \leq k} S_i - \min_{1 \leq i \leq k} S_i` (range)
   - :math:`S = \sqrt{\frac{1}{k} \sum_{i=1}^k (x_i - \bar{x})^2}` (standard deviation)
3. **Scaling**: :math:`R/S \sim k^H`

**Theoretical Foundation**:
R/S analysis measures the scaling of the range of partial sums, normalized by the standard deviation.

**Mathematical Formulation**:
For a segment of length :math:`k`, the rescaled range is:

.. math::

   R/S = \frac{\max_{1 \leq i \leq k} \sum_{j=1}^i (x_j - \bar{x}) - \min_{1 \leq i \leq k} \sum_{j=1}^i (x_j - \bar{x})}{\sqrt{\frac{1}{k} \sum_{i=1}^k (x_i - \bar{x})^2}}

Higuchi Method
~~~~~~~~~~~~~

**Algorithm**:

1. **Subseries Construction**: For each :math:`k`, construct :math:`k` subseries
2. **Length Calculation**: Calculate the length :math:`L_m(k)` of each subseries
3. **Average Length**: :math:`L(k) = \frac{1}{k} \sum_{m=1}^k L_m(k)`
4. **Scaling**: :math:`L(k) \sim k^{-D}` where :math:`D = 2 - H`

**Theoretical Foundation**:
The Higuchi method estimates the fractal dimension by measuring how the length of the time series changes with different sampling intervals.

**Mathematical Formulation**:
For a time series :math:`\{x_i\}` and lag :math:`k`, the length is:

.. math::

   L_m(k) = \frac{1}{k} \left[ \frac{N-1}{k^2} \sum_{i=1}^{[(N-m)/k]} |x_{m+ik} - x_{m+(i-1)k}| \right]

Spectral Domain Estimators
-------------------------

Geweke-Porter-Hudak (GPH) Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. **Periodogram**: Calculate :math:`I(f_j) = \frac{1}{2\pi N} |\sum_{t=1}^N x_t e^{-i2\pi f_j t}|^2`
2. **Log-Regression**: Fit :math:`\log I(f_j) = c - d \log(4\sin^2(\pi f_j)) + \epsilon_j`
3. **Estimation**: :math:`H = d + 0.5`

**Theoretical Foundation**:
The GPH estimator is based on the spectral representation of ARFIMA processes, where the log-periodogram follows a linear relationship with the log-frequency.

**Mathematical Formulation**:
For frequencies :math:`f_j = j/N`, the regression model is:

.. math::

   \log I(f_j) = c - d \log(4\sin^2(\pi f_j)) + \epsilon_j

where :math:`d` is the fractional differencing parameter and :math:`H = d + 0.5`.

Whittle Estimator
~~~~~~~~~~~~~~~~~

**Algorithm**:

1. **Spectral Density**: Assume parametric form :math:`S(f; \theta)`
2. **Whittle Likelihood**: :math:`L(\theta) = \sum_{j=1}^{N/2} \left[ \log S(f_j; \theta) + \frac{I(f_j)}{S(f_j; \theta)} \right]`
3. **Optimization**: Maximize :math:`L(\theta)` to estimate parameters

**Theoretical Foundation**:
The Whittle estimator maximizes an approximation to the likelihood function in the frequency domain, making it asymptotically efficient.

**Mathematical Formulation**:
The Whittle likelihood function is:

.. math::

   L(\theta) = \sum_{j=1}^{N/2} \left[ \log S(f_j; \theta) + \frac{I(f_j)}{S(f_j; \theta)} \right]

where :math:`S(f; \theta)` is the theoretical spectral density and :math:`I(f_j)` is the periodogram.

Wavelet Domain Estimators
------------------------

Wavelet Variance
~~~~~~~~~~~~~~~

**Algorithm**:

1. **Wavelet Decomposition**: Apply discrete wavelet transform
2. **Variance Calculation**: :math:`\sigma^2_j = \frac{1}{n_j} \sum_{k=1}^{n_j} d_{j,k}^2`
3. **Scaling**: :math:`\sigma^2_j \sim 2^{j(2H-1)}`

**Theoretical Foundation**:
Wavelet variance measures the energy at different scales, providing a robust estimate of the scaling exponent.

**Mathematical Formulation**:
For wavelet coefficients :math:`d_{j,k}` at scale :math:`j`, the variance is:

.. math::

   \sigma^2_j = \frac{1}{n_j} \sum_{k=1}^{n_j} d_{j,k}^2

where :math:`n_j` is the number of coefficients at scale :math:`j`.

Continuous Wavelet Transform (CWT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. **CWT Calculation**: :math:`W_x(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt`
2. **Wavelet Spectrum**: :math:`S_x(a) = \int_{-\infty}^{\infty} |W_x(a,b)|^2 db`
3. **Scaling**: :math:`S_x(a) \sim a^{2H+1}`

**Theoretical Foundation**:
CWT provides a time-scale representation that preserves both temporal and frequency information.

**Mathematical Formulation**:
The continuous wavelet transform is:

.. math::

   W_x(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt

where :math:`\psi(t)` is the mother wavelet and :math:`a, b` are scale and translation parameters.

Multifractal Estimators
----------------------

Multifractal Detrended Fluctuation Analysis (MFDFA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Algorithm**:

1. **Profile**: :math:`Y(i) = \sum_{k=1}^i (x_k - \bar{x})`
2. **Segmentation**: Divide into :math:`N_s = N/s` segments
3. **Detrending**: Fit polynomial to each segment
4. **Fluctuation**: :math:`F_q(s) = \left[ \frac{1}{N_s} \sum_{v=1}^{N_s} F^2(v,s)^{q/2} \right]^{1/q}`
5. **Scaling**: :math:`F_q(s) \sim s^{h(q)}`

**Theoretical Foundation**:
MFDFA extends DFA to capture multifractal scaling by considering different moments of the fluctuation function.

**Mathematical Formulation**:
The qth order fluctuation function is:

.. math::

   F_q(s) = \left[ \frac{1}{N_s} \sum_{v=1}^{N_s} F^2(v,s)^{q/2} \right]^{1/q}

The multifractal spectrum :math:`f(\alpha)` is obtained via Legendre transform:

.. math::

   \alpha = h(q) + qh'(q), \quad f(\alpha) = q[\alpha - h(q)] + 1

Validation Techniques
====================

Monte Carlo Simulations
----------------------

**Purpose**: Validate estimator performance on synthetic data with known parameters.

**Methodology**:

1. **Data Generation**: Generate :math:`N` realizations of synthetic data with known :math:`H`
2. **Estimation**: Apply estimators to each realization
3. **Statistics**: Calculate bias, variance, and mean squared error
4. **Confidence Intervals**: Construct empirical confidence intervals

**Mathematical Formulation**:
For estimator :math:`\hat{H}` and true value :math:`H_0`:

- **Bias**: :math:`\text{Bias} = \mathbb{E}[\hat{H}] - H_0`
- **Variance**: :math:`\text{Var}[\hat{H}] = \mathbb{E}[(\hat{H} - \mathbb{E}[\hat{H}])^2]`
- **MSE**: :math:`\text{MSE} = \mathbb{E}[(\hat{H} - H_0)^2] = \text{Bias}^2 + \text{Var}[\hat{H}]`

Bootstrap Methods
----------------

**Purpose**: Estimate confidence intervals and standard errors for estimators.

**Methodology**:

1. **Resampling**: Generate bootstrap samples by resampling with replacement
2. **Estimation**: Apply estimator to each bootstrap sample
3. **Statistics**: Calculate empirical distribution of estimates
4. **Confidence Intervals**: Use percentiles or bias-corrected methods

**Mathematical Formulation**:
For bootstrap samples :math:`\{\hat{H}_1^*, \ldots, \hat{H}_B^*\}`:

- **Bootstrap Mean**: :math:`\bar{H}^* = \frac{1}{B} \sum_{b=1}^B \hat{H}_b^*`
- **Bootstrap Variance**: :math:`s^2 = \frac{1}{B-1} \sum_{b=1}^B (\hat{H}_b^* - \bar{H}^*)^2`
- **Confidence Interval**: :math:`[\hat{H}_{\alpha/2}^*, \hat{H}_{1-\alpha/2}^*]`

Cross-Validation
---------------

**Purpose**: Assess estimator performance and prevent overfitting.

**Methodology**:

1. **Data Splitting**: Divide data into training and validation sets
2. **Parameter Tuning**: Optimize parameters on training set
3. **Validation**: Evaluate performance on validation set
4. **Cross-Validation**: Repeat with different splits

**Mathematical Formulation**:
For k-fold cross-validation with estimator :math:`f` and loss function :math:`L`:

.. math::

   \text{CV} = \frac{1}{k} \sum_{i=1}^k L(y_i, f^{-i}(x_i))

where :math:`f^{-i}` is the estimator trained on all folds except fold :math:`i`.

Robustness Analysis
------------------

**Purpose**: Assess estimator performance under various data conditions.

**Methodology**:

1. **Contamination**: Add noise, outliers, or trends to data
2. **Estimation**: Apply estimators to contaminated data
3. **Comparison**: Compare results with uncontaminated estimates
4. **Robustness Metrics**: Calculate breakdown points and influence functions

**Mathematical Formulation**:
For contamination level :math:`\epsilon` and contamination distribution :math:`G`:

.. math::

   F_\epsilon = (1-\epsilon)F + \epsilon G

where :math:`F` is the original distribution and :math:`F_\epsilon` is the contaminated distribution.

Performance Metrics
==================

Accuracy Metrics
---------------

**Mean Absolute Error (MAE)**:
.. math::

   \text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{H}_i - H_i|

**Root Mean Square Error (RMSE)**:
.. math::

   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{H}_i - H_i)^2}

**Mean Absolute Percentage Error (MAPE)**:
.. math::

   \text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{\hat{H}_i - H_i}{H_i}\right|

Precision Metrics
----------------

**Standard Error**:
.. math::

   \text{SE} = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (\hat{H}_i - \bar{H})^2}

**Coefficient of Variation**:
.. math::

   \text{CV} = \frac{\text{SE}}{\bar{H}} \times 100\%

**Confidence Interval Width**:
.. math::

   \text{CI Width} = \hat{H}_{1-\alpha/2} - \hat{H}_{\alpha/2}

Efficiency Metrics
-----------------

**Computational Complexity**: Big-O notation for time and space complexity
**Convergence Rate**: Rate at which estimator approaches true value
**Asymptotic Efficiency**: Ratio of estimator variance to Cramér-Rao lower bound

Statistical Tests
================

Hypothesis Testing
-----------------

**Null Hypothesis**: :math:`H_0: H = H_0`
**Alternative Hypothesis**: :math:`H_1: H \neq H_0`

**Test Statistic**:
.. math::

   T = \frac{\hat{H} - H_0}{\text{SE}(\hat{H})}

**Decision Rule**: Reject :math:`H_0` if :math:`|T| > z_{\alpha/2}`

Goodness-of-Fit Tests
--------------------

**Kolmogorov-Smirnov Test**:
Tests whether empirical distribution matches theoretical distribution.

**Anderson-Darling Test**:
Weighted version of KS test, more sensitive to tails.

**Chi-Square Test**:
Tests fit of observed frequencies to expected frequencies.

Model Selection
--------------

**Akaike Information Criterion (AIC)**:
.. math::

   \text{AIC} = 2k - 2\ln(L)

**Bayesian Information Criterion (BIC)**:
.. math::

   \text{BIC} = \ln(n)k - 2\ln(L)

where :math:`k` is the number of parameters and :math:`L` is the likelihood.

Practical Examples
=================

Monte Carlo Simulation Example
-----------------------------

.. code-block:: python

   import numpy as np
   from lrdbenchmark import FBMModel, ComprehensiveBenchmark
   import matplotlib.pyplot as plt

   def monte_carlo_validation_example():
       """Demonstrate Monte Carlo validation for estimator performance."""
       
       # Define parameter space
       H_values = np.linspace(0.3, 0.9, 13)  # 13 different H values
       sample_sizes = [500, 1000, 2000]
       n_realizations = 50
       
       # Initialize results storage
       results = {
           'dfa': {'bias': [], 'variance': [], 'mse': []},
           'gph': {'bias': [], 'variance': [], 'mse': []},
           'rs': {'bias': [], 'variance': [], 'mse': []}
       }
       
       print("Running Monte Carlo validation...")
       
       for H in H_values:
           print(f"Testing H = {H:.2f}")
           
           for n in sample_sizes:
               estimates = {'dfa': [], 'gph': [], 'rs': []}
               
               for i in range(n_realizations):
                   # Generate synthetic data
                   model = FBMModel(H=H, sigma=1.0)
                   data = model.generate(n, seed=i)
                   
                   # Apply estimators
                   benchmark = ComprehensiveBenchmark()
                   result = benchmark.run_classical_benchmark(
                       data_length=n,
                       estimators=['dfa', 'gph', 'rs']
                   )
                   
                   # Collect estimates
                   for estimator_name in estimates.keys():
                       if estimator_name in result.estimators:
                           estimates[estimator_name].append(
                               result.estimators[estimator_name].mean_estimate
                           )
               
               # Calculate statistics
               for estimator_name in estimates.keys():
                   if estimates[estimator_name]:
                       est_array = np.array(estimates[estimator_name])
                       bias = np.mean(est_array) - H
                       variance = np.var(est_array)
                       mse = bias**2 + variance
                       
                       results[estimator_name]['bias'].append(bias)
                       results[estimator_name]['variance'].append(variance)
                       results[estimator_name]['mse'].append(mse)
       
       # Plot results
       fig, axes = plt.subplots(1, 3, figsize=(15, 5))
       
       for i, metric in enumerate(['bias', 'variance', 'mse']):
           for estimator_name in results.keys():
               if results[estimator_name][metric]:
                   axes[i].plot(H_values, results[estimator_name][metric], 
                               label=estimator_name.upper(), marker='o')
           
           axes[i].set_xlabel('True Hurst Parameter (H)')
           axes[i].set_ylabel(metric.capitalize())
           axes[i].set_title(f'{metric.capitalize()} vs True H')
           axes[i].legend()
           axes[i].grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return results

   # Run the example
   if __name__ == "__main__":
       results = monte_carlo_validation_example()
       print("Monte Carlo validation completed!")

Power Spectral Density Analysis
------------------------------

.. code-block:: python

   import numpy as np
   from scipy import signal
   from lrdbenchmark import FBMModel, FGNModel
   import matplotlib.pyplot as plt

   def power_spectral_density_example():
       """Demonstrate power spectral density analysis for LRD processes."""
       
       # Generate data with different Hurst parameters
       models = {
           'FBM (H=0.3)': FBMModel(H=0.3, sigma=1.0),
           'FBM (H=0.5)': FBMModel(H=0.5, sigma=1.0),
           'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
           'FBM (H=0.9)': FBMModel(H=0.9, sigma=1.0)
       }
       
       plt.figure(figsize=(12, 8))
       
       for model_name, model in models.items():
           # Generate data
           data = model.generate(2000, seed=42)
           
           # Compute power spectral density
           freqs, psd = signal.welch(data, fs=1.0, nperseg=256)
           
           # Plot PSD
           plt.loglog(freqs, psd, label=model_name, linewidth=2)
       
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('Power Spectral Density')
       plt.title('Power Spectral Density of FBM Processes')
       plt.legend()
       plt.grid(True)
       plt.show()
       
       # Theoretical PSD for comparison
       plt.figure(figsize=(10, 6))
       freqs_theoretical = np.logspace(-3, 0, 100)
       
       for H in [0.3, 0.5, 0.7, 0.9]:
           # Theoretical PSD: S(f) ∝ f^(-2H+1)
           psd_theoretical = freqs_theoretical**(-2*H + 1)
           plt.loglog(freqs_theoretical, psd_theoretical, 
                     label=f'Theoretical (H={H})', linestyle='--')
       
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('Power Spectral Density')
       plt.title('Theoretical Power Spectral Density')
       plt.legend()
       plt.grid(True)
       plt.show()

   # Run the example
   if __name__ == "__main__":
       power_spectral_density_example()
       print("Power spectral density analysis completed!")

Autocorrelation Function Analysis
--------------------------------

.. code-block:: python

   import numpy as np
   from lrdbenchmark import FBMModel
   import matplotlib.pyplot as plt

   def autocorrelation_analysis_example():
       """Demonstrate autocorrelation function analysis for LRD processes."""
       
       # Generate FBM data with different H values
       H_values = [0.3, 0.5, 0.7, 0.9]
       max_lag = 100
       
       plt.figure(figsize=(12, 8))
       
       for H in H_values:
           # Generate data
           model = FBMModel(H=H, sigma=1.0)
           data = model.generate(2000, seed=42)
           
           # Compute autocorrelation function
           acf = np.correlate(data, data, mode='full')
           acf = acf[len(data)-1:len(data)-1+max_lag] / acf[len(data)-1]
           
           # Plot ACF
           lags = np.arange(max_lag)
           plt.plot(lags, acf, label=f'FBM (H={H})', linewidth=2)
       
       plt.xlabel('Lag')
       plt.ylabel('Autocorrelation')
       plt.title('Autocorrelation Function of FBM Processes')
       plt.legend()
       plt.grid(True)
       plt.show()
       
       # Theoretical ACF comparison
       plt.figure(figsize=(10, 6))
       lags_theoretical = np.arange(1, max_lag+1)
       
       for H in H_values:
           # Theoretical ACF: ρ(k) ∝ k^(2H-2)
           acf_theoretical = lags_theoretical**(2*H - 2)
           plt.loglog(lags_theoretical, acf_theoretical, 
                     label=f'Theoretical (H={H})', linestyle='--')
       
       plt.xlabel('Lag')
       plt.ylabel('Autocorrelation')
       plt.title('Theoretical Autocorrelation Function')
       plt.legend()
       plt.grid(True)
       plt.show()

   # Run the example
   if __name__ == "__main__":
       autocorrelation_analysis_example()
       print("Autocorrelation analysis completed!")

Theoretical References
=====================

1. Beran, J. (1994). Statistics for Long-Memory Processes. Chapman & Hall.
2. Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. SIAM Review, 10(4), 422-437.
3. Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. Physical Review E, 49(2), 1685.
4. Geweke, J., & Porter-Hudak, S. (1983). The estimation and application of long memory time series models. Journal of Time Series Analysis, 4(4), 221-238.
5. Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. Physica A, 316(1-4), 87-114.
6. Abry, P., & Veitch, D. (1998). Wavelet analysis of long-range-dependent traffic. IEEE Transactions on Information Theory, 44(1), 2-15.
