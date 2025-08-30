Validation Techniques and Statistical Tests
===========================================

This document provides comprehensive coverage of the validation techniques, statistical tests, and quality assurance methods used in LRDBench.

Monte Carlo Validation
=====================

Overview
---------

Monte Carlo validation is the primary method for assessing estimator performance on synthetic data with known parameters. This approach allows for controlled evaluation of estimator bias, variance, and overall accuracy.

Methodology
----------

**Data Generation Process**:

1. **Parameter Space Definition**: Define ranges for Hurst parameters (typically H ∈ [0.3, 0.9])
2. **Model Selection**: Choose data models (FBM, FGN, ARFIMA, MRW)
3. **Sample Size Selection**: Use multiple sample sizes (e.g., 500, 1000, 2000, 5000)
4. **Realization Count**: Generate N realizations for each parameter combination

**Mathematical Framework**:

For estimator :math:`\hat{H}` and true value :math:`H_0`:

.. math::

   \text{Bias} = \mathbb{E}[\hat{H}] - H_0 = \frac{1}{N} \sum_{i=1}^N \hat{H}_i - H_0

.. math::

   \text{Variance} = \mathbb{E}[(\hat{H} - \mathbb{E}[\hat{H}])^2] = \frac{1}{N-1} \sum_{i=1}^N (\hat{H}_i - \bar{H})^2

.. math::

   \text{MSE} = \mathbb{E}[(\hat{H} - H_0)^2] = \text{Bias}^2 + \text{Variance}

**Implementation**:

.. code-block:: python

   import numpy as np
   from lrdbenchmark import FBMModel, ComprehensiveBenchmark
   
   def monte_carlo_validation(H_values, sample_sizes, n_realizations=100):
       """Perform Monte Carlo validation."""
       results = {}
       
       for H in H_values:
           for n in sample_sizes:
               estimates = []
               
               for i in range(n_realizations):
                   # Generate synthetic data
                   model = FBMModel(H=H, sigma=1.0)
                   data = model.generate(n, seed=i)
                   
                   # Apply estimator
                   benchmark = ComprehensiveBenchmark()
                   result = benchmark.run_classical_benchmark(
                       data_length=n,
                       estimators=['dfa', 'gph', 'rs']
                   )
                   
                   # Collect estimates
                   for estimator_name, estimator_result in result.estimators.items():
                       if estimator_name not in results:
                           results[estimator_name] = {}
                       if (H, n) not in results[estimator_name]:
                           results[estimator_name][(H, n)] = []
                       
                       results[estimator_name][(H, n)].append(
                           estimator_result.mean_estimate
                       )
       
       return results

Bootstrap Methods
================

Overview
---------

Bootstrap methods provide non-parametric approaches to estimate confidence intervals, standard errors, and bias correction for estimators.

Types of Bootstrap
------------------

**1. Non-Parametric Bootstrap**:

Resample with replacement from the original data:

.. math::

   X^*_1, X^*_2, \ldots, X^*_n \sim \text{iid from } \{X_1, X_2, \ldots, X_n\}

**2. Parametric Bootstrap**:

Generate bootstrap samples from a fitted parametric model:

.. math::

   X^*_i \sim F_{\hat{\theta}}(x)

**3. Moving Block Bootstrap**:

Preserve temporal dependence by resampling blocks:

.. math::

   B_i = (X_i, X_{i+1}, \ldots, X_{i+l-1})

**Implementation**:

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   def bootstrap_confidence_interval(data, estimator_func, n_bootstrap=1000, 
                                   confidence_level=0.95):
       """Calculate bootstrap confidence interval."""
       bootstrap_estimates = []
       
       for _ in range(n_bootstrap):
           # Resample with replacement
           bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
           
           # Apply estimator
           estimate = estimator_func(bootstrap_sample)
           bootstrap_estimates.append(estimate)
       
       # Calculate confidence interval
       alpha = 1 - confidence_level
       lower_percentile = (alpha / 2) * 100
       upper_percentile = (1 - alpha / 2) * 100
       
       ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
       ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
       
       return ci_lower, ci_upper, bootstrap_estimates

Cross-Validation
================

Overview
--------

Cross-validation is essential for machine learning estimators to prevent overfitting and assess generalization performance.

K-Fold Cross-Validation
-----------------------

**Algorithm**:

1. **Data Partitioning**: Divide data into K folds
2. **Training/Validation**: For each fold k:
   - Train on K-1 folds
   - Validate on fold k
3. **Performance Aggregation**: Average performance across all folds

**Mathematical Formulation**:

.. math::

   \text{CV} = \frac{1}{K} \sum_{k=1}^K L(y_k, f^{-k}(x_k))

where :math:`f^{-k}` is the estimator trained on all folds except fold k.

**Implementation**:

.. code-block:: python

   from sklearn.model_selection import KFold
   from sklearn.metrics import mean_squared_error
   
   def cross_validate_estimator(data, labels, estimator_class, k_folds=5):
       """Perform k-fold cross-validation."""
       kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
       cv_scores = []
       
       for train_idx, val_idx in kf.split(data):
           # Split data
           X_train, X_val = data[train_idx], data[val_idx]
           y_train, y_val = labels[train_idx], labels[val_idx]
           
           # Train estimator
           estimator = estimator_class()
           estimator.fit(X_train, y_train)
           
           # Predict and evaluate
           y_pred = estimator.estimate(X_val)
           mse = mean_squared_error(y_val, y_pred)
           cv_scores.append(mse)
       
       return np.mean(cv_scores), np.std(cv_scores)

Time Series Cross-Validation
----------------------------

For time series data, standard k-fold CV can lead to data leakage. Time series CV uses expanding windows:

.. code-block:: python

   def time_series_cv(data, labels, estimator_class, min_train_size=100):
       """Time series cross-validation with expanding windows."""
       cv_scores = []
       
       for i in range(min_train_size, len(data)):
           # Training set: all data up to index i
           X_train = data[:i]
           y_train = labels[:i]
           
           # Validation set: next observation
           X_val = data[i:i+1]
           y_val = labels[i:i+1]
           
           # Train and predict
           estimator = estimator_class()
           estimator.fit(X_train, y_train)
           y_pred = estimator.estimate(X_val)
           
           # Calculate error
           mse = mean_squared_error(y_val, y_pred)
           cv_scores.append(mse)
       
       return np.mean(cv_scores), np.std(cv_scores)

Robustness Analysis
===================

Overview
--------

Robustness analysis assesses estimator performance under various data conditions, including contamination, noise, and model misspecification.

Contamination Models
--------------------

**1. Additive Noise**:

.. math::

   X_t = X_t^{(0)} + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)

**2. Outlier Contamination**:

.. math::

   X_t = \begin{cases}
   X_t^{(0)} & \text{with probability } 1-\epsilon \\
   X_t^{(0)} + \delta & \text{with probability } \epsilon
   \end{cases}

**3. Trend Contamination**:

.. math::

   X_t = X_t^{(0)} + \alpha t + \beta t^2

**Implementation**:

.. code-block:: python

   def robustness_analysis(data, estimator_func, contamination_levels):
       """Analyze estimator robustness to contamination."""
       results = {}
       
       for epsilon in contamination_levels:
           contaminated_data = data.copy()
           
           # Add contamination
           n_contaminated = int(epsilon * len(data))
           contamination_indices = np.random.choice(
               len(data), size=n_contaminated, replace=False
           )
           
           # Add outliers
           contaminated_data[contamination_indices] += np.random.normal(
               0, 5, size=n_contaminated
           )
           
           # Apply estimator
           estimate = estimator_func(contaminated_data)
           results[epsilon] = estimate
       
       return results

Breakdown Point Analysis
------------------------

The breakdown point is the smallest fraction of contaminated data that can cause the estimator to produce arbitrarily bad results.

**Implementation**:

.. code-block:: python

   def breakdown_point_analysis(data, estimator_func, max_contamination=0.5):
       """Estimate breakdown point of an estimator."""
       original_estimate = estimator_func(data)
       breakdown_point = None
       
       for epsilon in np.arange(0, max_contamination, 0.01):
           # Create contaminated data
           n_contaminated = int(epsilon * len(data))
           contaminated_data = data.copy()
           
           # Add extreme outliers
           contamination_indices = np.random.choice(
               len(data), size=n_contaminated, replace=False
           )
           contaminated_data[contamination_indices] = 1e6
           
           # Check if estimator breaks down
           try:
               contaminated_estimate = estimator_func(contaminated_data)
               
               # Check if estimate is reasonable
               if abs(contaminated_estimate - original_estimate) > 0.5:
                   breakdown_point = epsilon
                   break
           except:
               breakdown_point = epsilon
               break
       
       return breakdown_point

Statistical Tests
=================

Hypothesis Testing
------------------

**1. Test for Long-Range Dependence**:

Null hypothesis: :math:`H_0: H = 0.5` (no LRD)
Alternative hypothesis: :math:`H_1: H \neq 0.5` (LRD present)

Test statistic:
.. math::

   T = \frac{\hat{H} - 0.5}{\text{SE}(\hat{H})}

**2. Test for Parameter Equality**:

Null hypothesis: :math:`H_0: H_1 = H_2`
Alternative hypothesis: :math:`H_1: H_1 \neq H_2`

Test statistic:
.. math::

   T = \frac{\hat{H}_1 - \hat{H}_2}{\sqrt{\text{SE}(\hat{H}_1)^2 + \text{SE}(\hat{H}_2)^2}}

**Implementation**:

.. code-block:: python

   from scipy import stats
   
   def test_lrd_presence(data, estimator_func, alpha=0.05):
       """Test for presence of long-range dependence."""
       # Estimate H
       H_estimate = estimator_func(data)
       
       # Bootstrap standard error
       bootstrap_estimates = []
       for _ in range(1000):
           bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
           bootstrap_estimates.append(estimator_func(bootstrap_sample))
       
       se_H = np.std(bootstrap_estimates)
       
       # Test statistic
       test_statistic = (H_estimate - 0.5) / se_H
       
       # Critical value
       critical_value = stats.norm.ppf(1 - alpha/2)
       
       # Decision
       reject_null = abs(test_statistic) > critical_value
       p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
       
       return {
           'test_statistic': test_statistic,
           'p_value': p_value,
           'reject_null': reject_null,
           'H_estimate': H_estimate,
           'standard_error': se_H
       }

Goodness-of-Fit Tests
---------------------

**1. Kolmogorov-Smirnov Test**:

Tests whether empirical distribution matches theoretical distribution.

.. math::

   D_n = \sup_x |F_n(x) - F(x)|

**2. Anderson-Darling Test**:

Weighted version of KS test, more sensitive to tails.

.. math::

   A^2 = n \int_{-\infty}^{\infty} \frac{(F_n(x) - F(x))^2}{F(x)(1-F(x))} dF(x)

**3. Chi-Square Test**:

Tests fit of observed frequencies to expected frequencies.

.. math::

   \chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}

**Implementation**:

.. code-block:: python

   def goodness_of_fit_tests(data, theoretical_distribution):
       """Perform goodness-of-fit tests."""
       results = {}
       
       # Kolmogorov-Smirnov test
       ks_statistic, ks_pvalue = stats.kstest(data, theoretical_distribution)
       results['ks_test'] = {
           'statistic': ks_statistic,
           'p_value': ks_pvalue
       }
       
       # Anderson-Darling test
       ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(
           data, dist=theoretical_distribution
       )
       results['anderson_darling'] = {
           'statistic': ad_statistic,
           'critical_values': ad_critical_values,
           'significance_levels': ad_significance_levels
       }
       
       # Chi-square test
       observed, bins = np.histogram(data, bins='auto')
       expected = theoretical_distribution.pdf(bins[:-1]) * len(data) * np.diff(bins)
       
       chi2_statistic, chi2_pvalue = stats.chisquare(observed, expected)
       results['chi_square'] = {
           'statistic': chi2_statistic,
           'p_value': chi2_pvalue
       }
       
       return results

Model Selection
===============

Information Criteria
--------------------

**1. Akaike Information Criterion (AIC)**:

.. math::

   \text{AIC} = 2k - 2\ln(L)

where k is the number of parameters and L is the likelihood.

**2. Bayesian Information Criterion (BIC)**:

.. math::

   \text{BIC} = \ln(n)k - 2\ln(L)

where n is the sample size.

**3. Corrected AIC (AICc)**:

.. math::

   \text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}

**Implementation**:

.. code-block:: python

   def model_selection_criteria(models, data):
       """Calculate model selection criteria."""
       results = {}
       
       for model_name, model in models.items():
           # Fit model and get likelihood
           model.fit(data)
           log_likelihood = model.log_likelihood(data)
           n_params = model.n_parameters
           n_samples = len(data)
           
           # Calculate criteria
           aic = 2 * n_params - 2 * log_likelihood
           bic = np.log(n_samples) * n_params - 2 * log_likelihood
           aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
           
           results[model_name] = {
               'AIC': aic,
               'BIC': bic,
               'AICc': aicc,
               'log_likelihood': log_likelihood,
               'n_parameters': n_params
           }
       
       return results

Performance Metrics
===================

Accuracy Metrics
----------------

**1. Mean Absolute Error (MAE)**:

.. math::

   MAE = \frac{1}{n} \sum_{i=1}^n |\hat{H}_i - H_i|

**2. Root Mean Square Error (RMSE)**:

.. math::

   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{H}_i - H_i)^2}

**3. Mean Absolute Percentage Error (MAPE)**:

.. math::

   MAPE = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{\hat{H}_i - H_i}{H_i}\right|

**4. Symmetric Mean Absolute Percentage Error (SMAPE)**:

.. math::

   SMAPE = \frac{100\%}{n} \sum_{i=1}^n \frac{2|\hat{H}_i - H_i|}{|\hat{H}_i| + |H_i|}

Precision Metrics
-----------------

**1. Standard Error**:

.. math::

   SE = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (\hat{H}_i - \bar{H})^2}

**2. Coefficient of Variation**:

.. math::

   CV = \frac{SE}{\bar{H}} \times 100\%

**3. Confidence Interval Width**:

.. math::

   CI_{width} = \hat{H}_{1-\alpha/2} - \hat{H}_{\alpha/2}

**Implementation**:

.. code-block:: python

   def calculate_performance_metrics(true_values, estimated_values):
       """Calculate comprehensive performance metrics."""
       metrics = {}
       
       # Convert to numpy arrays
       true_vals = np.array(true_values)
       est_vals = np.array(estimated_values)
       
       # Accuracy metrics
       errors = est_vals - true_vals
       abs_errors = np.abs(errors)
       
       metrics['MAE'] = np.mean(abs_errors)
       metrics['RMSE'] = np.sqrt(np.mean(errors**2))
       metrics['MAPE'] = 100 * np.mean(np.abs(errors / true_vals))
       metrics['SMAPE'] = 100 * np.mean(2 * abs_errors / (np.abs(est_vals) + np.abs(true_vals)))
       
       # Precision metrics
       metrics['Standard_Error'] = np.std(est_vals)
       metrics['Coefficient_of_Variation'] = (np.std(est_vals) / np.mean(est_vals)) * 100
       
       # Bias
       metrics['Bias'] = np.mean(errors)
       
       # Correlation
       metrics['Correlation'] = np.corrcoef(true_vals, est_vals)[0, 1]
       
       return metrics

Efficiency Metrics
-----------------

**1. Computational Complexity**:
Big-O notation for time and space complexity of estimators.

**2. Convergence Rate**:
Rate at which estimator approaches true value with increasing sample size.

**3. Asymptotic Efficiency**:
Ratio of estimator variance to Cramér-Rao lower bound.

**Implementation**:

.. code-block:: python

   import time
   import numpy as np
   from lrdbenchmark import FBMModel, ComprehensiveBenchmark
   import matplotlib.pyplot as plt

   def efficiency_analysis_example():
       """Demonstrate efficiency analysis of estimators."""
       
       # Test different sample sizes
       sample_sizes = [500, 1000, 2000, 4000, 8000]
       n_runs = 10
       
       # Initialize results
       results = {
           'dfa': {'times': [], 'estimates': []},
           'gph': {'times': [], 'estimates': []},
           'rs': {'times': [], 'estimates': []}
       }
       
       print("Running efficiency analysis...")
       
       for n in sample_sizes:
           print(f"Testing sample size: {n}")
           
           for estimator_name in results.keys():
               times = []
               estimates = []
               
               for i in range(n_runs):
                   # Generate data
                   model = FBMModel(H=0.7, sigma=1.0)
                   data = model.generate(n, seed=i)
                   
                   # Time estimation
                   start_time = time.time()
                   
                   benchmark = ComprehensiveBenchmark()
                   result = benchmark.run_classical_benchmark(
                       data_length=n,
                       estimators=[estimator_name]
                   )
                   
                   end_time = time.time()
                   
                   if estimator_name in result.estimators:
                       times.append(end_time - start_time)
                       estimates.append(
                           result.estimators[estimator_name].mean_estimate
                       )
               
               if times:
                   results[estimator_name]['times'].append(np.mean(times))
                   results[estimator_name]['estimates'].append(np.mean(estimates))
       
       # Plot computational complexity
       plt.figure(figsize=(12, 5))
       
       plt.subplot(1, 2, 1)
       for estimator_name in results.keys():
           if results[estimator_name]['times']:
               plt.loglog(sample_sizes[:len(results[estimator_name]['times'])], 
                         results[estimator_name]['times'], 
                         label=estimator_name.upper(), marker='o')
       
       plt.xlabel('Sample Size')
       plt.ylabel('Execution Time (seconds)')
       plt.title('Computational Complexity')
       plt.legend()
       plt.grid(True)
       
       # Plot convergence
       plt.subplot(1, 2, 2)
       true_H = 0.7
       for estimator_name in results.keys():
           if results[estimator_name]['estimates']:
               errors = np.abs(np.array(results[estimator_name]['estimates']) - true_H)
               plt.loglog(sample_sizes[:len(errors)], errors, 
                         label=estimator_name.upper(), marker='o')
       
       plt.xlabel('Sample Size')
       plt.ylabel('Absolute Error')
       plt.title('Convergence Rate')
       plt.legend()
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return results

   # Run the example
   if __name__ == "__main__":
       results = efficiency_analysis_example()
       print("Efficiency analysis completed!")

Quality Assurance
=================

Data Quality Checks
------------------

**1. Stationarity Tests**:

.. code-block:: python

   from statsmodels.tsa.stattools import adfuller, kpss
   from scipy import stats
   import numpy as np
   from lrdbenchmark import FBMModel, FGNModel

   def data_quality_checks_example():
       """Demonstrate data quality checks for time series."""
       
       # Generate different types of data
       models = {
           'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
           'FGN (H=0.8)': FGNModel(H=0.8, sigma=1.0),
           'Non-stationary': lambda: np.cumsum(np.random.normal(0, 1, 1000))
       }
       
       print("=== DATA QUALITY CHECKS ===")
       
       for model_name, model in models.items():
           print(f"\n--- {model_name} ---")
           
           # Generate data
           if callable(model):
               data = model()
           else:
               data = model.generate(1000, seed=42)
           
           # ADF Test (Augmented Dickey-Fuller)
           adf_stat, adf_pvalue, adf_critical = adfuller(data)[:3]
           print(f"ADF Test:")
           print(f"  Statistic: {adf_stat:.4f}")
           print(f"  p-value: {adf_pvalue:.4f}")
           print(f"  Stationary: {'Yes' if adf_pvalue < 0.05 else 'No'}")
           
           # KPSS Test
           kpss_stat, kpss_pvalue, kpss_critical = kpss(data)[:3]
           print(f"KPSS Test:")
           print(f"  Statistic: {kpss_stat:.4f}")
           print(f"  p-value: {kpss_pvalue:.4f}")
           print(f"  Stationary: {'Yes' if kpss_pvalue > 0.05 else 'No'}")
           
           # Normality Test (Shapiro-Wilk)
           shapiro_stat, shapiro_pvalue = stats.shapiro(data)
           print(f"Shapiro-Wilk Test:")
           print(f"  Statistic: {shapiro_stat:.4f}")
           print(f"  p-value: {shapiro_pvalue:.4f}")
           print(f"  Normal: {'Yes' if shapiro_pvalue > 0.05 else 'No'}")
           
           # Basic statistics
           print(f"Basic Statistics:")
           print(f"  Mean: {np.mean(data):.4f}")
           print(f"  Std: {np.std(data):.4f}")
           print(f"  Skewness: {stats.skew(data):.4f}")
           print(f"  Kurtosis: {stats.kurtosis(data):.4f}")

   # Run the example
   if __name__ == "__main__":
       data_quality_checks_example()

Estimator Validation
-------------------

**1. Consistency Checks**:

.. code-block:: python

   import numpy as np
   from lrdbenchmark import FBMModel, ComprehensiveBenchmark
   import matplotlib.pyplot as plt

   def estimator_validation_example():
       """Demonstrate estimator validation procedures."""
       
       # Test consistency with increasing sample size
       sample_sizes = [500, 1000, 2000, 4000, 8000]
       true_H = 0.7
       n_runs = 20
       
       results = {
           'dfa': {'estimates': [], 'std': []},
           'gph': {'estimates': [], 'std': []},
           'rs': {'estimates': [], 'std': []}
       }
       
       print("Running estimator validation...")
       
       for n in sample_sizes:
           print(f"Sample size: {n}")
           
           for estimator_name in results.keys():
               estimates = []
               
               for i in range(n_runs):
                   # Generate data
                   model = FBMModel(H=true_H, sigma=1.0)
                   data = model.generate(n, seed=i)
                   
                   # Apply estimator
                   benchmark = ComprehensiveBenchmark()
                   result = benchmark.run_classical_benchmark(
                       data_length=n,
                       estimators=[estimator_name]
                   )
                   
                   if estimator_name in result.estimators:
                       estimates.append(
                           result.estimators[estimator_name].mean_estimate
                       )
               
               if estimates:
                   results[estimator_name]['estimates'].append(np.mean(estimates))
                   results[estimator_name]['std'].append(np.std(estimates))
       
       # Plot consistency
       plt.figure(figsize=(12, 5))
       
       plt.subplot(1, 2, 1)
       for estimator_name in results.keys():
           if results[estimator_name]['estimates']:
               plt.semilogx(sample_sizes[:len(results[estimator_name]['estimates'])], 
                           results[estimator_name]['estimates'], 
                           label=estimator_name.upper(), marker='o')
       
       plt.axhline(y=true_H, color='red', linestyle='--', label='True H')
       plt.xlabel('Sample Size')
       plt.ylabel('Estimated H')
       plt.title('Consistency Check')
       plt.legend()
       plt.grid(True)
       
       # Plot standard deviation
       plt.subplot(1, 2, 2)
       for estimator_name in results.keys():
           if results[estimator_name]['std']:
               plt.loglog(sample_sizes[:len(results[estimator_name]['std'])], 
                         results[estimator_name]['std'], 
                         label=estimator_name.upper(), marker='o')
       
       plt.xlabel('Sample Size')
       plt.ylabel('Standard Deviation')
       plt.title('Precision vs Sample Size')
       plt.legend()
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return results

   # Run the example
   if __name__ == "__main__":
       results = estimator_validation_example()
       print("Estimator validation completed!")

Comprehensive Validation Workflow
--------------------------------

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ComprehensiveBenchmark
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   def comprehensive_validation_workflow():
       """Complete validation workflow for LRDBench estimators."""
       
       print("=== COMPREHENSIVE VALIDATION WORKFLOW ===")
       
       # 1. Define validation parameters
       H_values = np.linspace(0.3, 0.9, 13)
       sample_sizes = [500, 1000, 2000]
       n_realizations = 50
       
       # 2. Initialize results storage
       validation_results = []
       
       # 3. Run comprehensive validation
       for H in H_values:
           print(f"Testing H = {H:.2f}")
           
           for n in sample_sizes:
               for i in range(n_realizations):
                   # Generate data
                   model = FBMModel(H=H, sigma=1.0)
                   data = model.generate(n, seed=int(H*1000 + i))
                   
                   # Run benchmark
                   benchmark = ComprehensiveBenchmark()
                   result = benchmark.run_classical_benchmark(
                       data_length=n,
                       estimators=['dfa', 'gph', 'rs', 'higuchi']
                   )
                   
                   # Store results
                   for estimator_name, estimator_result in result.estimators.items():
                       validation_results.append({
                           'true_H': H,
                           'sample_size': n,
                           'realization': i,
                           'estimator': estimator_name,
                           'estimated_H': estimator_result.mean_estimate,
                           'error': abs(estimator_result.mean_estimate - H)
                       })
       
       # 4. Analyze results
       df = pd.DataFrame(validation_results)
       
       print(f"\n=== VALIDATION SUMMARY ===")
       print(f"Total tests: {len(df)}")
       print(f"H range: {df['true_H'].min():.2f} to {df['true_H'].max():.2f}")
       print(f"Sample sizes: {sorted(df['sample_size'].unique())}")
       print(f"Estimators: {sorted(df['estimator'].unique())}")
       
       # 5. Calculate performance metrics
       performance = df.groupby('estimator').agg({
           'error': ['mean', 'std', 'min', 'max'],
           'estimated_H': ['mean', 'std']
       }).round(4)
       
       print(f"\n=== PERFORMANCE METRICS ===")
       print(performance)
       
       # 6. Create visualization
       plt.figure(figsize=(15, 10))
       
       # Error distribution
       plt.subplot(2, 3, 1)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.hist(subset['error'], alpha=0.7, label=estimator.upper(), bins=20)
       plt.xlabel('Absolute Error')
       plt.ylabel('Frequency')
       plt.title('Error Distribution')
       plt.legend()
       
       # Error vs True H
       plt.subplot(2, 3, 2)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['true_H'], subset['error'], alpha=0.6, label=estimator.upper())
       plt.xlabel('True H')
       plt.ylabel('Absolute Error')
       plt.title('Error vs True H')
       plt.legend()
       
       # Error vs Sample Size
       plt.subplot(2, 3, 3)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['sample_size'], subset['error'], alpha=0.6, label=estimator.upper())
       plt.xlabel('Sample Size')
       plt.ylabel('Absolute Error')
       plt.title('Error vs Sample Size')
       plt.legend()
       
       # Estimated vs True H
       plt.subplot(2, 3, 4)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['true_H'], subset['estimated_H'], alpha=0.6, label=estimator.upper())
       plt.plot([0.3, 0.9], [0.3, 0.9], 'r--', label='Perfect')
       plt.xlabel('True H')
       plt.ylabel('Estimated H')
       plt.title('Estimated vs True H')
       plt.legend()
       
       # Box plot by estimator
       plt.subplot(2, 3, 5)
       df.boxplot(column='error', by='estimator', ax=plt.gca())
       plt.title('Error Distribution by Estimator')
       plt.suptitle('')
       
       # Box plot by sample size
       plt.subplot(2, 3, 6)
       df.boxplot(column='error', by='sample_size', ax=plt.gca())
       plt.title('Error Distribution by Sample Size')
       plt.suptitle('')
       
       plt.tight_layout()
       plt.show()
       
       return df, performance

   # Run the comprehensive workflow
   if __name__ == "__main__":
       df, performance = comprehensive_validation_workflow()
       print("Comprehensive validation workflow completed!")

**1. Computational Complexity**: Big-O notation for time and space complexity
**2. Convergence Rate**: Rate at which estimator approaches true value
**3. Asymptotic Efficiency**: Ratio of estimator variance to Cramér-Rao lower bound

**Implementation**:

.. code-block:: python

   import time
   import psutil
   
   def efficiency_analysis(estimator_func, data_sizes):
       """Analyze computational efficiency."""
       results = {}
       
       for n in data_sizes:
           # Generate test data
           test_data = np.random.randn(n)
           
           # Measure execution time
           start_time = time.time()
           start_memory = psutil.Process().memory_info().rss
           
           estimator_func(test_data)
           
           end_time = time.time()
           end_memory = psutil.Process().memory_info().rss
           
           execution_time = end_time - start_time
           memory_usage = end_memory - start_memory
           
           results[n] = {
               'execution_time': execution_time,
               'memory_usage': memory_usage,
               'time_per_sample': execution_time / n,
               'memory_per_sample': memory_usage / n
           }
       
       return results

Quality Assurance
=================

Data Quality Checks
------------------

**1. Stationarity Tests**:

- **Augmented Dickey-Fuller Test**: Tests for unit roots
- **KPSS Test**: Tests for stationarity around a deterministic trend
- **Phillips-Perron Test**: Non-parametric version of ADF test

**2. Normality Tests**:

- **Shapiro-Wilk Test**: Tests for normality
- **Anderson-Darling Test**: Tests for normality with emphasis on tails
- **Jarque-Bera Test**: Tests for normality based on skewness and kurtosis

**Implementation**:

.. code-block:: python

   from statsmodels.tsa.stattools import adfuller, kpss
   from scipy import stats
   
   def data_quality_checks(data):
       """Perform comprehensive data quality checks."""
       results = {}
       
       # Stationarity tests
       adf_result = adfuller(data)
       results['adf_test'] = {
           'statistic': adf_result[0],
           'p_value': adf_result[1],
           'critical_values': adf_result[4]
       }
       
       kpss_result = kpss(data)
       results['kpss_test'] = {
           'statistic': kpss_result[0],
           'p_value': kpss_result[1],
           'critical_values': kpss_result[3]
       }
       
       # Normality tests
       shapiro_result = stats.shapiro(data)
       results['shapiro_test'] = {
           'statistic': shapiro_result[0],
           'p_value': shapiro_result[1]
       }
       
       jarque_bera_result = stats.jarque_bera(data)
       results['jarque_bera_test'] = {
           'statistic': jarque_bera_result[0],
           'p_value': jarque_bera_result[1]
       }
       
       # Basic statistics
       results['basic_stats'] = {
           'mean': np.mean(data),
           'std': np.std(data),
           'skewness': stats.skew(data),
           'kurtosis': stats.kurtosis(data),
           'min': np.min(data),
           'max': np.max(data)
       }
       
       return results

Estimator Validation
-------------------

**1. Consistency Checks**: Verify estimator converges to true value as sample size increases
**2. Unbiasedness Checks**: Verify estimator is unbiased across different parameter values
**3. Efficiency Checks**: Compare estimator variance to theoretical bounds

**Implementation**:

.. code-block:: python

   def estimator_validation(estimator_func, true_H_values, sample_sizes, n_replicates=100):
       """Validate estimator properties."""
       validation_results = {}
       
       for H in true_H_values:
           validation_results[H] = {}
           
           for n in sample_sizes:
               estimates = []
               
               for _ in range(n_replicates):
                   # Generate data with known H
                   model = FBMModel(H=H, sigma=1.0)
                   data = model.generate(n, seed=np.random.randint(10000))
                   
                   # Apply estimator
                   estimate = estimator_func(data)
                   estimates.append(estimate)
               
               # Calculate validation metrics
               estimates = np.array(estimates)
               bias = np.mean(estimates) - H
               variance = np.var(estimates)
               mse = bias**2 + variance
               
               validation_results[H][n] = {
                   'bias': bias,
                   'variance': variance,
                   'mse': mse,
                   'consistency': abs(bias) < 0.1,  # Bias < 0.1
                   'efficiency': variance < 0.01    # Variance < 0.01
               }
       
       return validation_results

Validation References
=====================

1. Beran, J. (1994). Statistics for Long-Memory Processes. Chapman & Hall.
2. Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. CRC Press.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
4. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice. OTexts.
5. Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis. Wiley.
6. Shumway, R. H., & Stoffer, D. S. (2017). Time Series Analysis and Its Applications. Springer.
