Benchmark API
============

LRDBench provides a comprehensive benchmarking framework for evaluating and comparing different estimators of long-range dependence.

Comprehensive Benchmark
---------------------

.. autoclass:: lrdbench.analysis.benchmark.ComprehensiveBenchmark
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: run_comprehensive_benchmark
   .. automethod:: run_classical_benchmark
   .. automethod:: run_ml_benchmark
   .. automethod:: run_neural_benchmark

Benchmark Results
----------------

.. autoclass:: lrdbench.analysis.benchmark.BenchmarkResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbench.analysis.benchmark.EstimatorResult
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark Configuration
----------------------

.. autoclass:: lrdbench.analysis.benchmark.BenchmarkConfig
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
-------------

Basic Benchmark
~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import pandas as pd

   # Create benchmark instance
   benchmark = ComprehensiveBenchmark()

   print("Running comprehensive benchmark...")
   print("This will test multiple estimators on various data models")
   
   # Run comprehensive benchmark
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       n_runs=10
   )

   # Access results
   print(f"\n=== BENCHMARK RESULTS ===")
   print(f"Number of estimators tested: {len(results.estimators)}")
   print(f"Number of datasets generated: {len(results.datasets)}")
   print(f"Total runs completed: {len(results.estimators) * len(results.datasets) * 10}")

   # Get summary statistics
   summary = results.get_summary()
   print(f"\n=== SUMMARY STATISTICS ===")
   print(summary)
   
   # Convert to DataFrame for detailed analysis
   df = results.to_dataframe()
   print(f"\n=== DETAILED RESULTS ===")
   print(f"DataFrame shape: {df.shape}")
   print(f"Columns: {list(df.columns)}")
   
   # Show top performing estimators
   estimator_performance = df.groupby('estimator')['estimated_H'].agg(['mean', 'std', 'count'])
   print(f"\n=== ESTIMATOR PERFORMANCE ===")
   print(estimator_performance.round(3))
   
   # Show results by data model
   model_performance = df.groupby('data_model')['estimated_H'].agg(['mean', 'std', 'count'])
   print(f"\n=== DATA MODEL PERFORMANCE ===")
   print(model_performance.round(3))

Classical Estimators Only
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark()
   
   # Run only classical estimators
   results = benchmark.run_classical_benchmark(
       data_length=1000,
       estimators=['dfa', 'rs', 'gph', 'wavelet_variance'],
       n_runs=5
   )
   
   # Get results for specific estimator
   dfa_results = results.get_estimator_results('dfa')
   print(f"DFA mean H estimate: {dfa_results.mean_estimate:.3f}")
   print(f"DFA standard error: {dfa_results.std_error:.3f}")

Machine Learning Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark()
   
   # Run ML estimators with custom parameters
   results = benchmark.run_ml_benchmark(
       data_length=1000,
       estimators=['random_forest', 'gradient_boosting', 'svr'],
       n_runs=3,
       train_test_split=0.8
   )
   
   # Get performance metrics
   for estimator_name, result in results.estimators.items():
       print(f"{estimator_name}:")
       print(f"  Mean H estimate: {result.mean_estimate:.3f}")
       print(f"  RMSE: {result.rmse:.3f}")
       print(f"  MAE: {result.mae:.3f}")

Neural Network Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark()
   
   # Run neural network estimators
   results = benchmark.run_neural_benchmark(
       data_length=1000,
       estimators=['cnn', 'lstm', 'transformer'],
       n_runs=2,
       epochs=50,
       batch_size=32
   )
   
   # Get training history
   for estimator_name, result in results.estimators.items():
       if hasattr(result, 'training_history'):
           print(f"{estimator_name} training completed")
           print(f"  Final loss: {result.training_history['loss'][-1]:.4f}")

Custom Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark, BenchmarkConfig
   
   # Create custom configuration
   config = BenchmarkConfig(
       data_models=['fbm', 'fgn', 'arfima'],
       estimators=['dfa', 'gph', 'random_forest'],
       data_lengths=[500, 1000, 2000],
       n_runs=5,
       random_seed=42
   )
   
   # Create benchmark with custom config
   benchmark = ComprehensiveBenchmark(config=config)
   
   # Run benchmark
   results = benchmark.run_comprehensive_benchmark()
   
   # Get results for specific data length
   results_1000 = results.get_results_by_length(1000)
   print(f"Results for length 1000: {len(results_1000.estimators)} estimators")

Advanced Usage
-------------

Parallel Processing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import multiprocessing as mp
   
   # Set number of processes
   mp.set_start_method('spawn', force=True)
   
   benchmark = ComprehensiveBenchmark()
   
   # Run benchmark with parallel processing
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       n_runs=20,
       n_jobs=4  # Use 4 parallel processes
   )

Custom Data Models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark, FBMModel, FGNModel
   
   # Create custom data models
   custom_models = {
       'fbm_high': FBMModel(H=0.8, sigma=1.0),
       'fbm_low': FBMModel(H=0.3, sigma=1.0),
       'fgn_medium': FGNModel(H=0.6, sigma=1.0)
   }
   
   benchmark = ComprehensiveBenchmark()
   
   # Run benchmark with custom models
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       custom_models=custom_models,
       n_runs=5
   )

Custom Estimators
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
   
   # Create custom estimator
   custom_dfa = DFAEstimator(
       min_scale=4,
       max_scale=100,
       num_scales=20,
       polynomial_order=2
   )
   
   custom_estimators = {
       'custom_dfa': custom_dfa
   }
   
   benchmark = ComprehensiveBenchmark()
   
   # Run benchmark with custom estimator
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       custom_estimators=custom_estimators,
       n_runs=5
   )

Results Analysis
---------------

Statistical Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import pandas as pd
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=10)
   
   # Convert to pandas DataFrame for analysis
   df = results.to_dataframe()
   
   # Group by estimator and calculate statistics
   stats = df.groupby('estimator')['estimated_H'].agg([
       'mean', 'std', 'min', 'max', 'count'
   ]).round(3)
   
   print("Estimator Statistics:")
   print(stats)
   
   # Calculate bias for each estimator
   true_H = df['true_H'].iloc[0]  # Assuming same true H for all
   bias = df.groupby('estimator')['estimated_H'].mean() - true_H
   
   print(f"\nBias (estimated - true H = {true_H}):")
   print(bias.round(3))

Visualization
~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=10)
   
   # Create box plot
   df = results.to_dataframe()
   
   plt.figure(figsize=(12, 6))
   sns.boxplot(data=df, x='estimator', y='estimated_H')
   plt.axhline(y=df['true_H'].iloc[0], color='red', linestyle='--', label='True H')
   plt.title('Hurst Parameter Estimates by Estimator')
   plt.xticks(rotation=45)
   plt.legend()
   plt.tight_layout()
   plt.show()
   
   # Create scatter plot
   plt.figure(figsize=(10, 6))
   for estimator in df['estimator'].unique():
       subset = df[df['estimator'] == estimator]
       plt.scatter(subset['true_H'], subset['estimated_H'], 
                  label=estimator, alpha=0.6)
   
   plt.plot([0.3, 0.9], [0.3, 0.9], 'k--', label='Perfect Estimation')
   plt.xlabel('True Hurst Parameter')
   plt.ylabel('Estimated Hurst Parameter')
   plt.title('True vs Estimated Hurst Parameters')
   plt.legend()
   plt.grid(True)
   plt.show()

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import time
   
   benchmark = ComprehensiveBenchmark()
   
   # Measure execution time
   estimators = ['dfa', 'rs', 'gph', 'wavelet_variance']
   execution_times = {}
   
   for estimator in estimators:
       start_time = time.time()
       results = benchmark.run_classical_benchmark(
           data_length=1000,
           estimators=[estimator],
           n_runs=5
       )
       execution_time = time.time() - start_time
       execution_times[estimator] = execution_time
   
   print("Execution Times:")
   for estimator, time_taken in execution_times.items():
       print(f"{estimator}: {time_taken:.2f} seconds")

Error Analysis
~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import numpy as np
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=10)
   
   df = results.to_dataframe()
   
   # Calculate errors
   df['error'] = df['estimated_H'] - df['true_H']
   df['abs_error'] = np.abs(df['error'])
   df['squared_error'] = df['error']**2
   
   # Error statistics by estimator
   error_stats = df.groupby('estimator').agg({
       'error': ['mean', 'std'],
       'abs_error': 'mean',
       'squared_error': 'mean'
   }).round(4)
   
   error_stats.columns = ['Bias', 'Bias_Std', 'MAE', 'MSE']
   print("Error Statistics:")
   print(error_stats)
   
   # Identify outliers
   Q1 = df.groupby('estimator')['error'].quantile(0.25)
   Q3 = df.groupby('estimator')['error'].quantile(0.75)
   IQR = Q3 - Q1
   
   outliers = df[
       (df['error'] < (Q1 - 1.5 * IQR).loc[df['estimator']]) |
       (df['error'] > (Q3 + 1.5 * IQR).loc[df['estimator']])
   ]
   
   print(f"\nNumber of outliers: {len(outliers)}")

Confidence Intervals
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import scipy.stats as stats
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=20)
   
   df = results.to_dataframe()
   
   # Calculate confidence intervals
   confidence_level = 0.95
   alpha = 1 - confidence_level
   
   ci_results = {}
   for estimator in df['estimator'].unique():
       subset = df[df['estimator'] == estimator]
       estimates = subset['estimated_H'].values
       
       # Bootstrap confidence interval
       n_bootstrap = 1000
       bootstrap_means = []
       
       for _ in range(n_bootstrap):
           bootstrap_sample = np.random.choice(estimates, size=len(estimates), replace=True)
           bootstrap_means.append(np.mean(bootstrap_sample))
       
       lower_ci = np.percentile(bootstrap_means, alpha/2 * 100)
       upper_ci = np.percentile(bootstrap_means, (1-alpha/2) * 100)
       
       ci_results[estimator] = {
           'mean': np.mean(estimates),
           'lower_ci': lower_ci,
           'upper_ci': upper_ci,
           'width': upper_ci - lower_ci
       }
   
   print("Confidence Intervals (95%):")
   for estimator, ci in ci_results.items():
       print(f"{estimator}: {ci['mean']:.3f} [{ci['lower_ci']:.3f}, {ci['upper_ci']:.3f}]")

Export and Reporting
-------------------

Export Results
~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   import json
   import pandas as pd
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=5)
   
   # Export to JSON
   results.save_json('benchmark_results.json')
   
   # Export to CSV
   df = results.to_dataframe()
   df.to_csv('benchmark_results.csv', index=False)
   
   # Export to Excel
   with pd.ExcelWriter('benchmark_results.xlsx') as writer:
       df.to_excel(writer, sheet_name='Results', index=False)
       
       # Create summary sheet
       summary = results.get_summary()
       summary_df = pd.DataFrame([summary])
       summary_df.to_excel(writer, sheet_name='Summary', index=False)

Generate Reports
~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000, n_runs=10)
   
   # Generate comprehensive report
   report = results.generate_report(
       include_plots=True,
       include_statistics=True,
       include_recommendations=True
   )
   
   # Save report
   with open('benchmark_report.html', 'w') as f:
       f.write(report)
   
   # Print summary
   print(results.get_summary())

Best Practices
-------------

1. **Sample Size**: Use at least 1000 data points for reliable estimates
2. **Number of Runs**: Use 10-20 runs for stable statistics
3. **Multiple Estimators**: Compare results from different estimator types
4. **Data Models**: Test on various synthetic data models
5. **Error Handling**: Always handle potential estimation failures
6. **Performance Monitoring**: Track execution times for large-scale benchmarks
7. **Result Validation**: Cross-validate results with known theoretical values

.. note::
   The benchmark system automatically handles parallel processing, error recovery,
   and result aggregation. For large-scale benchmarks, consider using the
   parallel processing capabilities.

.. warning::
   Some estimators may fail on certain data types or parameter combinations.
   The benchmark system will report failures but continue with other estimators.
