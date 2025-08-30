Comprehensive LRDBench Demonstration
===================================

This document provides comprehensive examples demonstrating all major features of lrdbenchmark, from basic usage to advanced analysis workflows.

Basic Data Generation and Analysis
==================================

Simple Hurst Parameter Estimation
----------------------------------

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ComprehensiveBenchmark
   import matplotlib.pyplot as plt
   import numpy as np

   def basic_analysis_demo():
       """Demonstrate basic data generation and analysis."""
       
       print("=== BASIC LRDBENCHMARK ANALYSIS ===")
       
       # 1. Generate data with known Hurst parameters
       models = {
           'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
           'FBM (H=0.3)': FBMModel(H=0.3, sigma=1.0),
           'FGN (H=0.8)': FGNModel(H=0.8, sigma=1.0)
       }
       
       results = {}
       
       for model_name, model in models.items():
           print(f"\n--- Analyzing {model_name} ---")
           
           # Generate data
           data = model.generate(1000, seed=42)
           true_H = model.H
           
           print(f"Generated {len(data)} samples")
           print(f"True H: {true_H:.3f}")
           print(f"Data statistics: mean={data.mean():.3f}, std={data.std():.3f}")
           
           # Run comprehensive benchmark
           benchmark = ComprehensiveBenchmark()
           benchmark_results = benchmark.run_comprehensive_benchmark(
               data_length=1000,
               n_runs=5
           )
           
           # Extract results
           model_results = {}
           for estimator_name, estimator_result in benchmark_results.estimators.items():
               estimated_H = estimator_result.mean_estimate
               error = abs(estimated_H - true_H)
               model_results[estimator_name] = {
                   'estimate': estimated_H,
                   'error': error
               }
               print(f"  {estimator_name}: H = {estimated_H:.3f} (error: {error:.3f})")
           
           results[model_name] = model_results
       
       return results

   # Run basic analysis
   basic_results = basic_analysis_demo()

Data Model Comparison
----------------------

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel
   import matplotlib.pyplot as plt
   import numpy as np

   def data_model_comparison_demo():
       """Compare different data models and their properties."""
       
       print("=== DATA MODEL COMPARISON ===")
       
       # Define models with similar Hurst parameters
       models = {
           'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
           'FGN (H=0.7)': FGNModel(H=0.7, sigma=1.0),
           'ARFIMA (H=0.7)': ARFIMAModel(d=0.2, p=1, q=1),
           'MRW (H=0.7)': MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
       }
       
       # Generate and plot data
       plt.figure(figsize=(15, 10))
       
       for i, (model_name, model) in enumerate(models.items(), 1):
           # Generate data
           data = model.generate(1000, seed=42)
           
           # Plot time series
           plt.subplot(2, 2, i)
           plt.plot(data[:200], linewidth=1)
           plt.title(f'{model_name}\nLength: {len(data)}')
           plt.xlabel('Time')
           plt.ylabel('Value')
           plt.grid(True, alpha=0.3)
           
           # Print statistics
           print(f"\n{model_name}:")
           print(f"  Mean: {data.mean():.4f}")
           print(f"  Std: {data.std():.4f}")
           print(f"  Min: {data.min():.4f}")
           print(f"  Max: {data.max():.4f}")
       
       plt.tight_layout()
       plt.show()
       
       # Compare autocorrelation functions
       plt.figure(figsize=(12, 8))
       
       max_lag = 50
       for model_name, model in models.items():
           data = model.generate(1000, seed=42)
           
           # Compute autocorrelation
           acf = np.correlate(data, data, mode='full')
           acf = acf[len(data)-1:len(data)-1+max_lag] / acf[len(data)-1]
           
           plt.plot(range(max_lag), acf, label=model_name, linewidth=2)
       
       plt.xlabel('Lag')
       plt.ylabel('Autocorrelation')
       plt.title('Autocorrelation Function Comparison')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()

   # Run data model comparison
   data_model_comparison_demo()

Advanced Benchmarking
=====================

Comprehensive Estimator Comparison
----------------------------------

.. code-block:: python

   from lrdbenchmark import ComprehensiveBenchmark, FBMModel
   import pandas as pd
   import matplotlib.pyplot as plt
   import numpy as np

   def comprehensive_estimator_comparison():
       """Compare all estimators across different data conditions."""
       
       print("=== COMPREHENSIVE ESTIMATOR COMPARISON ===")
       
       # Define test conditions
       H_values = np.linspace(0.3, 0.9, 7)
       sample_sizes = [500, 1000, 2000]
       
       # Initialize results storage
       all_results = []
       
       for H in H_values:
           print(f"Testing H = {H:.2f}")
           
           for n in sample_sizes:
               # Generate data
               model = FBMModel(H=H, sigma=1.0)
               data = model.generate(n, seed=int(H*100))
               
               # Run benchmark
               benchmark = ComprehensiveBenchmark()
               results = benchmark.run_comprehensive_benchmark(
                   data_length=n,
                   n_runs=10
               )
               
               # Store results
               for estimator_name, estimator_result in results.estimators.items():
                   all_results.append({
                       'true_H': H,
                       'sample_size': n,
                       'estimator': estimator_name,
                       'estimated_H': estimator_result.mean_estimate,
                       'error': abs(estimator_result.mean_estimate - H),
                       'std': estimator_result.std_estimate
                   })
       
       # Convert to DataFrame
       df = pd.DataFrame(all_results)
       
       # Analysis
       print(f"\n=== ANALYSIS SUMMARY ===")
       print(f"Total tests: {len(df)}")
       print(f"Estimators tested: {sorted(df['estimator'].unique())}")
       
       # Performance by estimator
       print(f"\n=== ESTIMATOR PERFORMANCE ===")
       estimator_performance = df.groupby('estimator')['error'].agg(['mean', 'std', 'min', 'max'])
       print(estimator_performance.round(4))
       
       # Performance by sample size
       print(f"\n=== PERFORMANCE BY SAMPLE SIZE ===")
       size_performance = df.groupby('sample_size')['error'].agg(['mean', 'std'])
       print(size_performance.round(4))
       
       # Create visualizations
       plt.figure(figsize=(15, 10))
       
       # Error distribution by estimator
       plt.subplot(2, 3, 1)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.hist(subset['error'], alpha=0.7, label=estimator, bins=15)
       plt.xlabel('Absolute Error')
       plt.ylabel('Frequency')
       plt.title('Error Distribution by Estimator')
       plt.legend()
       
       # Error vs True H
       plt.subplot(2, 3, 2)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['true_H'], subset['error'], alpha=0.6, label=estimator)
       plt.xlabel('True H')
       plt.ylabel('Absolute Error')
       plt.title('Error vs True H')
       plt.legend()
       
       # Error vs Sample Size
       plt.subplot(2, 3, 3)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['sample_size'], subset['error'], alpha=0.6, label=estimator)
       plt.xlabel('Sample Size')
       plt.ylabel('Absolute Error')
       plt.title('Error vs Sample Size')
       plt.legend()
       
       # Estimated vs True H
       plt.subplot(2, 3, 4)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['true_H'], subset['estimated_H'], alpha=0.6, label=estimator)
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
       
       return df

   # Run comprehensive comparison
   comparison_results = comprehensive_estimator_comparison()

Machine Learning and Neural Network Analysis
============================================

ML Estimator Training and Evaluation
-------------------------------------

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel
   from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
   from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
   from sklearn.model_selection import train_test_split
   import numpy as np
   import matplotlib.pyplot as plt

   def ml_analysis_demo():
       """Demonstrate machine learning estimator training and evaluation."""
       
       print("=== MACHINE LEARNING ANALYSIS ===")
       
       # Generate comprehensive training dataset
       print("Generating training dataset...")
       training_data = []
       training_labels = []
       
       # Create diverse training data
       H_values = np.linspace(0.3, 0.9, 20)
       models = {
           'FBM': FBMModel,
           'FGN': FGNModel,
           'ARFIMA': lambda H: ARFIMAModel(d=H-0.5, p=1, q=1)
       }
       
       for H in H_values:
           for model_name, model_class in models.items():
               if model_name == 'ARFIMA':
                   model = model_class(H)
               else:
                   model = model_class(H=H, sigma=1.0)
               
               # Generate multiple realizations
               for i in range(15):
                   data = model.generate(1000, seed=int(H*1000 + i))
                   training_data.append(data)
                   training_labels.append(H)
       
       print(f"Generated {len(training_data)} training samples")
       
       # Split into training and validation sets
       X_train, X_val, y_train, y_val = train_test_split(
           training_data, training_labels, test_size=0.2, random_state=42
       )
       
       # Train estimators
       estimators = {
           'Random Forest': RandomForestEstimator(n_estimators=100, random_state=42),
           'Gradient Boosting': GradientBoostingEstimator(n_estimators=100, random_state=42)
       }
       
       trained_estimators = {}
       
       for name, estimator in estimators.items():
           print(f"\nTraining {name}...")
           estimator.fit(X_train, y_train)
           trained_estimators[name] = estimator
           
           # Evaluate on validation set
           val_predictions = estimator.estimate(X_val)
           val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(y_val)))
           print(f"  Validation MAE: {val_mae:.4f}")
       
       # Test on new data
       print(f"\n=== TESTING ON NEW DATA ===")
       test_cases = [
           ('FBM (H=0.6)', FBMModel(H=0.6, sigma=1.0), 0.6),
           ('FGN (H=0.4)', FGNModel(H=0.4, sigma=1.0), 0.4),
           ('ARFIMA (H=0.75)', ARFIMAModel(d=0.25, p=1, q=1), 0.75)
       ]
       
       test_results = []
       
       for test_name, test_model, true_H in test_cases:
           test_data = test_model.generate(1000, seed=999)
           
           print(f"\n{test_name}:")
           print(f"  True H: {true_H:.3f}")
           
           for name, estimator in trained_estimators.items():
               prediction = estimator.estimate([test_data])[0]
               error = abs(prediction - true_H)
               print(f"  {name}: H = {prediction:.3f} (error: {error:.3f})")
               
               test_results.append({
                   'test_case': test_name,
                   'estimator': name,
                   'true_H': true_H,
                   'predicted_H': prediction,
                   'error': error
               })
       
       # Visualize results
       plt.figure(figsize=(12, 5))
       
       # Predictions vs True values
       plt.subplot(1, 2, 1)
       for estimator_name in trained_estimators.keys():
           subset = [r for r in test_results if r['estimator'] == estimator_name]
           true_vals = [r['true_H'] for r in subset]
           pred_vals = [r['predicted_H'] for r in subset]
           plt.scatter(true_vals, pred_vals, label=estimator_name, alpha=0.7)
       
       plt.plot([0.3, 0.9], [0.3, 0.9], 'r--', label='Perfect')
       plt.xlabel('True H')
       plt.ylabel('Predicted H')
       plt.title('ML Estimator Predictions')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Error comparison
       plt.subplot(1, 2, 2)
       for estimator_name in trained_estimators.keys():
           subset = [r for r in test_results if r['estimator'] == estimator_name]
           errors = [r['error'] for r in subset]
           plt.bar(estimator_name, np.mean(errors), alpha=0.7, label=estimator_name)
       
       plt.ylabel('Mean Absolute Error')
       plt.title('ML Estimator Performance')
       plt.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return trained_estimators, test_results

   # Run ML analysis
   ml_estimators, ml_results = ml_analysis_demo()

Analytics and Monitoring
========================

Comprehensive Analytics Demo
----------------------------

.. code-block:: python

   from lrdbenchmark import enable_analytics, get_analytics_summary
   from lrdbenchmark.analytics import AnalyticsDashboard
   from lrdbenchmark import FBMModel, FGNModel, ComprehensiveBenchmark
   import time

   def analytics_demo():
       """Demonstrate the analytics and monitoring capabilities."""
       
       print("=== ANALYTICS AND MONITORING DEMO ===")
       
       # Enable analytics
       print("Enabling analytics system...")
       enable_analytics()
       
       # Create dashboard
       dashboard = AnalyticsDashboard()
       
       # Run various analyses with tracking
       print("\nRunning analyses with analytics tracking...")
       
       # Analysis 1: Basic FBM analysis
       print("Analysis 1: FBM data analysis")
       model1 = FBMModel(H=0.7, sigma=1.0)
       data1 = model1.generate(1000, seed=42)
       
       benchmark1 = ComprehensiveBenchmark()
       results1 = benchmark1.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=5
       )
       
       # Analysis 2: FGN analysis
       print("Analysis 2: FGN data analysis")
       model2 = FGNModel(H=0.8, sigma=1.0)
       data2 = model2.generate(1000, seed=123)
       
       benchmark2 = ComprehensiveBenchmark()
       results2 = benchmark2.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=5
       )
       
       # Analysis 3: Parameter sweep
       print("Analysis 3: Parameter sweep")
       for H in [0.3, 0.5, 0.7, 0.9]:
           model = FBMModel(H=H, sigma=1.0)
           data = model.generate(500, seed=int(H*100))
           
           benchmark = ComprehensiveBenchmark()
           results = benchmark.run_classical_benchmark(
               data_length=500,
               estimators=['dfa', 'gph']
           )
       
       # Generate analytics reports
       print("\n=== ANALYTICS REPORTS ===")
       
       # Comprehensive summary
       print("1. Comprehensive Summary:")
       summary = dashboard.get_comprehensive_summary()
       print(summary)
       
       # Usage report
       print("\n2. Usage Report:")
       usage_report = dashboard.generate_usage_report()
       print(usage_report)
       
       # Performance report
       print("\n3. Performance Report:")
       performance_report = dashboard.generate_performance_report()
       print(performance_report)
       
       # Reliability report
       print("\n4. Reliability Report:")
       reliability_report = dashboard.generate_reliability_report()
       print(reliability_report)
       
       # Workflow report
       print("\n5. Workflow Report:")
       workflow_report = dashboard.generate_workflow_report()
       print(workflow_report)
       
       return dashboard

   # Run analytics demo
   analytics_dashboard = analytics_demo()

Real-World Application Example
==============================

Financial Time Series Analysis
------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from lrdbenchmark import ComprehensiveBenchmark
   from lrdbenchmark.analysis.temporal.dfa.dfa_estimator import DFAEstimator
   from lrdbenchmark.analysis.spectral.gph.gph_estimator import GPHEstimator

   def financial_analysis_demo():
       """Demonstrate LRDBench for financial time series analysis."""
       
       print("=== FINANCIAL TIME SERIES ANALYSIS ===")
       
       # Simulate financial returns with different persistence levels
       np.random.seed(42)
       
       # Generate synthetic financial data
       n_samples = 2000
       
       # High persistence (trending market)
       high_persistence = np.cumsum(np.random.normal(0, 0.01, n_samples))
       
       # Low persistence (mean-reverting market)
       low_persistence = np.zeros(n_samples)
       for i in range(1, n_samples):
           low_persistence[i] = 0.9 * low_persistence[i-1] + np.random.normal(0, 0.01)
       
       # Random walk (efficient market)
       random_walk = np.cumsum(np.random.normal(0, 0.01, n_samples))
       
       datasets = {
           'High Persistence': high_persistence,
           'Low Persistence': low_persistence,
           'Random Walk': random_walk
       }
       
       # Analyze each dataset
       results = {}
       
       for dataset_name, data in datasets.items():
           print(f"\n--- Analyzing {dataset_name} ---")
           
           # Calculate returns
           returns = np.diff(data)
           
           print(f"Data length: {len(data)}")
           print(f"Returns mean: {returns.mean():.6f}")
           print(f"Returns std: {returns.std():.6f}")
           
           # Apply estimators
           estimators = {
               'DFA': DFAEstimator(),
               'GPH': GPHEstimator()
           }
           
           dataset_results = {}
           
           for name, estimator in estimators.items():
               try:
                   H_estimate = estimator.estimate(returns)
                   dataset_results[name] = H_estimate
                   print(f"  {name}: H = {H_estimate:.3f}")
               except Exception as e:
                   print(f"  {name}: Error - {e}")
                   dataset_results[name] = None
           
           results[dataset_name] = dataset_results
       
       # Visualize results
       plt.figure(figsize=(15, 10))
       
       # Time series plots
       for i, (dataset_name, data) in enumerate(datasets.items(), 1):
           plt.subplot(3, 3, i)
           plt.plot(data[:500], linewidth=1)
           plt.title(f'{dataset_name}\nTime Series')
           plt.xlabel('Time')
           plt.ylabel('Price')
           plt.grid(True, alpha=0.3)
       
       # Returns plots
       for i, (dataset_name, data) in enumerate(datasets.items(), 4):
           returns = np.diff(data)
           plt.subplot(3, 3, i)
           plt.plot(returns[:500], linewidth=1)
           plt.title(f'{dataset_name}\nReturns')
           plt.xlabel('Time')
           plt.ylabel('Returns')
           plt.grid(True, alpha=0.3)
       
       # Hurst parameter comparison
       plt.subplot(3, 3, 7)
       dataset_names = list(results.keys())
       dfa_estimates = [results[name]['DFA'] for name in dataset_names if results[name]['DFA'] is not None]
       gph_estimates = [results[name]['GPH'] for name in dataset_names if results[name]['GPH'] is not None]
       
       x = np.arange(len(dataset_names))
       width = 0.35
       
       plt.bar(x - width/2, dfa_estimates, width, label='DFA', alpha=0.7)
       plt.bar(x + width/2, gph_estimates, width, label='GPH', alpha=0.7)
       
       plt.xlabel('Dataset')
       plt.ylabel('Hurst Parameter')
       plt.title('Hurst Parameter Estimates')
       plt.xticks(x, dataset_names)
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Market efficiency interpretation
       plt.subplot(3, 3, 8)
       efficiency_levels = []
       for name in dataset_names:
           if results[name]['DFA'] is not None:
               H = results[name]['DFA']
               if H < 0.5:
                   efficiency = 'Mean Reverting'
               elif H > 0.5:
                   efficiency = 'Trending'
               else:
                   efficiency = 'Random Walk'
               efficiency_levels.append(efficiency)
           else:
               efficiency_levels.append('Unknown')
       
       efficiency_counts = pd.Series(efficiency_levels).value_counts()
       plt.pie(efficiency_counts.values, labels=efficiency_counts.index, autopct='%1.1f%%')
       plt.title('Market Efficiency Classification')
       
       # Risk analysis
       plt.subplot(3, 3, 9)
       volatilities = [np.diff(data).std() for data in datasets.values()]
       plt.bar(dataset_names, volatilities, alpha=0.7)
       plt.xlabel('Dataset')
       plt.ylabel('Volatility')
       plt.title('Return Volatility')
       plt.xticks(rotation=45)
       plt.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return results

   # Run financial analysis
   financial_results = financial_analysis_demo()

Integration with External Libraries
===================================

HPFracc Integration Example
---------------------------

.. code-block:: python

   def hpfracc_integration_demo():
       """Demonstrate integration with HPFracc fractional neural networks."""
       
       print("=== HPFRACC INTEGRATION DEMO ===")
       
       try:
           from scripts.hpfracc_proper_benchmark import HPFraccProperBenchmark
           
           # Create benchmark
           print("Creating HPFracc benchmark...")
           benchmark = HPFraccProperBenchmark(
               series_length=1000,
               batch_size=32,
               input_window=10,
               prediction_horizon=1
           )
           
           # Run comparison
           print("Running HPFracc vs LRDBench comparison...")
           results = benchmark.run_benchmark()
           
           # Generate report
           print("Generating comparison report...")
           report = benchmark.generate_report()
           print(report)
           
           return results
           
       except ImportError:
           print("HPFracc not available. Install with: pip install hpfracc")
           return None
       except Exception as e:
           print(f"HPFracc integration failed: {e}")
           return None

   # Run HPFracc integration demo
   hpfracc_results = hpfracc_integration_demo()

Complete Workflow Example
=========================

End-to-End Analysis Pipeline
----------------------------

.. code-block:: python

   def complete_workflow_demo():
       """Demonstrate a complete end-to-end analysis workflow."""
       
       print("=== COMPLETE WORKFLOW DEMO ===")
       
       # Step 1: Data Generation
       print("Step 1: Generating synthetic data...")
       from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel
       
       models = {
           'FBM': FBMModel(H=0.7, sigma=1.0),
           'FGN': FGNModel(H=0.8, sigma=1.0),
           'ARFIMA': ARFIMAModel(d=0.2, p=1, q=1)
       }
       
       datasets = {}
       for name, model in models.items():
           data = model.generate(2000, seed=42)
           datasets[name] = data
           print(f"  Generated {name}: {len(data)} samples")
       
       # Step 2: Data Quality Assessment
       print("\nStep 2: Assessing data quality...")
       from scipy import stats
       
       for name, data in datasets.items():
           print(f"\n{name} Quality Assessment:")
           print(f"  Mean: {data.mean():.4f}")
           print(f"  Std: {data.std():.4f}")
           print(f"  Skewness: {stats.skew(data):.4f}")
           print(f"  Kurtosis: {stats.kurtosis(data):.4f}")
           
           # Stationarity test
           from statsmodels.tsa.stattools import adfuller
           adf_stat, adf_pvalue = adfuller(data)[:2]
           print(f"  ADF p-value: {adf_pvalue:.4f}")
           print(f"  Stationary: {'Yes' if adf_pvalue < 0.05 else 'No'}")
       
       # Step 3: Comprehensive Benchmarking
       print("\nStep 3: Running comprehensive benchmark...")
       from lrdbenchmark import ComprehensiveBenchmark
       
       benchmark = ComprehensiveBenchmark()
       benchmark_results = benchmark.run_comprehensive_benchmark(
           data_length=2000,
           n_runs=10
       )
       
       # Step 4: Results Analysis
       print("\nStep 4: Analyzing results...")
       import pandas as pd
       
       df = benchmark_results.to_dataframe()
       
       print(f"Benchmark completed:")
       print(f"  Total tests: {len(df)}")
       print(f"  Estimators: {sorted(df['estimator'].unique())}")
       print(f"  Data models: {sorted(df['data_model'].unique())}")
       
       # Performance summary
       performance = df.groupby('estimator')['estimated_H'].agg(['mean', 'std', 'count'])
       print(f"\nEstimator Performance:")
       print(performance.round(4))
       
       # Step 5: Visualization
       print("\nStep 5: Creating visualizations...")
       import matplotlib.pyplot as plt
       
       plt.figure(figsize=(15, 10))
       
       # Results by estimator
       plt.subplot(2, 3, 1)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.hist(subset['estimated_H'], alpha=0.7, label=estimator, bins=15)
       plt.xlabel('Estimated H')
       plt.ylabel('Frequency')
       plt.title('Distribution of Estimates')
       plt.legend()
       
       # Results by data model
       plt.subplot(2, 3, 2)
       for model in df['data_model'].unique():
           subset = df[df['data_model'] == model]
           plt.hist(subset['estimated_H'], alpha=0.7, label=model, bins=15)
       plt.xlabel('Estimated H')
       plt.ylabel('Frequency')
       plt.title('Estimates by Data Model')
       plt.legend()
       
       # Box plot by estimator
       plt.subplot(2, 3, 3)
       df.boxplot(column='estimated_H', by='estimator', ax=plt.gca())
       plt.title('Estimates by Estimator')
       plt.suptitle('')
       
       # Scatter plot: estimated vs true H
       plt.subplot(2, 3, 4)
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.scatter(subset['true_H'], subset['estimated_H'], 
                      alpha=0.6, label=estimator)
       plt.plot([0.3, 0.9], [0.3, 0.9], 'r--', label='Perfect')
       plt.xlabel('True H')
       plt.ylabel('Estimated H')
       plt.title('Estimated vs True H')
       plt.legend()
       
       # Error analysis
       plt.subplot(2, 3, 5)
       df['error'] = abs(df['estimated_H'] - df['true_H'])
       for estimator in df['estimator'].unique():
           subset = df[df['estimator'] == estimator]
           plt.hist(subset['error'], alpha=0.7, label=estimator, bins=15)
       plt.xlabel('Absolute Error')
       plt.ylabel('Frequency')
       plt.title('Error Distribution')
       plt.legend()
       
       # Performance comparison
       plt.subplot(2, 3, 6)
       error_by_estimator = df.groupby('estimator')['error'].mean()
       plt.bar(error_by_estimator.index, error_by_estimator.values, alpha=0.7)
       plt.xlabel('Estimator')
       plt.ylabel('Mean Absolute Error')
       plt.title('Estimator Performance')
       plt.xticks(rotation=45)
       
       plt.tight_layout()
       plt.show()
       
       # Step 6: Report Generation
       print("\nStep 6: Generating final report...")
       
       report = f"""
       === LRDBENCH ANALYSIS REPORT ===
       
       Data Generation:
       - Generated {len(datasets)} datasets
       - Total samples: {sum(len(data) for data in datasets.values())}
       
       Benchmark Results:
       - Total tests: {len(df)}
       - Estimators tested: {len(df['estimator'].unique())}
       - Data models: {len(df['data_model'].unique())}
       
       Performance Summary:
       - Best estimator: {error_by_estimator.idxmin()} (MAE: {error_by_estimator.min():.4f})
       - Worst estimator: {error_by_estimator.idxmax()} (MAE: {error_by_estimator.max():.4f})
       
       Recommendations:
       - Use {error_by_estimator.idxmin()} for highest accuracy
       - Consider multiple estimators for robust analysis
       - Validate results with different data models
       """
       
       print(report)
       
       return benchmark_results, df, report

   # Run complete workflow
   workflow_results, workflow_df, workflow_report = complete_workflow_demo()

Summary
=======

This comprehensive demonstration showcases the full capabilities of LRDBench:

1. **Basic Analysis**: Simple data generation and Hurst parameter estimation
2. **Data Model Comparison**: Understanding different LRD processes
3. **Advanced Benchmarking**: Comprehensive estimator comparison
4. **Machine Learning**: Training and evaluating ML-based estimators
5. **Analytics**: Monitoring and tracking analysis performance
6. **Real-World Applications**: Financial time series analysis
7. **External Integration**: HPFracc fractional neural networks
8. **Complete Workflow**: End-to-end analysis pipeline

Each example provides practical code that can be adapted for specific research needs. The demonstrations show how LRDBench can be used for both educational purposes and serious research applications in long-range dependence analysis.
