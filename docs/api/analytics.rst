Analytics API
============

lrdbenchmark provides a comprehensive analytics system for tracking usage, monitoring performance, analyzing errors, and understanding user workflows.

Analytics Dashboard
-----------------

.. autoclass:: lrdbenchmark.analytics.dashboard.AnalyticsDashboard
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: get_comprehensive_summary
   .. automethod:: generate_usage_report
   .. automethod:: generate_performance_report
   .. automethod:: generate_reliability_report
   .. automethod:: generate_workflow_report

Usage Tracking
-------------

.. autoclass:: lrdbenchmark.analytics.usage_tracker.UsageTracker
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: track_estimator_usage
   .. automethod:: track_benchmark_run
   .. automethod:: get_usage_summary

.. autoclass:: lrdbenchmark.analytics.usage_tracker.UsageEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbenchmark.analytics.usage_tracker.UsageSummary
   :members:
   :undoc-members:
   :show-inheritance:

Performance Monitoring
---------------------

.. autoclass:: lrdbenchmark.analytics.performance_monitor.PerformanceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: start_monitoring
   .. automethod:: stop_monitoring
   .. automethod:: get_performance_summary

.. autoclass:: lrdbenchmark.analytics.performance_monitor.PerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbenchmark.analytics.performance_monitor.PerformanceSummary
   :members:
   :undoc-members:
   :show-inheritance:

Error Analysis
-------------

.. autoclass:: lrdbenchmark.analytics.error_analyzer.ErrorAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: record_error
   .. automethod:: get_error_summary
   .. automethod:: get_improvement_recommendations

.. autoclass:: lrdbenchmark.analytics.error_analyzer.ErrorEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbenchmark.analytics.error_analyzer.ErrorSummary
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Analysis
----------------

.. autoclass:: lrdbenchmark.analytics.workflow_analyzer.WorkflowAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: track_workflow_step
   .. automethod:: get_workflow_summary
   .. automethod:: get_optimization_recommendations

.. autoclass:: lrdbench.analytics.workflow_analyzer.WorkflowStep
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lrdbench.analytics.workflow_analyzer.Workflow
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
--------------------

.. autofunction:: lrdbench.enable_analytics
.. autofunction:: lrdbench.get_analytics_summary
.. autofunction:: lrdbench.generate_analytics_report
.. autofunction:: lrdbench.track_usage
.. autofunction:: lrdbench.monitor_performance
.. autofunction:: lrdbench.track_errors
.. autofunction:: lrdbench.track_workflow

Usage Examples
-------------

Basic Analytics Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import enable_analytics, get_analytics_summary
   from lrdbenchmark.analytics import AnalyticsDashboard

   # Enable analytics system
   print("Enabling LRDBench analytics system...")
   enable_analytics()

   # Your analysis code here
   from lrdbenchmark import FBMModel, FGNModel, ComprehensiveBenchmark
   import time

   print("Running analysis with analytics tracking...")
   
   # Generate data with different models
   models = {
       'FBM (H=0.7)': FBMModel(H=0.7, sigma=1.0),
       'FBM (H=0.3)': FBMModel(H=0.3, sigma=1.0),
       'FGN (H=0.8)': FGNModel(H=0.8, sigma=1.0)
   }
   
   for model_name, model in models.items():
       print(f"Generating {model_name} data...")
       data = model.generate(1000, seed=42)
       
       # Run benchmark
       benchmark = ComprehensiveBenchmark()
       results = benchmark.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=5
       )
       
       print(f"Completed benchmark for {model_name}")

   # Get comprehensive analytics summary
   print("\n=== ANALYTICS SUMMARY ===")
   summary = get_analytics_summary()
   print(summary)
   
   # Create dashboard for detailed analysis
   dashboard = AnalyticsDashboard()
   
   # Generate specific reports
   print("\n=== USAGE REPORT ===")
   usage_report = dashboard.generate_usage_report()
   print(usage_report)
   
   print("\n=== PERFORMANCE REPORT ===")
   performance_report = dashboard.generate_performance_report()
   print(performance_report)
   
   print("\n=== RELIABILITY REPORT ===")
   reliability_report = dashboard.generate_reliability_report()
   print(reliability_report)

Usage Tracking with Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import track_usage, FBMModel
   
   @track_usage
   def analyze_fbm_data(H=0.7, length=1000):
       """Analyze FBM data with given parameters."""
       model = FBMModel(H=H, sigma=1.0)
       data = model.generate(length, seed=42)
       
       # Perform analysis
       return data.mean(), data.std()
   
   # Function calls will be automatically tracked
   mean_val, std_val = analyze_fbm_data(H=0.8, length=2000)
   mean_val2, std_val2 = analyze_fbm_data(H=0.6, length=1000)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import monitor_performance, ComprehensiveBenchmark
   
   @monitor_performance
   def run_benchmark_analysis():
       """Run comprehensive benchmark analysis."""
       benchmark = ComprehensiveBenchmark()
       results = benchmark.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=10
       )
       return results
   
   # Performance will be automatically monitored
   results = run_benchmark_analysis()
   
   # Get performance summary
   from lrdbenchmark.analytics import PerformanceMonitor
   monitor = PerformanceMonitor()
   perf_summary = monitor.get_performance_summary()
   print(f"Average execution time: {perf_summary.avg_execution_time:.2f}s")

Error Tracking
~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import track_errors, ComprehensiveBenchmark
   
   @track_errors
   def run_estimator_analysis():
       """Run estimator analysis with error tracking."""
       benchmark = ComprehensiveBenchmark()
       
       try:
           results = benchmark.run_comprehensive_benchmark(
               data_length=1000,
               n_runs=5
           )
           return results
       except Exception as e:
           # Errors will be automatically tracked
           raise e
   
   # Run analysis
   try:
       results = run_estimator_analysis()
   except Exception as e:
       print(f"Analysis failed: {e}")
   
   # Get error summary
   from lrdbench.analytics import ErrorAnalyzer
   error_analyzer = ErrorAnalyzer()
   error_summary = error_analyzer.get_error_summary()
   print(f"Total errors: {error_summary.total_errors}")

Workflow Tracking
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import track_workflow, FBMModel, ComprehensiveBenchmark
   
   @track_workflow
   def complete_analysis_workflow():
       """Complete analysis workflow with tracking."""
       # Step 1: Data generation
       model = FBMModel(H=0.7, sigma=1.0)
       data = model.generate(1000, seed=42)
       
       # Step 2: Benchmark execution
       benchmark = ComprehensiveBenchmark()
       results = benchmark.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=5
       )
       
       # Step 3: Results analysis
       summary = results.get_summary()
       
       return summary
   
   # Workflow will be automatically tracked
   summary = complete_analysis_workflow()
   
   # Get workflow summary
   from lrdbench.analytics import WorkflowAnalyzer
   workflow_analyzer = WorkflowAnalyzer()
   workflow_summary = workflow_analyzer.get_workflow_summary()
   print(f"Workflows completed: {workflow_summary.total_workflows}")

Advanced Analytics Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import AnalyticsDashboard
   
   # Create analytics dashboard
   dashboard = AnalyticsDashboard()
   
   # Generate comprehensive analytics report
   report = dashboard.get_comprehensive_summary()
   print("=== COMPREHENSIVE ANALYTICS REPORT ===")
   print(report)
   
   # Generate specific reports
   usage_report = dashboard.generate_usage_report()
   performance_report = dashboard.generate_performance_report()
   reliability_report = dashboard.generate_reliability_report()
   workflow_report = dashboard.generate_workflow_report()
   
   print("\n=== USAGE REPORT ===")
   print(usage_report)
   
   print("\n=== PERFORMANCE REPORT ===")
   print(performance_report)
   
   print("\n=== RELIABILITY REPORT ===")
   print(reliability_report)
   
   print("\n=== WORKFLOW REPORT ===")
   print(workflow_report)

Custom Analytics Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import (
       UsageTracker, PerformanceMonitor, ErrorAnalyzer, WorkflowAnalyzer
   )
   
   # Create custom analytics components
   usage_tracker = UsageTracker(
       track_user_id=True,
       track_parameters=True,
       track_timing=True
   )
   
   performance_monitor = PerformanceMonitor(
       track_memory=True,
       track_cpu=True,
       track_gpu=True
   )
   
   error_analyzer = ErrorAnalyzer(
       categorize_errors=True,
       track_stack_traces=True,
       generate_recommendations=True
   )
   
   workflow_analyzer = WorkflowAnalyzer(
       track_step_dependencies=True,
       analyze_patterns=True,
       generate_optimizations=True
   )
   
   # Use custom components
   usage_tracker.track_estimator_usage(
       estimator_name='dfa',
       parameters={'min_scale': 4, 'max_scale': 100},
       execution_time=1.23,
       success=True
   )
   
   performance_monitor.start_monitoring()
   # ... your code here ...
   performance_monitor.stop_monitoring()
   
   error_analyzer.record_error(
       error_type='ValueError',
       error_message='Invalid parameter value',
       context={'estimator': 'dfa', 'parameters': {'H': 1.5}},
       timestamp='2024-01-15T10:30:00'
   )

Data Export and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import AnalyticsDashboard
   import pandas as pd
   import matplotlib.pyplot as plt
   
   dashboard = AnalyticsDashboard()
   
   # Export analytics data
   analytics_data = dashboard.export_analytics_data()
   
   # Convert to pandas DataFrame
   df = pd.DataFrame(analytics_data['usage_events'])
   
   # Create visualizations
   plt.figure(figsize=(12, 8))
   
   # Usage by estimator
   plt.subplot(2, 2, 1)
   estimator_counts = df['estimator_name'].value_counts()
   estimator_counts.plot(kind='bar')
   plt.title('Usage by Estimator')
   plt.xticks(rotation=45)
   
   # Execution time distribution
   plt.subplot(2, 2, 2)
   plt.hist(df['execution_time'], bins=20, alpha=0.7)
   plt.title('Execution Time Distribution')
   plt.xlabel('Time (seconds)')
   
   # Success rate over time
   plt.subplot(2, 2, 3)
   df['date'] = pd.to_datetime(df['timestamp']).dt.date
   success_rate = df.groupby('date')['success'].mean()
   success_rate.plot(kind='line')
   plt.title('Success Rate Over Time')
   plt.ylabel('Success Rate')
   
   # Parameter usage heatmap
   plt.subplot(2, 2, 4)
   # Create heatmap of parameter usage
   plt.title('Parameter Usage Heatmap')
   
   plt.tight_layout()
   plt.show()

Real-time Analytics Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import AnalyticsDashboard
   import time
   import threading
   
   dashboard = AnalyticsDashboard()
   
   def monitor_analytics():
       """Monitor analytics in real-time."""
       while True:
           summary = dashboard.get_comprehensive_summary()
           print("\n" + "="*50)
           print("REAL-TIME ANALYTICS UPDATE")
           print("="*50)
           print(summary)
           time.sleep(60)  # Update every minute
   
   # Start monitoring in background
   monitor_thread = threading.Thread(target=monitor_analytics, daemon=True)
   monitor_thread.start()
   
   # Your analysis code here
   from lrdbenchmark import FBMModel, ComprehensiveBenchmark
   
   for i in range(5):
       model = FBMModel(H=0.5 + i*0.1, sigma=1.0)
       data = model.generate(1000, seed=i)
       
       benchmark = ComprehensiveBenchmark()
       results = benchmark.run_comprehensive_benchmark(
           data_length=1000,
           n_runs=2
       )
       
       time.sleep(30)  # Wait between runs

Analytics Integration with Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import ComprehensiveBenchmark, enable_analytics
   from lrdbench.analytics import AnalyticsDashboard
   
   # Enable analytics
   enable_analytics()
   
   # Create benchmark with analytics integration
   benchmark = ComprehensiveBenchmark()
   
   # Run benchmark with analytics tracking
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       n_runs=10,
       enable_analytics=True  # Enable analytics tracking
   )
   
   # Get analytics dashboard
   dashboard = AnalyticsDashboard()
   
   # Generate integrated report
   integrated_report = dashboard.generate_integrated_report(
       benchmark_results=results,
       include_performance=True,
       include_reliability=True,
       include_workflow=True
   )
   
   print("=== INTEGRATED BENCHMARK & ANALYTICS REPORT ===")
   print(integrated_report)

Best Practices
-------------

1. **Enable Early**: Enable analytics at the start of your analysis
2. **Use Decorators**: Use the provided decorators for automatic tracking
3. **Monitor Performance**: Track execution times for optimization
4. **Error Handling**: Always track errors for debugging
5. **Workflow Analysis**: Track complete workflows for optimization
6. **Regular Reports**: Generate regular analytics reports
7. **Data Export**: Export analytics data for external analysis
8. **Privacy**: Be mindful of sensitive data in analytics

Configuration Options
--------------------

Analytics Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import AnalyticsConfig
   
   # Configure analytics system
   config = AnalyticsConfig(
       # Usage tracking
       track_user_id=True,
       track_parameters=True,
       track_timing=True,
       
       # Performance monitoring
       track_memory=True,
       track_cpu=True,
       track_gpu=True,
       
       # Error analysis
       categorize_errors=True,
       track_stack_traces=True,
       generate_recommendations=True,
       
       # Workflow analysis
       track_step_dependencies=True,
       analyze_patterns=True,
       generate_optimizations=True,
       
       # Data retention
       max_events=10000,
       retention_days=30,
       
       # Privacy
       anonymize_user_ids=True,
       sanitize_parameters=True
   )
   
   # Apply configuration
   from lrdbench.analytics import configure_analytics
   configure_analytics(config)

Privacy and Security
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbench.analytics import UsageTracker
   
   # Create privacy-aware usage tracker
   usage_tracker = UsageTracker(
       track_user_id=False,  # Don't track user IDs
       sanitize_parameters=True,  # Remove sensitive parameters
       anonymize_data=True  # Anonymize all data
   )
   
   # Track usage with privacy protection
   usage_tracker.track_estimator_usage(
       estimator_name='dfa',
       parameters={'min_scale': 4, 'max_scale': 100},  # Will be sanitized
       execution_time=1.23,
       success=True
   )

.. note::
   The analytics system is designed to be privacy-aware and can be configured
   to protect sensitive information while still providing valuable insights.

.. warning::
   When using analytics in production environments, ensure compliance with
   data protection regulations and implement appropriate privacy controls.
