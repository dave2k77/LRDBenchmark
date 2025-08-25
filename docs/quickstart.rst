Quick Start Guide
================

This guide will get you up and running with LRDBench in minutes.

Basic Usage
----------

Generate synthetic data and run a benchmark:

.. code-block:: python

   import numpy as np
   from lrdbench import FBMModel, ComprehensiveBenchmark
   
   # Generate Fractional Brownian Motion data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # Run comprehensive benchmark
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000)
   
   print("Benchmark completed!")
   print(f"Results: {results}")

Data Models
----------

LRDBench provides several synthetic data models:

.. code-block:: python

   from lrdbench import FBMModel, FGNModel, ARFIMAModel, MRWModel
   
   # Fractional Brownian Motion
   fbm = FBMModel(H=0.7, sigma=1.0)
   fbm_data = fbm.generate(1000)
   
   # Fractional Gaussian Noise
   fgn = FGNModel(H=0.6, sigma=1.0)
   fgn_data = fgn.generate(1000)
   
   # ARFIMA process
   arfima = ARFIMAModel(d=0.3, sigma=1.0)
   arfima_data = arfima.generate(1000)
   
   # Multifractal Random Walk
   mrw = MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
   mrw_data = mrw.generate(1000)

Individual Estimators
--------------------

Use specific estimators directly:

.. code-block:: python

   from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
   from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
   
   # Detrended Fluctuation Analysis
   dfa = DFAEstimator()
   H_dfa = dfa.estimate(data)
   
   # Geweke-Porter-Hudak estimator
   gph = GPHEstimator()
   H_gph = gph.estimate(data)
   
   print(f"DFA H estimate: {H_dfa:.3f}")
   print(f"GPH H estimate: {H_gph:.3f}")

Analytics System
---------------

Track usage and performance:

.. code-block:: python

   from lrdbench import enable_analytics, get_analytics_summary
   
   # Enable analytics
   enable_analytics()
   
   # Run your analysis
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000)
   
   # Get analytics summary
   summary = get_analytics_summary()
   print(summary)

Advanced Usage
-------------

Custom benchmark configuration:

.. code-block:: python

   from lrdbench import ComprehensiveBenchmark
   
   # Configure benchmark
   benchmark = ComprehensiveBenchmark()
   
   # Run with specific parameters
   results = benchmark.run_comprehensive_benchmark(
       data_length=2000,
       estimators=['dfa', 'gph', 'rs'],
       data_models=['fbm', 'fgn'],
       n_runs=5
   )
   
   # Access detailed results
   for estimator, result in results.items():
       print(f"{estimator}: H={result['estimated_H']:.3f}")

Integration with HPFracc
-----------------------

Compare with fractional neural networks:

.. code-block:: python

   # This requires hpfracc to be installed
   try:
       from scripts.hpfracc_proper_benchmark import HPFraccProperBenchmark
       
       # Create benchmark
       benchmark = HPFraccProperBenchmark(
           series_length=1000,
           batch_size=32,
           input_window=10,
           prediction_horizon=1
       )
       
       # Run comparison
       results = benchmark.run_benchmark()
       
       # Generate report
       report = benchmark.generate_report()
       print(report)
       
   except ImportError:
       print("HPFracc not available. Install with: pip install hpfracc")

Visualization
------------

Plot results and data:

.. code-block:: python

   import matplotlib.pyplot as plt
   from lrdbench import FBMModel
   
   # Generate data with different H values
   H_values = [0.3, 0.5, 0.7, 0.9]
   datasets = {}
   
   for H in H_values:
       model = FBMModel(H=H, sigma=1.0)
       datasets[f'H={H}'] = model.generate(1000)
   
   # Plot
   plt.figure(figsize=(12, 8))
   for name, data in datasets.items():
       plt.plot(data[:200], label=name, alpha=0.7)
   
   plt.title('Fractional Brownian Motion with Different H Values')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.show()

Performance Tips
---------------

1. **Use GPU acceleration** when available
2. **Batch processing** for large datasets
3. **Enable analytics** for monitoring
4. **Use appropriate data lengths** (1000+ samples recommended)

Next Steps
----------

* :doc:`user_guide/getting_started` - Detailed getting started guide
* :doc:`user_guide/data_models` - Learn about data models
* :doc:`user_guide/estimators` - Explore available estimators
* :doc:`user_guide/examples` - More examples and use cases
