Data Models API
==============

lrdbenchmark provides several synthetic data models for generating time series with known long-range dependence properties.

Base Model
---------

.. autoclass:: lrdbenchmark.models.data_models.base_model.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Brownian Motion (FBM)
-------------------------------

.. autoclass:: lrdbenchmark.models.data_models.fbm.fbm_model.FractionalBrownianMotion
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: generate

Fractional Gaussian Noise (FGN)
------------------------------

.. autoclass:: lrdbenchmark.models.data_models.fgn.fgn_model.FractionalGaussianNoise
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: generate

ARFIMA Model
------------

.. autoclass:: lrdbenchmark.models.data_models.arfima.arfima_model.ARFIMAModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: generate

Multifractal Random Walk (MRW)
-----------------------------

.. autoclass:: lrdbenchmark.models.data_models.mrw.mrw_model.MultifractalRandomWalk
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: generate

Convenience Aliases
------------------

For easier usage, lrdbenchmark provides shortened aliases for all data models:

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel
   
   # These are equivalent to the full class names
   fbm = FBMModel(H=0.7, sigma=1.0)
   fgn = FGNModel(H=0.6, sigma=1.0)
   arfima = ARFIMAModel(d=0.3, sigma=1.0)
   mrw = MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)

Convenience Functions
--------------------

.. autofunction:: lrdbenchmark.models.data_models.create_fbm_model
.. autofunction:: lrdbenchmark.models.data_models.create_fgn_model
.. autofunction:: lrdbenchmark.models.data_models.create_arfima_model
.. autofunction:: lrdbenchmark.models.data_models.create_mrw_model

Usage Examples
-------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel
   
   # Generate FBM data
   fbm_model = FBMModel(H=0.7, sigma=1.0)
   fbm_data = fbm_model.generate(1000, seed=42)
   
   # Generate FGN data
   fgn_model = FGNModel(H=0.6, sigma=1.0)
   fgn_data = fgn_model.generate(1000, seed=42)

Multiple Realizations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import FBMModel
   import numpy as np
   
   # Generate multiple realizations
   model = FBMModel(H=0.7, sigma=1.0)
   realizations = []
   
   for i in range(10):
       data = model.generate(1000, seed=i)
       realizations.append(data)
   
   realizations = np.array(realizations)

Parameter Sweeps
~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import FBMModel
   import matplotlib.pyplot as plt
   
   # Generate data with different H values
   H_values = [0.3, 0.5, 0.7, 0.9]
   datasets = {}
   
   for H in H_values:
       model = FBMModel(H=H, sigma=1.0)
       datasets[f'H={H}'] = model.generate(1000, seed=42)
   
   # Plot results
   plt.figure(figsize=(12, 8))
   for name, data in datasets.items():
       plt.plot(data[:200], label=name, alpha=0.7)
   
   plt.title('FBM with Different H Values')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.show()

Model Comparison
~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark import FBMModel, FGNModel, ARFIMAModel, MRWModel
   import matplotlib.pyplot as plt
   
   # Generate data from different models
   models = {
       'FBM': FBMModel(H=0.7, sigma=1.0),
       'FGN': FGNModel(H=0.7, sigma=1.0),
       'ARFIMA': ARFIMAModel(d=0.3, sigma=1.0),
       'MRW': MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
   }
   
   datasets = {}
   for name, model in models.items():
       datasets[name] = model.generate(1000, seed=42)
   
   # Plot comparison
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   axes = axes.flatten()
   
   for i, (name, data) in enumerate(datasets.items()):
       axes[i].plot(data[:200])
       axes[i].set_title(name)
       axes[i].grid(True)
   
   plt.tight_layout()
   plt.show()

Error Handling
-------------

.. code-block:: python

   from lrdbenchmark import FBMModel
   
   try:
       # Invalid H value
       model = FBMModel(H=1.5, sigma=1.0)
   except ValueError as e:
       print(f"Error: {e}")
   
   try:
       # Invalid sigma value
       model = FBMModel(H=0.7, sigma=-1.0)
   except ValueError as e:
       print(f"Error: {e}")

Performance Considerations
-------------------------

* **Memory Usage**: Large datasets may require significant memory
* **GPU Acceleration**: Some models support GPU acceleration when available
* **Parallel Generation**: Use multiple processes for generating many realizations
* **Seed Management**: Use different seeds for independent realizations

.. note::
   All models generate numpy arrays by default. For GPU acceleration,
   the data can be converted to the appropriate tensor format.

.. warning::
   Very large datasets (>1M samples) may cause memory issues.
   Consider generating data in chunks or using streaming approaches.
