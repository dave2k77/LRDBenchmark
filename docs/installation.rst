Installation Guide
=================

This guide will help you install lrdbenchmark and its dependencies.

Requirements
------------

* Python 3.8 or higher
* pip (Python package installer)
* Optional: CUDA-compatible GPU for accelerated computations

Basic Installation
------------------

Install lrdbenchmark from PyPI:

.. code-block:: bash

   pip install lrdbenchmark

This will install lrdbenchmark with all required dependencies including enhanced ML and neural network estimators.

Installation with Optional Dependencies
--------------------------------------

For GPU acceleration and additional features:

.. code-block:: bash

   # Install with dashboard support (Streamlit + Plotly)
   pip install lrdbenchmark[dashboard]
   
   # Install with development dependencies
   pip install lrdbenchmark[dev]
   
   # Install with documentation dependencies
   pip install lrdbenchmark[docs]

Development Installation
------------------------

To install LRDBenchmark in development mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dave2k77/LRDBenchmark.git
   cd LRDBenchmark
   
   # Install in development mode
   pip install -e .
   
   # Install with dashboard support
   pip install -e .[dashboard]
   
   # Install development dependencies
   pip install -r requirements-dev.txt

Conda Installation
------------------

Using conda:

.. code-block:: bash

   # Create a new conda environment
   conda create -n lrdbenchmark python=3.9
   conda activate lrdbenchmark
   
   # Install lrdbenchmark
   pip install lrdbenchmark

Docker Installation
-------------------

Pull the official LRDBenchmark Docker image:

.. code-block:: bash

   docker pull lrdbenchmark/lrdbenchmark:latest
   
   # Run with GPU support
   docker run --gpus all -it lrdbenchmark/lrdbenchmark:latest

Or build from Dockerfile:

.. code-block:: bash

   git clone https://github.com/dave2k77/LRDBenchmark.git
   cd LRDBenchmark
   docker build -t lrdbenchmark .
   docker run -it lrdbenchmark

Verification
------------

After installation, verify that LRDBenchmark is working correctly:

.. code-block:: python

   import lrdbenchmark
   print(f"lrdbenchmark version: {lrdbenchmark.__version__}")
   
   # Test enhanced ML and neural network estimators
   from lrdbenchmark import CNNEstimator, LSTMEstimator
   print("Enhanced estimators imported successfully!")
   
   # Test basic functionality
   from lrdbenchmark import FBMModel
   model = FBMModel(H=0.7)
   data = model.generate(100)
   print(f"Generated {len(data)} samples")

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch separately: ``pip install torch``

**CUDA not found**
   Install CUDA toolkit or use CPU-only version: ``pip install lrdbenchmark[cpu]``

**JAX installation issues**
   On Windows, JAX may require special installation. See `JAX installation guide <https://github.com/google/jax#installation>`_.

**Memory issues with large datasets**
   Consider using smaller batch sizes or reducing data length in benchmarks.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance:

1. **Use GPU acceleration** when available
2. **Install optimized BLAS libraries** (Intel MKL, OpenBLAS)
3. **Enable JIT compilation** for JAX backends
4. **Use appropriate batch sizes** for your hardware

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Set these environment variables for optimal performance:

.. code-block:: bash

   # Enable JAX optimizations
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_ALLOCATOR=platform
   
   # PyTorch optimizations
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

Next Steps
-----------

After successful installation, proceed to:

* :doc:`quickstart` - Get started with LRDBench
* :doc:`user_guide/getting_started` - Detailed getting started guide
* :doc:`user_guide/examples` - Example notebooks and scripts
