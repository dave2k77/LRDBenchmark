Installation Guide
=================

This guide will help you install LRDBench and its dependencies.

Requirements
-----------

* Python 3.8 or higher
* pip (Python package installer)
* Optional: CUDA-compatible GPU for accelerated computations

Basic Installation
-----------------

Install LRDBench from PyPI:

.. code-block:: bash

   pip install lrdbench

This will install LRDBench with all required dependencies.

Installation with Optional Dependencies
-------------------------------------

For GPU acceleration and additional features:

.. code-block:: bash

   # Install with GPU support (PyTorch + CUDA)
   pip install lrdbench[gpu]
   
   # Install with JAX backend
   pip install lrdbench[jax]
   
   # Install with all optional dependencies
   pip install lrdbench[all]

Development Installation
-----------------------

To install LRDBench in development mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-username/lrdbench.git
   cd lrdbench
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

Conda Installation
-----------------

Using conda:

.. code-block:: bash

   # Create a new conda environment
   conda create -n lrdbench python=3.9
   conda activate lrdbench
   
   # Install LRDBench
   pip install lrdbench

Docker Installation
------------------

Pull the official LRDBench Docker image:

.. code-block:: bash

   docker pull lrdbench/lrdbench:latest
   
   # Run with GPU support
   docker run --gpus all -it lrdbench/lrdbench:latest

Or build from Dockerfile:

.. code-block:: bash

   git clone https://github.com/your-username/lrdbench.git
   cd lrdbench
   docker build -t lrdbench .
   docker run -it lrdbench

Verification
-----------

After installation, verify that LRDBench is working correctly:

.. code-block:: python

   import lrdbench
   print(f"LRDBench version: {lrdbench.__version__}")
   
   # Test basic functionality
   from lrdbench import FBMModel
   model = FBMModel(H=0.7)
   data = model.generate(100)
   print(f"Generated {len(data)} samples")

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch separately: ``pip install torch``

**CUDA not found**
   Install CUDA toolkit or use CPU-only version: ``pip install lrdbench[cpu]``

**JAX installation issues**
   On Windows, JAX may require special installation. See `JAX installation guide <https://github.com/google/jax#installation>`_.

**Memory issues with large datasets**
   Consider using smaller batch sizes or reducing data length in benchmarks.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance:

1. **Use GPU acceleration** when available
2. **Install optimized BLAS libraries** (Intel MKL, OpenBLAS)
3. **Enable JIT compilation** for JAX backends
4. **Use appropriate batch sizes** for your hardware

Environment Variables
~~~~~~~~~~~~~~~~~~~~

Set these environment variables for optimal performance:

.. code-block:: bash

   # Enable JAX optimizations
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_ALLOCATOR=platform
   
   # PyTorch optimizations
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

Next Steps
----------

After successful installation, proceed to:

* :doc:`quickstart` - Get started with LRDBench
* :doc:`user_guide/getting_started` - Detailed getting started guide
* :doc:`user_guide/examples` - Example notebooks and scripts
