.. LRDBench documentation master file, created by
   sphinx-quickstart on Sun Aug 25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LRDBench's documentation!
====================================

.. image:: https://img.shields.io/pypi/v/lrdbench.svg
   :target: https://pypi.org/project/lrdbench/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/lrdbench.svg
   :target: https://pypi.org/project/lrdbench/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/lrdbench/badge/?version=latest
   :target: https://lrdbench.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**LRDBench** is a comprehensive benchmarking framework for long-range dependence (LRD) analysis in time series data. It provides a unified platform for evaluating and comparing various estimators and models for detecting and quantifying long-range dependence patterns.

Key Features
-----------

* **Comprehensive Estimator Suite**: Classical, machine learning, and neural network estimators
* **Multiple Data Models**: FBM, FGN, ARFIMA, MRW with configurable parameters
* **High Performance**: GPU-accelerated implementations with JAX and PyTorch backends
* **Analytics System**: Built-in usage tracking and performance monitoring
* **Extensible Architecture**: Easy integration of new estimators and models
* **Production Ready**: Pre-trained models for deployment

Quick Start
----------

Install LRDBench:

.. code-block:: bash

   pip install lrdbench

Basic usage:

.. code-block:: python

   from lrdbench import FBMModel, ComprehensiveBenchmark
   
   # Generate synthetic data
   model = FBMModel(H=0.7, sigma=1.0)
   data = model.generate(1000, seed=42)
   
   # Run comprehensive benchmark
   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(data_length=1000)
   
   print(results)

Installation & Setup
-------------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart

API Reference
------------

.. toctree::
   :maxdepth: 2

   api/data_models
   api/estimators
   api/benchmark
   api/analytics

Research & Theory
----------------

.. toctree::
   :maxdepth: 2

   research/theory
   research/validation

Examples & Demos
---------------

.. toctree::
   :maxdepth: 2

   examples/comprehensive_demo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:

   installation
   quickstart
   api/data_models
   api/estimators
   api/benchmark
   api/analytics
   research/theory
   research/validation
   examples/comprehensive_demo
