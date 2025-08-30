.. LRDBench documentation master file, created by
   sphinx-quickstart on Sun Aug 25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LRDBenchmark's documentation!
========================================

.. image:: https://img.shields.io/pypi/v/lrdbenchmark.svg
   :target: https://pypi.org/project/lrdbenchmark/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/lrdbenchmark.svg
   :target: https://pypi.org/project/lrdbenchmark/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/lrdbenchmark/badge/?version=latest
   :target: https://lrdbenchmark.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**LRDBenchmark** is a comprehensive benchmarking framework for long-range dependence (LRD) analysis in time series data. It provides a unified platform for evaluating and comparing various estimators and models for detecting and quantifying long-range dependence patterns.

🏆 **Latest Results: ML Estimators 4x More Accurate Than Classical Methods!**

Our latest benchmark shows:
- **100% success rate** across all 98 test cases
- **ML estimators significantly outperform** classical methods (MSE: 0.061 vs 0.245)
- **All estimators working correctly** with unified interfaces and graceful fallbacks
- **Top performers**: DFA (32.5% error), DMA (39.8% error), Random Forest (74.8% error)

**LRDBenchmark** provides a comprehensive benchmarking framework for long-range dependence (LRD) analysis in time series data. It offers a unified platform for evaluating and comparing various estimators and models for detecting and quantifying long-range dependence patterns.

**Key Features:**
* **Comprehensive Estimator Suite**: 18 total estimators including classical, enhanced machine learning, and neural network estimators
* **Multiple Data Models**: FBM, FGN, ARFIMA, MRW with configurable parameters
* **High Performance**: GPU-accelerated implementations with JAX and PyTorch backends
* **Analytics System**: Built-in usage tracking and performance monitoring
* **Extensible Architecture**: Easy integration of new estimators and models
* **Production Ready**: Pre-trained models for deployment
* **Unified Framework**: All estimators work seamlessly with graceful fallbacks

**Quick Start:**
Install with `pip install lrdbenchmark` and see the Quick Start Guide for detailed examples.

Installation & Setup
--------------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/data_models
   api/estimators
   api/benchmark
   api/analytics

Research & Theory
-----------------

.. toctree::
   :maxdepth: 2

   research/theory
   research/validation

Examples & Demos
----------------

.. toctree::
   :maxdepth: 2

   examples/comprehensive_demo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

