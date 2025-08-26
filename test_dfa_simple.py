#!/usr/bin/env python3
"""Simple DFA test."""

import time
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbench.analysis.temporal.dfa.dfa_estimator_numba_optimized import NumbaOptimizedDFAEstimator

print("Testing DFA NUMBA estimator...")

try:
    # Generate test data
    fgn = FractionalGaussianNoise(H=0.7)
    data = fgn.generate(1000, seed=42)
    
    # Test estimator
    estimator = NumbaOptimizedDFAEstimator()
    
    start_time = time.time()
    result = estimator.estimate(data)
    execution_time = time.time() - start_time
    
    print(f"✅ Success!")
    print(f"Execution Time: {execution_time:.4f}s")
    print(f"Hurst Parameter: {result['hurst_parameter']:.6f}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
