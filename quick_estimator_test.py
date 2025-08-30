#!/usr/bin/env python3
"""
Quick test script to verify all estimators are working correctly.
"""

import numpy as np
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion

# Import all unified estimators
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.temporal.higuchi.higuchi_estimator_unified import HiguchiEstimator
from lrdbenchmark.analysis.temporal.dma.dma_estimator_unified import DMAEstimator

from lrdbenchmark.analysis.spectral.gph.gph_estimator_unified import GPHEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator
from lrdbenchmark.analysis.spectral.periodogram.periodogram_estimator_unified import PeriodogramEstimator

from lrdbenchmark.analysis.wavelet.variance.variance_estimator_unified import WaveletVarianceEstimator
from lrdbenchmark.analysis.wavelet.whittle.whittle_estimator_unified import WaveletWhittleEstimator
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator
from lrdbenchmark.analysis.wavelet.log_variance.log_variance_estimator_unified import WaveletLogVarianceEstimator

from lrdbenchmark.analysis.multifractal.mfdfa.mfdfa_estimator_unified import MFDFAEstimator
from lrdbenchmark.analysis.multifractal.wavelet_leaders.wavelet_leaders_estimator_unified import MultifractalWaveletLeadersEstimator

def test_estimator(estimator_class, estimator_name, data, hurst):
    """Test a single estimator."""
    try:
        estimator = estimator_class(use_optimization='numpy')
        result = estimator.estimate(data)
        
        if result and 'hurst_parameter' in result:
            hurst_est = result['hurst_parameter']
            method = result.get('method', 'unknown')
            print(f"✅ {estimator_name}: H_est={hurst_est:.3f}, Method={method}")
            return True
        else:
            print(f"❌ {estimator_name}: No valid result")
            return False
            
    except Exception as e:
        print(f"❌ {estimator_name}: Error - {str(e)}")
        return False

def main():
    """Test all estimators."""
    print("🚀 Quick Estimator Test - Verifying All Estimators")
    print("=" * 60)
    
    # Generate test data
    fbm = FractionalBrownianMotion(H=0.7)
    data = fbm.generate(1000)
    print(f"📊 Generated test data: {len(data)} points, H=0.7")
    print()
    
    # Test all estimators
    estimators = [
        (RSEstimator, "R/S Estimator"),
        (DFAEstimator, "DFA Estimator"),
        (HiguchiEstimator, "Higuchi Estimator"),
        (DMAEstimator, "DMA Estimator"),
        (GPHEstimator, "GPH Estimator"),
        (WhittleEstimator, "Whittle Estimator"),
        (PeriodogramEstimator, "Periodogram Estimator"),
        (WaveletVarianceEstimator, "Wavelet Variance Estimator"),
        (WaveletWhittleEstimator, "Wavelet Whittle Estimator"),
        (CWTEstimator, "CWT Estimator"),
        (WaveletLogVarianceEstimator, "Log Variance Estimator"),
        (MFDFAEstimator, "MFDFA Estimator"),
        (MultifractalWaveletLeadersEstimator, "Wavelet Leaders Estimator"),
    ]
    
    successful = 0
    total = len(estimators)
    
    for estimator_class, name in estimators:
        if test_estimator(estimator_class, name, data, 0.7):
            successful += 1
    
    print()
    print("=" * 60)
    print(f"📈 Test Results: {successful}/{total} estimators working ({successful/total*100:.1f}%)")
    
    if successful == total:
        print("🎉 All estimators are working correctly!")
    else:
        print("⚠️ Some estimators need attention.")

if __name__ == "__main__":
    main()
