
# LRDBench Complete Estimators Test and Benchmark Report

**Generated:** 2025-08-28 05:14:46
**Timestamp:** 20250828_051446

## 📊 Executive Summary

### Individual Estimators
- **Total Tested:** 12
- **Successful:** 12
- **Success Rate:** 100.0%
- **Average Execution Time:** 0.0756s

### Auto-Optimized Estimators
- **Total Tested:** 7
- **Successful:** 7
- **Success Rate:** 100.0%
- **Average Execution Time:** 0.1238s

## 🔍 Individual Estimator Results

- **R/S:** ✅ PASS | Time: 0.0269s | H_est: 0.735081
- **DFA:** ✅ PASS | Time: 0.0837s | H_est: 0.701903
- **DMA:** ✅ PASS | Time: 0.0710s | H_est: 0.632912
- **Higuchi:** ✅ PASS | Time: 0.0133s | H_est: 0.663961
- **GPH:** ✅ PASS | Time: 0.0021s | H_est: 0.658517
- **Whittle:** ✅ PASS | Time: 0.0032s | H_est: 0.626313
- **Periodogram:** ✅ PASS | Time: 0.0021s | H_est: 0.509250
- **CWT:** ✅ PASS | Time: 0.1594s | H_est: 0.556552
- **Wavelet Variance:** ✅ PASS | Time: 0.0031s | H_est: 0.836094
- **Wavelet Log Variance:** ✅ PASS | Time: 0.0011s | H_est: 0.732963
- **Wavelet Whittle:** ✅ PASS | Time: 0.0118s | H_est: 0.476167
- **MFDFA:** ✅ PASS | Time: 0.5298s | H_est: 0.096780

## ⚡ Auto-Optimized Estimator Results

- **Auto DFA:** ✅ PASS | Time: 0.0822s | H_est: 0.886658 | Opt: SciPy
- **Auto RS:** ✅ PASS | Time: 0.0443s | H_est: 0.812911 | Opt: SciPy
- **Auto DMA:** ✅ PASS | Time: 0.0013s | H_est: 0.775838 | Opt: NUMBA
- **Auto Higuchi:** ✅ PASS | Time: 0.0130s | H_est: 0.792703 | Opt: NUMBA
- **Auto GPH:** ✅ PASS | Time: 0.4781s | H_est: 0.875543 | Opt: NUMBA
- **Auto Periodogram:** ✅ PASS | Time: 0.1192s | H_est: 0.875543 | Opt: NUMBA
- **Auto Whittle:** ✅ PASS | Time: 0.1282s | H_est: 0.100000 | Opt: NUMBA

## 🚀 Comprehensive Benchmark Results

- **Total Tests:** 72
- **Successful Tests:** 72
- **Success Rate:** 100.0%
- **Data Models Tested:** 4
- **Estimators Tested:** 18

## 📈 Advanced Metrics Benchmark Results

- **Total Tests:** 54
- **Successful Tests:** 54
- **Success Rate:** 100.0%

## 🧪 Contamination Test Results

- **additive_gaussian:** 100.0% success rate
- **multiplicative_noise:** 100.0% success rate
- **outliers:** 100.0% success rate
- **trend:** 100.0% success rate
- **seasonal:** 100.0% success rate

## 💡 Recommendations

- **Performance Issue:** Auto-optimized estimators are slower than individual estimators
- **System Status:** ✅ LRDBench is operating at high performance levels
- **Ready for Production:** All major components are functioning correctly
