
# LRDBench Complete Estimators Test and Benchmark Report

**Generated:** 2025-08-28 15:57:29
**Timestamp:** 20250828_155729

## 📊 Executive Summary

### Individual Estimators
- **Total Tested:** 12
- **Successful:** 12
- **Success Rate:** 100.0%
- **Average Execution Time:** 0.0755s

### Auto-Optimized Estimators
- **Total Tested:** 7
- **Successful:** 7
- **Success Rate:** 100.0%
- **Average Execution Time:** 0.2000s

## 🔍 Individual Estimator Results

- **R/S:** ✅ PASS | Time: 0.1026s | H_est: 0.735081
- **DFA:** ✅ PASS | Time: 0.0423s | H_est: 0.707277
- **DMA:** ✅ PASS | Time: 0.0717s | H_est: 0.632912
- **Higuchi:** ✅ PASS | Time: 0.0135s | H_est: 0.663961
- **GPH:** ✅ PASS | Time: 0.0075s | H_est: 0.658517
- **Whittle:** ✅ PASS | Time: 0.0030s | H_est: 0.626313
- **Periodogram:** ✅ PASS | Time: 0.0020s | H_est: 0.509250
- **CWT:** ✅ PASS | Time: 0.1664s | H_est: 0.556552
- **Wavelet Variance:** ✅ PASS | Time: 0.0082s | H_est: 0.836094
- **Wavelet Log Variance:** ✅ PASS | Time: 0.0026s | H_est: 0.732963
- **Wavelet Whittle:** ✅ PASS | Time: 0.0125s | H_est: 0.476167
- **MFDFA:** ✅ PASS | Time: 0.4734s | H_est: 0.096780

## ⚡ Auto-Optimized Estimator Results

- **Auto DFA:** ✅ PASS | Time: 0.0805s | H_est: 0.886658 | Opt: SciPy
- **Auto RS:** ✅ PASS | Time: 0.0473s | H_est: 0.812911 | Opt: SciPy
- **Auto DMA:** ✅ PASS | Time: 0.0022s | H_est: 0.775838 | Opt: NUMBA
- **Auto Higuchi:** ✅ PASS | Time: 0.0134s | H_est: 0.792703 | Opt: NUMBA
- **Auto GPH:** ✅ PASS | Time: 1.0147s | H_est: 0.875543 | Opt: NUMBA
- **Auto Periodogram:** ✅ PASS | Time: 0.1185s | H_est: 0.875543 | Opt: NUMBA
- **Auto Whittle:** ✅ PASS | Time: 0.1235s | H_est: 0.100000 | Opt: NUMBA

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
