tfr
# LRDBench Complete Estimators Test and Benchmark Report

**Generated:** 2025-08-28 03:48:49
**Timestamp:** 20250828_034849

## 📊 Executive Summary

### Individual Estimators
- **Total Tested:** 12
- **Successful:** 12
- **Success Rate:** 100.0%
- **Average Execution Time:** 0.0768s

### Auto-Optimized Estimators
- **Total Tested:** 7
- **Successful:** 6
- **Success Rate:** 85.7%
- **Average Execution Time:** 0.0804s

## 🔍 Individual Estimator Results

- **R/S:** ✅ PASS | Time: 0.0260s | H_est: 0.735081
- **DFA:** ✅ PASS | Time: 0.0806s | H_est: 0.701903
- **DMA:** ✅ PASS | Time: 0.0715s | H_est: 0.632912
- **Higuchi:** ✅ PASS | Time: 0.0071s | H_est: 0.008536
- **GPH:** ✅ PASS | Time: 0.0019s | H_est: 0.658517
- **Whittle:** ✅ PASS | Time: 0.0071s | H_est: 0.990000
- **Periodogram:** ✅ PASS | Time: 0.0015s | H_est: 0.509250
- **CWT:** ✅ PASS | Time: 0.1673s | H_est: 0.556552
- **Wavelet Variance:** ✅ PASS | Time: 0.0028s | H_est: 0.836094
- **Wavelet Log Variance:** ✅ PASS | Time: 0.0013s | H_est: 0.732963
- **Wavelet Whittle:** ✅ PASS | Time: 0.0114s | H_est: 0.476167
- **MFDFA:** ✅ PASS | Time: 0.5431s | H_est: 0.096780

## ⚡ Auto-Optimized Estimator Results

- **Auto DFA:** ✅ PASS | Time: 0.0795s | H_est: 0.886658 | Opt: SciPy
- **Auto RS:** ✅ PASS | Time: 0.0431s | H_est: 0.812911 | Opt: SciPy
- **Auto DMA:** ✅ PASS | Time: 0.0015s | H_est: 0.775838 | Opt: NUMBA
- **Auto Higuchi:** ❌ FAIL | Time: N/A | H_est: N/A | Opt: N/A
  - Error: unexpected cycle in lookup()
- **Auto GPH:** ✅ PASS | Time: 0.1108s | H_est: 0.875543 | Opt: NUMBA
- **Auto Periodogram:** ✅ PASS | Time: 0.1244s | H_est: 0.875543 | Opt: NUMBA
- **Auto Whittle:** ✅ PASS | Time: 0.1231s | H_est: 0.100000 | Opt: NUMBA

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

- **Fix Auto-Optimized Estimators:** Auto Higuchi failed tests
