# Advanced Metrics Benchmark Report
## Convergence Rates and Mean Signed Error Analysis

**Date**: August 27, 2025  
**Benchmark Type**: Advanced Classical Estimators  
**Data Models**: fBm, fGn, MRW (ARFIMA skipped due to no true H value)  
**Estimators Tested**: 13 classical estimators  
**Monte Carlo Simulations**: 30 per estimator  
**Success Rate**: 100% (39/39 tests successful)

---

## 🎯 Executive Summary

The advanced metrics benchmark successfully demonstrated the new convergence rates and mean signed error functionality in LRDBench. All 13 classical estimators were tested across 3 data models with comprehensive performance profiling.

### Key Findings:
- **GPH estimator** achieved the highest comprehensive score (0.9821)
- **DMA estimator** showed excellent performance (0.9651)
- **R/S estimator** demonstrated strong reliability (0.9586)
- Most estimators showed **significant bias** in their estimates
- **Convergence rates** varied significantly across estimators and data models

---

## 📊 Detailed Performance Analysis

### Top 5 Performers by Comprehensive Score

| Rank | Estimator | Comprehensive Score | Data Models Tested |
|------|-----------|-------------------|-------------------|
| 1 | GPH | 0.9821 | 3 |
| 2 | DMA | 0.9651 | 3 |
| 3 | R/S | 0.9586 | 3 |
| 4 | DFA | 0.9439 | 3 |
| 5 | Periodogram | 0.9365 | 3 |

### Convergence Analysis Results

#### Convergence Achievement
- **GPH**: Achieved convergence in 2/3 data models
- **Whittle**: Achieved convergence in 2/3 data models  
- **WaveletWhittle**: Achieved convergence in 2/3 data models
- **Other estimators**: No convergence achieved within threshold

#### Convergence Rate Statistics
- **Best convergence rate**: GPH on fBm (0.233)
- **Worst convergence rate**: R/S on fBm (-1.611)
- **Average convergence rate**: -0.847 (negative indicates improvement with data size)

### Mean Signed Error Analysis

#### Bias Assessment
- **Most estimators showed significant bias** (p < 0.05)
- **Bias percentages ranged from -98.8% to +74.0%**
- **Higuchi estimator** consistently showed the largest negative bias
- **WaveletVar on fGn** showed the largest positive bias (+74.0%)

#### Bias Patterns by Data Model

**fBm Data Model:**
- **Least biased**: DFA (0.26% bias)
- **Most biased**: Higuchi (-98.8% bias)
- **Average bias**: -15.2%

**fGn Data Model:**
- **Least biased**: Whittle (5.38% bias)
- **Most biased**: WaveletVar (+74.0% bias)
- **Average bias**: +8.5%

**MRW Data Model:**
- **Least biased**: R/S (-1.35% bias)
- **Most biased**: Higuchi (-93.8% bias)
- **Average bias**: -12.1%

---

## 🔍 Detailed Estimator Analysis

### 1. GPH Estimator (Best Overall)
- **Comprehensive Score**: 0.9821
- **Convergence**: Achieved in 2/3 models
- **Bias Range**: -5.9% to +21.1%
- **Execution Time**: Very fast (0.001-0.004s)
- **Strengths**: Excellent convergence, low bias, fast execution
- **Weaknesses**: Moderate bias on fGn data

### 2. DMA Estimator (Second Best)
- **Comprehensive Score**: 0.9651
- **Convergence**: Not achieved in any model
- **Bias Range**: -9.6% to +10.5%
- **Execution Time**: Moderate (0.074-0.076s)
- **Strengths**: Consistent low bias across all models
- **Weaknesses**: No convergence achieved

### 3. R/S Estimator (Third Best)
- **Comprehensive Score**: 0.9586
- **Convergence**: Not achieved in any model
- **Bias Range**: -1.4% to +17.7%
- **Execution Time**: Fast (0.027-0.034s)
- **Strengths**: Very low bias on MRW, consistent performance
- **Weaknesses**: Poor convergence rates

### 4. DFA Estimator (Fourth Best)
- **Comprehensive Score**: 0.9439
- **Convergence**: Not achieved in any model
- **Bias Range**: +0.26% to +24.0%
- **Execution Time**: Moderate (0.080-0.086s)
- **Strengths**: Excellent bias control on fBm
- **Weaknesses**: High positive bias on fGn

### 5. Periodogram Estimator (Fifth Best)
- **Comprehensive Score**: 0.9365
- **Convergence**: Not achieved in any model
- **Bias Range**: -27.2% to +8.8%
- **Execution Time**: Very fast (0.001-0.002s)
- **Strengths**: Fast execution, moderate bias
- **Weaknesses**: High negative bias on fBm

---

## 📈 Performance Insights

### Convergence Rate Analysis
1. **GPH estimator** shows the best convergence behavior with positive rates indicating improvement with data size
2. **Most estimators** have negative convergence rates, suggesting they may not improve significantly with larger datasets
3. **Convergence achievement** is rare, indicating most estimators need more data or different parameters

### Bias Analysis
1. **Systematic bias** is present in most estimators
2. **Data model dependency**: Bias patterns vary significantly across data models
3. **Estimator-specific bias**: Some estimators (Higuchi, MFDFA) consistently show large negative bias
4. **Bias magnitude**: Ranges from negligible (0.26%) to severe (-98.8%)

### Stability Analysis
1. **Stability metrics** range from 0.0 (perfect stability) to 1.2 (high instability)
2. **WaveletWhittle** shows perfect stability (0.0) when convergence is achieved
3. **Higuchi** shows the highest instability (0.95-1.21)

---

## 🎯 Recommendations

### For Researchers
1. **Use GPH estimator** for applications requiring both accuracy and convergence
2. **Consider DMA estimator** for applications where bias consistency is critical
3. **Use R/S estimator** for MRW data due to its low bias
4. **Avoid Higuchi estimator** due to severe systematic bias

### For System Design
1. **Implement convergence monitoring** to detect when estimators stabilize
2. **Use bias correction** for estimators with known systematic bias
3. **Consider ensemble methods** combining multiple estimators
4. **Monitor stability metrics** to assess estimator reliability

### For Future Development
1. **Improve convergence** for estimators with poor convergence rates
2. **Develop bias correction methods** for estimators with systematic bias
3. **Optimize parameters** based on convergence and bias analysis
4. **Extend testing** to more data models and contamination scenarios

---

## 📋 Technical Notes

### Methodology
- **Convergence analysis**: Uses log-log regression on subset estimates
- **Bias analysis**: Monte Carlo simulations with 30 iterations
- **Stability metric**: Coefficient of variation of estimates
- **Comprehensive score**: Combines accuracy, speed, convergence, and bias

### Data Quality
- **100% success rate** indicates robust implementation
- **Warning messages** for spectral estimators on small subsets are expected
- **Boundary effects** in wavelet estimators are normal for short data

### Limitations
- **ARFIMA model** excluded due to no true H parameter
- **Short data subsets** may affect convergence analysis
- **Monte Carlo simulations** limited to 30 iterations for speed

---

## 🚀 Conclusion

The advanced metrics benchmark successfully demonstrates the new convergence rates and mean signed error functionality in LRDBench. The analysis reveals important insights about estimator performance, bias patterns, and convergence behavior that were not previously available.

**Key achievements:**
- ✅ Successfully integrated convergence rate analysis
- ✅ Successfully integrated mean signed error analysis  
- ✅ Comprehensive performance profiling working
- ✅ 100% test success rate
- ✅ Detailed bias and convergence insights

The new metrics provide researchers with deeper understanding of estimator behavior, enabling more informed selection and potential bias correction strategies.
