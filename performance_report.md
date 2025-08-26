# LRDBench Performance Profiling Report
==================================================

## Performance Summary

| Estimator | 1K | 5K | 10K | Status |
|-----------|----|----|----|--------|
| DFA | 0.0859s | 0.3736s | 0.6821s | ✅ All Pass |
| DMA | 0.0836s | 0.5623s | 1.2883s | ✅ All Pass |
| GPH | 0.0010s | 0.0010s | 0.0010s | ✅ All Pass |
| Higuchi | 0.0083s | 0.0470s | 0.1056s | ✅ All Pass |
| Periodogram | 0.0000s | 0.0010s | 0.0000s | ✅ All Pass |
| RS | 0.0312s | 0.1141s | 0.2012s | ✅ All Pass |
| Whittle | 0.0050s | 0.0041s | 0.0055s | ✅ All Pass |

## Detailed Analysis

### Data Size: 1000

#### RS

- **Execution Time**: 0.0312s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.824109037843658

#### DFA

- **Execution Time**: 0.0859s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.86825143960194

#### DMA

- **Execution Time**: 0.0836s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.8045619520676908

#### Higuchi

- **Execution Time**: 0.0083s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.04759044958586145

#### GPH

- **Execution Time**: 0.0010s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.911503262100257

#### Periodogram

- **Execution Time**: 0.0000s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.7617536450694748

#### Whittle

- **Execution Time**: 0.0050s
- **Success**: True
- **Data Size**: 1000
- **Hurst Estimate**: 0.7373660816123788

### Data Size: 5000

#### RS

- **Execution Time**: 0.1141s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.7730010723476779

**Top 10 Function Calls:**
```
         129347 function calls in 0.114 seconds

   Ordered by: cumulative time
   List reduced from 94 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.114    0.114 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\rs\rs_estimator.py:103(estimate)
       20    0.014    0.001    0.113    0.006 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\rs\rs_estimator.py:195(_calculate_rs)
     2262    0.002    0.000    0.056    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3605(std)
     2262    0.005    0.000    0.053    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:216(_std)
     2262    0.025    0.000    0.048    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:144(_var)
    11335    0.020    0.000    0.020    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     2284    0.002    0.000    0.017    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
     2285    0.005    0.000    0.015    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
     4526    0.004    0.000    0.014    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:69(_wrapreduction)
     2262    0.002    0.000    0.009    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:2781(max)



```

#### DFA

- **Execution Time**: 0.3736s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.7939875927619482

**Top 10 Function Calls:**
```
         377182 function calls in 0.373 seconds

   Ordered by: cumulative time
   List reduced from 116 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.373    0.373 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dfa\dfa_estimator.py:80(estimate)
       20    0.029    0.001    0.372    0.019 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dfa\dfa_estimator.py:159(_calculate_fluctuation)
     4953    0.056    0.000    0.233    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_polynomial_impl.py:442(polyfit)
     4953    0.071    0.000    0.117    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\linalg\_linalg.py:2382(lstsq)
     4953    0.035    0.000    0.052    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_polynomial_impl.py:694(polyval)
     4995    0.005    0.000    0.047    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
     4996    0.013    0.000    0.042    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
     4953    0.022    0.000    0.039    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_twodim_base_impl.py:539(vander)
     9951    0.025    0.000    0.025    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     4953    0.002    0.000    0.015    0.000 {method 'sum' of 'numpy.ndarray' objects}



```

#### DMA

- **Execution Time**: 0.5623s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.7653476705261029

**Top 10 Function Calls:**
```
         975836 function calls in 0.562 seconds

   Ordered by: cumulative time
   List reduced from 111 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.562    0.562 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dma\dma_estimator.py:76(estimate)
       15    0.092    0.006    0.561    0.037 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dma\dma_estimator.py:174(_calculate_fluctuation)
    75032    0.047    0.000    0.440    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
    75033    0.129    0.000    0.394    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
    75039    0.126    0.000    0.126    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    75033    0.088    0.000    0.099    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:76(_count_reduce_items)
   150066    0.013    0.000    0.013    0.000 {built-in method builtins.issubclass}
    75000    0.011    0.000    0.011    0.000 {built-in method builtins.max}
    75032    0.011    0.000    0.011    0.000 {built-in method builtins.hasattr}
    75034    0.010    0.000    0.010    0.000 {built-in method numpy.lib.array_utils.normalize_axis_index}



```

#### Higuchi

- **Execution Time**: 0.0470s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.03143445482119556

#### GPH

- **Execution Time**: 0.0010s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.8614166270837988

#### Periodogram

- **Execution Time**: 0.0010s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.7118073845499564

#### Whittle

- **Execution Time**: 0.0041s
- **Success**: True
- **Data Size**: 5000
- **Hurst Estimate**: 0.7905284765747579

### Data Size: 10000

#### RS

- **Execution Time**: 0.2012s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.7759822058455065

**Top 10 Function Calls:**
```
         230009 function calls in 0.201 seconds

   Ordered by: cumulative time
   List reduced from 94 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.201    0.201 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\rs\rs_estimator.py:103(estimate)
       20    0.025    0.001    0.200    0.010 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\rs\rs_estimator.py:195(_calculate_rs)
     4028    0.004    0.000    0.098    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3605(std)
     4028    0.009    0.000    0.094    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:216(_std)
     4028    0.045    0.000    0.084    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:144(_var)
    20165    0.035    0.000    0.035    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     4050    0.004    0.000    0.029    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
     4051    0.008    0.000    0.026    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
     8058    0.007    0.000    0.025    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:69(_wrapreduction)
     4028    0.003    0.000    0.017    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:2781(max)



```

#### DFA

- **Execution Time**: 0.6821s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.8040056200153931

**Top 10 Function Calls:**
```
         693418 function calls in 0.682 seconds

   Ordered by: cumulative time
   List reduced from 116 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.682    0.682 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dfa\dfa_estimator.py:80(estimate)
       20    0.054    0.003    0.681    0.034 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dfa\dfa_estimator.py:159(_calculate_fluctuation)
     9114    0.102    0.000    0.425    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_polynomial_impl.py:442(polyfit)
     9114    0.129    0.000    0.214    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\linalg\_linalg.py:2382(lstsq)
     9114    0.065    0.000    0.096    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_polynomial_impl.py:694(polyval)
     9156    0.010    0.000    0.087    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
     9157    0.024    0.000    0.077    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
     9114    0.040    0.000    0.069    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\lib\_twodim_base_impl.py:539(vander)
    18273    0.046    0.000    0.046    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     9114    0.003    0.000    0.029    0.000 {method 'sum' of 'numpy.ndarray' objects}



```

#### DMA

- **Execution Time**: 1.2883s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.788741218315334

**Top 10 Function Calls:**
```
         2210910 function calls in 1.288 seconds

   Ordered by: cumulative time
   List reduced from 111 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.288    1.288 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dma\dma_estimator.py:76(estimate)
       17    0.210    0.012    1.288    0.076 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\dma\dma_estimator.py:174(_calculate_fluctuation)
   170036    0.107    0.000    1.012    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
   170037    0.296    0.000    0.906    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
   170043    0.296    0.000    0.296    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   170037    0.201    0.000    0.224    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:76(_count_reduce_items)
   340074    0.030    0.000    0.030    0.000 {built-in method builtins.issubclass}
   170000    0.026    0.000    0.026    0.000 {built-in method builtins.max}
   170036    0.026    0.000    0.026    0.000 {built-in method builtins.hasattr}
   170038    0.024    0.000    0.024    0.000 {built-in method numpy.lib.array_utils.normalize_axis_index}



```

#### Higuchi

- **Execution Time**: 0.1056s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.02625368660920202

**Top 10 Function Calls:**
```
         190585 function calls in 0.106 seconds

   Ordered by: cumulative time
   List reduced from 102 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\higuchi\higuchi_estimator.py:60(estimate)
       19    0.090    0.005    0.105    0.006 C:\Users\davia\DataExploratoryProject\lrdbench\analysis\temporal\higuchi\higuchi_estimator.py:163(_calculate_curve_length)
   182801    0.013    0.000    0.013    0.000 {built-in method builtins.abs}
       21    0.000    0.000    0.001    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\fromnumeric.py:3476(mean)
       22    0.000    0.000    0.001    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\numpy\_core\_methods.py:110(_mean)
       30    0.001    0.000    0.001    0.000 {built-in method numpy.asanyarray}
     7243    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\scipy\stats\_stats_mstats_common.py:22(linregress)
       28    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        1    0.000    0.000    0.000    0.000 C:\Users\davia\miniconda3\envs\fractional_pinn_env\lib\site-packages\scipy\stats\_distn_infrastructure.py:2214(ppf)



```

#### GPH

- **Execution Time**: 0.0010s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.8163785628728384

#### Periodogram

- **Execution Time**: 0.0000s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.6667892136147081

#### Whittle

- **Execution Time**: 0.0055s
- **Success**: True
- **Data Size**: 10000
- **Hurst Estimate**: 0.8326683437568863
