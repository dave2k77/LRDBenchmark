# ML/NN Estimators Comprehensive Benchmark Report

**Generated:** 2025-08-28 09:16:17

## Summary Statistics

- **Total Tests:** 72
- **Successful Tests:** 72 (100.0%)
- **Failed Tests:** 0 (0.0%)

## Estimator Performance

| Estimator | Mean Error | Std Error | Avg Time (s) | Success Count |
|-----------|------------|-----------|--------------|---------------|
| CNN | 0.5633 | 0.2004 | 0.392 | 9.0 |
| GRU | 0.1794 | 0.1565 | 0.052 | 9.0 |
| GradientBoosting | 0.0301 | 0.0342 | 0.112 | 9.0 |
| LSTM | 0.1777 | 0.1560 | 0.022 | 9.0 |
| NeuralNetwork | 0.0383 | 0.0481 | 0.111 | 9.0 |
| RandomForest | 0.0213 | 0.0239 | 0.171 | 9.0 |
| SVR | 0.0375 | 0.0495 | 0.111 | 9.0 |
| Transformer | 0.3321 | 0.1960 | 0.161 | 9.0 |

## Data Type Performance

| Data Type | True H | Mean Error | Std Error |
|-----------|--------|------------|-----------|
| ARFIMA (H=0.7) | 0.7 | 0.1891 | 0.2533 |
| Random (H=0.5) | 0.5 | 0.1520 | 0.1781 |
| fBm (H=0.3) | 0.3 | 0.1589 | 0.1761 |
| fBm (H=0.5) | 0.5 | 0.1497 | 0.1798 |
| fBm (H=0.7) | 0.7 | 0.1532 | 0.2751 |
| fGn (H=0.3) | 0.3 | 0.1478 | 0.1843 |
| fGn (H=0.5) | 0.5 | 0.1667 | 0.1616 |
| fGn (H=0.7) | 0.7 | 0.1708 | 0.2592 |
| fGn (H=0.9) | 0.9 | 0.2641 | 0.3315 |

## Method Analysis

- **RandomForestEstimator (ML):** 9 uses
- **GradientBoostingEstimator (ML):** 9 uses
- **SVREstimator (ML):** 9 uses
- **CNN (Neural Network):** 9 uses
- **Transformer (Neural Network):** 9 uses
- **LSTMEstimator (LSTM):** 9 uses
- **GRUEstimator (GRU):** 9 uses
- **NeuralNetworkEstimator (ML):** 9 uses

## Detailed Results

| Estimator | ARFIMA (H=0.7) | Random (H=0.5) | fBm (H=0.3) | fBm (H=0.5) | fBm (H=0.7) | fGn (H=0.3) | fGn (H=0.5) | fGn (H=0.7) | fGn (H=0.9) |
|-----------|---|---|---|---|---|---|---|---|---|
| CNN | 0.002 | 0.005 | 0.005 | 0.005 | 0.005 | 0.002 | 0.002 | 0.002 | 0.002 |
| GRU | 0.701 | 0.703 | 0.703 | 0.703 | 0.702 | 0.702 | 0.702 | 0.702 | 0.702 |
| GradientBoosting | 0.792 | 0.506 | 0.309 | 0.509 | 0.685 | 0.311 | 0.540 | 0.783 | 0.892 |
| LSTM | 0.700 | 0.700 | 0.699 | 0.699 | 0.699 | 0.700 | 0.700 | 0.700 | 0.700 |
| NeuralNetwork | 0.826 | 0.491 | 0.291 | 0.501 | 0.703 | 0.292 | 0.390 | 0.754 | 0.924 |
| RandomForest | 0.722 | 0.492 | 0.248 | 0.496 | 0.698 | 0.292 | 0.526 | 0.768 | 0.899 |
| SVR | 0.814 | 0.488 | 0.280 | 0.496 | 0.725 | 0.293 | 0.491 | 0.687 | 0.766 |
| Transformer | 0.242 | 0.217 | 0.216 | 0.218 | 0.217 | 0.251 | 0.251 | 0.250 | 0.249 |
