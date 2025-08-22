# Fractional PINN Project: Optimized Long-Range Dependence Estimation

## 🎯 **Project Overview**

This project implements an optimized **Physics-Informed Neural Network (PINN)** for long-range dependence estimation using the `hpfracc` fractional calculus library. The goal is to achieve superior performance compared to classical estimators while maintaining physical interpretability.

## 🔬 **Key Features**

- **Real Fractional Calculus**: Uses `hpfracc` library for authentic fractional operators
- **Physics-Informed Constraints**: 7 different physical constraints for biological plausibility
- **Performance Optimization**: Built for speed and accuracy from the ground up
- **Comprehensive Benchmarking**: Full comparison with classical estimators
- **Research Ready**: Complete framework for academic publication

## 📁 **Project Structure**

```
fractional_pinn_project/
├── src/
│   ├── models/
│   │   ├── fractional_pinn.py      # Main PINN implementation
│   │   ├── fractional_operators.py # Fractional calculus operators
│   │   └── physics_constraints.py  # Physics-informed loss functions
│   ├── estimators/
│   │   ├── classical_estimators.py # DFA, R/S, GPH, CWT
│   │   └── pinn_estimator.py       # PINN-based estimator
│   ├── data/
│   │   ├── generators.py           # Synthetic data generation
│   │   └── loaders.py              # Data loading utilities
│   └── utils/
│       ├── benchmarking.py         # Performance evaluation
│       └── visualization.py        # Plotting and analysis
├── benchmarks/
│   ├── performance_benchmark.py    # Main benchmark script
│   └── results/                    # Benchmark results
├── tests/
│   └── test_fractional_pinn.py     # Unit tests
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🚀 **Quick Start**

1. **Activate the virtual environment:**
   ```bash
   cd fractional_pinn_project
   ..\fractional_pinn_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the benchmark:**
   ```bash
   python benchmarks/performance_benchmark.py
   ```

## 🔬 **Technical Implementation**

### **Fractional PINN Architecture**
- **Neural Network**: Multi-layer perceptron with fractional calculus integration
- **Fractional Operators**: Marchaud, Weyl, and custom derivatives via `hpfracc`
- **Physics Constraints**: 7 different physical laws enforced in loss function
- **Optimization**: Advanced training procedures with early stopping

### **Physics-Informed Constraints**
1. **Fractional Derivative Constraint**: Enforces fractional calculus relationships
2. **Scale Invariance**: Maintains power-law scaling properties
3. **Memory Constraint**: Preserves long-range memory effects
4. **Hurst Parameter Range**: Ensures biologically plausible values (0 < H < 1)
5. **Self-Similarity**: Enforces fractal-like properties
6. **Stationarity**: Maintains statistical stationarity
7. **Power Law Constraint**: Enforces power-law frequency scaling

## 📊 **Performance Goals**

- **Accuracy**: Target MAE < 0.05 (better than classical estimators)
- **Speed**: Real-time processing capability
- **Robustness**: Consistent performance under confounds
- **Interpretability**: Physical meaning in all outputs

## 🎯 **Research Applications**

- **Neurological Biomarkers**: Long-range dependence in brain signals
- **Clinical Monitoring**: Real-time patient monitoring
- **Early Detection**: Sensitive detection of pathological changes
- **Personalized Medicine**: Individual neural signature analysis

## 📚 **References**

- Research angles and motivation: `../RESEARCH_ANGLES_FOR_FRACTIONAL_PINNS.md`
- Original project analysis: `../FRACTIONAL_PINN_MOTIVATION_ANALYSIS.md`
- PINO fix summary: `../PINO_FIX_SUMMARY.md`

## 🏆 **Expected Outcomes**

This optimized implementation aims to:
1. **Demonstrate superior performance** compared to classical estimators
2. **Provide computational efficiency** for real-time applications
3. **Maintain physical interpretability** for clinical validation
4. **Enable research publication** with compelling results

---

**Status**: Development in Progress | Performance Optimization Focus | Research Ready  
**Next**: Implement core fractional PINN with hpfracc integration
