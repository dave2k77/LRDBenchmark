# Changelog

All notable changes to LRDBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2024-12-28

### 🚀 Added
- **Enhanced Neural Network Estimators**: Four new state-of-the-art neural network estimators
  - **CNN Estimator**: Convolutional Neural Network with residual connections and attention mechanisms
  - **LSTM Estimator**: Long Short-Term Memory with bidirectional architecture and multi-head attention
  - **GRU Estimator**: Gated Recurrent Unit with attention mechanisms and deep stacking
  - **Transformer Estimator**: Multi-head self-attention architecture with positional encoding
- **Pre-trained Models**: All neural estimators come with pre-trained PyTorch models for immediate use
- **Adaptive Input Handling**: Automatic adaptation to different sequence lengths
- **Enhanced Training Curriculum**: Comprehensive training with early stopping, learning rate scheduling, and gradient clipping
- **Robust Fallback System**: Graceful handling when models are not available
- **Production-Ready Workflow**: Development vs production workflow with automatic model management

### 🔧 Enhanced
- **Machine Learning Module**: Complete overhaul of ML estimators with enhanced architectures
- **Base ML Estimator**: Improved base class with better feature extraction and model management
- **Streamlit Dashboard**: Updated to support all 18 estimators including new ML and neural methods
- **Contamination Analysis**: Extended to include ML and neural estimators for comprehensive robustness testing
- **Benchmark System**: Enhanced to handle all estimator types with proper categorization

### 🐛 Fixed
- **Import Paths**: Corrected all import paths to use the new `lrdbench` package structure
- **Estimator Loading**: Fixed issues with ML estimators not appearing in benchmark results
- **Emoji Handling**: Corrected emoji removal logic in Streamlit dashboard for proper estimator matching
- **Data Shape Issues**: Resolved tensor dimension mismatches in neural estimators
- **Fallback Logic**: Fixed inheritance issues preventing proper PyTorch model usage
- **StandardScaler Mismatch**: Corrected feature extraction before scaling in ML estimators

### 📚 Documentation
- **Complete API Reference**: Updated for all 18 estimators
- **Enhanced Neural Models Guide**: Comprehensive documentation of new neural estimators
- **User Guides**: Updated with new estimator examples and usage patterns
- **README Files**: Updated across all components to reflect new capabilities
- **Installation Instructions**: Updated dependencies and requirements

### 🔄 Changed
- **Package Structure**: Reorganized ML estimators under enhanced architecture
- **API Interface**: Simplified imports with direct access to enhanced estimators
- **Estimator Count**: Increased from 12 to 18 total estimators
- **Version Number**: Bumped to 1.6.0 to reflect major feature additions
- **Dependencies**: Added PyTorch and enhanced scikit-learn requirements

### 🗑️ Removed
- **Legacy ML Estimators**: Replaced with enhanced versions
- **Old Import Paths**: Cleaned up deprecated import structures
- **Unused Code**: Removed obsolete estimator implementations

## [1.5.1] - 2024-12-20

### 🔧 Enhanced
- **Auto-Optimization System**: Revolutionary performance improvements with NUMBA and SciPy optimizations
- **Benchmark Performance**: Up to 850x speedup on optimized estimators
- **Web Dashboard**: Interactive Streamlit interface with real-time benchmarking

### 🐛 Fixed
- **Performance Issues**: Resolved bottlenecks in critical estimation algorithms
- **Memory Usage**: Optimized data structures for large-scale analysis
- **Error Handling**: Improved robustness and fallback mechanisms

## [1.5.0] - 2024-12-15

### 🚀 Added
- **Comprehensive Benchmarking System**: Systematic evaluation of all estimators
- **Contamination Testing**: Robustness assessment under various data conditions
- **Analytics Dashboard**: Usage tracking and performance monitoring
- **High-Performance Options**: JAX and Numba optimizations

### 🔧 Enhanced
- **Data Models**: Improved synthetic data generation
- **Estimator Implementations**: Enhanced classical methods
- **Documentation**: Comprehensive user guides and API reference

## [1.0.0] - 2024-12-01

### 🚀 Added
- **Initial Release**: Core long-range dependence estimation framework
- **Classical Estimators**: DFA, RS, DMA, Higuchi, GPH, Periodogram, Whittle, CWT, Wavelet methods, MFDFA
- **Data Models**: FBM, FGN, ARFIMA, MRW
- **Basic Benchmarking**: Simple performance comparison tools

---

## Migration Guide

### Upgrading from v1.5.x to v1.6.0

#### New Import Structure
```python
# Old way (v1.5.x)
from lrdbench.analysis.machine_learning.cnn_estimator import CNNEstimator

# New way (v1.6.0)
from lrdbench import CNNEstimator
```

#### Enhanced Estimators
```python
# All enhanced estimators are now available directly
from lrdbench import (
    CNNEstimator,      # Enhanced CNN with residual connections
    LSTMEstimator,     # Enhanced LSTM with bidirectional architecture
    GRUEstimator,      # Enhanced GRU with attention mechanisms
    TransformerEstimator  # Enhanced Transformer with self-attention
)

# Use immediately with pre-trained models
estimator = CNNEstimator()
result = estimator.estimate(data)
```

#### Streamlit Dashboard Updates
- Dashboard now supports all 18 estimators
- ML and neural estimators appear with 🤖 and 🧠 emojis
- Contamination analysis includes all estimator types
- Enhanced visualization and performance tracking

### Breaking Changes

1. **Import Paths**: Some internal import paths have changed
2. **Estimator Classes**: Old ML estimator classes have been replaced
3. **Model Files**: New model file structure for enhanced estimators

### Deprecation Warnings

- Old ML estimator classes will show deprecation warnings
- Legacy import paths will continue to work but are not recommended

---

## Contributing

When contributing to LRDBench:

1. **Follow Semantic Versioning**: Use conventional commit messages
2. **Update Changelog**: Document all user-facing changes
3. **Test Thoroughly**: Ensure all estimators work correctly
4. **Update Documentation**: Keep docs in sync with code changes
5. **Maintain Compatibility**: Consider backward compatibility for minor releases

---

## Support

For questions about upgrading or using new features:

- **Documentation**: [Enhanced Neural Models Guide](ENHANCED_NEURAL_MODELS.md)
- **API Reference**: [Complete API Documentation](documentation/api_reference/)
- **Issues**: [GitHub Issues](https://github.com/dave2k77/LRDBenchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/LRDBenchmark/discussions)
