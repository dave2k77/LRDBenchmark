# 🚀 **LRDBench Development Guide**

This document is for developers working on the LRDBench project. For user documentation, see the main `README.md`.

## 🎯 **Project Overview**

LRDBench is a comprehensive toolkit for benchmarking long-range dependence estimators on synthetic and real-world time series data. This project focuses on implementing and analyzing five key stochastic models with 18 different estimators.

## 🏆 **Project Status**

🎉 **PROJECT COMPLETE - 100%** 🎉

All major components have been successfully implemented and tested:

- ✅ **Data Models**: 5/5 models fully implemented and optimized
- ✅ **Estimators**: 18/18 estimators with comprehensive testing
- ✅ **High-Performance**: Sub-100ms estimation times with robust algorithms
- ✅ **Neural fSDE**: Components present with optional JAX/PyTorch dependencies
- ✅ **Auto-Discovery**: Intelligent component discovery and integration system
- ✅ **PyPI Ready**: Complete packaging configuration for distribution
- ✅ **Demos**: Comprehensive demonstration scripts and examples
- ✅ **Production Ready**: All models come pre-trained and ready to use

## 🏗️ **Project Structure**

### **📁 Root Level (Essential Files)**
```
DataExploratoryProject/
├── README.md                           # User-facing documentation (PyPI)
├── DEVELOPMENT.md                      # This development guide
├── requirements.txt                    # Dependencies
├── setup.py                          # PyPI packaging configuration
├── pyproject.toml                     # Modern Python packaging
├── MANIFEST.in                        # Package inclusion rules
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── auto_discovery_system.py           # Component discovery system
├── component_registry.json            # Component registry
├── lrdbench/                          # Main package directory
│   ├── __init__.py                    # Package initialization
│   ├── analysis/                      # Estimator implementations
│   ├── models/                        # Data model implementations
│   └── analytics/                     # Usage tracking and analytics
├── setup/                             # Setup and configuration files
├── scripts/                           # Main Python scripts
├── config/                            # Configuration files
├── assets/                            # Images and media files
├── documentation/                     # Documentation
├── demos/                             # Demo scripts
├── tests/                             # Test files
└── confound_results/                  # Quality leaderboard and benchmark results
```

### **📁 Organized Folders**

#### **🔧 setup/ - Setup & Configuration**
- Git Bash setup guides and configuration
- PowerShell profiles and terminal settings
- Git hooks and automation scripts
- Project cleanup documentation

#### **🐍 scripts/ - Main Python Scripts**
- Comprehensive benchmarking scripts
- Machine learning estimator analysis and training
- Confound analysis and robustness testing
- Machine learning vs classical comparison

#### **⚙️ config/ - Configuration & Registry**
- Component registry and discovery metadata
- Git configuration and project settings
- Auto-discovery system configuration

#### **🖼️ assets/ - Images & Media**
- Research visualizations and diagrams
- Neural fSDE framework analysis
- Machine learning estimator performance results
- Publication-quality figures

## 🚀 **Development Setup**

### **1. Environment Setup**

#### **Conda Environment (Recommended)**
```bash
# Create and activate the fracnn environment
conda create -n fracnn python=3.9
conda activate fracnn

# Install dependencies
pip install -r requirements.txt
```

#### **Virtual Environment**
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Local Development Installation**
```bash
# Install in development mode
pip install -e .

# Or build and install
python -m build
pip install dist/lrdbench-*.whl
```

## 🧪 **Testing & Development**

### **Running Tests**
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_fbm.py

# Run with coverage
python -m pytest --cov=lrdbench tests/
```

### **Testing Shortened Names**
```bash
# Test that all shortened model names work
python test_shortened_names.py
```

### **Testing Analytics System**
```bash
# Test analytics functionality
python -c "from lrdbench.analytics import UsageTracker; tracker = UsageTracker(); print('✅ Analytics working')"
```

## 📦 **Package Structure**

### **Core Components**
- **`lrdbench/analysis/`**: All estimator implementations (18 estimators)
- **`lrdbench/models/`**: Data model implementations (5 generators)
- **`lrdbench/analytics/`**: Usage tracking and analytics system

### **Model Naming Convention**
We use shortened names for user convenience:
- `FBMModel` = `FractionalBrownianMotion`
- `FGNModel` = `FractionalGaussianNoise`
- `ARFIMAModel` = `ARFIMAModel` (already short)
- `MRWModel` = `MultifractalRandomWalk`

### **Analytics System**
The analytics system provides:
- **Usage Tracking**: Monitor how estimators are used
- **Performance Monitoring**: Track execution times and resource usage
- **Error Analysis**: Categorize and analyze failures
- **Workflow Analysis**: Understand user behavior patterns
- **Dashboard**: Unified interface for all analytics data

## 🔧 **Development Workflow**

### **1. Making Changes**
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test thoroughly: `python test_shortened_names.py`
4. Update documentation if needed
5. Commit with descriptive message

### **2. Testing Changes**
```bash
# Test imports
python -c "from lrdbench import FBMModel; print('✅ Import working')"

# Test functionality
python -c "from lrdbench import FBMModel; fbm = FBMModel(H=0.7); data = fbm.generate(100); print(f'✅ Generated {len(data)} points')"

# Test analytics
python -c "from lrdbench import get_analytics_summary; print(get_analytics_summary())"
```

### **3. Building for PyPI**
```bash
# Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build package
python -m build

# Upload to PyPI (test first)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## 📚 **Documentation**

### **Documentation Structure**
- **`README.md`**: User-facing documentation (PyPI users)
- **`DEVELOPMENT.md`**: This development guide
- **`documentation/`**: Comprehensive API documentation
- **`examples/`**: Usage examples and demos

### **Updating Documentation**
When making changes:
1. Update relevant API documentation
2. Update examples if needed
3. Update this development guide
4. Ensure main README remains user-focused

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# If you get import errors, check:
python -c "import lrdbench; print(lrdbench.__file__)"
python -c "from lrdbench.models.data_models import FBMModel; print('✅ Models working')"
```

#### **Analytics Issues**
```bash
# Test analytics system
python -c "from lrdbench.analytics import UsageTracker; print('✅ Analytics working')"
```

#### **Model Issues**
```bash
# Test individual models
python -c "from lrdbench import FBMModel; fbm = FBMModel(H=0.7); print('✅ FBM working')"
```

## 🚀 **Next Steps & Future Development**

### **Immediate Priorities**
- [ ] Monitor PyPI package usage
- [ ] Collect user feedback
- [ ] Address any reported issues

### **Future Enhancements**
- [ ] Additional data models
- [ ] More estimator algorithms
- [ ] Enhanced analytics dashboard
- [ ] Web-based interface
- [ ] Performance optimizations

### **Research Areas**
- [ ] Advanced neural network architectures
- [ ] Hybrid estimation methods
- [ ] Real-world data validation
- [ ] Clinical applications

## 🤝 **Contributing**

### **Development Guidelines**
1. **Follow existing code style** and documentation patterns
2. **Add comprehensive tests** for new features
3. **Update documentation** for any API changes
4. **Use the auto-discovery system** for component integration
5. **Test shortened names** work correctly
6. **Ensure analytics system** remains functional

### **Code Review Process**
1. Create pull request with descriptive title
2. Include tests for new functionality
3. Update relevant documentation
4. Ensure all shortened names still work
5. Test analytics system integration

## 📊 **Performance Benchmarks**

### **Current Performance**
- **Data Generation**: < 10ms for 1000 points
- **Estimation**: < 100ms for most estimators
- **Memory Usage**: Optimized for large datasets
- **GPU Acceleration**: Available for JAX-based methods

### **Testing Performance**
```bash
# Run performance benchmarks
python scripts/comprehensive_estimator_benchmark.py

# Test specific estimators
python scripts/comprehensive_ml_classical_benchmark.py
```

## 🔍 **Debugging & Profiling**

### **Debug Mode**
```bash
# Enable debug logging
export LRDBENCH_DEBUG=1
python your_script.py
```

### **Performance Profiling**
```bash
# Profile with cProfile
python -m cProfile -o profile.stats your_script.py

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## 📞 **Support & Contact**

For development questions or technical support:
- **Repository Issues**: Use GitHub issues
- **Development Team**: Internal communication channels
- **Documentation**: Check this guide and `documentation/` folder

---

**Remember**: Always test that shortened names work after making changes, and ensure the analytics system remains functional!
