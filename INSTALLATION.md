# LRDBench Installation Guide

## Quick Installation

### Option 1: Install from source (recommended for development)

```bash
# Clone the repository
git clone https://github.com/dave2k77/LRDBenchmark.git
cd LRDBenchmark

# Create a virtual environment (recommended)
python -m venv lrdbench_env
source lrdbench_env/bin/activate  # On Windows: lrdbench_env\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 2: Install from PyPI (when available)

```bash
pip install lrdbenchmark
```

## Dependencies

The package will automatically install all required dependencies:

- **Core**: numpy, scipy, scikit-learn, pandas
- **ML/Neural**: torch, jax, jaxlib
- **Performance**: numba
- **Analysis**: pywavelets, matplotlib, seaborn
- **Monitoring**: psutil, networkx

## Verification

After installation, verify the package works:

```python
import lrdbenchmark
print(f"LRDBench version: {lrdbenchmark.__version__}")

# Test auto-discovery
from auto_discovery_system import AutoDiscoverySystem
discovery = AutoDiscoverySystem()
components = discovery.discover_components()
print(f"Components found: {sum(len(v) for v in components.values())}")
```

## Development Setup

For development work, install additional tools:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're using the virtual environment:

```bash
source lrdbench_env/bin/activate  # Linux/Mac
# or
lrdbench_env\Scripts\activate     # Windows
```

### Module Not Found
Some modules may not be fully implemented yet. The package includes placeholder classes that will raise informative errors when accessed.

### Performance Issues
For optimal performance, ensure you have:
- NumPy with optimized BLAS/LAPACK
- Numba for JIT compilation
- PyTorch with CUDA support (if using GPU)
