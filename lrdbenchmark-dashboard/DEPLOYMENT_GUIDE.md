# 🚀 LRDBenchmark Dashboard - Deployment Guide

This guide explains how to deploy the LRDBenchmark Dashboard to PyPI and Streamlit Cloud.

## 📦 **PyPI Package Deployment**

### **Prerequisites**

1. **PyPI Account**: Create an account at [PyPI](https://pypi.org/account/register/)
2. **TestPyPI Account**: Create an account at [TestPyPI](https://test.pypi.org/account/register/)
3. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
4. **Build Tools**: Install required build tools

```bash
pip install build twine setuptools wheel
```

### **Build and Upload Process**

#### **Step 1: Prepare the Package**

```bash
# Navigate to the dashboard package directory
cd lrdbenchmark-dashboard

# Clean any previous builds
rm -rf build/ dist/ *.egg-info/
```

#### **Step 2: Build the Package**

```bash
# Build the package
python -m build

# Check the built package
python -m twine check dist/*
```

#### **Step 3: Test Upload (Recommended)**

```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ lrdbenchmark-dashboard
```

#### **Step 4: Production Upload**

```bash
# Upload to PyPI (production)
python -m twine upload dist/*
```

### **Automated Build Script**

Use the provided build script for automated deployment:

```bash
python build_and_upload.py
```

This script will:
- Clean previous builds
- Build the package
- Check the package
- Offer upload options (PyPI/TestPyPI)

## 🌐 **Streamlit Cloud Deployment**

### **Automatic Deployment**

The dashboard is automatically deployed to Streamlit Cloud from the GitHub repository:

**Live URL**: [https://lrdbenchmark-dev.streamlit.app/](https://lrdbenchmark-dev.streamlit.app/)

### **Manual Deployment**

If you need to deploy manually:

1. **Fork the Repository**: Fork the LRDBenchmark repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select the forked repository
3. **Configure Deployment**:
   - **Main file path**: `lrdbenchmark-dashboard/lrdbenchmark_dashboard/app.py`
   - **Python version**: 3.9
   - **Requirements file**: `lrdbenchmark-dashboard/requirements.txt`

### **Streamlit Configuration**

The dashboard includes optimized Streamlit configuration:

```toml
[global]
developmentMode = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🔧 **Local Development**

### **Development Setup**

```bash
# Clone the repository
git clone https://github.com/dave2k77/LRDBenchmark.git
cd LRDBenchmark

# Install the dashboard package in development mode
cd lrdbenchmark-dashboard
pip install -e .

# Run the dashboard locally
lrdbenchmark-dashboard
```

### **Development Dependencies**

```bash
# Install with development dependencies
pip install -e .[dev]

# Install with deployment dependencies
pip install -e .[deploy]
```

## 📊 **Package Structure**

```
lrdbenchmark-dashboard/
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Dependencies
├── README.md              # Package documentation
├── MANIFEST.in            # Package files inclusion
├── build_and_upload.py    # Build and upload script
├── DEPLOYMENT_GUIDE.md    # This file
└── lrdbenchmark_dashboard/
    ├── __init__.py        # Package initialization
    ├── main.py           # Command-line entry point
    ├── app.py            # Main Streamlit application
    └── .streamlit/
        └── config.toml   # Streamlit configuration
```

## 🚀 **Usage After Deployment**

### **Installation**

```bash
# Install from PyPI
pip install lrdbenchmark-dashboard

# Run the dashboard
lrdbenchmark-dashboard
```

### **Alternative Usage**

```bash
# Run with streamlit directly
streamlit run -m lrdbenchmark_dashboard.app

# Run with Python
python -c "from lrdbenchmark_dashboard.app import main; main()"
```

## 🔍 **Verification**

### **Package Verification**

After uploading to PyPI, verify the package:

```bash
# Install the package
pip install lrdbenchmark-dashboard

# Test the command-line tool
lrdbenchmark-dashboard --help

# Test the Python import
python -c "import lrdbenchmark_dashboard; print('Package imported successfully')"
```

### **Dashboard Verification**

1. **Local Testing**: Run the dashboard locally and verify all features
2. **Cloud Testing**: Access the live dashboard and test all functionality
3. **Integration Testing**: Test with the main LRDBenchmark package

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Build Errors**:
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify package structure

2. **Upload Errors**:
   - Check PyPI credentials
   - Verify package name availability
   - Ensure package meets PyPI requirements

3. **Streamlit Deployment Issues**:
   - Check file paths in Streamlit Cloud
   - Verify requirements.txt compatibility
   - Check for import errors

### **Support**

For deployment issues:
- Check the [PyPI documentation](https://packaging.python.org/tutorials/packaging-projects/)
- Review [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-community-cloud)
- Create an issue on the GitHub repository

## 📈 **Monitoring**

### **PyPI Statistics**

Monitor package usage:
- [PyPI Statistics](https://pypi.org/project/lrdbenchmark-dashboard/#statistics)
- Download counts and version information

### **Streamlit Cloud Monitoring**

Monitor dashboard usage:
- Streamlit Cloud dashboard analytics
- User engagement metrics
- Performance monitoring

---

**For questions or issues with deployment, please create an issue on the GitHub repository.**
