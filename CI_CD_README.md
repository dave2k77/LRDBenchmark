# 🚀 CI/CD Pipeline Documentation

This document describes the automated CI/CD (Continuous Integration/Continuous Deployment) pipeline for the LRDBench project.

## 📋 Overview

The CI/CD pipeline consists of three main workflows:

1. **CI Pipeline** (`ci.yml`) - Runs on every push and pull request
2. **Deployment** (`deploy.yml`) - Automatically deploys to PyPI on releases
3. **Maintenance** (`maintenance.yml`) - Scheduled maintenance tasks

## 🔄 CI Pipeline (`ci.yml`)

### **Triggers**
- Push to `master`, `main`, or `develop` branches
- Pull requests to `master`, `main`, or `develop` branches
- Release publications

### **Jobs**

#### **1. Testing Matrix**
- **Operating Systems**: Ubuntu, Windows, macOS
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Exclusions**: Python 3.12 on Windows (compatibility issues)

#### **2. Code Quality Checks**
- **Black**: Code formatting validation
- **Flake8**: Style guide enforcement
- **MyPy**: Type checking

#### **3. Testing**
- **pytest**: Unit test execution
- **Coverage**: Code coverage reporting
- **Codecov**: Coverage upload

#### **4. Building**
- **Package Building**: Creates source and wheel distributions
- **Package Validation**: Ensures package integrity
- **Artifact Upload**: Stores build artifacts

#### **5. Security**
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency security checks

#### **6. Dependency Management**
- **Outdated Check**: Identifies outdated packages
- **Scheduled**: Runs weekly on Mondays

## 🚀 Deployment (`deploy.yml`)

### **Triggers**
- GitHub releases (published)

### **Process**
1. **Build**: Creates distribution packages
2. **Validate**: Checks package integrity
3. **Deploy**: Uploads to PyPI
4. **Release Assets**: Attaches packages to GitHub release
5. **Notification**: Confirms successful deployment

### **Environment Variables Required**
```bash
PYPI_API_TOKEN: Your PyPI API token
```

## 🔧 Maintenance (`maintenance.yml`)

### **Triggers**
- **Scheduled**: Every Monday at 6:00 UTC
- **Manual**: Can be triggered manually

### **Tasks**
1. **Dependency Updates**: Check for outdated packages
2. **Security**: Vulnerability scanning
3. **Code Quality**: Automated quality checks
4. **Documentation**: Validate package structure
5. **Reporting**: Generate maintenance summaries

## 🛠️ Local Development

### **Prerequisites**
```bash
pip install black flake8 mypy pytest pytest-cov
```

### **Running Checks Locally**

#### **Code Formatting**
```bash
# Check formatting
black --check --diff .

# Format code
black .
```

#### **Style Checking**
```bash
# Run Flake8
flake8 .

# Run with specific rules
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

#### **Type Checking**
```bash
# Run MyPy
mypy --ignore-missing-imports --no-strict-optional .
```

#### **Testing**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov=analysis --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py -v
```

## 📊 Code Quality Standards

### **Black Configuration**
- **Line Length**: 88 characters
- **Target Versions**: Python 3.9+
- **Exclusions**: Build directories, virtual environments

### **Flake8 Configuration**
- **Max Line Length**: 88 characters
- **Max Complexity**: 10
- **Ignores**: E203, W503 (compatible with Black)

### **MyPy Configuration**
- **Strict Mode**: Enabled
- **Type Checking**: Comprehensive
- **External Libraries**: Ignored (numpy, scipy, etc.)

## 🔐 Security

### **Dependencies**
- **Bandit**: Static security analysis
- **Safety**: Known vulnerability database
- **Regular Scans**: Weekly automated checks

### **Best Practices**
- Regular dependency updates
- Security vulnerability monitoring
- Automated security scanning

## 📈 Monitoring and Reporting

### **Coverage Reports**
- **Codecov Integration**: Automatic coverage upload
- **Thresholds**: Configurable coverage requirements
- **Trends**: Historical coverage tracking

### **Artifacts**
- **Build Outputs**: Distribution packages
- **Test Results**: Coverage and test reports
- **Security Reports**: Vulnerability assessments

## 🚨 Troubleshooting

### **Common Issues**

#### **Build Failures**
```bash
# Check package structure
python -m build --dry-run

# Validate package
twine check dist/*
```

#### **Test Failures**
```bash
# Run specific failing test
pytest tests/test_specific.py -v -s

# Check test dependencies
pip install -r requirements.txt
```

#### **Linting Issues**
```bash
# Auto-format with Black
black .

# Check specific file
flake8 specific_file.py
```

### **Debug Mode**
```bash
# Run with verbose output
pytest -v -s --tb=long

# Run specific job locally
python -m pytest tests/ -v
```

## 🔄 Workflow Customization

### **Adding New Jobs**
1. Edit the appropriate workflow file
2. Define job steps and requirements
3. Test locally before committing
4. Update this documentation

### **Modifying Triggers**
```yaml
on:
  push:
    branches: [ master, feature/* ]
  pull_request:
    branches: [ master ]
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6:00 UTC
```

### **Environment Variables**
```yaml
env:
  CUSTOM_VAR: value
  
jobs:
  job-name:
    env:
      JOB_SPECIFIC_VAR: value
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Style Guide](https://flake8.pycqa.org/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [pytest Testing Framework](https://docs.pytest.org/)

## 🤝 Contributing

When contributing to the CI/CD pipeline:

1. **Test Locally**: Run checks before committing
2. **Update Documentation**: Keep this file current
3. **Follow Standards**: Maintain consistent formatting
4. **Review Changes**: Ensure pipeline integrity

---

**Last Updated**: August 23, 2025  
**Maintainer**: Davian R. Chin  
**Status**: ✅ Active and Operational
