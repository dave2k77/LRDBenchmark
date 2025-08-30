Documentation Guide
==================

Welcome to the LRDBenchmark documentation! This directory contains the source files for building the comprehensive documentation for LRDBenchmark, a Long-Range Dependence Benchmarking Toolkit.

Building the Documentation
-------------------------

To build the documentation locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build HTML documentation:**
   ```bash
   make html
   ```

3. **View the documentation:**
   Open `_build/html/index.html` in your web browser

4. **Clean build files:**
   ```bash
   make clean
   ```

Documentation Structure
----------------------

- **index.rst**: Main documentation entry point
- **installation.rst**: Installation and setup instructions
- **quickstart.rst**: Quick start guide and basic usage
- **api/**: Complete API reference documentation
- **examples/**: Code examples and demonstrations
- **research/**: Theoretical background and validation studies

ReadTheDocs.io Integration
-------------------------

This documentation is automatically built and hosted on ReadTheDocs.io at:
https://lrdbenchmark.readthedocs.io/

The documentation is automatically updated when changes are pushed to the main branch.

Contributing to Documentation
----------------------------

To contribute to the documentation:

1. Make changes to the `.rst` files in this directory
2. Test locally with `make html`
3. Commit and push your changes
4. ReadTheDocs.io will automatically rebuild

For more information, see the main project README.md file.
