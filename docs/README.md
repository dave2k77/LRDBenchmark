# LRDBench Documentation

This directory contains the documentation for LRDBench, built using Sphinx and hosted on ReadTheDocs.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Install documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the documentation:
   ```bash
   make html
   ```

3. View the documentation:
   ```bash
   make serve
   ```
   Then open http://localhost:8000 in your browser.

## Directory Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── requirements.txt     # Documentation dependencies
├── Makefile            # Build commands
├── _static/            # Static files (CSS, images)
│   └── custom.css      # Custom styling
├── _templates/         # Custom templates
├── user_guide/         # User guide documentation
├── api/                # API reference
├── advanced/           # Advanced topics
├── development/        # Development documentation
└── research/           # Research documentation
```

## Building Documentation

### HTML Documentation
```bash
make html
```

### PDF Documentation
```bash
make pdf
```

### EPUB Documentation
```bash
make epub
```

### All Formats
```bash
make all
```

## Development

### Adding New Pages

1. Create a new `.rst` file in the appropriate directory
2. Add the file to the appropriate `toctree` in `index.rst`
3. Build and test the documentation

### Styling

Custom styles are defined in `_static/custom.css`. The documentation uses the ReadTheDocs theme with custom modifications.

### Testing

- Check for broken links: `make linkcheck`
- Spell check: `make spelling`
- Clean build: `make clean`

## ReadTheDocs Integration

The documentation is automatically built and deployed on ReadTheDocs when changes are pushed to the main branch.

### Configuration

The ReadTheDocs configuration is in `conf.py`:
- Project name: LRDBench
- Theme: sphinx_rtd_theme
- Extensions: autodoc, napoleon, mathjax, etc.
- Mock imports for optional dependencies

### Build Process

1. ReadTheDocs clones the repository
2. Installs dependencies from `requirements.txt`
3. Runs `make html` to build documentation
4. Deploys to https://lrdbench.readthedocs.io

## Contributing

When contributing to the documentation:

1. Follow the existing style and structure
2. Use reStructuredText (`.rst`) format
3. Include code examples where appropriate
4. Test your changes locally before submitting
5. Update the table of contents if adding new pages

## Troubleshooting

### Common Issues

**Import errors during build**
- Check that all dependencies are in `requirements.txt`
- Verify mock imports in `conf.py`

**Styling issues**
- Check `_static/custom.css`
- Verify theme configuration in `conf.py`

**Build failures**
- Run `make clean` and try again
- Check for syntax errors in `.rst` files
- Verify all referenced files exist

### Getting Help

- Check the Sphinx documentation: https://www.sphinx-doc.org/
- ReadTheDocs documentation: https://docs.readthedocs.io/
- Open an issue on GitHub for documentation-specific problems
