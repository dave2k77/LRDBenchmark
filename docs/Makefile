# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for LRDBench documentation

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "HTML documentation built in $(BUILDDIR)/html/"

# Build PDF documentation
pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX..."
	@cd "$(BUILDDIR)/latex" && make
	@echo "PDF documentation built in $(BUILDDIR)/latex/"

# Build EPUB documentation
epub:
	@echo "Building EPUB documentation..."
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "EPUB documentation built in $(BUILDDIR)/epub/"

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

# Serve documentation locally
serve: html
	@echo "Starting local server..."
	@cd "$(BUILDDIR)/html" && python -m http.server 8000
	@echo "Documentation available at http://localhost:8000"

# Check for broken links
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check completed. See $(BUILDDIR)/linkcheck/output.txt"

# Spell check
spelling:
	@echo "Running spell check..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)
	@echo "Spell check completed. See $(BUILDDIR)/spelling/output.txt"

# Build all formats
all: html pdf epub
	@echo "All documentation formats built."

# Install dependencies
install-deps:
	@echo "Installing documentation dependencies..."
	@pip install -r requirements.txt
	@echo "Dependencies installed."

# Setup development environment
setup-dev: install-deps
	@echo "Setting up development environment..."
	@mkdir -p _static _templates
	@echo "Development environment ready."

# Build for ReadTheDocs
rtd: html
	@echo "Documentation built for ReadTheDocs."
	@echo "Upload to ReadTheDocs or push to GitHub to trigger automatic build."
