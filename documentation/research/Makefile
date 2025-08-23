# LaTeX Makefile for Research Paper
# Usage: make all, make clean, make pdf

# Main document name (without extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex

# Bibliography processor
BIBTEX = bibtex

# Clean up extensions
CLEAN_EXTENSIONS = aux bbl blg log out toc lof lot fdb_latexmk fls synctex.gz

# Default target
all: pdf

# Compile PDF
pdf: $(MAIN).pdf

# Main compilation rule
$(MAIN).pdf: $(MAIN).tex abstract.tex sections/*.tex appendices/*.tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Clean up auxiliary files
clean:
	rm -f $(foreach ext,$(CLEAN_EXTENSIONS),$(MAIN).$(ext))
	rm -f sections/*.aux appendices/*.aux

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Quick compile (no bibliography)
quick: $(MAIN).tex
	$(LATEX) $(MAIN)

# View PDF (platform dependent)
view: pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v open > /dev/null; then \
		open $(MAIN).pdf; \
	else \
		echo "Please open $(MAIN).pdf manually"; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile complete PDF with bibliography"
	@echo "  pdf      - Same as 'all'"
	@echo "  quick    - Quick compile without bibliography"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove auxiliary files and PDF"
	@echo "  view     - Compile and open PDF"
	@echo "  help     - Show this help message"

.PHONY: all pdf quick clean cleanall view help
