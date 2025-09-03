# Makefile for REV Documentation

# Variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PYTHON        = python3
PIP           = pip
BROWSER       = open

# Colors for output
RED           = \033[0;31m
GREEN         = \033[0;32m
YELLOW        = \033[0;33m
BLUE          = \033[0;34m
NC            = \033[0m # No Color

.PHONY: help clean install html pdf epub latex markdown diagrams api serve watch test-docs check-links

help:
	@echo "$(BLUE)REV Documentation Build System$(NC)"
	@echo "================================"
	@echo ""
	@echo "$(GREEN)Basic commands:$(NC)"
	@echo "  make html        - Build HTML documentation"
	@echo "  make pdf         - Build PDF documentation"
	@echo "  make serve       - Serve documentation locally"
	@echo "  make watch       - Auto-rebuild on changes"
	@echo ""
	@echo "$(GREEN)Documentation types:$(NC)"
	@echo "  make api         - Generate API documentation from code"
	@echo "  make diagrams    - Generate architecture diagrams"
	@echo "  make markdown    - Convert to Markdown format"
	@echo "  make epub        - Build EPUB format"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make install     - Install documentation dependencies"
	@echo "  make test-docs   - Test documentation build"
	@echo "  make check-links - Check for broken links"
	@echo ""
	@echo "$(YELLOW)Advanced:$(NC)"
	@echo "  make all         - Build all documentation formats"
	@echo "  make deploy      - Deploy to documentation server"

# Install documentation dependencies
install:
	@echo "$(BLUE)Installing documentation dependencies...$(NC)"
	$(PIP) install -q sphinx sphinx-rtd-theme sphinx-autodoc-typehints
	$(PIP) install -q myst-parser sphinx-copybutton sphinx-tabs
	$(PIP) install -q nbsphinx jupyter matplotlib
	$(PIP) install -q sphinx-autobuild linkchecker
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Clean build artifacts
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILDDIR)
	rm -rf source/_build
	rm -rf source/generated
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Initialize Sphinx project if not exists
init:
	@if [ ! -f source/conf.py ]; then \
		echo "$(BLUE)Initializing Sphinx documentation...$(NC)"; \
		sphinx-quickstart -q -p "REV System" -a "Rohan Vinaik" \
			-v "3.0" --ext-autodoc --ext-viewcode --ext-napoleon \
			--makefile --no-batchfile source; \
		echo "$(GREEN)✓ Sphinx initialized$(NC)"; \
	else \
		echo "$(YELLOW)Sphinx already initialized$(NC)"; \
	fi

# Generate Sphinx configuration
source/conf.py: init
	@echo "$(BLUE)Generating Sphinx configuration...$(NC)"
	@cat > source/conf.py << 'EOF'
# Configuration file for REV documentation
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'REV System'
copyright = '2024, Rohan Vinaik'
author = 'Rohan Vinaik'
release = '3.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'nbsphinx',
]

# Templates
templates_path = ['_templates']
exclude_patterns = []

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
EOF
	@echo "$(GREEN)✓ Configuration generated$(NC)"

# Build HTML documentation
html: source/conf.py
	@echo "$(BLUE)Building HTML documentation...$(NC)"
	@mkdir -p source/generated
	@cp -r ../*.md source/
	@cp -r *.md source/ 2>/dev/null || true
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo ""
	@echo "$(GREEN)✓ HTML documentation built$(NC)"
	@echo "$(YELLOW)View at: file://$(PWD)/$(BUILDDIR)/html/index.html$(NC)"

# Generate API documentation from code
api:
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@mkdir -p source/api
	sphinx-apidoc -f -o source/api ../src \
		--separate --implicit-namespaces \
		--module-first --no-toc
	@echo "$(GREEN)✓ API documentation generated$(NC)"

# Generate architecture diagrams
diagrams:
	@echo "$(BLUE)Generating architecture diagrams...$(NC)"
	@$(PYTHON) << 'EOF'
import subprocess
import os

# Mermaid diagrams to generate
diagrams = [
    ("module_dependency", "graph TB..."),  # Add actual diagram content
    ("data_flow", "sequenceDiagram..."),
    ("class_hierarchy", "classDiagram...")
]

os.makedirs("source/_static/diagrams", exist_ok=True)

for name, content in diagrams:
    # Save Mermaid file
    with open(f"source/_static/diagrams/{name}.mmd", "w") as f:
        f.write(content)
    
    # Convert to SVG if mmdc is available
    try:
        subprocess.run([
            "mmdc", "-i", f"source/_static/diagrams/{name}.mmd",
            "-o", f"source/_static/diagrams/{name}.svg"
        ], check=True)
        print(f"✓ Generated {name}.svg")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠ Skipped {name}.svg (install mermaid-cli)")
EOF
	@echo "$(GREEN)✓ Diagrams generated$(NC)"

# Build PDF documentation
pdf: latex
	@echo "$(BLUE)Building PDF documentation...$(NC)"
	@cd $(BUILDDIR)/latex && make all-pdf
	@echo "$(GREEN)✓ PDF documentation built$(NC)"
	@echo "$(YELLOW)View at: $(BUILDDIR)/latex/revsystem.pdf$(NC)"

# Build LaTeX documentation
latex: source/conf.py
	@echo "$(BLUE)Building LaTeX documentation...$(NC)"
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "$(GREEN)✓ LaTeX documentation built$(NC)"

# Build EPUB documentation
epub: source/conf.py
	@echo "$(BLUE)Building EPUB documentation...$(NC)"
	$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS)
	@echo "$(GREEN)✓ EPUB documentation built$(NC)"
	@echo "$(YELLOW)View at: $(BUILDDIR)/epub/REVSystem.epub$(NC)"

# Convert documentation to Markdown
markdown:
	@echo "$(BLUE)Converting documentation to Markdown...$(NC)"
	@mkdir -p $(BUILDDIR)/markdown
	@for file in source/*.md; do \
		if [ -f "$$file" ]; then \
			cp "$$file" $(BUILDDIR)/markdown/; \
		fi; \
	done
	@$(PYTHON) -c "print('Converting RST to Markdown...')"
	@find source -name "*.rst" -exec pandoc {} -f rst -t markdown -o $(BUILDDIR)/markdown/{}.md \; 2>/dev/null || true
	@echo "$(GREEN)✓ Markdown documentation ready$(NC)"

# Serve documentation locally
serve: html
	@echo "$(BLUE)Starting documentation server...$(NC)"
	@echo "$(YELLOW)Documentation available at: http://localhost:8080$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@cd $(BUILDDIR)/html && $(PYTHON) -m http.server 8080

# Watch for changes and auto-rebuild
watch: install
	@echo "$(BLUE)Watching for documentation changes...$(NC)"
	@echo "$(YELLOW)Documentation available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 --port 8000 \
		--ignore "*.pyc" --ignore "__pycache__"

# Test documentation build
test-docs:
	@echo "$(BLUE)Testing documentation build...$(NC)"
	@$(MAKE) clean > /dev/null 2>&1
	@$(MAKE) html SPHINXOPTS="-W --keep-going" > /dev/null 2>&1 && \
		echo "$(GREEN)✓ HTML build test passed$(NC)" || \
		echo "$(RED)✗ HTML build test failed$(NC)"
	@$(MAKE) api > /dev/null 2>&1 && \
		echo "$(GREEN)✓ API generation test passed$(NC)" || \
		echo "$(RED)✗ API generation test failed$(NC)"
	@echo "$(GREEN)✓ Documentation tests complete$(NC)"

# Check for broken links
check-links: html
	@echo "$(BLUE)Checking for broken links...$(NC)"
	@command -v linkchecker >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing linkchecker...$(NC)"; \
		$(PIP) install linkchecker; \
	}
	linkchecker $(BUILDDIR)/html/index.html \
		--no-warnings --ignore-url=^mailto: --ignore-url=^tel:
	@echo "$(GREEN)✓ Link check complete$(NC)"

# Build all documentation formats
all: clean html api diagrams pdf epub markdown
	@echo "$(GREEN)✓ All documentation formats built$(NC)"
	@echo ""
	@echo "$(BLUE)Build Summary:$(NC)"
	@echo "  HTML:     $(BUILDDIR)/html/index.html"
	@echo "  PDF:      $(BUILDDIR)/latex/revsystem.pdf"
	@echo "  EPUB:     $(BUILDDIR)/epub/REVSystem.epub"
	@echo "  Markdown: $(BUILDDIR)/markdown/"
	@echo "  API Docs: source/api/"

# Deploy documentation (configure as needed)
deploy: html
	@echo "$(BLUE)Deploying documentation...$(NC)"
	@if [ -z "$(DOCS_SERVER)" ]; then \
		echo "$(RED)Error: DOCS_SERVER not set$(NC)"; \
		echo "$(YELLOW)Usage: make deploy DOCS_SERVER=user@host:/path$(NC)"; \
		exit 1; \
	fi
	rsync -avz --delete $(BUILDDIR)/html/ $(DOCS_SERVER)
	@echo "$(GREEN)✓ Documentation deployed to $(DOCS_SERVER)$(NC)"

# Quick build for development
quick: api html
	@echo "$(GREEN)✓ Quick build complete$(NC)"

# Generate requirements file for Read the Docs
rtd-requirements:
	@echo "$(BLUE)Generating Read the Docs requirements...$(NC)"
	@cat > requirements-docs.txt << 'EOF'
sphinx>=4.0
sphinx-rtd-theme>=1.0
sphinx-autodoc-typehints>=1.12
myst-parser>=0.18
sphinx-copybutton>=0.5
sphinx-tabs>=3.0
nbsphinx>=0.8
jupyter>=1.0
matplotlib>=3.0
EOF
	@echo "$(GREEN)✓ requirements-docs.txt generated$(NC)"

# Generate Read the Docs configuration
rtd-config:
	@echo "$(BLUE)Generating Read the Docs configuration...$(NC)"
	@cat > ../.readthedocs.yaml << 'EOF'
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/source/conf.py
  builder: html

formats:
  - pdf
  - epub

python:
  install:
    - requirements: docs/requirements-docs.txt
    - method: pip
      path: .

search:
  ranking:
    api/*: -1
EOF
	@echo "$(GREEN)✓ .readthedocs.yaml generated$(NC)"

# Print build information
info:
	@echo "$(BLUE)Documentation Build Information$(NC)"
	@echo "================================"
	@echo "Python:       $(shell $(PYTHON) --version)"
	@echo "Pip:          $(shell $(PIP) --version)"
	@echo "Sphinx:       $(shell $(SPHINXBUILD) --version 2>/dev/null || echo 'Not installed')"
	@echo "Working Dir:  $(PWD)"
	@echo "Source Dir:   $(SOURCEDIR)"
	@echo "Build Dir:    $(BUILDDIR)"

.DEFAULT_GOAL := help