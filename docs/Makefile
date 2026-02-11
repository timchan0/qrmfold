# Minimal Sphinx Makefile

SPHINXOPTS    ?=
PYTHON        ?= python
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = source
BUILDDIR      = build
APIDOCDIR     = $(SOURCEDIR)/api
PACKAGEDIR    = ../qrmfold

.PHONY: help clean apidoc html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(APIDOCDIR)"

apidoc:
	@$(PYTHON) -m sphinx.ext.apidoc --force --separate --module-first -o "$(APIDOCDIR)" "$(PACKAGEDIR)" "_*"


html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
