# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mass
SOURCEDIR     = .
BUILDDIR      = _build

# Workaround!
# It appears sphinx generates scripts with relative #! strings that get confused
# easily. Try to coerce them to do the right thing
REALSPHINXBUILD := python $(shell which $(SPHINXBUILD))

# Put it first so that "make" without argument is like "make help".
help:
	@$(REALSPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(REALSPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
