# Build mass
# J. Fowler, NIST
# Updated May 2023

TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = $(shell find . -name "*.py")
CYFILES = $(shell find . -name "*.pyx")
FORMFILES := $(shell find mass -name "*_form_ui.py")

.PHONY: all build clean clean_hdf5 test pep8 autopep8 lint ruff

all: build test

build:
	python -m build

clean: clean_hdf5
	rm -rf build || sudo rm -rf build
	rm -f `find . -name "*.pyc"`

clean_hdf5:
	rm -f */regression_test/*_mass.hdf5

test: clean_hdf5
	pytest

archive: $(TARGET_ZIP)

$(TARGET_ZIP): $(PYFILES) $(CYFILES) Makefile
	python setup.py sdist --format=gztar,zip

.PHONY: autopep8 pep8 lint
PEPFILES := $(PYFILES)  # Don't pep8 the $(CYFILES)
PEPFILES := $(filter-out $(FORMFILES), $(PEPFILES))  # Remove the UI.py forms

pep8: pep8-report.txt
pep8-report.txt: $(PEPFILES) Makefile
	pycodestyle --exclude=build,nonstandard . > $@ || true

autopep8: $(PEPFILES) Makefile
	autopep8 --verbose --in-place --recursive .

lint: lint-report.txt
lint-report.txt: $(PYFILES) Makefile
	ruff check --preview mass doc tests > $@

ruff:
	ruff check --preview mass doc tests
