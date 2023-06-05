# Build mass
# J. Fowler, NIST
# Updated May 2023

TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = $(shell find mass -name "*.py")
CYFILES = $(shell find mass -name "*.pyx")
FORMFILES := $(shell find mass -name "*_form_ui.py")

.PHONY: all build clean test pep8 autopep8 lint

all: build test

build:
	python -m build

clean:
	rm -rf build || sudo rm -rf build
	rm -f `find . -name "*.pyc"`

test:
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
lint-report.txt: pylintrc $(PYFILES) Makefile
	pylint --rcfile=$< mass > $@
