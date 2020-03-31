# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011


TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = $(shell find mass -name "*.py")
CYFILES = $(shell find mass -name "*.pyx")
FORMFILES := $(shell find mass -name "*_form_ui.py")

.PHONY: lint archive all build develop install clean test report_install_location pep8

all: build develop test

build:
	python setup.py build

develop: build
	sudo python setup.py develop

install: build
	sudo python setup.py install

clean:
	rm -rf build || sudo rm -rf build
	rm -f `find . -name "*.pyc"`

test:
	pytest

archive: $(TARGET_ZIP)

$(TARGET_ZIP): $(PYFILES) $(CYFILES) Makefile
	python setup.py sdist --format=gztar,zip

PEPFILES := $(PYFILES)  # Don't pep8 the Cython files $(CYFILES)
PEPFILES := $(filter-out $(FORMFILES), $(PEPFILES))

pep8: pep8-report.txt
pep8-report.txt: $(PEPFILES) MAKEFILE
	pycodestyle --statistics --max-line-length=150 $(PEPFILES) > $@

lint: lint-report.txt
lint-report.txt: pylintrc $(PYFILES) MAKEFILE
	pylint --rcfile=$< mass > $@
