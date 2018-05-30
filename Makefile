# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011


TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = $(shell find src/mass -name "*.py")
CYFILES = $(shell find src/mass -name "*.pyx")
FORMFILES := $(shell find src/mass -name "*_form_ui.py")

.PHONY: lint archive  build install clean test report_install_location pep8

build:
	python setup.py build

install: build
	sudo python setup.py install

clean:
	rm -rf build || sudo rm -rf build
	rm -f `find . -name "*.pyc"`

test:
	python runtests.py

archive: $(TARGET_ZIP)

$(TARGET_ZIP): $(PYFILES) $(CYFILES) Makefile
	python setup.py sdist --format=gztar,zip

PEPFILES := $(PYFILES) $(CYFILES)
PEPFILES := $(filter-out $(FORMFILES), $(PEPFILES))

pep8: pep8-report.txt
pep8-report.txt: $(PEPFILES) MAKEFILE
	pycodestyle --statistics --max-line-length=120 $(PEPFILES) > $@

lint: lint-report.txt
lint-report.txt: pylintrc $(PYFILES) MAKEFILE
	pylint-2.7 --rcfile=$< src/mass > $@
