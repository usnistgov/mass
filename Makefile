# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011


TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = $(shell find src -name "*.py")
CYFILES = $(shell find src -name "*.pyx")

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

pep8: pep8-report.txt
pep8-report.txt: $(PYFILES)
	pep8 --statistics --count --max-line-length=99 $(PYFILES) $(CYFILES) > $@

lint: lint-report.txt
lint-report.txt: pylintrc $(PYFILES)
	pylint-2.7 --rcfile=$< src/mass > $@
