# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011


TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = mass/*.py mass/*/*.py

.PHONY: lint archive  build install clean test report_install_location

archive: $(TARGET_ZIP)

$(TARGET_ZIP): $(PYFILES) Makefile
	python setup.py sdist --format=gztar,zip

lint: lint-report.txt
lint-report.txt: pylintrc $(PYFILES)
	pylint-2.7 --rcfile=$< mass > $@

build:
	python setup.py build

install: build
	sudo python setup.py install

clean: 
	rm -rf build || sudo rm -rf build
	rm -f `find . -name "*.pyc"`

test:
	@for dir in `find . -type d -name test`; do \
	    for pyfile in `find $${dir} -name "*.py"`; do \
	    	python $${pyfile}; \
	    done; \
	done
