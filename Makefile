# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011


TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = mass/*.py mass/*/*.py

.PHONY: archive built

build:
	python setup.py build

archive: $(TARGET_ZIP)

$(TARGET_ZIP): $(PYFILES) Makefile
	python setup.py sdist --format=gztar,zip

 
.PHONY: lint install clean test
lint: lint-report.txt
lint-report.txt: pylintrc mass/*/*.py
	pylint-2.6 --rcfile=$< mass > $@

TARGETDIR = /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/mass
install: build
	sudo python setup.py install
	ls -l $(TARGETDIR)  
	ls -l $(TARGETDIR)/core
	ls -l $(TARGETDIR)/mathstat
	
clean: 
	rm -rf build || sudo rm -rf build

	
test:
	@for dir in `find . -type d -name test`; do \
	    for pyfile in `find $${dir} -name "*.py"`; do \
	    	python $${pyfile}; \
	    done; \
	done
