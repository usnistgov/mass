# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011

EXTENSION_DIR=extensions

$(OBJECT_DIR)/%.so: $(EXTENSION_DIR)/%.f90 $(EXTENSION_DIR)/%.pyf
	f2py -c $(EXTENSION_DIR)/$*.pyf $<
	mv $*.so $(OBJECT_DIR)/ 

TARGET_ZIP = mass.zip
TARGET_TAR = mass.tgz
PYFILES = mass/*.py

.PHONY: archive all

archive: $(TARGET_ZIP)
all: $(TARGET_ZIP) $(TARGET_TAR)

$(TARGET_ZIP): $(PYFILES) Makefile
	zip -v $@ $(PYFILES)

$(TARGET_TAR): $(PYFILES) Makefile
	tar -zvcf $@ $(PYFILES)
 
 
.PHONY: lint install clean
lint: lint-report.txt
lint-report.txt: $(OBJECT_DIR)/*.py
	pylint-2.6 --ignore=deprecated.py mass > $@

TARGETDIR = /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/mass
install:
	python setup.py build
	sudo python setup.py install
	ls -l $(TARGETDIR)  
	ls -l $(TARGETDIR)/core
	ls -l $(TARGETDIR)/mathstat
	
clean: 
	rm -rf build || sudo rm -rf build
