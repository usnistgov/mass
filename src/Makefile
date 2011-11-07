# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011

OBJECT_DIR=mass
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
 
 
 .PHONY: lint
 lint: lint-report.py
 lint-report.py: $(OBJECT_DIR)/*.py
	pylint-2.6 --ignore=deprecated.py mass > $@
 