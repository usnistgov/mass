# Build mass into a zip or a gzipped tar archive for distribution.
# J. Fowler, NIST
# June 16, 2011

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
 