.PHONY: doc

doc: html/index.html

html/index.html: doxygen.conf Makefile ../mass/*.py ../mass/*/*.py
	doxygen doxygen.conf
