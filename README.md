

          MASS: The Microcalorimeter Analysis Software System

                    Joe Fowler, NIST Boulder Labs

                        November 2010-present

# Test Status
* Develop, Python 2.7 [![Build Status](https://semaphoreci.com/api/v1/projects/682fce58-5d81-4d08-bb85-78a6edd0a4c2/946875/badge.svg)](https://semaphoreci.com/drjoefowler/mass)

# Documentation
[Online documentation](https://oneilg.bitbucket.io/mass/)

The documentation of mass is a work in progress, it is worth reviewing. It is easy to help with the documentation. There are two ways to help.

1. [Write a docstring.](https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings) (or update an old docstring and fix the formatting while you're at it)
  * A poorly formatted docstring is better than no docstring, don't be shy.
2. [Add to (or write a new) .md file.](http://commonmark.org/help/)
  * put a header (line starting with pound) as the first line
  * again, poorly formatted markdown is better than no markdown  
  * [link to WYSIWIG online markdown editor/viewer.](https://dillinger.io/) Just copy and paste once it looks like you want it to.
  * look in mass/doc for .md files
3. Submit a pull request with your changes (or just copy and paste it into an issue with a description of where it goes, and I'll do the git work)

After that I (Galen) will update the docs page with your changes.

# Intro


Mass is a software suite to analyze data from microcalorimeters.  It is meant to be exactly as general as NIST users and their collaborators require, without being any more general than that.  

With Mass and a little bit of Python knowledge, you can:

* Process data from one or multiple detectors at one time.
* Analyze data from normal TDM systems or from flux-summed CDM.
* Choose data cuts.
* Compute and apply "optimal filters" of various types.
* Win friends and influence people.

As of this writing (September 28, 2012), it is 11,000 lines of Python. It has a single extension module in FORTRAN90 due to Brad Alpert, which runs a compute-intensive calculation very, very fast. It has a
single extension module in Cython that I wrote to execute some calculations from the field of "robust statistics" very quickly.  I expect more extension in F90, C, C++, and/or Cython for performance reasons.

Mass is being shifted from a personal project to a system that can be shared by microcalorimeters users at NIST and elsewhere.  This step is only a year old, so please be patient!  Let the author know what's missing or wrong or useful.



## Tests

If you look for files named `test_*.py` they will have examples of how the tested functions are called. You can learn a fair amount from looking at them.

Run all the tests on your systemto make sure they pass!. Do `python runtests.py` in the `mass` directory.

## Tutorials and demos

These are all probably out of date. They live in `mass/mass/demo`. If you want to contribute to mass, but don't want to do a bunch of programming, updating the demos would be a huge help!
