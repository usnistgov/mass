[![Build Status](https://semaphoreci.com/api/v1/projects/682fce58-5d81-4d08-bb85-78a6edd0a4c2/946875/badge.svg)](https://semaphoreci.com/drjoefowler/mass)

          MASS: The Microcalorimeter Analysis Software System

                    Joe Fowler, NIST Boulder Labs

                        November 2010-present



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


# DOCUMENTATION

## Doc md files

Look in `mass/doc` for `*.md` files. These will look nice if viewed on bitbucket or with a markdown viewer. They contain example code for many of the features of mass. As of August 2016 they are the main way we are tying to add documentation.

## Docstrings


The most complete documentation (though certainly not complete) can be found in the docstrings that accompany every (well, almost every) module, class, function, and method in the system.  Thus:
```
import mass
help mass.TESGroup
help data.datasets[0].obscure_method_name
```
(Notice that help is a built-in function, and if you don't run in ipython, then you'll have to surround the argument with parentheses. Ipython, being totally sweet, will do this for you.) In iipython you can also do
```
data.datasets[0].obscure_method_name?
```
## Doxygen site

The subdirectory doc contains configuration and a Makefile to generate a local copy of a doxygen website.  For a mac:
```
$ cd ~/trunk/python/mass/doc
$ make doc
$ open -n html/index.html
```

For linux, I think you can replace the last with
`firefox html/index.html`, but I can't say for sure.

This is probably more useful to someone planning to add code to Mass than to the casual user.  Still, it's kind of fun to see what doxygen can do more or less by magic.

I will try to copy the doxygen site to the group webserver someday, and to keep it updated.  More on that later.


## Tutorials and demos

These are all probably out of date. They live in `mass/mass/demo`. If you want to contribute to mass, but don't want to do a bunch of programming, updating the demos would be a huge help!
