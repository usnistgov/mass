

          MASS: The Microcalorimeter Analysis Software System

                    Joe Fowler, NIST Boulder Labs

                        November 2010-present

# Installation
We reccomend you use Python 3, although as of March 2020 we are still supporting python 2.7.
  
```  
pip install -e git+https://oneilg@bitbucket.org/joe_fowler/mass.git#egg=mass
```

When installed with the `-e` argument the mass source code will be in `src/mass` relative to where you ran the install command. You can install without `-e` if you don't need to modify the source, but the tests won't pass. To get them to pass we need to install all the test data with the source code and we currently don't do that.

Dependency managment in python is a mess. Ubuntu tries to install packages via apt, then you can also install them via pip. Often if you have them installed via apt, then pip will lie to you about what version is available. A solution is to use virtual env. It changes your terminal to use a certain version of python with its own new set of packages. Here are commands to make a virtualenv and activate it. You should put the line that starts with `source` in `~/.profile` so it always starts new terminals with this virtual enviroment activated. Do not skip the pip upgrade, or else anything with PyQt will fail.
```
python3 -m venv ~/venv/qsp
source  ~/venv/qsp/bin/activate
pip install pip --upgrade
```

## Scripts
Mass installs 3 scripts (as of March 2020). These are `ljh_truncate`, `make_projectors`, and `ljh2off`. You can check `setup.py` and look for either `scripts` or `console_scripts` to see if any others have been added. These should be executable from your terminal from anywhere without typing `python` before them. They all have help accessible via eg `ljh2off --help`.

# Documentation
Use ipython to do something like `mass.function?` to see docstrings, or browse the source code. Also there are some overview `.md` files in the `doc` folder. I like to view these `.md` files in bitbucket.


How to help with the documentation:
1. [Write a docstring.](https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings) (or update an old docstring and fix the formatting while you're at it)
  * A poorly formatted docstring is better than no docstring, don't be shy.
2. [Add to (or write a new) .md file.](http://commonmark.org/help/)
  * put a header (line starting with pound) as the first line
  * again, poorly formatted markdown is better than no markdown  
  * [link to WYSIWIG online markdown editor/viewer.](https://dillinger.io/) Just copy and paste once it looks like you want it to.
  * look in mass/doc for .md files

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

# Realtime Analysis
Realtime analysis is implemented by writing filtered values as well as "svd components" represting the shape of each pulse to a `.off` file. This requires a substanial change is how things in mass work, thus there is a new interface that replaces larges parts of mass in `mass.off`. Look at `mass/off/test_channels.py` for a test script that shows basic usage.

## Tests

If you look for files named `test_*.py` they will have examples of how the tested functions are called. You can learn a fair amount from looking at them.

Run all the tests on your systemto make sure they pass!. Do `pytest` in the `mass` directory. Tests require that you install via `pip install -e ...` or `python setup.py develop`.

## Tutorials and demos

These are all probably out of date. They live in `mass/mass/demo`. If you want to contribute to mass, but don't want to do a bunch of programming, updating the demos would be a huge help!
