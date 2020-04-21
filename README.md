

          MASS: The Microcalorimeter Analysis Software System

                    Joe Fowler, NIST Boulder Labs

                        November 2010-present

# Installation
Requires Python 3.

```  
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git#egg=mass
```

See [`nist-qsp-tdm`](https://bitbucket.org/nist_microcal/nist-qsp-tdm) README for instructions to install all tdm python software simultaneously, and how to setup venv.

## Windows
You may need to install Visual Studio Community Edition. 

## Sudo
Try installing without `sudo` first, though I find on NIST macs that I can't get away without sudo.

## Scripts
Mass installs 3 scripts (as of March 2020). These are `ljh_truncate`, `make_projectors`, and `ljh2off`. You can check `setup.py` and look for either `scripts` or `console_scripts` to see if any others have been added. These should be executable from your terminal from anywhere without typing `python` before them, though you may need to add something to your path. Please update this if you need to add something to your path. They all have help accessible via eg `ljh2off --help`.

### Python 2.7
If you really want to use Python 2.7, version 0.7.5 is the last version tested on Python 2.7, you can install it with the following command:
```  
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git@versions/0.7.5#egg=mass
```

# Documentation
[Docs for master](https://oneilg.bitbucket.io/mass/) 
[Docs for latest push to any non master branch](https://oneilg.bitbucket.io/mass_non_master/) 


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

Run all the tests on your system and make sure they pass!. Do `pytest` in the `mass` directory. Tests require that you install via `pip install -e ...` or `python setup.py develop`.

## Tutorials and demos

These are all probably out of date. They live in `mass/mass/demo`. If you want to contribute to mass, but don't want to do a bunch of programming, updating the demos would be a huge help!

# Development Tips

## Running Tests
Run tests from within the source directory with `pytest`

### Auto-run subsets of tests
`pytest-watch --pdb -- mass/off` run from the source directory will run only the tests in `mass/off`, automatically, each time a file is saved. Upon any error it will drop into pdb for debugging. You will need to `pip install pytest-watch` first.

## Install location
If you intall in a venv, the install location will be inside the `venv/src/mass` where `venv` is the name of your venv.
Otherwise will will just be in mass relative to where you run the pip command.

## -e
The -e command makes development really easy, you can change python files, then the next time you import mass the new files will be used. If you change Cython files or other complied files you should install again. Either do `pip install -e .` from within the soure directory, or call `python setup.py develop` from within the directory.

## reload
You may find `reload(mass.mathstat)` useful for interactive code development, but be aware it can't reload complied files. 
