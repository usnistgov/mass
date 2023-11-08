

# MASS: The Microcalorimeter Analysis Software System

MASS is the work of [Joe Fowler](https://bitbucket.org/joe_fowler/) and [Galen O'Neil](https://bitbucket.org/oneilg/) of NIST Boulder Labs and the University of Colorado, with substantial contributions from:

* Dan Becker
* Young-Il Joe
* Jamie Titus
* Many collaborators, who have made many bug reports, bug fixes, and feature requests.

MASS was begin in November 2010, and development continues.

## Introduction

MASS is a software suite to analyze data from high-resolution, cryogenic microcalorimeters. It is meant to be exactly as general as NIST's users and their collaborators require, without being any more general than that. We use it with x-ray and gamma-ray spectrometers.

With MASS and a little bit of Python knowledge, you can:

* Process data from one or multiple detectors at one time.
* Analyze data from time-division multiplexed and microwave-multiplexed systems.
* Choose and apply data cuts.
* Compute and apply "optimal filters" of various types.
* Fix complex line shapes in an energy spectrum.
* Estimate and apply accurate functions for absolute-energy calibration.
* Win friends and influence people.

As of this writing (November 8, 2023), it is 12,000 lines of Python (plus 3000 lines of test code). It has some extension modules in Cython.


## Installation
Mass requires Python version 3.8 or higher. You will might need to add an ssh key to your bitbucket account to get the installation to work.

```
cd ~/somewhere/to/install/code
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git#egg=mass
```
In the above instance, mass will be installed as `~/somewhere/to/install/code/src/mass`. That penultimate directory (`src/`) follows from pip's rules.

If you want to install a certain branch "branchname", you can go to the installation directory and use the usual git commands to change branches, or you can install directly from the branch of choice like this:
```
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git@branchname#egg=mass
```

If you want to update the installation, you can do so via the usual git (`git pull`) from the installed directory, or you can:
```
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git#egg=mass --upgrade
```
It's possible that the above would also help you to re-install if you do something drastic such as change from using Python 3.9 to 3.10. (Not tested!)

See the [`nist-qsp-tdm README`](https://bitbucket.org/nist_microcal/nist-qsp-tdm) for instructions to install all Python software simultaneously for a TDM operation, and how to setup venv.

You may need to install Visual Studio Community Edition in order to run on Windows.

### Install location
If you install in a virtual environment (a "venv"), the install location will be inside the `MYVENV/src/mass` where `MYVENV` is the name of your venv.
Otherwise will will just be in mass relative to where you run the pip command.

### -e
The `-e` argument to the `pip install` command makes development really easy, you can change python files; the next time you import mass the new files will be used. If you change Cython files or other complied files you should install again. Do `pip install -e .` from within the source directory.



### Scripts
Mass installs 2 scripts (as of November 2023). These are `make_projectors` and `ljh2off`. You can check `setup.cfg` and look for either `scripts` or `console_scripts` to see if the list has been changed. These should be executable from your terminal from anywhere without typing `python` before them, though you may need to add something to your path. If you need to add something to your path, please use this approach to make them part of the MASS installation. The scripts all have help accessible via, e.g., `ljh2off --help`.


### Python 2.7
If you really want to use Python 2.7, know that MASS version 0.7.5 is the last one tested on Python 2.7. You can install it with the following command:
```
pip install -e git+ssh://git@bitbucket.org/joe_fowler/mass.git@versions/0.7.5#egg=mass
```

## Documentation

MASS automatically deploys two versions of the documentation during the Bitbucket pipeline run:

* [Docs for master](https://oneilg.bitbucket.io/mass/)
* [Docs for latest push to any non master branch](https://oneilg.bitbucket.io/mass_non_master/)


How you can help with the documentation:

1. [Write a docstring.](https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings) (or update an old docstring and fix the formatting while you're at it). A poorly formatted docstring is better than no docstring; don't be shy.
2. [Add to (or write a new) .rst file.](http://commonmark.org/help/).

General advice on updating/creating documentation files:

* Put a header (line starting with pound) as the first line.
* Again, poorly formatted markdown is better than no markdown.
* [Here's a WYSIWIG online markdown editor/viewer.](https://dillinger.io/) Just copy and paste once it looks like you want it to.
* Look in `doc/` for `.md` files and `.rst` files. The latter contain sphinx doctests. We should probably use these for all new documentation.
* If there's something worth updating in an existing `.md` file, then update it. Better yet, change it to an `.rst` file with doctests.


## Realtime Analysis
Realtime analysis is implemented by writing filtered values as well as "SVD components" represting the shape of each pulse to a `.off` file. This requires a substanial change is how things in mass work, thus there is a new interface that replaces larges parts of mass in `mass.off`. Look at `mass/off/test_channels.py` for a test script that shows basic usage.

# Development Tips

## Tests

If you look for files in the `tests/` directory, they will have examples of how the tested functions are called. You can learn a fair amount from looking at them.

Run all the tests on your system and make sure they pass!. Do `pytest .` from the `mass` directory. Tests require that you install via `pip install -e ...`.

### Auto-run subsets of tests
`pytest-watch --pdb -- mass/off` run from the source directory will run only the tests in `mass/off`, automatically, each time a file is saved. Upon any error, it will drop into pdb for debugging. You will need to `pip install pytest-watch` first.

## Working on docs + tests
Change directory into `doc` then

  * posix (mac/linux) `make doctest html && open _build/html/index.html`
  * windows cmd shell `make doctest html && start _build/html/index.html`
  * windows powershell `./make doctest;./make html;start _build/html/index.html`

Read about RST (reStructuredText) format. It is weird. My most common mistake is forgetting the blank line between `.. blah` statements and the following text. See the [Sphinx docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for details about its syntax.

# massGui, the GUI for Mass
massGui is an attempt to bring the core features of Mass into a graphical user interface (GUI). You can find more information and install it [here](https://github.com/gmondee/massGui).