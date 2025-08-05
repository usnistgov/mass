

# MASS: The Microcalorimeter Analysis Software System

MASS is the work of [Joe Fowler](https://github.com/joefowler/) and [Galen O'Neil](https://github.com/ggggggggg/) of NIST Boulder Labs and the University of Colorado, with substantial contributions from:

* [Dan Becker](https://github.com/danbek/)
* Young-Il Joe
* Jamie Titus
* Many collaborators, who have made many bug reports, bug fixes, and feature requests.

MASS was begun in November 2010, and development continues. See [Migration instructions](#migrating-from-bitbucket-to-github) for more info about the May 2024 move from Bitbucket to GitHub for hosting this project.

## Introduction

MASS is a software suite designed to analyze pulse records from high-resolution, cryogenic microcalorimeters. It is meant to be exactly as general as NIST's users and their collaborators require, without being any more general than that. Specifically, most of the algorithms are closedly tied to the data structures and file storage systems we use. A long-term goal is to fully separate the analysis algorithms from the data representation, but to date, we have pursued it only occasionally. We use MASS with pulse records from x-ray and gamma-ray spectrometers.

With MASS and a little bit of Python knowledge, you can:

* Process data from one or multiple detectors at one time.
* Analyze data from time-division multiplexed (TDM) and microwave-multiplexed (µMUX) systems.
* Choose and apply data cuts.
* Compute and apply "optimal filters" of various types.
* Fix complex line shapes in an energy spectrum.
* Estimate and apply accurate functions for absolute-energy calibration.
* Win friends and influence people.

As of this writing (June 12, 2025), MASS consists of nearly 13,000 lines of Python (plus over 3500 lines of test code). It used to use extension modules in Cython, but we removed the last of these in early 2025.


## Installation
Mass requires Python version 3.8 or higher. (We test it automatically with versions 3.9 and 3.13.) You have three choices. You can install inside a virtual environment, in a conda environment, or without either. We recommend the first.

### 1. Virtual environment (recommended approach, for people who already have a working non-conda python)

Python virtual environments provide some of the same benefits as a conda environments, but they are not the same. They are easy to set up. They let you keep up with separate dependencies for separate projects. However, you might want a more inclusive name, particularly on a data acquisition server. The venv you make should probably include MASS and other DAQ software; for that case, follow the instructions at https://github.com/usnistgov/microcal-daq/

The following assumes that you want your virtual environment to be named `analysis` and you are _not_ installing on a data acquisition computer:
```bash
python3 -m venv ~/analysis
source ~/analysis/bin/activate
pip install --upgrade pip
pip install -e git+https://github.com/usnistgov/mass.git#egg=mass
```


The above (HTTPS) cloning method is probably simpler initially (no ssh setup), but users who contribute to MASS might prefer to set up password-free connections with an ssh key. For them, instead of using the last line above, contributors might want to use ssh-based cloning:
```bash
pip install -e git+ssh://git@github.com/usnistgov/mass.git#egg=mass
```

Comments on these commands:
1. The first line is safe (but optional) if you already have a virtualenv at `~/analysis/`. If you don't, it creates one.
2. The second line must be used in _every_ shell where you want that virtualenv to be active. We suggest making a short alias, or (if you're willing to work in this enviroment by default) running this among your shell startup commands.
3. The third is optional but not a bad idea.
4. The fourth installs mass in `~/analysis/src/mass/`, which is within your virtual environment.

If you install in any virtual environment, the install location will be inside the `MYVENV/src/mass` where `MYVENV` is the name of your venv. You can switch git branches and update from GitHub in that directory and have everything take effect immediately.


### 2. Conda environment (an approach we used to recommend, but no longer do)

At NIST we are having trouble with conda under the Anaconda team's new licensing rules, so we leave our old conda advice here, but we are not using or testing it for accuracy.

If you are installing Python through the conda package manager, either the full Anaconda Python, or the slimmed-down Miniconda or Miniforge, you will install MASS into a conda environment.

#### Install miniconda (might cause licensing problems!)

See [Installing miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) for instructions on how to install miniconda. It's a 3-liner at the terminal: you download the installation script, run it, and delete it. Easy!

If you have a good reason to prefer the full Anaconda Python distribution, use [Installing Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install). That is an alternative to Miniconda. They share a package manager, differing on whether to install a minimal number of packages (Miniconda) or a huge suite of scientific Python tools (Anaconda).

Now, it is possible to install MASS in your base Conda environment, but you shouldn't. That would defeat a key purpose of Conda, which is to let you work on separate projects that might require conflicting versions of libraries such as `numpy`. In this example, we assume you want your Conda environment to be named `analysis`:
```bash
conda create --name analysis -y
conda activate analysis
pip install --upgrade pip
pip install -e git+https://github.com/usnistgov/mass.git#egg=mass
```

The above (HTTPS) cloning method is probably simpler initially (no ssh setup), but users who contribute to MASS might prefer to set up password-free connections with an ssh key. For them, instead of using the last line above, contributors might want to use ssh-based cloning:
```bash
pip install -e git+ssh://git@github.com/usnistgov/mass.git#egg=mass
```

You'll also need to remember to activate the `analysis` environment in each terminal where you want to use MASS, via
```bash
conda activate analysis
```
This could be made automatic in your `.bashrc` or `.profile` file, if you like.

When I tried this, the MASS source code was installed in `~/src/mass`.


### 3. No virtual environment

To install mass in `~/somewhere/to/install/code` you do this:
```
cd ~/somewhere/to/install/code
pip install -e git+ssh://git@github.com/usnistgov/mass.git#egg=mass
```

If you don't add an ssh key to your account the following might work (depending on GitHub's current security policies), but you'll need to type in a password for installation and each update:
```
pip install -e git+https://github.com/usnistgov/mass#egg=mass
```

In the above instance, mass will be installed as `~/somewhere/to/install/code/src/mass`. That penultimate directory (`src/`) follows from pip's rules.

### Using nonstandard releases, tags, branches, or commits
If you want to install a certain branch `branchname`, you can go to the installation directory and use the usual git commands to change branches, or you can install directly from the branch of choice with the syntax `@branchname`, like this:
```
pip install -e git+ssh://git@github.com/usnistgov/mass.git@branchname#egg=mass
```

The same syntax `@something`

### Updating the installation

The `-e` argument to the `pip install` command makes development really easy: you can change python files, and the next time you import mass the new files will be used.

See the [`Microcal DAQ README`](https://github.com/usnistgov/microcal-daq) for further instructions to install all relevant Python software simultaneously for a TDM operation, and how to setup venv.

**Windows users:** You may need to install Visual Studio Community Edition to run on Windows.



### Scripts
Mass installs 2 scripts (as of November 2023). These are `make_projectors` and `ljh2off`. You can check `setup.cfg` and look for either `scripts` or `console_scripts` to see if the list has been changed. These should be executable from your terminal from anywhere without typing `python` before them, though you may need to add something to your path. If you need to add something to your path, please use this approach to make them part of the MASS installation. The scripts all have help accessible via, e.g., `ljh2off --help`.


### Python 2.7
If you really want to use Python 2.7, you can use MASS version 0.7.5; it's the last one compatible with Python 2.7. You can install the tag `versions/0.7.5` with the following command:
```
pip install -e git+ssh://git@github.com/usnistgov/mass.git@versions/0.7.5#egg=mass
```

### Migrating from Bitbucket to GitHub

On May 6, 2024, the primary hosting service for MASS moved from Bitbucket to GitHub to comport with changes in Bitbucket _and_ NIST policies. What used to be at https://bitbucket.org/joe_fowler/mass/ is now here, at https://github.com/usnistgov/mass/. We had an excellent overall experience with Bitbucket over the last twelve years, but we anticipate an equally good experience now that we're hosted at GitHub.

If you have already checked out a copy of MASS and wish to continue being able to connect to the main repository for code updates, then one small piece of git magic will make this possible.

1. At a terminal, change directories to the main MASS directory. See [installation instruction](#installation) for clues about where that might be.
2. Check the current settings, `git remote -v` .
3. Say `git remote set-url origin ssh://git@github.com/usnistgov/mass.git`
4. Say `git remote -v` again to verify that the `origin` URL has changed.
5. If you happen to have more than one remote URL configured (e.g., an ssh-based and an http-based one), repeat step 3 for all. (This is unusual, but we have done it on a few lab computers.)

## Documentation

Automatic deployment of documentation has been paused while we figure out how to make it compatible with GitHub (May 10, 2024).
If static documentation from May 2024 is useful, you can still find two versions of the documentation:

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

# User Tips

## Configuring iPython and Matplotlib

I like to have Matplotlib start automatic in "interactive mode" and to use the Qt5 backend. Therefore I have a file `~/.matplotlib/matplotlibrc` whose contents are 
```yaml
# Also see defaults, which live in a file whose partial path is
# .../site-packages/matplotlib/mpl-data/matplotlibrc
backend:       Qt5Agg
interactive:   True
timezone:      US/Mountain
```

That's for a Mac or Linux. I understand that on Windows, you'd put the file in `C:\Users\YourUsername\.matplotlib\matplotlibrc`.

There are also settings I like to automatically make for any iPython session. You can create any 1+ python scripts in the `~/.ipython/profile_default/startup/` directory. They will be executed in lexical order. I have the following in mine. You'll notice that I specifically avoid importing mass, so as not to slow down iPython when I am doing something that doesn't involve MASS.

File `~/.ipython/profile_default/startup/50-imports.py`
```python
import pylab as plt
import numpy as np
import scipy as sp
import h5py
print("Imported pylab, numpy, scipy, and h5py")
plt.ion()  # Make pylab start in interactive mode

# ENABLE AUTORELOAD
ip = get_ipython()
ip.run_line_magic('load_ext', 'autoreload')
ip.run_line_magic('autoreload', '2')
print("Imported autoreload. (Use magic '%autoreload 0' to disable,)")%                  
```

File `~/.ipython/profile_default/startup/60-favorites.py`:
```python
"""
favorites.py

Functions that I, Joe Fowler, want loaded in every ipython session.
"""

import os
import numpy as np
import pylab as plt

from pylab import clf, plot, subplot, scatter, semilogy, semilogx, loglog


def myfigure(size, fignum=9):
    """Generate a figure #9 to replace the current figure. Do so only if
    the requested size tuple (width,height) is different by more than 0.01 inches
    in either dimension from the current figure's size."""
    curr_size = plt.gcf().get_size_inches()
    if abs(curr_size[0] - size[0]) + abs(curr_size[1] - size[1]) > .01:
        plt.close(fignum)
    return plt.figure(fignum, figsize=size)


def imshow(matrix, fraction=.09, *args, **kwargs):
    """Plot a matrix using pylab.imshow with rectangular pixels
    and a color bar. Argument 'fraction' is passed to the colorbar.
    All others go to pylab.imshow()"""
    plt.clf()
    plt.imshow(matrix, interpolation='none', *args, **kwargs)
    plt.colorbar(fraction=fraction)


def hist(x, bins=100, range=None, *args, **kwargs):
    """Plot a histogram using the (non-default) 'histtype="step"' argument
    and default bins=100.."""
    kwargs.setdefault('histtype', 'step')
    return plt.hist(x, bins=bins, range=range, *args, **kwargs)


print("Loaded favorites.")
```

# Development Tips

## Code style

We are using the tools `ruff` and `pycodestyle` to check for violations of Python programming norms. You can manually check for these by running either of these two complementary commands from the top-level directory of the MASS repository:

```
ruff check

# Or if the following, equivalent statement is more memorable:
make ruff
```

or

```
make pep8
cat pep8-report.txt
```

You'll need to install `ruff` and `pycodestyle` via pip, macports, or whatever for these to work.

Ideally, you'll have zero warnings from ruff and no contents in the `pep8-report.txt` file. This was not true for a very long time, but we finally reached full compliance with MASS release `v0.8.2`.

To _automate_ these tests so that it's easy to notice noncompliant new code as you develop MASS, there are tools you can install and activate within the VS Code development platform.

1. Install the tool `Ruff` for VS code. The offical identifier is `charliermarsh.ruff`.
2. Checking its settings (there are 15 of them at this time, which you can find by checking the VS code settings for `@ext:charliermarsh.ruff`). I found most default settings worked fine, but I did have to change:
   1. I added two lines to the `Ruff > Lint: Args` setting, to make it run the way we want:
      * `--line-length=135`
      * `--preview`
   2. I chose to have Ruff run `onType` rather than `onSave`, because the former did not cause noticeable burdens.
3. Install the tool `Flake8` for VS code, official identifier: `ms-python.flake8`
4. In its settings, add the line `--max-line-length 135` to the setting `Flake8: Args`



## Tests

If you look for files in the `tests/` directory, they will have examples of how the tested functions are called. You can learn a fair amount from looking at them.

Run all the tests on your system and make sure they pass!. From the `mass` directory, say `pytest`. Tests require that you installed via `pip install -e ...`.

### Auto-run subsets of tests
`pytest-watch --pdb -- mass/off` run from the source directory will run only the tests in `mass/off`, automatically, each time a file is saved. Upon any error, it will drop into pdb for debugging. You will need to `pip install pytest-watch` first.

## Working on docs + tests
Change directory into `doc`, then:

  * For Posix (Mac/Linux) `make doctest html && open _build/html/index.html`
  * For Windows cmd shell `make doctest html && start _build/html/index.html`
  * For Windows Powershell `./make doctest;./make html;start _build/html/index.html`

Read about RST (reStructuredText) format. It is weird. My most common mistake is forgetting the blank line between `.. blah` statements and the following text. See the [Sphinx docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for details about its syntax.

# massGui, the GUI for Mass
massGui is an attempt to bring the core features of Mass into a graphical user interface (GUI). You can find more information and install it at [https://github.com/gmondee/massGui](https://github.com/gmondee/massGui).
