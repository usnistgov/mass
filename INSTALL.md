# MASS: The Microcalorimeter Analysis Software Suite

### Joe Fowler, Galen O'Neil, and Young-Il Joe, NIST Boulder Labs

The MASS project dates from November 2010-present

This file last updated: April 10, 2017

## Contents

* How to download Mass
* How to install Mass
  * Possible installation problems
* How to import Mass within Python/Ipython
* Package dependencies
* Matplotlib preferences


## How to Download MASS

At the moment, MASS is available on a private repository at bitbucket.
Ask Joe Fowler to join the MASS users group (you'll need to have a bitbucket
account already, or you'll get one when you join that users group).

Create the directory where you want to have the mass source live,
then clone a git repository into it.

```bash
cd ~/software/awesome_software
git clone git@bitbucket.org:joe_fowler/mass.git

# To install:
cd mass
python setup.py build
sudo python setup.py install
```



## How to Install MASS

Mass is a  mostly Python system for analyzing microcalorimeter data.
A small amount of code has been written in Cythonfor the
execution speed advantages of compiled code.

The preferred approach is simply to use python setuptools to
copy the Mass code to the known, standard install location.  You will
probably need sudo power to do this, but the process is otherwise
very simple.  From the top-level Mass directory (where you find this
file), you need only two commands:

```bash
python setup.py build
sudo python setup.py install
```
NOTE FOR ANACONDA USERS, DO NOT USE SUDO.  INSTEAD: `python setup.py install`

(You could omit the build step, but in that event, it will be done
implicitly while you are sudoing.  The result is that your build
directory gets filled with build files *owned by root*, which is
annoying if not terrible.)

If you wish to test Mass without installing, or if you wish to play
with the latest edition from subversion, then I suggest that you
modify your PYTHONPATH shell variable.  The Mass code lives in
./mass/ (were "." refers to the location of this INSTALL file).
You just need your PYTHONPATH to point to this.

One problem with running a test version of Mass is that the compiled
extensions do not get built into the "live" tree.  One possible workaround:

```bash
export PYTHONPATH=${PYTHONPATH}:~/where_i_keep_software/mass/build/lib*/mass/mathstat/
```

Or another is just to copy the compiled extensions into the live tree:

```bash
cp build/lib*/mass/mathstat/*so mass/mathstat/
```

Let me know what works for you!


### Installation on Ubuntu
The following should get you the required packages:

```
sudo apt-get install python-qt4 ipython python-numpy python-matplotlib \
    python-scipy pyqt4-dev-tools cython gfortran python-sklearn python-h5py \
    python-statsmodels
```


### Possible Installation Problems

There is no limit to the possibilities.  One I have found, which seems like
it ought to have an obvious solution, is that my `setup.py` script tends to
install SOME (but not all) of the *.py files as world-readable.  If any
py file is root-owned and world-UNreadable, then you have a problem.

A workaround is to run `sudo chown -R a+rX .../site-packages/mass`, but
for this you'll have to figure out where your Python site-packages are.
I believe that the Makefile target "install" does this correctly, so that
you can just say `make build install`.  I have only tested this on macports,
but it ought to work on Linux, too.  `make report_install_location` will
tell you where your python site-packages is located.

Beware that Python can reload pure Python code from the interpreter, e.g.,
reload(mass.mathstat)`.  But it cannot reload compiled code generated
from Cython, C, C++, or FORTRAN.  If you change any Cython or FORTRAN
code, rebuild, and reinstall, then those changes will be invisible to any
python session already in progress.  I'm sorry, but that's a limitation
far above my level of expertise.

With Macports 2.1.2 we have seen the following error when trying to build mass:
```
mass/mathstat/robust.c:253:10: fatal error: 'numpy/arrayobject.h' file not found
#include "numpy/arrayobject.h"
         ^
1 error generated.
```

The ugly hack to fix this is to create the following symlink:

```bash
export TMPROOTDIR=/opt/local/Library/Frameworks/Python.framework/Versions/2.7
ln -s ${TMPROOTDIR}/lib/python2.7/site-packages/numpy/core/include/numpy/ \
    ${TMPROOTDIR}/include/python2.7/numpy
```

Frank Schima believes this is a bug in the Cython project.


## How to import MASS within Python or Ipython

```python
>>> import mass
>>> import mass.gui # < this bit is optional
```

Note that `mass.gui` can be used only if your matplotlib "backend" is Qt4. See
your `~/.matplotlibrc` file (or create one) for further information on that.




## Package Dependencies

Mass has a few required packages, though I expect they are mostly things
that any physicist would already have when using Python for scientific
work:

- numpy
- scipy
- matplotlib (a.k.a. pylab)
- cython
- Qt4 (optional)
- PyQt4 (optional)

The Qt4 packages are optional in the sense that the optional GUI is
built on PyQt4.  You can certainly use mass without the GUI, in which
case the matplotlib plotting package is free to use a non-Qt backend.
I would not do this is you don't have to.  As of late April 2012, I
finally added a few useful GUI components.  I hope more will come in time.

I am sure that Mass uses certain features of each package that are
only available starting with certain version numbers.  Unfortunately,
I have no idea what they are.  I can say that at the time of this
writing (April 10, 2017), my own computer uses versions:

- python     2.7.13
- ipython    5.3.0
- numpy      1.12.1
- scipy      0.19.0
- matplotlib 2.0.0
- cython	  0.25.2
- PyQt4      4.12.0
- Qt4        4.8.7
- gcc        6.3.0  (known to work with gcc 4.5-4.8 and gcc 5, too)

If you are using macports and lack a C compiler, or if your compiler
is not "selected", then you might want one or both of these commands,
without some have reported that they could not install Mass:

``` bash
sudo port install gcc6
sudo port select gcc mp-gcc6
```


## Matplotlib preferences

The Qt4 GUI features will conflict with your matplotlib GUI
unless you are using Qt4 as your "backend" for matplotlib. (Event-driven
frameworks like Qt and Tk are incompatible and cannot both be run at once.)
You can specify the choice of Qt4 by using the following line in your configuration file,
which is generally ~/.matplotlib/matplotlibrc :

```
backend     : Qt4Agg

# Here are some other things I find very useful in that file but which
# have nothing to do with Mass:

interactive : True
keymap.grid : g                     # switching on/off a grid in current axes
keymap.yscale : l                   # toggle scaling of y-axes ('log'/'linear')
keymap.xscale : L, k                # toggle scaling of x-axes ('log'/'linear')
```
