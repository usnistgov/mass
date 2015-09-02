#!/usr/bin/env python
"""
setup.py  distutils build/install file for Mass, the Microcalorimeter Analysis Software Suite

Joe Fowler, NIST Boulder Labs

This setup file must be able to build both FORTRAN and Cython extension modules, which
requires using the numpy+f2py distutils for the former and Cython.Distutils for the
latter.  

I found it not at all clear how to mix the two types until discovering the two recent
discussions:

1. http://stackoverflow.com/questions/7932028/setup-py-for-packages-that-depend-on-both-cython-and-f2py
2. http://answerpot.com/showthread.php?601643-cython%20and%20f2py
"""

import os.path
from distutils.command.build import build as basic_build


def parse_version_number(VERSIONFILE=None):
    # Parse the version number out of the _version.py file without importing it
    import re

    VERSIONFILE = os.path.join("mass", "_version.py")
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

MASS_VERSION = parse_version_number()


def configuration_fortran(parent_package='', top_path=None):
    """Configure FORTRAN extensions only."""
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mass', parent_package, top_path)

    # Extensions in FORTRAN90
    sourcename = os.path.join('mass', 'mathstat', 'factor_covariance')
    config.add_extension('mathstat._factor_covariance',
                         [sourcename + ext for ext in [".pyf", ".f90"]])

    return config


class QtBuilder(basic_build):
    """Subclass the usual distutils builder so that it can convert Qt Designer files
    *.ui and *.rc to python files."""

    @staticmethod
    def compile_ui(ui_file, py_file=None):
        # Search for pyuic4 in python bin dir, then in the $Path.
        if py_file is None:
            py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        try:
            from PyQt4 import uic
            fp = open(py_file, 'w')
            uic.compileUi(ui_file, fp, indent=4)
            fp.close()
            print("compiled", ui_file, "into", py_file)
        except Exception as e:
            print('Unable to compile user interface', e)
            return

    @staticmethod
    def compile_rc(qrc_file, py_file=None):
        # Search for pyuic4 in python bin dir, then in the $Path.
        if py_file is None:
            py_file = os.path.splitext(qrc_file)[0] + "_rc.py"
        if os.system('pyrcc4 "%s" -o "%s"' % (qrc_file, py_file)) > 0:
            print("Unable to generate python module for resource file", qrc_file)

    def run(self):
        # Compile the Qt files to Python files, then call the base class run() method
        for dirpath, _, filenames in os.walk('mass'):
            for filename in filenames:
                if filename.endswith('.ui'):
                    self.compile_ui(os.path.join(dirpath, filename))
                elif filename.endswith('.qrc'):
                    self.compile_rc(os.path.join(dirpath, filename))
        basic_build.run(self)


if __name__ == "__main__":
    import sys
    from distutils.extension import Extension

    import numpy as np
    from numpy.distutils.core import setup
    from Cython.Build import cythonize

    if sys.platform != 'win32':
        setup(configuration=configuration_fortran)

    setup(name='mass',
          version=MASS_VERSION,
          author='Joe Fowler',
          author_email='joe.fowler@nist.gov',
          url='https://bitbucket.org/joe_fowler/mass',
          description='Microcalorimeter Analysis Software Suite',
          packages=['mass', 'mass.core', 'mass.mathstat', 'mass.calibration',
                    'mass.demo', 'mass.gui', 'mass.nonstandard'],
          ext_modules=cythonize([Extension('mass.mathstat._robust',
                                           [os.path.join('mass', 'mathstat', 'robust.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.mathstat.nearest_arrivals',
                                           [os.path.join('mass', 'mathstat', 'nearest_arrivals.pyx')],
                                           include_dirs=[np.get_include()])]),
          package_data={'mass.gui': ['*.ui'],   # Copy the Qt Designer user interface files
                        'mass.calibration': ['nist_xray_data.dat', 'low_z_xray_data.dat']
                        },
          cmdclass={'build': QtBuilder}
          )
