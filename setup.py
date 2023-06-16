#!/usr/bin/env python
"""
setup.py  Build/install file for Mass, the Microcalorimeter Analysis Software Suite

Python version note: this package contains some Python "f-strings", a system for string
formatting that was introduced in Python 3.6. That means MASS won't work on Python 2 or any
other version < 3.6. As of 13 September 2020, all versions of Python older than 3.6 were
considered beyond end-of-life, so the authors of MASS think this a reasonable limitation.

Joe Fowler and Galen O'Neil, NIST Boulder Labs
"""

import os.path
import numpy as np

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

BASEDIR = os.path.dirname(os.path.realpath(__file__))

requirements = ["numpy>=1.11", "scipy>=0.19", "Cython", "pandas", "scikit-learn",
                "h5py>=2.7", "palettable", "cycler", "fastdtw", "progress", "lmfit>=0.9.11", "pytest",
                "uncertainties", "dill", "xraydb", "matplotlib>1.5", "statsmodels>0.8",
                "packaging", "deprecated"]

# dill 0.3.6 breaks our recipebook saving test, so for a while we pinned to 0.3.5.1
# See https://bitbucket.org/joe_fowler/mass/pipelines/results/1037, which fails
# and https://bitbucket.org/joe_fowler/mass/pipelines/results/1033,
# which succeeds on the same commit. We are disabling that test for now, so we can
# lift the dill version requirement.


def parse_version_number(VERSIONFILE=None):
    # Parse the version number out of the _version.py file without importing it
    import re

    if not VERSIONFILE:
        VERSIONFILE = os.path.join(BASEDIR, 'mass', "_version.py")

    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


MASS_VERSION = parse_version_number()


def generate_sourceroot_file():
    """We need a file to point back to the root of the source directory. This is needed only
    for the demos, and it wouldn't be neccesary when installed with `pip -e`.
    """

    root = os.path.dirname(os.path.abspath(__file__))
    code = """
# An auto-generated file. DO NOT EDIT!

import os.path

sourceroot = r'%s'


def source_file(item=""):
    \"\"\"A function to remember the directory from which mass was installed.\"\"\"
    return os.path.join(sourceroot, item)
""" % root
    with open(os.path.join(BASEDIR, "mass", "demo", "sourceroot.py"), "w") as fp:
        fp.write(code)


if __name__ == "__main__":
    generate_sourceroot_file()

    setup(name='mass',
          version=MASS_VERSION,
          author='Joe Fowler',
          author_email='joe.fowler@nist.gov',
          url='https://bitbucket.org/joe_fowler/mass',
          description='Microcalorimeter Analysis Software Suite',
          python_requires=">=3.6",  # See this file's docstring for discussion.
          packages=['mass', 'mass.core', 'mass.mathstat', 'mass.calibration',
                    'mass.demo', 'mass.off', 'mass.materials'],
          ext_modules=cythonize([Extension('mass.core.cython_channel',
                                           [os.path.join(BASEDIR, 'mass',
                                                         'core', 'cython_channel.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.mathstat.robust',
                                           [os.path.join(BASEDIR, 'mass',
                                                         'mathstat', 'robust.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.core.analysis_algorithms',
                                           [os.path.join(BASEDIR, 'mass', 'core',
                                                         'analysis_algorithms.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.mathstat.entropy',
                                           [os.path.join(BASEDIR, 'mass',
                                                         'mathstat', 'entropy.pyx')],
                                           include_dirs=[np.get_include()])
                                 ],
                                compiler_directives={'language_level': "3"}),
          # Installs non .py files that are needed. We could make tests pass in non-develop
          # mode by installing test required files here.
          package_data={'mass.calibration': ['nist_xray_data.dat', 'low_z_xray_data.dat', 'nist_asd.pickle']
                        },
          package_dir={'mass': "mass"},
          install_requires=requirements,
          scripts=["bin/ljh_truncate"],
          entry_points={
              'console_scripts': ['ljh2off=mass.core.ljh2off:main',
                                  'make_projectors=mass.core.projectors_script:main'], }
          )
