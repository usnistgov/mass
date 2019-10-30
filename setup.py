#!/usr/bin/env python
"""
setup.py  distutils build/install file for Mass, the Microcalorimeter Analysis Software Suite

Joe Fowler, NIST Boulder Labs
"""

import os.path
from distutils.command.build import build as basic_build

BASEDIR = os.path.dirname(os.path.realpath(__file__))

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs_path = os.path.join(BASEDIR,"requirements.txt")
# apparently parsing the requirements.txt file is not advised see:
# http://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py

reqs = parse_requirements(reqs_path)
# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']


def parse_version_number(VERSIONFILE=None):
    # Parse the version number out of the _version.py file without importing it
    import re

    if not VERSIONFILE:
        VERSIONFILE = os.path.join(BASEDIR, "src", 'mass', "_version.py")

    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


MASS_VERSION = parse_version_number()


def generate_sourceroot_file():
    """We need a file to point back to the root of the source directory"""

    root = os.path.dirname(os.path.abspath(__file__))
    code = """
# An auto-generated file. DO NOT EDIT!

import os.path

sourceroot = r'%s'


def source_file(item=""):
    \"\"\"A function to remember the directory from which mass was installed.\"\"\"
    return os.path.join(sourceroot, item)
""" % root
    with open(os.path.join(BASEDIR,"src", "mass", "demo", "sourceroot.py"), "w") as fp:
        fp.write(code)


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
            print("Compiled '%s' into '%s'"%(ui_file, py_file))
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
        for dirpath, _, filenames in os.walk('src'):
            for filename in filenames:
                if filename.endswith('.ui'):
                    self.compile_ui(os.path.join(dirpath, filename))
                elif filename.endswith('.qrc'):
                    self.compile_rc(os.path.join(dirpath, filename))
        basic_build.run(self)


if __name__ == "__main__":
    import numpy as np
    from Cython.Build import cythonize

    from setuptools import setup
    from setuptools.extension import Extension

    generate_sourceroot_file()

    setup(name='mass',
          version=MASS_VERSION,
          author='Joe Fowler',
          author_email='joe.fowler@nist.gov',
          url='https://bitbucket.org/joe_fowler/mass',
          description='Microcalorimeter Analysis Software Suite',
          packages=['mass', 'mass.core', 'mass.mathstat', 'mass.calibration',
                    'mass.demo', 'mass.gui', 'mass.off'],
          ext_modules=cythonize([Extension('mass.core.cython_channel',
                                           [os.path.join(BASEDIR,'src', 'mass', 'core', 'cython_channel.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.mathstat.robust',
                                           [os.path.join(BASEDIR,'src', 'mass', 'mathstat', 'robust.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.core.analysis_algorithms',
                                           [os.path.join(BASEDIR,'src', 'mass', 'core', 'analysis_algorithms.pyx')],
                                           include_dirs=[np.get_include()]),
                                 Extension('mass.mathstat.entropy',
                                           [os.path.join(BASEDIR,'src', 'mass', 'mathstat', 'entropy.pyx')],
                                           include_dirs=[np.get_include()])
                                 ]),
          package_data={'mass.gui': ['*.ui'],   # Copy the Qt Designer user interface files
                        'mass.calibration': ['nist_xray_data.dat', 'low_z_xray_data.dat']
                        },
          cmdclass={'build': QtBuilder},
          package_dir={'': os.path.join(BASEDIR,'src')},
          scripts=[os.path.join(BASEDIR,"bin","ljh_truncate")],
          install_requires=reqs
          )
