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

def configuration_fortran(parent_package='',top_path=None):
    """Configure FORTRAN extensions only."""
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mass',parent_package,top_path)

    # Extensions in FORTRAN90
    sourcename = os.path.join('mass','mathstat','factor_covariance')
    config.add_extension('mathstat._factor_covariance', 
                         [sourcename+ext for ext in ".pyf",".f90"])
    
    return config



if __name__ == "__main__":
        
    from numpy.distutils.core import setup
    setup(version='0.2.1',
          author='Joe Fowler',
          author_email='joe.fowler@nist.gov',
          url = 'http://dummy.broken.nist.gov/',
          description='Microcalorimeter Analysis Software Suite',
          packages=['mass','mass.core', 'mass.mathstat', 'mass.calibration', 
                    'mass.demo', 'mass.gui']
          )

    setup( configuration=configuration_fortran )


    # Now configure all Cython modules
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    
    setup(
      ext_modules = [Extension('mass.mathstat._robust', 
                               [os.path.join('mass','mathstat','robust')+ext for ext in (".pyx",)])],
      cmdclass = {'build_ext': build_ext},
#      script_args = ['build_ext', '--inplace'],
    )
