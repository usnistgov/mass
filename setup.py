#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = []

# The following did NOT work 11/4/11
#ext1 = Extension(name='covar',
#                 language='fortran',
#                 sources=['src/extensions/covar.f90','src/extensions/covar.pyf'])

# The following DOES work...but disable for now since it's not a core part of mass 11/4/11
#ext_modules.append( Extension(name = "mass.solve_R2",
#                 sources=["src/extensions/solve_R2.pyx"]))

setup(name="mass",
      version='0.2',
      author='Joe Fowler',
      author_email='joe.fowler@nist.gov',
      description='Microcalorimeter Analysis Software Suite',
      packages=['mass'],
      package_dir={'mass':'src/mass'},
      cmdclass = {'build_ext':build_ext},
      ext_modules = ext_modules,
      )
