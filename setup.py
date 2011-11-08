#!/usr/bin/env python
#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#import numpy.distutils
#
#ext_modules = []
#
## The following did NOT work 11/4/11
##ext1 = Extension(name='covar',
##                 language='fortran',
##                 sources=['src/extensions/covar.f90','src/extensions/covar.pyf'])
#
## Hmm, also no good!
#ext2 = numpy.distutils.misc_util.Configuration('factor_covariance','mass.math',
#                                               top_path='mass/math')
#ext2.add_extension('covar', sources=['covar.pyf','covar.f90'])
#
## The following DOES work...but disable for now since it's not a core part of mass 11/4/11
##ext_modules.append( Extension(name = "mass.solve_R2",
##                 sources=["src/extensions/solve_R2.pyx"]))
#
#setup(name="mass",
#      version='0.2',
#      author='Joe Fowler',
#      author_email='joe.fowler@nist.gov',
#      description='Microcalorimeter Analysis Software Suite',
#      packages=['mass.core', 'mass.math', 'mass.calibration', 'mass.gui'],
##      package_dir={'mass':'mass'},
#      cmdclass = {'build_ext':build_ext},
#      ext_modules = ext_modules,
#      package_data={'mass':['math/*.so']}
#      )


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mass',parent_package,top_path)
    config.add_extension('mathstat._factor_covariance', ['mass/mathstat/factor_covariance.f90'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name="mass",
          version='0.2',
          author='Joe Fowler',
          author_email='joe.fowler@nist.gov',
          description='Microcalorimeter Analysis Software Suite',
          packages=['mass','mass.core', 'mass.mathstat', 'mass.calibration', 'mass.gui'],
          configuration=configuration)
