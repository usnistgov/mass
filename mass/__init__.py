## \mainpage Overview of Mass
#
# \section summary Summary
#
# This summarizes some things.
# 
# \section install Installation
# 
# \section starting Getting started
# 
# Here is how we get started.
# 
# \section requirements Requirements
# 
# There are some requirements.


## \package  mass
#
# \brief Microcalorimeter Analysis Software Suite
# 
# Python tools to analyze microcalorimeter data offline.

"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.

Joe Fowler, NIST Boulder Labs.  November 2010--
"""

import core
import calibration
import mathstat
#import gui

from core import *
from calibration import *
from mathstat import *

__all__ = []
__all__.extend(core.__all__)
__all__.extend(calibration.__all__)
__all__.extend(mathstat.__all__)


print """The Microcalorimeter Analysis Software System (MASS) is now imported."""
print 'All is ', __all__
