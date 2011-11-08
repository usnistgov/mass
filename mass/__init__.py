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

__all__ = []

import core
import calibration
import mathstat
#import gui

__all__.extend(core.__all__)
__all__.extend(calibration.__all__)
__all__.extend(mathstat.__all__)

#from core.files import root2ljh_translate_all, root2ljh_translator
#from core.channel_group import TESGroup, CDMGroup


print """The Microcalorimeter Analysis Software System (MASS) is now imported."""
