## \mainpage Overview of Mass
#
# \section summary Summary
#
# 
# \section install Installation
# 
# \section starting Getting started
# 
# here is how we get started.
# 
# \section requirements Requirements
# 
# 


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
from core import *
import calibration
from calibration import *
import math
from math import *
import gui

from core.files import root2ljh_translate_all, root2ljh_translator
from core.channel_group import TESGroup, CDMGroup


print """The Microcalorimeter Analysis Software System (MASS) is now imported."""
