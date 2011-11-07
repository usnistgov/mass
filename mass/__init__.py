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

__all__ = ['channel_group','channel','controller','power_spectrum']

import channel
import channel_group
import controller
import energy_calibration
import files
import fluorescence_lines
import mass_GUI
import power_spectrum
import utilities 
import workarounds

from files import root2ljh_translate_all, root2ljh_translator
from channel_group import TESGroup, CDMGroup


print """The Microcalorimeter Analysis Software System (MASS) is now imported."""
