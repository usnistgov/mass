"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.

For a demonstration of some capabilities:
>>> import mass.demo
>>> print mass.demo.helptxt # and then follow the directions


Joe Fowler, NIST Boulder Labs.  November 2010--
"""

# import os

from ._version import __version__, __version_info__
from .core import *
from .calibration import *
from .mathstat import *
from .common import *
