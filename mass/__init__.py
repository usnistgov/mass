"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.


Joe Fowler, NIST Boulder Labs.  November 2010--
"""

# ruff: noqa: F403, F401

# This is the unique source of truth about the version number (since May 26, 2023)
# [Recommendation 1 in https://packaging.python.org/en/latest/guides/single-sourcing-package-version/]
__version__ = "0.8.4pre1"

from .core import *
from .calibration import *
from .mathstat import *
from .common import *
