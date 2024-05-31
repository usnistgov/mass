"""
Mass: a Microcalorimeter Analysis Software Suite

Python tools to analyze microcalorimeter data offline.


Joe Fowler, NIST Boulder Labs.  November 2010--
"""

# ruff: noqa: F403, F401
# flake8: noqa: F403, F401


try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from .core import *
from .calibration import *
from .mathstat import *
from .common import *
