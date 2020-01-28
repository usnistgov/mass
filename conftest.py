"""
Configuration file for pytest
"""

# import pytest
import warnings
import logging
import matplotlib

# Suppress matplotlib warnings during tests. See
# https://stackoverflow.com/questions/55109716/c-argument-looks-like-a-single-numeric-rgb-or-rgba-sequence
# from matplotlib.axes._axes import _log as matplotlib_axes_logger
# matplotlib_axes_logger.setLevel('ERROR')

matplotlib.use("svg")  # set to common backend so will run on semphora ci with fewer dependencies
warnings.filterwarnings("ignore")

# Raise the logging threshold, to reduce extraneous output during tests
LOG = logging.getLogger("mass")
LOG.setLevel(logging.ERROR)
