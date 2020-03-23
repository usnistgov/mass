"""
mass.gui module

Joe Fowler, NIST Boulder Labs

Contains various objects to simplify life with a (Qt4) GUI.
"""

import pylab
import matplotlib.backends

import mass.gui.load_data_dialog
import mass.gui.make_cuts_dialog

from mass.gui.load_data_dialog import *
from mass.gui.make_cuts_dialog import *

# The following magic will open a figure, which suffices
# to start the Qt4 event loop and prevent crashes.
# I was trying to close it, too, but that led to a RuntimeError.
assert 'Qt4' in matplotlib.backends.backend
pylab.figure(1)

__all__ = ['load_data_dialog', 'make_cuts_dialog']

__all__.extend(load_data_dialog.__all__)
__all__.extend(make_cuts_dialog.__all__)
