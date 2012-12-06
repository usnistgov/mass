
__all__ = ['load_data_dialog', 'make_cuts_dialog']



# The following magic will open and close a figure, which suffices
# to start the Qt4 event loop and prevent crashes.
import pylab, matplotlib.backends
assert 'qt4' in matplotlib.backends.backend
pylab.figure(56)
pylab.close(56)

import load_data_dialog
import make_cuts_dialog

from load_data_dialog import *
from make_cuts_dialog import *

__all__.extend(load_data_dialog.__all__)
__all__.extend(make_cuts_dialog.__all__)
