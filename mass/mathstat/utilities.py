"""
mass.utilities

Several math utilities, including:
* A mouse click capturer for mouse feedback from plots.

Joe Fowler, NIST

Started March 24, 2011
"""

## \file utilities.py
# \brief Several utilities used by Mass, including math, plotting, and other functions.
#
# Other utilities:
# -# plot_as_stepped_hist, to draw an already computed histogram in the same way that
#    pylab.hist() would do it.
# -# MouseClickReader, to capture pointer location on a plot.

__all__ = ['plot_as_stepped_hist', 'MouseClickReader']


import numpy

class MissingLibrary(object):
    """Class to raise ImportError only after python tries to use the import.
    Intended for use with shared objects built from Fortran or Cython source."""
    def __init__(self, libname):
        self.libname = libname
        self.error = ImportError( """This copy of Mass could not import the compiled '%s'
This happens when you run from a source tree, among other possibilities.  You can
either try using an installed version or do a 'python setup.py build' and copy
the .so file from build/lib*/mass/mathstat/ to mass/mathstat/  Note that this is 
a delayed error.  If it is raised, then you know that you needed the library!"""%self.libname)
    def __getattr__(self, attr):
        raise self.error


def plot_as_stepped_hist(axis, bin_ctrs, data, **kwargs):
    """Plot onto <axis> the histogram <bin_ctrs>,<data> in stepped-histogram format.
    \param axis     The pylab Axes object to plot onto.
    \param bin_ctrs An array of bin centers.  (Bin spacing will be inferred from the first two).
    \param data     Bin contents.   data and bin_ctrs will only be used to the shorter of the two arrays.
    \param kwargs   All other keyword arguments will be passed to axis.plot().
    """
    x = numpy.zeros(2+2*len(bin_ctrs), dtype=numpy.float)
    y = numpy.zeros_like(x)
    dx = bin_ctrs[1]-bin_ctrs[0]
    x[0:-2:2] = bin_ctrs-dx*.5
    x[1:-2:2] = bin_ctrs-dx*.5
    x[-2:] = bin_ctrs[-1]+dx*.5
    y[1:-1:2] = data
    y[2:-1:2] = data
    axis.plot(x, y, **kwargs)
    axis.set_xlim([x[0],x[-1]])



class MouseClickReader(object):
    """Object to serve as a callback for reading mouse clicks in data coordinates
    in pylab plots.  Will store self.b, .x, .y giving the button pressed,
    and the x,y data coordinates of the pointer when clicked.
    
    Usage example (ought to be here...):
    """
    def __init__(self, figure):
        """Connect to button press events on a pylab figure.
        \param figure The matplotlib.figure.Figure from which to capture mouse click events."""
        ## The button number of the last mouse click inside a plot.
        self.b = 0
        ## The x location of the last mouse click inside a plot.
        self.x = 0
        ## The y location of the last mouse click inside a plot.
        self.y = 0
        ## The Figure to whose events we are connected.
        self.fig=figure
        ## The connection ID for matplotlib event handling.
        self.cid = self.fig.canvas.mpl_connect('button_press_event',self)
        
    def __call__(self, event):
        """When called, capture the latest button number and the x,y location in 
        plot units.  Store in self.b, .x, and .y."""
        self.b, self.x, self.y =  event.button, event.xdata, event.ydata
    def __del__(self):
        """Disconnect the button press event from this object."""
        self.fig.canvas.mpl_disconnect(self.cid)
        
