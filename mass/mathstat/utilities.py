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

__all__ = ['plot_as_stepped_hist', 'plot_stepped_hist_poisson_errors', 'savitzky_golay', 'MouseClickReader']


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


def plot_as_stepped_hist(axis, data, bins, **kwargs):
    """Plot onto <axis> the histogram <bin_ctrs>,<data> in stepped-histogram format.
    \param axis     The pylab Axes object to plot onto.
    \param data     Bin contents.
    \param bins     An array of bin centers or of bin edges.  (Bin spacing will be 
                    inferred from the first two elements).  If len(bin_ctrs)==len(data)+1, then
                    bin_ctrs will be assumed to be bin edges; otherwise it will be assumed centers.
    \param kwargs   All other keyword arguments will be passed to axis.plot().
    """
    if len(bins)==len(data)+1:
        bin_edges = bins
        x = numpy.zeros(2*len(bin_edges), dtype=numpy.float)
        x[0::2] = bin_edges
        x[1::2] = bin_edges
    else:
        x = numpy.zeros(2+2*len(bins), dtype=numpy.float)
        dx = bins[1]-bins[0]
        x[0:-2:2] = bins-dx*.5
        x[1:-2:2] = bins-dx*.5
        x[-2:] = bins[-1]+dx*.5

    y = numpy.zeros_like(x)
    y[1:-1:2] = data
    y[2:-1:2] = data
    axis.plot(x, y, **kwargs)
    axis.set_xlim([x[0],x[-1]])



def plot_stepped_hist_poisson_errors(axis, counts, bin_ctrs, scale=1.0, **kwargs):
    """Use plot_as_stepped_hist to plot a histogram of <counts>*<scale>, where also
    an error band is plotted, assuming <counts> are Poisson-distributed variates.
    \param axis     The pylab Axes object to plot onto.
    \param counts   Bin contents.
    \param bin_ctrs An array of bin centers or of bin edges.  (Bin spacing will be 
                    inferred from the first two elements).  If len(bin_ctrs)==len(data)+1, then
                    bin_ctrs will be assumed to be bin edges; otherwise it will be assumed centers.
    \param scale    Plot counts*scale if you need to convert counts to some physical units.
    \param kwargs   All other keyword arguments will be passed to axis.plot().
    """
    if len(bin_ctrs)==len(counts)+1:
        bin_ctrs = 0.5*(bin_ctrs[1:]+bin_ctrs[:-1])
    elif len(bin_ctrs) != len(counts):
        raise ValueError("bin_ctrs must be either the same length as counts, or 1 longer.")
    smooth_counts = savitzky_golay(counts, 7, 4)
    errors = numpy.sqrt(counts)*scale
    axis.fill_between(bin_ctrs, smooth_counts-errors, smooth_counts+errors, alpha=0.25, **kwargs)
    plot_as_stepped_hist(axis, counts*scale, bin_ctrs, **kwargs)



def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = numpy.abs(numpy.int(window_size))
        order = numpy.abs(numpy.int(order))
    except ValueError, _msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = numpy.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = numpy.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve( m, y, mode='valid')


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
        
