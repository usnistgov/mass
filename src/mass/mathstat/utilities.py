"""
mass.mathstat.utilities

Several math and plotting utilities:
* plot_as_stepped_hist
* plot_stepped_hist_poisson_errors
* savitzky_golay
* CheckForMissingLibrary class

Joe Fowler, NIST

Started March 24, 2011
"""

import numpy as np

__all__ = ['plot_as_stepped_hist', 'plot_stepped_hist_poisson_errors', 'savitzky_golay',
           'find_svd_randomly', 'find_range_randomly']


class CheckForMissingLibrary(object):
    """Class to raise ImportError only after python tries to use the import.

    Intended for use with shared objects built from Fortran or Cython source.
    """
    def __init__(self, libname):
        self.libname = libname
        self.error = ImportError("""This copy of Mass could not import the compiled '%s'
This happens when you run from a source tree, among other possibilities.  You can
either try using an installed version or do a 'python setup.py build' and copy
the .so file from build/lib*/mass/mathstat/ to mass/mathstat/  Note that this is
a delayed error.  If it is raised, then you know that you needed the library!""" % self.libname)

    def __getattr__(self, attr):
        raise self.error


def plot_as_stepped_hist(axis, data, bins, **kwargs):
    """Plot data in stepped-histogram format.

    Args:
        axis: The pylab Axes object to plot onto.
        data: Bin contents.
        bins: An array of bin centers or of bin edges.  (Bin spacing will be
            inferred from the first two elements).  If len(bin_ctrs) == len(data)+1, then
            bin_ctrs will be assumed to be bin edges; otherwise it will be assumed to be
            the bin centers.
        **kwargs: All other keyword arguments will be passed to axis.plot().
    """
    if len(bins) == len(data)+1:
        bin_edges = bins
        x = np.zeros(2*len(bin_edges), dtype=np.float)
        x[0::2] = bin_edges
        x[1::2] = bin_edges
    else:
        x = np.zeros(2+2*len(bins), dtype=np.float)
        dx = bins[1]-bins[0]
        x[0:-2:2] = bins-dx*.5
        x[1:-2:2] = bins-dx*.5
        x[-2:] = bins[-1]+dx*.5

    y = np.zeros_like(x)
    y[1:-1:2] = data
    y[2:-1:2] = data
    axis.plot(x, y, **kwargs)
    axis.set_xlim([x[0], x[-1]])


def plot_stepped_hist_poisson_errors(axis, counts, bin_ctrs, scale=1.0, offset=0.0, **kwargs):
    """Use plot_as_stepped_hist to plot a histogram, where also
    an error band is plotted, assuming data are Poisson-distributed.

    Args:
        axis: The pylab Axes object to plot onto.
        data: Bin contents.
        bin_ctrs: An array of bin centers or of bin edges.  (Bin spacing will be
            inferred from the first two elements).  If len(bin_ctrs) == len(data)+1, then
            bin_ctrs will be assumed to be bin edges; otherwise it will be assumed to be
            the bin centers.
        scale: Plot counts*scale+offset if you need to convert counts to some physical units.
        offset: Plot counts*scale+offset if you need to convert counts to some physical units.
        **kwargs: All other keyword arguments will be passed to axis.plot().
    """
    if len(bin_ctrs) == len(counts)+1:
        bin_ctrs = 0.5*(bin_ctrs[1:]+bin_ctrs[:-1])
    elif len(bin_ctrs) != len(counts):
        raise ValueError("bin_ctrs must be either the same length as counts, or 1 longer.")
    smooth_counts = savitzky_golay(counts*scale, 7, 4)
    errors = np.sqrt(counts)*scale
    fill_lower = smooth_counts-errors
    fill_upper = smooth_counts+errors
    fill_lower[fill_lower < 0] = 0
    fill_upper[fill_upper < 0] = 0
    axis.fill_between(bin_ctrs, fill_lower+offset, fill_upper+offset, alpha=0.25, **kwargs)
    plot_as_stepped_hist(axis, counts*scale+offset, bin_ctrs, **kwargs)


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
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as _msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size-1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m, y, mode='valid')


def find_range_randomly(A, nl, q=1):
    """Find approximate range of matrix A using nl random vectors and
    with q power iterations (q>=0). Based on Halko Martinsson & Tropp Algorithm 4.3

    Suggest q=1 or larger particularly when the singular values of A decay slowly
    enough that the singular vectors associated with the smaller singular values
    are interfering with the computation.

    See "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions." by N Halko, P Martinsson, and J Tropp. *SIAM
    Review* v53 #2 (2011) pp217-288. http://epubs.siam.org/doi/abs/10.1137/090771806
    """
    if q < 0:
        msg = "The number of power iterations q=%d needs to be at least 0"%q
        raise ValueError(msg)
    A = np.asarray(A)
    m,n = A.shape
    Omega = np.random.standard_normal((n,nl))
    Y = np.dot(A, Omega)
    for _ in range(q):
        Y = np.dot(A.T, Y)
        Y = np.dot(A, Y)
    Q,R = np.linalg.qr(Y)
    return Q

def find_svd_randomly(A, nl, q=2):
    """Find approximate SVD of matrix A using nl random vectors and
    with q power iterations. Based on Halko Martinsson & Tropp Algorithm 5.1

    See "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions." by N Halko, P Martinsson, and J Tropp. *SIAM
    Review* v53 #2 (2011) pp217-288. http://epubs.siam.org/doi/abs/10.1137/090771806
    """
    Q = find_range_randomly(A, nl, q=q)
    B = np.dot(Q.T, A)
    u_b,w,v = np.linalg.svd(B, 0)
    u = np.dot(Q, u_b)
    return u,w,v
