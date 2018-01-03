"""
Fix numpy bug 10319 by monkey-patching, if it's real.

See https://github.com/numpy/numpy/issues/10319 and pull request 9189, too.
"""

import numpy as np
import pylab as plt

def is_histogram_bug_there():
    """
    Does the imported version of numpy have the histogram bug for single-precision
    floating point input data?
    """
    # Make a histogram from these two values. Put the lower bin limit just above the lower value.
    a64=np.array([25552.234, 26000])
    bin_limits = (a64[0]+0.0005, 26200)

    # Histogramming the float32 version of the data can raise an unwanted ValueError.
    a32=a64.astype(np.float32)
    try:
        counts, binedges = np.histogram(a32, 10, bin_limits)
        return False
    except ValueError:
        return True


def histogram(a, bins=10, range=None, normed=False, weights=None,
              density=None):
    """MONKEY-PATCHED VERSION OF np.histogram to fix bug #10319.
    Works by converting input data from float32/float16 to float (if possible,
    do the conversion only after selecting in-bounds values).
    """
    a_corrected = np.asarray(a)
    if a_corrected.dtype in (np.float32, np.float16):
        if isinstance(bins, basestring) or range is None:
            # For automatic bin computation, convert all values to float
            a_corrected = a_corrected.astype(np.float)
        else:
            # When bins or bin number+range is given, convert only the in-bounds
            # values to float.
            if np.iterable(bins):
                keep = a_corrected >= bins[0]
                keep &= a_corrected < bins[-1]
            else:
                keep = a_corrected >= range[0]
                keep &= a_corrected < range[1]
            a_corrected = a_corrected[keep].astype(np.float)
    return np._old_histogram(a_corrected, bins=bins, range=range, normed=normed,
                             weights=weights, density=density)

# If the bug exists, replace calls to np.histogram with safe_histogram.
if is_histogram_bug_there():
    histogram.__doc__ = "".join([histogram.__doc__, np.histogram.__doc__])
    np._old_histogram = np.histogram
    np.histogram = histogram
    plt.histogram = histogram
