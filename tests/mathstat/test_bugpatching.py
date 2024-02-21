import numpy as np
import pylab as plt


class TestNumpyHistogramBug:
    """There was a bug in numpy 1.13 (and earlier?) that can arise when
    one passes float32 data to np.histogram, and a value is just below the
    lowest bin edge but appears to be equal to the edge if tested as a float32.
    See numpy issues 9189 and 10319 (https://github.com/numpy/numpy/issues/10319).

    This tests whether the bug exists. (We require numpy 1.14+, so bug should
    not still be present.)
    """

    def test_histogram_precision_bug(self):
        a64 = np.array([25552.234, 26000])

        # Make a histogram from these two values. Put the lower bin limit just above the lower value.
        bin_limits = (a64[0] + 0.0005, 26200)
        counts, binedges = np.histogram(a64, 10, bin_limits)
        assert counts[0] == 0

        # That worked, but histogram again with the data converted to lower precision.
        a32 = a64.astype(np.float32)

        # ... The following 2 lines raise ValueErrors if numpy has the bug and
        # the mass patch fails to fix it.
        counts, binedges = np.histogram(a32, 10, bin_limits)
        counts, binedges, patches = plt.hist(a32, 10, bin_limits)
