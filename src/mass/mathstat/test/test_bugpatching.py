import mass
import numpy as np
import pylab as plt
import unittest as ut


class TestNumpyHistogramBug(ut.TestCase):
    """There's a bug in numpy 1.13 that can arise when one passes float32 data
    to np.histogram, and a value is just below the lowest bin edge but appears
    to be equal to the edge if tested as a float32. See numpy issues 9189 and
    10319 (https://github.com/numpy/numpy/issues/10319).

    This tests whether the bug exists. MASS attempts to monkey-patch np.histogram
    to prevent the bug from arising, and we are effectively checking that here.
    """
    def test_histogram_precision_bug(self):
        a64 = np.array([25552.234, 26000])

        # Make a histogram from these two values. Put the lower bin limit just above the lower value.
        bin_limits = (a64[0]+0.0005, 26200)
        counts, binedges = np.histogram(a64, 10, bin_limits)
        self.assertEqual(counts[0], 0)

        # That worked, but histogram again with the data converted to lower precision.
        a32 = a64.astype(np.float32)
        a16 = a64.astype(np.float16)

        # ... The following 2 lines raise ValueErrors if numpy has the bug.
        counts, binedges = np.histogram(a32, 10, bin_limits)
        counts, binedges, patches = plt.hist(a32, 10, bin_limits)
        self.assertEqual(counts[0], 0)

        # ... The following 2 lines raise ValueErrors if numpy has the bug.
        counts, binedges = np.histogram(a16, 10, bin_limits)
        counts, binedges, patches = plt.hist(a16, 10, bin_limits)
        self.assertEqual(counts[0], 0)


if __name__ == '__main__':
    ut.main()
