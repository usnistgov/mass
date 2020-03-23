'''
test_robust.py

Test functions in mass.mathstat.robust

Created on Feb 9, 2012

@author: fowlerj
'''

import unittest
import numpy
from mass.mathstat.robust import shorth_range, high_median, Qscale


class Test_Shorth(unittest.TestCase):
    """Test the function shorth_range, which computes the range of the shortest half."""

    def testUnnormalized(self):
        """Verify that you get actual shorth when normalize=False for odd and even size lists."""
        r = shorth_range([1, 4, 6, 8, 11], normalize=False)
        self.assertEqual(r, 4, msg="Did not find shortest half range in length-5 list")

        x = numpy.array([1, 4.6, 6, 8, 11])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        self.assertEqual(r, x[3]-x[1],
                         msg="Did not find shortest half range in length-5 list")
        self.assertEqual(shr_mean, x[1:4].mean(),
                         msg="Did not find shortest half mean in length-5 list")
        self.assertEqual(shr_ctr, 0.5*(x[1]+x[3]),
                         msg="Did not find shortest half center in length-5 list")

        r = shorth_range([2, 4, 6, 8, 11, 15], normalize=False)
        self.assertEqual(r, 6, msg="Did not find shortest half range in length-6 list")

        x = numpy.array([1, 4.6, 6, 8, 11, 100])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        self.assertEqual(r, x[4]-x[1], msg="Did not find shortest half range in length-6 list")
        self.assertEqual(shr_mean, x[1:5].mean(),
                         msg="Did not find shortest half mean in length-6 list")
        self.assertEqual(shr_ctr, 0.5*(x[1]+x[4]),
                         msg="Did not find shortest half center in length-6 list")

    def testSortInplace(self):
        """Verify behavior of the sort_inplace argument."""
        x = [7, 1, 2, 3, 4, 5, 6]
        y = numpy.array(x)
        self.assertIsNotNone(shorth_range(x, sort_inplace=False))
        self.assertIsNotNone(shorth_range(y, sort_inplace=False))
        self.assertEqual(x[0], 7, msg="shorth_range has reordered a list when asked not to.")
        self.assertEqual(y[0], 7, msg="shorth_range has reordered a ndarray when asked not to.")

        # If sort_inplace=True on a non-array, a Value Error is supposed to be raised.
        self.assertRaises(ValueError, shorth_range, x, sort_inplace=True)

        self.assertIsNotNone(shorth_range(y, sort_inplace=True))
        self.assertEqual(y[0], 1,
                         msg="shorth_range has not sorted a ndarray in place when requested to do so.")


class Test_High_Median(unittest.TestCase):
    """"Test the function high_median, which returns the high weighted median of a data set."""

    def testUnweighted(self):
        """Verify high_median when the weights are implied equal to 1"""
        x = numpy.arange(7)
        self.assertEqual(high_median(x), 3, msg="Did not get HM([0,1...6]) = 3.")
        x = numpy.arange(8)
        self.assertEqual(high_median(x), 4, msg="Did not get HM([0,1...,7]) = 4.")
        x = numpy.arange(7, -1, -1)
        self.assertEqual(high_median(x), 4, msg="Did not get HM([7,6,...0]) = 4.")
        x = numpy.array([4])
        self.assertEqual(high_median(x), 4, msg="Did not get HM([4]) = 4.")

    def testWeighted(self):
        """Verify simple cases of high_median."""
        new_order = [3, 0, 1, 4, 2]

        def scramble(x, w):
            return [x[i] for i in new_order], [w[i] for i in new_order]

        x, w = [1, 2, 3, 4, 5], [3, 1, 1, 1, 3]
        self.assertEqual(high_median(x, w), 3,
                         msg="Failed high_median on balanced, odd-summed weights.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 3,
                         msg="Failed high_median on balanced, odd-summed weights.")

        x, w = [1, 2, 3, 4, 5], [3, 1, 2, 1, 3]
        self.assertEqual(high_median(x, w), 3,
                         msg="Failed high_median on balanced, even-summed weights.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 3,
                         msg="Failed high_median on balanced, even-summed weights.")

        x, w = [1, 2, 3, 4, 5], [3, 1, 1, 1, 1]
        self.assertEqual(high_median(x, w), 2,
                         msg="Failed high_median on unbalanced odd-summed weights.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 2,
                         msg="Failed high_median on unbalanced odd-summed weights.")

        x, w = [1, 2, 3, 4, 5], [4, 1, 1, 1, 1]
        self.assertEqual(high_median(x, w), 2,
                         msg="Failed high_median on even-summed weights.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 2,
                         msg="Failed high_median on even-summed weights.")

        x, w = [1, 2, 3, 4, 5], [5, 1, 1, 1, 1]
        self.assertEqual(high_median(x, w), 1, msg="Failed high_median on answer=lowest.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 1, msg="Failed high_median on answer=lowest.")

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 3]
        self.assertEqual(high_median(x, w), 4, msg="Failed high_median on answer=highest-1.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 4, msg="Failed high_median on answer=highest-1.")

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 4]
        self.assertEqual(high_median(x, w), 5, msg="Failed high_median on answer=highest.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 5, msg="Failed high_median on answer=highest.")

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 5]
        self.assertEqual(high_median(x, w), 5, msg="Failed high_median on answer=highest.")
        x, w = scramble(x, w)
        self.assertEqual(high_median(x, w), 5, msg="Failed high_median on answer=highest.")


class Test_Qscale(unittest.TestCase):
    """Test the Qscale() statistic."""

    def testSimple(self):
        x = numpy.array([1, 4, 5, 6, 8], dtype=numpy.float)
        self.assertAlmostEqual(Qscale(x), 2.0*2.2219*.844, 3)

    def testInplace(self):
        _ = Qscale([1, 2, 3, 4], sort_inplace=False)
        self.assertRaises(ValueError, Qscale, [1, 2, 3, 4], sort_inplace=True)

        x = numpy.array([4, 3, 2, 1])
        _ = Qscale(x, sort_inplace=False)
        self.assertEqual(x[0], 4, msg="Qscale sorted data when asked not to.")
        _ = Qscale(x, sort_inplace=True)
        self.assertEqual(x[3], 4, msg="Qscale did not sort data when asked to sort.")

    def Qslow(self, x):
        """Compute Q the simple, slow way, an O(n^2) calculation to verify the fast one."""
        x = numpy.array(x)
        x.sort()
        n = len(x)

        prefactor = 2.2219
        if n <= 9:
            prefactor *= [0, 0, 0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872][n]
        else:
            if n % 2 == 1:
                prefactor *= n/(n+1.4)
            else:
                prefactor *= n/(n+3.8)

        dist = numpy.hstack([x[j]-x[:j] for j in range(1, n)])
        dist.sort()
        h = n // 2 + 1
        k = h*(h-1) // 2 - 1
        return prefactor * dist[k]

    def testRandom(self):
        """Test some random (normal) data to be sure that the fast and Qslow
        operations give the same result.
        """
        numpy.random.seed(0)  # improve test repeatability
        for size in (3, 6, 9, 15, 20, 25, 30, 40, 45, 50, 75, 100, 140):
            data = numpy.random.standard_normal(size=size)
            qs = self.Qslow(data)
            qf = Qscale(data)
            self.assertIsNotNone(qs)
            self.assertIsNotNone(qf)
            self.assertAlmostEqual(qs, qf, 5)


if __name__ == "__main__":
    unittest.main()
