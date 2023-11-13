'''
test_robust.py

Test functions in mass.mathstat.robust

Created on Feb 9, 2012

@author: fowlerj
'''

import pytest
import numpy as np
from mass.mathstat.robust import shorth_range, high_median, Qscale


class Test_Shorth:
    """Test the function shorth_range, which computes the range of the shortest half."""

    def test_unnormalized(self):
        """Verify that you get actual shorth when normalize=False for odd and even size lists."""
        r = shorth_range([1, 4, 6, 8, 11], normalize=False)
        assert r == 4, "Did not find shortest half range in length-5 list"

        x = np.array([1, 4.6, 6, 8, 11])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        assert r == x[3]-x[1], \
            "Did not find shortest half range in length-5 list"
        assert shr_mean == x[1:4].mean(), \
            "Did not find shortest half mean in length-5 list"
        assert shr_ctr == 0.5*(x[1]+x[3]), \
            "Did not find shortest half center in length-5 list"

        r = shorth_range([2, 4, 6, 8, 11, 15], normalize=False)
        assert r == 6, "Did not find shortest half range in length-6 list"

        x = np.array([1, 4.6, 6, 8, 11, 100])
        r, shr_mean, shr_ctr = shorth_range(x, normalize=False, location=True)
        assert r == x[4]-x[1], "Did not find shortest half range in length-6 list"
        assert shr_mean == x[1:5].mean(), \
            "Did not find shortest half mean in length-6 list"
        assert shr_ctr == 0.5*(x[1]+x[4]), \
            "Did not find shortest half center in length-6 list"

    def test_sort_inplace(self):
        """Verify behavior of the sort_inplace argument."""
        x = [7, 1, 2, 3, 4, 5, 6]
        y = np.array(x)
        assert shorth_range(x, sort_inplace=False) is not None
        assert shorth_range(y, sort_inplace=False) is not None
        assert x[0] == 7, "shorth_range has reordered a list when asked not to."
        assert y[0] == 7, "shorth_range has reordered a ndarray when asked not to."

        # If sort_inplace=True on a non-array, a Value Error is supposed to be raised.
        pytest.raises(ValueError, shorth_range, x, sort_inplace=True)

        assert shorth_range(y, sort_inplace=True) is not None
        assert y[0] == 1, \
            "shorth_range has not sorted a ndarray in place when requested to do so."


class Test_High_Median:
    """"Test the function high_median, which returns the high weighted median of a data set."""

    def test_unweighted(self):
        """Verify high_median when the weights are implied equal to 1"""
        x = np.arange(7)
        assert high_median(x) == 3, "Did not get HM([0,1...6]) = 3."
        x = np.arange(8)
        assert high_median(x) == 4, "Did not get HM([0,1...,7]) = 4."
        x = np.arange(7, -1, -1)
        assert high_median(x) == 4, "Did not get HM([7,6,...0]) = 4."
        x = np.array([4])
        assert high_median(x) == 4, "Did not get HM([4]) = 4."

    def test_weighted(self):
        """Verify simple cases of high_median."""
        new_order = [3, 0, 1, 4, 2]

        def scramble(x, w):
            return [x[i] for i in new_order], [w[i] for i in new_order]

        x, w = [1, 2, 3, 4, 5], [3, 1, 1, 1, 3]
        assert high_median(x, w) == 3, \
            "Failed high_median on balanced, odd-summed weights."
        x, w = scramble(x, w)
        assert high_median(x, w) == 3, \
            "Failed high_median on balanced, odd-summed weights."

        x, w = [1, 2, 3, 4, 5], [3, 1, 2, 1, 3]
        assert high_median(x, w) == 3, \
            "Failed high_median on balanced, even-summed weights."
        x, w = scramble(x, w)
        assert high_median(x, w) == 3, \
            "Failed high_median on balanced, even-summed weights."

        x, w = [1, 2, 3, 4, 5], [3, 1, 1, 1, 1]
        assert high_median(x, w) == 2, \
            "Failed high_median on unbalanced odd-summed weights."
        x, w = scramble(x, w)
        assert high_median(x, w) == 2, \
            "Failed high_median on unbalanced odd-summed weights."

        x, w = [1, 2, 3, 4, 5], [4, 1, 1, 1, 1]
        assert high_median(x, w) == 2, \
            "Failed high_median on even-summed weights."
        x, w = scramble(x, w)
        assert high_median(x, w) == 2, \
            "Failed high_median on even-summed weights."

        x, w = [1, 2, 3, 4, 5], [5, 1, 1, 1, 1]
        assert high_median(x, w) == 1, "Failed high_median on answer=lowest."
        x, w = scramble(x, w)
        assert high_median(x, w) == 1, "Failed high_median on answer=lowest."

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 3]
        assert high_median(x, w) == 4, "Failed high_median on answer=highest-1."
        x, w = scramble(x, w)
        assert high_median(x, w) == 4, "Failed high_median on answer=highest-1."

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 4]
        assert high_median(x, w) == 5, "Failed high_median on answer=highest."
        x, w = scramble(x, w)
        assert high_median(x, w) == 5, "Failed high_median on answer=highest."

        x, w = [1, 2, 3, 4, 5], [1, 1, 1, 1, 5]
        assert high_median(x, w) == 5, "Failed high_median on answer=highest."
        x, w = scramble(x, w)
        assert high_median(x, w) == 5, "Failed high_median on answer=highest."


class Test_Qscale:
    """Test the Qscale() statistic."""

    def test_simple(self):
        x = np.array([1, 4, 5, 6, 8], dtype=float)
        assert Qscale(x) == pytest.approx(2.0*2.2219*.844, abs=1e-3)

    def test_inplace(self):
        _ = Qscale([1, 2, 3, 4], sort_inplace=False)
        pytest.raises(ValueError, Qscale, [1, 2, 3, 4], sort_inplace=True)

        x = np.array([4, 3, 2, 1])
        _ = Qscale(x, sort_inplace=False)
        assert x[0] == 4, "Qscale sorted data when asked not to."
        _ = Qscale(x, sort_inplace=True)
        assert x[3] == 4, "Qscale did not sort data when asked to sort."

    def Qslow(self, x):
        """Compute Q the simple, slow way, an O(n^2) calculation to verify the fast one."""
        x = np.array(x)
        x.sort()
        n = len(x)

        prefactor = 2.2219
        if n <= 9:
            prefactor *= [0, 0, 0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872][n]
        elif n % 2 == 1:
            prefactor *= n/(n+1.4)
        else:
            prefactor *= n/(n+3.8)

        dist = np.hstack([x[j]-x[:j] for j in range(1, n)])
        dist.sort()
        h = n // 2 + 1
        k = h*(h-1) // 2 - 1
        return prefactor * dist[k]

    def test_random(self):
        """Test some random (normal) data to be sure that the fast and Qslow
        operations give the same result.
        """
        rng = np.random.default_rng(0)  # improve test repeatability
        for size in (3, 6, 9, 15, 20, 25, 30, 40, 45, 50, 75, 100, 140):
            data = rng.standard_normal(size=size)
            qs = self.Qslow(data)
            qf = Qscale(data)
            assert qs is not None
            assert qf is not None
            assert qs == pytest.approx(qf, abs=1e-5)
