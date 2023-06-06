"""
Unit tests for MASS utilities.
(So far, only the Toeplitz solver)

Also contains a speed test object TestToeplitzSpeed

J. Fowler, NIST

March 30, 2011
"""

from mass.mathstat.toeplitz import ToeplitzSolver
import numpy as np
import scipy.linalg as linalg
import time
import unittest


class TestToeplitzSolverSmallSymmetric(unittest.TestCase):
    """Test ToeplitzSolver on a 5x5 symmetric matrix."""

    def setUp(self):
        self.autocorr = np.array((6., 4., 2., 1., 0.))
        self.n = len(self.autocorr)
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5*np.random.standard_normal(self.n)
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)


class TestToeplitzSolverSmallAsymmetric(unittest.TestCase):
    """Test ToeplitzSolver on a 5x5 non-symmetric matrix."""

    def setUp(self):
        self.autocorr = np.asarray((-1, -2, 0, 3, 6., 4., 2., 1., 0.))
        self.n = (len(self.autocorr) + 1) // 2
        self.solver = ToeplitzSolver(self.autocorr, symmetric=False)
        self.R = linalg.toeplitz(self.autocorr[self.n-1:], self.autocorr[self.n-1::-1])

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)

    def test_arb_vectors(self):
        for _i in range(self.n):
            x_in = 5*np.random.standard_normal(self.n)
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 12)


class TestToeplitzSolver_32(unittest.TestCase):
    """Test ToeplitzSolver on a 32x32 symmetric matrix."""

    def setUp(self):
        self.n = 32
        t = np.arange(self.n)
        t[0] = 1
        pi = np.pi
        T = 1.0*self.n
        self.autocorr = np.sin(pi*t/T)/(pi*t/T)
        self.autocorr[0] = 1
        self.autocorr[:5] *= np.arange(5, .5, -1)
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_all_unit_vectors(self):
        for i in range(self.n):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(
                0, big_dif, 10, msg='Unit vector trial i=%2d gives x_out=%s' % (i, x_out))


class TestToeplitzSolver_512(unittest.TestCase):
    """Test ToeplitzSolver on a 512x512 symmetric matrix."""

    def setUp(self):
        self.n = 512
        t = np.arange(self.n)
        self.autocorr = 1.0+3.2*np.exp(-t/100.)
        self.autocorr[0] = 9
        self.solver = ToeplitzSolver(self.autocorr, symmetric=True)
        self.R = linalg.toeplitz(self.autocorr)

    def test_some_unit_vectors(self):
        for i in (0, 20, 128, 256, 500, 512-1):
            x_in = np.zeros(self.n, dtype=float)
            x_in[i] = 1.0
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(
                0, big_dif, 10, msg='Unit vector trial i=%2d gives x_out=%s' % (i, x_out))

    def test_arb_vectors(self):
        for _i in range(5):
            x_in = 5*np.random.standard_normal(self.n)
            y = np.dot(self.R, x_in)
            x_out = self.solver(y)
            big_dif = np.abs(x_out-x_in).max()
            self.assertAlmostEqual(0, big_dif, 10,
                                   msg='Random vector trial gives rms diff=%sf' % (x_out-x_in).std())


class toeplitzSpeed:
    """Test the speed of the Toeplitz solver.

    This is NOT a unit test. Usage:

>>> from mass.mathstat.test import test_toeplitz
>>> t=test_toeplitz.toeplitzSpeed()
>>> t.plot()
    """

    def __init__(self, maxsize=8192):
        self.sizes = np.hstack((100, 200, np.arange(500, 5500, 500), 6144, 8192, 10000,
                                20000, 30000, 50000))
        t = np.arange(100000)
        self.autocorr = 1.0+3.2*np.exp(-t/100.)
        self.autocorr[0] = 9

        self.ts_time = np.zeros(len(self.sizes), dtype=float)
        self.build_time = np.zeros_like(self.ts_time)
        self.mult_time = np.zeros_like(self.ts_time)
        self.solve_time = np.zeros_like(self.ts_time)
        self.lu_time = np.zeros_like(self.ts_time)
        for i, s in enumerate(self.sizes):
            times = self.test(s, maxsize)
            (self.ts_time[i], self.build_time[i], self.mult_time[i],
             self.solve_time[i], self.lu_time[i]) = times

    def test(self, size, maxsize=8192):
        if size > 150000:
            return 5*[np.NaN]

        ac = self.autocorr[:size]
        v = np.random.standard_normal(size)

        t0 = time.time()
        solver = ToeplitzSolver(ac, symmetric=True)
        x = solver(v)
        dt = [time.time()-t0]

        if size <= maxsize:
            # dt[1] = creating R time
            t0 = time.time()
            R = linalg.toeplitz(ac)
            dt.append(time.time()-t0)

            # dt[2] = R * vector time
            t0 = time.time()
            v2 = np.dot(R, x)
            dt.append(time.time()-t0)

            # dt[3] = solve(R,v) time
            t0 = time.time()
            x2 = np.linalg.solve(R, v)
            dt.append(time.time()-t0)

            t0 = time.time()
            lu_piv = linalg.lu_factor(R)
            x3 = linalg.lu_solve(lu_piv, v, overwrite_b=False)
            dt.append(time.time()-t0)
            print('rms rhs diff: %.3g, solution diff: %.3g %.3g' %
                  ((v-v2).std(), (x-x2).std(), (x-x3).std()))

        else:
            dt.extend(4*[np.NaN])
        print(size, ['%6.3f' % t for t in dt])
        return dt

    def plot(self):
        import pylab as plt
        plt.clf()
        plt.plot(self.sizes, self.ts_time, label='Toeplitz solver')
        plt.plot(self.sizes, self.build_time, label='Matrix build')
        plt.plot(self.sizes, self.mult_time, label='Matrix-vector mult')
        plt.plot(self.sizes, self.solve_time, label='Matrix solve')
        plt.plot(self.sizes, self.lu_time, label='LU solve')
        plt.legend(loc='upper left')
        plt.xlabel('Vector size')
        plt.ylabel('Time (sec)')
        plt.grid()
        plt.loglog()


if __name__ == '__main__':
    unittest.main()
