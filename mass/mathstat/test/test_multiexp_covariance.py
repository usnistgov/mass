'''
Test framework for the MultiExponentialCovarianceSolver

Checks a 2x2 matrix (where we know the answers), including a 2x2 sub-matrix
of larger matrices.  Also checks a few random instances of 100x100 matrices.

Created on Dec 2, 2011

@author: fowlerj
'''
import numpy, scipy.linalg
import unittest
from mass.mathstat import MultiExponentialCovarianceSolver

class ParentTest2x2(object):
    """Test a 2x2 matrix with [1, .5] elements.
    This is not a full test case, but just supplies 3 tests for any full case that
    uses the same matrix.  A full test case inherits this AND unittest.TestCase."""

    def test_product(self):
        """Verify product Rx for 2x2 matrix."""
        z = self.solver.covariance_product([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 14) 
        y = self.solver.covariance_product([4, 3])
        self.assertAlmostEqual(y[0], 5.5, 14)
        self.assertAlmostEqual(y[1], 5.0, 14)

    def test_cholesky_product(self):
        """Verify product Lx for 2x2 matrix."""
        z = self.solver.cholesky_product([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 14) 
        y = self.solver.cholesky_product([5, 3**(-0.5)])
        self.assertAlmostEqual(y[0], 5.0, 14)
        self.assertAlmostEqual(y[1], 3.0, 14)
        
    def test_solve(self):
        """Verify solution Rx=b for 2x2 matrix."""
        z = self.solver.solve([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 14) 
        y = self.solver.solve([1, -1])
        self.assertAlmostEqual(y[0], 2.0, 14)
        self.assertAlmostEqual(y[1], -2.0, 14)


class Test2x2_full_matrix(unittest.TestCase, ParentTest2x2):
    """Test a 2x2 matrix with [1, .5] elements."""
    def setUp(self):
        self.solver = MultiExponentialCovarianceSolver([1,], [.5], 2)


class Test2x2_submatrix4(unittest.TestCase, ParentTest2x2):
    """Test a 2x2 submatrix with [1, .5] elements out of a larger 4x4 matrix."""
    def setUp(self):
        self.solver = MultiExponentialCovarianceSolver([.5, .5], [.2, .8], 4)


class Test2x2_submatrix99(unittest.TestCase, ParentTest2x2):
    """Test a 2x2 submatrix with [1, .5] elements out of a larger 99x99 matrix."""
    def setUp(self):
        self.solver = MultiExponentialCovarianceSolver([.5, .5], [.2, .8], 99)


class Test100x100(unittest.TestCase):
    """Test a 100x100 matrix."""
    
    def setUp(self):
        self.nsamp = 100
        self.nvectors = 3
        amps = numpy.random.uniform(size=5)
        bases = numpy.random.uniform(size=5)
        self.solver = MultiExponentialCovarianceSolver(amps, bases, self.nsamp)
        t = numpy.arange(self.nsamp, dtype=numpy.float)
        x = numpy.zeros_like(t)
        for a,b in zip(amps, bases):
            x += a*(b**t)
        self.matrix = scipy.linalg.toeplitz(x)

    def test_product(self):
        """Verify product Rx for 100x100 matrix."""
        z = self.solver.covariance_product([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 12)
            
        for _ in range(self.nvectors):
            x = numpy.random.standard_normal(self.nsamp) 
            y = self.solver.covariance_product(x)
            ytrue = numpy.dot(self.matrix, x)
            for a, b in zip(y, ytrue):
                self.assertAlmostEqual(a, b, 12)

    def test_cholesky_product(self):
        """Verify product Lx for 100x100 matrix"""
        z = self.solver.cholesky_product([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 12) 

        for _ in range(self.nvectors):
            x = numpy.random.standard_normal(self.nsamp) 
            y = self.solver.cholesky_product(x)
            ytrue = numpy.dot(numpy.linalg.cholesky(self.matrix), x)
            for a, b in zip(y, ytrue):
                self.assertAlmostEqual(a, b, 12)
        
    def test_solve(self):
        """Verify solution Rx=b for 100x100 matrix"""
        z = self.solver.solve([0, 0])
        for x in z:
            self.assertAlmostEqual(x, 0, 12) 

        for _ in range(self.nvectors):
            x = numpy.random.standard_normal(self.nsamp) 
            y = self.solver(x)
            ytrue = numpy.linalg.solve(self.matrix, x)
            for a, b in zip(y, ytrue):
                self.assertAlmostEqual(a, b, 12)

    
if __name__ == "__main__":
    unittest.main()