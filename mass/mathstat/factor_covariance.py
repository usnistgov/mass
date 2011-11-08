'''
Factor an approximate covariance matrix (a sum of exponentials).

Factor a time covariance matrix, assuming that the covariance function
is a short sum of (possibly complex) exponentials.

Wraps a few functions written by Brad Alpert in FORTRAN 90.

Created on Nov 8, 2011

@author: fowlerj
'''

__all__ = ['MultiExponentialCovarianceSolver']

import numpy
from mass.mathstat import _factor_covariance

class MultiExponentialCovarianceSolver(object):
    """
    Solver for a covariance matrix R composed of a short sum of exponentials.

    TO DO:
    1. Method prod(x): Use the function covprod to compute the product Rx=y given x.
    
    2. It's my plan that someday we can add a method .expand(nsamp), which will
    grow the factored matrix R and allow us to expand dynamically the maximum
    length of vector that can be solved.... if this ever proves useful.
    """
    
    def __init__(self, amplitudes, bases, nsamp):
        """
        Define the covariance matrix to be solved as a sum of exponential terms.
        The covariance is stationary and symmetric:
        R_ij = Sum [m=1 to k]    amplitudes[m] * (bases[m]**|i-j|)
        
        <amplitudes>  A sequence of k amplitudes
        <bases>       A sequence of k bases, where |b|<=1 for each element.
        <nsamp>       The maximum number of data samples for which you will want to solve R.
        
        Work taking O(k*nsamp) will be done in the constructor to factor R.
        """
        
        # Parameter validation
        na = len(amplitudes)
        nb = len(bases)
        if na != nb:
            raise ValueError("The number of amplitudes (%d) != number of bases (%d)" % (na, nb))
        self.rank = na
        if nsamp < 2*self.rank:
            raise ValueError("The number of samples (%d) is not at least twice the rank (%d)"
                             % (nsamp, self.rank))
        if numpy.abs(bases).max() > 1.0:
            raise ValueError("The bases must not have absolute values greater than 1.")

        # Save the input parameters
        self.amplitudes = numpy.asarray(amplitudes, dtype=numpy.complex)
        self.bases = numpy.asarray(bases, dtype=numpy.complex)
        self.nsamp = nsamp
        
        # Cholesky factor the matrix and save the results in the opaque vector self.cholesky_saved
        self.cholesky_saved = _factor_covariance.covchol(self.amplitudes, #@UndefinedVariable
                                                self.bases, self.nsamp) 
        
    
    def __repr__(self):
        return "%s(amplitudes=%s, bases=%s, nsamp=%d)" % (self.__class__.__name__,
                                                        self.amplitudes, self.bases, self.nsamp)
    
    def __str__(self):
        return "%s of rank %d for vectors of length <= %d" % (self.__class__.__name__,
                                                        self.rank, self.nsamp)

    def __call__(self, b):
        """Equivalent to self.solve(b)"""
        return self.solve(b)
    
    def solve(self, b):
        """Solve the covariance matrix equation Rx=b for x.
        Requires that len(b) <= self.nsamp
        Return: <x>"""
        n = len(b)
        if n > self.nsamp:
            raise ValueError("The covariance matrix was factored for only "+
                             "%d samples and cannot solve size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        return _factor_covariance.covsolv(b, self.cholesky_saved) #@UndefinedVariable

