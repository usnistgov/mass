'''
Approximate a covariance matrix as a sum of exponentials, and factor it.

Classes include:
* FitExponentialSum - Fit a sum of exponentials to a data vector
* MultiExponentialCovarianceSolver - Factor and solve such a covariance matrix. 


Created on Nov 8, 2011

@author: fowlerj
'''

__all__ = ['MultiExponentialCovarianceSolver', 'FitExponentialSum']

import numpy
from mass.mathstat import _factor_covariance

class MultiExponentialCovarianceSolver(object):
    """
    Solver for a covariance matrix R composed of a short sum of exponentials.

    Cholesky factor a time covariance matrix, assuming that the covariance function
    is a short sum of (possibly complex) exponentials.  Use this factorization
    LL' to solve R or to apply L to a vector.  
    
    The factorization step requires O(k^2 n) operations.  Once this is done,
    the solve or Lx multiplication requires only O(kn).
    
    Wraps a few functions written by Brad Alpert in FORTRAN 90.

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
    
    def cholesky_product(self, x):
        """Return Lx where LL'=R (that is, L is the lower-triangular Cholesky
        factor of R).  This is useful in that if x is iid Gaussian noise of unit
        variance, then Lx has expected covariance matrix equal to R"""
        n = len(x)
        if n > self.nsamp:
            raise ValueError("The covariance matrix was factored for only "+
                             "%d samples.  Its Cholesky factor cannot multiply size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        return _factor_covariance.cholprod(x, self.cholesky_saved) #@UndefinedVariable
        
    def simulate_noise(self, n):
        """Return a vector of length <n> containing correlated multivariate Gaussian
        noise.  The expected covariance of this noise is R."""
        if n > self.nsamp:
            raise ValueError("The covariance matrix was factored for only "+
                             "%d samples.  Its Cholesky factor cannot multiply size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        white = numpy.random.standard_normal(n)
        return self.cholesky_product(white)

    def plot_covariance(self, nsamp=None, axis=None, **kwargs):
        """Plot the approximated covariance function for the first <nsamp> samples.
        Use matplotlib.Axes <axis> or (if None) a new full-figure subplot.
        All other keyword arguments are passed to pylab.plot"""
        if nsamp is None:
            nsamp = self.nsamp
        elif nsamp > self.nsamp:
            nsamp = self.nsamp
        import pylab
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        i = numpy.arange(nsamp, dtype=numpy.float)
        covar = numpy.array([a*(b**i) for a,b in zip(self.amplitudes, self.bases)]).sum(axis=0)
        axis.plot(covar.real, **kwargs)
        axis.set_xlabel("Samples")


class FitExponentialSum(object):
    """
    Fit a function as a sum of (possibly complex) exponentials.
    
    This is a subtle problem, until and unless the exponentials are
    known.  (Once they are known, it's a simple linear least-squares
    problem.)
    
    Typical usage when you don't know in advance how many singular values
    are important:
    
    fitter = FitExponentialSum(covariance_data, sval_thresh=None)
    fitter.plot_singular_values()  
    # You notice that almost all singular values are < 1e-3 times the highest, so...
    fitter.cut_svd(sval_thresh=1e-3)
    .... (INCOMPLETE DESCRIPTION)
    """
    
    def __init__(self, data, nsamp=None, rectangle_aspect=10, sval_thresh=None):
        """
        Store the data and plan the fit.
        
        data  - The data set to be fit.
        nsamp - The number of data values to fit.  If None, then use the entire data.
        rectangle_aspect - Aspect ratio of the Hankel matrix used in finding the exponentials
        sval_thresh - The least singular value to be used in the second stage.  If None,
                      then only the first stage will be done.
        """
        
        if nsamp is None:
            nsamp = len(data)
        elif len(data) < nsamp:
            raise ValueError("nsamp (%d) cannot exceed the length of data (%d)" % 
                             (nsamp, len(data)))
        if nsamp < 3:
            raise ValueError("The fit must be to at least 3 samples of data.")
        
        self.ncol = int(0.5+nsamp/(rectangle_aspect+1.0))
        if self.ncol<2:
            self.ncol = 2
        self.nrow = nsamp-1-self.ncol
        self.nsamp = nsamp
        self.data = numpy.asarray(data, dtype=numpy.float)[:self.nsamp].copy()
        self.svalues = None
        self.all_svalues = None
        self._hankel_svd()

        if sval_thresh is not None:
            self.cut_svd(sval_thresh)
        
    def _hankel_svd(self):
        """Compute the Hankel matrix used in the fit and do a singular
        value decomposition of all but its lowest row."""
        
        # Build the Hankel matrix
        A=numpy.zeros((self.nrow, self.ncol), dtype=numpy.float)
        A[:,0] = self.data[:self.nrow]  # col 0
        A[-1,1:] = self.data[self.nrow:self.nrow+self.ncol-1]  # bottom row
        for col in range(1, self.ncol):
            A[:-1,col] = A[1:,col-1]
        
        self.hankel2 = A[1:, :] 
        # Note that numpy.linalg.svd returns U, Sigma, and what is usually called V_transpose 
        self.svdu, self.all_svalues, self.svdv_t = \
            numpy.linalg.svd(A[:-1,:], full_matrices=False, compute_uv=True)

    def cut_svd(self, sval_thresh, min_values=None, max_values=None):
        """Reduce the Hankel matrix to keep only those parts of the SVD
        where the singular values are at least <sval_thresh> times the
        highest singular value.
        
        sval_thresh - Keep singular values down to this fractional level times the highest.
        min_values - Keep at least this many singular values.
        max_values - Keep no more than this many singular values.
        """
        min_sval = sval_thresh * self.all_svalues[0]
        ngood = (self.svalues>=min_sval).sum()
        
        # Some constraints on how many "good" singular values to use
        if min_values is not None:
            if min_values > self.nsamp/2:
                raise ValueError("min_values cannot exceed %d." % (self.nsamp/2))
            if ngood < min_values:
                ngood = min_values 

        if ngood > self.nsamp/2:
            ngood = self.nsamp/2
        elif max_values is not None:
            if ngood > max_values:
                ngood = max_values
        
        # Cut the matrices of the SVD, saving the full list of singular values.
        self.svalues = self.all_svalues[:ngood]
        self.svdu = self.svdu[:, :ngood]
        self.svdv_t = self.svdv_t[:ngood, :]
        
        # Now estimate the "system matrix" of time translation
        from numpy import dot
        fplus = (self.svdu*(self.svalues**-0.5)).T
        gplus = (self.svdv_t.T*(self.svalues**-0.5))
        self.system = dot(dot(fplus, self.hankel2), gplus)
        
        eigval, _evec = numpy.linalg.eig(self.system)
        self.bases = eigval
        print 'Bases are: ', eigval
        print 'Decay times (samples): ', -1./numpy.log(numpy.abs(eigval))
        
    def plot_singular_values(self):
        """Plot the full set of singular values, to help user decide where to cut"""
        import pylab
        pylab.clf()
        maxsval = self.all_svalues[0]
        pylab.semilogy(self.all_svalues/maxsval, 'rx-')
        if self.svalues is not None:
            pylab.plot(self.svalues/maxsval, 'ob')
        pylab.grid()
        ymax = 1.1
        ymin = 1e-10
        if ymin < (self.all_svalues[-1]/maxsval)*.9:
            ymin = (self.all_svalues[-1]/maxsval)*.9
        pylab.ylim([ymin, ymax])
        pylab.xlabel("Singular value number")
        pylab.ylabel("Arb scale (highest singular value scaled to 1)")
        pylab.title("Singular values ranked from highest to lowest")
        pylab.xlim([-0.5, len(self.all_svalues)+0.5])
    
    
    