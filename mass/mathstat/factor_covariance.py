'''
Approximate a covariance matrix as a sum of exponentials, and factor it.

Classes include:
* FitExponentialSum - Fit a sum of exponentials to a data vector
* MultiExponentialCovarianceSolver - Factor and solve such a covariance matrix. 
* WhiteNoiseCovarianceSolver - Factor and solve the trivial covariance matrix. 


Created on Nov 8, 2011

@author: fowlerj
'''

__all__ = ['WhiteNoiseCovarianceSolver', 'MultiExponentialCovarianceSolver', 'FitExponentialSum']

import numpy
import scipy.optimize

try:
    from mass.mathstat import _factor_covariance
except ImportError:
    from mass.mathstat.utilities import MissingLibrary
    _factor_covariance = MissingLibrary("_factor_covariance.so")
    

class WhiteNoiseCovarianceSolver(object):
    """
    Solver for a covariance matrix R equal to the identity matrix times a constant.
    
    This is meant to share the same interface as MultiExponentialCovarianceSolver,
    but with trivial operations.  Using this object, you have a drop-in replacement
    for the fancier (slower) solver, which simplifies tests of the question: "how
    different would it be if we pretended the noise were white?"     
    """
    
    def __init__(self, variance=1.0):
        self.variance = variance

    def __repr__(self):
        return "%s()"%(self.__class__.__name__)

    def __str__(self):
        return "%s"%self.__class__.__name__
    
    def __call__(self, b):
        return self.solve(b)
    
    def solve(self, b):
        """Solve the covariance matrix equation Rx=b for x.
        Requires that len(b) <= self.nsamp
        Return: <x>"""
        return numpy.array(b)/self.variance
    
    def cholesky_product(self, x):
        """Return Lx where LL'=R (that is, L is the lower-triangular Cholesky
        factor of R).  This is useful in that if x is iid Gaussian noise of unit
        variance, then Lx has expected covariance matrix equal to R"""
        return numpy.array(x)*(self.variance**0.5)

    def cholesky_solve(self, x):
        """Return L^-1 x where LL'=R (that is, L is the lower-triangular Cholesky
        factor of R).  This is useful if we want to compute many vector-matrix-vector
        products of the form (a' R^-1 b).  We can instead compute A=L^-1 a and B=L^-1 b
        and the product becomes a simple dot product (a' R^-1 b) = A'B."""
        return numpy.array(x)/(self.variance**0.5)
      
    def covariance_product(self, x):
        """Return Rx."""
        return numpy.array(x)*self.variance
        
    def simulate_noise(self, n):
        """Return a vector of length <n> containing correlated multivariate Gaussian
        noise.  The expected covariance of this noise is R."""
        return numpy.random.standard_normal(n)*(self.variance**0.5)

    def plot_covariance(self, nsamp=100, axis=None, **kwargs):
        """Plot the approximated covariance function for the first <nsamp> samples.
        Use matplotlib.Axes <axis> or (if None) a new full-figure subplot.
        All other keyword arguments are passed to pylab.plot"""
        import pylab
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        i = numpy.arange(nsamp, dtype=numpy.float)
        covar = numpy.zeros(i)
        covar[0]=self.variance
        axis.plot(covar.real, **kwargs)
        axis.set_xlabel("Samples")



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
    1. It's my plan that someday we can add a method .expand(nsamp), which will
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
    
    def cholesky_solve(self, x):
        """Return L^-1 x where LL'=R (that is, L is the lower-triangular Cholesky
        factor of R).  This is useful if we want to compute many vector-matrix-vector
        products of the form (a' R^-1 b).  We can instead compute A=L^-1 a and B=L^-1 b
        and the product becomes a simple dot product (a' R^-1 b) = A'B."""
        n = len(x)
        if n > self.nsamp:
            raise ValueError("The covariance matrix was factored for only "+
                             "%d samples.  Its Cholesky factor cannot multiply size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        return _factor_covariance.cholsolv(x, self.cholesky_saved) #@UndefinedVariable

        
    def covariance_product(self, x):
        """Return Rx."""
        n = len(x)
        if n > self.nsamp:
            raise ValueError("The covariance matrix was factored for only "+
                             "%d samples.  It cannot multiply size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        return _factor_covariance.covprod(self.amplitudes, self.bases, x) #@UndefinedVariable
        
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
    fitter.fit_amplitudes()
    fitter.plot()
    print fitter.summary()
    
    If unhappy with the absolute deviation, repeat the steps starting at cut_svd and
    use a lower sval_thresh.
    .... (INCOMPLETE DESCRIPTION)
    """
    
    def __init__(self, data, nsamp=None, rectangle_aspect=10, sval_thresh=None,
                 randomize_columns = False):
        """
        Store the data and plan the fit.
        
        data  - The data set to be fit.
        nsamp - The number of data values to fit.  If None, then use the entire data.
        rectangle_aspect - Aspect ratio of the Hankel matrix used in finding the exponentials
        sval_thresh - The least singular value to be used in the second stage.  If None,
                      then only the first stage will be done.
        randomize_columns - Whether to fill the second half of all columns considered with
                    random choices.  (By default, False: use the first N columns.)
        """
        
        if nsamp is None:
            nsamp = len(data)
        elif len(data) < nsamp:
            raise ValueError("nsamp (%d) cannot exceed the length of data (%d)" % 
                             (nsamp, len(data)))
        if nsamp < 3:
            raise ValueError("The fit must be to at least 3 samples of data.")
        
        if nsamp*(nsamp/rectangle_aspect) > 100000000:
            nmax = 100000000/(nsamp**2)
            raise ValueError("The rectangle_aspect must be at least %d for samples of length %d"%(nmax, nsamp))
        
        if randomize_columns:
            self.nrow = nsamp/2
            self.ncol = int(0.5+self.nrow/rectangle_aspect)
        else:
            self.ncol = int(0.5+nsamp/(rectangle_aspect+1.0))
            if self.ncol<2:
                self.ncol = 2
            self.nrow = nsamp-1-self.ncol
        self.nsamp = nsamp
        self.data = numpy.asarray(data, dtype=numpy.float)[:self.nsamp].copy()
        self.svalues = None
        self.lowest_allowed_sval = 0.0
        self.all_svalues = None
        self.is_cut = False
        self.amplitudes = None
        self.complex_bases = None
        self.real_bases = None
        self.negative_bases = None
        self.randomize = randomize_columns
        
        self._hankel_svd()

        if sval_thresh is not None:
            self.cut_svd(sval_thresh)
        
    def _hankel_svd(self):
        """Compute the Hankel matrix used in the fit and do a singular
        value decomposition of all but its lowest row."""
        
        # Build the matrix of data columns
        A=numpy.zeros((self.nrow, self.ncol), dtype=numpy.float)
        A[:,0] = self.data[:self.nrow]  # col 0
        if self.randomize:
            # If random columns, then it's not a Hankel matrix.  Sorry.
            ncol_notrandom = self.ncol/2
            A[-1,1:ncol_notrandom] = self.data[self.nrow:self.nrow+ncol_notrandom-1]  # bottom row
            for col in range(1, ncol_notrandom):
                A[:-1,col] = A[1:,col-1]
            ncol_random = self.ncol - ncol_notrandom
            col_choices = numpy.random.permutation( numpy.arange(ncol_notrandom, self.nsamp-self.nrow))[:ncol_random]
            for col,choice in zip(range(ncol_notrandom,self.ncol), col_choices):
                A[:,col] = self.data[choice:choice+self.nrow]
        else:
            A[-1,1:] = self.data[self.nrow:self.nrow+self.ncol-1]  # bottom row
            for col in range(1, self.ncol):
                A[:-1,col] = A[1:,col-1]
        
        self.hankel2 = A[1:, :] 
        # Note that numpy.linalg.svd returns U, Sigma, and what is usually called V_transpose 
        self.svdu, self.all_svalues, self.svdv_t = \
            numpy.linalg.svd(A[:-1,:], full_matrices=False, compute_uv=True)
        self.svalues = self.all_svalues
        self.lowest_allowed_sval = 0.0

    def cut_svd(self, sval_thresh, min_values=None, max_values=None):
        """Reduce the Hankel matrix to keep only those parts of the SVD
        where the singular values are at least <sval_thresh> times the
        highest singular value.
        
        sval_thresh - Keep singular values down to this fractional level times the highest.
        min_values - Keep at least this many singular values.
        max_values - Keep no more than this many singular values.
        
        If you have previously cut to a higher (more restrictive) <sval_thresh>, then
        self._hankel_svd() will be called to repeat construction of the Hankel matrix
        and to compute its SVD.
        """
        
        # If you've already cut the SVD more severely than you want to cut it now, then the
        # SVD will need to be recomputed.
        min_sval = sval_thresh * self.all_svalues[0]
        if self.is_cut and min_sval < self.lowest_allowed_sval:
            self._hankel_svd()
        self.is_cut = True

        ngood = (self.all_svalues>=min_sval).sum()
        
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
        if ngood<1: 
            ngood=1
        print "Using %d singular values exceeding %f" % (ngood, min_sval)
        if ngood > len(self.svalues):
            print "Have to recompute Handle SVD"
            self._hankel_svd()

        # Cut the matrices of the SVD, saving the full list of singular values.
        self.svdu = self.svdu[:, :ngood]
        self.svalues = self.all_svalues[:ngood]
        self.svdv_t = self.svdv_t[:ngood, :]
        self.lowest_allowed_sval = min_sval
        
        # Now estimate the "system matrix" of time translation
        from numpy import dot
        fplus = (self.svdu*(self.svalues**-0.5)).T
        gplus = (self.svdv_t.T*(self.svalues**-0.5))
        self.system = dot(dot(fplus, self.hankel2), gplus)
        
        print 'Solving system of shape ', self.system.shape
        print '  |system| = %.4g' % numpy.linalg.det(self.system)
        eigval, _evec = numpy.linalg.eig(self.system)
        self.bases = eigval
#        print 'Bases are: ', eigval
#        print 'Decay times (samples): ', -1./numpy.log(numpy.abs(eigval))
        
    def plot_singular_values(self):
        """Plot the full set of singular values, to help user decide where to cut"""
        import pylab
        pylab.clf()
        maxsval = self.all_svalues[0]
        pylab.semilogy(self.all_svalues/maxsval, 'rx-')
        if self.is_cut:
            pylab.plot(self.svalues/maxsval, 'ob')
            m = self.lowest_allowed_sval/maxsval
            pylab.plot(pylab.xlim(), [m,m], color='gray')
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
    
    def _model(self, weights, real_bases, negative_bases, complex_bases, x):
        """Compute and return the multi-exponential model at a vector of sample numbers <x>
        given the array of <weights>, a sequence of <real_bases> and a sequence of
        <complex_bases>.  The number of weights should be len(real_bases) + 2*len(complex_bases)."""
        
        # Positive real bases
        m = numpy.array([w*(b**x) for w,b in zip(weights, real_bases)]).sum(axis=0)
        
        # Negative bases
        nr = len(real_bases)
        m += numpy.array([w*((-b)**x)*numpy.cos(numpy.pi*x) for w,b in zip(weights[nr:], negative_bases)]).sum(axis=0)
        
        # Complex bases
        nr = len(real_bases)+len(negative_bases)
        for i,cb in enumerate(complex_bases):
            amplitude = weights[i*2+nr] + weights[i*2+nr+1]*1j
            m += (amplitude*(cb**x)).real
        return m
 
    def model(self, x):
        """Given that the model is already completely fit..."""
        return self._model(self.amplitudes, self.real_bases, self.negative_bases, self.complex_bases, x)


    def fit_amplitudes(self, verbose=False):
        """Solve for the amplitudes of the exponential 'bases' found in self.cut_svd()."""
        
        if not self.is_cut:
            raise ValueError("Cannot fit amplitudes until fitter.cut_svd() is called to compute and cut on singular values.")
    
        
        ############################
        def residual(weights, real_bases, negative_bases, complex_bases, x, y):
            """Compute and return the residual between a data vector <y> 
            and the multi-exponential model at a vector of sample numbers <x>.
            This requires the array of <weights>, a sequence of <real_bases> and a sequence of
            <complex_bases>.  The number of weights should be 
            len(real_bases) + len(negative_bases) + 2*len(complex_bases)."""
            return y-self._model(weights, real_bases, negative_bases, complex_bases, x)
        
        # Separate real bases from CC pairs.  Sort by base.imag and pair off complex ones that way
        idx = self.bases.imag.argsort()
        pairs = []
        while len(idx)>0 and numpy.abs(self.bases.imag[idx[0]]) > 1./self.nsamp:
            pairs.append(idx[-1])
            idx=idx[1:-1]
        solos=idx
        real_bases = self.bases[solos].real
        negative_bases = real_bases[real_bases<0]
        real_bases = real_bases[real_bases>0]
        complex_bases = self.bases[pairs]
#        print 'Real bases: ',real_bases
#        print 'Complex bases: ',complex_bases
    
        powers = numpy.arange(self.nsamp, dtype=numpy.float)
        fweights = numpy.ones(len(self.bases), dtype=numpy.float)
        fweights, _stat =  scipy.optimize.leastsq(residual, fweights, 
                                                  args=(real_bases, negative_bases, 
                                                        complex_bases, powers, self.data))
        self.amplitudes = fweights
        self.real_bases = real_bases
        self.negative_bases = negative_bases
        self.complex_bases = complex_bases
        if verbose:
            for w,b in zip(fweights, real_bases):
                log=numpy.log(b)
                if numpy.isnan(log):
                    log = numpy.log(-b)+numpy.pi*1j
                print " %10.5f*[(%9.6f+%8.5fj)**m] or exp((%8.5f+%8.5fj)m)"%(w,b.real, b.imag, log.real, log.imag)
            
            nr = len(real_bases)
            for i,nb in enumerate(negative_bases):
                w = fweights[i+nr]
                log=numpy.log(-nb)
                print " %10.5f*[(%9.6f+%8.5fj)**m] or exp((%8.5f+%8.5fj)m)"%(w,nb.real, nb.imag, log.real, log.imag)
            
            nr = len(real_bases) + len(negative_bases)
            for i,cb in enumerate(complex_bases):
                log=numpy.log(cb)
                if numpy.isnan(log):
                    log = numpy.log(-cb)+numpy.pi*1j
                w=fweights[2*i+nr:2*i+nr+2]
                print " %10.5f *[(%9.6f+%8.5fj)**m] or exp((%8.5f+%8.5fj)m)"%(w[0],cb.real, cb.imag, log.real, log.imag)
                print "+%10.5fj*[(%9.6f+%8.5fj)**m] or exp((%8.5f+%8.5fj)m)"%(w[1],cb.real, cb.imag, log.real, log.imag)

    
    def results(self):
        """This takes the "packed" results and returns them in a form suitable for use by
        the MultiExponentialCovaranceSolver."""
        amplitudes = []
        bases = []
        stored_amp = list(self.amplitudes)
        for b in self.real_bases:
            bases.append(b)
            amplitudes.append(stored_amp.pop(0))
        for b in self.negative_bases:
            bases.append(b)
            amplitudes.append(stored_amp.pop(0))
        for b in self.complex_bases:
            bases.append(b)
            r = stored_amp.pop(0)
            c = stored_amp.pop(0)
            amplitudes.append(r + 1j*c)
        return numpy.array(amplitudes), numpy.array(bases)
 
    
    def plot(self, axis=None, axis2=None):
        import pylab
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(211)
            axis2 = pylab.subplot(212, sharex=axis)

        axis.plot(self.data, label='Data')
        if self.amplitudes is not None:
            x = numpy.arange(self.nsamp)
            y = self.model(x)
            axis.plot(x, y, label='Model')
            axis.set_xlabel("Sample number")
            
            nr = len(self.real_bases)
            nn = len(self.negative_bases)
            for i,b in enumerate(self.real_bases):
                axis.plot(x, self._model([self.amplitudes[i]], [b], [], [], x), '--', label='Component %.4g' % b)
            for i,b in enumerate(self.negative_bases):
                axis.plot(x, self._model([self.amplitudes[i+nr]], [], [b], [], x), '--', label='Component %.4g' % b)
            for i,c in enumerate(self.complex_bases):
                j = i*2 + nr + nn
                axis.plot(x, self._model(self.amplitudes[j:j+2], [], [], [c], x), '--', label='Component %.4g %.4gj' % (c.real, c.imag))
        axis.legend()
        
        if axis2 is not None:
            residual = self.data-self.model(x)
            axis2.plot(residual)
            axis2.text(.1, .9, 'rms deviation: %f' % residual.std(), transform=axis2.transAxes)
            axis2.set_xlabel("Sample number")
    
    def summary(self, dt=1.0):
        s=['Summary of exponential sum fit:']
        for i,b in enumerate(self.real_bases):
            if b>0:
                s.append("%11.4f exp(%10.5f t)  [x = %10.6f  tau  = %10.4f]" % (self.amplitudes[i], numpy.log(b)/dt, b, -dt/numpy.log(b)))
            else:
                s.append("%11.4f exp(%10.5f t)  [x = %10.6f  tau  = %10.4f] (-1)^k" % (self.amplitudes[i], numpy.log(-b)/dt, b, -dt/numpy.log(-b)))
        for i,c in enumerate(self.complex_bases):
            j = i*2 + len(self.real_bases)
            s.append("%11.4f exp(%10.5f t) cos(%10.5f t) tau  = %10.4f" % (self.amplitudes[j], numpy.log(c.real)/dt, c.imag/dt, -dt/numpy.log(c.real)))
            s.append("%11.4f exp(%10.5f t) sin(%10.5f t) period %10.4f" % (-self.amplitudes[j+1], numpy.log(c.real)/dt, c.imag/dt, 2*numpy.pi*dt/c.imag))
        return "\n".join(s)

