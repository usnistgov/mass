"""
mass.utilities

Several math utilities, including:
* Toeplitz matrix solver (useful for computing time-domain optimal filters)
* A histogram fitter that uses a full maximum likelihood fit.
* A mouse click capturer for mouse feedback from plots.

Joe Fowler, NIST

Started March 24, 2011
"""

## \file utilities.py
# \brief Several utilities used by Mass, including math, plotting, and other functions.
#
# Math utilities include:
# -# CholeskySolver (meant to supersede toeplitz.ToeplitzSolver)
# -# MaximumLikelihoodHistogramFitter
#
# Other utilities:
# -# plot_as_stepped_hist, to draw an already computed histogram in the same way that
#    pylab.hist() would do it.
# -# MouseClickReader, to capture pointer location on a plot.


import numpy
import scipy.linalg


def plot_as_stepped_hist(axis, bin_ctrs, data, **kwargs):
    """Plot onto <axis> the histogram <bin_ctrs>,<data> in stepped-histogram format.
    \param axis     The pylab Axes object to plot onto.
    \param bin_ctrs An array of bin centers.  (Bin spacing will be inferred from the first two).
    \param data     Bin contents.   data and bin_ctrs will only be used to the shorter of the two arrays.
    \param kwargs   All other keyword arguments will be passed to axis.plot().
    """
    x = numpy.zeros(2+2*len(bin_ctrs), dtype=numpy.float)
    y = numpy.zeros_like(x)
    dx = bin_ctrs[1]-bin_ctrs[0]
    x[0:-2:2] = bin_ctrs-dx*.5
    x[1:-2:2] = bin_ctrs-dx*.5
    x[-2:] = bin_ctrs[-1]+dx*.5
    y[1:-1:2] = data
    y[2:-1:2] = data
    axis.plot(x, y, **kwargs)
    axis.set_xlim([x[0],x[-1]])



import covar as f90covar #@UnresolvedImport

class CovarianceSolver(object): 
    """
    """
    
    def __init__(self, covariance):
        """
        <covariance>  The noise covariance function, starting at zero lag 
        """
        self.covariance = numpy.asarray(covariance, dtype=numpy.float)
        self.N = len(covariance)
#        self.cholesky_saved = f90covar.cholsav()
        raise NotImplementedError("Full Covariance Solver doesn't exist yet.")
        
        

class MultiExponentialCovarianceSolver(object):
    """
    It's my plan that someday we can add a method .expand(nsamp), which will
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
        if na!=nb:
            raise ValueError("The number of amplitudes (%d) != number of bases (%d)"%(na,nb))
        self.rank=na
        if nsamp<2*self.rank:
            raise ValueError("The number of samples (%d) is not at least twice the rank (%d)"%(nsamp, self.nrank))
        if numpy.abs(bases).max() > 1.0:
            raise ValueError("The bases must not have absolute values greater than 1.")

        # Save the input parameters
        self.amplitudes = numpy.asarray(amplitudes, dtype=numpy.complex)
        self.bases = numpy.asarray(bases, dtype=numpy.complex)
        self.nsamp = nsamp
        
        # Cholesky factor the matrix and save the results in the opaque vector self.cholesky_saved
        self.cholesky_saved=f90covar.covchol(self.amplitudes, self.bases, self.nsamp)
        
    
    def __repr__(self):
        return "%s(amplitudes=%s, bases=%s, nsamp=%d)"%(self.__class__.__name__,
                                                        self.amplitudes, self.bases, self.nsamp)
    
    def __str__(self):
        return "%s of rank %d for vectors of length <= %d"%(self.__class__.__name__,
                                                        self.rank, self.nsamp)

    def __call__(self, b):
        """Equivalent to self.solve(b)"""
        return self.solve(b)
    
    def solve(self, b):
        """Solve the covariance matrix equation Rx=b for x.
        Requires that len(b) <= self.nsamp
        Return: <x>"""
        n = len(b)
        if n>self.nsamp:
            raise ValueError("The covariance matrix was factored for only %d samples and cannot solve size %d>%d"%(
                                    self.nsamp, n, self.nsamp))
        return f90covar.covsolv(b, self.cholesky_saved)



class MaximumLikelihoodHistogramFitter(object):
    """
    Object to fit a theory having 1 or more free parameters to a histogram, using the
    proper likelihood.  That is, assume that events in each bin are independent
    and are Poisson-distributed with an expectation equal to the theory.
    
    This implementation is fast (only requires 2x as long as the chisquared
    minimizer scipy.optimize.leastsq, which assumes Gaussian-distributed data),
    and it removes the biases that arise from fitting parameters by minimizing
    chi-squared, as if the data were Gaussian.
    
    User must supply the bin centers, bin contents, an initial guess at the 
    theory parameters, and the theory function.  An optional theory gradient
    can be supplied to compute gradient of the theory w.r.t. its parameters.
    If not supplied, gradients will be computed by one-sided finite differences.
    
    For information on the algorithm and its implementation, see 
    MaximumLikelihoodHistogramFitter.NOTES  
    """
    
    ## Further algorithm notes beyond the docstring.
    NOTES="""
    The likelihood being maximized is converted to a "MLE Chi^2" by defining
    chi^2_MLE = -2 log(LR).  LR is here a likelihood RATIO, between the Poisson  
    likelihood of the data given the theory and the likelihood given a "perfect theory"
    that predicts the exact observed data. (We assume flat priors on the expected number
    of events per bin, rather than flat priors on the parameters of the theory.) 
    The chi^2 so defined is what appears as the attribute chisq of this class
    after doing a fit.
    
    Expressions for this likelihood appear in many places, including equation (2)
    of S Baker & R.D. Cousins "Clarification of the use of chi-square and likelihood
    functions in fits to Histograms", NIM 221 (1984) pages 437-442.  The MLE chi^2
    appears as "chi^2_{lambda,p}" near the end of section 2.  The MLE chi^2 is also
    equation (1) in T.A. Laurence & B.A. Chromy "Efficient maximum likelihood estimator
    fitting of histograms", Nature Methods v.7 no.5 (2010) page 338-339 and its
    online supplement. 
    
    The algorithm for rapidly minimizing this chi-squared is a slight variation
    on the usual Levenberg-Marquardt method for minimizing a sum of true squares,
    as described in Laurence & Chromy.  The implementation is a translation to 
    Python+numpy of the Levenberg-Marquardt solver class Fitmrq appearing in C++ in
    Numerical Recipes, 3rd Edition, with appropriate changes so as to minimize
    -2 times the log of the likelihood ratio, rather than a true chi-squared.
    """
    
    ## This many steps with negligible "chisq" change cause normal exit
    DONE=4
    
    ## This many steps cause RuntimeError for excessive iterations
    ITMAX=500  
    
    def __init__(self, x, nobs, params, theory_function, theory_gradient=None, 
                 epsilon=1e-5, TOL=1e-3):
        """
        Initialize the fitter, making copies of the input data.
        
        Inputs:
        <x>          The histogram bin centers (used in computing model values)
        <nobs>       The histogram bin contents (must be same length as <x>)
        <params>     The theory's vector of parameter starting guesses
        <theory_function>   (callable) A function of (params, x) returning the modeled bin contents as
                            a vector of the same length as x
        <theory_gradient>   (callable) A function of (params, x) returning the gradient of the modeled 
                            bin contents as a 2D vector of shape (len(params), len(x)).  If None (the 
                            default), then the gradient is computed by one-sided finite differences.
        <epsilon>    (float or ndarray).  If theory_gradient is to be approximated,
                    use this value for the step size. 
        <TOL>       The fractional or absolute tolerance on the minimum "MLE Chi^2".
                    When self.DONE successive iterations fail to improve the MLE Chi^2 by this
                    much (aboslutely or fractionally), then fitting will return successfully.
        """
        self.x = numpy.array(x)
        self.ndat = len(x)
        self.nobs = numpy.array(nobs)
        if len(self.nobs) != self.ndat:
            raise ValueError("x and nobs must have the same length")

        self.mfit = self.nparam = 0
        self.set_parameters(params)
        self.theory_function = theory_function
        if theory_gradient is None:
            self.theory_gradient = self.__discrete_gradient
            if numpy.isscalar(epsilon):
                self.epsilon = epsilon + numpy.zeros_like(params)
            else:
                if len(epsilon) != self.nparam:
                    raise ValueError("epsilon must be a scalar or if a vector, a vector of the same length as params")
                self.epsilon = numpy.array(epsilon)
        else:
            self.theory_gradient = theory_gradient
        self.TOL = TOL
        self.chisq = 0.0


    def set_parameters(self, params):
        """
        Set the initial guess at the parameters.  This call will clear
        the "hold state" of any parameters that used to be held.  This
        call also erases the self.alpha and self.r matrices.
        """
        self.mfit = self.nparam = len(params)
        self.params = numpy.array(params)
        self.ia = numpy.ones(self.nparam, dtype=numpy.bool)
        self.alpha=numpy.zeros((self.nparam,self.nparam), dtype=numpy.float)
        self.covar=numpy.zeros((self.nparam,self.nparam), dtype=numpy.float)


    def hold(self, i, val=None):
        """
        Hold parameter number <i> fixed either at value <val> or (by default) at its present 
        value.  Parameter is fixed until method free(i) is called.
        """
        self.ia[i] = False
#        print 'Holding param %d'%i
        if val is not None:
            self.params[i] = val
        

    def free(self, i):
        """
        Release parameter <i> to float in the next fit.  If already free, then no effect.
        """
        self.ia[i] = True


    def __discrete_gradient(self, p, x): 
        """
        Estimate the gradient of our minimization function self.theory_function
        with respect to each of the parameters when we don't have an exact expression for it.
        Use one-sided differences with steps of size self.epsilon away
        from the test point <p> at an array of points <x>.
        
        If you have a way to return the true gradient dy/dp_i | p,x, then you
        should use that instead as function self.theory_gradient.r
        """
        nx = len(x)
        np = len(p)
        dyda=numpy.zeros((np, nx), dtype=numpy.float)
        yp = self.theory_function(p,x)
        for i,dx in enumerate(self.epsilon):
            p2 = p.copy()
            p2[i]+=dx
            dyda[i,:] = (self.theory_function(p2,x)-yp)/dx
        return dyda


    def fit(self):
        """
        Iterate to reduce the "maximum likelihood chi-squared" of a fit between a histogram 
        of data points self.x, self.nobs and a nonlinear function that depends on the 
        coefficients self.param.
        
        When chisq is no longer decreasing for self.DONE iterations, return (params, covariance)
        where <params> is the best-fit value of each parameter (recall that some can optionally
        be held out of the fit by calling self.free and self.hold), and where <covariance> is the
        square matrix of estimated covariances between the parameters.
        
        When self.ITMAX iterations are reached, this method raises a RuntimeError.  
        """
        
        done = 0
        alambda = 0.01
        self.mfit = self.ia.sum()
        self.alpha, beta = self.__mrqcof(self.params)
        
        atry = self.params.copy()
        ochisq = self.chisq
        for iter_number in range(self.ITMAX):
            temp = numpy.array(self.alpha)
            if done==self.DONE:
                alambda = 0.0 # use alambda=0 on last pass
            else:
                for j in range(self.mfit):
                    temp[j,j] *= 1.0+alambda
        
            try:
                da = scipy.linalg.solve(temp, beta[self.ia])
                scipy.linalg.inv(temp, overwrite_a = True)
            except scipy.linalg.LinAlgError, e:
                print 'temp (lambda=%f, iteration %d) is singular:'%(alambda,iter_number), temp, beta
                raise e

            if done==self.DONE:
                self.covar = numpy.zeros((self.nparam, self.nparam), dtype=numpy.float)
                self.covar[:self.mfit, :self.mfit] = temp
                self.__cov_sort_in_place(self.covar)                
#                self.__cov_sort_in_place(self.alpha)
                return self.params, self.covar
            
            # Did the trial succeed?
            atry[self.ia] = self.params[self.ia] + da
            
            self.covar, da = self.__mrqcof(atry)
            if abs(self.chisq-ochisq) < max(self.TOL, self.TOL*self.chisq): done+=1
            if (self.chisq < ochisq ): # success: we've improved
                alambda *= 0.1
                ochisq = self.chisq
                self.alpha = self.covar.copy()
                beta = da.copy()
                self.params = atry.copy()
            else:   # failure.  Increase lambda and return to previous starting point.
                alambda *= 10
                self.chisq = ochisq
        
        raise RuntimeError("lev_marq_min.fit too many iterations")
        
    
    def __mrqcof(self, params):
        """Used by fit to evaluate the linearized fitting matrix alpha and vector beta,
        and to calculate chisq.  Returns (alpha,beta) and stores chisq in self.chisq"""
        
        # Careful!  Do this smart, or this routine will bog down the entire
        # fit hugely.
        y_model = self.theory_function(params, self.x)
        dyda = self.theory_gradient(params, self.x)
        if dyda[0].sum()==0:
            print 'Problem:',self.epsilon, dyda[:,:4], params
        dyda_over_y = dyda/y_model
        nobs = self.nobs
        y_resid = nobs - y_model
        
        beta = (y_resid*dyda_over_y).sum(axis=1)
        
        alpha = numpy.zeros((self.mfit,self.mfit), dtype=numpy.float)
        for i in range(self.mfit):
            for j in range(i+1):
                alpha[i,j] = (nobs*dyda_over_y[i,:]*dyda_over_y[j,:]).sum()
                alpha[j,i] = alpha[i,j]
            
        nzobs = nobs>0
        self.chisq = 2*(y_model.sum()-nobs.sum())  \
                + 2*(nobs[nzobs]*numpy.log((nobs/y_model)[nzobs])).sum()
        return alpha, beta
    
    
    def __cov_sort_in_place(self, C):
        """Expand the matrix C in place, so as to account
        for parameters being held fixed."""
        C[self.mfit:, :] = 0.0
        C[:, self.mfit:] = 0.0
        k = self.mfit - 1
        for j in range(self.nparam-1, -1, -1):
            if j==k: break
            if self.ia[j]:
                for m in range(self.nparam):
                    C[m,k], C[m,j] = C[m,j], C[m,k]
                    C[k,m], C[j,m] = C[j,m], C[k,m]
                k -= 1




class MouseClickReader(object):
    """Object to serve as a callback for reading mouse clicks in data coordinates
    in pylab plots.  Will store self.b, .x, .y giving the button pressed,
    and the x,y data coordinates of the pointer when clicked.
    
    Usage example (ought to be here...):
    """
    def __init__(self, figure):
        """Connect to button press events on a pylab figure.
        \param figure The matplotlib.figure.Figure from which to capture mouse click events."""
        ## The button number of the last mouse click inside a plot.
        self.b = 0
        ## The x location of the last mouse click inside a plot.
        self.x = 0
        ## The y location of the last mouse click inside a plot.
        self.y = 0
        ## The Figure to whose events we are connected.
        self.fig=figure
        ## The connection ID for matplotlib event handling.
        self.cid = self.fig.canvas.mpl_connect('button_press_event',self)
        
    def __call__(self, event):
        """When called, capture the latest button number and the x,y location in 
        plot units.  Store in self.b, .x, and .y."""
        self.b, self.x, self.y =  event.button, event.xdata, event.ydata
    def __del__(self):
        """Disconnect the button press event from this object."""
        self.fig.canvas.mpl_disconnect(self.cid)
        
