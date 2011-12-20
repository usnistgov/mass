"""
mass.mathstat.fitting

Several model- utilities, including:
* A histogram fitter that uses a full maximum likelihood fit.

Joe Fowler, NIST

Started March 24, 2011
December 15, 2011: forked from utilities.py
"""

## \file fitting.py
# \brief Model-fitting procedures used by Mass.
#
# Math utilities include:
# -# MaximumLikelihoodHistogramFitter
#

__all__ = ['MaximumLikelihoodHistogramFitter',
           'MaximumLikelihoodGaussianFitter',]


import numpy
import scipy.linalg

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
    ITMAX=1000  
    
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
        self.total_obs = self.nobs.sum()
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
        self.param_free = numpy.ones(self.nparam, dtype=numpy.bool)
        self.alpha=numpy.zeros((self.nparam,self.nparam), dtype=numpy.float)
        self.covar=numpy.zeros((self.nparam,self.nparam), dtype=numpy.float)


    def hold(self, i, val=None):
        """
        Hold parameter number <i> fixed either at value <val> or (by default) at its present 
        value.  Parameter is fixed until method free(i) is called.
        """
        self.param_free[i] = False
#        print 'Holding param %d'%i
        if val is not None:
            self.params[i] = val
        

    def free(self, i):
        """
        Release parameter <i> to float in the next fit.  If already free, then no effect.
        """
        self.param_free[i] = True


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
        
        no_change_ctr = 0
        alambda = 0.01
        self.mfit = self.param_free.sum()
        self.alpha, beta = self._mrqcof(self.params)
        
        atry = self.params.copy()
        prev_chisq = self.chisq
        for iter_number in range(self.ITMAX):
            temp = numpy.array(self.alpha)
            if no_change_ctr==self.DONE:
                alambda = 0.0 # use alambda=0 on last pass
            else:
                for j in range(self.mfit):
                    temp[j,j] += alambda*temp[j,j]
        
            try:
                da = scipy.linalg.solve(temp, beta[self.param_free])
            except scipy.linalg.LinAlgError, e:
                print 'temp (lambda=%f, iteration %d) is singular:'%(alambda,iter_number), temp, beta
                raise e

            if no_change_ctr==self.DONE:
                self.covar = numpy.zeros((self.nparam, self.nparam), dtype=numpy.float)
                self.covar[:self.mfit, :self.mfit] = scipy.linalg.inv(temp)
                self.__cov_sort_in_place(self.covar)
                return self.params, self.covar
            
            # Did the trial succeed?
            atry[self.param_free] = self.params[self.param_free] + da
            
            self.covar, da = self._mrqcof(atry)
            if abs(self.chisq-prev_chisq) < max(self.TOL, self.TOL*self.chisq): 
                no_change_ctr+=1
            if (self.chisq < prev_chisq ): # success: we've improved
                alambda *= 0.1
                prev_chisq = self.chisq
                self.alpha = self.covar.copy()
                beta = da.copy()
                self.params = atry.copy()
            else:   # failure.  Increase lambda and return to previous starting point.
                alambda *= 10
                self.chisq = prev_chisq
        
        raise RuntimeError("lev_marq_min.fit too many iterations")
        
    
    def _mrqcof(self, params):
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
        self.chisq = 2*(y_model.sum()-self.total_obs)  \
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
            if self.param_free[j]:
                for m in range(self.nparam):
                    C[m,k], C[m,j] = C[m,j], C[m,k]
                    C[k,m], C[j,m] = C[j,m], C[k,m]
                k -= 1




class MaximumLikelihoodGaussianFitter(MaximumLikelihoodHistogramFitter):
    """
    Object to fit a gaussian to a histogram, using the proper likelihood.
    That is, assume that events in each bin are independent and are
    Poisson-distributed with an expectation equal to the theory.
    
    This implementation is a special case of the more general 
    MaximumLikelihoodHistogramFitter, which it subclasses.  Unlike the general
    fitter, we don't require a model (theory) function nor a way to compute a
    discretized gradient, because these are done on paper.
    
    User must supply the bin centers, bin contents, an initial guess at the 
    theory parameters.
    
    To do: we currently include a constant background level.  There's no reason
    this could not be an Nth order polynomial.
    
    For information on the algorithm and its implementation, see 
    MaximumLikelihoodHistogramFitter.NOTES  
    """
    
    def __init__(self, x, nobs, params, TOL=1e-3):
        """
        Initialize the fitter, making copies of the input data.
        
        Inputs:
        <x>          The histogram bin centers (used in computing model values)
        <nobs>       The histogram bin contents (must be same length as <x>)
        <params>     The theory's vector of parameter starting guesses.  They are [FWHM, centroid,
                     peak value, (constant) background, background slope].  These last two are
                     optional and will be both set to zero and fixed if not given as input.
        <TOL>       The fractional or absolute tolerance on the minimum "MLE Chi^2".
                    When self.DONE successive iterations fail to improve the MLE Chi^2 by this
                    much (aboslutely or fractionally), then fitting will return successfully.
        """
        self.x = numpy.array(x)
        self.ndat = len(x)
        self.scaled_x = numpy.arange(self.ndat)*2.0/self.ndat - 1.0
        self.nobs = numpy.array(nobs)
        self.total_obs = self.nobs.sum()
        if len(self.nobs) != self.ndat:
            raise ValueError("x and nobs must have the same length")

        self.mfit = self.nparam = 0
        self.set_parameters(params)
        if self.nparam<3 or self.nparam>5:
            raise ValueError("params requires 3 to 5 values")
        elif self.nparam == 3:
            self.set_parameters(numpy.hstack((params,[0,0])))
            self.hold(3, 0.0)
            self.hold(4, 0.0)
        elif self.nparam == 4:
            self.set_parameters(numpy.hstack((params,[0])))
            self.hold(4, 0.0)
            
        self.TOL = TOL
        self.chisq = 0.0

    EIGHT_LN2 = 8*numpy.log(2)
    FOUR_LN2 = 4*numpy.log(2)
    
    def _mrqcof(self, params):
        """Used by fit to evaluate the linearized fitting matrix alpha and vector beta,
        and to calculate chisq.  Returns (alpha,beta) and stores chisq in self.chisq"""
        
        # Careful!  Do this smart, or this routine will bog down the entire
        # fit hugely.
        # Precompute g = (x-param1)/param0
        # and h = exp(-ln2* g**2)
        
        g = (self.x - params[1])/params[0]
        h = numpy.exp(-self.FOUR_LN2*g*g)
        params[2] = abs(params[2])
        params[3] = abs(params[3])
        y_model = params[2]*h
        if params[3] > 0:
            y_model += params[3]
        if params[4] != 0:
            y_model += params[4]*self.scaled_x
        y_model[y_model<1e-10] = 1e-10
        
        dy_dp1 = self.EIGHT_LN2 * g*h*params[2]/params[0]
        dyda = numpy.vstack(( g*dy_dp1, dy_dp1, h, numpy.ones_like(h), self.scaled_x ))
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
        self.chisq = 2*(y_model.sum()-self.total_obs)  \
                + 2*(nobs[nzobs]*numpy.log((nobs/y_model)[nzobs])).sum()
        return alpha, beta
    
    
