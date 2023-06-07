"""
mass.mathstat.fitting

Model-fitting utilities, including at this point only:
* MaximumLikelihoodHistogramFitter: a histogram fitter that performs a full
    maximum-likelihood fit.

Joe Fowler, NIST
"""

import numpy as np
import scipy as sp
import time
import logging

LOG = logging.getLogger("mass")

__all__ = ['MaximumLikelihoodHistogramFitter', 'kink_model', 'fit_kink_model']


class MaximumLikelihoodHistogramFitter:
    """Object to fit a theory having 1 or more free parameters to a histogram,
    using the proper likelihood.  That is, assume that events in each bin are
    independent and are Poisson-distributed with an expectation equal to the
    theory.

    This implementation is fast (only requires 2x as long as the chisquared
    minimizer sp.optimize.leastsq, which assumes Gaussian-distributed data),
    and it removes the biases that arise from fitting parameters by minimizing
    chi-squared, as if the data were Gaussian.

    User must supply the bin centers, bin contents, an initial guess at the
    theory parameters, and the theory function.  An optional theory gradient
    can be supplied to compute gradient of the theory w.r.t. its parameters.
    If not supplied, gradients will be computed by one-sided finite differences.

    An optional upper or lower bound is possible on each parameter, either in
    the constructor or by calling the setbounds() method. The bounds are handled by
    a transformation between "parameters" and "internal parameters". These trans-
    formations are based on what MINUIT does.  For more information, see
    http://github.com/jjhelmus/leastsqbound-scipy or
    http://lmfit.github.io/lmfit-py/bounds.html

    For information on the fitting algorithm and its implementation, see
    MaximumLikelihoodHistogramFitter.NOTES
    """

    # Further algorithm notes beyond the docstring.
    NOTES = """
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
    Python+np of the Levenberg-Marquardt solver class Fitmrq appearing in C++ in
    Numerical Recipes, 3rd Edition, with appropriate changes so as to minimize
    -2 times the log of the likelihood ratio, rather than a true chi-squared.
    """

    # This many steps with negligible "chisq" change cause normal exit
    DONE = 4

    # This many steps cause RuntimeError for excessive iterations
    ITMAX = 1000

    def __init__(self, x, nobs, params, theory_function, theory_gradient=None,
                 epsilon=1e-5, TOL=1e-5, timeout=30.0, seed=None):
        """Initialize the fitter, making copies of the input data.

        Args:
            x: The histogram bin centers (used in computing model values)
            nobs: The histogram bin contents (must be same length as <x>)
            params: The theory's vector of parameter starting guesses
            theory_function (callable): A function of (params, x) returning the modeled bin contents as
                a vector of the same length as x
            theory_gradient (callable): A function of (params, x) returning the gradient of the modeled
                bin contents as a 2D vector of shape (len(params), len(x)).  If None (the
                default), then the gradient is computed by one-sided finite differences.
            epsilon (float or ndarray):  If theory_gradient is to be approximated,
                use this value for the step size.
            TOL: The fractional or absolute tolerance on the minimum "MLE Chi^2".
                When self.DONE successive iterations fail to improve the MLE Chi^2 by this
                much (absolutely or fractionally), then fitting will return successfully.
            timeout: Fail if the fitting takes more than this many seconds.
        """
        self.rng = np.random.default_rng(seed)
        self.x = np.array(x)
        self.ndat = len(x)
        self.nobs = np.array(nobs)
        self.total_obs = self.nobs.sum()
        if len(self.nobs) != self.ndat:
            raise ValueError("x and nobs must have the same length")

        self.mfit = 0
        self.nparam = len(params)
        self.timeout = timeout

        # Handle bounded parameters with translations between internal (-inf,+inf) and bounded
        # parameters. Until and unless self.setbounds is called, we'll assume that no
        # parameters have bounds.
        self.lowerbound = [None for _ in range(self.nparam)]
        self.upperbound = [None for _ in range(self.nparam)]
        self.internal2bounded = [lambda x:x for _ in range(self.nparam)]
        self.bounded2internal = [lambda x:x for _ in range(self.nparam)]
        self.boundedinternal_grad = [lambda x:x for _ in range(self.nparam)]
        for pnum in range(self.nparam):
            self.setbounds(pnum, None, None)

        self.params = self.param_free = self.covar = self.chisq = None
        self.set_parameters(params)
        self.penalty = None

        self.theory_function = theory_function
        if np.isscalar(epsilon):
            self.epsilon = epsilon + np.zeros_like(params)
        else:
            if len(epsilon) != self.nparam:
                msg = "epsilon must be a scalar or if a vector, " + \
                    "a vector of the same length as params"
                raise ValueError(msg)
            self.epsilon = np.array(epsilon)
        self.theory_gradient = theory_gradient
        if theory_gradient is None:
            self.theory_gradient = self.__discrete_gradient
        self.TOL = TOL
        self.chisq = 0.0
        self.iterations = 0

    def set_parameters(self, params):
        """Set the initial guess at the parameters.

        Args:
            params (ndarray): a vector containing the parameter initial guess.

        This call will clear the "hold state" of any parameters that used to be
        held.  This call also erases the self.covar matrix and sets the current
        chisq.
        """
        self.mfit = self.nparam = len(params)
        self.params = np.array(params)
        self.param_free = np.ones(self.nparam, dtype=bool)
        self.covar = np.zeros((self.nparam, self.nparam), dtype=float)
        self.chisq = 1e99
        # Force parameters into bounded range
        for pnum in range(self.nparam):
            a, b = self.lowerbound[pnum], self.upperbound[pnum]
            if a is not None and self.params[pnum] < a:
                print("Warning: param[%d] = %f was forced to lower limit %f" % (pnum, params[pnum], a))
                self.params[pnum] = a
            if self.upperbound[pnum] is not None:
                print("Warning: param[%d] = %f was forced to upper limit %f" % (pnum, params[pnum], b))
                self.params[pnum] = b

    def hold(self, i, val=None):
        """Force one parameter to be held.

        Args:
            i (int): The parameter number to be held.
            val (float): The value to hold it at, or if None, then do not change
                the current value (default None)

        Parameter number i is fixed until method free(i) is called.
        """
        self.param_free[i] = False
        if val is not None:
            self.params[i] = val

    def free(self, i):
        """Allow one held parameter to be very freely.

        Args:
            i (int): The parameter number to be freed.

        Parameter number i is freed to vary in the next fit. If already free, then no effect.
        """
        self.param_free[i] = True

    def setbounds(self, pnum, lower=None, upper=None):
        """Set lower and/or upper limits on the value that parameter pnum can take."""
        if pnum < 0 or pnum >= self.nparam:
            raise ValueError("pnum must be in range [0,%d]" % (self.nparam-1))
        self.lowerbound[pnum] = lower
        self.upperbound[pnum] = upper
        if lower is None and upper is None:  # no constraints
            self.internal2bounded[pnum] = lambda x: x
            self.bounded2internal[pnum] = lambda x: x
            self.boundedinternal_grad[pnum] = lambda x: 1.0
        elif upper is None:     # only lower bound
            self.internal2bounded[pnum] = lambda x: lower - 1. + np.sqrt(x * x + 1.)
            self.bounded2internal[pnum] = lambda x: np.sqrt((1.0 + (x - lower))**2 - 1.)
            self.boundedinternal_grad[pnum] = lambda x: x/np.sqrt(x * x + 1.)
        elif lower is None:     # only upper bound
            self.internal2bounded[pnum] = lambda x: upper + 1. - np.sqrt(x * x + 1.)
            self.bounded2internal[pnum] = lambda x: np.sqrt(((upper - x) + 1.)**2 - 1.)
            self.boundedinternal_grad[pnum] = lambda x: -x/np.sqrt(x * x + 1.)
        else:                   # lower and upper bounds
            self.internal2bounded[pnum] = lambda x: lower + ((upper - lower) / 2.) * (np.sin(x) + 1.)
            self.bounded2internal[pnum] = lambda x: np.arcsin(2.*(x-lower)/(upper-lower) - 1.)
            self.boundedinternal_grad[pnum] = lambda x: ((upper - lower) / 2.) * np.cos(x)

    def set_penalty(self, penalty=None):
        """Set a penalty function, to be added to chisquared.

        This allows parameters to be 'nudged' without enforcing hard limits.
        The function `penalty` must be of the form

        (P, gradP, hessianP) = penalty(params).

        P is the scalar value of the penalty;
        gradP is its derivative w.r.t. each parameter;
        hessianP is its matrix of second derivatives w.r.t. each parameter.

        Or if no penalty is desired, `penalty=None` should be chosen (which is the default).
        """
        # Be sure the penalty returns something sensible
        if penalty is not None:
            example_params = self.params
            npar = len(example_params)
            pen, grad, hess = penalty(example_params)
            assert np.isscalar(pen)
            assert len(grad) == npar
            assert hess.shape == (npar, npar)
        self.penalty = penalty

    def __discrete_gradient(self, p, x):
        """
        Estimate the gradient of our minimization function self.theory_function
        with respect to each of the parameters when we don't have an exact expression for it.
        Use 2-sided differences with steps of size self.epsilon away
        from the test point <p> at an array of points <x>.

        The gradient will be zero for any parameters held (self.param_free[i] == False)

        Args:
            p (array): the parameters at which to find the gradient.
            x (array): the measured data values.

        If you have a way to return the true gradient dy/dp_i | p,x, then you
        should use that instead as function self.theory_gradient.
        """
        nx = len(x)
        npar = len(p)
        dyda = np.zeros((npar, nx), dtype=float)
        for i, dx in enumerate(self.epsilon):
            # Don't compute gradient on parameters being held fixed
            if not self.param_free[i]:
                continue
            dxminus = dxplus = dx
            if self.upperbound[i] is not None and p[i] + dxplus >= self.upperbound[i]:
                dxplus = .9*(self.upperbound[i]-p[i])
            if self.lowerbound[i] is not None and p[i] - dxminus <= self.lowerbound[i]:
                dxminus = .9*(p[i]-self.lowerbound[i])

            # Use a symmetric interval, when possible. But if one side must take a zero-sized
            # step, then obviously don't make them be symmetrically zero. Duh.
            if dxplus*dxminus > 0:
                dxplus = dxminus = min(dxplus, dxminus)

            p2 = p.copy()
            p2[i] = p[i] + dxplus
            tf_plus = self.theory_function(p2, x)

            p2[i] = p[i] - dxminus
            tf_minus = self.theory_function(p2, x)
            dyda[i, :] = (tf_plus - tf_minus)/(dxplus + dxminus)
        return dyda

    def fit(self, verbose=False):
        """
        Iterate to reduce the "maximum likelihood chi-squared" of a fit between a histogram
        of data points self.x, self.nobs and a nonlinear function that depends on the
        coefficients self.param.

        When chisq is no longer decreasing for self.DONE iterations, return
        (params, covariance) where <params> is the best-fit value of each
        parameter (recall that some can optionally be held out of the fit by
        calling self.free and self.hold), and where <covariance> is the square
        matrix of estimated covariances between the parameters.

        When self.ITMAX iterations are reached, this method raises a RuntimeError.
        """

        # Parameters that are bounded but not held need to be placed between the bounds.
        # Also if at the bound, they should be "pinged" away from the bound.
        for i in range(self.nparam):
            if self.param_free[i]:
                if self.upperbound[i] == self.lowerbound[i] and self.lowerbound[i] is not None:
                    self.param_free[i] = False
                    continue

                if self.lowerbound[i] is not None:
                    if self.lowerbound[i] > self.params[i]:
                        self.params[i] = self.lowerbound[i]
                    if self.lowerbound[i] == self.params[i]:
                        self.params[i] += self.rng.uniform(0, self.epsilon[i])
                if self.upperbound[i] is not None:
                    if self.upperbound[i] < self.params[i]:
                        self.params[i] = self.upperbound[i]
                    if self.upperbound[i] == self.params[i]:
                        self.params[i] -= self.rng.uniform(0, self.epsilon[i])

        no_change_counter = 0
        lambda_coef = 0.01
        self.mfit = self.param_free.sum()
        self.internal = np.array([f(p) for f, p in zip(self.bounded2internal, self.params)])
        alpha, beta, prev_chisq = self._mrqcof(self.internal)

        atry = self.internal.copy()
        t_start = time.time()
        for iter_number in range(self.ITMAX):

            alpha_prime = np.array(alpha)
            for j in range(self.mfit):
                alpha_prime[j, j] += lambda_coef*alpha_prime[j, j]

            try:
                delta_params = sp.linalg.solve(alpha_prime, beta[self.param_free],
                                               overwrite_a=False, overwrite_b=False)
            except (sp.linalg.LinAlgError, ValueError) as ex:
                LOG.debug('alpha (lambda=%f, iteration %d) is singular or has NaN:' %
                          (lambda_coef, iter_number))
                LOG.debug('Internal: ', self.internal)
                LOG.debug('Params: ', self.params)
                raise ex

            # Did the trial succeed?
            atry[self.param_free] = self.internal[self.param_free] + delta_params
            trial_alpha, trial_beta, trial_chisq = self._mrqcof(atry)

            # When the chisq hasn't changed appreciably in self.DONE iterations, we return with success.
            # All other exits from this method are exceptions.
            if abs(trial_chisq-prev_chisq) < max(self.TOL, self.TOL*self.chisq):
                no_change_counter += 1

                if no_change_counter >= self.DONE:
                    internal_covar = sp.linalg.inv(alpha)
                    dpdi_grad = np.array([f(p) for f, p in zip(self.boundedinternal_grad,
                                                               self.internal)])[self.param_free]
                    external_covar = ((internal_covar*dpdi_grad).T*dpdi_grad).T
                    self.covar[:self.mfit, :self.mfit] = external_covar
                    self.__cov_sort_in_place(self.covar)
                    self.iterations = iter_number
                    self.chisq = trial_chisq
                    return self.params, self.covar
            else:
                no_change_counter = 0

            if (trial_chisq < prev_chisq):  # success: we've improved chisq
                lambda_coef *= 0.1
                alpha = trial_alpha
                beta = trial_beta
                self.internal = atry.copy()
                self.params = np.array([f(p) for f, p in zip(self.internal2bounded, self.internal)])
                if verbose:
                    print(f"Improved: chisq={trial_chisq:9.4e}->{prev_chisq:9.4e} l={lambda_coef:.1e} params={self.params[:2]}...")
                self.chisq = prev_chisq = trial_chisq
            else:   # failure.  Increase lambda and return to previous starting point.
                lambda_coef *= 10.0
                if verbose:
                    print(f"No imprv: chisq={trial_chisq:9.4e} >= {prev_chisq:9.4e} l={lambda_coef:.1e} params={self.params[:2]}...")
                self.chisq = prev_chisq

            dt = time.time()-t_start
            if dt > self.timeout:
                raise RuntimeError("MaximumLikelihoodHistogramFitter.fit() timed out after %.2f sec (%d iterations)" %
                                   (dt, iter_number))

        raise RuntimeError("MaximumLikelihoodHistogramFitter.fit() reached ITMAX=%d iterations" % self.ITMAX)

    def _mrqcof(self, internal):
        """Used by fit to evaluate the linearized fitting matrix alpha and vector beta,
        and to calculate chisq.  Returns (alpha,beta,chisq).  Does NOT update self.chisq

        Careful! When any number of the M parameters are 'held', then the
        returned Hessian alpha has size NxN, while the gradient beta is of size
        M>=N.  (That is, there are M total parameters and only N of them are
        variable parameters.)
        """

        # Careful!  Do this smart, or this routine will bog down the entire
        # fit hugely.
        params = np.array([f(p) for f, p in zip(self.internal2bounded, internal)])
        dpdi_grad = np.array([f(p) for f, p in zip(self.boundedinternal_grad, internal)])

        y_model = self.theory_function(params, self.x)
        dyda = (self.theory_gradient(params, self.x).T * dpdi_grad).T
        if y_model.min() <= 0:
            y_model[y_model <= 0] = 1e-10
        if np.any(np.isnan(y_model)):
            y_model[np.isnan(y_model)] = 1e-10
        dyda_over_y = (dyda/y_model)
        nobs = self.nobs
        y_resid = nobs - y_model

        beta = (y_resid*dyda_over_y).sum(axis=1)

        alpha = np.zeros((self.mfit, self.mfit), dtype=float)
        pfnz = self.param_free.nonzero()[0]
        for i in range(self.mfit):
            overallrow = pfnz[i]  # convert from the ith free row to the overall row #
            alpha[i, i] = (nobs*(dyda_over_y[overallrow, :]**2)).sum()
            for j in range(i):
                overallcol = pfnz[j]  # convert from the jth free col to the overall col #
                alpha[i, j] = (nobs*dyda_over_y[overallrow, :]*dyda_over_y[overallcol, :]).sum()
                alpha[j, i] = alpha[i, j]
            if alpha[:, i].sum() == 0:
                alpha[i, i] = 1.0  # This prevents a singular matrix error

        nonzero_obs = nobs > 0
        nobsNZ = nobs[nonzero_obs]
        y_modelNZ = y_model[nonzero_obs]
        chisq = 2*(y_model.sum()-self.total_obs) + \
            2*(nobsNZ*(np.log(nobsNZ)-np.log(y_modelNZ))).sum()

        # If a penalty is being imposed on the parameters, change the return values
        # of the Hessian, the gradient, and the function value accordingly.
        if self.penalty is not None:
            penalty, grad, hessian = self.penalty(params)
            chisq += penalty
            beta -= 0.5 * grad
            for i, pnum in enumerate(pfnz):
                alpha[i, :] += 0.5*hessian[pnum, pfnz]

        return alpha, beta, chisq

    def __cov_sort_in_place(self, C):
        """Expand the matrix C in place, so as to account
        for parameters being held fixed."""
        C[self.mfit:, :] = 0.0
        C[:, self.mfit:] = 0.0
        k = self.mfit - 1
        for j in range(self.nparam-1, -1, -1):
            if j == k:
                break
            if self.param_free[j]:
                for m in range(self.nparam):
                    C[m, k], C[m, j] = C[m, j], C[m, k]
                    C[k, m], C[j, m] = C[j, m], C[k, m]
                k -= 1


def kink_model(k, x, y):
    """Compute a kinked-linear model on data {x,y} with kink at x=k.

    The model is f(x) = a+b(x-k) for x<k and f(x)=a+c(x-k) for x>=k, where
    the 4 parameters are {k,a,b,c}, representing the kink at (x,y)=(k,a) and
    slopes of b and c for x<k and x>= k.

    For a fixed k, the model is linear in the other parameters, whose linear
    least-squares values can thus be found exactly by linear algebra. This
    function computes them.

    Returns (model_y, (a,b,c), X2) where:
    model_y is an array of the model y-values;
    (a,b,c) are the best-fit values of the linear parameters;
    X2 is the sum of square differences between y and model_y.

    Fails (raising ValueError) if k doesn't satisfy x.min() < k < x.max().
    """
    xi = x[x < k]
    yi = y[x < k]
    xj = x[x >= k]
    yj = y[x >= k]
    N = len(x)
    if len(xi) == 0 or len(xj) == 0:
        xmin = x.min()
        xmax = x.max()
        raise ValueError(f"k={k:g} should be in range [xmin,xmax], or [{xmin:g},{xmax:g}].")

    dxi = xi-k
    dxj = xj-k
    si = dxi.sum()
    sj = dxj.sum()
    si2 = (dxi**2).sum()
    sj2 = (dxj**2).sum()
    A = np.array([[N, si, sj],
                  [si, si2, 0],
                  [sj, 0, sj2]])
    v = np.array([y.sum(), (yi*dxi).sum(), (yj*dxj).sum()])
    a, b, c = abc = np.linalg.solve(A, v)
    model = np.hstack([a+b*dxi, a+c*dxj])
    X2 = ((model-y)**2).sum()
    return model, abc, X2


def fit_kink_model(x, y, kbounds=None):
    """Find the linear least-squares solution for a kinked-linear model.

    The model is f(x) = a+b(x-k) for x<k and f(x)=a+c(x-k) for x>=k, where
    the 4 parameters are {k,a,b,c}, representing the kink at (x,y)=(k,a) and
    slopes of b and c for x<k and x>= k.

    Given k, the model is linear in the other parameters, which can thus be
    found exactly by linear algebra. The best value of k is found by use of
    the Bounded method of the sp.optimize.minimize_scalar() routine.

    x - The input data x-values;
    y - The input data y-values;
    kbounds - Bounds on k. If (u,v), then the minimize_scalar is
        used to find the best k strictly in u<=k<=v. If None, then use the Brent
        method, which will start with (b1,b2) as a search bracket where b1 and b2
        are the 2nd lowest and 2nd highest values of x.

    Examples
    --------
    x = np.arange(10, dtype=float)
    y = np.array(x)
    truek = 4.6
    y[x>truek] = truek
    y += np.random.default_rng().standard_normal(len(x))*.15
    model, (kbest,a,b,c), X2 = fit_kink_model(x, y, kbounds=(3,6))
    plt.clf()
    plt.plot(x, y, "or", label="Noisy data to be fit")
    xi = np.linspace(x[0], kbest, 200)
    xj = np.linspace(kbest, x[-1], 200)
    plt.plot(xi, a+b*(xi-kbest), "--k", label="Best-fit kinked model")
    plt.plot(xj, a+c*(xj-kbest), "--k")
    plt.legend()
    """
    def penalty(k, x, y):
        _, _, X2 = kink_model(k, x, y)
        return X2

    if kbounds is None:
        kbounds = (x.min(), x.max())
    elif kbounds[0] < x.min() or kbounds[1] > x.max():
        raise ValueError("kbounds (%s) must be within the range of x data" % kbounds)
    optimum = sp.optimize.minimize_scalar(penalty, args=(x, y), method="Bounded",
                                          bounds=kbounds)
    kbest = optimum.x
    model, abc, X2 = kink_model(kbest, x, y)
    return model, np.hstack([[kbest], abc]), X2
