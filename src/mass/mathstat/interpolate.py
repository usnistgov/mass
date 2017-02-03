"""
Created on Feb 20, 2014

@author: fowlerj
interpolate.py

Module mass.mathstat.interpolate

Contains interpolations functions not readily available elsewhere.

CubicSpline   - Perform an exact cubic spline through the data, with
                either specified slope at the end of the interval or
                'natural boundary conditions' (y''=0 at ends).

LinterpCubicSpline - Create a new CubicSpline that's the linear interpolation
                of two existing ones.

SmoothingSpline - Create a smoothing spline that does not exactly interpolate
                  the data, but finds the cubic spline with lowest "curvature
                  energy" among all splines that meet the maximum allowed value
                  of chi-squared.

SmoothingSplineLog - Create a SmoothingSpline using the log of the x,y points.

NaturalBsplineBasis - A tool for expressing a spline basis using B-splines but
                  also enforcing 'natural boundary conditions'.

Joe Fowler, NIST
"""

__all__ = ['CubicSpline', 'SmoothingSpline', 'SmoothingSplineLog']

import numpy as np
import scipy as sp
from scipy.interpolate import splev

from mass.mathstat.derivative import *


class CubicSpline(object):
    """An exact cubic spline, with either a specified slope or 'natural boundary
    conditions' (y''=0) at ends of interval.

    Note that the interface is similar
    to scipy.interpolate.InterpolatedUnivariateSpline, but the behavior is different.
    The scipy version will remove the 2nd and 2nd-to-last data points from the
    set of knots as a way of using the 2 extra degrees of freedom. This class
    instead sets the 1st or 2nd derivatives at the end of the interval to use
    the extra degrees of freedom.

    This code is inspired by section 3.3. of Numerical Recipes, 3rd Edition.

    Usage:
    x=np.linspace(4,12,20)
    y=(x-6)**2+np.random.standard_normal(20)
    cs = mass.CubicSpline(x, y)
    plt.clf()
    plt.plot(x,y,'ok')
    xa = np.linspace(0,16,200)
    plt.plot(xa, cs(xa), 'b-')
    """

    def __init__(self, x, y, yprime1=None, yprimeN=None):
        """
        Create an exact cubic spline representation for the function y(x).
        'Exact' means that the spline will strictly pass through the given points.

        The user can give specific values for the slope at either boundary through
        <yprime1> and <yprimeN>, or can use the default value of None. The
        slope of None means to use 'natural boundary conditions' by fixing the
        second derivative to zero at that boundary.
        """
        argsort = np.argsort(x)
        self._x = np.array(x, dtype=np.float)[argsort]
        self._y = np.array(y, dtype=np.float)[argsort]
        self._n = len(x)
        self._y2 = np.zeros(self._n, dtype=np.float)
        self.yprime1 = yprime1
        self.yprimeN = yprimeN
        self._compute_y2()

    def _compute_y2(self):
        self.dy = self._y[1:]-self._y[:-1]
        self.dx = self._x[1:] - self._x[:-1]

        u = self.dy/self.dx
        u[1:] = u[1:] - u[:-1]

        # For natural boundary conditions, u[0]=y2[0]=0.
        if self.yprime1 is None:
            u[0] = 0
            self._y2[0] = 0
        else:
            u[0] = (3.0/self.dx[0])*(self.dy[0]/self.dx[0]-self.yprime1)
            self._y2[0] = -0.5

        for i in range(1, self._n-1):
            sig = self.dx[i-1]/(self._x[i+1]-self._x[i-1])
            p = sig*self._y2[i-1]+2.0
            self._y2[i] = (sig-1.0)/p
            u[i] = (6*u[i]/(self._x[i+1]-self._x[i-1])-sig*u[i-1])/p

        # Again, the following is only for natural boundary conditions
        if self.yprimeN is None:
            qn = un = 0.0
        else:
            qn = 0.5
            un = (3.0/self.dx[-1])*(self.yprimeN-self.dy[-1]/self.dx[-1])
        self._y2[self._n-1] = (un-qn*u[self._n-2])/(qn*self._y2[self._n-2]+1.0)

        # Backsubstitution:
        for k in range(self._n-2, -1, -1):
            self._y2[k] = self._y2[k]*self._y2[k+1]+u[k]

        if self.yprime1 is None:
            self.yprime1 = self.dy[0]/self.dx[0] - self.dx[0]*(self._y2[0]/3.+self._y2[1]/6.)
        if self.yprimeN is None:
            self.yprimeN = self.dy[-1]/self.dx[-1] + self.dx[-1]*(self._y2[-2]/6.+self._y2[-1]/3.)

    def __call__(self, x, der=0):
        scalar = np.isscalar(x)
        x = np.asarray(x)
        if x.size == 0:
            return np.array([])
        elif x.size == 1:
            x.shape = (1,)
        result = np.zeros_like(x, dtype=np.float)

        # Find which interval 0,...self._n-2 contains the points (or extrapolates to the points)
        position = np.searchsorted(self._x, x)-1

        # Here, position == -1 means extrapolate below the first interval.
        extrap_low = position < 0
        if extrap_low.any():
            if der == 0:
                h = x[extrap_low]-self._x[0]  # will be negative
                result[extrap_low] = self._y[0] + h*self.yprime1
            elif der == 1:
                result[extrap_low] = self.yprime1
            elif der > 1:
                result[extrap_low] = .0

        # position = self._n-1 means extrapolate above the last interval.
        extrap_hi = position >= self._n-1
        if extrap_hi.any():
            if der == 0:
                h = x[extrap_hi] - self._x[-1]  # will be positive
                result[extrap_hi] = self._y[-1] + h*self.yprimeN
            elif der == 1:
                result[extrap_hi] = self.yprimeN
            elif der > 1:
                result[extrap_hi] = .0

        interp = np.logical_and(position >= 0, position < self._n-1)
        if interp.any():
            klo = position[interp]
            khi = klo+1
            dx = self.dx[klo]
            a = (self._x[khi] - x[interp]) / dx
            b = (x[interp] - self._x[klo]) / dx

            if der == 0:
                result[interp] = a * self._y[klo] + b * self._y[khi] \
                                                  + ((a**3 - a) * self._y2[klo] +
                                                     (b**3 - b)*self._y2[khi]) * dx * dx / 6.0
            elif der == 1:
                result[interp] = -self._y[klo] / dx + self._y[khi] / dx \
                                                    + ((-a**2 + 1.0 / 3) * self._y2[klo] +
                                                      (b**2 - 1.0 / 3) * self._y2[khi]) * dx / 2.0
            elif der == 2:
                result[interp] = a * self._y2[klo] + b * self._y2[khi]
            elif der == 3:
                result[interp] = (-self._y2[klo] + self._y2[khi]) * dx
            elif der > 3:
                result[interp] = .0

        if scalar:
            result = result[()]
        return result


class LinterpCubicSpline(CubicSpline):
    """A CubicSpline object which is a linear combination of CubicSpline objects
    s1 and s2, effectively fraction*s1 + (1-fraction)*s2
    """

    def __init__(self, s1, s2, fraction):
        if s1._n != s2._n:
            raise ValueError("Splines must be of the same length to be linearly interpolated")
        if np.max(np.abs(s1._x - s2._x)) > 1e-3:
            raise ValueError("Splines must have same abscissa values to be interpolated")

        wtsum = lambda a,b,frac: a*frac + b*(1-frac)
        self._n = s1._n
        self._x = s1._x
        self._y = wtsum(s1._y, s2._y, fraction)
        self._y2 = wtsum(s1._y2, s2._y2, fraction)
        self.yprime1 = wtsum(s1.yprime1, s2.yprime1, fraction)
        self.yprimeN = wtsum(s1.yprimeN, s2.yprimeN, fraction)
        self.dx = wtsum(s1.dx, s2.dx, fraction)
        self.dy = wtsum(s1.dy, s2.dy, fraction)


class NaturalBsplineBasis(object):
    """Represent a cubic B-spline basis in 1D with natural boundary conditions.
    That is, f''(x)=0 at the first and last knots. This constraint reduces the
    effective number of basis functions from (2+Nknots) to Nknots.

    Usage:
    knots = [0,5,8,9,10,12]
    basis = NaturalBsplineBasis(knots)
    x = np.linspace(0, 12, 200)
    plt.clf()
    for id in range(len(knots)):
        plt.plot(x, basis(x, id))
    """

    def __init__(self, knots):
        """Initialization requires only the list of knots.
        """
        Nk = len(knots)
        b, e = knots[0], knots[-1]
        padknots = np.hstack([[b, b, b], knots, [e, e, e]])

        # Combinations of basis function #1 into 2 and 3 (and #N+2 into N+1
        # and N) are used to enforce the natural B.C. of f''(x)=0 at the ends.
        lowfpp = np.zeros(3, dtype=float)
        hifpp = np.zeros(3, dtype=float)
        for i in (0, 1, 2):
            scoef = np.zeros(Nk + 2, dtype=float)
            scoef[i] = 1.0
            lowfpp[i] = splev(b, (padknots, scoef, 3), der=2)
        for i in (0, 1, 2):
            scoef = np.zeros(Nk + 2, dtype=float)
            scoef[Nk + 1 - i] = 1.0  # go from last to 3rd-to-last
            hifpp[i] = splev(e, (padknots, scoef, 3), der=2)
        self.coef_b = -lowfpp[1:3] / lowfpp[0]
        self.coef_e = -hifpp[1:3] / hifpp[0]

        self.Nk = Nk
        self.knots = np.array(knots)
        self.padknots = padknots

    def __call__(self, x, id, der=0):
        if id < 0 or id >= self.Nk:
            raise ValueError("Require 0 <= id < Nk=%d" % self.Nk)
        coef = np.zeros(self.Nk + 2, dtype=float)
        coef[id + 1] = 1.0
        if id < 2:
            coef[0] = self.coef_b[id]
        elif id >= self.Nk - 2:
            coef[-1] = self.coef_e[self.Nk-id-1]
        return splev(x, (self.padknots, coef, 3), der=der)

    def values_matrix(self, der=0):
        """Return matrix M where M_ij = value at knot i for basis function j.
        If der>0, then return the derivative of that order instead of the value."""
        # Note the array is naturally built by vstack as the Transpose of what we want.
        return np.vstack([self(self.knots, id, der=der)] for id in range(self.Nk)).T

    def expand_coeff(self, beta):
        """Given coefficients of this length-Nk basis, return the coefficients
        needed by FITPACK, which are of length Nk+2."""
        c = np.hstack([[0], beta, [0]])
        c[0] = beta[0]*self.coef_b[0] + beta[1]*self.coef_b[1]
        c[-1] = beta[-1]*self.coef_e[0] + beta[-2]*self.coef_e[1]
        return c


class SmoothingSpline(object):
    """A callable object that performs a smoothing cubic spline operation, using
    the NaturalBsplineBasis object for the basis representation of splines.

    The smoothing spline is the cubic spline minimizing the "curvature
    energy" subject to a constraint that the maximum allowed chi-squared is
    equal to the number of data points. Here curvature energy is defined as
    the integral of the square of the second derivative from the lowest to
    the highest knots.

    For a proof see Reinsch, C. H. (1967). "Smoothing by spline functions."
    Numerische Mathematik, 10(3), 177–183. http://doi.org/10.1007/BF02162161
    """

    def __init__(self, x, y, dy, dx=None, maxchisq=None):
        """Smoothing spline for data {x,y} with errors {dy} on the y values
        and {dx} on the x values (or zero if not given).

        If dx errors are given, a global quadratic fit is done to the data to
        estimate the local slope. If that's a poor choice, then you should
        combine your dx and dy errors to create a sensible single error list,
        and you should pass that in as dy.

        maxchisq specifies the largest allowed value of chi-squared (the sum of
        the squares of the differences y_i-f(x_i), divided by the variance v_i).
        If not given, this defaults to the number of data values. When a
        (weighted) least squares fit of a line to the data meets the maxchisq
        constraint, then the actual chi-squared will be less than maxchisq.
        """
        if dx is None:
            err = np.array(np.abs(dy))
        else:
            roughfit = np.polyfit(x, y, 2)
            slope = np.poly1d(np.polyder(roughfit, 1))(x)
            err = np.sqrt((dx*slope)**2 + dy*dy)

        self.x = np.array(x)
        self.y = np.array(y)
        self.dx = np.array(dx)
        self.dy = np.array(dy)
        self.err = err
        self.Nk = len(x)
        if maxchisq is None:
            self.maxchisq = self.Nk

        self.basis = NaturalBsplineBasis(self.x)
        self.N0 = self.basis.values_matrix(0)
        self.N2 = self.basis.values_matrix(2)
        self.Omega = self._compute_Omega(self.x, self.N2)
        self.smooth(chisq=self.maxchisq)

    def _compute_Omega(self, knots, N2):
        """Given the matrix M2 of second derivates at the knots (that is, M2_ij is
        the value of B_j''(x_i), second derivative of basis function #j at knot i),
        compute the matrix Omega, where Omega_ij is the integral over the entire
        domain of the product B_i''(x) B_j''(x). This can be done because each B_i(x)
        is piecewise linear, with the slope changes at each knot location."""

        Nk = len(knots)
        assert N2.shape[0] == Nk
        assert N2.shape[1] == Nk
        Omega = np.zeros_like(N2)
        for i in range(Nk):
            for j in range(i+1):
                for k in range(Nk-1):
                    Omega[i,j] += (N2[k+1,i]*N2[k,j]+N2[k+1,j]*N2[k,i])*(knots[k+1]-knots[k])/6.0
                for k in range(Nk):
                    Omega[i,j] += N2[k,i]*N2[k,j]*(knots[min(k+1,Nk-1)]-knots[max(0,k-1)])/3.0
                Omega[j,i] = Omega[i,j]
        return Omega

    def smooth(self, chisq=None):
        """Choose the value of the curve at the knots so as to achieve the
        smallest possible curvature subject to the constraint that the
        sum over all {x,y} pairs S = [(y-f(x))/dy]^2 <= chisq """
        Nk = self.Nk
        if chisq is None:
            chisq = Nk

        Dinv = self.err**(-2) # Vector but stands for diagonals of a diagonal matrix.
        NTDinv = self.N0.T * Dinv
        lhs = np.dot(NTDinv, self.N0)
        rhs = np.dot(self.N0.T, Dinv*self.y)

        def best_params(p):
            return np.linalg.solve(p*(lhs-self.Omega) + self.Omega, p * rhs)

        def chisq_difference(p, target_chisq):
            # If curvature is too small, the computation can become singular.
            # Avoid this by returning a crazy-high chisquared, as needed.
            try:
                beta = best_params(p)
            except np.linalg.LinAlgError:
                return 1e99
            ys = np.dot(self.N0, beta)
            chisq = np.sum(((self.y-ys)/self.err)**2)
            return chisq - target_chisq

        mincurvature = 1e-20
        pbest = sp.optimize.brentq(chisq_difference, mincurvature, 1, args=(chisq,))
        beta = best_params(pbest)
        self.coeff = self.basis.expand_coeff(beta)

        # Store the linear extrapolation outside the knotted region.
        val = self([self.x[0], self.x[-1]], 0)
        slope = self([self.x[0], self.x[-1]], 1)
        self.lowline = np.poly1d([slope[0], val[0]])
        self.highline = np.poly1d([slope[1], val[1]])

    def __call__(self, x, der=0):
        """Return the value of (the `der`th derivative of) the smoothing spline
        at data points `x`."""
        scalar = np.isscalar(x)
        x = np.asarray(x)
        splresult = splev(x, (self.basis.padknots, self.coeff, 3), der=der)
        low = x < self.x[0]
        high = x > self.x[-1]
        if np.any(low):
            splresult[low] = self.lowline(x[low]-self.x[0])
        if np.any(high):
            splresult[high] = self.highline(x[high]-self.x[-1])
        if scalar:
            splresult = splresult[()]
        return splresult


class SmoothingSplineFunction(SmoothingSpline):
    def __init__(self, x, y, dy, dx=None, maxchisq=None, der=0):
        super(SmoothingSplineFunction, self).__init__(x, y, dy, dx=dx, maxchisq=maxchisq)
        self.der = der

    def derivative(self, der=1):
        if self.der + der > 3:
            return ConstantFunction(0)

        return SmoothingSplineFunction(self.x, self.y, self.dy, self.dx, self.maxchisq, der=self.der + der)

    def __call__(self, x, der=0):
        if self.der + der > 3:
            return np.zeros_like(x)
        return super(SmoothingSplineFunction, self).__call__(x, der=self.der + der)


class SmoothingSplineLog(object):
    def __init__(self, x, y, dy, dx=None, maxchisq=None):
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("The x and y data must all be positive to use a SmoothingSplineLog")
        if dx is not None:
            dx /= x
        self.linear_model = SmoothingSpline(np.log(x), np.log(y), dy/y, dx, maxchisq=maxchisq)

    def __call__(self, x, der=0):
        return np.exp(self.linear_model(np.log(x), der=der))
