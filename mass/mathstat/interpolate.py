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

Joe Fowler, NIST
"""

__all__ = ['CubicSpline']

import numpy as np


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

        return result



class LinterpCubicSpline(CubicSpline):
    '''A CubicSpline object which is a linear combination of CubicSpline objects
    s1 and s2, effectively fraction*s1 + (1-fraction)*s2'''

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
        self.yprimeN  = wtsum(s1.yprimeN, s2.yprimeN, fraction)
        self.dx = wtsum(s1.dx, s2.dx, fraction)
        self.dy = wtsum(s1.dy, s2.dy, fraction)
