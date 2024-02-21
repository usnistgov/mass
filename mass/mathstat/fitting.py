"""
mass.mathstat.fitting

Model-fitting utilities.

Joe Fowler, NIST
"""

import numpy as np
import scipy as sp
import logging

LOG = logging.getLogger("mass")

__all__ = ['kink_model', 'fit_kink_model']


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

    dxi = xi - k
    dxj = xj - k
    si = dxi.sum()
    sj = dxj.sum()
    si2 = (dxi**2).sum()
    sj2 = (dxj**2).sum()
    A = np.array([[N, si, sj],
                  [si, si2, 0],
                  [sj, 0, sj2]])
    v = np.array([y.sum(), (yi * dxi).sum(), (yj * dxj).sum()])
    a, b, c = abc = np.linalg.solve(A, v)
    model = np.hstack([a + b * dxi, a + c * dxj])
    X2 = ((model - y)**2).sum()
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
