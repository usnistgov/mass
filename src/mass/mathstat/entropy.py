"""
entropy.py

Estimates of the distribution entropy computed using kernel-density estimates
of the distribution.

* `laplace_entropy(x, w=1.0)` - Compute the entropy of data set `x` where the
kernel is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w).
* `laplace_KL_divergence(x, y, w=1.0)` - Compute the Kullback-Leibler Divergence
of data set `y` from `x`, where the
kernel is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w).

The K-L divergence of Q(x) from P(x) is defined as the integral over the full
x domain of P(x) log[P(x)/Q(x)].

This equals the cross-entropy H(P,Q) - H(P). Note that K-L divergence is not
symmetric with respect to reversal of `x` and `y`.

Eventually, plan to translate this to Cython.
"""

import numpy as np

def laplace_entropy(x, w=1.0):
    """`laplace_entropy(x, w=1.0)`

    Compute the entropy of data set `x` where the
    kernel is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w)."""

    N = len(x)
    if N == 0:
        raise ValueError("laplace_entropy(x) needs at least 1 element in `x`.")
    if w <= 0.0:
        raise ValueError("laplace_entropy(x, w) needs `w>0`.")
    c = np.zeros(N, dtype=float)
    d = np.zeros(N, dtype=float)
    y = np.sort(x)/w

    e = np.exp(y[:-1]-y[1:])
    c[0] = 1.0
    for i in range(1,N):
        c[i] = e[i-1]*c[i-1] + 1.0
    d[N-1] = 1.0
    for i in range(N-2, -1, -1):
        d[i] = e[i]*d[i+1] + 1.0
    c /= 2*w*N
    d /= 2*w*N
    H = w*d[0]*(1-np.log(d[0])) + w*c[N-1]*(1-np.log(c[N-1]))
    for i in range(N-1):
        dp = d[i+1]*e[i]
        r = (c[i]/dp)**0.5
        r1 = c[i]/d[i+1]
        e2 = e[i]**0.5
        H += 4*w*(c[i]*dp)**0.5*np.arctan((e2-1.0/e2)*r1**0.5/(r1+1.0))
        H += w*(dp-c[i])*(np.log(c[i]+dp)-1)
        A,B = d[i+1], c[i]*e[i]
        H -= w*(A-B)*(np.log(A+B)-1)
    return H



def _merge_orderedlists(x1, x2):
    """Given two lists that are assumed to be sorted (in ascending order),
    return `(x, wasfirst)` where `x` is an array that contains all the values
    from `x1` and `x2` in sorted order, and where `wasfirst` is a boolean array
    whose values are True if and only if the corresponding value of `x` was
    found in the `x1` input.

    Behavior is undefined if either `x1` or `x2` is not sorted."""
    N1 = len(x1)
    N2 = len(x2)
    if N2 == 0:
        if N1 == 0:
            raise ValueError("_merge_orderedlists(x,y) requires at least one list of positive length.")
        return x1, np.ones(N1, dtype=np.bool)
    elif N1 == 0:
        return x2, np.ones(N2, dtype=np.bool)

    out = np.zeros(N1+N2, dtype=float)
    wasfirst = np.zeros(N1+N2, dtype=np.bool)
    i = j = k = 0
    while True:
        if x1[i] < x2[j]:
            out[k] = x1[i]
            wasfirst[k] = True
            k += 1
            i += 1
            if i >= N1:
                out[k:] = x2[j:]
                wasfirst[k:] = False
                return out, wasfirst
        else:
            out[k] = x2[j]
            wasfirst[k] = False
            k += 1
            j += 1
            if j >= N2:
                out[k:] = x1[i:]
                wasfirst[k:] = True
                return out, wasfirst


def laplace_KL_divergence(x, y, w=1.0):
    if w <= 0.0:
        raise ValueError("laplace_KL_divergence(x, y, w) needs `w>0`.")
    Nx = len(x)
    Ny = len(y)
    N = Nx+Ny
    if Nx == 0 or Ny == 0:
        raise ValueError("laplace_KL_divergence(x, y) needs at least 1 element apiece in `x` and `y`.")

    x = np.sort(x)/w
    y = np.sort(y)/w
    nodes, isx = _merge_orderedlists(x, y)

    a = np.zeros(N, dtype=float)
    b = np.zeros(N, dtype=float)
    c = np.zeros(N, dtype=float)
    d = np.zeros(N, dtype=float)
    decayfactor = np.zeros(N, dtype=float)

    if isx[0]:
        c[0] = 1.0
    else:
        a[0] = 1.0
    for i in range(1,N):
        decayfactor[i] = np.exp(nodes[i-1]-nodes[i])
        factor = decayfactor[i]
        a[i] = factor*a[i-1]
        c[i] = factor*c[i-1]
        if isx[i]:
            c[i] += 1.0
        else:
            a[i] += 1.0
    if isx[N-1]:
        d[N-1] = 1.0
    else:
        b[N-1] = 1.0
    for i in range(N-2,-1,-1):
        factor = decayfactor[i+1]
        b[i] = factor*b[i+1]
        d[i] = factor*d[i+1]
        if isx[i]:
            d[i] += 1.0
        else:
            b[i] += 1.0

    c /= 2*Nx
    d /= 2*Nx
    a /= 2*Ny
    b /= 2*Ny

    H = 0.0
    H -= _antideriv_F(0.0, b[0], 0.0, d[0])
    H += _antideriv_F(a[-1], 0.0, c[-1], 0.0)
    for i in range(1,N):
        factor = decayfactor[i]
        H += _antideriv_F(a[i-1], b[i]*factor, c[i-1], d[i]*factor)
        H -= _antideriv_F(a[i-1]*factor, b[i], c[i-1]*factor, d[i])
    return H


def _antideriv_F(A, B, C, D):
    if A<0 or B<0 or C<0 or D<0:
        raise ValueError
    r = (D-C)*(np.log(A+B)-1.0)
    if A==0:
        return r-2*C
    elif B==0:
        return r+2*D
    return r - 2*(A*D+B*C)/(A*B)**0.5 * np.arctan((A/B)**0.5)
