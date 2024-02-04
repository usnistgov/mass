r"""
entropy.py

Estimates of the distribution entropy computed using kernel-density estimates
of the distribution.

* `laplace_entropy(x, w=1.0)` - Compute the entropy H(p) of data set `x` where the
kernel used to estimate p from x is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w).
* `laplace_cross_entropy(x, y, w=1.0)` - Compute the cross entropy of q from p,
where q and p are the kernel-density estimates taken from data set
`y` and data set `x`, and where the kernel is the Laplace kernel.
* `KL_divergence(x, y, w=1.0)` - Compute the Kullback-Leibler Divergence
of data set `y` from `x`, where the kernel is the Laplace kernel.

The K-L divergence of Q(x) from P(x) is defined as the integral over the full
x domain of P(x) log[P(x)/Q(x)].

This equals the cross-entropy H(P,Q) - H(P). Note that cross-entropy and K-L divergence
are not symmetric with respect to reversal of `x` and `y`.
"""

import numpy as np
import scipy as sp

from libc.math cimport exp, atan, log, sqrt, abs
cimport numpy as np
cimport cython

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64
BYTPE = bool
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t


@cython.embedsignature(True)
cpdef double laplace_entropy(x_in, double w=1.0, approx_mode="size") except? -9999:
    r"""Compute the entropy of data set `x` where the
    kernel is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w).

    Args:
        x_in (array): The vector of data of which we want the entropy.
        w (double): The width (exponential scale length) of the Laplace distribution
            to be used in kernel-density estimation.
        approx_mode (string): How to balance execution speed and accuracy
            (default "size").

    The `approx_mode` can be one of:
    ``exact``  The exact integral is computed (can take ~0.25 sec per 10^6 values).
    ``approx`` The integral is approximated by histogramming the data, smoothing
               that, and using Simpson's rule on the PDF samples that result.
    ``size``   Uses "approx" if len(x)>200000, or "exact" otherwise.
    """
    cdef int N = len(x_in)
    if N == 0:
        raise ValueError("laplace_entropy(x) needs at least 1 element in `x`.")
    if w <= 0.0:
        raise ValueError("laplace_entropy(x, w) needs `w>0`.")
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.asarray(x_in, dtype=DTYPE)

    if approx_mode == "size":
        if N <= 200000:
            approx_mode = "exact"
        else:
            approx_mode = "approx"
    if approx_mode.startswith("exact"):
        return laplace_entropy_array(x, w)
    else:
        return laplace_entropy_approx(x, w)


cdef double laplace_entropy_array(np.ndarray[DTYPE_t, ndim=1] x, double w=1.0):
    cdef int i
    cdef int N = len(x)
    cdef np.ndarray[DTYPE_t, ndim=1] c = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.sort(x)/w

    cdef np.ndarray[DTYPE_t, ndim=1] e = np.zeros(N, dtype=DTYPE)
    cdef double stepsize = 1.0/(2*w*N)
    c[0] = stepsize
    for i in range(1, N):
        e[i-1] = exp(y[i-1]-y[i])
        c[i] = e[i-1]*c[i-1] + stepsize
    d[N-1] = stepsize
    for i in range(N-2, -1, -1):
        d[i] = e[i]*d[i+1] + stepsize

    cdef double H, dp, r, r1, e2, A, B
    H = w*d[0]*(1-log(d[0])) + w*c[N-1]*(1-log(c[N-1]))
    for i in range(N-1):
        dp = d[i+1]*e[i]
        r = sqrt(c[i]/dp)
        r1 = c[i]/d[i+1]
        e2 = sqrt(e[i])
        H += 4*w*sqrt(c[i]*dp)*atan((e2-1.0/e2)*r1**0.5/(r1+1.0))
        H += w*(dp-c[i])*(log(c[i]+dp)-1)
        A, B = d[i+1], c[i]*e[i]
        H -= w*(A-B)*(log(A+B)-1)
    return H


cdef laplace_entropy_approx(x, w=1.0):
    """Approximate the entropy with a binned histogram and the Laplace-distribution
    kernel-density estimator of the probability distribtion."""
    cdef double EXTEND_DATA = 5*w
    cdef double BINS_PER_W = 20
    cdef double KERNEL_WIDTH_IN_WS = 15.0

    cdef double xmin = x.min()-EXTEND_DATA
    cdef double xmax = x.max()+EXTEND_DATA
    cdef int nbins = int(0.5+(xmax-xmin)*BINS_PER_W/w)
    c, b = np.histogram(x, nbins, [xmin, xmax])
    cdef double db = b[1]-b[0]
    cdef int nx = int(0.5+KERNEL_WIDTH_IN_WS*w/db)

    cdef np.ndarray[DTYPE_t, ndim=1] kernel = np.zeros(2*nx+1)
    cdef double kx
    for i in range(2*nx+1):
        kx = (i-nx)*db
        kernel[i] = exp(-abs(kx/w))

    # kde = unnormalized kernel-density estimator.
    kde = sp.signal.fftconvolve(c, kernel, mode="full")[nx:-nx]
    kde[kde < kernel.min()] = kernel.min()

    # p = normalized probability distribution.
    cdef double norm = 1.0/sp.integrate.simpson(kde, dx=db, even="first")
    p = kde*norm
    return -sp.integrate.simpson(p*np.log(p), dx=db, even="first")


@cython.embedsignature(True)
cpdef _merge_orderedlists(x1_in, x2_in):
    """Given two lists that are assumed to be sorted (in ascending order),
    return `(x, wasfirst)` where `x` is an array that contains all the values
    from `x1` and `x2` in sorted order, and where `wasfirst` is a boolean array
    whose values are True if and only if the corresponding value of `x` was
    found in the `x1` input.

    Behavior is undefined if either `x1` or `x2` is not sorted.
    """

    x1 = np.asarray(x1_in, dtype=DTYPE)
    x2 = np.asarray(x2_in, dtype=DTYPE)
    cdef int N1 = len(x1)
    cdef int N2 = len(x2)
    if N2 == 0:
        if N1 == 0:
            raise ValueError("_merge_orderedlists(x,y) requires at least one list of positive length.")
        return x1, np.ones(N1, dtype=bool)
    elif N1 == 0:
        return x2, np.ones(N2, dtype=bool)

    out = np.zeros(N1+N2, dtype=DTYPE)
    wasfirst = np.zeros(N1+N2, dtype=bool)
    _merge_orderedlists_arrays(out, wasfirst, x1, x2)
    return out, wasfirst


@cython.embedsignature(True)
cdef _merge_orderedlists_arrays(np.ndarray[DTYPE_t, ndim=1] out,
                                np.ndarray wasfirst,
                                np.ndarray[DTYPE_t, ndim=1] x1,
                                np.ndarray[DTYPE_t, ndim=1] x2):
    cdef int i, j, k
    cdef int N1 = len(x1)
    cdef int N2 = len(x2)
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
                return
        else:
            out[k] = x2[j]
            wasfirst[k] = False
            k += 1
            j += 1
            if j >= N2:
                out[k:] = x1[i:]
                wasfirst[k:] = True
                return


@cython.embedsignature(True)
cpdef double laplace_KL_divergence(x, y, double w=1.0, approx_mode="size") except? -9999:
    r"""Compute the Kullback-Leibler divergence of data set `y` from data set `x`.

    Use kernel-density estimation, where the kernel is the Laplace kernel
    k(x) \propto exp(-abs(x-x0)/w).

    The `approx_mode` can be one of:
    ``exact``  The exact integral is computed (can take ~0.25 sec per 10^6 values).
    ``approx`` The integral is approximated by histogramming the data, smoothing
               that, and using Simpson's rule on the PDF samples that result.
    ``size``   Uses "approx" if len(x)+len(y)>200000, or "exact" otherwise.
    """
    return laplace_cross_entropy(x, y, w, approx_mode=approx_mode) - \
        laplace_entropy(x, w, approx_mode=approx_mode)


@cython.embedsignature(True)
cpdef double laplace_cross_entropy(x, y, double w=1.0, approx_mode="size") except? -9999:
    r"""`laplace_cross_entropy(x, y, w=1.0, approx_mode="size")`

    Compute the cross-entropy of data set `x` from data set `y`, where the
    kernel for x is the Laplace kernel k(x) \propto exp(-abs(x-x0)/w).

    The kernel for the y data is the piecewise-constant (top-hat) kernel. We choose this
    because a Laplace kernel for y led to possible divergences when the y-distribtion q
    is exceedingly small, but the x-distribution p nevertheless is non-zero because of a
    random x-value lying far from any random y-values. The constant kernel is given a
    non-zero floor value, so that q is never so small as to make any x-value impossible.

    Args:
        x (array): One vector of data.
        y (array): The other vector of data.
        w (double): The width (exponential scale length) of the Laplace distribution
            to be used in kernel-density estimation.
        approx_mode (string): How to balance execution speed and accuracy
            (default "size").

    The `approx_mode` can be one of:
    ``exact``  The exact integral is computed (can take ~0.25 sec per 10^6 values).
    ``approx`` The integral is approximated by histogramming the data, smoothing
               that, and using Simpson's rule on the PDF samples that result.
    ``size``   Uses "approx" if len(x)+len(y)>200000, or "exact" otherwise.
    """
    if w <= 0.0:
        raise ValueError("laplace_cross_entropy(x, y, w) needs `w>0`.")
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    if Nx == 0 or Ny == 0:
        raise ValueError("laplace_cross_entropy(x, y) needs at least 1 element apiece in `x` and `y`.")

    if approx_mode == "size":
        if Nx+Ny <= 200000:
            approx_mode = "exact"
        else:
            approx_mode = "approx"
    if approx_mode.startswith("exact"):
        xsorted = np.asarray(np.sort(x)/w, dtype=DTYPE)
        ysorted = np.asarray(np.sort(y)/w, dtype=DTYPE)
        return laplace_cross_entropy_arrays(xsorted, ysorted)+log(w)
    else:
        return laplace_cross_entropy_approx(np.asarray(x, dtype=DTYPE),
                                            np.asarray(y, dtype=DTYPE), w)


cdef double laplace_cross_entropy_arrays(np.ndarray[DTYPE_t, ndim=1] x,
                                         np.ndarray[DTYPE_t, ndim=1] y):

    # List of all places where q(u) increases or decreases because of a y-point.
    cdef double Qstepwidth = 2*sqrt(6)
    ynodes, qstep_is_up = _merge_orderedlists(y-0.5*Qstepwidth, y+0.5*Qstepwidth)

    # List of all places where p(u) or q(u) changes because of an x- a y-point.
    nodes, isx = _merge_orderedlists(x, ynodes)

    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int N = Nx+Ny*2
    cdef int i
    cdef double factor

    # Pretend q(u) is never lower than this value, and spread this probability across
    # the range 10 less than the lowest to 10 more than the highest node.
    cdef double Qmin_sum = 1.0/sqrt(Ny+3)
    cdef double Qmin = Qmin_sum / (nodes[-1]+10 - (nodes[0]-10))
    cdef double Qstep = (1.0-Qmin_sum) / (Ny * Qstepwidth)

    # Initialize the vectors decayfactor, c, and d.
    cdef np.ndarray[DTYPE_t, ndim=1] decayfactor = np.zeros(N, dtype=DTYPE)
    for i in range(1, N):
        decayfactor[i] = exp(nodes[i-1]-nodes[i])

    # c requires a left-right pass over all nodes.
    cdef np.ndarray[DTYPE_t, ndim=1] c = np.zeros(N, dtype=DTYPE)
    cdef double stepX = 1.0/(2*Nx)
    cdef int j = 0
    if isx[0]:
        c[0] = stepX
    else:
        j = 1
    for i in range(1, N):
        factor = decayfactor[i]
        c[i] = factor*c[i-1]
        if isx[i]:
            c[i] += stepX

    # d requires a right-left pass over all nodes.
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.zeros(N, dtype=DTYPE)
    if isx[N-1]:
        d[N-1] = stepX
    for i in range(N-2, -1, -1):
        factor = decayfactor[i+1]
        d[i] = factor*d[i+1]
        if isx[i]:
            d[i] += stepX

    # Now a left-right pass over all nodes to compute the H integral.
    cdef int net_up_qsteps = 0
    if not isx[0]:
        net_up_qsteps = 1

    cdef double H = -d[0] * log(Qmin)  # H due to the open first interval [-inf, nodes[0]]
    cdef double q
    for i in range(1, N):
        factor = decayfactor[i]
        q = Qmin + Qstep*net_up_qsteps
        H -= (c[i-1]+d[i])*(1-factor)*log(q)

        if not isx[i]:
            if qstep_is_up[j]:
                net_up_qsteps += 1
            else:
                net_up_qsteps -= 1
            j += 1
    H -= c[-1] * log(Qmin)  # H due to the open last interval [nodes[-1], +inf]
    return H


cdef laplace_cross_entropy_approx(np.ndarray[DTYPE_t, ndim=1] x,
                                  np.ndarray[DTYPE_t, ndim=1] y, double w=1.0):
    """Approximate the cross-entropy with a binned histogram and the
    Laplace-distribution kernel-density estimator of the probability distribtion.
    """
    cdef double EXTEND_DATA = 5*w
    cdef double BINS_PER_W = 20
    cdef double KERNEL_WIDTH_IN_WS = 15.0

    cdef double xmin = min(x.min(), y.min())-EXTEND_DATA
    cdef double xmax = max(x.max(), y.max())+EXTEND_DATA
    cdef int nbins = int(0.5+(xmax-xmin)*BINS_PER_W/w)
    cx, b = np.histogram(x, nbins, [xmin, xmax])
    cy, b = np.histogram(y, nbins, [xmin, xmax])
    cdef double db = b[1]-b[0]
    cdef int nx = int(0.5+KERNEL_WIDTH_IN_WS*w/db)

    cdef np.ndarray[DTYPE_t, ndim=1] kernel = np.zeros(2*nx+1)
    cdef double kx
    for i in range(2*nx+1):
        kx = (i-nx)*db
        kernel[i] = exp(-abs(kx/w))

    # kde = unnormalized kernel-density estimator.
    kde = sp.signal.fftconvolve(cx, kernel, mode="full")[nx:-nx]
    kde[kde < kernel.min()] = kernel.min()

    # p = normalized probability distribution.
    cdef double norm = 1.0/sp.integrate.simpson(kde, dx=db, even="first")
    p = kde*norm

    kde = sp.signal.fftconvolve(cy, kernel, mode="full")[nx:-nx]
    kde[kde < kernel.min()] = kernel.min()
    norm = 1.0/sp.integrate.simpson(kde, dx=db, even="first")
    q = kde*norm
    return -sp.integrate.simpson(p*np.log(q), dx=db, even="first")
