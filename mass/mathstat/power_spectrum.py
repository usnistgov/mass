"""
A class and functions to compute a power spectrum using some of the
sophistications given in Numerical Recipes, including windowing and
overlapping data segments.

Use the class PowerSpectrum in the case that you are compute-limited
and PowerSpectrumOverlap in the case that you are data-limited.  The
latter uses k segments of data two segments at a time to make (k-1)
estimates and makes fuller use of all data (except in the first and
last segment).

Joe Fowler, NIST

October 13, 2010


Usage:

import power_spectrum as ps
import pylab
N=1024
M=N/4
data=np.random.standard_normal(N)
spec = ps.PowerSpectrum(M, dt=1e-6)
window = ps.hann(2*M)
for i in range(3):
    spec.addDataSegment(data[i*M : (i+2)*M], window=window)
pylab.plot(spec.frequencies(), spec.spectrum())

Or you can use the convenience function that hides the class objects
from you and simply returns a (frequency,spectrum) pair of arrays:

N=1024
data=np.random.standard_normal(N)
pylab.clf()
for i in (2,4,8,1):
    f,s = ps.computeSpectrum(data, segfactor=i, dt=1e-6, window=np.hanning)
    pylab.plot(f, s)

Window choices are:
bartlett - Triangle shape
hann     - Sine-squared
hamming  - 0.08 + 0.92*(sine-squared)
welch    - Parabolic
None     - Square (no windowing)
***      - Any other vector of length 2m OR any callable accepting
           2m as an argument and returning a sequence of that length.

Each window take an argument (n), the number of data points per
segment.  When using the PowerSpectrum or PowerSpectrumOverlap classes
or the convenience function computeSpectrum, you have a choice.  You
can call the window and pass in the resulting vector, or you can pass
in the callable function itself.  It is allowed to use different windows
on different data segments, though honestly that would be really weird.
"""

import numpy as np
import matplotlib.pylab as pylab

__all__ = ['PowerSpectrum', 'PowerSpectrumOverlap',
           'bartlett', 'welch', 'hann', 'hamming',
           'computeSpectrum']


class PowerSpectrum(object):
    """Object for accumulating power spectrum estimates from one or more data segments.

    If you want to use multiple overlapping segments, use class
    PowerSpectrumOvelap.

    Based on Num Rec 3rd Ed section 13.4"""

    def __init__(self, m, dt=1.0):
        """Sets up to estimate PSD at m+1 frequencies (counting DC) given
        data segments of length 2m.  Optional dt is the time step Delta"""
        self.m = m
        self.m2 = 2*m
        self.nsegments = 0
        self.specsum = np.zeros(m+1, dtype=float)
        self.dt = dt
        if dt is None:
            self.dt = 1.0

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to recompute everything, but
        you do want to update the method definitions."""
        c = PowerSpectrum(self.m, dt=self.dt)
        c.__dict__.update(self.__dict__)
        return c

    def addDataSegment(self, data, window=None):
        """Process a data segment of length 2m using the window function
        given.  window can be None (square window), a callable taking the
        length and returning a sequence, or a sequence."""
        if len(data) != self.m2:
            raise ValueError("wrong size data segment.  len(data)=%d but require %d" %
                             (len(data), self.m2))
        if np.isnan(data).any():
            raise ValueError("data contains NaN")
        if window is None:
            wksp = data
            sum_window = self.m2
        else:
            try:
                w = window(self.m2)
            except TypeError:
                w = np.array(window)
            wksp = w*data
            sum_window = (w**2).sum()

        scale_factor = 2./(sum_window*self.m2)
        if True:  # we want real units
            scale_factor *= self.dt*self.m2
        wksp = np.fft.rfft(wksp)

        # The first line adds 2x too much to the first/last bins.
        ps = np.abs(wksp)**2
        self.specsum += scale_factor*ps
        self.nsegments += 1

    def addLongData(self, data, window=None):
        """Process a long vector of data as non-overlapping segments of length 2m."""
        nt = len(data)
        nk = nt//self.m2
        for k in range(nk):
            noff = k*self.m2
            PowerSpectrum.addDataSegment(self,
                                         data[noff:noff+self.m2],
                                         window=window)

    def spectrum(self, nbins=None):
        """If <nbins> is given, the data are averaged into <nbins> bins."""
        if nbins is None:
            return self.specsum / self.nsegments
        if nbins > self.m:
            raise ValueError("Cannot rebin into more than m=%d bins" % self.m)

        newbin = np.asarray(0.5+np.arange(self.m+1, dtype=float)/(self.m+1)*nbins, dtype=int)
        result = np.zeros(nbins+1, dtype=float)
        for i in range(nbins+1):
            result[i] = self.specsum[newbin == i].mean()
        return result/self.nsegments

    def autocorrelation(self):
        """Return the autocorrelation (the DFT of this power spectrum)"""
        raise NotImplementedError("The autocorrelation method is not yet implemented.")

    def frequencies(self, nbins=None):
        """If <nbins> is given, the data are averaged into <nbins> bins."""
        if nbins is None:
            nbins = self.m
        if nbins > self.m:
            raise ValueError("Cannot rebin into more than m=%d bins" % self.m)
        return np.arange(nbins+1, dtype=float)/(2*self.dt*nbins)


class PowerSpectrumOverlap(PowerSpectrum):
    """Object for power spectral estimation using overlapping data segments.

    User sends non-overlapping segments of length m,
    and they are processed in pairs of length 2m with overlap (except
    on the first and last segment).
    """

    def __init__(self, m, dt=1.0):
        PowerSpectrum.__init__(self, m, dt=dt)
        self.first = True

    def addDataSegment(self, data, window=None):
        "Process a data segment of length m using window."
        if self.first:
            self.first = False
            self.fullseg = np.concatenate((
                np.zeros_like(data),
                np.array(data)))
        else:
            self.fullseg[0:self.m] = self.fullseg[self.m:]
            self.fullseg[self.m:] = data
            PowerSpectrum.addDataSegment(self, self.fullseg, window=window)

    def addLongData(self, data, window=None):
        """Process a long vector of data as overlapping segments of
        length 2m."""
        nt = len(data)
        nk = (nt-1)//self.m
        if nk > 1:
            delta_el = (nt-self.m2)/(nk-1.0)
        else:
            delta_el = 0.0
        for k in range(nk):
            noff = int(k*delta_el+0.5)
            PowerSpectrum.addDataSegment(self,
                                         data[noff:noff+self.m2],
                                         window=window)

# Commonly used window functions


def bartlett(n):
    """A Bartlett window (triangle shape) of length n"""
    return np.bartlett(n)


def welch(n):
    """A Welch window (parabolic) of length n"""
    return 1 - (2*np.arange(n, dtype=float)/(n - 1.) - 1)**2


def hann(n):
    """A Hann window (sine-squared) of length n"""
    # twopi = np.pi*2
    # i = np.arange(n, dtype=float)
    # return  0.5*(1.0-np.cos(i*twopi/(n-1)))
    return np.hanning(n)


def hamming(n):
    """A Hamming window (0.08 + 0.92*sine-squared) of length n"""
    return np.hamming(n)

# Convenience functions


def computeSpectrum(data, segfactor=1, dt=None, window=None):
    """Convenience function to compute the power spectrum of a single data array.

    Args:
        <data>  Data for finding the spectrum
        <segfactor>   How many segments to break up the data into.  The spectrum
            will be found on each consecutive pair of segments and
            will be averaged over all pairs.
        <dt>      The sample spacing, in time.
        <window>  The window function to apply.  Should be a function that accepts
            a number of samples and returns an array of that length. Possible
            values are bartlett, welch, hann, and hamming in this module, or use
            a function of your choosing.

    Returns:
        Either the PSD estimate as an array (non-negative frequencies only),
        *OR* the tuple (frequencies, PSD).  The latter returns when <dt> is not None.
    """

    N = len(data)
    M = N//(2*segfactor)
    try:
        window = window(2*M)  # precompute
    except TypeError:
        window = None

    if segfactor == 1:
        spec = PowerSpectrum(M, dt=dt)
        # Ensure that the datasegment has even length
        spec.addDataSegment(data[:2*(len(data)//2)], window=window)
    else:
        spec = PowerSpectrumOverlap(M, dt=dt)
        for i in range(2*segfactor-1):
            spec.addDataSegment(data[i*M:(i+1)*M], window=window)

    if dt is None:
        return spec.spectrum()
    else:
        return spec.frequencies(), spec.spectrum()


def demo(N=1024, window=np.hanning):
    data = np.random.standard_normal(N)
    pylab.clf()
    for i in (2, 4, 8, 1):
        f, s = computeSpectrum(data, segfactor=i, dt=1e0, window=window)
        pylab.plot(f, s)
