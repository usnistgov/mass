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
data=numpy.random.standard_normal(N)
spec = ps.PowerSpectrum(M, dt=1e-6)
window = ps.hann(2*M)
for i in range(3):
    spec.addDataSegment(data[i*M : (i+2)*M], window=window)
pylab.plot(spec.frequencies(), spec.spectrum())

Or you can use the convenience function that hides the class objects
from you and simply returns a (frequency,spectrum) pair of arrays:

N=1024
data=numpy.random.standard_normal(N)
pylab.clf()
for i in (2,4,8,1):
    f,s = ps.computeSpectrum(data, segfactor=i, dt=1e-6, window=numpy.hanning)
    pylab.plot(f, s)

Window choices are:
bartlett - Triangle shape
hamm     - Sine-squared
hanning  - 0.08 + 0.92*(sine-squared)
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

import numpy


class PowerSpectrum(object):
    """Object for accumulating power spectrum estimates from one
    or more segments of data.  If you want to use multiple overlapping
    segments, use class PowerSpectrumOvelap.  

    Based on Num Rec 3rd Ed section 13.4"""
    
    def __init__(self, m, dt=1.0):
        """Sets up to estimate PSD at m+1 frequencies (counting DC) given
        data segments of length 2m.  Optional dt is the time step Delta"""
        self.m = m
        self.m2 = 2*m
        self.nsegments = 0
        self.specsum = numpy.zeros(m+1, dtype=numpy.float)
        self.dt = dt
        if dt is None:
            self.dt = 1.0

    def addDataSegment(self, data, window=None):
        """Process a data segment of length 2m using the window function
        given.  window can be None (square window), a callable taking the
        length and returning a sequence, or a sequence."""
        if len(data) != self.m2:
            raise ValueError("wrong size data segment")
        if window is None:
            wksp = data
            sum_window = self.m2
        else:
            try:
                w = window(self.m2)
            except TypeError:
                w = numpy.array(window)
            wksp = w*data
            sum_window = (w**2).sum()

        scale_factor = 2./(sum_window*self.m2)
        if True: # we want real units
            scale_factor *=  self.dt*self.m2
        wksp = numpy.fft.rfft(wksp)
        
        # The first line adds 2x too much to the first/last bins.
        ps = numpy.abs(wksp)**2
#        ps[0] *= 0.5
#        ps[-1] *= 0.5
        self.specsum += scale_factor*ps
        self.nsegments += 1
        
    def spectrum(self):
        return self.specsum / self.nsegments

    def autocorrelation(self):
        "Return the autocorrelation (the DFT of this power spectrum)"
        

    def frequencies(self):
        return numpy.arange(self.m+1, dtype=numpy.float)/(2*self.dt*self.m)

class PowerSpectrumOverlap(PowerSpectrum):
    """Object for power spectral estimation using overlapping
    data segments.  User sends non-overlapping segments of length m,
    and they are processed in pairs of length 2m with overlap (except
    on the first and last segment)."""

    def __init__(self, m, dt=1.0):
        PowerSpectrum.__init__(self, m, dt=dt)
        self.first = True

    def addDataSegment(self, data, window=None):
        "Process a data segment of length m using window."
        if self.first:
            self.first = False
            self.fullseg = numpy.concatenate((
                numpy.zeros_like(data),
                numpy.array(data)))
        else:
            self.fullseg[0:self.m] = self.fullseg[self.m:]
            self.fullseg[self.m:] = data
            PowerSpectrum.addDataSegment(self, self.fullseg, window=window)

    def addLongData(self, data, window=None):
        """Process a long vector of data as overlapping segments of
        length 2m."""
        nt = len(data)
        nk = (nt-1)/self.m
        if nk>1:
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
    "A Bartlett window (triangle shape) of length n"
    return numpy.bartlett(n)

def welch(n):
    "A Welch window (parabolic) of length n"
    return 1-(2*numpy.arange(n, dtype=numpy.float)/(n-1.)-1)**2

def hann(n):
    "A Hann window (sine-squared) of length n"
    # twopi = numpy.pi*2
    # i = numpy.arange(n, dtype=numpy.float)
    # return  0.5*(1.0-numpy.cos(i*twopi/(n-1)))
    return numpy.hanning(n)

def hamming(n):
    "A Hamming window (0.08 + 0.92*sine-squared) of length n"
    return numpy.hamming(n)

# Convenience functions

def computeSpectrum(data, segfactor=1, dt=None, window=None):
    """Convenience function to compute the power spectrum of a single data array.
    
    <data>  Data for finding the spectrum
    <segfactor>   How many segments to break up the data into.  The spectrum
                  will be found on each consecutive pair of segments and
                  will be averaged over all pairs.
    <dt>      The sample spacing, in time.
    <window>  The window function to apply.  Should be a function that accepts
              a number of samples and returns an array of that length.  
              Possible values are bartlett, welch,
              hann, and hamming in this module, or use a function of your choosing.
              
    Return: either the PSD estimate as an array (non-negative frequencies only), 
    *OR* the tuple (frequencies, PSD).  The latter returns when <dt> is not None.
    """
    
    N = len(data)
    M = N/(2*segfactor)
    try:
        window = window(2*M) # precompute
    except TypeError:
        window = None

    if segfactor == 1:
        spec = PowerSpectrum(M, dt=dt)
        spec.addDataSegment(data, window=window)
    else:
        spec = PowerSpectrumOverlap(M, dt=dt)
        for i in range(2*segfactor-1):
            spec.addDataSegment( data[i*M : (i+1)*M], window=window)

    if dt is None:
        return spec.spectrum()
    else:
        return spec.frequencies(), spec.spectrum()


import matplotlib.pylab as pylab
def demo(N=1024, window=numpy.hanning):
    data=numpy.random.standard_normal(N)
    pylab.clf()
    for i in (2,4,8,1):
        f,s = computeSpectrum(data, segfactor=i, dt=1e0, 
                                 window=window)
        pylab.plot(f, s)
