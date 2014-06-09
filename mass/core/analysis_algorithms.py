'''
mass.core.analysis_algorithms - main algorithms used in data analysis

Designed to ... ?

Created on Jun 9, 2014

@author: fowlerj
'''

__all__ = ['make_smooth_histogram', 'HistogramSmoother']

import numpy as np
import scipy as sp
# import pylab as plt


class HistogramSmoother(object):
    """Object that can repeatedly smooth histograms with the same bin
    count and width to the same Gaussian width.  By pre-computing the
    smoothing kernel for that histogram, we can smooth multiple histograms
    with the same geometry."""
    
    def __init__(self, smooth_sigma, limits):
        """Give the smoothing Gaussian's width as <smooth_sigma> and the
        [lower,upper] histogram limits as <limits>."""
        
        self.limits = np.asarray(limits, dtype=float)
        self.smooth_sigma = smooth_sigma

        # Choose a reasonable # of bins, at least 1024 and a power of 2
        stepsize = 0.4*smooth_sigma
        dlimits = self.limits[1] - self.limits[0]
        nbins = int(dlimits/stepsize+0.5)
        pow2 = 1024
        while pow2<nbins:
            pow2 *= 2
        self.nbins = pow2
        self.stepsize = dlimits / self.nbins
        
        # Compute the Fourier-space smoothing kernel
        kernel = np.exp(-0.5*(np.arange(self.nbins)*self.stepsize/self.smooth_sigma)**2)
        kernel[1:] += kernel[-1:0:-1] # Handle the negative frequencies 
        kernel /= kernel.sum()
        self.kernel_ft = np.fft.rfft(kernel)

    
    def __call__(self, values):
        """Return a smoothed histogram of the data vector <values>"""
        contents, _ = np.histogram(values, self.nbins, self.limits)
        ftc = np.fft.rfft(contents) 
        csmooth = np.fft.irfft(self.kernel_ft * ftc)
        csmooth[csmooth<0] = 0
        return csmooth


def make_smooth_histogram(values, smooth_sigma, limit, upper_limit=None):
    """
    Convert a vector of arbitrary <values> info a smoothed histogram by 
    histogramming it and smoothing. The smoothing Gaussian's width is 
    <smooth_sigma> and the histogram limits are [limit,upper_limit] or
    [0,limit] if upper_limit is None.

    This is a convenience function using the HistogramSmoother class. 
    """
    if upper_limit is None:
        limit, upper_limit = 0, limit
    return HistogramSmoother(smooth_sigma, [limit,upper_limit])(values)



def drift_correct(dataset):
    """Apply a drift correction to the data in <dataset>, which should be
    a mass.MicrocalDataset object.
    
    The model is that the filtered pulse height PH should be scaled by
    (1 + a*PTM) where a is an arbitrary parameter computed here, and 
    PTM is the difference between each record's pretrigger mean and the
    median value of all pretrigger means.
    """
    g = dataset.cuts.good()
    vec = dataset.p_filt_value[g]
    ptm = dataset.p_pretrig_mean[g]
    ptm_offset = np.median(ptm)
    ptm -= ptm_offset
    limit = 7000
    
    smoother = HistogramSmoother(0.5, [0,limit])
    def entropy(param, ptm, vec, smoother):
        vec_dc = vec * (1+ptm*param)
        hsmooth = smoother(vec_dc)
        w = hsmooth>0
        return -(np.log(hsmooth[w])*hsmooth[w]).sum()
    
    param = sp.optimize.brent(entropy, (ptm, vec, smoother), brack=[0, .001])
    print 'Best drift correction parameter: %.6f'%param
    
    # Apply correction and remember what it was
    dataset.p_filt_value_dc = dataset.p_filt_value* \
        (1+(dataset.p_pretrig_mean-ptm_offset)*param)
    
    dataset.drift_correct_info = {'type':'ptmean_gain',
                                  'slope': param,
                                  'median_pretrig_mean': ptm_offset}
