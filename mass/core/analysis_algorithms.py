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


########################################################################################
# Pulse summary quantities

def estimateRiseTime(pulse_data, timebase, nPretrig):
    """Compute the rise time of timeseries <pulse_data>, where the time steps are <timebase>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.
    
    If <nPretrig> >= 4, then the samples pulse_data[:nPretrig] are averaged to estimate
    the baseline.  Otherwise, the minimum of pulse_data is assumed to be the baseline.
    
    Specifically, take the first and last of the rising points in the range of 
    10% to 90% of the peak value, interpolate a line between the two, and use its
    slope to find the time to rise from 0 to the peak.
    
    See also mass.nonstandard.deprecated.fitExponentialRiseTime for a more traditional
    (and computationally expensive) definition.
    """
    MINTHRESH, MAXTHRESH = 0.1, 0.9
    
    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    nd = len(pulse_data.shape)
    if nd>2 or nd<1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if nd==1:
        pulse_data.shape (1, pulse_data.shape[0])

    # The following requires a lot of numpy foo to read. Sorry!
    if nPretrig >= 4:
        baseline_value = pulse_data[:, 0:nPretrig].mean(axis=1)
    else:
        baseline_value = pulse_data.min(axis=1)
        nPretrig = 0
    value_at_peak = pulse_data.max(axis=1) - baseline_value
    idx_last_pk = pulse_data.argmax(axis=1).max()
    
    npulses = pulse_data.shape[0]
    try:
        rising_data = ((pulse_data[:,nPretrig:idx_last_pk+1].T - baseline_value) / value_at_peak).T
        # Find the last and first indices at which the data are in (0.1, 0.9] times the
        # peak value. Then make sure last is at least 1 past first.
        last_idx = (rising_data>MAXTHRESH).argmax(axis=1)-1
        first_idx = (rising_data>MINTHRESH).argmax(axis=1)
        last_idx[last_idx<first_idx] = first_idx[last_idx<first_idx]+1
        
        pulsenum = np.arange(npulses)
        y_diff = np.asarray(rising_data[pulsenum,last_idx]-rising_data[pulsenum,first_idx],
                            dtype=float)
        y_diff[y_diff<timebase] = timebase
        time_diff = timebase*(last_idx-first_idx)
        rise_time = time_diff / y_diff
        rise_time[y_diff <= 0] = -9.9e-6
        return rise_time
    
    except ValueError:
        return -9.9e-6+np.zeros(npulses, dtype=float)



########################################################################################
# Drift correction

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



def drift_correct(indicator, uncorrected, limit=None):
    """Compute a drift correction that minimizes the spectral entropy of
    <uncorrected> (a filtered pulse height vector) after correction.
    The <indicator> vector should be of equal length and should have some 
    linear correlation with the pulse gain. Generally it will be the
    pretrigger mean of the pulses, but you can experiment with other choices.
    
    The entropy will be computed on corrected values only in the range
    [0, <limit>], so <limit> should be set to a characteristic large
    value of <uncorrected>. If <limit> is None (the default), then
    in will be compute as 25% larger than the 99%ile point of
    <uncorrected>
    
    The model is that the filtered pulse height PH should be scaled by
    (1 + a*PTM) where a is an arbitrary parameter computed here, and 
    PTM is the difference between each record's pretrigger mean and the
    median value of all pretrigger means. (Or replace "pretrigger mean"
    with whatever quantity you passed in as <indicator>.)
    """
    ptm_offset = np.median(indicator)
    indicator = np.array(indicator) - ptm_offset
    
    if limit is None:
        pct99 = sp.stats.scoreatpercentile(uncorrected, 99)
        limit = 1.25 * pct99
    
    smoother = HistogramSmoother(0.5, [0,limit])
    def entropy(param, indicator, uncorrected, smoother):
        corrected = uncorrected * (1+indicator*param)
        hsmooth = smoother(corrected)
        w = hsmooth>0
        return -(np.log(hsmooth[w])*hsmooth[w]).sum()
    
    drift_corr_param = sp.optimize.brent(entropy, (indicator, uncorrected, smoother), brack=[0, .001])
    
    drift_correct_info = {'type':'ptmean_gain',
                                  'slope': drift_corr_param,
                                  'median_pretrig_mean': ptm_offset}
    return drift_corr_param, drift_correct_info
