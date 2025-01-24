"""
mass.core.analysis_algorithms - main algorithms used in data analysis

Designed to abstract certain key algorithms out of the class `MicrocalDataSet`
and be able to run them fast.

Created on Jun 9, 2014

@author: fowlerj
"""
cimport cython
cimport numpy as np

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import sklearn.cluster
import mass.mathstat

from libc.math cimport sqrt
from mass.mathstat.entropy import laplace_entropy
from mass.core.utilities import show_progress


########################################################################################
# Pulse summary quantities

@cython.embedsignature(True)
def estimateRiseTime(pulse_data, timebase, nPretrig):
    """Computes the rise time of timeseries <pulse_data>, where the time steps are <timebase>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.

    If nPretrig >= 4, then the samples pulse_data[:nPretrig] are averaged to estimate
    the baseline.  Otherwise, the minimum of pulse_data is assumed to be the baseline.

    Specifically, take the first and last of the rising points in the range of
    10% to 90% of the peak value, interpolate a line between the two, and use its
    slope to find the time to rise from 0 to the peak.

    See also mass.nonstandard.deprecated.fitExponentialRiseTime for a more traditional
    (and computationally expensive) definition.

    Args:
        pulse_data: An np.ndarray of dimension 1 (a single pulse record) or 2 (an
            array with each row being a pulse record).
        timebase: The sampling time.
        nPretrig: The number of samples that are recorded before the trigger.

    Returns:
        An ndarray of dimension 1, giving the rise times.
    """
    MINTHRESH, MAXTHRESH = 0.1, 0.9

    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim > 2 or ndim < 1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim == 1:
        pulse_data.shape = (1, pulse_data.shape[0])

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
        rising_data = ((pulse_data[:, nPretrig:idx_last_pk+1] - baseline_value[:, np.newaxis]) /
                       value_at_peak[:, np.newaxis])
        # Find the last and first indices at which the data are in (0.1, 0.9] times the
        # peak value. Then make sure last is at least 1 past first.
        last_idx = (rising_data > MAXTHRESH).argmax(axis=1)-1
        first_idx = (rising_data > MINTHRESH).argmax(axis=1)
        last_idx[last_idx < first_idx] = first_idx[last_idx < first_idx]+1
        last_idx[last_idx == rising_data.shape[1]] = rising_data.shape[1]-1

        pulsenum = np.arange(npulses)
        y_diff = np.asarray(rising_data[pulsenum, last_idx]-rising_data[pulsenum, first_idx],
                            dtype=float)
        y_diff[y_diff < timebase] = timebase
        time_diff = timebase*(last_idx-first_idx)
        rise_time = time_diff / y_diff
        rise_time[y_diff <= 0] = -9.9e-6
        return rise_time

    except ValueError:
        return -9.9e-6+np.zeros(npulses, dtype=float)


@cython.embedsignature(True)
def python_compute_max_deriv(pulse_data, ignore_leading, spike_reject=True, kernel=None):
    """Equivalent to compute_max_deriv(...)"""
    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim > 2 or ndim < 1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim == 1:
        pulse_data.shape = (1, pulse_data.shape[0])
    pulse_data = np.array(pulse_data[:, ignore_leading:], dtype=float)
    NPulse, NSamp = pulse_data.shape

    # The default filter:
    filter_coef = np.array([+.2, +.1, 0, -.1, -.2])
    if kernel == "SG":
        # This filter is the Savitzky-Golay filter of n_L=1, n_R=3 and M=3, to use the
        # language of Numerical Recipes 3rd edition.  It amounts to least-squares fitting
        # of an M=3rd order polynomial to the five points [-1,+3] and
        # finding the slope of the polynomial at 0.
        # Note that we reverse the order of coefficients because convolution will re-reverse
        filter_coef = np.array([-0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905])[::-1]

    elif kernel is not None:
        filter_coef = np.array(kernel).ravel()

    # Use np.convolve, not scipy.signal.fftconvolve for small kernels.  A test showed that
    # np.convolve was 10x faster in typical use.
    max_deriv = np.zeros(NPulse, dtype=float)
    for i, data in enumerate(pulse_data):
        conv = np.convolve(data, filter_coef, mode='valid')
        if spike_reject:
            conv = np.fmin(conv[2:], conv[:-2])
        max_deriv[i] = np.max(conv)

    return max_deriv


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_max_deriv(pulse_data, ignore_leading, spike_reject=True, kernel=None):
    """Computes the maximum derivative in timeseries <pulse_data>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.

    Args:
        pulse_data:
        ignore_leading:
        spike_reject: (default True)
        kernel: the linear filter against which the signals will be convolved
            (CONVOLED, not correlated, so reverse the filter as needed). If None,
            then the default kernel of [+.2 +.1 0 -.1 -.2] will be used. If
            "SG", then the cubic 5-point Savitzky-Golay filter will be used (see
            below). Otherwise, kernel needs to be a (short) array which will
            be converted to a 1xN 2-dimensional np.ndarray. (default None)

    Returns:
        An np.ndarray, dimension 1: the value of the maximum derivative (units of <pulse_data units> per sample).

    When kernel=="SG", then we estimate the derivative by Savitzky-Golay filtering
    (with 1 point before/3 points after the point in question and fitting polynomial
    of order 3).  Find the right general area by first doing a simple difference.
    """
    cdef:
        double f0, f1, f2, f3, f4
        double t0, t1, t2, t3, t_max_deriv
        Py_ssize_t i, j
        const unsigned short[:, :] pulse_view
        const unsigned short[:] pulses
        double[:] max_deriv

    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim > 2 or ndim < 1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim == 1:
        pulse_data.shape = (1, pulse_data.shape[0])
    pulse_view = pulse_data[:, ignore_leading:]
    NPulse = pulse_view.shape[0]
    NSamp = pulse_view.shape[1]

    # The default filter:
    filter_coef = np.array([+.2, +.1, 0, -.1, -.2])
    if kernel == "SG":
        # This filter is the Savitzky-Golay filter of n_L=1, n_R=3 and M=3, to use the
        # language of Numerical Recipes 3rd edition.  It amounts to least-squares fitting
        # of an M=3rd order polynomial to the five points [-1,+3] and
        # finding the slope of the polynomial at 0.
        # Note that we reverse the order of coefficients because convolution will re-reverse
        filter_coef = np.array([-0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905])[::-1]

    elif kernel is not None:
        filter_coef = np.array(kernel).ravel()

    f0, f1, f2, f3, f4 = filter_coef

    max_deriv = np.zeros(NPulse, dtype=np.float64)

    if spike_reject:
        for i in range(NPulse):
            pulses = pulse_view[i]
            t0 = f4 * pulses[0] + f3 * pulses[1] + f2 * pulses[2] + f1 * pulses[3] + f0 * pulses[4]
            t1 = f4 * pulses[1] + f3 * pulses[2] + f2 * pulses[3] + f1 * pulses[4] + f0 * pulses[5]
            t2 = f4 * pulses[2] + f3 * pulses[3] + f2 * pulses[4] + f1 * pulses[5] + f0 * pulses[6]
            t_max_deriv = t2 if t2 < t0 else t0

            for j in range(7, NSamp):
                t3 = f4 * pulses[j - 4] + f3 * pulses[j - 3] + \
                    f2 * pulses[j - 2] + f1 * pulses[j - 1] + f0 * pulses[j]
                t4 = t3 if t3 < t1 else t1
                if t4 > t_max_deriv:
                    t_max_deriv = t4

                t0, t1, t2 = t1, t2, t3

            max_deriv[i] = t_max_deriv
    else:
        for i in range(NPulse):
            pulses = pulse_view[i]
            t0 = f4 * pulses[0] + f3 * pulses[1] + f2 * pulses[2] + f1 * pulses[3] + f0 * pulses[4]
            t_max_deriv = t0

            for j in range(5, NSamp):
                t0 = f4 * pulses[j - 4] + f3 * pulses[j - 3] + \
                    f2 * pulses[j - 2] + f1 * pulses[j - 1] + f0 * pulses[j]
                if t0 > t_max_deriv:
                    t_max_deriv = t0
            max_deriv[i] = t_max_deriv

    return np.asarray(max_deriv, dtype=np.float32)


########################################################################################
# Drift correction and related algorithms

class HistogramSmoother:
    """Object that can repeatedly smooth histograms with the same bin count and
    width to the same Gaussian width.  By pre-computing the smoothing kernel for
    that histogram, we can smooth multiple histograms with the same geometry.
    """

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
        while pow2 < nbins:
            pow2 *= 2
        self.nbins = pow2
        self.stepsize = dlimits / self.nbins

        # Compute the Fourier-space smoothing kernel
        kernel = np.exp(-0.5*(np.arange(self.nbins)*self.stepsize/self.smooth_sigma)**2)
        kernel[1:] += kernel[-1:0:-1]  # Handle the negative frequencies
        kernel /= kernel.sum()
        self.kernel_ft = np.fft.rfft(kernel)

    def __call__(self, values):
        """Return a smoothed histogram of the data vector <values>"""
        contents, _ = np.histogram(values, self.nbins, self.limits)
        ftc = np.fft.rfft(contents)
        csmooth = np.fft.irfft(self.kernel_ft * ftc)
        csmooth[csmooth < 0] = 0
        return csmooth


@cython.embedsignature(True)
def make_smooth_histogram(values, smooth_sigma, limit, upper_limit=None):
    """Convert a vector of arbitrary <values> info a smoothed histogram by
    histogramming it and smoothing.

    This is a convenience function using the HistogramSmoother class.

    Args:
        values: The vector of data to be histogrammed.
        smooth_sigma: The smoothing Gaussian's width (FWHM)
        limit, upper_limit: The histogram limits are [limit,upper_limit] or
            [0,limit] if upper_limit is None.

    Returns:
        The smoothed histogram as an array.
    """
    if upper_limit is None:
        limit, upper_limit = 0, limit
    return HistogramSmoother(smooth_sigma, [limit, upper_limit])(values)


@cython.embedsignature(True)
def drift_correct(indicator, uncorrected, limit=None):
    """Compute a drift correction that minimizes the spectral entropy.

    Args:
        indicator: The "x-axis", which indicates the size of the correction.
        uncorrected: A filtered pulse height vector. Same length as indicator.
            Assumed to have some gain that is linearly related to indicator.
        limit: The upper limit of uncorrected values over which entropy is
            computed (default None).

    Generally indicator will be the pretrigger mean of the pulses, but you can
    experiment with other choices.

    The entropy will be computed on corrected values only in the range
    [0, limit], so limit should be set to a characteristic large value of
    uncorrected. If limit is None (the default), then in will be compute as
    25% larger than the 99%ile point of uncorrected.

    The model is that the filtered pulse height PH should be scaled by (1 +
    a*PTM) where a is an arbitrary parameter computed here, and PTM is the
    difference between each record's pretrigger mean and the median value of all
    pretrigger means. (Or replace "pretrigger mean" with whatever quantity you
    passed in as <indicator>.)
    """
    ptm_offset = np.median(indicator)
    indicator = np.array(indicator) - ptm_offset

    if limit is None:
        pct99 = np.percentile(uncorrected, 99)
        limit = 1.25 * pct99

    smoother = HistogramSmoother(0.5, [0, limit])

    def entropy(param, indicator, uncorrected, smoother):
        corrected = uncorrected * (1+indicator*param)
        hsmooth = smoother(corrected)
        w = hsmooth > 0
        return -(np.log(hsmooth[w])*hsmooth[w]).sum()

    drift_corr_param = sp.optimize.brent(entropy, (indicator, uncorrected, smoother), brack=[0, .001])

    drift_correct_info = {'type': 'ptmean_gain',
                                  'slope': drift_corr_param,
                                  'median_pretrig_mean': ptm_offset}
    return drift_corr_param, drift_correct_info


@cython.embedsignature(True)
def python_nearest_arrivals(reference_times, other_times):
    """Identical to nearest_arrivals(...)."""
    nearest_after_index = np.searchsorted(other_times, reference_times)
    # because both sets of arrival times should be sorted, there are faster algorithms than searchsorted
    # for example: https://github.com/kwgoodman/bottleneck/issues/47
    # we could use one if performance becomes an issue
    last_index = np.searchsorted(nearest_after_index, other_times.size, side="left")
    first_index = np.searchsorted(nearest_after_index, 1)

    nearest_before_index = np.copy(nearest_after_index)
    nearest_before_index[:first_index] = 1
    nearest_before_index -= 1
    before_times = reference_times-other_times[nearest_before_index]
    before_times[:first_index] = np.Inf

    nearest_after_index[last_index:] = other_times.size-1
    after_times = other_times[nearest_after_index]-reference_times
    after_times[last_index:] = np.Inf

    return before_times, after_times


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def nearest_arrivals(long long[:] pulse_timestamps, long long[:] external_trigger_timestamps):
    """Find the external trigger time immediately before and after each pulse timestamp

    Args:
        pulse_timestamps - 1d array of pulse timestamps whose nearest neighbors
            need to be found.
        external_trigger_timestamps - 1d array of possible nearest neighbors.

    Returns:
        (before_times, after_times)

    before_times is an ndarray of the same size as pulse_timestamps.
    before_times[i] contains the difference between the closest lesser time
    contained in external_trigger_timestamps and pulse_timestamps[i]  or inf if there was no
    earlier time in other_times Note that before_times is always a positive
    number even though the time difference it represents is negative.

    after_times is an ndarray of the same size as pulse_timestamps.
    after_times[i] contains the difference between pulse_timestamps[i] and the
    closest greater time contained in other_times or a inf number if there was
    no later time in external_trigger_timestamps.
    """
    cdef:
        Py_ssize_t num_pulses, num_triggers
        Py_ssize_t i = 0, j = 0, t
        long long[:] delay_from_last_trigger
        long long[:] delay_until_next_trigger
        long long a, b, pt
        long long max_value

    num_pulses = pulse_timestamps.shape[0]
    num_triggers = external_trigger_timestamps.shape[0]

    if num_pulses < 1:
        return np.array([], dtype=np.int64)

    delay_from_last_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)
    delay_until_next_trigger = np.zeros_like(pulse_timestamps, dtype=np.int64)

    max_value = np.iinfo(np.int64).max

    if num_triggers > 1:
        a = external_trigger_timestamps[0]
        b = external_trigger_timestamps[1]
        j = 1

        # handle the case where pulses arrive before the fist external trigger
        while True:
            pt = pulse_timestamps[i]
            if pt < a:
                delay_from_last_trigger[i] = max_value
                delay_until_next_trigger[i] = a - pt
                i += 1
                if i >= num_pulses:
                    return np.asarray(delay_from_last_trigger, dtype=np.int64),\
                           np.asarray(delay_until_next_trigger, dtype=np.int64)
            else:
                break

        # At this point in the code a and b are values from
        # external_trigger_timestamps that bracket pulse_timestamp[i]
        while True:
            pt = pulse_timestamps[i]
            if pt < b:
                delay_from_last_trigger[i] = pt - a
                delay_until_next_trigger[i] = b - pt
                i += 1
                if i >= num_pulses:
                    break
            else:
                j += 1
                if j >= num_triggers:
                    break
                else:
                    a, b = b, external_trigger_timestamps[j]

        # handle the case where pulses arrive after the last external trigger
        for t in range(i, num_pulses):
            delay_from_last_trigger[t] = pulse_timestamps[t] - b
            delay_until_next_trigger[t] = max_value
    elif num_triggers > 0:
        a = b = external_trigger_timestamps[0]

        for i in range(num_pulses):
            pt = pulse_timestamps[i]
            if pt > a:
                delay_from_last_trigger[i] = pt - a
                delay_until_next_trigger[i] = max_value
            else:
                delay_from_last_trigger[i] = max_value
                dealay_until_next_trigger = a - pt
    else:
        for i in range(num_pulses):
            delay_from_last_trigger[i] = max_value
            delay_until_next_trigger[i] = max_value

    return (np.asarray(delay_from_last_trigger, dtype=np.int64),
            np.asarray(delay_until_next_trigger, dtype=np.int64))


def filter_signal_lowpass(sig, fs, fcut):
    """Tophat lowpass filter using an FFT

    Args:
        sig - the signal to be filtered
        fs - the sampling frequency of the signal
        fcut - the frequency at which to cutoff the signal

    Returns:
        the filtered signal
    """
    N = sig.shape[0]
    SIG = np.fft.fft(sig)
    freqs = (fs/N) * np.concatenate((np.arange(0, N/2+1), np.arange(N/2-1, 0, -1)))
    filt = np.zeros_like(SIG)
    filt[freqs < fcut] = 1.0
    sig_filt = np.fft.ifft(SIG * filt)
    return sig_filt


def correct_flux_jumps(vals, g, flux_quant):
    '''Remove 'flux' jumps' from pretrigger mean.

    When using umux readout, if a pulse is recorded that has a very fast rising
    edge (e.g. a cosmic ray), the readout system will "slip" an integer number
    of flux quanta. This means that the baseline level returned to after the
    pulse will different from the pretrigger value by an integer number of flux
    quanta. This causes that pretrigger mean summary quantity to jump around in
    a way that causes trouble for the rest of MASS. This function attempts to
    correct these jumps.

    Arguments:
    vals -- array of values to correct
    g -- mask indentifying "good" pulses
    flux_quant -- size of 1 flux quanta

    Returns:
    Array with values corrected
    '''
    # The naive thing is to simply replace each value with its value mod
    # the flux quantum. But of the baseline value turns out to fluctuate
    # about an integer number of flux quanta, this will introduce new
    # jumps. I don't know the best way to handle this in general. For now,
    # if there are still jumps after the mod, I add 1/4 of a flux quanta
    # before modding, then mod, then subtract the 1/4 flux quantum and then
    # *add* a single flux quantum so that the values never go negative.
    #
    # To determine whether there are "still jumps after the mod" I look at the
    # difference between the largest and smallest values for "good" pulses. If
    # you don't exclude "bad" pulses, this check can be tricked in cases where
    # the pretrigger section contains a (sufficiently large) tail.
    if (np.amax(vals) - np.amin(vals)) >= flux_quant:
        corrected = vals % (flux_quant)
        if (np.amax(corrected[g]) - np.amin(corrected[g])) > 0.75*flux_quant:
            corrected = (vals + flux_quant/4) % (flux_quant)
            corrected = corrected - flux_quant/4 + flux_quant
        corrected = corrected - (corrected[0] - vals[0])
        return corrected
    else:
        return vals


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def filter_data_5lag_cython(
    const unsigned short[:,:] rawdata,
    double[:] filter_values,
    int pulses_per_seg=0):
    """Filter the complete data file one chunk at a time."""
    cdef:
        Py_ssize_t i, j, k
        int n_segments, pulse_size_bytes, bigblock
        int nPulses, nSamples, filter_length
        double conv0, conv1, conv2, conv3, conv4
        float[:] filt_phase_scratch_array, filt_value_scratch_array
        unsigned short sample
        double f0, f1, f2, f3, f4
        double p0, p1, p2

    nPulses = rawdata.shape[0]
    nSamples = rawdata.shape[1]
    filter_length = nSamples - 4
    if pulses_per_seg <= 0:
        pulse_size_bytes = 16+2*nSamples
        bigblock = 2**22
        pulses_per_seg = max(bigblock // pulse_size_bytes, 1)
    n_segments = 1+(nPulses-1) // pulses_per_seg

    filt_phase = np.zeros(nPulses, dtype=np.float64)
    filt_value = np.zeros(nPulses, dtype=np.float64)

    for i in range(nPulses):
        pulse = rawdata[i, :]

        f0, f1, f2, f3 = filter_values[0], filter_values[1], filter_values[2], filter_values[3]

        conv0 = pulse[0] * f0 + pulse[1] * f1 + pulse[2] * f2 + pulse[3] * f3
        conv1 = pulse[1] * f0 + pulse[2] * f1 + pulse[3] * f2
        conv2 = pulse[2] * f0 + pulse[3] * f1
        conv3 = pulse[3] * f0
        conv4 = 0.0

        for k in range(4, nSamples - 4):
            f4 = filter_values[k]
            sample = pulse[k]
            conv0 += sample * f4
            conv1 += sample * f3
            conv2 += sample * f2
            conv3 += sample * f1
            conv4 += sample * f0
            f0, f1, f2, f3 = f1, f2, f3, f4

        conv4 += pulse[nSamples-4] * f0 + pulse[nSamples-3] * f1 +\
            pulse[nSamples-2] * f2 + pulse[nSamples-1] * f3
        conv3 += pulse[nSamples-4] * f1 + pulse[nSamples-3] * f2 + pulse[nSamples-2] * f3
        conv2 += pulse[nSamples-4] * f2 + pulse[nSamples-3] * f3
        conv1 += pulse[nSamples-4] * f3

        p0 = conv0*(-6.0/70) + conv1*(24.0/70) + conv2*(34.0/70) + conv3*(24.0/70) + conv4*(-6.0/70)
        p1 = conv0*(-14.0/70) + conv1*(-7.0/70) + conv3*(7.0/70) + conv4*(14.0/70)
        p2 = conv0*(10.0/70) + conv1*(-5.0/70) + conv2*(-10.0/70) + conv3*(-5.0/70) + conv4*(10.0/70)

        filt_phase[i] = -0.5*p1 / p2
        filt_value[i] = p0 - 0.25*p1**2 / p2
    return filt_value, filt_phase