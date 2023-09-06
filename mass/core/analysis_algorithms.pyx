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
    """Compute the maximum derivative in each pulse

    :param pulse_data: pulse data
    :type pulse_data: ndarray
    :param ignore_leading: ignore this many samples (presumably, samples where usual pulses are rising)
    :type ignore_leading: int
    :param spike_reject: whether to use the spike-reject algorithm, defaults to True
    :type spike_reject: bool, optional
    :param kernel: convolution kernel for estimating derivatives, defaults to None (in which case it uses (+.2, +.1, 0, -.1, -.2))
    :type kernel: various, optional
    :raises ValueError: input `pulse_data` should be a 1d or 2d array
    :return: the maximum derivative for each record in `pulse_data`
    :rtype: ndarray
    """
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
    """Compute the maximum derivative in each pulse

    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.

    When kernel=="SG", then we estimate the derivative by Savitzky-Golay filtering
    (with 1 point before/3 points after the point in question and fitting polynomial
    of order 3).  Find the right general area by first doing a simple difference.

    :param pulse_data: pulse data
    :type pulse_data: ndarray
    :param ignore_leading: ignore this many samples (presumably, samples where usual pulses are rising)
    :type ignore_leading: int
    :param spike_reject: whether to use the spike-reject algorithm, defaults to True
    :type spike_reject: bool, optional
    :param kernel:the linear filter against which the signals will be convolved
            (CONVOLED, not correlated, so reverse the filter as needed). If None,
            then the default kernel of [+.2 +.1 0 -.1 -.2] will be used. If
            "SG", then the cubic 5-point Savitzky-Golay filter will be used (see
            below). Otherwise, kernel needs to be a (short) array which will
            be converted to a 1xN 2-dimensional np.ndarray, defaults to None
    :type kernel: various, optional
    :raises ValueError: input `pulse_data` should be a 1d or 2d array
    :return: the maximum derivative for each record in `pulse_data`
    :rtype: ndarray
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
    """Convert a vector of arbitrary <values> info a smoothed histogram by histogramming it and smoothing.

    This is a convenience function using the HistogramSmoother class.

    :param values: The vector of data to be histogrammed
    :type values: ndarray
    :param smooth_sigma: The smoothing Gaussian's width (FWHM)
    :type smooth_sigma: float
    :param limit: The histogram limits are [limit,upper_limit] or [0,limit] if upper_limit is None.
    :type limit: float
    :param upper_limit: histogram upper limit, defaults to None
    :type upper_limit: float, optional
    :return: The smoothed histogram as an array.
    :rtype: ndarray
    """
    if upper_limit is None:
        limit, upper_limit = 0, limit
    return HistogramSmoother(smooth_sigma, [limit, upper_limit])(values)


@cython.embedsignature(True)
def drift_correct(indicator, uncorrected, limit=None):
    """Compute a drift correction that minimizes the spectral entropy.

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

    :param indicator: The "x-axis", which indicates the size of the correction.
    :type indicator: ndarray
    :param uncorrected: A filtered pulse height vector. Same length as indicator.
        Assumed to have some gain that is linearly related to indicator.
    :type uncorrected: ndarray
    :param limit: The upper limit of uncorrected values over which entropy is computed, defaults to None
    :type limit: _type_, optional
    :return: _description_
    :rtype: _type_
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


# @cython.embedsignature(True)


########################################################################################
# Arrival-time correction

class FilterTimeCorrection:
    """Represent the phase-dependent correction to a filter, based on
    running model pulses through the filter.  Developed November 2013 to
    June 2014.

    Unlike a phase-appropriate filter approach, the idea here is that all pulses
    will be passed through the same filter. This object studies the systematics
    that results and attempts to find a good correction for them

    WARNING: This seems to work great on calibronium data (4 to 8 keV with lines) on
    the Tupac system, but it didn't do so well on Mn fluorescence in Dixie's 2011
    snout with best-ever TDM-8. This is a work in progress.
    """

    def __init__(self, trainingPulses, promptness, energy,
                 linearFilter, nPresamples, typicalResolution=None, labels=None, maxorder=6, verbose=0):
        """
        Create a filtered pulse height time correction from various ingredients:
        trainingPulses  An NxM array of N pulse records with M samples each
        promptness      A length-N vector with the promptness (arrival time proxy)
        energy          A length-N vector with any reasonable proxy for pulse energy
        linearFilter    The filter, of length F <= M
        nPresamples     Number of samples that precede the edge trigger nominal position
        typicalResolution  The rough idea of the energy resolution, in same units as
                        the energy vector. (optional)
        labels          A length-N vector with integer values [-1,L). Pulses with
                        label -1 will be ignored. Pulses with non-negative label will
                        be combined with others of the same label. (optional)
        maxorder        ??

        Cuts: no data cuts are performed. Ensure that the trainingPulses, promptness,
        and energy are already selected for events that pass cuts.

        If F<M, then we assume that (M-F) is even and represents and equal number
        of samples omitted from the start and end of a pulse for the purpose of
        being able to do the multi-lag correlation.

        If labels is None, then a clustering algorithm will be run to choose the
        labeling. In that event, typicalResolution must be given.

        If trainingPulses is None, then no computations will be done (this is
        for use in the copy method only, really).
        """

        self.pulse_model = None
        self.max_poly_order = maxorder
        self.filter = np.array(linearFilter)
        self.nPresamples = nPresamples
        self.verbose = verbose
        if trainingPulses is None:
            return  # used in self.copy() only

        _, M = trainingPulses.shape
        F = len(linearFilter)
        if F > M or (M-F) % 2 != 0:
            raise RuntimeError(
                """The filter (length %d) should be equal in length to training pulses (%d)
                "or shorter by an even number of samples""" % (F, M))

        if labels is None:
            if typicalResolution is None:
                raise RuntimeError("Must give either labels or typicalResolution as inputs!")

            cluster_width = 2*typicalResolution
            labels = self._label_data(energy, cluster_width)
        else:
            labels = np.array(labels)

        self._sort_labels(labels, energy)
        self.labels = labels
        self._make_pulse_model(trainingPulses, promptness, energy, labels)
        Nlabels = 1+labels.max()
        self._filter_model(Nlabels, linearFilter)
        self._create_interpolation(Nlabels)

    def copy(self):
        """Return a new object with all the properties of this"""
        new = FilterTimeCorrection(None, None, None, None)
        new.__dict__.update(self.__dict__)
        return new

    def _label_data(self, energy, res):
        """Label all pulses by clustering them in energy.
        <energy> is an energy indicator (generally, a pulse height)
        <res> is the cluster width in the same units as <energy>"""

        # Ignore clusters containing < 2% of the data or <50 pulses
        MIN_PCT = 2.0
        MIN_PULSES = 50
        N = len(energy)
        min_samples = max(MIN_PULSES, int(0.5 + 0.01*MIN_PCT*N))

        _core_samples, labels = sklearn.cluster.dbscan(energy.reshape((N, 1)), eps=res,
                                                       min_samples=min_samples)
        labels = np.asarray(labels, dtype=int)
        labelCounts, _ = np.histogram(labels, 1+labels.max(), [-.5, .5+labels.max()])
        if self.verbose > 0:
            print('Label counts: ', labelCounts)
        return labels

    def _sort_labels(self, labels, energy):
        # Now sort the labels from low to high energy
        NL = int(1+labels.max())
        self.meanEnergyByLabel = np.zeros(NL, dtype=float)
        for i in range(NL):
            self.meanEnergyByLabel[i] = energy[labels == i].mean()

        args = self.meanEnergyByLabel.argsort()
        self.meanEnergyByLabel = self.meanEnergyByLabel[args]
        labels[labels >= 0] = args.argsort()[labels[labels >= 0]]

    def _make_pulse_model(self, trainingPulses, promptness, energy, labels):
        """Fit the data samples."""

        _nPulses, nSamp = trainingPulses.shape
        self.num_zeros = self.nPresamples-1
        self.nSamp = nSamp

        self.raw_fits = {}
        self.prompt_range = {}

        # Loop over spectral lines (or clusters)
        Nlabels = 1+labels.max()
        for i in range(Nlabels):
            self.raw_fits[i] = np.zeros((nSamp - self.num_zeros,
                                         self.max_poly_order+1), dtype=float)

            use = (labels == i)
            if self.verbose > 0:
                print('Using %4d pulses for cluster %d' % (use.sum(), i))

            prompt = promptness[use]
            ptmean = trainingPulses[use, :self.nPresamples-1].mean(axis=1)
            med = np.median(prompt)
            self.prompt_range[i] = np.array((np.percentile(prompt, 1),
                                             med, np.percentile(prompt, 99)))

            later_order = min(self.max_poly_order, 3)
            for j in range(self.num_zeros, nSamp):
                # For the first few samples, do a high-order fit
                if j <= 18 + self.num_zeros:
                    fit = np.polyfit(prompt-med, trainingPulses[use, j]-ptmean, self.max_poly_order)
                    self.raw_fits[i][j-self.num_zeros, :] = fit

                # For the later samples, a cubic will suffice (?)
                else:
                    fit = np.polyfit(prompt-med, trainingPulses[use, j]-ptmean, later_order)
                    self.raw_fits[i][j-self.num_zeros, -1-later_order:] = fit

    def _filter_model(self, Nlabels, linearFilter, plot=True):
        self.lag0_results = {}
        self.parab_results = {}

        if plot:
            plt.clf()
            axes = [plt.subplot(Nlabels, 2, 1)]
            for i in range(2, 1 + 2*Nlabels):
                if i % 2 == 0:
                    axes.append(plt.subplot(Nlabels, 2, i, sharex=axes[0], sharey=axes[-1]))
                else:
                    axes.append(plt.subplot(Nlabels, 2, i, sharex=axes[0]))
            axes[-2].set_xlabel("Promptness")
            axes[-1].set_xlabel("Promptness")
            axes[0].set_title("Lag 0 filter output")
            axes[1].set_title("5-lag parab filter output")
            colors = [plt.cm.jet(float(i)/(Nlabels-1.)) for i in range(Nlabels)]

        # Loop over labels
        for i in range(Nlabels):
            fit = np.zeros((self.nSamp, self.raw_fits[i].shape[1]), dtype=float)
            fit[self.num_zeros:, :] = self.raw_fits[i]

            # These parameters fit a parabola to any 5 evenly-spaced points
            fit_array = np.array((
                    (-6, 24, 34, 24, -6),
                    (-14, -7, 0, 7, 14),
                    (10, -5, -10, -5, 10)), dtype=float)/70.0

            pvalues = np.linspace(self.prompt_range[i][0]-.003, self.prompt_range[i][2]+.003, 60)
            med_prompt = self.prompt_range[i][1]

            output_lag0 = np.zeros_like(pvalues)
            output_fit = np.zeros_like(pvalues)

            for ip, prompt in enumerate(pvalues):
                pwrs_prompt = (prompt-med_prompt)**np.arange(self.max_poly_order, -0.5, -1)
                model = np.dot(fit, pwrs_prompt)

                conv = np.zeros(5, dtype=float)
                conv[:4] = [np.dot(model[k:k-4], linearFilter) for k in range(4)]
                conv[4] = np.dot(model[4:], linearFilter)

                parab_param = np.dot(fit_array, conv)
                peak_x = -0.5*parab_param[1]/parab_param[2]
                peak_y = parab_param[0] - 0.25*parab_param[1]**2 / parab_param[2]

                output_lag0[ip] = conv[2]
                output_fit[ip] = peak_y

            self.lag0_results[i] = (pvalues, np.array(output_lag0))
            self.parab_results[i] = (pvalues, np.array(output_fit))
            if self.verbose > 0:
                print("Cluster %2d: FWHM lag 0: %.3f  5-lag fit: %.3f" %
                      (i, 2.3548*np.std(output_lag0), 2.3548*np.std(output_fit)))

            if plot:
                ax = axes[(Nlabels-1-i)*2]
                ax.plot(pvalues, output_lag0, 'o', color=colors[i])

                ax.text(.1, .85, 'Cluster %2d:  FWHM: %.2f arbs' % (i, 2.3548*np.std(output_lag0)),
                        transform=ax.transAxes)
                ax = axes[(Nlabels-1-i)*2+1]
                ax.plot(pvalues, output_fit, 'd-', color=colors[i])
                ax.text(.1, .85, 'Cluster %2d:  FWHM: %.2f arbs' % (i, 2.3548*np.std(output_fit)),
                        transform=ax.transAxes)

    def _create_interpolation(self, Nlabels):
        """generate cubic spline for each curve, and then use the linear
        interp of those, based one which interval of the four: <Cr, Cr-Mn, Mn-Fe, >Fe"""
        self.meanEnergyByLabel

        self.splines = []
        for i in range(Nlabels):
            x, y = self.parab_results[i]
            y = y-y.mean()
            spl = sp.interpolate.UnivariateSpline(x, y, w=25+np.zeros_like(y), s=len(x), k=3,
                                                  bbox=[x[0]-.05, x[-1]+.05])
            self.splines.append(spl)
        self.interval_boundaries = self.meanEnergyByLabel

    def __call__(self, prompt, pulse_rms):
        """Compute and return a pulse height correction for a filtered pulse with
        promptness 'prompt' and pulse_rms 'pulse_rms'. Can be arrays."""
        if not isinstance(prompt, np.ndarray) or not isinstance(pulse_rms, np.ndarray):
            return self(np.asarray(prompt), np.asarray(pulse_rms))
        if prompt.ndim == 0:
            prompt = np.asarray([prompt])
        if pulse_rms.ndim == 0:
            pulse_rms = np.asarray([pulse_rms])
        result = np.zeros_like(prompt)

        # If there are multiple pulse_rms each with their own curves in self.splines,
        # then we want to interpolate (linearly) between the nearest 2 results.
        # But first, handle the case of only one cluster=one interval.
        n_intervals = len(self.interval_boundaries)-1
        if n_intervals == 0:
            return self.splines[0](prompt)

        pulse_interval = np.zeros(len(pulse_rms))
        for ib in range(1, n_intervals):
            pulse_interval[pulse_rms > self.interval_boundaries[ib]] = ib

        for interval in range(n_intervals):
            a, b = self.interval_boundaries[interval:interval+2]
            use = pulse_interval == interval
            if use.sum() <= 0:
                continue
            fraction = (pulse_rms[use]-a)/(b-a)
            # Limit extrapolation
            fraction[fraction < -0.5] = -0.5
            fraction[fraction > 1.5] = 1.5
            f_a = self.splines[interval](prompt[use])
            f_b = self.splines[interval+1](prompt[use])
            result[use] = f_a + (f_b-f_a)*fraction
        return result

    def plot_corrections(self, center_x=True, scale_x=True):
        plt.clf()
        Nlabels = len(self.raw_fits)
        colors = [plt.cm.jet(float(i)/(Nlabels-1.)) for i in range(Nlabels)]
        for i in range(Nlabels):
            xraw, y = self.parab_results[i]
            x = xraw.copy()
            y = y-y.mean()

            if center_x:
                x -= self.prompt_range[i][1]
            if scale_x:
                x /= (self.prompt_range[i][2]-self.prompt_range[i][0])
            plt.plot(x, self.splines[i](xraw), 'gray')
            plt.plot(x, y, 'd-', color=colors[i])
            xlab = "Promptness"
            if center_x:
                xlab += ', centered'
            if scale_x:
                xlab += ', scaled'
            plt.xlabel(xlab)
            plt.ylabel("Correction, in raw (filtered) units")


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

    before_times is an ndarray of the same size as pulse_timestamps.
    before_times[i] contains the difference between the closest lesser time
    contained in external_trigger_timestamps and pulse_timestamps[i]  or inf if there was no
    earlier time in other_times Note that before_times is always a positive
    number even though the time difference it represents is negative.

    after_times is an ndarray of the same size as pulse_timestamps.
    after_times[i] contains the difference between pulse_timestamps[i] and the
    closest greater time contained in other_times or a inf number if there was
    no later time in external_trigger_timestamps.

    :param pulse_timestamps: 1d array of pulse timestamps whose nearest neighbors
            need to be found
    :type pulse_timestamps: ndarray
    :param external_trigger_timestamps: d array of possible nearest neighbors
    :type external_trigger_timestamps: ndarray
    :return: (before_times, after_times)
    :rtype: (ndarray, ndarray)
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