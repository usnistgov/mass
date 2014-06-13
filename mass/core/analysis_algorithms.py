'''
mass.core.analysis_algorithms - main algorithms used in data analysis

Designed to ... ?

Created on Jun 9, 2014

@author: fowlerj
'''

__all__ = ['make_smooth_histogram', 'HistogramSmoother', 'FilterTimeCorrection']

import numpy as np
import scipy as sp
import pylab as plt
import sklearn.cluster


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
    ndim = len(pulse_data.shape)
    if ndim>2 or ndim<1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim==1:
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



def compute_max_deriv(pulse_data, ignore_leading, spike_reject=True, kernel=None):
    """Compute the maximum derivative in timeseries <pulse_data>.
    <pulse_data> can be a 2D array where each row is a different pulse record, in which case
    the return value will be an array last long as the number of rows in <pulse_data>.
    
    Return the value of the maximum derivative (units of <pulse_data units> per sample).
    
    If <spike_reject>, then 
    
    <kernel> is the linear filter against which the signals will be convolved
    (CONVOLED, no correlated, so reverse the filter as needed). If None, then the
    default kernel of [+.2 +.1 0 -.1 -.2] will be used. If "SG", then the cubic 5-point
    Savitzky-Golay filter will be used (see below). Otherwise, <kernel> needs to be a
    (short) array which will be converted to a 1xN 2-dimensional np.ndarray.
    
    When kernel=="SG", then we estimate the derivative by Savitzky-Golay filtering 
    (with 1 point before/3 points after the point in question and fitting polynomial 
    of order 3).  Find the right general area by first doing a simple difference.
    """
    
    # If pulse_data is a 1D array, turn it into 2
    pulse_data = np.asarray(pulse_data)
    ndim = len(pulse_data.shape)
    if ndim>2 or ndim<1:
        raise ValueError("input pulse_data should be a 1d or 2d array.")
    if ndim==1:
        pulse_data.shape = (1, pulse_data.shape[0])
    pulse_data = np.array(pulse_data[:,ignore_leading:], dtype=float)
    NPulse, NSamp = pulse_data.shape

#     # Get a rough estimate of the place where pulse derivative is largest
#     # by taking difference between samples i+1 and i-1.
#     rough_imax = 1 + (pulse_data[:,2:]-pulse_data[:,:-2]).argmax(axis=1)
#     HALFRANGE=12
#     
#     # If samples are too few, then rough is all we get.
#     if NSamp < 2*HALFRANGE:
#         approx_deriv = (pulse_data[:,2:]-pulse_data[:,:-2]).max(axis=1)
#         if return_index_too:
#             return approx_deriv, rough_imax
#         return approx_deriv
#         
#     # Go +- 12 around that rough estimate, but bring into allowed
#     # range [0,NSamp)
#     first,end = rough_imax-HALFRANGE, rough_imax+HALFRANGE
#     end[first<0] = 2*HALFRANGE
#     first[first<0] = 0
#     first[end>=NSamp] = NSamp-2*HALFRANGE
#     end[end>=NSamp] = NSamp
#     
#     # Now trim the pulse_data, allowing us to filter only 
#     trimmed_data = np.zeros((NPulse, 2*HALFRANGE),dtype=float)
#     for r,data in enumerate(pulse_data):
#         trimmed_data[r,:] = data[first[r]:end[r]]

    # The default filter:    
    filter_coef = np.array([+.2, +.1, 0, -.1, -.2])
    if kernel == "SG":
        # This filter is the Savitzky-Golay filter of n_L=1, n_R=3 and M=3, to use the
        # language of Numerical Recipes 3rd edition.  It amounts to least-squares fitting
        # of an M=3rd order polynomial to the five points [-1,+3] and
        # finding the slope of the polynomial at 0.
        # Note that we reverse the order of coefficients because convolution will re-reverse
        filter_coef = np.array([ -0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905,   ])[::-1]
        
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



########################################################################################
# Drift correction and related algorithms

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



########################################################################################
# Arrival-time correction

class FilterTimeCorrection(object):
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
                 linearFilter, nPresamples, typicalResolution=None, labels=None, maxorder=6):
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
        if trainingPulses is  None: return  # used in self.copy() only
        
        _,M = trainingPulses.shape
        F = len(linearFilter)
        if F>M or (M-F)%2 != 0:
            raise RuntimeError(
                "The filter (length %d) should be equal in length to training pulses (%d)\n"%(F,M)+
                "or shorter by an even number of samples")
        
        if labels is None:
            if typicalResolution is None:
                raise RuntimeError("Must give either labels or typicalResolution as inputs!")
            
            cluster_width = 2*typicalResolution
            labels = self._label_data(energy, cluster_width)
        else:
            labels = np.array(labels)
        
        self._sort_labels(labels, energy)
        self.labels=labels
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
        min_samples = max(MIN_PULSES, int(0.5+ 0.01*MIN_PCT*N))

        _core_samples, labels = sklearn.cluster.dbscan(energy.reshape((N,1)), eps=res, 
                                                       min_samples=min_samples)
        labels = np.asarray(labels, dtype=int)
        labelCounts,_ = np.histogram(labels, 1+labels.max(), [-.5, .5+labels.max()])
        print 'Label counts: ', labelCounts
        return labels
    

    def _sort_labels(self, labels, energy):
        # Now sort the labels from low to high energy
        NL = int(1+labels.max())
        self.meanEnergyByLabel = np.zeros(NL, dtype=float)
        for i in range(NL):
            self.meanEnergyByLabel[i] = energy[labels==i].mean()
        
        args = self.meanEnergyByLabel.argsort()
        self.meanEnergyByLabel = self.meanEnergyByLabel[args]
        labels[labels>=0] = args.argsort()[labels[labels>=0]]
        

    def _make_pulse_model(self, trainingPulses, promptness, energy, labels):
        """Fit the data samples."""
        
        _nPulses, nSamp = trainingPulses.shape
        self.num_zeros = self.nPresamples+2
        self.nSamp = nSamp
        
        self.raw_fits={}
        self.prompt_range={}
    
        # Loop over spectral lines (or clusters)
        Nlabels = 1+labels.max()
        for i in range(Nlabels):
            self.raw_fits[i] = np.zeros((nSamp-(self.num_zeros), 
                                         self.max_poly_order+1), dtype=np.float)
            
            use = (labels==i)
            print 'Using %4d pulses for cluster %d'%(use.sum(), i)
            
            prompt = promptness[use]
#             pulse_rms = energy[use]
            ptmean = trainingPulses[use, :self.nPresamples].mean(axis=1)
            med = np.median(prompt)
            self.prompt_range[i] = np.array((sp.stats.scoreatpercentile(prompt, 1),
                med, sp.stats.scoreatpercentile(prompt, 99)))
                
            later_order = min(self.max_poly_order,3)
            for j in range(self.num_zeros, nSamp):
                # For the first few samples, do a high-order fit
                if j<=18+self.num_zeros:
                    fit = np.polyfit(prompt-med, trainingPulses[use,j]-ptmean, self.max_poly_order)
                    self.raw_fits[i][j-self.num_zeros,:] = fit
                    
                # For the later samples, a cubic will suffice (?)
                else:
                    fit = np.polyfit(prompt-med, trainingPulses[use,j]-ptmean, later_order)
                    self.raw_fits[i][j-self.num_zeros,-1-later_order:] = fit


    def _filter_model(self, Nlabels, linearFilter, plot=True):
        self.lag0_results={}
        self.parab_results={}
         
        if plot: 
            plt.clf()
            axes = [plt.subplot(Nlabels,2,1)]
            for i in range(2,1+2*Nlabels):
                if i%2==0:
                    axes.append(plt.subplot(Nlabels,2,i, sharex=axes[0], sharey=axes[-1]))
                else:
                    axes.append(plt.subplot(Nlabels,2,i, sharex=axes[0]))
            axes[-2].set_xlabel("Promptness")
            axes[-1].set_xlabel("Promptness")
            axes[0].set_title("Lag 0 filter output")
            axes[1].set_title("5-lag parab filter output")
            colors = [plt.cm.jet(float(i)/(Nlabels-1.)) for i in range(Nlabels)]
         
        # Loop over labels
        for i in range(Nlabels):
            fit = np.zeros((self.nSamp, self.raw_fits[i].shape[1]), dtype=np.float)
            fit[self.num_zeros:,:] = self.raw_fits[i]
              
            # These parameters fit a parabola to any 5 evenly-spaced points
            fit_array = np.array((
                    ( -6,24, 34,24,-6),
                    (-14,-7,  0, 7,14),
                    ( 10,-5,-10,-5,10)), dtype=np.float)/70.0
      
            pvalues = np.linspace(self.prompt_range[i][0]-.003, self.prompt_range[i][2]+.003, 60) 
            med_prompt = self.prompt_range[i][1]
              
            output_lag0 = np.zeros_like(pvalues)
            output_fit = np.zeros_like(pvalues)
              
            for ip,prompt in enumerate(pvalues):
                pwrs_prompt = (prompt-med_prompt)**np.arange(self.max_poly_order,-0.5,-1)
                model = np.dot(fit, pwrs_prompt)
  
                conv = np.zeros(5, dtype=np.float)
                conv[:4] = [np.dot(model[k:k-4], linearFilter) for k in range(4)]
                conv[4] = np.dot(model[4:], linearFilter)
              
                parab_param = np.dot(fit_array, conv)
                peak_x = -0.5*parab_param[1]/parab_param[2]
                peak_y = parab_param[0] - 0.25*parab_param[1]**2 / parab_param[2] 
                  
                output_lag0[ip] = conv[2]
                output_fit[ip] = peak_y
  
            self.lag0_results[i] = (pvalues, np.array(output_lag0))
            self.parab_results[i] = (pvalues, np.array(output_fit))
            print "Cluster %2d: FWHM lag 0: %.3f  5-lag fit: %.3f"%(i, 2.3548*np.std(output_lag0), 2.3548*np.std(output_fit))
              
            if plot:
                ax = axes[(Nlabels-1-i)*2]
                ax.plot(pvalues, output_lag0, 'o', color=colors[i])
  
                ax.text(.1,.85, 'Cluster %2d:  FWHM: %.2f arbs'%(i,2.3548*np.std(output_lag0)), transform=ax.transAxes)
                ax = axes[(Nlabels-1-i)*2+1]
                ax.plot(pvalues, output_fit, 'd-', color=colors[i])
                ax.text(.1,.85, 'Cluster %2d:  FWHM: %.2f arbs'%(i, 2.3548*np.std(output_fit)), transform=ax.transAxes)
 
 
    def _create_interpolation(self, Nlabels):
        """generate cubic spline for each curve, and then use the linear
        interp of those, based one which interval of the four: <Cr, Cr-Mn, Mn-Fe, >Fe"""
        self.meanEnergyByLabel
        
        self.splines=[] 
        for i in range(Nlabels):
            x,y = self.parab_results[i]
            y = y-y.mean()
            spl = sp.interpolate.UnivariateSpline(x, y, w=25+np.zeros_like(y), s=len(x), k=3,
                                                  bbox=[x[0]-.05, x[-1]+.05])
            self.splines.append(spl)
        self.interval_boundaries=self.meanEnergyByLabel
        
        
    def __call__(self, prompt, pulse_rms):
        """Compute and return a pulse height correction for a filtered pulse with
        promptness 'prompt' and pulse_rms 'pulse_rms'. Can be arrays."""
        if not isinstance(prompt, np.ndarray) or not isinstance(pulse_rms, np.ndarray):
            return  self(np.asarray(prompt), np.asarray(pulse_rms))
        if prompt.ndim==0: prompt = np.asarray([prompt])
        if pulse_rms.ndim==0: pulse_rms = np.asarray([pulse_rms])
        #@todo: handle case where one prompt or one pulse_rms is given, by broadcasting
        result = np.zeros_like(prompt)
 
        n_intervals = len(self.interval_boundaries)-1
        pulse_interval = np.zeros(len(pulse_rms))
        for ib in range(1,n_intervals):
            pulse_interval[pulse_rms > self.interval_boundaries[ib]] = ib
 
        for interval in range(n_intervals):
            a,b = self.interval_boundaries[interval:interval+2]
            use = pulse_interval == interval
            if use.sum() <= 0: continue
            fraction = (pulse_rms[use]-a)/(b-a)
#             # Limit extrapolation
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
            xraw,y= self.parab_results[i]
            x = xraw.copy()
            y = y-y.mean()
 
            if center_x: x -= self.prompt_range[i][1]
            if scale_x: x /= (self.prompt_range[i][2]-self.prompt_range[i][0])
            plt.plot(x, self.splines[i](xraw), 'gray')
            plt.plot(x,y,'d-', color=colors[i])
            xlab = "Promptness"
            if center_x: xlab += ', centered'
            if scale_x: xlab += ', scaled'
            plt.xlabel(xlab)
            plt.ylabel("Correction, in raw (filtered) units")

