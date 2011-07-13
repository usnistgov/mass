"""
Created on Feb 16, 2011

@author: fowlerj
"""

import numpy
import scipy.linalg
import scipy.optimize
import scipy.special
from matplotlib import pylab
import os.path
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle    

# MASS modules
import controller
import files
import utilities
import power_spectrum
import energy_calibration
import fluorescence_lines


class NoiseRecords(object):
    """
    Encapsulate a set of noise records, which can either be
    assumed continuous or arbitrarily separated in time.
    """
    
    def __init__(self, filename, records_are_continuous=False):
        """
        Load a 

        If <records_are_continuous> is True, then treat all pulses as a continuous timestream.
        """
        self.__open_file(filename)
        self.continuous = records_are_continuous
        self.spectrum = None
        self.autocorrelation = None
        
     
    def __open_file(self, filename):
        """Detect the filetype and open it."""

        # For now, we have only one file type, so let's just assume it!
        MAXSEGMENTSIZE = 80000000
        self.datafile = files.LJHFile(filename, segmentsize=MAXSEGMENTSIZE)
        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples","nPresamples","nPulses", "timebase"):
            self.__dict__[attr] = self.datafile.__dict__[attr]

        for first_pnum, end_pnum, seg_num, data in self.datafile.iter_segments():
            if seg_num > 0 or first_pnum>0 or end_pnum != self.nPulses:
                msg = "NoiseRecords objects can't (yet) handle multi-segment noise files.\n"+\
                    "File size %d exceeds maximum allowed segment size of %d"%(
                    self.filename, self.datafile.binary_size, MAXSEGMENTSIZE)
                raise NotImplementedError(msg)
            self.data = data

    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = NoiseRecords(self.filename)
        c.__dict__.update( self.__dict__ )
        c.datafile = self.datafile.copy()
        return c

    
    def compute_power_spectrum(self, window=power_spectrum.hann, plot=True):
        self.spectrum = self.compute_power_spectrum_reshape(window=window, nsegments=None)
        if plot: self.plot_power_spectrum()


    def compute_power_spectrum_reshape(self, window=power_spectrum.hann, nsegments=None):
        """Compute the noise power spectrum with noise "records" reparsed into 
        <nsegments> separate records.  (If None, then self.data.shape[0] which is self.data.nPulses,
        will be used as the number of segments.)
        
        By making <nsegments> large, you improve the noise on the PSD estimates at the price of poor
        frequency resolution.  By making it small, you get good frequency resolution with worse
        uncertainty on each PSD estimate.  No free lunch, know what I mean?
        """
        
        if not self.continuous and nsegments is not None:
            raise ValueError("This NoiseRecords object does not have continuous noise records, so it can't be resegmented.")
        
        if self.continuous and nsegments is not None:
            data = self.data.ravel()
            n=len(data)
            n = n-n%nsegments
            data=data[:n].reshape((nsegments,n/nsegments))
        else:
            data=self.data

        seg_length = data.shape[1]
        if window is None:
            window = numpy.ones(seg_length)
        else:
            window = window(seg_length)
        spectrum = power_spectrum.PowerSpectrum(seg_length/2, dt=self.timebase)
        for d in data:
            spectrum.addDataSegment(d-d.mean(), window=window)
        return spectrum


    def compute_fancy_power_spectrum(self, window=power_spectrum.hann, plot=True, nseg_choices=None):
        assert self.continuous

        n = numpy.prod(self.data.shape)
        if nseg_choices is None:
            nseg_choices = [16]
            while nseg_choices[-1]<=n/16 and nseg_choices[-1]<20000:
                nseg_choices.append(nseg_choices[-1]*8)
        print nseg_choices

        spectra = [self.compute_power_spectrum_reshape(window=window, nsegments=ns) for ns in nseg_choices]
        if plot:
            pylab.clf()
            lowest_freq = numpy.array([1./(sp.dt*sp.m2) for sp in spectra])
            
            start_freq=0.0
            for i,sp in enumerate(spectra):
                x,y = sp.frequencies(), sp.spectrum()
                if i==len(spectra)-1:
                    good = x>=start_freq
                else:
                    good = numpy.logical_and(x>=start_freq, x<4*lowest_freq[i+1])
                    start_freq = 1*lowest_freq[i+1]
                pylab.loglog(x[good],y[good],'-')
    
    def plot_power_spectrum(self, axis=None, scale=1.0, sqrt_psd=False, **kwarg):
        """
        Plot the power spectrum of this noise record.
        
        <axis>     Which pylab.Axes object to plot on.  If none, clear the figure and plot there.
        <scale>    Scale all raw units by this number to convert counts to physical
        <sqrt_psd> Whether to take the sqrt(PSD) for plotting.  Default is no sqrt
        """
        if self.spectrum is None:
            self.compute_power_spectrum(plot=False)
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        spec = self.spectrum
        yvalue = spec.spectrum()[1:] * (scale**2)
        if sqrt_psd: 
            yvalue = numpy.sqrt(yvalue)
        axis.plot(spec.frequencies()[1:], yvalue, **kwarg)
        pylab.loglog()
        axis.grid()
        axis.set_xlim([10,3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title("Noise power spectrum for %s"%self.filename)

        
    def compute_autocorrelation(self, n_lags=None, n_data=None, plot=True):
        """
        Compute the autocorrelation averaged across all "pulses" in the file.
        """
        
        if self.continuous:
            if n_data is None:
                n_data = self.nSamples*self.nPulses
            if n_lags is None:
                n_lags = n_data
            if n_lags > n_data:
                n_lags = n_data
            
            def padded_length(n):
                """Return a sensible number in the range [n, 2n] which is not too
                much larger than n, yet is good for FFTs.
                That is, choose (1, 3, or 5)*(a power of two), whichever is smallest
                """
                pow2 = numpy.round(2**numpy.ceil(numpy.log2(n)))
                if n==pow2: return n
                elif n>0.75*pow2: return pow2
                elif n>0.625*pow2: return numpy.round(0.75*pow2)
                else: return numpy.round(0.625*pow2)
            
        
            # When there are 10 million data points and only 10,000 lags wanted,
            # it's hugely inefficient to compute the full autocorrelation, especially
            # in memory.  Instead, compute it on chunks 7* the length of the desired
            # correlation, and average.
            CHUNK_MULTIPLE=31
            if n_data >= (1+CHUNK_MULTIPLE)*n_lags:
                # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
                # unwanted wraparound in the autocorrelation.
                # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
                chunksize=CHUNK_MULTIPLE*n_lags
                padded_data = numpy.zeros(padded_length(n_lags+chunksize), dtype=numpy.float)
                ac = numpy.zeros(n_lags, dtype=numpy.float)
                
                data_used=0
                entries = 0.0
                data_mean = self.data.mean()
                data = self.data.ravel()-data_mean
                t0=time.time()
                
                # Notice that the following loop might ignore the last data values, up to as many
                # as (chunksize-1) values, unless the data are an exact multiple of chunksize.
                while data_used+chunksize <= n_data:
                    padded_data[:chunksize] = data[data_used:data_used+chunksize]
                    padded_data[chunksize:] = 0.0
                    data_used += chunksize
                    
                    ft = numpy.fft.rfft(padded_data)
                    ft[0] = 0 # this redundantly removes the mean of the data set
                    power = (ft*ft.conj()).real
                    acsum = numpy.fft.irfft(power)
                    ac += acsum[:n_lags] 
                    entries += 1.0
                    
                    # A message for the first time through:
                    if data_used==chunksize:
                        dt = time.time()-t0
                        print 'Analyzed %d samples in %.2f sec'%(data_used, dt)
                        print '....expect total time %.2f sec'%(dt*n_data/chunksize)

                ac /= entries
                ac /= (numpy.arange(chunksize, chunksize-n_lags+0.5, -1.0, dtype=numpy.float))
                    
            # compute the full autocorrelation                
            else:
                padded_data = numpy.zeros(padded_length(n_lags+n_data), dtype=numpy.float)
                padded_data[:n_data] = numpy.array(self.data.ravel())[:n_data] - self.data.mean()
                padded_data[n_data:] = 0.0
                
                ft = numpy.fft.rfft(padded_data)
                del padded_data
                ft[0] = 0 # this redundantly removes the mean of the data set
                ft *= ft.conj()
                ft = ft.real
                acsum = numpy.fft.irfft(ft)
                del ft
                ac = acsum[:n_lags+1] / (n_data-numpy.arange(n_lags+1.0))
                del acsum
         
        else:
            ac=numpy.zeros(self.nSamples, dtype=numpy.float)
            for i in range(self.nPulses):
                data = 1.0*self.data[i,:]
                data -= data.mean()
                ac += numpy.correlate(data,data,'full')[self.nSamples-1:]
            ac /= self.nPulses
            ac /= self.nSamples - numpy.arange(self.nSamples, dtype=numpy.float)
        self.autocorrelation = ac
        
        if plot: self.plot_autocorrelation()
        
    def plot_autocorrelation(self, axis=None):
        if self.autocorrelation is None:
            print "Autocorrelation will be computed first"
            self.compute_autocorrelation(plot=False)
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        t = self.timebase * 1e3 * numpy.arange(len(self.autocorrelation))
        axis.plot(t,self.autocorrelation)
        axis.plot([0],[self.autocorrelation[0]],'o')
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel("Autocorrelation (counts$^2$)")
        

    
class PulseRecords(object):
    """
    Encapsulate a set of data containing multiple triggered pulse traces.
    The pulses should not be noise records."""
    
    
    def __init__(self, filename, file_format=None):
        self.nSamples = 0
        self.nPresamples = 0
        self.nPulses = 0
        self.segmentsize = 0
        self.timebase = None
        
        self.p_timestamp = None
        self.p_peak_index = None
        self.p_peak_value = None
        self.p_peak_time = None
        self.p_min_value = None
        self.p_pretrig_mean = None
        self.p_pretrig_rms = None
        self.p_pulse_average = None
        self.p_rise_time = None
        self.p_max_posttrig_deriv = None
        
        self.cuts = None
        self.bad = None
        self.good = None

        self.__open_file(filename, file_format = file_format)
#        self.__setup_vectors()

    def __open_file(self, filename, file_format=None):
        """Detect the filetype and open it."""

        ALLOWED_TYPES=("ljh","root")
        if file_format is None:
            if filename.endswith("root"):
                file_format = "root"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s"%ALLOWED_TYPES)

        if file_format == "ljh":
            self.datafile = files.LJHFile(filename)
        elif file_format == "root":
            self.datafile = files.LANLFile(filename)
        else:
            raise RuntimeError("It is a programming error to get here")
        
        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples","nPresamples","nPulses", "timebase", 
                     "n_segments", "pulses_per_seg"):
            self.__dict__[attr] = self.datafile.__dict__[attr]


#    def __setup_vectors(self):
#        """Given the number of pulses, build arrays to hold the relevant facts 
#        about each pulse in memory."""
#        
#        assert self.nPulses > 0
#        self.p_timestamp = numpy.zeros(self.nPulses, dtype=numpy.int32)
#        self.p_peak_index = numpy.zeros(self.nPulses, dtype=numpy.uint16)
#        self.p_peak_value = numpy.zeros(self.nPulses, dtype=numpy.uint16)
#        self.p_peak_time = numpy.zeros(self.nPulses, dtype=numpy.float)
#        self.p_min_value = numpy.zeros(self.nPulses, dtype=numpy.uint16)
#        self.p_pretrig_mean = numpy.zeros(self.nPulses, dtype=numpy.float)
#        self.p_pretrig_rms = numpy.zeros(self.nPulses, dtype=numpy.float)
#        self.p_pulse_average = numpy.zeros(self.nPulses, dtype=numpy.float)
#        self.p_rise_time = numpy.zeros(self.nPulses, dtype=numpy.float)
#        self.p_max_posttrig_deriv = numpy.zeros(self.nPulses, dtype=numpy.float)
#        
#        self.cuts = Cuts(self.nPulses)
#        self.good = self.cuts.good()
#        self.bad = self.cuts.bad()


    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)

    
    def read_segment(self, segment_num):
        """Read the requested segment of the raw data file and return  (first,end,data)
        meaning: the first record number, 1 more than the last record number,
        and the nPulse x nSamples array."""
        if segment_num >= self.n_segments:
            return -1,-1
        first_pnum, end_pnum, data = self.datafile.read_segment(segment_num)
        self.data = data
        return first_pnum, end_pnum
    
    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = PulseRecords(self.filename)
        c.__dict__.update( self.__dict__ )
        c.datafile = self.datafile.copy()
        return c
        

    def serialize(self, serialfile):
        """Store object in a pickle file"""
        fp = open(serialfile, "wb")
        pickle.dump(self, fp, protocol=2)
        fp.close()


        
##########################################################################################

def estimateRiseTime(ts, dt=1.0, nPretrig=0):
    """Compute the rise time of timeseries <ts>, where the time steps are <dt>.
    If <nPretrig> >= 4, then the samples ts[:nPretrig] are averaged to estimate
    the baseline.  Otherwise, the minimum of ts is assumed to be the baseline.
    
    Specifically, take the first and last of the rising points in the range of 
    10% to 90% of the peak value, interpolate a line between the two, and use its
    slope to find the time to rise from 0 to the peak.
    
    See also fitExponentialRiseTime for a more traditional (and computationally
    expensive) definition.
    """
    MINTHRESH, MAXTHRESH = 0.1, 0.9
    
    
    if nPretrig >= 4:
        baseline_value = ts[0:nPretrig-5].mean()
    else:
        baseline_value = ts.min()
        nPretrig = 0
    value_at_peak = ts.max() - baseline_value
    idxpk = ts.argmax()

    try:
        rising_data = (ts[nPretrig:idxpk+1] - baseline_value) / value_at_peak
        idx = numpy.arange(len(rising_data), dtype=numpy.int)
        last_idx = idx[rising_data<MAXTHRESH].max()
        first_idx = idx[rising_data>MINTHRESH].min()
        y_diff = rising_data[last_idx]-rising_data[first_idx]
        if y_diff <= 0:
            return -9.9e-6
        time_diff = dt*(last_idx-first_idx)
        return time_diff / y_diff
    except ValueError:
        return -9.9e-6



#def fitExponentialRiseTime(ts, dt=1.0, nPretrig=0):
#    """Compute the rise time of timeseries <ts>, where the time steps are <dt>.
#    If <nPretrig> >0, then the samples ts[:nPretrig] are averaged to estimate
#    the baseline.  Otherwise, the minimum of ts is assumed to be the baseline.
#    
#    Specifically, fit an exponential to the rising points in the range of 10% to 90% of the peak
#    value and use its slope to compute the time to rise from 0 to the peak.
#    
#    See also estimateRiseTime
#    """
#    MINTHRESH, MAXTHRESH = 0.1, 0.9
#    
#    if nPretrig > 0:
#        baseline_value = ts[0:nPretrig].mean()
#    else:
#        baseline_value = ts.min()
#        nPretrig = 0
#    valpk = ts.max() - baseline_value
#    idxpk = ts.argmax()
#    useable = ts[nPretrig:idxpk] - baseline_value
#    idx = numpy.arange(len(useable))
#    last_idx = idx[useable<MAXTHRESH*valpk].max()
#    first_idx = idx[useable>MINTHRESH*valpk].min()
#    if (last_idx-first_idx) < 4:
#        raise ValueError("Cannot make an exponential fit to only %d points!"%
#                         (last_idx-first_idx+1))
#    
#    x, y = idx[first_idx:last_idx+1], useable[first_idx:last_idx+1]
#    
#    fitfunc = lambda p, x: p[0]*numpy.exp(-x/p[1])+p[2]
#    errfunc = lambda p, x, y: fitfunc(p, x) - y
#    
#    p0 = -useable.max(), 0.6*(x[-1]-x[0]), useable.max()  
#    p, _stat = scipy.optimize.leastsq(errfunc, p0, args=(x,y))
#    return dt * p[1]



def compute_max_deriv(ts, return_index_too=False):
    """Compute the maximum derivative in timeseries <ts>.
    
    Return the value of the maximum derivative (units of <ts units> per sample).
    
    If <return_index_too> then return a tuple with the derivative AND the index from
    <ts> where the derivative is highest.
    
    We estimate it by Savitzky-Golay filtering (with 1 point before/3 points after
    the point in question and fitting polynomial of order 3).  Find the right general
    area by first doing a simple difference."""
    
    ts = 1.0*ts # so that there are no unsigned ints!
    rough_imax = 1 + (ts[2:]-ts[:-2]).argmax()
    
    first,end = rough_imax-12, rough_imax+12
    if first<0: first=0
    if end>len(ts): end=len(ts)
    
    # Can't even do convolution if there aren't enough data.
    if end-first < 9: 
        return 0.5*(ts[rough_imax+1]-ts[rough_imax-1])
    
    # This filter is the Savitzky-Golay filter of n_L=1, n_R=3 and M=3, to use the
    # language of Numerical Recipes 3rd edition.  It amounts to least-squares fitting
    # of an M=3rd order polynomial to the five points [-1,+3] and
    # finding the slope of the polynomial at 0.
    filter = numpy.array([ -0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905,   ])[::-1]
    conv = numpy.convolve(ts[first:end], filter, mode='valid')
    
    if return_index_too:
        return first + 2 + conv.argmax() # This would be the index.
    return conv.max()



class Cuts(object):
    "Object to hold a mask for each trigger."
    
    def __init__(self, n):
        "Create an object to hold n masks of 32 bits each"
        self._mask = numpy.zeros( n, dtype=numpy.int32 )
        
    def cut(self, cutnum, mask):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        bitval = 1<<cutnum
        self._mask[mask] |= bitval

    def clearCut(self, cutnum):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        bitmask = ~(1<<cutnum)
        self._mask &= bitmask
        
    def good(self):
        return numpy.logical_not(self._mask)
    
    def bad(self):
        return self._mask != 0
    
    def isCut(self, cutnum=None):
        if cutnum is None: return self.bad()
        return (self._mask & (1<<cutnum)) != 0
    
    def isUncut(self, cutnum=None):
        if cutnum is None: return self.good()
        return (self._mask & (1<<cutnum)) == 0
    
    def nCut(self):
        return (self._mask != 0).sum()
    
    def nUncut(self):
        return (self._mask == 0).sum()

    def __repr__(self):
        return "Cuts(%d)"%len(self._mask)
    
    def __str__(self):
        return ("Cuts(%d) with %d cut and %d uncut"%(len(self._mask), self.nCut(), self.nUncut()))



class Filter(object):
    """A set of optimal filters based on a single signal and noise set."""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None, 
                 fmax=None, f_3db=None, sample_time=None, shorten=0):
        """
        Create a set of filters under various assumptions and for various purposes.
        
        <avg_signal>     The average signal shape.  Filters will be rescaled so that the output
                         upon putting this signal into the filter equals the *peak value* of this
                         filter (that is, peak value relative to the baseline level).
        <n_pretrigger>   The number of leading samples in the average signal that are considered
                         to be pre-trigger samples.  The avg_signal in this section is replaced by
                         its constant averaged value before creating filters.  Also, one filter
                         (filt_baseline_pretrig) is designed to infer the baseline using only
                         <n_pretrigger> samples at the start of a record.
        <noise_psd>      The noise power spectral density.  If None, then filt_fourier won't be
                         computed.  If not None, then it must be of length (2N+1), where N is the
                         length of <avg_signal>, and its values are assumed to cover the non-negative
                         frequencies from 0, 1/Delta, 2/Delta,.... up to the Nyquist frequency.
        <noise_autocorr> The autocorrelation function of the noise, where the lag spacing is
                         assumed to be the same as the sample period of <avg_signal>.  If None,
                         then several filters won't be computed.  (One of <noise_psd> or 
                         <noise_autocorr> must be a valid array.)
        <fmax>           The strict maximum frequency to be passed in all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
        <f_3db>          The 3 dB point for a one-pole low-pass filter to be applied to all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
                         Either or both of <fmax> and <f_3db> are allowed.
        <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                         This must be given if <fmax> or <f_3db> are ever to be used.
        <shorten>        The time-domain filters should be shortened by removing this many
                         samples from each end.  (Do this for convenience of convolution over
                         multiple lags.)
        """
        self.sample_time = sample_time
        self.shorten = shorten
        pre_avg = avg_signal[:n_pretrigger].mean()
        
        # If signal is negative-going, 
        is_negative =  (avg_signal.min()-pre_avg)/(avg_signal.max()-pre_avg) < -1
        if is_negative:
            self.peak_signal = avg_signal.min() - pre_avg
        else:
            self.peak_signal = avg_signal.max() - pre_avg

        # self.avg_signal is normalized to have unit peak
        self.avg_signal = (avg_signal - pre_avg) / self.peak_signal
        self.avg_signal[:n_pretrigger] = 0.0
        
        self.n_pretrigger = n_pretrigger
        if noise_psd is None:
            self.noise_psd = None
        else:
            self.noise_psd = numpy.array(noise_psd)
        if noise_autocorr is None:
            self.noise_autocorr = None
        else:
            self.noise_autocorr = numpy.array(noise_autocorr)
        if noise_psd is None and noise_autocorr is None:
            raise ValueError("Filter must have noise_psd or noise_autocorr arguments (or both)")
        
        self.compute(fmax=fmax, f_3db=f_3db)


    def normalize_filter(self, q): 
        "Rescale filter <q> so that it gives unit response to self.avg_signal"
        if len(q) == len(self.avg_signal):
            q *= 1 / numpy.dot(q, self.avg_signal)
        else:  
#                print "scaling by 1/%f"%numpy.dot(q, self.avg_signal[self.shorten:-self.shorten])
#                print self.peak_signal, q
            q *= 1 / numpy.dot(q, self.avg_signal[self.shorten:-self.shorten]) 


    def _compute_fourier_filter(self, fmax=None, f_3db=None):
        "Compute the Fourier-domain filter"
        if self.noise_psd is None: return
        
        # Careful: let's be sure that the Fourier domain filter is done consistently in Filter and
        # its child classes.
        
        n = len(self.noise_psd)
#        window = power_spectrum.hamming(2*(n-1-self.shorten))
        window = 1.0

        if self.shorten>0:
            sig_ft = numpy.fft.rfft(self.avg_signal[self.shorten:-self.shorten]*window)
        else:
            sig_ft = numpy.fft.rfft(self.avg_signal * window)

        if len(sig_ft) != n-self.shorten:
            raise ValueError("signal real DFT and noise PSD are not the same length (%d and %d)"
                             %(len(sig_ft), n))
            
        # Careful with PSD: "shorten" it by converting into a real space autocorrelation, 
        # truncating the middle, and going back to Fourier space
        if self.shorten>0:
            noise_autocorr = numpy.fft.irfft(self.noise_psd)
            noise_autocorr = numpy.hstack((noise_autocorr[:n-self.shorten-1], noise_autocorr[-n+self.shorten:]))
            noise_psd = numpy.fft.rfft(noise_autocorr)
        else:
            noise_psd = self.noise_psd
        sig_ft_weighted = sig_ft/noise_psd
        
        # Band-limit
        if fmax is not None or f_3db is not None:
            freq = numpy.arange(0,n-self.shorten,dtype=numpy.float)*0.5/((n-1)*self.sample_time)
            if fmax is not None:
                sig_ft_weighted[freq>fmax] = 0.0
            if f_3db is not None:
                sig_ft_weighted /= (1+(freq*1.0/f_3db)**2)

        # Compute both the normal (DC-free) and the full (with DC) filters.
        self.filt_fourierfull = numpy.fft.irfft(sig_ft_weighted)/window
        sig_ft_weighted[0] = 0.0
        self.filt_fourier = numpy.fft.irfft(sig_ft_weighted)/window
        self.normalize_filter(self.filt_fourierfull)
        self.normalize_filter(self.filt_fourier)
        
        # How we compute the uncertainty depends on whether there's a noise autocorrelation result
        if self.noise_autocorr is None:
            noise_ft_squared = (len(self.noise_psd)-1)/self.sample_time * self.noise_psd
            kappa = (numpy.abs(sig_ft*self.peak_signal)**2/noise_ft_squared)[:].sum()
            self.variances['fourierfull'] = 1./kappa
            
            kappa = (numpy.abs(sig_ft*self.peak_signal)**2/noise_ft_squared)[1:].sum()
            self.variances['fourier'] = 1./kappa
        else:
            ac = self.noise_autocorr[:len(self.filt_fourier)].copy()
            self.variances['fourier'] = self.bracketR(self.filt_fourier, ac)/self.peak_signal**2
            self.variances['fourierfull'] = self.bracketR(self.filt_fourierfull, ac)/self.peak_signal**2
#        print 'Fourier filter done.  Variance: ',self.variances['fourier'], 'V/dV: ',self.variances['fourier']**(-0.5)/2.35482


    def compute(self, fmax=None, f_3db=None, use_toeplitz_solver=True):
        """
        Compute a set of filters.  This is called once on construction, but you can call it
        again if you want to change the frequency cutoff or f_3db rolloff point.
        """

        self.fmax=fmax
        self.f_3db=f_3db
        self.variances={}
        self._compute_fourier_filter(fmax=fmax, f_3db=f_3db)

        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2*self.shorten
            assert len(self.noise_autocorr) >= n
            if self.shorten>0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            
            noise_corr = self.noise_autocorr[:n]/self.peak_signal**2
            if use_toeplitz_solver:
                ts = utilities.ToeplitzSolver(noise_corr, symmetric=True)
                Rinv_sig = ts(avg_signal)
                Rinv_1 = ts(numpy.ones(n))
            else:
                if n>6000: raise ValueError("Not allowed to use generic solver for vectors longer than 6000, because it's slow-ass.")
                R =  scipy.linalg.toeplitz(noise_corr)
                Rinv_sig = numpy.linalg.solve(R, avg_signal)
                Rinv_1 = numpy.linalg.solve(R, numpy.ones(n))
            
            self.filt_noconst = Rinv_1.sum()*Rinv_sig - Rinv_sig.sum()*Rinv_1

            # Band-limit
            if fmax is not None or f_3db is not None:
                sig_ft = numpy.fft.rfft(self.filt_noconst)
                freq = numpy.arange(0,n/2+1,dtype=numpy.float)*0.5/self.sample_time/(n/2)
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1.+(1.0*freq/f_3db)**2)
                self.filt_noconst = numpy.fft.irfft(sig_ft)

            self.normalize_filter(self.filt_noconst)

            self.filt_baseline = numpy.dot(avg_signal, Rinv_sig)*Rinv_1 - Rinv_sig.sum()*Rinv_sig
            self.filt_baseline /=  self.filt_baseline.sum()
            
            Rpretrig = scipy.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger]/self.peak_signal**2)
            self.filt_baseline_pretrig = numpy.linalg.solve(Rpretrig, numpy.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()

            self.variances['noconst'] = self.bracketR(self.filt_noconst, noise_corr) 
            self.variances['baseline'] = self.bracketR(self.filt_baseline, noise_corr)
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig, Rpretrig[0,:])

                
    def bracketR(self, q, noise):
        """Return the dot product (q^T R q) for vector <q> and matrix R constructed from
        the vector <noise> by R_ij = noise_|i-j|.  We don't want to construct the full matrix
        R because for records as long as 10,000 samples, the matrix will consist of 10^8 floats
        (800 MB of memory)."""
        
        if len(noise) < len(q):
            raise ValueError("Vector q (length %d) cannot be longer than the noise (length %d)"%
                             (len(q),len(noise)))
        n=len(q)
        r = numpy.zeros(2*n-1, dtype=numpy.float)
        r[n-1:] = noise[:n]
        r[n-1::-1] = noise[:n]
        dot = 0.0
        for i in range(n):
            dot += q[i]*numpy.dot(r[n-i-1:2*n-i-1], q)
        return dot
    
            
    def plot(self, axes=None):
        if axes is None:
            pylab.clf()
            axis1 = pylab.subplot(211)
            axis2 = pylab.subplot(212)
        else:
            axis1,axis2 = axes
        try:
            axis1.plot(self.filt_noconst,color='red')
            axis2.plot(self.filt_baseline,color='purple')
            axis2.plot(self.filt_baseline_pretrig,color='blue')
        except AttributeError: pass
        try:
            axis1.plot(self.filt_fourier,color='gold')
        except AttributeError: pass


    def report(self, filters=None):
        """Report on V/dV for all filters
        
        <filters>   Either the name of one filter or a sequence of names.  If not given, then all filters
                    not starting with "baseline" will be reported
        """
        
        # Handle <filters> is a single string --> convert to tuple of 1 string
        if isinstance(filters,str):
            filters=(filters,)
            
        # Handle default <filters> not given.
        if filters is None:
            filters = list(self.variances.keys())
            for f in self.variances:
                if f.startswith("baseline"):
                    filters.remove(f)
            filters.sort()

        for f in filters:
            try:
                var = self.variances[f]
                v_dv = var**(-.5) / numpy.sqrt(8*numpy.log(2))
                print "%-20s  %10.3f  %10.4e"%(f, v_dv, var)
            except KeyError:
                print "%-20s not known"%f



class ExperimentalFilter(Filter):
    """Compute and all filters for pulses given an <avgpulse>, the
    <noise_autocorr>, and an expected time constant <tau> for decaying exponentials.
    Shorten the filters w.r.t. the avgpulse function by <shorten> samples on each end.
    
    CAUTION: THESE ARE EXPERIMENTAL!  Don't use yet if you don't know what you're doing!"""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None, 
                 fmax=None, f_3db=None, sample_time=None, shorten=0, tau=2.0):
        """
        Create a set of filters under various assumptions and for various purposes.
        
        <avg_signal>     The average signal shape.  Filters will be rescaled so that the output
                         upon putting this signal into the filter equals the *peak value* of this
                         filter (that is, peak value relative to the baseline level).
        <n_pretrigger>   The number of leading samples in the average signal that are considered
                         to be pre-trigger samples.  The avg_signal in this section is replaced by
                         its constant averaged value before creating filters.  Also, one filter
                         (filt_baseline_pretrig) is designed to infer the baseline using only
                         <n_pretrigger> samples at the start of a record.
        <noise_psd>      The noise power spectral density.  If None, then filt_fourier won't be
                         computed.  If not None, then it must be of length (2N+1), where N is the
                         length of <avg_signal>, and its values are assumed to cover the non-negative
                         frequencies from 0, 1/Delta, 2/Delta,.... up to the Nyquist frequency.
        <noise_autocorr> The autocorrelation function of the noise, where the lag spacing is
                         assumed to be the same as the sample period of <avg_signal>.  If None,
                         then several filters won't be computed.  (One of <noise_psd> or 
                         <noise_autocorr> must be a valid array.)
        <fmax>           The strict maximum frequency to be passed in all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
        <f_3db>          The 3 dB point for a one-pole low-pass filter to be applied to all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
                         Either or both of <fmax> and <f_3db> are allowed.
        <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                         This must be given if <fmax> or <f_3db> are ever to be used.
        <shorten>        The time-domain filters should be shortened by removing this many
                         samples from each end.  (Do this for convenience of convolution over
                         multiple lags.)
        <tau>            Time constant of exponential to filter out (in milliseconds)
        """
        
        self.tau = tau # in milliseconds
        super(self.__class__, self).__init__(avg_signal, n_pretrigger, noise_psd,
                                             noise_autocorr, fmax, f_3db, sample_time, shorten)

        
    
    def compute(self, fmax=None, f_3db=None):
        """
        Compute a set of filters.  This is called once on construction, but you can call it
        again if you want to change the frequency cutoff or rolloff points.
        
        Set is:
        filt_fourier    Fourier filter for signals
        filt_full       Alpert basic filter
        filt_noconst    Alpert filter insensitive to constants
        filt_noexp      Alpert filter insensitive to exp(-t/tau)
        filt_noexpcon   Alpert filter insensitive to exp(-t/tau) and to constants
        filt_noslope    Alpert filter insensitive to slopes
        filt_nopoly1    Alpert filter insensitive to Chebyshev polynomials order 0 to 1
        filt_nopoly2    Alpert filter insensitive to Chebyshev polynomials order 0 to 2
        filt_nopoly3    Alpert filter insensitive to Chebyshev polynomials order 0 to 3
        """

        self.fmax=fmax
        self.f_3db=f_3db
        self.variances={}
        
        self._compute_fourier_filter(fmax=fmax, f_3db=f_3db)
        
        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2*self.shorten
            if self.shorten>0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            assert len(self.noise_autocorr) >= n
            
            expx = numpy.arange(n, dtype=numpy.float)*self.sample_time*1e3 # in ms
            chebyx = numpy.linspace(-1, 1, n)
            
            R = self.noise_autocorr[:n]/self.peak_signal**2 # A *vector*, not a matrix
            ts = utilities.ToeplitzSolver(R, symmetric=True)
            
            unit = numpy.ones(n)
            exp  = numpy.exp(-expx/self.tau)
            cht1 = scipy.special.chebyt(1)(chebyx)
            cht2 = scipy.special.chebyt(2)(chebyx)
            cht3 = scipy.special.chebyt(3)(chebyx)
            
            Rinv_sig  = ts(avg_signal)
            Rinv_unit = ts(unit)
            Rinv_exp  = ts(exp)
            Rinv_cht1 = ts(cht1)
            Rinv_cht2 = ts(cht2)
            Rinv_cht3 = ts(cht3)
            
            # Band-limit
            def band_limit(vector, fmax, f_3db):
                sig_ft = numpy.fft.rfft(vector)
                freq = numpy.fft.fftfreq(n, d=self.sample_time) 
                freq=freq[:n/2+1]
                freq[-1] *= -1
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1+(freq/f_3db)**2)
                vector[:] = numpy.fft.irfft(sig_ft)
                
            if fmax is not None or f_3db is not None:
                for vector in Rinv_sig, Rinv_unit, Rinv_exp, Rinv_cht1, Rinv_cht2, Rinv_cht3:
                    band_limit(vector, fmax, f_3db)

            
            orthogonalities={
                'filt_full':(),
                'filt_noconst' :('unit',),
                'filt_noexp'   :('exp',),
                'filt_noexpcon':('unit', 'exp'),
                'filt_noslope' :('cht1',),
                'filt_nopoly1' :('unit', 'cht1'),
                'filt_nopoly2' :('unit', 'cht1', 'cht2'),
                'filt_nopoly3' :('unit', 'cht1', 'cht2', 'cht3'),
                }
            
            pylab.clf()
            pylab.plot(self.filt_fourier, color='gold',label='Fourier')
#            for shortname in ('full','noconst','noexpcon','nopoly1'):
            for shortname in ('full','noconst','noexp','noexpcon','noslope','nopoly1','nopoly2','nopoly3'):
#            for shortname in ('noexp','noconst','noexpcon','nopoly1'):
                name = 'filt_%s'%shortname
                orthnames = orthogonalities[name]
                filt = Rinv_sig
                
                N_orth = len(orthnames) # To how many vectors are we orthgonal?
                if N_orth > 0:
                    u = numpy.vstack((Rinv_sig, [eval('Rinv_%s'%v) for v in orthnames]))
                else:
                    u = Rinv_sig.reshape((1,n))
                M = numpy.zeros((1+N_orth,1+N_orth), dtype=numpy.float)
                for i in range(1+N_orth):
                    M[0,i] = numpy.dot(avg_signal, u[i,:])
                    for j in range(1,1+N_orth):
                        M[j,i] = numpy.dot(eval(orthnames[j-1]), u[i,:])
                Minv = numpy.linalg.inv(M)
                weights = Minv[:,0]

                filt = numpy.dot(weights, u)
                filt = u[0,:]*weights[0]
                for i in range(1,1+N_orth):
                    filt += u[i,:]*weights[i]
                
                
                self.normalize_filter(filt)
                self.__dict__[name] = filt
                
                print '%15s'%name,
                pylab.plot(filt, label=name)
                for v in (avg_signal,numpy.ones(n),numpy.exp(-expx/self.tau),scipy.special.chebyt(1)(chebyx),
                          scipy.special.chebyt(2)(chebyx)):
                    print '%10.5f '%numpy.dot(v,filt),
                    
                self.variances[shortname] = self.bracketR(filt, R)
                print 'Res=%6.3f eV = %.5f'%(5898.801*numpy.sqrt(8*numpy.log(2))*self.variances[shortname]**(.5), (self.variances[shortname]/self.variances['full'])**.5)
            pylab.legend()

            self.filt_baseline = numpy.dot(avg_signal, Rinv_sig)*Rinv_unit - Rinv_sig.sum()*Rinv_sig
            self.filt_baseline /=  self.filt_baseline.sum()
            self.variances['baseline'] = self.bracketR(self.filt_baseline, R)
            
            Rpretrig = scipy.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger]/self.peak_signal**2)
            self.filt_baseline_pretrig = numpy.linalg.solve(Rpretrig, numpy.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig, R[:self.n_pretrigger])

            if self.noise_psd is not None:
                r =  self.noise_autocorr[:len(self.filt_fourier)]/self.peak_signal**2
                self.variances['fourier'] = self.bracketR(self.filt_fourier, r)


            
    def plot(self, axes=None):
        if axes is None:
            pylab.clf()
            axis1 = pylab.subplot(211)
            axis2 = pylab.subplot(212)
        else:
            axis1,axis2 = axes
        try:
            axis1.plot(self.filt_noconst,color='red')
            axis2.plot(self.filt_baseline,color='purple')
            axis2.plot(self.filt_baseline_pretrig,color='blue')
        except AttributeError: pass
        try:
            axis1.plot(self.filt_fourier,color='gold')
        except AttributeError: pass


class MicrocalDataSet(object):
    """
    Represent a single microcalorimeter's PROCESSED data.
    This channel can be directly from a TDM detector, or it
    can be the demodulated result of a CDM modulation.
    """
    
    ( CUT_PRETRIG_MEAN,
      CUT_PRETRIG_RMS,
      CUT_RETRIGGER,
      CUT_BIAS_PULSE,
      CUT_RISETIME,
      CUT_UNLOCK,
      CUT_TIMESTAMP,
       ) = range(7)
    
    def __init__(self, pulserec_dict):
        """
        Pass in a dictionary (presumably that of a PulseRecords object)
        containing the expected attributes that must be copied to this
        MicrocalDataSet.
        """
        self.filters = None
        self.noise_spectrum = None
        self.noise_autocorr = None 
        self.noise_demodulated = None
        self.calibration = {'p_filt_value':energy_calibration.EnergyCalibration('p_filt_value')}

        expected_attributes=("nSamples","nPresamples","nPulses","timebase")
        for a in expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename','virtual data set')
        self.__setup_vectors()
        self.gain = 1.0
        self.pretrigger_ignore_microsec = 20 # Cut this long before trigger in computing pretrig values
        self.peak_time_microsec = 220.0   # Look for retriggers only after this time. 


    def __setup_vectors(self):
        """Given the number of pulses, build arrays to hold the relevant facts 
        about each pulse in memory."""
        
        assert self.nPulses > 0
        self.p_timestamp = numpy.zeros(self.nPulses, dtype=numpy.int32)
        self.p_peak_index = numpy.zeros(self.nPulses, dtype=numpy.uint16)
        self.p_peak_value = numpy.zeros(self.nPulses, dtype=numpy.uint16)
        self.p_peak_time = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_min_value = numpy.zeros(self.nPulses, dtype=numpy.uint16)
        self.p_pretrig_mean = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_pretrig_rms = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_pulse_average = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_rise_time = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_max_posttrig_deriv = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_filt_phase = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_filt_value = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_filt_value_phc = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_filt_value_dc = numpy.zeros(self.nPulses, dtype=numpy.float)
        self.p_energy = numpy.zeros(self.nPulses, dtype=numpy.float)
        
        self.cuts = Cuts(self.nPulses)
        self.good = self.cuts.good()
        self.bad = self.cuts.bad()


    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)
    
    def resize(self, nPulses):
        if self.nPulses < nPulses:
            raise ValueError("Can only shrink using resize(), but the requested size %d is larger than current %d"%
                             (nPulses, self.nPulses))
        self.nPulses = nPulses
        self.__setup_vectors()
    
    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = MicrocalDataSet(self.__dict__ )
        c.__dict__.update( self.__dict__ )
        for k in self.calibration.keys():
            c.calibration[k] = self.calibration[k].copy()
        return c

    
    def summarize_data(self, first, end):
        """Summarize the complete data file"""
        
        if first >= self.nPulses:
            return
        maxderiv_holdoff = int(self.peak_time_microsec*1e-6/self.timebase) # don't look for retriggers before this # of samples
        self.pretrigger_ignore_samples = int(self.pretrigger_ignore_microsec*1e-6/self.timebase)

        seg_size = end-first
        self.p_timestamp[first:end] = self.times[:seg_size]
        self.p_pretrig_mean[first:end] = self.data[:seg_size,:self.nPresamples-self.pretrigger_ignore_samples].mean(axis=1)
        self.p_pretrig_rms[first:end] = self.data[:seg_size,:self.nPresamples-self.pretrigger_ignore_samples].std(axis=1)
        self.p_peak_index[first:end] = self.data[:seg_size,:].argmax(axis=1)
        self.p_peak_value[first:end] = self.data[:seg_size,:].max(axis=1)
        self.p_min_value[first:end] = self.data[:seg_size,:].min(axis=1)
        self.p_pulse_average[first:end] = self.data[:seg_size,self.nPresamples:].mean(axis=1)
        
        # Remove the pretrigger mean from the peak value and the pulse average figures. 
        self.p_peak_value[first:end] -= self.p_pretrig_mean[first:end]
        self.p_pulse_average[first:end] -= self.p_pretrig_mean[first:end]
        # Careful: p_peak_index is unsigned, so make it signed before subtracting nPresamples:
        self.p_peak_time[first:end] = (numpy.asarray(self.p_peak_index[first:end], dtype=numpy.int)-self.nPresamples)*self.timebase

        # Compute things that have to be computed one at a time:
        for pulsenum,pulse in enumerate(self.data):
            if pulsenum>=seg_size: break
            self.p_rise_time[first+pulsenum] = estimateRiseTime(pulse, 
                                                dt=self.timebase, nPretrig = self.nPresamples)
            self.p_max_posttrig_deriv[first+pulsenum] = \
                compute_max_deriv(pulse[self.nPresamples + maxderiv_holdoff:])


    def filter_data(self, filter, first, end):
        if first >= self.nPulses:
            return None,None

        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = numpy.array((
                ( -6,24, 34,24,-6),
                (-14,-7,  0, 7,14),
                ( 10,-5,-10,-5,10)), dtype=numpy.float)/70.0
        
        assert len(filter)+4 == self.nSamples

        seg_size = min(end-first, self.data.shape[0])
        conv = numpy.zeros((5, seg_size), dtype=numpy.float)
        for i in range(5):
            if i-4 == 0:
                conv[i,:] = (filter*self.data[:seg_size,i:]).sum(axis=1)
            else:
#                print conv[i,:].shape, self.data.shape, (filter*self.data[:seg_size,i:i-4]).shape
                conv[i,:] = (filter*self.data[:seg_size,i:i-4]).sum(axis=1)
        param = numpy.dot(fit_array, conv)
        peak_x = -0.5*param[1,:]/param[2,:]
        peak_y = param[0,:] - 0.25*param[1,:]**2 / param[2,:] 
        return peak_x, peak_y

    
    def plot_summaries(self, valid='uncut', downsample=None, log=False):
        """Plot a summary of the data set, including time series and histograms of
        key pulse properties.
        
        <valid> An array of booleans self.nPulses long saying which pulses are to be plotted
                *OR* 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted 
                *OR* None, meaning that all pulses should be plotted.
                
        <downsample> To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points.
                     
        <log>  Use logarithmic y-axis on the histograms (right panels).
        """
        
        # Convert "uncut" or "cut" to array of all good or all bad data
        if isinstance(valid, str):
            if "uncut" in valid.lower():
                valid = self.cuts.good()
                print "Plotting only uncut data",
            elif "cut" in valid.lower():
                valid = self.cuts.bad()  
                print "Plotting only cut data",
            elif 'all' in valid.lower():
                valid = None
                print "Plotting all data, cut or uncut",
            else:
                raise ValueError("If valid is a string, it must contain 'all', 'uncut' or 'cut'.")
                
        if valid is not None:
            nrecs = valid.sum()
            if downsample is None:
                downsample=nrecs/10000
                if downsample < 1: downsample = 1
            hour = self.p_timestamp[valid][::downsample]/3.6e6
        else:
            nrecs = self.nPulses
            if downsample is None:
                downsample = self.nPulses / 10000
                if downsample < 1: downsample = 1
            hour = self.p_timestamp[::downsample]/3.6e6
        print " (%d records; %d in scatter plots)"%(
            nrecs,len(hour))
    
        plottables = (
            (self.p_pulse_average, 'Pulse Avg', 'purple', None),
            (self.p_pretrig_rms, 'Pretrig RMS', 'blue', [0,4000]),
            (self.p_pretrig_mean, 'Pretrig Mean', 'green', None),
            (self.p_peak_value, 'Peak value', '#88cc00',None),
            (self.p_max_posttrig_deriv, 'Max PT deriv', 'gold', [0,700]),
            (self.p_rise_time*1e3, 'Rise time (ms)', 'orange', [0,12]),
            (self.p_peak_time*1e3, 'Peak time (ms)', 'red', [-3,9])
          ) 
        
        pylab.clf()
        for i,(vect, label, color, limits) in enumerate(plottables):
            pylab.subplot(len(plottables),2,1+i*2)
            pylab.ylabel(label)
            
            if valid is not None:
                vect = vect[valid]
            
            pylab.plot(hour, vect[::downsample],',', color=color)
            pylab.subplot(len(plottables),2,2+i*2)
            if limits is None:
                in_limit = numpy.ones(len(vect), dtype=numpy.bool)
            else:
                in_limit= numpy.logical_and(vect>limits[0], vect<limits[1])
            contents, _bins, _patches = pylab.hist(vect[in_limit],200, log=log, 
                           histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                pylab.ylim(ymin = contents.min())


    def cut_parameter(self, data, allowed, cut_id):
        """Apply a cut on some per-pulse parameter.  
        
        <data>    The per-pulse parameter to cut on.  It can be an attribute of self, or it
                  can be computed from one (or more), 
                  but it must be an array of length self.nPulses
        <allowed> The cut to apply (see below).
        <cut_id>  The bit number (range [0,31]) to identify this cut.  Should be one of
                  self.CUT_* (see the set of class attributes)
        
        <allowed> is a 2-element sequence (a,b), then the cut requires a < data < b. 
        Either a or b may be None, indicating no cut."""
        
        if allowed is None: # no cut here! 
            return
        if cut_id <0 or cut_id >=32:
            raise ValueError("cut_id must be in the range [0,31]")
        
        try:
            a,b = allowed
            if a is not None:
                self.cuts.cut(cut_id, data <= a)
            if b is not None:
                self.cuts.cut(cut_id, data >= b)
        except ValueError:
            pass
    
    
    def apply_cuts(self, controls=None, clear=False):
        if clear: self.clear_cuts()
        
        if controls is None:
            controls = controller.standardControl()
    
        pretrigger_rms_cut = controls.cuts_prm['pretrigger_rms']
        pretrigger_mean_cut = controls.cuts_prm['pretrigger_mean']
        pretrigger_mean_dep_cut = controls.cuts_prm['pretrigger_mean_departure_from_median']
        peak_time_ms_cut = controls.cuts_prm['peak_time_ms']
        rise_time_ms_cut = controls.cuts_prm['rise_time_ms']
        max_posttrig_deriv_cut = controls.cuts_prm['max_posttrig_deriv']
        pulse_average_cut = controls.cuts_prm['pulse_average']
        min_value_cut = controls.cuts_prm['min_value']
        timestamp_cut = controls.cuts_prm['timestamp_ms']
        
        self.cut_parameter(self.p_pretrig_rms, pretrigger_rms_cut, self.CUT_PRETRIG_RMS)
        self.cut_parameter(self.p_pretrig_mean, pretrigger_mean_cut, self.CUT_PRETRIG_MEAN)
        self.cut_parameter(self.p_peak_time*1e3, peak_time_ms_cut, self.CUT_RISETIME)
        self.cut_parameter(self.p_rise_time*1e3, rise_time_ms_cut, self.CUT_RISETIME)
        self.cut_parameter(self.p_max_posttrig_deriv, max_posttrig_deriv_cut, self.CUT_RETRIGGER)
        self.cut_parameter(self.p_pulse_average, pulse_average_cut, self.CUT_UNLOCK)
        self.cut_parameter(self.p_min_value-self.p_pretrig_mean, min_value_cut, self.CUT_UNLOCK)
        self.cut_parameter(self.p_timestamp, timestamp_cut, self.CUT_TIMESTAMP)
        if pretrigger_mean_dep_cut is not None:
            median = numpy.median(self.p_pretrig_mean[self.cuts.good()])
            print'applying cut',pretrigger_mean_dep_cut,' around median of ',median
            self.cut_parameter(self.p_pretrig_mean-median, pretrigger_mean_dep_cut, self.CUT_PRETRIG_MEAN)
    
        
    def clear_cuts(self):
        self.cuts = Cuts(self.nPulses)
    
    
    def phase_correct(self, prange=None, times=None, plot=True):
        """Apply a correction for pulse variation with arrival phase.
        
        prange:  use only filtered values in this range for correction 
        times: if not None, use this range of p_timestamps instead of all data (units are millisec
               since server started--ugly but that's what we have to work with)
        plot:  whether to display the result
        """
        
        # Choose number and size of bins
        phase_step=.05
        nstep = int(.5+1.0/phase_step)
        phases = (0.5+numpy.arange(nstep))/nstep - 0.5
        phase_step = 1.0/nstep
        
        # Default: use the calibration to pick a prange
        if prange is None:
            calibration = self.calibration['p_filt_value']
            ph_estimate = calibration.name2ph('Mn Ka1')
            prange = numpy.array((ph_estimate*.98, ph_estimate*1.02))

        # Estimate corrections in a few different pieces
        corrections = []
        valid = self.cuts.good()
        if prange is not None:
            valid = numpy.logical_and(valid, self.p_filt_value<prange[1])
            valid = numpy.logical_and(valid, self.p_filt_value>prange[0])
        if times is not None:
            valid = numpy.logical_and(valid, self.p_timestamp<times[1])
            valid = numpy.logical_and(valid, self.p_timestamp>times[0])

        # Plot the raw filtered value vs phase
        if plot:
            pylab.clf()
            pylab.subplot(211)
            pylab.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value[valid],',',color='orange')
            pylab.xlim([-.55,.55])
            if prange is not None:
                pylab.ylim(prange)
                
        for ctr_phase in phases:
            valid_ph = numpy.logical_and(valid,
                                         numpy.abs((self.p_filt_phase - ctr_phase)%1) < phase_step*0.5)
#            print valid_ph.sum(),"   ",
            mean = self.p_filt_value[valid_ph].mean()
            median = numpy.median(self.p_filt_value[valid_ph])
            corrections.append(mean) # not obvious that mean vs median matters
            if plot:
                pylab.plot(ctr_phase, mean, 'or')
                pylab.plot(ctr_phase, median, 'vk', ms=10)
        corrections = numpy.array(corrections)
        assert numpy.isfinite(corrections).all()
        
        def model(params, phase):
            "Params are (phase of center, curvature, mean peak height)"
            phase = (phase - params[0]+.5)%1 - 0.5
            return 4*params[1]*(phase**2 - 0.125) + params[2]
        errfunc = lambda p,x,y: y-model(p,x)
        
        params = (-0.25, 0, corrections.mean())
        fitparams, _iflag = scipy.optimize.leastsq(errfunc, params, args=(self.p_filt_phase[valid], self.p_filt_value[valid]))
        phases = numpy.arange(-0.6,0.5001,.01)
        if plot: pylab.plot(phases, model(fitparams, phases), color='blue')
        
        self.phase_correction={'phase':fitparams[0],
                            'amplitude':fitparams[1],
                            'mean':fitparams[2]}
        fitparams[2] = 0
        correction = model(fitparams, self.p_filt_phase)
        self.p_filt_value_phc = self.p_filt_value - correction
        print 'RMS phase correction is: %9.3f (%6.2f parts/thousand)'%(correction.std(), 
                                            1e3*correction.std()/self.p_filt_value.mean())
        if plot:
            pylab.subplot(212)
            pylab.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value_phc[valid],',b')
            pylab.xlim([-.55,.55])
            if prange is not None:
                pylab.ylim(prange)


    def auto_drift_correct_rms(self, prange=None, times=None, plot=False, slopes=None):
        """Apply a correction for pulse variation with pretrigger mean, which we've found
        to be a pretty good indicator of drift.  Use the rms width of the Mn Kalpha line
        rather than actually fitting for the resolution.  (THIS IS THE OLD WAY TO DO IT.
        SUGGEST YOU USE self.auto_drift_correct instead....)
        
        prange:  use only filtered values in this range for correction 
        times: if not None, use this range of p_timestamps instead of all data (units are millisec
               since server started--ugly but that's what we have to work with)
        plot:  whether to display the result
        """
        if plot:
            pylab.clf()
            axis1=pylab.subplot(211)
        if self.p_filt_value_phc[0] ==0:
            self.p_filt_value_phc = self.p_filt_value.copy()
        
        # Default: use the calibration to pick a prange
        if prange is None:
            calibration = self.calibration['p_filt_value']
            ph_estimate = calibration.name2ph('Mn Ka1')
            prange = numpy.array((ph_estimate*.99, ph_estimate*1.01))
        
        range_ctr = 0.5*(prange[0]+prange[1])
        half_range = numpy.abs(range_ctr-prange[0])
        valid = numpy.logical_and(self.cuts.good(), numpy.abs(self.p_filt_value_phc-range_ctr)<half_range)
        if times is not None:
            valid = numpy.logical_and(valid, self.p_timestamp<times[1])
            valid = numpy.logical_and(valid, self.p_timestamp>times[0])

        data = self.p_filt_value_phc[valid]
        corrector = self.p_pretrig_mean[valid]
        mean_pretrig_mean = corrector.mean()
        corrector -= mean_pretrig_mean
        if slopes is None: slopes = numpy.arange(-.2,.9,.05)
        rms_widths=[]
        for sl in slopes:
            rms = (data+corrector*sl).std()
            rms_widths.append(rms)
#            print "%6.3f %7.2f"%(sl,rms)
            if plot: pylab.plot(sl,rms,'bo')
        poly_coef = scipy.polyfit(slopes, rms_widths, 2)
        best_slope = -0.5*poly_coef[1]/poly_coef[0]
        print "Drift correction requires slope %6.3f"%best_slope
        self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
        
        self.calibration['p_filt_value_dc'] = energy_calibration.EnergyCalibration('p_filt_value_dc')
        
        if plot:
            pylab.subplot(212)
            pylab.plot(corrector, data, ',')
            xlim = pylab.xlim()
            c = numpy.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
            pylab.plot(c, -c*best_slope + data.mean(),color='green')
            pylab.ylim(prange)
            axis1.plot(slopes, numpy.poly1d(poly_coef)(slopes),color='red')             


    def auto_drift_correct(self, prange=None, times=None, plot=False, slopes=None):
        """Apply a correction for pulse variation with pretrigger mean.
        This attempts to replace the previous version by using a fit to the
        Mn K alpha complex
        
        prange:  use only filtered values in this range for correction 
        times: if not None, use this range of p_timestamps instead of all data (units are millisec
               since server started--ugly but that's what we have to work with)
        plot:  whether to display the result
        """
        if plot:
            pylab.clf()
            axis1=pylab.subplot(211)
        if self.p_filt_value_phc[0] ==0:
            self.p_filt_value_phc = self.p_filt_value.copy()
        
        # Default: use the calibration to pick a prange
        if prange is None:
            calibration = self.calibration['p_filt_value']
            ph_estimate = calibration.name2ph('Mn Ka1')
            prange = numpy.array((ph_estimate*.99, ph_estimate*1.01))
        
        range_ctr = 0.5*(prange[0]+prange[1])
        half_range = numpy.abs(range_ctr-prange[0])
        valid = numpy.logical_and(self.cuts.good(), numpy.abs(self.p_filt_value_phc-range_ctr)<half_range)
        if times is not None:
            valid = numpy.logical_and(valid, self.p_timestamp<times[1])
            valid = numpy.logical_and(valid, self.p_timestamp>times[0])

        data = self.p_filt_value_phc[valid]
        corrector = self.p_pretrig_mean[valid]
        mean_pretrig_mean = corrector.mean()
        corrector -= mean_pretrig_mean
        if slopes is None: slopes = numpy.arange(0,1.,.09)
        
        fit_resolutions=[]
        for sl in slopes:
            self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*sl
            params,_covar = self.fit_spectral_line(prange=prange, times=times, plot=False,
                                                   type='dc', line='MnKAlpha', verbose=False)
#            print "%5.1f %s"%(sl, params[:4])
            fit_resolutions.append(params[0])
            if plot: pylab.plot(sl,params[0],'go')
        poly_coef = scipy.polyfit(slopes, fit_resolutions, 2)
        best_slope = -0.5*poly_coef[1]/poly_coef[0]
        print "Drift correction requires slope %6.3f"%best_slope
        self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
        
        self.calibration['p_filt_value_dc'] = energy_calibration.EnergyCalibration('p_filt_value_dc')
        
        if plot:
            pylab.subplot(212)
            pylab.plot(corrector, data, ',')
            xlim = pylab.xlim()
            c = numpy.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
            pylab.plot(c, -c*best_slope + data.mean(),color='green')
            pylab.ylim(prange)
            
            axis1.plot(slopes, numpy.poly1d(poly_coef)(slopes),color='red')


    def fit_spectral_line(self, prange, times=None, type='dc', line='MnKAlpha', verbose=True, plot=True, **kwargs):
        all_values={'filt': self.p_filt_value,
                    'phc': self.p_filt_value_phc,
                    'dc': self.p_filt_value_dc,
                    'energy': self.p_energy,
                    }[type]
        valid = self.cuts.good()
        if times is not None:
            valid = numpy.logical_and(valid, self.p_timestamp<times[1])
            valid = numpy.logical_and(valid, self.p_timestamp>times[0])
        good_values = all_values[valid]
        contents,bin_edges = numpy.histogram(good_values, 200, prange)
        if verbose: print "%d events pass cuts; %d are in histogram range"%(len(good_values),contents.sum())
        bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        # Temporary hack for Lorentzian lines that we'll approximate as Gaussian
        if line in ('AlKalpha', 'SiKalpha'):
            fittername = 'fluorescence_lines.GaussianFitter(fluorescence_lines.%s())'%line
        else:
            fittername = 'fluorescence_lines.%sFitter()'%line
        fitter = eval(fittername)
        print 'fittername: ',fittername
        print fitter
        params, covar = fitter.fit(contents, bin_ctrs, plot=plot, **kwargs)
        if verbose: print 'Resolution: %5.2f +- %5.2f eV'%(params[0],numpy.sqrt(covar[0,0]))
        return params, covar
    
    
    def fit_MnK_lines(self, times=None, update_energy=True, verbose=False, plot=True):
        """"""
        
        if plot:
            pylab.clf()
            ax1 = pylab.subplot(221)
            ax2 = pylab.subplot(222)
            ax3 = pylab.subplot(223)
        else:
            ax1 = ax2 = ax3 = None
        
        calib = self.calibration['p_filt_value_dc']
        mnka_range = calib.name2ph('Mn Ka1') * numpy.array((.99,1.01))
        params, _covar = self.fit_spectral_line(prange=mnka_range, times=times, type='dc', line='MnKAlpha', verbose=verbose, plot=plot, axis=ax1)
        calib.add_cal_point(params[1], 'Mn Ka1')

        mnkb_range = calib.name2ph('Mn Kb') * numpy.array((.94,1.02))
#        params[1] = calib.name2ph('Mn Kb')
#        params[3] *= 0.50
#        params[4] = 0.0
        try:
            params, _covar = self.fit_spectral_line(prange=mnkb_range, times=times, type='dc', line='MnKBeta', 
                                                    verbose=verbose, plot=plot, axis=ax2)
            calib.add_cal_point(params[1], 'Mn Kb')
        except scipy.linalg.LinAlgError:
            print "Failed to fit Mn K-beta!"
        if update_energy: self.p_energy = calib(self.p_filt_value_dc)
        
        if plot:
            calib.plot(axis=pylab.subplot(224))
            self.fit_spectral_line(prange=(5850,5930), times=times, type='energy', line='MnKAlpha', verbose=verbose, plot=plot, axis=ax3)
            ax1.set_xlabel("Filtered, drift-corr. PH")
            ax2.set_xlabel("Filtered, drift-corr. PH")
            ax3.set_xlabel("Energy (eV)")
            ax1.text(.06,.8,'Mn K$\\alpha$', transform=ax1.transAxes)
            ax2.text(.06,.8,'Mn K$\\beta$', transform=ax2.transAxes)
            ax3.text(.06,.8,'Mn K$\\alpha$', transform=ax3.transAxes)



################################################################################################

def create_pulse_and_noise_records(fname, noisename=None, records_are_continuous=True, 
                                   noise_only=False, pulse_only=False):
    """
    Factory function to create a PulseRecords and a NoiseRecords object
    from a raw LJH file name, with optional LJH-style noise file name.
    
    Return two objects: a PulseRecords, NoiseRecords tuple.
    
    <fname>     The path of the raw data file containing pulse data
    <noisename> The path of the noise data file.  If None, it will be inferred
                by replacing the file extension in <fname> with ".noi".
    <records_are_continuous> Whether the noise file's data are auto-triggered
                fast enough to have no gaps in the data.
    <noise_only> The <fname> are noise files, and there are no pulse files.
    """
    if noisename is None:
        try:
            root, _ext = os.path.splitext(fname)
            noisename = root+".noi"
        except:
            raise ValueError("If noisename is not given, it must be constructable by replacing file suffix with '.noi'")
    
    assert not (noise_only and pulse_only)
    
    if pulse_only:
        pr = PulseRecords(fname)
        nr = None
    elif noise_only:
        pr = PulseRecords(fname)
        nr = NoiseRecords(fname, records_are_continuous)
    else:
        pr = PulseRecords(fname)
        nr = NoiseRecords(noisename, records_are_continuous)
    
    return (pr, nr)
