"""
Created on Feb 16, 2011

@author: fowlerj
"""

__all__=['NoiseRecords', 'PulseRecords', 'Cuts',
         'MicrocalDataSet', 'create_pulse_and_noise_records']

import numpy as np
import scipy as sp
import pylab as plt
import os.path

try:
    import cPickle as pickle
except ImportError:
    import pickle    

# MASS modules
#import mass.mathstat
import mass.mathstat.power_spectrum
from mass.core.files import VirtualFile, LJHFile, LANLFile

class NoiseRecords(object):
    """
    Encapsulate a set of noise records, which can either be
    assumed continuous or arbitrarily separated in time.
    """
    DEFAULT_MAXSEGMENTSIZE = 32000000
    
    def __init__(self, filename, records_are_continuous=False, use_records=None, maxsegmentsize=None):
        """
        Load a noise records file.

        If <records_are_continuous> is True, then treat all pulses as a continuous timestream.
        <use_records>  can be a sequence (first,end) to use only a limited section of the file.
        """
        if maxsegmentsize is not None:
            self.maxsegmentsize = maxsegmentsize
        else:
            self.maxsegmentsize = self.DEFAULT_MAXSEGMENTSIZE
            
        self.nSamples = self.nPresamples = self.nPulses = 0
        self.timebase = 0.0
        self.__open_file(filename, use_records=use_records)
        self.continuous = records_are_continuous
        self.spectrum = None
        self.autocorrelation = None
     
    def __open_file(self, filename, use_records=None, file_format=None):
        """Detect the filetype and open it."""

        ALLOWED_TYPES=("ljh","root","virtual")
        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("root"):
                file_format = "root"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s"%ALLOWED_TYPES)

        if file_format == "ljh":
            self.datafile = LJHFile(filename, segmentsize=self.maxsegmentsize)
        elif file_format == "root":
            self.datafile = LANLFile(filename)
        elif file_format == "virtual":
            vfile = filename # Aha!  It must not be a string
            self.datafile = vfile
            self.datafile.segmentsize = vfile.nPulses*(6+2*vfile.nSamples)
            filename = 'Virtual file'
        else:
            raise RuntimeError("It is a programming error to get here")
        self.filename = filename
        self.records_per_segment = self.datafile.segmentsize / (6+2*self.datafile.nSamples)
        
        if use_records is not None:
            if use_records < self.datafile.nPulses:
                self.datafile.nPulses = use_records
                self.datafile.n_segments = use_records / self.records_per_segment

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum"):
            self.__dict__[attr] = self.datafile.__dict__[attr]

#        for first_pnum, end_pnum, seg_num, data in self.datafile.iter_segments():
#            if seg_num > 0 or first_pnum>0 or end_pnum != self.nPulses:
#                msg = "NoiseRecords objects can't (yet) handle multi-segment noise files.\n"+\
#                    "File size %d exceeds maximum allowed segment size of %d"%(
#                    self.datafile.binary_size, self.maxsegmentsize)
#                raise NotImplementedError(msg)
#            self.data = data

    def clear_cache(self):
        self.datafile.clear_cache()

    def set_fake_data(self):
        """Use when this does not correspond to a real datafile (e.g., CDM data)"""
        self.datafile = mass.VirtualFile(np.zeros((0,0)))
        

    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = NoiseRecords(self.filename)
        c.__dict__.update( self.__dict__ )
        c.datafile = self.datafile.copy()
        return c

    
    def compute_power_spectrum(self, window=mass.mathstat.power_spectrum.hann, plot=True, max_excursion=9e9):
        self.compute_power_spectrum_reshape(window=window, nsegments=None,
                                            max_excursion=max_excursion)
        if plot: self.plot_power_spectrum()


    def compute_power_spectrum_reshape(self, window=mass.mathstat.power_spectrum.hann, seg_length=None, 
                                       max_excursion=9e9):
        """Compute the noise power spectrum with noise "records" reparsed into 
        separate records of <seg_length> length.  (If None, then self.data.shape[0] which is self.data.nPulses,
        will be used as the number of segments, each having length self.data.nSamples.)
        
        By making <nsegments> large, you improve the noise on the PSD estimates at the price of poor
        frequency resolution.  By making it small, you get good frequency resolution with worse
        uncertainty on each PSD estimate.  No free lunch, know what I mean?
        """
        
        if not self.continuous and seg_length is not None:
            raise ValueError("This NoiseRecords object does not have continuous noise records, so it can't be resegmented.")
        
        if seg_length is None:
            seg_length = self.nSamples
        
        self.spectrum = mass.mathstat.power_spectrum.PowerSpectrum(seg_length/2, dt=self.timebase)
        if window is None:
            window = np.ones(seg_length)
        else:
            window = window(seg_length)
            
        for _first_pnum, _end_pnum, _seg_num, data in self.datafile.iter_segments():
            if self.continuous and seg_length is not None:
                data = data.ravel()
                n=len(data)
                n = n-n%seg_length
                data=data[:n].reshape((n/seg_length, seg_length))
    
            for d in data:
                y = d-d.mean()
                if y.max() - y.min() < max_excursion and len(y)==self.spectrum.m2:
                    self.spectrum.addDataSegment(y, window=window)


    def compute_fancy_power_spectrum(self, window=mass.mathstat.power_spectrum.hann, plot=True, nseg_choices=None):
        assert self.continuous

        n = np.prod(self.data.shape)
        if nseg_choices is None:
            nseg_choices = [16]
            while nseg_choices[-1]<=n/16 and nseg_choices[-1]<20000:
                nseg_choices.append(nseg_choices[-1]*8)
        print nseg_choices

        spectra = [self.compute_power_spectrum_reshape(window=window, nsegments=ns) for ns in nseg_choices]
        if plot:
            plt.clf()
            lowest_freq = np.array([1./(sp.dt*sp.m2) for sp in spectra])
            
            start_freq=0.0
            for i,sp in enumerate(spectra):
                x,y = sp.frequencies(), sp.spectrum()
                if i==len(spectra)-1:
                    good = x>=start_freq
                else:
                    good = np.logical_and(x>=start_freq, x<4*lowest_freq[i+1])
                    start_freq = 1*lowest_freq[i+1]
                plt.loglog(x[good],y[good],'-')
    
    def plot_power_spectrum(self, axis=None, scale=1.0, sqrt_psd=False, **kwarg):
        """
        Plot the power spectrum of this noise record.
        
        <axis>     Which plt.Axes object to plot on.  If none, clear the figure and plot there.
        <scale>    Scale all raw units by this number to convert counts to physical
        <sqrt_psd> Whether to take the sqrt(PSD) for plotting.  Default is no sqrt
        """
        if self.spectrum is None:
            self.compute_power_spectrum(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        spec = self.spectrum
        yvalue = spec.spectrum()[1:] * (scale**2)
        if sqrt_psd: 
            yvalue = np.sqrt(yvalue)
        axis.plot(spec.frequencies()[1:], yvalue, **kwarg)
        plt.loglog()
        axis.grid()
        axis.set_xlim([10,3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title("Noise power spectrum for %s"%self.filename)

        
    def _compute_continuous_autocorrelation(self, n_lags=None, data_samples=None, 
                                            max_excursion=9e9):
        if data_samples is None:
            data_samples = [0, self.nSamples*self.nPulses]
        n_data = data_samples[1] - data_samples[0]
            
        samples_per_segment = self.records_per_segment*self.nSamples
        if n_lags is None:
            n_lags = samples_per_segment
        if n_lags > samples_per_segment:
            n_lags = samples_per_segment
            
#        print 'Compute continuous autocorr (%d lags on %d data) '%(n_lags, n_data),
        
        def padded_length(n):
            """Return a sensible number in the range [n, 2n] which is not too
            much larger than n, yet is good for FFTs.
            That is, choose (1, 3, or 5)*(a power of two), whichever is smallest
            """
            pow2 = np.round(2**np.ceil(np.log2(n)))
            if n==pow2: return n
            elif n>0.75*pow2: return pow2
            elif n>0.625*pow2: return np.round(0.75*pow2)
            else: return np.round(0.625*pow2)
        
    
        # When there are 10 million data points and only 10,000 lags wanted,
        # it's hugely inefficient to compute the full autocorrelation, especially
        # in memory.  Instead, compute it on chunks several times the length of the desired
        # correlation, and average.
        CHUNK_MULTIPLE=31
        if n_data >= (1+CHUNK_MULTIPLE)*n_lags:
            # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
            # unwanted wraparound in the autocorrelation.
            # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
            chunksize=CHUNK_MULTIPLE*n_lags
            padsize = n_lags
            padded_data = np.zeros(padded_length(padsize+chunksize), dtype=np.float)
#            print 'with chunks of %d, padsize %d'%(chunksize,padsize)
            
            ac = np.zeros(n_lags, dtype=np.float)
            
            entries = 0.0
#            t0=time.time()
            
            for first_pnum, end_pnum, _seg_num, data in self.datafile.iter_segments():
#                print "Using pulses %d to %d (seg=%3d)"%(first_pnum, end_pnum, seg_num)
                data_consumed=0
                data = data.ravel()
                samples_this_segment = len(data)
                if data_samples[0] > self.nSamples*first_pnum:
                    data_consumed = data_samples[0]-self.nSamples*first_pnum
                if data_samples[1] < self.nSamples*end_pnum:
                    samples_this_segment = data_samples[1]-self.nSamples*first_pnum
#                print data_consumed, samples_this_segment, "used, sthisseg", data.shape
                data_mean = data[data_consumed:samples_this_segment].mean()

                # Notice that the following loop might ignore the last data values, up to as many
                # as (chunksize-1) values, unless the data are an exact multiple of chunksize.
                while data_consumed+chunksize <= samples_this_segment:
                    padded_data[:chunksize] = data[data_consumed:data_consumed+chunksize] - data_mean
                    data_consumed += chunksize
                    padded_data[chunksize:] = 0.0
                    if np.abs(padded_data).max() > max_excursion:
                        continue
                    
                    ft = np.fft.rfft(padded_data)
                    ft[0] = 0 # this redundantly removes the mean of the data set
                    power = (ft*ft.conj()).real
                    acsum = np.fft.irfft(power)
                    ac += acsum[:n_lags] 
                    entries += 1.0
                    if entries*chunksize > n_data:
                        break

            ac /= entries
            ac /= (np.arange(chunksize, chunksize-n_lags+0.5, -1.0, dtype=np.float))
                
        # compute the full autocorrelation                
        else:
            raise NotImplementedError("Now that Joe has chunkified the noise, we can no longer compute full continuous autocorrelations")
            padded_data = np.zeros(padded_length(n_lags+n_data), dtype=np.float)
            padded_data[:n_data] = np.array(self.data.ravel())[:n_data] - self.data.mean()
            padded_data[n_data:] = 0.0
            
            ft = np.fft.rfft(padded_data)
            del padded_data
            ft[0] = 0 # this redundantly removes the mean of the data set
            ft *= ft.conj()
            ft = ft.real
            acsum = np.fft.irfft(ft)
            del ft
            ac = acsum[:n_lags+1] / (n_data-np.arange(n_lags+1.0))
            del acsum
            
        self.autocorrelation = ac

        
    def compute_autocorrelation(self, n_lags=None, data_samples=None, plot=True, max_excursion=9e9):
        """
        Compute the autocorrelation averaged across all "pulses" in the file.
        <n_lags>
        <data_samples> If not None, then a range [a,b] to use.
        """
        
        if self.continuous:
            self._compute_continuous_autocorrelation(n_lags=n_lags, data_samples=data_samples, 
                                                     max_excursion=max_excursion)

        else:
            if n_lags is not None and n_lags > self.nSamples:
                raise ValueError("The autocorrelation requires n_lags<=%d when data are not continuous"%self.nSamples)

            class TooMuchData(StopIteration):
                "Use to signal that the computation loop is done" 
                pass
            
            if data_samples is None:
                data_samples = [0, self.nSamples*self.nPulses]
            n_data = data_samples[1] - data_samples[0]

            records_used = samples_used = 0
            ac=np.zeros(self.nSamples, dtype=np.float)
            try:
                for first_pnum, end_pnum, _seg_num, intdata in self.datafile.iter_segments():
                    if end_pnum <= data_samples[0]: continue
                    if first_pnum >= data_samples[1]: break
                    for i in range(first_pnum, end_pnum):
                        if i < data_samples[0]: continue
                        if i >= data_samples[1]: break

                        data = 1.0*(intdata[i-first_pnum,:])
                        if data.max() - data.min() > max_excursion: continue
                        data -= data.mean()
    
                        ac += np.correlate(data,data,'full')[self.nSamples-1:]
                        samples_used += self.nSamples
                        records_used += 1
                        if n_data is not None and samples_used >= n_data: 
                            raise TooMuchData()
            except TooMuchData: 
                pass
            
            ac /= records_used
            ac /= self.nSamples - np.arange(self.nSamples, dtype=np.float)
            if n_lags is not None and n_lags < self.nSamples:
                ac=ac[:n_lags]
            self.autocorrelation = ac
        if plot: self.plot_autocorrelation()
        

    def plot_autocorrelation(self, axis=None, color='blue', label=None):
        if self.autocorrelation is None:
            print "Autocorrelation will be computed first"
            self.compute_autocorrelation(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        t = self.timebase * 1e3 * np.arange(len(self.autocorrelation))
        axis.plot(t,self.autocorrelation, label=label, color=color)
        axis.plot([0],[self.autocorrelation[0]],'o', color=color)
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
        self.pulses_per_seg = 0
        self.timebase = None
        self.timestamp_offset = None
        
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

        ALLOWED_TYPES=("ljh","root","virtual")
        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("root"):
                file_format = "root"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s"%ALLOWED_TYPES)

        if file_format == "ljh":
            self.datafile = LJHFile(filename)
        elif file_format == "root":
            self.datafile = LANLFile(filename)
        elif file_format == "virtual":
            vfile = filename # Aha!  It must not be a string
            self.datafile = vfile
        else:
            raise RuntimeError("It is a programming error to get here")
        
        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples","nPresamples","nPulses", "timebase", "channum",
                     "n_segments", "pulses_per_seg", "segmentsize", "timestamp_offset"):
            self.__dict__[attr] = self.datafile.__dict__[attr]


    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)

    
    def set_segment_size(self, seg_size):
        """Update the underlying file's segment (read chunk) size in bytes."""
        self.datafile.set_segment_size(seg_size)
        self.n_segments = self.datafile.n_segments
        self.pulses_per_seg = self.datafile.pulses_per_seg
        self.segmentsize = self.datafile.segmentsize
        
    
    def read_segment(self, segment_num):
        """Read the requested segment of the raw data file and return  (first,end)
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
        idx = np.arange(len(rising_data), dtype=np.int)
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
#    idx = np.arange(len(useable))
#    last_idx = idx[useable<MAXTHRESH*valpk].max()
#    first_idx = idx[useable>MINTHRESH*valpk].min()
#    if (last_idx-first_idx) < 4:
#        raise ValueError("Cannot make an exponential fit to only %d points!"%
#                         (last_idx-first_idx+1))
#    
#    x, y = idx[first_idx:last_idx+1], useable[first_idx:last_idx+1]
#    
#    fitfunc = lambda p, x: p[0]*np.exp(-x/p[1])+p[2]
#    errfunc = lambda p, x, y: fitfunc(p, x) - y
#    
#    p0 = -useable.max(), 0.6*(x[-1]-x[0]), useable.max()  
#    p, _stat = sp.optimize.leastsq(errfunc, p0, args=(x,y))
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
    filter_coef = np.array([ -0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905,   ])[::-1]
    conv = np.convolve(ts[first:end], filter_coef, mode='valid')
    
    if return_index_too:
        return first + 2 + conv.argmax() # This would be the index.
    return conv.max()



class Cuts(object):
    "Object to hold a mask for each trigger."
    
    def __init__(self, n):
        "Create an object to hold n masks of 32 bits each"
        self._mask = np.zeros( n, dtype=np.int32 )
        
    def cut(self, cutnum, mask):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        assert(mask.size == self._mask.size)
        bitval = 1<<cutnum
        self._mask[mask] |= bitval

    def clearCut(self, cutnum):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        bitmask = ~(1<<cutnum)
        self._mask &= bitmask
        
    def good(self):
        return np.logical_not(self._mask)
    
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
    
    def copy(self):
        c = Cuts(len(self._mask))
        c._mask = self._mask.copy()
        return c
        



class MicrocalDataSet(object):
    """
    Represent a single microcalorimeter's PROCESSED data.
    This channel can be directly from a TDM detector, or it
    can be the demodulated result of a CDM modulation.
    """
    CUT_NAME = ['pretrigger_rms',
                'pretrigger_mean',
                'pretrigger_mean_departure_from_median',
                'peak_time_ms',
                'rise_time_ms',
                'max_posttrig_deriv',
                'pulse_average',
                'min_value',
                'timestamp_sec',
                'timestamp_diff_sec',
                'peak_value',
                'energy'] 

    # Attributes that all such objects must have.
    expected_attributes=("nSamples","nPresamples","nPulses","timebase", "channum", 
                         "timestamp_offset")



    def __init__(self, pulserec_dict, auto_pickle = True):
        """
        Pass in a dictionary (presumably that of a PulseRecords object)
        containing the expected attributes that must be copied to this
        MicrocalDataSet.
        """
        self.auto_pickle = auto_pickle
        self.filter = {}
        self.lastUsedFilterHash = -1
        self.drift_correct_info = {}
        self.phase_correct_info = {}
        self.noise_spectrum = None
        self.noise_autocorr = None 
        self.noise_demodulated = None
        self.calibration = {'p_filt_value':mass.calibration.energy_calibration.EnergyCalibration('p_filt_value')}

        for a in self.expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename','virtual data set')
        self.gain = 1.0
        self.pretrigger_ignore_microsec = None # Cut this long before trigger in computing pretrig values
        self.peak_time_microsec = None   # Look for retriggers only after this time. 
        self.index = None   # Index in the larger TESGroup or CDMGroup object
        self.__setup_vectors(npulses=self.nPulses)
        if self.auto_pickle:
            self.unpickle()
            



    def __setup_vectors(self, npulses=None):
        """Given the number of pulses, build arrays to hold the relevant facts 
        about each pulse in memory.
        p_filt_value = pulse height after running through filter
        P_filt_value_phc = phase corrected
        p_fil_value_dc = pulse height after running through filter and applying drift correction
        p_energy = pulse energy determined from applying a calibration to one of the p_filt_value??? variables"""
        
        if npulses is None:
            assert self.nPulses > 0
            npulses = self.nPulses
        self.p_timestamp = np.zeros(npulses, dtype=np.float64)
        self.p_peak_index = np.zeros(npulses, dtype=np.uint16)
        self.p_peak_value = np.zeros(npulses, dtype=np.uint16)
        self.p_min_value = np.zeros(npulses, dtype=np.uint16)
        self.p_pretrig_mean = np.zeros(npulses, dtype=np.float32)
        self.p_pretrig_rms = np.zeros(npulses, dtype=np.float32)
        self.p_pulse_average = np.zeros(npulses, dtype=np.float32)
        self.p_pulse_rms = np.zeros(npulses, dtype=np.float32)
        self.p_promptness = np.zeros(npulses, dtype=np.float32)
        self.p_rise_time = np.zeros(npulses, dtype=np.float32)
        self.p_max_posttrig_deriv = np.zeros(npulses, dtype=np.float32)
        self.p_filt_phase = np.zeros(npulses, dtype=np.float64) # float32 for p_filt_phase makes energy resolution worse, gco, 20130516, it should be possible to use 32 but probably requires rescaling phase
        # maybe converting phase to int16, where 0 is 0, -max is -2, max is 2?
        self.p_filt_value = np.zeros(npulses, dtype=np.float32) 
        self.p_filt_value_phc = np.zeros(npulses, dtype=np.float32) 
        self.p_filt_value_dc = np.zeros(npulses, dtype=np.float32)
        self.p_energy = np.zeros(npulses, dtype=np.float32)
        
        self.cuts = Cuts(self.nPulses)
    
    @property
    def p_peak_time(self):
        # this is a property to reduce memory usage, I hope it works
        return (np.asarray(self.p_peak_index, dtype=np.int)-self.nPresamples)*self.timebase

    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)


    def good(self):
        """Return a boolean vector, one per pulse record, saying whether record is good"""
        return self.cuts.good()
    
    def bad(self):
        """Return a boolean vector, one per pulse record, saying whether record is bad"""
        return self.cuts.bad()
    
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
        c.cuts = self.cuts.copy()
        if self.noise_spectrum is None:
            c.noise_spectrum = None
        else:
            c.noise_spectrum = self.noise_spectrum.copy()
        return c

#    def save_mass_data(self, dataname, *args, **kwargs):
#        basedir, basename = os.path.split(self.filename)
#        basename, baseext = os.path.splitext(basename)
#        massdir = os.path.join(basedir, "mass")
#        if not os.path.isdir(massdir):
#            os.mkdir(massdir, 0775)
#        filename = os.path.join(massdir, '%s_%s'%(basename, dataname))
#        np.savez(filename, *args, **kwargs)
#        
#    def load_mass_data(self, dataname):
#        basedir, basename = os.path.split(self.filename)
#        basename, baseext = os.path.splitext(basename)
#        massdir = os.path.join(basedir, 'mass')
#        filename = os.path.join(massdir, '%s_%s.%s'%(basename, dataname,'npz'))
#        try:
#            data_out = np.load(filename)
#        except IOError:
#            data_out = None
#        return data_out

    def pickle(self, filename=None, verbose=True):
        """Pickle the _contents_ of the MicrocalDataSet object.
        <filename>    The output pickle name.  If not given, then it will be the data file name
                      with the suffix replaced by '.pkl' and in a subdirectory mass under the
                      main file's location."""
    
        if filename is None:
            basedir = os.path.dirname(self.filename)
            massdir = os.path.join(basedir, "mass")
            if not os.path.isdir(massdir):
                os.mkdir(massdir, 0775)
            filename = os.path.join(massdir, "%s.pkl"%os.path.basename(self.filename))
        
        import cPickle
        fp = open(filename, "wb")
        pickler = cPickle.Pickler(fp, protocol=2)
        
        # Pickle the self.expected_attributes
        exp_at = {}
        for k in self.expected_attributes:
            exp_at[k] = self.__dict__[k]
        pickler.dump(exp_at)
        
        # Pickle the cuts mask
        pickler.dump(self.cuts._mask)
        
        # Pickle all attributes noise_*, p_*, peak_time_microsec, pretrigger_*, timebase, times
        # Approach is to dump the attribute NAME then value.
        attr_starts = ("noise_", "p_", "pretrigger_")
        attr_names = ("peak_time_microsec", "timebase", "times", "average_pulse",
                      "calibration", "drift_correct_info", "phase_correct_info", "filter" )
        for attr in self.__dict__:
            store_this_attr = attr in attr_names
            for ast in attr_starts:
                if attr.startswith(ast):
                    store_this_attr = True
                    break
            if store_this_attr:
                pickler.dump(attr)
                pickler.dump(self.__dict__[attr])
        fp.close()
        if verbose:
            print "Stored %9d bytes %s"%(os.stat(filename).st_size, filename)
    
        
    def unpickle(self, filename=None):
        """
        Unpickle a MicrocalDataSet pickled by its .pickle() method.
        
        Data structure must be:
        1. A dictionary with simple values, whose keys include at least all strings in
           the tuple MicrocalDataSet.expected_attributes.
        2. The dataset cuts._mask
        3. Any string.  If #4 also loads, then this will be the attribute name.
        4. Any pickleable object.  This will become an attribute value
           (prev item gives its name)
        ... Repeat items (3,4) as needed to load all attribute (name,value) pairs.
        """
        if filename is None:
            basedir = os.path.dirname(self.filename)
            massdir = os.path.join(basedir, "mass")
            if not os.path.isdir(massdir):
                os.mkdir(massdir, 0775)
            filename = os.path.join(massdir, "%s.pkl"%os.path.basename(self.filename))
        if not os.path.isfile(filename):
            return
        
        import cPickle
        fp = open(filename, "rb")
        unpickler = cPickle.Unpickler(fp)
        _expected_attr = unpickler.load()
        # ignore the expected_attr
        self.cuts._mask = unpickler.load()
        try:
            while True:
                k = unpickler.load()
                v = unpickler.load()
                self.__dict__[k] = v
        except EOFError:
            pass
        fp.close()

    def summarize_data_tdm(self, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0, forceNew = False):
        """summarized the complete data file one chunk at a time
        this version does the whole dataset at once (instead of previous segment at a time for all datasets)
        """
        if len(self.p_timestamp) < self.pulse_records.nPulses:
            self.__setup_vectors(nPulses=self.pulse_records.nPulses)
        elif forceNew or all(self.p_timestamp==0):
            self.pretrigger_ignore_samples = int(pretrigger_ignore_microsec*1e-6/self.timebase)   
            # consider setting segment size first
            printUpdater = mass.calibration.inlineUpdater.InlineUpdater('channel.summarize_data_tdm chan %d'%self.channum)

            for s in range(self.pulse_records.n_segments):
                first, last = self.pulse_records.read_segment(s) # this reloads self.data to contain new pulses
                self.p_timestamp[first:last] = self.pulse_records.datafile.datatimes_float
                (self.p_pretrig_mean[first:last], self.p_pretrig_rms[first:last],
                self.p_peak_index[first:last], self.p_peak_value[first:last], self.p_min_value[first:last],
                self.p_pulse_average[first:last], self.p_rise_time[first:last], 
                self.p_max_posttrig_deriv[first:last]) = mass.mathstat.summarize_and_filter.summarize_old(self.pulse_records.data, 
                    self.nPresamples, self.pretrigger_ignore_samples, self.timebase, peak_time_microsec)
                printUpdater.update((s+1)/float(self.pulse_records.n_segments))
            self.pulse_records.datafile.clear_cached_segment()      
            if self.auto_pickle:
                self.pickle(verbose=False)
        else:
            print('\nchan %d did not summarize because results were already preloaded'%self.channum)

    def summarize_data(self, first, end, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0):
        """Summarize the complete data file
        summarize_data(self, first, end, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0)
        peak_time_microsec is used when calculating max dp/dt after trigger
        
        """ 
        self.peak_time_microsec = peak_time_microsec
        self.pretrigger_ignore_microsec = pretrigger_ignore_microsec
        if first >= self.nPulses:
            return
        if end > self.nPulses:
            end = self.nPulses
        if len(self.p_timestamp) <= 0:
            self.__setup_vectors(npulses=self.nPulses)

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
        PTM = self.p_pretrig_mean[first:end]
        self.p_peak_value[first:end] -= PTM   # subtract float from int, remains an int
        self.p_pulse_average[first:end] -= PTM
        self.p_pulse_rms[first:end] = np.sqrt(
                (self.data[:seg_size,self.nPresamples:]**2.0).mean(axis=1) -
                PTM*(PTM + 2*self.p_pulse_average[first:end]))
        self.p_promptness[first:end] = (
                self.data[:seg_size,self.nPresamples+3:self.nPresamples+7].mean(axis=1)-PTM)/self.p_peak_value[first:end]

        # Compute things that have to be computed one at a time:
        for pulsenum,pulse in enumerate(self.data):
            if pulsenum>=seg_size: break
            self.p_rise_time[first+pulsenum] = estimateRiseTime(pulse, 
                                                dt=self.timebase, nPretrig = self.nPresamples)
            self.p_max_posttrig_deriv[first+pulsenum] = \
                compute_max_deriv(pulse[self.nPresamples + maxderiv_holdoff:])

    
    def filter_data_tdm(self, filter_name='filt_noconst', transform=None, forceNew=False):
        """filter the complete data file one chunk at a time
        this version does the whole dataset at once (instead of previous segment at a time for all datasets)
        """
        filter_values = self.filter.__dict__[filter_name]
        if forceNew or all(self.p_filt_value == 0): # determine if we need to do anything
            printUpdater = mass.calibration.inlineUpdater.InlineUpdater('channel.filter_data_tdm chan %d'%self.channum)
            for s in range(self.pulse_records.n_segments):
                first, last = self.pulse_records.read_segment(s) # this reloads self.data to contain new pulses
                (self.p_filt_phase[first:last], self.p_filt_value[first:last]) = mass.mathstat.summarize_and_filter.filter_data_old(
                filter_values, self.pulse_records.data, transform, self.p_pretrig_mean[first:last])
                printUpdater.update((s+1)/float(self.pulse_records.n_segments))
                
            self.pulse_records.datafile.clear_cached_segment()    
            if self.auto_pickle:
                self.pickle(verbose=False)  
        else:
            print('\nchan %d did not filter because results were already loaded'%self.channum)
        

    def filter_data(self, filter_values, first, end, transform=None):
        if first >= self.nPulses:
            return None,None

        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = np.array((
                ( -6,24, 34,24,-6),
                (-14,-7,  0, 7,14),
                ( 10,-5,-10,-5,10)), dtype=np.float)/70.0
        
        assert len(filter_values)+4 == self.nSamples

        seg_size = min(end-first, self.data.shape[0])
        conv = np.zeros((5, seg_size), dtype=np.float)
        if transform is not None:
            ptmean = self.p_pretrig_mean[first:end]
            ptmean.shape = (len(ptmean),1)
            data = transform(self.data-ptmean)
        for i in range(5):
            if i-4 == 0:
                # previous method in comments, converted to dot product based on ~30% speed boost in tests
#                    conv[i,:] = (filter_values*self.data[:seg_size,i:]).sum(axis=1)
                conv[i,:] = np.dot(self.data[:,i:], filter_values)
            else:
#                    conv[i,:] = (filter_values*self.data[:seg_size,i:i-4]).sum(axis=1)
                conv[i,:] = np.dot(self.data[:seg_size,i:i-4], filter_values)


        param = np.dot(fit_array, conv)
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
            hour = self.p_timestamp[valid][::downsample]/3600.0
        else:
            nrecs = self.nPulses
            if downsample is None:
                downsample = self.nPulses / 10000
                if downsample < 1: downsample = 1
            hour = self.p_timestamp[::downsample]/3600.0
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
        
        plt.clf()
        for i,(vect, label, color, limits) in enumerate(plottables):

            # Time series scatter plots (left-hand panels)
            plt.subplot(len(plottables), 2, 1+i*2)
            plt.ylabel(label)
            if valid is not None:
                vect = vect[valid]
            plt.plot(hour, vect[::downsample],',', color=color)

            # Histogram (right-hand panels)            
            plt.subplot(len(plottables), 2, 2+i*2)
            if limits is None:
                in_limit = np.ones(len(vect), dtype=np.bool)
            else:
                in_limit = np.logical_and(vect>limits[0], vect<limits[1])
            contents, _bins, _patches = plt.hist(vect[in_limit],200, log=log, 
                           histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin = contents.min())


    def cut_parameter(self, data, allowed, cut_id):
        """Apply a cut on some per-pulse parameter.  
        
        <data>    The per-pulse parameter to cut on.  It can be an attribute of self, or it
                  can be computed from one (or more), 
                  but it must be an array of length self.nPulses
        <allowed> The cut to apply (see below).
        <cut_id>  The bit number (range [0,31]) to identify this cut.  Should be one of
                  self.CUT_NAME
        
        <allowed> is a 2-element sequence (a,b), then the cut requires a < data < b. 
        Either a or b may be None, indicating no cut.
        OR
        <allowed> is a sequence of 2-element sequences (a,b), then the cut cuts data that does not meet a <= data <=b
        for any of the two element sequences, if any element in allowed is ''invert'' then it swaps cut and uncut
        """
        
        if allowed is None: # no cut here! 
            return
        if cut_id <0 or cut_id >=32:
            raise ValueError("cut_id must be in the range [0,31]")
        
        # determine if allowed is a sequence or a sequence of sequences
        if np.size(allowed[0]) == 2 or allowed[0] == 'invert':
            doInvert = False
            cut_vec = np.ones_like(data, dtype='bool')
            for element in allowed:
                if np.size(element) == 2:
                    try:
                        a,b = element
                        if a is not None and b is not None:
                            index = np.logical_and(data >= a, data <= b)
                        elif a is not None:
                            index = data >= a
                        elif b is not None:
                            index = data <= b
                        cut_vec[index] = False
                    except:
                        raise ValueError('%s was passed as a cut element, only two element lists or tuples are valid'%str(element))
                elif element == 'invert':
                    doInvert = True
            if doInvert:
                self.cuts.cut(cut_id, ~cut_vec)
            else:
                self.cuts.cut(cut_id, cut_vec)
        else:
            try:
                a,b = allowed
                if a is not None:
                    self.cuts.cut(cut_id, data <= a)
                if b is not None:
                    self.cuts.cut(cut_id, data >= b)
            except ValueError:
                raise ValueError('%s was passed as a cut element, but only two-element sequences are valid.'%str(allowed))

    
    
    def apply_cuts(self, controls=None, clear=False, verbose=1):
        """
        <clear>  Whether to clear previous cuts first (by default, do not clear).
        <verbose> How much to print to screen.  Level 1 (default) counts all pulses good/bad/total.
                    Level 2 adds some stuff about the departure-from-median pretrigger mean cut.
        """
        if clear: self.clear_cuts()
        
        if controls is None:
            controls = mass.controller.standardControl()
        c = controls.cuts_prm
              
        self.cut_parameter(self.p_energy, c['energy'], self.CUT_NAME.index('energy'))
        self.cut_parameter(self.p_pretrig_rms, c['pretrigger_rms'], 
                           self.CUT_NAME.index('pretrigger_rms'))
        self.cut_parameter(self.p_pretrig_mean, c['pretrigger_mean'],
                           self.CUT_NAME.index('pretrigger_mean'))

        # Careful: p_peak_index is unsigned, so make it signed before subtracting nPresamples:
        self.cut_parameter(1e3*self.p_peak_time, c['peak_time_ms'],
                           self.CUT_NAME.index('peak_time_ms'))
        self.cut_parameter(self.p_rise_time*1e3, c['rise_time_ms'],
                           self.CUT_NAME.index('rise_time_ms'))
        self.cut_parameter(self.p_max_posttrig_deriv, c['max_posttrig_deriv'],
                           self.CUT_NAME.index('max_posttrig_deriv'))
        self.cut_parameter(self.p_pulse_average, c['pulse_average'],
                           self.CUT_NAME.index('pulse_average'))
        self.cut_parameter(self.p_peak_value, c['peak_value'],
                           self.CUT_NAME.index('peak_value'))
        self.cut_parameter(self.p_min_value-self.p_pretrig_mean, c['min_value'],
                           self.CUT_NAME.index('min_value'))
        self.cut_parameter(self.p_timestamp, c['timestamp_sec'],
                           self.CUT_NAME.index('timestamp_sec'))
        if c['timestamp_diff_sec'] is not None:
            self.cut_parameter(np.hstack((0.0, np.diff(self.p_timestamp))),
                               c['timestamp_diff_sec'],
                               self.CUT_NAME.index('timestamp_diff_sec'))        
        if c['pretrigger_mean_departure_from_median'] is not None:
            median = np.median(self.p_pretrig_mean[self.cuts.good()])
            if verbose>1:
                print'applying cut on pretrigger mean around its median value of ',median
            self.cut_parameter(self.p_pretrig_mean-median,
                               c['pretrigger_mean_departure_from_median'],
                               self.CUT_NAME.index('pretrigger_mean_departure_from_median'))
        if verbose>0:
            print "Chan %d after cuts, %d are good, %d are bad of %d total pulses"%(
                self.channum, self.cuts.nUncut(), 
                self.cuts.nCut(), self.nPulses)

        
    def clear_cuts(self):
        self.cuts = Cuts(self.nPulses)
    
    
#    def phase_correct(self, prange=None, times=None, plot=True):
#        """Apply a correction for pulse variation with arrival phase.
#        Model is a parabolic correction with cups at +-180 degrees away from the "center".
#        
#        prange:  use only filtered values in this range for correction 
#        times: if not None, use this range of p_timestamps instead of all data (units are seconds
#               since server started--ugly but that's what we have to work with)
#        plot:  whether to display the result
#        """
#        
#        # Choose number and size of bins
#        phase_step=.05
#        nstep = int(.5+1.0/phase_step)
#        phases = (0.5+np.arange(nstep))/nstep - 0.5
#        phase_step = 1.0/nstep
#        
#        # Default: use the calibration to pick a prange
#        if prange is None:
#            calibration = self.calibration['p_filt_value']
#            ph_estimate = calibration.name2ph('Mn Ka1')
#            prange = np.array((ph_estimate*.98, ph_estimate*1.02))
#
#        # Estimate corrections in a few different pieces
#        corrections = []
#        valid = self.cuts.good()
#        if prange is not None:
#            valid = np.logical_and(valid, self.p_filt_value<prange[1])
#            valid = np.logical_and(valid, self.p_filt_value>prange[0])
#        if times is not None:
#            valid = np.logical_and(valid, self.p_timestamp<times[1])
#            valid = np.logical_and(valid, self.p_timestamp>times[0])
#
#        # Plot the raw filtered value vs phase
#        if plot:
#            plt.clf()
#            plt.subplot(211)
#            plt.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value[valid],',',color='orange')
#            plt.xlabel("Hypothetical 'center phase'")
#            plt.ylabel("Filtered PH")
#            plt.xlim([-.55,.55])
#            if prange is not None:
#                plt.ylim(prange)
#                
#        for ctr_phase in phases:
#            valid_ph = np.logical_and(valid,
#                                         np.abs((self.p_filt_phase - ctr_phase)%1) < phase_step*0.5)
##            print valid_ph.sum(),"   ",
#            mean = self.p_filt_value[valid_ph].mean()
#            median = np.median(self.p_filt_value[valid_ph])
#            corrections.append(mean) # not obvious that mean vs median matters
#            if plot:
#                plt.plot(ctr_phase, mean, 'or')
#                plt.plot(ctr_phase, median, 'vk', ms=10)
#        corrections = np.array(corrections)
#        assert np.isfinite(corrections).all()
#        
#        def model(params, phase):
#            "Params are (phase of center, curvature, mean peak height)"
#            phase = (phase - params[0]+.5)%1 - 0.5
#            return 4*params[1]*(phase**2 - 0.125) + params[2]
#        errfunc = lambda p,x,y: y-model(p,x)
#        
#        params = (0., 4, corrections.mean())
#        fitparams, _iflag = sp.optimize.leastsq(errfunc, params, args=(self.p_filt_phase[valid], self.p_filt_value[valid]))
#        phases = np.arange(-0.6,0.5001,.01)
#        if plot: plt.plot(phases, model(fitparams, phases), color='blue')
#        
#        
#        self.phase_correction={'phase':fitparams[0],
#                            'amplitude':fitparams[1],
#                            'mean':fitparams[2]}
#        fitparams[2] = 0
#        correction = model(fitparams, self.p_filt_phase)
#        self.p_filt_value_phc = self.p_filt_value - correction
#        self.p_filt_value_dc = self.p_filt_value_phc.copy()
#        print 'RMS phase correction is: %9.3f (%6.2f parts/thousand)'%(correction.std(), 
#                                            1e3*correction.std()/self.p_filt_value.mean())
#        
#        if plot:
#            plt.subplot(212)
#            plt.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value_phc[valid],',b')
#            plt.xlim([-.55,.55])
#            if prange is not None:
#                plt.ylim(prange)

    # galen 20130211 - I think this can be replaced by polyfit with 1 dimension, its faster, more obvious what is going on, and in my test yielded the same answer to 3 decimal places
    def auto_drift_correct_rms(self, prange=None, times=None, ptrange=None, plot=False, 
                               slopes=None, line_name="MnKAlpha"):
        """Apply a correction for pulse variation with pretrigger mean, which we've found
        to be a pretty good indicator of drift.  Use the rms width of the Mn Kalpha line
        rather than actually fitting for the resolution.  (THIS IS THE OLD WAY TO DO IT.
        SUGGEST YOU USE self.auto_drift_correct instead....)
        
        prange:  use only filtered values in this range for correction 
        ptrange: use only pretrigger means in this range for correction
        times: if not None, use this range of p_timestamps instead of all data (units are seconds
               since server started--ugly but that's what we have to work with)
        plot:  whether to display the result
        line_name: Line to calibrate on, if prange is None 
        ===============================================
        returns best_slope 
        units = 
        """
        if plot:
            plt.clf()
            axis1=plt.subplot(211)
            plt.xlabel("Drift correction slope")
            plt.ylabel("RMS of selected, corrected pulse heights")
        if self.p_filt_value_phc[0] ==0:
            self.p_filt_value_phc = self.p_filt_value.copy()
        
        # Default: use the calibration to pick a prange
        if prange is None:
            calibration = self.calibration['p_filt_value']
            ph_estimate = calibration.name2ph(line_name)
            prange = np.array((ph_estimate*.99, ph_estimate*1.01))
        
        range_ctr = 0.5*(prange[0]+prange[1])
        half_range = np.abs(range_ctr-prange[0])
        valid = np.logical_and(self.cuts.good(), np.abs(self.p_filt_value_phc-range_ctr)<half_range)
        if times is not None:
            valid = np.logical_and(valid, self.p_timestamp<times[1])
            valid = np.logical_and(valid, self.p_timestamp>times[0])
            
        if ptrange is not None:
            valid = np.logical_and(valid, self.p_pretrig_mean<ptrange[1])
            valid = np.logical_and(valid, self.p_pretrig_mean>ptrange[0])

        data = self.p_filt_value_phc[valid]
        corrector = self.p_pretrig_mean[valid]
        mean_pretrig_mean = corrector.mean()
        corrector -= mean_pretrig_mean
        if slopes is None: slopes = np.arange(-.2,.9,.05)
        rms_widths=[]
        for sl in slopes:
            rms = (data+corrector*sl).std()
            rms_widths.append(rms)
#            print "%6.3f %7.2f"%(sl,rms)
            if plot: 
                plt.plot(sl,rms,'bo')
        poly_coef = sp.polyfit(slopes, rms_widths, 2)
        best_slope = -0.5*poly_coef[1]/poly_coef[0]
        print "Drift correction requires slope %6.3f"%best_slope
        self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
        
        if plot:
            plt.subplot(212)
            plt.plot(corrector, data, ',')
            xlim = plt.xlim()
            c = np.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
            plt.plot(c, -c*best_slope + data.mean(),color='green')
            plt.ylim(prange)
            axis1.plot(slopes, np.poly1d(poly_coef)(slopes),color='red')
            plt.xlabel("Pretrigger mean - mean(PT mean)")
            plt.ylabel("Selected, uncorrected pulse heights")
        return best_slope
    
           
    def auto_drift_correct(self, prange=None, times=None, plot=False, slopes=None, line_name='MnKAlpha'):
        """Apply a correction for pulse variation with pretrigger mean.
        This attempts to replace the previous version by using a fit to the
        Mn K alpha complex
        
        prange:  use only filtered values in this range for correction 
        times: if not None, use this range of p_timestamps instead of all data (units are seconds
               since server started--ugly but that's what we have to work with)
        plot:  whether to display the result
        line_name: name of the element whose Kalpha complex you want to fit for drift correction
        """

        if self.p_filt_value_phc[0] ==0:
            self.p_filt_value_phc = self.p_filt_value.copy()
        
        # Default: use the calibration to pick a prange
        if prange is None:
            calibration = self.calibration['p_filt_value']
            ph_estimate = calibration.name2ph('MnKAlpha')
            prange = np.array((ph_estimate*.99, ph_estimate*1.01))
        
        range_ctr = 0.5*(prange[0]+prange[1])
        half_range = np.abs(range_ctr-prange[0])
        valid = np.logical_and(self.cuts.good(), np.abs(self.p_filt_value_phc-range_ctr)<half_range)
        if times is not None:
            valid = np.logical_and(valid, self.p_timestamp<times[1])
            valid = np.logical_and(valid, self.p_timestamp>times[0])

        data = self.p_filt_value_phc[valid]
        corrector = self.p_pretrig_mean[valid]
        mean_pretrig_mean = corrector.mean()
        corrector -= mean_pretrig_mean
        if slopes is None: slopes = np.arange(0,1.,.09)
        
        fit_resolutions=[]
        for sl in slopes:
            self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*sl
            params,_covar,_fitter = self.fit_spectral_line(prange=prange, times=times, plot=False,
                                                   fit_type='dc', line=line_name, verbose=False)
#            print "%5.1f %s"%(sl, params[:4])
            fit_resolutions.append(params[0])
#        print(fit_resolutions)
        poly_coef = sp.polyfit(slopes, fit_resolutions, 2)
#        best_slope = -0.5*poly_coef[1]/poly_coef[0] # this could be better in principle, but in practice is often way worse
        # some code to check if the best slope from the quatratic fit is reasonable, like near the minimum could
        # be used to get the best of both worlds
        # or start with a sweep then add a binary search at the end
        best_slope = slopes[np.argmin(fit_resolutions)]
        best_slope_resolution = np.interp(best_slope, slopes, fit_resolutions)
        
        print "Drift correction requires slope (using min not quadratic fit) %6.3f"%best_slope
        self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
        
        if plot:
            plt.clf()
            plt.subplot(211)
            plt.plot(slopes, fit_resolutions,'go')
            plt.plot(best_slope, best_slope_resolution,'bo')
            plt.plot(slopes, np.polyval(poly_coef, slopes),color='red')
            plt.xlabel("Drift correction slope")
            plt.ylabel("Fit resolution from selected, corrected pulse heights")
            plt.title('auto_drift_correct fitting %s'%line_name)
            
            plt.subplot(212)
            plt.plot(corrector, data, ',')
            xlim = plt.xlim()
            c = np.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
            plt.plot(c, -c*best_slope + data.mean(),color='green')
            plt.ylim(prange)
            plt.xlabel("Pretrigger mean - mean(PT mean)")
            plt.ylabel("Selected, uncorrected pulse heights")
            
        return best_slope, mean_pretrig_mean


    def fit_spectral_line(self, prange, mask=None, times=None, fit_type='dc', line='MnKAlpha', 
                          nbins=200, verbose=True, plot=True, **kwargs):
        """
        <line> can be one of the fitters in mass.calibration.fluorescence_lines (e.g. 'MnKAlpha', 'CuKBeta') or
        in mass.calibration.gaussian_lines (e.g. 'Gd97'), or a number.  In this last case, it is assumed to
        be a single Gaussian line.
        """
        all_values={'filt': self.p_filt_value,
                    'phc': self.p_filt_value_phc,
                    'dc': self.p_filt_value_dc,
                    'energy': self.p_energy,
                    }[fit_type]
        if mask is not None:
            valid = np.array(mask)
        else:
            valid = self.cuts.good()
        if times is not None:
            valid = np.logical_and(valid, self.p_timestamp<times[1])
            valid = np.logical_and(valid, self.p_timestamp>times[0])
        good_values = all_values[valid]
        contents,bin_edges = np.histogram(good_values, nbins, prange)
        if verbose: print "%d events pass cuts; %d are in histogram range"%(len(good_values),contents.sum())
        bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        # Try line first as a number, then as a fluorescence line, then as a Gaussian
        try:
            energy = float(line)
            module = 'mass.calibration.gaussian_lines'
            fittername = '%s.GaussianFitter(%s.GaussianLine())'%(module,module)
            fitter = eval(fittername)
        except ValueError:
            energy = None
            try:
                module = 'mass.calibration.fluorescence_lines'
                fittername = '%s.%sFitter()'%(module,line)
                fitter = eval(fittername)
            except AttributeError:
                try:
                    module = 'mass.calibration.gaussian_lines'
                    fittername = '%s.%sFitter()'%(module,line)
                    fitter = eval(fittername)
                except AttributeError:
                    raise ValueError("Cannot understand line=%s as an energy or a known calibration line."%line)

        params, covar = fitter.fit(contents, bin_ctrs, plot=plot, **kwargs)
        if plot:
            mass.plot_as_stepped_hist(plt.gca(), contents, bin_ctrs)
        if energy is not None:
            scale = energy/params[1]
        else:
            scale=1.0
        if verbose: print 'Resolution: %5.2f +- %5.2f eV'%(params[0]*scale,np.sqrt(covar[0,0])*scale)
        return params, covar, fitter
    
    
    def fit_MnK_lines(self, mask=None, times=None, update_energy=True, verbose=False, plot=True):
        """"""
        
        if plot:
            plt.clf()
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
        else:
            ax1 = ax2 = ax3 = None
        
        calib = self.calibration['p_filt_value']
        mnka_range = calib.name2ph('MnKAlpha') * np.array((.99,1.01))
        params, _covar, _fitter = self.fit_spectral_line(prange=mnka_range, mask=mask, times=times, fit_type='dc', line='MnKAlpha', verbose=verbose, plot=plot, axis=ax1)
        calib.add_cal_point(params[1], 'MnKAlpha')

        mnkb_range = calib.name2ph('MnKBeta') * np.array((.95,1.02))
#        params[1] = calib.name2ph('Mn Kb')
#        params[3] *= 0.50
#        params[4] = 0.0
        try:
            mnkb_range = calib.name2ph('MnKBeta') * np.array((.985,1.015))
            params, _covar, _fitter = self.fit_spectral_line(prange=mnkb_range, mask=mask, times=times, fit_type='dc', line='MnKBeta', 
                                                    verbose=verbose, plot=plot, axis=ax2)
            calib.add_cal_point(params[1], 'MnKBeta')
        except sp.linalg.LinAlgError:
            print "Failed to fit Mn K-beta!"
        if update_energy: self.p_energy = calib(self.p_filt_value_dc)
        
        if plot:
            calib.plot(axis=plt.subplot(224))
            self.fit_spectral_line(prange=(5850,5930), mask=mask, times=times, fit_type='energy', line='MnKAlpha', verbose=verbose, plot=plot, axis=ax3)
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
    if noisename is None and not pulse_only:
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


def unpickle_MicrocalDataSet(filename):
    """
    Factory function to unpickle a MicrocalDataSet pickled by its .pickle() method.
    Note that you might be better off creating a MicrocalDataSet the usual way, then
    loading it with its .unpickle() method.
    
    Data structure must be:
    1. A dictionary with simple values, whose keys include at least all strings in
       the tuple MicrocalDataSet.expected_attributes.
    2. The dataset cuts._mask
    3. Any string.  If #4 also loads, then this will be the attribute name.
    4. Any pickleable object.  This will become an attribute value (prev item gives its name)
    ... Repeat items (3,4) as many times as necessary to load attribute (name,value) pairs.
    """
    import cPickle
    fp = open(filename, "rb")
    unpickler = cPickle.Unpickler(fp)
    expected_attr = unpickler.load()
    ds = MicrocalDataSet(expected_attr)
    ds.cuts._mask = unpickler.load()
    try:
        while True:
            k = unpickler.load()
            v = unpickler.load()
            ds.__dict__[k] = v
    except EOFError:
        pass
    fp.close()
    return ds
