"""
Created on Feb 16, 2011

@author: fowlerj
"""

import numpy
import scipy.linalg
import scipy.optimize
from matplotlib import pylab
import os.path

try:
    import cPickle as pickle
except ImportError:
    import pickle    

# MASS modules
import controller
import files
import power_spectrum
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
        self.datafile = files.LJHFile(filename, segmentsize=50000000)
        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples","nPresamples","nPulses", "timebase"):
            self.__dict__[attr] = self.datafile.__dict__[attr]

        for first_pnum, end_pnum, seg_num, data in self.datafile.iter_segments():
            if seg_num > 0 or first_pnum>0 or end_pnum != self.nPulses:
                msg = "NoiseRecords objects can't (yet) handle multi-segment noise files.\n"+\
                    "This is file %s seg_num %d with first/end pulse# %d, %d"%(
                    self.filename,seg_num, first_pnum, end_pnum)
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
        spec = power_spectrum.PowerSpectrum(self.nSamples/2, dt=self.timebase)
        if window is None:
            window = numpy.ones(self.nSamples)
        else:
            window = window(self.nSamples)
        for d in self.data:
            spec.addDataSegment(d-d.mean(), window=window)
        self.spectrum = spec
        if plot:
            self.plot_power_spectrum()


    def plot_power_spectrum(self, axis=None):
        if self.spectrum is None:
            self.compute_power_spectrum(plot=False)
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        spec = self.spectrum
        axis.plot(spec.frequencies()[1:], spec.spectrum()[1:])
        pylab.loglog()
        axis.grid()
        axis.set_xlim([10,3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title("Noise power spectrum for %s"%self.filename)

        
    def compute_autocorrelation(self, n_lags=None, plot=True):
        """
        Compute the autocorrelation averaged across all "pulses" in the file.
        """
        
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
            
        
        if self.continuous:
            n_data = self.nSamples*self.nPulses
            if n_lags is None:
                n_lags = n_data
            if n_lags > n_data:
                n_lags = n_data

            paddedData = numpy.zeros(padded_length(n_lags+n_data), dtype=numpy.float)
            paddedData[:n_data] = numpy.array(self.data.ravel())[:n_data] - self.data.mean()
            paddedData[n_data:] = 0.0
            
            ft = numpy.fft.rfft(paddedData)
            del paddedData
            ft[0] = 0 # this redundantly removes the mean of the data set
            ft = (ft*ft.conj()).real
            acsum = numpy.fft.irfft(ft)
            ac = acsum[:n_lags+1] / (n_data-numpy.arange(n_lags+1.0))
         
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
    
    
    def __init__(self, filename):
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

        self.__open_file(filename)
#        self.__setup_vectors()

    def __open_file(self, filename):
        """Detect the filetype and open it."""

        # For now, we have only one file type, so let's just assume it!
        self.datafile = files.LJHFile(filename)
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


    def DEPRECATED_summarize_data(self):
        """Summarize the complete data file"""
        
        maxderiv_holdoff = int(100e-6/self.timebase) # don't look for retriggers before this # of samples

        for first, end, segnum, data in self.datafile.iter_segments():
            print 'Read segment #%2d with %d pulses'%(segnum, self.datafile.datatimes.shape[0])

            self.p_timestamp[first:end] = self.datafile.datatimes
            self.p_peak_index[first:end] = data.argmax(axis=1)
            self.p_peak_value[first:end] = data.max(axis=1)
            self.p_min_value[first:end] = data.min(axis=1)
            self.p_pretrig_mean[first:end] = data[:,:self.nPresamples-2].mean(axis=1)
            self.p_pretrig_rms[first:end] = data[:,:self.nPresamples-2].std(axis=1)
            self.p_pulse_average[first:end] = data[:,self.nPresamples:].mean(axis=1)
            self.p_pulse_average[first:end] -= self.p_pretrig_mean[first:end]

            for pulsenum,pulse in enumerate(data):
                self.p_rise_time[first+pulsenum] = estimateRiseTime(pulse, 
                                                    dt=self.timebase, nPretrig = self.nPresamples)
                self.p_max_posttrig_deriv[first+pulsenum] = \
                    compute_max_deriv(pulse[self.nPresamples + maxderiv_holdoff:])
                    
        # Careful: p_peak_index is unsigned, so make it signed before subtracting nPresamples:
        self.p_peak_time = (numpy.asarray(self.p_peak_index, dtype=numpy.int)-self.nPresamples)*self.timebase



#    def find_data(self, *args):
#        """"Return an array of pulses satisfying a certain criterion (or several).
#        
#        Example:  pulse_record.find_data('self.p_peak_time > .250', 'self.p_rise_time>1e-3')"""
#        
#        good = numpy.array(self.nPulses, dtype=numpy.bool)
#        good = numpy.arange(self.nPulses)
#        for test in args:
#            test_result = eval('numpy.arange(self.nPulses)[%s]'%test, None, locals())
#            print test_result
#            good = numpy.logical_and(good, test_result)
#            print good
#        return numpy.arange(self.nPulses)[good]
        

    def DEPRECATED_plot_summaries(self, valid='uncut', downsample=None, log=False):
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
                print "Plotting only uncut data"
            elif "cut" in valid.lower():
                valid = self.cuts.bad()  
                print "Plotting only cut data"
            else:
                raise ValueError("If valid is a string, it must contain 'uncut' or 'cut'.")
                
        if valid is not None:
            if downsample is None:
                downsample=valid.sum()/10000
            hour = self.p_timestamp[valid][::downsample]/3.6e6
        else:
            if downsample is None:
                downsample = self.nPulses / 10000
            hour = self.p_timestamp[::downsample]/3.6e6
    
        plottables = (
            (self.p_pulse_average, 'Pulse Avg', 'purple', None),
            (self.p_pretrig_rms, 'Pretrig RMS', 'blue', [0,4000]),
            (self.p_pretrig_mean, 'Pretrig Mean', 'green', [0,6000]),
            (self.p_peak_value, 'Peak value', '#88cc00',[0,10000]),
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

    
    def DEPRECATED_plot_traces(self, pulsenums, pulse_summary=True, axis=None):
        """Plot some example pulses, given by sample number.
        <pulsenums>  A sequence of sample numbers, or a single one.
        
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A pylab axis to plot on.
        """
        if isinstance(pulsenums, int):
            pulsenums = (pulsenums,)
        pulsenums = numpy.asarray(pulsenums)
            
        dt = (numpy.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        color= 'magenta','purple','blue','green','#88cc00','gold','orange','red', 'brown','gray','#444444'
        MAX_TO_SUMMARIZE = 20
        
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        axis.set_xlabel("Time after trigger (ms)")
        axis.set_ylabel("Feedback (or mix) in [Volts/16384]")
        if pulse_summary:
            axis.text(.975, .97, r"              -PreTrigger-   Max  Rise t Peak   Pulse", 
                       size='medium', family='monospace', transform = axis.transAxes, ha='right')
            axis.text(.975, .95, r"Cut P#    Mean     rms PTDeriv  ($\mu$s) value   mean", 
                       size='medium', family='monospace', transform = axis.transAxes, ha='right')

        cuts_good = self.cuts.good()[pulsenums]
        for i,pn in enumerate(pulsenums):
            data = self.datafile.read_trace(pn)
            cutchar,alpha,linestyle,linewidth = ' ',1.0,'-',1
            if not cuts_good[i]:
                cutchar,alpha,linestyle,linewidth = 'X',1.0,'--' ,1
            axis.plot(dt, data, color=color[i%len(color)], linestyle=linestyle, alpha=alpha,
                       linewidth=linewidth)
            if pulse_summary and i<MAX_TO_SUMMARIZE:
                summary = "%s%6d: %5.0f %7.2f %6.1f %5.0f %5.0f %7.1f"%(
                            cutchar, pn, self.p_pretrig_mean[pn], self.p_pretrig_rms[pn],
                            self.p_max_posttrig_deriv[pn], self.p_rise_time[pn]*1e6,
                            self.p_peak_value[pn], self.p_pulse_average[pn])
                axis.text(.975, .93-.02*i, summary, color=color[i%len(color)], 
                           family='monospace', size='medium', transform = axis.transAxes, ha='right')


    def DEPRECATED_compute_average_pulse(self, controls=None):
        
        if controls is None:
            controls = controller.standardControl()
            
        ph_bins = controls.analysis_prm['pulse_averaging_ranges']
        nbins = ph_bins.shape[0]
        pulse_sums = numpy.zeros((nbins,self.nSamples), dtype=numpy.float)
        pulse_counts = numpy.zeros(nbins)
        
        self.good = self.cuts.good()
        for first, end, _seg_num, data in self.datafile.iter_segments():
#            n = self.datafile.data.shape[0]
            for ibin, bin in enumerate(ph_bins):
                bin_ctr = 0.5*(bin[0]+bin[1])
                bin_hw = numpy.abs(bin_ctr-bin[0])
                cuts = numpy.logical_and(
                        numpy.abs(bin_ctr - data.max(axis=1)) < bin_hw,
                        self.good[first:end])
                good_pulses = data[cuts, :]
                pulse_counts[ibin] += good_pulses.shape[0]
                pulse_sums[ibin,:] += good_pulses.sum(axis=0)

        self.average_pulse = (pulse_sums.T/pulse_counts).T


    def DEPRECATED_filter_data(self, filter):
        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = numpy.array((
                (-6,24,34,24,-6),
                (-14,-7,0,7,14),
                (10,-5,-10,-5,10)), dtype=numpy.float)/35.0
        
        assert len(filter)+4 == self.nSamples
        peak_x = numpy.zeros(self.nPulses, dtype=numpy.float)
        peak_y = numpy.zeros(self.nPulses, dtype=numpy.float)
        for first, end, _seg_num, data in self.datafile.iter_segments():
            conv = numpy.zeros((5, end-first), dtype=numpy.float)
            for i in range(5):
                if i-4 == 0:
                    conv[i,:] = (filter*data[:,i:]).sum(axis=1)
                else:
                    conv[i,:] = (filter*data[:,i:i-4]).sum(axis=1)
            param = numpy.dot(fit_array, conv)
            peak_x[first:end] = -0.5*param[1,:]/param[2,:]
            peak_y[first:end] = param[0,:] - 0.25*param[1,:]**2 / param[2,:] 
        return peak_x, peak_y
        
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
    """A set of optimal filters based on a single signal and noise set"""

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
                         Either or both of <fmax> and <f_3db> are allowed. *NEITHER ARE IMPLMENETED YET*.
        <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                         This must be given if <fmax> or <f_3db> are ever to be used.
        <shorten>        The time-domain filters should be shortened by removing this many
                         samples from each end.  (Do this for convenience of convolution over
                         multiple lags.)
        """
        self.sample_time = sample_time
        self.shorten = shorten
        pre_avg = avg_signal[:n_pretrigger].mean()
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
            raise ValueError("Filter must have noise_psd and/or noise_autocorr arguments")
        
        self.compute(fmax=fmax, f_3db=f_3db)

        
    def compute(self, fmax=None, f_3db=None):
        """"""
#        if fmax is not None:
#            raise NotImplementedError("Use of fmax on Filters is not yet supported.")
#        if f_3db is not None:
#            raise NotImplementedError("Use of f_3db on Filters is not yet supported.")
        
        self.variances={}
        
        def normalize_filter(q): 
            if len(q) == len(self.avg_signal):
                q *= 1 / numpy.dot(q, self.avg_signal)
            else:  
                q *= 1 / numpy.dot(q, self.avg_signal[self.shorten:-self.shorten]) 
                
        # Fourier domain filters
        if self.noise_psd is not None:
            n = len(self.noise_psd)
            sig_ft = numpy.fft.rfft(self.avg_signal)
            if len(sig_ft) != n:
                raise ValueError("signal real DFT and noise PSD are not the same length (%d and %d)"
                                 %(len(sig_ft), n))
            
            sig_ft /= self.noise_psd
            sig_ft[0] = 0.0
            
            # Band-limit
            if fmax is not None or f_3db is not None:
                freq = numpy.fft.fftfreq(n, d=self.sample_time) 
                freq=freq[:n/2+1]
                freq[-1] *= -1
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1+(freq/f_3db)**2)

            self.filt_fourier = numpy.fft.irfft(sig_ft)
            normalize_filter(self.filt_fourier)
        
        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2*self.shorten
            if self.shorten>0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            assert len(self.noise_autocorr) >= n
            R =  scipy.linalg.toeplitz(self.noise_autocorr[:n]/self.peak_signal**2)
            Rinv_sig = numpy.linalg.solve(R, avg_signal)
            Rinv_1 = numpy.linalg.solve(R, numpy.ones(n))
            
            self.filt_noconst = Rinv_1.sum()*Rinv_sig - Rinv_sig.sum()*Rinv_1

            # Band-limit
            if fmax is not None or f_3db is not None:
                sig_ft = numpy.fft.rfft(self.filt_noconst)
                freq = numpy.fft.fftfreq(n, d=self.sample_time) 
                freq=freq[:n/2+1]
                freq[-1] *= -1
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1+(freq/f_3db)**2)
                self.filt_noconst = numpy.fft.irfft(sig_ft)

            normalize_filter(self.filt_noconst)

            self.filt_baseline = numpy.dot(avg_signal, Rinv_sig)*Rinv_1 - Rinv_sig.sum()*Rinv_sig
            self.filt_baseline /=  self.filt_baseline.sum()
            
            Rpretrig = scipy.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger]/self.peak_signal**2)
            self.filt_baseline_pretrig = numpy.linalg.solve(Rpretrig, numpy.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()

            bracketR = lambda a, R: numpy.dot(a, numpy.dot(R, a))            
            self.variances['noconst'] = bracketR(self.filt_noconst, R) 
            self.variances['baseline'] = bracketR(self.filt_baseline, R)
            self.variances['baseline_pretrig'] = bracketR(self.filt_baseline_pretrig, Rpretrig)
            if self.noise_psd is not None:
                R =  scipy.linalg.toeplitz(self.noise_autocorr[:len(self.filt_fourier)]/self.peak_signal**2)
                self.variances['fourier'] = bracketR(self.filt_fourier, R)
            
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

        expected_attributes=("nSamples","nPresamples","nPulses","timebase")
        for a in expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename','virtual data set')
        self.__setup_vectors()
        self.gain = 1.0


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
        return c

    
    def summarize_data(self, first, end):
        """Summarize the complete data file"""
        
        if first >= self.nPulses:
            return
        maxderiv_holdoff = int(220e-6/self.timebase) # don't look for retriggers before this # of samples

        seg_size = end-first
        self.p_timestamp[first:end] = self.times[:seg_size]
        self.p_pretrig_mean[first:end] = self.data[:seg_size,:self.nPresamples-2].mean(axis=1)
        self.p_pretrig_rms[first:end] = self.data[:seg_size,:self.nPresamples-2].std(axis=1)
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
                ( 10,-5,-10,-5,10)), dtype=numpy.float)/35.0
        
        assert len(filter)+4 == self.nSamples

        seg_size = min(end-first, self.data.shape[0])
        conv = numpy.zeros((5, seg_size), dtype=numpy.float)
        for i in range(5):
            if i-4 == 0:
                conv[i,:] = (filter*self.data[:,i:]).sum(axis=1)
            else:
                print conv[i,:].shape, self.data.shape, (filter*self.data[:,i:i-4]).shape
                conv[i,:] = (filter*self.data[:,i:i-4]).sum(axis=1)
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
    
    
    def phase_correct(self, ylim=None, plot=True):
        """Apply a correction for pulse variation with arrival phase"""
        
        # Choose number and size of bins
        phase_step=.05
        nstep = int(.5+1.0/phase_step)
        phases = (0.5+numpy.arange(nstep))/nstep - 0.5
        phase_step = 1.0/nstep
        
        # Plot the raw filtered value vs phase
        if plot:
            pylab.clf()
            pylab.subplot(211)
            pylab.plot((self.p_filt_phase+.5)%1-.5, self.p_filt_value,'.',color='orange')
            pylab.xlim([-.55,.55])
            if ylim is not None:
                pylab.ylim(ylim)
                
        # Estimate corrections in a few different pieces
        corrections = []
        valid = self.cuts.good()
        if ylim is not None:
            valid = numpy.logical_and(valid, self.p_filt_value<ylim[1])
            valid = numpy.logical_and(valid, self.p_filt_value>ylim[0])

        for ctr_phase in phases:
            valid_ph = numpy.logical_and(valid,
                                         numpy.abs((self.p_filt_phase - ctr_phase)%1) < phase_step*0.5)
            print valid_ph.sum(),"   ",
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
        pylab.plot(phases, model(fitparams, phases), color='blue')
        
        self.phase_correction={'phase':fitparams[0],
                            'amplitude':fitparams[1],
                            'mean':fitparams[2]}
        fitparams[2] = 0
        self.p_filt_value_phc = self.p_filt_value - model(fitparams, self.p_filt_phase)
        if plot:
            pylab.subplot(212)
            pylab.plot((self.p_filt_phase+.5)%1-.5, self.p_filt_value_phc,'.b')
            pylab.xlim([-.55,.55])
            if ylim is not None:
                pylab.ylim(ylim)


    def fit_mn_kalpha(self, prange, type='phc',**kwargs):
        all_values={'filt': self.p_filt_value,
                    'phc': self.p_filt_value_phc,
                    'dc': self.p_filt_value_dc,
                    }[type]
        good_values = all_values[self.cuts.good()]
        contents,bin_edges = numpy.histogram(good_values, 200, prange)
        print "%d events pass cuts; %d are in histogram range"%(len(good_values),contents.sum())
        bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])
        fitter = fluorescence_lines.MnKAlphaFitter()
        params, covar = fitter.fit(contents, bin_ctrs, **kwargs)
        print 'Resolution: %5.2f +- %5.2f eV'%(params[0],numpy.sqrt(covar[0,0]))



def create_pulse_and_noise_records(fname, noisename=None, records_are_continuous=True, 
                                   noise_only=False):
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
    
    if noise_only:
        pr = PulseRecords(fname)
        nr = NoiseRecords(fname, records_are_continuous)
    else:
        pr = PulseRecords(fname)
        nr = NoiseRecords(noisename, records_are_continuous)
    
    return (pr, nr)
