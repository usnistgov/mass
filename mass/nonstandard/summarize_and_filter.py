import numpy
import mass
from mass.core.utilities import InlineUpdater

__all__ = ['summarize_old']


def MicrocalDataSet_summarize_data_tdm(self, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0, forceNew = False):
    """summarized the complete data file one chunk at a time
    this version does the whole dataset at once (instead of previous segment at a time for all datasets)
    """
    if len(self.p_timestamp) < self.pulse_records.nPulses:
        self.__setup_vectors(nPulses=self.pulse_records.nPulses)
    elif forceNew or all(self.p_timestamp==0):
        self.pretrigger_ignore_samples = int(pretrigger_ignore_microsec*1e-6/self.timebase)   
        # consider setting segment size first
        printUpdater = InlineUpdater('channel.summarize_data_tdm chan %d'%self.channum)

        for s in range(self.pulse_records.n_segments):
            first, last = self.pulse_records.read_segment(s) # this reloads self.data to contain new pulses
            self.p_timestamp[first:last] = self.pulse_records.datafile.datatimes_float
            (self.p_pretrig_mean[first:last], self.p_pretrig_rms[first:last],
            self.p_peak_index[first:last], self.p_peak_value[first:last], self.p_min_value[first:last],
            self.p_pulse_average[first:last], self.p_rise_time[first:last], 
            self.p_max_posttrig_deriv[first:last]) = mass.nonstandard.summarize_and_filter.summarize_old(self.pulse_records.data, 
                self.nPresamples, self.pretrigger_ignore_samples, self.timebase, peak_time_microsec)
            printUpdater.update((s+1)/float(self.pulse_records.n_segments))
        self.pulse_records.datafile.clear_cached_segment()      
        if self.auto_pickle:
            self.pickle(verbose=False)
    else:
        print('\nchan %d did not summarize because results were already preloaded'%self.channum)


def MicrocalDataSet_filter_data_tdm(self, filter_name='filt_noconst', transform=None, forceNew=False):
    """filter the complete data file one chunk at a time
    this version does the whole dataset at once (instead of previous segment at a time for all datasets)
    """
    filter_values = self.filter.__dict__[filter_name]
    if forceNew or all(self.p_filt_value == 0): # determine if we need to do anything
        printUpdater = InlineUpdater('channel.filter_data_tdm chan %d'%self.channum)
        for s in range(self.pulse_records.n_segments):
            first, last = self.pulse_records.read_segment(s) # this reloads self.data to contain new pulses
            (self.p_filt_phase[first:last], self.p_filt_value[first:last]) = mass.nonstandard.summarize_and_filter.filter_data_old(
            filter_values, self.pulse_records.data, transform, self.p_pretrig_mean[first:last])
            printUpdater.update((s+1)/float(self.pulse_records.n_segments))
            
        self.pulse_records.datafile.clear_cached_segment()    
        if self.auto_pickle:
            self.pickle(verbose=False)  
    else:
        print('\nchan %d did not filter because results were already loaded'%self.channum)
        


try:
    from numba import autojit
    @autojit
    def summarize_numba(data, nPresamples, pretrigger_ignore_samples):
        # this is faster mainly because it only loops through each pulse once, 
        # instead of six times with the 6 separate numpy functions
        # numba recompiles it in c or something so its not running slow 
        # python loops, but is still single threaded
        numPulses = data.shape[0]
        pulseLength = data.shape[1]
        pretrig_end_average_index = nPresamples-pretrigger_ignore_samples
        
        max = numpy.empty(numPulses, dtype=data.dtype)
        max[:] = data[:,0]
        
        min = numpy.empty(numPulses, dtype=data.dtype)
        min[:] = data[:,0]
        
        posttrig_sum = numpy.zeros(numPulses)
        pretrig_sum = numpy.zeros(numPulses)
        pretrig_sumsq = numpy.zeros(numPulses)
    
        argmax = numpy.zeros(numPulses)
        argmin = numpy.zeros(numPulses)
        
        for p in xrange(numPulses):
            for j in xrange(pretrig_end_average_index):
                d = data[p, j]
                pretrig_sum[p] += d
                pretrig_sumsq[p] += d*d
                if d > max[p]: 
                    max[p] = d
                    argmax[p] = j
                elif d < min[p]: 
                    min[p] = d
                    argmin[p] = j
            for j in xrange(pretrig_end_average_index, nPresamples):
                d = data[p, j]
                if d > max[p]: 
                    max[p] = d
                    argmax[p] = j
                elif d < min[p]: 
                    min[p] = d
            for j in xrange(nPresamples, pulseLength):
                d = data[p, j]
                posttrig_sum[p] +=d
                if d > max[p]: 
                    max[p] = d
                    argmax[p] = j
                elif d < min[p]: 
                    min[p] = d
    
                    
        p_pretrig_mean = pretrig_sum/pretrig_end_average_index
        p_pretrig_rms = numpy.sqrt((pretrig_sumsq/pretrig_end_average_index) - (p_pretrig_mean*p_pretrig_mean))
        p_pulse_average = posttrig_sum/(pulseLength-nPresamples) - p_pretrig_mean
        p_peak_value = max - p_pretrig_mean
        
        return p_pretrig_mean, p_pretrig_rms, argmax, p_peak_value, min, p_pulse_average

    __all__.append('summarize_numba')
    
except ImportError:
    print('numba not installed, disabled summarize_numba')
    
except Exception, e:
    print("Galen's summarize_numba did not compile, even though numba is installed.")
    print("Please suggest to Galen that he not commit code without testing it.")


def summarize_old(data, nPresamples, pretrigger_ignore_samples, timebase, peak_time_microsec):
    p_pretrig_mean = data[:,:nPresamples-pretrigger_ignore_samples].mean(axis=1, dtype=numpy.float32)
    p_pretrig_rms = data[:,:nPresamples-pretrigger_ignore_samples].std(axis=1, dtype=numpy.float32)
    p_peak_index = numpy.array(data.argmax(axis=1), dtype=numpy.uint16)
    p_peak_value = data.max(axis=1)
    p_min_value = data.min(axis=1)
    p_pulse_average = data[:,nPresamples:].mean(axis=1, dtype=numpy.float32)
    
    # Remove the pretrigger mean from the peak value and the pulse average figures. 
    p_peak_value -= p_pretrig_mean
    p_pulse_average -= p_pretrig_mean
    
    maxderiv_holdoff_samples = int(peak_time_microsec*1e-6/timebase) # don't look for retriggers before this # of samples
    # Compute things that have to be computed one at a time:
    p_rise_time = numpy.zeros_like(p_pulse_average)
    p_max_posttrig_deriv = numpy.zeros_like(p_pulse_average)
    for pulsenum,pulse in enumerate(data):
        p_rise_time[pulsenum] = estimateRiseTime(pulse, dt=timebase, nPretrig = nPresamples)
        p_max_posttrig_deriv[pulsenum] = compute_max_deriv(pulse[nPresamples + maxderiv_holdoff_samples:])
#        self.p_timestamp[first:last], self.p_pretrig_mean[first:last], self.p_pretrig_rms[first:last],
#        self.p_peak_index[first:last], self.p_peak_value[first:last], self.p_min_value[first:last],
#        self.p_pulse_average[first:last], self.p_rise_time[first:last], 
#        self.p_max_posttrig_deriv[first:last]
    return p_pretrig_mean, p_pretrig_rms, p_peak_index, p_peak_value, p_min_value, p_pulse_average, p_rise_time, p_max_posttrig_deriv

def compare_summarize(data, nPresamples, pretrigger_ignore_samples):
    import time
    out_numba = summarize_numba(numpy.random.rand(100,500), 5, 0) # forces numba to compile summarize_numba

    t0 = time.time()
    out_numba = summarize_numba(data, nPresamples, pretrigger_ignore_samples)
    t_numba = time.time()-t0
    t0 = time.time()
    out_old = summarize_old(data, nPresamples, pretrigger_ignore_samples)
    t_old = time.time()-t0
    
    print('for data.shape=(%d, %d), nPresamples = %d, pretrigger_ignore_samples = %d'%(data.shape[0], data.shape[1], nPresamples, pretrigger_ignore_samples))
    varnames = ['p_pretrig_mean', 'p_pretrig_rms', 'p_peak_index', 'p_peak_value', 'p_min_value', 'p_pulse_average']
    for i in range(len(out_numba)):
        print(varnames[i])
        print(i, all(out_numba[i]==out_old[i]), numpy.max(out_numba[i]-out_old[i]))
    print('numba took %f ms, old took %f ms, data.dtype=%s'%(1e3*t_numba, 1e3*t_old, data.dtype))
    
def estimateRiseTime(pulse_data, dt=1.0, nPretrig=0):
    """Compute the rise time of timeseries <pulse_data>, where the time steps are <dt>.
    If <nPretrig> >= 4, then the samples pulse_data[:nPretrig] are averaged to estimate
    the baseline.  Otherwise, the minimum of pulse_data is assumed to be the baseline.
    
    Specifically, take the first and last of the rising points in the range of 
    10% to 90% of the peak value, interpolate a line between the two, and use its
    slope to find the time to rise from 0 to the peak.
    
    See also fitExponentialRiseTime for a more traditional (and computationally
    expensive) definition.
    """
    MINTHRESH, MAXTHRESH = 0.1, 0.9
    
    
    if nPretrig >= 4:
        baseline_value = pulse_data[0:nPretrig-5].mean()
    else:
        baseline_value = pulse_data.min()
        nPretrig = 0
    value_at_peak = pulse_data.max() - baseline_value
    idxpk = pulse_data.argmax()

    try:
        rising_data = (pulse_data[nPretrig:idxpk+1] - baseline_value) / value_at_peak
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
    filter_coef = numpy.array([ -0.45238,   -0.02381,    0.28571,    0.30952,   -0.11905,   ])[::-1]
    conv = numpy.convolve(ts[first:end], filter_coef, mode='valid')
    
    if return_index_too:
        return first + 2 + conv.argmax() # This would be the index.
    return conv.max()

def filter_data_old(filter_values, data, transform=None, ptmean = None):
    # These parameters fit a parabola to any 5 evenly-spaced points
    fit_array = numpy.array((
            ( -6,24, 34,24,-6),
            (-14,-7,  0, 7,14),
            ( 10,-5,-10,-5,10)), dtype=numpy.float)/70.0
    
    assert len(filter_values)+4 == data.shape[1]

    conv = numpy.zeros((5, data.shape[0]), dtype=numpy.float32)
    if transform is not None:
        ptmean.shape = (len(ptmean),1)
        data = transform(data-ptmean)
    for i in range(5):
        if i-4 == 0:
            # previous method in comments, converted to dot product based on ~30% speed boost in tests
#                    conv[i,:] = (filter_values*self.data[:seg_size,i:]).sum(axis=1)
            conv[i,:] = numpy.dot(data[:,i:], filter_values)
        else:
#                    conv[i,:] = (filter_values*self.data[:seg_size,i:i-4]).sum(axis=1)
            conv[i,:] = numpy.dot(data[:,i:i-4], filter_values)


    param = numpy.dot(fit_array, conv)
    peak_x = -0.5*param[1,:]/param[2,:] #phase
    peak_y = param[0,:] - 0.25*param[1,:]**2 / param[2,:] #amplitude
    return peak_x, peak_y


    
    
if __name__ == '__main__':
    data=numpy.array(numpy.random.rand(10000, 500)*16363, dtype='int16')
#    data= numpy.random.rand(10000, 500)
    compare_summarize(data[2000:6000,:], nPresamples=200, pretrigger_ignore_samples=4)