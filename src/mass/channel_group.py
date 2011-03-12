"""
channel_group.py

Part of the Microcalorimeter Analysis Software System (MASS).

This module defines classes that handle one or more TES data streams 
together.  While these classes are indispensable for code-
division multiplexed (CDM) systems, they are also useful for the
simpler time-division multiplexed (TDM) systems in that they allow
the same interface to handle both types of data.

That's the goal, at least.

Author: Joe Fowler, NIST

Started March 2, 2011
"""

import os.path
import numpy
from matplotlib import pylab
import scipy.linalg

import mass.channel
#import mass.controller



class BaseChannelGroup(object):
    """
    Provides the interface for a group of one or more microcalorimeters,
    whether the detectors are multiplexed with time division or code
    division.
    
    This is an abstract base class, and the appropriate concrete class
    is the TESGroup or the CDMGroup, depending on the multiplexing scheme. 
    """
    def __init__(self, filenames, noise_filenames):
        # Convert a single filename to a tuple of size one
        if isinstance(filenames, str):
            filenames = (filenames,)
            
        self.filenames = tuple(filenames)
        self.n_channels = len(self.filenames)
        self.noise_filenames = tuple(filenames)
        assert self.n_channels == len(self.noise_filenames)
        
        self.n_segments = None
        self._cached_segment = None
        self._cached_pnum_range = None
        self.pulses_per_seg = None
        self.filters = None
        self.colors=("blue", "#aaaa00","green","red")
        
        
    def iter_segments(self, first_seg=0, end_seg=-1):
        if end_seg < 0: 
            end_seg = self.n_segments
        for i in range(first_seg, end_seg):
            first_rnum, end_rnum = self.read_segment(i)
            yield first_rnum, end_rnum


    def summarize_data(self):
        """
        ...?
        """

        for first, end in self.iter_segments():
            if end>self.nPulses:
                end = self.nPulses 
            print "Records %d to %d loaded"%(first,end-1)
            for dset in self.datasets:
                dset.nPulses = self.nPulses
                dset.summarize_data(first, end)

    
    def read_trace(self, record_num, chan_num=0):
        """Read (from cache or disk) and return the pulse numbered <record_num> for channel
        number <chan_num>.  If this is a CDMGroup, then the pulse is the demodulated
        channel by that number."""
        seg_num = record_num / self.pulses_per_seg
        self.read_segment(seg_num)
        return self.datasets[chan_num].data[record_num % self.pulses_per_seg]
        
        
        
        
    def plot_traces(self, pulsenums, channum=0, pulse_summary=True, axis=None):
        """Plot some example pulses, given by sample number.
        <pulsenums>  A sequence of sample numbers, or a single one.
        
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A pylab axis to plot on.
        """
        if isinstance(pulsenums, int):
            pulsenums = (pulsenums,)
        pulsenums = numpy.asarray(pulsenums)
        dataset = self.datasets[channum]
            
        dt = (numpy.arange(dataset.nSamples)-dataset.nPresamples)*dataset.timebase*1e3
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

        cuts_good = dataset.cuts.good()[pulsenums]
        for i,pn in enumerate(pulsenums):
            data = self.read_trace(pn, channum)
            cutchar,alpha,linestyle,linewidth = ' ',1.0,'-',1
            if not cuts_good[i]:
                cutchar,alpha,linestyle,linewidth = 'X',1.0,'--' ,1
            axis.plot(dt, data, color=color[i%len(color)], linestyle=linestyle, alpha=alpha,
                       linewidth=linewidth)
            if pulse_summary and i<MAX_TO_SUMMARIZE:
                summary = "%s%6d: %5.0f %7.2f %6.1f %5.0f %5.0f %7.1f"%(
                            cutchar, pn, dataset.p_pretrig_mean[pn], dataset.p_pretrig_rms[pn],
                            dataset.p_max_posttrig_deriv[pn], dataset.p_rise_time[pn]*1e6,
                            dataset.p_peak_value[pn], dataset.p_pulse_average[pn])
                axis.text(.975, .93-.02*i, summary, color=color[i%len(color)], 
                           family='monospace', size='medium', transform = axis.transAxes, ha='right')


    def make_masks(self, pulse_avg_ranges=None, pulse_peak_ranges=None, 
                   use_gains=True, gains=None, cut_crosstalk=False):
        """Generate a sequence of masks for use in compute_average_pulses().
        
        <use_gains>   Rescale the pulses by a set of "gains", either from <gains> or from
                      the MicrocalDataSet.gain parameter if <gains> is None.
        <gains>       The set of gains to use, overriding the self.datasets[*].gain, if
                      <use_gains> is True.  (If False, this argument is ignored.)
        <cut_crosstalk> ????
        """
        
        nhits = numpy.array([d.p_pulse_average>50 for d in self.datasets]).sum(axis=0)
        masks = []
        if use_gains:
            if gains is None:
                gains = [d.gain for d in self.datasets]
        else:
            gains = numpy.ones(self.n_channels)
            
        if pulse_avg_ranges is not None:
            if isinstance(pulse_avg_ranges[0], int) and len(pulse_avg_ranges)==2:
                pulse_avg_ranges = tuple(pulse_avg_ranges) 
            for r in pulse_avg_ranges:
                middle = 0.5*(r[0]+r[1])
                abslim = 0.5*numpy.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = numpy.abs(dataset.p_pulse_average/gain-middle) <= abslim
                    if cut_crosstalk:
                        m = numpy.logical_and(m, nhits==1)
                    masks.append(m)
        elif pulse_peak_ranges is not None:
            if isinstance(pulse_peak_ranges[0], int) and len(pulse_peak_ranges)==2:
                pulse_peak_ranges = tuple(pulse_peak_ranges) 
            for r in pulse_peak_ranges:
                middle = 0.5*(r[0]+r[1])
                abslim = 0.5*numpy.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = numpy.abs(dataset.p_peak_value/gain-middle) <= abslim
                    if cut_crosstalk:
                        m = numpy.logical_and(m, nhits==1)
                    masks.append(m)
        else:
            raise ValueError("Call make_masks with only one of pulse_avg_ranges and pulse_peak_ranges specified.")
        
        return numpy.array(masks)

        
    def compute_average_pulse(self, masks, subtract_mean=True):
        """
        Compute several average pulses in each TES channel, one per mask given in
        <masks>.  Store the averages in self.datasets.average_pulses with shape (m,n)
        where m is the number of masks and n equals self.nPulses (the # of records).
        
        Note that this method replaces any previously computed self.datasets.average_pulses
        
        <masks> is either an array of shape (m,n) or an array (or other sequence) of length
        (m*n).  It's required that n equal self.nPulses.   In the second case,
        m must be an integer.  The elements of <masks> should be booleans or interpretable
        as booleans.
        
        If <subtract_mean> is True, then each average pulse will subtract a constant
        to ensure that the pretrigger mean (first self.nPresamples elements) is zero.
        """

        # Make sure that masks is either a 2D or 1D array of the right shape,
        # or a sequence of 1D arrays of the right shape
        if isinstance(masks, numpy.ndarray):
            nd = len(masks.shape)
            if nd==1:
                n = len(masks)
                masks = masks.reshape((n/self.nPulses, self.nPulses))
            elif nd>2:
                raise ValueError("masks argument should be a 2D array or a sequence of 1D arrays")
            nbins = masks.shape[0]
        else:
            nbins = len(masks)
        for i,m in enumerate(masks):
            if not isinstance(m, numpy.ndarray):
                raise ValueError("masks[%d] is not a numpy.ndarray"%i)
            if m.shape != (self.nPulses,):
                raise ValueError("masks[%d] has shape %s, but it needs to be (%d,)"%
                     (i, m.shape, self.nPulses ))
            
        pulse_counts = numpy.zeros((self.n_channels,nbins))
        pulse_sums = numpy.zeros((self.n_channels,nbins,self.nSamples), dtype=numpy.float)

        for first, end in self.iter_segments():
            for imask,mask in enumerate(masks):
                valid = mask[first:end]
                for ichan,chan in enumerate(self.datasets):
                    good_pulses = chan.data[valid, :]
                    pulse_counts[ichan,imask] += good_pulses.shape[0]
                    pulse_sums[ichan,imask,:] += good_pulses.sum(axis=0)

        # Rescale and store result to each MicrocalDataSet
        pulse_sums /= pulse_counts.reshape((self.n_channels, nbins,1))
        for ichan,chan in enumerate(self.datasets):
            chan.average_pulses = pulse_sums[ichan,:,:]
            if subtract_mean:
                for mask in range(chan.average_pulses.shape[0]):
                    chan.average_pulses[mask,:] -= chan.average_pulses[mask,:self.nPresamples].mean()
    
    
    def plot_average_pulses(self, id, axis=None):
        """Plot average pulse number <id> on matplotlib.Axes <axis>, or
        on a new Axes if <axis> is None."""
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        axis.set_color_cycle(("blue","gold","green","red"))
        dt = (numpy.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        for i,d in enumerate(self.datasets):
            pylab.plot(dt,d.average_pulses[id], label="Demod TES %d"%i)
            pylab.xlabel("Time past trigger (ms)")


    def set_gains(self, gains):
        """Set the datasets to have the given gains.  These gains will be used when
        averaging pulses in self.compute_average_pulse() and in ...***?"""
        if len(gains) != self.n_channels:
            raise ValueError("gains must have the same length as the number of datasets (%d)"
                             %self.n_channels)
            
        for g,d in zip(gains, self.datasets):
            d.gain = g
    
    
    def compute_filters(self):
        
        # Analyze the noise, if not already done
        for n in self.datasets:
            if n.noise_autocorr is None or n.noise_spectrum is None:
                print "Computing noise autocorrelation and spectrum"
                self.compute_noise_spectra
                break
            
        self.filters=[]
        for i,ds in enumerate(self.datasets):
            print "Computing filter %d of %d"%(i, self.n_channels)
            avg_signal = ds.average_pulses[i]
            f = mass.channel.Filter(avg_signal, self.nPresamples, ds.noise_spectrum.spectrum(),
                                    ds.noise_autocorr, sample_time=self.timebase,
                                    shorten=2)
            self.filters.append(f)
            
    
    def filter_data(self):
        """Filter data sets and store in datasets[*].p_filt_phase and _value.
        The filters are currently self.filter[*].filt_noconst"""
        if self.filters is None:
            self.compute_filters()
        
        for dset in self.datasets:
            dset.p_filt_phase = numpy.zeros(dset.nPulses, dtype=numpy.float)
            dset.p_filt_value = numpy.zeros(dset.nPulses, dtype=numpy.float)
            
        for first, end in self.iter_segments():
            print "Records %d to %d loaded"%(first,end-1)
            for filter,dset in zip(self.filters,self.datasets):
                peak_x, peak_y = dset.filter_data(filter.filt_noconst,first, end)
                dset.p_filt_phase[first:end] = peak_x
                dset.p_filt_value[first:end] = peak_y
    
    
    def estimate_crosstalk(self, plot=True):
        """Work in progress..."""

        if plot: 
            pylab.clf()
            ds0 = self.datasets[0]
            dt = (numpy.arange(ds0.nSamples)-ds0.nPresamples)*ds0.timebase*1e3

        crosstalk = []
        from numpy import dot
        for i,(n,ds) in enumerate(zip(self.noise_channels, self.datasets)):
            if n.autocorrelation is None:
                print "Computing noise autocorrelation for %s"%os.path.basename(n.filename)
                n.compute_autocorrelation(self.nSamples, plot=False)

            p = ds.average_pulses[i]
            d = numpy.zeros_like(p)
            d[1:] = p[1:]-p[:-1]
            
            if 'pulse_filt' in ds.__dict__ and 'deriv_filt' in ds.__dict__:
                qp = ds.pulse_filt
                qd = ds.deriv_filt
            else:
                print "Solving for TES %2d noise-inv-weighted pulse"%i
                Rp = numpy.linalg.solve(scipy.linalg.toeplitz(n.autocorrelation[:8192]), p)
                print "Solving for TES %2d noise-inv-weighted derivative"%i
                Rd = numpy.linalg.solve(scipy.linalg.toeplitz(n.autocorrelation[:8192]), d)
                
                qp = dot(d,Rd)*Rp - dot(d,Rp)*Rd
                qd = dot(p,Rp)*Rd - dot(p,Rd)*Rp
                qp /= dot(p,qp)
                qd /= dot(d,qd)
            
                # Save filters and the derivative
                ds.pulse_filt = qp
                ds.deriv_filt = qd
                ds.pulse_deriv = d
            
            # Apply the filters to cross-talk and self-talk.  (!)
            ct = []
            for j,ds1 in enumerate(self.datasets):
                sig = ds1.average_pulses[i]
                print "%8.4f %8.4f"%(dot(sig,qp), dot(sig,qd))
                ct.append(dot(sig,qp))
                if plot and (i != j):
                    ax = pylab.subplot(4,4,1+4*j+i)
                    sig2 = sig-dot(sig,qp)*ds.average_pulses[i]
                    sig3 = sig2-dot(sig,qd)*ds.pulse_deriv
                    pylab.plot(dt,sig,color='green')
                    pylab.plot(dt,sig2,color='red')
                    pylab.plot(dt,sig3,color='blue')
                    pylab.xlim([-.5,3])
                    pylab.xticks((0,1,2,3))
                    pylab.ylim([-100,100])
                    pylab.title("TES %d when %d hit"%(j,i))
                    rmss=["%5.2f"%(s.std()) for s  in sig,sig2,sig3]
                    if sig[100+ds.nPresamples] < 0:
                        pylab.text(.08,.85,",".join(rmss), transform=ax.transAxes)
                    else:
                        pylab.text(.08,.05,",".join(rmss), transform=ax.transAxes)
            crosstalk.append(ct)
        return numpy.array(crosstalk)
    
    
    def compare_noise(self, axis=None):
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        axis.set_color_cycle(self.colors)
        for noise in self.noise_channels:
            noise.plot_power_spectrum(axis=axis)



class TESGroup(BaseChannelGroup):
    """
    A group of one or more *independent* microcalorimeters, in that
    they are time-division multiplexed.  It might be convenient to use
    this for multiple TDM channels, or for singles.  The key is that
    this object offers the same interface as the CDMGroup object
    (which has to be more complex under the hood).
    """
    def __init__(self, filenames, noise_filenames=None, noise_only=False):

        super(self.__class__, self).__init__(filenames, noise_filenames)
        self.noise_only = noise_only
        
        pulse_list = []
        noise_list = []
        dset_list = []
        for fname in self.filenames:
            pulse, noise = mass.channel.create_pulse_and_noise_records(fname, noise_only=noise_only)
            dset = mass.channel.MicrocalDataSet(pulse.__dict__)
            pulse_list.append(pulse)
            noise_list.append(noise)
            dset_list.append(dset)
            
            if self.n_segments is None:
                for attr in "n_segments","nPulses","nSamples","nPresamples", "timebase":
                    self.__dict__[attr] = pulse.__dict__[attr]
            else:
                assert self.n_segments == pulse.n_segments
            
        self.channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)
        if len(pulse_list)>0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg
    

    def copy(self):
        g = TESGroup([])
        g.__dict__.update(self.__dict__)
        g.channels = tuple([c.copy() for c in self.channels])
        return g
        

    def read_segment(self, segnum):
        """Read segment number <segnum> into memory for each of the
        channels in the group.  Return (first,end) where these are the
        number of the first record in that segment and 1 more than the
        number of the last record."""
        if segnum == self._cached_segment:
            return self._cached_pnum_range
        for chan, dset in zip(self.channels, self.datasets):
            first_pnum, end_pnum = chan.read_segment(segnum)
            dset.data = chan.data
            dset.times = chan.datafile.datatimes
        self._cached_segment = segnum
        self._cached_pnum_range = first_pnum, end_pnum
        return first_pnum, end_pnum
    
    
    def compute_noise_spectra(self):
        for dataset,noise in zip(self.datasets,self.noise_channels):
            noise.compute_power_spectrum(plot=False)
            dataset.noise_spectrum = noise.spectrum
#            noise.compute_autocorrelation(plot=False)
   

class CDMGroup(BaseChannelGroup):
    """
    A group of *CDM-coupled* microcalorimeters, in that they are code-division
    multiplexing a set of calorimeters into a set of output data streams.The key
    is that this object offers the same interface as the TESGroup object (which
    is rather less complex under the hood than this object).
    """
    
    def __init__(self, filenames, noise_filenames=None, demodulation=None, noise_only=False):
        """
        modulation[i,j] (i.e. row i, column j) means contribution to signal in
        channel i due to detector j.   
        """
        super(self.__class__, self).__init__(filenames, noise_filenames)

        if demodulation is None:
            demodulation = numpy.array(
                (( 1, 1, 1, 1),
                 (-1, 1, 1,-1),
                 (-1,-1, 1, 1),
                 (-1, 1,-1, 1)), dtype=numpy.float)
        assert demodulation.shape[0] == demodulation.shape[1]
        assert demodulation.shape[0] == self.n_channels
        self.demodulation = demodulation
        self.idealized_walsh = numpy.array(demodulation.round(), dtype=numpy.int16)
        self.noise_only = noise_only

        pulse_list = []
        noise_list = [] 
        demod_list = []
        for i,fname in enumerate(self.filenames):
            if noise_filenames is None:
                pulse, noise = mass.channel.create_pulse_and_noise_records(fname, noise_only=noise_only)
            else:
                nf = noise_filenames[i]
                pulse, noise = mass.channel.create_pulse_and_noise_records(fname, noisename=nf,
                                                                           noise_only=noise_only)
            demod = mass.channel.MicrocalDataSet(pulse.__dict__)
            
            pulse_list.append(pulse)
            noise_list.append(noise)
            demod_list.append(demod)

            if self.n_segments is None:
                for attr in "n_segments","nPulses","nSamples","nPresamples","timebase":
                    self.__dict__[attr] = pulse.__dict__[attr]
            else:
                assert self.n_segments == pulse.n_segments
                assert self.nSamples == pulse.nSamples
                assert self.nPresamples == pulse.nPresamples
                if self.nPulses > pulse.nPulses:
                    self.nPulses = pulse.nPulses
            
        self.raw_channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets= tuple(demod_list)
        for ds in self.datasets:
            if ds.nPulses > self.nPulses:
                ds.resize(self.nPulses)
                
        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].datafile.pulses_per_seg
        

    def copy(self):
        g = CDMGroup(self.filenames, self.noise_filenames,
                     demodulation=self.demodulation, noise_only=self.noise_only)
        g.__dict__.update(self.__dict__)
        g.raw_channels = tuple([c.copy() for c in self.raw_channels])
        g.datasets = tuple([c.copy() for c in self.datasets])
        return g
        

    def read_segment(self, segnum):
        """
        Read segment number <segnum> into memory for each of the
        channels in the group.  
        
        Perform linear drift correction and demodulation.
        
        Return (first,end) where these are the
        number of the first record in that segment and 1 more than the
        number of the last record."""

        if segnum == self._cached_segment:
            return self._cached_pnum_range

        for chan, dset in zip(self.raw_channels, self.datasets):
            first, end = chan.read_segment(segnum)
            dset.times = chan.datafile.datatimes

        # Remove linear drift
        REMOVE_DRIFT=True
        seg_size = min([rc.data.shape[0] for rc in self.raw_channels]) # Last seg can be of unequal size!
        mod_data = numpy.zeros([self.n_channels, seg_size, self.nSamples], dtype=numpy.int32)
        mod_data[0, :, :] = numpy.array(self.raw_channels[0].data[:seg_size, :])
        for i in range(1, self.n_channels):
            if REMOVE_DRIFT:
                mod_data[i, :, 1:] = i*self.raw_channels[i].data[:seg_size,1:]
                mod_data[i, :, 1:] += (self.n_channels-i) *self.raw_channels[i].data[:seg_size,:-1]
                mod_data[i, :, 0] = self.n_channels*self.raw_channels[i].data[:seg_size,0]
                mod_data[i, :, :] /= self.n_channels
            else:
                mod_data[i, :, :] = numpy.array(self.raw_channels[i].data[:,:])
        
        # Demodulate
        for i_det,dset in enumerate(self.datasets):   
            dset.data = numpy.zeros((seg_size, self.nSamples), dtype=numpy.float)
            for j in range(self.n_channels):
                dset.data += self.demodulation[i_det,j]*mod_data[j, :, :]
    
        self._cached_segment = segnum
        self._cached_pnum_range = first,end
        return first, end


    def compute_noise_spectra(self, compute_raw_spectra=False):
        """Compute the noise power spectral density for demodulated data.
        
        <compute_raw_spectra> If True, also compute it for the raw data
        """

        # First, find out the length of the shortest noise set
        nrec = 9999999
        for n in self.noise_channels:
            if compute_raw_spectra:
                n.compute_power_spectrum(plot=False)
            if nrec>n.data.shape[0]:
                nrec = n.data.shape[0]
        
        # Now generate a set of fake channels, where we'll store the demodulated data
        fake_noise_chan = [n.copy() for n in self.noise_channels]
        shape = (nrec,self.noise_channels[0].data.shape[1])

        # Demodulate noise
        for i,nc in enumerate(fake_noise_chan):
            nc.nPulses = nrec
            nc.data = numpy.zeros(shape, dtype=numpy.float)
            for j,n in enumerate(self.noise_channels):
                nc.data += self.demodulation[i,j] * n.data[:nrec,:]
            nc.data -= nc.data.mean()
            nc.data *= 0.5
        
        # Compute spectra    
        for nc,ds in zip(fake_noise_chan, self.datasets):
            nc.compute_autocorrelation(n_lags=self.nSamples, plot=False)
            nc.compute_power_spectrum(plot=False)
            ds.noise_spectrum = nc.spectrum
            ds.noise_autocorr = nc.autocorrelation


    def compare_noise(self):
        """Check whether noise spectra differ greatly between modulated channels
        or between demodulated channels."""
        
        nrec = 9999999
        for n in self.noise_channels:
            n.compute_power_spectrum(plot=False)
            if nrec>n.data.shape[0]:
                nrec = n.data.shape[0]
        
        # Demodulate noise
        dmn = [n.copy() for n in self.noise_channels]
        shape = (nrec,self.noise_channels[0].data.shape[1])
        for i,d in enumerate(dmn):
            d.data = numpy.zeros(shape, dtype=numpy.float)
            for j,n in enumerate(self.noise_channels):
                d.data += self.demodulation[i,j] * n.data[:nrec,:]
            d.data -= d.data.mean()
            d.data *= 0.5
        for n in dmn:
            n.compute_power_spectrum(plot=False)
            
        pylab.clf()
        ax1=pylab.subplot(211)
        ax2=pylab.subplot(212, sharex=ax1, sharey=ax1)
        for a in ax1,ax2:
            a.set_color_cycle(("blue","#cccc00","green","red"))
            a.loglog()
            a.grid()
        for n in self.noise_channels:
            n.plot_power_spectrum(axis=ax1)
        for n in dmn:
            n.plot_power_spectrum(axis=ax2)
        ax1.set_title("Raw (modulated) channel noise power spectrum")
        ax2.set_title("Demodulated TES noise power spectrum")
    
    
    def update_demodulation(self, relative_response):
        relative_response = numpy.asmatrix(relative_response)
        if relative_response.shape != (self.n_channels,self.n_channels):
            raise ValueError("The relative_response matrix needs to be of shape (%d,%d)"%
                             (self.n_channels, self.n_channels))
            
        self.demodulation = numpy.dot( relative_response.I, self.demodulation)