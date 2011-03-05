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

import numpy
from matplotlib import pylab

import mass.channel
import mass.controller



class BaseChannelGroup(object):
    """
    Provides the interface for a group of one or more microcalorimeters,
    whether the detectors are multiplexed with time division or code
    division.
    
    This is an abstract base class, and the appropriate concrete class
    is the TESGroup or the CDMGroup, depending on the multiplexing scheme. 
    """
    def __init__(self, filenames):
        # Convert a single filename to a tuple of size one
        if isinstance(filenames, str):
            filenames = (filenames,)
        self.filenames = tuple(filenames)
        self.n_channels = len(self.filenames)
        self.n_segments = None
        self._cached_segment = None
        self._cached_pnum_range = None
        self.pulses_per_seg = None
        
        
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
            print "Records %d to %d loaded"%(first,end-1)
            for dset in self.datasets:
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


    def compute_average_pulse(self, masks):

        # Make sure that masks is either a 2D or 1D array of the right shape,
        # or a sequence of 1D arrays of the right shape
        if isinstance(masks, numpy.ndarray):
            nd = len(masks.shape)
            if nd==1:
                masks = masks.reshape((1,len(masks)))
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

        # Rescale
        pulse_sums /= pulse_counts.reshape((self.n_channels, nbins,1))
        for ichan,chan in enumerate(self.datasets):
            chan.average_pulses = pulse_sums[ichan,:,:]

##            n = self.datafile.data.shape[0]
#            for ibin, bin in enumerate(ph_bins):
#                bin_ctr = 0.5*(bin[0]+bin[1])
#                bin_hw = numpy.abs(bin_ctr-bin[0])
#                cuts = numpy.logical_and(
#                        numpy.abs(bin_ctr - data.max(axis=1)) < bin_hw,
#                        self.good[first:end])
#                good_pulses = data[cuts, :]
#                pulse_counts[ibin] += good_pulses.shape[0]
#                pulse_sums[ibin,:] += good_pulses.sum(axis=0)
#
#        self.average_pulse = (pulse_sums.T/pulse_counts).T





class TESGroup(BaseChannelGroup):
    """
    A group of one or more *independent* microcalorimeters, in that
    they are time-division multiplexed.  It might be convenient to use
    this for multiple TDM channels, or for singles.  The key is that
    this object offers the same interface as the CDMGroup object
    (which has to be more complex under the hood).
    """
    def __init__(self, filenames):

        super(self.__class__, self).__init__(filenames)
        
        pulse_list = []
        noise_list = []
        dset_list = []
        for fname in self.filenames:
            pulse, noise = mass.channel.create_pulse_and_noise_records(fname)
            dset = mass.channel.MicrocalDataSet(pulse.__dict__)
            pulse_list.append(pulse)
            noise_list.append(noise)
            dset_list.append(dset)
            
            if self.n_segments is None:
                self.n_segments = pulse.n_segments
                self.nPulses = pulse.nPulses
                self.nSamples = pulse.nSamples
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

   

class CDMGroup(BaseChannelGroup):
    """
    A group of *CDM-coupled* microcalorimeters, in that they are code-division
    multiplexing a set of calorimeters into a set of output data streams.The key
    is that this object offers the same interface as the TESGroup object (which
    is rather less complex under the hood than this object).
    """
    
    def __init__(self, filenames, walsh=None):
        
        super(self.__class__, self).__init__(filenames)

        if walsh is None:
            walsh = numpy.array(
                (( 1, 1, 1, 1),
                 (-1, 1, 1,-1),
                 (-1,-1, 1, 1),
                 (-1, 1,-1, 1)), dtype=numpy.int16)
        assert walsh.shape[0] == walsh.shape[1]
        assert walsh.shape[0] == self.n_channels
        self.walsh = walsh

        pulse_list = []
        noise_list = [] 
        demod_list = []
        for fname in self.filenames:
            pulse, noise = mass.channel.create_pulse_and_noise_records(fname)
            demod = mass.channel.MicrocalDataSet(pulse.__dict__)
            
            pulse_list.append(pulse)
            noise_list.append(noise)
            demod_list.append(demod)

            if self.n_segments is None:
                self.n_segments = pulse.n_segments
                self.nPulses = pulse.nPulses
                self.nSamples = pulse.nSamples
            else:
                assert self.n_segments == pulse.n_segments
            
        self.raw_channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets= tuple(demod_list)
        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].datafile.pulses_per_seg
        

    def copy(self):
        g = CDMGroup(self.filenames, walsh=self.walsh)
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
        shape = self.raw_channels[0].data.shape
        mod_data = numpy.zeros([self.n_channels]+list(shape), dtype=numpy.int32)
        mod_data[0, :, :] = self.raw_channels[0].data
        for i in range(1, self.n_channels):
            mod_data[i, :, 1:] = i*self.raw_channels[i].data[:,1:]
            mod_data[i, :, 1:] += (self.n_channels-i) *self.raw_channels[i].data[:,:-1]
            mod_data[i, :, 0] = self.n_channels*self.raw_channels[i].data[:,0]
            mod_data[i, :, :] /= self.n_channels
        
        # Demodulate
        for i,dset in enumerate(self.datasets):   
            dset.data = numpy.zeros(shape, dtype=numpy.int32)
            for j in range(self.n_channels):
                dset.data += self.walsh[i,j]*mod_data[j, :, :]
    
        self._cached_segment = segnum
        self._cached_pnum_range = first,end
        return first, end


