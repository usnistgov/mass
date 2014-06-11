'''
Created on Jun 11, 2014

@author: fowlerj
'''

import numpy as np
import pylab as plt
import time
import mass
from mass.core.channel_group import BaseChannelGroup

class CDMGroup(BaseChannelGroup):
    """
    NOT MAINTAINED SINCE 2012, so I don't know how well this works, if at all.
    J Fowler, 11 June 2014
    
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
        BaseChannelGroup.__init__(self, filenames, noise_filenames)
        if noise_only:
            self.n_cdm = len(filenames)
        else:
            self.n_cdm = self.n_channels  # If >1 column, this won't be true anymore

#        if demodulation is None:
#            demodulation = np.array(
#                (( 1, 1, 1, 1),
#                 (-1, 1, 1,-1),
#                 (-1,-1, 1, 1),
#                 (-1, 1,-1, 1)), dtype=np.float)
        assert demodulation.shape[0] == demodulation.shape[1]
        assert demodulation.shape[0] == self.n_cdm
        self.demodulation = demodulation
        self.idealized_walsh = np.array(demodulation.round(), dtype=np.int16)
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
        self._setup_channels_list()
        for ds in self.datasets:
            if ds.nPulses > self.nPulses:
                ds.resize(self.nPulses)
                
        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].datafile.pulses_per_seg
        self.noise_filenames = [n.datafile.filename for n in self.noise_channels]
        self.REMOVE_INFRAME_DRIFT=True
        
        # Set master timestamp_offset (seconds)
        self.timestamp_offset = self.raw_channels[0].timestamp_offset
        for ch in self.raw_channels:
            if ch.timestamp_offset != self.timestamp_offset:
                self.timestamp_offset = None
                break
    
        

    def copy(self):
        self.clear_cache()
        g = CDMGroup(self.filenames, self.noise_filenames,
                     demodulation=self.demodulation, noise_only=self.noise_only)
        g.__dict__.update(self.__dict__)
        g.raw_channels = tuple([c.copy() for c in self.raw_channels])
        g.noise_channels = tuple([c.copy() for c in self.noise_channels])
        g.noise_channels_demod = tuple([c.copy() for c in self.noise_channels_demod])
        g.datasets = tuple([c.copy() for c in self.datasets])
#        g.filters = tuple([f.copy() for f in self.filters])
        return g
        

    def set_segment_size(self, seg_size):
        self.clear_cache()
        self.n_segments = 0
        for chan in self.raw_channels:
            chan.set_segment_size(seg_size)
            self.n_segments = max(self.n_segments, chan.n_segments)
        self.pulses_per_seg = self.raw_channels[0].pulses_per_seg
        for chan in self.raw_channels:
            assert chan.pulses_per_seg == self.pulses_per_seg


    def read_segment(self, segnum, use_cache=True):
        """
        Read segment number <segnum> into memory for each of the
        channels in the group.  
        When <use_cache> is true, we use cached value when possible
        
        Perform linear drift correction and demodulation.
        
        Return (first,end) where these are the
        number of the first record in that segment and 1 more than the
        number of the last record."""

        if segnum == self._cached_segment and use_cache:
            return self._cached_pnum_range

        for chan, dset in zip(self.raw_channels, self.datasets):
            first, end = chan.read_segment(segnum)
            dset.times = chan.datafile.datatimes_float

        # Remove linear drift
        seg_size = min([rc.data.shape[0] for rc in self.raw_channels]) # Last seg can be of unequal size!
        mod_data = np.zeros([self.n_channels, seg_size, self.nSamples], dtype=np.int32)
        wide_holder = np.zeros([seg_size, self.nSamples], dtype=np.int32)
        
#        mod_data[0, :, :] = np.array(self.raw_channels[0].data[:seg_size, :])
#        for i in range(1, self.n_channels):
#            if self.REMOVE_INFRAME_DRIFT:
#                mod_data[i, :, :] =  self.raw_channels[i].data[:seg_size,:]
#                wide_holder[:, 1:] = self.raw_channels[i].data[:seg_size,:-1]  # Careful never to mult int16 by more than 4! 
#                mod_data[i, :, 1:] *= (self.n_channels-i)   # Weight to future value
#                mod_data[i, :, 1:] += i  *wide_holder[:,1:] # Weight to prev value
#
#            else:
#                mod_data[i, :, :] = np.array(self.raw_channels[i].data[:seg_size,:])
    
        for i, raw_ch in enumerate(self.raw_channels):
            mod_data[i, :, :] = np.array(raw_ch.data[:seg_size, :])
            
        if self.REMOVE_INFRAME_DRIFT:
            mod_data[0, :, :] *= self.n_channels
            for i in range(1, self.n_channels):
                wide_holder[:, 1:] = self.raw_channels[i].data[:seg_size,:-1]  # Careful never to mult int16 by more than 4! 
                wide_holder[:, 0] = wide_holder[:, 1]       # Handle boudary case of first sample, where there is no prev value to mix in 
                mod_data[i, :, :] *= (self.n_channels-i)    # Weight to future value
                mod_data[i, :, :] += i*wide_holder[:,:]     # Weight to prev value

        # Demodulate.  
        # If we've done INFRAME DRIFT, then mod_data is too large by a factor of n_channels.  Correct for that: 
        demodulation = self.demodulation
        if self.REMOVE_INFRAME_DRIFT:
            demodulation = self.demodulation.copy()/self.n_channels

        for i_det,dset in enumerate(self.datasets):
            
            # For efficiency, don't create a new dset.data vector is it already exists and is the right shape.   
            try:
                assert dset.data.shape == (seg_size, self.nSamples)
                dset.data.flat = 0.0
            except:
                dset.data = np.zeros((seg_size, self.nSamples), dtype=np.float)
            
            # Multiply by the demodulation matrix    
            for j in range(self.n_channels):
                dset.data += demodulation[i_det,j]*mod_data[j, :, :]
    
        self._cached_segment = segnum
        self._cached_pnum_range = first,end
        return first, end


    def summarize_data(self):
        """
        Compute summary quantities for each pulse.
        """

        t0 = time.time()
        BaseChannelGroup.summarize_data(self)

        # How many detectors were hit in each record?
        self.nhits = np.array([d.p_pulse_average>50 for d in self.datasets]).sum(axis=0)
        print "Summarized data in %.0f seconds" %(time.time()-t0)
        

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
        BaseChannelGroup.compute_average_pulse(self, masks, use_crosstalk_masks=True, subtract_mean=subtract_mean)
        
        
    def compute_noise_spectra(self, compute_raw_spectra=False):
        """
        Compute the noise power spectral density for demodulated data.
        
        <compute_raw_spectra> If True, also compute it for the raw data
        """

        # First, find out the length of the shortest noise set
        nrec = 9999999
        for n in self.noise_channels:
            if compute_raw_spectra:
                print "Computing raw power spectrum for %s"%n.filename
                n.compute_power_spectrum(plot=False)
            if nrec>n.nPulses:
                nrec = n.nPulses
        
        # Now generate a set of fake channels, where we'll store the demodulated data
        self.noise_channels_demod = [n.copy() for n in self.noise_channels]
        for nc in self.noise_channels_demod: nc.data=None
        shape = (nrec,self.noise_channels[0].nSamples)

        # Demodulate noise
        print "Demodulating noise for nrec=%d"%nrec
        for i,nc in enumerate(self.noise_channels_demod):
            nc.set_fake_data()
            nc.nPulses = nrec
            nc.data = np.zeros(shape, dtype=np.float)
            for j,n in enumerate(self.noise_channels):
                for first, end, _segnum, data in n.datafile.iter_segments():
                    if end > nrec:
                        end = nrec
                    print i, j, first, end, data.shape, 'is here', self.demodulation[i,j]
                    nc.data[first:end] += self.demodulation[i,j] * data[:(end-first),:]
            nc.data -= nc.data.mean()
            
        # Compute spectra    
        for nc,ds in zip(self.noise_channels_demod, self.datasets):
            print "Computing demodulated noise autocorrelation for %s"%ds
            nc.compute_autocorrelation(n_lags=self.nSamples, plot=False)
            print "Computing demodulated power spectrum for %s"%ds
            nc.compute_power_spectrum(plot=False)
            ds.noise_spectrum = nc.spectrum
            ds.noise_autocorr = nc.autocorrelation
#            ds.noise_demodulated = nc


    def plot_noise(self, show_modulated=False, channels=None, scale_factor=1.0, sqrt_psd=False):
        """Compare the noise power spectra.
        
        <show_modulated> Whether to show the raw (modulated) noise spectra, or
                         only the demodulated spectra.
        <channels>    Sequence of channels to display.  If None, then show all.
        <scale_factor> Multiply counts by this number to get physical units. 
        <sqrt_psd>     Whether to show the sqrt(PSD) or (by default) the PSD itself.
        """
        
        if channels is None:
            channels = np.arange(self.n_cdm)
            
        for ds in self.datasets:
            if ds.noise_spectrum is None:
                self.compute_noise_spectra(compute_raw_spectra=True)
                break
            
        
        plt.clf()
        if show_modulated:
            ax1=plt.subplot(211)
            ax1.set_title("Raw (modulated) channel noise power spectrum")
            ax2=plt.subplot(212, sharex=ax1, sharey=ax1)
            axes=(ax1,ax2)
        else:
            ax2=plt.subplot(111)
            axes=(ax2,)
            
        ax2.set_title("Demodulated TES noise power spectrum SCALED BY 1/%d"%self.n_cdm)
        for a in axes:
            a.set_color_cycle(self.colors)
            a.set_xlabel("Frequency (Hz)")
            a.loglog()
            a.grid()
        
        if show_modulated:
            for i,n in enumerate(self.noise_channels):
                n.plot_power_spectrum(axis=ax1, label='Row %d'%i)
            plt.legend(loc='upper right')
            
        for i,ds in enumerate(self.datasets):
            if i not in channels: continue
            yvalue = ds.noise_spectrum.spectrum()*scale_factor**2
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
            plt.plot(ds.noise_spectrum.frequencies(), yvalue,
                       label='TES %d'%i, color=self.colors[i])
        plt.legend(loc='lower left')


    def plot_noise_autocorrelation(self, axis=None, channels=None):
        """Compare the noise autocorrelation functions.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        """
        
        if channels is None:
            channels = np.arange(self.n_channels)

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            
        axis.grid(True)
        for i,noise in enumerate(self.noise_channels_demod):
            if i not in channels: continue
            noise.plot_autocorrelation(axis=axis, label='TES %d'%i, color=self.colors[i%len(self.colors)])
#        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        plt.legend(loc='upper right')    

    
    def plot_undecimated_noise(self, ndata=None):
        """Show the noise as if """
        np = np.array([nc.nPulses for nc in self.noise_channels]).min()
        if ndata is None:
            ndata = np*self.noise_channels[0].nSamples
        assert ndata<=1024*1024
        data = np.asarray(np.vstack([nc.data[:np,:].ravel()[:ndata] for nc in self.noise_channels]), dtype=np.int16)
        for r in data:
            r -= r.mean()
        data=data.T.ravel()

        segfactor=max(8, ndata/1024)
        
        freq, psd = mass.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=self.timebase/self.n_cdm, window=mass.power_spectrum.hamming)
        plt.clf()
        plt.plot(freq, psd)
    
    def update_demodulation(self, relative_response):
        relative_response = np.asmatrix(relative_response)
        if relative_response.shape != (self.n_channels,self.n_channels):
            raise ValueError("The relative_response matrix needs to be of shape (%d,%d)"%
                             (self.n_channels, self.n_channels))
            
        self.demodulation = np.dot( relative_response.I, self.demodulation)


    def plot_modulated_demodulated(self, pulsenum=119148, modulated_offsets=np.arange(15000,-1,-5000), xlim=[-5,15.726]):
        "Plot one record both modulated and demodulated"
        
        self.ms = (np.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        plt.clf()
        
        self.read_trace(pulsenum)
        n = pulsenum - self._cached_pnum_range[0]
        plt.subplot(211)
        if modulated_offsets is None:
            modulated_offsets = (0,0,0,0)
        for i,rc in enumerate(self.raw_channels):
            plt.plot(self.ms, rc.data[n,:]+modulated_offsets[i]-rc.data[n,:self.nPresamples].mean(), color=self.colors[i], label='SQUID sw%d'%i)
        if xlim is not None: plt.xlim(xlim)
        plt.legend(loc='upper left')
        plt.title("Modulated (raw) signal")
        

        plt.subplot(212)
        for i,ds in enumerate(self.datasets):
            plt.plot(self.ms, ds.data[n,:]-ds.p_pretrig_mean[pulsenum], color=self.colors[i], label='TES %d'%i)
        if xlim is not None: plt.xlim(xlim)
        plt.legend(loc='upper left')
        plt.title("Demodulated signal")
        plt.xlabel("Time since trigger (ms)")


