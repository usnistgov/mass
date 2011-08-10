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

import time
import numpy
from matplotlib import pylab
import scipy.linalg

import mass.channel
import mass.utilities
import mass.energy_calibration
import mass.power_spectrum
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
        if noise_filenames is None:
            self.noise_filenames = None
        else:
            if isinstance(noise_filenames, str):
                noise_filenames = (noise_filenames,)
            self.noise_filenames = noise_filenames
        
        self.nhits = None
        self.n_segments = None
        self._cached_segment = None
        self._cached_pnum_range = None
        self.pulses_per_seg = None
        self.filters = None
        
        if self.n_channels <=4:
            self.colors=("blue", "#aaaa00","green","red")
        else:
            BRIGHTORANGE='#ff7700'
            self.colors=('purple',"blue","cyan","green","gold",BRIGHTORANGE,"red","brown")
    
    def get_channel_dataset(self, channum):
        for i,fn in enumerate(self.filenames):
            if "chan%2d"%channum in fn: return self.datasets[i]
        print "No filename contains 'chan%2d', so dataset not found"%channum
        return None
        

    def clear_cache(self):
        self._cached_segment = None
        self._cached_pnum_range = None
        
        
    def iter_segments(self, first_seg=0, end_seg=-1):
        if end_seg < 0: 
            end_seg = self.n_segments
        for i in range(first_seg, end_seg):
            first_rnum, end_rnum = self.read_segment(i)
            yield first_rnum, end_rnum


    def summarize_data(self):
        """
        Compute summary quantities for each pulse.  Subclasses override this with methods
        that ought to call this!
        """

        print "This data set has (up to) %d records with %d samples apiece."%(self.nPulses, self.nSamples)  
        for first, end in self.iter_segments():
            if end>self.nPulses:
                end = self.nPulses 
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
        
        
        
        
    def plot_traces(self, pulsenums, channum=0, pulse_summary=True, axis=None, difference=False):
        """Plot some example pulses, given by sample number.
        <pulsenums>  A sequence of sample numbers, or a single one.
        
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A pylab axis to plot on.
        <difference> Whether to show successive differences or the raw data
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
            if difference:
                data = data*1.0-numpy.roll(data,1)
                data[0] = 0
                data += numpy.roll(data,1) + numpy.roll(data,-1)
                data[0] = 0
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


    def plot_summaries(self, quantity, valid='uncut', downsample=None, log=False, hist_limits=None):
        """Plot a summary of one quantity from the data set, including time series and histograms of
        this quantity.  This method plots all channels in the group, but only one quantity.  If you
        would rather see all quantities for one channel, then use the group's 
        group.dataset[i].plot_summaries() method. 
        
        <quantity> A case-insensitive whitespace-ignored one of the following list, or the numbers that
                   go with it:
                   "Pulse Avg" (0)
                   "Pretrig RMS" (1)
                   "Pretrig Mean" (2)
                   "Peak Value" (3)
                   "Max PT Deriv" (4)
                   "Rise Time" (5)
                   "Peak Time" (6)
                   
        <valid> The words 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted 
                *OR* None, meaning that all pulses should be plotted.
                
        <downsample> To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points.
                     
        <log>  Use logarithmic y-axis on the histograms (right panels).
        """
        
        plottables = (
            ("p_pulse_average", 'Pulse Avg', 'purple', [0,5000]),
            ("p_pretrig_rms", 'Pretrig RMS', 'blue', [0,4000]),
            ("p_pretrig_mean", 'Pretrig Mean', 'green', None),
            ("p_peak_value", 'Peak value', '#88cc00',None),
            ("p_max_posttrig_deriv", 'Max PT deriv', 'gold', [0,700]),
            ("p_rise_time*1e3", 'Rise time (ms)', 'orange', [0,12]),
            ("p_peak_time*1e3", 'Peak time (ms)', 'red', [-3,9])
          ) 
        
        quant_names = [p[1].lower().replace(" ","") for p in plottables]
        if quantity in range(len(quant_names)):
            plottable = plottables[quantity]
        else:
            i = quant_names.index(quantity.lower().replace(" ",""))
            plottable = plottables[i]
                
        pylab.clf()
        for i,ds in enumerate(self.datasets):
            print 'TES%2d '%i,
            
            # Convert "uncut" or "cut" to array of all good or all bad data
            if isinstance(valid, str):
                if "uncut" in valid.lower():
                    valid_mask = ds.cuts.good()
                    print "Plotting only uncut data",
                elif "cut" in valid.lower():
                    valid_mask = ds.cuts.bad()  
                    print "Plotting only cut data",
                elif 'all' in valid.lower():
                    valid_mask = None
                    print "Plotting all data, cut or uncut",
                else:
                    raise ValueError("If valid is a string, it must contain 'all', 'uncut' or 'cut'.")
                    
            if valid_mask is not None:
                nrecs = valid_mask.sum()
                if downsample is None:
                    downsample=nrecs/10000
                    if downsample < 1: downsample = 1
                hour = ds.p_timestamp[valid_mask][::downsample]/3.6e6
            else:
                nrecs = ds.nPulses
                if downsample is None:
                    downsample = ds.nPulses / 10000
                    if downsample < 1: downsample = 1
                hour = ds.p_timestamp[::downsample]/3.6e6
            print " (%d records; %d in scatter plots)"%(
                nrecs,len(hour))
        
            (vect, label, color, default_limits) = plottable
            if hist_limits is None:
                limits = default_limits
            else:
                limits = hist_limits
            
            vect=eval("ds.%s"%vect)[valid_mask]
            
            if i==0:
                ax_master=pylab.subplot(self.n_channels, 2, 1+i*2)
            else:
                pylab.subplot(self.n_channels, 2, 1+i*2, sharex=ax_master)
                
            pylab.plot(hour, vect[::downsample],',', color=color)
            if i==0: pylab.title(label)
            pylab.ylabel("TES %d"%i)

            if i==0:
                axh_master = pylab.subplot(self.n_channels, 2, 2+i*2)
            else:
                if 'Pretrig Mean'==label:
                    pylab.subplot(self.n_channels, 2, 2+i*2)
                else:
                    pylab.subplot(self.n_channels, 2, 2+i*2, sharex=axh_master)
                
            if limits is None:
                in_limit = numpy.ones(len(vect), dtype=numpy.bool)
            else:
                in_limit= numpy.logical_and(vect>limits[0], vect<limits[1])
            contents, _bins, _patches = pylab.hist(vect[in_limit],200, log=log, 
                           histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                pylab.ylim(ymin = contents.min())


    def make_masks(self, pulse_avg_ranges=None, pulse_peak_ranges=None, 
                   use_gains=True, gains=None, cut_crosstalk=False,
                   max_ptrms=None, max_post_deriv=None):
        """Generate a sequence of masks for use in compute_average_pulses().
        
        <use_gains>   Rescale the pulses by a set of "gains", either from <gains> or from
                      the MicrocalDataSet.gain parameter if <gains> is None.
        <gains>       The set of gains to use, overriding the self.datasets[*].gain, if
                      <use_gains> is True.  (If False, this argument is ignored.)
        <cut_crosstalk>  Whether to mask out events having nhits>1.  (Makes no sense in TDM data).
        <max_ptrms>      When <cut_crosstalk>, we can also mask out events where any other channel
                         has p_pretrig_rms exceeding <max_ptrms>
        <max_post_deriv> When <cut_crosstalk>, we can also mask out events where any other channel
                         has p_max_posttrig_deriv exceeding <max_post_deriv>
        """
        
        masks = []
        if use_gains:
            if gains is None:
                gains = [d.gain for d in self.datasets]
        else:
            gains = numpy.ones(self.n_channels)
            
        # Cut crosstalk only makes sense in CDM data
        if cut_crosstalk and not isinstance(self, CDMGroup):
            print 'Cannot cut crosstalk because this is not CDM data'
            cut_crosstalk = False 
            
        if not cut_crosstalk: 
            if max_ptrms is not None:
                print "Warning: make_masks ignores max_ptrms when not cut_crosstalk"
            if max_post_deriv is not None:
                print "Warning: make_masks ignores max_post_deriv when not cut_crosstalk"
            
        if pulse_avg_ranges is not None:
            if pulse_peak_ranges is not None:
                print "Warning: make_masks uses only one range argument.  Ignoring pulse_peak_ranges."
            
            if isinstance(pulse_avg_ranges[0], (int,float)) and len(pulse_avg_ranges)==2:
                pulse_avg_ranges = tuple(pulse_avg_ranges),
            for r in pulse_avg_ranges:
                middle = 0.5*(r[0]+r[1])
                abslim = 0.5*numpy.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = numpy.abs(dataset.p_pulse_average/gain-middle) <= abslim
                    if cut_crosstalk: 
                        m = numpy.logical_and(m, self.nhits==1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = numpy.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = numpy.logical_and(m, ds.p_max_posttrig_deriv < max_post_deriv)
                    m = numpy.logical_and(m, dataset.cuts.good())
                    masks.append(m)
                    
        elif pulse_peak_ranges is not None:
            if isinstance(pulse_peak_ranges[0], (int,float)) and len(pulse_peak_ranges)==2:
                pulse_peak_ranges = tuple(pulse_peak_ranges),
            for r in pulse_peak_ranges:
                middle = 0.5*(r[0]+r[1])
                abslim = 0.5*numpy.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = numpy.abs(dataset.p_peak_value/gain-middle) <= abslim
                    if cut_crosstalk:
                        m = numpy.logical_and(m, self.nhits==1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = numpy.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = numpy.logical_and(m, ds.p_max_posttrig_deriv < max_post_deriv)
                    m = numpy.logical_and(m, dataset.cuts.good())
                    masks.append(m)
        else:
            raise ValueError("Call make_masks with only one of pulse_avg_ranges and pulse_peak_ranges specified.")
        
        return masks
        

    def compute_average_pulse(self, masks, use_crosstalk_masks, subtract_mean=True):
        """
        Compute several average pulses in each TES channel, one per mask given in
        <masks>.  Store the averages in self.datasets.average_pulses with shape (m,n)
        where m is the number of masks and n equals self.nPulses (the # of records).
        
        Note that this method replaces any previously computed self.datasets.average_pulses
        
        <masks> is either an array of shape (m,n) or an array (or other sequence) of length
        (m*n).  It's required that n equal self.nPulses.   In the second case,
        m must be an integer.  The elements of <masks> should be booleans or interpretable
        as booleans.
        
        <use_crosstalk_masks> Says whether to compute, e.g., averages for mask[1] on dataset[0].
                              Normally you'd set this to True for CDM data and False for TDM.
                              And the subclass methods use these settings.
        
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
            
        pulse_counts = numpy.zeros((self.n_channels,nbins))
        pulse_sums = numpy.zeros((self.n_channels,nbins,self.nSamples), dtype=numpy.float)

        for first, end in self.iter_segments():
            print "Records %d to %d loaded"%(first,end-1)
            for imask,mask in enumerate(masks):
                valid = mask[first:end]
                for ichan,chan in enumerate(self.datasets):
                    if not (use_crosstalk_masks or (imask%self.n_channels) == ichan):
                        continue 
                    
                    if mask.shape != (chan.nPulses,):
                        raise ValueError("masks[%d] has shape %s, but it needs to be (%d,)"%
                             (imask, mask.shape, chan.nPulses ))
                    good_pulses = chan.data[valid, :]
                    pulse_counts[ichan,imask] += good_pulses.shape[0]
                    pulse_sums[ichan,imask,:] += good_pulses.sum(axis=0)

        # Rescale and store result to each MicrocalDataSet
        pulse_sums /= pulse_counts.reshape((self.n_channels, nbins,1))
        for ichan,ds in enumerate(self.datasets):
            ds.average_pulses = pulse_sums[ichan,:,:]
            if subtract_mean:
                for imask in range(ds.average_pulses.shape[0]):
                    ds.average_pulses[imask,:] -= ds.average_pulses[imask,:self.nPresamples-ds.pretrigger_ignore_samples].mean()
    
    
    def plot_average_pulses(self, id, axis=None):
        """Plot average pulse number <id> on matplotlib.Axes <axis>, or
        on a new Axes if <axis> is None."""
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        axis.set_color_cycle(self.colors)
        dt = (numpy.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        
        if id in range(self.n_channels):
            for i,d in enumerate(self.datasets):
                pylab.plot(dt,d.average_pulses[id], label="Demod TES %d"%i)
        else:
            for i,d in enumerate(self.datasets):
                pylab.plot(dt,d.average_pulses[i], label="Demod TES %d"%i)
        pylab.xlabel("Time past trigger (ms)")
        pylab.legend(loc='best')


    def plot_raw_spectra(self):
        """Plot distribution of raw pulse averages, with and without gain"""
        ds = self.datasets[0]
        meangain = ds.p_pulse_average[ds.cuts.good()].mean()/ds.gain
        pylab.clf()
        pylab.subplot(211)
        for ds in self.datasets:
            gain = ds.gain
            _=pylab.hist(ds.p_pulse_average[ds.cuts.good()], 200, [meangain*.8, meangain*1.2], alpha=0.5)
            
        pylab.subplot(212)
        for ds in self.datasets:
            gain = ds.gain
            _=pylab.hist(ds.p_pulse_average[ds.cuts.good()]/gain, 200, [meangain*.8,meangain*1.2], alpha=0.5)
            print ds.p_pulse_average[ds.cuts.good()].mean()
        return meangain
        

    def set_gains(self, gains):
        """Set the datasets to have the given gains.  These gains will be used when
        averaging pulses in self.compute_average_pulse() and in ...***?"""
        if len(gains) != self.n_channels:
            raise ValueError("gains must have the same length as the number of datasets (%d)"
                             %self.n_channels)
            
        for g,d in zip(gains, self.datasets):
            d.gain = g
    
    
    def compute_filters(self, fmax=None, f_3db=None):
        
        # Analyze the noise, if not already done
        for n in self.datasets:
            if n.noise_autocorr is None or n.noise_spectrum is None:
                print "Computing noise autocorrelation and spectrum"
                self.compute_noise_spectra()
                break
            
        self.filters=[]
        for i,ds in enumerate(self.datasets):
            if ds.cuts.good().sum() < 10:
                print 'Cannot compute filter for channel %d'%i 
                self.filters.append(None)
                continue
            print "Computing filter %d of %d"%(i, self.n_channels)
            avg_signal = ds.average_pulses[i].copy()
            
            f = mass.channel.Filter(avg_signal, self.nPresamples-ds.pretrigger_ignore_samples, ds.noise_spectrum.spectrum(),
                                    ds.noise_autocorr, sample_time=self.timebase,
                                    fmax=fmax, f_3db=f_3db, shorten=2)
            self.filters.append(f)
            

    def plot_filters(self, first=0, end=-1):
        """Plot the filters from <first> through <end>-1.  By default, plots all filters,
        except that the maximum number is 8.  Left panels are the Fourier and time-domain
        X-ray energy filters.  Right panels are two different filters for estimating the 
        baseline level.
        """
        pylab.clf()
        if end<=first: end=self.n_channels
        if first >= self.n_channels:
            raise ValueError("First channel must be less than %d"%self.n_channels)
        nplot = min(end-first, 8)
        filters = self.filters[first:first+nplot]
        for i,f in enumerate(filters):
            ax1 = pylab.subplot(nplot,2,1+2*i)
            ax2 = pylab.subplot(nplot,2,2+2*i)
            ax1.set_title("TES %d signal"%(first+i))
            ax2.set_title("TES %d baseline"%(first+i))
            for ax in (ax1,ax2): ax.set_xlim([0,self.nSamples])
            f.plot(axes=(ax1,ax2))

        
    
    def summarize_filters(self):
        rms_fwhm = numpy.sqrt(numpy.log(2)*8) # FWHM is this much times the RMS
        print 'V/dV for time, Fourier filters: '
        for i,f in enumerate(self.filters):
            try:
                rms = numpy.array((f.variances['noconst'],
                                   f.variances['fourier'],
                                   ))**0.5
                v_dv = (1/rms)/rms_fwhm
                print "[ %6.1f  %6.1f ]"%(v_dv[0],v_dv[1]) , 'Predicted res: [ %6.3f %6.3f ] (eV)'%(5898.8/v_dv[0],5898.8/v_dv[1])
            except:
                print "Filter %d can't be used"%i
                            
            
    
    def filter_data(self, filter_name=None):
        """Filter data sets and store in datasets[*].p_filt_phase and _value.
        The filters are currently self.filter[*].filt_noconst"""
        if self.filters is None:
            self.compute_filters()
            
        if filter_name is None: filter_name='filt_noconst'
        
        for dset in self.datasets:
            dset.p_filt_phase = numpy.zeros(dset.nPulses, dtype=numpy.float)
            dset.p_filt_value = numpy.zeros(dset.nPulses, dtype=numpy.float)
            
        for first, end in self.iter_segments():
            if end>self.nPulses:
                end = self.nPulses 
            print "Records %d to %d loaded"%(first,end-1)
            for _i,(filter,dset) in enumerate(zip(self.filters,self.datasets)):
                filt_vector = filter.__dict__[filter_name]
                peak_x, peak_y = dset.filter_data(filt_vector,first, end)
                dset.p_filt_phase[first:end] = peak_x
                dset.p_filt_value[first:end] = peak_y
            
    def plot_crosstalk(self, xlim=None, ylim=None, use_legend=True):
        pylab.clf()
        dt = (numpy.arange(self.nSamples)-self.nPresamples)*1e3*self.timebase
        
        ndet = self.n_channels
        plots_nx, plots_ny = ndet/2, 2
        for i in range(ndet):
            ax=pylab.subplot(plots_nx, plots_ny, 1+i)
            self.plot_average_pulses(i, axis=ax)
            pylab.plot(dt,self.datasets[i].average_pulses[i]/100,'k--', label="Main pulse/100")
            if xlim is None:
                xlim=[-.2,.2]
            if ylim is None:
                ylim=[-200,200]
            pylab.xlim(xlim)
            pylab.ylim(ylim)
            if use_legend: pylab.legend(loc='upper left')
            pylab.grid()
            pylab.title("Mean record when TES %d is hit"%i)
    
             
    def estimate_crosstalk(self, plot=True):
        """Work in progress..."""

        NMUX = self.n_channels

        if plot: 
            pylab.clf()
            ds0 = self.datasets[0]
            dt = (numpy.arange(ds0.nSamples)-ds0.nPresamples)*ds0.timebase*1e3
            ax0 = pylab.subplot(NMUX,NMUX,1)
        crosstalk = []
        dot = numpy.dot
        if self.datasets[0].noise_autocorr is None:
            self.compute_noise_spectra()
        
        for i,ds in enumerate( self.datasets):
            p = ds.average_pulses[i]
            d = numpy.zeros_like(p)
            d[1:] = p[1:]-p[:-1]
            
            if 'pulse_filt' in ds.__dict__ and 'deriv_filt' in ds.__dict__:
                qp = ds.pulse_filt
                qd = ds.deriv_filt
            else:
                print "Solving for TES %2d noise-inv-weighted pulse"%i
                R = scipy.linalg.toeplitz(ds.noise_autocorr[:len(p)])
                Rp = numpy.linalg.solve(R, p)
                print "Solving for TES %2d noise-inv-weighted derivative"%i
                Rd = numpy.linalg.solve(R, d)
                del R
                
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
                ct.append(dot(sig,qp)) # row i, col j
                if plot and (i != j):
                    ax = pylab.subplot(NMUX,NMUX,1+NMUX*j+i, sharex=ax0, sharey=ax0) # row j, col i
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
    
    
    def drift_correct_ptmean(self, filt_ranges):
        """A hack, but a great simple start on drift correction using the pretrigger mean.
        DEPRECATED: see MicrocalDataSet.auto_drift_correct instead."""
        
        class LineDrawer(object):
            def __init__(self, figure):
                self.b, self.x, self.y = 0,0,0
                self.fig = figure
                self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
                print "Connected cid=%d"%self.cid
            def __call__(self, event):
                self.b, self.x, self.y =  event.button, event.xdata, event.ydata
            def __del__(self):
                self.fig.canvas.mpl_disconnect(self.cid)
                print 'Disconnected cid=%d'%self.cid
        
        for _i,(rng,ds) in enumerate(zip(filt_ranges,self.datasets)):
            pylab.clf()
            fig = pylab.gcf()
            good = ds.cuts.good()
            pylab.plot(ds.p_pretrig_mean[good], ds.p_filt_value[good],'.')
            pylab.ylim(rng)
            print "Click button 1 twice to draw a line; click button 2 to quit"
            pf = LineDrawer(fig)
            x,y=[],[]
            line = None
            offset,slope = 0.0,0.0
            while True:
                pylab.waitforbuttonpress()
                if pf.b > 1: break
                x.append(pf.x)
                y.append(pf.y)
                if len(x)==2:
                    if line is not None:
                        line.remove()
                    line, = pylab.plot(x, y, 'r')
                    offset = x[0]
                    slope = (y[1]-y[0])/(x[1]-x[0])
                    if ds.p_filt_value_phc[0]==0: ds.p_filt_value_phc=ds.p_filt_value
                    ds.p_filt_value_dc = ds.p_filt_value_phc - (ds.p_pretrig_mean-offset)*slope
                    ds.energy = ds.p_filt_value_dc[good]    
                    print offset,slope, ' corrects the energy.  Hit button 2 to move on, or try again.'
                    x,y = [],[]
                
            del pf


    def find_features_with_mouse(self, channame='p_filt_value', nclicks=1, xrange=None, trange=None):
        """
        Plot histograms of each channel's "energy" spectrum, one channel at a time.
        After recording the x-coordinate of <nclicks> mouse clicks per plot, return an
        array of shape (N_channels, N_click) containing the "energy" of each click.
        
        <channame>  A string to choose the desired energy-like parameter.  Probably you want
                    to start with p_filt_value or p_filt_value_dc and later (once an energy
                    calibration is in place) p_energy.
        <nclicks>   The number of x coordinates to record per detector.  If you want to get
                    for example, a K-alpha and K-beta line in one go, then choose 2.
        <xrange>    A 2-element sequence giving the limits to histogram.  If None, then the
                    histogram will show all data.
        <trange>    A 2-element sequence giving the time limits to use (in ms).  If None, then the
                    histogram will show all data.
                    
        Returns:
        A numpy.ndarray of shape (self.n_channels, nclicks).  
        """
        x = []
        for i,ds in enumerate(self.datasets):
            pylab.clf()
            g = ds.cuts.good()
            if trange is not None:
                g = numpy.logical_and(g, ds.p_timestamp>trange[0])
                g = numpy.logical_and(g, ds.p_timestamp<trange[1])
            pylab.hist(ds.__dict__[channame][g], 200, range=xrange)
            pylab.xlabel(channame)
            pylab.title("Detector %d: attribute %s"%(i, channame))
            fig = pylab.gcf()
            pf = mass.utilities.MouseClickReader(fig)
            for i in range(nclicks):
                while True:
                    pylab.waitforbuttonpress()
                    try:
                        pfx = '%g'%pf.x
                    except TypeError:
                        continue
                    print 'Click on line #%d at %s'%(i+1, pfx)
                    x.append(pf.x)
                    break
            del pf
        xvalues = numpy.array(x)
        xvalues.shape=(self.n_channels, nclicks)
        return xvalues


    def find_named_features_with_mouse(self, name='Mn Ka1', channame='p_filt_value', xrange=None, trange=None, energy=None):
        
        if energy is None:
            energy = mass.energy_calibration.STANDARD_FEATURES[name]
        
        print "Please click with the mouse on each channel's histogram at the %s line"%name
        xvalues = self.find_features_with_mouse(channame=channame, nclicks=1, xrange=xrange, trange=trange).ravel()
        for ds,xval in zip(self.datasets, xvalues):
            calibration = ds.calibration[channame]
            calibration.add_cal_point(xval, energy, name)


    def report(self):
        """
        Report on the number of data points and similar
        """
        for i,ds in enumerate(self.datasets):
            ng = ds.cuts.nUncut()
            good = ds.cuts.good()
            dt = (ds.p_timestamp[good][-1]*1.0 - ds.p_timestamp[good][0])/1e3  # seconds
            np = numpy.arange(len(good))[good][-1] - good.argmax() + 1
            rate = (np-1.0)/dt
#            grate = (ng-1.0)/dt
            print 'TES %2d %6d pulses (%6.3f Hz over %6.4f hr) %6.3f%% good'%(i, np, rate, dt/3600., 100.0*ng/np)


    def plot_noise_autocorrelation(self, axis=None, channels=None):
        """Compare the noise autocorrelation functions.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        """
        
        if channels is None:
            channels = numpy.arange(self.n_channels)

        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        axis.grid(True)
        for i,noise in enumerate(self.noise_channels):
            if i not in channels: continue
            noise.plot_autocorrelation(axis=axis, label='TES %d'%i, color=self.colors[i%len(self.colors)])
#        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        pylab.legend(loc='upper right')
    
    


class TESGroup(BaseChannelGroup):
    """
    A group of one or more *independent* microcalorimeters, in that
    they are time-division multiplexed.  It might be convenient to use
    this for multiple TDM channels, or for singles.  The key is that
    this object offers the same interface as the CDMGroup object
    (which has to be more complex under the hood).
    """
    def __init__(self, filenames, noise_filenames=None, noise_only=False, pulse_only=False):
        super(self.__class__, self).__init__(filenames, noise_filenames)
        self.noise_only = noise_only
        
        pulse_list = []
        noise_list = []
        dset_list = []
        for i,fname in enumerate(self.filenames):
            if noise_filenames is None:
                pulse, noise = mass.channel.create_pulse_and_noise_records(fname, noise_only=noise_only,
                                                                           pulse_only=pulse_only)
            else:
                nf = self.noise_filenames[i]
                print "nf='%s'"%nf
                pulse, noise = mass.channel.create_pulse_and_noise_records(fname, noisename=nf,
                                                                           noise_only=noise_only,
                                                                           pulse_only=pulse_only)
            dset = mass.channel.MicrocalDataSet(pulse.__dict__)
            pulse_list.append(pulse)
            if noise is not None:
                noise_list.append(noise)
            dset_list.append(dset)
            
            if self.n_segments is None:
                for attr in "n_segments","nPulses","nSamples","nPresamples", "timebase":
                    self.__dict__[attr] = pulse.__dict__[attr]
            else:
                for attr in "nSamples","nPresamples", "timebase":
                    if self.__dict__[attr] != pulse.__dict__[attr]:
                        raise ValueError("Unequal values of %s: %f != %f"%(attr,float(self.__dict__[attr]),
                                                                            float(pulse.__dict__[attr])))
                self.n_segments = max(self.n_segments, pulse.n_segments)
                self.nPulses = max(self.nPulses, pulse.nPulses)
            
        self.channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)
        if len(pulse_list)>0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg
        self.noise_filenames = [n.datafile.filename for n in self.noise_channels]
    

    def copy(self):
        g = TESGroup([])
        g.__dict__.update(self.__dict__)
        g.channels = tuple([c.copy() for c in self.channels])
        g.datasets = tuple([d.copy() for d in self.datasets])
        g.noise_channels = tuple([c.copy() for c in self.noise_channels])
#        g.filters = tuple([f.copy() for f in self.filters])
        return g
        
    
    def join(self, *others):
        # Ensure they are compatible
        for g in others:
            for attr in ('nPresamples','nSamples', 'noise_only', 'timebase'):
                if g.__dict__[attr] != self.__dict__[attr]:
                    raise RuntimeError("All objects must agree on group.%s"%attr)
            
        for g in others:
            n_extra = self.n_channels
            for ds in g.datasets:
                ds.average_pulses = numpy.vstack((numpy.zeros((n_extra,self.nSamples),dtype=numpy.float),
                                                  ds.average_pulses))
            
            self.channels += g.channels
            self.datasets += g.datasets
            self.filters += g.filters()
            self.noise_channels += g.noise_channels
            self.n_channels += g.n_channels
            self.n_segments = max(self.n_segments, g.n_segments)
        
        self.clear_cache()
        
    
    def set_segment_length(self, seg_length):
        self.clear_cache()
        raise NotImplementedError("ugh!")
#        for chan, dset in zip(self.channels, self.datasets):
        
        
    def read_segment(self, segnum, use_cache=True):
        """Read segment number <segnum> into memory for each of the
        channels in the group.  Return (first,end) where these are the
        number of the first record in that segment and 1 more than the
        number of the last record.
        
        When <use_cache> is true, we use cached value when possible.
        """
        if segnum == self._cached_segment and use_cache:
            return self._cached_pnum_range
        
        first_pnum,end_pnum = -1,-1
        for chan, dset in zip(self.channels, self.datasets):
            a,b = chan.read_segment(segnum)
            dset.data = chan.data
            dset.times = chan.datafile.datatimes

            # Possibly some channels are shorter than others (in TDM data)
            # Make sure to return first_pnum,end_pnum for longest VALID channel only 
            if a>=0:
                if first_pnum>=0:
                    assert a==first_pnum 
                first_pnum=a
            if b>=end_pnum: end_pnum=b
        self._cached_segment = segnum
        self._cached_pnum_range = first_pnum, end_pnum
        return first_pnum, end_pnum
    
    
    def summarize_data(self):
        """
        Compute summary quantities for each pulse.
        """

        t0 = time.time()
        super(self.__class__, self).summarize_data()
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
        BaseChannelGroup.compute_average_pulse(self, masks, use_crosstalk_masks=False, subtract_mean=subtract_mean)
        
        
    def plot_noise(self, axis=None, channels=None, scale_factor=1.0, sqrt_psd=False):
        """Compare the noise power spectra.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        <scale_factor> Multiply counts by this number to get physical units. 
        <sqrt_psd>     Whether to show the sqrt(PSD) or (by default) the PSD itself.
        """
        
        if channels is None:
            channels = numpy.arange(self.n_channels)

        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        axis.set_color_cycle(self.colors)
        axis.grid(True)
        for i,noise in enumerate(self.noise_channels):
            if i not in channels: continue
            yvalue = noise.spectrum.spectrum()*scale_factor**2
            if sqrt_psd:
                yvalue = numpy.sqrt(yvalue)
            axis.plot(noise.spectrum.frequencies(), yvalue, label='TES %d'%i)
        f=self.noise_channels[0].spectrum.frequencies()
        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        axis.loglog()
        pylab.legend(loc='upper right')
    
    
    def compute_noise_spectra(self, max_excursion=9e9):
        for dataset,noise in zip(self.datasets,self.noise_channels):
            noise.compute_power_spectrum(plot=False, max_excursion=max_excursion)
            dataset.noise_spectrum = noise.spectrum
            noise.compute_autocorrelation(n_lags=self.nSamples, plot=False, max_excursion=max_excursion)
            dataset.noise_autocorr = noise.autocorrelation



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
        if noise_only:
            self.n_cdm = len(filenames)
        else:
            self.n_cdm = self.n_channels  # If >1 column, this won't be true anymore

#        if demodulation is None:
#            demodulation = numpy.array(
#                (( 1, 1, 1, 1),
#                 (-1, 1, 1,-1),
#                 (-1,-1, 1, 1),
#                 (-1, 1,-1, 1)), dtype=numpy.float)
        assert demodulation.shape[0] == demodulation.shape[1]
        assert demodulation.shape[0] == self.n_cdm
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
        self.noise_filenames = [n.datafile.filename for n in self.noise_channels]
        self.REMOVE_INFRAME_DRIFT=True
        

    def copy(self):
        g = CDMGroup(self.filenames, self.noise_filenames,
                     demodulation=self.demodulation, noise_only=self.noise_only)
        g.__dict__.update(self.__dict__)
        g.raw_channels = tuple([c.copy() for c in self.raw_channels])
        g.noise_channels = tuple([c.copy() for c in self.noise_channels])
        g.datasets = tuple([c.copy() for c in self.datasets])
#        g.filters = tuple([f.copy() for f in self.filters])
        return g
        

    def set_segment_size(self, seg_length):
        self.clear_cache()
        for chan in self.raw_channels:
            chan.set_segment_size(seg_length)
        self.n_segments = self.raw_channels[0].n_segments

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
            dset.times = chan.datafile.datatimes

        # Remove linear drift
        seg_size = min([rc.data.shape[0] for rc in self.raw_channels]) # Last seg can be of unequal size!
        mod_data = numpy.zeros([self.n_channels, seg_size, self.nSamples], dtype=numpy.int32)
        wide_holder = numpy.zeros([seg_size, self.nSamples], dtype=numpy.int32)
        
        mod_data[0, :, :] = numpy.array(self.raw_channels[0].data[:seg_size, :])
        for i in range(1, self.n_channels):
            if self.REMOVE_INFRAME_DRIFT:
                mod_data[i, :, :] =  self.raw_channels[i].data[:seg_size,:]
                wide_holder[:, 1:] = self.raw_channels[i].data[:seg_size,:-1]  # Careful never to mult int16 by more than 4! 
                mod_data[i, :, 1:] *= (self.n_channels-i)   # Weight to future value
                mod_data[i, :, 1:] += i  *wide_holder[:,1:] # Weight to prev value
                mod_data[i, :, 1:] /= self.n_channels       # Divide by total weight
            else:
                mod_data[i, :, :] = numpy.array(self.raw_channels[i].data[:seg_size,:])
        
        # Demodulate
        for i_det,dset in enumerate(self.datasets):   
            dset.data = numpy.zeros((seg_size, self.nSamples), dtype=numpy.float)
            for j in range(self.n_channels):
                dset.data += self.demodulation[i_det,j]*mod_data[j, :, :]
    
        self._cached_segment = segnum
        self._cached_pnum_range = first,end
        return first, end


    def summarize_data(self):
        """
        Compute summary quantities for each pulse.
        """

        t0 = time.time()
        super(self.__class__, self).summarize_data()

        # How many detectors were hit in each record?
        self.nhits = numpy.array([d.p_pulse_average>50 for d in self.datasets]).sum(axis=0)
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
            nc.data = numpy.zeros(shape, dtype=numpy.float)
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
            nc.compute_autocorrelation(n_lags=self.nSamples, n_data=self.nSamples*nrec, plot=False)
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
            channels = numpy.arange(self.n_cdm)
            
        for ds in self.datasets:
            if ds.noise_spectrum is None:
                self.compute_noise_spectra(compute_raw_spectra=True)
                break
            
        
        pylab.clf()
        if show_modulated:
            ax1=pylab.subplot(211)
            ax1.set_title("Raw (modulated) channel noise power spectrum")
            ax2=pylab.subplot(212, sharex=ax1, sharey=ax1)
            axes=(ax1,ax2)
        else:
            ax2=pylab.subplot(111)
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
            pylab.legend(loc='upper right')
            
        for i,ds in enumerate(self.datasets):
            if i not in channels: continue
            yvalue = ds.noise_spectrum.spectrum()*scale_factor**2
            if sqrt_psd:
                yvalue = numpy.sqrt(yvalue)
            pylab.plot(ds.noise_spectrum.frequencies(), yvalue,
                       label='TES %d'%i, color=self.colors[i])
        pylab.legend(loc='lower left')


    def plot_noise_autocorrelation(self, axis=None, channels=None):
        """Compare the noise autocorrelation functions.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        """
        
        if channels is None:
            channels = numpy.arange(self.n_channels)

        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            
        axis.grid(True)
        for i,noise in enumerate(self.noise_channels_demod):
            if i not in channels: continue
            noise.plot_autocorrelation(axis=axis, label='TES %d'%i, color=self.colors[i%len(self.colors)])
#        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        pylab.legend(loc='upper right')    

    
    def plot_undecimated_noise(self, ndata=None):
        """Show the noise as if """
        np = numpy.array([nc.nPulses for nc in self.noise_channels]).min()
        if ndata is None:
            ndata = np*self.noise_channels[0].nSamples
        assert ndata<=1024*1024
        data = numpy.asarray(numpy.vstack([nc.data[:np,:].ravel()[:ndata] for nc in self.noise_channels]), dtype=numpy.int16)
        for r in data:
            r -= r.mean()
        data=data.T.ravel()

        segfactor=max(8, ndata/1024)
        
        freq, psd = mass.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=self.timebase/self.n_cdm, window=mass.power_spectrum.hamming)
        pylab.clf()
        pylab.plot(freq, psd)
    
    def update_demodulation(self, relative_response):
        relative_response = numpy.asmatrix(relative_response)
        if relative_response.shape != (self.n_channels,self.n_channels):
            raise ValueError("The relative_response matrix needs to be of shape (%d,%d)"%
                             (self.n_channels, self.n_channels))
            
        self.demodulation = numpy.dot( relative_response.I, self.demodulation)


    def plot_modulated_demodulated(self, pulsenum=119148, modulated_offsets=numpy.arange(15000,-1,-5000), xlim=[-5,15.726]):
        "Plot one record both modulated and demodulated"
        
        self.ms = (numpy.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        pylab.clf()
        
        self.read_trace(pulsenum)
        n = pulsenum - self._cached_pnum_range[0]
        pylab.subplot(211)
        if modulated_offsets is None:
            modulated_offsets = (0,0,0,0)
        for i,rc in enumerate(self.raw_channels):
            pylab.plot(self.ms, rc.data[n,:]+modulated_offsets[i]-rc.data[n,:self.nPresamples].mean(), color=self.colors[i], label='SQUID sw%d'%i)
        if xlim is not None: pylab.xlim(xlim)
        pylab.legend(loc='upper left')
        pylab.title("Modulated (raw) signal")
        

        pylab.subplot(212)
        for i,ds in enumerate(self.datasets):
            pylab.plot(self.ms, ds.data[n,:]-ds.p_pretrig_mean[pulsenum], color=self.colors[i], label='TES %d'%i)
        if xlim is not None: pylab.xlim(xlim)
        pylab.legend(loc='upper left')
        pylab.title("Demodulated signal")
        pylab.xlabel("Time since trigger (ms)")





class CrosstalkVeto(object):
    """
    An object to allow vetoing of data in 1 channel when another is hit
    """
    
    def __init__(self, datagroup, window_ms=(-10,3), pileup_limit=100):
        if datagroup is None:
            return
        
        window_ms = numpy.array(window_ms, dtype=numpy.int)
        self.window_ms = window_ms
        self.n_channels = datagroup.n_channels
        self.n_pulses = datagroup.nPulses
#        self.veto = numpy.zeros((self.n_channels, self.n_pulses), dtype=numpy.bool8)
        
        ms0 = numpy.array([ds.p_timestamp[0] for ds in datagroup.datasets]).min() + window_ms[0]
        ms9 = numpy.array([ds.p_timestamp.max() for ds in datagroup.datasets]).max() + window_ms[1]
        self.nhits = numpy.zeros(ms9-ms0+1, dtype=numpy.int8)
        self.time0 = ms0
        
        
        for ds in datagroup.datasets:
            g = numpy.ones(ds.nPulses, dtype=numpy.bool8)
            g = ds.cuts.good()
            vetotimes = ds.p_timestamp[g]-ms0
            vetotimes[vetotimes<0] = 0
            print vetotimes, len(vetotimes), 1.0e3*ds.nPulses/(ms9-ms0),
            a,b = window_ms
            b+=1
            for t in vetotimes:
                self.nhits[t+a:t+b] += 1
                
            pileuptimes = vetotimes[ds.p_max_posttrig_deriv[g]>pileup_limit]
            print len(pileuptimes)
            for t in pileuptimes:
                self.nhits[t+b:t+b+8] += 1
                
    def copy(self):
        v = CrosstalkVeto(None)
        v.__dict__ = self.__dict__.copy()
        return v
    
    
    def veto(self, times):    
        index = times-self.time0
        return self.nhits[index]>1