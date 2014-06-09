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
__all__=['TESGroup','CDMGroup','CrosstalkVeto', 'unpickle_TESGroup']

import numpy as np
import pylab as plt

import time, os
import cPickle

import mass.calibration
import mass.calibration.inlineUpdater as inlineUpdater

from mass.core.channel import create_pulse_and_noise_records

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
        self._allowed_pnum_ranges = None
        self._allowed_segnums = None
        self.pulses_per_seg = None
        self._bad_channums=set()
        
        if self.n_channels <=4:
            self.colors=("blue", "#aaaa00","green","red")
        else:
            BRIGHTORANGE='#ff7700'
            self.colors=('purple',"blue","cyan","green","gold",BRIGHTORANGE,"red","brown")


    def __iter__(self):
        """Iterator over the self.datasets in channel number order"""
        for ds in self.iter_channels():
            yield ds
    
    def iter_channels(self, include_badchan=False):
        """Iterator over the self.datasets in channel number order
        include_badchan : whether to include officially bad channels in the result."""
        channum = self.channel.keys()
        if not include_badchan:
            channum = list(set(channum) - set(self._bad_channums))
        channum.sort()
        for c in channum:
            yield self.channel[c]
    
    def iter_channel_numbers(self, include_badchan=False):
        """Iterator over the channel numbers in numerical order
        include_badchan : whether to include officially bad channels in the result."""
        channum = self.channel.keys()
        if not include_badchan:
            channum = list(set(channum) - set(self._bad_channums))
        channum.sort()
        for c in channum:
            yield c
    
    def set_chan_good(self, *args):
        """Set one or more channels to be good.  (No effect for channels already listed
        as good.)
        *args  Arguments to this function are integers or containers of integers.  Each 
               integer is removed from the bad-channels list."""
        added_to_list = set()
        for a in args:
            try:
                goodones = set(a)
            except TypeError:
                goodones = set([a])
            self._bad_channums -= goodones
            added_to_list.update(goodones)
        added_to_list = list(added_to_list)
        added_to_list.sort()
        print "Removed channels %s from bad channel list"%(added_to_list)
        self.update_chan_info()
    
    def set_chan_bad(self, *args):
        """Set one or more channels to be bad.  (No effect for channels already listed
        as bad.)
        *args  Arguments to this function are integers or containers of integers.  Each 
               integer is added to the bad-channels list."""
        added_to_list = set()
        comment = ''
        for a in args:
            if type(a) is type(comment):
                comment = a
                continue
            try:
                badones = set(a)
            except TypeError:
                badones = set([a])
            self._bad_channums.update(badones)
            added_to_list.update(badones)

        if len(added_to_list) > 0:
            added_to_list = list(added_to_list)
            added_to_list.sort()
            log_string = 'chan %s flagged bad because %s'%(added_to_list, comment)
            self.why_chan_bad.append(log_string)
            print log_string
        self.update_chan_info()


    def update_chan_info(self):        
        channum = self.channel.keys()
        channum = list(set(channum) - set(self._bad_channums))
        channum.sort()
        self.num_good_channels = len(channum)
        self.good_channels = list(channum)
        if self.num_good_channels>0: 
            self.first_good_dataset = self.channel[channum[0]]
        elif len(channum) > 0:
            print("WARNING: All datasets flagged bad, most things won't work.")
            self.first_good_dataset = None
        
    def _setup_channels_list(self):
        self.channel = {}
        self.why_chan_bad = []
        for ds_num,ds in enumerate(self.datasets):
            try:
                ds.index = ds_num
                self.channel[ds.channum] = ds
            except AttributeError:
                pass
        self.update_chan_info()

    
#    def pop_dataset(self, dataset_number):
#        """Remove self.datasets[dataset_number] from the channel group.
#        Returns the offending dataset just in case you need it."""
#        if dataset_number >= self.n_channels or dataset_number<0:
#            raise ValueError("Cannnot remove dataset_number=%d, but only [0,%d]%"%
#                             (dataset_number, self.n_channels-1))
#        datasets = list(self.datasets)
#        removed = datasets.pop(dataset_number)
#        self.datasets = tuple(datasets)
#        self.n_channels -= 1
#        self._setup_channels_list()
#        return removed
#
#
#    def insert_dataset(self, dataset_number, dataset):
#        """Insert <dataset> into this group of channels at position
#        <dataset_number>, which follows the semantics of list.insert()."""
#        if not isinstance(dataset, mass.MicrocalDataSet):
#            raise ValueError("Argument dataset must be a mass.MicrocalDataSet.")
#        datasets = list(self.datasets)
#        datasets.insert(dataset_number, dataset)
#        self.datasets = tuple(datasets)
#        self.n_channels += 1
#        self._setup_channels_list()

        
    def clear_cache(self):
        """Invalidate any cached raw data."""
        self._cached_segment = None
        self._cached_pnum_range = None
        for ds in self.datasets: ds.data=None
        if 'raw_channels' in self.__dict__:
            for rc in self.raw_channels: rc.data=None
        if 'noise_channels' in self.__dict__:
            for nc in self.noise_channels: nc.datafile.clear_cache()
 
    def sample2segnum(self, samplenum):
        """Returns the segment number of sample number <samplenum>."""
        if samplenum >= self.nPulses:
            samplenum = self.nPulses-1
        return samplenum/self.pulses_per_seg


    def segnum2sample_range(self, segnum):
        """Return the (first,end) sample numbers of the segment numbered <segnum>.
        Note that <end> is 1 beyond the last sample number in that segment."""
        return (segnum*self.pulses_per_seg, (segnum+1)*self.pulses_per_seg)


    def set_data_use_ranges(self, ranges=None):
        """Set the range of sample numbers that this object will use when iterating over
        raw data.
        
        <ranges> can be None (which causes all samples to be used, the default);
                or a 2-element sequence (a,b), which causes only a through b-1 inclusive to be used;
                or a sequence of 2-element sequences, which is like the previous but with multiple sample ranges allowed. 
        """
        allowed_ranges=[]
        if ranges is None:
            allowed_ranges=[[0, self.nPulses]]
        elif len(ranges)==2 and np.isscalar(ranges[0]) and np.isscalar(ranges[1]):
            allowed_ranges = [[ranges[0], ranges[1]]]
        else:
            allowed_ranges = [r for r in ranges]
        
        allowed_segnums = np.zeros(self.n_segments, dtype=np.bool)
        for first,end in allowed_ranges:
            assert first <= end
            for sn in range(self.sample2segnum(first), self.sample2segnum(end-1)+1):
                allowed_segnums[sn] = True
            
        self._allowed_pnum_ranges = allowed_ranges
        self._allowed_segnums = allowed_segnums
        
        if ranges is not None:
            print 'Warning!  This feature is only half-complete.  Currently, the granularity is limited.'
            print '   Only full "segments" of size %d records can be ignored.'%self.pulses_per_seg
            print '   Will use %d segments and ignore %d.'%(self._allowed_segnums.sum(), self.n_segments-self._allowed_segnums.sum())

    def iter_segments(self, first_seg=0, end_seg=-1, sample_mask=None, segment_mask=None):
        if self._allowed_segnums is None:
            self.set_data_use_ranges(None)
            
        if end_seg < 0: 
            end_seg = self.n_segments
        for i in range(first_seg, end_seg):
            if not self._allowed_segnums[i]: 
                continue
            a,b = self.segnum2sample_range(i)
            if sample_mask is not None:
                if b>len(sample_mask):
                    b=len(sample_mask)
                if not sample_mask[a:b].any():
                    print 'We can skip segment %4d'%i
                    continue # Don't need anything in this segment.  Sweet!
            if segment_mask is not None:
                if not segment_mask[i]:
                    print 'We can skip segment %4d'%i
                    continue # Don't need anything in this segment.  Sweet!
            first_rnum, end_rnum = self.read_segment(i)
            yield first_rnum, end_rnum


    def summarize_data(self, peak_time_microsec = 220.0, pretrigger_ignore_microsec = 20.0, include_badchan = False):
        """
        Compute summary quantities for each pulse.  Subclasses override this with methods
        that ought to call this!
        """
        printUpdater = inlineUpdater.InlineUpdater('BaseChannelGroup.summarize_data')
        print "summarize_data: This data set has (up to) %d records with %d samples apiece."%(
            self.nPulses, self.nSamples)  
        for first, end in self.iter_segments():
            if end>self.nPulses:
                end = self.nPulses 
            printUpdater.update(end/float(self.nPulses))
            for ds in self.iter_channels(include_badchan):
                ds.summarize_data(first, end, peak_time_microsec, pretrigger_ignore_microsec)

    def summarize_data_tdm(self, peak_time_microsec = 220.0, pretrigger_ignore_microsec = 20.0, include_badchan = False, forceNew=False):
        printUpdater = inlineUpdater.InlineUpdater('summarize_data_tdm')
        for chan in self.iter_channel_numbers(include_badchan):
            self.channel[chan].summarize_data_tdm(peak_time_microsec, pretrigger_ignore_microsec, forceNew)    
            if include_badchan:
                printUpdater.update((chan/2+1)/float(len(self.channel.keys())))
            else:
                printUpdater.update((chan/2+1)/float(self.num_good_channels))
    
    def read_trace(self, record_num, dataset_num=0, chan_num=None):
        """Read (from cache or disk) and return the pulse numbered <record_num> for 
        dataset number <dataset_num> or channel number <chan_num>.
        If both are given, then <chan_num> will be used when valid.  
        If this is a CDMGroup, then the pulse is the demodulated
        channel by that number."""
        ds = self.channel.get(chan_num, self.datasets[dataset_num])
        seg_num = record_num / self.pulses_per_seg
        self.read_segment(seg_num)
        return ds.data[record_num % self.pulses_per_seg]
        
        
        
        
    def plot_traces(self, pulsenums, dataset_num=0, chan_num=None, pulse_summary=True, axis=None, difference=False,
                    residual=False, valid_status=None):
        """Plot some example pulses, given by sample number.
        <pulsenums>   A sequence of sample numbers, or a single one.
        <dataset_num> Dataset index (0 to n_dets-1, inclusive).  Will be used only if 
                      <chan_num> is invalid.
        <chan_num>    Dataset channel number.  If valid, it will be used instead of dataset_num.
        
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A plt axis to plot on.
        <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
        <residual>   Whether to show the residual between data and opt filtered model, or just raw data.
        <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                     that have been cut.  If "cut", show only those that have been cut.
        """
        if isinstance(pulsenums, int):
            pulsenums = (pulsenums,)
        pulsenums = np.asarray(pulsenums)
        if chan_num in self.channel:
            dataset = self.channel[chan_num]
            dataset_num = dataset.index
        else:
            dataset = self.datasets[dataset_num]
            if chan_num is not None:
                print "Cannot find chan_num[%d], so using dataset #%d"%(
                                            chan_num, dataset_num)
        
        # Don't print pulse summaries if the summary data is not available
        if pulse_summary:
            try:
                if len(dataset.p_pretrig_mean) == 0:
                    pulse_summary = False
            except AttributeError:
                pulse_summary = False
        
        if valid_status not in (None, "valid", "cut"):
            raise ValueError("valid_status must be one of [None, 'valid', or 'cut']")
        if residual and difference:
            raise ValueError("Only one of residual and difference can be True.")
            
        dt = (np.arange(dataset.nSamples)-dataset.nPresamples)*dataset.timebase*1e3
        color= 'purple','blue','green','#88cc00','gold','orange','red', 'brown','gray','#444444','magenta'
#        ncolors = max(len(pulsenums), 20)
#        cmap = plt.cm.get_cmap(name='spectral')
#        color = cmap(np.arange(ncolors,dtype=np.float)/ncolors)
        MAX_TO_SUMMARIZE = 20
        
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        axis.set_xlabel("Time after trigger (ms)")
        axis.set_xlim([dt[0], dt[-1]])
        axis.set_ylabel("Feedback (or mix) in [Volts/16384]")
        if pulse_summary:
            axis.text(.975, .97, r"              -PreTrigger-   Max  Rise t Peak   Pulse", 
                       size='medium', family='monospace', transform = axis.transAxes, ha='right')
            axis.text(.975, .95, r"Cut P#    Mean     rms PTDeriv  ($\mu$s) value   mean", 
                       size='medium', family='monospace', transform = axis.transAxes, ha='right')

        cuts_good = dataset.cuts.good()[pulsenums]
        pulses_plotted = -1
        for i,pn in enumerate(pulsenums):
            if valid_status == 'cut' and cuts_good[i]: continue
            if valid_status == 'valid' and not cuts_good[i]: continue
            pulses_plotted += 1
            
            data = self.read_trace(pn, dataset_num=dataset_num)
            if difference:
                data = data*1.0-np.roll(data,1)
                data[0] = 0
                data += np.roll(data,1) + np.roll(data,-1)
                data[0] = 0
            elif residual:
                model = dataset.p_filt_value[pn] * dataset.average_pulse / dataset.average_pulse.max()
                data = data-model
                
            cutchar,alpha,linestyle,linewidth = ' ',1.0,'-',1
            
            # When plotting both cut and valid, mark the cut data with x and dashed lines
            if valid_status is None and not cuts_good[i]:
                cutchar,alpha,linestyle,linewidth = 'X',1.0,'--' ,1
            axis.plot(dt, data, color=color[pulses_plotted%len(color)], linestyle=linestyle, alpha=alpha,
                       linewidth=linewidth)
            if pulse_summary and pulses_plotted<MAX_TO_SUMMARIZE and len(dataset.p_pretrig_mean)>=pn:
                try:
                    summary = "%s%6d: %5.0f %7.2f %6.1f %5.0f %5.0f %7.1f"%(
                                cutchar, pn, dataset.p_pretrig_mean[pn], dataset.p_pretrig_rms[pn],
                                dataset.p_max_posttrig_deriv[pn], dataset.p_rise_time[pn]*1e6,
                                dataset.p_peak_value[pn], dataset.p_pulse_average[pn])
                except IndexError:
                    pulse_summary = False
                    continue
                axis.text(.975, .93-.02*pulses_plotted, summary, color=color[pulses_plotted%len(color)], 
                           family='monospace', size='medium', transform = axis.transAxes, ha='right')


    def plot_summaries(self, quantity, valid='uncut', downsample=None, log=False, hist_limits=None,
                        dataset_numbers=None):
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
                     
        <log>              Use logarithmic y-axis on the histograms (right panels).
        <hist_limits>
        <dataset_numbers>  A sequence of the datasets [0...n_channels-1] to plot.  If None (the default)
                           then plot all datasets in numerical order.
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
                
        if dataset_numbers is None:
            datasets = self.datasets
            dataset_numbers = range(len(datasets))
        else:
            datasets = [self.datasets[i] for i in dataset_numbers]

        plt.clf()
        ny_plots = len(datasets)
        for i,(channum,ds) in enumerate(zip(dataset_numbers, datasets)):
            print 'TES%2d '%channum,
            
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
                hour = ds.p_timestamp[valid_mask][::downsample]/3600.0
            else:
                nrecs = ds.nPulses
                if downsample is None:
                    downsample = ds.nPulses / 10000
                    if downsample < 1: downsample = 1
                hour = ds.p_timestamp[::downsample]/3600.0
            print " (%d records; %d in scatter plots)"%(
                nrecs,len(hour))
        
            (vect, label, color, default_limits) = plottable
            if hist_limits is None:
                limits = default_limits
            else:
                limits = hist_limits
            
            vect=eval("ds.%s"%vect)[valid_mask]
            
            if i==0:
                ax_master=plt.subplot(ny_plots, 2, 1+i*2)
            else:
                plt.subplot(ny_plots, 2, 1+i*2, sharex=ax_master)
                
            if len(vect)>0:
                plt.plot(hour, vect[::downsample],',', color=color)
            else:
                plt.text(.5,.5,'empty', ha='center', va='center', size='large', transform=plt.gca().transAxes)
            if i==0: plt.title(label)
            plt.ylabel("TES %d"%channum)
            if i==ny_plots-1:
                plt.xlabel("Time since server start (hours)")

            if i==0:
                axh_master = plt.subplot(ny_plots, 2, 2+i*2)
            else:
                if 'Pretrig Mean'==label:
                    plt.subplot(ny_plots, 2, 2+i*2)
                else:
                    plt.subplot(ny_plots, 2, 2+i*2, sharex=axh_master)
                
            if limits is None:
                in_limit = np.ones(len(vect), dtype=np.bool)
            else:
                in_limit= np.logical_and(vect>limits[0], vect<limits[1])
            if in_limit.sum()<=0:
                plt.text(.5,.5,'empty', ha='center', va='center', size='large', transform=plt.gca().transAxes)
            else:
                contents, _bins, _patches = plt.hist(vect[in_limit],200, log=log, 
                                                       histtype='stepfilled', fc=color, alpha=0.5)
            if i==ny_plots-1:
                plt.xlabel(label)
            if log:
                plt.ylim(ymin = contents.min())


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
            gains = np.ones(self.n_channels)
            
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
                abslim = 0.5*np.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = np.abs(dataset.p_pulse_average/gain-middle) <= abslim
                    if cut_crosstalk: 
                        m = np.logical_and(m, self.nhits==1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = np.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = np.logical_and(m, ds.p_max_posttrig_deriv < max_post_deriv)
                    m = np.logical_and(m, dataset.cuts.good())
                    masks.append(m)
                    
        elif pulse_peak_ranges is not None:
            if isinstance(pulse_peak_ranges[0], (int,float)) and len(pulse_peak_ranges)==2:
                pulse_peak_ranges = tuple(pulse_peak_ranges),
            for r in pulse_peak_ranges:
                middle = 0.5*(r[0]+r[1])
                abslim = 0.5*np.abs(r[1]-r[0])
                for gain,dataset in zip(gains,self.datasets):
                    m = np.abs(dataset.p_peak_value/gain-middle) <= abslim
                    if cut_crosstalk:
                        m = np.logical_and(m, self.nhits==1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = np.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds==dataset: continue 
                                m = np.logical_and(m, ds.p_max_posttrig_deriv < max_post_deriv)
                    m = np.logical_and(m, dataset.cuts.good())
                    masks.append(m)
        else:
            raise ValueError("Call make_masks with only one of pulse_avg_ranges and pulse_peak_ranges specified.")
        
        return masks
    
    def compute_average_pulse(self, masks, use_crosstalk_masks, subtract_mean=True):
        """
        Compute several average pulses in each TES channel, one per mask given in
        <masks>.  Store the averages in self.datasets.average_pulse with shape (m,n)
        where m is the number of masks and n equals self.nPulses (the # of records).
        
        Note that this method replaces any previously computed self.datasets.average_pulse
        
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
        if isinstance(masks, np.ndarray):
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
            if not isinstance(m, np.ndarray):
                raise ValueError("masks[%d] is not a np.ndarray"%i)
            
        pulse_counts = np.zeros((self.n_channels, nbins))
        pulse_sums = np.zeros((self.n_channels, nbins, self.nSamples), dtype=np.float)

        # Compute a master mask to say whether ANY mask wants a pulse from each segment
        # This can speed up work a lot when the pulses being averaged are from certain times only.
        segment_mask = np.zeros(self.n_segments, dtype=np.bool)
        for m in masks:
            n = len(m)
            nseg = 1+(n-1)/self.pulses_per_seg
            for i in range(nseg):
                if segment_mask[i]:
                    continue
                a,b = self.segnum2sample_range(i)
                if m[a:b].any():
                    segment_mask[i] = True
            a,b = self.segnum2sample_range(nseg+1)
            if a<n and m[a:].any():
                segment_mask[nseg+1] = True 
                
        printUpdater = inlineUpdater.InlineUpdater('compute_average_pulse')
        for first, end in self.iter_segments(segment_mask=segment_mask):
            printUpdater.update(end/float(self.nPulses))
            for imask,mask in enumerate(masks):
                valid = mask[first:end]
                for ichan,chan in enumerate(self.datasets):
                    if not (use_crosstalk_masks or (imask%self.n_channels) == ichan):
                        continue 
                    
                    if mask.shape != (chan.nPulses,):
                        raise ValueError("\nmasks[%d] has shape %s, but it needs to be (%d,)"%
                             (imask, mask.shape, chan.nPulses ))
                    if len(valid)>chan.data.shape[0]:
                        good_pulses = chan.data[valid[:chan.data.shape[0]], :]
                    else:
                        good_pulses = chan.data[valid, :]
                    pulse_counts[ichan,imask] += good_pulses.shape[0]
                    pulse_sums[ichan,imask,:] += good_pulses.sum(axis=0)


        # Rescale and store result to each MicrocalDataSet
        pulse_sums /= pulse_counts.reshape((self.n_channels, nbins,1))
        for ichan,ds in enumerate(self.datasets):
            average_pulses = pulse_sums[ichan,:,:]
            if subtract_mean:
                for imask in range(average_pulses.shape[0]):
                    average_pulses[imask,:] -= average_pulses[imask,:self.nPresamples-ds.pretrigger_ignore_samples].mean()
            ds.average_pulse = average_pulses[ichan,:]
    
    def plot_average_pulses(self, pulse_id, axis=None, use_legend=True):
        """Plot average pulse number <pulse_id> on matplotlib.Axes <axis>, or
        on a new Axes if <axis> is None.  If <pulse_id> is not a valid channel
        number, then plot all average pulses."""
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            
        axis.set_color_cycle(self.colors)
        dt = (np.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        
        for ds in self:
            plt.plot(dt,ds.average_pulse, label="chan %d"%ds.channum)

        axis.set_title("Average pulse for each channel when it is hit")

        plt.xlabel("Time past trigger (ms)")
        plt.ylabel("Raw counts")
        plt.xlim([dt[0], dt[-1]])
        if use_legend: plt.legend(loc='best')


    def plot_raw_spectra(self):
        """Plot distribution of raw pulse averages, with and without gain"""
        ds = self.first_good_dataset
        meangain = ds.p_pulse_average[ds.cuts.good()].mean()/ds.gain
        plt.clf()
        plt.subplot(211)
        for ds in self.datasets:
            gain = ds.gain
            _=plt.hist(ds.p_pulse_average[ds.cuts.good()], 200, [meangain*.8, meangain*1.2], alpha=0.5)
            
        plt.subplot(212)
        for ds in self.datasets:
            gain = ds.gain
            _=plt.hist(ds.p_pulse_average[ds.cuts.good()]/gain, 200, [meangain*.8,meangain*1.2], alpha=0.5)
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
        for ds in self:
            if ds.noise_autocorr is None or ds.noise_spectrum is None:
                print "Computing noise autocorrelation and spectrum"
                self.compute_noise_spectra()
                break
        
        printUpdater = inlineUpdater.InlineUpdater('compute_filters')
        for ds_num,ds in enumerate(self):
            if ds.cuts.good().sum() < 10:
                ds.filter = None
                self.set_chan_bad(ds.channum, 'cannot compute filter')
                continue
            printUpdater.update((ds_num+1)/float(self.n_channels))
            avg_signal = ds.average_pulse.copy()
            
            try:
                spectrum = ds.noise_spectrum.spectrum()
            except:
                spectrum = None
            f = mass.core.Filter(avg_signal, self.nPresamples-ds.pretrigger_ignore_samples, spectrum,
                                 ds.noise_autocorr, sample_time=self.timebase,
                                 fmax=fmax, f_3db=f_3db, shorten=2)
            ds.filter = f
            

    def plot_filters(self, first=0, end=-1):
        """Plot the filters from <first> through <end>-1.  By default, plots all filters,
        except that the maximum number is 8.  Left panels are the Fourier and time-domain
        X-ray energy filters.  Right panels are two different filters for estimating the 
        baseline level.
        """
        plt.clf()
        if end<=first: end=self.n_channels
        if first >= self.n_channels:
            raise ValueError("First channel must be less than %d"%self.n_channels)
        nplot = min(end-first, 8)
        for i,ds in enumerate(self.datasets[first:first+nplot]):
            ax1 = plt.subplot(nplot,2,1+2*i)
            ax2 = plt.subplot(nplot,2,2+2*i)
            ax1.set_title("chan %d signal"%(ds.channum))
            ax2.set_title("chan %d baseline"%(ds.channum))
            for ax in (ax1,ax2): ax.set_xlim([0,self.nSamples])
            ds.filter.plot(axes=(ax1,ax2))

        
    
    def summarize_filters(self, filter_name='noconst', std_energy=5898.8):
        rms_fwhm = np.sqrt(np.log(2)*8) # FWHM is this much times the RMS
        print 'V/dV for time, Fourier filters: '
        for i,ds in enumerate(self):
    
            try:
                rms = ds.filter.variances[filter_name]**0.5
                v_dv = (1/rms)/rms_fwhm
                print "Chan %3d filter %-15s Predicted V/dV %6.1f  Predicted res at %.1f eV: %6.1f eV" % (
                                ds.channum, filter_name, v_dv, std_energy, std_energy/v_dv)
            except Exception, e:
                print "Filter %d can't be used"%i
                print e
            
    def filter_data_tdm(self, filter_name='filt_noconst', transform=None, include_badchan=False, forceNew=False):
        printUpdater = inlineUpdater.InlineUpdater('filter_data_tdm')
        for chan in self.iter_channel_numbers(include_badchan):
            self.channel[chan].filter_data_tdm(filter_name, transform, forceNew)
            if include_badchan:
                printUpdater.update((chan/2+1)/float(len(self.channel.keys())))
            else:
                printUpdater.update((chan/2+1)/float(self.num_good_channels))

    def filter_data(self, filter_name=None, transform=None):
        """Filter data sets and store in datasets[*].p_filt_phase and _value.
        The filters are currently self.filter[*].filt_noconst"""
        if self.first_good_dataset.filter is None:
            self.compute_filters()
            
        if filter_name is None: filter_name='filt_noconst'
        
        for ds in self.datasets:
            ds.p_filt_phase = np.zeros_like(ds.p_filt_phase) # be sure not to change the data type of these arrays, they should follow the type from channel._setup_vectors
            ds.p_filt_value = np.zeros_like(ds.p_filt_value)
            
        printUpdater = inlineUpdater.InlineUpdater('BaseChannelGroup.filter_data')
        for first, end in self.iter_segments():
            if end>self.nPulses:
                end = self.nPulses 
            printUpdater.update(end/float(self.nPulses))
            for ds in self:
                if ds.filter is None:
                    continue
                filt_vector = ds.filter.__dict__[filter_name]
                peak_x, peak_y = ds.filter_data(filt_vector,first, end, transform=transform)
                ds.p_filt_phase[first:end] = peak_x
                ds.p_filt_value[first:end] = peak_y
        
        # Reset the phase-corrected and drift-corrected values
        for ds in self.datasets:
            ds.p_filt_value_phc = ds.p_filt_value.copy()
            ds.p_filt_value_dc  = ds.p_filt_value.copy()

            
    def plot_crosstalk(self, xlim=None, ylim=None, use_legend=True):
        print('plot_crosstalk no longer works, galen broke it when moving things inside datasets')
#        plt.clf()
#        dt = (np.arange(self.nSamples)-self.nPresamples)*1e3*self.timebase
#        
#        ndet = self.n_channels
#        plots_nx, plots_ny = ndet/2, 2
#        for i in range(ndet):
#            ax=plt.subplot(plots_nx, plots_ny, 1+i)
#            self.plot_average_pulses(i, axis=ax, use_legend=use_legend)
#            plt.plot(dt,self.datasets[i].average_pulses[i]/100,'k--', label="Main pulse/100")
#            if i+plots_ny<ndet:
#                ax.set_xlabel("")
#                ax.set_xticklabels([""])
#            if xlim is None:
#                xlim=[-.2,.2]
#            if ylim is None:
#                ylim=[-200,200]
#            plt.xlim(xlim)
#            plt.ylim(ylim)
#            if use_legend: 
#                plt.legend(loc='upper left')
#            plt.grid()
#            plt.title("Mean record when TES %d is hit"%i)
    
             
    def estimate_crosstalk(self, plot=True):
        """Work in progress..."""
        print('estimate_crosstalk definitley doesnt work even if it ever did, since galen broke it moving things inside datasets')
#        NMUX = self.n_channels
#
#        if plot: 
#            plt.clf()
#            ds0 = self.datasets[0]
#            dt = (np.arange(ds0.nSamples)-ds0.nPresamples)*ds0.timebase*1e3
#            ax0 = plt.subplot(NMUX,NMUX,1)
#        crosstalk = []
#        dot = np.dot
#        if self.datasets[0].noise_autocorr is None:
#            self.compute_noise_spectra()
#        
#        for i,ds in enumerate( self.datasets):
#            p = ds.average_pulses[i]
#            d = np.zeros_like(p)
#            d[1:] = p[1:]-p[:-1]
#            
#            if 'pulse_filt' in ds.__dict__ and 'deriv_filt' in ds.__dict__:
#                qp = ds.pulse_filt
#                qd = ds.deriv_filt
#            else:
#                print "Solving for TES %2d noise-inv-weighted pulse"%i
#                R = sp.linalg.toeplitz(ds.noise_autocorr[:len(p)])
#                Rp = np.linalg.solve(R, p)
#                print "Solving for TES %2d noise-inv-weighted derivative"%i
#                Rd = np.linalg.solve(R, d)
#                del R
#                
#                qp = dot(d,Rd)*Rp - dot(d,Rp)*Rd
#                qd = dot(p,Rp)*Rd - dot(p,Rd)*Rp
#                qp /= dot(p,qp)
#                qd /= dot(d,qd)
#            
#                # Save filters and the derivative
#                ds.pulse_filt = qp
#                ds.deriv_filt = qd
#                ds.pulse_deriv = d
#            
#            # Apply the filters to cross-talk and self-talk.  (!)
#            ct = []
#            for j,ds1 in enumerate(self.datasets):
#                sig = ds1.average_pulses[i]
#                print "%8.4f %8.4f"%(dot(sig,qp), dot(sig,qd))
#                ct.append(dot(sig,qp)) # row i, col j
#                if plot and (i != j):
#                    ax = plt.subplot(NMUX,NMUX,1+NMUX*j+i, sharex=ax0, sharey=ax0) # row j, col i
#                    sig2 = sig-dot(sig,qp)*ds.average_pulses[i]
#                    sig3 = sig2-dot(sig,qd)*ds.pulse_deriv
#                    plt.plot(dt,sig,color='green')
#                    plt.plot(dt,sig2,color='red')
#                    plt.plot(dt,sig3,color='blue')
#                    plt.xlim([-.5,3])
#                    plt.xticks((0,1,2,3))
#                    plt.ylim([-100,100])
#                    plt.title("TES %d when %d hit"%(j,i))
#                    rmss=["%5.2f"%(s.std()) for s  in sig,sig2,sig3]
#                    if sig[100+ds.nPresamples] < 0:
#                        plt.text(.08,.85,",".join(rmss), transform=ax.transAxes)
#                    else:
#                        plt.text(.08,.05,",".join(rmss), transform=ax.transAxes)
#            crosstalk.append(ct)
#        return np.array(crosstalk)
    
    
#    def drift_correct_ptmean(self, filt_ranges):
#        """A hack, but a great simple start on drift correction using the pretrigger mean.
#        DEPRECATED: see MicrocalDataSet.auto_drift_correct instead."""
#        
#        class LineDrawer(object):
#            def __init__(self, figure):
#                self.b, self.x, self.y = 0,0,0
#                self.fig = figure
#                self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
#                print "Connected cid=%d"%self.cid
#            def __call__(self, event):
#                self.b, self.x, self.y =  event.button, event.xdata, event.ydata
#            def __del__(self):
#                self.fig.canvas.mpl_disconnect(self.cid)
#                print 'Disconnected cid=%d'%self.cid
#        
#        for _i,(rng,ds) in enumerate(zip(filt_ranges,self.datasets)):
#            plt.clf()
#            fig = plt.gcf()
#            good = ds.cuts.good()
#            plt.plot(ds.p_pretrig_mean[good], ds.p_filt_value[good],'.')
#            plt.ylim(rng)
#            print "Click button 1 twice to draw a line; click button 2 to quit"
#            pf = LineDrawer(fig)
#            x,y=[],[]
#            line = None
#            offset,slope = 0.0,0.0
#            while True:
#                plt.waitforbuttonpress()
#                if pf.b > 1: break
#                x.append(pf.x)
#                y.append(pf.y)
#                if len(x)==2:
#                    if line is not None:
#                        line.remove()
#                    line, = plt.plot(x, y, 'r')
#                    offset = x[0]
#                    slope = (y[1]-y[0])/(x[1]-x[0])
#                    if ds.p_filt_value_phc[0]==0: ds.p_filt_value_phc=ds.p_filt_value
#                    ds.p_filt_value_dc = ds.p_filt_value_phc - (ds.p_pretrig_mean-offset)*slope
#                    ds.energy = ds.p_filt_value_dc[good]    
#                    print offset,slope, ' corrects the energy.  Hit button 2 to move on, or try again.'
#                    x,y = [],[]
#                
#            del pf


    def find_features_with_mouse(self, channame='p_filt_value', nclicks=1, prange=None, trange=None):
        """
        Plot histograms of each channel's "energy" spectrum, one channel at a time.
        After recording the x-coordinate of <nclicks> mouse clicks per plot, return an
        array of shape (N_channels, N_click) containing the "energy" of each click.
        
        <channame>  A string to choose the desired energy-like parameter.  Probably you want
                    to start with p_filt_value or p_filt_value_dc and later (once an energy
                    calibration is in place) p_energy.
        <nclicks>   The number of x coordinates to record per detector.  If you want to get
                    for example, a K-alpha and K-beta line in one go, then choose 2.
        <prange>    A 2-element sequence giving the limits to histogram.  If None, then the
                    histogram will show all data.
        <trange>    A 2-element sequence giving the time limits to use (in sec).  If None, then the
                    histogram will show all data.
                    
        Returns:
        A np.ndarray of shape (self.n_channels, nclicks).  
        """
        x = []
        for i,ds in enumerate(self.datasets):
            plt.clf()
            g = ds.cuts.good()
            if trange is not None:
                g = np.logical_and(g, ds.p_timestamp>trange[0])
                g = np.logical_and(g, ds.p_timestamp<trange[1])
            plt.hist(ds.__dict__[channame][g], 200, range=prange)
            plt.xlabel(channame)
            plt.title("Detector %d: attribute %s"%(i, channame))
            fig = plt.gcf()
            pf = mass.core.utilities.MouseClickReader(fig)
            for i in range(nclicks):
                while True:
                    plt.waitforbuttonpress()
                    try:
                        pfx = '%g'%pf.x
                    except TypeError:
                        continue
                    print 'Click on line #%d at %s'%(i+1, pfx)
                    x.append(pf.x)
                    break
            del pf
        xvalues = np.array(x)
        xvalues.shape=(self.n_channels, nclicks)
        return xvalues


    def find_named_features_with_mouse(self, name='Mn Ka1', channame='p_filt_value', prange=None, trange=None, energy=None):
        
        if energy is None:
            energy = mass.calibration.energy_calibration.STANDARD_FEATURES[name]
        
        print "Please click with the mouse on each channel's histogram at the %s line"%name
        xvalues = self.find_features_with_mouse(channame=channame, nclicks=1, prange=prange, trange=trange).ravel()
        for ds,xval in zip(self.datasets, xvalues):
            calibration = ds.calibration[channame]
            calibration.add_cal_point(xval, energy, name)


    def report(self):
        """
        Report on the number of data points and similar
        """
        for ds in self.datasets:
            ng = ds.cuts.nUncut()
            good = ds.cuts.good()
            dt = (ds.p_timestamp[good][-1]*1.0 - ds.p_timestamp[good][0])  # seconds
            np = np.arange(len(good))[good][-1] - good.argmax() + 1
            rate = (np-1.0)/dt
#            grate = (ng-1.0)/dt
            print 'chan %2d %6d pulses (%6.3f Hz over %6.4f hr) %6.3f%% good'%(ds.channum, np, rate, dt/3600., 100.0*ng/np)


    def plot_noise_autocorrelation(self, axis=None, channels=None, cmap=None):
        """Compare the noise autocorrelation functions.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        """
        
        if channels is None:
            channels = np.arange(self.n_channels)

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            
        if cmap is None:
            cmap = plt.cm.get_cmap("spectral")
            
        axis.grid(True)
        for i,ds in enumerate(self.datasets):
            if i not in channels: continue
            noise = ds.noise_records
            noise.plot_autocorrelation(axis=axis, label='TES %d'%i, color=cmap(float(i)/self.n_channels))
#        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        axis.set_xlabel("Time lag (ms)")
        plt.legend(loc='best')
        ltext = axis.get_legend().get_texts()
        plt.setp(ltext, fontsize='small')
        

    def save_pulse_energies_ascii(self, filename='all'):
        filename += '.energies'
        energy=[]
        for ds in self:
            energy=np.hstack((energy,ds.p_energy[ds.cuts.good()]))
        np.savetxt(filename, energy, fmt='%.10e')
#        f=open(filename+'.energies','w')
#        for line in energy:
#            f.write(line + '\n')
#        f.close()
    



class TESGroup(BaseChannelGroup):
    """
    A group of one or more *independent* microcalorimeters, in that
    they are time-division multiplexed.  It might be convenient to use
    this for multiple TDM channels, or for singles.  The key is that
    this object offers the same interface as the CDMGroup object
    (which has to be more complex under the hood).
    """
    def __init__(self, filenames, noise_filenames=None, noise_only=False, pulse_only=False,
                 noise_is_continuous=True, max_cachesize=None,
                 auto_pickle=True):
        if noise_filenames is not None and len(noise_filenames)==0:
            noise_filenames = None
        BaseChannelGroup.__init__(self, filenames, noise_filenames)
        self.noise_only = noise_only
        
        pulse_list = []
        noise_list = []
        dset_list = []
        for i,fname in enumerate(self.filenames):
            if noise_filenames is None:
                pulse, noise = create_pulse_and_noise_records(
                      fname, noise_only=noise_only, pulse_only=pulse_only, 
                      records_are_continuous = noise_is_continuous)
            else:
                nf = self.noise_filenames[i]
                pulse, noise = create_pulse_and_noise_records(fname, noisename=nf,
                      noise_only=noise_only, pulse_only=pulse_only, 
                      records_are_continuous = noise_is_continuous)
                
            pulse_list.append(pulse)
            if noise is None:
                dset = mass.channel.MicrocalDataSet(pulse.__dict__, auto_pickle=auto_pickle)
                dset_list.append(dset)
            elif pulse.channum == noise.channum:
                dset = mass.channel.MicrocalDataSet(pulse.__dict__, auto_pickle=auto_pickle)
                dset.noise_records = noise
                assert(dset.channum == dset.noise_records.channum)
                dset_list.append(dset)
                noise_list.append(noise)
            else:
                print('TESGroup did not add data because channums dont match %s, %s'%(fname, nf))
            
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
        for chan,ds in zip(self.channels, self.datasets):
            ds.pulse_records = chan
        self._setup_channels_list()
        if len(pulse_list)>0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg
        if len(self.datasets)>0:
            # Set master timestamp_offset (seconds)
            self.timestamp_offset = self.first_good_dataset.timestamp_offset
            
        for ds in self:
            if ds.timestamp_offset != self.timestamp_offset:
                self.timestamp_offset = None
                break

        if max_cachesize is not None:
            if max_cachesize < self.n_channels * self._pulse_records[0].segmentsize:
                self.set_segment_size(max_cachesize / self.n_channels)
    

    def copy(self):
        self.clear_cache()
        g = TESGroup([])
        g.__dict__.update(self.__dict__)
        g.datasets = tuple([d.copy() for d in self.datasets])
        return g
        
    
    def join(self, *others):
        # Ensure they are compatible
        print('join probably doesnt work since galen messed with it moving things inside datasets')
        for g in others:
            for attr in ('nPresamples','nSamples', 'noise_only', 'timebase'):
                if g.__dict__[attr] != self.__dict__[attr]:
                    raise RuntimeError("All objects must agree on group.%s"%attr)
            
        for g in others:
            n_extra = self.n_channels          
            self.datasets += g.datasets
            self.n_channels += g.n_channels
            self.n_segments = max(self.n_segments, g.n_segments)
        
        self.clear_cache()
        
    
    def set_segment_size(self, seg_size):
        self.clear_cache()
        self.n_segments = 0
        for ds in self:
            ds.pulse_records.set_segment_size(seg_size)
            self.n_segments = max(self.n_segments, ds.pulse_records.pulses_per_seg)
        self.pulses_per_seg = self.first_good_dataset.pulse_records.pulses_per_seg
        for ds in self:
            assert ds.pulse_records.pulses_per_seg == self.pulses_per_seg

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
        for ds in self.datasets:
            a,b = ds.pulse_records.read_segment(segnum)
            ds.data = ds.pulse_records.data
            try:
                ds.times = ds.pulse_records.datafile.datatimes_float
            except AttributeError:  
                ds.times = ds.pulse_records.datafile.datatimes/1e3

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
        
        
    def plot_noise(self, axis=None, channels=None, scale_factor=1.0, sqrt_psd=False, cmap=None):
        """Compare the noise power spectra.
        
        <channels>    Sequence of channels to display.  If None, then show all. 
        <scale_factor> Multiply counts by this number to get physical units. 
        <sqrt_psd>     Whether to show the sqrt(PSD) or (by default) the PSD itself.
        <cmap>         A matplotlib color map.  Defaults to something.
        """
        
        if channels is None:
            channels = np.arange(self.n_channels)

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            
        if scale_factor==1.0:
            units="Counts"
        else:
            units = "Scaled counts"
            
        axis.grid(True)
        if cmap is None:
            cmap = plt.cm.get_cmap("spectral")
        for ds_num, ds in enumerate(self):
            yvalue = ds.noise_records.spectrum.spectrum()*scale_factor**2
            axis.set_ylabel("Power Spectral Density (%s^2/Hz)"%units)
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
                axis.set_ylabel("PSD$^{1/2}$ (%s/Hz$^{1/2}$)"%units)
            axis.plot(ds.noise_records.spectrum.frequencies(), yvalue, label='TES chan %d'%ds.channum,
                      color=cmap(float(ds_num)/self.n_channels))
        f=self.first_good_dataset.noise_records.spectrum.frequencies()
        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        axis.set_xlabel("Frequency (Hz)")
        
        axis.loglog()
        plt.legend(loc='best')
        ltext = axis.get_legend().get_texts()
        plt.setp(ltext, fontsize='small')
    
    
    def compute_noise_spectra(self, max_excursion=9e9, n_lags=None):
        """<n_lags>, if not None, is the number of lags in each noise spectrum and the max lag 
        for the autocorrelation.  If None, the record length is used."""
        if n_lags is None:
            n_lags = self.nSamples
        for ds in self:
            ds.noise_records.compute_power_spectrum_reshape(max_excursion=max_excursion, seg_length=n_lags)
            ds.noise_spectrum = ds.noise_records.spectrum
            ds.noise_records.compute_autocorrelation(n_lags=n_lags, plot=False, max_excursion=max_excursion)
            ds.noise_autocorr = ds.noise_records.autocorrelation
            ds.noise_records.clear_cache()
        
    def pickle_datasets(self):
        for ds in self:
            ds.pickle()
            
    def pickle(self, filename=None, dirname=None):
        """Pickle the object by pickling its important contents
           <filename>    The output pickle name.  If not given, then it will be the data file name
                         with the suffix replaced by '.pkl' and in a subdirectory mass under the
                         main file's location."""
    
        if filename is None:
            ljhfilename = self.first_good_dataset.filename
            ljhbasename = ljhfilename.split("_chan")[0]
            basedir = os.path.dirname(ljhfilename)
            if dirname is None:
                dirname = basedir
            massdir = os.path.join(dirname,'mass')
            if not os.path.isdir(massdir):
                os.mkdir(massdir, 0775)
            filename = os.path.join(massdir, os.path.basename(ljhbasename+"_mass.pkl"))
        
        fp = open(filename, "wb")
        pickler = cPickle.Pickler(fp, protocol=2)
        pickler.dump(self.noise_only)
        pickler.dump(self._bad_channums)
        filenames = [ds.filename for ds in self.iter_channels(include_badchan=True)]
        pickler.dump(filenames)
        if self.noise_filenames is None:
            noise_filenames = None
        else:
            noise_filenames = [ds.noise_records.filename for ds in self.iter_channels(include_badchan=True)]
        pickler.dump(noise_filenames)
        fp.close()
        print "Stored %9d bytes %s"%(os.stat(filename).st_size, filename)
        for ds in self.datasets:
            ds.pickle()
        

def _sort_filenames_numerically(fnames, inclusion_list=None):
    """Take a sequence of filenames of the form '*_chanXXX.*'
    and sort it according to the numerical value of channel number XXX.
    If inclusion_list is not None, then it must be a container with the
    channel numbers to be included in the output.
    """
    if fnames is None or len(fnames)==0:
        return None
    chan2fname={}
    for name in fnames:
        channum = int(name.split('_chan')[1].split(".")[0])
        if inclusion_list is not None and channum not in inclusion_list:
            continue
        print channum, name
        chan2fname[channum] = name
    sorted_chan = chan2fname.keys()
    sorted_chan.sort()
    sorted_fnames = [chan2fname[key] for key in sorted_chan]
    return sorted_fnames


def _replace_path(fnames, newpath):
    """Take a sequence of filenames <fnames> and replace the directories leading to each
    with <newpath>"""
    if fnames is None or len(fnames)==0:
        return None
    result=[]
    for f in fnames:
        _,name = os.path.split(f)
        result.append(os.path.join(newpath,name))
    return result


def unpickle_TESGroup(filename, rawpath=None, inclusion_list=None):
    """
    Factory function to unpickle a TESGroup pickled by its .pickle() method.
    <filename>   The pickle file containing the group information.  (It's expected
                 that the per-channel pickle files will live in the standard place.)
    <rawpath>    If None, then assume the raw files live at the path indicated in the
                 pickle file.  Otherwise, they live in <rawpath>.
    <inclusion_list> A list of channel numbers to include in the result.  If None,
                 then all channels are included.
                 
    Returns a valid TESGroup object.  I hope.
    """
    if not filename[-8:] == 'mass.pkl':
        baseDir, fName = os.path.split(filename)
        massDir = os.path.join(baseDir, 'mass/')
        massFilename = fName.replace(fName[fName.rfind('chan'):],'mass.pkl')
        massFilename = os.path.join(massDir, massFilename)
        print('unpickle_TESGroup given %s, found %s'%(filename, massFilename))
        filename = massFilename

    fp = open(filename, "rb")
    unpickler = cPickle.Unpickler(fp)
    noise_only = unpickler.load()
    bad_channums = unpickler.load()
    filenames = _sort_filenames_numerically(unpickler.load(), inclusion_list)
    noise_filenames = _sort_filenames_numerically(unpickler.load(), inclusion_list)
    pulse_only = (not noise_only and (noise_filenames is None or len(noise_filenames)==0))
    if rawpath is not None:
        filenames = _replace_path(filenames, rawpath)
        noise_filenames = _replace_path(noise_filenames, rawpath)
        
    data = TESGroup(filenames, noise_filenames, pulse_only=pulse_only,
                    noise_only=noise_only)
    data.set_chan_bad(bad_channums, 'was bad when saved')
    printUpdater = inlineUpdater.InlineUpdater('unpickle_TESGroup')
    for ds in data.datasets:
        printUpdater.update((ds.index+1)/float(len(data.datasets)))
        ds.unpickle()
    
#     expected_attr = unpickler.load()
#     ds = MicrocalDataSet(expected_attr)
#     ds.cuts._mask = unpickler.load()
#     try:
#         while True:
#             k = unpickler.load()
#             v = unpickler.load()
#             ds.__dict__[k] = v
#     except EOFError:
#         pass
#     fp.close()
    return data
        

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





class CrosstalkVeto(object):
    """
    An object to allow vetoing of data in 1 channel when another is hit
    """
    
    def __init__(self, datagroup, window_ms=(-10,3), pileup_limit=100):
        if datagroup is None:
            return
        
        window_ms = np.array(window_ms, dtype=np.int)
        self.window_ms = window_ms
        self.n_channels = datagroup.n_channels
        self.n_pulses = datagroup.nPulses
#        self.veto = np.zeros((self.n_channels, self.n_pulses), dtype=np.bool8)
        
        ms0 = np.array([ds.p_timestamp[0] for ds in datagroup.datasets]).min()*1e3 + window_ms[0]
        ms9 = np.array([ds.p_timestamp.max() for ds in datagroup.datasets]).max()*1e3 + window_ms[1]
        self.nhits = np.zeros(ms9-ms0+1, dtype=np.int8)
        self.time0 = ms0
        
        
        for ds in datagroup.datasets:
            g = ds.cuts.good()
            vetotimes = ds.p_timestamp[g]*1e3-ms0
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
    
    
    def veto(self, times_sec):
        """Return boolean vector for whether a given moment is vetoed.  Times are given in
        seconds.  Resolution is 1 ms for the veto."""  
        index = np.asarray(times_sec*1e3-self.time0+0.5, dtype=np.int)
        return self.nhits[index]>1
