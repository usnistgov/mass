"""
channel_group.py

Part of the Microcalorimeter Analysis Software System (MASS).

This module defines classes that handle one or more TES data streams
together.  While these classes are indispensable for code-
division multiplexed (CDM) systems, they are also useful for the
simpler time-division multiplexed (TDM) systems in that they allow
the same interface to handle both types of data.

That's the goal, at least.

Notice that no one has used CDM data from 2012 to present (June 2014),
so I moved the CDMGroup class to mass.nonstandard.CDM module. Still, I
am preserving the separation of BaseChannelGroup (methods common to TDM
or CDM data) and TESGroup (for TDM data only).

Author: Joe Fowler, NIST

Started March 2, 2011
"""
from collections import Iterable
from functools import reduce
import glob
import os

import numpy as np
import matplotlib.pylab as plt
import h5py

import mass.core.analysis_algorithms
import mass.calibration.energy_calibration
import mass.nonstandard.CDM

from mass.calibration.energy_calibration import EnergyCalibration
from mass.core.channel import MicrocalDataSet, PulseRecords, NoiseRecords
from mass.core.cython_channel import CythonMicrocalDataSet
from mass.core.cut import CutFieldMixin
from mass.core.optimal_filtering import Filter
from mass.core.utilities import InlineUpdater, show_progress


class FilterCanvas(object):
    pass


def _generate_hdf5_filename(rawname):
    """Generate the appropriate HDF5 filename based on a file's LJH name.
    Takes /path/to/data_chan33.ljh --> /path/to/data_mass.hdf5"""
    import re
    fparts = re.split(r"_chan\d+", rawname)
    prefix_path = fparts[0]
    if rawname.endswith("noi"):
        prefix_path += '_noise'
    return prefix_path + "_mass.hdf5"


def RestoreTESGroup(hdf5filename, hdf5noisename=None):
    """Generate a TESGroup object from a data summary HDF5 filename 'hdf5filename'
    and optionally an 'hdf5noisename', though the latter can often be inferred from
    the noise raw filenames, which are stored in the pulse HDF5 file (assuming you
    aren't doing something weird).

    TODO: make this function accept a sequence of channel numbers and load only those
    channels into the TESGroup.
    """
    pulsefiles = []
    channum = []
    noisefiles = []
    generated_noise_hdf5_name = None

    h5file = h5py.File(hdf5filename, "r")
    for name, group in h5file.iteritems():
        if not name.startswith("chan"):
            continue
        pulsefiles.append(group.attrs['filename'])
        channum.append(group.attrs['channum'])

        if hdf5noisename is None:
            fname = group.attrs['noise_filename']
            if generated_noise_hdf5_name is None:
                generated_noise_hdf5_name = _generate_hdf5_filename(fname)
            elif generated_noise_hdf5_name != _generate_hdf5_filename(fname):
                raise RuntimeError("""The implied HDF5 noise files names are not the same for all channels.
                The first channel implies '%s'
                and another implies '%s'.
                Instead, you should run RestoreTESGroup with an explicit hdf5noisename argument.""" %
                                   (generated_noise_hdf5_name, _generate_hdf5_filename(fname)))
            noisefiles.append(fname)
    h5file.close()

    if hdf5noisename is not None:
        h5file = h5py.File(hdf5noisename, "r")
        for ch in channum:
            group = h5file['chan%d' % ch]
            noisefiles.append(group.attrs['filename'])
        h5file.close()
    else:
        hdf5noisename = generated_noise_hdf5_name

    return TESGroup(pulsefiles, noisefiles, hdf5_filename=hdf5filename,
                    hdf5_noisefilename=hdf5noisename)


class TESGroup(CutFieldMixin):
    """
    Provides the interface for a group of one or more microcalorimeters,
    multiplexed by TDM.
    """

    BRIGHT_ORANGE = '#ff7700'

    def __init__(self, filenames, noise_filenames=None, noise_only=False,
                 noise_is_continuous=True, max_cachesize=None,
                 hdf5_filename=None, hdf5_noisefilename=None,
                 never_use=None, use_only=None):

        if noise_filenames is not None and len(noise_filenames) == 0:
            noise_filenames = None

        # In the noise_only case, you can put the noise file names either in the
        # usual (pulse) filenames argument or in the noise_filenames argument.
        self.noise_only = noise_only
        if noise_only and noise_filenames is None:
            filenames, noise_filenames = (), filenames

        # Handle the case that either filename list is a glob pattern (e.g., "files_chan*.ljh")
        filenames = _glob_expand(filenames)
        noise_filenames = _glob_expand(noise_filenames)

        # If using a glob pattern especially, we have to be careful to eliminate files that are
        # missing a partner, either noise without pulse or pulse without noise.
        _remove_unmatched_channums(filenames, noise_filenames, never_use=never_use, use_only=use_only)

        # Figure out where the 2 HDF5 files are to live, if the default argument
        # was given for their paths.
        if hdf5_filename is None and not noise_only:
            hdf5_filename = _generate_hdf5_filename(filenames[0])
        if hdf5_noisefilename is None and noise_filenames is not None:
            hdf5_noisefilename = _generate_hdf5_filename(noise_filenames[0])

        # Handle the pulse files.
        if noise_only:
            self.filenames = ()
            self.hdf5_file = None
        else:
            # Convert a single filename to a tuple of size one
            if isinstance(filenames, str):
                filenames = (filenames,)
            self.filenames = tuple(filenames)
            self.n_channels = len(self.filenames)
            self.hdf5_file = h5py.File(hdf5_filename, 'a')

        # Cut parameter description need to initialized.
        self.cut_field_desc_init()

        # Same for noise filenames
        self.noise_filenames = None
        self.hdf5_noisefile = None
        if noise_filenames is not None:
            if isinstance(noise_filenames, str):
                noise_filenames = (noise_filenames,)
            self.noise_filenames = noise_filenames
            self.hdf5_noisefile = h5py.File(hdf5_noisefilename, 'a')
            if noise_only:
                self.n_channels = len(self.noise_filenames)

        # Set up other aspects of the object
        self.nhits = None
        self.n_segments = 0

        self.nPulses = 0
        self.nPresamples = 0
        self.nSamples = 0
        self.timebase = 0.0

        self._cached_segment = None
        self._cached_pnum_range = None
        self._allowed_pnum_ranges = None
        self._allowed_segnums = None
        self.pulses_per_seg = None
        self._bad_channums = dict()

        if self.n_channels <= 4:
            self.colors = ("blue", "#aaaa00", "green", "red")
        else:
            self.colors = ('purple', "blue", "cyan", "green", "gold", self.BRIGHT_ORANGE, "red", "brown")

        if self.noise_only:
            self._setup_per_channel_objects_noiseonly(noise_is_continuous)
        else:
            self._setup_per_channel_objects(noise_is_continuous)

        if max_cachesize is not None:
            if max_cachesize < self.n_channels * self.channels[0].segmentsize:
                self.set_segment_size(max_cachesize // self.n_channels)

        self.updater = InlineUpdater

    def _setup_per_channel_objects(self, noise_is_continuous=True):
        pulse_list = []
        noise_list = []
        dset_list = []

        for i, fname in enumerate(self.filenames):
            # Create the pulse records file interface and the overall MicrocalDataSet
            pulse = PulseRecords(fname)
            print("%s %i" % (fname, pulse.nPulses))
            if pulse.nPulses == 0:
                print("TESGroup is skipping a file that has zero pulses: %s" % fname)
                continue  # don't load files with zero pulses

            hdf5_group = self.hdf5_file.require_group("chan%d" % pulse.channum)
            hdf5_group.attrs['filename'] = fname

            dset = CythonMicrocalDataSet(pulse.__dict__, tes_group=self, hdf5_group=hdf5_group)

            if 'calibration' in hdf5_group:
                hdf5_cal_grp = hdf5_group['calibration']
                for cal_name in hdf5_cal_grp:
                    dset.calibration[cal_name] = EnergyCalibration.load_from_hdf5(hdf5_cal_grp, cal_name)

            if 'why_bad' in hdf5_group.attrs:
                self._bad_channums[dset.channum] = [comment.decode() for comment in hdf5_group.attrs['why_bad']]

            # If appropriate, add to the MicrocalDataSet the NoiseRecords file interface
            if self.noise_filenames is not None:
                nf = self.noise_filenames[i]
                hdf5_group.attrs['noise_filename'] = nf
                try:
                    hdf5_noisegroup = self.hdf5_noisefile.require_group("chan%d" % pulse.channum)
                    hdf5_noisegroup.attrs['filename'] = nf
                except:
                    hdf5_noisegroup = None
                noise = NoiseRecords(nf, records_are_continuous=noise_is_continuous,
                                     hdf5_group=hdf5_noisegroup)

                if pulse.channum != noise.channum:
                    print("TESGroup did not add data: channums don't match %s, %s" % (fname, nf))
                    continue
                dset.noise_records = noise
                assert(dset.channum == dset.noise_records.channum)
                noise_list.append(noise)

            pulse_list.append(pulse)
            dset_list.append(dset)

            if self.n_segments == 0:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    self.__dict__[attr] = pulse.__dict__[attr]
            else:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    if self.__dict__[attr] != pulse.__dict__[attr]:
                        raise ValueError("Unequal values of %s: %f != %f" % (attr, float(self.__dict__[attr]),
                                                                             float(pulse.__dict__[attr])))
            self.n_segments = max(self.n_segments, pulse.n_segments)
            self.nPulses = max(self.nPulses, pulse.nPulses)

        # Store relevant facts as attributes to the HDF5 file
        if self.hdf5_file is not None:
            self.hdf5_file.attrs.update({'npulses': self.nPulses,
                                         'nsamples': self.nSamples,
                                         'npresamples': self.nPresamples,
                                         'frametime': self.timebase})

        self.channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)

        for index, (pr, ds) in enumerate(zip(self.channels, self.datasets)):
            ds.pulse_records = pr
            ds.index = index

        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg

    def _setup_per_channel_objects_noiseonly(self, noise_is_continuous=True):
        noise_list = []
        dset_list = []
        for fname in self.noise_filenames:

            noise = NoiseRecords(fname, records_are_continuous=noise_is_continuous)
            try:
                hdf5_group = self.hdf5_noisefile.require_group("chan%d" % noise.channum)
                hdf5_group.attrs['filename'] = fname
                noise.hdf5_group = hdf5_group
            except:
                hdf5_group = None

            dset = MicrocalDataSet(noise.__dict__, hdf5_group=hdf5_group)
            dset.noise_records = noise
            noise_list.append(noise)
            dset_list.append(dset)

            if self.n_segments == 0:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    self.__dict__[attr] = noise.__dict__[attr]
            else:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    if self.__dict__[attr] != noise.__dict__[attr]:
                        raise ValueError(
                            "Unequal values of %s: %f != %f" % (attr, float(self.__dict__[attr]),
                                                                float(noise.__dict__[attr])))
            self.n_segments = max(self.n_segments, noise.n_segments)
            self.nPulses = max(self.nPulses, noise.nPulses)

        # Store relevant facts as attributes to the HDF5 file
        if self.hdf5_file is not None:
            self.hdf5_file.attrs.update({'npulses': self.nPulses,
                                         'nsamples': self.nSamples,
                                         'npresamples': self.nPresamples,
                                         'frametime': self.timebase})

        self.channels = ()
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)

        for index, (pr, ds) in enumerate(zip(self.channels, self.datasets)):
            ds.pulse_records = pr
            ds.index = index

    def __iter__(self):
        """Iterator over the self.datasets in channel number order"""
        for ds in self.iter_channels():
            yield ds

    def iter_channels(self, include_badchan=False):
        """Iterator over the self.datasets in channel number order
        include_badchan : whether to include officially bad channels in the result.
        """
        for ds in self.datasets:
            if not include_badchan:
                if ds.channum in self._bad_channums:
                    continue
            yield ds

    def iter_channel_numbers(self, include_badchan=False):
        """Iterator over the channel numbers in numerical order
        include_badchan : whether to include officially bad channels in the result.
        """
        for ds in self.iter_channels(include_badchan=include_badchan):
            yield ds.channum

    def set_chan_good(self, *args):
        """Set one or more channels to be good.  (No effect for channels already listed
        as good.)
        *args  Arguments to this function are integers or containers of integers.  Each
               integer is removed from the bad-channels list."""
        added_to_list = set.union(*[set(x) if isinstance(x, Iterable) else {x} for x in args])

        for channum in added_to_list:
            if channum in self._bad_channums:
                comment = self._bad_channums.pop(channum)
                del self.hdf5_file["chan{0:d}".format(channum)].attrs['why_bad']
                print("chan %d set good, had previously been set bad for %s" % (channum, str(comment)))
            else:
                print("chan %d not set good because it was not set bad" % channum)

    def set_chan_bad(self, *args):
        """Set one or more channels to be bad.  (No effect for channels already listed
        as bad.)

        Args:
            *args  Arguments to this function are integers or containers of integers.  Each
                integer is added to the bad-channels list."""
        added_to_list = set.union(*[set(x) if isinstance(x, Iterable) else {x} for x in args if not isinstance(x, str)])
        comment = reduce(lambda x, y: y, [x for x in args if isinstance(x, str)], '')

        for channum in added_to_list:
            new_comment = self._bad_channums.get(channum, []) + [comment]
            self._bad_channums[channum] = new_comment
            print('chan %s flagged bad because %s' % (channum, comment))
            self.hdf5_file["chan{0:d}".format(channum)].attrs['why_bad'] = np.asarray(new_comment, dtype=np.bytes_)

    @property
    def timestamp_offset(self):
        ts = set([ds.timestamp_offset for ds in self if ds.channum not in self._bad_channums])
        if len(ts) == 1:
            return ts.pop()
        else:
            return None

    @property
    def channel(self):
        return {ds.channum: ds for ds in self.datasets}

    @property
    def good_channels(self):
        return [ds.channum for ds in self if ds.channum not in self._bad_channums]

    @property
    def num_good_channels(self):
        return len(self.good_channels)

    @property
    def first_good_dataset(self):
        if self.num_good_channels > 0:
            return self.channel[self.good_channels[0]]
        else:
            raise IndexError("WARNING: All datasets flagged bad, most things won't work.")

    @property
    def why_chan_bad(self):
        return self._bad_channums.copy()

    def clear_cache(self):
        """Invalidate any cached raw data."""
        self._cached_segment = None
        self._cached_pnum_range = None
        for ds in self.datasets:
            ds.data = None
        if 'raw_channels' in self.__dict__:
            for rc in self.raw_channels:
                rc.data = None
        if 'noise_channels' in self.__dict__:
            for nc in self.noise_channels:
                nc.datafile.clear_cache()

    def sample2segnum(self, samplenum):
        """Returns the segment number of sample number <samplenum>."""
        if samplenum >= self.nPulses:
            samplenum = self.nPulses - 1
        return samplenum // self.pulses_per_seg

    def segnum2sample_range(self, segnum):
        """Return the (first,end) sample numbers of the segment numbered <segnum>.
        Note that <end> is 1 beyond the last sample number in that segment."""
        return segnum * self.pulses_per_seg, (segnum + 1) * self.pulses_per_seg

    def set_data_use_ranges(self, ranges=None):
        """Set the range of sample numbers that this object will use when iterating over
        raw data.

        <ranges> can be None (which causes all samples to be used, the default);
                or a 2-element sequence (a,b), which causes only a through b-1 inclusive to be used;
                or a sequence of 2-element sequences, which is like the previous
                but with multiple sample ranges allowed.
        """
        if ranges is None:
            allowed_ranges = [[0, self.nPulses]]
        elif len(ranges) == 2 and np.isscalar(ranges[0]) and np.isscalar(ranges[1]):
            allowed_ranges = [[ranges[0], ranges[1]]]
        else:
            allowed_ranges = [r for r in ranges]

        allowed_segnums = np.zeros(self.n_segments, dtype=np.bool)
        for first, end in allowed_ranges:
            assert first <= end
            for sn in range(self.sample2segnum(first), self.sample2segnum(end - 1) + 1):
                allowed_segnums[sn] = True

        self._allowed_pnum_ranges = allowed_ranges
        self._allowed_segnums = allowed_segnums

        if ranges is not None:
            print('Warning!  This feature is only half-complete.  Currently, granularity is limited.')
            print('   Only full "segments" of size %d records can be ignored.' % self.pulses_per_seg)
            print('   Will use %d segments and ignore %d.' % (self._allowed_segnums.sum(),
                                                              self.n_segments - self._allowed_segnums.sum()))

    def iter_segments(self, first_seg=0, end_seg=-1, sample_mask=None, segment_mask=None):
        if self._allowed_segnums is None:
            self.set_data_use_ranges(None)

        if end_seg < 0:
            end_seg = self.n_segments
        for i in range(first_seg, end_seg):
            if not self._allowed_segnums[i]:
                continue
            a, b = self.segnum2sample_range(i)
            if sample_mask is not None:
                if b > len(sample_mask):
                    b = len(sample_mask)
                if not sample_mask[a:b].any():
                    print('We can skip segment %4d' % i)
                    continue  # Don't need anything in this segment.  Sweet!
            if segment_mask is not None:
                if not segment_mask[i]:
                    print('We can skip segment %4d' % i)
                    continue  # Don't need anything in this segment.  Sweet!
            first_rnum, end_rnum = self.read_segment(i)
            yield first_rnum, end_rnum

    @show_progress("summarize_data")
    def summarize_data(self, peak_time_microsec=220.0, pretrigger_ignore_microsec=20.0,
                       include_badchan=False, forceNew=False, use_cython=True):
        """Compute summary quantities for each pulse.
        We are (July 2014) developing a Julia replacement for this, but you can use Python
        if you wish.
        """
        nchan = float(len(self.channel.keys())) if include_badchan else float(self.num_good_channels)

        for i, ds in enumerate(self.iter_channels(include_badchan)):
            try:
                ds.summarize_data(peak_time_microsec, pretrigger_ignore_microsec, forceNew, use_cython=use_cython)
                yield (i + 1) / nchan
                self.hdf5_file.flush()
            except:
                self.set_chan_bad(ds.channum, "summarize_data")

    def calc_external_trigger_timing(self, after_last=False, until_next=False, from_nearest=False, forceNew=False):
        if not (after_last or until_next or from_nearest):
            raise ValueError("at least one of from_last, until_next, or from_nearest should be True")
        ds = self.first_good_dataset

        # loading this dataset can be slow, so lets do it only once for the whole ChannelGroup
        external_trigger_rowcount = np.asarray(ds.external_trigger_rowcount[:], dtype=np.int64)

        for ds in self:
            try:
                if forceNew or\
                        ("rows_after_last_external_trigger" not in ds.hdf5_group and after_last) or\
                        ("rows_until_next_external_trigger" not in ds.hdf5_group and until_next) or\
                        ("rows_from_nearest_external_trigger" not in ds.hdf5_group and from_nearest):
                    rows_after_last_external_trigger, rows_until_next_external_trigger = \
                        mass.core.analysis_algorithms.nearest_arrivals(ds.p_rowcount[:], external_trigger_rowcount)
                    if after_last:
                        g = ds.hdf5_group.require_dataset("rows_after_last_external_trigger",
                                                          (ds.nPulses,), dtype=np.int64)
                        g[:] = rows_after_last_external_trigger
                    if until_next:
                        g = ds.hdf5_group.require_dataset("rows_until_next_external_trigger",
                                                          (ds.nPulses,), dtype=np.int64)
                        g[:] = rows_until_next_external_trigger
                    if from_nearest:
                        g = ds.hdf5_group.require_dataset("rows_from_nearest_external_trigger",
                                                          (ds.nPulses,), dtype=np.int64)
                        g[:] = np.fmin(rows_after_last_external_trigger, rows_until_next_external_trigger)
            except Exception:
                self.set_chan_bad(ds.channum, "calc_external_trigger_timing")

    def read_trace(self, record_num, dataset_num=0, channum=None):
        """Read (from cache or disk) and return the pulse numbered <record_num> for
        dataset number <dataset_num> or channel number <channum>.
        If both are given, then <channum> will be used when valid.
        If this is a CDMGroup, then the pulse is the demodulated
        channel by that number."""
        ds = self.channel.get(channum, self.datasets[dataset_num])
        return ds.read_trace(record_num)

    def plot_traces(self, pulsenums, dataset_num=0, channum=None, pulse_summary=True, axis=None,
                    difference=False, residual=False, valid_status=None, shift1=False):
        """Plot some example pulses, given by sample number.
        <pulsenums>   A sequence of sample numbers, or a single one.
        <dataset_num> Dataset index (0 to n_dets-1, inclusive).  Will be used only if
                      <channum> is invalid.
        <channum>    Dataset channel number.  If valid, it will be used instead of dataset_num.

        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A plt axis to plot on.
        <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
        <residual>   Whether to show the residual between data and opt filtered model,
                     or just raw data.
        <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                     that have been cut.  If "cut", show only those that have been cut.
        <shift1>     Whether to take pulses with p_shift1==True and delay them by 1 sample
        """

        if channum in self.channel:
            dataset = self.channel[channum]
            dataset_num = dataset.index
        else:
            dataset = self.datasets[dataset_num]
            if channum is not None:
                print("Cannot find channum[%d], so using dataset #%d" % (channum, dataset_num))
        return dataset.plot_traces(pulsenums, pulse_summary, axis, difference,
                                   residual, valid_status, shift1)

    def plot_summaries(self, quantity, valid='uncut', downsample=None, log=False, hist_limits=None,
                       channel_numbers=None, dataset_numbers=None):
        """Plot a summary of one quantity from the data set, including time series and histograms of
        this quantity.  This method plots all channels in the group, but only one quantity.  If you
        would rather see all quantities for one channel, then use the group's
        group.dataset[i].plot_summaries() method.

        <quantity> A case-insensitive whitespace-ignored one of the following list, or the numbers
                   that go with it:
                   "Pulse RMS" (0)
                   "Pulse Avg" (1)
                   "Peak Value" (2)
                   "Pretrig RMS" (3)
                   "Pretrig Mean" (4)
                   "Max PT Deriv" (5)
                   "Rise Time" (6)
                   "Peak Time" (7)

        <valid> The words 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted
                *OR* None, meaning that all pulses should be plotted.

        <downsample> To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points.

        <log>              Use logarithmic y-axis on the histograms (right panels).
        <hist_limits>
        <channel_numbers>  A sequence of channel numbers to plot. If None, then plot all.
        <dataset_numbers>  A sequence of the datasets [0...n_channels-1] to plot.  If None
                           (the default) then plot all datasets in numerical order.
        """

        plottables = (
            ("p_pulse_rms", 'Pulse RMS', 'magenta', None),
            ("p_pulse_average", 'Pulse Avg', 'purple', [0,5000]),
            ("p_peak_value", 'Peak value', 'blue', None),
            ("p_pretrig_rms", 'Pretrig RMS', 'green', [0, 4000]),
            ("p_pretrig_mean", 'Pretrig Mean', '#00ff26', None),
            ("p_postpeak_deriv", 'Max PostPk deriv', 'gold', [0, 700]),
            ("p_rise_time[:]*1e3", 'Rise time (ms)', 'orange', [0, 12]),
            ("p_peak_time[:]*1e3", 'Peak time (ms)', 'red', [-3, 9])
        )

        quant_names = [p[1].lower().replace(" ", "") for p in plottables]
        if quantity in range(len(quant_names)):
            plottable = plottables[quantity]
        else:
            quantity = quant_names.index(quantity.lower().replace(" ", ""))
            plottable = plottables[quantity]

        MAX_TO_PLOT = 16
        if channel_numbers is None:
            if dataset_numbers is None:
                datasets = [ds for ds in self]
                if len(datasets) > MAX_TO_PLOT:
                    datasets = datasets[:MAX_TO_PLOT]
            else:
                datasets = [self.datasets[i] for i in dataset_numbers]
            channel_numbers = [ds.channum for ds in datasets]
        else:
            datasets = [self.channel[i] for i in channel_numbers]

        # Plot timeseries with 0 = the last 00 UT during or before the run.
        last_record = np.max([ds.p_timestamp[-1] for ds in self])
        last_midnight = last_record - (last_record%86400)
        hour_offset = last_midnight/3600.

        plt.clf()
        ny_plots = len(datasets)
        for i, (channum, ds) in enumerate(zip(channel_numbers, datasets)):

            # Convert "uncut" or "cut" to array of all good or all bad data
            if isinstance(valid, str):
                if "uncut" in valid.lower():
                    valid_mask = ds.cuts.good()
                    print("Plotting only uncut data"),
                elif "cut" in valid.lower():
                    valid_mask = ds.cuts.bad()
                    print("Plotting only cut data"),
                elif 'all' in valid.lower():
                    valid_mask = None
                    print("Plotting all data, cut or uncut"),
                else:
                    raise ValueError("If valid is a string, it must contain 'all', 'uncut' or 'cut'.")

            if valid_mask is not None:
                nrecs = valid_mask.sum()
                if downsample is None:
                    downsample = nrecs // 10000
                    if downsample < 1:
                        downsample = 1
                hour = ds.p_timestamp[valid_mask][::downsample] / 3600.0
            else:
                nrecs = ds.nPulses
                if downsample is None:
                    downsample = ds.nPulses // 10000
                    if downsample < 1:
                        downsample = 1
                hour = ds.p_timestamp[::downsample] / 3600.0
            print("Chan %3d (%d records; %d in scatter plots)" % (channum, nrecs, hour.shape[0]))

            (vect, label, color, default_limits) = plottable
            if hist_limits is None:
                limits = default_limits
            else:
                limits = hist_limits

            # Vectors are being sampled and multiplied, so eval() is needed.
            vect = eval("ds.%s"%vect)[valid_mask]

            # Scatter plots on left half of figure
            if i == 0:
                ax_master = plt.subplot(ny_plots, 2, 1 + i * 2)
            else:
                plt.subplot(ny_plots, 2, 1 + i * 2, sharex=ax_master)

            if len(vect) > 0:
                plt.plot(hour-hour_offset, vect[::downsample], '.', ms=1, color=color)
            else:
                plt.text(.5, .5, 'empty', ha='center', va='center', size='large',
                         transform=plt.gca().transAxes)
            if i == 0:
                plt.title(label)
            plt.ylabel("Ch %d" % channum)
            if i == ny_plots - 1:
                plt.xlabel("Time since last UT midnight (hours)")

            # Histograms on right half of figure
            if i == 0:
                axh_master = plt.subplot(ny_plots, 2, 2 + i * 2)
            else:
                if 'Pretrig Mean' == label:
                    plt.subplot(ny_plots, 2, 2 + i * 2)
                else:
                    plt.subplot(ny_plots, 2, 2 + i * 2, sharex=axh_master)

            if limits is None:
                in_limit = np.ones(len(vect), dtype=np.bool)
            else:
                in_limit = np.logical_and(vect > limits[0], vect < limits[1])
            if in_limit.sum() <= 0:
                plt.text(.5, .5, 'empty', ha='center', va='center', size='large',
                         transform=plt.gca().transAxes)
            else:
                contents, _bins, _patches = plt.hist(vect[in_limit], 200, log=log,
                                                     histtype='stepfilled', fc=color, alpha=0.5)
            if i == ny_plots - 1:
                plt.xlabel(label)
            if log:
                plt.ylim(ymin=contents.min())
            plt.tight_layout()

    def make_masks(self, pulse_avg_range=None,
                   pulse_peak_range=None,
                   pulse_rms_range=None,
                   use_gains=True, gains=None):
        """Generate a sequence of masks for use in compute_average_pulses().

        Arguments:
        pulse_avg_range -- A 2-sequence giving the (minimum,maximum) p_pulse_average
        pulse_peak_range -- A 2-sequence giving the (minimum,maximum) p_peak_value
        pulse_rms_range --  A 2-sequence giving the (minimum,maximum) p_pulse_rms
        use_gains -- Whether to rescale the pulses by a set of "gains", either from
                     `gains` or from the ds.gain parameter if `gains` is None.
        gains -- The set of gains to use, overriding the self.datasets[*].gain, if
                 `use_gains` is True.  (If False, this argument is ignored.)
        """

        for ds in self:
            if ds.nPulses == 0:
                self.set_chan_bad(ds.channum, "has 0 pulses")

        masks = []
        if use_gains:
            if gains is None:
                gains = [d.gain for d in self.datasets]
        else:
            gains = np.ones(self.n_channels)

        nranges = 0
        if pulse_avg_range is not None:
            nranges += 1
            vectname = "p_pulse_average"
            pmin, pmax = pulse_avg_range
        if pulse_peak_range is not None:
            nranges += 1
            vectname = "p_peak_value"
            pmin, pmax = pulse_peak_range
        if pulse_rms_range is not None:
            nranges += 1
            vectname = "p_pulse_rms"
            pmin, pmax = pulse_rms_range

        if nranges == 0:
            raise ValueError("Call make_masks with one of pulse_avg_range"
                             " pulse_rms_range, or pulse_peak_range specified.")
        elif nranges > 1:
            print("Warning: make_masks uses only one range argument.  Checking only '%s'." % vectname)

        middle = 0.5 * (pmin + pmax)
        abs_lim = 0.5 * np.abs(pmax - pmin)
        for gain, dataset in zip(gains, self.datasets):
            v = dataset.__dict__[vectname][:]
            m = np.abs(v / gain - middle) <= abs_lim
            m = np.logical_and(m, dataset.cuts.good())
            masks.append(m)
        return masks

    def compute_average_pulse(self, masks, subtract_mean=True, forceNew=False):
        """Compute an average pulse in each TES channel.

        Store the averages in self.datasets.average_pulse, a length nSamp vector.
        Note that this method replaces any previously computed self.datasets.average_pulse

        `masks` -- a sequence of length self.n_channels, one sequence per channel.
        The elements of `masks` should be booleans or interpretable as booleans.

        `subtract_mean` -- whether each average pulse will subtract a constant
        to ensure that the pretrigger mean (first self.nPresamples elements) is zero.
        """
        if len(masks) != len(self.datasets):
            raise ValueError("masks must include exactly one mask per data channel")

        # Make sure that masks is a sequence of 1D arrays of the right shape
        for i, m in enumerate(masks):
            if not isinstance(m, np.ndarray):
                raise ValueError("masks[%d] is not a np.ndarray" % i)

        for (mask, ds) in zip(masks, self.datasets):
            if ds.channum not in self.good_channels:
                continue
            ds.compute_average_pulse(mask, subtract_mean=subtract_mean, forceNew=forceNew)

    def plot_average_pulses(self, channum=None, axis=None, use_legend=True):
        """Plot average pulse for cahannel number <channum> on matplotlib.Axes <axis>, or
        on a new Axes if <axis> is None.  If <channum> is not a valid channel
        number, then plot all average pulses."""
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        axis.set_color_cycle(self.colors)
        dt = (np.arange(self.nSamples) - self.nPresamples) * self.timebase * 1e3

        if channum in self.channel:
            plt.plot(dt, self.channel[channum].average_pulse, label='Chan %d' % channum)
        else:
            for ds in self:
                plt.plot(dt, ds.average_pulse, label="Chan %d" % ds.channum)

        axis.set_title("Average pulse for each channel when it is hit")

        plt.xlabel("Time past trigger (ms)")
        plt.ylabel("Raw counts")
        plt.xlim([dt[0], dt[-1]])
        if use_legend:
            plt.legend(loc='best')

    def plot_raw_spectra(self):
        """Plot distribution of raw pulse averages, with and without gain"""
        ds = self.first_good_dataset
        meangain = ds.p_pulse_average[ds.cuts.good()].mean() / ds.gain
        plt.clf()
        plt.subplot(211)
        for ds in self.datasets:
            gain = ds.gain
            _ = plt.hist(ds.p_pulse_average[ds.cuts.good()], 200,
                         [meangain * .8, meangain * 1.2], alpha=0.5)

        plt.subplot(212)
        for ds in self.datasets:
            gain = ds.gain
            _ = plt.hist(ds.p_pulse_average[ds.cuts.good()] / gain, 200,
                         [meangain * .8, meangain * 1.2], alpha=0.5)
            print(ds.p_pulse_average[ds.cuts.good()].mean())
        return meangain

    def set_gains(self, gains):
        """Set the datasets to have the given gains.  These gains will be used when
        averaging pulses in self.compute_average_pulse() and in ...***?"""
        if len(gains) != self.n_channels:
            raise ValueError("gains must have the same length as the number of datasets (%d)"
                             % self.n_channels)

        for g, d in zip(gains, self.datasets):
            d.gain = g

    @show_progress("compute_filters")
    def compute_filters(self, fmax=None, f_3db=None, forceNew=False):

        # Analyze the noise, if not already done
        needs_noise = any([ds.noise_autocorr[0] == 0.0 or
                           ds.noise_psd[1] == 0 for ds in self])
        if needs_noise:
            print("Computing noise autocorrelation and spectrum")
            self.compute_noise_spectra()

        for ds_num, ds in enumerate(self):
            if "filters" not in ds.hdf5_group or forceNew:
                if ds.cuts.good().sum() < 10:
                    ds.filter = None
                    self.set_chan_bad(ds.channum, 'cannot compute filter, too few good pulses')
                    continue
                if ds._use_new_filters:
                    f = ds.compute_newfilter(fmax=fmax, f_3db=f_3db)
                else:
                    f = ds.compute_oldfilter(fmax=fmax, f_3db=f_3db)
                ds.filter = f
                yield (ds_num + 1) / float(self.n_channels)

                # Store all filters created to a new HDF5 group
                h5grp = ds.hdf5_group.require_group('filters')
                if f.f_3db is not None:
                    h5grp.attrs['f_3db'] = f.f_3db
                if f.fmax is not None:
                    h5grp.attrs['fmax'] = f.fmax
                h5grp.attrs['peak'] = f.peak_signal
                h5grp.attrs['shorten'] = f.shorten
                h5grp.attrs['newfilter'] = ds._use_new_filters
                for k in ["filt_fourier", "filt_fourier_full", "filt_noconst",
                          "filt_baseline", "filt_baseline_pretrig", 'filt_aterms']:
                    if k in h5grp:
                        del h5grp[k]
                    if getattr(f, k, None) is not None:
                        vec = h5grp.create_dataset(k, data=getattr(f, k))
                        vec.attrs['variance'] = f.variances.get(k.split('filt_')[1], 0.0)
                        vec.attrs['predicted_v_over_dv'] = f.predicted_v_over_dv.get(k.split('filt_')[1], 0.0)
            else:
                print("chan %d skipping compute_filter because already done, and loading filter" % ds.channum)
                h5grp = ds.hdf5_group['filters']
                ds.filter = Filter(ds.average_pulse[...], self.nPresamples - ds.pretrigger_ignore_samples,
                                   ds.noise_psd[...], ds.noise_autocorr[...], sample_time=self.timebase,
                                   fmax=fmax, f_3db=f_3db, shorten=2)
                ds.filter.peak_signal = h5grp.attrs['peak']
                ds.filter.shorten = h5grp.attrs['shorten']
                ds.filter.f_3db = h5grp.attrs['f_3db'] if 'f_3db' in h5grp.attrs else None
                ds.filter.fmax = h5grp.attrs['fmax'] if 'fmax' in h5grp.attrs else None
                ds.filter.variances = {}
                for name in h5grp:
                    if name.startswith("filt_"):
                        setattr(ds.filter, name, h5grp[name][:])
                        if 'variance' in h5grp[name].attrs:
                            ds.filter.variances[name] = h5grp[name].attrs['variance']
                        if 'predicted_v_over_dv' in h5grp[name].attrs:
                            ds.filter.predicted_v_over_dv[name] = \
                                h5grp[name].attrs['predicted_v_over_dv']

    def plot_filters(self, first=0, end=-1):
        """Plot the filters from <first> through <end>-1.  By default, plots all filters,
        except that the maximum number is 8.  Left panels are the Fourier and time-domain
        X-ray energy filters.  Right panels are two different filters for estimating the
        baseline level.
        """
        plt.clf()
        if end <= first:
            end = self.n_channels
        if first >= self.n_channels:
            raise ValueError("First channel must be less than %d" % self.n_channels)
        nplot = min(end - first, 8)
        for i, ds in enumerate(self.datasets[first:first + nplot]):
            ax1 = plt.subplot(nplot, 2, 1 + 2 * i)
            ax2 = plt.subplot(nplot, 2, 2 + 2 * i)
            ax1.set_title("chan %d signal" % ds.channum)
            ax2.set_title("chan %d baseline" % ds.channum)
            for ax in (ax1, ax2):
                ax.set_xlim([0, self.nSamples])
                if hasattr(ds, 'filter'):
                    ds.filter.plot(axes=(ax1, ax2))
        plt.show()

    def summarize_filters(self, filter_name='noconst', std_energy=5898.8):
        rms_fwhm = np.sqrt(np.log(2) * 8)  # FWHM is this much times the RMS
        print('V/dV for time, Fourier filters: ')
        for i, ds in enumerate(self):
            try:
                if ds.filter is not None:
                    rms = ds.filter.variances[filter_name]**0.5
                else:
                    rms = ds.hdf5_group['filters/filt_%s' % filter_name].attrs['variance']**0.5
                v_dv = (1 / rms) / rms_fwhm
                print("Chan %3d filter %-15s Predicted V/dV %6.1f  Predicted res at %.1f eV: %6.1f eV" %
                      (ds.channum, filter_name, v_dv, std_energy, std_energy / v_dv))
            except Exception as e:
                print("Filter %d can't be used" % i)
                print(e)

    @show_progress("filter_data")
    def filter_data(self, filter_name='filt_noconst', transform=None, include_badchan=False, forceNew=False, use_cython=True):
        nchan = float(len(self.datasets)) if include_badchan else float(self.num_good_channels)

        for i, ds in enumerate(self.iter_channels(include_badchan)):
            ds.filter_data(filter_name, transform, forceNew, use_cython=use_cython)
            yield (i+1) / nchan

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
        for i, ds in enumerate(self.datasets):
            plt.clf()
            g = ds.cuts.good()
            if trange is not None:
                g = np.logical_and(g, ds.p_timestamp > trange[0])
                g = np.logical_and(g, ds.p_timestamp < trange[1])
            plt.hist(ds.__dict__[channame][g], 200, range=prange)
            plt.xlabel(channame)
            plt.title("Detector %d: attribute %s" % (i, channame))
            fig = plt.gcf()
            pf = mass.core.utilities.MouseClickReader(fig)

            for j in range(nclicks):
                while True:
                    plt.waitforbuttonpress()
                    try:
                        pfx = '%g' % pf.x
                    except TypeError:
                        continue
                    print('Click on line #%d at %s' % (j + 1, pfx))
                    x.append(pf.x)
                    break

        xvalues = np.array(x)
        xvalues.shape = (self.n_channels, nclicks)
        return xvalues

    def find_named_features_with_mouse(self, name='Mn Ka1', channame='p_filt_value',
                                       prange=None, trange=None, energy=None):

        if energy is None:
            energy = mass.calibration.energy_calibration.STANDARD_FEATURES[name]

        print("Please click with the mouse on each channel's histogram at the %s line" % name)
        xvalues = self.find_features_with_mouse(channame=channame, nclicks=1,
                                                prange=prange, trange=trange).ravel()
        for ds, xval in zip(self.datasets, xvalues):
            calibration = ds.calibration[channame]
            calibration.add_cal_point(xval, energy, name)

    def report(self):
        """
        Report on the number of data points and similar
        """
        for ds in self.datasets:
            good = ds.cuts.good()
            ng = ds.cuts.good().sum()
            dt = (ds.p_timestamp[good][-1] * 1.0 - ds.p_timestamp[good][0])  # seconds
            npulse = np.arange(len(good))[good][-1] - good.argmax() + 1
            rate = (npulse - 1.0) / dt
            print('chan %2d %6d pulses (%6.3f Hz over %6.4f hr) %6.3f%% good' %
                  (ds.channum, npulse, rate, dt / 3600., 100.0 * ng / npulse))

    def plot_noise_autocorrelation(self, axis=None, channels=None, cmap=None,
                                   legend=True):
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
        for i, ds in enumerate(self.datasets):
            if i not in channels:
                continue
            noise = ds.noise_records
            noise.plot_autocorrelation(axis=axis, label='TES %d' % i,
                                       color=cmap(float(i) / self.n_channels))
        axis.set_xlabel("Time lag (ms)")
        if legend:
            plt.legend(loc='best')
            ltext = axis.get_legend().get_texts()
            plt.setp(ltext, fontsize='small')

    def save_pulse_energies_ascii(self, filename='all'):
        filename += '.energies'
        energy = []
        for ds in self:
            energy = np.hstack((energy, ds.p_energy[ds.cuts.good()]))
        np.savetxt(filename, energy, fmt='%.10e')

    def copy(self):
        self.clear_cache()
        g = TESGroup(self.filenames, self.noise_filenames)
        g.__dict__.update(self.__dict__)
        g.datasets = tuple([d.copy() for d in self.datasets])
        return g

    def join(self, *others):
        # Ensure they are compatible
        print('join probably doesnt work since galen messed with it moving things inside datasets')
        for g in others:
            for attr in ('nPresamples', 'nSamples', 'noise_only', 'timebase'):
                if g.__dict__[attr] != self.__dict__[attr]:
                    raise RuntimeError("All objects must agree on group.%s" % attr)

        for g in others:
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

        first_pnum, end_pnum = -1, -1
        for ds in self.datasets:
            a, b = ds.read_segment(segnum)

            # Possibly some channels are shorter than others (in TDM data)
            # Make sure to return first_pnum,end_pnum for longest VALID channel only
            if a >= 0:
                if first_pnum >= 0:
                    assert a == first_pnum
                first_pnum = a
            if b >= end_pnum:
                end_pnum = b
        self._cached_segment = segnum
        self._cached_pnum_range = first_pnum, end_pnum
        return first_pnum, end_pnum

    def plot_noise(self, axis=None, channels=None, scale_factor=1.0, sqrt_psd=False,
                   cmap=None, legend=True):
        """Compare the noise power spectra.

        <channels>    Sequence of channels to display.  If None, then show all.
        <scale_factor> Multiply counts by this number to get physical units.
        <sqrt_psd>     Whether to show the sqrt(PSD) or (by default) the PSD itself.
        <cmap>         A matplotlib color map.  Defaults to something.
        `legend` -- Whether to plot the legend
        """

        if channels is None:
            channels = list(self.channel.keys())
            channels.sort()

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if scale_factor == 1.0:
            units = "Counts"
        else:
            units = "Scaled counts"

        axis.grid(True)
        if cmap is None:
            cmap = plt.cm.get_cmap("spectral")
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel: continue
            ds = self.channel[channum]
            yvalue = ds.noise_psd[:] * scale_factor**2
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
                axis.set_ylabel("PSD$^{1/2}$ (%s/Hz$^{1/2}$)" % units)
            try:
                df = ds.noise_psd.attrs['delta_f']
                freq = np.arange(1, 1 + len(yvalue)) * df
                axis.plot(freq, yvalue, label='TES chan %d' % channum,
                          color=cmap(float(ds_num) / len(channels)))
            except:
                print("Could not plot channel %4d." % channum)
        axis.set_xlim([freq[1] * 0.9, freq[-1] * 1.1])
        axis.set_ylabel("Power Spectral Density (%s^2/Hz)" % units)
        axis.set_xlabel("Frequency (Hz)")
        axis.loglog()
        if legend:
            plt.legend(loc='best')
            ltext = axis.get_legend().get_texts()
            plt.setp(ltext, fontsize='small')

    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        for ds in self:
            ds.compute_noise_spectra(max_excursion, n_lags, forceNew)

    def apply_cuts(self, cuts, forceNew=True):
        for ds in self:
            ds.apply_cuts(cuts, forceNew)

    def avg_pulses_auto_masks(self, max_pulses_to_use=7000, forceNew=False):
        """
        Compute average pulse using an automatically generated mask of
        +- 5%% around the median pulse_average value. Use no more than
        the first `max_pulses_to_use` good pulses.
        """
        median_pulse_avg = np.ones(self.n_channels, dtype=np.float)
        for i, ds in enumerate(self.datasets):
            if ds.good().sum() > 0:
                median_pulse_avg[i] = np.median(ds.p_pulse_average[ds.good()])
            else:
                self.set_chan_bad(ds.channum, "No good pulses")
        masks = self.make_masks([.95, 1.05], use_gains=True, gains=median_pulse_avg)
        for m in masks:
            if np.sum(m) > max_pulses_to_use:
                good_so_far = np.cumsum(m)
                stop_at = (good_so_far == max_pulses_to_use).argmax()
                m[stop_at+1:] = False
        self.compute_average_pulse(masks, forceNew=forceNew)

    def drift_correct(self, forceNew=False, category=None):
        for ds in self:
            try:
                ds.drift_correct(forceNew, category)
            except:
                self.set_chan_bad(ds.channum, "failed drift correct")

    def phase_correct(self, plot=False, forceNew=False, category=None):
        for ds in self:
            try:
                ds.phase_correct(forceNew=forceNew, category=category)
            except Exception as e:
                self.set_chan_bad(ds.channum, "failed phase_correct with %s" % e)

    def phase_correct2014(self, typical_resolution, maximum_num_records=50000,
                          plot=False, forceNew=False, pre_sanitize_p_filt_phase=True, category=None):
        if pre_sanitize_p_filt_phase:
            self.sanitize_p_filt_phase()
        for ds in self:
            try:
                ds.phase_correct2014(typical_resolution, maximum_num_records, plot, forceNew, category)
            except:
                self.set_chan_bad(ds.channum, "failed phase_correct2014")

    def sanitize_p_filt_phase(self):
        ds = self.first_good_dataset
        self.register_boolean_cut_fields("filt_phase")
        print("filt_phase cut")
        for ds in self:
            ds.cuts.cut("filt_phase", np.abs(ds.p_filt_phase[:])>2)

    def calibrate(self, attr, line_names, name_ext="", size_related_to_energy_resolution=10,
                  fit_range_ev=200, excl=(), plot_on_fail=False,
                  bin_size_ev=2, category=None, forceNew=False, maxacc=0.015, nextra=3,
                  param_adjust_closure=None):
        for ds in self:
            try:
                ds.calibrate(attr, line_names, name_ext, size_related_to_energy_resolution,
                             fit_range_ev, excl, plot_on_fail,
                             bin_size_ev, category, forceNew, maxacc, nextra,
                             param_adjust_closure=param_adjust_closure)
            except:
                self.set_chan_bad(ds.channum, "failed calibration %s" % attr + name_ext)
        self.convert_to_energy(attr, attr + name_ext)

    def convert_to_energy(self, attr, calname=None):
        if calname is None:
            calname = attr
        print("for all channels converting %s to energy with calibration %s" % (attr, calname))
        for ds in self:
            ds.convert_to_energy(attr, calname)

    def time_drift_correct(self, poly_order=1, attr='p_filt_value_phc',
                           num_lines=None, forceNew=False):
        for ds in self:
            if poly_order == 1:
                ds.time_drift_correct(attr, forceNew)
            elif poly_order > 1:
                ds.time_drift_correct_polynomial(poly_order, attr, num_lines, forceNew)
            else:
                raise ValueError('%g is invalid value of poly_order' % poly_order)

    def plot_count_rate(self, bin_s=60, title=""):
        bin_edge = np.arange(self.first_good_dataset.p_timestamp[0],
                             np.amax(self.first_good_dataset.p_timestamp), bin_s)
        bin_centers = bin_edge[:-1] + 0.5 * (bin_edge[1] - bin_edge[0])
        rates_all = np.array([ds.count_rate(False, bin_edge)[1] for ds in self])
        rates_good = np.array([ds.count_rate(True, bin_edge)[1] for ds in self])
        plt.figure()
        plt.subplot(311)
        plt.plot(bin_centers, rates_all.T)
        plt.ylabel("all by chan")
        plt.subplot(312)
        plt.plot(bin_centers, rates_good.T)
        plt.ylabel("good by chan")
        plt.subplot(313)
        print(rates_all.sum(axis=-1).shape)
        plt.plot(bin_centers, rates_all.sum(axis=0))
        plt.ylabel("all array")
        plt.grid("on")

        plt.figure()
        plt.plot([ds.channum for ds in self], rates_all.mean(axis=1), 'o', label="all")
        plt.plot([ds.channum for ds in self], rates_good.mean(axis=1), 'o', label="good")
        plt.xlabel("channel number")
        plt.ylabel("average trigger/s")
        plt.grid("on")
        plt.legend()

    def smart_cuts(self, threshold=10.0, n_trainings=10000, forceNew=False):
        for ds in self:
            ds.smart_cuts(threshold, n_trainings, forceNew)


def _extract_channum(name):
    return int(name.split('_chan')[1].split(".")[0])


def _remove_unmatched_channums(filenames1, filenames2, never_use=None, use_only=None):
    """Extract the channel number in the filenames appearing in both lists.
    Remove from each list any file whose channel number doesn't appear on both lists.
    Also remove any file whose channel number is in the `never_use` list.
    If `use_only` is a sequence of channel numbers, use only the channels on that list.

    If either `filenames1` or `filenames2` is empty, do nothing."""

    # If one list is empty, then matching is not required or expected.
    if filenames1 is None or len(filenames1) == 0 \
        or filenames2 is None or len(filenames2) == 0:
        return

    # Now make a mapping of channel numbers to names.
    names1 = {_extract_channum(f):f for f in filenames1}
    names2 = {_extract_channum(f):f for f in filenames2}
    cnum1 = set(names1.keys())
    cnum2 = set(names2.keys())

    # Find the set of valid channel numbers.
    valid_cnum = cnum1.intersection(cnum2)
    if never_use is not None:
        valid_cnum -= set(never_use)
    if use_only is not None:
        valid_cnum = valid_cnum.intersection(set(use_only))

    # Remove invalid channel numbers
    for c in (cnum1-valid_cnum):
        filenames1.remove(names1[c])
    for c in (cnum2-valid_cnum):
        filenames2.remove(names2[c])


def _sort_filenames_numerically(fnames, inclusion_list=None):
    """Take a sequence of filenames of the form '*_chanXXX.*'
    and sort it according to the numerical value of channel number XXX.
    If inclusion_list is not None, then it must be a container with the
    channel numbers to be included in the output.
    """
    if fnames is None or len(fnames) == 0:
        return None
    chan2fname = {}
    for name in fnames:
        channum = _extract_channum(name)
        if inclusion_list is not None and channum not in inclusion_list:
            continue
        chan2fname[channum] = name
    sorted_chan = list(chan2fname.keys())
    sorted_chan.sort()
    sorted_fnames = [chan2fname[key] for key in sorted_chan]
    return sorted_fnames


def _glob_expand(pattern):
    """If `pattern` is a string, treat it as a glob pattern and return the glob-result
    as a list. If it isn't a string, return it unchanged (presumably then it's already
    a sequence)."""
    if not isinstance(pattern, str):
        return pattern

    result = glob.glob(pattern)
    return _sort_filenames_numerically(result)


def _replace_path(fnames, newpath):
    """Take a sequence of filenames <fnames> and replace the directories leading to each
    with <newpath>"""
    if fnames is None or len(fnames) == 0:
        return None
    result = []
    for f in fnames:
        _, name = os.path.split(f)
        result.append(os.path.join(newpath, name))
    return result


class CrosstalkVeto(object):
    """
    An object to allow vetoing of data in 1 channel when another is hit
    """

    def __init__(self, datagroup=None, window_ms=(-10, 3), pileup_limit=100):
        if datagroup is None:
            return

        window_ms = np.array(window_ms, dtype=np.int)
        self.window_ms = window_ms
        self.n_channels = datagroup.n_channels
        self.n_pulses = datagroup.nPulses
#        self.veto = np.zeros((self.n_channels, self.n_pulses), dtype=np.bool8)

        ms0 = np.array([ds.p_timestamp[0] for ds in datagroup.datasets]).min() * 1e3 + window_ms[0]
        ms9 = np.array([ds.p_timestamp[-1] for ds in datagroup.datasets]).max() * 1e3 + window_ms[1]
        self.nhits = np.zeros(ms9 - ms0 + 1, dtype=np.int8)
        self.time0 = ms0

        for ds in datagroup.datasets:
            g = ds.cuts.good()
            vetotimes = np.asarray(ds.p_timestamp[g] * 1e3 - ms0, dtype=np.int64)
            vetotimes[vetotimes < 0] = 0
            print(vetotimes, len(vetotimes), 1.0e3 * ds.nPulses / (ms9 - ms0)),
            a, b = window_ms
            b += 1
            for t in vetotimes:
                self.nhits[t + a:t + b] += 1

            pileuptimes = vetotimes[ds.p_postpeak_deriv[g] > pileup_limit]
            print(len(pileuptimes))
            for t in pileuptimes:
                self.nhits[t + b:t + b + 8] += 1

    def copy(self):
        v = CrosstalkVeto()
        v.__dict__ = self.__dict__.copy()
        return v

    def veto(self, times_sec):
        """Return boolean vector for whether a given moment is vetoed.  Times are given in
        seconds.  Resolution is 1 ms for the veto."""
        index = np.asarray(times_sec * 1e3 - self.time0 + 0.5, dtype=np.int)
        return self.nhits[index] > 1