"""
channel_group.py

Part of the Microcalorimeter Analysis Software System (MASS).

This module defines classes that handle one or more TES data streams together.
"""

from collections import Iterable
from functools import reduce
import os
import re

import six

import numpy as np
import matplotlib.pylab as plt
import palettable
import h5py

import mass.core.analysis_algorithms
import mass.calibration.energy_calibration

from mass.calibration.energy_calibration import EnergyCalibration
from mass.core.channel import MicrocalDataSet, PulseRecords, NoiseRecords, GroupLooper
from mass.core.cython_channel import CythonMicrocalDataSet
from mass.core.cut import CutFieldMixin
from mass.core.optimal_filtering import Filter
from mass.core.utilities import InlineUpdater, show_progress
from mass.core.ljh_util import remove_unpaired_channel_files, \
    filename_glob_expand

import logging
LOG = logging.getLogger("mass")


def _generate_hdf5_filename(rawname):
    """Generate the appropriate HDF5 filename based on a file's LJH name.

    Takes /path/to/data_chan33.ljh --> /path/to/data_mass.hdf5
    """
    import re
    fparts = re.split(r"_chan\d+", rawname)
    prefix_path = fparts[0]
    if rawname.endswith("noi"):
        prefix_path += '_noise'
    return prefix_path + "_mass.hdf5"


def RestoreTESGroup(hdf5filename, hdf5noisename=None):
    """Generate a TESGroup object from a data summary HDF5 file.

    Args:
        hdf5filename (string): the data summary file
        hdf5noisename (string): the noise summary file; this can often be inferred from
            the noise raw filenames, which are stored in the pulse HDF5 file (assuming you
            aren't doing something weird). (default None)
    """
    pulsefiles = []
    channum = []
    noisefiles = []
    generated_noise_hdf5_name = None

    with h5py.File(hdf5filename, "r") as h5file:
        for name, group in h5file.items():
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
        with h5py.File(hdf5noisename, "r") as h5file:
            for ch in channum:
                group = h5file['chan%d' % ch]
                noisefiles.append(group.attrs['filename'])
            h5file.close()
    else:
        hdf5noisename = generated_noise_hdf5_name

    return TESGroup(pulsefiles, noisefiles, hdf5_filename=hdf5filename,
                    hdf5_noisefilename=hdf5noisename)


class TESGroup(CutFieldMixin, GroupLooper):
    """The interface for a group of one or more microcalorimeters."""

    def __init__(self, filenames, noise_filenames=None, noise_only=False,
                 noise_is_continuous=True, max_cachesize=None,
                 hdf5_filename=None, hdf5_noisefilename=None,
                 never_use=None, use_only=None):
        """Set up a group of related data sets by their filenames.

        Args:
            filenames: either a sequence of pulse filenames, or a shell "glob pattern"
                that can be expanded to a list of filenames
            noise_filenames: either a sequence of noise filenames, or a shell
                "glob pattern" that can be expanded to a list of filenames, or
                None (if there are to be no noise data). (default None)
            noise_only (bool): if True, then treat this as a pulse-free analysis
                and take filenames to be a list of noise files (default False)
            noise_is_continuous (bool): whether to treat sequential noise records
                as continuous in time (default True)
            max_cachesize: the maximum number of bytes to read when reading and
                caching raw data. If None, use the default of ??.
            hdf5_filename: if not None, the filename to use for backing the
                analyzed data in HDF5 (default None). If None, choose a sensible
                filename based on the input data filenames.
            hdf5_noisefilename: if not None, the filename to use for backing the
                analyzed noise data in HDF5 (default None). If None, choose a sensible
                filename based on the input data filenames.
            never_use: if not None, a sequence of channel numbers to ignore
                (default None).
            use_only:  if not None, a sequence of channel numbers to use, i.e.
                ignore all channels not on this list (default None).
        """

        if noise_filenames is not None and len(noise_filenames) == 0:
            noise_filenames = None

        # In the noise_only case, you can put the noise file names either in the
        # usual (pulse) filenames argument or in the noise_filenames argument.
        self.noise_only = noise_only
        if noise_only and noise_filenames is None:
            filenames, noise_filenames = (), filenames

        # Handle the case that either filename list is a glob pattern (e.g.,
        # "files_chan*.ljh"). Note that this will return a list, never a string,
        # even if there is only one result from the pattern matching.
        filenames = filename_glob_expand(filenames)
        noise_filenames = filename_glob_expand(noise_filenames)

        # If using a glob pattern especially, we have to be careful to eliminate files that are
        # missing a partner, either noise without pulse or pulse without noise.
        remove_unpaired_channel_files(filenames, noise_filenames, never_use=never_use, use_only=use_only)

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
            if isinstance(filenames, six.string_types):
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
            if isinstance(noise_filenames, six.string_types):
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
            LOG.info("%s %i", fname, pulse.nPulses)
            if pulse.nPulses == 0:
                LOG.info("TESGroup is skipping a file that has zero pulses: %s", fname)
                continue  # don't load files with zero pulses

            hdf5_group = self.hdf5_file.require_group("chan%d" % pulse.channum)
            hdf5_group.attrs['filename'] = fname

            dset = CythonMicrocalDataSet(pulse.__dict__, tes_group=self, hdf5_group=hdf5_group)

            if 'calibration' in hdf5_group:
                hdf5_cal_grp = hdf5_group['calibration']
                for cal_name in hdf5_cal_grp:
                    dset.calibration[cal_name] = EnergyCalibration.load_from_hdf5(hdf5_cal_grp, cal_name)

            if 'why_bad' in hdf5_group.attrs:
                self._bad_channums[dset.channum] = [comment.decode() for comment in
                                                    hdf5_group.attrs['why_bad']]

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
                    LOG.warn("WARNING: TESGroup did not add data: channums don't match %s, %s", fname, nf)
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

    def __repr__(self):
        if self.noise_only:
            return "{0:s}(noise={1:s}, noise_only=True)".format(self.__class__.__name__,
                                                                os.path.dirname(self.noise_filenames[0]))
        else:
            return "{0:s}(pulse={1:s}, noise={2:s})".format(self.__class__.__name__,
                                                            os.path.dirname(self.filenames[0]),
                                                            os.path.dirname(self.noise_filenames[0]))

    def __iter__(self):
        """Iterator over the self.datasets in channel number order"""
        for ds in self.iter_channels():
            yield ds

    def iter_channels(self, include_badchan=False):
        """Iterator over the self.datasets in channel number order

        Args:
            include_badchan (bool): whether to include officially bad channels
                in the result (default False).
        """
        for ds in self.datasets:
            if not include_badchan:
                if ds.channum in self._bad_channums:
                    continue
            yield ds

    def iter_channel_numbers(self, include_badchan=False):
        """Iterator over  the channel numbers in channel number order

        Args:
            include_badchan (bool): whether to include officially bad channels
                in the result (default False).
        """
        for ds in self.iter_channels(include_badchan=include_badchan):
            yield ds.channum

    def set_chan_good(self, *args):
        """Set one or more channels to be good.

        (No effect for channels already listed as good.)

        Args:
            *args  Arguments to this function are integers or containers of integers.  Each
                integer is removed from the bad-channels list.
        """
        added_to_list = set.union(*[set(x) if isinstance(x, Iterable) else {x} for x in args])

        for channum in added_to_list:
            if channum in self._bad_channums:
                comment = self._bad_channums.pop(channum)
                del self.hdf5_file["chan{0:d}".format(channum)].attrs['why_bad']
                LOG.info("chan %d set good, had previously been set bad for %s", channum, str(comment))
            else:
                LOG.info("chan %d not set good because it was not set bad", channum)

    def set_chan_bad(self, *args):
        """Set one or more channels to be bad.

        (No effect for channels already listed as bad.)

        Args:
            *args  Arguments to this function are integers or containers of integers.  Each
                integer is added to the bad-channels list.

        Examples:
            data.set_chan_bad(1, "too few good pulses")
            data.set_chan_bad(103, [1, 3, 5], "detector unstable")
        """
        added_to_list = set.union(*[set(x) if isinstance(x, Iterable) else {x} for x in args
                                    if not isinstance(x, six.string_types)])
        comment = reduce(lambda x, y: y, [x for x in args if isinstance(x, six.string_types)], '')

        for channum in added_to_list:
            new_comment = self._bad_channums.get(channum, []) + [comment]
            self._bad_channums[channum] = new_comment
            LOG.warn('WARNING: Chan %s flagged bad because %s', channum, comment)
            self.hdf5_file["chan{0:d}".format(channum)].attrs['why_bad'] =  \
                np.asarray(new_comment, dtype=np.bytes_)

    def n_good_channels(self):
        return self.n_channels - len(self._bad_channums.keys())

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
            LOG.warn("""Warning!  This feature is only half-complete.  Currently, granularity is limited.
    Only full "segments" of size %d records can be ignored.
    Will use %d segments and ignore %d.""", self.pulses_per_seg, self._allowed_segnums.sum(),
                     self.n_segments - self._allowed_segnums.sum())

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
                    LOG.info('We can skip segment %4d', i)
                    continue  # Don't need anything in this segment.  Sweet!
            if segment_mask is not None:
                if not segment_mask[i]:
                    LOG.info('We can skip segment %4d', i)
                    continue  # Don't need anything in this segment.  Sweet!
            first_rnum, end_rnum = self.read_segment(i)
            yield first_rnum, end_rnum

    @show_progress("summarize_data")
    def summarize_data(self, peak_time_microsec=None, pretrigger_ignore_microsec=None,
                       include_badchan=False, forceNew=False, use_cython=True):
        """Summarize the data with per-pulse summary quantities for each channel.

        peak_time_microsec will be determined automatically if None, and will be
        stored in channels as ds.peak_samplenumber.

        Args:
            use_cython uses a cython (aka faster) implementation of summarize.
        """
        nchan = float(len(self.channel.keys())) if include_badchan else float(self.num_good_channels)

        for i, ds in enumerate(self.iter_channels(include_badchan)):
            try:
                ds.summarize_data(peak_time_microsec=peak_time_microsec,
                                  pretrigger_ignore_microsec=pretrigger_ignore_microsec,
                                  forceNew=forceNew, use_cython=use_cython)
                yield (i + 1.0) / nchan
                self.hdf5_file.flush()
            except Exception as e:
                self.set_chan_bad(ds.channum, "summarize_data failed with %s" % e)

    def calc_external_trigger_timing(self, after_last=False, until_next=False,
                                     from_nearest=False, forceNew=False):
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
                        mass.core.analysis_algorithms.nearest_arrivals(ds.p_rowcount[:],
                                                                       external_trigger_rowcount)
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
        """Read one trace from cache or disk.

        Args:
            record_num (int): the pulse record number to read.
            dataset_num (int): the dataset number to use
            channum (int): the channel number to use (if both this and dataset_num
                are given, use channum in preference).

        Returns:
            an ndarray: the pulse numbered <record_num>
        """
        ds = self.channel.get(channum, self.datasets[dataset_num])
        return ds.read_trace(record_num)

    def plot_traces(self, pulsenums, dataset_num=0, channum=None, pulse_summary=True, axis=None,
                    difference=False, residual=False, valid_status=None, shift1=False):
        """Plot some example pulses, given by record number.

        Args:
            <pulsenums>   A sequence of record numbers, or a single number.
            <dataset_num> Dataset index (0 to n_dets-1, inclusive).  Will be used only if
                          <channum> is invalid.
            <channum>    Channel number.  If valid, it will be used instead of dataset_num.
            <pulse_summary> Whether to put text about the first few pulses on the plot
                (default True)
            <axis>       A plt axis to plot on (default None, i.e., create a new axis)
            <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
                (default False).
            <residual>   Whether to show the residual between data and opt filtered model,
                 or just raw data (default False).
            <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                that have been cut.  If "cut", show only those that have been cut.
                (default None).
            <shift1>     Whether to take pulses with p_shift1==True and delay them by
                1 sample (default False, i.e., show the pure raw data w/o shifting).
        """

        if channum in self.channel:
            dataset = self.channel[channum]
            dataset_num = dataset.index
        else:
            dataset = self.datasets[dataset_num]
            if channum is not None:
                LOG.info("Cannot find channum[%d], so using dataset #%d", channum, dataset_num)
        return dataset.plot_traces(pulsenums, pulse_summary, axis, difference,
                                   residual, valid_status, shift1)

    def plot_summaries(self, quantity, valid='uncut', downsample=None, log=False, hist_limits=None,
                       channel_numbers=None, dataset_numbers=None):
        """Plot a summary of one quantity from the data set.

        This plot includes time series and histograms of this quantity.  This
        method plots all channels in the group, but only one quantity.  If you
        would rather see all quantities for one channel, then use the group's
        group.channel[i].plot_summaries() method.

        Args:
            quantity: A case-insensitive whitespace-ignored one of the following list, or the numbers
               that go with it:
               "Pulse RMS" (0)
               "Pulse Avg" (1)
               "Peak Value" (2)
               "Pretrig RMS" (3)
               "Pretrig Mean" (4)
               "Max PT Deriv" (5)
               "Rise Time" (6)
               "Peak Time" (7)
               "Peak Index" (8)

            valid: The words 'uncut' or 'cut', meaning that only uncut or cut data
                are to be plotted *OR* None, meaning that all pulses should be plotted.

            downsample (int): To prevent the scatter plots (left panels) from getting too crowded,
                 plot only one out of this many samples.  If None, then plot will be
                 downsampled to 10,000 total points.
            log (bool): Use logarithmic y-axis on the histograms (right panels).
            hist_limits: if not None, limit the right-panel histograms to this range.
            channel_numbers: A sequence of channel numbers to plot. If None, then plot all.
            dataset_numbers: A sequence of the datasets [0...n_channels-1] to plot.  If None
                (the default) then plot all datasets in numerical order. But ignored
                if channel_numbers is not None.
        """

        plottables = (
            ("p_pulse_rms", 'Pulse RMS', 'magenta', None),
            ("p_pulse_average", 'Pulse Avg', 'purple', [0, 5000]),
            ("p_peak_value", 'Peak value', 'blue', None),
            ("p_pretrig_rms", 'Pretrig RMS', 'green', [0, 4000]),
            ("p_pretrig_mean", 'Pretrig Mean', '#00ff26', None),
            ("p_postpeak_deriv", 'Max PostPk deriv', 'gold', [0, 700]),
            ("p_rise_time[:]*1e3", 'Rise time (ms)', 'orange', [0, 12]),
            ("p_peak_time[:]*1e3", 'Peak time (ms)', 'red', [-3, 9]),
            ("p_peak_index[:]", 'Peak index', 'red', [600, 800])
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
        last_midnight = last_record - (last_record % 86400)
        hour_offset = last_midnight/3600.

        plt.clf()
        ny_plots = len(datasets)
        for i, (channum, ds) in enumerate(zip(channel_numbers, datasets)):

            # Convert "uncut" or "cut" to array of all good or all bad data
            if isinstance(valid, six.string_types):
                if "uncut" in valid.lower():
                    valid_mask = ds.cuts.good()
                    LOG.info("Plotting only uncut data"),
                elif "cut" in valid.lower():
                    valid_mask = ds.cuts.bad()
                    LOG.info("Plotting only cut data"),
                elif 'all' in valid.lower():
                    valid_mask = None
                    LOG.info("Plotting all data, cut or uncut"),
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
            LOG.info("Chan %3d (%d records; %d in scatter plots)", channum, nrecs, hour.shape[0])

            (vect, label, color, default_limits) = plottable
            if hist_limits is None:
                limits = default_limits
            else:
                limits = hist_limits

            # Vectors are being sampled and multiplied, so eval() is needed.
            vect = eval("ds.%s" % vect)[valid_mask]

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
                   pulse_rms_range=None, gains=None):
        """Generate a sequence of masks for use in compute_average_pulses().

        Args:
            pulse_avg_range -- A 2-sequence giving the (minimum,maximum) p_pulse_average
            pulse_peak_range -- A 2-sequence giving the (minimum,maximum) p_peak_value
            pulse_rms_range --  A 2-sequence giving the (minimum,maximum) p_pulse_rms
            gains -- The set of gains to use, if any.

        Returns:
            a list of ndvectors of boolean dtype, one list per channel.
            Each vector says whether each pulse in that channel is in the given
            range of allowed pulse sizes.
        """

        for ds in self:
            if ds.nPulses == 0:
                self.set_chan_bad(ds.channum, "has 0 pulses")

        masks = []
        if gains is None:
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
            LOG.warn("Warning: make_masks uses only one range argument.  Checking only '%s'.", vectname)

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

        Args:
            masks: A sequence of length self.n_channels, one sequence per channel.
                The elements of `masks` should be booleans or interpretable as booleans.

            subtract_mean (bool): whether each average pulse will subtract a constant
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

    def plot_average_pulses(self, axis=None, channels=None, cmap=None, legend=True, fcut=None):
        """Plot average pulse for channel number <channum> on matplotlib.Axes
        <axis>, or on a new Axes if <axis> is None. If <channum> is not a valid
        channel number, then plot all average pulses. If <fcut> is not None,
        then lowpass filter the traces with this cutoff frequency prior to
        plotting.
        """

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if channels is None:
            channels = list(self.channel.keys())
            channels.sort()

        if cmap is None:
            cmap = plt.cm.get_cmap("nipy_spectral")

        dt = (np.arange(self.nSamples) - self.nPresamples) * self.timebase * 1e3

        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            avg_pulse = ds.average_pulse[:].copy()
            if fcut != None:
                avg_pulse = mass.core.analysis_algorithms.filter_signal_lowpass(avg_pulse, 1./self.timebase, fcut)
            plt.plot(dt, avg_pulse, label="Chan %d" % ds.channum,
                     color=cmap(float(ds_num) / len(channels)))

        plt.title("Average pulse for each channel when it is hit")

        plt.xlabel("Time past trigger (ms)")
        plt.ylabel("Raw counts")
        plt.xlim([dt[0], dt[-1]])
        if legend:
            plt.legend(loc='best')
            if len(channels) > 12:
                ltext = axis.get_legend().get_texts()
                plt.setp(ltext, fontsize='small')

    @show_progress("compute_filters")
    def compute_filters(self, fmax=None, f_3db=None, forceNew=False):
        """
        compute_filters(self, fmax=None, f_3db=None, forceNew=False)

        Looks at ds._use_new_filters to decide which type of filter to use.
        """
        # Analyze the noise, if not already done
        needs_noise = any([ds.noise_autocorr[0] == 0.0 or
                           ds.noise_psd[1] == 0 for ds in self])
        if needs_noise:
            LOG.debug("Computing noise autocorrelation and spectrum")
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
                LOG.info("chan %d skipping compute_filter because already done, and loading filter",
                         ds.channum)
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

    def plot_filters(self, axis=None, channels=None, cmap=None,
                     filtname="filt_noconst", legend=True):
        """Plot the optimal filters.

        Args:
            channels: Sequence of channel numbers to display.  If None, then show all.
        """

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if channels is None:
            channels = list(self.channel.keys())
            channels.sort()

        if cmap is None:
            cmap = plt.cm.get_cmap("nipy_spectral")

        axis.grid(True)
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            plt.plot(ds.filter.__dict__[filtname], label="Chan %d" % channum,
                     color=cmap(float(ds_num) / len(channels)))

        plt.xlabel("Sample number")
        if legend:
            plt.legend(loc='best')
            if len(channels) > 12:
                ltext = axis.get_legend().get_texts()
                plt.setp(ltext, fontsize='small')

    def summarize_filters(self, filter_name='noconst', std_energy=5898.8):
        rms_fwhm = np.sqrt(np.log(2) * 8)  # FWHM is this much times the RMS
        LOG.info('V/dV for time, Fourier filters: ')
        for i, ds in enumerate(self):
            try:
                if ds.filter is not None:
                    rms = ds.filter.variances[filter_name]**0.5
                else:
                    rms = ds.hdf5_group['filters/filt_%s' % filter_name].attrs['variance']**0.5
                v_dv = (1 / rms) / rms_fwhm
                LOG.info("Chan %3d filter %-15s Predicted V/dV %6.1f  Predicted res at %.1f eV: %6.1f eV",
                         ds.channum, filter_name, v_dv, std_energy, std_energy / v_dv)
            except Exception as e:
                LOG.warn("Filter %d can't be used", i)
                LOG.warn(e)

    @show_progress("filter_data")
    def filter_data(self, filter_name='filt_noconst', transform=None, include_badchan=False,
                    forceNew=False, use_cython=True):
        nchan = float(len(self.datasets)) if include_badchan else float(self.num_good_channels)

        for i, ds in enumerate(self.iter_channels(include_badchan)):
            ds.filter_data(filter_name, transform, forceNew, use_cython=use_cython)
            yield (i+1.0) / nchan

    def report(self):
        """Report on the number of data points and similar."""
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
        """Plot the noise autocorrelation functions.

        Args:
            channels: Sequence of channel numbers to display.  If None, then show all.
        """

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if channels is None:
            channels = list(self.channel.keys())
            channels.sort()

        if cmap is None:
            cmap = plt.cm.get_cmap("nipy_spectral")

        axis.grid(True)
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            noise = ds.noise_records
            noise.plot_autocorrelation(axis=axis, label='Chan %d' % channum,
                                       color=cmap(float(ds_num) / len(channels)))
        plt.xlabel("Time lag (ms)")
        if legend:
            plt.legend(loc='best')
            if len(channels) > 12:
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
        channels in the group.

        Args:
            use_cache (bool): if True, use the cached value when possible.

        Returns:
            (first,end) where these are the number of the first record in
                that segment and 1 more than the number of the last record.
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

    def plot_noise(self, axis=None, channels=None, cmap=None, scale_factor=1.0,
                   sqrt_psd=False, legend=True):
        """Plot the noise power spectra.

        Args:
            channels:    Sequence of channels to display.  If None, then show all.
            scale_factor: Multiply counts by this number to get physical units.
            sqrt_psd:     Whether to show the sqrt(PSD) or (by default) the PSD itself.
            cmap:         A matplotlib color map.  Defaults to something.
            legend (bool): Whether to plot the legend (default True)
        """

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if channels is None:
            channels = list(self.channel.keys())
            channels.sort()

        if cmap is None:
            cmap = plt.cm.get_cmap("nipy_spectral")

        if scale_factor == 1.0:
            units = "Counts"
        else:
            units = "Scaled counts"

        axis.grid(True)
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            yvalue = ds.noise_psd[:] * scale_factor**2
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
                axis.set_ylabel("PSD$^{1/2}$ (%s/Hz$^{1/2}$)" % units)
            try:
                df = ds.noise_psd.attrs['delta_f']
                freq = np.arange(1, 1 + len(yvalue)) * df
                axis.plot(freq, yvalue, label='Chan %d' % channum,
                          color=cmap(float(ds_num) / len(channels)))
            except:
                LOG.warn("WARNING: Could not plot channel %4d.", channum)
        axis.set_xlim([freq[1] * 0.9, freq[-1] * 1.1])
        axis.set_ylabel("Power Spectral Density (%s^2/Hz)" % units)
        axis.set_xlabel("Frequency (Hz)")
        axis.loglog()
        if legend:
            plt.legend(loc='best')
            if len(channels) > 12:
                ltext = axis.get_legend().get_texts()
                plt.setp(ltext, fontsize='small')

    def avg_pulses_auto_masks(self, max_pulses_to_use=7000, forceNew=False):
        """Compute an average pulse.

        Compute average pulse using an automatically generated mask of
        +- 5%% around the median pulse_average value.

        Args:
            max_pulses_to_use (int): Use no more than
                the first this many good pulses (default 7000).
            forceNew (bool): whether to re-compute if results already exist (default False)
        """
        median_pulse_avg = np.ones(self.n_channels, dtype=np.float)
        for i, ds in enumerate(self.datasets):
            if ds.good().sum() > 0:
                median_pulse_avg[i] = np.median(ds.p_pulse_average[ds.good()])
            else:
                self.set_chan_bad(ds.channum, "No good pulses")
        masks = self.make_masks([.95, 1.05], gains=median_pulse_avg)
        for m in masks:
            if np.sum(m) > max_pulses_to_use:
                good_so_far = np.cumsum(m)
                stop_at = (good_so_far == max_pulses_to_use).argmax()
                m[stop_at+1:] = False
        self.compute_average_pulse(masks, forceNew=forceNew)

    def correct_flux_jumps(self, flux_quant):
        '''Remove 'flux' jumps' from pretrigger mean.

        When using umux readout, if a pulse is recorded that has a very fast
        rising edge (e.g. a cosmic ray), the readout system will "slip" an
        integer number of flux quanta. This means that the baseline level
        returned to after the pulse will different from the pretrigger value by
        an integer number of flux quanta. This causes that pretrigger mean
        summary quantity to jump around in a way that causes trouble for the
        rest of MASS. This function attempts to correct these jumps.

        Arguments:
        flux_quant -- size of 1 flux quantum
        '''
        for ds in self:
            try:
                ds.correct_flux_jumps(flux_quant)
            except Exception as e:
                self.set_chan_bad(ds.channum, "failed to correct flux jumps")

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
        self.register_boolean_cut_fields("filt_phase")
        for ds in self:
            ds.cuts.cut("filt_phase", np.abs(ds.p_filt_phase[:]) > 2)

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
        LOG.info("for all channels converting %s to energy with calibration %s", attr, calname)
        for ds in self:
            ds.convert_to_energy(attr, calname)

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
        LOG.info(rates_all.sum(axis=-1).shape)
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


class CrosstalkVeto(object):
    """An object to allow vetoing of data in 1 channel when another is hit."""

    def __init__(self, datagroup=None, window_ms=(-10, 3), pileup_limit=100):
        if datagroup is None:
            return

        window_ms = np.array(window_ms, dtype=np.int)
        self.window_ms = window_ms
        self.n_channels = datagroup.n_channels
        self.n_pulses = datagroup.nPulses

        ms0 = np.array([ds.p_timestamp[0] for ds in datagroup.datasets]).min() * 1e3 + window_ms[0]
        ms9 = np.array([ds.p_timestamp[-1] for ds in datagroup.datasets]).max() * 1e3 + window_ms[1]
        self.nhits = np.zeros(int(ms9 - ms0 + 1), dtype=np.int8)
        self.time0 = ms0

        for ds in datagroup.datasets:
            g = ds.cuts.good()
            vetotimes = np.asarray(ds.p_timestamp[g] * 1e3 - ms0, dtype=np.int64)
            vetotimes[vetotimes < 0] = 0
            a, b = window_ms
            b += 1
            for t in vetotimes:
                self.nhits[t + a:t + b] += 1

            pileuptimes = vetotimes[ds.p_postpeak_deriv[g] > pileup_limit]
            LOG.info("%s %d %f %d", vetotimes, len(vetotimes), 1.0e3 * ds.nPulses / (ms9 - ms0), len(pileuptimes))
            for t in pileuptimes:
                self.nhits[t + b:t + b + 8] += 1

    def copy(self):
        """Deep copy."""
        v = CrosstalkVeto()
        v.__dict__ = self.__dict__.copy()
        return v

    def veto(self, times_sec):
        """Return boolean vector for whether a given moment is vetoed.  Times are given in
        seconds.  Resolution is 1 ms for the veto."""
        index = np.asarray(times_sec * 1e3 - self.time0 + 0.5, dtype=np.int)
        return self.nhits[index] > 1
