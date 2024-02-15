"""
channel_group.py

Part of the Microcalorimeter Analysis Software System (MASS).

This module defines classes that handle one or more TES data streams together.
"""

import os
import logging
import re
from collections.abc import Iterable
from functools import reduce
from deprecated import deprecated

import numpy as np
import matplotlib.pylab as plt
import h5py

import mass.core.analysis_algorithms
import mass.calibration.energy_calibration

from mass.calibration.energy_calibration import EnergyCalibration
from mass.core.channel import MicrocalDataSet, PulseRecords, NoiseRecords, GroupLooper
from mass.core.cut import CutFieldMixin
from mass.core.utilities import InlineUpdater, show_progress, plot_multipage
from mass.core.ljh_util import remove_unpaired_channel_files, filename_glob_expand, ljh_get_extern_trig_fnames
from ..common import isstr

LOG = logging.getLogger("mass")


def _generate_hdf5_filename(rawname):
    """Generate the appropriate HDF5 filename based on a file's LJH name.

    Takes /path/to/data_chan33.ljh --> /path/to/data_mass.hdf5
    """
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
                    msg = f"""The implied HDF5 noise files names are not the same for all channels.
The first channel implies '{generated_noise_hdf5_name}'
and another implies '{_generate_hdf5_filename(fname)}'.
Instead, you should run RestoreTESGroup with an explicit hdf5noisename argument."""
                    raise RuntimeError(msg)
                noisefiles.append(fname)
        h5file.close()

    if hdf5noisename is not None:
        with h5py.File(hdf5noisename, "r") as h5file:
            for ch in channum:
                group = h5file[f'chan{ch}']
                noisefiles.append(group.attrs['filename'])
            h5file.close()
    else:
        hdf5noisename = generated_noise_hdf5_name

    return TESGroup(pulsefiles, noisefiles, hdf5_filename=hdf5filename,
                    hdf5_noisefilename=hdf5noisename)


class TESGroup(CutFieldMixin, GroupLooper):
    """The interface for a group of one or more microcalorimeters."""

    def __init__(self, filenames, noise_filenames=None, noise_only=False,
                 noise_is_continuous=True,
                 hdf5_filename=None, hdf5_noisefilename=None,
                 never_use=None, use_only=None, max_chans=None,
                 experimentStateFile=None, excludeStates="auto", overwrite_hdf5_file=False):
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
            max_chans: open at most this many ljh files
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
        pattern = filenames
        filenames = filename_glob_expand(filenames)
        if (filenames is None or len(filenames) == 0) and (not noise_only):
            raise ValueError(f"Pulse filename pattern {pattern} expanded to no files")
        if noise_filenames is not None:
            pattern = noise_filenames
            noise_filenames = filename_glob_expand(noise_filenames)
            if noise_filenames is None or len(noise_filenames) == 0:
                raise ValueError(f"Noise filename pattern {pattern} expanded to no files")

        # If using a glob pattern especially, we have to be careful to eliminate files that are
        # missing a partner, either noise without pulse or pulse without noise.
        remove_unpaired_channel_files(filenames, noise_filenames,
                                      never_use=never_use, use_only=use_only)

        # enforce max_chans
        if max_chans is not None:
            n = min(max_chans, len(filenames))
            filenames = filenames[:n]
            noise_filenames = noise_filenames[:n]

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
            if isstr(filenames):
                filenames = (filenames,)
            self.filenames = tuple(filenames)
            self.n_channels = len(self.filenames)
            if overwrite_hdf5_file:
                self.hdf5_file = h5py.File(hdf5_filename, 'w')
            else:
                self.hdf5_file = h5py.File(hdf5_filename, 'a')

        # Cut parameter description need to initialized.
        self.cut_field_desc_init()

        # Same for noise filenames
        self.noise_filenames = None
        self.hdf5_noisefile = None
        if noise_filenames is not None:
            if isstr(noise_filenames):
                noise_filenames = (noise_filenames,)
            self.noise_filenames = noise_filenames
            try:
                self.hdf5_noisefile = h5py.File(hdf5_noisefilename, 'a')
            except OSError:
                # if the noise file is corrupted, we will get an OSError
                # open with write intent, which will clobber the existing file
                self.hdf5_noisefile = h5py.File(hdf5_noisefilename, 'w')
            if noise_only:
                self.n_channels = len(self.noise_filenames)

        # Load up experiment state file
        self.experimentStateFile = None
        if not noise_only:
            if experimentStateFile is None:
                try:
                    self.experimentStateFile = mass.off.ExperimentStateFile(
                        datasetFilename=self.filenames[0], excludeStates=excludeStates)
                except OSError as e:
                    LOG.debug('Skipping loading of experiment state file because %s', e)
            else:
                self.experimentStateFile = mass.off.channels.ExperimentStateFile(
                    experimentStateFile, excludeStates=excludeStates)
            if self.experimentStateFile is not None:
                valid_state_labels = self.experimentStateFile.labels
                self.register_categorical_cut_field("state", valid_state_labels)

        # Set up other aspects of the object
        self.nhits = None
        self.n_segments = 0

        self.nPulses = 0
        self.nPresamples = 0
        self.nSamples = 0
        self.timebase = 0.0

        self._allowed_pnum_ranges = None
        self.pulses_per_seg = None
        self._bad_channums = {}
        self._external_trigger_subframe_count = None

        if self.noise_only:
            self._setup_per_channel_objects_noiseonly(noise_is_continuous)
        else:
            self._setup_per_channel_objects(noise_is_continuous)

        self.updater = InlineUpdater

    def toOffStyle(self):
        channels = [ds.toOffStyle() for ds in self]
        g = mass.off.channels.ChannelGroupFromNpArrays(channels,
                                                       shortname=self.shortname,
                                                       experimentStateFile=self.experimentStateFile)
        return g

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

            hdf5_group = self.hdf5_file.require_group(f"chan{pulse.channum}")
            hdf5_group.attrs['filename'] = fname

            dset = MicrocalDataSet(pulse.__dict__, tes_group=self, hdf5_group=hdf5_group)

            if 'calibration' in hdf5_group:
                hdf5_cal_grp = hdf5_group['calibration']
                for cal_name in hdf5_cal_grp:
                    dset.calibration[cal_name] = EnergyCalibration.load_from_hdf5(
                        hdf5_cal_grp, cal_name)

            if 'why_bad' in hdf5_group.attrs:
                self._bad_channums[dset.channum] = [comment.decode() for comment in
                                                    hdf5_group.attrs['why_bad']]

            # If appropriate, add to the MicrocalDataSet the NoiseRecords file interface
            if self.noise_filenames is not None:
                nf = self.noise_filenames[i]
                hdf5_group.attrs['noise_filename'] = nf
                try:
                    hdf5_noisegroup = self.hdf5_noisefile.require_group(f"chan{pulse.channum}")
                    hdf5_noisegroup.attrs['filename'] = nf
                except Exception:
                    hdf5_noisegroup = None
                noise = NoiseRecords(nf, records_are_continuous=noise_is_continuous)
                noise.set_hdf5_group(hdf5_noisegroup)

                if pulse.channum != noise.channum:
                    LOG.warning(
                        "WARNING: TESGroup did not add data: channums don't match %s, %s", fname, nf)
                    continue
                dset.noise_records = noise
                assert (dset.channum == dset.noise_records.channum)
                noise_list.append(noise)

            pulse_list.append(pulse)
            dset_list.append(dset)

            if self.n_segments == 0:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    self.__dict__[attr] = pulse.__dict__[attr]
            else:
                for attr in ("nSamples", "nPresamples", "timebase"):
                    if self.__dict__[attr] != pulse.__dict__[attr]:
                        msg = f"Unequal values of '{attr}': {float(self.__dict__[attr])} != {float(pulse.__dict__[attr])}"
                        raise ValueError(msg)
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
            ds.data = pr.datafile.alldata
            ds.times = pr.datafile.datatimes_float
            ds.subframecount = pr.datafile.subframecount
            ds.index = index

        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg

    def _setup_per_channel_objects_noiseonly(self, noise_is_continuous=True):
        noise_list = []
        dset_list = []
        for fname in self.noise_filenames:

            noise = NoiseRecords(fname, records_are_continuous=noise_is_continuous)
            hdf5_group = self.hdf5_noisefile.require_group(f"chan{noise.channum}")
            hdf5_group.attrs['filename'] = fname
            noise.set_hdf5_group(hdf5_group)

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
                        msg = f"Unequal values of '{attr}': {float(self.__dict__[attr])} != {float(noise.__dict__[attr])}"
                        raise ValueError(msg)
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
        clname = self.__class__.__name__
        if self.noise_filenames is None:
            pname = os.path.dirname(self.filenames[0])
            return f"{clname}(pulse={pname}, noise=None)"

        nname = os.path.dirname(self.noise_filenames[0])
        if self.noise_only:
            return f"{clname}(noise={nname}, noise_only=True)"
        pname = os.path.dirname(self.filenames[0])
        return f"{clname}(pulse={pname}, noise={nname})"

    def __iter__(self):
        """Iterator over the self.datasets in channel number order"""
        yield from self.iter_channels()

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
                del self.hdf5_file[f"chan{channum}"].attrs['why_bad']
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
        added_to_list = set.union(*[set(x) if isinstance(x, Iterable)
                                    else {x} for x in args if not isstr(x)])
        comment = reduce(lambda x, y: y, [x for x in args if isstr(x)], '')

        for channum in added_to_list:
            new_comment = self._bad_channums.get(channum, []) + [comment]
            self._bad_channums[channum] = new_comment
            LOG.warning('WARNING: Chan %s flagged bad because %s', channum, comment)
            self.hdf5_file[f"chan{channum}"].attrs['why_bad'] =  \
                np.asarray(new_comment, dtype=np.bytes_)

    def set_all_chan_good(self):
        """Set all channels to be good."""
        # Must do it this way (copying the list) so that you aren't iterating over a list
        # while also changing that list
        bad_chan_list = [ch for ch in self._bad_channums]
        for channum in bad_chan_list:
            self.set_chan_good(channum)

    def n_good_channels(self):
        return self.n_channels - len(self._bad_channums.keys())

    @property
    def timestamp_offset(self):
        ts = set([ds.timestamp_offset for ds in self if ds.channum not in self._bad_channums])
        if len(ts) == 1:
            return ts.pop()
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
        if self.num_good_channels <= 0:
            raise IndexError("WARNING: All datasets flagged bad, most things won't work.")
        return self.channel[self.good_channels[0]]

    @property
    def why_chan_bad(self):
        return self._bad_channums.copy()

    @deprecated(version="0.7.9", reason="Use compute_noise(), which is equivalent but better named")
    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        """Replaced by the equivalent compute_noise(...)"""
        # This is needed because the @_add_group_loop decorator does not preserve warnings
        # and hand them up.
        for ds in self:
            ds.compute_noise(max_excursion=max_excursion, n_lags=n_lags, forceNew=forceNew)

    def sample2segnum(self, samplenum):
        """Returns the segment number of sample number <samplenum>."""
        if samplenum >= self.nPulses:
            samplenum = self.nPulses - 1
        return samplenum // self.pulses_per_seg

    def segnum2sample_range(self, segnum):
        """Return the (first,end) sample numbers of the segment numbered <segnum>.
        Note that <end> is 1 beyond the last sample number in that segment."""
        return segnum * self.pulses_per_seg, (segnum + 1) * self.pulses_per_seg

    @property
    def shortname(self):
        """Return a string containing part of the filename and the number of good channels"""
        ngoodchan = len([ds for ds in self])
        return mass.ljh_util.ljh_basename_channum(os.path.split(self.datasets[0].filename)[-1])[0]+f", {ngoodchan} chans"

    @show_progress("summarize_data")
    def summarize_data(self, peak_time_microsec=None, pretrigger_ignore_microsec=None,
                       cut_pre=0, cut_post=0,
                       include_badchan=False, forceNew=False, use_cython=True, doPretrigFit=False):
        """Summarize the data with per-pulse summary quantities for each channel.

        peak_time_microsec will be determined automatically if None, and will be
        stored in channels as ds.peak_samplenumber.

        Args:
            use_cython uses a cython (aka faster) implementation of summarize.
        """
        nchan = float(len(self.channel.keys())) if include_badchan else float(
            self.num_good_channels)

        for i, ds in enumerate(self.iter_channels(include_badchan)):
            try:
                ds.summarize_data(peak_time_microsec=peak_time_microsec,
                                  pretrigger_ignore_microsec=pretrigger_ignore_microsec,
                                  cut_pre=cut_pre,
                                  cut_post=cut_post,
                                  forceNew=forceNew, use_cython=use_cython,
                                  doPretrigFit=doPretrigFit)
                yield (i + 1.0) / nchan
                self.hdf5_file.flush()
            except Exception as e:
                self.set_chan_bad(ds.channum, f"summarize_data failed with {e}")

    def compute_filters(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0, forceNew=False, category=None, filter_type="ats"):
        if category is None:
            category = {}
        LOG.warning(
            'compute_filters is deprecated and will eventually be removed, please '
            'use compute_ats_filter or compute_5lag_filter directly')
        for ds in self.datasets:
            if hasattr(ds, "_use_new_filters"):
                raise Exception(
                    "ds._use_new_filters is deprecated, use the filter_type argument to this function instead")
        if filter_type == "ats":
            self.compute_ats_filter(fmax=fmax, f_3db=f_3db, cut_pre=cut_pre,
                                    cut_post=cut_post, forceNew=forceNew, category=category)
        elif filter_type == "5lag":
            self.compute_5lag_filter(fmax=fmax, f_3db=f_3db, cut_pre=cut_pre,
                                     cut_post=cut_post, forceNew=forceNew, category=category)
        else:
            raise Exception("filter_type must be one of `ats` or `5lag`")

    def pulse_model_to_hdf5(self, hdf5_file=None, n_basis=6, replace_output=False,
                            maximum_n_pulses=4000, extra_n_basis_5lag=0, noise_weight_basis=True,
                            category=None, f_3db_5lag=None, _rethrow=False):
        if category is None:
            category = {}
        if hdf5_file is None:
            basename, _ = self.datasets[0].filename.split("chan")
            hdf5_filename = basename+"model.hdf5"
            if os.path.isfile(hdf5_filename):
                if not replace_output:
                    raise Exception(
                        f"file {hdf5_filename} already exists, pass replace_output = True to overwrite")
            with h5py.File(hdf5_filename, "w") as hdf5_file:
                self._pulse_model_to_hdf5(
                    hdf5_file, n_basis, pulses_for_svd=None,
                    extra_n_basis_5lag=extra_n_basis_5lag, maximum_n_pulses=maximum_n_pulses,
                    category=category, noise_weight_basis=noise_weight_basis, f_3db_5lag=f_3db_5lag, _rethrow=_rethrow)
                LOG.info("writing pulse_model to %s", hdf5_filename)
        else:
            hdf5_filename = hdf5_file.filename
            LOG.info("writing pulse_model to %s", hdf5_filename)
            self._pulse_model_to_hdf5(
                hdf5_file, n_basis, maximum_n_pulses=maximum_n_pulses,
                extra_n_basis_5lag=extra_n_basis_5lag, f_3db_5lag=f_3db_5lag, category=category)
        return hdf5_filename

    @property
    def external_trigger_subframe_count(self):
        if self._external_trigger_subframe_count is None:
            self.subframe_divisions = self.first_good_dataset.subframe_divisions
            possible_files = ljh_get_extern_trig_fnames(self.first_good_dataset.filename)
            if os.path.isfile(possible_files["hdf5"]):
                h5 = h5py.File(possible_files["hdf5"], "r")
                ds_name = "trig_times_w_offsets" if "trig_times_w_offsets" in h5 else "trig_times"
                self._external_trigger_subframe_count = h5[ds_name]
            elif os.path.isfile(possible_files["binary"]):
                with open(possible_files["binary"], "rb") as f:
                    header_text = f.readline().decode()
                    m = re.match(r".*\(nrow=(.*)\)\n", header_text)
                    if m is not None:
                        self.subframe_divisions = int(m.groups()[0])
                        for ds in self.datasets:
                            ds.subframe_divisions = self.subframe_divisions
                    self._external_trigger_subframe_count = np.fromfile(f, dtype="int64")
            else:
                raise OSError("No external trigger files found: ", possible_files)
            self.subframe_timebase = self.timebase/float(self.subframe_divisions)
            for ds in self.datasets:
                ds.subframe_timebase = self.subframe_timebase
        return self._external_trigger_subframe_count

    @property
    def external_trigger_subframe_as_seconds(self):
        """This is not a posix timestamp, it is just the external trigger subframecount converted to seconds
        based on the nominal clock rate of the crate.
        """
        return self.external_trigger_subframe_count[:]/float(self.subframe_divisions)*self.timebase

    def calc_external_trigger_timing(self, forceNew=False):
        ds = self.first_good_dataset
        external_trigger_subframe_count = np.asarray(ds.external_trigger_subframe_count[:], dtype=np.int64)

        for ds in self:
            try:
                if ("subframes_after_last_external_trigger" in ds.hdf5_group) and \
                    ("subframes_until_next_external_trigger" in ds.hdf5_group) and \
                    ("subframes_from_nearest_external_trigger" in ds.hdf5_group) and \
                    (not forceNew):
                        continue

                subframes_after_last, subframes_until_next = \
                    mass.core.analysis_algorithms.nearest_arrivals(ds.p_subframecount[:],
                                                                   external_trigger_subframe_count)
                nearest = np.fmin(subframes_after_last, subframes_until_next)
                for name, values in zip(
                    ("subframes_after_last_external_trigger",
                     "subframes_until_next_external_trigger",
                     "subframes_from_nearest_external_trigger"),
                    (subframes_after_last, subframes_until_next, nearest)):
                    h5dset = ds.hdf5_group.require_dataset(name, (ds.nPulses,), dtype=np.int64)
                    h5dset[:] = values
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
                datasets = list(self)
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
            if isstr(valid):
                if "uncut" in valid.lower():
                    valid_mask = ds.cuts.good()
                    LOG.info("Plotting only uncut data")
                elif "cut" in valid.lower():
                    valid_mask = ds.cuts.bad()
                    LOG.info("Plotting only cut data")
                elif 'all' in valid.lower():
                    valid_mask = None
                    LOG.info("Plotting all data, cut or uncut")
                else:
                    raise ValueError(
                        "If valid is a string, it must contain 'all', 'uncut' or 'cut'.")

            if valid_mask is not None:
                nrecs = valid_mask.sum()
                if downsample is None:
                    downsample = max(nrecs // 10000, 1)
                hour = ds.p_timestamp[valid_mask][::downsample] / 3600.0
            else:
                nrecs = ds.nPulses
                if downsample is None:
                    downsample = max(ds.nPulses // 10000, 1)
                hour = ds.p_timestamp[::downsample] / 3600.0
            LOG.info("Chan %3d (%d records; %d in scatter plots)", channum, nrecs, hour.shape[0])

            (vect, label, color, default_limits) = plottable
            if hist_limits is None:
                limits = default_limits
            else:
                limits = hist_limits

            # Vectors are being sampled and multiplied, so eval() is needed.
            vect = eval(f"ds.{vect}")[valid_mask]

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
            plt.ylabel(f"Ch {channum}")
            if i == ny_plots - 1:
                plt.xlabel("Time since last UT midnight (hours)")

            # Histograms on right half of figure
            if i == 0:
                axh_master = plt.subplot(ny_plots, 2, 2 + i * 2)
            elif 'Pretrig Mean' == label:
                plt.subplot(ny_plots, 2, 2 + i * 2)
            else:
                plt.subplot(ny_plots, 2, 2 + i * 2, sharex=axh_master)

            if limits is None:
                in_limit = np.ones(len(vect), dtype=bool)
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
        if nranges > 1:
            LOG.warning(
                "Warning: make_masks uses only one range argument.  Checking only '%s'.", vectname)

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
                raise ValueError(f"masks[{i}] is not a np.ndarray")

        for (mask, ds) in zip(masks, self.datasets):
            if ds.channum not in self.good_channels:
                continue
            ds.compute_average_pulse(mask, subtract_mean=subtract_mean, forceNew=forceNew)

    def plot_average_pulses(self, axis=None, channels=None, cmap=None, legend=True,
                            fcut=None, include_badchan=False):
        """Plot average pulse for channel number <channum> on matplotlib.Axes
        <axis>, or on a new Axes if <axis> is None. If <channum> is not a valid
        channel number, then plot all average pulses. If <fcut> is not None,
        then lowpass filter the traces with this cutoff frequency prior to
        plotting.
        """

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)

        if cmap is None:
            cmap = plt.cm.nipy_spectral

        dt = (np.arange(self.nSamples) - self.nPresamples) * self.timebase * 1e3

        if channels is None:
            dsets = list(self.iter_channels(include_badchan=include_badchan))
        else:
            dsets = [self.channel[c] for c in channels]
        nplot = len(dsets)

        for i, ds in enumerate(dsets):
            avg_pulse = ds.average_pulse[:].copy()
            if fcut is not None:
                avg_pulse = mass.core.analysis_algorithms.filter_signal_lowpass(
                    avg_pulse, 1./self.timebase, fcut)
            plt.plot(dt, avg_pulse, label=f"Chan {ds.channum}", color=cmap(float(i) / nplot))

        plt.title("Average pulse for each channel when it is hit")

        plt.xlabel("Time past trigger (ms)")
        plt.ylabel("Raw counts")
        plt.xlim([dt[0], dt[-1]])
        if legend:
            plt.legend(loc='best')
            if nplot > 12:
                ltext = axis.get_legend().get_texts()
                plt.setp(ltext, fontsize='small')

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
            cmap = plt.cm.nipy_spectral

        axis.grid(True)
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            if ds.filter is None:
                continue
            plt.plot(ds.filter.__dict__[filtname], label=f"Chan {ds.channum}",
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
                    rms = ds.hdf5_group[f'filters/filt_{filter_name}'].attrs['variance']**0.5
                v_dv = (1 / rms) / rms_fwhm
                LOG.info("Chan %3d filter %-15s Predicted V/dV %6.1f  Predicted res at %.1f eV: %6.1f eV",
                         ds.channum, filter_name, v_dv, std_energy, std_energy / v_dv)
            except Exception as e:
                LOG.warning("Filter %d can't be used", i)
                LOG.warning(e)

    def report(self):
        """Report on the number of data points and similar."""
        for ds in self.datasets:
            good = ds.cuts.good()
            ng = ds.cuts.good().sum()
            dt = (ds.p_timestamp[good][-1] * 1.0 - ds.p_timestamp[good][0])  # seconds
            npulse = np.arange(len(good))[good][-1] - good.argmax() + 1
            rate = (npulse - 1.0) / dt
            print(f'chan {ds.channum:3d} {npulse:6d} pulses ({rate:6.3f} Hz over {dt/3600.:6.4f} hr) {100.*ng/npulse:6.3f}% good')

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
            cmap = plt.cm.nipy_spectral

        axis.grid(True)
        for ds_num, channum in enumerate(channels):
            if channum not in self.channel:
                continue
            ds = self.channel[channum]
            noise = ds.noise_records
            noise.plot_autocorrelation(axis=axis, label=f'Chan {channum}',
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
        g = TESGroup(self.filenames, self.noise_filenames)
        g.__dict__.update(self.__dict__)
        g.datasets = tuple([d.copy() for d in self.datasets])
        return g

    def set_segment_size(self, seg_size):
        self.n_segments = 0
        for ds in self:
            ds.pulse_records.set_segment_size(seg_size)
            self.n_segments = max(self.n_segments, ds.pulse_records.pulses_per_seg)
        self.pulses_per_seg = self.first_good_dataset.pulse_records.pulses_per_seg
        for ds in self:
            assert ds.pulse_records.pulses_per_seg == self.pulses_per_seg

    def plot_noise(self, axis=None, channels=None, cmap=None, scale_factor=1.0,
                   sqrt_psd=False, legend=True, include_badchan=False):
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
            dsets = list(self.iter_channels(include_badchan=include_badchan))
        else:
            dsets = [self.channel[c] for c in channels]
        nplot = len(dsets)

        if cmap is None:
            cmap = plt.cm.nipy_spectral

        if scale_factor == 1.0:
            units = "Counts"
        else:
            units = "Scaled counts"

        axis.grid(True)
        for i, ds in enumerate(dsets):
            channum = ds.channum
            yvalue = ds.noise_psd[:] * scale_factor**2
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
                axis.set_ylabel(f"PSD$^{1/2}$ ({units}/Hz$^{1/2}$)")
            try:
                df = ds.noise_psd.attrs['delta_f']
                freq = np.arange(1, 1 + len(yvalue)) * df
                axis.plot(freq, yvalue, label=f'Chan {channum}',
                          color=cmap(float(i) / nplot))
            except Exception:
                LOG.warning("WARNING: Could not plot channel %4d.", channum)
                continue
        axis.set_xlim([freq[1] * 0.9, freq[-1] * 1.1])
        axis.set_ylabel(f"Power Spectral Density ({units}^2/Hz)")
        axis.set_xlabel("Frequency (Hz)")
        axis.loglog()
        if legend:
            plt.legend(loc='best')
            if nplot > 12:
                ltext = axis.get_legend().get_texts()
                plt.setp(ltext, fontsize='small')

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
            except Exception:
                self.set_chan_bad(ds.channum, "failed to correct flux jumps")

    def sanitize_p_filt_phase(self):
        self.register_boolean_cut_fields("filt_phase")
        for ds in self:
            ds.cuts.cut("filt_phase", np.abs(ds.p_filt_phase[:]) > 2)

    def plot_count_rate(self, bin_s=60):
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

    def plot_summary_pages(self, x_attr, y_attr, x_range=None, y_range=None, subplot_shape=(3, 4),
                           suffix=None, lines=None, down=10, fileformat='png', one_file=False):
        '''Make scatter plots of summary quantities for all channels.

        This creates the plots for each good channel, placing multiple plots on
        each page, and saves each page to its own file. Pulses that pass cuts are
        plotted in blue, and cut pulses are plotted in gray. The file names have
        the form "<x_attr>.vs.<y-attr>-<suffix>-<page number>.png". The default
        value for the suffix is that pulsefile's base name.

        Arguments:
          x_attr -- string containing name of X value attribute
          y_attr -- string containing name of Y value attribute
          x_range -- if not None, values to use for x limits. Defaults to None.
          y_range -- if not None, values to use for y limits. Defaults to None.
          subplot_shape -- tuple indicating shape of subplots. First element is
                           number of rows, second is number of columns.
          suffix -- suffix to use for filenames. Defaults to None, which causes the
                    function to use the first 15 characters of the pulse filename
                    for the first data set (which typically will have a value
                    like '20171017_103454')
          lines -- if not None, must contain a hashtable, keyed off of channel
                   number. The value for each channel is a list of numbers. A
                   dashed horizontal line is plotted for each value in this list.
                   Defaults to None.
          down -- downsample by this factor. Defaults to 10
          fileformat -- output format ('png', 'pdf', etc). Must be a value supported by
                    your installation of matplotlib.
          one_file -- If True, combine all pages to one pdf file. If False, use
                      separate files for all pages. Defaults to False. If format is
                      something other than 'pdf', this uses the ImageMagick program
                      `convert` to combine the files. You can install it on ubuntu
                      via `apt-get install imagemagick`.
        '''
        if suffix is None:
            suffix = os.path.basename(self.channels[0].datafile.filename)[:15]

        filename_template_per_file = f'{y_attr}.vs.{x_attr}-{suffix}-%%03d.{fileformat}'
        filename_template_glob = f'{y_attr}.vs.{x_attr}-{suffix}-[0-9][0-9][0-9].{fileformat}'
        filename_one_file = f'{y_attr}.vs.{x_attr}-{suffix}.pdf' % (y_attr, x_attr, suffix)

        def helper(ds, ax):
            ch = ds.channum
            g = ds.good()
            b = np.logical_not(g)

            x_g = getattr(ds, x_attr)[g][::down]
            x_b = getattr(ds, x_attr)[b][::down]
            y_g = getattr(ds, y_attr)[g][::down]
            y_b = getattr(ds, y_attr)[b][::down]

            if x_attr == 'p_timestamp':
                x_g = (x_g - getattr(ds, x_attr)[0]) / (60*60)
                x_b = (x_b - getattr(ds, x_attr)[0]) / (60*60)

            plt.plot(x_b, y_b, '.', markersize=2.5, color='gray')
            plt.plot(x_g, y_g, '.', markersize=2.5, color='blue')

            if lines is not None:
                x_lo = min(np.amin(x_g), np.amin(x_b))
                x_hi = max(np.amax(x_g), np.amax(x_b))
                for line in lines[ch]:
                    plt.plot([x_lo, x_hi], [line, line], '--k')

            if x_range is not None:
                plt.xlim(x_range)
            if y_range is not None:
                plt.ylim(y_range)

            if x_attr == 'p_timestamp':
                plt.xlabel('Time (hours)')
            else:
                plt.xlabel(x_attr, fontsize=8)
            plt.ylabel(y_attr, fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            plt.title('MATTER Ch%d' % ch, fontsize=10)

        plot_multipage(self, subplot_shape, helper, filename_template_per_file,
                       filename_template_glob, filename_one_file, format, one_file)

    def plot_histogram_pages(self, attr, valrange, bins, y_range=None, subplot_shape=(3, 4),
                             suffix=None, lines=None, fileformat='png', one_file=False):
        '''Make plots of histograms for all channels.

        This creates the plots for each good channel, placing multiple plots on
        each page, and saves each page to its own file. Only pulses that pass cuts
        are included. The file names have the form "<attr>-hist-<suffix>-<page
        number>.png". The default value for the suffix is that pulsefile's base
        name.

        Arguments:
          attr -- string containing name of attribute to plot
          valrange -- range of value over which to histogram (passed into histogram function)
          bins -- number of bins (passed into histogram function)
          y_range -- if not None, values to use for y limits. Defaults to None.
          subplot_shape -- tuple indicating shape of subplots. First element is
                           number of rows, second is number of columns.
          suffix -- suffix to use for filenames. Defaults to None, which causes the
                    function to use the first 15 characters of the pulse filename
                    for the first data set (which typically will have a value
                    like '20171017_103454')
          lines -- if not None, must contain a hashtable, keyed off of channel
                   number. The value for each channel is a list of numbers. A
                   dashed horizontal line is plotted for each value in this list.
                   Defaults to None.
          fileformat -- output format ('png', 'pdf', etc). Must be a value supported by
                    your installation of matplotlib.
          one_file -- If True, combine all pages to one pdf file. If False, use
                      separate files for all pages. Defaults to False. If format is
                      something other than 'pdf', this uses the ImageMagick program
                      `convert` to combine the files. You can install it on ubuntu
                      via `apt-get install imagemagick`.
        '''
        if suffix is None:
            suffix = os.path.basename(self.channels[0].datafile.filename)[:15]

        filename_template_per_file = f'{attr}-hist-{suffix}-%%03d.{fileformat}'
        filename_template_glob = f'{attr}-hist-{suffix}-[0-9][0-9][0-9].{fileformat}'
        filename_one_file = f'{attr}-hist-{suffix}.pdf'

        def helper(ds, ax):
            g = ds.good()
            x_g = getattr(ds, attr)[g]

            # I generally prefer the "stepped" histtype, but that seems to interact
            # poorly with log scale - the automatic choice of axis limits gets
            # screwed up.
            plt.hist(x_g, range=range, bins=bins, histtype='bar')
            plt.yscale('log')

            if lines is not None:
                x_lo = np.amin(x_g)
                x_hi = np.amax(x_g)
                for line in lines[ds.channum]:
                    plt.plot([x_lo, x_hi], [line, line], '-k')

            if y_range is not None:
                plt.ylim(y_range)

            plt.xlabel(attr, fontsize=8)
            plt.ylabel('Counts / bin', fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            plt.title(f'MATTER Ch{ds.channum}', fontsize=10)

        plot_multipage(self, subplot_shape, helper, filename_template_per_file,
                       filename_template_glob, filename_one_file, format, one_file)

    def hists(self, bin_edges, attr="p_energy", t0=0, tlast=1e20, category={}, g_func=None):
        """return a tuple of (bin_centers, countsdict). automatically filters out nan values
        where countsdict is a dictionary mapping channel numbers to numpy arrays of counts
        bin_edges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        t0 and tlast -- cuts all pulses outside this timerange before fitting
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer    """
        bin_edges = np.array(bin_edges)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        countsdict = {ds.channum: ds.hist(bin_edges, attr, t0, tlast, category, g_func)[1] for ds in self}
        return bin_centers, countsdict

    def hist(self, bin_edges, attr="p_energy", t0=0, tlast=1e20, category={}, g_func=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses in all good datasets
          (use .hists to get the histograms individually). filters out nan values
        bin_edges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        t0 and tlast -- cuts all pulses outside this timerange before fitting
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer
        """
        bin_centers, countsdict = self.hists(bin_edges, attr, t0, tlast, category, g_func)
        counts = np.zeros_like(bin_centers, dtype="int")
        for (k, v) in countsdict.items():
            counts += v
        return bin_centers, counts

    def linefit(self, line_name="MnKAlpha", t0=0, tlast=1e20, axis=None, dlo=50, dhi=50,
                binsize=1, bin_edges=None, attr="p_energy", label="full", plot=True,
                guess_params=None, ph_units="eV", category={}, g_func=None, has_tails=False):
        """Do a fit to `line_name` and return the fitter. You can get the params results with
        fitter.last_fit_params_dict or any other way you like.

        line_name -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
        t0 and tlast -- cuts all pulses outside this timerange before fitting
        axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
        dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo,
            fitter.spect.nominal_peak_energy+dhi, binsize)
        bin_edges -- pass the bin_edges you want as a numpy array
        attr -- default is "p_energy", you could pick "p_filt_value" or others. be sure to pass in bin_edges as well because
            the default calculation will probably fail for anything other than p_energy
        label -- passed to fitter.plot
        plot -- passed to fitter.fit, determine if plot happens
        guess_params -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
        ph_units -- passed to fitter.fit, used in plot label
        category -- pass {"side":"A"} or similar to use categorical cuts
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
            This vector is anded with the vector calculated by the histogrammer
        this should be the same as ds.linefit, but for now I've just copied and pasted the code
        """
        assert "energy" in attr
        model = mass.getmodel(line_name, has_tails=has_tails)
        nominal_peak_energy = model.spect.nominal_peak_energy
        if bin_edges is None:
            bin_edges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)

        bin_centers, counts = self.hist(bin_edges, attr, t0, tlast, category, g_func)

        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(vary=False)
        result = model.fit(counts, params=params, bin_centers=bin_centers)
        if plot:
            result.plotm(ax=axis, xlabel=f"{attr} ({ph_units})",
                         ylabel=f"counts per {binsize:0.2f} ({ph_units}) bin",
                         title=f"{self.shortname}\n{model.spect}")

        return result


class CrosstalkVeto:
    """An object to allow vetoing of data in 1 channel when another is hit."""

    def __init__(self, datagroup=None, window_ms=(-10, 3), pileup_limit=100):
        if datagroup is None:
            return

        window_ms = np.array(window_ms, dtype=int)
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
            LOG.info("%s %d %f %d", vetotimes, len(vetotimes), 1.0e3
                     * ds.nPulses / (ms9 - ms0), len(pileuptimes))
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
        index = np.asarray(times_sec * 1e3 - self.time0 + 0.5, dtype=int)
        return self.nhits[index] > 1
