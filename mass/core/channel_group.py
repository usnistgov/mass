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
import numpy as np
import matplotlib.pylab as plt
import os
import h5py

import mass.core.analysis_algorithms
import mass.calibration.energy_calibration
import mass.nonstandard.CDM

from mass.core.channel import MicrocalDataSet, PulseRecords, NoiseRecords
from mass.core.cython_channel import CythonMicrocalDataSet
from mass.core.optimal_filtering import Filter
from mass.core.utilities import InlineUpdater


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


class TESGroup(object):
    """
    Provides the interface for a group of one or more microcalorimeters,
    multiplexed by TDM.
    """

    BRIGHT_ORANGE = '#ff7700'

    BUILTIN_BOOLEAN_CUT_FIELDS = ['pretrigger_rms',
                                  'pretrigger_mean',
                                  'pretrigger_mean_departure_from_median',
                                  'peak_time_ms',
                                  'rise_time_ms',
                                  'postpeak_deriv',
                                  'pulse_average',
                                  'min_value',
                                  'timestamp_sec',
                                  'timestamp_diff_sec',
                                  'peak_value',
                                  'energy',
                                  'timing',
                                  "p_filt_phase",
                                  'smart_cuts']

    # Categorical cut field item format
    # [name of field, list of categories, default category]
    BUILTIN_CATEGORICAL_CUT_FIELDS = [
        ['calibration', ['in', 'out'], 'in'],
    ]

    __cut_boolean_field_desc_dtype = np.dtype([("name", np.bytes_, 64),
                                               ("mask", np.uint32)])
    __cut_categorical_field_desc_dtype = np.dtype([("name", np.bytes_, 64),
                                                   ("pos", np.uint8),
                                                   ("mask", np.uint32)])
    __cut_category_list_dtype = np.dtype([("field", np.bytes_, 64),
                                          ("category", np.bytes_, 64),
                                          ("index", np.uint8)])

    def __init__(self, filenames, noise_filenames=None, noise_only=False,
                 noise_is_continuous=True, max_cachesize=None,
                 hdf5_filename=None, hdf5_noisefilename=None):

        if noise_filenames is not None and len(noise_filenames) == 0:
            noise_filenames = None

        # In the noise_only case, you can put the noise file names either in the
        # usual (pulse) filenames argument or in the noise_filenames argument.
        self.noise_only = noise_only
        if noise_only and noise_filenames is None:
            filenames, noise_filenames = (), filenames

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
        if self.hdf5_file:
            if "cut_used_bit_flags" not in self.hdf5_file.attrs:
                # This condition is for the backward-compatibility.
                if "cut_num_used_bits" in self.hdf5_file.attrs:
                    cut_num_used_bits = np.uint32(self.hdf5_file.attrs["cut_num_used_bits"])
                else:
                    cut_num_used_bits = np.uint32(0)
                self.hdf5_file.attrs["cut_used_bit_flags"] = np.uint32((np.uint64(1) << cut_num_used_bits) - 1)

            if "cut_boolean_field_desc" not in self.hdf5_file.attrs:
                self.hdf5_file.attrs["cut_boolean_field_desc"] = np.zeros(32, dtype=self.__cut_boolean_field_desc_dtype)
                self.register_boolean_cut_fields(*self.BUILTIN_BOOLEAN_CUT_FIELDS)

            if ("cut_categorical_field_desc" not in self.hdf5_file.attrs) and \
                    ("cut_category_list" not in self.hdf5_file):
                self.hdf5_file.attrs["cut_categorical_field_desc"] = \
                    np.zeros(0, dtype=self.__cut_categorical_field_desc_dtype)
                self.hdf5_file.attrs["cut_category_list"] =\
                    np.zeros(0, dtype=self.__cut_category_list_dtype)

                for categorical_desc in self.BUILTIN_CATEGORICAL_CUT_FIELDS:
                    self.register_categorical_cut_field(*categorical_desc)

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

        self.num_good_channels = 0
        self.good_channels = []
        self.first_good_dataset = None

        if self.noise_only:
            self._setup_per_channel_objects_noiseonly(noise_is_continuous)
        else:
            self._setup_per_channel_objects(noise_is_continuous)

        if max_cachesize is not None:
            if max_cachesize < self.n_channels * self.channels[0].segmentsize:
                self.set_segment_size(max_cachesize // self.n_channels)

    @property
    def boolean_cut_desc(self):
        return self.hdf5_file.attrs["cut_boolean_field_desc"]

    @boolean_cut_desc.setter
    def boolean_cut_desc(self, value):
        self.hdf5_file.attrs["cut_boolean_field_desc"] = value

    @property
    def categorical_cut_desc(self):
        return self.hdf5_file.attrs["cut_categorical_field_desc"]

    @categorical_cut_desc.setter
    def categorical_cut_desc(self, value):
        self.hdf5_file.attrs["cut_categorical_field_desc"] = value

    @property
    def cut_category_list(self):
        return self.hdf5_file.attrs["cut_category_list"]

    @cut_category_list.setter
    def cut_category_list(self, value):
        self.hdf5_file.attrs["cut_category_list"] = value

    @property
    def cut_num_used_bits(self):
        return self.hdf5_file.attrs["cut_num_used_bits"]

    @cut_num_used_bits.setter
    def cut_num_used_bits(self, value):
        self.hdf5_file.attrs["cut_num_used_bits"] = value

    @property
    def cut_used_bit_flags(self):
        return self.hdf5_file.attrs["cut_used_bit_flags"]

    @cut_used_bit_flags.setter
    def cut_used_bit_flags(self, value):
        self.hdf5_file.attrs["cut_used_bit_flags"] = np.uint32(value)

    def cut_field_categories(self, field_name):
        category_list = self.cut_category_list

        return {name.decode(): index for field, name, index in category_list if field == field_name.encode()}

    @staticmethod
    def __lowest_available_cut_bit(cut_used_bit_flags):
        mask = np.uint32(1)

        for i in range(32):
            trial_bit = mask << np.uint32(i)
            if cut_used_bit_flags & trial_bit == 0:
                return np.uint8(i)

        raise Exception("No available cut bit.")

    def register_boolean_cut_fields(self, *names):
        boolean_fields = self.boolean_cut_desc
        cut_used_bit_flags = self.cut_used_bit_flags

        new_fields = [n.encode() for n in names if n.encode() not in boolean_fields["name"]]

        for new_field in new_fields:
            available_bit = self.__lowest_available_cut_bit(cut_used_bit_flags)
            boolean_fields[available_bit] = (new_field, np.uint32(1) << available_bit)
            cut_used_bit_flags |= (np.uint32(1) << available_bit)

        self.boolean_cut_desc = boolean_fields
        self.cut_used_bit_flags = cut_used_bit_flags

    def unregister_boolean_cut_fields(self, *names):
        boolean_fields = self.boolean_cut_desc

        enc_names = [name.encode() for name in names]

        for name in enc_names:
            if not name or name not in boolean_fields['name']:
                raise ValueError("{0:s} is not a registered boolean field.".format(name))

        clear_mask = np.uint32(0)

        for i in range(32):
            if boolean_fields[i][0] in enc_names:
                clear_mask |= boolean_fields[i][1]
                boolean_fields[i] = (b'', 0)

        self.boolean_cut_desc = boolean_fields
        self.cut_used_bit_flags &= ~clear_mask

    def register_categorical_cut_field(self, name, categories, default="uncategorized"):
        categorical_fields = self.categorical_cut_desc
        cut_used_bit_flags = self.cut_used_bit_flags

        if name.encode() in categorical_fields["name"]:
            return

        # categories might be an immutable tuple.
        category_list = list(categories)

        # if the default category is already included, it's temporarily removed from the category_list
        # and insert into at the head of the category_list.
        if default in category_list:
            category_list.remove(default)
        category_list.insert(0, default)

        # Updates the 'cut_category_list' attribute
        new_list = np.array([(name.encode(), category.encode(), i) for i, category in enumerate(category_list)],
                            dtype=self.__cut_category_list_dtype)
        self.cut_category_list = np.hstack([self.cut_category_list,
                                            new_list])

        # Needs to update the 'cut_categorical_field_desc' attribute.

        num_bits = 1
        while (1 << num_bits) < len(category_list):
            num_bits += 1

        mask_pos = self.__lowest_available_cut_bit(cut_used_bit_flags)
        bit_mask = np.uint32(0)
        for _ in range(num_bits):
            bit_mask |= (np.uint32(1) << self.__lowest_available_cut_bit(cut_used_bit_flags | bit_mask))

        field_desc_item = np.array([(name.encode(),
                                     mask_pos,
                                     bit_mask)],
                                   dtype=self.__cut_categorical_field_desc_dtype)
        self.categorical_cut_desc = np.hstack([categorical_fields, field_desc_item])
        self.cut_used_bit_flags |= bit_mask

    def unregister_categorical_cut_field(self, name):
        categorical_fields = self.categorical_cut_desc
        category_list = self.cut_category_list
        cut_used_bit_flags = self.cut_used_bit_flags

        new_categorical_fields = categorical_fields[categorical_fields['name'] != name.encode()]
        new_category_list = category_list[category_list['field'] != name.encode()]

        clear_mask = categorical_fields['mask'][categorical_fields['name'] == name.encode()][0]

        self.categorical_cut_desc = new_categorical_fields
        self.cut_category_list = new_category_list
        self.cut_used_bit_flags &= ~clear_mask

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

            try:
                hdf5_group = self.hdf5_file.require_group("chan%d" % pulse.channum)
                hdf5_group.attrs['filename'] = fname
            except:
                hdf5_group = None

            dset = CythonMicrocalDataSet(pulse.__dict__, tes_group=self, hdf5_group=hdf5_group)

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
            self.hdf5_file.attrs['npulses'] = self.nPulses
            self.hdf5_file.attrs['nsamples'] = self.nSamples
            self.hdf5_file.attrs['npresamples'] = self.nPresamples
            self.hdf5_file.attrs['frametime'] = self.timebase

        self.channels = tuple(pulse_list)
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)

        for chan, ds in zip(self.channels, self.datasets):
            ds.pulse_records = chan

        self._setup_channels_list()

        if len(pulse_list) > 0:
            self.pulses_per_seg = pulse_list[0].pulses_per_seg
        if len(self.datasets) > 0:
            # Set master timestamp_offset (seconds)
            self.timestamp_offset = self.first_good_dataset.timestamp_offset

        for ds in self:
            if ds.timestamp_offset != self.timestamp_offset:
                self.timestamp_offset = None
                break

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
            self.hdf5_file.attrs['npulses'] = self.nPulses
            self.hdf5_file.attrs['nsamples'] = self.nSamples
            self.hdf5_file.attrs['npresamples'] = self.nPresamples
            self.hdf5_file.attrs['frametime'] = self.timebase

        self.channels = ()
        self.noise_channels = tuple(noise_list)
        self.datasets = tuple(dset_list)
        for chan, ds in zip(self.channels, self.datasets):
            ds.pulse_records = chan
        self._setup_channels_list()
        if len(self.datasets) > 0:
            self.timestamp_offset = self.first_good_dataset.timestamp_offset

        for ds in self:
            if ds.timestamp_offset != self.timestamp_offset:
                self.timestamp_offset = None
                break

    def __iter__(self):
        """Iterator over the self.datasets in channel number order"""
        for ds in self.iter_channels():
            yield ds

    def iter_channels(self, include_badchan=False):
        """Iterator over the self.datasets in channel number order
        include_badchan : whether to include officially bad channels in the result."""
        channum = self.channel.keys()
        if not include_badchan:
            channum = list(set(channum) - set(self._bad_channums.keys()))
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
                goodones = {a}
            added_to_list.update(goodones)
        for k in added_to_list:
            if k in self._bad_channums:
                comment = self._bad_channums.pop(k)
                print("chan %d set good, had previously been set bad for %s" % (k, str(comment)))
            else:
                print("chan %d not set good because it was not set bad" % k)
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
                badones = {a}
            added_to_list.update(badones)

        for k in added_to_list:
            self._bad_channums[k] = self._bad_channums.get(k, []) + [comment]
            print('chan %s flagged bad because %s' % (k, comment))

        self.update_chan_info()

    def update_chan_info(self):
        channum = self.channel.keys()
        channum = list(set(channum) - set(self._bad_channums.keys()))
        channum.sort()
        self.num_good_channels = len(channum)
        self.good_channels = list(channum)
        if self.num_good_channels > 0:
            self.first_good_dataset = self.channel[channum[0]]
        elif len(channum) > 0:
            print("WARNING: All datasets flagged bad, most things won't work.")
            self.first_good_dataset = None

    def _setup_channels_list(self):
        self.channel = {}
        for ds_num, ds in enumerate(self.datasets):
            try:
                ds.index = ds_num
                self.channel[ds.channum] = ds
            except AttributeError:
                pass
        self.update_chan_info()

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

    def summarize_data(self, peak_time_microsec=220.0, pretrigger_ignore_microsec=20.0,
                       include_badchan=False, forceNew=False, use_cython=True):
        """
        Compute summary quantities for each pulse.
        We are (July 2014) developing a Julia replacement for this, but you can use Python
        if you wish.
        """
        printUpdater = InlineUpdater('summarize_data')
        if include_badchan:
            nchan = float(len(self.channel.keys()))
        else:
            nchan = float(self.num_good_channels)

        for i, chan in enumerate(self.iter_channel_numbers(include_badchan)):
            try:
                self.channel[chan].summarize_data(peak_time_microsec,
                                                  pretrigger_ignore_microsec, forceNew, use_cython=use_cython)
                printUpdater.update((i + 1) / nchan)
                self.hdf5_file.flush()
            except:
                self.set_chan_bad(chan, "summarize_data")

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
                        #ds._rows_after_last_external_trigger = g
                    if until_next:
                        g = ds.hdf5_group.require_dataset("rows_until_next_external_trigger",
                                                          (ds.nPulses,), dtype=np.int64)
                        g[:] = rows_until_next_external_trigger
                        #ds._rows_until_next_external_trigger = g
                    if from_nearest:
                        g = ds.hdf5_group.require_dataset("rows_from_nearest_external_trigger",
                                                          (ds.nPulses,), dtype=np.int64)
                        g[:] = np.fmin(rows_after_last_external_trigger, rows_until_next_external_trigger)
                        #ds._rows_from_nearest_external_trigger = g
            except Exception:
                self.set_chan_bad(ds.channum, "calc_external_trigger_timing")

    def read_trace(self, record_num, dataset_num=0, chan_num=None):
        """Read (from cache or disk) and return the pulse numbered <record_num> for
        dataset number <dataset_num> or channel number <chan_num>.
        If both are given, then <chan_num> will be used when valid.
        If this is a CDMGroup, then the pulse is the demodulated
        channel by that number."""
        ds = self.channel.get(chan_num, self.datasets[dataset_num])
        return ds.read_trace(record_num)

    def plot_traces(self, pulsenums, dataset_num=0, chan_num=None, pulse_summary=True, axis=None,
                    difference=False, residual=False, valid_status=None, channum=None, shift1=False):
        """Plot some example pulses, given by sample number.
        <pulsenums>   A sequence of sample numbers, or a single one.
        <dataset_num> Dataset index (0 to n_dets-1, inclusive).  Will be used only if
                      <chan_num> is invalid.
        <chan_num>    Dataset channel number.  If valid, it will be used instead of dataset_num.

        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A plt axis to plot on.
        <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
        <residual>   Whether to show the residual between data and opt filtered model,
                     or just raw data.
        <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                     that have been cut.  If "cut", show only those that have been cut.
        <channum>    Synonym for chan_num (an unfortunate but old choice)
        <shift1>     Whether to take pulses with p_shift1==True and delay them by 1 sample
        """

        if chan_num is None:
            chan_num = channum
        if chan_num in self.channel:
            dataset = self.channel[chan_num]
            dataset_num = dataset.index
        else:
            dataset = self.datasets[dataset_num]
            if chan_num is not None:
                print("Cannot find chan_num[%d], so using dataset #%d" % (chan_num, dataset_num))
        return dataset.plot_traces(pulsenums, pulse_summary, axis, difference,
                                   residual, valid_status, shift1)

    def plot_summaries(self, quantity, valid='uncut', downsample=None, log=False, hist_limits=None,
                       dataset_numbers=None):
        """Plot a summary of one quantity from the data set, including time series and histograms of
        this quantity.  This method plots all channels in the group, but only one quantity.  If you
        would rather see all quantities for one channel, then use the group's
        group.dataset[i].plot_summaries() method.

        <quantity> A case-insensitive whitespace-ignored one of the following list, or the numbers
                   that go with it:
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
        <dataset_numbers>  A sequence of the datasets [0...n_channels-1] to plot.  If None
                           (the default) then plot all datasets in numerical order.
        """

        plottables = (
            ("p_pulse_average", 'Pulse Avg', 'purple', [0, 5000]),
            ("p_pretrig_rms", 'Pretrig RMS', 'blue', [0, 4000]),
            ("p_pretrig_mean", 'Pretrig Mean', 'green', None),
            ("p_peak_value", 'Peak value', '#88cc00', None),
            ("p_postpeak_deriv", 'Max PT deriv', 'gold', [0, 700]),
            ("p_rise_time[:]*1e3", 'Rise time (ms)', 'orange', [0, 12]),
            ("p_peak_time[:]*1e3", 'Peak time (ms)', 'red', [-3, 9])
        )

        quant_names = [p[1].lower().replace(" ", "") for p in plottables]
        if quantity in range(len(quant_names)):
            plottable = plottables[quantity]
        else:
            i = quant_names.index(quantity.lower().replace(" ", ""))
            plottable = plottables[i]

        if dataset_numbers is None:
            datasets = self.datasets
            dataset_numbers = range(len(datasets))
        else:
            datasets = [self.datasets[i] for i in dataset_numbers]

        plt.clf()
        ny_plots = len(datasets)
        for i, (channum, ds) in enumerate(zip(dataset_numbers, datasets)):
            print('TES%2d ' % channum),

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
            print(" (%d records; %d in scatter plots)" % (nrecs, hour.shape[0]))

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
                plt.plot(hour, vect[::downsample], '.', ms=1, color=color)
            else:
                plt.text(.5, .5, 'empty', ha='center', va='center', size='large',
                         transform=plt.gca().transAxes)
            if i == 0:
                plt.title(label)
            plt.ylabel("TES %d" % channum)
            if i == ny_plots - 1:
                plt.xlabel("Time since server start (hours)")

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
                         has p_postpeak_deriv exceeding <max_post_deriv>
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

        # Cut crosstalk only makes sense in CDM data
        if cut_crosstalk and not isinstance(self, mass.nonstandard.CDM.CDMGroup):
            print('Cannot cut crosstalk because this is not CDM data')
            cut_crosstalk = False

        if not cut_crosstalk:
            if max_ptrms is not None:
                print("Warning: make_masks ignores max_ptrms when not cut_crosstalk")
            if max_post_deriv is not None:
                print("Warning: make_masks ignores max_post_deriv when not cut_crosstalk")

        if pulse_avg_ranges is not None:
            if pulse_peak_ranges is not None:
                print("Warning: make_masks uses only one range argument.  Ignoring pulse_peak_ranges.")

            if isinstance(pulse_avg_ranges[0], (int, float)) and len(pulse_avg_ranges) == 2:
                pulse_avg_ranges = tuple(pulse_avg_ranges),

            for r in pulse_avg_ranges:
                middle = 0.5 * (r[0] + r[1])
                abslim = 0.5 * np.abs(r[1] - r[0])
                for gain, dataset in zip(gains, self.datasets):
                    m = np.abs(dataset.p_pulse_average[:] / gain - middle) <= abslim
                    if cut_crosstalk:
                        m = np.logical_and(m, self.nhits == 1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds == dataset:
                                    continue
                                m = np.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds == dataset:
                                    continue
                                m = np.logical_and(m, ds.p_postpeak_deriv < max_post_deriv)
                    m = np.logical_and(m, dataset.cuts.good())
                    masks.append(m)

        elif pulse_peak_ranges is not None:
            if isinstance(pulse_peak_ranges[0], (int, float)) and len(pulse_peak_ranges) == 2:
                pulse_peak_ranges = tuple(pulse_peak_ranges),
            for r in pulse_peak_ranges:
                middle = 0.5 * (r[0] + r[1])
                abslim = 0.5 * np.abs(r[1] - r[0])
                for gain, dataset in zip(gains, self.datasets):
                    m = np.abs(dataset.p_peak_value[:] / gain - middle) <= abslim
                    if cut_crosstalk:
                        m = np.logical_and(m, self.nhits == 1)
                        if max_ptrms is not None:
                            for ds in self.datasets:
                                if ds == dataset:
                                    continue
                                m = np.logical_and(m, ds.p_pretrig_rms < max_ptrms)
                        if max_post_deriv is not None:
                            for ds in self.datasets:
                                if ds == dataset:
                                    continue
                                m = np.logical_and(m, ds.p_postpeak_deriv < max_post_deriv)
                    m = np.logical_and(m, dataset.cuts.good())
                    masks.append(m)
        else:
            raise ValueError("Call make_masks with only one of pulse_avg_ranges"
                             " and pulse_peak_ranges specified.")

        return masks

    def compute_average_pulse(self, masks, subtract_mean=True, forceNew=False):
        """
        Compute several average pulses in each TES channel, one per mask given in
        <masks>.  Store the averages in self.datasets.average_pulse with shape (m,n)
        where m is the number of masks and n equals self.nPulses (the # of records).

        Note that this method replaces any previously computed self.datasets.average_pulse

        <masks> is either an array of shape (m,n) or an array (or other sequence) of length
        (m*n).  It's required that n equal self.nPulses.   In the second case,
        m must be an integer.  The elements of <masks> should be booleans or interpretable
        as booleans.

        If <subtract_mean> is True, then each average pulse will subtract a constant
        to ensure that the pretrigger mean (first self.nPresamples elements) is zero.
        """

        # Don't proceed if not necessary and not forced
        already_done = all([ds.average_pulse[-1] != 0 for ds in self])
        if already_done and not forceNew:
            print("skipping compute average pulse")
            return

        # Make sure that masks is either a 2D or 1D array of the right shape,
        # or a sequence of 1D arrays of the right shape
        if isinstance(masks, np.ndarray):
            nd = masks.ndim
            if nd == 1:
                n = len(masks)
                masks = masks.reshape((n // self.nPulses, self.nPulses))
            elif nd > 2:
                raise ValueError("masks argument should be a 2D array or a sequence of 1D arrays")
            nbins = masks.shape[0]
        else:
            nbins = len(masks)

        for i, m in enumerate(masks):
            if not isinstance(m, np.ndarray):
                raise ValueError("masks[%d] is not a np.ndarray" % i)

        pulse_counts = np.zeros((self.n_channels, nbins))
        pulse_sums = np.zeros((self.n_channels, nbins, self.nSamples), dtype=np.float)

        # Compute a master mask to say whether ANY mask wants a pulse from each segment
        # This can speed up work a lot when the pulses being averaged are from certain times only.
        segment_mask = np.zeros(self.n_segments, dtype=np.bool)
        for m in masks:
            n = len(m)
            nseg = 1 + (n - 1) // self.pulses_per_seg
            for i in range(nseg):
                if segment_mask[i]:
                    continue
                a, b = self.segnum2sample_range(i)
                if m[a:b].any():
                    segment_mask[i] = True
            a, b = self.segnum2sample_range(nseg + 1)
            if a < n and m[a:].any():
                segment_mask[nseg + 1] = True

        printUpdater = InlineUpdater('compute_average_pulse')
        for first, end in self.iter_segments(segment_mask=segment_mask):
            printUpdater.update(end / float(self.nPulses))
            for imask, mask in enumerate(masks):
                valid = mask[first:end]
                for ichan, chan in enumerate(self):  # loop over only valid datasets
                    if chan.channum not in self.why_chan_bad:
                        if (imask % self.n_channels) != ichan:
                            continue

                        if mask.shape != (chan.nPulses,):
                            raise ValueError("\nmasks[%d] has shape %s, but it needs to be (%d,)" %
                                             (imask, mask.shape, chan.nPulses))
                        if len(valid) > chan.data.shape[0]:
                            good_pulses = chan.data[valid[:chan.data.shape[0]], :]
                        else:
                            good_pulses = chan.data[valid, :]
                        pulse_counts[ichan, imask] += good_pulses.shape[0]
                        pulse_sums[ichan, imask, :] += good_pulses.sum(axis=0)

        # Rescale and store result to each MicrocalDataSet
        pulse_sums /= pulse_counts.reshape((self.n_channels, nbins, 1))
        for ichan, ds in enumerate(self.datasets):
            average_pulses = pulse_sums[ichan, :, :]
            if subtract_mean:
                for imask in range(average_pulses.shape[0]):
                    average_pulses[imask, :] -= np.mean(average_pulses[imask,
                                                                       :self.nPresamples - ds.pretrigger_ignore_samples])
            ds.average_pulse[:] = average_pulses[ichan, :]

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

    def compute_filters(self, fmax=None, f_3db=None, forceNew=False):

        # Analyze the noise, if not already done
        needs_noise = any([ds.noise_autocorr[0] == 0.0 or
                           ds.noise_psd[1] == 0 for ds in self])
        if needs_noise:
            print("Computing noise autocorrelation and spectrum")
            self.compute_noise_spectra()

        print_updater = InlineUpdater('compute_filters')
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
                print_updater.update((ds_num + 1) / float(self.n_channels))

                # Store all filters created to a new HDF5 group
                h5grp = ds.hdf5_group.require_group('filters')
                if f.f_3db is not None:
                    h5grp.attrs['f_3db'] = f.f_3db
                if f.fmax is not None:
                    h5grp.attrs['fmax'] = f.fmax
                h5grp.attrs['peak'] = f.peak_signal
                h5grp.attrs['shorten'] = f.shorten
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

    def filter_data(self, filter_name='filt_noconst', transform=None, include_badchan=False, forceNew=False, use_cython=True):
        printUpdater = InlineUpdater('filter_data')
        if include_badchan:
            nchan = float(len(self.channel.keys()))
        else:
            nchan = float(self.num_good_channels)

        for i, chan in enumerate(self.iter_channel_numbers(include_badchan)):
            self.channel[chan].filter_data(filter_name, transform, forceNew, use_cython=use_cython)

            printUpdater.update((i + 1) / nchan)

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
#            grate = (ng-1.0)/dt
            print('chan %2d %6d pulses (%6.3f Hz over %6.4f hr) %6.3f%% good' %
                  (ds.channum, npulse, rate, dt / 3600., 100.0 * ng / npulse))

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
        for i, ds in enumerate(self.datasets):
            if i not in channels:
                continue
            noise = ds.noise_records
            noise.plot_autocorrelation(axis=axis, label='TES %d' % i,
                                       color=cmap(float(i) / self.n_channels))
#        axis.set_xlim([f[1]*0.9,f[-1]*1.1])
        axis.set_xlabel("Time lag (ms)")
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

    def plot_noise(self, axis=None, channels=None, scale_factor=1.0, sqrt_psd=False, cmap=None):
        """Compare the noise power spectra.

        <channels>    Sequence of channels to display.  If None, then show all.
        <scale_factor> Multiply counts by this number to get physical units.
        <sqrt_psd>     Whether to show the sqrt(PSD) or (by default) the PSD itself.
        <cmap>         A matplotlib color map.  Defaults to something.
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
        for ds_num,channum in enumerate(channels):
            if channum not in self.channel: continue
            ds = self.channel[channum]
            yvalue = ds.noise_records.noise_psd[:] * scale_factor**2
            if sqrt_psd:
                yvalue = np.sqrt(yvalue)
                axis.set_ylabel("PSD$^{1/2}$ (%s/Hz$^{1/2}$)" % units)
            df = ds.noise_records.noise_psd.attrs['delta_f']
            freq = np.arange(1, 1 + len(yvalue)) * df
            axis.plot(freq, yvalue, label='TES chan %d' % channum,
                      color=cmap(float(ds_num) / len(channels)))
        axis.set_xlim([freq[1] * 0.9, freq[-1] * 1.1])
        axis.set_ylabel("Power Spectral Density (%s^2/Hz)" % units)
        axis.set_xlabel("Frequency (Hz)")

        axis.loglog()
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
        median_pulse_avg = np.array([np.median(ds.p_pulse_average[ds.good()]) for ds in self])
        masks = self.make_masks([.95, 1.05], use_gains=True, gains=median_pulse_avg)
        for m in masks:
            if len(m) > max_pulses_to_use:
                m[max_pulses_to_use:] = False
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
                self.set_chan_bad(ds.channum, "failed phase_correct with %s"%e)

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
                  bin_size_ev=2, category=None, forceNew=False, maxacc=0.015, nextra=3):
        for ds in self:
            try:
                ds.calibrate(attr, line_names, name_ext, size_related_to_energy_resolution,
                             fit_range_ev, excl, plot_on_fail,
                             bin_size_ev, category, forceNew, maxacc, nextra)
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
        channum = int(name.split('_chan')[1].split(".")[0])
        if inclusion_list is not None and channum not in inclusion_list:
            continue
        print(channum, name)
        chan2fname[channum] = name
    sorted_chan = chan2fname.keys()
    sorted_chan.sort()
    sorted_fnames = [chan2fname[key] for key in sorted_chan]
    return sorted_fnames


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
