"""
Single-channel classes:

* NoiseRecords: encapsulate a file with noise records
* PulseRecords: encapsulate a file with pulse records
* MicrocalDataSet: encapsulate basically everything about 1 channel's pulses and noise
"""

import h5py
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pylab as plt
import inspect

# MASS modules
import mass.mathstat.power_spectrum
import mass.mathstat.interpolate
import mass.mathstat.robust
import mass.core.analysis_algorithms

from mass.core.cut import Cuts
from mass.core.files import VirtualFile, LJHFile
from mass.core.optimal_filtering import Filter, ArrivalTimeSafeFilter
from mass.core.utilities import show_progress
from mass.calibration.energy_calibration import EnergyCalibration
from mass.calibration.algorithms import EnergyCalibrationAutocal
from mass.mathstat.entropy import laplace_entropy

from mass.core import ljh_util
import logging
LOG = logging.getLogger("mass")


class NoiseRecords(object):
    """Encapsulate a set of noise records.

    The noise records can either be assumed continuous or arbitrarily separated in time.
    """

    DEFAULT_MAXSEGMENTSIZE = 32000000
    ALLOWED_TYPES = ("ljh", "virtual")

    def __init__(self, filename, records_are_continuous=False, use_records=None,
                 maxsegmentsize=None, hdf5_group=None):
        """Contain and analyze a noise records file.

        Args:
            filename: name of the noise data file
            records_are_continuous: whether to treat all pulses as a continuous
                timestream (default False)
            use_records: (default None), or can be a sequence (first,end) to use
                only a limited section of the file from record first to the one
                before end.
            maxsegmentsize: the number of bytes to be read at once in a segment
                (default self.DEFAULT_MAXSEGMENTSIZE)
            hdf5_group: the HDF5 group to be associated with this noise (default None)
        """
        self.hdf5_group = hdf5_group

        if maxsegmentsize is not None:
            self.maxsegmentsize = maxsegmentsize
        else:
            self.maxsegmentsize = self.DEFAULT_MAXSEGMENTSIZE

        self.channum = None
        self.nSamples = self.nPresamples = self.nPulses = 0
        self.n_segments = 0
        self.timebase = 0.0

        self.datafile = None
        self.data = None

        self.__open_file(filename, use_records=use_records)
        self.continuous = records_are_continuous
        self.noise_psd = None
        if self.hdf5_group is not None:
            self.autocorrelation = self.hdf5_group.require_dataset(
                "autocorrelation", shape=(self.nSamples,),
                dtype=np.float64)
            nfreq = 1 + self.nSamples // 2
            self.noise_psd = self.hdf5_group.require_dataset(
                'noise_psd', shape=(nfreq,),
                dtype=np.float64)

    def __open_file(self, filename, use_records=None, file_format=None):
        """Detect the filetype and open it."""

        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in self.ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s" % ",".join(self.ALLOWED_TYPES))

        if file_format == "ljh":
            self.datafile = LJHFile(filename, segmentsize=self.maxsegmentsize)
        elif file_format == "virtual":
            vfile = filename  # Aha!  It must not be a string
            self.datafile = vfile
            self.datafile.segmentsize = vfile.nPulses*(6+2*vfile.nSamples)
            filename = 'Virtual file'
        else:
            raise RuntimeError("It is a programming error to get here")
        self.filename = filename
        self.records_per_segment = self.datafile.segmentsize // (6+2*self.datafile.nSamples)

        if use_records is not None:
            if use_records < self.datafile.nPulses:
                self.datafile.nPulses = use_records
                self.datafile.n_segments = use_records // self.records_per_segment

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum", "n_segments"):
            self.__dict__[attr] = self.datafile.__dict__[attr]
            if self.hdf5_group is not None:
                self.hdf5_group.attrs[attr] = self.datafile.__dict__[attr]

    def clear_cache(self):
        self.datafile.clear_cache()

    def set_fake_data(self):
        """Use when this does not correspond to a real datafile."""
        self.datafile = VirtualFile(np.zeros((0, 0)))

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = NoiseRecords(self.filename)
        c.__dict__.update(self.__dict__)
        c.datafile = self.datafile.copy()
        return c

    def compute_power_spectrum(self, window=mass.mathstat.power_spectrum.hann, plot=True,
                               max_excursion=1000):
        self.compute_power_spectrum_reshape(window=window, seg_length=None,
                                            max_excursion=max_excursion)
        if plot:
            self.plot_power_spectrum()

    def compute_power_spectrum_reshape(self, window=mass.mathstat.power_spectrum.hann,
                                       seg_length=None, max_excursion=1000):
        """Compute the noise power spectrum with noise "records" reparsed into
        separate records of a given length.

        Args:
            window (ndarray): a window function weighting, or a function that will
                return a weighting.
            seg_length (int): length of the noise segments (default None). If None,
                then use self.data.shape[0], which is self.data.nPulses, will be
                used as the number of segments, each having length self.data.nSamples.
            max_excursion (number): the biggest excursion from the median allowed
                in each data segment, or else it will be ignored (default 1000).

        By making <seg_length> small, you improve the noise on the PSD estimates at the price of poor
        frequency resolution.  By making it large, you get good frequency resolution with worse
        uncertainty on each PSD estimate.  No free lunch, know what I mean?
        """

        if not self.continuous and seg_length is not None:
            raise ValueError("This NoiseRecords doesn't have continuous noise; it can't be resegmented.")

        if seg_length is None:
            seg_length = self.nSamples

        spectrum = mass.mathstat.power_spectrum.PowerSpectrum(seg_length // 2, dt=self.timebase)
        if window is None:
            window = np.ones(seg_length)
        else:
            window = window(seg_length)

        for _first_pnum, _end_pnum, _seg_num, data in self.datafile.iter_segments():
            if self.continuous and seg_length is not None:
                data = data.ravel()
                n = len(data)

                # Would it be a problem if n % seg_length is non-zero?
                n -= n % seg_length
                data = data[:n].reshape((n // seg_length, seg_length))

            for d in data:
                y = d-d.mean()
                if y.max() - y.min() < max_excursion and len(y) == spectrum.m2:
                    spectrum.addDataSegment(y, window=window)

        freq = spectrum.frequencies()
        psd = spectrum.spectrum()
        if self.hdf5_group is not None:
            self.noise_psd[:] = psd
            self.noise_psd.attrs['delta_f'] = freq[1] - freq[0]
        else:
            self.noise_psd = psd
        return spectrum

    def compute_fancy_power_spectra(self, window=mass.mathstat.power_spectrum.hann,
                                    plot=True, seglength_choices=None):
        """Compute a power spectrum using a few long segments for the low freq.
        and many short ones for the higher frequencies.
        """
        assert self.continuous

        # Does it assume that all data fit into a single segment?
        self.datafile.read_segment(0)
        n = np.prod(self.datafile.data.shape)
        if seglength_choices is None:
            longest_seg = 1
            while longest_seg <= n // 16:
                longest_seg *= 2
            seglength_choices = [longest_seg]
            while seglength_choices[-1] > 256:
                seglength_choices.append(seglength_choices[-1] // 4)
            LOG.debug("Will use segments of length: %s", seglength_choices)

        spectra = [self.compute_power_spectrum_reshape(window=window, seg_length=seglen)
                   for seglen in seglength_choices]
        if plot:
            plt.clf()
            lowest_freq = np.array([1./(s.dt * s.m2) for s in spectra])

            start_freq = 0.0
            for i, s in enumerate(spectra):
                x, y = s.frequencies(), s.spectrum()
                if i == len(spectra)-1:
                    good = x >= start_freq
                else:
                    good = np.logical_and(x >= start_freq, x < 10*lowest_freq[i+1])
                plt.loglog(x[good], y[good], '-')
                start_freq = lowest_freq[i] * 10
        return spectra

    def plot_power_spectrum(self, axis=None, scale=1.0, sqrt_psd=False, **kwarg):
        """Plot the power spectrum of this noise record.

        Args:
            <axis>     Which plt.Axes object to plot on.  If none, clear the figure and plot there.
            <scale>    Scale all raw units by this number to convert counts to physical
            <sqrt_psd> Whether to take the sqrt(PSD) for plotting.  Default is no sqrt
        """
        if all(self.noise_psd[:] == 0):
            self.compute_power_spectrum(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        yvalue = self.noise_psd[1:] * (scale**2)
        if sqrt_psd:
            yvalue = np.sqrt(yvalue)
        freq = np.arange(1, 1 + len(yvalue))*self.noise_psd.attrs['delta_f']
        axis.plot(freq, yvalue, **kwarg)
        plt.loglog()
        axis.grid()
        axis.set_xlim([10, 3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel(r"Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title("Noise power spectrum for %s" % self.filename)

    def _compute_continuous_autocorrelation(self, n_lags=None, data_samples=None,
                                            max_excursion=1000):
        if data_samples is None:
            data_samples = [0, self.nSamples*self.nPulses]
        n_data = data_samples[1] - data_samples[0]

        samples_per_segment = self.records_per_segment*self.nSamples
        if n_lags is None:
            n_lags = samples_per_segment
        if n_lags > samples_per_segment:
            n_lags = samples_per_segment

        def padded_length(n):
            """Return a sensible number in the range [n, 2n] which is not too
            much larger than n, yet is good for FFTs.

            Returns:
                A number: (1, 3, or 5)*(a power of two), whichever is smallest.
            """
            pow2 = np.round(2**np.ceil(np.log2(n)))
            if n == pow2:
                return int(n)
            elif n > 0.75*pow2:
                return int(pow2)
            elif n > 0.625*pow2:
                return int(np.round(0.75*pow2))
            else:
                return int(np.round(0.625*pow2))

        # When there are 10 million data points and only 10,000 lags wanted,
        # it's hugely inefficient to compute the full autocorrelation, especially
        # in memory.  Instead, compute it on chunks several times the length of the desired
        # correlation, and average.
        CHUNK_MULTIPLE = 15
        if n_data >= (1 + CHUNK_MULTIPLE) * n_lags:
            # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
            # unwanted wraparound in the autocorrelation.
            # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
            chunksize = CHUNK_MULTIPLE * n_lags
            padsize = n_lags
            padded_data = np.zeros(padded_length(padsize+chunksize), dtype=np.float)

            ac = np.zeros(n_lags, dtype=np.float)

            entries = 0.0

            for first_pnum, end_pnum, _seg_num, data in self.datafile.iter_segments():
                data_consumed = 0
                data = data.ravel()
                samples_this_segment = len(data)
                if data_samples[0] > self.nSamples*first_pnum:
                    data_consumed = data_samples[0]-self.nSamples*first_pnum
                if data_samples[1] < self.nSamples*end_pnum:
                    samples_this_segment = data_samples[1]-self.nSamples*first_pnum

                #
                # umux data will "slip" flux quanta when pulses with a very
                # fast leading edge occur. When this happens the baseline level
                # that is returned to after a pulse may be an integer number of
                # flux quanta different from the baseline level before the
                # pulse. If this happens to noise data within a segment, the
                # excursions algorithm will needlessly reject the entire segment.
                #
                # Ideally we would recognize and "correct" these flux jumps.
                # But for now, I am calculating `data_mean` at the chunk level
                # instead of the segment level. This means that a jump will
                # cause data for just that chunk to be thrown away, instead of
                # for the entire segment containing the chunk.
                #

                # Notice that the following loop might ignore the last data values, up to as many
                # as (chunksize-1) values, unless the data are an exact multiple of chunksize.
                while data_consumed+chunksize <= samples_this_segment:
                    data_mean = data[data_consumed:data_consumed+chunksize].mean()
                    padded_data[:chunksize] = data[data_consumed:data_consumed+chunksize] - data_mean
                    data_consumed += chunksize
                    padded_data[chunksize:] = 0.0
                    if np.abs(padded_data).max() > max_excursion:
                        continue

                    ft = np.fft.rfft(padded_data)
                    ft[0] = 0  # this redundantly removes the mean of the data set
                    power = (ft*ft.conj()).real
                    acsum = np.fft.irfft(power)
                    ac += acsum[:n_lags]
                    entries += 1.0
                    if entries*chunksize > n_data:
                        break

            if entries == 0:
                raise Exception("Apparently all chunks had excusions, so no autocorrelation was computed")

            ac /= entries
            ac /= (np.arange(chunksize, chunksize-n_lags+0.5, -1.0, dtype=np.float))

        # compute the full autocorrelation
        else:
            raise NotImplementedError("Now that Joe has chunkified the noise, we can "
                                      "no longer compute full continuous autocorrelations")

        self.autocorrelation[:] = ac

    def compute_autocorrelation(self, n_lags=None, data_samples=None, plot=True, max_excursion=1000):
        """Compute the autocorrelation averaged across all "pulses" in the file.

        Args:
            <n_lags>
            <data_samples> If not None, then a range [a,b] to use.
        """

        if self.continuous:
            self._compute_continuous_autocorrelation(n_lags=n_lags, data_samples=data_samples,
                                                     max_excursion=max_excursion)

        else:
            if n_lags is not None and n_lags > self.nSamples:
                raise ValueError("The autocorrelation requires "
                                 "n_lags<=%d when data are not continuous" % self.nSamples)

            class TooMuchData(StopIteration):
                """Use to signal that the computation loop is done"""
                pass

            if data_samples is None:
                data_samples = [0, self.nSamples*self.nPulses]
            n_data = data_samples[1] - data_samples[0]

            records_used = samples_used = 0
            ac = np.zeros(self.nSamples, dtype=np.float)
            try:
                for first_pnum, end_pnum, _seg_num, intdata in self.datafile.iter_segments():
                    if end_pnum <= data_samples[0]:
                        continue
                    if first_pnum >= data_samples[1]:
                        break
                    for i in range(first_pnum, end_pnum):
                        if i < data_samples[0]:
                            continue
                        if i >= data_samples[1]:
                            break

                        data = 1.0*(intdata[i-first_pnum, :])
                        if data.max() - data.min() > max_excursion:
                            continue
                        data -= data.mean()

                        ac += np.correlate(data, data, 'full')[self.nSamples-1:]
                        samples_used += self.nSamples
                        records_used += 1
                        if n_data is not None and samples_used >= n_data:
                            raise TooMuchData()
            except TooMuchData:
                pass

            ac /= records_used
            ac /= self.nSamples - np.arange(self.nSamples, dtype=np.float)
            if n_lags is not None and n_lags < self.nSamples:
                ac = ac[:n_lags]
            self.autocorrelation[:] = ac

        if self.hdf5_group is not None:
            grp = self.hdf5_group.require_group("reclen%d" % n_lags)
            ds = grp.require_dataset("autocorrelation", shape=(n_lags,), dtype=np.float64)
            ds[:] = self.autocorrelation[:]

        if plot:
            self.plot_autocorrelation()

    def plot_autocorrelation(self, axis=None, color='blue', label=None):
        """Plot the autocorrelation function."""
        if all(self.autocorrelation[:] == 0):
            LOG.info("Autocorrelation must be computed before it can be plotted")
            self.compute_autocorrelation(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        t = self.timebase * 1e3 * np.arange(len(self.autocorrelation))
        axis.plot(t, self.autocorrelation, label=label, color=color)
        axis.plot([0], [self.autocorrelation[0]], 'o', color=color)
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel(r"Autocorrelation (counts$^2$)")


class PulseRecords(object):
    """
    Encapsulate a set of data containing multiple triggered pulse traces.
    The pulses should not be noise records.

    This object will not contain derived facts such as pulse summaries, filtered values,
    and so forth. It is meant to be only a file interface.
    """

    ALLOWED_TYPES = ("ljh", "virtual")

    def __init__(self, filename, file_format=None):
        """Contain and analyze a noise records file.

        Args:
            filename: name of the pulse data file
        """
        self.nSamples = 0
        self.nPresamples = 0
        self.nPulses = 0
        self.channum = 0
        self.n_segments = 0
        self.segmentsize = 0
        self.pulses_per_seg = 0
        self.timebase = None
        self.timestamp_offset = None

        self.datafile = None
        self.__open_file(filename, file_format=file_format)

        self.data = np.array([], ndmin=2)
        self.times = np.array([], ndmin=2)
        self.rowcount = None

    def __open_file(self, filename, file_format=None):
        """Detect the filetype and open it."""

        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in self.ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s" % ",".join(self.ALLOWED_TYPES))

        if file_format == "ljh":
            self.datafile = LJHFile(filename)
        elif file_format == "virtual":
            vfile = filename  # Aha!  It must not be a string
            self.datafile = vfile
        else:
            raise RuntimeError("It is a programming error to get here")

        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum",
                     "n_segments", "pulses_per_seg", "segmentsize", "timestamp_offset"):
            self.__dict__[attr] = self.datafile.__dict__[attr]

    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time" % (
            self.__class__.__name__, self.filename, self.nSamples, self.nPresamples,
            1e6*self.timebase)

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.filename)

    def set_segment_size(self, seg_size):
        """Update the underlying file's segment size in bytes."""
        self.datafile.set_segment_size(seg_size)
        self.n_segments = self.datafile.n_segments
        self.pulses_per_seg = self.datafile.pulses_per_seg
        self.segmentsize = self.datafile.segmentsize

    def read_segment(self, segment_num):
        """Read the requested segment of the raw data file and return  (first,end)
        meaning: the first record number, 1 more than the last record number,
        and the nPulse x nSamples array."""
        if segment_num >= self.n_segments:
            return -1, -1
        first_pnum, end_pnum, data = self.datafile.read_segment(segment_num)

        self.data = data
        self.rowcount = self.datafile.rowcount
        try:
            self.times = self.datafile.datatimes_float
        except AttributeError:
            self.times = self.datafile.datatimes/1e3
        return first_pnum, end_pnum

    def clear_cache(self):
        self.data = None
        self.rowcount = None
        self.times = None
        self.datafile.clear_cached_segment()

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = PulseRecords(self.filename)
        c.__dict__.update(self.__dict__)
        c.datafile = self.datafile.copy()
        return c


class GroupLooper(object):
    """A mixin class to allow TESGroup objects to hold methods that loop over
    their constituent channels. (Has to be a mixin, in order to break the import
    cycle that would otherwise occur.)"""
    pass

def _add_group_loop(method):
    """Add MicrocalDataSet method `method` to GroupLooper (and hence, to TESGroup).

    This is a decorator to add before method definitions inside class MicrocalDataSet.
    Usage is:

    class MicrocalDataSet(...):
        ...

        @_add_group_loop
        def awesome_fuction(self, ...):
            ...
    """

    method_name = method.__name__
    # print "Adding method named '%s'"%method_name

    def wrapper(self, *args, **kwargs):
        for ds in self:
            try:
                method(ds, *args, **kwargs)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                self.set_chan_bad(ds.channum, "failed %s with %s" % (method_name, e))

    wrapper.__name__ = method_name

    # Generate a good doc-string.
    lines = ["Loop over self, calling the %s(...) method for each channel."%method_name]
    arginfo = inspect.getargspec(method)
    argtext = inspect.formatargspec(*arginfo)
    if method.__doc__ is None:
        lines.append("\n%s%s has no docstring"%(method_name, argtext))
    else:
        lines.append("\n%s%s docstring reads:"%(method_name, argtext))
        lines.append( method.__doc__)
    wrapper.__doc__ = "\n".join(lines)

    setattr(GroupLooper, method_name, wrapper)
    return method


class MicrocalDataSet(object):
    """Represent a single microcalorimeter's PROCESSED data."""

    # Attributes that all such objects must have.
    expected_attributes = ("nSamples", "nPresamples", "nPulses", "timebase", "channum",
                           "timestamp_offset")
    HDF5_CHUNK_SIZE = 256

    def __init__(self, pulserec_dict, tes_group=None, hdf5_group=None):
        """
        Args:
            pulserec_dict: a dictionary (presumably that of a PulseRecords object)
                containing the expected attributes that must be copied to this
                MicrocalDataSet.
            tes_group: the parent TESGroup object of which this is a member
                (default None).
            hdf5_group: the HDF5 group in which the relevant per-pulse data are
                cached. You really want this to exist, for reasons of both performance
                and data backup. (default None)
        """
        self.nSamples = 0
        self.nPresamples = 0
        self.nPulses = 0
        self.timebase = 0.0
        self.channum = None
        self.timestamp_offset = 0

        self.filter = None
        self.lastUsedFilterHash = -1
        self.drift_correct_info = {}
        self.phase_correct_info = {}
        self.time_drift_correct_info = {}
        self.noise_autocorr = None
        self.noise_demodulated = None
        self.calibration = {}

        for a in self.expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename', 'virtual data set')
        self.pretrigger_ignore_samples = 0  # Cut this long before trigger in computing pretrig values
        self.cut_pre = 0 # Number of presamples to ignore at start of pulse
        self.cut_post = 0  # Number of samples to ignore at end of pulse
        self.peak_samplenumber = None   # Look for retriggers only after this time.
        self.index = None   # Index in the larger TESGroup object
        self.last_used_calibration = None

        self.data = None
        self.times = None
        self.rowcount = None

        self.number_of_rows = None
        self.row_number = None
        self.number_of_columns = None
        self.column_number = None

        self._external_trigger_rowcount = None
        self._use_new_filters = True

        self.row_timebase = None

        self.nearest_neighbors_dictionary = {}

        self.tes_group = tes_group

        try:
            self.hdf5_group = hdf5_group
            self.hdf5_group.attrs['npulses'] = self.nPulses
            self.hdf5_group.attrs['channum'] = self.channum
        except KeyError:
            self.hdf5_group = None

        self.__setup_vectors(npulses=self.nPulses)
        self.__load_cals_from_hdf5()
        self.__load_auto_cuts()

    def __setup_vectors(self, npulses=None):
        """Given the number of pulses, build arrays to hold the relevant facts
        about each pulse in memory.

        These will include:
        * p_filt_value = pulse height after running through filter
        * p_fil_value_dc = pulse height after running through filter and applying drift correction
        * p_filt_value_phc = phase corrected pulse height
        * p_energy = pulse energy determined from applying a calibration to one of the p_filt_value??? variables
        * ...and many others.
        """

        if npulses is None:
            assert self.nPulses > 0
            npulses = self.nPulses

        h5grp = self.hdf5_group

        # Set up the per-pulse vectors
        float64_fields = ('timestamp',)
        float32_fields = ('pretrig_mean', 'pretrig_rms', 'pulse_average', 'pulse_rms',
                          'promptness', 'rise_time', 'postpeak_deriv',
                          'pretrig_deriv', 'pretrig_offset',
                          'filt_phase', 'filt_phase_corr', 'filt_value', 'filt_value_dc',
                          'filt_value_phc', 'filt_value_tdc',
                          'energy')
        uint16_fields = ('peak_index', 'peak_value', 'min_value')
        int64_fields = ('rowcount',)
        bool_fields = ('shift1',)
        for dtype, fieldnames in ((np.float64, float64_fields),
                                  (np.float32, float32_fields),
                                  (np.uint16, uint16_fields),
                                  (np.bool, bool_fields),
                                  (np.int64, int64_fields)):
            for field in fieldnames:
                self.__dict__['p_%s' % field] = h5grp.require_dataset(field, shape=(npulses,),
                                                                      dtype=dtype)
        if "peak_samplenumber" in self.p_peak_index.attrs:
            self.peak_samplenumber = self.p_peak_index.attrs["peak_samplenumber"]

        # Other vectors needed per-channel
        self.average_pulse = h5grp.require_dataset('average_pulse', shape=(self.nSamples,),
                                                   dtype=np.float32)
        self.noise_autocorr = h5grp.require_dataset('noise_autocorr', shape=(self.nSamples,),
                                                    dtype=np.float64)
        nfreq = 1 + self.nSamples // 2
        self.noise_psd = h5grp.require_dataset('noise_psd', shape=(nfreq,),
                                               dtype=np.float64)

        if 'filters' in h5grp:
            filter_group = h5grp['filters']

            fmax = filter_group.attrs['fmax'] if 'fmax' in filter_group.attrs else None
            f_3db = filter_group.attrs['f_3db'] if 'f_3db' in filter_group.attrs else None
            shorten = filter_group.attrs['shorten'] if 'shorten' in filter_group.attrs else None
            if "newfilter" in filter_group.attrs:
                newfilter = filter_group.attrs["newfilter"]
            else:
                newfilter = "filt_aterms" in filter_group.keys()
            self._use_new_filters = newfilter

            if newfilter:
                aterms = filter_group["filt_aterms"][:]
                model = np.vstack([self.average_pulse[1:], aterms]).T
                modelpeak = np.max(self.average_pulse)
                self.filter = ArrivalTimeSafeFilter(model,
                                                    self.nPresamples - self.pretrigger_ignore_samples,
                                                    self.noise_autocorr,
                                                    sample_time=self.timebase,
                                                    peak=modelpeak)
            else:
                self.filter = Filter(self.average_pulse[...],
                                     self.nPresamples - self.pretrigger_ignore_samples,
                                     self.noise_psd[...],
                                     self.noise_autocorr, sample_time=self.timebase,
                                     shorten=shorten)
            self.filter.fmax = fmax
            self.filter.f_3db = f_3db

            for k in ["filt_fourier", "filt_fourier_full", "filt_noconst",
                      "filt_baseline", "filt_baseline_pretrig", "filt_aterms"]:
                if k in filter_group:
                    filter_ds = filter_group[k]
                    setattr(self.filter, k, filter_ds[...])
                    if 'variance' in filter_ds.attrs:
                        self.filter.variances[k.split("filt_")[1]] = filter_ds.attrs['variance']
                    if 'predicted_v_over_dv' in filter_ds.attrs:
                        self.filter.predicted_v_over_dv[k.split("filt_")[1]] = filter_ds.attrs['predicted_v_over_dv']

        grp = self.hdf5_group.require_group('cuts')
        self.cuts = Cuts(self.nPulses, self.tes_group, hdf5_group=grp)

    def __load_cals_from_hdf5(self, overwrite=False):
        """Load all calibrations in self.hdf5_group["calibration"] into the dict
        self.calibration.
        """
        hdf5_cal_group = self.hdf5_group.require_group('calibration')
        for k in hdf5_cal_group.keys():
            if not overwrite:
                if k in self.calibration.keys():
                    raise ValueError("trying to load over existing calibration, consider passing overwrite=True")
            self.calibration[k] = EnergyCalibration.load_from_hdf5(hdf5_cal_group, k)

    @property
    def p_peak_time(self):
        # this is a property to reduce memory usage, I hope it works
        return (self.p_peak_index[:] - self.nPresamples) * self.timebase

    @property
    def external_trigger_rowcount(self):
        if not self._external_trigger_rowcount:
            filename = ljh_util.ljh_get_extern_trig_fname(self.filename)
            h5 = h5py.File(filename, "r")
            ds_name = "trig_times_w_offsets" if "trig_times_w_offsets" in h5 else "trig_times"
            self._external_trigger_rowcount = h5[ds_name]
            self.row_timebase = self.timebase/float(self.number_of_rows)
        return self._external_trigger_rowcount

    @property
    def external_trigger_rowcount_as_seconds(self):
        """This is not a posix timestamp, it is just the external trigger rowcount converted to seconds
        based on the nominal clock rate of the crate.
        """
        return self.external_trigger_rowcount[:]*self.timebase/float(self.number_of_rows)

    @property
    def rows_after_last_external_trigger(self):
        try:
            return self.hdf5_group["rows_after_last_external_trigger"]
        except KeyError:
            raise ValueError("run tes_group.calc_external_trigger_timing with after_last=True before accessing this")

    @property
    def rows_until_next_external_trigger(self):
        try:
            return self.hdf5_group["rows_until_next_external_trigger"]
        except KeyError:
            raise ValueError("run tes_group.calc_external_trigger_timing with until_next=True before accessing this")

    @property
    def rows_from_nearest_external_trigger(self):
        try:
            return self.hdf5_group["rows_from_nearest_external_trigger"]
        except KeyError:
            raise ValueError("run tes_group.calc_external_trigger_timing with from_nearest=True before accessing this")

    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time" % (
            self.__class__.__name__, self.filename, self.nSamples, self.nPresamples,
            1e6*self.timebase)

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.filename)

    def updater(self, name):
        return self.tes_group.updater(name + " chan {0:d}".format(self.channum))

    def good(self, *args, **kwargs):
        """Returns a boolean vector, one per pulse record, saying whether record is good"""
        return self.cuts.good(*args, **kwargs)

    def bad(self, *args, **kwargs):
        """Returns a boolean vector, one per pulse record, saying whether record is bad"""
        return self.cuts.bad(*args, **kwargs)

    def resize(self, nPulses):
        if self.nPulses < nPulses:
            raise ValueError("Can only shrink using resize(), but the requested size %d is larger than current %d" %
                             (nPulses, self.nPulses))
        self.nPulses = nPulses
        self.__setup_vectors()

    def copy(self):
        """Return a deep copy of the object."""
        c = MicrocalDataSet(self.__dict__)
        c.__dict__.update(self.__dict__)
        for k in self.calibration.keys():
            c.calibration[k] = self.calibration[k].copy()
        c.cuts = self.cuts.copy()
        return c

    def _compute_peak_samplenumber(self):
        if self.data is None:
            self.read_segment(0)
        peak_idx = self.data[:,self.cut_pre:self.nSamples-self.cut_post].argmax(axis=1)+self.cut_pre
        self.peak_samplenumber = int(sp.stats.mode(peak_idx)[0][0])
        self.p_peak_index.attrs["peak_samplenumber"] = self.peak_samplenumber
        return self.peak_samplenumber
    
    @show_progress("channel.summarize_data")
    def summarize_data(self, peak_time_microsec=None, pretrigger_ignore_microsec=None,
                       cut_pre = 0, cut_post = 0,
                       forceNew=False, use_cython=True,
                       doPretrigFit = False):
        """Summarize the complete data set one chunk at a time.

        Store results in the HDF5 datasets p_pretrig_mean and similar.

        Args:
            peak_time_microsec: the time in microseconds at which this channel's
                pulses typically peak (default None). You should leave this as None,
                and let the value be estimated from the data.
            pretrigger_ignore_microsec: how much time before the trigger to ignore
                when computing pretrigger mean (default None). If None, it will
                be chosen sensibly.
            cut_pre: Cut this many samples from the start of a pulse record when calculating summary values
            cut_post: Cut this many samples from the end of the a record when calculating summary values              
            forceNew: whether to re-compute summaries if they exist (default False)
            use_cython: whether to use cython for summarizing the data (default True).
                If this object is not a CythonMicrocalDataSet, then Cython cannot
                be used, and this value is ignored.
            doPretrigFit: whether to do a linear fit of the pretrigger data
        """
        # Don't proceed if not necessary and not forced
        self.number_of_rows = self.pulse_records.datafile.number_of_rows
        self.row_number = self.pulse_records.datafile.row_number
        self.number_of_columns = self.pulse_records.datafile.number_of_columns
        self.column_number = self.pulse_records.datafile.column_number

        if self.number_of_rows is not None and self.timebase is not None:
            self.row_timebase = self.timebase / float(self.number_of_rows)

        not_done = all(self.p_pretrig_mean[:] == 0)
        if not (not_done or forceNew):
            LOG.info('\nchan %d did not summarize because results were already preloaded', self.channum)
            return

        if len(self.p_timestamp) < self.pulse_records.nPulses:
            self.__setup_vectors(npulses=self.pulse_records.nPulses)  # make sure vectors are setup correctly

        if peak_time_microsec is None:
            self.peak_samplenumber = None
        else:
            self.peak_samplenumber = 2+self.nPresamples+int(peak_time_microsec*1e-6/self.timebase)
        if pretrigger_ignore_microsec is None:
            self.pretrigger_ignore_samples = 3
        else:
            self.pretrigger_ignore_samples = int(pretrigger_ignore_microsec*1e-6/self.timebase)
            
        self.cut_pre = cut_pre
        self.cut_post = cut_post

        for segnum in range(self.pulse_records.n_segments):
            if use_cython:
                self._summarize_data_segment(segnum)
            else:
                MicrocalDataSet._summarize_data_segment(self, segnum, doPretrigFit=doPretrigFit)
            yield (segnum+1.0) / self.pulse_records.n_segments

        self.pulse_records.datafile.clear_cached_segment()
        self.clear_cache()
        self.hdf5_group.file.flush()

    def _summarize_data_segment(self, segnum, doPretrigFit=False):
        """Summarize one segment of the data file, loading it into cache."""
        first, end = self.read_segment(segnum)  # this reloads self.data to contain new pulses
        if first >= self.nPulses:
            return
        if end > self.nPulses:
            end = self.nPulses

        if len(self.p_timestamp) <= 0:
            self.__setup_vectors(npulses=self.nPulses)

        # Don't look for retriggers before this # of samples. Use the most common
        # value of the peak index in the currently-loaded segment.
        if self.peak_samplenumber is None:
            self._compute_peak_samplenumber()

        seg_size = end-first
        self.p_timestamp[first:end] = self.times[:seg_size]
        self.p_rowcount[first:end] = self.rowcount[:seg_size]
        
        # Fit line to pretrigger and save the derivative and offset
        if doPretrigFit:
            presampleNumbers = np.arange(self.cut_pre,self.nPresamples-self.pretrigger_ignore_samples)
            self.p_pretrig_deriv[first:end], self.p_pretrig_offset[first:end] = \
                np.polyfit(presampleNumbers, self.data[:seg_size, self.cut_pre:self.nPresamples-self.pretrigger_ignore_samples].T, deg=1)  
            
        self.p_pretrig_mean[first:end] = \
            self.data[:seg_size, self.cut_pre:self.nPresamples-self.pretrigger_ignore_samples].mean(axis=1)
        self.p_pretrig_rms[first:end] = \
            self.data[:seg_size, self.cut_pre:self.nPresamples-self.pretrigger_ignore_samples].std(axis=1)
        self.p_peak_index[first:end] = self.data[:seg_size, self.cut_pre:self.nSamples-self.cut_post].argmax(axis=1)+self.cut_pre
        self.p_peak_value[first:end] = self.data[:seg_size, self.cut_pre:self.nSamples-self.cut_post].max(axis=1)
        self.p_min_value[first:end] = self.data[:seg_size, self.cut_pre:self.nSamples-self.cut_post].min(axis=1)
        self.p_pulse_average[first:end] = self.data[:seg_size, self.nPresamples:self.nSamples-self.cut_post].mean(axis=1)


        # Remove the pretrigger mean from the peak value and the pulse average figures.
        ptm = self.p_pretrig_mean[first:end]
        self.p_pulse_average[first:end] -= ptm
        self.p_peak_value[first:end] -= np.asarray(ptm, dtype=self.p_peak_value.dtype)
        self.p_pulse_rms[first:end] = np.sqrt(
            (self.data[:seg_size, self.nPresamples:self.nSamples-self.cut_post]**2.0).mean(axis=1) -
            ptm*(ptm + 2*self.p_pulse_average[first:end]))

        shift1 = (self.data[:seg_size, self.nPresamples+2]-ptm >
                  4.3*self.p_pretrig_rms[first:end])
        self.p_shift1[first:end] = shift1

        halfidx = (self.nPresamples+5+self.peak_samplenumber)//2
        pkval = self.p_peak_value[first:end]
        prompt = (self.data[:seg_size, self.nPresamples+5:halfidx].mean(axis=1) -
                  ptm) / pkval
        prompt[shift1] = (self.data[shift1, self.nPresamples+4:halfidx-1].mean(axis=1) -
                          ptm[shift1]) / pkval[shift1]
        self.p_promptness[first:end] = prompt

        self.p_rise_time[first:end] = \
            mass.core.analysis_algorithms.estimateRiseTime(self.data[:seg_size, self.cut_pre:self.nSamples-self.cut_post],
                                                           timebase=self.timebase,
                                                           nPretrig=self.nPresamples-self.cut_pre)

        self.p_postpeak_deriv[first:end] = \
            mass.core.analysis_algorithms.compute_max_deriv(self.data[:seg_size, self.cut_pre:self.nSamples-self.cut_post],
                                                            ignore_leading=self.peak_samplenumber-self.cut_pre)
    
    
    @show_progress("compute_average_pulse")
    def compute_average_pulse(self, mask, subtract_mean=True, forceNew=False):
        """Compute the average pulse this channel.

        Store as self.average_pulse

        Args:
            mask -- A boolean array saying which records to average.
            subtract_mean -- Whether to subtract the pretrigger mean and set the
                pretrigger period to strictly zero (default True).
            forceNew -- Whether to recompute when already exists (default False)
        """
        # Don't proceed if not necessary and not forced
        already_done = self.average_pulse[-1] != 0
        if already_done and not forceNew:
            LOG.info("skipping compute average pulse on chan %d", self.channum)
            return

        pulse_count = 0
        pulse_sum = np.zeros(self.nSamples, dtype=np.float)

        # Compute a master mask to say whether ANY mask wants a pulse from each segment
        # This can speed up work a lot when the pulses being averaged are from certain times only.
        segment_mask = np.zeros(self.pulse_records.n_segments, dtype=np.bool)
        n = len(mask)
        ppseg = self.pulse_records.pulses_per_seg
        nseg = 1 + (n - 1) // ppseg
        for i in range(nseg):
            a = i * ppseg
            b = a + ppseg
            if b >= len(mask):
                b = len(mask) - 1
            if mask[a:b].any():
                segment_mask[i] = True

        for iseg in range(nseg):
            if not segment_mask[iseg]:
                continue
            first, end = self.read_segment(iseg)
            yield end / float(self.nPulses)
            valid = mask[first:end]

            if mask.shape != (self.nPulses,):
                raise ValueError("\nmasks[chan %d] has shape %s, but it needs to be (%d,)" %
                                 (self.channum, mask.shape, self.nPulses))
            if len(valid) > self.data.shape[0]:
                good_pulses = self.data[valid[:self.data.shape[0]], :]
            else:
                good_pulses = self.data[valid, :]
            pulse_count += good_pulses.shape[0]
            pulse_sum[:] += good_pulses.sum(axis=0)

        # Rescale and store result to each MicrocalDataSet

        average_pulse = pulse_sum
        if pulse_count > 0:
            average_pulse /= pulse_count
        if subtract_mean:
            average_pulse -= np.mean(average_pulse[:self.nPresamples - self.pretrigger_ignore_samples])
        self.average_pulse[:] = average_pulse

    @_add_group_loop
    def avg_pulses_auto_masks(self, max_pulses_to_use=7000, subtract_mean=True, forceNew=False):
        """Compute an average pulse.

        Compute average pulse using an automatically generated mask of
        +- 5%% around the median pulse_average value.

        Args:
            max_pulses_to_use (int): Use no more than
                the first this many good pulses (default 7000).
            forceNew (bool): whether to re-compute if results already exist (default False)
        """
        use = self.good()
        if use.sum() <= 0:
            raise ValueError("No good pulses")
        median_pulse_avg = np.median(self.p_pulse_average[use])
        mask = np.abs(self.p_pulse_average[:]/median_pulse_avg-1) < 0.05
        mask = np.logical_and(mask, use)
        if mask.sum() <= 0:
            raise ValueError("No good pulses within 5%% of median size.")

        if np.sum(mask) > max_pulses_to_use:
            good_so_far = np.cumsum(mask)
            stop_at = (good_so_far == max_pulses_to_use).argmax()
            mask[stop_at+1:] = False
        self.compute_average_pulse(mask, subtract_mean=subtract_mean,
                                   forceNew=forceNew)

    def compute_oldfilter(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        try:
            spectrum = self.noise_spectrum.spectrum()
        except:
            spectrum = self.noise_psd[:]

        avg_signal = np.array(self.average_pulse)
        f = mass.core.Filter(avg_signal, self.nPresamples-self.pretrigger_ignore_samples,
                             spectrum, self.noise_autocorr, sample_time=self.timebase,
                             shorten=2, cut_pre=cut_pre, cut_post=cut_post)
        f.compute(fmax=fmax, f_3db=f_3db)
        return f

    def compute_newfilter(self, fmax=None, f_3db=None, transform=None, cut_pre=0, cut_post=0):
        """Compute a new-style filter to model the pulse and its time-derivative.

        Args:
            fmax: if not None, the hard cutoff in frequency space, above which
                the DFT of the filter will be set to zero (default None)
            f_3db: if not None, the 3 dB rolloff point in frequency space, above which
                the DFT of the filter will rolled off with a 1-pole filter
                (default None)
            transform: a callable object that will be called on all data records
                before filtering (default None)
            cut_pre: Cut this many samples from the start of the filter, giving them 0 weight.
            cut_post: Cut this many samples from the end of the filter, giving them 0 weight.

        Returns:
            the filter (an ndarray)

        Modified in April 2017 to make the model for the rising edge and the rest of
        the pulse differently. For the rising edge, we use entropy minimization to understand
        the pulse shape dependence on arrival-time. For the rest of the pulse, it
        is less noisy and in fact more robust to rely on the finite-difference of
        the pulse average to get the arrival-time dependence.
        """

        # At the moment, 1st-order model vs arrival-time is required.
        DEGREE = 1

        # The raw training data, which is shifted (trigger-aligned)
        data, pulsenums = self.first_n_good_pulses(3000)
        raw = data[:, 1:]
        shift1 = self.p_shift1[:][pulsenums]
        raw[shift1, :] = data[shift1, 0:-1]

        # Center promptness around 0, using a simple function of Prms
        prompt = self.p_promptness[:][pulsenums]
        prms = self.p_pulse_rms[:][pulsenums]
        mprms = np.median(prms)
        use = np.abs(prms/mprms-1.0) < 0.3
        promptshift = np.poly1d(np.polyfit(prms[use], prompt[use], 1))
        prompt -= promptshift(prms)

        # Scale promptness quadratically to cover the range -0.5 to +0.5, approximately
        x, y, z = sp.stats.scoreatpercentile(prompt[use], [10, 50, 90])
        A = np.array([[x*x, x, 1],
                      [y*y, y, 1],
                      [z*z, z, 1]])
        param = np.linalg.solve(A, [-.4, 0, +.4])
        ATime = np.poly1d(param)(prompt)
        use = np.logical_and(use, np.abs(ATime) < 0.45)
        ATime = ATime[use]

        ptm = self.p_pretrig_mean[:][pulsenums]
        ptm.shape = (len(pulsenums), 1)
        raw = (raw-ptm)[use, :]
        if transform is not None:
            raw = transform(raw)
        rawscale = raw.max(axis=1)

        # The 0 component of the model is an average pulse, but do not use
        # self.average_pulse, because it doesn't account for the shift1.
        model = np.zeros((self.nSamples-1, 1+DEGREE), dtype=float)
        ap = (raw.T/rawscale).mean(axis=1)
        apmax = np.max(ap)
        model[:, 0] = ap/apmax
        model[1:-1, 1] = (ap[2:] - ap[:-2])*0.5/apmax
        model[-1, 1] = (ap[-1]-ap[-2])/apmax
        model[:self.nPresamples+2, :] = 0

        # Now use min-entropy computation to model dp/dt on the rising edge
        def cost(slope, x, y):
            return mass.mathstat.entropy.laplace_entropy(y-x*slope, 0.002)

        if self.peak_samplenumber is None:
            self._compute_peak_samplenumber()
        for samplenum in range(self.nPresamples+2, self.peak_samplenumber):
            y = raw[:, samplenum]/rawscale
            bestslope = sp.optimize.brent(cost, (ATime, y), brack=[-.1, .25], tol=1e-7)
            model[samplenum, 1] = bestslope

        modelpeak = np.median(rawscale)
        self.pulsemodel = model
        f = ArrivalTimeSafeFilter(model, self.nPresamples, self.noise_autocorr,
                                  sample_time=self.timebase, peak=modelpeak)
        f.compute(fmax=fmax, f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)
        self.filter = f
        return f

    @_add_group_loop
    @show_progress("channel.filter_data_tdm")
    def filter_data(self, filter_name='filt_noconst', transform=None, forceNew=False,
                    use_cython=False):
        """Filter the complete data file one chunk at a time.

        Args:
            filter_name: the object under self.filter to use for filtering the
                data records (default 'filt_noconst')
            transform: a callable object that will be called on all data records
                before filtering (default None)
            forceNew: Whether to recompute when already exists (default False)
        """
        if not(forceNew or all(self.p_filt_value[:] == 0)):
            LOG.info('\nchan %d did not filter because results were already loaded', self.channum)
            return

        if self.filter is not None:
            filter_values = self.filter.__dict__[filter_name]
        else:
            filter_values = self.hdf5_group['filters/%s' % filter_name].value

        if self._use_new_filters:
            filterfunction = self._filter_data_segment_new
            filter_AT = self.filter.filt_aterms[0]
        else:
            filterfunction = self._filter_data_segment_old
            filter_AT = None

        for s in range(self.pulse_records.n_segments):
            first, end = self.read_segment(s)  # this reloads self.data to contain new pulses
            (self.p_filt_phase[first:end],
             self.p_filt_value[first:end]) = \
                filterfunction(filter_values, filter_AT, first, end, transform)
            yield (end+1)/float(self.nPulses)

        self.pulse_records.datafile.clear_cached_segment()
        self.hdf5_group.file.flush()

    def _filter_data_segment_old(self, filter_values, _filter_AT, first, end, transform=None):
        """Traditional 5-lag filter used by default until 2015."""
        if first >= self.nPulses:
            return None, None

        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = np.array((
            (-6, 24, 34, 24, -6),
            (-14, -7, 0, 7, 14),
            (10, -5, -10, -5, 10)), dtype=np.float)/70.0

        assert len(filter_values) + 4 == self.nSamples

        seg_size = end - first
        assert seg_size == self.data.shape[0]
        data = self.data
        conv = np.zeros((5, seg_size), dtype=np.float)
        if transform is not None:
            ptmean = self.p_pretrig_mean[first:end]
            ptmean.shape = (seg_size, 1)
            data = transform(data - ptmean)
        conv[0, :] = np.dot(data[:, 0:-4], filter_values)
        conv[1, :] = np.dot(data[:, 1:-3], filter_values)
        conv[2, :] = np.dot(data[:, 2:-2], filter_values)
        conv[3, :] = np.dot(data[:, 3:-1], filter_values)
        conv[4, :] = np.dot(data[:, 4:], filter_values)

        param = np.dot(fit_array, conv)
        peak_x = -0.5 * param[1, :] / param[2, :]
        peak_y = param[0, :] - 0.25 * param[1, :]**2 / param[2, :]
        return peak_x, peak_y

    def _filter_data_segment_new(self, filter_values, filter_AT, first, end, transform=None):
        """single-lag filter developed in 2015"""
        if first >= self.nPulses:
            return None, None

        assert len(filter_values) + 1 == self.nSamples

        seg_size = end - first
        assert seg_size == self.data.shape[0]
        ptmean = self.p_pretrig_mean[first:end]
        data = self.data
        if transform is not None:
            ptmean.shape = (seg_size, 1)
            data = transform(self.data - ptmean)
            ptmean.shape = (seg_size,)
        conv0 = np.dot(data[:, 1:], filter_values)
        conv1 = np.dot(data[:, 1:], filter_AT)

        # Find pulses that triggered 1 sample too late and "want to shift"
        want_to_shift = self.p_shift1[first:end]
        conv0[want_to_shift] = np.dot(data[want_to_shift, :-1], filter_values)
        conv1[want_to_shift] = np.dot(data[want_to_shift, :-1], filter_AT)
        AT = conv1 / conv0
        return AT, conv0

    def plot_summaries(self, valid='uncut', downsample=None, log=False):
        """Plot a summary of the data set, including time series and histograms of
        key pulse properties.

        Args:
            valid: An array of booleans self.nPulses long saying which pulses are to be plotted
                *OR* 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted
                *OR* None, meaning that all pulses should be plotted.

            downsample: To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points (default None).

            log (bool):  Use logarithmic y-axis on the histograms (right panels). (Default False)
        """

        # Convert "uncut" or "cut" to array of all good or all bad data
        if isinstance(valid, basestring):
            if "uncut" in valid.lower():
                valid = self.cuts.good()
                status = "Plotting only uncut data"
            elif "cut" in valid.lower():
                valid = self.cuts.bad()
                status = "Plotting only cut data"
            elif 'all' in valid.lower():
                valid = None
                status = "Plotting all data, cut or uncut"
            else:
                raise ValueError("If valid is a string, it must contain 'all', 'uncut' or 'cut'.")

        if valid is not None:
            nrecs = valid.sum()
            if downsample is None:
                downsample = nrecs // 10000
                if downsample < 1:
                    downsample = 1
            hour = self.p_timestamp[valid][::downsample] / 3600.0
        else:
            nrecs = self.nPulses
            if downsample is None:
                downsample = self.nPulses // 10000
                if downsample < 1:
                    downsample = 1
            hour = self.p_timestamp[::downsample] / 3600.0
        LOG.info("%s (%d records; %d in scatter plots)", status, nrecs, len(hour))

        plottables = (
            (self.p_pulse_rms, 'Pulse RMS', 'magenta', None),
            (self.p_pulse_average, 'Pulse Avg', 'purple', None),
            (self.p_peak_value, 'Peak value', 'blue', None),
            (self.p_pretrig_rms, 'Pretrig RMS', 'green', [0, 4000]),
            (self.p_pretrig_mean, 'Pretrig Mean', '#00ff26', None),
            (self.p_postpeak_deriv, 'Max PostPk deriv', 'gold', [0, 700]),
            (self.p_rise_time[:]*1e3, 'Rise time (ms)', 'orange', [0, 12]),
            (self.p_peak_time[:]*1e3, 'Peak time (ms)', 'red', [-3, 9])
        )

        # Plot timeseries with 0 = the last 00 UT during or before the run.
        last_record = np.max(self.p_timestamp)
        last_midnight = last_record - (last_record % 86400)
        hour_offset = last_midnight/3600.

        plt.clf()
        for i, (vect, label, color, limits) in enumerate(plottables):

            # Time series scatter plots (left-hand panels)
            plt.subplot(len(plottables), 2, 1+i*2)
            plt.ylabel(label)
            if valid is not None:
                vect = vect[valid]
            plt.plot(hour-hour_offset, vect[::downsample], '.', ms=1, color=color)
            if i == len(plottables) - 1:
                plt.xlabel("Time since last UT midnight (hours)")

            # Histogram (right-hand panels)
            plt.subplot(len(plottables), 2, 2+i*2)
            if limits is None:
                in_limit = np.ones(len(vect), dtype=np.bool)
            else:
                in_limit = np.logical_and(vect[:] > limits[0], vect[:] < limits[1])
            contents, _bins, _patches = plt.hist(vect[in_limit], 200, log=log,
                                                 histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin=contents.min())

    @_add_group_loop
    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        """Compute the noise power spectrum of this channel.

        Args:
            max_excursion (number): the biggest excursion from the median allowed
                in each data segment, or else it will be ignored (default 1000).
            n_lags: if not None, the number of lags in each noise spectrum and the max lag
                for the autocorrelation.  If None, the record length is used
                (default None).
            forceNew (bool): whether to recompute if it already exists (default False).
        """
        if n_lags is None:
            n_lags = self.nSamples
        if forceNew or all(self.noise_autocorr[:] == 0):
            self.noise_records.compute_power_spectrum_reshape(max_excursion=max_excursion, seg_length=n_lags)
            self.noise_records.compute_autocorrelation(n_lags=n_lags, plot=False, max_excursion=max_excursion)
            self.noise_records.clear_cache()

            self.noise_autocorr[:] = self.noise_records.autocorrelation[:]
            self.noise_psd[:] = self.noise_records.noise_psd[:]
            self.noise_psd.attrs['delta_f'] = self.noise_records.noise_psd.attrs['delta_f']
        else:
            LOG.info("chan %d skipping compute_noise_spectra because already done", self.channum)

    @_add_group_loop
    def apply_cuts(self, controls, clear=False, forceNew=True):
        """Apply the cuts.

        Args:
            controls (AnalysisControl): contains the cuts to apply.
            clear (bool):  Whether to clear previous cuts first (default False).
            forceNew (bool): whether to recompute if it already exists (default False).
        """
        if self.nPulses == 0:
            return  # don't bother current if there are no pulses
        if not forceNew:
            if self.cuts.good().sum() != self.nPulses:
                LOG.info("Chan %d skipped cuts: after %d are good, %d are bad of %d total pulses",
                         self.channum, self.cuts.good().sum(), self.cuts.bad().sum(), self.nPulses)

        if clear:
            self.clear_cuts()

        if controls is None:
            controls = mass.controller.standardControl()
        c = controls.cuts_prm

        self.cuts.cut_parameter(self.p_energy, c['energy'], 'energy')
        self.cuts.cut_parameter(self.p_pretrig_rms, c['pretrigger_rms'], 'pretrigger_rms')
        self.cuts.cut_parameter(self.p_pretrig_mean, c['pretrigger_mean'], 'pretrigger_mean')

        self.cuts.cut_parameter(self.p_peak_time[:]*1e3, c['peak_time_ms'], 'peak_time_ms')
        self.cuts.cut_parameter(self.p_rise_time[:]*1e3, c['rise_time_ms'], 'rise_time_ms')
        self.cuts.cut_parameter(self.p_postpeak_deriv, c['postpeak_deriv'], 'postpeak_deriv')
        self.cuts.cut_parameter(self.p_pulse_average, c['pulse_average'], 'pulse_average')
        self.cuts.cut_parameter(self.p_peak_value, c['peak_value'], 'peak_value')
        self.cuts.cut_parameter(self.p_min_value[:] - self.p_pretrig_mean[:], c['min_value'], 'min_value')
        self.cuts.cut_parameter(self.p_timestamp[:], c['timestamp_sec'], 'timestamp_sec')

        if c['timestamp_diff_sec'] is not None:
            self.cuts.cut_parameter(np.hstack((np.inf, np.diff(self.p_timestamp))),
                                    c['timestamp_diff_sec'], 'timestamp_diff_sec')
        if c['rowcount_diff_sec'] is not None:
            self.cuts.cut_parameter(np.hstack((np.inf, np.diff(self.p_rowcount[:] * self.row_timebase))),
                                    c['rowcount_diff_sec'], 'rowcount_diff_sec')
        if c['pretrigger_mean_departure_from_median'] is not None and self.cuts.good().sum() > 0:
            median = np.median(self.p_pretrig_mean[self.cuts.good()])
            LOG.debug('applying cut on pretrigger mean around its median value of %f', median)
            self.cuts.cut_parameter(self.p_pretrig_mean-median,
                                    c['pretrigger_mean_departure_from_median'],
                                    'pretrigger_mean_departure_from_median')
        LOG.info("Chan %d after cuts, %d are good, %d are bad of %d total pulses",
                 self.channum, self.cuts.good().sum(), self.cuts.bad().sum(), self.nPulses)

    @_add_group_loop
    def clear_cuts(self):
        """Clear all cuts."""
        self.cuts.clear_cut()

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
        # remember original value, just in case we need it
        self.p_pretrig_mean_orig = self.p_pretrig_mean[:]
        corrected = mass.core.analysis_algorithms.correct_flux_jumps(self.p_pretrig_mean[:], self.good(), flux_quant)
        self.p_pretrig_mean[:] = corrected

    @_add_group_loop
    def drift_correct(self, forceNew=False, category=None):
        """Drift correct using the standard entropy-minimizing algorithm"""
        doesnt_exist = all(self.p_filt_value_dc[:] == 0) or all(self.p_filt_value_dc[:] == self.p_filt_value[:])
        if not (forceNew or doesnt_exist):
            LOG.info("chan %d drift correction skipped, because p_filt_value_dc already populated", self.channum)
            return
        if category is None:
            category = {"calibration": "in"}
        g = self.cuts.good(**category)
        uncorrected = self.p_filt_value[g]
        indicator = self.p_pretrig_mean[g]
        drift_corr_param, self.drift_correct_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        self.p_filt_value_dc.attrs.update(self.drift_correct_info)  # Store in hdf5 file
        LOG.info('chan %d best drift correction parameter: %.6f', self.channum, drift_corr_param)
        self._apply_drift_correction()

    def _apply_drift_correction(self):
        # Apply correction
        assert self.p_filt_value_dc.attrs["type"] == "ptmean_gain"
        ptm_offset = self.p_filt_value_dc.attrs["median_pretrig_mean"]
        gain = 1+(self.p_pretrig_mean[:]-ptm_offset)*self.p_filt_value_dc.attrs["slope"]
        self.p_filt_value_dc[:] = self.p_filt_value[:]*gain
        self.hdf5_group.file.flush()

    @_add_group_loop
    def phase_correct2014(self, typical_resolution, maximum_num_records=50000, plot=False,
                          forceNew=False, category=None):
        """Apply the phase correction that worked for calibronium-like data as of June 2014.

        For more notes, do help(mass.core.analysis_algorithms.FilterTimeCorrection)

        Args:
            typical_resolution (number): should be an estimated energy resolution in UNITS OF
                self.p_pulse_rms. This helps the peak-finding (clustering) algorithm decide
                which pulses go together into a single peak.  Be careful to use a semi-reasonable
                quantity here.
            maximum_num_records (int): don't use more than this many records to learn
                the correction (default 50000).
            plot (bool): whether to make a relevant plot
            forceNew (bool): whether to recompute if it already exists (default False).
            category (dict): if not None, then a dict giving a category name and the
                required category label.
        """
        doesnt_exist = all(self.p_filt_value_phc[:] == 0) or all(self.p_filt_value_phc[:] == self.p_filt_value_dc[:])
        if not (forceNew or doesnt_exist):
            LOG.info("channel %d skipping phase_correct2014", self.channum)
            return

        if category is None:
            category = {"calibration": "in"}
        data, g = self.first_n_good_pulses(maximum_num_records, category)
        LOG.info("channel %d doing phase_correct2014 with %d good pulses", self.channum, data.shape[0])
        prompt = self.p_promptness[:]
        prms = self.p_pulse_rms[:]

        if self.filter is not None:
            dataFilter = self.filter.__dict__['filt_noconst']
        else:
            dataFilter = self.hdf5_group['filters/filt_noconst'][:]
        tc = mass.core.analysis_algorithms.FilterTimeCorrection(
            data, prompt[g], prms[g], dataFilter,
            self.nPresamples, typicalResolution=typical_resolution)

        self.p_filt_value_phc[:] = self.p_filt_value_dc[:]
        self.p_filt_value_phc[:] -= tc(prompt, prms)
        if plot:
            fnum = plt.gcf().number
            plt.figure(5)
            plt.clf()
            g = self.cuts.good()
            plt.plot(prompt[g], self.p_filt_value_dc[g], 'g.')
            plt.plot(prompt[g], self.p_filt_value_phc[g], 'b.')
            plt.figure(fnum)

    def _find_peaks_heuristic(self, phnorm):
        """A heuristic method to identify the peaks in a spectrum.

        This can be used to design the arrival-time-bias correction. Of course,
        you might have better luck finding peaks by an experiment-specific
        method, but this will stand in if you cannot or do not want to find
        peaks another way.

        Args:
            phnorm: a vector of pulse heights, found by whatever means you like.
                Normally it will be the self.p_filt_value_dc AFTER CUTS.

        Returns:
            ndarray of the various peaks found in the input vector.
        """
        median_scale = np.median(phnorm)

        # First make histogram with bins = 0.2% of median PH
        hist, bins = np.histogram(phnorm, 1000, [0, 2*median_scale])
        binctr = bins[1:] - 0.5 * (bins[1] - bins[0])

        # Scipy continuous wavelet transform
        pk1 = np.array(sp.signal.find_peaks_cwt(hist, np.array([2, 4, 8, 12])))

        # A peak must contain 0.5% of the data or 500 events, whichever is more,
        # but the requirement is not more than 5% of data (for meager data sets)
        Ntotal = len(phnorm)
        MinCountsInPeak = min(max(500, Ntotal//200), Ntotal//20)
        pk2 = pk1[hist[pk1] > MinCountsInPeak]

        # Now take peaks from highest to lowest, provided they are at least 40 bins from any neighbor
        ordering = hist[pk2].argsort()
        pk2 = pk2[ordering]
        peaks = [pk2[0]]

        for pk in pk2[1:]:
            if (np.abs(peaks-pk) > 10).all():
                peaks.append(pk)
        peaks.sort()
        return np.array(binctr[peaks])

    @_add_group_loop
    def phase_correct(self, forceNew=False, category=None, ph_peaks=None, method2017=False,
                      kernel_width=None):
        """Apply the 2017 or 2015 phase correction method.

        Args:
            forceNew (bool): whether to recompute if it already exists (default False).
            category (dict): if not None, then a dict giving a category name and the
                required category label.
            ph_peaks:  Peaks to use for alignment. If None, then use self._find_peaks_heuristic()
            kernel_width: Width (in PH units) of the kernel-smearing function. If None, use a heuristic.
        """

        doesnt_exist = all(self.p_filt_value_phc[:] == 0) or all(self.p_filt_value_phc[:] == self.p_filt_value_dc[:])
        if not (forceNew or doesnt_exist):
            LOG.info("channel %d skipping phase_correct", self.channum)
            return

        if category is None:
            category = {"calibration": "in"}
        good = self.cuts.good(**category)

        if ph_peaks is None:
            ph_peaks = self._find_peaks_heuristic(self.p_filt_value_dc[good])
        if len(ph_peaks) <= 0:
            LOG.info("Could not phase_correct on chan %3d because no peaks", self.channum)
            return
        ph_peaks = np.asarray(ph_peaks)
        ph_peaks.sort()

        # Compute a correction function at each line in ph_peaks
        corrections = []
        median_phase = []
        if kernel_width is None:
            kernel_width = np.max(ph_peaks)/1000.0
        for pk in ph_peaks:
            c, mphase = _phasecorr_find_alignment(self.p_filt_phase[good],
                                                  self.p_filt_value_dc[good], pk,
                                                  .012*np.mean(ph_peaks),
                                                  method2017=method2017,
                                                  kernel_width=kernel_width)
            corrections.append(c)
            median_phase.append(mphase)
        median_phase = np.array(median_phase)

        # Store the info needed to reconstruct corrections
        nc = np.hstack([len(c._x) for c in corrections])
        cx = np.hstack([c._x for c in corrections])
        cy = np.hstack([c._y for c in corrections])
        for name, data in zip(("phase_corrector_x", "phase_corrector_y", "phase_corrector_n"),
                              (cx, cy, nc)):
            if name in self.hdf5_group:
                del self.hdf5_group[name]
            self.hdf5_group.create_dataset(name, data=data)

        NC = len(corrections)
        if NC > 3:
            phase_corrector = mass.mathstat.interpolate.CubicSpline(ph_peaks, median_phase)
        else:
            # Too few peaks to spline, so just bin and take the median per bin, then
            # interpolated (approximating) spline through/near these points.
            NBINS = 10
            dc = self.p_filt_value_dc[good]
            ph = self.p_filt_phase[good]
            top = min(dc.max(), 1.2*sp.stats.scoreatpercentile(dc, 98))
            bin = np.digitize(dc, np.linspace(0, top, 1+NBINS))-1
            x = np.zeros(NBINS, dtype=float)
            y = np.zeros(NBINS, dtype=float)
            w = np.zeros(NBINS, dtype=float)
            for i in range(NBINS):
                w[i] = (bin == i).sum()
                if w[i] == 0:
                    continue
                x[i] = np.median(dc[bin == i])
                y[i] = np.median(ph[bin == i])

            nonempty = w > 0
            # Use sp.interpolate.UnivariateSpline because it can make an approximating
            # spline. But then use its x/y data and knots to create a Mass CubicSpline,
            # because that one can have natural boundary conditions instead of insane
            # cubic functions in the extrapolation.
            crazy_spline = sp.interpolate.UnivariateSpline(x[nonempty], y[nonempty], w=w[nonempty]*(12**-0.5))
            phase_corrector = mass.mathstat.interpolate.CubicSpline(crazy_spline._data[0], crazy_spline._data[1])
        self.p_filt_phase_corr[:] = self.p_filt_phase[:] - phase_corrector(self.p_filt_value_dc[:])
        return self._apply_phase_correction(category=category)

    def _apply_phase_correction(self, category=None):
        if category is None:
            category = {"calibration": "in"}
        good = self.cuts.good(**category)

        # Compute a correction for each pulse for each correction-line energy
        # For the actual correction, don't let |ph| > 0.6 sample
        corrected_phase = np.clip(self.p_filt_phase_corr[:], -0.6, 0.6)

        nc = self.hdf5_group["phase_corrector_n"][...]
        cx = self.hdf5_group["phase_corrector_x"][...]
        cy = self.hdf5_group["phase_corrector_y"][...]
        corrections = []
        idx = 0
        for n in nc:
            x = cx[idx:idx+n]
            y = cy[idx:idx+n]
            idx += n
            spl = mass.mathstat.interpolate.CubicSpline(x, y)
            corrections.append(spl)

        self.p_filt_value_phc[:] = _phase_corrected_filtvals(corrected_phase, self.p_filt_value_dc, corrections)

        LOG.info('Channel %3d phase corrected. Correction size: %.2f',
                 self.channum, mass.mathstat.robust.median_abs_dev(self.p_filt_value_phc[good] -
                                                                   self.p_filt_value_dc[good], True))
        self.phase_corrections = corrections
        return corrections

    def first_n_good_pulses(self, n=50000, category=None):
        """Return the first good pulse records.

        Args:
            n: maximum number of good pulses to include (default 50000).

        Returns:
            (data, g)
            data is a (X,Y) array where X is number of records, and Y is number of samples per record
            g is a 1d array of of pulse record numbers of the pulses in data.

        If we did load all of ds.data at once, this would be roughly equivalent to
        return ds.data[ds.cuts.good()][:n], np.nonzero(ds.cuts.good())[0][:n]
        """
        if category is None:
            category = {"calibration": "in"}
        g = self.cuts.good(**category)

        first, end = self.read_segment(0)
        data = self.data[g[first:end]]
        for j in range(1, self.pulse_records.n_segments):
            if data.shape[0] > n:
                break
            first, end = self.read_segment(j)
            data = np.vstack((data, self.data[g[first:end], :]))
        nrecords = np.amin([n, data.shape[0]])
        return data[:nrecords], np.nonzero(g)[0][:nrecords]

    def fit_spectral_line(self, prange, mask=None, times=None, fit_type='dc', line='MnKAlpha',
                          nbins=200, plot=True, **kwargs):
        all_values = {'filt': self.p_filt_value,
                      'phc': self.p_filt_value_phc,
                      'dc': self.p_filt_value_dc,
                      'energy': self.p_energy,
                      }[fit_type]
        if mask is not None:
            valid = np.array(mask)
        else:
            valid = self.cuts.good()
        if times is not None:
            valid = np.logical_and(valid, self.p_timestamp < times[1])
            valid = np.logical_and(valid, self.p_timestamp > times[0])
        good_values = all_values[valid]
        contents, bin_edges = np.histogram(good_values, nbins, prange)
        LOG.info("%d events pass cuts; %d are in histogram range", len(good_values), contents.sum())
        bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])

        # Try line first as a number, then as a fluorescence line, then as a Gaussian
        try:
            energy = float(line)
            module = 'mass.calibration.gaussian_lines'
            fittername = '%s.GaussianFitter(%s.GaussianLine())' % (module, module)
            fitter = eval(fittername)
        except ValueError:
            energy = None
            try:
                module = 'mass.calibration.fluorescence_lines'
                fittername = '%s.%sFitter()' % (module, line)
                fitter = eval(fittername)
            except AttributeError:
                try:
                    module = 'mass.calibration.gaussian_lines'
                    fittername = '%s.%sFitter()' % (module, line)
                    fitter = eval(fittername)
                except AttributeError:
                    raise ValueError("Cannot understand line=%s as an energy or a known calibration line." % line)

        params, covar = fitter.fit(contents, bin_ctrs, plot=plot, **kwargs)
        if plot:
            mass.plot_as_stepped_hist(plt.gca(), contents, bin_ctrs)
        if energy is not None:
            scale = energy/params[1]
        else:
            scale = 1.0
        LOG.info('Resolution: %5.2f +- %5.2f eV', params[0]*scale, np.sqrt(covar[0, 0])*scale)
        return params, covar, fitter

    @property
    def pkl_fname(self):
        return ljh_util.mass_folder_from_ljh_fname(self.filename, filename="ch%d_calibration.pkl" % self.channum)

    @_add_group_loop
    def calibrate(self, attr, line_names, name_ext="", size_related_to_energy_resolution=10,
                  fit_range_ev=200, excl=(), plot_on_fail=False,
                  bin_size_ev=2.0, category=None, forceNew=False, maxacc=0.015, nextra=3,
                  param_adjust_closure=None, diagnose=False):
        calname = attr+name_ext

        if not forceNew and calname in self.calibration:
            return self.calibration[calname]

        LOG.info("Calibrating chan %d to create %s", self.channum, calname)
        cal = EnergyCalibration()
        cal.set_use_approximation(False)

        if category is None:
            category = {"calibration": "in"}

        # It tries to calibrate detector using mass.calibration.algorithm.EnergyCalibrationAutocal.
        auto_cal = EnergyCalibrationAutocal(cal,
                                            getattr(self, attr)[self.cuts.good(**category)],
                                            line_names)
        auto_cal.guess_fit_params(smoothing_res_ph=size_related_to_energy_resolution,
                                  fit_range_ev=fit_range_ev,
                                  binsize_ev=bin_size_ev,
                                  nextra=nextra, maxacc=maxacc)
        if param_adjust_closure:
            param_adjust_closure(self, auto_cal)
        auto_cal.fit_lines()

        if auto_cal.anyfailed:
            LOG.warning("chan %d failed calibration because on of the fitter was a FailedFitter", self.channum)
            raise Exception()

        self.calibration[calname] = cal
        hdf5_cal_group = self.hdf5_group.require_group('calibration')
        cal.save_to_hdf5(hdf5_cal_group, calname)

        if diagnose:
            auto_cal.diagnose()
        self.convert_to_energy(attr, attr + name_ext)

    @_add_group_loop
    def convert_to_energy(self, attr, calname=None):
        if calname is None:
            calname = attr
        if calname not in self.calibration:
            raise ValueError("For chan %d calibration %s does not exist" % (self.channum, calname))
        cal = self.calibration[calname]
        self.p_energy[:] = cal.ph2energy(getattr(self, attr))
        self.last_used_calibration = cal

    def read_segment(self, n):
        first, end = self.pulse_records.read_segment(n)
        self.data = self.pulse_records.data
        self.times = self.pulse_records.times
        self.rowcount = self.pulse_records.rowcount

        # If you want to invert all data on read, then set self.invert_data=True.
        if self.__dict__.get("invert_data", False):
            self.data = ~self.data
        return first, end

    def clear_cache(self):
        self.data = None
        self.rowcount = None
        self.times = None
        self.pulse_records.clear_cache()

    def plot_traces(self, pulsenums, pulse_summary=True, axis=None, difference=False,
                    residual=False, valid_status=None, shift1=False,
                    subtract_baseline=False, fcut=None):
        """Plot some example pulses, given by sample number.

        Args:
            <pulsenums>   A sequence of sample numbers, or a single one.
            <pulse_summary> Whether to put text about the first few pulses on the plot
            <axis>       A plt axis to plot on.
            <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
            <residual>   Whether to show the residual between data and opt filtered model, or just raw data.
            <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                         that have been cut.  If "cut", show only those that have been cut.
            <shift1>     Whether to take pulses with p_shift1==True and delay them by 1 sample
            <subtract_baseline>  Whether to subtract pretrigger mean prior to plotting the pulse
            <fcut>  If not none, apply a lowpass filter with this cutoff frequency prior to plotting
        """

        if isinstance(pulsenums, int):
            pulsenums = (pulsenums,)
        pulsenums = np.asarray(pulsenums)
        if pulse_summary:
            try:
                if len(self.p_pretrig_mean) == 0:
                    pulse_summary = False
            except AttributeError:
                pulse_summary = False

        if valid_status not in (None, "valid", "cut"):
            raise ValueError("valid_status must be one of [None, 'valid', or 'cut']")
        if residual and difference:
            raise ValueError("Only one of residual and difference can be True.")

        dt = (np.arange(self.nSamples)-self.nPresamples)*self.timebase*1e3
        cm = plt.cm.jet
        MAX_TO_SUMMARIZE = 30

        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        axis.set_xlabel("Time after trigger (ms)")
        axis.set_xlim([dt[0], dt[-1]])
        axis.set_ylabel("Feedback (or mix) in [Volts/16384]")
        if pulse_summary:
            axis.text(.975, .97, r"              -PreTrigger-   Max  Rise t Peak   Pulse",
                      size='medium', family='monospace', transform=axis.transAxes, ha='right')
            axis.text(.975, .95, r"Cut P#    Mean     rms PTDeriv  ($\mu$s) value   mean",
                      size='medium', family='monospace', transform=axis.transAxes, ha='right')

        cuts_good = self.cuts.good()[pulsenums]
        pulses_plotted = -1
        nplottable = cuts_good.sum()
        for i, pn in enumerate(pulsenums):
            if valid_status == 'cut' and cuts_good[i]:
                continue
            if valid_status == 'valid' and not cuts_good[i]:
                continue
            pulses_plotted += 1

            data = self.read_trace(pn)
            if difference:
                data = data*1.0 - np.roll(data, 1)
                data[0] = 0
                data += np.roll(data, 1) + np.roll(data, -1)
                data[0] = 0
            elif residual:
                model = self.p_filt_value[pn] * self.average_pulse[:] / np.max(self.average_pulse)
                data -= model
            if shift1 and self.p_shift1[pn]:
                data = np.hstack([data[0], data[:-1]])
            if fcut != None:
                data = mass.core.analysis_algorithms.filter_signal_lowpass(data, 1./self.timebase, fcut)
            if subtract_baseline:
                # Recalculate the pretrigger mean here, to avoid issues due to flux slipping when
                # plotting umux data
                data = data - np.mean(data[:self.nPresamples - self.pretrigger_ignore_samples])

            cutchar, alpha, linestyle, linewidth = ' ', 1.0, '-', 1

            # When plotting both cut and valid, mark the cut data with x and dashed lines
            if valid_status is None and not cuts_good[i]:
                cutchar, alpha, linestyle, linewidth = 'X', 1.0, '--', 1
            color = cm(pulses_plotted*1.0/len(cuts_good))
            axis.plot(dt, data, color=color,
                      linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            if pulse_summary and pulses_plotted < MAX_TO_SUMMARIZE and len(self.p_pretrig_mean) >= pn:
                try:
                    summary = "%s%6d: %5.0f %7.2f %6.1f %5.0f %5.0f %7.1f" % (
                        cutchar, pn, self.p_pretrig_mean[pn], self.p_pretrig_rms[pn],
                        self.p_postpeak_deriv[pn], self.p_rise_time[pn]*1e6,
                        self.p_peak_value[pn], self.p_pulse_average[pn])
                except IndexError:
                    pulse_summary = False
                    continue
                axis.text(.975, .93-.025*pulses_plotted, summary, color=color,
                          family='monospace', size='medium',
                          transform=axis.transAxes, ha='right')

    def read_trace(self, record_num):
        """Read (from cache or disk) and return the pulse numbered `record_num`."""
        seg_num = record_num // self.pulse_records.pulses_per_seg
        self.read_segment(seg_num)
        return self.data[record_num % self.pulse_records.pulses_per_seg, :]

    @_add_group_loop
    def time_drift_correct(self, attr="p_filt_value_phc", sec_per_degree = 2000,
                           pulses_per_degree = 2000, max_degrees = 20, forceNew=False):
        """Drift correct over long times with an entropy-minimizing algorithm.
        Here we correct as a low-ish-order Legendre polynomial in time.

        attr: the attribute of self that is to be corrected. (The result
                will be stored in self.p_filt_value_tdc[:]).
        sec_per_degree: assign as many as one polynomial degree per this many seconds
        pulses_per_degree: assign as many as one polynomial degree per this many pulses
        max_degrees: never use more than this many degrees of Legendre polynomial.

        forceNew: whether to do this step, if it appears already to have been done.
        """
        if all(self.p_filt_value_tdc[:] == 0.0) or forceNew:
            LOG.info("chan %d doing time_drift_correct", self.channum)
            attr = getattr(self, attr)
            g = self.cuts.good()
            pk = np.median(attr[g])
            g = np.logical_and(g, np.abs(attr[:]/pk-1) < 0.5)
            w = max(pk/3000., 1.0)
            info = time_drift_correct(self.p_timestamp[g], attr[g], w,
                                      limit=[0.5*pk, 2*pk])
            tnorm = info["normalize"](self.p_timestamp[:])
            corrected = attr[:]*(1+info["model"](tnorm))
            self.p_filt_value_tdc[:] = corrected
            self.time_drift_correct_info = info
        else:
            LOG.info("chan %d skipping time_drift_correct", self.channum)
            corrected, info = self.p_filt_value_tdc[:], {}
        return corrected, info

    def compare_calibrations(self):
        plt.figure()
        for key in self.calibration:
            cal = self.calibration[key]
            try:
                plt.plot(cal.peak_energies, cal.energy_resolutions, 'o', label=key)
            except:
                pass
        plt.legend()
        plt.xlabel("energy (eV)")
        plt.ylabel("energy resolution fwhm (eV)")
        plt.grid("on")
        plt.title("chan %d cal comparison" % self.channum)

    def count_rate(self, goodonly=False, bin_s=60):
        g = self.cuts.good()
        if not goodonly:
            g[:] = True
        if isinstance(bin_s, (int, float)):
            bin_edge = np.arange(self.p_timestamp[g][0], self.p_timestamp[g][-1], bin_s)
        else:
            bin_edge = bin_s
        counts, bin_edge = np.histogram(self.p_timestamp[g], bin_edge)
        bin_centers = bin_edge[:-1]+0.5*(bin_edge[1]-bin_edge[0])
        rate = counts/float(bin_edge[1]-bin_edge[0])

        return bin_centers, rate

    def cut_summary(self):
        boolean_fields = [name.decode() for (name, _) in self.tes_group.boolean_cut_desc if name]

        for c1 in boolean_fields:
            bad1 = self.cuts.bad(c1)
            for c2 in boolean_fields:
                if c1 is c2:
                    continue
                bad2 = self.cuts.bad(c2)
                n_and = np.logical_and(bad1, bad2).sum()
                n_or = np.logical_or(bad1, bad2).sum()
                print("%6d (and) %6d (or) pulses cut by [%s and/or %s]" %
                      (n_and, n_or, c1.upper(), c2.upper()))
        print()
        for cut_name in boolean_fields:
            print("%6d pulses cut by %s" % (self.cuts.bad(cut_name).sum(), cut_name.upper()))
        print("%6d pulses total" % self.nPulses)

    @_add_group_loop
    def auto_cuts(self, nsigma_pt_rms=8.0, nsigma_max_deriv=8.0, pretrig_rms_percentile=None, forceNew=False, clearCuts=True):
        """Compute and apply an appropriate set of automatically generated cuts.

        The peak time and rise time come from the measured most-common peak time.
        The pulse RMS and postpeak-derivative cuts are based on what's observed in
        the (presumably) pulse-free noise file associated with this data file.

        Args:
            nsigma_pt_rms (float):  How big an excursion is allowed in pretrig RMS
                (default 8.0).
            nsigma_max_deriv (float): How big an excursion is allowed in max
                post-peak derivative (default 8.0).
            pretrig_rms_percentile (float): Make upper limit for pretrig_rms at
                least as large as this percentile of the data. I.e., if you
                pass in 99, then the upper limit for pretrig_rms will exclude
                no more than the 1 % largest values. This number is a
                percentage, *not* a fraction. This should not be routinely used
                - it is intended to help auto_cuts work even if there is a
                problem during a data acquisition that causes large drifts in
                noise properties.
            forceNew (bool): Whether to perform auto-cuts even if cuts already exist.
            clearCuts (bool): Whether to clear any existing cuts first (default
                True).

        The two excursion limits are given in units of equivalent sigma from the
        noise file. "Equivalent" meaning that the noise file was assessed not for
        RMS but for median absolute deviation, normalized to Gaussian distributions.

        Returns:
            The cut object that was applied.
        """
        # These are based on function calc_cuts_from_noise in make_preknowledge.py
        # in Galen's project POPE.jl.

        if not (all(self.cuts.good()) or forceNew):
            LOG.info("channel %g skipping auto cuts because cuts exist", self.channum)
            return

        if clearCuts:
            self.clear_cuts()

        # Step 1: peak and rise times
        if self.peak_samplenumber is None:
            self._compute_peak_samplenumber()
        MARGIN = 3  # step at least this many samples forward before cutting.
        peak_time_ms = (MARGIN + self.peak_samplenumber-self.nPresamples)*self.timebase*1000

        # Step 2: analyze *noise* so we know how to cut on pretrig rms postpeak_deriv
        max_deriv = np.zeros(self.noise_records.nPulses)
        pretrigger_rms = np.zeros(self.noise_records.nPulses)
        for first_pnum, end_pnum, _seg_num, data_seg in self.noise_records.datafile.iter_segments():
            max_deriv[first_pnum:end_pnum] = mass.analysis_algorithms.compute_max_deriv(
                data_seg, ignore_leading=0)
            pretrigger_rms[first_pnum:end_pnum] = data_seg[:, :self.nPresamples].std(axis=1)

        # Multiply MAD by 1.4826 to get into terms of sigma, if distribution were Gaussian.
        md_med = np.median(max_deriv)
        pt_med = np.median(pretrigger_rms)
        md_madn = np.median(np.abs(max_deriv-md_med))*1.4826
        pt_madn = np.median(np.abs(pretrigger_rms-pt_med))*1.4826
        md_max = md_med + md_madn*nsigma_max_deriv
        pt_max = max(0.0, pt_med + pt_madn*nsigma_pt_rms)

        # Step 2.5: In the case of pretrig_rms, cut no more than pretrig_rms_percentile percent
        # of the pulses on the upper end. This appears to be appropriate for
        # SLEDGEHAMMER gamma devices, but may not be appropriate in cases where
        # there are many pulses riding on tails, so by default we don't do
        # this.
        if pretrig_rms_percentile != None:
            pt_max = max(pt_max, np.percentile(self.p_pretrig_rms, pretrig_rms_percentile))

        # Step 3: make the cuts
        cuts = mass.core.controller.AnalysisControl(
            peak_time_ms=(0, peak_time_ms*1.25),
            rise_time_ms=(0, peak_time_ms*1.10),
            pretrigger_rms=(None, pt_max),
            postpeak_deriv=(None, md_max),
        )
        self.apply_cuts(cuts, forceNew=True, clear=False)
        self.__save_auto_cuts(cuts)
        return cuts

    def __save_auto_cuts(self, cuts):
        """Store the results of auto-cuts internally and in HDF5."""
        self.saved_auto_cuts = cuts
        g = self.hdf5_group["cuts"].require_group("auto_cuts")
        for attrname in ("peak_time_ms", "rise_time_ms", "pretrigger_rms",
                         "postpeak_deriv"):
            g.attrs[attrname] = cuts.cuts_prm[attrname][1]

    def __load_auto_cuts(self):
        """Load the results of auto-cuts, if any, from HDF5."""
        try:
            g = self.hdf5_group["cuts/auto_cuts"]
        except KeyError:
            return
        cuts = mass.AnalysisControl()
        for attrname in ("peak_time_ms", "rise_time_ms", "pretrigger_rms",
                         "postpeak_deriv"):
            cuts.cuts_prm[attrname] = (None, g.attrs[attrname])
        self.saved_auto_cuts = cuts

    @_add_group_loop
    def smart_cuts(self, threshold=10.0, n_trainings=10000, forceNew=False):
        """Young! Why is there no doc string here??"""
        # first check to see if this had already been done
        if all(self.cuts.good("smart_cuts")) or forceNew:
            from sklearn.covariance import MinCovDet

            mdata = np.vstack([self.p_pretrig_mean[:n_trainings], self.p_pretrig_rms[:n_trainings],
                               self.p_min_value[:n_trainings], self.p_postpeak_deriv[:n_trainings]])
            mdata = mdata.transpose()

            robust = MinCovDet().fit(mdata)

            # It excludes only extreme outliers.
            mdata = np.vstack([self.p_pretrig_mean[...], self.p_pretrig_rms[...],
                               self.p_min_value[...], self.p_postpeak_deriv[...]])
            mdata = mdata.transpose()
            flag = robust.mahalanobis(mdata) > threshold**2

            self.cuts.cut("smart_cuts", flag)
            LOG.info("channel %g ran smart cuts, %g of %g pulses passed",
                     self.channum, self.cuts.good("smart_cuts").sum(), self.nPulses)
        else:
            LOG.info("channel %g skipping smart cuts because it was already done", self.channum)

    @_add_group_loop
    def nearest_neighbor_crosstalk_cuts(self, priorVetoTime, postVetoTime, forceNew=False):
        ''' Uses a list of nearest neighbor channels to cut pulses in current channel based
            on arrival times of pulses in neighboring channels

            Args:
            priorVetoTime (float): amount of time to check for before the pulse arrival time
            postVetoTime (float): amount of time to check for after the pulse arrival time
            forceNew (bool): whether to re-compute the crosstalk cuts (default False)
        '''

        groupName = 'nearest_neighbors'
        if groupName in self.hdf5_group.keys() and (all(self.cuts.good("crosstalk")) or forceNew):
            # Combine all nearest neighbor pairs for this channel into a single list
            combinedNearestNeighbors = np.array([])
            # Loop through nearest neighbor categories
            for neighborCategory in self.hdf5_group['nearest_neighbors']:
                subgroupName = groupName + '/' + neighborCategory
                tempNeighbors = self.hdf5_group[subgroupName].value
                # Remove duplicates, sort
                combinedNearestNeighbors = np.unique(np.append(combinedNearestNeighbors, tempNeighbors).astype(int))

            # Convert from ms input to s used in rest of MASS
            priorVetoTime /= 1000.0
            postVetoTime /= 1000.0

            # Cuts pulses in current channels by comparing to pulse times in neighboring channels
            channum1 = self.channum
            LOG.info('Checking crosstalk between channel %d and neighbors...', channum1)

            # Create uneven histogram edges, with a specified amount of time before and after a photon event
            pulseTimes = self.p_rowcount[:] * self.row_timebase

            # Create start and stop edges around pulses corresponding to veto times
            startEdges = pulseTimes - priorVetoTime
            stopEdges = pulseTimes + postVetoTime
            combinedEdges = np.sort(np.append(startEdges, stopEdges))

            # Initialize array that will include the pulses from all neighboring channels
            neighboringChannelsPulsesList = np.array([])
            # Iterate through all neighboring channels that you will veto against
            for channum2 in combinedNearestNeighbors:
                dsToCompare = self.tes_group.channel[channum2]
                # Combine the pulses from all neighboring channels into a single array
                neighboringChannelsPulsesList = np.append(neighboringChannelsPulsesList, dsToCompare.p_rowcount[:] * dsToCompare.row_timebase)

            # Create a histogram of the neighboring channel pulses using the bin edges from the channel you are flagging
            hist, bin_edges = np.histogram(neighboringChannelsPulsesList, bins=combinedEdges)

            # Even corresponds to bins with a photon in channel 1 (crosstalk), odd are empty bins (no crosstalk)
            badCountsHist = hist[::2]

            # Even only histogram indices map directly to previously good flagged pulse indices for channel 1
            isCrosstalking = badCountsHist > 0.0

            # Apply crosstalk cuts
            self.cuts.cut("crosstalk", isCrosstalking)

        else:
            LOG.info("channel %d skipping crosstalk cuts because it was already done", self.channum)

    @_add_group_loop
    def set_nearest_neighbors_list(self, mapFilename, nearestNeighborCategory = 'physical', forceNew=False):
        ''' Finds the nearest neighbors in a given space for all channels in a data set

        Args:
        mapFilename (str): Location of map file in the following format
            Column 0 - list of channel numbers.
            Remaining column(s) - coordinates that define a particular column in a given space.
                For example, can be the row and column number in a physical space
                or the frequency order number in a frequency space (umux readout).
        nearestNeighborCategory (str): name used to categorize the type of nearest neighbor.
            This will be the name given to the subgroup of the hdf5 file under the nearest_neighbor group.
            This will also be a key for dictionary nearest_neighbors_dictionary
        forceNew (bool): whether to re-compute nearest neighbors list if it exists (default False)
        '''

        # Create hdf5 group for nearest neighbors
        h5grp = self.hdf5_group.require_group('nearest_neighbors')

        # Check to see if if data set already exists or if forceNew is set to True
        if 'nearest_neighbors/' + nearestNeighborCategory not in self.hdf5_group or forceNew:

            # Load channel numbers and positions from map file, define number of dimensions
            mapData = np.loadtxt(mapFilename, dtype=int)
            channelNumbers = mapData[:,0]
            positionValues = mapData[:,1:]
            nDims = positionValues.shape[1]

            # Extract channel number and position of current channel
            channum = self.channum
            channelPos = np.array(positionValues[channum == channelNumbers][0],ndmin=1)

            # Initialize array for storing nearest neighbors of current channel
            channelsList = np.array([]).astype(int)

            '''
            Returns the channel number of a neighboring position after checking for goodness

            Args:
            positionToCompare (int array) - position to check for nearest neighbor match
            '''
            def process_matching_channel(positionToCompare):
                # Find the channel number corresponding to the compare position
                channelToCompare = channelNumbers[np.all(positionToCompare == positionValues, axis=1)]
                # If the new position exists in map file and the channel to compare to is good, return the channel number
                if (positionToCompare in positionValues) & (channelToCompare in self.tes_group.good_channels):
                    return channelToCompare
                # Return an empty array if not actually a good nearest neighbor
                else:
                    return np.array([], dtype=int)

            # Check the lower and upper position for each dimension in the given space
            for iDim in range(nDims):
                lowerPosition = np.array(channelPos)
                lowerPosition[iDim] -= 1
                channelsList = np.append(channelsList, process_matching_channel(lowerPosition))
                upperPosition = np.array(channelPos)
                upperPosition[iDim] += 1
                channelsList = np.append(channelsList, process_matching_channel(upperPosition))

            # Save nearest neighbor data into hdf5 file in nearestNeighborCategory subgroup
            if nearestNeighborCategory in h5grp:
                del h5grp[nearestNeighborCategory]
            h5grp.create_dataset(nearestNeighborCategory, data = channelsList)

        # Also save the data into a dictionary with nearestNeighborCategory as the key
        self.nearest_neighbors_dictionary[nearestNeighborCategory] = h5grp[nearestNeighborCategory].value




# Below here, these are functions that we might consider moving to Cython for speed.
# But at any rate, they do not require any MicrocalDataSet attributes, so they are
# pure functions, not methods.

def _phasecorr_find_alignment(phase_indicator, pulse_heights, peak, delta_ph,
                              method2017=False, nf=10, kernel_width=2.0):
    """Find the way to align (flatten) `pulse_heights` as a function of `phase_indicator`
    working only within the range [peak-delta_ph, peak+delta_ph].

    If `method2017`, then use a scipy LSQUnivariateSpline with a reasonable (?)
    number of knots. Otherwise, use `nf` bins in `phase_indicator`, shifting each
    such that its `pulse_heights` histogram best aligns with the overall histogram.
    `method2017==False` (the 2015 way) is subject to particular problems when
    there are not a lot of counts in the peak.
    """
    phrange = np.array([-delta_ph, delta_ph])+peak
    use = np.logical_and(np.abs(pulse_heights[:]-peak) < delta_ph,
                         np.abs(phase_indicator) < 2)
    low_phase, median_phase, high_phase = \
        sp.stats.scoreatpercentile(phase_indicator[use], [3, 50, 97])

    if method2017:
        x = phase_indicator[use]
        y = pulse_heights[use]
        NBINS = len(x) // 300
        NBINS = max(2, NBINS)
        NBINS = min(12, NBINS)

        bin_edge = np.linspace(low_phase, high_phase, NBINS+1)
        dx = high_phase-low_phase
        bin_edge[0] -= dx
        bin_edge[-1] += dx
        bins = np.digitize(x, bin_edge)-1

        knots = np.zeros(NBINS, dtype=float)
        yknot = np.zeros(NBINS, dtype=float)
        iter1 = 0
        for i in range(NBINS):
            yu = y[bins == i]
            yo = y[bins != i]
            knots[i] = np.median(x[bins == i])

            def target(shift):
                return mass.mathstat.entropy.laplace_cross_entropy(yo, yu+shift, kernel_width)
            brack = 0.002*np.array([-1, 1], dtype=float)
            sbest, KLbest, niter, _ = sp.optimize.brent(target, (), brack=brack, full_output=True, tol=3e-4)
            iter1 += niter
            yknot[i] = sbest

        yknot -= yknot.mean()
        correction1 = mass.CubicSpline(knots, yknot)
        ycorr = y + correction1(x)

        iter2 = 0
        yknot2 = np.zeros(NBINS, dtype=float)
        for i in range(NBINS):
            yu = ycorr[bins == i]
            yo = ycorr[bins != i]

            def target(shift):
                return mass.mathstat.entropy.laplace_cross_entropy(yo, yu+shift, kernel_width)
            brack = 0.002*np.array([-1, 1], dtype=float)
            sbest, KLbest, niter, _ = sp.optimize.brent(target, (), brack=brack, full_output=True, tol=1e-4)
            iter2 += niter
            yknot2[i] = sbest
        correction = mass.CubicSpline(knots, yknot+yknot2)
        H0 = mass.mathstat.entropy.laplace_entropy(y, kernel_width)
        H1 = mass.mathstat.entropy.laplace_entropy(ycorr, kernel_width)
        H2 = mass.mathstat.entropy.laplace_entropy(y+correction(x), kernel_width)
        LOG.info("Laplace entropy before/middle/after: %.4f, %.4f %.4f (%d+%d iterations, %d phase groups)",
                 H0, H1, H2, iter1, iter2, NBINS)

        curve = mass.CubicSpline(knots-median_phase, peak-(yknot+yknot2))
        return curve, median_phase

    # Below here is "method2015", in which we perform correlations and fit to quadratics.
    # It is basically unsuitable for small statistics, so it is no longer preferred.
    Pedges = np.linspace(low_phase, high_phase, nf+1)
    Pctrs = 0.5*(Pedges[1:]+Pedges[:-1])
    dP = 2.0/nf
    Pbin = np.digitize(phase_indicator, Pedges)-1

    NBINS = 200
    hists = np.zeros((nf, NBINS), dtype=float)
    for i, P in enumerate(Pctrs):
        use = (Pbin == i)
        c, b = np.histogram(pulse_heights[use], NBINS, phrange)
        hists[i] = c
    bctr = 0.5*(b[1]-b[0])+b[:-1]

    kernel = np.mean(hists, axis=0)[::-1]
    peaks = np.zeros(nf, dtype=float)
    for i in range(nf):
        # Find the PH of this ridge by fitting quadratic to the correlation
        # of histogram #i and the mean histogram, then finding its local max.
        conv = sp.signal.fftconvolve(kernel, hists[i], 'same')
        m = conv.argmax()
        if conv[m] <= 0:
            continue
        p = np.poly1d(np.polyfit(bctr[m-2:m+3], conv[m-2:m+3], 2))
        # p = np.poly1d(np.polyfit(b[m-2:m+3], conv[m-2:m+3], 2))
        peak = p.deriv(m=1).r[0]
        # if peak < bctr[m-2]: peak = bctr[m]
        # if peak > bctr[m+2]: peak = bctr[m]
        peaks[i] = peak
    # use = peaks>0
    # if use.sum() >= 2:
    #     curve = mass.mathstat.interpolate.CubicSpline(Pctrs[use]-median_phase, peaks[use])
    # else:
    #     curve = mass.mathstat.interpolate.CubicSpline(Pctrs-median_phase, np.mean(phrange)+np.zeros_like(Pctrs))
    curve = mass.mathstat.interpolate.CubicSpline(Pctrs-median_phase, peaks)
    return curve, median_phase


def _phase_corrected_filtvals(phase, uncorrected, corrections):
    """Apply phase correction to `uncorrected`.

    Returns:
        the corrected vector.
    """
    NC = len(corrections)
    NP = len(phase)
    assert NP == len(uncorrected)
    phase = np.asarray(phase)
    uncorrected = np.asarray(uncorrected)

    ph = np.hstack([0] + [c(0) for c in corrections])
    assert (ph[1:] > ph[:-1]).all()  # corrections should be sorted by PH
    corr = np.zeros((NC+1, NP), dtype=float)
    for i, c in enumerate(corrections):
        corr[i+1] = c(0) - c(phase)

    # Now apply the appropriate correction (a linear interp between 2 neighboring values)
    corrected = uncorrected.copy()
    binnum = np.digitize(uncorrected, ph)
    for b in range(NC):
        # Don't correct binnum=0, which would be negative PH
        use = (binnum == 1+b)
        if b+1 == NC:  # For the last bin, extrapolate
            use = (binnum >= 1+b)
        if use.sum() == 0:
            continue
        frac = (uncorrected[use]-ph[b])/(ph[b+1]-ph[b])
        corrected[use] += frac*corr[b+1, use] + (1-frac)*corr[b, use]
    return corrected


def time_drift_correct(time, uncorrected, w, sec_per_degree = 2000,
                       pulses_per_degree = 2000, max_degrees = 20, limit=None):
    """Compute a time-based drift correction that minimizes the spectral entropy.

    Args:
        time: The "time-axis". Correction will be a low-order polynomial in this.
        uncorrected: A filtered pulse height vector. Same length as indicator.
            Assumed to have some gain that is linearly related to indicator.
        w: the kernel width for the Laplace KDE density estimator
        sec_per_degree: assign as many as one polynomial degree per this many seconds
        pulses_per_degree: assign as many as one polynomial degree per this many pulses
        max_degrees: never use more than this many degrees of Legendre polynomial.
        limit: The [lower,upper] limit of uncorrected values over which entropy is
            computed (default None).

    The entropy will be computed on corrected values only in the range
    [limit[0], limit[1]], so limit should be set to a characteristic large value
    of uncorrected. If limit is None (the default), then it will be computed as
    25%% larger than the 99%%ile point of uncorrected.

    Possible improvements in the future:
    * Move this routine to Cython.
    * Allow the parameters to be function arguments with defaults: photons per
      degree of freedom, seconds per degree of freedom, and max degrees of freedom.
    * Figure out how to span the available time with more than one set of legendre
      polynomials, so that we can have more than 20 d.o.f. eventually, for long runs.
    """
    if limit is None:
        pct99 = sp.stats.scoreatpercentile(uncorrected, 99)
        limit = [0, 1.25 * pct99]

    use = np.logical_and(uncorrected > limit[0], uncorrected < limit[1])
    tmin, tmax = np.min(time), np.max(time)

    def normalize(t):
        return (t-tmin)/(tmax-tmin)*2-1

    info = {
        "tmin": tmin,
        "tmax": tmax,
        "normalize": normalize,
        }

    dtime = tmax-tmin
    N = len(time)
    ndeg = int(np.minimum(dtime/sec_per_degree, N/pulses_per_degree))
    ndeg = min(ndeg, max_degrees)
    ndeg = max(ndeg, 1)
    phot_per_degree = N/float(ndeg)

    if phot_per_degree >= 2*pulses_per_degree:
        downsample = int(phot_per_degree/pulses_per_degree)
        time = time[::downsample]
        uncorrected = uncorrected[::downsample]
        N = len(time)
    else:
        downsample = 1

    LOG.info("Using %2d degrees for %6d photons (after %d downsample)", ndeg, N, downsample)
    LOG.info("That's %6.1f photons per degree, and %6.1f seconds per degree.", N/float(ndeg), dtime/ndeg)

    def model1(pi, i, param, basis):
        pcopy = np.array(param)
        pcopy[i] = pi
        return 1 + np.dot(basis.T, pcopy)

    def cost1(pi, i, param, y, w, basis):
        return laplace_entropy(y*model1(pi, i, param, basis), w=w)

    param = np.zeros(ndeg, dtype=float)
    xnorm = np.asarray(normalize(time), dtype=float)
    basis = np.vstack([sp.special.legendre(i+1)(xnorm) for i in range(ndeg)])

    fc = 0
    model = np.poly1d([0])
    info["coefficients"] = np.zeros(ndeg, dtype=float)
    for i in range(ndeg):
        result, fval, iter, funcalls = sp.optimize.brent(
            cost1, (i, param, uncorrected, w, basis), [-.001, .001], tol=1e-5, full_output=True)
        param[i] = result
        fc += funcalls
        model += sp.special.legendre(i+1) * result
        info["coefficients"][i] = result
    info["funccalls"] = fc

    xk = np.linspace(-1, 1, 1+2*ndeg)
    model2 = mass.mathstat.interpolate.CubicSpline(xk, model(xk))
    H1 = laplace_entropy(uncorrected, w=w)
    H2 = laplace_entropy(uncorrected*(1+model(xnorm)), w=w)
    H3 = laplace_entropy(uncorrected*(1+model2(xnorm)), w=w)
    if H2 <= 0 or H2-H1 > 0.0:
        model = np.poly1d([0])
    elif H3 <= 0 or H3-H2 > .00001:
        model2 = model

    info["entropies"] = (H1, H2, H3)
    info["model"] = model
    return info
