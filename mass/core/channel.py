"""
Single-channel classes:

* NoiseRecords: encapsulate a file with noise records
* PulseRecords: encapsulate a file with pulse records
* MicrocalDataSet: encapsulate basically everything about 1 channel's pulses and noise
"""

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import inspect
import os
import sys
import traceback
import sklearn
from packaging import version
from deprecation import deprecated
from indexedproperty import indexedproperty

# MASS modules
import mass.mathstat.power_spectrum
import mass.mathstat.interpolate
import mass.mathstat.robust
import mass.core.analysis_algorithms
from . import phase_correct
from .pulse_model import PulseModel
from .channel_summarize import summarize_data_numba

from mass.core.cut import Cuts
from mass.core.files import VirtualFile, LJHFile
from mass.core.optimal_filtering import FilterMaker
from mass.core.utilities import show_progress
from mass.calibration.energy_calibration import EnergyCalibration
from mass.calibration.algorithms import EnergyCalibrationAutocal
from mass.mathstat.entropy import laplace_entropy
import mass.off

from mass.core import ljh_util
import logging
LOG = logging.getLogger("mass")


class NoiseRecords:
    """Encapsulate a set of noise records.

    The noise records can either be assumed continuous or arbitrarily separated in time.
    """

    DEFAULT_MAXSEGMENTSIZE = 32000000
    ALLOWED_TYPES = ("ljh", "virtual")

    def __init__(self, filename, records_are_continuous=False, use_records=None,
                 maxsegmentsize=None):
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
        """

        if maxsegmentsize is not None:
            self.maxsegmentsize = maxsegmentsize
        else:
            self.maxsegmentsize = self.DEFAULT_MAXSEGMENTSIZE

        self.channum = ljh_util.ljh_channum(filename)
        self.nSamples = self.nPresamples = self.nPulses = 0
        self.n_segments = 0
        self.timebase = 0.0
        self.timestamp_offset = 0
        self.subframe_divisions = 1
        self.subframe_offset = 0

        self.datafile = None
        self.data = None
        self.saved_auto_cuts = None

        self.__open_file(filename, use_records=use_records)
        self.continuous = records_are_continuous
        self.noise_psd = None

    def set_hdf5_group(self, hdf5_group):
        if hdf5_group is None:
            raise ValueError("hdf5_group should not be None")

        self.hdf5_group = hdf5_group

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum", "n_segments"):
            setattr(self, attr, getattr(self.datafile, attr))
            self.hdf5_group.attrs[attr] = getattr(self.datafile, attr)

        self.autocorrelation = self.hdf5_group.require_dataset(
            "autocorrelation", shape=(self.nSamples,), dtype=np.float64)
        nfreq = 1 + self.nSamples // 2
        self.noise_psd = self.hdf5_group.require_dataset(
            "noise_psd", shape=(nfreq,), dtype=np.float64)

    def __open_file(self, filename, use_records=None, file_format=None):
        """Detect the filetype and open it."""

        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = "virtual"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in self.ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of {}".format(",".join(self.ALLOWED_TYPES)))

        if file_format == "ljh":
            self.datafile = LJHFile.open(filename)
        elif file_format == "virtual":
            vfile = filename  # Aha!  It must not be a string
            self.datafile = vfile
            self.datafile.segmentsize = vfile.nPulses * (6 + 2 * vfile.nSamples)
            filename = "Virtual file"
        else:
            raise RuntimeError("It is a programming error to get here")
        self.filename = filename
        self.records_per_segment = self.datafile.segmentsize // (6 + 2 * self.datafile.nSamples)
        self.data = self.datafile.alldata

        if use_records is not None:
            if use_records < self.datafile.nPulses:
                self.datafile.nPulses = use_records
                self.datafile.n_segments = use_records // self.records_per_segment

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

    def compute_power_spectrum(self, window=None, plot=True,
                               max_excursion=1000):
        if window is None:
            window = mass.mathstat.power_spectrum.hann
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

        if not self.continuous and seg_length is not None and seg_length != self.nSamples:
            raise ValueError(
                "This NoiseRecords doesn't have continuous noise; it can't be resegmented.")

        if seg_length is None:
            seg_length = self.nSamples

        spectrum = mass.mathstat.power_spectrum.PowerSpectrum(seg_length // 2, dt=self.timebase)
        if window is None:
            window = np.ones(seg_length)
        else:
            window = window(seg_length)

        data = self.datafile.alldata
        if self.continuous and seg_length is not None:
            n = np.prod(data.shape)
            n -= n % seg_length
            data = data[:n].reshape((n // seg_length, seg_length))

        for d in data:
            y = d - d.mean()
            if y.max() - y.min() < max_excursion and len(y) == spectrum.m2:
                spectrum.addDataSegment(y, window=window)

        freq = spectrum.frequencies()
        psd = spectrum.spectrum()
        if self.hdf5_group is not None:
            self.noise_psd[:] = psd
            self.noise_psd.attrs["delta_f"] = freq[1] - freq[0]
        else:
            self.noise_psd = psd
        return spectrum

    def compute_fancy_power_spectra(self, window=mass.mathstat.power_spectrum.hann,
                                    plot=True, seglength_choices=None):
        """Compute a power spectrum using a few long segments for the low freq.
        and many short ones for the higher frequencies.
        """
        assert self.continuous

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
            lowest_freq = np.array([1. / (s.dt * s.m2) for s in spectra])

            start_freq = 0.0
            for i, s in enumerate(spectra):
                x, y = s.frequencies(), s.spectrum()
                if i == len(spectra) - 1:
                    good = x >= start_freq
                else:
                    good = np.logical_and(x >= start_freq, x < 10 * lowest_freq[i + 1])
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
        freq = np.arange(1, 1 + len(yvalue)) * self.noise_psd.attrs['delta_f']
        axis.plot(freq, yvalue, **kwarg)
        plt.loglog()
        axis.grid()
        axis.set_xlim([10, 3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel(r"Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title(f"Noise power spectrum for {self.filename}")

    def _compute_continuous_autocorrelation(self, n_lags=None, data_samples=None,
                                            max_excursion=1000):
        if data_samples is None:
            data_samples = [0, self.nSamples * self.nPulses]
        n_data = data_samples[1] - data_samples[0]

        samples_per_segment = self.records_per_segment * self.nSamples
        if n_lags is None:
            n_lags = samples_per_segment
        n_lags = min(n_lags, samples_per_segment)

        def padded_length(n):
            """Return a sensible number in the range [n, 2n] which is not too
            much larger than n, yet is good for FFTs.

            Returns:
                A number: (1, 3, or 5)*(a power of two), whichever is smallest.
            """
            pow2 = np.round(2**np.ceil(np.log2(n)))
            if n == pow2:
                return int(n)
            elif n > 0.75 * pow2:
                return int(pow2)
            elif n > 0.625 * pow2:
                return int(np.round(0.75 * pow2))
            else:
                return int(np.round(0.625 * pow2))

        # When there are 10 million data points and only 10,000 lags wanted,
        # it's hugely inefficient to compute the full autocorrelation, especially
        # in memory.  Instead, compute it on chunks several times the length of the desired
        # correlation, and average.
        CHUNK_MULTIPLE = 15
        if n_data < CHUNK_MULTIPLE * n_lags:
            n_lags = n_data // CHUNK_MULTIPLE
            if n_lags < self.nSamples:
                msg = f"There are not enough data to compute at least {self.nSamples} lags."
                msg += f"\nn_data={n_data}, n_lags * {CHUNK_MULTIPLE} = {CHUNK_MULTIPLE * n_lags}"
                raise ValueError(msg)

        # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
        # unwanted wraparound in the autocorrelation.
        # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
        chunksize = CHUNK_MULTIPLE * n_lags
        padsize = n_lags
        padded_data = np.zeros(padded_length(padsize + chunksize), dtype=float)

        ac = np.zeros(n_lags, dtype=float)

        entries = 0.0

        first, last = data_samples
        data = self.datafile.alldata[first:last].ravel()
        Nchunks = np.prod(data.shape) // chunksize
        datachunks = data[:Nchunks * chunksize].reshape(Nchunks, chunksize)
        for data in datachunks:
            padded_data[:chunksize] = data - data.mean()
            padded_data[chunksize:] = 0.0
            if np.abs(padded_data).max() > max_excursion:
                continue

            ft = np.fft.rfft(padded_data)
            ft[0] = 0  # this redundantly removes the mean of the data set
            power = (ft * ft.conj()).real
            acsum = np.fft.irfft(power)
            ac += acsum[:n_lags]
            entries += 1.0

        if entries == 0:
            raise Exception(
                "Apparently all chunks had excusions, so no autocorrelation was computed")

        ac /= entries
        ac /= (np.arange(chunksize, chunksize - n_lags + 0.5, -1.0, dtype=float))
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
            self._compute_broken_autocorrelation(n_lags=n_lags, data_samples=data_samples,
                                                 max_excursion=max_excursion)

        if self.hdf5_group is not None:
            grp = self.hdf5_group.require_group(f"reclen{n_lags}")
            ds = grp.require_dataset("autocorrelation", shape=(n_lags,), dtype=np.float64)
            ds[:] = self.autocorrelation[:]

        if plot:
            self.plot_autocorrelation()

    def _compute_broken_autocorrelation(self, n_lags=None, data_samples=None,
                                        max_excursion=1000):
        if n_lags is None:
            n_lags = self.nSamples
        if n_lags > self.nSamples:
            raise ValueError("The autocorrelation can't be computed for "
                             f"n_lags>nsamp={self.nSamples} when data are not continuous")

        if data_samples is None:
            data_samples = [0, self.nSamples * self.nPulses]

        records_used = samples_used = 0
        ac = np.zeros(self.nSamples, dtype=float)
        first = data_samples[0]
        idx_first = first // self.nSamples
        if first % self.nSamples > 0:
            idx_first += 1
        idx_last = data_samples[1] // self.nSamples

        for i in range(idx_first, idx_last):
            data = 1.0 * self.datafile.alldata[i]
            if data.max() - data.min() > max_excursion:
                continue
            data -= data.mean()

            ac += np.correlate(data, data, 'full')[self.nSamples - 1:]
            samples_used += self.nSamples
            records_used += 1

        ac /= records_used
        ac /= self.nSamples - np.arange(self.nSamples, dtype=float)
        if n_lags < self.nSamples:
            ac = ac[:n_lags]
        self.autocorrelation[:] = ac

    def plot_autocorrelation(self, axis=None, color='blue', label=None):
        """Plot the autocorrelation function."""
        if all(self.autocorrelation[:] == 0):
            LOG.info("Autocorrelation must be computed before it can be plotted")
            self.compute_autocorrelation(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        t = self.timebase * 1e3 * np.arange(len(self.autocorrelation))
        axis.plot(t, self.autocorrelation[:], label=label, color=color)
        axis.plot([0], [self.autocorrelation[0]], 'o', color=color)
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel(r"Autocorrelation (counts$^2$)")


class PulseRecords:
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
        self.n_segments = 0
        self.segmentsize = 0
        self.pulses_per_seg = 0
        self.timebase = None
        self.timestamp_offset = 0
        self.channum = ljh_util.ljh_channum(filename)

        self.datafile = None
        self.__open_file(filename, file_format=file_format)

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
            raise ValueError("file_format must be None or one of {}".format(",".join(self.ALLOWED_TYPES)))

        if file_format == "ljh":
            self.datafile = LJHFile.open(filename)
        elif file_format == "virtual":
            vfile = filename  # Aha!  It must not be a string
            self.datafile = vfile
        else:
            raise RuntimeError("It is a programming error to get here")

        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum",
                     "n_segments", "pulses_per_seg", "segmentsize", "timestamp_offset",
                     "subframe_offset", "subframe_divisions"):
            setattr(self, attr, getattr(self.datafile, attr))

    def __str__(self):
        line1 = f"{self.__class__.__name__} path '{self.filename}'\n"
        line2 = f"{self.nSamples} samples ({self.nPresamples} pretriggger) at {1e6 * self.timebase:.2f} µs sample time"
        return "\n".join((line1, line2))

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.filename}')"

    def set_segment_size(self, seg_size):
        """Update the underlying file's segment size in bytes."""
        self.datafile.set_segment_size(seg_size)
        self.n_segments = self.datafile.n_segments
        self.pulses_per_seg = self.datafile.pulses_per_seg
        self.segmentsize = self.datafile.segmentsize

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = PulseRecords(self.filename)
        c.__dict__.update(self.__dict__)
        c.datafile = self.datafile.copy()
        return c


class GroupLooper:
    """A mixin class to allow TESGroup objects to hold methods that loop over
    their constituent channels. (Has to be a mixin, in order to break the import
    cycle that would otherwise occur.)"""
    pass


def _add_group_loop():
    """Add MicrocalDataSet method `method` to GroupLooper (and hence, to TESGroup).

    This is a decorator to add before method definitions inside class MicrocalDataSet.
    Usage is:

    class MicrocalDataSet(...):
        ...

        @_add_group_loop()
        def awesome_fuction(self, ...):
            ...
    """
    is_running_tests = "pytest" in sys.modules

    def decorator(method):
        method_name = method.__name__

        def wrapper(self, *args, **kwargs):
            rethrow = kwargs.pop("_rethrow", is_running_tests)  # always throw errors when testing
            for ds in self:
                try:
                    method(ds, *args, **kwargs)
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    s = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    if rethrow:
                        raise
                    self.set_chan_bad(ds.channum, f"failed {method_name} with {e}\ntraceback:\n{s}")

        wrapper.__name__ = method_name

        # Generate a good doc-string.
        lines = [f"Loop over self, calling the {method_name}(...) method for each channel."]
        try:
            argtext = inspect.signature(method)  # Python 3.3 and later
        except AttributeError:
            arginfo = inspect.getargspec(method)
            argtext = inspect.formatargspec(*arginfo)

        if method.__doc__ is None:
            lines.append(f"\n{method_name}{argtext} has no docstring")
        else:
            lines.append(f"\n{method_name}{argtext} docstring reads:")
            lines.append(method.__doc__)
        wrapper.__doc__ = "\n".join(lines)

        setattr(GroupLooper, method_name, wrapper)
        return method
    return decorator


class MicrocalDataSet:  # noqa: PLR0904
    """Represent a single microcalorimeter's PROCESSED data."""

    # Attributes that all such objects must have.
    expected_attributes = ("nSamples", "nPresamples", "nPulses", "timebase", "channum",
                           "timestamp_offset", "subframe_divisions", "subframe_offset")
    HDF5_CHUNK_SIZE = 256

    @property
    def invert_data(self):
        return self._invert_data

    @invert_data.setter
    def invert_data(self, is_inverted):
        if self.pulse_records is not None:
            self.pulse_records.datafile.invert_data = is_inverted
        self._invert_data = is_inverted

    @indexedproperty
    def data(self, key):
        return self.pulse_records.datafile[key]

    @property
    def alldata(self):
        return self.pulse_records.datafile.alldata

    def __init__(self, pulserec_dict, tes_group=None, hdf5_group=None, invert_data=False):
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
        self._invert_data = invert_data
        self.nSamples = 0
        self.nPresamples = 0
        self.nPulses = 0
        self.timebase = 0.0
        self.channum = None
        self.timestamp_offset = 0
        self.subframe_divisions = None
        self.subframe_offset = None
        self.subframe_timebase = None

        self.filter = None
        self.lastUsedFilterHash = -1
        self.drift_correct_info = {}
        self.phase_correct_info = {}
        self.time_drift_correct_info = {}
        self.noise_autocorr = None
        self.noise_demodulated = None
        self.calibration = {}

        for a in self.expected_attributes:
            setattr(self, a, pulserec_dict[a])
        self.filename = pulserec_dict.get('filename', 'virtual data set')
        self.pretrigger_ignore_samples = 0  # Cut this long before trigger in computing pretrig values
        self.cut_pre = 0  # Number of presamples to ignore at start of pulse
        self.cut_post = 0  # Number of samples to ignore at end of pulse
        self.peak_samplenumber = None   # Look for retriggers only after this time.
        self.index = None   # Index in the larger TESGroup object
        self.last_used_calibration = None

        self.pulse_records = None
        self.times = None
        self.subframecount = None

        self.number_of_rows = None
        self.row_number = None
        self.number_of_columns = None
        self.column_number = None

        self.tes_group = tes_group

        try:
            self.hdf5_group = hdf5_group
            if "npulses" not in self.hdf5_group.attrs:  # to allow TESGroupHDF5 with in read only mode
                self.hdf5_group.attrs['npulses'] = self.nPulses
            elif self.hdf5_group.attrs['npulses'] != self.nPulses:
                msg = f"""Could not use the existing HDF5 file, which has {self.hdf5_group.attrs["npulses"]} pulses,
while the data file has {self.nPulses} pulses in channel {self.channum}.

Try creating with the argument mass.TESGroup(..., overwite_hdf5_file=True)
"""
                raise ValueError(msg)
            if "channum" not in self.hdf5_group.attrs:  # to allow TESGroupHDF5 with in read only mode
                self.hdf5_group.attrs['channum'] = self.channum
            else:
                assert self.hdf5_group.attrs['channum'] == self.channum
        except KeyError:
            self.hdf5_group = None

        self.__setup_vectors(npulses=self.nPulses)
        self.__load_filters_from_hdf5()
        self.__load_cals_from_hdf5()
        self.__load_auto_cuts()
        self.__load_corrections()

    @property
    def _filter_type(self):
        try:
            if isinstance(self.filter, mass.FilterATS):
                return "ats"
            if isinstance(self.filter, mass.Filter5Lag):
                return "5lag"
        except AttributeError:
            return None
        return None

    def toOffStyle(self):
        a = self._makeNumpyArray()
        return mass.off.channels.ChannelFromNpArray(a,
                                                    channum=self.channum,
                                                    shortname=self.shortname,
                                                    experimentStateFile=self.tes_group.experimentStateFile)

    def _makeNumpyArray(self, fields=None, prefix="p_"):
        if fields is None:
            fields = [k for k in self.__dict__ if k.startswith(prefix)]
        _dtypes = [self.__dict__[k].dtype for k in fields]
        dtlist = list(zip(fields, _dtypes))
        dtype = np.dtype(dtlist)
        a = np.zeros(len(self.__dict__[fields[0]]), dtype)
        for k in fields:
            a[k] = self.__dict__[k][:]
        return a

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
        int64_fields = ()
        bool_fields = ('shift1',)
        for dtype, fieldnames in ((np.float64, float64_fields),
                                  (np.float32, float32_fields),
                                  (np.uint16, uint16_fields),
                                  (bool, bool_fields),
                                  (np.int64, int64_fields)):
            for field in fieldnames:
                self.__dict__[f'p_{field}'] = h5grp.require_dataset(
                    field, shape=(npulses,), dtype=dtype)

        # Workaround for fact that this value changed names in Feb 2024 from "rowcount" to "subframecount"
        if "rowcount" in h5grp:
            self.p_subframecount = h5grp.require_dataset(
                "rowcount", shape=(npulses,), dtype=np.int64)
        else:
            self.p_subframecount = h5grp.require_dataset(
                "subframecount", shape=(npulses,), dtype=np.int64)

        if "peak_samplenumber" in self.p_peak_index.attrs:
            self.peak_samplenumber = self.p_peak_index.attrs["peak_samplenumber"]

        # Other vectors needed per-channel
        self.average_pulse = h5grp.require_dataset(
            'average_pulse', shape=(self.nSamples,), dtype=np.float32)
        self.noise_autocorr = h5grp.require_dataset(
            'noise_autocorr', shape=(self.nSamples,), dtype=np.float64)
        nfreq = 1 + self.nSamples // 2
        self.noise_psd = h5grp.require_dataset('noise_psd', shape=(nfreq,),
                                               dtype=np.float64)

        grp = self.hdf5_group.require_group('cuts')
        self.cuts = Cuts(self.nPulses, self.tes_group, hdf5_group=grp)

    def __load_filters_from_hdf5(self, overwrite=False):
        if 'filters' not in self.hdf5_group:
            return
        filter_group = self.hdf5_group['filters']

        fmax = filter_group.attrs.get("fmax", None)
        f_3db = filter_group.attrs.get("f_3db", None)

        version = filter_group.attrs.get("version", 0)
        if version not in {1, 2}:  # don't support the older HDF5 without version numbers
            return

        filter_type = filter_group.attrs.get("filter_type", "ats")
        if isinstance(filter_type, bytes):
            filter_type = filter_type.decode()

        if version == 1:
            # Version 1 avg_signal was an attribute until Nov 2021, when we fixed #208.
            # Try to read as a dataset, then as attribute so that old HDF5 files still work.
            if filter_type == "ats":
                # arrival time safe filter can be shorter than records by 1 sample, or equal in length
                try:
                    avg_signal = filter_group["avg_signal"][()]
                except KeyError:
                    avg_signal = filter_group.attrs["avg_signal"][()]
                aterms = filter_group["filt_aterms"][()].ravel()
                modelpeak = np.max(avg_signal)
                maker = FilterMaker(avg_signal, self.nPresamples - self.pretrigger_ignore_samples,
                                    self.noise_autocorr, self.noise_psd, aterms,
                                    sample_time_sec=self.timebase, peak=modelpeak)
                self.filter = maker.compute_ats(fmax=fmax, f_3db=f_3db)
        if version == 2:
            if filter_type == "ats":
                values = filter_group["values"][:]
                dt_values = filter_group["dt_values"][:]
                variance = filter_group["values"].attrs.get("variance", 0.0)
                vdv = filter_group["values"].attrs.get("predicted_v_over_dv", 0.0)
                peak = filter_group["values"].attrs.get("nominal_peak", 0.0)
                self.filter = mass.FilterATS(values, peak, variance, vdv, dt_values, None, None, None,
                                             1, fmax=fmax, f_3db=f_3db)

            else:
                modelpeak = np.max(self.average_pulse)
                maker = FilterMaker(self.average_pulse[:], self.nPresamples - self.pretrigger_ignore_samples,
                                    self.noise_autocorr, self.noise_psd,
                                    sample_time_sec=self.timebase, peak=modelpeak)
                if filter_type == "5lag":
                    self.filter = maker.compute_5lag(fmax=fmax, f_3db=f_3db)
                elif filter_type == "fourier":
                    self.filter = maker.compute_fourier(fmax=fmax, f_3db=f_3db)
                else:
                    raise Exception(f"filter_type={filter_type}, must be `ats`, `5lag`, or `fourier`")

    def __load_cals_from_hdf5(self, overwrite=False):
        """Load all calibrations in self.hdf5_group["calibration"] into the dict
        self.calibration.
        """
        hdf5_cal_group = self.hdf5_group.require_group('calibration')
        for k in hdf5_cal_group.keys():
            if not overwrite:
                if k in self.calibration.keys():
                    raise ValueError(
                        "trying to load over existing calibration, consider passing overwrite=True")
            self.calibration[k] = EnergyCalibration.load_from_hdf5(hdf5_cal_group, k)

    def __load_corrections(self):
        # drift correction should be loaded here, but I don't think it is loaded at all, here or anywhere!
        if "phase_correction" in self.hdf5_group:
            self.phaseCorrector = phase_correct.PhaseCorrector.fromHDF5(
                self.hdf5_group, name="phase_correction")

    @property
    @deprecated(deprecated_in="0.8.2", details="Use subframecount, which is equivalent but better named")
    def rowcount(self):
        return self.subframecount

    @property
    def p_peak_time(self):
        peak_index = np.asarray(self.p_peak_index[:], dtype=float)
        return (peak_index - self.nPresamples) * self.timebase

    @property
    def subframes_after_last_external_trigger(self):
        try:
            return self.hdf5_group["subframes_after_last_external_trigger"]
        except KeyError:
            raise ValueError(
                "run tes_group.calc_external_trigger_timing before accessing this")

    @property
    def subframes_until_next_external_trigger(self):
        try:
            return self.hdf5_group["subframes_until_next_external_trigger"]
        except KeyError:
            raise ValueError(
                "run tes_group.calc_external_trigger_timing before accessing this")

    @property
    def subframes_from_nearest_external_trigger(self):
        try:
            return self.hdf5_group["subframes_from_nearest_external_trigger"]
        except KeyError:
            raise ValueError(
                "run tes_group.calc_external_trigger_timing before accessing this")

    def __str__(self):
        line1 = f"{self.__class__.__name__} path '{self.filename}'\n"
        line2 = f"{self.nSamples} samples ({self.nPresamples} pretriggger) at {1e6 * self.timebase:.2f} µs sample time"
        return "\n".join((line1, line2))

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.filename}')"

    def updater(self, name):
        return self.tes_group.updater(name + f" chan {self.channum:d}")

    def good(self, *args, **kwargs):
        """Returns a boolean vector, one per pulse record, saying whether record is good"""
        return self.cuts.good(*args, **kwargs)

    def bad(self, *args, **kwargs):
        """Returns a boolean vector, one per pulse record, saying whether record is bad"""
        return self.cuts.bad(*args, **kwargs)

    def resize(self, nPulses):
        if self.nPulses < nPulses:
            raise ValueError(f"Can only shrink using resize(), but the requested size {nPulses} > than current {self.nPulses}")
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
        peak_idx = self.data[:, self.cut_pre:self.nSamples
                             - self.cut_post].argmax(axis=1) + self.cut_pre
        if version.parse(sp.__version__) >= version.parse("1.9.0"):
            self.peak_samplenumber = int(sp.stats.mode(peak_idx, keepdims=False).mode)
        else:
            # The old way of using sp.stats.mode is deprecated in sp 1.9.0 and will be removed.
            self.peak_samplenumber = int(sp.stats.mode(peak_idx)[0][0])
        self.p_peak_index.attrs["peak_samplenumber"] = self.peak_samplenumber
        return self.peak_samplenumber

    @show_progress("channel.summarize_data")
    def summarize_data(self, peak_time_microsec=None, pretrigger_ignore_microsec=None,
                       cut_pre=0, cut_post=0,
                       forceNew=False, doPretrigFit=False):
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
        doPretrigFit: whether to do a linear fit of the pretrigger data
        """
        for name in ("number_of_rows", "row_number", "number_of_columns", "column_number",
                     "subframe_divisions", "subframe_offset"):
            setattr(self, name, getattr(self.pulse_records.datafile, name))
        self.subframe_timebase = self.timebase / float(self.subframe_divisions)

        not_done = all(self.p_pretrig_mean[:] == 0)
        if not (not_done or forceNew):
            LOG.info('\nchan %d did not summarize because results were already preloaded', self.channum)
            return

        if len(self.p_timestamp) < self.pulse_records.nPulses:
            # make sure vectors are setup correctly
            self.__setup_vectors(npulses=self.pulse_records.nPulses)

        if peak_time_microsec is None:
            self.peak_samplenumber = None
        else:
            self.peak_samplenumber = self.nPresamples + int(peak_time_microsec * 1e-6 / self.timebase)
        if pretrigger_ignore_microsec is None:
            self.pretrigger_ignore_samples = 3
        else:
            self.pretrigger_ignore_samples = int(pretrigger_ignore_microsec * 1e-6 / self.timebase)

        self.cut_pre = cut_pre
        self.cut_post = cut_post

        if self.peak_samplenumber is None:
            self._compute_peak_samplenumber()
        self.p_timestamp[:] = self.times[:]
        self.p_subframecount[:] = self.subframecount[:]

        for segnum in range(self.pulse_records.n_segments):
            first = segnum * self.pulse_records.pulses_per_seg
            end = min(first + self.pulse_records.pulses_per_seg, self.nPulses)
            idx_slice = slice(first, end)
            results = summarize_data_numba(
                self.data[idx_slice], self.timebase, self.peak_samplenumber,
                self.pretrigger_ignore_samples, self.nPresamples)
            self.p_pretrig_mean[idx_slice] = results["pretrig_mean"][:]
            self.p_pretrig_rms[idx_slice] = results["pretrig_rms"][:]
            self.p_pulse_average[idx_slice] = results["pulse_average"][:]
            self.p_pulse_rms[idx_slice] = results["pulse_rms"][:]
            self.p_promptness[idx_slice] = results["promptness"][:]
            self.p_postpeak_deriv[idx_slice] = results["postpeak_deriv"][:]
            self.p_peak_index[idx_slice] = results["peak_index"][:]
            self.p_peak_value[idx_slice] = results["peak_value"][:]
            self.p_min_value[idx_slice] = results["min_value"][:]
            self.p_rise_time[idx_slice] = results["rise_times"][:]
            self.p_shift1[idx_slice] = results["shift1"][:]

            yield (segnum + 1.0) / self.pulse_records.n_segments

        self.hdf5_group.file.flush()
        self.__parse_expt_states()

    def __parse_expt_states(self):
        """
        Load experiment states from the state file and store the slices found for each state
        as a categorical cut.
        """
        esf = self.tes_group.experimentStateFile
        if esf is None:
            return
        nano = self.p_timestamp[:] * 1e9
        slicedict = esf.calcStatesDict(nano)

        state_codes = np.zeros(self.nPulses, dtype=np.uint32)
        for id, state in enumerate(slicedict.keys()):
            slices = slicedict[state]

            # Ensure `slices` is a list of slices, even if it's a single slice
            if not isinstance(slices, list):
                slices = [slices]

            # Assign codes for each slice in the list
            for slice in slices:
                state_codes[slice.start:slice.stop] = id + 1

        self.cuts.cut("state", state_codes)

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
        already_done = np.any(self.average_pulse[:] != 0)
        if already_done and not forceNew:
            LOG.info("skipping compute average pulse on chan %d", self.channum)
            return

        average_pulse = self.data[mask, :].mean(axis=0)
        if subtract_mean:
            nsamp = self.nPresamples - self.pretrigger_ignore_samples
            average_pulse -= np.mean(average_pulse[:nsamp])
        self.average_pulse[:] = average_pulse

    @_add_group_loop()
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
        mask = np.abs(self.p_pulse_average[:] / median_pulse_avg - 1) < 0.05
        mask = np.logical_and(mask, use)
        if mask.sum() <= 0:
            raise ValueError("No good pulses within 5%% of median size.")

        if np.sum(mask) > max_pulses_to_use:
            good_so_far = np.cumsum(mask)
            stop_at = (good_so_far == max_pulses_to_use).argmax()
            mask[stop_at + 1:] = False
        self.compute_average_pulse(mask, subtract_mean=subtract_mean,
                                   forceNew=forceNew)

    def _filter_to_hdf5(self):
        # Store any optimal filter to a new HDF5 group
        if "filters" in self.hdf5_group:
            del self.hdf5_group["filters"]
        h5grp = self.hdf5_group.require_group('filters')
        if self.filter.f_3db is not None:
            h5grp.attrs['f_3db'] = self.filter.f_3db
        if self.filter.fmax is not None:
            h5grp.attrs['fmax'] = self.filter.fmax
        h5grp.attrs['filter_type'] = self.filter._filter_type
        h5grp.attrs["version"] = 2
        for name in ("values", "dt_values"):
            if name in h5grp:
                del h5grp[name]
        vec = h5grp.create_dataset("values", data=self.filter.values)
        vec.attrs["nominal_peak"] = self.filter.nominal_peak
        vec.attrs["variance"] = self.filter.variance
        vec.attrs["predicted_v_over_dv"] = self.filter.predicted_v_over_dv
        if isinstance(self.filter, mass.FilterATS):
            h5grp.create_dataset("dt_values", data=self.filter.dt_values)

    @property
    def shortname(self):
        """return a string containing part of the filename and the channel number, useful for labelling plots"""
        s = os.path.split(self.filename)[-1]
        chanstr = f"chan{self.channum:g}"
        if chanstr not in s:
            s += chanstr
        return s

    @_add_group_loop()
    def compute_5lag_filter(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0, category={}, forceNew=False):
        """Requires that compute_noise has been run and that average pulse has been computed"""
        if "filters" in self.hdf5_group and not forceNew:
            LOG.info(f"ch {self.channum} skipping compute 5-lag filter because it is already done")
            return
        if all(self.noise_autocorr[:] == 0):
            raise Exception("compute noise first")
        if not (category is None or category == {}):
            raise Exception(
                "category argument has no effect on compute_oldfilter, pass None or {}. compute_oldfilter uses self.average_pulse")
        f = self._compute_5lag_filter_no_mutation(fmax, f_3db, cut_pre, cut_post)
        self.filter = f
        self._filter_to_hdf5()
        return f

    def _compute_5lag_filter_no_mutation(self, fmax, f_3db, cut_pre, cut_post):
        try:
            spectrum = self.noise_spectrum.spectrum()
        except Exception:
            spectrum = self.noise_psd[:]
        avg_signal = self.average_pulse[:]
        if np.all(np.abs(self.average_pulse) == 0):
            raise Exception("average pulse is all zeros, try avg_pulses_auto_masks first")
        maker = FilterMaker(avg_signal, self.nPresamples - self.pretrigger_ignore_samples,
                            self.noise_autocorr, spectrum, sample_time_sec=self.timebase)
        f = maker.compute_5lag(fmax=fmax, f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)
        return f

    @_add_group_loop()
    def compute_ats_filter(self, fmax=None, f_3db=None, transform=None, cut_pre=0, cut_post=0,  # noqa: PLR0917, PLR0914
                           category={}, shift1=True, forceNew=False, minimum_n_pulses=20,
                           maximum_n_pulses=4000, optimize_dp_dt=True):
        """Compute a arrival-time-safe filter to model the pulse and its time-derivative.
        Requires that `compute_noise` has been run.

        Args:
            fmax: if not None, the hard cutoff in frequency space, above which
                the DFT of the filter will be set to zero (default None)
            f_3db: if not None, the 3 dB rolloff point in frequency space, above which
                the DFT of the filter will rolled off with a 1-pole filter
                (default None)
            transform: a callable object that will be called on all data records
                before filtering (default None)
            optimize_dp_dt: bool, try a more elaborate approach to dp_dt than just the finite
                difference (works well for x-ray, bad for gamma rays)
            cut_pre: Cut this many samples from the start of the filter, giving them 0 weight.
            cut_post: Cut this many samples from the end of the filter, giving them 0 weight.
            shift1: Potentially shift each pulse by one sample based on ds.shift1 value,
            resulting filter is one sample shorter than pulse records.
            If you used a zero threshold trigger (eg dastard egdeMulti you can likely use shift1=False)

        Returns:
            the filter (an ndarray)

        Modified in April 2017 to make the model for the rising edge and the rest of
        the pulse differently. For the rising edge, we use entropy minimization to understand
        the pulse shape dependence on arrival-time. For the rest of the pulse, it
        is less noisy and in fact more robust to rely on the finite-difference of
        the pulse average to get the arrival-time dependence.
        """
        if "filters" in self.hdf5_group and not forceNew:
            LOG.info(f"ch {self.channum} skipping compute_ats_filter because it is already done")
            return
        if all(self.noise_autocorr[:] == 0):
            raise Exception("compute noise first")
        # At the moment, 1st-order model vs arrival-time is required.
        DEGREE = 1

        # The raw training data, which is shifted (trigger-aligned)
        data, pulsenums = self.first_n_good_pulses(maximum_n_pulses, category=category)
        if len(pulsenums) < minimum_n_pulses:
            raise Exception(f"too few good pulses, ngood={len(pulsenums)}")
        if shift1:
            raw = data[:, 1:]
            _shift1 = self.p_shift1[:][pulsenums]
            raw[_shift1, :] = data[_shift1, 0:-1]
        else:
            raw = data[:, :]

        # Center promptness around 0, using a simple function of Prms
        prompt = self.p_promptness[:][pulsenums]
        prms = self.p_pulse_rms[:][pulsenums]
        mprms = np.median(prms)
        use = np.abs(prms / mprms - 1.0) < 0.3
        promptshift = np.poly1d(np.polyfit(prms[use], prompt[use], 1))
        prompt -= promptshift(prms)

        # Scale promptness quadratically to cover the range -0.5 to +0.5, approximately
        x, y, z = np.percentile(prompt[use], [10, 50, 90])
        A = np.array([[x * x, x, 1],
                      [y * y, y, 1],
                      [z * z, z, 1]])
        param = np.linalg.solve(A, [-.4, 0, +.4])
        ATime = np.poly1d(param)(prompt)
        use = np.logical_and(use, np.abs(ATime) < 0.45)
        ATime = ATime[use]

        ptm = self.p_pretrig_mean[:][pulsenums]
        ptm.shape = (len(pulsenums), 1)
        raw = (raw - ptm)[use, :]
        if transform is not None:
            raw = transform(raw)
        rawscale = raw.max(axis=1)

        # The 0 component of the model is an average pulse, but do not use
        # self.average_pulse, because it doesn't account for the shift1.
        if shift1:
            model = np.zeros((self.nSamples - 1, 1 + DEGREE), dtype=float)
        else:
            model = np.zeros((self.nSamples, 1 + DEGREE), dtype=float)
        ap = (raw.T / rawscale).mean(axis=1)
        apmax = np.max(ap)
        model[:, 0] = ap / apmax
        model[1:-1, 1] = (ap[2:] - ap[:-2]) * 0.5 / apmax
        model[-1, 1] = (ap[-1] - ap[-2]) / apmax
        model[:self.nPresamples - 1, :] = 0

        if optimize_dp_dt:
            # Now use min-entropy computation to model dp/dt on the rising edge
            def cost(slope, x, y):
                return mass.mathstat.entropy.laplace_entropy(y - x * slope, 0.002)

            if self.peak_samplenumber is None:
                self._compute_peak_samplenumber()
            for samplenum in range(self.nPresamples - 1, self.peak_samplenumber):
                y = raw[:, samplenum] / rawscale
                bestslope = sp.optimize.brent(cost, (ATime, y), brack=[-.1, .25], tol=1e-7)
                model[samplenum, 1] = bestslope

        modelpeak = np.median(rawscale)
        self.pulsemodel = model

        maker = FilterMaker(model[:, 0], self.nPresamples, self.noise_autocorr,
                            dt_model=model[:, 1], sample_time_sec=self.timebase, peak=modelpeak)
        f = maker.compute_ats(fmax=fmax, f_3db=f_3db, cut_pre=cut_pre, cut_post=cut_post)
        self.filter = f
        if np.any(np.isnan(f.values)) or np.any(np.isnan(f.dt_values)):
            raise ValueError("{}. model {}, nPresamples {}, noise_autcorr {}, timebase {}, modelpeak {}".format(
                "there are nan values in your filters!! BAD",
                model, self.nPresamples, self.noise_autocorr, self.timebase, modelpeak))
        self._filter_to_hdf5()
        return f

    @_add_group_loop()
    def filter_data(self, transform=None, forceNew=False, use_cython=None):
        """Filter the complete data file one chunk at a time.

        Args:
            transform: a callable object that will be called on all data records
                before filtering (default None)
            forceNew: Whether to recompute when already exists (default False)
        """
        if not (forceNew or all(self.p_filt_value[:] == 0)):
            LOG.info('chan %d did not filter because results were already loaded', self.channum)
            return

        if use_cython is None:
            use_cython = self._filter_type == "5lag"

        assert self.filter is not None
        filter_values = self.filter.values

        if use_cython:
            if self._filter_type == "ats":
                raise ValueError("Cannot perform Arrival-Time-Safe filtering in Cython yet")
            fdata = mass.core.analysis_algorithms.filter_data_5lag_cython
            fv, fp = fdata(self.alldata, filter_values)
            self.p_filt_value[:] = fv[:]
            self.p_filt_phase[:] = fp[:]
            self.hdf5_group.file.flush()
            return

        self._filter_data_nocython(filter_values, transform=transform)

    @show_progress("channel.filter_data_tdm")
    def _filter_data_nocython(self, filter_values, transform=None):
        # when dastard uses kink model for determining trigger location, we don't need to shift1
        # this code path should be followed when filters are created with the shift1=False argument
        effective_filter_length = len(filter_values) + self.filter.convolution_lags - 1
        use_shift1 = (effective_filter_length == self.nSamples - 1)
        if not use_shift1:
            assert effective_filter_length == self.nSamples

        for s in range(self.pulse_records.n_segments):
            first = s * self.pulse_records.pulses_per_seg
            end = min(self.nPulses, first + self.pulse_records.pulses_per_seg)
            seg_size = end - first
            data = self.data[first:end]

            # Handle "shift1" data by removing the 1st sample from most records, but the
            # last sample from those with self.p_shift1 set to True.
            if use_shift1:
                shift_these_records = self.p_shift1[first:end]
                if np.any(shift_these_records):
                    data = np.array(data)
                    data[shift_these_records, 1:] = data[shift_these_records, :-1]
                data = data[:, 1:]

            if transform is not None:
                ptmean = self.p_pretrig_mean[first:end]
                ptmean.shape = (seg_size, 1)
                data = transform(data - ptmean)

            (self.p_filt_value[first:end],
             self.p_filt_phase[first:end]) = \
                self.filter.filter_records(data)
            yield (end + 1) / float(self.nPulses)

        self.hdf5_group.file.flush()

    def get_pulse_model(self, f, f_5lag, n_basis, pulses_for_svd, extra_n_basis_5lag=0,
                        maximum_n_pulses=4000, noise_weight_basis=True, category={}):
        assert n_basis >= 3
        assert f.is_arrival_time_safe, "requires arrival-time-safe filter"
        assert noise_weight_basis, "basis not noise weighted is not implemented"

        deriv_like_model = f.dt_model
        pulse_like_model = f.signal_model
        v_dv = f.predicted_v_over_dv
        if not len(pulse_like_model) == self.nSamples:
            raise Exception(f"filter length {len(pulse_like_model)} and nSamples {self.nSamples} don't match, "
                            "you likely need to use shift1=False in compute_ats_filter")

        projectors1 = np.vstack([f.const_values,
                                 f.dt_values,
                                 f.values])
        basis1 = np.vstack([
            np.ones(len(pulse_like_model), dtype=float),
            deriv_like_model,
            pulse_like_model,
        ]).T

        if pulses_for_svd is None:
            pulses_for_svd, _ = self.first_n_good_pulses(maximum_n_pulses, category=category)
            pulses_for_svd = pulses_for_svd.T
        if hasattr(self, "saved_auto_cuts"):
            pretrig_rms_median = self.saved_auto_cuts._pretrig_rms_median
            pretrig_rms_sigma = self.saved_auto_cuts._pretrig_rms_sigma
        else:
            raise Exception(
                "use autocuts when making projectors, so it can save more info about desired cuts")
        self.pulse_model = PulseModel(projectors1, basis1, n_basis, pulses_for_svd,
                                      v_dv, pretrig_rms_median, pretrig_rms_sigma, self.filename,
                                      extra_n_basis_5lag, f_5lag.values,
                                      self.average_pulse[:], self.noise_psd[:], self.noise_psd.attrs['delta_f'],
                                      self.noise_autocorr[:])
        return self.pulse_model

    @_add_group_loop()
    def _pulse_model_to_hdf5(self, hdf5_file, n_basis, pulses_for_svd=None, extra_n_basis_5lag=0,
                             maximum_n_pulses=4000, noise_weight_basis=True, f_3db_5lag=None, category={}):
        self.avg_pulses_auto_masks(forceNew=False, max_pulses_to_use=maximum_n_pulses)
        f_5lag = self._compute_5lag_filter_no_mutation(
            fmax=None, f_3db=f_3db_5lag, cut_pre=0, cut_post=0)
        pulse_model = self.get_pulse_model(
            self.filter, f_5lag, n_basis, pulses_for_svd, extra_n_basis_5lag,
            maximum_n_pulses=maximum_n_pulses, noise_weight_basis=noise_weight_basis, category=category)
        save_inverted = self.invert_data
        hdf5_group = hdf5_file.create_group(f"{self.channum}")
        pulse_model.toHDF5(hdf5_group, save_inverted)

    def plot_summaries(self, valid='uncut', downsample=None, log=False):
        """Plot a summary of the data set, including time series and histograms of
        key pulse properties.

        Args:
            valid: An array of booleans self.nPulses long saying which pulses are to be plotted
                *OR* 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted
                *OR* None or 'all', meaning that all pulses should be plotted.

            downsample: To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points (default None).

            log (bool):  Use logarithmic y-axis on the histograms (right panels). (Default False)
        """

        # Convert "uncut" or "cut" to array of all good or all bad data
        def isstr(x):
            return isinstance(x, ("".__class__, "".__class__))

        status = "Plotting selected data"
        if valid is None:
            status = "Plotting all data, cut or uncut"
        elif isstr(valid):
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
                downsample = max(downsample, 1)
            hour = self.p_timestamp[valid][::downsample] / 3600.0
        else:
            nrecs = self.nPulses
            if downsample is None:
                downsample = self.nPulses // 10000
                downsample = max(downsample, 1)
            hour = self.p_timestamp[::downsample] / 3600.0
        LOG.info("%s (%d records; %d in scatter plots)", status, nrecs, len(hour))

        plottables = (
            (self.p_pulse_rms, 'Pulse RMS', 'magenta', None),
            (self.p_pulse_average, 'Pulse Avg', 'purple', None),
            (self.p_peak_value, 'Peak value', 'blue', None),
            (self.p_pretrig_rms, 'Pretrig RMS', 'green', [0, 4000]),
            (self.p_pretrig_mean, 'Pretrig Mean', '#00ff26', None),
            (self.p_postpeak_deriv, 'Max PostPk deriv', 'gold', [0, 700]),
            (self.p_rise_time[:] * 1e3, 'Rise time (ms)', 'orange', [0, 12]),
            (self.p_peak_time[:] * 1e3, 'Peak time (ms)', 'red', [-3, 9])
        )

        # Plot timeseries with 0 = the last 00 UT during or before the run.
        last_record = np.max(self.p_timestamp)
        last_midnight = last_record - (last_record % 86400)
        hour_offset = last_midnight / 3600.

        plt.clf()
        for i, (vect, label, color, limits) in enumerate(plottables):

            # Time series scatter plots (left-hand panels)
            plt.subplot(len(plottables), 2, 1 + i * 2)
            plt.ylabel(label)
            use_vect = vect
            if valid is not None:
                use_vect = vect[valid]
            plt.plot(hour - hour_offset, use_vect[::downsample], '.', ms=1, color=color)
            if i == len(plottables) - 1:
                plt.xlabel("Time since last UT midnight (hours)")

            # Histogram (right-hand panels)
            plt.subplot(len(plottables), 2, 2 + i * 2)
            if limits is None:
                in_limit = np.ones(len(use_vect), dtype=bool)
            else:
                in_limit = np.logical_and(use_vect[:] > limits[0], use_vect[:] < limits[1])
            contents, _bins, _patches = plt.hist(use_vect[in_limit], 200, log=log,
                                                 histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin=contents.min())

    @_add_group_loop()
    def assume_white_noise(self, noise_variance=1.0, forceNew=False):
        """Set the noise variance to `noise_variance` and the spectrum to be white.

        This is appropriate when no noise files were taken.
        Though you may set `noise_variance` to a value other than 1, this will affect only the
        predicted resolution, and will not change the optimal filters that get computed/used.

        Args:
            noise_variance(number): what to set as the lag-0 noise autocorrelation.
            forceNew (bool): whether to update the noise autocorrelation if it's already
                been set (default False).
        """
        if forceNew or all(self.noise_autocorr[:] == 0):
            self.noise_autocorr[1:] = 0.0
            self.noise_autocorr[0] = noise_variance
            psd = 2.0 * noise_variance * self.timebase
            self.noise_psd[:] = psd

    @_add_group_loop()
    def compute_noise_nlags(self, n_lags, max_excursion=1000, plot=False):
        """Compute the noise autocorrelation and power spectrum of this channel using records of length nlags.
        Treats data in separate noise traces as continuous.

        Args:
            max_excursion (number): the biggest excursion from the median allowed
                in each data segment, or else it will be ignored (default 1000).
            n_lags: if not None, the number of lags in each noise spectrum and the max lag
                for the autocorrelation.  If None, the record length is used
                (default None).
            forceNew (bool): whether to recompute if it already exists (default False).
        """
        self.noise_records_nlags = NoiseRecords(self.noise_records.filename)
        self.noise_records_nlags.compute_power_spectrum_reshape(
            max_excursion=max_excursion, seg_length=n_lags)
        self.noise_records_nlags.compute_autocorrelation(
            n_lags=n_lags, plot=False, max_excursion=max_excursion)
        if plot:
            self.noise_records_nlags.plot_power_spectrum(sqrt_psd=False)

    @_add_group_loop()
    def compute_noise(self, max_excursion=1000, forceNew=False):
        """Compute the noise autocorrelation and power spectrum of this channel.

        Args:
            max_excursion (number): the biggest excursion from the median allowed
                in each data segment, or else it will be ignored (default 1000).
            n_lags: if not None, the number of lags in each noise spectrum and the max lag
                for the autocorrelation.  If None, the record length is used
                (default None).
            forceNew (bool): whether to recompute if it already exists (default False).
        """
        n_lags = self.noise_records.nSamples
        if forceNew or all(self.noise_autocorr[:] == 0):
            self.noise_records.compute_power_spectrum_reshape(
                max_excursion=max_excursion, seg_length=n_lags)
            self.noise_records.compute_autocorrelation(
                n_lags=n_lags, plot=False, max_excursion=max_excursion)

            self.noise_autocorr[:] = self.noise_records.autocorrelation[:len(
                self.noise_autocorr[:])]
            self.noise_psd[:] = self.noise_records.noise_psd[:len(self.noise_psd[:])]
            self.noise_psd.attrs['delta_f'] = self.noise_records.noise_psd.attrs['delta_f']
        else:
            LOG.info("chan %d skipping compute_noise because already done", self.channum)

    # Rename compute_noise_spectra -> compute_noise, because the latter is a better name!
    # But use deprecation to not immediately break all code.
    @_add_group_loop()
    @deprecated(deprecated_in="0.7.9", details="Use compute_noise(), which is equivalent but better named")
    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        """Replaced by the equivalent compute_noise(...)"""
        return self.compute_noise(max_excursion=max_excursion, n_lags=n_lags, forceNew=forceNew)

    @_add_group_loop()
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

        self.cuts.cut_parameter(self.p_peak_time[:] * 1e3, c['peak_time_ms'], 'peak_time_ms')
        self.cuts.cut_parameter(self.p_rise_time[:] * 1e3, c['rise_time_ms'], 'rise_time_ms')
        self.cuts.cut_parameter(self.p_postpeak_deriv, c['postpeak_deriv'], 'postpeak_deriv')
        self.cuts.cut_parameter(self.p_pulse_average, c['pulse_average'], 'pulse_average')
        self.cuts.cut_parameter(self.p_peak_value, c['peak_value'], 'peak_value')
        self.cuts.cut_parameter(
            self.p_min_value[:] - self.p_pretrig_mean[:], c['min_value'], 'min_value')
        self.cuts.cut_parameter(self.p_timestamp[:], c['timestamp_sec'], 'timestamp_sec')

        if c['timestamp_diff_sec'] is not None:
            self.cuts.cut_parameter(np.hstack((np.inf, np.diff(self.p_timestamp))),
                                    c['timestamp_diff_sec'], 'timestamp_diff_sec')
        if c['subframecount_diff_sec'] is not None:
            self.cuts.cut_parameter(np.hstack((np.inf, np.diff(self.p_subframecount[:] * self.subframe_timebase))),
                                    c['subframecount_diff_sec'], 'subframecount_diff_sec')
        if c['pretrigger_mean_departure_from_median'] is not None and self.cuts.good().sum() > 0:
            median = np.median(self.p_pretrig_mean[self.cuts.good()])
            LOG.debug('applying cut on pretrigger mean around its median value of %f', median)
            self.cuts.cut_parameter(self.p_pretrig_mean - median,
                                    c['pretrigger_mean_departure_from_median'],
                                    'pretrigger_mean_departure_from_median')
        LOG.info("Chan %d after cuts, %d are good, %d are bad of %d total pulses",
                 self.channum, self.cuts.good().sum(), self.cuts.bad().sum(), self.nPulses)

    @_add_group_loop()
    def clear_cuts(self):
        """Clear all cuts."""
        self.cuts.clear_cut()
        self.saved_auto_cuts = None

    @_add_group_loop()
    def correct_flux_jumps(self, flux_quant, algorithm="Baker"):
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
        algorithm -- {"Baker", "orig"}
        '''
        methods = {
            "Baker": mass.core.analysis_algorithms.correct_flux_jumps,
            "orig": mass.core.analysis_algorithms.correct_flux_jumps_original
        }
        method = methods[algorithm]
        corrected = method(self.p_pretrig_mean[:], self.good(), flux_quant)
        self.p_pretrig_mean[:] = corrected

    @_add_group_loop()
    def drift_correct(self, attr="p_filt_value", forceNew=False, category={}):
        """Drift correct using the standard entropy-minimizing algorithm"""
        doesnt_exist = all(self.p_filt_value_dc[:] == 0) or all(
            self.p_filt_value_dc[:] == self.p_filt_value[:])
        if not (forceNew or doesnt_exist):
            LOG.info(
                "chan %d drift correction skipped, because p_filt_value_dc already populated", self.channum)
            return
        g = self.cuts.good(**category)
        uncorrected = getattr(self, attr)[g]
        indicator = self.p_pretrig_mean[g]
        drift_corr_param, self.drift_correct_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        self.p_filt_value_dc.attrs.update(self.drift_correct_info)  # Store in hdf5 file
        LOG.info('chan %d best drift correction parameter: %.6fe6', self.channum, 1e6 * drift_corr_param)
        self._apply_drift_correction(attr=attr)

    def _apply_drift_correction(self, attr):
        # Apply correction
        assert self.p_filt_value_dc.attrs["type"] == "ptmean_gain"
        ptm_offset = self.p_filt_value_dc.attrs["median_pretrig_mean"]
        uncorrected = getattr(self, attr)[:]
        gain = 1 + (self.p_pretrig_mean[:] - ptm_offset) * self.p_filt_value_dc.attrs["slope"]
        self.p_filt_value_dc[:] = uncorrected * gain
        self.hdf5_group.file.flush()

    @_add_group_loop()
    def phase_correct(self, attr="p_filt_value_dc", forceNew=False, category={}, ph_peaks=None,
                      method2017=True, kernel_width=None, save_to_hdf5=True):
        """Apply the 2017 or 2015 phase correction method.

        Args:
            forceNew (bool): whether to recompute if it already exists (default False).
            category (dict): if not None, then a dict giving a category name and the
                required category label.
            ph_peaks:  Peaks to use for alignment. If None, then use _find_peaks_heuristic()
            kernel_width: Width (in PH units) of the kernel-smearing function. If None, use a heuristic.
        """

        doesnt_exist = not hasattr(self, "phaseCorrector")
        if not (forceNew or doesnt_exist):
            LOG.info("channel %d skipping phase_correct", self.channum)
            return
        good = self.cuts.good(**category)

        self.phaseCorrector = phase_correct.phase_correct(self.p_filt_phase[good],
                                                          getattr(self, attr)[good],
                                                          ph_peaks=ph_peaks, method2017=method2017, kernel_width=kernel_width,
                                                          indicatorName="p_filt_phase", uncorrectedName="p_filt_value_dc")

        self.p_filt_phase_corr[:] = self.phaseCorrector.phase_uniformifier(self.p_filt_phase[:])
        self.p_filt_value_phc[:] = self.phaseCorrector(self.p_filt_phase[:], getattr(self, attr)[:])

        if save_to_hdf5:
            self.phaseCorrector.toHDF5(self.hdf5_group, overwrite=True)

        LOG.info('Channel %3d phase corrected. Correction size: %.2f',
                 self.channum, mass.mathstat.robust.median_abs_dev(self.p_filt_value_phc[good]
                                                                   - getattr(self, attr)[good], True))

        return self.phaseCorrector

    def first_n_good_pulses(self, n=50000, category={}):
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
        g = self.cuts.good(**category)
        if g.sum() > n:
            dont_use = np.nonzero(g)[0][n:]
            g[dont_use] = False

        return self.data[g], np.nonzero(g)[0]

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
        bin_ctrs = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Try line first as a number, then as a fluorescence line, then as a Gaussian
        try:
            energy = float(line)
            module = 'mass.calibration.gaussian_lines'
            fittername = f'{module}.GaussianFitter({module}.GaussianLine())'
            fitter = eval(fittername)
        except ValueError:
            energy = None
            try:
                module = 'mass.calibration.fluorescence_lines'
                fittername = f'{module}.{line}Fitter()'
                fitter = eval(fittername)
            except AttributeError:
                try:
                    module = 'mass.calibration.gaussian_lines'
                    fittername = f'{module}.{line}Fitter()'
                    fitter = eval(fittername)
                except AttributeError:
                    raise ValueError(
                        f"Cannot understand line={line} as an energy or a known calibration line.")

        params, covar = fitter.fit(contents, bin_ctrs, plot=plot, **kwargs)
        if plot:
            mass.plot_as_stepped_hist(plt.gca(), contents, bin_ctrs)
        if energy is not None:
            scale = energy / params[1]
        else:
            scale = 1.0
        LOG.info('Resolution: %5.2f +- %5.2f eV', params[0] * scale, np.sqrt(covar[0, 0]) * scale)
        return params, covar, fitter

    @property
    def pkl_fname(self):
        return ljh_util.mass_folder_from_ljh_fname(self.filename, filename=f"ch{self.channum}_calibration.pkl")

    @_add_group_loop()
    def calibrate(self, attr, line_names, name_ext="", size_related_to_energy_resolution=10,  # noqa: PLR0917
                  fit_range_ev=200, excl=(), plot_on_fail=False,
                  bin_size_ev=2.0, category={}, forceNew=False, maxacc=0.015, nextra=3,
                  param_adjust_closure=None, curvetype="gain", approximate=False,
                  diagnose=False):
        calname = attr + name_ext

        if not forceNew and calname in self.calibration:
            return self.calibration[calname]

        LOG.info("Calibrating chan %d to create %s", self.channum, calname)
        # It tries to calibrate detector using mass.calibration.algorithm.EnergyCalibrationAutocal.
        auto_cal = EnergyCalibrationAutocal(
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
            LOG.warning(
                "chan %d failed calibration because on of the fitter was a FailedFitter", self.channum)
            raise Exception()

        cal = auto_cal.cal_factory.make_calibration(curvename=curvetype, approximate=approximate)
        self.calibration[calname] = cal
        hdf5_cal_group = self.hdf5_group.require_group('calibration')
        cal.save_to_hdf5(hdf5_cal_group, calname)

        if diagnose:
            auto_cal.diagnose()
        self.convert_to_energy(attr, attr + name_ext)

    @_add_group_loop()
    def convert_to_energy(self, attr, calname=None):
        if calname is None:
            calname = attr
        if calname not in self.calibration:
            raise ValueError(f"For chan {self.channum} calibration {calname} does not exist")
        cal = self.calibration[calname]
        self.p_energy[:] = cal.ph2energy(getattr(self, attr))
        self.last_used_calibration = cal

    def plot_traces(self, pulsenums, pulse_summary=True, axis=None, difference=False,  # noqa: PLR0917
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

        if valid_status not in {None, "valid", "cut"}:
            raise ValueError("valid_status must be one of [None, 'valid', or 'cut']")
        if residual and difference:
            raise ValueError("Only one of residual and difference can be True.")

        dt = (np.arange(self.nSamples) - self.nPresamples) * self.timebase * 1e3
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
        for i, pn in enumerate(pulsenums):
            if valid_status == 'cut' and cuts_good[i]:
                continue
            if valid_status == 'valid' and not cuts_good[i]:
                continue
            pulses_plotted += 1

            data = self.read_trace(pn)
            if difference:
                data = data * 1.0 - np.roll(data, 1)
                data[0] = 0
                data += np.roll(data, 1) + np.roll(data, -1)
                data[0] = 0
            elif residual:
                model = self.p_filt_value[pn] * self.average_pulse[:] / np.max(self.average_pulse)
                # `data = data - model` rebinds data to a numpy vector, which is allowed.
                # Disable QA test for augmented assignment, b/c we're replacing a memmap with a vector
                data = data - model  # NOQA: PLR6104
            if shift1 and self.p_shift1[pn]:
                data = np.hstack([data[0], data[:-1]])
            if fcut is not None:
                data = mass.core.analysis_algorithms.filter_signal_lowpass(
                    data, 1. / self.timebase, fcut)
            if subtract_baseline:
                # Recalculate the pretrigger mean here, to avoid issues due to flux slipping when
                # plotting umux data.
                # Disable QA test for augmented assignment, b/c we're replacing a memmap with a vector
                data = data - np.mean(data[:self.nPresamples - self.pretrigger_ignore_samples])  # NOQA: PLR6104

            cutchar, alpha, linestyle, linewidth = ' ', 1.0, '-', 1

            # When plotting both cut and valid, mark the cut data with x and dashed lines
            if valid_status is None and not cuts_good[i]:
                cutchar, alpha, linestyle, linewidth = 'X', 1.0, '--', 1
            color = cm(pulses_plotted * 1.0 / len(cuts_good))
            axis.plot(dt, data, color=color,
                      linestyle=linestyle, alpha=alpha, linewidth=linewidth)
            if pulse_summary and pulses_plotted < MAX_TO_SUMMARIZE and len(self.p_pretrig_mean) >= pn:
                try:
                    summary = f"{cutchar}{pn:6d}: {self.p_pretrig_mean[pn]:5.0f} {self.p_pretrig_rms[pn]:7.2f} "
                    summary += f"{self.p_postpeak_deriv[pn]:6.1f} {self.p_rise_time[pn] * 1e6:5.0f} "
                    summary += f"{self.p_peak_value[pn]:5.0f} {self.p_pulse_average[pn]:7.1f}"
                except IndexError:
                    pulse_summary = False
                    continue
                axis.text(.975, .93 - .025 * pulses_plotted, summary, color=color,
                          family='monospace', size='medium',
                          transform=axis.transAxes, ha='right')

    def read_trace(self, record_num):
        """Read (from cache or disk) and return the pulse numbered `record_num`."""
        return self.data[record_num, :]

    @_add_group_loop()
    def time_drift_correct(self, attr="p_filt_value_phc", sec_per_degree=2000,
                           pulses_per_degree=2000, max_degrees=20, forceNew=False,
                           category={}):
        """Drift correct over long times with an entropy-minimizing algorithm.
        Here we correct as a low-ish-order Legendre polynomial in time.

        attr: the attribute of self that is to be corrected. (The result
                will be stored in self.p_filt_value_tdc[:]).
        sec_per_degree: assign as many as one polynomial degree per this many seconds
        pulses_per_degree: assign as many as one polynomial degree per this many pulses
        max_degrees: never use more than this many degrees of Legendre polynomial.

        forceNew: whether to do this step, if it appears already to have been done.
        category: choices for categorical cuts
        """
        if all(self.p_filt_value_tdc[:] == 0.0) or forceNew:
            LOG.info("chan %d doing time_drift_correct", self.channum)
            attr = getattr(self, attr)
            g = self.cuts.good(**category)
            pk = np.median(attr[g])
            g = np.logical_and(g, np.abs(attr[:] / pk - 1) < 0.5)
            w = max(pk / 3000., 1.0)
            info = time_drift_correct(self.p_timestamp[g], attr[g], w,
                                      limit=[0.5 * pk, 2 * pk])
            tnorm = info["normalize"](self.p_timestamp[:])
            corrected = attr[:] * (1 + info["model"](tnorm))
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
            except Exception:
                pass
        plt.legend()
        plt.xlabel("energy (eV)")
        plt.ylabel("energy resolution fwhm (eV)")
        plt.grid("on")
        plt.title(f"chan {self.channum} cal comparison")

    def count_rate(self, goodonly=False, bin_s=60):
        g = self.cuts.good()
        if not goodonly:
            g[:] = True
        if isinstance(bin_s, (int, float)):
            bin_edge = np.arange(self.p_timestamp[g][0], self.p_timestamp[g][-1], bin_s)
        else:
            bin_edge = bin_s
        counts, bin_edge = np.histogram(self.p_timestamp[g], bin_edge)
        bin_centers = bin_edge[:-1] + 0.5 * (bin_edge[1] - bin_edge[0])
        rate = counts / float(bin_edge[1] - bin_edge[0])

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
                print(f"{n_and:6d} (and) {n_or:6d} (or) pulses cut by [{c1.upper()} and/or {c2.upper()}]")
        print()
        for cut_name in boolean_fields:
            print(f"{self.cuts.bad(cut_name).sum():6d} pulses cut by {cut_name.upper()}")
        print(f"{self.nPulses:6d} pulses total")

    @_add_group_loop()
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
        if self.saved_auto_cuts is None:
            forceNew = True
        if not forceNew:
            LOG.info("channel %g skipping auto cuts because cuts exist", self.channum)
            return

        if clearCuts:
            self.clear_cuts()

        # Step 1: peak and rise times
        if self.peak_samplenumber is None:
            self._compute_peak_samplenumber()
        MARGIN = 3  # step at least this many samples forward before cutting.
        peak_time_ms = (MARGIN + self.peak_samplenumber - self.nPresamples) * self.timebase * 1000

        # Step 2: analyze *noise* so we know how to cut on pretrig rms postpeak_deriv
        pretrigger_rms = np.zeros(self.noise_records.nPulses)
        for i in range(self.noise_records.nPulses):
            data = self.noise_records.datafile.alldata[i]
            pretrigger_rms[i] = data[:self.nPresamples].std()
        max_deriv = mass.analysis_algorithms.compute_max_deriv(data, ignore_leading=0)

        # Multiply MAD by 1.4826 to get into terms of sigma, if distribution were Gaussian.
        md_med = np.median(max_deriv)
        pt_med = np.median(pretrigger_rms)
        md_madn = np.median(np.abs(max_deriv - md_med)) * 1.4826
        pt_madn = np.median(np.abs(pretrigger_rms - pt_med)) * 1.4826
        md_max = md_med + md_madn * nsigma_max_deriv
        pt_max = max(0.0, pt_med + pt_madn * nsigma_pt_rms)

        # Step 2.5: In the case of pretrig_rms, cut no more than pretrig_rms_percentile percent
        # of the pulses on the upper end. This appears to be appropriate for
        # SLEDGEHAMMER gamma devices, but may not be appropriate in cases where
        # there are many pulses riding on tails, so by default we don't do
        # this.
        if pretrig_rms_percentile is not None:
            pt_max = max(pt_max, np.percentile(self.p_pretrig_rms, pretrig_rms_percentile))

        # Step 3: make the cuts
        cuts = mass.core.controller.AnalysisControl(
            peak_time_ms=(0, peak_time_ms * 1.25),
            rise_time_ms=(0, peak_time_ms * 1.10),
            pretrigger_rms=(None, pt_max),
            postpeak_deriv=(None, md_max),
        )
        cuts._pretrig_rms_median = pt_med  # store these so we can acess them when writing projectors to hdf5
        cuts._pretrig_rms_sigma = pt_madn

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
            self.saved_auto_cuts = None
            return
        cuts = mass.AnalysisControl()
        for attrname in ("peak_time_ms", "rise_time_ms", "pretrigger_rms",
                         "postpeak_deriv"):
            cuts.cuts_prm[attrname] = (None, g.attrs[attrname])
        self.saved_auto_cuts = cuts

    @_add_group_loop()
    def smart_cuts(self, threshold=10.0, n_trainings=10000, forceNew=False):
        """Young! Why is there no doc string here??"""
        # first check to see if this had already been done
        if all(self.cuts.good("smart_cuts")) or forceNew:

            mdata = np.vstack([self.p_pretrig_mean[:n_trainings], self.p_pretrig_rms[:n_trainings],
                               self.p_min_value[:n_trainings], self.p_postpeak_deriv[:n_trainings]])
            mdata = mdata.transpose()

            robust = sklearn.covariance.MinCovDet().fit(mdata)

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

    @_add_group_loop()
    def flag_crosstalking_pulses(self, priorTime, postTime, combineCategories=True,
                                 nearestNeighborsDistances=1, crosstalk_key='is_crosstalking',
                                 forceNew=False):
        ''' Uses a list of nearest neighbor channels to flag pulses in current channel based
            on arrival times of pulses in neighboring channels

            Args:
            priorTime (float): amount of time to check, in ms, before the pulse arrival time
            postTime (float): amount of time to check, in ms, after the pulse arrival time
            combineChannels (bool): whether to combine all neighboring channel pulses for flagging crosstalk
            nearestNeighborDistances (int or int array): nearest neighbor distances to use for flagging,
                i.e. 1 = 1st nearest neighbors, 2 = 2nd nearest neighbors, etc.
            forceNew (bool): whether to re-compute the crosstalk cuts (default False)
        '''

        def crosstalk_flagging_loop(channelsToCompare):
            ''' Main algorithm for flagging crosstalking pulses in current victim channel,
                given a list of perpetrator channels

                Args:
                channelsToCompare: (int array): list of perpetrator channel numbers to compare against
            '''
            # Initialize array that will include the pulses from all neighboring channels
            compareChannelsPulsesList = np.array([])
            # Iterate through all neighboring channels and put pulse timestamps into combined list
            for compare_channum in channelsToCompare:
                if compare_channum not in self.tes_group.channel:
                    continue
                dsToCompare = self.tes_group.channel[compare_channum]
                # Combine the pulses from all neighboring channels into a single array
                compareChannelsPulsesList = np.append(compareChannelsPulsesList,
                                                      dsToCompare.p_subframecount[:] * dsToCompare.subframe_timebase)
            # Create a histogram of the neighboring channel pulses using the bin edges from the channel you are flagging
            hist, _bin_edges = np.histogram(compareChannelsPulsesList, bins=combinedEdges)
            # Even corresponds to bins with a photon in channel 1 (crosstalk), odd are empty bins (no crosstalk)
            badCountsHist = hist[::2]
            # Even only histogram indices map directly to previously good flagged pulse indices for victim channel
            crosstalking_pulses = badCountsHist > 0.0
            return crosstalking_pulses

        # Check to see if nearest neighbors list has already been set, otherwise skip
        nn_channel_key = 'nearest_neighbors'
        if nn_channel_key in self.hdf5_group.keys():

            # Set up combined crosstalk flag array in hdf5 file
            h5grp = self.hdf5_group.require_group('crosstalk_flags')

            crosstalk_array_dtype = bool
            self.__dict__[f'p_{crosstalk_key}'] = h5grp.require_dataset(
                crosstalk_key, shape=(self.nPulses,), dtype=crosstalk_array_dtype)

            if not combineCategories:
                for neighborCategory in self.hdf5_group[nn_channel_key]:
                    categoryField = str(crosstalk_key + '_' + neighborCategory)
                    self.__dict__[f'p_{categoryField}'] = h5grp.require_dataset(
                        categoryField, shape=(self.nPulses,), dtype=crosstalk_array_dtype)

            # Check to see if crosstalk list has already been written and skip, unless forceNew
            if (not np.any(h5grp[crosstalk_key][:]) or forceNew):

                # Convert from ms input to s used in rest of MASS
                priorTime /= 1000.0
                postTime /= 1000.0

                # Create uneven histogram edges, with a specified amount of time before and after a photon event
                pulseTimes = self.p_subframecount[:] * self.subframe_timebase

                # Create start and stop edges around pulses corresponding to veto times
                startEdges = pulseTimes - priorTime
                stopEdges = pulseTimes + postTime
                combinedEdges = np.sort(np.append(startEdges, stopEdges))

                if combineCategories:
                    LOG.info('Checking crosstalk between channel %d and combined neighbors...', self.channum)
                    # Combine all nearest neighbor pairs for this channel into a single list
                    combinedNearestNeighbors = np.array([])
                    # Loop through nearest neighbor categories
                    for neighborCategory in self.hdf5_group[nn_channel_key]:
                        subgroupName = nn_channel_key + '/' + neighborCategory
                        subgroupNeighbors = self.hdf5_group[subgroupName + '/neighbors_list'].value
                        # Remove duplicates, sort
                        selectNeighbors = subgroupNeighbors[:, 0][np.isin(
                            subgroupNeighbors[:, 2], nearestNeighborsDistances)]
                        combinedNearestNeighbors = np.unique(
                            np.append(combinedNearestNeighbors, selectNeighbors).astype(int))
                    if np.sum(np.isin(self.tes_group.channel.keys(), selectNeighbors)) > 0:
                        h5grp[crosstalk_key][:] = crosstalk_flagging_loop(combinedNearestNeighbors)
                    else:
                        msg = f"Channel {self.channum} skipping crosstalk cuts: no nearest neighbors matching criteria"
                        LOG.info(msg)

                else:
                    for neighborCategory in self.hdf5_group[nn_channel_key]:
                        categoryField = str(crosstalk_key + '_' + neighborCategory)
                        subgroupName = nn_channel_key + '/' + neighborCategory
                        subgroupNeighbors = self.hdf5_group[subgroupName + '/neighbors_list'].value
                        selectNeighbors = subgroupNeighbors[:, 0][np.isin(
                            subgroupNeighbors[:, 2], nearestNeighborsDistances)]
                        if np.sum(np.isin(self.tes_group.channel.keys(), selectNeighbors)) > 0:
                            LOG.info(f'Checking crosstalk between channel {self.channum} and {neighborCategory} neighbors...')
                            h5grp[categoryField][:] = crosstalk_flagging_loop(selectNeighbors)
                            h5grp[crosstalk_key][:] = np.logical_or(
                                h5grp[crosstalk_key], h5grp[categoryField])
                        else:
                            msg = f"channel {self.channum} skipping {neighborCategory} crosstalk cuts because"
                            msg += " no nearest neighbors matching criteria in category"
                            LOG.info(msg)

            else:
                LOG.info("channel %d skipping crosstalk cuts because it was already done", self.channum)

        else:
            LOG.info("channel %d skipping crosstalk cuts because nearest neighbors not set", self.channum)

    @_add_group_loop()
    def set_nearest_neighbors_list(self, mapFilename, nearestNeighborCategory='physical',
                                   distanceType='cartesian', forceNew=False):
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
        distanceType (str): Type of distance to measure between nearest neighbors, i.e. cartesian
        forceNew (bool): whether to re-compute nearest neighbors list if it exists (default False)
        '''

        def calculate_cartesian_squared_distances():
            '''
            Stores the cartesian distance from the current channel to all other channels
            on the array within nearest_neighbors_dictionary.
            '''

            # Create a dictionary to link all squared distances in the given space
            # to n for the n-th nearest neighbor.
            # Find the maximum distance in each dimension
            maxDistances = np.amax(positionValues, axis=0) - np.amin(positionValues, axis=0)
            # Create a list of possible squared distances for each dimension
            singleDimensionalSquaredDistances = []
            for iDim in range(nDims):
                tempSquaredDistances = []
                for iLength in range(maxDistances[iDim] + 1):
                    tempSquaredDistances.append(iLength**2)
                singleDimensionalSquaredDistances.append(tempSquaredDistances)
            # Use np.meshgrid to make an array of all possible combinations of the arrays in each dimension
            possibleSquaredCombinations = np.meshgrid(
                *[iArray for iArray in singleDimensionalSquaredDistances])
            # Sum the squared distances along the corresponding axis to get squared distance at each point
            allSquaredDistances = possibleSquaredCombinations[0]
            for iDim in range(1, nDims):
                allSquaredDistances += possibleSquaredCombinations[iDim]
            # Make a sorted list of the unique squared values
            uniqueSquaredDistances = np.unique(allSquaredDistances)
            # Create a dictionary with squared distance keys and the index of the squared distance array
            # as the n value for n-th nearest neighbor
            squaredDistanceDictionary = {}
            for nthNeighbor, nthNeighborSquaredDistance in enumerate(uniqueSquaredDistances):
                squaredDistanceDictionary[nthNeighborSquaredDistance] = nthNeighbor

            # Loop through channel map and save an hdf5 dataset including the
            # channum, cartesian squared distance, and nearest neighbor n
            for channelIndex, distant_channum in enumerate(channelNumbers):
                distantPos = np.array(positionValues[distant_channum == channelNumbers][0], ndmin=1)
                squaredDistance = 0.0
                for iDim in range(nDims):
                    squaredDistance += (channelPos[iDim] - distantPos[iDim])**2.0
                h5grp[nearestNeighborCategory]['neighbors_list'][channelIndex, 0] = distant_channum
                h5grp[nearestNeighborCategory]['neighbors_list'][channelIndex, 1] = squaredDistance
                h5grp[nearestNeighborCategory]['neighbors_list'][channelIndex, 2] = \
                    squaredDistanceDictionary[squaredDistance]

        def calculate_manhattan_distances():
            raise Exception("not implemented")

        def process_matching_channel(positionToCompare):
            '''
            Returns the channel number of a neighboring position after checking for goodness

            Args:
            positionToCompare (int array) - position to check for nearest neighbor match
            '''
            # Find the channel number corresponding to the compare position
            channelToCompare = channelNumbers[np.all(positionToCompare == positionValues, axis=1)]
            # If the new position exists in map file and the channel to compare to is good, return the channel number
            if (positionToCompare in positionValues) & (channelToCompare in self.tes_group.good_channels):
                return channelToCompare
            # Return an empty array if not actually a good nearest neighbor
            else:
                return np.array([], dtype=int)

        # Load channel numbers and positions from map file, define number of dimensions
        mapData = np.loadtxt(mapFilename, dtype=int)
        channelNumbers = np.array(mapData[:, 0], dtype=int)
        positionValues = mapData[:, 1:]
        nDims = positionValues.shape[1]

        # Set up hdf5 group and repopulate arrays, if already calculated earlier
        h5grp = self.hdf5_group.require_group('nearest_neighbors')
        h5grp.require_group(nearestNeighborCategory)

        # Victim channel position dataset
        self.__dict__[f'position_{nearestNeighborCategory}'] = \
            h5grp[nearestNeighborCategory].require_dataset(
                'position', shape=(nDims,), dtype=np.int64)

        # Perpetrator channels dataset
        hnnc = h5grp[nearestNeighborCategory]
        self.__dict__[f'neighbors_list_{nearestNeighborCategory}'] = \
            hnnc.require_dataset('neighbors_list', shape=((len(channelNumbers), 3)),
                                 dtype=np.int64)

        # Check to see if if data set already exists or if forceNew is set to True
        if not np.any(h5grp[nearestNeighborCategory]['neighbors_list'][:]) or forceNew:

            # Extract channel number and position of current channel, store in hdf5 file
            channum = self.channum
            channelPos = np.array(positionValues[channum == channelNumbers][0], ndmin=1)
            h5grp[nearestNeighborCategory]['position'][:] = channelPos

            # Calculate distances and store in hdf5 file
            if distanceType == 'cartesian':
                calculate_cartesian_squared_distances()
            elif distanceType == 'manhattan':
                calculate_manhattan_distances()
            else:
                raise Exception('Distance type ' + distanceType + ' not recognized.')

    def hist(self, bin_edges, attr="p_energy", t0=0, tlast=1e20, category={}, g_func=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute).

        Automatically filtes out nan values

        Parameters
        ----------
        bin_edges : _type_
            edges of bins unsed for histogram
        attr : str, optional
            which attribute to histogram "p_energy" or "p_filt_value", by default "p_energy"
        t0 : int, optional
            cuts all pulses before this time before fitting, by default 0
        tlast : _type_, optional
            cuts all pulses after this time before fitting, by default 1e20
        category : dict, optional
            _description_, by default {}
        g_func : _type_, optional
            a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer, by default None

        Returns
        -------
        ndarray, ndarray
            Histogram bin *centers*, counts
        """
        bin_edges = np.array(bin_edges)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        vals = getattr(self, attr)[:]
        # sanitize the data bit
        tg = np.logical_and(self.p_timestamp[:] > t0, self.p_timestamp[:] < tlast)
        g = np.logical_and(tg, self.good(**category))
        g = np.logical_and(g, ~np.isnan(vals))
        if g_func is not None:
            g &= g_func(self)

        counts, _ = np.histogram(vals[g], bin_edges)
        return bin_centers, counts

    def plot_hist(self, bin_edges, attr="p_energy", axis=None, label_lines=[], category={}, g_func=None):
        """plot a coadded histogram from all good datasets and all good pulses
        bin_edges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        axis -- if None, then create a new figure, otherwise plot onto this axis
        annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer    """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        x, y = self.hist(bin_edges, attr, category=category, g_func=g_func)
        axis.plot(x, y, drawstyle="steps-mid")
        axis.set_xlabel(attr)
        de = bin_edges[1] - bin_edges[0]
        axis.set_ylabel(f"counts per {de:0.1f} unit bin")
        axis.set_title(self.shortname)
        mass.core.utilities.annotate_lines(axis, label_lines)

    def linefit(self, line_name="MnKAlpha", t0=0, tlast=1e20, axis=None, dlo=50, dhi=50,  # noqa: PLR0917
                binsize=1, bin_edges=None, attr="p_energy", label="full", plot=True,
                guess_params=None, ph_units="eV", category={}, g_func=None,
                has_tails=False):
        """Do a fit to `line_name` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other
        way you like.

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
        """
        assert "energy" in attr
        model = mass.getmodel(line_name, has_tails=has_tails)
        nominal_peak_energy = model.spect.nominal_peak_energy
        if bin_edges is None:
            bin_edges = np.arange(nominal_peak_energy - dlo, nominal_peak_energy + dhi, binsize)

        bin_centers, counts = self.hist(bin_edges, attr, t0, tlast, category, g_func)

        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(vary=False)
        result = model.fit(counts, params=params, bin_centers=bin_centers)
        if plot:
            result.plotm(ax=axis, xlabel=f"{attr} ({ph_units})",
                         ylabel=f"counts per {binsize:0.2f} ({ph_units}) bin",
                         title=f"{self.shortname}\n{model.spect}")

        return result


def time_drift_correct(time, uncorrected, w, sec_per_degree=2000,  # noqa: PLR0914
                       pulses_per_degree=2000, max_degrees=20, ndeg=None, limit=None):
    """Compute a time-based drift correction that minimizes the spectral entropy.

    Args:
        time: The "time-axis". Correction will be a low-order polynomial in this.
        uncorrected: A filtered pulse height vector. Same length as indicator.
            Assumed to have some gain that is linearly related to indicator.
        w: the kernel width for the Laplace KDE density estimator
        sec_per_degree: assign as many as one polynomial degree per this many seconds
        pulses_per_degree: assign as many as one polynomial degree per this many pulses
        max_degrees: never use more than this many degrees of Legendre polynomial.
        n_deg: If not None, use this many degrees, regardless of the values of
               sec_per_degree, pulses_per_degree, and max_degress. In this case, never downsample.
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
        pct99 = np.percentile(uncorrected, 99)
        limit = [0, 1.25 * pct99]

    use = np.logical_and(uncorrected > limit[0], uncorrected < limit[1])
    time = time[use]
    uncorrected = uncorrected[use]

    tmin, tmax = np.min(time), np.max(time)

    def normalize(t):
        return (t - tmin) / (tmax - tmin) * 2 - 1

    info = {
        "tmin": tmin,
        "tmax": tmax,
        "normalize": normalize,
    }

    dtime = tmax - tmin
    N = len(time)
    if ndeg is None:
        ndeg = int(np.minimum(dtime / sec_per_degree, N / pulses_per_degree))
        ndeg = min(ndeg, max_degrees)
        ndeg = max(ndeg, 1)
        phot_per_degree = N / float(ndeg)
        if phot_per_degree >= 2 * pulses_per_degree:
            downsample = int(phot_per_degree / pulses_per_degree)
            time = time[::downsample]
            uncorrected = uncorrected[::downsample]
            N = len(time)
        else:
            downsample = 1
    else:
        downsample = 1

    LOG.info("Using %2d degrees for %6d photons (after %d downsample)", ndeg, N, downsample)
    LOG.info("That's %6.1f photons per degree, and %6.1f seconds per degree.", N / float(ndeg), dtime / ndeg)

    def model1(pi, i, param, basis):
        pcopy = np.array(param)
        pcopy[i] = pi
        return 1 + np.dot(basis.T, pcopy)

    def cost1(pi, i, param, y, w, basis):
        return laplace_entropy(y * model1(pi, i, param, basis), w=w)

    param = np.zeros(ndeg, dtype=float)
    xnorm = np.asarray(normalize(time), dtype=float)
    basis = np.vstack([sp.special.legendre(i + 1)(xnorm) for i in range(ndeg)])

    fc = 0
    model = np.poly1d([0])
    info["coefficients"] = np.zeros(ndeg, dtype=float)
    for i in range(ndeg):
        result, _fval, _iter, funcalls = sp.optimize.brent(
            cost1, (i, param, uncorrected, w, basis), [-.001, .001], tol=1e-5, full_output=True)
        param[i] = result
        fc += funcalls
        model += sp.special.legendre(i + 1) * result
        info["coefficients"][i] = result
    info["funccalls"] = fc

    xk = np.linspace(-1, 1, 1 + 2 * ndeg)
    model2 = mass.mathstat.interpolate.CubicSpline(xk, model(xk))
    H1 = laplace_entropy(uncorrected, w=w)
    H2 = laplace_entropy(uncorrected * (1 + model(xnorm)), w=w)
    H3 = laplace_entropy(uncorrected * (1 + model2(xnorm)), w=w)
    if H2 <= 0 or H2 - H1 > 0.0:
        model = np.poly1d([0])
    elif H3 <= 0 or H3 - H2 > .00001:
        model2 = model

    info["entropies"] = (H1, H2, H3)
    info["model"] = model
    return info
