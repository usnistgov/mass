"""
Created on Feb 16, 2011

@author: fowlerj
"""

from functools import reduce

try:
    import cPickle as pickle
except ImportError:
    import pickle

import h5py
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pylab as plt

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

from mass.core import ljh_util


def log_and(a, b, *args):
    """Generalize np.logical_and() to 2 OR MORE arguments."""
    return reduce(np.logical_and, args, np.logical_and(a, b))


class NoiseRecords(object):
    """
    Encapsulate a set of noise records, which can either be
    assumed continuous or arbitrarily separated in time.
    """
    DEFAULT_MAXSEGMENTSIZE = 32000000

    ALLOWED_TYPES = ("ljh", "virtual")

    def __init__(self, filename, records_are_continuous=False, use_records=None,
                 maxsegmentsize=None, hdf5_group=None):
        """
        Load a noise records file.

        If <records_are_continuous> is True, then treat all pulses as a continuous timestream.
        <use_records>  can be a sequence (first,end) to use only a limited section of the file.
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
        """Use when this does not correspond to a real datafile (e.g., CDM data)"""
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
        separate records of <seg_length> length.  (If None, then self.data.shape[0] which is
        self.data.nPulses, will be used as the number of segments, each having length
        self.data.nSamples.)

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
        assert self.continuous

        # Does it assume that all data fit into a single segment?
        self.datafile.read_segment(0)
        n = np.prod(self.datafile.data.shape)
        if seglength_choices is None:
            longest_seg = 1
            while longest_seg <= n//16:
                longest_seg *= 2
            seglength_choices = [longest_seg]
            while seglength_choices[-1] > 256:
                seglength_choices.append(seglength_choices[-1]//4)
            print("Will use segments of length: %s"%seglength_choices)

        spectra = [self.compute_power_spectrum_reshape(window=window, seg_length=seglen)
                   for seglen in seglength_choices]
        if plot:
            plt.clf()
            lowest_freq = np.array([1./(sp.dt*sp.m2) for sp in spectra])

            start_freq = 0.0
            for i, sp in enumerate(spectra):
                x, y = sp.frequencies(), sp.spectrum()
                if i == len(spectra)-1:
                    good = x >= start_freq
                else:
                    good = np.logical_and(x >= start_freq, x < 10*lowest_freq[i+1])
                plt.loglog(x[good], y[good], '-')
                start_freq = lowest_freq[i] * 10
        return spectra

    def plot_power_spectrum(self, axis=None, scale=1.0, sqrt_psd=False, **kwarg):
        """
        Plot the power spectrum of this noise record.

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

#        print 'Compute continuous autocorr (%d lags on %d data) '%(n_lags, n_data),

        def padded_length(n):
            """Return a sensible number in the range [n, 2n] which is not too
            much larger than n, yet is good for FFTs.
            That is, choose (1, 3, or 5)*(a power of two), whichever is smallest
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
        CHUNK_MULTIPLE=15
        if n_data >= (1 + CHUNK_MULTIPLE) * n_lags:
            # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
            # unwanted wraparound in the autocorrelation.
            # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
            chunksize=CHUNK_MULTIPLE * n_lags
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
                data_mean = data[data_consumed:samples_this_segment].mean()

                # Notice that the following loop might ignore the last data values, up to as many
                # as (chunksize-1) values, unless the data are an exact multiple of chunksize.
                while data_consumed+chunksize <= samples_this_segment:
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

            ac /= entries
            ac /= (np.arange(chunksize, chunksize-n_lags+0.5, -1.0, dtype=np.float))

        # compute the full autocorrelation
        else:
            raise NotImplementedError("Now that Joe has chunkified the noise, we can "
                                      "no longer compute full continuous autocorrelations")
#             padded_data = np.zeros(padded_length(n_lags+n_data), dtype=np.float)
#             padded_data[:n_data] = np.array(self.data.ravel())[:n_data] - self.data.mean()
#             padded_data[n_data:] = 0.0
#
#             ft = np.fft.rfft(padded_data)
#             del padded_data
#             ft[0] = 0  # this redundantly removes the mean of the data set
#             ft *= ft.conj()
#             ft = ft.real
#             acsum = np.fft.irfft(ft)
#             del ft
#             ac = acsum[:n_lags+1] / (n_data-np.arange(n_lags + 1.0))
#             del acsum

        self.autocorrelation[:] = ac

    def compute_autocorrelation(self, n_lags=None, data_samples=None, plot=True, max_excursion=1000):
        """
        Compute the autocorrelation averaged across all "pulses" in the file.
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
        if all(self.autocorrelation[:] == 0):
            print("Autocorrelation will be computed first")
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
    and so forth. It is meant to be only a file interface (though until July 2014, this
    was not exactly the case).
    """

    ALLOWED_TYPES = ("ljh", "virtual")

    def __init__(self, filename, file_format=None):
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

        self.cuts = None
        self.bad = None
        self.good = None
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
        """Update the underlying file's segment (read chunk) size in bytes."""
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


class MicrocalDataSet(object):
    """
    Represent a single microcalorimeter's PROCESSED data.
    This channel can be directly from a TDM detector, or it
    can be the demodulated result of a CDM modulation.
    """

    # Attributes that all such objects must have.
    expected_attributes = ("nSamples", "nPresamples", "nPulses", "timebase", "channum",
                           "timestamp_offset")

    HDF5_CHUNK_SIZE = 256

    def __init__(self, pulserec_dict, tes_group=None, hdf5_group=None):
        """
        Pass in a dictionary (presumably that of a PulseRecords object)
        containing the expected attributes that must be copied to this
        MicrocalDataSet.
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
        self.noise_autocorr = None
        self.noise_demodulated = None
        self.calibration = {}

        for a in self.expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename', 'virtual data set')
        self.gain = 1.0
        self.pretrigger_ignore_microsec = None  # Cut this long before trigger in computing pretrig values
        self.pretrigger_ignore_samples = 0
        self.peak_time_microsec = None   # Look for retriggers only after this time.
        self.index = None   # Index in the larger TESGroup or CDMGroup object
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

        self.tes_group = tes_group

        try:
            self.hdf5_group = hdf5_group
            self.hdf5_group.attrs['npulses'] = self.nPulses
            self.hdf5_group.attrs['channum'] = self.channum
        except KeyError:
            self.hdf5_group = None

        self.__setup_vectors(npulses=self.nPulses)

    def __setup_vectors(self, npulses=None):
        """Given the number of pulses, build arrays to hold the relevant facts
        about each pulse in memory.
        p_filt_value = pulse height after running through filter
        P_filt_value_phc = phase corrected
        p_fil_value_dc = pulse height after running through filter and applying drift correction
        p_energy = pulse energy determined from applying a calibration to one of the p_filt_value??? variables"""

        if npulses is None:
            assert self.nPulses > 0
            npulses = self.nPulses

        h5grp = self.hdf5_group

        # Set up the per-pulse vectors
        float64_fields = ('timestamp',)
        float32_fields = ('pretrig_mean', 'pretrig_rms', 'pulse_average', 'pulse_rms',
                          'promptness', 'rise_time', 'postpeak_deriv',
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
                                                    self.tes_group.nPresamples - self.pretrigger_ignore_samples,
                                                    self.noise_autocorr,
                                                    fmax=fmax, f_3db=f_3db,
                                                    sample_time=self.timebase,
                                                    peak=modelpeak)
            else:
                self.filter = Filter(self.average_pulse[...],
                                     self.tes_group.nPresamples - self.pretrigger_ignore_samples,
                                     self.noise_psd[...],
                                     self.noise_autocorr, sample_time=self.timebase,
                                     fmax=fmax, f_3db=f_3db,
                                     shorten=shorten)

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

    @property
    def p_peak_time(self):
        # this is a property to reduce memory usage, I hope it works
        return (self.p_peak_index[:] - self.nPresamples) * self.timebase

    @property
    def external_trigger_rowcount(self):
        if not self._external_trigger_rowcount:
            filename = ljh_util.ljh_get_extern_trig_fname(self.filename)
            h5 = h5py.File(filename,"r")
            ds_name = "trig_times_w_offsets" if "trig_times_w_offsets" in h5 else "trig_times"
            self._external_trigger_rowcount = h5[ds_name]
            self.row_timebase = self.timebase/float(self.number_of_rows)
        return self._external_trigger_rowcount

    @property
    def external_trigger_rowcount_as_seconds(self):
        """
        This is not a posix timestamp, it is just the external trigger rowcount converted to seconds
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
        """Return a boolean vector, one per pulse record, saying whether record is good"""
        return self.cuts.good(*args, **kwargs)

    def bad(self, *args, **kwargs):
        """Return a boolean vector, one per pulse record, saying whether record is bad"""
        return self.cuts.bad(*args, **kwargs)

    def resize(self, nPulses):
        if self.nPulses < nPulses:
            raise ValueError("Can only shrink using resize(), but the requested size %d is larger than current %d" %
                             (nPulses, self.nPulses))
        self.nPulses = nPulses
        self.__setup_vectors()

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = MicrocalDataSet(self.__dict__)
        c.__dict__.update(self.__dict__)
        for k in self.calibration.keys():
            c.calibration[k] = self.calibration[k].copy()
        c.cuts = self.cuts.copy()
        return c

    @show_progress("channel.summarize_data_tdm")
    def summarize_data(self, peak_time_microsec=None, pretrigger_ignore_microsec=None, forceNew=False):
        """Summarize the complete data set one chunk at a time.
        """
        # Don't proceed if not necessary and not forced
        self.number_of_rows = self.pulse_records.datafile.number_of_rows
        self.row_number = self.pulse_records.datafile.row_number
        self.number_of_columns = self.pulse_records.datafile.number_of_columns
        self.column_number = self.pulse_records.datafile.column_number

        not_done = all(self.p_pretrig_mean[:] == 0)
        if not (not_done or forceNew):
            print('\nchan %d did not summarize because results were already preloaded' % self.channum)
            return

        if len(self.p_timestamp) < self.pulse_records.nPulses:
            self.__setup_vectors(npulses=self.pulse_records.nPulses)  # make sure vectors are setup correctly

        if peak_time_microsec is None:
            self.peak_samplenumber = None
        else:
            self.peak_samplenumber = int(peak_time_microsec*1e-6/self.timebase)
        if pretrigger_ignore_samples is None:
            self.pretrigger_ignore_samples = 3
        else:
            self.pretrigger_ignore_samples = int(peak_time_microsec*1e-6/self.timebase)

        for segnum in range(self.pulse_records.n_segments):
            self._summarize_data_segment(segnum)
            yield (segnum+1) / float(self.pulse_records.n_segments)
        self.pulse_records.datafile.clear_cached_segment()
        self.hdf5_group.file.flush()

    def _summarize_data_segment(self, segnum):
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
            peak_idx = self.data.argmax(axis=1)
            self.peak_samplenumber = sp.stats.mode(peak_idx)[0][0]

        seg_size = end-first
        self.p_timestamp[first:end] = self.times[:seg_size]
        self.p_rowcount[first:end] = self.rowcount[:seg_size]
        self.p_pretrig_mean[first:end] = \
            self.data[:seg_size, :self.nPresamples-self.pretrigger_ignore_samples].mean(axis=1)
        self.p_pretrig_rms[first:end] = \
            self.data[:seg_size, :self.nPresamples-self.pretrigger_ignore_samples].std(axis=1)
        self.p_peak_index[first:end] = self.data[:seg_size, :].argmax(axis=1)
        self.p_peak_value[first:end] = self.data[:seg_size, :].max(axis=1)
        self.p_min_value[first:end] = self.data[:seg_size, :].min(axis=1)
        self.p_pulse_average[first:end] = self.data[:seg_size, self.nPresamples:].mean(axis=1)

        # Remove the pretrigger mean from the peak value and the pulse average figures.
        ptm = self.p_pretrig_mean[first:end]
        self.p_pulse_average[first:end] -= ptm
        self.p_peak_value[first:end] -= np.asarray(ptm, dtype=self.p_peak_value.dtype)
        self.p_pulse_rms[first:end] = np.sqrt(
            (self.data[:seg_size, self.nPresamples:]**2.0).mean(axis=1) -
            ptm*(ptm + 2*self.p_pulse_average[first:end]))

        shift1 = (self.data[:seg_size,self.nPresamples+2]-ptm >
                  4.3*self.p_pretrig_rms[first:end])
        self.p_shift1[first:end] = shift1

        halfidx = (self.nPresamples+5+self.peak_samplenumber)//2
        pkval = self.p_peak_value[first:end]
        prompt = (self.data[:seg_size, self.nPresamples+5:halfidx].mean(axis=1)
                  - ptm) / pkval
        prompt[shift1] = (self.data[shift1, self.nPresamples+4:halfidx-1].mean(axis=1)
                  - ptm[shift1]) / pkval[shift1]
        self.p_promptness[first:end] = prompt

        self.p_rise_time[first:end] = \
            mass.core.analysis_algorithms.estimateRiseTime(self.data[:seg_size],
                                                           timebase=self.timebase,
                                                           nPretrig=self.nPresamples)

        self.p_postpeak_deriv[first:end] = \
            mass.core.analysis_algorithms.compute_max_deriv(self.data[:seg_size],
                                                            ignore_leading=self.peak_samplenumber)

    @show_progress("compute_average_pulse")
    def compute_average_pulse(self, mask, subtract_mean=True, forceNew=False):
        """Compute the average pulse this channel.

        mask -- A boolean array saying which records to average.
        subtract_mean -- Whether to subtract the pretrigger mean and set the
            pretrigger period to strictly zero.
        forceNew -- Whether to recompute when already exists
        """
        # Don't proceed if not necessary and not forced
        already_done = self.average_pulse[-1] != 0
        if already_done and not forceNew:
            print("skipping compute average pulse on chan %d" % self.channum)
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
        average_pulse = pulse_sum / pulse_count
        if subtract_mean:
            average_pulse -= np.mean(average_pulse[:self.nPresamples - self.pretrigger_ignore_samples])
        self.average_pulse[:] = average_pulse
        print()

    def compute_oldfilter(self, fmax=None, f_3db=None):
        try:
            spectrum = self.noise_spectrum.spectrum()
        except:
            spectrum = self.noise_psd[:]

        avg_signal = np.array(self.average_pulse)
        f = mass.core.Filter(avg_signal, self.nPresamples-self.pretrigger_ignore_samples,
                             spectrum, self.noise_autocorr, sample_time=self.timebase,
                             fmax=fmax, f_3db=f_3db, shorten=2)
        f.compute()
        return f

    def compute_newfilter(self, fmax=None, f_3db=None, transform=None, DEGREE = 1):
        data, pulsenums = self.first_n_good_pulses(1000)

        # The raw training data, which is shifted (trigger-aligned)
        raw = data[:, 1:]
        shift1 = self.p_shift1[:][pulsenums]
        raw[shift1, :] = data[shift1, 0:-1]

        # Center promptness around 0, using a simple function of Prms
        prompt = self.p_promptness[:][pulsenums]
        prms = self.p_pulse_rms[:][pulsenums]
        mprms = np.median(prms)
        use = np.abs(prms/mprms-1.0) < 0.4
        promptshift = np.poly1d(np.polyfit(prms[use], prompt[use], 1))
        prompt -= promptshift(prms)

        # Scale it quadratically to cover the range -0.5 to +0.5, approximately
        x, y, z = sp.stats.scoreatpercentile(prompt[use], [10, 50, 90])
        A = np.array([[x*x, x, 1],
                      [y*y, y, 1],
                      [z*z, z, 1]])
        param = np.linalg.solve(A, [-.4, 0, +.4])
        ATime = np.poly1d(param)(prompt)
        use = np.logical_and(use, np.abs(ATime)<0.45)

        ptm = self.p_pretrig_mean[:][pulsenums]
        ptm.shape = (len(pulsenums), 1)
        raw = (raw-ptm)[use,:]
        if transform is not None:
            raw = transform(raw)
        rawscale = raw.max(axis=1)

        # Arrival time and a binned version of it
        ATime = ATime[use]
        NBINS = 9
        bins = np.digitize(ATime, np.linspace(ATime.min(), ATime.max(), NBINS+1))-1

        # Are all bins populated with at least 5 pulses?
        valid_bins = []
        for i in range(NBINS):
            if (bins==i).sum() >= 5:
                valid_bins.append(i)
        valid_bins = np.array(valid_bins)

        # Are there enough populated bins to use DEGREE?
        n_valid = len(valid_bins)
        if n_valid < 2:
            raise RuntimeError("Only %d valid arrival-time bins were found in compute_newfilter"%n_valid)
        if n_valid <= DEGREE:
            DEGREE = n_valid-1

        model = np.zeros((self.nSamples-1, 1+DEGREE), dtype=float)
        for s in range(self.nPresamples+2, self.nSamples-1):
            y = raw[:, s]/rawscale
            xmed = [np.median(ATime[bins == i]) for i in valid_bins]
            ymed = [np.median(y[bins == i]) for i in valid_bins]
            fit = np.polyfit(xmed, ymed, DEGREE)
            model[s, :] = fit[::-1]  # Reverse so order is [const, lin, quad...] terms

        modelpeak = np.median(rawscale)
        self.pulsemodel = model
        f = ArrivalTimeSafeFilter(model, self.nPresamples, self.noise_autocorr, fmax=fmax,
                                  f_3db=f_3db, sample_time=self.timebase, peak=modelpeak)
        f.compute(fmax=fmax, f_3db=f_3db)
        self.filter = f
        return f

    @show_progress("channel.filter_data_tdm")
    def filter_data(self, filter_name='filt_noconst', transform=None, forceNew=False):
        """Filter the complete data file one chunk at a time.
        """
        if not(forceNew or all(self.p_filt_value[:] == 0)):
            print('\nchan %d did not filter because results were already loaded' % self.channum)
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
        """Traditional 5-lag filter used by default until 2015"""
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

    def clear_cache(self):
        self.data = None
        self.rowcount = None
        self.times = None
        self.pulse_records.clear_cache()

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

        <valid> An array of booleans self.nPulses long saying which pulses are to be plotted
                *OR* 'uncut' or 'cut', meaning that only uncut or cut data are to be plotted
                *OR* None, meaning that all pulses should be plotted.

        <downsample> To prevent the scatter plots (left panels) from getting too crowded,
                     plot only one out of this many samples.  If None, then plot will be
                     downsampled to 10,000 total points.

        <log>  Use logarithmic y-axis on the histograms (right panels).
        """

        # Convert "uncut" or "cut" to array of all good or all bad data
        if isinstance(valid, str):
            if "uncut" in valid.lower():
                valid = self.cuts.good()
                print("Plotting only uncut data"),
            elif "cut" in valid.lower():
                valid = self.cuts.bad()
                print("Plotting only cut data"),
            elif 'all' in valid.lower():
                valid = None
                print("Plotting all data, cut or uncut"),
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
        print(" (%d records; %d in scatter plots)" % (nrecs, len(hour)))

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
        last_midnight = last_record - (last_record%86400)
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

    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        """<n_lags>, if not None, is the number of lags in each noise spectrum and the max lag
        for the autocorrelation.  If None, the record length is used."""
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
            print("chan %d skipping compute_noise_spectra because already done" % self.channum)

    def apply_cuts(self, controls=None, clear=False, verbose=1, forceNew=True):
        """
        <clear>  Whether to clear previous cuts first (by default, do not clear).
        <verbose> How much to print to screen.  Level 1 (default) counts all pulses good/bad/total.
                    Level 2 adds some stuff about the departure-from-median pretrigger mean cut.
        """
        if self.nPulses == 0:
            return  # don't bother current if there are no pulses
        if not forceNew:
            if self.cuts.good().sum() != self.nPulses:
                print("Chan %d skipped cuts: after %d are good, %d are bad of %d total pulses" %
                      (self.channum, self.cuts.good().sum(), self.cuts.bad().sum(), self.nPulses))

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
            self.cuts.cut_parameter(np.hstack((0.0, np.diff(self.p_timestamp))),
                                    c['timestamp_diff_sec'], 'timestamp_diff_sec')
        if c['pretrigger_mean_departure_from_median'] is not None and self.cuts.good().sum() > 0:
            median = np.median(self.p_pretrig_mean[self.cuts.good()])
            if verbose > 1:
                print('applying cut on pretrigger mean around its median value of ', median)
            self.cuts.cut_parameter(self.p_pretrig_mean-median,
                                    c['pretrigger_mean_departure_from_median'],
                                    'pretrigger_mean_departure_from_median')
        if verbose > 0:
            print("Chan %d after cuts, %d are good, %d are bad of %d total pulses" % (
                self.channum, self.cuts.good().sum(), self.cuts.bad().sum(), self.nPulses))

    def clear_cuts(self):
        self.cuts.clear_cut()

    def drift_correct(self, forceNew=False, category=None):
        """Drift correct using the standard entropy-minimizing algorithm"""
        doesnt_exist = all(self.p_filt_value_dc[:] == 0) or all(self.p_filt_value_dc[:] == self.p_filt_value[:])
        if not (forceNew or doesnt_exist):
            print("chan %d not drift correction, p_filt_value_dc already populated" % self.channum)
            return
        if category is None:
            category = {"calibration": "in"}
        g = self.cuts.good(**category)
        uncorrected = self.p_filt_value[g]
        indicator = self.p_pretrig_mean[g]
        drift_corr_param, self.drift_correct_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        self.p_filt_value_dc.attrs.update(self.drift_correct_info) # Store in hdf5 file
        print('chan %d best drift correction parameter: %.6f' % (self.channum, drift_corr_param))
        self._apply_drift_correction()

    def _apply_drift_correction(self):
        # Apply correction
        assert self.p_filt_value_dc.attrs["type"] == "ptmean_gain"
        ptm_offset = self.p_filt_value_dc.attrs["median_pretrig_mean"]
        gain = 1+(self.p_pretrig_mean[:]-ptm_offset)*self.p_filt_value_dc.attrs["slope"]
        self.p_filt_value_dc[:] = self.p_filt_value[:]*gain
        self.hdf5_group.file.flush()

    def phase_correct2014(self, typical_resolution, maximum_num_records=50000, plot=False, forceNew=False, category=None):
        """Apply the phase correction that seems good for calibronium-like
        data as of June 2014. For more notes, do
        help(mass.core.analysis_algorithms.FilterTimeCorrection)

        <typical_resolution> should be an estimated energy resolution in UNITS OF
        self.p_pulse_rms. This helps the peak-finding (clustering) algorithm decide
        which pulses go together into a single peak.  Be careful to use a semi-reasonable
        quantity here.
        """
        doesnt_exist = all(self.p_filt_value_phc[:] == 0) or all(self.p_filt_value_phc[:] == self.p_filt_value_dc[:])
        if not (forceNew or doesnt_exist):
            print("channel %d skipping phase_correct2014" % self.channum)
            return

        if category is None:
            category = {"calibration": "in"}
        data, g = self.first_n_good_pulses(maximum_num_records, category)
        print("channel %d doing phase_correct2014 with %d good pulses" % (self.channum, data.shape[0]))
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
        """A heuristic method to identify the peaks in a spectrum that can be used to
        design the arrival-time-bias correction. Of course, you might have better luck
        finding peaks by an experiment-specific method, but this will stand in if you
        cannot or do not want to find peaks another way.

        phnorm should be a vector of pulse heights, found by whatever means you like.
        Normally it will be the self.p_filt_value_dc AFTER CUTS.
        """
        median_scale = np.median(phnorm)

        # First make histogram with bins = 0.2% of median PH
        hist, bins = np.histogram(phnorm, 1000, [0, 2*median_scale])
        binctr = bins[1:] - 0.5 * (bins[1] - bins[0])

        # Scipy continuous wavelet transform
        pk1 = np.array(sp.signal.find_peaks_cwt(hist, np.array([2,4,8,12])))

        # A peak must contain 0.5% of the data or 500 events, whichever is more,
        # but the requirement is not more than 5% of data (for meager data sets)
        Ntotal = len(phnorm)
        MinCountsInPeak = min(max(500, Ntotal//200), Ntotal//20)
        pk2 = pk1[hist[pk1]>MinCountsInPeak]

        # Now take peaks from highest to lowest, provided they are at least 40 bins from any neighbor
        ordering = hist[pk2].argsort()
        pk2 = pk2[ordering]
        peaks = [pk2[0]]

        for pk in pk2[1:]:
            if (np.abs(peaks-pk) > 10).all():
                peaks.append(pk)
        peaks.sort()
        return np.array(binctr[peaks])


    def phase_correct(self, forceNew=False, category=None, ph_peaks=None, method2017=False,
                      kernel_width=None):
        """2017 or 2015 phase correction method. Arguments are:
        `forceNew`  To repeat computation if it already exists.
        `category`  From the new named/categorical cuts system.
        `ph_peaks`  Peaks to use for alignment. If None, then use self._find_peaks_heuristic()
        `kernel_width` Width (in PH units) of the kernel-smearing function. If None, use a heuristic.
        """
        doesnt_exist = all(self.p_filt_value_phc[:] == 0) or all(self.p_filt_value_phc[:] == self.p_filt_value_dc[:])
        if not (forceNew or doesnt_exist):
            print("channel %d skipping phase_correct" % self.channum)
            return

        if category is None:
            category = {"calibration": "in"}
        good = self.cuts.good(**category)

        if ph_peaks is None:
            ph_peaks = self._find_peaks_heuristic(self.p_filt_value_dc[good])
        if len(ph_peaks) <= 0:
            print ("Could not phase_correct on chan %3d because no peaks"%self.channum)
            return
        ph_peaks = np.asarray(ph_peaks)
        ph_peaks.sort()

        # Compute a correction function at each line in ph_peaks
        corrections = []
        median_phase = []
        if kernel_width is None:
            kernel_width = np.max(ph_peaks)/1000.0
        for pk in ph_peaks:
            c, mphase = phasecorr_find_alignment(self.p_filt_phase[good],
                                self.p_filt_value_dc[good], pk, .012*np.mean(ph_peaks),
                                method2017=method2017, kernel_width=kernel_width)
            corrections.append(c)
            median_phase.append(mphase)
        median_phase = np.array(median_phase)

        # Store the info needed to reconstruct corrections
        nc = np.hstack([len(c._x) for c in corrections])
        cx = np.hstack([c._x for c in corrections])
        cy = np.hstack([c._y for c in corrections])
        for name,data in zip(("phase_corrector_x", "phase_corrector_y", "phase_corrector_n"),
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
            NBINS=10
            dc = self.p_filt_value_dc[good]
            ph = self.p_filt_phase[good]
            top = min(dc.max(), 1.2*sp.stats.scoreatpercentile(dc, 98))
            bin = np.digitize(dc, np.linspace(0, top, 1+NBINS))-1
            x = np.zeros(NBINS, dtype=float)
            y = np.zeros(NBINS, dtype=float)
            w = np.zeros(NBINS, dtype=float)
            for i in range(NBINS):
                w[i] = (bin==i).sum()
                if w[i] == 0: continue
                x[i] = np.median(dc[bin==i])
                y[i] = np.median(ph[bin==i])

            nonempty = w>0
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
        corrected_phase = self.p_filt_phase_corr[:]
        corrected_phase[corrected_phase>0.6] = 0.6
        corrected_phase[corrected_phase<-0.6] = -0.6

        nc = self.hdf5_group["phase_corrector_n"][...]
        cx = self.hdf5_group["phase_corrector_x"][...]
        cy = self.hdf5_group["phase_corrector_y"][...]
        corrections = []
        idx=0
        for n in nc:
            x = cx[idx:idx+n]
            y = cy[idx:idx+n]
            idx += n
            spl = mass.mathstat.interpolate.CubicSpline(x,y)
            corrections.append(spl)

        self.p_filt_value_phc[:] = _phase_corrected_filtvals(corrected_phase, self.p_filt_value_dc, corrections)

        print('Channel %3d phase corrected. Correction size: %.2f' % (
            self.channum, mass.mathstat.robust.median_abs_dev(self.p_filt_value_phc[good] -
                                                              self.p_filt_value_dc[good], True)))
        self.phase_corrections = corrections
        return corrections


    def first_n_good_pulses(self, n=50000, category=None):
        """
        :param n: maximum number of good pulses to include
        :return: data, g
        data is a (X,Y) array where X is number of records, and Y is number of samples per record
        g is a 1d array of of pulse record numbers of the pulses in data
        if we  did load all of ds.data at once, this would be roughly equivalent to
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
                          nbins=200, verbose=True, plot=True, **kwargs):
        """
        <line> can be one of the fitters in mass.calibration.fluorescence_lines (e.g. 'MnKAlpha', 'CuKBeta') or
        in mass.calibration.gaussian_lines (e.g. 'Gd97'), or a number.  In this last case, it is assumed to
        be a single Gaussian line.
        """
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
        if verbose:
            print("%d events pass cuts; %d are in histogram range" % (len(good_values), contents.sum()))
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
        if verbose:
            print('Resolution: %5.2f +- %5.2f eV' % (params[0]*scale, np.sqrt(covar[0, 0])*scale))
        return params, covar, fitter

    @property
    def pkl_fname(self):
        return ljh_util.mass_folder_from_ljh_fname(self.filename, filename="ch%d_calibration.pkl" % self.channum)

    def calibrate(self, attr, line_names, name_ext="", size_related_to_energy_resolution=10,
                  fit_range_ev=200, excl=(), plot_on_fail=False,
                  bin_size_ev=2.0, category=None, forceNew=False, maxacc=0.015, nextra=3,
                  param_adjust_closure=None, diagnose=False):
        calname = attr+name_ext

        if not forceNew and calname in self.calibration:
            return self.calibration[calname]

        print("Calibrating chan %d to create %s" % (self.channum, calname))
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
            print("chan %d failed calibration because on of the fitter was a FailedFitter" % self.channum)
            raise Exception()

        self.calibration[calname] = cal
        hdf5_cal_group = self.hdf5_group.require_group('calibration')
        cal.save_to_hdf5(hdf5_cal_group, calname)

        if diagnose:
            auto_cal.diagnose()

    def convert_to_energy(self, attr, calname=None):
        if calname is None:
            calname = attr
        if calname not in self.calibration:
            raise ValueError("For chan %d calibration %s does not exist"(self.channum, calname))
        cal = self.calibration[calname]
        self.p_energy[:] = cal.ph2energy(getattr(self, attr))
        self.last_used_calibration = cal

    def read_segment(self, n):
        first, end = self.pulse_records.read_segment(n)
        self.data = self.pulse_records.data
        self.times = self.pulse_records.times
        self.rowcount = self.pulse_records.rowcount
        return first, end

    def plot_traces(self, pulsenums, pulse_summary=True, axis=None, difference=False,
                    residual=False, valid_status=None, shift1=False):
        """Plot some example pulses, given by sample number.
        <pulsenums>   A sequence of sample numbers, or a single one.
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A plt axis to plot on.
        <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
        <residual>   Whether to show the residual between data and opt filtered model, or just raw data.
        <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                     that have been cut.  If "cut", show only those that have been cut.
        <shift1>     Whether to take pulses with p_shift1==True and delay them by 1 sample
        """
        # Don't print pulse summaries if the summary data is not available

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

    def time_drift_correct(self, attr="p_filt_value_phc", forceNew=False):
        if all(self.p_filt_value_tdc[:] == 0.0) or forceNew:
            print("chan %d doing time_drift_correct" % self.channum)
            attr = getattr(self, attr)
            _, info = mass.analysis_algorithms.drift_correct(self.p_timestamp[self.cuts.good()], attr[self.cuts.good()])
            median_timestamp = info['median_pretrig_mean']
            slope = info['slope']

            new_info = {'type': 'time_gain',
                        'slope': slope,
                        'median_timestamp': median_timestamp}

            corrected = attr*(1+slope*(self.p_timestamp[:]-median_timestamp))
            self.p_filt_value_tdc[:] = corrected
        else:
            print("chan %d skipping time_drift_correct" % self.channum)
            corrected, new_info = self.p_filt_value_tdc[:], {}
        return corrected, new_info

    def time_drift_correct_polynomial(self, poly_order=2, attr='p_filt_value_phc', num_lines=None, forceNew=False):
        """assumes the gain is a polynomial in time
        estimates that polynomial by fitting a polynomial to each line in the calibration with the same name as the attribute
         and taking an appropriate average of the polyonomials from each line weighted by the counts in each line
        """
        if not hasattr(self, 'p_filt_value_tdc') or forceNew:
            print("chan %d doing time_drift_correct_polynomail with order %d" % (self.channum, poly_order))
            cal = self.calibration[attr]
            attr = getattr(self, attr)
            attr_good = attr[self.cuts.good()]

            if num_lines is None:
                num_lines = len(cal.elements)

            t0 = np.median(self.p_timestamp)
            counts = [h[0].sum() for h in cal.histograms]
            pfits = []
            counts = [h[0].sum() for h in cal.histograms]
            for i in np.argsort(counts)[-1:-num_lines-1:-1]:
                line_name = cal.elements[i]
                low, high = cal.histograms[i][1][[0, -1]]
                use = np.logical_and(attr_good > low, attr_good < high)
                use_time = self.p_timestamp[self.cuts.good()][use]-t0
                pfit = np.polyfit(use_time, attr_good[use], poly_order)
                pfits.append(pfit)
            pfits = np.array(pfits)

            pfits_slope = np.average(pfits/np.repeat(np.array(pfits[:, -1], ndmin=2).T,
                                                     pfits.shape[-1], 1),
                                     axis=0, weights=np.array(sorted(counts))[-1:-num_lines-1:-1])

            p_corrector = pfits_slope.copy()
            p_corrector[:-1] *= -1
            corrected = attr*np.polyval(p_corrector, self.p_timestamp-t0)
            self.p_filt_value_tdc = corrected

            new_info = {'poly_gain': p_corrector, 't0': t0, 'type': 'time_gain_polynomial'}
        else:
            print("chan %d skipping time_drift_correct_polynomial_dataset" % self.channum)
            corrected, new_info = self.p_filt_value_tdc, {}
        return corrected, new_info

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
        if isinstance(bin_s, float) or isinstance(bin_s, int):
            bin_edge = np.arange(self.p_timestamp[g][0], self.p_timestamp[g][-1], bin_s)
        else:
            bin_edge = bin_s
        counts, bin_edge = np.histogram(self.p_timestamp[g], bin_edge)
        bin_centers = bin_edge[:-1]+0.5*(bin_edge[1]-bin_edge[0])
        rate = counts/float(bin_edge[1]-bin_edge[0])

        return bin_centers, rate

    def cut_summary(self):
        boolean_fields = [name.decode() for name, _ in self.tes_group.boolean_cut_desc if name]

        for c1 in boolean_fields:
            for c2 in boolean_fields:
                print("%d pulses cut by both %s and %s" % (
                    self.cuts.bad(c1, c2).sum(), c1.upper(), c2.upper()))
        for cut_name in boolean_fields:
            print("%d pulses cut by %s" % (self.cuts.bad(cut_name).sum(), cut_name.upper()))
        print("%d pulses total" % self.nPulses)

    def smart_cuts(self, threshold=10.0, n_trainings=10000, forceNew=False):
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
            print("channel %g ran smart cuts, %g of %g pulses passed" % (self.channum,
                                                                         self.cuts.good("smart_cuts").sum(),
                                                                         self.nPulses))
        else:
            print("channel %g skipping smart cuts because it was already done" % self.channum)


# Below here, these are functions that we might consider moving to Cython for speed.
# But at any rate, they do not require any MicrocalDataSet attributes, so they are
# pure functions, not methods.

def phasecorr_find_alignment(phase_indicator, pulse_heights, peak, delta_ph,
                             method2017=False, nf=10, kernel_width=2.0):
    """Find the way to align (flatten) `pulse_heights` as a function of `phase_indicator`
    working only within the range [peak-delta_ph, peak+delta_ph].

    If `method2017`, then use a scipy LSQUnivariateSpline with a reasonable (?)
    number of knots. Otherwise, use `nf` bins in `phase_indicator`, shifting each
    such that its `pulse_heights` histogram best aligns with the overall histogram.
    `method2017==False` (the 2015 way) is subject to particular problems when
    there are not a lot of counts in the peak.
    """
    phrange = np.array([-delta_ph,delta_ph])+peak
    use = np.logical_and(np.abs(pulse_heights[:]-peak)<delta_ph,
        np.abs(phase_indicator)<2)
    low_phase, median_phase, high_phase = \
        sp.stats.scoreatpercentile(phase_indicator[use], [3,50,97])

    if method2017:
        x = phase_indicator[use]
        y = pulse_heights[use]
        NBINS = len(x) // 300
        if NBINS<2: NBINS=2
        if NBINS>12: NBINS=12

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
            knots[i] = np.median(x[bins==i])
            f = lambda shift: mass.mathstat.entropy.laplace_cross_entropy(yo, yu+shift, kernel_width)
            brack = 0.002*np.array([-1,1], dtype=float)
            sbest, KLbest, niter, _ = sp.optimize.brent(f, (), brack=brack, full_output=True, tol=3e-4)
            # print ("Best KL-div is %7.4f at s[%d]=%.4f after %2d iterations"%(KLbest, i, sbest, niter))
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
            f = lambda shift: mass.mathstat.entropy.laplace_cross_entropy(yo, yu+shift, kernel_width)
            brack = 0.002*np.array([-1,1], dtype=float)
            sbest, KLbest, niter, _ = sp.optimize.brent(f, (), brack=brack, full_output=True, tol=1e-4)
            iter2 += niter
            yknot2[i] = sbest
        correction = mass.CubicSpline(knots, yknot+yknot2)
        H0 = mass.mathstat.entropy.laplace_entropy(y, kernel_width)
        H1 = mass.mathstat.entropy.laplace_entropy(ycorr, kernel_width)
        H2 = mass.mathstat.entropy.laplace_entropy(y+correction(x), kernel_width)
        print("Laplace entropy before/middle/after: %.4f, %.4f %.4f (%d+%d iterations, %d phase groups)"%(H0, H1, H2, iter1, iter2, NBINS))

        curve = mass.CubicSpline(knots-median_phase, peak-(yknot+yknot2))
        return curve, median_phase

    # Below here is "method2015", in which we perform correlations and fit to quadratics.
    # It is basically unsuitable for small statistics, so it is no longer preferred.
    Pedges = np.linspace(low_phase, high_phase, nf+1)
    Pctrs = 0.5*(Pedges[1:]+Pedges[:-1])
    dP = 2.0/nf
    Pbin = np.digitize(phase_indicator, Pedges)-1

    NBINS = 200
    hists=np.zeros((nf, NBINS), dtype=float)
    for i,P in enumerate(Pctrs):
        use = (Pbin==i)
        c,b = np.histogram(pulse_heights[use], NBINS, phrange)
        hists[i] = c
    bctr = 0.5*(b[1]-b[0])+b[:-1]

    kernel = np.mean(hists, axis=0)[::-1]
    peaks = np.zeros(nf, dtype=float)
    for i in range(nf):
        # Find the PH of this ridge by fitting quadratic to the correlation
        # of histogram #i and the mean histogram, then finding its local max.
        conv = sp.signal.fftconvolve(kernel, hists[i], 'same')
        m = conv.argmax()
        if conv[m] <= 0: continue
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
    """Apply phase correction to `uncorrected` and return the corrected
    vector."""
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
        if b+1 == NC: # For the last bin, extrapolate
            use = (binnum >= 1+b)
        if use.sum() == 0:
            continue
        frac = (uncorrected[use]-ph[b])/(ph[b+1]-ph[b])
        corrected[use] += frac*corr[b+1, use] + (1-frac)*corr[b, use]
    return corrected
