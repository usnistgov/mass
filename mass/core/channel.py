"""
Created on Feb 16, 2011

@author: fowlerj
"""

__all__=['NoiseRecords', 'PulseRecords', 'Cuts', 'MicrocalDataSet']

import numpy as np
import scipy as sp
import pylab as plt
import os.path

import cPickle

# MASS modules
#import mass.mathstat
import mass.mathstat.power_spectrum
from mass.core.files import VirtualFile, LJHFile, LANLFile
from mass.core.utilities import InlineUpdater
from mass.calibration import young

class NoiseRecords(object):
    """
    Encapsulate a set of noise records, which can either be
    assumed continuous or arbitrarily separated in time.
    """
    DEFAULT_MAXSEGMENTSIZE = 32000000

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

        self.nSamples = self.nPresamples = self.nPulses = 0
        self.timebase = 0.0
        self.__open_file(filename, use_records=use_records)
        self.continuous = records_are_continuous
        self.noise_psd = None
        if self.hdf5_group is not None:
            self.autocorrelation = self.hdf5_group.require_dataset(
                                            "autocorrelation", shape=(self.nSamples,),
                                            dtype=np.float64)
            nfreq = 1+self.nSamples/2
            self.noise_psd = self.hdf5_group.require_dataset(
                                            'noise_psd', shape=(nfreq,),
                                            dtype=np.float64)


    def __open_file(self, filename, use_records=None, file_format=None):
        """Detect the filetype and open it."""

        ALLOWED_TYPES=("ljh","root","virtual")
        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("root"):
                file_format = "root"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s"%ALLOWED_TYPES)

        if file_format == "ljh":
            self.datafile = LJHFile(filename, segmentsize=self.maxsegmentsize)
        elif file_format == "root":
            self.datafile = LANLFile(filename)
        elif file_format == "virtual":
            vfile = filename # Aha!  It must not be a string
            self.datafile = vfile
            self.datafile.segmentsize = vfile.nPulses*(6+2*vfile.nSamples)
            filename = 'Virtual file'
        else:
            raise RuntimeError("It is a programming error to get here")
        self.filename = filename
        self.records_per_segment = self.datafile.segmentsize / (6+2*self.datafile.nSamples)

        if use_records is not None:
            if use_records < self.datafile.nPulses:
                self.datafile.nPulses = use_records
                self.datafile.n_segments = use_records / self.records_per_segment

        # Copy up some of the most important attributes
        for attr in ("nSamples", "nPresamples", "nPulses", "timebase", "channum"):
            self.__dict__[attr] = self.datafile.__dict__[attr]
            self.hdf5_group.attrs[attr] = self.datafile.__dict__[attr]

#        for first_pnum, end_pnum, seg_num, data in self.datafile.iter_segments():
#            if seg_num > 0 or first_pnum>0 or end_pnum != self.nPulses:
#                msg = "NoiseRecords objects can't (yet) handle multi-segment noise files.\n"+\
#                    "File size %d exceeds maximum allowed segment size of %d"%(
#                    self.datafile.binary_size, self.maxsegmentsize)
#                raise NotImplementedError(msg)
#            self.data = data

    def clear_cache(self):
        self.datafile.clear_cache()

    def set_fake_data(self):
        """Use when this does not correspond to a real datafile (e.g., CDM data)"""
        self.datafile = mass.VirtualFile(np.zeros((0,0)))


    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = NoiseRecords(self.filename)
        c.__dict__.update( self.__dict__ )
        c.datafile = self.datafile.copy()
        return c


    def compute_power_spectrum(self, window=mass.mathstat.power_spectrum.hann, plot=True,
                               max_excursion=1000):
        self.compute_power_spectrum_reshape(window=window, nsegments=None,
                                            max_excursion=max_excursion)
        if plot: self.plot_power_spectrum()


    def compute_power_spectrum_reshape(self, window=mass.mathstat.power_spectrum.hann,
                                       seg_length=None, max_excursion=1000):
        """Compute the noise power spectrum with noise "records" reparsed into
        separate records of <seg_length> length.  (If None, then self.data.shape[0] which is
        self.data.nPulses, will be used as the number of segments, each having length
        self.data.nSamples.)

        By making <nsegments> large, you improve the noise on the PSD estimates at the price of poor
        frequency resolution.  By making it small, you get good frequency resolution with worse
        uncertainty on each PSD estimate.  No free lunch, know what I mean?
        """

        if not self.continuous and seg_length is not None:
            raise ValueError("This NoiseRecords doesn't have continuous noise; it can't be resegmented.")

        if seg_length is None:
            seg_length = self.nSamples

        spectrum = mass.mathstat.power_spectrum.PowerSpectrum(seg_length/2, dt=self.timebase)
        if window is None:
            window = np.ones(seg_length)
        else:
            window = window(seg_length)

        for _first_pnum, _end_pnum, _seg_num, data in self.datafile.iter_segments():
            if self.continuous and seg_length is not None:
                data = data.ravel()
                n=len(data)
                n = n-n%seg_length
                data=data[:n].reshape((n/seg_length, seg_length))

            for d in data:
                y = d-d.mean()
                if y.max() - y.min() < max_excursion and len(y)==spectrum.m2:
                    spectrum.addDataSegment(y, window=window)

        freq = spectrum.frequencies()
        self.noise_psd.attrs['delta_f'] = freq[1]-freq[0]
        self.noise_psd[:] = spectrum.spectrum()



    def compute_fancy_power_spectrum(self, window=mass.mathstat.power_spectrum.hann,
                                     plot=True, nseg_choices=None):
        assert self.continuous

        n = np.prod(self.data.shape)
        if nseg_choices is None:
            nseg_choices = [16]
            while nseg_choices[-1]<=n/16 and nseg_choices[-1]<20000:
                nseg_choices.append(nseg_choices[-1]*8)
        print nseg_choices

        spectra = [self.compute_power_spectrum_reshape(window=window, nsegments=ns)
                   for ns in nseg_choices]
        if plot:
            plt.clf()
            lowest_freq = np.array([1./(sp.dt*sp.m2) for sp in spectra])

            start_freq=0.0
            for i,sp in enumerate(spectra):
                x,y = sp.frequencies(), sp.spectrum()
                if i==len(spectra)-1:
                    good = x>=start_freq
                else:
                    good = np.logical_and(x>=start_freq, x<4*lowest_freq[i+1])
                    start_freq = 1*lowest_freq[i+1]
                plt.loglog(x[good],y[good],'-')

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
        freq = np.arange(1, 1+len(yvalue))*self.noise_psd.attrs['delta_f']
        axis.plot(freq, yvalue, **kwarg)
        plt.loglog()
        axis.grid()
        axis.set_xlim([10,3e5])
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power Spectral Density (counts$^2$ Hz$^{-1}$)")
        axis.set_title("Noise power spectrum for %s"%self.filename)


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
            if n==pow2: return int(n)
            elif n>0.75*pow2: return int(pow2)
            elif n>0.625*pow2: return int(np.round(0.75*pow2))
            else: return int(np.round(0.625*pow2))


        # When there are 10 million data points and only 10,000 lags wanted,
        # it's hugely inefficient to compute the full autocorrelation, especially
        # in memory.  Instead, compute it on chunks several times the length of the desired
        # correlation, and average.
        CHUNK_MULTIPLE=15
        if n_data >= (1+CHUNK_MULTIPLE)*n_lags:
            # Be sure to pad chunksize samples by AT LEAST n_lags zeros, to prevent
            # unwanted wraparound in the autocorrelation.
            # padded_data is what we do DFT/InvDFT on; ac is the unnormalized output.
            chunksize=CHUNK_MULTIPLE*n_lags
            padsize = n_lags
            padded_data = np.zeros(padded_length(padsize+chunksize), dtype=np.float)

            ac = np.zeros(n_lags, dtype=np.float)

            entries = 0.0

            for first_pnum, end_pnum, _seg_num, data in self.datafile.iter_segments():
#                print "Using pulses %d to %d (seg=%3d)"%(first_pnum, end_pnum, seg_num)
                data_consumed=0
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
                    ft[0] = 0 # this redundantly removes the mean of the data set
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
            padded_data = np.zeros(padded_length(n_lags+n_data), dtype=np.float)
            padded_data[:n_data] = np.array(self.data.ravel())[:n_data] - self.data.mean()
            padded_data[n_data:] = 0.0

            ft = np.fft.rfft(padded_data)
            del padded_data
            ft[0] = 0 # this redundantly removes the mean of the data set
            ft *= ft.conj()
            ft = ft.real
            acsum = np.fft.irfft(ft)
            del ft
            ac = acsum[:n_lags+1] / (n_data-np.arange(n_lags+1.0))
            del acsum

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
                                 "n_lags<=%d when data are not continuous"%self.nSamples)

            class TooMuchData(StopIteration):
                "Use to signal that the computation loop is done"
                pass

            if data_samples is None:
                data_samples = [0, self.nSamples*self.nPulses]
            n_data = data_samples[1] - data_samples[0]

            records_used = samples_used = 0
            ac=np.zeros(self.nSamples, dtype=np.float)
            try:
                for first_pnum, end_pnum, _seg_num, intdata in self.datafile.iter_segments():
                    if end_pnum <= data_samples[0]: continue
                    if first_pnum >= data_samples[1]: break
                    for i in range(first_pnum, end_pnum):
                        if i < data_samples[0]: continue
                        if i >= data_samples[1]: break

                        data = 1.0*(intdata[i-first_pnum,:])
                        if data.max() - data.min() > max_excursion: continue
                        data -= data.mean()

                        ac += np.correlate(data,data,'full')[self.nSamples-1:]
                        samples_used += self.nSamples
                        records_used += 1
                        if n_data is not None and samples_used >= n_data:
                            raise TooMuchData()
            except TooMuchData:
                pass

            ac /= records_used
            ac /= self.nSamples - np.arange(self.nSamples, dtype=np.float)
            if n_lags is not None and n_lags < self.nSamples:
                ac=ac[:n_lags]
            self.autocorrelation[:] = ac

        if self.hdf5_group is not None:
            grp = self.hdf5_group.require_group("reclen%d"%n_lags)
            ds = grp.require_dataset("autocorrelation", shape=(n_lags,), dtype=np.float64)
            ds[:] = self.autocorrelation[:]

        if plot: self.plot_autocorrelation()


    def plot_autocorrelation(self, axis=None, color='blue', label=None):
        if all(self.autocorrelation[:]==0):
            print "Autocorrelation will be computed first"
            self.compute_autocorrelation(plot=False)
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        t = self.timebase * 1e3 * np.arange(len(self.autocorrelation))
        axis.plot(t, self.autocorrelation, label=label, color=color)
        axis.plot([0],[self.autocorrelation[0]],'o', color=color)
        axis.set_xlabel("Lag (ms)")
        axis.set_ylabel("Autocorrelation (counts$^2$)")


class PulseRecords(object):
    """
    Encapsulate a set of data containing multiple triggered pulse traces.
    The pulses should not be noise records.

    This object will not contain derived facts such as pulse summaries, filtered values,
    and so forth. It is meant to be only a file interface (though until July 2014, this
    was not exactly the case).
    """

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

        self.__open_file(filename, file_format=file_format)

        self.cuts = None
        self.bad = None
        self.good = None


    def __open_file(self, filename, file_format=None):
        """Detect the filetype and open it."""

        ALLOWED_TYPES=("ljh","root","virtual")
        if file_format is None:
            if isinstance(filename, VirtualFile):
                file_format = 'virtual'
            elif filename.endswith("root"):
                file_format = "root"
            elif filename.endswith("ljh"):
                file_format = "ljh"
            else:
                file_format = "ljh"
        if file_format not in ALLOWED_TYPES:
            raise ValueError("file_format must be None or one of %s"%ALLOWED_TYPES)

        if file_format == "ljh":
            self.datafile = LJHFile(filename)
        elif file_format == "root":
            self.datafile = LANLFile(filename)
        elif file_format == "virtual":
            vfile = filename # Aha!  It must not be a string
            self.datafile = vfile
        else:
            raise RuntimeError("It is a programming error to get here")

        self.filename = filename

        # Copy up some of the most important attributes
        for attr in ("nSamples","nPresamples","nPulses", "timebase", "channum",
                     "n_segments", "pulses_per_seg", "segmentsize", "timestamp_offset"):
            self.__dict__[attr] = self.datafile.__dict__[attr]


    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples,
                1e6*self.timebase)

    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)


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
            return -1,-1
        first_pnum, end_pnum, data = self.datafile.read_segment(segment_num)
        self.data = data
        try:
            self.times = self.datafile.datatimes_float
        except AttributeError:
            self.times = self.datafile.datatimes/1e3
        return first_pnum, end_pnum

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = PulseRecords(self.filename)
        c.__dict__.update( self.__dict__ )
        c.datafile = self.datafile.copy()
        return c


    def serialize(self, serialfile):
        """Store object in a pickle file"""
        fp = open(serialfile, "wb")
        cPickle.dump(self, fp, protocol=2)
        fp.close()


class Cuts(object):
    "Object to hold a 32-bit cut mask for each triggered record."

    def __init__(self, n, hdf5_group=None):
        "Create an object to hold n masks of 32 bits each"
        self.hdf5_group = hdf5_group
        if hdf5_group is None:
            self._mask = np.zeros( n, dtype=np.int32 )
        else:
            self._mask = hdf5_group.require_dataset('mask', shape=(n,), dtype=np.int32)

    def cut(self, cutnum, mask):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        assert(mask.size == self._mask.size)
        bitval = 1<<cutnum
        self._mask[mask] |= bitval

    def clearCut(self, cutnum):
        if cutnum < 0 or cutnum >= 32:
            raise ValueError("cutnum must be in the range [0,31] inclusive")
        bitmask = ~(1<<cutnum)
        self._mask[:] &= bitmask

    def good(self):
        return np.logical_not(self._mask)

    def bad(self):
        return self._mask[:] != 0

    def isCut(self, cutnum=None):
        if cutnum is None: return self.bad()
        return (self._mask[:] & (1<<cutnum)) != 0

    def isUncut(self, cutnum=None):
        if cutnum is None: return self.good()
        return (self._mask[:] & (1<<cutnum)) == 0

    def nCut(self):
        return (self._mask[:] != 0).sum()

    def nUncut(self):
        return (self._mask[:] == 0).sum()

    def __repr__(self):
        return "Cuts(%d)"%len(self._mask)

    def __str__(self):
        return ("Cuts(%d) with %d cut and %d uncut"%(len(self._mask), self.nCut(), self.nUncut()))

    def copy(self):
        c = Cuts(len(self._mask), hdf5_group=self.hdf5_group)
        c._mask = self._mask.copy()
        return c



class MicrocalDataSet(object):
    """
    Represent a single microcalorimeter's PROCESSED data.
    This channel can be directly from a TDM detector, or it
    can be the demodulated result of a CDM modulation.
    """
    CUT_NAME = ['pretrigger_rms',
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
                'timing']

    # Attributes that all such objects must have.
    expected_attributes=("nSamples","nPresamples","nPulses","timebase", "channum",
                         "timestamp_offset")

    HDF5_CHUNK_SIZE = 256


    def __init__(self, pulserec_dict, auto_pickle = True, hdf5_group=None):
        """
        Pass in a dictionary (presumably that of a PulseRecords object)
        containing the expected attributes that must be copied to this
        MicrocalDataSet.
        """
        self.auto_pickle = auto_pickle
        self.filter = None
        self.lastUsedFilterHash = -1
        self.drift_correct_info = {}
        self.phase_correct_info = {}
        self.noise_autocorr = None
        self.noise_demodulated = None
        self.calibration = {'p_filt_value':mass.calibration.energy_calibration.EnergyCalibration('p_filt_value')}

        for a in self.expected_attributes:
            self.__dict__[a] = pulserec_dict[a]
        self.filename = pulserec_dict.get('filename','virtual data set')
        self.gain = 1.0
        self.pretrigger_ignore_microsec = None # Cut this long before trigger in computing pretrig values
        self.pretrigger_ignore_samples = 0
        self.peak_time_microsec = None   # Look for retriggers only after this time.
        self.index = None   # Index in the larger TESGroup or CDMGroup object
        self.last_used_calibration = None
        self.pumped_band_knowledge = None

        try:
            self.hdf5_group = hdf5_group
            self.hdf5_group.attrs['npulses'] = self.nPulses
            self.hdf5_group.attrs['channum'] = self.channum
        except KeyError:
            self.hdf5_group = None
        self.__setup_vectors(npulses=self.nPulses)
        if self.auto_pickle:
            self.unpickle()



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
        float32_fields = ('pretrig_mean','pretrig_rms', 'pulse_average', 'pulse_rms',
                          'promptness', 'rise_time','postpeak_deriv',
                          'filt_phase','filt_value','filt_value_dc','filt_value_phc',
                          'energy')
        uint16_fields = ('peak_index', 'peak_value', 'min_value')
        for dtype,fieldnames in ((np.float64, float64_fields),
                                 (np.float32, float32_fields),
                                 (np.uint16, uint16_fields)):
            for field in fieldnames:
                self.__dict__['p_%s'%field] = h5grp.require_dataset(field, shape=(npulses,),
                                                                    dtype=dtype)

        # Other vectors needed per-channel
        self.average_pulse= h5grp.require_dataset('average_pulse', shape=(self.nSamples,),
                                                    dtype=np.float32)
        self.noise_autocorr = h5grp.require_dataset('noise_autocorr', shape=(self.nSamples,),
                                                    dtype=np.float64)
        nfreq = 1+self.nSamples/2
        self.noise_psd = h5grp.require_dataset('noise_psd', shape=(nfreq,),
                                                    dtype=np.float64)
        grp = self.hdf5_group.require_group('cuts')
        self.cuts = Cuts(self.nPulses, hdf5_group=grp)

    @property
    def p_peak_time(self):
        # this is a property to reduce memory usage, I hope it works
        return (np.asarray(self.p_peak_index, dtype=np.int)-self.nPresamples)*self.timebase

    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples,
                1e6*self.timebase)

    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)


    def good(self):
        """Return a boolean vector, one per pulse record, saying whether record is good"""
        return self.cuts.good()

    def bad(self):
        """Return a boolean vector, one per pulse record, saying whether record is bad"""
        return self.cuts.bad()

    def resize(self, nPulses):
        if self.nPulses < nPulses:
            raise ValueError("Can only shrink using resize(), but the requested size %d is larger than current %d"%
                             (nPulses, self.nPulses))
        self.nPulses = nPulses
        self.__setup_vectors()

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = MicrocalDataSet(self.__dict__ )
        c.__dict__.update( self.__dict__ )
        for k in self.calibration.keys():
            c.calibration[k] = self.calibration[k].copy()
        c.cuts = self.cuts.copy()
        return c


    def pickle(self, filename=None, verbose=True):
        """Pickle the _contents_ of the MicrocalDataSet object.
        <filename>    The output pickle name.  If not given, then it will be the data file name
                      with the suffix replaced by '.pkl' and in a subdirectory mass under the
                      main file's location."""

        if filename is None:
            basedir = os.path.dirname(self.filename)
            massdir = os.path.join(basedir, "mass")
            if not os.path.isdir(massdir):
                os.mkdir(massdir, 0775)
            filename = os.path.join(massdir, "%s.pkl"%os.path.basename(self.filename))

        fp = open(filename, "wb")
        pickler = cPickle.Pickler(fp, protocol=2)

        # Pickle the self.expected_attributes
        exp_at = {}
        for k in self.expected_attributes:
            exp_at[k] = self.__dict__[k]
        pickler.dump(exp_at)

        # Pickle the cuts mask
        pickler.dump(self.cuts._mask)

        # Pickle all attributes noise_*, p_*, peak_time_microsec, pretrigger_*, timebase, times
        # Approach is to dump the attribute NAME then value.
        attr_starts = ("pretrigger_",)
        attr_names = ("peak_time_microsec", "timebase", "times", "average_pulse",
                      "calibration", "drift_correct_info", "phase_correct_info", "filter",
                      "pumped_band_knowledge")
        for attr in self.__dict__:
            store_this_attr = attr in attr_names
            for ast in attr_starts:
                if attr.startswith(ast):
                    store_this_attr = True
                    break
            if store_this_attr:
                pickler.dump(attr)
                pickler.dump(self.__dict__[attr])
        fp.close()
        if verbose:
            print "Stored %9d bytes %s"%(os.stat(filename).st_size, filename)


    def unpickle(self, filename=None):
        """
        Unpickle a MicrocalDataSet pickled by its .pickle() method.

        Data structure must be:
        1. A dictionary with simple values, whose keys include at least all strings in
           the tuple MicrocalDataSet.expected_attributes.
        2. The dataset cuts._mask
        3. Any string.  If #4 also loads, then this will be the attribute name.
        4. Any pickleable object.  This will become an attribute value
           (prev item gives its name)
        ... Repeat items (3,4) as needed to load all attribute (name,value) pairs.
        """
        if filename is None:
            basedir = os.path.dirname(self.filename)
            massdir = os.path.join(basedir, "mass")
            if not os.path.isdir(massdir):
                os.mkdir(massdir, 0775)
            filename = os.path.join(massdir, "%s.pkl"%os.path.basename(self.filename))
        if not os.path.isfile(filename):
            return

        fp = open(filename, "rb")
        unpickler = cPickle.Unpickler(fp)
        # ignore the expected_attr and the cuts mask
        _expected_attr = unpickler.load()
        _ = unpickler.load()
        #self.cuts._mask = unpickler.load()
        try:
            while True:
                try:
                    k = unpickler.load()
                    v = unpickler.load()
                except TypeError:
                    continue
                self.__dict__[k] = v
        except (EOFError, cPickle.UnpicklingError):
            pass
        fp.close()

    def summarize_data(self, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0, forceNew=False):
        """Summarize the complete data set one chunk at a time.
        """
        # Don't proceed if not necessary and not forced
        not_done =  all(self.p_pretrig_mean[:]==0)
        if not (not_done or forceNew):
            print('\nchan %d did not summarize because results were already preloaded'%self.channum)
            return

        if len(self.p_timestamp) < self.pulse_records.nPulses:
            self.__setup_vectors(npulses=self.pulse_records.nPulses) # make sure vectors are setup correctly
        self.pretrigger_ignore_samples = int(pretrigger_ignore_microsec*1e-6/self.timebase)

        printUpdater = InlineUpdater('channel.summarize_data_tdm chan %d'%self.channum)
        for s in range(self.pulse_records.n_segments):
            first, end = self.read_segment(s) # this reloads self.data to contain new pulses
            self._summarize_data_segment(first, end, peak_time_microsec, pretrigger_ignore_microsec)
            printUpdater.update((s+1)/float(self.pulse_records.n_segments))
        self.pulse_records.datafile.clear_cached_segment()
        if self.auto_pickle:
            self.pickle(verbose=False)


    def _summarize_data_segment(self, first, end, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0):
        """Summarize the complete data file
        summarize_data(self, first, end, peak_time_microsec=220.0, pretrigger_ignore_microsec = 20.0)
        peak_time_microsec is used when calculating max dp/dt after trigger

        """
        self.peak_time_microsec = peak_time_microsec
        self.pretrigger_ignore_microsec = pretrigger_ignore_microsec
        if first >= self.nPulses:
            return
        if end > self.nPulses:
            end = self.nPulses
        if len(self.p_timestamp) <= 0:
            self.__setup_vectors(npulses=self.nPulses)

        maxderiv_holdoff = int(self.peak_time_microsec*1e-6/self.timebase) # don't look for retriggers before this # of samples
        self.pretrigger_ignore_samples = int(self.pretrigger_ignore_microsec*1e-6/self.timebase)

        seg_size = end-first
        self.p_timestamp[first:end] = self.times[:seg_size]
        self.p_pretrig_mean[first:end] = self.data[:seg_size,:self.nPresamples-self.pretrigger_ignore_samples].mean(axis=1)
        self.p_pretrig_rms[first:end] = self.data[:seg_size,:self.nPresamples-self.pretrigger_ignore_samples].std(axis=1)
        self.p_peak_index[first:end] = self.data[:seg_size,:].argmax(axis=1)
        self.p_peak_value[first:end] = self.data[:seg_size,:].max(axis=1)
        self.p_min_value[first:end] = self.data[:seg_size,:].min(axis=1)
        self.p_pulse_average[first:end] = self.data[:seg_size,self.nPresamples:].mean(axis=1)

        # Remove the pretrigger mean from the peak value and the pulse average figures.
        PTM = self.p_pretrig_mean[first:end]
        self.p_pulse_average[first:end] -= PTM
        self.p_peak_value[first:end] -= np.asarray(PTM, dtype=self.p_peak_value.dtype)
        self.p_pulse_rms[first:end] = np.sqrt(
                (self.data[:seg_size,self.nPresamples:]**2.0).mean(axis=1) -
                PTM*(PTM + 2*self.p_pulse_average[first:end]))
        self.p_promptness[first:end] = (
                self.data[:seg_size,self.nPresamples+6:self.nPresamples+12].mean(axis=1)-PTM)/ \
                self.p_peak_value[first:end]

        self.p_rise_time[first:end] = \
            mass.core.analysis_algorithms.estimateRiseTime(self.data[:seg_size],
                                                           timebase=self.timebase,
                                                           nPretrig=self.nPresamples)

        self.p_postpeak_deriv[first:end] = \
            mass.core.analysis_algorithms.compute_max_deriv(self.data[:seg_size], ignore_leading =
                                                            self.nPresamples+maxderiv_holdoff)

    def filter_data(self, filter_name='filt_noconst', transform=None, forceNew=False):
        """Filter the complete data file one chunk at a time.
        """
        if not(forceNew or all(self.p_filt_value[:]==0)):
            print('\nchan %d did not filter because results were already loaded'%self.channum)
            return

        if self.filter is not None:
            filter_values = self.filter.__dict__[filter_name]
        else:
            filter_values = self.hdf5_group['filters/%s'%filter_name].value
        printUpdater = InlineUpdater('channel.filter_data_tdm chan %d'%self.channum)
        for s in range(self.pulse_records.n_segments):
            first, end = self.read_segment(s) # this reloads self.data to contain new pulses
            (self.p_filt_phase[first:end],
             self.p_filt_value[first:end]) = \
                self._filter_data_segment(filter_values, first, end, transform)
            printUpdater.update((end+1)/float(self.nPulses))

        self.pulse_records.datafile.clear_cached_segment()
        if self.auto_pickle:
            self.pickle(verbose=False)


    def _filter_data_segment(self, filter_values, first, end, transform=None):
        if first >= self.nPulses:
            return None,None

        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = np.array((
                ( -6,24, 34,24,-6),
                (-14,-7,  0, 7,14),
                ( 10,-5,-10,-5,10)), dtype=np.float)/70.0

        assert len(filter_values)+4 == self.nSamples

        seg_size = min(end-first, self.data.shape[0])
        conv = np.zeros((5, seg_size), dtype=np.float)
        if transform is not None:
            ptmean = self.p_pretrig_mean[first:end]
            ptmean.shape = (len(ptmean),1)
            data = transform(self.data-ptmean)
        else:
            data = self.data
        for i in range(4):
            conv[i,:] = np.dot(data[:seg_size,i:i-4], filter_values)
        conv[4,:] = np.dot(data[:,4:], filter_values)


        param = np.dot(fit_array, conv)
        peak_x = -0.5*param[1,:]/param[2,:]
        peak_y = param[0,:] - 0.25*param[1,:]**2 / param[2,:]
        return peak_x, peak_y


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
                print "Plotting only uncut data",
            elif "cut" in valid.lower():
                valid = self.cuts.bad()
                print "Plotting only cut data",
            elif 'all' in valid.lower():
                valid = None
                print "Plotting all data, cut or uncut",
            else:
                raise ValueError("If valid is a string, it must contain 'all', 'uncut' or 'cut'.")

        if valid is not None:
            nrecs = valid.sum()
            if downsample is None:
                downsample=nrecs/10000
                if downsample < 1: downsample = 1
            hour = self.p_timestamp[valid][::downsample]/3600.0
        else:
            nrecs = self.nPulses
            if downsample is None:
                downsample = self.nPulses / 10000
                if downsample < 1: downsample = 1
            hour = self.p_timestamp[::downsample]/3600.0
        print " (%d records; %d in scatter plots)"%(
            nrecs,len(hour))

        plottables = (
            (self.p_pulse_average, 'Pulse Avg', 'purple', None),
            (self.p_pretrig_rms, 'Pretrig RMS', 'blue', [0,4000]),
            (self.p_pretrig_mean, 'Pretrig Mean', 'green', None),
            (self.p_peak_value, 'Peak value', '#88cc00',None),
            (self.p_postpeak_deriv, 'Max PostPk deriv', 'gold', [0,700]),
            (self.p_rise_time[:]*1e3, 'Rise time (ms)', 'orange', [0,12]),
            (self.p_peak_time[:]*1e3, 'Peak time (ms)', 'red', [-3,9])
          )

        plt.clf()
        for i,(vect, label, color, limits) in enumerate(plottables):

            # Time series scatter plots (left-hand panels)
            plt.subplot(len(plottables), 2, 1+i*2)
            plt.ylabel(label)
            if valid is not None:
                vect = vect[valid]
            plt.plot(hour, vect[::downsample],',', color=color)

            # Histogram (right-hand panels)
            plt.subplot(len(plottables), 2, 2+i*2)
            if limits is None:
                in_limit = np.ones(len(vect), dtype=np.bool)
            else:
                in_limit = np.logical_and(vect>limits[0], vect<limits[1])
            contents, _bins, _patches = plt.hist(vect[in_limit],200, log=log,
                           histtype='stepfilled', fc=color, alpha=0.5)
            if log:
                plt.ylim(ymin = contents.min())


    def cut_parameter(self, data, allowed, cut_id):
        """Apply a cut on some per-pulse parameter.

        <data>    The per-pulse parameter to cut on.  It can be an attribute of self, or it
                  can be computed from one (or more),
                  but it must be an array of length self.nPulses
        <allowed> The cut to apply (see below).
        <cut_id>  The bit number (range [0,31]) to identify this cut.  Should be one of
                  self.CUT_NAME

        <allowed> is a 2-element sequence (a,b), then the cut requires a < data < b.
        Either a or b may be None, indicating no cut.
        OR
        <allowed> is a sequence of 2-element sequences (a,b), then the cut cuts data that does not meet a <= data <=b
        for any of the two element sequences, if any element in allowed is ''invert'' then it swaps cut and uncut
        """

        if allowed is None: # no cut here!
            return
        if cut_id <0 or cut_id >=32:
            raise ValueError("cut_id must be in the range [0,31]")

        # determine if allowed is a sequence or a sequence of sequences
        if np.size(allowed[0]) == 2 or allowed[0] == 'invert':
            doInvert = False
            cut_vec = np.ones_like(data, dtype='bool')
            for element in allowed:
                if np.size(element) == 2:
                    try:
                        a,b = element
                        if a is not None and b is not None:
                            index = np.logical_and(data[:] >= a, data[:] <= b)
                        elif a is not None:
                            index = data[:] >= a
                        elif b is not None:
                            index = data[:] <= b
                        cut_vec[index] = False
                    except:
                        raise ValueError('%s was passed as a cut element, only two element lists or tuples are valid'%str(element))
                elif element == 'invert':
                    doInvert = True
            if doInvert:
                self.cuts.cut(cut_id, ~cut_vec)
            else:
                self.cuts.cut(cut_id, cut_vec)
        else:
            try:
                a,b = allowed
                if a is not None:
                    self.cuts.cut(cut_id, data[:] <= a)
                if b is not None:
                    self.cuts.cut(cut_id, data[:] >= b)
            except ValueError:
                raise ValueError('%s was passed as a cut element, but only two-element sequences are valid.'%str(allowed))

    def compute_noise_spectra(self, max_excursion=1000, n_lags=None, forceNew=False):
        """<n_lags>, if not None, is the number of lags in each noise spectrum and the max lag
        for the autocorrelation.  If None, the record length is used."""
        if n_lags is None:
            n_lags = self.nSamples
        if forceNew or all(self.noise_autocorr[:]==0):
            self.noise_records.compute_power_spectrum_reshape(max_excursion=max_excursion, seg_length=n_lags)
            self.noise_records.compute_autocorrelation(n_lags=n_lags, plot=False, max_excursion=max_excursion)
            self.noise_records.clear_cache()

            self.noise_autocorr[:] = self.noise_records.autocorrelation[:]
            self.noise_psd[:] = self.noise_records.noise_psd[:]
            self.noise_psd.attrs['delta_f'] = self.noise_records.noise_psd.attrs['delta_f']
        else:
            print("chan %d skipping compute_noise_spectra because already done"%self.channum)

    def apply_cuts(self, controls=None, clear=False, verbose=1, forceNew=True):
        """
        <clear>  Whether to clear previous cuts first (by default, do not clear).
        <verbose> How much to print to screen.  Level 1 (default) counts all pulses good/bad/total.
                    Level 2 adds some stuff about the departure-from-median pretrigger mean cut.
        """
        if forceNew == False:
            if self.cuts.good().sum() != self.nPulses:
                print("Chan %d skipped cuts: after %d are good, %d are bad of %d total pulses"%
                      (self.channum, self.cuts.nUncut(),self.cuts.nCut(), self.nPulses))

        if clear: self.clear_cuts()

        if controls is None:
            controls = mass.controller.standardControl()
        c = controls.cuts_prm

        self.cut_parameter(self.p_energy, c['energy'], self.CUT_NAME.index('energy'))
        self.cut_parameter(self.p_pretrig_rms, c['pretrigger_rms'],
                           self.CUT_NAME.index('pretrigger_rms'))
        self.cut_parameter(self.p_pretrig_mean, c['pretrigger_mean'],
                           self.CUT_NAME.index('pretrigger_mean'))

        self.cut_parameter(self.p_peak_time[:]*1e3, c['peak_time_ms'],
                           self.CUT_NAME.index('peak_time_ms'))
        self.cut_parameter(self.p_rise_time[:]*1e3, c['rise_time_ms'],
                           self.CUT_NAME.index('rise_time_ms'))
        self.cut_parameter(self.p_postpeak_deriv, c['postpeak_deriv'],
                           self.CUT_NAME.index('postpeak_deriv'))
        self.cut_parameter(self.p_pulse_average, c['pulse_average'],
                           self.CUT_NAME.index('pulse_average'))
        self.cut_parameter(self.p_peak_value, c['peak_value'],
                           self.CUT_NAME.index('peak_value'))
        self.cut_parameter(self.p_min_value[:]-self.p_pretrig_mean[:], c['min_value'],
                           self.CUT_NAME.index('min_value'))
        self.cut_parameter(self.p_timestamp, c['timestamp_sec'],
                           self.CUT_NAME.index('timestamp_sec'))
        if c['timestamp_diff_sec'] is not None:
            self.cut_parameter(np.hstack((0.0, np.diff(self.p_timestamp))),
                               c['timestamp_diff_sec'],
                               self.CUT_NAME.index('timestamp_diff_sec'))
        if c['pretrigger_mean_departure_from_median'] is not None:
            median = np.median(self.p_pretrig_mean[self.cuts.good()])
            if verbose>1:
                print'applying cut on pretrigger mean around its median value of ',median
            self.cut_parameter(self.p_pretrig_mean-median,
                               c['pretrigger_mean_departure_from_median'],
                               self.CUT_NAME.index('pretrigger_mean_departure_from_median'))
        if verbose>0:
            print "Chan %d after cuts, %d are good, %d are bad of %d total pulses"%(
                self.channum, self.cuts.nUncut(),
                self.cuts.nCut(), self.nPulses)


    def clear_cuts(self):
        self.cuts = Cuts(self.nPulses)


    def drift_correct(self, forceNew=False):
        """Drift correct using the standard entropy-minimizing algorithm"""
        doesnt_exist = all(self.p_filt_value_dc[:]==0) or  \
                    all(self.p_filt_value_dc[:]==self.p_filt_value[:])
        if not (forceNew or doesnt_exist):
            print("chan %d not drift correction, p_filt_value_dc already populated"%self.channum)
            return
        g = self.cuts.good()
        uncorrected = self.p_filt_value[g]
        indicator = self.p_pretrig_mean[g]
        drift_corr_param, self.drift_correct_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        print 'chan %d best drift correction parameter: %.6f'%(self.channum, drift_corr_param)

        # Apply correction
        ptm_offset = self.drift_correct_info['median_pretrig_mean']
        gain = 1+(self.p_pretrig_mean-ptm_offset)*drift_corr_param
        self.p_filt_value_dc = self.p_filt_value*gain
        if self.auto_pickle:
            self.pickle(verbose=False)


    def phase_correct2014(self, typical_resolution, maximum_num_records = 50000, plot=False, forceNew=False):
        """Apply the phase correction that seems good for calibronium-like
        data as of June 2014. For more notes, do
        help(mass.core.analysis_algorithms.FilterTimeCorrection)

        <typical_resolution> should be an estimated energy resolution in UNITS OF
        self.p_pulse_rms. This helps the peak-finding (clustering) algorithm decide
        which pulses go together into a single peak.  Be careful to use a semi-reasonable
        quantity here.
        """
        doesnt_exist = all(self.p_filt_value_phc[:]==0) or  \
                    all(self.p_filt_value_phc[:]==self.p_filt_value_dc[:])
        if forceNew or doesnt_exist:
            data,g = self.first_n_good_pulses(maximum_num_records)
            print("channel %d doing phase_correct2014 with %d good pulses"%(self.channum, data.shape[0]))
            prompt = self.p_promptness

            dataFilter = self.filter.filt_noconst
            tc = mass.core.analysis_algorithms.FilterTimeCorrection(
                    data, prompt[:][g], self.p_pulse_rms[:][g], dataFilter,
                    self.nPresamples, typicalResolution=typical_resolution)

            self.p_filt_value_phc = self.p_filt_value_dc - tc(prompt, self.p_pulse_rms)
        else:
            print("channel %d skipping phase_correct2014"%self.channum)

        if plot:
            plt.clf()
            g = self.cuts.good()
            plt.plot(prompt[:][g], self.p_filt_value_dc[:][g], 'g.')
            plt.plot(prompt[:][g], self.p_filt_value_phc[:][g], 'b.')


    def first_n_good_pulses(self, n=50000):
        """
        :param n: maximum number of good pulses to include
        :return: data, g
        data is a (X,Y) array where X is number of records, and Y is number of samples per record
        g is a 1d array of of pulse record numbers of the pulses in data
        if we  did load all of ds.data at once, this would be roughly equivalent to
        return ds.data[ds.cuts.good()][:n], np.nonzero(ds.cuts.good())[0][:n]
        """
        first, end = self.read_segment(0)
        g = self.cuts.good()
        data = self.data[g[first:end]]
        for j in xrange(1,self.pulse_records.n_segments):
            first, end = self.read_segment(j)
            data = np.vstack((data, self.data[g[first:end],:]))
            if data.shape[0]>n:
                break
        nrecords = np.amin([n, data.shape[0]])
        return data[:nrecords], np.nonzero(g)[0][:nrecords]


    def fit_spectral_line(self, prange, mask=None, times=None, fit_type='dc', line='MnKAlpha',
                          nbins=200, verbose=True, plot=True, **kwargs):
        """
        <line> can be one of the fitters in mass.calibration.fluorescence_lines (e.g. 'MnKAlpha', 'CuKBeta') or
        in mass.calibration.gaussian_lines (e.g. 'Gd97'), or a number.  In this last case, it is assumed to
        be a single Gaussian line.
        """
        all_values={'filt': self.p_filt_value,
                    'phc': self.p_filt_value_phc,
                    'dc': self.p_filt_value_dc,
                    'energy': self.p_energy,
                    }[fit_type]
        if mask is not None:
            valid = np.array(mask)
        else:
            valid = self.cuts.good()
        if times is not None:
            valid = np.logical_and(valid, self.p_timestamp<times[1])
            valid = np.logical_and(valid, self.p_timestamp>times[0])
        good_values = all_values[valid]
        contents,bin_edges = np.histogram(good_values, nbins, prange)
        if verbose: print "%d events pass cuts; %d are in histogram range"%(len(good_values),contents.sum())
        bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])

        # Try line first as a number, then as a fluorescence line, then as a Gaussian
        try:
            energy = float(line)
            module = 'mass.calibration.gaussian_lines'
            fittername = '%s.GaussianFitter(%s.GaussianLine())'%(module,module)
            fitter = eval(fittername)
        except ValueError:
            energy = None
            try:
                module = 'mass.calibration.fluorescence_lines'
                fittername = '%s.%sFitter()'%(module,line)
                fitter = eval(fittername)
            except AttributeError:
                try:
                    module = 'mass.calibration.gaussian_lines'
                    fittername = '%s.%sFitter()'%(module,line)
                    fitter = eval(fittername)
                except AttributeError:
                    raise ValueError("Cannot understand line=%s as an energy or a known calibration line."%line)

        params, covar = fitter.fit(contents, bin_ctrs, plot=plot, **kwargs)
        if plot:
            mass.plot_as_stepped_hist(plt.gca(), contents, bin_ctrs)
        if energy is not None:
            scale = energy/params[1]
        else:
            scale=1.0
        if verbose: print 'Resolution: %5.2f +- %5.2f eV'%(params[0]*scale,np.sqrt(covar[0,0])*scale)
        return params, covar, fitter


    def calibrate(self, attr, line_names,name_ext="",size_related_to_energy_resolution=10, min_counts_per_cluster=20,
                  fit_range_ev=200, excl=(), plot_on_fail=False,max_num_clusters=np.inf,max_pulses_for_dbscan=1e5, forceNew=False):
        calname = attr+name_ext
        if self.calibration.has_key(calname):
            cal = self.calibration[calname]
            if young.is_calibrated(cal) and not forceNew:
                print("Not calibrating chan %d %s because it already exists"%(self.channum, calname))
                return None
            # first does this already exist? if the calibration already exists and has more than 1 pt,
            # we probably dont need to redo it
        print("Calibrating chan %d to create %s"%(self.channum, calname))
        cal = young.EnergyCalibration(size_related_to_energy_resolution, min_counts_per_cluster, fit_range_ev, excl,
                 plot_on_fail,max_num_clusters, max_pulses_for_dbscan)
        cal.fit(getattr(self, attr)[self.cuts.good()], line_names)
        self.calibration[calname]=cal
        if self.auto_pickle:
            self.pickle(verbose=False)


    def convert_to_energy(self, attr, calname=None):
        if calname is None: calname = attr
        if not self.calibration.has_key(calname):
            raise ValueError("For chan %d calibration %s does not exist"(self.channum, calname))
        cal = self.calibration[calname]
        self.p_energy = cal.ph2energy(getattr(self, attr))
        self.last_used_calibration = cal


    def read_segment(self, n):
        first, end = self.pulse_records.read_segment(n)
        self.data = self.pulse_records.data
        self.times = self.pulse_records.times
        return first, end


    def plot_traces(self, pulsenums, pulse_summary=True, axis=None, difference=False,
                    residual=False, valid_status=None):
        """Plot some example pulses, given by sample number.
        <pulsenums>   A sequence of sample numbers, or a single one.
        <pulse_summary> Whether to put text about the first few pulses on the plot
        <axis>       A plt axis to plot on.
        <difference> Whether to show successive differences (that is, d(pulse)/dt) or the raw data
        <residual>   Whether to show the residual between data and opt filtered model, or just raw data.
        <valid_status> If None, plot all pulses in <pulsenums>.  If "valid" omit any from that set
                     that have been cut.  If "cut", show only those that have been cut.
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
        color= 'purple','blue','green','#88cc00','gold','orange','red', 'brown','gray','#444444','magenta'
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

        cuts_good = self.cuts.good()[pulsenums]
        pulses_plotted = -1
        for i,pn in enumerate(pulsenums):
            if valid_status == 'cut' and cuts_good[i]: continue
            if valid_status == 'valid' and not cuts_good[i]: continue
            pulses_plotted += 1

            data = self.read_trace(pn)
            if difference:
                data = data*1.0-np.roll(data,1)
                data[0] = 0
                data += np.roll(data,1) + np.roll(data,-1)
                data[0] = 0
            elif residual:
                model = self.p_filt_value[pn] * self.average_pulse[:] / np.max(self.average_pulse)
                data = data-model

            cutchar,alpha,linestyle,linewidth = ' ',1.0,'-',1

            # When plotting both cut and valid, mark the cut data with x and dashed lines
            if valid_status is None and not cuts_good[i]:
                cutchar,alpha,linestyle,linewidth = 'X',1.0,'--' ,1
            axis.plot(dt, data, color=color[pulses_plotted%len(color)], linestyle=linestyle, alpha=alpha,
                       linewidth=linewidth)
            if pulse_summary and pulses_plotted<MAX_TO_SUMMARIZE and len(self.p_pretrig_mean)>=pn:
                try:
                    summary = "%s%6d: %5.0f %7.2f %6.1f %5.0f %5.0f %7.1f"%(
                                cutchar, pn, self.p_pretrig_mean[pn], self.p_pretrig_rms[pn],
                                self.p_postpeak_deriv[pn], self.p_rise_time[pn]*1e6,
                                self.p_peak_value[pn], self.p_pulse_average[pn])
                except IndexError:
                    pulse_summary = False
                    continue
                axis.text(.975, .93-.02*pulses_plotted, summary, color=color[pulses_plotted%len(color)],
                           family='monospace', size='medium', transform = axis.transAxes, ha='right')


    def read_trace(self, record_num):
        """Read (from cache or disk) and return the pulse numbered <record_num> for
        dataset number <dataset_num> or channel number <chan_num>.
        If both are given, then <chan_num> will be used when valid.
        If this is a CDMGroup, then the pulse is the demodulated
        channel by that number."""
        seg_num = record_num / self.pulse_records.pulses_per_seg
        self.read_segment(seg_num)
        return self.data[record_num % self.pulse_records.pulses_per_seg,:]


    def time_drift_correct(self, attr="p_filt_value_phc", forceNew=False):
        if not hasattr(self, 'p_filt_value_tdc') or forceNew:
            print("chan %d doing time_drift_correct"%self.channum)
            attr = getattr(self, attr)
            _, info = mass.analysis_algorithms.drift_correct(self.p_timestamp[self.cuts.good()],attr[self.cuts.good()])
            median_timestamp = info['median_pretrig_mean']
            slope = info['slope']

            new_info = {}
            new_info['type']='time_gain'
            new_info['slope']=slope
            new_info['median_timestamp']=median_timestamp

            corrected = attr*(1+slope*(self.p_timestamp-median_timestamp))
            self.p_filt_value_tdc = corrected
        else:
            print("chan %d skipping time_drift_correct"%self.channum)
            corrected, new_info = self.p_filt_value_tdc, {}
        return corrected, new_info


    def time_drift_correct_polynomial(self, poly_order=2,attr='p_filt_value_phc', num_lines = None, forceNew=False):
        """assumes the gain is a polynomial in time
        estimates that polynomial by fitting a polynomial to each line in the calibration with the same name as the attribute
         and taking an appropriate average of the polyonomials from each line weighted by the counts in each line
        """
        if not hasattr(self, 'p_filt_value_tdc') or forceNew:
            print("chan %d doing time_drift_correct_polynomail with order %d"%(self.channum, poly_order))
            cal = self.calibration[attr]
            attr = getattr(self, attr)
            attr_good = attr[self.cuts.good()]

            if num_lines is None: num_lines = len(cal.elements)

            t0 = np.median(self.p_timestamp)
            counts = [h[0].sum() for h in cal.histograms]
            pfits = []
            counts = [h[0].sum() for h in cal.histograms]
            for i in np.argsort(counts)[-1:-num_lines-1:-1]:
                line_name = cal.elements[i]
                low,high =cal.histograms[i][1][[0,-1]]
                use = np.logical_and(attr_good>low, attr_good<high)
                use_time = self.p_timestamp[self.cuts.good()][use]-t0
                pfit = np.polyfit(use_time, attr_good[use],poly_order)
                pfits.append(pfit)
            pfits = np.array(pfits)

            pfits_slope = np.average(pfits/np.repeat(np.array(pfits[:,-1],ndmin=2).T,pfits.shape[-1],1),axis=0, weights=np.array(sorted(counts))[-1:-num_lines-1:-1])

            p_corrector = pfits_slope.copy()
            p_corrector[:-1] *=-1
            corrected = attr*np.polyval(p_corrector, self.p_timestamp-t0)
            self.p_filt_value_tdc = corrected

            new_info = {'poly_gain':p_corrector, 't0':t0, 'type':'time_gain_polynomial'}
        else:
            print("chan %d skipping time_drift_correct_polynomial_dataset"%self.channum)
            corrected, new_info = self.p_filt_value_tdc, {}
        return corrected, new_info


    def compare_calibrations(self):
        plt.figure()
        for key in self.calibration:
            cal = self.calibration[key]
            try:
                plt.plot(cal.peak_energies, cal.energy_resolutions,'o', label=key)
            except:
                pass
        plt.legend()
        plt.xlabel("energy (eV)")
        plt.ylabel("energy resolution fwhm (eV)")
        plt.grid("on")
        plt.title("chan %d cal comparison"%self.channum)


    def count_rate(self, goodonly=False,bin_s=60):
        g = self.cuts.good()
        if not goodonly: g[:]=True
        if isinstance(bin_s,float) or isinstance(bin_s, int):
            bin_edge = np.arange(self.p_timestamp[g][0], self.p_timestamp[g][-1], bin_s)
        else:
            bin_edge = bin_s
        counts, bin_edge = np.histogram(self.p_timestamp[g], bin_edge)
        bin_centers = bin_edge[:-1]+0.5*(bin_edge[1]-bin_edge[0])
        rate = counts/float(bin_edge[1]-bin_edge[0])

        return bin_centers, rate

    def cut_summary(self):
        for i,c1 in enumerate(self.CUT_NAME):
            for j,c2 in enumerate(self.CUT_NAME):
                print("%d pulses cut by both %s and %s"%(
                np.sum( np.logical_and(self.cuts.isCut(i), self.cuts.isCut(j))),c1.upper(), c2.upper()))
        for j,cutname in enumerate(self.CUT_NAME):
            print("%d pulses cut by %s"%(np.sum(self.cuts.isCut(j)), cutname.upper()))
        print("%d pulses total"%self.nPulses)



################################################################################################


def unpickle_MicrocalDataSet(filename):
    """
    Factory function to unpickle a MicrocalDataSet pickled by its .pickle() method.
    Note that you might be better off creating a MicrocalDataSet the usual way, then
    loading it with its .unpickle() method.

    Data structure must be:
    1. A dictionary with simple values, whose keys include at least all strings in
       the tuple MicrocalDataSet.expected_attributes.
    2. The dataset cuts._mask
    3. Any string.  If #4 also loads, then this will be the attribute name.
    4. Any pickleable object.  This will become an attribute value (prev item gives its name)
    ... Repeat items (3,4) as many times as necessary to load attribute (name,value) pairs.
    """
    fp = open(filename, "rb")
    unpickler = cPickle.Unpickler(fp)
    expected_attr = unpickler.load()
    ds = MicrocalDataSet(expected_attr)
    ds.cuts._mask = unpickler.load()
    try:
        while True:
            k = unpickler.load()
            v = unpickler.load()
            ds.__dict__[k] = v
    except EOFError:
        pass
    fp.close()
    return ds
