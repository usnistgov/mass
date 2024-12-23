"""
fake_data - Objects to make fake data for use, e.g., in demonstration scripts.

Joe Fowler, NIST

November 7, 2011
"""

import numpy as np

from mass.core.files import VirtualFile
from mass.core.channel import PulseRecords
from mass.core.channel_group import TESGroup

__all__ = ['FakeDataGenerator']


class FakeDataGenerator:
    """An object to create fake data in memory.

    Can generate a single mass.MicrocalDataSet or a 1+channel mass.TESGroup.

    Only basic functionality is here so far.

    Many interesting randomizations could be added in the future, such as
    gain variation, baseline drift.  Pulse pileup should definitely be added.
    """

    def __init__(self, sample_time, n_samples, n_presamples=None, model_peak=None, seed=None):
        self.rng = np.random.default_rng(seed)
        # Some defaults that can be overridden before generating fake data
        self.pretrig_level = 1000
        self.rise_speed_us = 200.  # in us
        self.fall_speed_us = 1200.  # in us
        self.white_noise = 30.0
        self.model = None

        self.sample_time_us = sample_time  # in us
        self.n_samples = n_samples
        if n_presamples is None:
            self.n_presamples = self.n_samples / 4
        else:
            self.n_presamples = n_presamples
        self.compute_model(model_peak=model_peak)

    def compute_model(self, model_peak=None):
        """Compute the noise-free model pulse shape, given the 2 time constants."""
        dt_us = (np.arange(self.n_samples) - self.n_presamples - 0.5) * self.sample_time_us
        self.model = np.exp(-dt_us / self.fall_speed_us) - np.exp(-dt_us / self.rise_speed_us)
        self.model[dt_us <= 0] = 0
        if model_peak is not None:
            self.model = model_peak * self.model / self.model.max()

    def _generate_virtual_file(self, n_pulses, distributions=None,
                               distribution_weights=None, rate=1.0, channum=1):
        """Return a VirtualFile object with random pulses.

        Args:
            n_pulses      number of pulses to put in the "file"
            distributions random distribution of scale factors.  If none, all pulses are of unit height
            distribution_weights  relative contribution from each distribution in <distributions>.
                          if None, then all will be weighted equally
            rate          expected number of pulses per second.
        """

        data = np.zeros((n_pulses, self.n_samples), dtype=np.uint16)
        pulse_times = self.rng.exponential(1.0 / rate, size=n_pulses).cumsum()

        if distributions is None:
            scale = np.ones(n_pulses, dtype=float)
        else:
            weights = np.asarray(distribution_weights, dtype=float)
            weights = n_pulses * weights / weights.sum()
            weights = np.asarray(weights, dtype=int)
            weights[weights.argmax()] += n_pulses - weights.sum()
            scale = []
            for n, distrib in zip(weights, distributions):
                scale.append(distrib.rvs(size=n))
            scale = np.hstack(scale)
            self.rng.shuffle(scale)

        for i in range(n_pulses):
            data[i, :] = self.model * scale[i] + self.pretrig_level + \
                0.5 + self.rng.standard_normal(self.n_samples) * self.white_noise
        vfile = VirtualFile(data, times=pulse_times)
        vfile.filename = f"virtual_file_chan{channum}.vtf"
        vfile.timebase = self.sample_time_us / 1e6
        vfile.nPresamples = self.n_presamples
        return vfile

    def _generate_virtual_noise_file(self, n_pulses, lowpass_kludge=0):
        """Return a VirtualFile object with random noise.

        Args:
            n_pulses: number of pulses to put in the "file"
        """

        print('Making fake noise')
        data = np.zeros((n_pulses, self.n_samples), dtype=np.uint16)
        pulse_times = np.arange(n_pulses, dtype=float) * self.sample_time_us / 1e6

        raw_noise = self.rng.standard_normal((n_pulses, self.n_samples)) * self.white_noise
        for i in range(lowpass_kludge):
            raw_noise = 0.5 * (raw_noise + np.roll(raw_noise, 2**i))

        data[:, :] = 0.5 + raw_noise
        vfile = VirtualFile(data, times=pulse_times)
        vfile.timebase = self.sample_time_us / 1e6
        vfile.nPresamples = self.n_presamples
        return vfile

    def generate_microcal_dataset(self, n_pulses, distributions=None):
        """Return a single mass.MicrocalDataset"""
        vfile = self._generate_virtual_file(n_pulses, distributions=distributions)
        return PulseRecords(vfile), None

    def generate_tesgroup(self, n_pulses, n_noise=1024, distributions=None,
                          distribution_weights=None, nchan=1):
        """Return a mass.TESGroup with multiple channels in it.

        Args:
            n_pulses (int): the number of pulses per channel
            n_noise (int): the number of noise records per channel
            distributions: ??
            distribution_weights: ??
            nchan (int): how many channels to generate (default 1).
        """
        vfiles = [self._generate_virtual_file(n_pulses, distributions=distributions,
                                              distribution_weights=distribution_weights)
                  for _i in range(nchan)]
        nfiles = [self._generate_virtual_noise_file(n_noise)
                  for _i in range(nchan)]
        data = TESGroup(vfiles, nfiles)

        # Have to fake the channel numbers, b/c they aren't encoded in filename
        for i, ds in enumerate(data.datasets):
            ds.channum = i * 2 + 1
        data.channel = {}
        for ds in data.datasets:
            data.channel[ds.channum] = ds
        return data
