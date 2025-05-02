import numpy as np
from mass.core.analysis_algorithms import unwrap_n

import logging
LOG = logging.getLogger("mass")

rng = np.random.default_rng(7923532)  # make tests not fail randomly


class Test_unwrap_n:

    def setup_method(self):
        self.noise_size = 10.0
        self.dlength = 1000
        self.data = rng.uniform(size=self.dlength) * self.noise_size

    def test_no_unwrap(self):
        unwrapped = unwrap_n(self.data, self.noise_size / 10, n=0)
        assert np.array_equal(unwrapped, self.data)

    def test_period_limits(self):
        max_diff = np.max(np.abs(np.diff(self.data)))
        # Pick a period to ensure that at least one point should move
        period = max_diff * 1.99
        unwrapped = unwrap_n(self.data, period, n=1)
        assert not np.array_equal(unwrapped, self.data)

        # Should do nothing
        period = max_diff * 2.01
        unwrapped = unwrap_n(self.data, period, n=1)
        assert np.array_equal(unwrapped, self.data)

    def test_same_as_numpy(self):
        unwrapped = unwrap_n(self.data, self.noise_size / 10, n=1)
        np_unwrapped = np.unwrap(self.data, period=self.noise_size / 10)
        assert np.allclose(unwrapped, np_unwrapped, rtol=1e-9)

    def test_lengths(self):
        # Check that an array of size 1 is unaffected
        data = rng.uniform(size=1) * self.noise_size
        unwrapped = unwrap_n(data, self.noise_size / 10, n=5)
        assert np.array_equal(data, unwrapped)

        # Check that an array shorter than the averaging length will not
        # break anything. Also, check that a length-3 array is affected
        # only when the chosen period is small enough.
        data = rng.uniform(size=3) * self.noise_size
        diff1 = abs(data[1] - data[0])
        diff2 = abs(data[2] - (data[0] + data[1]) / 2)
        max_diff = max(diff1, diff2)

        period = max_diff * 1.99
        unwrapped = unwrap_n(data, period, n=5)
        assert len(unwrapped) == len(data)
        assert not np.allclose(unwrapped, data, rtol=1e-9)

        period = max_diff * 2.01
        unwrapped = unwrap_n(data, period, n=5)
        assert np.array_equal(unwrapped, data)
