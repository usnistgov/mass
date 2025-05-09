import numpy as np
from mass.core.analysis_algorithms import unwrap_n, correct_flux_jumps
from mass.core.analysis_algorithms import correct_flux_jumps_original

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


class TestOriginalAlgorithm:
    @staticmethod
    def make_trend_linear(sz):
        b = rng.integers(0, 2**16 - 1)
        m = 4 * rng.uniform() - 2
        trend = b + m * (sz / 2.**12) * np.arange(sz)
        return trend

    @staticmethod
    def make_trend_poly(sz, deg):
        max_phi0 = 2
        p = np.zeros(deg + 1)
        p[:-1] = (2 * max_phi0 * rng.uniform(deg) - max_phi0) * 2.**12 * (1. / sz)**(np.arange(deg, 0, -1))
        p[-1] = 2**14 + rng.integers(0, 2 * 2**14)
        trend = np.polyval(p, np.arange(sz))
        return trend

    @staticmethod
    def make_trend_poly_plus_sine(sz, deg):
        max_phi0 = 2
        p = np.zeros(deg + 1)
        p[:-1] = (0.1 * max_phi0 * rng.uniform() * deg - 0.05 * max_phi0) * 2.**12 * (1. / sz)**(np.arange(deg, 0, -1))
        p[-1] = 2**14 + rng.integers(0, 2 * 2**14)
        trend = np.polyval(p, np.arange(sz))

        phase = 2 * np.pi * rng.uniform()
        amp = 0.1 * 2**12 * rng.uniform()
        freq = 20 * rng.uniform()

        trend += amp * np.cos(2 * np.pi * (1.0 * np.arange(sz) / sz) * freq + phase)

        return trend

    @staticmethod
    def add_jumps(vals):
        njumps = 30
        for k in range(njumps):
            start = rng.integers(1, len(vals))
            vals[start:] += 2**12 * rng.integers(-4, 5)
        return vals

    @staticmethod
    def verify(orig_vals, corrected):
        assert np.max(np.abs(orig_vals - corrected)) < 1e-6

    def test_algorithm(self, N=100):
        sz = 2048
        g = np.full(sz, True, dtype=bool)
        g[1000] = g[1010] = g[1100:1110] = False
        
        for k in range(N):
            noise = np.abs(100 * rng.standard_normal(sz))
            vals_orig = self.make_trend_poly_plus_sine(sz, 2) + noise
            corrupted_vals = self.add_jumps(vals_orig.copy())
            corrupted_vals[1100:1110] += 3000
            # new_vals = correct_flux_jumps_original(corrupted_vals, g, 2**12)
            # self.verify(vals_orig, new_vals)

            new_vals = correct_flux_jumps(corrupted_vals, g, 2**12)
            self.verify(vals_orig, new_vals)
