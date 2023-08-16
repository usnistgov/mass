'''
Tests for fit_kink_model
'''
import mass
import numpy as np


class Test_fit_kink:
    """Test the mass.mathstat.fitting.fit_kink_model() function."""

    def setup_method(self):
        x = np.arange(10, dtype=float)
        y = np.array(x)
        truek = 4.6
        y[x > truek] = truek
        self.x = x
        self.y = y
        self.truek = truek

    def test_noisless_fit(self):
        """Make sure fit_kink_model gets very close to exact answer without noise."""
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, self.y, kbounds=(3, 6))
        assert X2 < 1e-8
        assert abs(kbest-self.truek) < 1e-5
        assert abs(a-self.truek) < 1e-5
        assert abs(b-1) < 1e-5
        assert abs(c) < 1e-5

    def test_noisless_fit_no_bounds(self):
        """Make sure fit_kink_model gets very close to exact answer without noise and
        using maximal bounds."""
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, self.y, kbounds=None)
        assert X2 < 1e-8
        assert abs(kbest-self.truek) < 1e-5
        assert abs(a-self.truek) < 1e-5
        assert abs(b-1) < 1e-5
        assert abs(c) < 1e-5

    def test_noisy_fit(self):
        """Make sure fit_kink_model gets close enough to exact answer with noise."""
        rng = np.random.default_rng(9090)
        noisy_y = self.y + rng.standard_normal(len(self.x))*.2
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, noisy_y, kbounds=(3, 6))
        assert X2 < 1.0
        assert abs(kbest-self.truek) < 0.3
        assert abs(a-self.truek) < 0.3
        assert abs(b-1) < 0.1
        assert abs(c) < 0.1
