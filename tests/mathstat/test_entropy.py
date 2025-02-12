"""
test_entropy.py

Test that Laplace KDE entropy code works

20 March 2017
Joe Fowler
"""

import pytest
import numpy as np
import mass
from mass.mathstat.entropy import laplace_entropy, _merge_orderedlists, \
    laplace_cross_entropy, laplace_KL_divergence  # noqa


class Test_LaplaceEntropy:
    """Test the entropy of the Laplace kernel-density estimator."""

    @staticmethod
    def test_entropy1():
        """Entropy on a distribution with 1 value."""
        for w in [.1, .2, .5, 1, 2, 5]:
            expected = 1 + np.log(2 * w)
            for i in [1.1, 0.0, -39]:
                d = np.array([i], dtype=float)
                assert laplace_entropy(d, w) == pytest.approx(expected, rel=1e-4)

    @staticmethod
    def test_specific_values_entropy():
        """Entropy on some specific values tested in Julia already."""
        e = laplace_entropy([1, 2, 3.0], .2)
        assert e == pytest.approx(1.0188440534191967)
        e = laplace_entropy([1, 2, 3.0], 1.0)
        assert e == pytest.approx(1.8865648057292637)
        e = laplace_entropy([1, 2, 3.0], 5.0)
        assert e == pytest.approx(3.3145184966015533)
        e = laplace_entropy([1, 2, 3, .1, .5, .3, .9, -5, 3], 0.2)
        assert e == pytest.approx(1.4343995252696242)
        e = laplace_entropy([1, 2, 3, .1, .5, .3, .9, -5, 3], 1.0)
        assert e == pytest.approx(2.2473133161879617)
        e = laplace_entropy([1, 2, 3, .1, .5, .3, .9, -5, 3], 5.0)
        assert e == pytest.approx(3.3724487611280987)

    @staticmethod
    def test_merge():
        """Test that _merge_orderedlists(x,y) works as intended."""
        x = [1, 2, 4, 6, 7, 8]
        y = [3, 5]
        m, is1 = _merge_orderedlists(x, y)
        expect_m = np.sort(np.hstack([x, y]))
        expect_i = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)

        for a, b in zip(m, expect_m):
            assert a == b
        for a, b in zip(is1, expect_i):
            assert a == b

    @staticmethod
    def test_specific_values_cross_h():
        """Cross-entropy on some specific values."""
        e = laplace_cross_entropy([1.0], [0.0], 1)
        assert e == pytest.approx(2.34308882742)
        e = laplace_cross_entropy([0.0], [1.0], 1)
        assert e == pytest.approx(2.34308882742)
        e = laplace_cross_entropy([0.0], [5.0], 5)
        assert e == pytest.approx(2.34308882742 + np.log(5))
        e = laplace_cross_entropy(np.linspace(1, 3, 30), [1, 2, 3.], .2)
        assert e == pytest.approx(1.39061512214)
        e = laplace_cross_entropy(np.linspace(1, 3, 30), [1, 2, 3.], 1)
        assert e == pytest.approx(2.19810978807)
        e = laplace_cross_entropy(np.linspace(1, 3, 30), [1, 2, 3.], 5)
        assert e == pytest.approx(3.77065709074)
        e = laplace_cross_entropy(np.linspace(1, 3, 30), np.linspace(1, 3, 30), .2)
        assert e == pytest.approx(1.07166443142)
        e = laplace_cross_entropy(np.linspace(1, 3, 30), np.linspace(1, 3, 30), 1)
        assert e == pytest.approx(1.99188741398)
        e = laplace_cross_entropy(np.linspace(1, 3, 30), np.linspace(1, 3, 30), 5)
        assert e == pytest.approx(3.60725329485)

    @staticmethod
    def test_exact_approx_entropy():
        """Test the exact vs approximated modes of laplace_entropy."""
        x = np.linspace(-1, 1, 1001)
        z = np.hstack([x - .001, x - .0005, x, x + .0002, x + .0008])
        # Because these are size 5005 vectors, they should default to "exact" mode.
        e = laplace_entropy(z, 1, "exact")
        assert e == pytest.approx(1.8064846705587594)
        e = laplace_entropy(z, 1)
        assert e == pytest.approx(1.8064846705587594)
        e = laplace_entropy(z, 1, "approx")
        assert e == pytest.approx(1.7862710795706667)

        e = laplace_entropy(z, .1, "exact")
        assert e == pytest.approx(0.8215581872670996)
        e = laplace_entropy(z, .1)
        assert e == pytest.approx(0.8215581872670996)
        e = laplace_entropy(z, .1, "approx")
        assert e == pytest.approx(0.8189686027988935)

    @staticmethod
    def test_exact_approx_cross_entropy():
        """Test the exact vs approximated modes of laplace_cross_entropy."""
        x = np.linspace(-1, 1, 1001)
        z = np.hstack([x - .001, x - .0005, x - .0001, x + .0002, x + .0008])
        # Because these are size 5005 vectors, they should default to "exact" mode.
        e = laplace_cross_entropy(z, x, 1, "exact")
        assert e == pytest.approx(1.94136653687)
        e = laplace_cross_entropy(z, x, 1)
        assert e == pytest.approx(1.94136653687)
        e = laplace_cross_entropy(x, z, 1, "exact")
        assert e == pytest.approx(1.95882859388)
        e = laplace_cross_entropy(x, z, 1)
        assert e == pytest.approx(1.95882859388)
        e = laplace_cross_entropy(z, x, 1, "approx")
        assert e == pytest.approx(1.786246409970764)
        e = laplace_cross_entropy(x, z, 1, "approx")
        assert e == pytest.approx(1.7861438931128626)
        e = laplace_cross_entropy(z, x, .1, "exact")
        assert e == pytest.approx(0.8366756130721216)
        e = laplace_cross_entropy(z, x, .1)
        assert e == pytest.approx(0.8366756130721216)
        e = laplace_cross_entropy(x, z, .1, "exact")
        assert e == pytest.approx(0.8321207511380377)
        e = laplace_cross_entropy(x, z, .1)
        assert e == pytest.approx(0.8321207511380377)
        e = laplace_cross_entropy(z, x, .1, "approx")
        assert e == pytest.approx(0.8189687816571245)
        e = laplace_cross_entropy(x, z, .1, "approx")
        assert e == pytest.approx(0.8189542, rel=1e-6)

    @staticmethod
    def test_empty():
        """Make sure entropy raises ValueError when either input is empty."""
        pytest.raises(ValueError, laplace_entropy, [], 1.0)
        pytest.raises(ValueError, laplace_KL_divergence, [], [], 1.0)
        pytest.raises(ValueError, laplace_KL_divergence, [], [0.0], 1.0)
        pytest.raises(ValueError, laplace_KL_divergence, [0.0], [], 1.0)

    @staticmethod
    def test_non_positive_widths():
        """Make sure entropy raises ValueError when width is not positive."""
        pytest.raises(ValueError, laplace_entropy, [1, 2], 0.0)
        pytest.raises(ValueError, laplace_entropy, [1, 2], -1.0)
        pytest.raises(ValueError, laplace_KL_divergence, [1, 2], [1, 2], 0.0)
        pytest.raises(ValueError, laplace_KL_divergence, [1, 2], [1, 2], -1.0)

    @staticmethod
    def test_types():
        """Make sure entropy works when inputs are not float64."""
        for t2 in (int, float, np.float32):
            e = laplace_entropy(np.array([1, 2, 3], dtype=t2), 1.0)
            assert e == pytest.approx(1.8865648057292637)

            for t1 in (float, np.float32):
                e = laplace_cross_entropy(np.linspace(1, 3, 30, dtype=t1),
                                          np.array([1, 2, 3], dtype=t2), .2)
                assert e == pytest.approx(1.39061512214)
                e = laplace_KL_divergence(np.linspace(1, 3, 30, dtype=t1),
                                          np.array([1, 2, 3], dtype=t2), .2)
                assert e == pytest.approx(0.41768265864979814)

    @staticmethod
    def test_bug115():
        """See MASS issue #115: nonsense values are appearing in KL divergence."""
        rng = np.random.default_rng(100)
        x = mass.MnKAlpha.rvs(size=100, rng=rng, instrument_gaussian_fwhm=0)
        gain = np.linspace(-.01, .01, 21)
        for p in gain:
            xg = np.exp(p) * x
            D = mass.mathstat.entropy.laplace_cross_entropy(xg, x, w=3.0, approx_mode="exact")
            assert abs(D) < 20

    @staticmethod
    def test_bug116():
        """See MASS issue #116: nonsense values are STILL appearing in KL divergence."""
        rng = np.random.default_rng(100)
        x = mass.MnKAlpha.rvs(size=1000, rng=rng, instrument_gaussian_fwhm=0)
        gain = np.linspace(-.005, .005, 11)
        for p in gain:
            xg = np.exp(p) * x
            D = mass.mathstat.entropy.laplace_cross_entropy(xg, x, w=1.0, approx_mode="exact")
            assert abs(D) < 20
