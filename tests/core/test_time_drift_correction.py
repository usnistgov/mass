import numpy as np

import mass
import logging
LOG = logging.getLogger("mass")

rng = np.random.default_rng(19324234)  # make tests not fail randomly


def make_arrival_times(cps, duration_s):
    """Return a distribution of arrival times where time differences are drawn from
    from an exponential distribution to reflect real x-ray arrival times. The last time
    will be less than duration_s, the first time will be zero.
    """
    nmult = 2
    while True:
        tdiffs = rng.exponential(1 / float(cps), int(np.ceil(nmult * cps * duration_s)))
        t = np.cumsum(np.hstack((0, tdiffs)))
        if t[-1] > duration_s:
            return t[t < duration_s]


def make_drifting_data(distrib, res_fwhm_ev, cps, duration_s, gain_of_t):
    """Return a tuple (t, energies) where t is an array of arrival times with realistic
    distribution with counts per second cps and duration duration_s, and energies
    is drawn from distribution, has added detector energy resoltion res_fwhm_ev,
    and the energies are warped by a gain equal to gain_of_t(t).
    """
    res_sigma = res_fwhm_ev / 2.3548
    t = make_arrival_times(cps, duration_s)
    energies0 = distrib.rvs(size=len(t), instrument_gaussian_fwhm=0, rng=rng)
    gain = gain_of_t(t)
    energies = gain * energies0 + rng.standard_normal(len(t)) * res_sigma
    return t, energies


class TestTimeDriftCorrection:

    @staticmethod
    def test_make_arrival_times():
        for cps in [0.1, 1, 10, 100]:
            for duration_s in [10, 100, 1000, 10000]:
                t = make_arrival_times(cps, duration_s)
                assert t[-1] < duration_s
                # t should have N=cps*duration_s entries with std deviation sqrt(N)
                # assert that it is within 10 stdevs
                Nexpected = cps * duration_s
                assert np.abs(len(t) - Nexpected) < 10 * np.sqrt(Nexpected)

    @staticmethod
    def gain_of_t(t):
        return 1 + 0.005 * np.sin(2 * np.pi * t / 10000.)

    def test_make_drifting_data(self):
        distrib = mass.calibration.MnKAlpha
        res_fwhm_ev = 3.0
        cps = 1
        duration_s = 10000

        _t, _energy = make_drifting_data(distrib, res_fwhm_ev, cps, duration_s,
                                         self.gain_of_t)
