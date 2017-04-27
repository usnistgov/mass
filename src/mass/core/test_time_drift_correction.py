import tempfile
import os.path

import numpy as np
import os
import shutil
import unittest as ut

import mass
import logging
LOG = logging.getLogger("mass")
# LOG.setLevel(logging.NOTSET)

np.random.seed(19324234) # make tests not fail randomly

def make_arrival_times(cps, duration_s):
    """
    Return a distribution of arrival times where time differences are drawn from
    from an exponential distribution to reflect real x-ray arrival times. The last time
    will be less than duration_s, the first time will be after zero.
    """
    nmult=2
    while True:
        tdiffs = np.random.exponential(1/float(cps), int(np.ceil(nmult*cps*duration_s)))
        t=np.cumsum(tdiffs)
        if t[-1] < duration_s:
            nmult*=2
            continue
        t=t[t<duration_s]
        return t

def make_drifting_data(distrib, res_fwhm_ev, cps, duration_s, gain_of_t):
    res_sigma = res_fwhm_ev / 2.3548
    t = make_arrival_times(cps, duration_s)
    energies0 = distrib.rvs(size=len(t))
    #energies0 += np.random.standard_normal(len(t))*res_sigma
    gain = gain_of_t(t)
    energies = gain*energies0
    return t,energies


class TestTimeDriftCorrection(ut.TestCase):

    def test_make_arrival_times(self):
        for cps in [0.1,1,10,100]:
            for duration_s in [10,100,1000,10000]:
                t = make_arrival_times(cps, duration_s)
                self.assertTrue(t[-1]<duration_s)
                # t should have N=cps*duration_s entries with std deviation sqrt(N)
                # assert that it is within 10 stdevs
                Nexpected = cps*duration_s
                self.assertTrue(np.abs(len(t)-Nexpected)<10*np.sqrt(Nexpected))

    def test_make_drifting_data(self):
        distrib = mass.calibration.MnKAlphaDistribution()
        res_fwhm_ev = 3.0
        cps=1
        duration_s = 10000
        gain_of_t = lambda t: 1+0.005*np.sin(2*np.pi*t/10000.)
        t,energy = make_drifting_data(distrib, res_fwhm_ev, cps, duration_s, gain_of_t)


if __name__ == '__main__':
    ut.main()
