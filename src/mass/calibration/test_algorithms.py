import unittest
import numpy as np
import pylab as plt
import mass
from mass.calibration.algorithms import *
import itertools


class TestAlgorithms(unittest.TestCase):

    def test_find_opt_assignment(self):
        known_energies = np.array([3100,3200,3300,3600,4000,4500,5200,5800,6500,8300,9200,10200])
        ph = known_energies**0.95
        combos = itertools.combinations(range(len(ph)), 8)
        tries = 0
        passes = 0
        for combo in combos:
            tries += 1
            inds = np.array(combo)
            name_e, energies_out, opt_assignments = find_opt_assignment(ph, known_energies[inds])
            if all(energies_out == known_energies[inds]) and all(opt_assignments == ph[inds]):
                passes += 1
        self.assertTrue(passes > tries*0.9)

    def test_find_local_maxima(self):
        np.random.seed(100)
        ph = np.random.randn(10000)+7000
        ph = np.hstack((ph, np.random.randn(5000)+4000))
        ph = np.hstack((ph, np.random.randn(1000)+1000))
        local_maxima, _ = find_local_maxima(ph, 10)
        rounded = np.round(local_maxima)
        self.assertTrue(all(rounded[:3] == np.array([7000, 4000, 1000])))

    def test_build_fit_ranges(self):
        known_energies = np.array([1000, 2000, 2050, 3000])
        # make a 1 to 1 calibration
        cal1 = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for ke in known_energies:
            cal1.add_cal_point(ke, ke)

        eout, fit_lo_hi, slopes_de_dph = build_fit_ranges(known_energies, [3050], cal1, 100)
        self.assertTrue(all(eout == known_energies))
        self.assertEqual(len(fit_lo_hi), len(known_energies))
        lo, hi = fit_lo_hi[0]
        self.assertAlmostEqual(lo, 950)
        self.assertAlmostEqual(hi, 1050)
        lo, hi = fit_lo_hi[1]
        self.assertAlmostEqual(lo, 1950)
        self.assertAlmostEqual(hi, 2025)
        lo, hi = fit_lo_hi[2]
        self.assertAlmostEqual(lo, 2025)
        self.assertAlmostEqual(hi, 2100)
        lo, hi = fit_lo_hi[3]
        self.assertAlmostEqual(lo, 2950)
        self.assertAlmostEqual(hi, 3025)

    def test_complete(self):
        # generate pulseheights from known spectrum
        spect = {}
        dist = {}
        num_samples = {k:1000*k for k in [1,2,3,4,5]}
        spect[1] = mass.fluorescence_lines.MnKAlpha()
        spect[1].set_gauss_fwhm(2)
        spect[2] = mass.fluorescence_lines.MnKBeta()
        spect[2].set_gauss_fwhm(3)
        spect[3] = mass.fluorescence_lines.CuKAlpha()
        spect[3].set_gauss_fwhm(4)
        spect[4] = mass.fluorescence_lines.TiKAlpha()
        spect[4].set_gauss_fwhm(5)
        spect[5] = mass.fluorescence_lines.FeKAlpha()
        spect[5].set_gauss_fwhm(6)
        dist[1]  = mass.fluorescence_lines.MnKAlphaDistribution()
        dist[2]  = mass.fluorescence_lines.MnKBetaDistribution()
        dist[3]  = mass.fluorescence_lines.CuKAlphaDistribution()
        dist[4]  = mass.fluorescence_lines.TiKAlphaDistribution()
        dist[5]  = mass.fluorescence_lines.FeKAlphaDistribution()
        dist[1].distribution.set_gauss_fwhm(4)
        dist[2].distribution.set_gauss_fwhm(4)
        dist[3].distribution.set_gauss_fwhm(5)
        dist[4].distribution.set_gauss_fwhm(3)
        dist[5].distribution.set_gauss_fwhm(4)
        e =[]
        for (k, v) in spect.items():
            sampler = dist[k]
            e.extend(sampler.rvs(size=num_samples[k]))
        e = np.array(e)
        e = e[e > 0]   # The wide-tailed distributions will occasionally produce negative e. Bad!
        ph = 2*e**0.9
        smoothing_res_ph = 20
        lm, _ = find_local_maxima(ph, smoothing_res_ph)
        line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "TiKAlpha", "FeKAlpha"]

        names_e, energies_opt, ph_opt = find_opt_assignment(lm, line_names)

        approxcal = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for (ee, phph) in zip(energies_opt, ph_opt):
            approxcal.add_cal_point(phph, ee)

        energies, fit_lo_hi, slopes_de_dph = build_fit_ranges_ph(energies_opt, [], approxcal, 100)
        binsize_ev = 1.0
        fitters = multifit(ph, line_names, fit_lo_hi, np.ones_like(slopes_de_dph)*binsize_ev, slopes_de_dph)

        line_names_2 = ["MnKAlpha", "MnKBeta", "CuKAlpha", mass.calibration.STANDARD_FEATURES["TiKAlpha"], "FeKAlpha"]
        fitters2 = multifit(ph, line_names_2, fit_lo_hi, np.ones_like(slopes_de_dph)*binsize_ev, slopes_de_dph)

    def test_autocal(self):
        # generate pulseheights from known spectrum
        spect = {}
        dist = {}
        num_samples = {k: 1000*k for k in [1, 2, 3, 4, 5]}
        spect[1] = mass.fluorescence_lines.MnKAlpha()
        spect[1].set_gauss_fwhm(2)
        spect[2] = mass.fluorescence_lines.MnKBeta()
        spect[2].set_gauss_fwhm(3)
        spect[3] = mass.fluorescence_lines.CuKAlpha()
        spect[3].set_gauss_fwhm(4)
        spect[4] = mass.fluorescence_lines.TiKAlpha()
        spect[4].set_gauss_fwhm(5)
        spect[5] = mass.fluorescence_lines.FeKAlpha()
        spect[5].set_gauss_fwhm(6)
        dist[1]  = mass.fluorescence_lines.MnKAlphaDistribution()
        dist[2]  = mass.fluorescence_lines.MnKBetaDistribution()
        dist[3]  = mass.fluorescence_lines.CuKAlphaDistribution()
        dist[4]  = mass.fluorescence_lines.TiKAlphaDistribution()
        dist[5]  = mass.fluorescence_lines.FeKAlphaDistribution()
        dist[1].distribution.set_gauss_fwhm(4)
        dist[2].distribution.set_gauss_fwhm(4)
        dist[3].distribution.set_gauss_fwhm(5)
        dist[4].distribution.set_gauss_fwhm(3)
        dist[5].distribution.set_gauss_fwhm(4)

        e = []
        for (k, v) in spect.items():
            sampler = dist[k]
            e.extend(sampler.rvs(size=num_samples[k]))
        e = np.array(e)
        e = e[e > 0]   # The wide-tailed distributions will occasionally produce negative e. Bad!
        ph = 2*e**0.9

        line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "TiKAlpha", "FeKAlpha"]

        cal = EnergyCalibration()
        auto_cal = EnergyCalibrationAutocal(cal, ph, line_names)
        auto_cal.autocal()
        auto_cal.diagnose()


if __name__ == "__main__":
    unittest.main()
