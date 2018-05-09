"""
Test code for mass.calibration.algorithms.
"""

import unittest
import numpy as np
import mass
from mass.calibration.algorithms import *
import itertools

np.random.seed(2)

class TestAlgorithms(unittest.TestCase):

    def test_find_opt_assignment(self):
        known_energies = np.array([3100, 3200, 3300, 3600, 4000, 4500, 5200, 5800,
                                   6500, 8300, 9200, 10200])
        ph = known_energies**0.95
        combos = itertools.combinations(range(len(ph)), 8)
        tries = 0
        passes = 0
        for combo in combos:
            tries += 1
            inds = np.array(combo)
            _name_e, energies_out, opt_assignments = find_opt_assignment(ph, known_energies[inds])
            if all(energies_out == known_energies[inds]) and all(opt_assignments == ph[inds]):
                passes += 1
        self.assertTrue(passes > tries*0.9)

    def test_find_local_maxima(self):
        np.random.seed(100)
        ph = np.random.randn(10000)+7000
        ph = np.hstack((ph, np.random.randn(5000)+4000))
        ph = np.hstack((ph, np.random.randn(1000)+1000))
        local_maxima, _ = find_local_maxima(ph, 10)
        local_maxima, _peak_heights = find_local_maxima(ph, 10)
        rounded = np.round(local_maxima)
        self.assertTrue(all(rounded[:3] == np.array([7000, 4000, 1000])))

    def test_build_fit_ranges(self):
        known_energies = np.array([1000, 2000, 2050, 3000])
        # make a 1 to 10 calibration
        cal1 = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for ke in known_energies:
            # args are (pulseheight, energy)
            cal1.add_cal_point(0.1*ke, ke)

        # this call asks for fit ranges at each known energy, and asks to avoid the line at 3050,
        # uses cal1 for the apprixmate calibraiton and asks for 100 eV wide fit ranges
        eout, fit_lo_hi_energy, slopes_de_dph = build_fit_ranges(known_energies, [3050], cal1, 100)
        self.assertTrue(all(eout == known_energies))
        self.assertEqual(len(fit_lo_hi_energy), len(known_energies))
        lo, hi = fit_lo_hi_energy[0]
        self.assertAlmostEqual(lo, 950)
        self.assertAlmostEqual(hi, 1050)
        lo, hi = fit_lo_hi_energy[1]
        self.assertAlmostEqual(lo, 1950)
        self.assertAlmostEqual(hi, 2025)
        lo, hi = fit_lo_hi_energy[2]
        self.assertAlmostEqual(lo, 2025)
        self.assertAlmostEqual(hi, 2100)
        lo, hi = fit_lo_hi_energy[3]
        self.assertAlmostEqual(lo, 2950)
        self.assertAlmostEqual(hi, 3025)

    def test_build_fit_ranges_ph(self):
        known_energies = np.array([1000, 2000, 2050, 3000])
        # make a 1 to 10 calibration
        cal1 = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for ke in known_energies:
            # args are (pulseheight, energy)
            cal1.add_cal_point(0.1*ke, ke)

        # this call asks for fit ranges at each known energy, and asks to avoid the line at 3050,
        # uses cal1 for the apprixmate calibraiton and asks for 100 eV wide fit ranges
        eout, fit_lo_hi, slopes_de_dph = build_fit_ranges_ph(known_energies, [3050], cal1, 100)
        self.assertTrue(all(eout == known_energies))
        self.assertEqual(len(fit_lo_hi), len(known_energies))
        lo, hi = fit_lo_hi[0]
        self.assertAlmostEqual(lo, 950*0.1)
        self.assertAlmostEqual(hi, 1050*0.1)
        lo, hi = fit_lo_hi[1]
        self.assertAlmostEqual(lo, 1950*0.1)
        self.assertAlmostEqual(hi, 2025*0.1)
        lo, hi = fit_lo_hi[2]
        self.assertAlmostEqual(lo, 2025*0.1)
        self.assertAlmostEqual(hi, 2100*0.1)
        lo, hi = fit_lo_hi[3]
        self.assertAlmostEqual(lo, 2950*0.1)
        self.assertAlmostEqual(hi, 3025*0.1)

    def test_complete(self):
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
        e = []
        for k,s in spect.items():
            e.extend(s.rvs(size=num_samples[k]))
        e = np.array(e)
        e = e[e > 0]   # The wide-tailed distributions will occasionally produce negative e. Bad!
        ph = 2*e**0.9
        smoothing_res_ph = 20
        lm, _lm_heights = find_local_maxima(ph, smoothing_res_ph)
        line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "TiKAlpha", "FeKAlpha"]

        _names_e, energies_opt, ph_opt = find_opt_assignment(lm, line_names)

        approxcal = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for (ee, phph) in zip(energies_opt, ph_opt):
            approxcal.add_cal_point(phph, ee)

        _energies, fit_lo_hi, slopes_de_dph = build_fit_ranges_ph(energies_opt, [], approxcal, 100)
        binsize_ev = 1.0
        fitters = multifit(ph, line_names, fit_lo_hi, np.ones_like(slopes_de_dph)*binsize_ev, slopes_de_dph)


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
        e = []
        for k,s in spect.items():
            e.extend(s.rvs(size=num_samples[k]))
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
