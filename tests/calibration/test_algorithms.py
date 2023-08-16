"""
Test code for mass.calibration.algorithms.
"""

from pytest import approx
import numpy as np
import mass
from mass.calibration.algorithms import find_opt_assignment, find_local_maxima, build_fit_ranges, \
    build_fit_ranges_ph, multifit, EnergyCalibration, EnergyCalibrationAutocal
import itertools

rng = np.random.default_rng(2)


class TestAlgorithms:

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
        assert passes > tries*0.9

    def test_find_local_maxima(self):
        rng = np.random.default_rng(100)
        ph = rng.standard_normal(10000)+7000
        ph = np.hstack((ph, rng.standard_normal(5000)+4000))
        ph = np.hstack((ph, rng.standard_normal(1000)+1000))
        local_maxima, _ = find_local_maxima(ph, 10)
        local_maxima, _peak_heights = find_local_maxima(ph, 10)
        rounded = np.round(local_maxima)
        assert all(rounded[:3] == np.array([7000, 4000, 1000]))

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
        assert all(eout == known_energies)
        assert len(fit_lo_hi_energy) == len(known_energies)
        lo, hi = fit_lo_hi_energy[0]
        assert lo == approx(950)
        assert hi == approx(1050)
        lo, hi = fit_lo_hi_energy[1]
        assert lo == approx(1950)
        assert hi == approx(2025)
        lo, hi = fit_lo_hi_energy[2]
        assert lo == approx(2025)
        assert hi == approx(2100)
        lo, hi = fit_lo_hi_energy[3]
        assert lo == approx(2950)
        assert hi == approx(3025)

    def test_build_fit_ranges_ph(self):
        known_energies = np.array([1000, 2000, 2050, 3000])
        # make a 1 to 10 calibration
        cal1 = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for ke in known_energies:
            # args are (pulseheight, energy)
            cal1.add_cal_point(0.1*ke, ke)

        # this call asks for fit ranges at each known energy, and asks to avoid the line at 3050,
        # uses cal1 for the apprixmate calibraiton and asks for 100 eV wide fit ranges
        eout, fit_lo_hi_energy, slopes_de_dph = build_fit_ranges_ph(known_energies, [3050], cal1, 100)
        assert all(eout == known_energies)
        assert len(fit_lo_hi_energy) == len(known_energies)
        lo, hi = fit_lo_hi_energy[0]
        assert lo == approx(950*0.1)
        assert hi == approx(1050*0.1)
        lo, hi = fit_lo_hi_energy[1]
        assert lo == approx(1950*0.1)
        assert hi == approx(2025*0.1)
        lo, hi = fit_lo_hi_energy[2]
        assert lo == approx(2025*0.1)
        assert hi == approx(2100*0.1)
        lo, hi = fit_lo_hi_energy[3]
        assert lo == approx(2950*0.1)
        assert hi == approx(3025*0.1)

    def test_complete(self):
        # generate pulseheights from known spectrum
        num_samples = {k: 1000*(k+1) for k in range(5)}
        line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "TiKAlpha", "FeKAlpha"]
        spect = {i: mass.spectra[n] for (i, n) in enumerate(line_names)}
        e = []
        for (k, s) in spect.items():
            e.extend(s.rvs(size=num_samples[k], instrument_gaussian_fwhm=k+3.0, rng=rng))
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
        results = multifit(ph, line_names, fit_lo_hi, np.ones_like(
            slopes_de_dph)*binsize_ev, slopes_de_dph, hide_deprecation=True)
        assert results is not None

    def test_autocal(self):
        # generate pulseheights from known spectrum
        num_samples = {k: 1000*(k+1) for k in range(5)}
        line_names = ["MnKAlpha", "MnKBeta", "CuKAlpha", "TiKAlpha", "FeKAlpha"]
        spect = {i: mass.spectra[n] for (i, n) in enumerate(line_names)}
        e = []
        for (k, s) in spect.items():
            e.extend(s.rvs(size=num_samples[k], instrument_gaussian_fwhm=k+3.0, rng=rng))
        e = np.array(e)
        e = e[e > 0]   # The wide-tailed distributions will occasionally produce negative e. Bad!
        ph = 2*e**0.9

        cal = EnergyCalibration()
        auto_cal = EnergyCalibrationAutocal(cal, ph, line_names)
        auto_cal.autocal()
        auto_cal.diagnose()
        cal.diagnose()
        assert hasattr(cal, "autocal")
        # test fitters are correct type, and ordered by line energy
        e0 = 0
        for r in auto_cal.results:
            assert isinstance(r.model, mass.calibration.line_models.GenericLineModel)
            peak = r.model.spect.peak_energy
            assert e0 < peak
            e0 = peak
