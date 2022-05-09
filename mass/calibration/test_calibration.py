"""
Test code for mass.calibration.

18 May 2016
Galen O'Neil
"""

import unittest
import pytest
import numpy as np
import mass
import h5py
import os


def compare_curves(curvetype1, use_approximation1, curvetype2, use_approximation2, refenergy=5100, npoints=10):
    cal1 = mass.calibration.energy_calibration.EnergyCalibration()
    cal1.set_curvetype(curvetype1)
    cal1.set_use_approximation(use_approximation1)
    for energy in np.linspace(3000, 6000, npoints):
        ph = energy**0.8
        cal1.add_cal_point(ph, energy)
    cal2 = cal1.copy()
    cal2.set_curvetype(curvetype2)
    cal2.set_use_approximation(use_approximation2)

    # Careful here: don't use a point in linspace(3000,6000,npoints),
    # or you'll get exact agreement when you don't expect/want it.
    ph1 = cal1.energy2ph(refenergy)
    ph2 = cal2.energy2ph(refenergy)
    e1 = cal1.ph2energy(refenergy**0.8)
    e2 = cal2.ph2energy(refenergy**0.8)
    doe1 = cal1.drop_one_errors()
    doe2 = cal2.drop_one_errors()
    return ph1, e1, doe1, ph2, e2, doe2, cal1, cal2


class TestLineDatabase(unittest.TestCase):

    def test_synonyms(self):
        """Test that there are multiple equivalent synonyms for the K-alpha1 line."""
        E = mass.STANDARD_FEATURES
        e = E["MnKAlpha"]
        for name in ("MnKA", "MnKA1", "MnKL3", "MnKAlpha1"):
            self.assertEqual(e, E[name])

    def check_elements(self):
        """Check that elements appear in the list that were absent before 2017."""
        E = mass.STANDARD_FEATURES
        for element in ("U", "Pr", "Ar", "Pt", "Au", "Hg"):
            self.assertGreater(E["%sKAlpha" % element], 0.0)
        self.assertGreater(E["MnKAlpha1"], E["MnKAlpha2"])


@pytest.mark.filterwarnings("ignore:divide by zero encountered")
class TestJoeStyleEnergyCalibration(unittest.TestCase):

    def test_copy_equality(self):
        """Test that any deep-copied calibration object is equivalent."""
        for curvetype in ['loglog', 'linear', 'linear+0', 'gain', 'invgain', 'loggain']:
            for use_approximation in [True, False]:
                cal1 = mass.calibration.energy_calibration.EnergyCalibration()
                cal1.set_curvetype(curvetype)
                cal1.set_use_approximation(use_approximation)
                for energy in np.linspace(3000, 6000, 10):
                    ph = energy**0.8
                    cal1.add_cal_point(ph, energy)
                cal2 = cal1.copy()
                ph1, e1 = cal1.energy2ph(5000), cal1.ph2energy(5000)
                ph2, e2 = cal2.energy2ph(5000), cal2.ph2energy(5000)
                # drop1e, drop1err = cal1.drop_one_errors()
                # drop2e, drop2err = cal2.drop_one_errors()
                self.assertEqual(e1, e2)
                self.assertEqual(ph1, ph2)

    def test_loglog_exact_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_loglog_approx_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_zerozero_exact_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_zerozero_approx_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_approx_loglog_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=True,
            curvetype2="loglog", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_approx_zerozero_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=True,
            curvetype2="linear+0", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_approx_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="linear", use_approximation1=True,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_gain_invgain_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="gain", use_approximation1=True,
            curvetype2="invgain", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_gain_loglog_2pts(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e, drop2err), cal1, cal2 = compare_curves(
            curvetype1="gain", use_approximation1=True,
            curvetype2="loglog", use_approximation2=False, npoints=2)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err == drop2err))

    def test_basic_energy(self):
        cal1 = mass.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500, 5500, 10):
            ph = energy**0.8
            self.assertAlmostEqual(ph, cal1.energy2ph(energy), places=1)

    def test_basic_ph(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500, 5500, 10):
            ph = energy**0.8
            self.assertAlmostEqual(energy, cal1.ph2energy(ph), places=1)

    def test_notlog_ph(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        cal1.set_curvetype("linear+0")
        cal1.set_use_approximation(True)
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500, 5500, 10):
            ph = energy**0.8
            self.assertAlmostEqual(energy, cal1.ph2energy(ph), places=-1)

    def test_unordered_entries(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        cal1.set_curvetype("gain")
        cal1.set_use_approximation(True)
        energies = np.array([6000, 3000, 4500, 4000, 5000, 5500], dtype=float)
        for energy in energies:
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        cal1(np.array([2200, 4200, 4400], dtype=float))

    def test_nonmonotonic_fail(self):
        "Check that issue 216 is fixed: non-monotone {E,PH} pairs should cause exceptions."
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        cal1.set_curvetype("gain")
        cal1.set_use_approximation(True)
        energies = np.array([6000, 3000, 4500, 4000], dtype=float)
        phvec = np.array([3000, 6000, 4500, 5000], dtype=float)
        with self.assertRaises(Exception):
            for ph, energy in zip(phvec, energies):
                cal1.add_cal_point(ph, energy)

    def test_save_and_load_to_hdf5(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        fname = "to_be_deleted.hdf5"
        for ctype in (0, "gain"):
            cal1.set_curvetype(ctype)
            with h5py.File(fname, "w") as h5:
                grp = h5.require_group("calibration")
                cal1.save_to_hdf5(grp, "cal1")
                cal2 = mass.calibration.energy_calibration.EnergyCalibration.load_from_hdf5(
                    grp, "cal1")
                self.assertEqual(len(grp.keys()), 1)
            self.assertTrue(all(cal1._ph == cal2._ph))
            self.assertTrue(all(cal2._energies == cal2._energies))
            self.assertTrue(all(cal1._dph == cal2._dph))
            self.assertTrue(all(cal1._de == cal2._de))
            self.assertEqual(cal1.nonlinearity, cal2.nonlinearity)
            self.assertEqual(cal1.CURVETYPE, cal2.CURVETYPE)
            self.assertEqual(cal1._use_approximation, cal2._use_approximation)
            os.remove(fname)

    def test_negative_inputs(self):
        """Negative or zero pulse-heights shouldn't produce NaN or Inf energies."""
        cal = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.9
            cal.add_cal_point(ph, energy)
        cal.set_use_approximation(True)

        ph = np.arange(-10, 10, dtype=float)*1000.
        for ct in cal.CURVETYPE:
            cal.set_curvetype(ct)
            e = cal(ph)
            self.assertFalse(any(np.isnan(e)))
            self.assertFalse(any(np.isinf(e)))

    def test_pre_GPRcal(self):
        "Test the pre-2021 calibration still works"
        ph = np.array([17157.08056038, 18532.35241609, 18667.38206583, 19942.89858008,
                       20181.77187566, 21382.23254964, 21727.54556571, 22848.99053659,
                       23300.78419074, 24340.78447936, 24899.9448867, 25239.423792,
                       26526.9605488, 28420.90655636, 29111.86088393])
        e = np.array([5414.8045, 5898.801, 5946.823, 6404.0062, 6490.585, 6930.378,
                      7058.175, 7478.2521, 7649.445, 8047.8227, 8264.775, 8398.242,
                      8905.413, 9672.575, 9964.133])
        dph = np.array([0.07974264, 0.12733291, 0.1818605, 0.10211174, 0.25735167,
                        0.05803863, 0.28282476, 0.07275278, 0.17837849, 0.0820041,
                        0.22199391, 0.54440676, 0.26877157, 2.36176241, 1.74482802])
        de = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                       0.01, 0.01, 0.01, 0.01])
        cal1 = mass.EnergyCalibration(curvetype="gain", approximate=True, useGPR=False)
        cal2 = mass.EnergyCalibration(curvetype="gain", approximate=True, useGPR=True)
        for cal in (cal1, cal2):
            for a, b, c, d in zip(ph, e, dph, de):
                cal.add_cal_point(a, b, pht_error=c, e_error=d, name="{:.3f} eV".format(b))
            cal._update_converters()
        self.assertTrue((np.abs(cal1(ph)-e) < 1.2*dph).all())
        self.assertTrue((np.abs(cal2(ph)-e) < 0.7*dph).all())

        # Be sure that the old-style (non-GPR) spline finds the right curvature
        self.assertAlmostEqual(cal1._underlying_spline.actualchisq, len(ph), 2)

        # Test for a problem in extrapolated gain that I had: gain was extrapolated with zero slope!
        for cal in (cal1, cal2):
            g1k = 10000/cal(10000)
            g4k = 40000/cal(40000)
            self.assertGreater(g1k, 3.2)
            self.assertLess(g4k, 2.9)


if __name__ == "__main__":
    unittest.main()
