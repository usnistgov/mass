"""
test_fits.py

Test that Maximum Likelihood Fits work

5 May 2016
Joe Fowler
"""

import unittest
import numpy as np
import pylab as plt
import mass


def test_options(use_loglog1, use_zerozero1, use_approximation1, use_loglog2, use_zerozero2, use_approximation2):
    cal1 = mass.calibration.energy_calibration.EnergyCalibration()
    cal1.set_use_loglog(use_loglog1)
    cal1.set_use_zerozero(use_zerozero1)
    cal1.set_use_approximation(use_approximation1)
    for energy in np.linspace(3000,6000,10):
        ph = energy**0.8
        cal1.add_cal_point(ph, energy)
    cal2 = cal1.copy()
    cal2.set_use_loglog(use_loglog2)
    cal2.set_use_zerozero(use_zerozero2)
    cal2.set_use_approximation(use_approximation2)
    return cal1.energy2ph(5000), cal1.ph2energy(5000), cal1.drop_one_errors(), cal2.energy2ph(5000), cal2.ph2energy(5000), cal2.drop_one_errors(),


class TestJoeStyleEnegyCalibration(unittest.TestCase):


    def test_copy_equality(self):
        for use_loglog in [True,False]:
            for use_zerozero in [True, False]:
                for use_approximation in [True, False]:
                    if use_approximation and not use_loglog:
                        continue # for now this crashes
                    cal1 = mass.calibration.energy_calibration.EnergyCalibration()
                    cal1.set_use_loglog(use_loglog)
                    cal1.set_use_zerozero(use_zerozero)
                    cal1.set_use_approximation(use_approximation)
                    for energy in np.linspace(3000,6000,10):
                        ph = energy**0.8
                        cal1.add_cal_point(ph, energy)
                    cal2 = cal1.copy()
                    ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = cal1.energy2ph(5000), cal1.ph2energy(5000), cal1.drop_one_errors(), cal2.energy2ph(5000), cal2.ph2energy(5000), cal2.drop_one_errors(),
                    # self.assertEqual(e1,e2)
                    # self.assertEqual(ph1,ph2)
                    assert(e1==e2)
                    assert(ph1==ph2)


    def test_loglog_exact_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            use_loglog1=True, use_zerozero1=False, use_approximation1=False,
            use_loglog2=False, use_zerozero2=False, use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_loglog_approx_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            use_loglog1=True, use_zerozero1=False, use_approximation1=False,
            use_loglog2=False, use_zerozero2=False, use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_zerozero_exact_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            use_loglog1=False, use_zerozero1=True, use_approximation1=False,
            use_loglog2=False, use_zerozero2=False, use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    # def test_zerozero_approx_diff(self):
    #     # loglog=True makes use_zerozero not matter
    #     ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
    #         use_loglog1=False, use_zerozero1=True, use_approximation1=True,
    #         use_loglog2=False, use_zerozero2=False, use_approximation2=True,)
    #     self.assertNotEqual(ph1, ph2)
    #     self.assertNotEqual(e1, e2)
    #     self.assertFalse(all(drop1err==drop2err))

    def test_approx_loglog_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            use_loglog1=True, use_zerozero1=False, use_approximation1=True,
            use_loglog2=True, use_zerozero2=False, use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    # def test_approx_zerozero_diff(self):
    #     ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
    #         use_loglog1=False, use_zerozero1=True, use_approximation1=True,
    #         use_loglog2=False, use_zerozero2=True, use_approximation2=False,)
    #     self.assertNotEqual(ph1, ph2)
    #     self.assertNotEqual(e1, e2)
    #     self.assertFalse(all(drop1err==drop2err))

    # def test_approx_diff(self):
    #     ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
    #         use_loglog1=False, use_zerozero1=False, use_approximation1=True,
    #         use_loglog2=False, use_zerozero2=False, use_approximation2=False,)
    #     self.assertNotEqual(ph1, ph2)
    #     self.assertNotEqual(e1, e2)
    #     self.assertFalse(all(drop1err==drop2err))

    def test_basic_energy(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000,6000,10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500,5500,10):
            ph = energy**0.8
            self.assertAlmostEqual(ph, cal1.energy2ph(energy),places=1)

    def test_basic_ph(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000,6000,10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500,5500,10):
            ph = energy**0.8
            self.assertAlmostEqual(energy, cal1.ph2energy(ph),places=1)

if __name__ == "__main__":
    unittest.main()
