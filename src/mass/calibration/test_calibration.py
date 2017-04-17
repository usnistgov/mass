#!/usr/bin/env python

"""
test_calibration.py

18 May 2016
Galen O'Neil
"""

import unittest
import numpy as np
import pylab as plt
import mass
import h5py
import os


def test_options(curvetype1, use_approximation1, curvetype2, use_approximation2, refenergy=5100):
    cal1 = mass.calibration.energy_calibration.EnergyCalibration()
    cal1.set_curvetype(curvetype1)
    cal1.set_use_approximation(use_approximation1)
    for energy in np.linspace(3000,6000,10):
        ph = energy**0.8
        cal1.add_cal_point(ph, energy)
    cal2 = cal1.copy()
    cal2.set_curvetype(curvetype2)
    cal2.set_use_approximation(use_approximation2)

    # Careful here: don't use a point in linspace(3000,6000,10),
    # or you'll get exact agreement when you don't expect/want it.
    ph1 = cal1.energy2ph(refenergy)
    ph2 = cal2.energy2ph(refenergy)
    e1 = cal1.ph2energy(refenergy**0.8)
    e2 = cal2.ph2energy(refenergy**0.8)
    doe1 = cal1.drop_one_errors()
    doe2 = cal2.drop_one_errors()
    return ph1, e1, doe1, ph2, e2, doe2


class TestJoeStyleEnegyCalibration(unittest.TestCase):

    def test_copy_equality(self):
        for curvetype in ['loglog', 'linear', 'linear+0', 'gain', 'invgain', 'loggain']:
            for use_approximation in [True, False]:
                cal1 = mass.calibration.energy_calibration.EnergyCalibration()
                cal1.set_curvetype(curvetype)
                cal1.set_use_approximation(use_approximation)
                for energy in np.linspace(3000, 6000, 10):
                    ph = energy**0.8
                    cal1.add_cal_point(ph, energy)
                cal2 = cal1.copy()
                ph1, e1, (drop1e, drop1err), \
                    ph2, e2, (drop2e,drop2err) = (cal1.energy2ph(5000), cal1.ph2energy(5000),
                                                  cal1.drop_one_errors(), cal2.energy2ph(5000),
                                                  cal2.ph2energy(5000), cal2.drop_one_errors())
                # self.assertEqual(e1,e2)
                # self.assertEqual(ph1,ph2)
                assert(e1 == e2)
                assert(ph1 == ph2)

    def test_loglog_exact_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_loglog_approx_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_zerozero_exact_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_zerozero_approx_diff(self):
        # loglog=True makes use_zerozero not matter
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_approx_loglog_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="loglog", use_approximation1=True,
            curvetype2="loglog", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_approx_zerozero_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="linear+0", use_approximation1=True,
            curvetype2="linear+0", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_approx_diff(self):
        ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
            curvetype1="linear", use_approximation1=True,
            curvetype2="linear", use_approximation2=False,)
        self.assertNotEqual(ph1, ph2)
        self.assertNotEqual(e1, e2)
        self.assertFalse(all(drop1err==drop2err))

    def test_basic_energy(self):
        cal1 = mass.energy_calibration.EnergyCalibration()
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

    def test_notlog_ph(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        cal1.set_curvetype("linear+0")
        cal1.set_use_approximation(True)
        for energy in np.linspace(3000,6000,10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        for energy in np.linspace(3500,5500,10):
            ph = energy**0.8
            self.assertAlmostEqual(energy, cal1.ph2energy(ph),places=-1)

    def test_unordered_entries(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        cal1.set_curvetype("gain")
        cal1.set_use_approximation(True)
        energies = np.array([6000,3000,4500,4000,5000,5500], dtype=float)
        for energy in energies:
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        cal1(np.array([2200,4200,4400], dtype=float))

    def test_save_and_load_to_hdf5(self):
        cal1 = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000,6000,10):
            ph = energy**0.8
            cal1.add_cal_point(ph, energy)
        fname = "to_be_delete.hdf5"
        with h5py.File(fname,"w") as h5:
            grp = h5.require_group("calibration")
            cal1.save_to_hdf5(grp,"cal1")
            cal2 = mass.calibration.energy_calibration.EnergyCalibration.load_from_hdf5(grp,"cal1")
            self.assertEqual(len(grp.keys()),1)
        self.assertTrue(all(cal1._ph==cal2._ph))
        self.assertTrue(all(cal2._energies==cal2._energies))
        self.assertTrue(all(cal1._dph==cal2._dph))
        self.assertTrue(all(cal1._de==cal2._de))
        self.assertEqual(cal1.nonlinearity, cal2.nonlinearity)
        self.assertEqual(cal1.CURVETYPE, cal2.CURVETYPE)
        self.assertEqual(cal1._use_approximation, cal2._use_approximation)
        os.remove(fname)
        
if __name__ == "__main__":
    unittest.main()

curvetype1="linear+0"
use_approximation1=False,
curvetype2="linear"
use_approximation2=False


cal1 = mass.calibration.energy_calibration.EnergyCalibration()
cal1.set_curvetype(curvetype1)
cal1.set_use_approximation(use_approximation1)
for energy in np.linspace(3000,6000,10):
    ph = energy**0.8
    cal1.add_cal_point(ph, energy)
cal2 = cal1.copy()
cal2.set_curvetype(curvetype2)
cal2.set_use_approximation(use_approximation2)

# Careful here: don't use a point in linspace(3000,6000,10),
# or you'll get exact agreement when you don't expect/want it.
ph1 = cal1.energy2ph(5100)
ph2 = cal2.energy2ph(5100)
e1 = cal1.ph2energy(5100**0.8)
e2 = cal2.ph2energy(5100**0.8)
doe1 = cal1.drop_one_errors()
doe2 = cal2.drop_one_errors()

ph1, e1, (drop1e, drop1err), ph2, e2, (drop2e,drop2err) = test_options(
    curvetype1="linear+0", use_approximation1=False,
    curvetype2="linear", use_approximation2=False,refenergy=10)
