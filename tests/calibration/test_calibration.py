"""
Test code for mass.calibration.

18 May 2016
Galen O'Neil
"""

import pytest
import numpy as np
import mass
import h5py
import os
from mass import EnergyCalibrationMaker


def basic_nonlinearity(e: np.ndarray) -> np.ndarray:
    return e**0.8


def basic_factory(npoints=10):
    energy = np.linspace(3000, 6000, npoints)
    ph = basic_nonlinearity(energy)
    return EnergyCalibrationMaker.init(ph, energy)


def compare_curves(curvetype1, use_approximation1, curvetype2, use_approximation2, npoints=10):
    factory = basic_factory()
    cal1 = factory.make_calibration(curvetype1, use_approximation1)
    cal2 = factory.make_calibration(curvetype2, use_approximation2)

    # Careful here: don't use a point in linspace(3000,6000,npoints),
    # or you'll get exact agreement when you don't expect/want it.
    refenergy = 5100.0
    ph1 = cal1.energy2ph(refenergy)
    ph2 = cal2.energy2ph(refenergy)
    e1 = cal1.ph2energy(basic_nonlinearity(refenergy))
    e2 = cal2.ph2energy(basic_nonlinearity(refenergy))
    doe1 = factory.drop_one_errors(curvetype1, use_approximation1)
    doe2 = factory.drop_one_errors(curvetype2, use_approximation2)
    return ph1, e1, doe1, ph2, e2, doe2, cal1, cal2


class TestLineDatabase:

    @staticmethod
    def test_synonyms():
        """Test that there are multiple equivalent synonyms for the K-alpha1 line."""
        E = mass.STANDARD_FEATURES
        e = E["MnKAlpha"]
        for name in ("MnKA", "MnKA1", "MnKL3", "MnKAlpha1"):
            assert e == E[name]

    @staticmethod
    def check_elements():
        """Check that elements appear in the list that were absent before 2017."""
        E = mass.STANDARD_FEATURES
        for element in ("U", "Pr", "Ar", "Pt", "Au", "Hg"):
            assert E[f"{element}KAlpha"] > 0.0
        assert E["MnKAlpha1"] > E["MnKAlpha2"]


@pytest.mark.filterwarnings("ignore:divide by zero encountered")
class TestJoeStyleEnergyCalibration:

    @staticmethod
    def test_copy_equality():
        """Test that any deep-copied calibration object is equivalent."""
        factory = basic_factory()
        for curvetype in factory.ALLOWED_CURVENAMES:
            for use_approximation in [True, False]:
                print(curvetype, use_approximation, "DDDDDD")
                cal1 = factory.make_calibration(curvetype, use_approximation)
                cal2 = cal1.copy()
                ph1, e1 = cal1.energy2ph(5000), cal1.ph2energy(5000)
                ph2, e2 = cal2.energy2ph(5000), cal2.ph2energy(5000)
                # drop1e, drop1err = cal1.drop_one_errors()
                # drop2e, drop2err = cal2.drop_one_errors()
                assert e1 == e2
                assert ph1 == ph2

    @staticmethod
    def test_loglog_exact_diff():
        # loglog=True makes use_zerozero not matter
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_loglog_approx_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_zerozero_exact_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_zerozero_approx_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=False,
            curvetype2="linear", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_approx_loglog_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="loglog", use_approximation1=True,
            curvetype2="loglog", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_approx_zerozero_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="linear+0", use_approximation1=True,
            curvetype2="linear+0", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_approx_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="linear", use_approximation1=True,
            curvetype2="linear", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_gain_invgain_diff():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="gain", use_approximation1=True,
            curvetype2="invgain", use_approximation2=False,)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_gain_loglog_2pts():
        ph1, e1, (_drop1e, drop1err), ph2, e2, (_drop2e, drop2err), _cal1, _cal2 = compare_curves(
            curvetype1="gain", use_approximation1=True,
            curvetype2="loglog", use_approximation2=False, npoints=3)
        assert ph1 != ph2
        assert e1 != e2
        assert not all(drop1err == drop2err)

    @staticmethod
    def test_basic_conversions():
        factory = basic_factory()
        cal1 = factory.make_calibration()
        for energy in np.linspace(3500, 5500, 10):
            ph = basic_nonlinearity(energy)
            assert ph == pytest.approx(cal1.energy2ph(energy), abs=0.1)
            assert energy == pytest.approx(cal1.ph2energy(ph), abs=0.1)

    @staticmethod
    def test_notlog_ph():
        factory = basic_factory()
        cal1 = factory.make_calibration(curvename="linear+0", approximate=True)
        for energy in np.linspace(3500, 5500, 10):
            ph = basic_nonlinearity(energy)
            assert energy == pytest.approx(cal1.ph2energy(ph), abs=10)

    @staticmethod
    def test_unordered_entries():
        energy = np.array([6000, 3000, 4500, 4000, 5000, 5500], dtype=float)
        ph = basic_nonlinearity(energy)
        dph = ph * 1e-3
        de = energy * 1e-3
        names = len(ph) * [""]
        factory = EnergyCalibrationMaker(ph, energy, dph, de, names)
        assert np.all(np.diff(factory.ph) > 0)
        cal1 = factory.make_calibration("gain", True)
        cal1(np.array([2200, 4200, 4400], dtype=float))

    @staticmethod
    def test_nonmonotonic_fail():
        "Check that issue 216 is fixed: non-monotone {E,PH} pairs should cause exceptions."
        energy = np.array([6000, 3000, 4500, 4000], dtype=float)
        ph = np.array([3000, 6000, 4500, 5000], dtype=float)
        dph = ph * 1e-3
        de = energy * 1e-3
        names = len(ph) * [""]
        with pytest.raises(Exception):
            factory = EnergyCalibrationMaker(ph, energy, dph, de, names)
            factory.make_calibration(curvename="gain", approximate=True)

    @staticmethod
    def test_save_and_load_to_hdf5():
        factory = basic_factory()
        fname = "to_be_deleted.hdf5"
        for ctype in ("loglog", "gain"):
            cal1 = factory.make_calibration(curvename=ctype)
            with h5py.File(fname, "w") as h5:
                grp = h5.require_group("calibration")
                cal1.save_to_hdf5(grp, "cal1")
                cal2 = mass.EnergyCalibration.load_from_hdf5(grp, "cal1")
                assert len(grp.keys()) == 1
            assert all(cal1.ph == cal2.ph)
            assert all(cal2.energy == cal2.energy)
            assert all(cal1.dph == cal2.dph)
            assert all(cal1.de == cal2.de)
            assert cal1.curvename == cal2.curvename
            assert cal1.approximating == cal2.approximating
            os.remove(fname)

    @staticmethod
    def test_negative_inputs():
        """Negative or zero pulse-heights shouldn't produce NaN or Inf energies, except in loglog cal."""
        factory = basic_factory()

        ph = np.arange(-10, 10, dtype=float) * 1000.
        for ct in EnergyCalibrationMaker.ALLOWED_CURVENAMES:
            if ct == "loglog":
                continue
            cal = factory.make_calibration(ct, approximate=True)
            e = cal(ph)
            print(ct, e)
            assert not any(np.isnan(e))
            assert not any(np.isinf(e))

    @staticmethod
    def test_pre_gprcal():
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
        factory = mass.EnergyCalibrationMaker.init(ph, e, dph, de)
        cal = factory.make_calibration(curvename="gain", approximate=True)
        assert (np.abs(cal(ph) - e) < 0.9 * dph).all()

        # Test for a problem in extrapolated gain that I had: gain was extrapolated with zero slope!
        g1k = 10000 / cal(10000)
        g4k = 40000 / cal(40000)
        assert g1k > 3.2
        assert g4k < 2.9

    @staticmethod
    def test_monotonic():
        "Generate 2 cal curves: cal1 is monotonic; cal2 is not. Verify this."
        names = ["CKAlpha", "NKAlpha", "OKAlpha", "FeLAlpha1", "NiLAlpha1", "CuLAlpha1"]
        e = np.array([mass.STANDARD_FEATURES[n] for n in names])
        ph1 = e * 10 / (1 + e / 2500)
        ph2 = ph1.copy()
        ph2[5] *= (ph2[5] / ph2[4])**(-0.85)
        factory1 = mass.EnergyCalibrationMaker(ph1, e, ph1 * 1e-4, np.zeros_like(e), names)
        factory2 = mass.EnergyCalibrationMaker(ph2, e, ph2 * 1e-4, np.zeros_like(e), names)
        cal1 = factory1.make_calibration("gain")
        cal2 = factory2.make_calibration("gain")
        assert cal1.ismonotonic
        assert not cal2.ismonotonic
