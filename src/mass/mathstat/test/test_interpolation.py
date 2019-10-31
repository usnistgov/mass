"""
test_interpolation.py

Test that interpolation code works.

25 October 2016
Joe Fowler
"""

import unittest
import numpy as np
import mass


class Test_SmoothingSpline(unittest.TestCase):

    def test_issue74(self):
        """This is a regression test to ensure that issue #74 is fixed and
        remains fixed.
        """
        ph = np.array([604.9186911,    658.11682861,   710.25965219,   761.46157549,
                       811.81613372,   861.40110546,   910.28210151,   958.51515166,
                       1006.14861411])
        e = np.array([3000.,          3333.33333333,  3666.66666667,  4000.,
                      4333.33333333,  4666.66666667,  5000.,          5333.33333333,
                      5666.66666667])
        de = np.array([0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01])
        dph = np.array([0.60491869,  0.65811683,  0.71025965,  0.76146158,  0.81181613,
                        0.86140111,  0.9102821,  0.95851515,  1.00614861])

        # At time of issue #74, this crashed on next line for Joe, but not for Galen.
        mass.mathstat.interpolate.SmoothingSplineLog(ph, e, de, dph)

        cal = mass.calibration.energy_calibration.EnergyCalibration()
        for energy in np.linspace(3000, 6000, 10):
            ph = energy**0.8
            cal.add_cal_point(ph, energy)
        # At time of issue #74, this crashed on next line for Galen, but not for Joe.
        cal.drop_one_errors()


if __name__ == "__main__":
    unittest.main()
