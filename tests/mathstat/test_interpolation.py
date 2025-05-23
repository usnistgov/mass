"""
test_interpolation.py

Test that interpolation code works.

25 October 2016
Joe Fowler
"""

import numpy as np
import mass
from mass.mathstat.interpolate import k_spline, GPRSpline


class Test_SmoothingSpline:

    @staticmethod
    def test_issue74():
        """This is a regression test to ensure that issue #74 is fixed and
        remains fixed.
        """
        ph = np.array([604.9186911, 658.11682861, 710.25965219, 761.46157549,
                       811.81613372, 861.40110546, 910.28210151, 958.51515166,
                       1006.14861411])
        e = np.array([3000., 3333.33333333, 3666.66666667, 4000.,
                      4333.33333333, 4666.66666667, 5000., 5333.33333333,
                      5666.66666667])
        de = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        dph = np.array([0.60491869, 0.65811683, 0.71025965, 0.76146158, 0.81181613,
                        0.86140111, 0.9102821, 0.95851515, 1.00614861])

        # At time of issue #74, this crashed on next line for Joe, but not for Galen.
        mass.mathstat.interpolate.SmoothingSplineLog(ph, e, de, dph)

        cal = mass.EnergyCalibrationMaker.init(e**0.8, e, dph, de)
        # At time of issue #74, this crashed on next line for Galen, but not for Joe.
        cal.drop_one_errors()


class Test_GPR:

    @staticmethod
    def test_spline_covar():
        for x in np.linspace(0, 10):
            assert x**3 / 3 == k_spline(x, x)
            assert k_spline(x, 5.5) == k_spline(5.5, x)

    @staticmethod
    def test_gprspline():
        x = np.linspace(2, 10, 9)
        s = 1.0
        delta = np.array([-0.08414947, 0.25100057, 0.70287457, -0.9225354, -0.56127467,
                          0.99469994, 0.50381756, -0.53460321, -0.49538835])

        def actualf(x):
            return 25 - 10 * x + x * x
        y = actualf(x) + s * delta
        dy = np.ones_like(y)
        spl = GPRSpline(x, y, dy)
        yspl = spl(x)
        assert np.mean((yspl - y)**2) < s**2

        xtest = np.array([0, 5, 7, 10, 12, 15])
        var = spl.variance(xtest)
        assert (var[np.logical_and(xtest > x[0] - 1, xtest < x[-1] + 1)] < 2).all()
        allowed_diff = 2 * np.sqrt(var)
        for x, ad in zip(xtest, allowed_diff):
            assert abs(spl(x) - actualf(x)) < ad
