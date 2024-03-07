'''
Test framework for the mass.mathstat.special functions

Created on Feb 3, 2012

Joe Fowler, NIST
'''

import pytest
import numpy as np
from mass.mathstat import special


class TestVoigtWidth:
    """Test the Voigt width approximation."""

    @staticmethod
    def test_voigt_gaussian_limit():
        """Verify Voigt function in Gaussian limit for a range of x and sigma."""

        def gaussian(x, sigma):
            return np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

        x = np.hstack(([0], 10**np.arange(-2, 2, .1)))
        for gauss_width in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            sigma = gauss_width / (8 * np.log(2))**.5
            v = special.voigt(x, xctr=0, hwhm=0.0, sigma=sigma)
            for vi, xi in zip(v, x):
                assert vi == pytest.approx(gaussian(xi, sigma=sigma), 7)

    @staticmethod
    def test_voigt_lorentzian_limit():
        """Verify Voigt function in Lorentzian limit for a range of x and FWHM."""

        def lorentzian(x, hwhm):
            return (hwhm / np.pi) / (x * x + hwhm * hwhm)

        x = np.hstack(([0], 10**np.arange(-2, 2, .1)))
        for hwhm in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            v = special.voigt(x, xctr=0, hwhm=hwhm, sigma=0.0)
            for vi, xi in zip(v, x):
                assert vi == pytest.approx(lorentzian(xi, hwhm=hwhm), 7)

    # def test_general_voigt():
    #     """I'd love to test the Voigt profile at a generic point, but I don't know how! """
    #     pass

    @staticmethod
    def test_voigt_width_limits():
        """Verify FWHM calculation of Voigt in all-Gaussian and all-Lorentz limits."""
        assert special.voigt_approx_fwhm(0, 0) == 0
        assert special.voigt_approx_fwhm(0, 1) == pytest.approx(1, abs=1e-8)
        assert special.voigt_approx_fwhm(1, 0) == pytest.approx(1, abs=1e-8)
        assert special.voigt_approx_fwhm(0, 2) == pytest.approx(2, abs=1e-8)
        assert special.voigt_approx_fwhm(2, 0) == pytest.approx(2, abs=1e-8)

    @staticmethod
    def test_voigt_width():
        """Verify FWHM approximation of Voigt with function results."""
        lor_width = 1.0
        for gauss_width in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            sigma = gauss_width / (8 * np.log(2))**.5
            vaf = special.voigt_approx_fwhm(lor_width, gauss_width)
            vhalf = special.voigt(x=vaf * .5, xctr=0, hwhm=0.5 * lor_width, sigma=sigma)
            vpeak = special.voigt(x=0., xctr=0., hwhm=0.5 * lor_width, sigma=sigma)
            assert 0.5 == pytest.approx(vhalf / vpeak, abs=1e-3)
