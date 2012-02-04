'''
Test framework for the mass.mathstat.special functions

Created on Feb 3, 2012

Joe Fowler, NIST
'''

import numpy
import unittest
from mass.mathstat import special

class TestVoigtWidth(unittest.TestCase):
    """Test the Voigt width approximation."""
    
#    def setUp(self):
#        self.nsamp = 100


    def test_voigt_gaussian_limit(self):
        """Verify Voigt function in Gaussian limit for a range of x and sigma"""

        def gaussian(x, sigma):
            return numpy.exp(-0.5*(x/sigma)**2)/(sigma*numpy.sqrt(2*numpy.pi))

        x = numpy.hstack(([0],10**numpy.arange(-2, 2, .1)))
        for gauss_width in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            sigma = gauss_width / (8*numpy.log(2))**.5
            v = special.voigt(x, x0=0, hwhm=0.0, sigma=sigma)
            for vi, xi in zip(v, x):
                self.assertAlmostEqual(vi, gaussian(xi, sigma=sigma), 7)

                
    def test_voigt_lorentzian_limit(self):
        """Verify Voigt function in Lorentzian limit for a range of x and FWHM"""

        def lorentzian(x, hwhm):
            return (hwhm/numpy.pi)/(x*x+hwhm*hwhm)

        x = numpy.hstack(([0],10**numpy.arange(-2, 2, .1)))
        for hwhm in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            v = special.voigt(x, x0=0, hwhm=hwhm, sigma=0.0)
            for vi, xi in zip(v, x):
                self.assertAlmostEqual(vi, lorentzian(xi, hwhm=hwhm), 7)

    def test_general_voigt(self):
        """I'd love to test the Voigt profile at a generic point, but I don't know how! """
        pass

    def test_voigt_width_limits(self):
        """Verify FWHM calculation of Voigt in all-Gaussian and all-Lorentz limits."""
        self.assertEqual( special.voigt_approx_fwhm(0, 0), 0)
        self.assertAlmostEqual( special.voigt_approx_fwhm(0, 1), 1, 8)
        self.assertAlmostEqual( special.voigt_approx_fwhm(1, 0), 1, 8)
        self.assertAlmostEqual( special.voigt_approx_fwhm(0, 2), 2, 8)
        self.assertAlmostEqual( special.voigt_approx_fwhm(2, 0), 2, 8)

    def test_voigt_width(self):
        """Verify FWHM approximation of Voigt with function results."""
        lor_width = 1.0
        for gauss_width in (.01, .1, .2, .5, 1, 2, 5, 10, 20):
            sigma = gauss_width / (8*numpy.log(2))**.5
            vaf = special.voigt_approx_fwhm(lor_width, gauss_width)
            vhalf = special.voigt(x=vaf*.5, x0=0, hwhm=0.5*lor_width, sigma=sigma)
            vpeak = special.voigt(x=0., x0=0., hwhm=0.5*lor_width, sigma=sigma)
            self.assertAlmostEqual( 0.5, vhalf/vpeak, 3)
                                    
    
if __name__ == "__main__":
    unittest.main()
