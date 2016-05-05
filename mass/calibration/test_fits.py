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

class Test_Gaussian(unittest.TestCase):

    def setUp(self):
        sigma = 1.5
        center = 15
        ampl = 1000
        self.params = [2.3548*sigma, center, ampl, 0.1, 0, 1e-9, 10]
        self.x = np.linspace(10,20,200)
        self.y = ampl * np.exp(-0.5*(self.x-center)**2/(sigma**2))
        self.obs = np.array([np.random.poisson(lam=y0) for y0 in self.y])
        self.fitter = mass.calibration.line_fits.GaussianFitter()

    def test_fit(self):
        self.fitter.phscale_positive = True
        param, covar = self.fitter.fit(self.obs, self.x, self.params, plot=True)
        plt.savefig("/tmp/test_gaussian1.pdf")
        self.assertAlmostEqual(param[0], self.params[0], 1) # FWHM
        self.assertAlmostEqual(param[1], self.params[1], 1) # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1) # Amplitude

    def test_fit_offset(self):
        center = self.params[1]
        self.fitter.phscale_positive = False
        guess = np.array(self.params)
        guess[1] = 0
        param, covar = self.fitter.fit(self.obs, self.x-center, guess, plot=True)
        # plt.clf()
        # plt.plot(self.x-center, self.obs, "r")
        plt.savefig("/tmp/test_gaussian2.pdf")
        self.assertAlmostEqual(param[0], self.params[0], 1) # FWHM
        self.assertAlmostEqual(param[1], 0, 1) # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1) # Amplitude

if __name__ == "__main__":
    unittest.main()
