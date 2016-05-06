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
        plt.savefig("/tmp/testfit_gaussian1.pdf")
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
        plt.savefig("/tmp/testfit_gaussian2.pdf")
        self.assertAlmostEqual(param[0], self.params[0], 1) # FWHM
        self.assertAlmostEqual(param[1], 0, 1) # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1) # Amplitude

class Test_MnKA(unittest.TestCase):
    def setUp(self):
        self.fitter = mass.calibration.line_fits.MnKAlphaFitter()
        self.distrib = mass.calibration.fluorescence_lines.MnKAlphaDistribution

    def do_test(self, n=50000, resolution=2.5, tailfrac=0, tailtau=17,
              bg = 10, nbins=150, vary_tail=False):
        bmin, bmax = 5875,5910

        values = self.distrib.rvs(size=n)
        sigma = resolution/2.3548
        values += sigma*np.random.standard_normal(size=n)

        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau
        obs,bins = np.histogram(values, nbins, [bmin, bmax])
        obs += np.random.poisson(size=nbins, lam=bg)

        params = np.array([resolution, 5898.8, 1.0, n, bg, 0, tailfrac, tailtau])
        twiddle = np.random.standard_normal(len(params))*[.05, .2, .001, n/1e3, 1,
                                                          0.001, .001, 0.1]
        if not vary_tail:
            twiddle[6:8] = 0.0
        guess = params + twiddle
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(obs, bins, guess, plot=True, axis=ax,
                                      vary_tail=vary_tail)
        plt.text(.05, .8, "Fit   : %s"%pfit, transform=ax.transAxes)
        plt.text(.05, .9, "Actual: %s"%params, transform=ax.transAxes)

    def test_basic(self):
        self.do_test()
        plt.savefig("/tmp/testfit_mnka1.pdf")

    def test_tail(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=1)
        plt.savefig("/tmp/testfit_mnka2.pdf")


class Test_MnKB(unittest.TestCase):
    def setUp(self):
        self.fitter = mass.calibration.line_fits.MnKBetaFitter()
        self.distrib = mass.calibration.fluorescence_lines.MnKBetaDistribution

    def do_test(self, n=50000, resolution=2.5, tailfrac=0, tailtau=17,
              bg = 10, nbins=150, vary_tail=False):
        bmin, bmax = 6460,6510

        values = self.distrib.rvs(size=n)
        sigma = resolution/2.3548
        values += sigma*np.random.standard_normal(size=n)

        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau
        obs,bins = np.histogram(values, nbins, [bmin, bmax])
        obs += np.random.poisson(size=nbins, lam=bg)

        params = np.array([resolution, 6490.5, 1.0, n, bg, 0, tailfrac, tailtau])
        twiddle = np.random.standard_normal(len(params))*[.0, .2, 0, n/1e3, 1,
                                                          0.001, .001, 0.1]
        if not vary_tail:
            twiddle[6:8] = 0.0
        guess = params + twiddle
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(obs, bins, guess, plot=True, axis=ax,
                                      hold=(0,2,), vary_tail=vary_tail)
        plt.text(.05, .8, "Fit   : %s"%pfit, transform=ax.transAxes)
        plt.text(.05, .9, "Actual: %s"%params, transform=ax.transAxes)

    def test_basic(self):
        self.do_test()
        plt.savefig("/tmp/testfit_mnkb1.pdf")

    def test_tail(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=1)
        plt.savefig("/tmp/testfit_mnkb2.pdf")

class Test_Voigt(unittest.TestCase):

    def setUp(self):
        self.fitter = mass.calibration.line_fits.VoigtFitter()

    def singletest(self, gauss_fwhm = 0.1, fwhm = 5, center = 100, ampl = 5000,
            bg = 200, nbins = 200, tailfrac = 1e-9, tailtau = 3,
            vary_resolution=False, vary_tail = False, hold=None):
        sigma = gauss_fwhm/2.3548

        params = [gauss_fwhm, center, fwhm, ampl, bg, 0, tailfrac, tailtau]
        throw = 3.5*fwhm+6*gauss_fwhm
        self.x = np.linspace(center-throw, center+throw, nbins)
        db = self.x[1]-self.x[0]
        bmin = self.x[0]-0.5*db
        bmax = self.x[-1]+0.5*db
        self.y = ampl/(1+((self.x-center)/(0.5*fwhm))**2)
        n = self.y.sum()
        values = np.random.standard_cauchy(size=n)*fwhm*0.5 + center
        values += sigma*np.random.standard_normal(size=n)
        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau

        self.obs,_ = np.histogram(values, nbins, [bmin, bmax])
        self.obs += np.random.poisson(size=nbins, lam=bg)
        twiddle = np.random.standard_normal(len(params))*0.03+1
        if hold is None:
            hold = []
        hold = list(hold)
        if not vary_resolution:
            hold.append(0)
        if not vary_tail:
            hold.extend([6,7])
        for h in hold:
            twiddle[h] = 1.0
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(self.obs, self.x, params*twiddle,
                               plot=True, vary_resolution=vary_resolution,
                               vary_tail=vary_tail, hold=hold, axis=ax)
        plt.text(.05, .9, "Actual: %s"%params, transform=ax.transAxes)
        plt.text(.05, .8, "Fit   : %s"%pfit, transform=ax.transAxes)
        return pfit, covar, params

    def test_fit(self):
        pfit, covar, params = self.singletest(ampl=20000, tailfrac = 0)
        plt.savefig("/tmp/testfit_voigt1.pdf")
        self.assertAlmostEqual(pfit[0], params[0], 1) # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1) # Center
        self.assertAlmostEqual(pfit[2], params[2], 0) # Lorentz FWHM

    def xxtest_fit_tail(self):
        pfit, covar, params = self.singletest(ampl=40000, tailfrac=0.20,
                                              tailtau=25, vary_tail=True)
        plt.savefig("/tmp/testfit_voigt2.pdf")
        self.assertAlmostEqual(pfit[0], params[0], 1) # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1) # Center
        self.assertAlmostEqual(pfit[2], params[2], 0) # Lorentz FWHM
        self.assertAlmostEqual(pfit[6], params[6], 1) # Tail frac
        self.assertAlmostEqual(pfit[7], params[7], -1) # Tail tau

    def test_fit_vary_res(self):
        pfit, covar, params = self.singletest(ampl=100000, gauss_fwhm=2, fwhm=2.5,
                                              bg=100, tailfrac=0, vary_resolution=True)
        plt.savefig("/tmp/testfit_voigt3.pdf")
        self.assertAlmostEqual(pfit[0], params[0], 0) # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1) # Center
        self.assertAlmostEqual(pfit[2], params[2], 0) # Lorentz FWHM



if __name__ == "__main__":
    unittest.main()
