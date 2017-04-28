'''
Test framework for MaximumLikelihoodGaussianFitter and MaximumLikelihoodHistogramFitter

Created on Jan 13, 2012

@author: fowlerj
'''
import unittest
import mass
import numpy as np


def weightavg(a, w):
    return (a*w).sum() / w.sum()


class Test_ratio_weighted_averages(unittest.TestCase):
    """Run test with a known, constant histogram"""

    def setUp(self):
        self.nobs = np.array([
            0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,
            0,  0,  1,  1,  0,  2,  0,  2,  1,  1,  3,  2,  2,  6,  2,  5,  3,
            6,  5, 15, 17,  9, 18, 12,  9, 17, 17, 14, 11, 22, 28, 21, 16, 14,
            19, 16, 14, 24, 16, 7, 15,  8,  5, 15, 12, 13,  6,  8,  6,  6,  7,
            7,  4,  2,  3,  5,  2,  1,  1,  1,  1,  0,  0,  1,  2,  0,  0,  0,
            0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

        self.x = np.linspace(-1.98, 1.98, 100)
        self.fitter = mass.calibration.line_fits.GaussianFitter()
        self.guess = np.array([1.09, 0, 17.7, 0.0, 0.0, 0, 25])

    def test_weighted_averages_nobg(self):
        """Check that wt avg (data/model)==1 for weight = model when the model has no
        background. This property should be guaranteed when using a Maximum-Likelihood fitter.
        """
        self.guess[3:] = 0
        params, _ = self.fitter.fit(self.nobs, self.x, self.guess, hold=(3, 4, 5, 6), plot=False)

        self.assertAlmostEqual(params[0], 1.09, 2)
        self.assertAlmostEqual(params[1], 0, 2)
        self.assertAlmostEqual(params[2], 17.7115968, 1)

        y = self.fitter.last_fit_result
        ratio = self.nobs/y
        self.assertAlmostEqual(weightavg(ratio, y), 1.0, 1)

    def test_weighted_averages_constbg(self):
        """Check that wt avg (data/model)=1 for weight = either one of {constant, model}
        when the model has a constant background.
        This property should be guaranteed when using a Maximum Likelihood fitter.
        """
        self.guess[3:] = 0
        self.fitter.fit(self.nobs, self.x, self.guess, hold=(4, 5, 6), plot=False)
        y = self.fitter.last_fit_result
        ratio = self.nobs/y
        self.assertAlmostEqual(weightavg(ratio, y), 1.0, 1)

    def test_weighted_averages_slopedbg(self):
        """Check that wt avg (data/model)=1 for weight = any one of {constant, model, x}
        when the model has a linear background.
        This property should be guaranteed when using a Maximum Likelihood fitter.
        """
        self.guess[3:] = 0
        params, _ = self.fitter.fit(self.nobs, self.x, self.guess, hold=(5, 6), plot=False)
        params[3] = 0
        params, _ = self.fitter.fit(self.nobs, self.x, params, hold=(5, 6), plot=False)
        y = self.fitter.last_fit_result
        ratio = self.nobs/y
        self.assertAlmostEqual(weightavg(ratio, y), 1.0, 1)


class Test_gaussian(unittest.TestCase):
    """Simulate some Gaussian data, fit the histograms, and make sure that the results are
    consistent with the expectation at the 2-sigma level.
    """
    def setUp(self):
        np.random.seed(12348)

    def generate_data(self, N, fwhm=1.0, ctr=0.0, nbins=100, N_bg=0):
        self.x = x = np.arange(.5, nbins)*4.0/nbins-2.0

        n_signal = np.random.poisson(N)
        n_bg = np.random.poisson(N_bg)

        data = np.random.standard_normal(size=n_signal)*fwhm/np.sqrt(8*np.log(2))
        if N_bg > 0:
            data = np.hstack((data, np.random.uniform(size=n_bg)*4.0-2.0))
        nobs, _bins = np.histogram(data, np.linspace(-2, 2, nbins+1))
        self.sum = nobs.sum()
        self.mean = (x*nobs).sum()/nobs.sum()
        self.var = (x*x*nobs).sum()/nobs.sum() - self.mean**2
        self.nobs = nobs

        self.true_params = (fwhm, self.mean, N/fwhm*4.0/nbins/1.06, N_bg*1.0/nbins, 0, 0, 25)
        self.guess = np.array([1.09, 0, 17.7, 0.0, 0.0, 0, 25])
        self.hold = [4, 5, 6]
        if N_bg <= 0:
            self.hold.append(3)

    def run_several_fits(self, N=1000, nfits=10, fwhm=1.0, ctr=0.0, nbins=100, N_bg=10, penalty=None):
        self.fitter = mass.calibration.line_fits.GaussianFitter()
        self.fitter.phscale_positive = False
        self.fitter.set_penalty(penalty)
        correct_params = (fwhm, ctr, .037535932*N, 0, 0, 0, 25)
        sigma_errors = np.zeros((7, nfits), dtype=np.float)
        params = np.zeros((7, nfits), dtype=np.float)
        for i in range(nfits):
            self.generate_data(N, fwhm, ctr, nbins, N_bg)
            try:
                p, covar = self.fitter.fit(self.nobs, self.x, self.true_params,
                                           hold=self.hold, plot=False)
                params[:, i] = p

            except np.linalg.LinAlgError:
                continue

            invcovar = np.array(covar.diagonal())
            invcovar[invcovar <= 0.0] = 1.0
            invcovar = 1.0/invcovar
            sigma_errors[:, i] = (p-correct_params)*(invcovar**0.5)

        sigma_errors[4, :] = 0
        maxparam = 4
        if N_bg <= 0:
            sigma_errors[3, :] = 0
            maxparam = 3
        self.assertTrue(np.all(sigma_errors[:maxparam].std(axis=1) < 1+2/nfits**0.5))
        return params

    def test_30fits_with_bg(self):
        "Run 30 fits with nonzero background and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 30
        N_bg = 100
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, N_bg)

    def test_30fits_small_bg(self):
        "Run 30 fits with one background event and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 30
        N_bg = 1
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, N_bg)

    def test_fits_zero_bg(self):
        "Run 50 fits with one background event and verify consistent parameters"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 50
        N_bg = 0
        self.run_several_fits(1000, nfits, fwhm, ctr, nbins, N_bg)

    def test_penalty_noeffect(self):
        "Test the regularization penalty in case where it has no effect"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 50
        N_bg = 0
        penalty = SimplePenalty(10, 0.2, 10)
        params = self.run_several_fits(1000, nfits, fwhm, ctr, nbins, N_bg, penalty=penalty)
        # Be sure that the center is -.05 < c < +.05
        self.assertTrue(abs(params[1, :].mean()) < 0.05)

    def test_penalty_haseffect(self):
        "Test the regularization penalty in case where it has an effect"
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 40
        N_bg = 0
        penalty = SimplePenalty(-1, 0.3, 30)
        params = self.run_several_fits(1000, nfits, fwhm, ctr, nbins, N_bg, penalty=penalty)
        # Be sure that the center is c < -.05, because of the penalty
        self.assertTrue(params[1, :].mean() < -0.05)


class SimplePenalty(object):
    """Function object to penalize param[1] exceeding pmax."""
    def __init__(self, pmax, pscale, amplitude):
        self.pmax = pmax
        self.pscale = pscale
        self.amplitude = amplitude

    def __call__(self, param):
        Npar = len(param)
        grad = np.zeros(Npar, dtype=float)
        hess = np.zeros((Npar, Npar), dtype=float)
        p = param[1]
        if p < self.pmax:
            return 0.0, grad, hess

        ae = self.amplitude*np.exp((p-self.pmax)/self.pscale)
        B = ae-self.amplitude
        grad[1] = ae/self.pscale
        hess[1, 1] = ae/self.pscale**2
        return B, grad, hess


class Test_fluorescence(unittest.TestCase):
    """Simulate some fluorescence data, fit the histograms, and make sure that the results are
    consistent with the expectation at the 2-sigma level.
    """

    def setUp(self):
        np.random.seed(12349)
        self.fitter = mass.MnKAlphaFitter()

    def generate_and_fit_data(self, N, fwhm=1.0, nbins=100, N_bg=0):
        n_signal = np.random.poisson(N)
        n_bg = np.random.poisson(N_bg)

        distrib = mass.calibration.fluorescence_lines.MnKAlphaDistribution()
        distrib.distribution.set_gauss_fwhm(fwhm)
        data = distrib.rvs(size=n_signal)
        if N_bg > 0:
            data = np.hstack((data, np.random.uniform(size=n_bg)*4.0-2.0))
        nobs, bin_edges = np.histogram(data, nbins, range=[5850, 5950])
        bins = bin_edges[1:]-0.5*(bin_edges[1]-bin_edges[0])
        params, covar = self.fitter.fit(nobs, bins, params=(fwhm, 5898, 1, N, 1., 0, 0, 25), plot=False)

        # Check uncertainties
        d_res, d_ectr, d_scale = covar.diagonal()[:3]**0.5
        expect_d_res = 5.0*fwhm/(N**0.5)
        self.assertLessEqual(d_res, 2*expect_d_res,
                             "dres=%.4f not less than 2*expected (%.4f)" % (d_res, expect_d_res))
        expect_d_ectr = 1.5*fwhm/(N**0.5)
        self.assertLessEqual(d_ectr, 2*expect_d_ectr,
                             "dectr=%.4f not less than 2*expected (%.4f)" % (d_ectr, expect_d_ectr))
        expect_d_scale = 1.0/(N**0.5)
        self.assertLessEqual(d_scale, 2*expect_d_scale,
                             "dscale=%.4f not less than 2*expected (%.4f)" % (d_scale, expect_d_scale))

        # Check data consistent with uncertainties
        res, ectr, scale = params[:3]
        msg = "Disagree at 4-sigma: Fit fwhm: %.4f  actual: %.4f; expect unc %.4f" % (res, fwhm, d_res)
        self.assertLessEqual(abs(res-fwhm), 4*d_res, msg)
        msg = "Disagree at 4-sigma: Fit Ectr: %.4f  actual: %.4f; expect unc %.4f" % (ectr, 5898.802, d_ectr)
        self.assertLessEqual(abs(ectr-5898.802), 4*d_ectr, msg)
        msg = "Disagree at 4-sigma: Fit scale: %.4f  actual: %.4f; expect unc %.4f" % (scale, 1.0, d_scale)
        self.assertLessEqual(abs(scale-1.0), 4*d_scale, msg)

    def test_mn_k_alpha_no_background(self):
        """Test that we can do Mn K-alpha fits without background"""
        N_bg = 0
        for _ in range(10):
            fwhm = np.random.uniform(1.8, 6.5, size=1)[0]
            nbins = int(np.random.uniform(50, 200, size=1)[0])
            self.generate_and_fit_data(3000, fwhm, nbins, N_bg)

    def test_mn_k_alpha_with_background(self):
        """Test that we can do Mn K-alpha fits without background"""
        for _ in range(10):
            fwhm = np.random.uniform(1.8, 6.5, size=1)[0]
            nbins = int(np.random.uniform(50, 200, size=1)[0])
            N_bg = int(np.random.uniform(0, 200, size=1)[0])
            self.generate_and_fit_data(3000, fwhm, nbins, N_bg)


if __name__ == "__main__":
    unittest.main()
