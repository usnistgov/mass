'''
Test framework for MaximumLikelihoodHistogramFitter

Created on Jan 13, 2012

@author: fowlerj
'''
import pytest
from pytest import approx
import mass
import numpy as np


rng = np.random.default_rng(34234)


def weightavg(a, w):
    return (a*w).sum() / w.sum()


class Test_ratio_weighted_averages:
    """Run test with a known, constant histogram. Verify the weighted-average ratio=1 of
    maxmimum-likelihood fitters."""

    @pytest.fixture(autouse=True)
    def set_up_weighted_average_tests(self):
        self.nobs = np.array([
            0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,
            0,  0,  1,  1,  0,  2,  0,  2,  1,  1,  3,  2,  2,  6,  2,  5,  3,
            6,  5, 15, 17,  9, 18, 12,  9, 17, 17, 14, 11, 22, 28, 21, 16, 14,
            19, 16, 14, 24, 16, 7, 15,  8,  5, 15, 12, 13,  6,  8,  6,  6,  7,
            7,  4,  2,  3,  5,  2,  1,  1,  1,  1,  0,  0,  1,  2,  0,  0,  0,
            0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        self.x = np.linspace(-1.98, 1.98, 100)
        line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", 0.0, 0.0, 0.0)
        line.linetype = "Gaussian"
        self.model = line.model()
        self.params = self.model.guess(self.nobs, bin_centers=self.x)
        self.params["fwhm"].set(1.09)
        self.params["peak_ph"].set(0)
        self.params["integral"].set(self.nobs.sum())
        self.params["dph_de"].set(1, vary=False)
        self.params["background"].set(0, vary=False)
        self.params["bg_slope"].set(0, vary=False)

    def test_weighted_averages_nobg(self):
        results = self.model.fit(self.nobs, self.params, bin_centers=self.x)
        pfit = results.best_values
        assert pfit["fwhm"] == approx(1.09, 0.01)
        assert pfit["peak_ph"] == approx(0.0, abs=1e-6)
        assert pfit["integral"] == approx(self.nobs.sum(), rel=1e-4)
        y = results.best_fit
        ratio = self.nobs / y
        assert weightavg(ratio, y) == approx(1.0, abs=0.1)

    def test_weighted_averages_constbg(self):
        """Check that wt avg (data/model)=1 for weight = either one of {constant, model}
        when the model has a constant background.
        This property should be guaranteed when using a Maximum Likelihood fitter.
        """
        self.params["background"].set(0, vary=True)
        results = self.model.fit(self.nobs, self.params, bin_centers=self.x)
        y = results.best_fit
        ratio = self.nobs / y
        assert weightavg(ratio, y) == approx(1.0, abs=0.1)

    def test_weighted_averages_slopedbg(self):
        """Check that wt avg (data/model)=1 for weight = any one of {constant, model, x}
        when the model has a linear background.
        This property should be guaranteed when using a Maximum Likelihood fitter.
        """
        self.params["background"].set(0, vary=True)
        self.params["bg_slope"].set(0, vary=True)
        results = self.model.fit(self.nobs, self.params, bin_centers=self.x)
        y = results.best_fit
        ratio = self.nobs / y
        assert weightavg(ratio, y) == approx(1.0, abs=0.1)


class Test_gaussian:
    """Simulate some Gaussian data, fit the histograms, and make sure that the results are
    consistent with the expectation at the 2-sigma level.
    """

    def generate_data(self, N, fwhm=1.0, ctr=0.0, nbins=100, expected_bg=0):
        n_signal = rng.poisson(N)
        n_bg = rng.poisson(expected_bg)

        data = rng.standard_normal(size=n_signal)*fwhm/np.sqrt(8*np.log(2))
        if n_bg > 0:
            data = np.hstack((data, rng.uniform(size=n_bg)*4.0-2.0))
        nobs, _ = np.histogram(data, np.linspace(-2, 2, nbins+1))
        self.sum = nobs.sum()
        self.mean = (nobs*self.x).sum()/nobs.sum()
        self.var = (nobs*self.x**2).sum()/nobs.sum() - self.mean**2
        self.nobs = nobs

        self.true_params = (fwhm, self.mean, N/fwhm*4.0/nbins/1.06, expected_bg*1.0/nbins, 0, 0, 25)
        self.guess = np.array([1.09, 0, 17.7, 0.0, 0.0, 0, 25])

    def run_several_fits(self, N=1000, nfits=10, fwhm=1.0, ctr=0.0, nbins=100, N_bg=10):
        bin_edges = np.linspace(-2, 2, nbins+1)
        self.x = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

        for _ in range(nfits):
            self.generate_data(N, fwhm, ctr, nbins, N_bg)
            params = self.model.guess(self.nobs, bin_centers=self.x)
            params["fwhm"].set(1.09)
            params["peak_ph"].set(0)
            params["integral"].set(self.nobs.sum())
            params["dph_de"].set(1, vary=False)
            varybg = N_bg > 0
            params["background"].set(0, vary=varybg)
            params["bg_slope"].set(0, vary=False)
            results = self.model.fit(self.nobs, params, bin_centers=self.x)

            assert results.params["fwhm"] == approx(fwhm, rel=5/N**0.5)
            assert results.params["peak_ph"] == approx(ctr, abs=3*fwhm/N**0.5)
            assert results.params["integral"] == approx(N, abs=3*N**0.5)
            assert results.params["fwhm"].stderr < 2*fwhm/N**0.5
            assert results.params["peak_ph"].stderr < 2*fwhm/N**0.5
            assert results.params["integral"].stderr < 2*N**0.5
            if N_bg > 0:
                assert results.params["background"] == approx(N_bg/nbins, abs=10*N**0.5)
                assert results.params["background"].stderr < 10*N**0.5

    def test_gauss(self):
        "Run 30 fits apiece with 3 background levels; verify consistent parameters"
        line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", 0.0, 0.0, 0.0)
        line.linetype = "Gaussian"
        self.model = line.model()

        Nsignal = 1000
        fwhm = 1.0
        ctr = 0.0
        nbins = 100
        nfits = 30
        self.run_several_fits(Nsignal, nfits, fwhm, ctr, nbins, N_bg=100)
        self.run_several_fits(Nsignal, nfits, fwhm, ctr, nbins, N_bg=1)
        self.run_several_fits(Nsignal, nfits, fwhm, ctr, nbins, N_bg=0)


# class Test_fluorescence(unittest.TestCase):
#     """Simulate some fluorescence data, fit the histograms, and make sure that the results are
#     consistent with the expectation at the 2-sigma level.
#     """

#     def setUp(self):
#         self.rng = np.random.default_rng(121312)
#         self.fitter = mass.MnKAlphaFitter()
#         self.fitter._have_warned = True  # eliminate deprecation warnings

#     def generate_and_fit_data(self, N, fwhm=1.0, nbins=100, N_bg=0):
#         n_signal = self.rng.poisson(N)
#         n_bg = self.rng.poisson(N_bg)

#         distrib = mass.calibration.fluorescence_lines.MnKAlpha
#         data = distrib.rvs(size=n_signal, instrument_gaussian_fwhm=fwhm)
#         if N_bg > 0:
#             data = np.hstack((data, self.rng.uniform(size=n_bg)*4.0-2.0))
#         nobs, bin_edges = np.histogram(data, nbins, range=[5850, 5950])
#         bins = bin_edges[1:]-0.5*(bin_edges[1]-bin_edges[0])
#         params, covar = self.fitter.fit(nobs, bins, params=(
#             fwhm, 5898, 1, N, 1., 0, 0, 25), plot=False)

#         # Check uncertainties
#         d_res, d_ectr, d_scale = covar.diagonal()[:3]**0.5
#         expect_d_res = 5.0*fwhm/(N**0.5)
#         self.assertLessEqual(d_res, 2*expect_d_res,
#                              f"dres={d_res:.4f} not less than 2*expected ({expect_d_res:.4f})")
#         expect_d_ectr = 1.5*fwhm/(N**0.5)
#         self.assertLessEqual(d_ectr, 2*expect_d_ectr,
#                              f"dectr={d_ectr:.4f} not less than 2*expected ({expect_d_ectr:.4f})")
#         expect_d_scale = 1.0/(N**0.5)
#         self.assertLessEqual(d_scale, 2*expect_d_scale,
#                              f"dscale={d_scale:.4f} not less than 2*expected ({expect_d_scale:.4f})")

#         # Check data consistent with uncertainties
#         res, ectr, scale = params[:3]
#         msg = "fhwm: {:0.2f}, nbins {:g}, N_bg {:g}: Disagree at 4-sigma: Fit fwhm: {:.4f}  actual: " \
#             "{:.4f}; expect unc {:.4f}".format(fwhm, nbins, N_bg, res, fwhm, d_res)
#         self.assertLessEqual(abs(res-fwhm), 4*d_res, msg)
#         msg = "fhwm: {:0.2f}, nbins {:g}, N_bg {:g}: Disagree at 4-sigma: Fit Ectr: {:.4f}  actual: "\
#             "{:.4f}; expect unc {:.4f}".format(fwhm, nbins, N_bg, ectr, 5898.802, d_ectr)
#         self.assertLessEqual(abs(ectr-5898.802), 4*d_ectr, msg)
#         msg = "fhwm: {:0.2f}, nbins {:g}, N_bg {:g}: Disagree at 4-sigma: Fit scale: {:.4f}  actual: "\
#             {:.4f}; expect unc {:.4f}".format(fwhm, nbins, N_bg, scale, 1.0, d_scale)
#         self.assertLessEqual(abs(scale-1.0), 4*d_scale, msg)

#     def test_mn_k_alpha_no_background(self):
#         """Test that we can do Mn K-alpha fits without background"""
#         N_bg = 0
#         for _ in range(10):
#             fwhm = self.rng.uniform(1.8, 6.5, size=1)[0]
#             nbins = int(self.rng.uniform(50, 200, size=1)[0])
#             self.generate_and_fit_data(3000, fwhm, nbins, N_bg)

#     def test_mn_k_alpha_with_background(self):
#         """Test that we can do Mn K-alpha fits without background"""
#         for _ in range(10):
#             fwhm = self.rng.uniform(1.8, 6.5, size=1)[0]
#             nbins = int(self.rng.uniform(50, 200, size=1)[0])
#             N_bg = int(self.rng.uniform(0, 200, size=1)[0])
#             self.generate_and_fit_data(3000, fwhm, nbins, N_bg)


class Test_fit_kink:
    """Test the mass.mathstat.fitting.fit_kink_model() function."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        x = np.arange(10, dtype=float)
        y = np.array(x)
        truek = 4.6
        y[x > truek] = truek
        self.x = x
        self.y = y
        self.truek = truek

    def test_noisless_fit(self):
        """Make sure fit_kink_model gets very close to exact answer without noise."""
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, self.y, kbounds=(3, 6))
        assert X2 < 1e-8
        assert abs(kbest-self.truek) < 1e-5
        assert abs(a-self.truek) < 1e-5
        assert abs(b-1) < 1e-5
        assert abs(c) < 1e-5

    def test_noisless_fit_no_bounds(self):
        """Make sure fit_kink_model gets very close to exact answer without noise and
        using maximal bounds."""
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, self.y, kbounds=None)
        assert X2 < 1e-8
        assert abs(kbest-self.truek) < 1e-5
        assert abs(a-self.truek) < 1e-5
        assert abs(b-1) < 1e-5
        assert abs(c) < 1e-5

    def test_noisy_fit(self):
        """Make sure fit_kink_model gets close enough to exact answer with noise."""
        rng = np.random.default_rng(9090)
        noisy_y = self.y + rng.standard_normal(len(self.x))*.2
        _, (kbest, a, b, c), X2 = mass.mathstat.fitting.fit_kink_model(
            self.x, noisy_y, kbounds=(3, 6))
        assert X2 < 1.0
        assert abs(kbest-self.truek) < 0.3
        assert abs(a-self.truek) < 0.3
        assert abs(b-1) < 0.1
        assert abs(c) < 0.1


def Test_Issue_125():
    """Test that issue 125 is fixed. The following fit used to take infinte time/memory.
    If this returns, then consider that a passing test."""

    e = np.linspace(5870.25, 5909.25, 80)
    contents = np.array([
        36,  49,  39,  41,  46,  46,  42,  42,  46,  52,  51,  53,  54,  48,  58,  46,  57,  51,
        61,   68,  63,  68,  73,  79,  66,  84,  78,  94,  84,  84,  74,  85,  76,  81,  83,  84,
        100,  95,  93,  82,  74,  83,  93, 102,  98,  79, 100, 113,  95,  88, 104,  94,  95, 110,
        112,  81, 106, 104, 110,  97, 105,  95, 103,  97,  95, 103,  84,  97,  79,  85,  84,  87,
        80,   71,  77,  89,  83,  81,  59])
    line = mass.MnKAlpha
    model = line.model()
    params = model.guess(contents, bin_centers=e)
    params["fwhm"].set(20)
    params["peak_ph"].set(5898)
    params["dph_de"].set(1.0)
    params["background"].set(20.0)
    _ = model.fit(contents, params, bin_centers=e)

    model = line.model(has_tails=True)
    params = model.guess(contents, bin_centers=e)
    params["fwhm"].set(20)
    params["peak_ph"].set(5898)
    params["dph_de"].set(1.0, vary=False)
    params["background"].set(20.0)
    params["tail_frac"].set(.1)
    _ = model.fit(contents, params, bin_centers=e)
