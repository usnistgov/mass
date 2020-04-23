"""
test_fits.py

Test that Maximum Likelihood Fits work

5 May 2016
Joe Fowler
"""

import tempfile
import os.path
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
        self.x = np.linspace(10, 20, 200)
        self.y = ampl * np.exp(-0.5*(self.x-center)**2/(sigma**2))
        np.random.seed(94)
        self.obs = np.array([np.random.poisson(lam=y0) for y0 in self.y])
        self.fitter = mass.calibration.GaussianFitter()

    def test_failed_fit(self):
        # pass nans
        param, covar = self.fitter.fit(
            np.array([np.nan]*len(self.obs)), self.x, self.params, plot=True)
        self.assertFalse(self.fitter.fit_success)
        self.assertTrue(np.isnan(self.fitter.last_fit_params[0]))
        self.assertTrue(np.isnan(self.fitter.last_fit_params_dict["peak_ph"][0]))
        self.assertTrue(np.isnan(self.fitter.last_fit_params_dict["peak_ph"][1]))
        self.assertTrue(isinstance(self.fitter.failed_fit_exception, Exception))
        self.assertTrue(all([self.params[i] == self.fitter.failed_fit_params[i]
                             for i in range(len(self.params))]))

    def test_fit(self):
        self.fitter.phscale_positive = True
        param, covar = self.fitter.fit(self.obs, self.x, self.params, plot=True)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_gaussian1.pdf"))
        self.assertAlmostEqual(param[0], self.params[0], 1)  # FWHM
        self.assertAlmostEqual(param[1], self.params[1], 1)  # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1)  # Amplitude
        self.assertAlmostEqual(param[0], self.fitter.last_fit_params_dict["resolution"][0], 1)
        self.assertAlmostEqual(param[1], self.fitter.last_fit_params_dict["peak_ph"][0], 1)
        self.assertAlmostEqual(param[2], self.fitter.last_fit_params_dict["amplitude"][0], 1)
        self.assertTrue(self.fitter.fit_success)

    def test_fit_offset(self):
        center = self.params[1]
        self.fitter.phscale_positive = False
        guess = np.array(self.params)
        guess[1] = 0
        param, covar = self.fitter.fit(self.obs, self.x-center, guess, plot=True)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_gaussian2.pdf"))
        self.assertAlmostEqual(param[0], self.params[0], 1)  # FWHM
        self.assertAlmostEqual(param[1], 0, 1)  # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1)  # Amplitude
        self.assertTrue(self.fitter.fit_success)

    def test_fit_zero_bg(self):
        param_guess = self.params[:]
        param_guess[self.fitter.param_meaning["background"]] = 0
        self.fitter.phscale_positive = True
        param, covar = self.fitter.fit(self.obs, self.x, param_guess, plot=True)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_gaussian1.pdf"))
        self.assertAlmostEqual(param[0], self.params[0], 1)  # FWHM
        self.assertAlmostEqual(param[1], self.params[1], 1)  # Center
        self.assertAlmostEqual(param[2]/self.params[2], 1, 1)  # Amplitude

    def test_negative_background_issue126(self):
        """This fit gives negative BG in all bins before the fix of issue #126."""
        obs = np.exp(-0.5*(self.x-self.params[1])**2/(self.params[0]/2.3548)**2) + 0
        param, covar = self.fitter.fit(obs, self.x, self.params, plot=True)
        bg_bin0 = self.fitter.last_fit_params[-4]
        bg_binEnd = bg_bin0 + (len(self.x)-1)*self.fitter.last_fit_params[-3]
        self.assertTrue(bg_bin0 >= 0)
        self.assertTrue(bg_binEnd >= 0)

    def test_wide_bins_issue162(self):
        """Does Gaussian fit give unbiased width when bin width = Gaussian width?

        Fit where bins and Gaussian have approximately equal width is biased. It
        used to overestimate the Gaussian's width, as shown in issue 162.
        Test that it's been fixed."""
        for fwhm in [0.7, 1.0, 1.5]:
            sigma = fwhm/2.3548
            nsim = 30
            N = 10000
            w = np.zeros(nsim, dtype=float)
            for i in range(nsim):
                x = np.random.standard_normal(N)*sigma
                c, b = np.histogram(x, 100, [-50, 50])
                param, covar = self.fitter.fit(c, b, [fwhm, 0, c.max(), 0, 0, 0, 25],
                                               plot=False, hold=(3, 4, 5, 6))
                w[i] = param[0]
            typical_width = mass.robust.trimean(w)
            self.assertLess(typical_width/fwhm, 1.05)  # was typically ~1.18 before fix

    def test_numerical_integration(self):
        """Test that the integrate_n_points argument works as expected."""
        fwhm = 1.0
        sigma = fwhm/2.3548
        nsim = 30
        N = 10000
        for npoints in (1, 3, 5, 7):
            w = np.zeros(nsim, dtype=float)
            for i in range(nsim):
                x = np.random.standard_normal(N)*sigma
                c, b = np.histogram(x, 100, [-50, 50])
                param, covar = self.fitter.fit(c, b, [fwhm, 0, c.max(), 0, 0, 0, 25],
                                               plot=False, hold=(3, 4, 5, 6), integrate_n_points=npoints)
                w[i] = param[0]
            typical_width = mass.robust.trimean(w)
            if npoints > 1:
                self.assertLess(typical_width/fwhm, 1.05)
            self.assertAlmostEqual(typical_width/fwhm, 1.0, delta=1.0)

        for npoints in (-5, 0, 2):
            with self.assertRaises(ValueError):
                self.fitter.fit(c, b, [fwhm, 0, c.max(), 0, 0, 0, 25], rethrow=True,
                                plot=False, hold=(3, 4, 5, 6), integrate_n_points=npoints)


class Test_MnKA(unittest.TestCase):
    def setUp(self):
        self.fitter = mass.calibration.MnKAlphaFitter()
        self.distrib = mass.calibration.fluorescence_lines.MnKAlpha
        self.tempdir = tempfile.gettempdir()
        mass.logging.log(mass.logging.INFO, "K-alpha fits stored to %s" % self.tempdir)
        np.random.seed(96)

    def do_test(self, n=50000, resolution=2.5, tailfrac=0, tailtau=17, bg=10,
                nbins=150, vary_bg_slope=False, vary_tail=False):
        bmin, bmax = 5875, 5910

        values = self.distrib.rvs(size=n, instrument_gaussian_fwhm=0)
        sigma = resolution/2.3548
        values += sigma*np.random.standard_normal(size=n)

        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau
        obs, bins = np.histogram(values, nbins, [bmin, bmax])
        obs += np.random.poisson(size=nbins, lam=bg)

        params = np.array([resolution, 5898.8, 1.0, n, bg, 0, tailfrac, tailtau])
        twiddle = np.random.standard_normal(len(params))*[.05, .2, .001, n/1e3, 1,
                                                          0.001, .001, 0.1]
        if not vary_bg_slope:
            twiddle[-4] = abs(twiddle[-4])  # non-negative BG guess
            twiddle[-3] = 0.0
        if not vary_tail:
            twiddle[-2:] = 0.0
        guess = params + twiddle
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(obs, bins, guess, plot=True, axis=ax,
                                      vary_bg_slope=vary_bg_slope, vary_tail=vary_tail)
        plt.text(.05, .76, "Actual: %s" % params, transform=ax.transAxes)
        plt.text(.05, .66, "Fit   : %s" % pfit, transform=ax.transAxes)
        self.assertTrue(self.fitter.fit_success)

    def test_basic(self):
        self.do_test()
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_mnka1.pdf"))

    def test_tail(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=1)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_mnka2.pdf"))

    def test_bg_slope(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=False, vary_bg_slope=True)
        plt.savefig(os.path.join(self.tempdir, "testfit_mnkb3.pdf"))

    def test_zero_bg(self):
        self.do_test(bg=0)

    def test_plot_and_result_string(self):
        self.do_test()
        self.fitter.plot(label="full", ph_units="arb", color="r")
        self.assertIsNotNone(self.fitter.result_string)


class Test_MnKB(unittest.TestCase):
    def setUp(self):
        self.fitter = mass.calibration.MnKBetaFitter()
        self.distrib = mass.calibration.fluorescence_lines.MnKBeta
        np.random.seed(97)
        self.tempdir = tempfile.gettempdir()
        mass.logging.log(mass.logging.INFO, "K-beta fits stored to %s" % self.tempdir)

    def do_test(self, n=50000, resolution=2.5, tailfrac=0, tailtau=17,
                bg=10, nbins=150, vary_bg_slope=False, vary_tail=False):
        bmin, bmax = 6460, 6510

        values = self.distrib.rvs(size=n, instrument_gaussian_fwhm=0)
        sigma = resolution/2.3548
        values += sigma*np.random.standard_normal(size=n)

        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau
        obs, bins = np.histogram(values, nbins, [bmin, bmax])
        obs += np.random.poisson(size=nbins, lam=bg)

        params = np.array([resolution, 6490.5, 1.0, n, bg, 0, tailfrac, tailtau])
        twiddle = np.random.standard_normal(len(params))*[.0, .2, 0, n/1e3, 1,
                                                          0.001, .001, 0.1]
        if not vary_bg_slope:
            twiddle[-4] = abs(twiddle[-4])  # non-negative BG guess
            twiddle[-3] = 0.0
        if not vary_tail:
            twiddle[-2:] = 0.0
        guess = params + twiddle
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(obs, bins, guess, plot=True, axis=ax,
                                      hold=(0, 2,), vary_tail=vary_tail,
                                      vary_bg_slope=vary_bg_slope)
        plt.text(.05, .76, "Actual: %s" % params, transform=ax.transAxes)
        plt.text(.05, .66, "Fit   : %s" % pfit, transform=ax.transAxes)

    def test_basic(self):
        self.do_test()
        plt.savefig(os.path.join(self.tempdir, "testfit_mnkb1.pdf"))

    def test_tail(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=True)
        plt.savefig(os.path.join(self.tempdir, "testfit_mnkb2.pdf"))

    def test_bg_slope(self):
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=False, vary_bg_slope=True)
        plt.savefig(os.path.join(self.tempdir, "testfit_mnkb3.pdf"))

    def test_zero_bg(self):
        self.do_test(bg=0)

    def test_plot_and_result_string(self):
        self.do_test()
        self.fitter.plot(label="full", ph_units="arb", color="r")
        self.assertIsNotNone(self.fitter.result_string)


class Test_Voigt(unittest.TestCase):

    def setUp(self):
        self.fitter = mass.calibration.VoigtFitter()

    def singletest(self, gauss_fwhm=0.1, fwhm=5, center=100, ampl=5000,
                   bg=200, nbins=200, tailfrac=1e-9, tailtau=3,
                   vary_resolution=False, vary_tail=False, hold=None):
        sigma = gauss_fwhm/2.3548

        params = [gauss_fwhm, center, fwhm, ampl, bg, 0, tailfrac, tailtau]
        throw = 3.5*fwhm+6*gauss_fwhm
        self.x = np.linspace(center-throw, center+throw, nbins)
        db = self.x[1]-self.x[0]
        bmin = self.x[0]-0.5*db
        bmax = self.x[-1]+0.5*db
        self.y = ampl/(1+((self.x-center)/(0.5*fwhm))**2)
        n = int(self.y.sum())
        values = np.random.standard_cauchy(size=n)*fwhm*0.5 + center
        values += sigma*np.random.standard_normal(size=n)
        tweak = np.random.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= np.random.standard_exponential(size=ntweak)*tailtau

        self.obs, _ = np.histogram(values, nbins, [bmin, bmax])
        self.obs += np.random.poisson(size=nbins, lam=bg)
        twiddle = np.random.standard_normal(len(params))*0.03+1
        if hold is None:
            hold = []
        hold = list(hold)
        if not vary_resolution:
            hold.append(0)
        if not vary_tail:
            hold.extend([6, 7])
        for h in hold:
            twiddle[h] = 1.0
        plt.clf()
        ax = plt.subplot(111)
        pfit, covar = self.fitter.fit(self.obs, self.x, params*twiddle,
                                      plot=True, vary_resolution=vary_resolution,
                                      vary_tail=vary_tail, hold=hold, axis=ax)
        plt.text(.05, .76, "Actual: %s" % params, transform=ax.transAxes)
        plt.text(.05, .66, "Fit   : %s" % pfit, transform=ax.transAxes)
        return pfit, covar, params

    def test_fit(self):
        pfit, covar, params = self.singletest(ampl=20000, tailfrac=0)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_voigt1.pdf"))
        self.assertAlmostEqual(pfit[0], params[0], 1)  # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1)  # Center
        self.assertAlmostEqual(pfit[2], params[2], 0)  # Lorentz FWHM

    def xxtest_fit_tail(self):
        pfit, covar, params = self.singletest(ampl=40000, tailfrac=0.20,
                                              tailtau=25, vary_tail=True)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_voigt2.pdf"))
        self.assertAlmostEqual(pfit[0], params[0], 1)  # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1)  # Center
        self.assertAlmostEqual(pfit[2], params[2], 0)  # Lorentz FWHM
        self.assertAlmostEqual(pfit[6], params[6], 1)  # Tail frac
        self.assertAlmostEqual(pfit[7], params[7], -1)  # Tail tau

    def test_fit_vary_res(self):
        pfit, covar, params = self.singletest(ampl=100000, gauss_fwhm=2, fwhm=2.5,
                                              bg=100, tailfrac=0, vary_resolution=True)
        plt.savefig(os.path.join(tempfile.gettempdir(), "testfit_voigt3.pdf"))
        self.assertAlmostEqual(pfit[0], params[0], 0)  # Gauss FWHM
        self.assertAlmostEqual(pfit[1], params[1], 1)  # Center
        self.assertAlmostEqual(pfit[2], params[2], 0)  # Lorentz FWHM

    def test_zero_bg(self):
        self.singletest(bg=0)


class Test_MnKA_lmfit(unittest.TestCase):
    def test_lmfit(self):
        n = 10000
        resolution = 2.5
        bin_edges = np.arange(5850, 5950, 0.5)
        # generate random x-ray pulse energies following MnKAlpha distribution
        line = mass.calibration.fluorescence_lines.MnKAlpha
        np.random.seed(154)
        values = line.rvs(size=n, instrument_gaussian_fwhm=0)
        # add gaussian oise to each x-ray energy
        sigma = resolution/2.3548
        values += sigma*np.random.standard_normal(size=n)
        # histogram
        counts, _ = np.histogram(values, bin_edges)
        model = line.model()
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        params = model.guess(counts, bin_centers=bin_centers)
        result = model.fit(counts, bin_centers=bin_centers, params=params)
        fitter = mass.MnKAlphaFitter()
        fitter.fit(counts, bin_centers, plot=False)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["resolution"][0], result.params["fwhm"].value, delta=2*result.params["fwhm"].stderr)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["resolution"][1], result.params["fwhm"].stderr, places=1)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["amplitude"][0], result.params["amplitude"].value, delta=2*result.params["amplitude"].stderr)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["amplitude"][1], result.params["amplitude"].stderr, places=-3)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["bg_slope"][0], result.params["bg_slope"].value, delta=2*result.params["bg_slope"].stderr)
        self.assertAlmostEqual(
            fitter.last_fit_params_dict["bg_slope"][1], result.params["bg_slope"].stderr, places=1)

class Test_Composites_lmfit(unittest.TestCase):
    def setUp(self):
        if 'dummy1' not in mass.spectrum_classes.keys():
            mass.calibration.fluorescence_lines.addline(
                element="dummy",
                material="dummy_material",
                linetype="1",
                reference_short='NIST ASD',
                fitter_type=mass.line_fits.GenericKBetaFitter,
                reference_plot_instrument_gaussian_fwhm=0.5,
                nominal_peak_energy=653.493657,
                energies=np.array([653.493657]), lorentzian_fwhm=np.array([0.1]),
                reference_amplitude=np.array([1]),
                reference_amplitude_type=mass.LORENTZIAN_PEAK_HEIGHT, ka12_energy_diff=None
            )
        if 'dummy2' not in mass.spectrum_classes.keys():
            mass.calibration.fluorescence_lines.addline(
                element="dummy",
                material="dummy_material",
                linetype="2",
                reference_short='NIST ASD',
                fitter_type=mass.line_fits.GenericKBetaFitter,
                reference_plot_instrument_gaussian_fwhm=0.5,
                nominal_peak_energy=653.679946,
                energies=np.array([653.679946]), lorentzian_fwhm=np.array([0.1]),
                reference_amplitude=np.array([1]),
                reference_amplitude_type=mass.LORENTZIAN_PEAK_HEIGHT, ka12_energy_diff=None
            )
        np.random.seed(131)
        bin_edges = np.arange(600, 700, 0.4)
        resolution = 4.0
        sigma = resolution/2.3548
        n1 = 10000
        n2 = 20000
        self.n = n1+n2
        self.line1 = mass.spectrum_classes['dummy1']()
        self.line2 = mass.spectrum_classes['dummy2']()
        self.nominal_separation = self.line2.nominal_peak_energy - self.line1.nominal_peak_energy
        values1 = self.line1.rvs(size=n1, instrument_gaussian_fwhm=resolution)
        values2 = self.line2.rvs(size=n2, instrument_gaussian_fwhm=resolution)
        self.counts1, _ = np.histogram(values1, bin_edges)
        self.counts2, _ = np.histogram(values2, bin_edges)
        self.counts = self.counts1 + self.counts2
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
    def test_FitToModelWithoutPrefix(self):
        model1_noprefix = mass.fluorescence_lines.make_line_model(self.line1)
        assert(model1_noprefix.prefix == '')
        params1_noprefix = model1_noprefix.guess(self.counts1, bin_centers=self.bin_centers)
        result1_noprefix = model1_noprefix.fit(self.counts1, params=params1_noprefix, bin_centers=self.bin_centers)
        for iComp in result1_noprefix.components:
            assert(iComp.prefix == '')

    def test_NonUniqueParamsFails(self):
        model1_noprefix = mass.fluorescence_lines.make_line_model(self.line1)
        model2_noprefix = mass.fluorescence_lines.make_line_model(self.line2)
        with self.assertRaises(NameError):
            composite_model = model1_noprefix + model2_noprefix
    
    def test_CompositeModelFit(self):
        prefix1 = 'p1_'
        prefix2 = 'p2_'
        model1 = mass.fluorescence_lines.make_line_model(self.line1, prefix=prefix1)
        model2 = mass.fluorescence_lines.make_line_model(self.line2, prefix=prefix2)
        assert(model1.prefix == prefix1)
        assert(model2.prefix == prefix2)
        params1 = model1.guess(self.counts1, bin_centers=self.bin_centers)
        params2 = model2.guess(self.counts2, bin_centers=self.bin_centers)
        params1['{}dph_de'.format(prefix1)].set(value=1.0, vary=False)
        params2['{}dph_de'.format(prefix2)].set(value=1.0, vary=False)
        result1 = model1.fit(self.counts1, params=params1, bin_centers=self.bin_centers)
        result2 = model2.fit(self.counts2, params=params2, bin_centers=self.bin_centers)
        compositeModel = model1 + model2
        modelComponentPrefixes = [iComp.prefix for iComp in compositeModel.components]
        assert(np.logical_and(prefix1 in modelComponentPrefixes, prefix2 in modelComponentPrefixes))
        compositeParams = result1.params + result2.params
        compositeParams['{}fwhm'.format(prefix1)].expr = '{}fwhm'.format(prefix2)
        compositeParams['{}peak_ph'.format(prefix1)].expr = '{}peak_ph - {}'.format(prefix2, self.nominal_separation)
        compositeParams.add(name='ampRatio', value = 0.5, vary=False)
        compositeParams['{}amplitude'.format(prefix1)].expr='{}amplitude * ampRatio'.format(prefix2)
        compositeResult = compositeModel.fit(self.counts, params=compositeParams, bin_centers=self.bin_centers)
        resultComponentPrefixes = [iComp.prefix for iComp in compositeResult.components]
        assert(np.logical_and(prefix1 in resultComponentPrefixes, prefix2 in resultComponentPrefixes))
        compositeResult._validate_bins_per_fwhm(minimum_bins_per_fwhm=3)

if __name__ == "__main__":
    unittest.main()
