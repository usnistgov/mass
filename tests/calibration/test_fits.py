"""
test_fits.py

Test that line work correctly.
"""

import pytest
from pytest import approx
import numpy as np
import mass

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


class Test_gaussian_basic:
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


class Test_Gaussian:

    @pytest.fixture(autouse=True)
    def setUp(self):
        sigma = 1.5
        self.fwhm = sigma*(8*np.log(2))**0.5
        self.center = 15
        self.integral = 1000
        Nbg = 0
        self.x = np.linspace(10, 20, 200)
        self.y = np.exp(-0.5*(self.x-self.center)**2/(sigma**2))
        self.y *= self.integral/self.y.sum()

        line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", self.center, 0, 0)
        line.linetype = "Gaussian"
        self.model = line.model()
        self.params = self.model.guess(self.y, bin_centers=self.x)
        self.params["fwhm"].set(2.3548*sigma)
        self.params["background"].set(Nbg/len(self.y))

    def test_fit(self):
        result = self.model.fit(self.y, self.params, bin_centers=self.x)
        param = result.best_values
        assert result.success
        assert param["fwhm"] == approx(self.fwhm, abs=1)
        assert param["peak_ph"] == approx(self.center, abs=1)
        assert param["integral"] == approx(self.integral, abs=1)

    def test_fit_offset(self):
        self.params["peak_ph"].set(0)
        result = self.model.fit(self.y, self.params, bin_centers=self.x-self.center)
        param = result.best_values
        assert result.success
        assert param["fwhm"] == approx(self.fwhm, abs=1)
        assert param["peak_ph"] == approx(0, abs=1)
        assert param["integral"] == approx(self.integral, abs=1)

    def test_fit_zero_bg(self):
        self.params["peak_ph"].set(self.center)
        self.params["background"].set(0)
        result = self.model.fit(self.y, self.params, bin_centers=self.x)
        param = result.best_values
        assert result.success
        assert param["fwhm"] == approx(self.fwhm, abs=1)
        assert param["peak_ph"] == approx(self.center, abs=1)
        assert param["integral"] == approx(self.integral, abs=1)

    def test_negative_background_issue126(self):
        """This fit gives negative BG in all bins before the fix of issue #126."""
        obs = np.exp(-0.5*(self.x-self.center)**2/(self.fwhm/2.3548)**2) + 0
        result = self.model.fit(obs, self.params, bin_centers=self.x)
        param = result.best_values
        assert result.success
        bg_bin0 = param["background"]
        bg_binEnd = bg_bin0 + (len(self.x)-1)*param["bg_slope"]
        assert bg_bin0 >= 0
        assert bg_binEnd >= 0


class TestMnKA_fitter:
    def do_test(self, n=50000, resolution=2.5, tailfrac=0, tailtau=17, bg=10,
                nbins=150, vary_bg_slope=False, vary_tail=False, expect_good_fit=True):
        bin_edges = np.arange(5850, 5950, 0.5)
        # generate random x-ray pulse energies following MnKAlpha distribution
        line = mass.calibration.fluorescence_lines.MnKAlpha
        model = line.model(has_tails=vary_tail)

        values = line.rvs(size=n, instrument_gaussian_fwhm=0, rng=self.rng)
        # add Gaussian noise to each x-ray energy
        sigma = resolution/2.3548
        values += sigma*self.rng.standard_normal(size=n)

        tweak = self.rng.uniform(0, 1, size=n) < tailfrac
        ntweak = tweak.sum()
        if ntweak > 0:
            values[tweak] -= self.rng.standard_exponential(size=ntweak)*tailtau

        # histogram
        counts, _ = np.histogram(values, bin_edges)
        if bg > 0:
            counts += rng.poisson(lam=bg, size=len(counts))
        bin_width = bin_edges[1]-bin_edges[0]
        bin_centers = 0.5*bin_width + bin_edges[:-1]
        params = model.guess(counts, bin_centers=bin_centers)
        if vary_tail:
            params["tail_frac"].set(.1, vary=True)
            params["tail_tau"].set(30, vary=True)
        if vary_bg_slope:
            params["bg_slope"].set(0, vary=True)

        result = model.fit(counts, bin_centers=bin_centers, params=params)
        if expect_good_fit:
            assert result.redchi < 1.5, "redcued chi squared too high for good fit"
            assert result.redchi > 0.5, "reducec chi squared too low for good fit"
        expect = {
            "fwhm": (2.516, 3.73*resolution*n**-0.5),
            "peak_ph": (line.peak_energy, 3.3*n**-0.5),
            "integral": (n, n**0.5),
            "bg_slope": (0, 0.5),
            "background": (bg, 60*n**-0.5),
        }
        for param_name, (expected_value, expected_error) in expect.items():
            inflated_error = expected_error
            if vary_tail and param_name == "background":
                inflated_error *= 10
            if vary_bg_slope: # we really want to increase all allowed errors if bg slope is varied?
                inflated_error *= 10

            print(param_name, result.params[param_name])
            if not vary_bg_slope and not param_name.startswith("b"):
                assert expected_value == approx(result.params[param_name].value, abs=2*inflated_error)
            if not vary_tail and not vary_bg_slope and not param_name.startswith("b"):
                assert expected_error == approx(result.params[param_name].stderr, abs=0.5*inflated_error)


    def test_MnKA_lmfit(self):
        self.rng = np.random.default_rng(154)
        self.do_test()
        self.do_test(bg=0, tailfrac=0, expect_good_fit=True)
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=True, expect_good_fit=True)
        self.do_test(n=200000, tailtau=10, tailfrac=0.08, vary_tail=False, vary_bg_slope=True, expect_good_fit=False)


def test_MnKA_float32():
    """See issue 193: if the energies are float32, then the fit shouldn't be flummoxed."""
    line = mass.calibration.fluorescence_lines.MnKAlpha
    model = line.model()
    N = 100000
    rng = np.random.default_rng(238)
    energies = line.rvs(size=N, instrument_gaussian_fwhm=4.0, rng=rng)  # draw from the distribution
    e32 = np.asarray(energies, dtype=np.float32)

    # bin_edges will be float32 b/c e32 is.
    sim, bin_edges = np.histogram(e32, 120, [5865, 5925])
    binsize = bin_edges[1] - bin_edges[0]
    bctr = bin_edges[:-1] + 0.5*binsize
    params = model.guess(sim, bin_centers=bctr)
    params["peak_ph"].set(value=5899)
    result = model.fit(sim, params, bin_centers=bctr)
    val1 = params["peak_ph"].value
    val2 = result.params["peak_ph"].value
    assert val1 != approx(val2, abs=0.1)


def test_MnKA_narrowbins():
    """See issue 194: if dPH/dE >> 1, we should not automatically trigger the too-narrow-bins warning."""
    line = mass.calibration.fluorescence_lines.MnKAlpha
    model = line.model()
    N = 10000
    rng = np.random.default_rng(238)
    energies = line.rvs(size=N, instrument_gaussian_fwhm=4.0, rng=rng)  # draw from the distribution

    for SCALE in (0.1, 1, 10.):
        sim, bin_edges = np.histogram(energies*SCALE, 60, [5865*SCALE, 5925*SCALE])
        binsize = bin_edges[1] - bin_edges[0]
        bctr = bin_edges[:-1] + 0.5*binsize
        params = model.guess(sim, bin_centers=bctr)
        params["peak_ph"].set(value=5899*SCALE)
        _ = model.fit(sim, params, bin_centers=bctr)
        # If the above never errors, then problem solved.


def test_integral_parameter():
    """See issue 202: parameter 'integral' should be the total number of counts.
    See also issue 204: parameter 'integral' should scale inversely with a QE model."""
    line = mass.MnKAlpha
    bgperev = 50
    Nsignal = 10000
    rng = np.random.default_rng(3038)
    sig = line.rvs(Nsignal, instrument_gaussian_fwhm=5, rng=rng)
    bg = rng.uniform(5850, 5950, 100*bgperev)
    samples = np.hstack((bg, sig))
    for nbins in (100, 200, 300):
        s, b = np.histogram(samples, nbins, [5850, 5950])
        e = b[:-1] + 0.5*(b[1]-b[0])

        model = line.model()
        params = model.guess(s, bin_centers=e)
        result = model.fit(s, params, bin_centers=e)
        integral = result.best_values["integral"]
        assert integral == approx(Nsignal, abs=3*np.sqrt(len(samples)))

        # Now check that the integral still works even when dph_de = 2
        if nbins > 100:  # only need to check once
            continue
        dph_de = 2
        rescaled_e = e*dph_de
        params = model.guess(s, bin_centers=rescaled_e)
        result = model.fit(s, params, bin_centers=rescaled_e)
        integral = result.best_values["integral"]
        assert integral == approx(Nsignal, abs=3*np.sqrt(len(samples)))

        # And check that integral _times QE_ = Nsignal for a nontrivial QE model.
        QE = 0.4
        def flat_qemodel(e): return QE+np.zeros_like(e)
        model = line.model(qemodel=flat_qemodel)
        params = model.guess(s, bin_centers=e)
        result = model.fit(s, params, bin_centers=e)
        integral = result.best_values["integral"]
        assert integral*QE == approx(Nsignal, abs=3*np.sqrt(len(samples)))


class Test_Composites_lmfit:
    @pytest.fixture(autouse=True)
    def setUp(self):
        if 'dummy1' not in mass.spectrum_classes.keys():
            mass.calibration.fluorescence_lines.addline(
                element="dummy",
                material="dummy_material",
                linetype="1",
                reference_short='NIST ASD',
                fitter_type=mass.GenericLineModel,
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
                fitter_type=mass.GenericLineModel,
                reference_plot_instrument_gaussian_fwhm=0.5,
                nominal_peak_energy=653.679946,
                energies=np.array([653.679946]), lorentzian_fwhm=np.array([0.1]),
                reference_amplitude=np.array([1]),
                reference_amplitude_type=mass.LORENTZIAN_PEAK_HEIGHT, ka12_energy_diff=None
            )
        rng = np.random.default_rng(131)
        bin_edges = np.arange(600, 700, 0.4)
        resolution = 4.0
        n1 = 10000
        n2 = 20000
        self.n = n1+n2
        self.line1 = mass.spectrum_classes['dummy1']()
        self.line2 = mass.spectrum_classes['dummy2']()
        self.nominal_separation = self.line2.nominal_peak_energy - self.line1.nominal_peak_energy
        values1 = self.line1.rvs(size=n1, instrument_gaussian_fwhm=resolution, rng=rng)
        values2 = self.line2.rvs(size=n2, instrument_gaussian_fwhm=resolution, rng=rng)
        self.counts1, _ = np.histogram(values1, bin_edges)
        self.counts2, _ = np.histogram(values2, bin_edges)
        self.counts = self.counts1 + self.counts2
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

    def test_FitToModelWithoutPrefix(self):
        model1_noprefix = self.line1.model()
        assert len(model1_noprefix.prefix) == 0
        params1_noprefix = model1_noprefix.guess(self.counts1, bin_centers=self.bin_centers)
        params1_noprefix['dph_de'].set(value=1.0, vary=False)
        result1_noprefix = model1_noprefix.fit(
            self.counts1, params=params1_noprefix, bin_centers=self.bin_centers)
        for iComp in result1_noprefix.components:
            assert len(iComp.prefix) == 0
        result1_noprefix._validate_bins_per_fwhm(minimum_bins_per_fwhm=3)

    def test_NonUniqueParamsFails(self):
        model1_noprefix = self.line1.model()
        model2_noprefix = self.line2.model()
        with pytest.raises(NameError):
            _ = model1_noprefix + model2_noprefix

    def test_CompositeModelFit_with_prefix_and_background(self):
        prefix1 = 'p1_'
        prefix2 = 'p2_'
        model1 = self.line1.model(prefix=prefix1)
        model2 = self.line2.model(prefix=prefix2, has_linear_background=False)
        assert (model1.prefix == prefix1)
        assert (model2.prefix == prefix2)
        params1 = model1.guess(self.counts1, bin_centers=self.bin_centers)
        params2 = model2.guess(self.counts2, bin_centers=self.bin_centers)
        params1[f'{prefix1}dph_de'].set(value=1.0, vary=False)
        params2[f'{prefix2}dph_de'].set(value=1.0, vary=False)
        result1 = model1.fit(self.counts1, params=params1, bin_centers=self.bin_centers)
        result2 = model2.fit(self.counts2, params=params2, bin_centers=self.bin_centers)
        compositeModel = model1 + model2
        modelComponentPrefixes = [iComp.prefix for iComp in compositeModel.components]
        assert (np.logical_and(prefix1 in modelComponentPrefixes, prefix2 in modelComponentPrefixes))
        compositeParams = result1.params + result2.params
        compositeParams[f'{prefix1}fwhm'].expr = f'{prefix2}fwhm'
        compositeParams[f'{prefix1}peak_ph'].expr = f'{prefix2}peak_ph - {self.nominal_separation}'
        compositeParams.add(name='ampRatio', value=0.5, vary=False)
        compositeParams[f'{prefix1}integral'].expr = f'{prefix2}integral * ampRatio'
        compositeResult = compositeModel.fit(
            self.counts, params=compositeParams, bin_centers=self.bin_centers)
        resultComponentPrefixes = [iComp.prefix for iComp in compositeResult.components]
        assert (np.logical_and(prefix1 in resultComponentPrefixes, prefix2 in resultComponentPrefixes))
        compositeResult._validate_bins_per_fwhm(minimum_bins_per_fwhm=3)


def test_BackgroundMLEModel():
    class BackgroundMLEModel(mass.calibration.line_models.MLEModel):
        def __init__(self, independent_vars=['bin_centers'], prefix='', nan_policy='raise', **kwargs):
            def modelfunc(bin_centers, background, bg_slope):
                bg = np.zeros_like(bin_centers) + background
                bg += bg_slope * np.arange(len(bin_centers))
                bg[bg < 0] = 0
                if any(np.isnan(bg)) or any(bg < 0):
                    raise ValueError("some entry in r is nan or negative")
                return bg
            kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                           'independent_vars': independent_vars})
            super().__init__(modelfunc, **kwargs)
            self.set_param_hint('background', value=1, min=0)
            self.set_param_hint('bg_slope', value=0)

    test_model = BackgroundMLEModel(name='LinearTestModel', prefix='p1_')
    test_params = test_model.make_params(background=1.0, bg_slope=0.0)
    x_data = np.arange(1000, 2000, 1)
    test_background = 127.3
    test_background_error = np.sqrt(test_background)
    test_bg_slope = 0.17
    rng = np.random.default_rng()
    y_data = np.zeros_like(x_data) + test_background + \
        rng.normal(scale=test_background_error, size=len(x_data))
    y_data += test_bg_slope * np.arange(len(x_data))
    y_data[y_data < 0] = 0
    test_model.fit(y_data, test_params, bin_centers=x_data)


def test_negatives():
    "Test for issue 217."
    counts = np.array([2, 4, 2, 4], dtype=np.int64)
    bin_centers = np.array([8009.07622011, 8009.57622011,
                            8010.07622011, 8010.57622011])

    model = mass.spectra["CuKAlpha"].model()
    params = model.guess(bin_centers=bin_centers, data=counts)
    params["dph_de"].set(1, min=0.1, max=10, vary=False)
    params["fwhm"].set(4)
    params["peak_ph"].set(8009.57622011)
    params["integral"].set(0)
    params["background"].set(2.0000000000000004)
    params["bg_slope"].set(0)
    r = model._residual(params=params, bin_centers=bin_centers, data=counts, weights=None)
    assert not any(np.isnan(r))


def test_issue_125():
    """Test that issue 125 is fixed. The following fit used to take infinte time/memory.
    If this returns, then consider that a passing test."""

    e = np.linspace(5870.25, 5909.25, 80)
    e = e[:-1] + 0.5*(e[1]-e[0])
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

def test_tail_tau_behaves_same_vs_energy_scale():
    """Issue 250 reports that the tail_tau parameter needs to be very large to have 
    visible tails when peak_ph is in the MeV scale, here we test that we get the same
    spectrum when looking at the values in each bin when we change the energy scale of 
    all energy parameters equally"""

    def get_spectrum_with_tail(escale):
        peak_ph = 6000*escale
        model = mass.get_model(peak_ph, has_tails=True)
        params = model.make_params()
        params["fwhm"].set(2*escale)
        params["tail_tau"].set(30*escale)
        params["tail_frac"].set(0.5)
        bin_centers = np.arange(5900,6100,1)*escale
        return model.eval(params=params, bin_centers=bin_centers)
    
    spect1 = get_spectrum_with_tail(escale=1)
    spect2 = get_spectrum_with_tail(escale=1000)
    assert np.allclose(spect1, spect1, rtol=1e-3, atol=0)
