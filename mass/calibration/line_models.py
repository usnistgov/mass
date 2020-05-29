"""
Implements MLEModel, CompositeMLEModel, GenericLineModel,
GenericKAlphaModel
"""

import lmfit
import numpy as np
import pylab as plt

from . import line_fits

VALIDATE_BIN_SIZE = True


class MLEModel(lmfit.Model):
    """A version of lmfit.Model that uses Maximum Likelihood weights
    in place of chisq, as described in: doi:10.1007/s10909-014-1098-4
    "Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"
    """

    def _residual(self, params, data, weights, **kwargs):
        """Calculate the chi_MLE^2 value from Joe Fowler's Paper
        doi:10.1007/s10909-014-1098-4 "Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"
        """
        y = self.eval(params, **kwargs)
        if data is None:
            return y
        r2 = y-data
        nonzero = data > 0
        r2[nonzero] += data[nonzero]*np.log((data/y)[nonzero])
        vals = (2*r2) ** 0.5
        vals[y < data] *= -1
        return vals

    def __repr__(self):
        """Return representation of Model."""
        return "<MLEModel: %s>" % (self.name)

    def _reprstring(self, long=False):
        out = self._name
        opts = []
        if len(self._prefix) > 0:
            opts.append("prefix='%s'" % (self._prefix))
        if long:
            for k, v in self.opts.items():
                opts.append("%s='%s'" % (k, v))
        if len(opts) > 0:
            out = "%s, %s" % (out, ', '.join(opts))
        return "MLEModel(%s)" % out

    def __add__(self, other):
        """+"""
        return CompositeMLEModel(self, other, lmfit.model.operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeMLEModel(self, other, lmfit.model.operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeMLEModel(self, other, lmfit.model.operator.mul)

    def __truediv__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)

    # def exponentialTail(self):
    #     """Return a new CompositeModel that convolves this object with a 2-sided exponential tail kernel."""
    #     def kernelfunc(bin_centers, frac_lo, length_lo, frac_hi, length_hi):
    #         pass
    #
    #     kmodel = MLEModel(kernelfunc)
    #     kmodel.set_param_hint('frac_lo', value=0.1, min=0, max=1, vary=False)
    #     kmodel.set_param_hint('frac_hi', value=0.0, min=0, max=1, vary=False)
    #     kmodel.set_param_hint('length_lo', value=1.0, min=0, vary=False)
    #     kmodel.set_param_hint('length_hi', value=1.0, min=0, vary=False)
    #
    #     dummy_op = lmfit.model.operator.add  # won't actually use this, b/c overriding c.eval method.
    #     c = CompositeMLEModel(self, kmodel, dummy_op)
    #
    #     def blah():
    #         pass
    #     c.eval = blah
    #     return c

    def fit(self, *args, **kwargs):
        """as lmfit.Model.fit except
        1. the default method is "least_squares because it gives error bars more often at 1.5-2.0X speed penalty
        2. supports "leastsq_refit" which uses "leastsq" to fit, but if there are no error bars, refits with "least_squares"
        call result.set_label_hints(...) then result.plotm() for a nice plot
        """
        if "method" not in kwargs:
            # change default method
            kwargs["method"] = "least_squares"
            # least_squares always gives uncertainties, while the normal default leastsq often does not
            # leastsq fails to give uncertaities if parameters are near bounds or at their initial value
            # least_squares is about 1.5X to 2.0X slower based on two test case
        if "minimum_bins_per_fwhm" not in kwargs:
            minimum_bins_per_fwhm = 3  # provide default value
        else:
            minimum_bins_per_fwhm = kwargs["minimum_bins_per_fwhm"]
            # remove this argument before passwing kwargs to ._fit
            del kwargs["minimum_bins_per_fwhm"]
        result = self._fit(*args, **kwargs)
        result.__class__ = LineModelResult
        result._validate_bins_per_fwhm(minimum_bins_per_fwhm)
        return result

    def _fit(self, *args, **kwargs):
        """internal implementation of fit to add support for "leastsq_refit" method"""
        if kwargs["method"] == "leastsq_refit":
            # fit fit with leastsq, then if we dont have unceratinties, fit again with least_squares
            kwargs["method"] = "leastsq"
            result0 = lmfit.Model.fit(self, *args, **kwargs)
            if result0.success and result0.errorbars:
                return result0
            kwargs["method"] = "least_squares"
            if "params" in kwargs:
                kwargs["params"] = result0.params
            elif len(args) > 1:
                args = [result0.params if i == 1 else arg for (i, arg) in enumerate(args)]
            result = lmfit.Model.fit(self, *args, **kwargs)
        else:
            result = lmfit.Model.fit(self, *args, **kwargs)
        return result


# first parent has precedence for repeated method (eg fit)
class CompositeMLEModel(MLEModel, lmfit.CompositeModel):
    """A version of lmfit.CompositeModel that uses Maximum Likelihood weights
    in place of chisq, as described in: doi:10.1007/s10909-014-1098-4
    "Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"
    """

    def _residual(self, params, data, weights, **kwargs):
        """Calculate the chi_MLE^2 value from Joe Fowler's Paper
        doi:10.1007/s10909-014-1098-4 Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation
        """
        y = self.eval(params, **kwargs)
        if data is None:
            return y
        r2 = y-data
        nonzero = data > 0
        r2[nonzero] += data[nonzero]*np.log((data/y)[nonzero])
        vals = (2*r2) ** 0.5
        vals[y < data] *= -1
        return vals

    def __repr__(self):
        """Return representation of Model."""
        return "<CompositeMLEModel: %s>" % (self.name)

    def __add__(self, other):
        """+"""
        return CompositeMLEModel(self, other, lmfit.model.operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeMLEModel(self, other, lmfit.model.operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeMLEModel(self, other, lmfit.model.operator.mul)

    def __truediv__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)


class GenericLineModel(MLEModel):
    def __init__(self, spect, independent_vars=['bin_centers'], prefix='', nan_policy='raise',
                 has_linear_background=True, has_tails=False, **kwargs):
        self.spect = spect
        self._has_tails = has_tails
        self._has_linear_background = has_linear_background
        if has_tails:
            def modelfunc(bin_centers, fwhm, peak_ph, dph_de, amplitude,
                          background=0, bg_slope=0, tail_frac=0, tail_tau=8, tail_frac_hi=0, tail_tau_hi=8):
                energy = (bin_centers - peak_ph) / dph_de + self.spect.peak_energy
                def cleanspectrum_fn(x): return self.spect.pdf(x, instrument_gaussian_fwhm=fwhm)
                # Convert tau values (in eV units) to
                # lengths in bin units, which _smear_exponential_tail expects
                binwidth = bin_centers[1]-bin_centers[0]
                length_lo = tail_tau*dph_de/binwidth
                length_hi = tail_tau_hi*dph_de/binwidth
                spectrum = line_fits._smear_exponential_tail(
                    cleanspectrum_fn, energy, fwhm, tail_frac, length_lo, tail_frac_hi, length_hi)
                r = line_fits._scale_add_bg(spectrum, amplitude, background, bg_slope)
                if any(np.isnan(r)) or any(r < 0):
                    raise ValueError("some entry in r is nan or negative")
                return r
        else:
            def modelfunc(bin_centers, fwhm, peak_ph, dph_de, amplitude, background=0, bg_slope=0):
                energy = (bin_centers - peak_ph) / dph_de + self.spect.peak_energy
                spectrum = self.spect.pdf(energy, fwhm)
                r = line_fits._scale_add_bg(spectrum, amplitude, background, bg_slope)
                if any(np.isnan(r)) or any(r < 0):
                    raise ValueError("some entry in r is nan or negative")
                return r
        param_names = ["fwhm", "peak_ph", "dph_de", "amplitude"]
        if self._has_linear_background:
            param_names += ["background", "bg_slope"]
        if self._has_tails:
            param_names += ["tail_frac", "tail_tau", "tail_frac_hi", "tail_tau_hi"]
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars, "param_names" : param_names})
        super(GenericLineModel, self).__init__(modelfunc, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('fwhm', value=4, min=0)
        self.set_param_hint('peak_ph', min=0, max=2**16)
        self.set_param_hint("dph_de", value=1, min=.01, max=100)
        self.set_param_hint("amplitude", value=100, min=0)
        if self._has_linear_background:
            self.set_param_hint('background', value=1, min=0)
            self.set_param_hint('bg_slope', value=0, vary=False)
        if self._has_tails:
            self.set_param_hint('tail_frac', value=0.05, min=0, max=1, vary=True)
            self.set_param_hint('tail_tau', value=30, min=0, max=100, vary=True)
            self.set_param_hint('tail_frac_hi', value=0, min=0, max=1, vary=False)
            self.set_param_hint('tail_tau_hi', value=0, min=0, max=100, vary=False)

    def guess(self, data, bin_centers=None, **kwargs):
        "Guess values for the peak_ph, amplitude, and background."
        # if data.sum() <= 0:
        #     pars = self.make_params()
        #     for k,v in pars.items():
        #         v.set(0,vary=False)
        #     pars["dph_de"].set(1,vary=False)
        # else:
        peak_ph = bin_centers[data.argmax()]
        ampl = data.max() * 9.4  # this number is taken from the GenericKBetaFitter
        if len(data) > 20:
            # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
            baseline = max(data[0:10].mean(), 1.0/len(data))
        else:
            baseline = 0.1
        pars = self.make_params(peak_ph=peak_ph, background=baseline, amplitude=ampl)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class GenericKAlphaModel(GenericLineModel):
    "Overrides GenericLineModel.guess to make guesses appropriate for K-alpha lines."

    def guess(self, data, bin_centers=None, **kwargs):
        "Guess values for the peak_ph, amplitude, and background, and dph_de"
        if data.sum() <= 0:
            raise ValueError("This histogram has no contents")
        # Heuristic: find the Ka1 line as the peak bin, and then make
        # assumptions about the full width (from 1/4-peak to 1/4-peak) and
        # how that relates to the PH spacing between Ka1 and Ka2
        peak_val = data.max()
        peak_ph = bin_centers[data.argmax()]
        lowqtr = bin_centers[(data > peak_val * 0.25).argmax()]
        N = len(data)
        topqtr = bin_centers[N - 1 - (data > peak_val * 0.25)[::-1].argmax()]

        ph_ka1 = peak_ph
        dph = 0.66 * (topqtr - lowqtr)
        dE = self.spect.ka12_energy_diff  # eV difference between KAlpha peaks
        ampl = data.max() * 9.4
        if len(data) > 20:
            # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
            baseline = max(data[0:10].mean(), 1.0/len(data))
        else:
            baseline = 0.1
        pars = self.make_params(peak_ph=ph_ka1, dph_de=dph/dE, background=baseline, amplitude=ampl)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class LineModelResult(lmfit.model.ModelResult):
    """like lmfit.model.Model result, but with some convenient plotting functions for line spectra fits"""

    def _compact_fit_report(self):
        s = ""
        sn = {"background": "bg", "amplitude": "ampl", "bg_slope": "bg_slp"}
        for k in sorted(self.params.keys()):
            v = self.params[k]
            if v.vary:
                if v.stderr is None:
                    sig_figs = 2
                    s += f"{sn.get(k,k):7} {v.value:.{sig_figs}g}±None\n"
                else:
                    sig_figs = int(np.ceil(np.log10(np.abs(v.value/v.stderr)))+1)
                    sig_figs = max(1, sig_figs)
                    s += f"{sn.get(k,k):7} {v.value:.{sig_figs}g}±{v.stderr:.2g}\n"
            else:
                s += f"{sn.get(k,k):7} {v.value:.{sig_figs}g} HELD\n"
        return s[:-1]

    def plotm(self, ax=None, title=None, xlabel=None, ylabel=None):
        """plot the data, the fit, and annotate the plot with the parameters"""
        title, xlabel, ylabel = self._handle_default_labels(title, xlabel, ylabel)
        ax = lmfit.model.ModelResult.plot_fit(self, ax=ax,
                                              xlabel=xlabel, ylabel=ylabel)
        if title is not None:
            plt.title(title)
        ax.text(0.05, 0.95, self._compact_fit_report(), transform=ax.transAxes,
                verticalalignment="top", bbox=dict(facecolor='w', alpha=0.5), family="monospace")
        # ax.legend(["data", self._compact_fit_report()],loc='best', frameon=True, framealpha = 0.5)
        ax.legend(loc="upper right")

    def set_label_hints(self, binsize, ds_shortname, attr_str, unit_str, cut_hint, states_hint=""):
        self._binsize = binsize
        self._ds_shortname = ds_shortname
        self._attr_str = attr_str
        self._unit_str = unit_str
        self._cut_hint = cut_hint
        self._states_hint = states_hint
        self._has_label_hints = True

    def _handle_default_labels(self, title, xlabel, ylabel):
        if hasattr(self, "_has_label_hints"):
            if title is None:
                title = f"{self._ds_shortname}: {self.model.spect.shortname}"
            if ylabel is None:
                ylabel = f"counts per {self._binsize:g} {self._unit_str} bin"
                if self._states_hint != "":
                    ylabel += f"\nstates={self._states_hint}: {self._cut_hint}"
            if xlabel is None:
                xlabel = f"{self._attr_str} ({self._unit_str})"
        else:
            if ylabel is None and "bin_centers" in self.userkws:
                binsize = self.userkws["bin_centers"][1]-self.userkws["bin_centers"][0]
                ylabel = f"counts per {binsize:g} unit bin"
        return title, xlabel, ylabel

    def _validate_bins_per_fwhm(self, minimum_bins_per_fwhm):
        if "bin_centers" not in self.userkws:
            return  # i guess someone used this for a non histogram fit
        if not VALIDATE_BIN_SIZE:
            return        
        bin_centers = self.userkws["bin_centers"]
        bin_size = bin_centers[1]-bin_centers[0]
        for iComp in self.components:
            prefix = iComp.prefix
            if (prefix+"dph_de" in self.params) and (prefix+"fwhm" in self.params):
                bin_size_energy = bin_size/self.params[prefix+"dph_de"]
                instrument_gaussian_fwhm_energy = self.params[prefix+"fwhm"].value/self.params[prefix+"dph_de"]
                minimum_fwhm_energy = iComp.spect.minimum_fwhm(instrument_gaussian_fwhm_energy)
                bins_per_fwhm = minimum_fwhm_energy/bin_size_energy
                if bins_per_fwhm < minimum_bins_per_fwhm:
                    raise Exception(f"""your bins are too large. bin_size (energy units) = {bin_size_energy:.3g}, fit_fwhm (energy units) = {instrument_gaussian_fwhm_energy:.3g}\n
                minimum fwhm accounting for narrowest lorentzian in spectrum (energy units) = {minimum_fwhm_energy:.3g}\n
                bins_per_fwhm = {bins_per_fwhm:.3g}, minimum_bins_per_fwhm = {minimum_bins_per_fwhm:.3g}\n
                to avoid this error:\n
                1. use smaller bins\n
                or 2. pass a smaller value of `minimum_bins_per_fwhm` to .fit\n
                or 3. set `mass.line_models.VALIDATE_BIN_SIZE = False`
                see https://bitbucket.org/joe_fowler/mass/issues/162/resolution-bias-in-fits-where-bin-size-is for discussion on this issue""")
