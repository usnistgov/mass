"""
Implements MLEModel, CompositeMLEModel, GenericLineModel,
GenericKAlphaModel
"""

import lmfit
import numpy as np
from . import line_fits


class MLEModel(lmfit.Model):
    """A version of lmfit.Model that uses Maximum Likelihood weights
    in place of chisq, as described in: doi:10.1007/s10909-014-1098-4
    "Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"
    """

    require_errorbars = True

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
        result = lmfit.Model.fit(self, *args, **kwargs)
        if self.require_errorbars and (not result.errorbars):
            raise(Exception("error bars not computed, are some of your guess values equal to max or min? you can set .require_errorbars=False to not error here"))
        return result


class CompositeMLEModel(lmfit.CompositeModel):
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
                 has_tails=False, **kwargs):
        self.spect = spect
        self._has_tails = has_tails
        if has_tails:
            def modelfunc(bin_centers, fwhm, peak_ph, dph_de, amplitude,
                          background, bg_slope, tail_frac, tail_tau, tail_frac_hi, tail_tau_hi):
                energy = (bin_centers - peak_ph) / dph_de + self.spect.peak_energy
                self.spect.set_gauss_fwhm(fwhm)
                cleanspectrum_fn = self.spect.pdf
                # Convert tau values (in eV units) to
                # lengths in bin units, which _smear_exponential_tail expects
                binwidth = bin_centers[1]-bin_centers[0]
                length_lo = tail_tau*dph_de/binwidth
                length_hi = tail_tau_hi*dph_de/binwidth
                spectrum = line_fits._smear_exponential_tail(
                    cleanspectrum_fn, energy, fwhm, tail_frac, length_lo, tail_frac_hi, length_hi)
                retval = line_fits._scale_add_bg(spectrum, amplitude, background, bg_slope)
                if any(np.isnan(retval)) or any(retval < 0):
                    raise ValueError
                return retval
        else:
            def modelfunc(bin_centers, fwhm, peak_ph, dph_de, amplitude, background, bg_slope):
                energy = (bin_centers - peak_ph) / dph_de + self.spect.peak_energy
                self.spect.set_gauss_fwhm(fwhm)
                spectrum = self.spect.pdf(energy)
                retval = line_fits._scale_add_bg(spectrum, amplitude, background, bg_slope)
                if any(np.isnan(retval)) or any(retval < 0):
                    raise ValueError
                return retval
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(GenericLineModel, self).__init__(modelfunc, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('fwhm', value=4, min=0)
        self.set_param_hint('peak_ph', min=0)
        self.set_param_hint("dph_de", value=1, min=.01, max=100)
        self.set_param_hint("amplitude", value=100, min=0)
        self.set_param_hint('background', value=1, min=0)
        self.set_param_hint('bg_slope', value=0, vary=False)
        if self._has_tails:
            self.set_param_hint('tail_frac', value=0.05, min=0, max=1, vary=True)
            self.set_param_hint('tail_tau', value=30, min=0, max=100, vary=True)
            self.set_param_hint('tail_frac_hi', value=0, min=0, max=1, vary=False)
            self.set_param_hint('tail_tau_hi', value=0, min=0, max=100, vary=False)

    def guess(self, data, bin_centers=None, **kwargs):
        "Guess values for the peak_ph, amplitude, and background."
        if data.sum() <= 0:
            raise ValueError("This histogram has no contents")
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