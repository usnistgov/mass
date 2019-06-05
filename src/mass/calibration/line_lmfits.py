import lmfit
import pylab as plt
import numpy as np
from . import line_fits
from . import fluorescence_lines

class MLEModel(lmfit.Model):
    """ A version of lmfit.Model that uses Maximum Likeliehood Estimates weights in place of chisq
    following:
    doi:10.1007/s10909-014-1098-4 Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"""
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

    def __div__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)

    def __truediv__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)

    def fit(self, *args, **kwargs):
        result = lmfit.Model.fit(self, *args, **kwargs)
        if not result.errorbars:
            raise(Exception("error bars not computed, are some of your guess values equal to max or min?"))
        return result

class CompositeMLEModel(lmfit.CompositeModel):
    """ A version of lmfit.CompositeModel that uses Maximum Likeliehood Estimates weights in place of chisq
    following:
    doi:10.1007/s10909-014-1098-4 Maximum-Likelihood Fits to Histograms for Improved Parameter Estimation"""
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

    def __div__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)

    def __truediv__(self, other):
        """/"""
        return CompositeMLEModel(self, other, lmfit.model.operator.truediv)


class GenericKAlphaModel(MLEModel):
    def __init__(self, independent_vars=['bin_centers'], prefix='', nan_policy='raise',
                 **kwargs):
        # spect must be defined by inheriting classes
        def modelfunc(bin_centers, fwhm, peak_ph, dph_de, amplitude,
         background, bg_slope, tail_frac, tail_tau):
            energy = (bin_centers - peak_ph) / dph_de + self.spect.peak_energy
            self.spect.set_gauss_fwhm(fwhm)
            cleanspectrum_fn = self.spect.pdf
            spectrum = line_fits._smear_lowEtail(cleanspectrum_fn, energy, fwhm, tail_frac, tail_tau)
            retval = line_fits._scale_add_bg(spectrum, amplitude, background, bg_slope)
            if any(np.isnan(retval)) or any(retval < 0):
                raise ValueError
            return retval
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars, 'name': "MnKAlpha"})
        super(GenericKAlphaModel, self).__init__(modelfunc, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('fwhm', value=4, min=0)
        self.set_param_hint('peak_ph', min=0)
        self.set_param_hint("dph_de", value=1, min=.01, max=100)
        self.set_param_hint("amplitude", value=100, min=0)
        self.set_param_hint('background', value=1, min=0)
        self.set_param_hint('bg_slope', value=0, min=0, vary=False)
        self.set_param_hint('tail_frac', value=0, min=0, vary=False)
        self.set_param_hint('tail_tau', value=0, min=0, vary=False)

    def guess(self, data, bin_centers=None, **kwargs):
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
        res = 4.0
        if len(data) > 20:
            # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
            baseline = max(data[0:10].mean(), 1.0/len(data))
        else:
            baseline = 0.1
        baseline_slope = 0.0
        pars = self.make_params(fwhm=res, peak_ph=ph_ka1, dph_de= dph/dE, bg=baseline, bgslope=baseline_slope)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

class MnKAlphaModel(GenericKAlphaModel):
    spect = fluorescence_lines.spectrum_classes["MnKAlpha"]()
