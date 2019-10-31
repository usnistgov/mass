"""
line_fits.py

Addded Joe Fowler 5 May, 2016

Separated line fits (here) from the line shapes (still in fluorescence_lines.py)
"""

import numpy as np
import pylab as plt

from mass.mathstat.fitting import MaximumLikelihoodHistogramFitter
from mass.mathstat.utilities import plot_as_stepped_hist
from mass.mathstat.special import voigt, voigt_approx_fwhm


def _smear_lowEtail(cleanspectrum_fn, x, P_resolution, P_tailfrac, P_tailtau):
    """Evaluate cleanspectrum_fn(x), but padded and smeared to add a low-E tail."""
    if P_tailfrac <= 1e-5:
        return cleanspectrum_fn(x)

    # Compute the low-E-tailed spectrum. This is done by
    # convolution, which is computed using DFT methods.
    # A wider energy range must be used, or wrap-around effects of
    # tails will corrupt the model.
    # Go 6*tau or up to 500 eV low; go res + tail (up to 50 eV) high, up to 1000 bins
    dx = x[1] - x[0]
    nlow = int(min(P_tailtau*6, 500) / dx + 0.5)
    nhi = int((P_resolution + min(P_tailtau, 50)) / dx + 0.5)
    nhi = min(1000, nhi)  # A practical limit
    nlow = max(nlow, nhi)
    x_wide = np.arange(-nlow, nhi+len(x)) * dx + x[0]
    if len(x_wide) > 100000:
        msg = "you're trying to fft data of length %i (bad fit param?)" % len(x_wide)
        raise ValueError(msg)

    freq = np.fft.rfftfreq(len(x_wide), d=dx)
    rawspectrum = cleanspectrum_fn(x_wide)
    ft = np.fft.rfft(rawspectrum)
    ft += ft * P_tailfrac * (1.0 / (1 - 2j * np.pi * freq * P_tailtau) - 1)
    smoothspectrum = np.fft.irfft(ft, n=len(x_wide))
    # in pathalogical cases, convolutuion can cause negative values
    # this is a hacky way to protect against that
    smoothspectrum[smoothspectrum < 0] = 0
    return smoothspectrum[nlow:nlow + len(x)]


def _scale_add_bg(spectrum, P_amplitude, P_bg=0, P_bgslope=0):
    """Scale a spectrum and add a sloped background. BG<0 is replaced with BG=0."""
    bg = np.zeros_like(spectrum) + P_bg
    if P_bgslope != 0:
        bg += P_bgslope * np.arange(len(spectrum))
    bg[bg < 0] = 0
    spectrum = spectrum * P_amplitude + bg  # Change in place and return changed vector
    return spectrum


class LineFitter(object):
    """Abstract base class for line fitting objects."""

    def __init__(self):
        # Parameters+covariance from last successful fit
        self.last_fit_params = None
        self.last_fit_cov = None
        # Fit function samples from last successful fit
        self.last_fit_result = None
        self.tailfrac = 0.0
        self.tailtau = 25
        # Whether pulse heights are necessarily non-negative.
        self.phscale_positive = True
        self.penalty_function = None
        self.hold = set([])
        self.last_fit_bins = None
        self.last_fit_contents = None
        self.fit_success = False
        self.last_fit_chisq = np.inf
        self.failed_fit_exception = None
        self.failed_fit_params = None
        self.failed_fit_starting_fitfunc = None

    def set_penalty(self, penalty):
        """Set a regularizer, or penalty function, for the fitter. For its requirements,
        see MaximumLikelihoodHistogramFitter.set_penalty()."""
        self.penalty_function = penalty

    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None,
            color=None, label=True, vary_resolution=True, vary_bg=True,
            vary_bg_slope=False, vary_tail=False, hold=None, verbose=False, ph_units="arb",
            integrate_n_points=None, rethrow=False):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the
        set of histogram bins <pulseheights>.

        On a succesful fit self.fit_success is set to True. You can use self.plot() after the fact
        to make the same plot as if you passed plot=True.

        On a failed fit, self.fit_success is set to False. self.failed_fit_exception contains the
        exception thrown. self.plot will still work, and will indicate a failed fit. You can disable
        this behavior, and just have it throw an exception if you pass rethrow=True.

        Args:
            pulseheights -- the histogram bin centers or bin edges.

            params: see self.__doc__, because the group of parameters and their numbering
                    depends on the exact line shape being fit.

            plot:   Whether to make a plot.  If not, then the next few args are ignored
            axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather
                    than on the current figure.
            color:  Color for drawing the histogram contents behind the fit.
            label:  (True/False) Label for the fit line to go into the plot (usually used for
                    resolution and uncertainty)
                    "full" label with all fit params including chi sqaured (w/ an "H" if it was held)
            ph_units: "arb" by default, used in x and y labels on plot (pass "eV" if you have eV!)

            vary_resolution Whether to let the Gaussian resolution vary in the fit
            vary_bg:       Whether to let a constant background level vary in the fit
            vary_bg_slope: Whether to let a slope on the background level vary in the fit
            vary_tail:     Whether to let a low-energy exponential tail to vary.
            hold:      A sequence of parameter numbers to keep fixed.  Resolution, BG
                       BG slope, or tail will be held if relevant parameter number
                       appears in the hold sequence OR if relevant boolean vary_* tests False.
            integrate_n_points: Perform numerical integration across each bin with this many points
                        per bin. Default: None means use a heuristic to decide. For narrow bins,
                        generally this will choose 1, i.e., the midpoint method. For wide ones,
                        Simpson's method for 3, 5, or more will be appropriate
            rethrow: Throw any generated exceptions instead of catching them and setting fit_success=False.

        Returns:
            (fitparams, covariance)
            fitparams has same format as input variable params.
        """

        # Convert bin edges to centers
        pulseheights = np.asarray(pulseheights)
        if len(pulseheights) == len(data) + 1:
            dp = pulseheights[1] - pulseheights[0]
            pulseheights = 0.5 * dp + pulseheights[:-1]

        # Pulseheights doesn't make sense as bin centers, either.
        # So just use the integers starting at zero.
        elif len(pulseheights) != len(data):
            pulseheights = np.arange(len(data), dtype=np.float)

        self.hold = hold
        if self.hold is None:
            self.hold = set([])
        else:
            self.hold = set(self.hold)
        if not vary_resolution:
            self.hold.add(self.param_meaning["resolution"])
        if not vary_bg:
            self.hold.add(self.param_meaning["background"])
        if not vary_bg_slope:
            self.hold.add(self.param_meaning["bg_slope"])
        if not vary_tail:
            self.hold.add(self.param_meaning["tail_frac"])
            self.hold.add(self.param_meaning["tail_length"])

        if (params is not None) and (not len(params) == self.nparam):
            raise ValueError("params has wrong length")

        self.last_fit_bins = pulseheights.copy()
        self.last_fit_contents = data.copy()

        try:
            if params is None:
                params = self.guess_starting_params(data, pulseheights)

            if integrate_n_points is None:
                integrate_n_points = 1
                # Given no direction, we have to use a heuristic here to decide how densely
                # to perform numerical integration within each bin.
                w = self.feature_scale(params)
                binwidth = pulseheights[1] - pulseheights[0]
                if w/binwidth < 6.5:
                    integrate_n_points = 3
                if w/binwidth < 1.5:
                    integrate_n_points = 5
                if w/binwidth < 0.9:
                    integrate_n_points = 7

            if integrate_n_points % 2 != 1 or integrate_n_points < 1:
                raise ValueError("integrate_n_points=%d, want an odd, positive number" %
                                 integrate_n_points)

            # In this block, replace fitfunc with the version that integrates numerically across bins
            fitfunc = self.fitfunc
            if integrate_n_points > 1:
                dx = pulseheights[1] - pulseheights[0]
                x0 = pulseheights[0] - 0.5*dx
                x1 = pulseheights[-1] + 0.5*dx
                x_values = np.linspace(x0, x1, 1+(integrate_n_points-1)*len(pulseheights))
                if integrate_n_points == 3:
                    def integrated_model(params, _x):
                        y = self.fitfunc(params, x_values)
                        return (y[0:-1:2] + 4.0*y[1::2] + y[2::2])/6.0
                elif integrate_n_points == 5:
                    def integrated_model(params, _x):
                        y = self.fitfunc(params, x_values)
                        return (y[0:-1:4] + 4.0*y[1::4] + 2.0*y[2::4] + 4.0*y[3::4] + y[4::4])/12.0
                else:
                    def integrated_model(params, _x):
                        y = self.fitfunc(params, x_values)
                        ninterv = integrate_n_points-1
                        # dx = 1.0/ninterv
                        z = y[0:-1:ninterv] + y[ninterv::ninterv]
                        for i in range(1, ninterv, 2):
                            z += 4.0*y[i::ninterv]
                        for i in range(2, ninterv-1, 2):
                            z += 2.0*y[i::ninterv]
                        return z / (3.0*ninterv)
                fitfunc = integrated_model

            # Max-likelihood histogram fitter
            epsilon = self.stepsize(params)
            fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params,
                                                      fitfunc, TOL=1e-4, epsilon=epsilon)
            self.setbounds(params, pulseheights)
            for i, b in enumerate(self.bounds):
                fitter.setbounds(i, b[0], b[1])

            for h in self.hold:
                fitter.hold(h)

            if self.penalty_function is not None:
                fitter.set_penalty(self.penalty_function)

            self.last_fit_params, self.last_fit_cov = fitter.fit(verbose=verbose)
            self.fit_success = True
            self.last_fit_chisq = fitter.chisq
            self.last_fit_result = self.fitfunc(self.last_fit_params, self.last_fit_bins)

        except Exception as e:
            if rethrow:
                raise e
            self.fit_success = False
            self.last_fit_params = np.ones(self.nparam)*np.nan
            self.last_fit_cov = np.ones((self.nparam, self.nparam))*np.nan
            self.last_fit_chisq = np.nan
            self.last_fit_result = np.ones(self.nparam)*np.nan

            self.failed_fit_exception = e
            self.failed_fit_params = params
            if params is None:
                self.failed_fit_starting_fitfunc = np.ones(len(self.last_fit_contents))*np.nan
            else:
                self.failed_fit_starting_fitfunc = self.fitfunc(
                    self.failed_fit_params, self.last_fit_bins)

        if plot:
            self.plot(color, axis, label, ph_units)

        return self.last_fit_params, self.last_fit_cov

    def setbounds(self, params, ph):
        msg = "%s is an abstract base class; cannot be used without implementing setbounds" % type(
            self)
        raise NotImplementedError(msg)

    def _minBG0(self, params, ph):
        """Lower bound for the bin-0 background.
        It should be bounded IF the BG slope is held, but not otherwise."""
        minBG0 = None
        bg_slope_param = self.param_meaning["bg_slope"]
        if bg_slope_param in self.hold:
            bg_slope = params[bg_slope_param]
            if bg_slope >= 0:
                minBG0 = 0.0
            else:
                nbins = len(ph)
                minBG0 = -bg_slope * (nbins-1)
        return minBG0

    def result_string(self):
        """Return a string describing the fit result, including
        the value and uncertainty for each parameter.
        An "H" after the parameter indicates it was held.
        """
        labeldict = {meaning: meaning+" %.3g +- %.3g" for meaning in self.param_meaning.keys()}
        labeldict["resolution"] = "FWHM: %.3g +- %.3g"
        labeldict["tail_frac"] = "f$_\\mathrm{tail}$: %.3f +- %.3f"
        labeldict["peak_ph"] = "peak_ph: %.7g +- %.3g"
        slabel = ""
        for (meaning, i) in self.param_meaning.items():
            val = self.last_fit_params[i]
            err = self.last_fit_cov[i, i]**0.5
            s = labeldict[meaning] % (val, err)
            if i in self.hold:
                s += " H"
            s += "\n"
            slabel += s
        slabel += "reduced chisq %4g" % self.last_fit_reduced_chisq
        return slabel

    def _plot_failed_fit(self, color, axis, label, ph_units):
        """ Overrides part of self.plot if self.fit_success==False """
        plot_as_stepped_hist(axis, self.last_fit_contents, self.last_fit_bins, color=color)
        axis.plot(self.last_fit_bins, self.failed_fit_starting_fitfunc, color='m',
                  label="failed fit\nguess params shown")
        ph_binsize = self.last_fit_bins[1] - self.last_fit_bins[0]
        axis.set_xlim([self.last_fit_bins[0] - 0.5 * ph_binsize,
                       self.last_fit_bins[-1] + 0.5 * ph_binsize])
        axis.set_xlabel("energy (%s)" % ph_units)
        axis.set_ylabel("counts per %0.2f %s bin" % (ph_binsize, ph_units))
        axis.set_title("failed fit")
        axis.legend(loc="best", frameon=False)

    def plot(self, color=None, axis=None, label=True, ph_units="arb"):
        """Plot the fit.

        Args:
            color = color of the data
            axis = axis on which to plot, if it is None, the current figure is cleared
            label = True, False or "full"
                "full" includes more info than True
            ph_units = used for the ylabel
        """

        if color is None:
            color = 'blue'
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        if not self.fit_success:
            return self._plot_failed_fit(color, axis, label, ph_units)

        plot_as_stepped_hist(axis, self.last_fit_contents, self.last_fit_bins, color=color)
        ph_binsize = self.last_fit_bins[1] - self.last_fit_bins[0]
        axis.set_xlim([self.last_fit_bins[0] - 0.5 * ph_binsize,
                       self.last_fit_bins[-1] + 0.5 * ph_binsize])
        axis.set_xlabel("energy (%s)" % ph_units)
        axis.set_ylabel("counts per %0.2f %s bin" % (ph_binsize, ph_units))

        pnum_res = self.param_meaning["resolution"]
        slabel = ""
        if label == "full":
            slabel = self.result_string()
        elif label and pnum_res not in self.hold:
            pnum_tf = self.param_meaning["tail_frac"]
            res = self.last_fit_params[pnum_res]
            err = self.last_fit_cov[pnum_res, pnum_res]**0.5
            tf = self.last_fit_params[pnum_tf]
            slabel = "FWHM: %.2f +- %.2f" % (res, err)
            if tf > 0.001:
                slabel += "\nf$_\\mathrm{tail}$: %.1f%%" % (tf*100)
        axis.plot(self.last_fit_bins, self.last_fit_result, color='#666666',
                  label=slabel)
        if slabel:
            axis.legend(loc='best', frameon=False)

    @property
    def n_degree_of_freedom(self):
        """return the number of degrees of freedom"""
        return len(self.last_fit_bins)-self.nparam+len(self.hold)

    @property
    def last_fit_reduced_chisq(self):
        return self.last_fit_chisq/self.n_degree_of_freedom

    @property
    def last_fit_params_dict(self):
        """return a dictionary mapping a param meaning (like "resolution") to a tuple of value and uncertainty"""
        return {k: (self.last_fit_params[i], np.sqrt(self.last_fit_cov[i][i])) for (k, i) in self.param_meaning.items()}


class VoigtFitter(LineFitter):
    """Fit a single Lorentzian line, with Gaussian smearing and potentially a low-E tail.

    Parameters are:
    0 - Gaussian resolution (FWHM)
    1 - Pulse height (x-value) of the line peak
    2 - Lorentzian component width (FWHM)
    3 - Amplitude (y-value) of the line peak
    4 - Mean background counts per bin
    5 - Background slope (counts per bin per bin)
    6 - Low-energy tail fraction (0 <= f <= 1)
    7 - Low-energy tail scale length

    The units of 0, 1, 2, and 7 are all whatsoever units are used for pulse heights.
    """

    param_meaning = {
        "resolution": 0,
        "peak_ph": 1,
        "lorentz_fwhm": 2,
        "amplitude": 3,
        "background": 4,
        "bg_slope": 5,
        "tail_frac": 6,
        "tail_length": 7
    }
    nparam = 8

    def __init__(self):
        super(VoigtFitter, self).__init__()

    def feature_scale(self, params):
        res = params[self.param_meaning["resolution"]]
        lw = params[self.param_meaning["lorentz_fwhm"]]
        return voigt_approx_fwhm(lw, res)

    def guess_starting_params(self, data, binctrs, tailf=0.0, tailt=25.0):
        order_stat = np.array(data.cumsum(), dtype=np.float) / data.sum()

        def percentiles(p):
            return binctrs[(order_stat > p).argmax()]

        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75) - percentiles(0.25))
        res = iqr * 0.7
        lor_fwhm = res
        # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
        baseline = max(data[0:10].mean(), 1.0/len(data))
        baseline_slope = (data[-10:].mean() - baseline) / len(data)
        ampl = (data.max() - baseline) * np.pi
        return [res, peak_loc, lor_fwhm, ampl, baseline, baseline_slope, tailf, tailt]

    # Compute the smeared line value.
    #
    # @param params  The 8 parameters of the fit (see self.__doc__ for details).
    # @param x       An array of pulse heights.
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 8 parameters of the fit (see self.__doc__ for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:
            The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_lorenzfwhm, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params
        sigma = P_gaussfwhm / (8 * np.log(2))**0.5
        lorentz_hwhm = P_lorenzfwhm*0.5

        def cleanspectrum_fn(x):
            return voigt(x, P_phpeak, lorentz_hwhm, sigma)

        spectrum = _smear_lowEtail(cleanspectrum_fn, x, P_gaussfwhm, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def setbounds(self, params, ph):
        # bin-0 background should be bounded IF the BG slope is held
        minBG0 = self._minBG0(params, ph)

        self.bounds = []
        self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None))  # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))      # Lorentz FWHM
        self.bounds.append((0, None))       # Amplitude
        self.bounds.append((minBG0, None))  # Background level (bin 0)
        self.bounds.append((None, None))    # Background slope (counts/bin)
        self.bounds.append((0, 1))          # Tail fraction
        self.bounds.append((0, None))       # Tail scale length

    def stepsize(self, params):
        eps = np.array((1e-3, params[0]/1e5, 1e-3, params[3]/1e5, 1e-3, 1e-3, 1e-4, 1e-2))
        return eps


class NVoigtFitter(LineFitter):
    """Fit a set of N Lorentzian lines, with Gaussian smearing.

    So far, I don't know how to guess the starting parameters, so you have to supply all 3N+5.
    (See method fit() for explanation).
    """

    def __init__(self, Nlines):
        """This fitter will have `Nlines` Lorentzian lines."""
        self.Nlines = Nlines
        assert Nlines >= 1
        super(NVoigtFitter, self).__init__()
        self.nparam = 5+3*Nlines
        self.param_meaning = {
            "resolution": 0,
            "background": self.nparam-4,
            "bg_slope": self.nparam-3,
            "tail_frac": self.nparam-2,
            "tail_length": self.nparam-1
        }
        for i in range(Nlines):
            j = i+1
            self.param_meaning["peak_ph%d" % j] = i*3+1
            self.param_meaning["lorentz_fwhm%d" % j] = i*3+2
            self.param_meaning["amplitude%d" % j] = i*3+3

    def feature_scale(self, params):
        res = params[self.param_meaning["resolution"]]
        lw = params[self.param_meaning["lorentz_fwhm0"]]
        for i in range(1, self.Nlines):
            lw = min(lw, params[self.param_meaning["lorentz_fwhm%d" % i]])
        return voigt_approx_fwhm(lw, res)

    def guess_starting_params(self, data, binctrs):
        raise NotImplementedError(
            "I don't know how to guess starting parameters for a %d-peak Voigt." % self.Nlines)

    # Compute the smeared line value.
    #
    # @param params  The parameters of the fit (see self.fit for details).
    # @param x       An array of pulse heights (params will scale them to energy).
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 3N+5 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        x = np.asarray(x)
        if not x.shape:
            x = x.reshape(1)
        P_gaussfwhm = params[0]
        P_amplitude = 1.0  # overall scale factor covered by the individual Lorentzians
        P_bg, P_bgslope, P_tailfrac, P_tailtau = params[-4:]
        sigma = params[0] / (8 * np.log(2))**0.5

        def cleanspectrum_fn(x):
            s = np.zeros_like(x)
            for i in range(self.Nlines):
                s += voigt(x, params[1+3*i], params[2+3*i]*0.5, sigma) * params[3+3*i]
            return s
        spectrum = _smear_lowEtail(cleanspectrum_fn, x, P_gaussfwhm, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def setbounds(self, params, ph):
        # bin-0 background should be bounded IF the BG slope is held
        minBG0 = self._minBG0(params, ph)

        self.bounds = []
        self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))  # Gauss FWHM
        for _ in range(self.Nlines):
            self.bounds.append((np.min(ph), np.max(ph)))
            self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))  # Lorentz FWHM
            self.bounds.append((0, None))   # Amplitude
        self.bounds.append((minBG0, None))  # Background level (bin 0)
        self.bounds.append((None, None))    # Background slope (counts/bin)
        self.bounds.append((0, 1))          # Tail fraction
        self.bounds.append((0, None))       # Tail scale length

    def stepsize(self, params):
        epsilon = np.copy(params)
        epsilon[0] = 1e-3
        epsilon[-4] = 1e-3  # BG level
        epsilon[-3] = 1e-3  # BG slope
        epsilon[-2] = 1e-4
        epsilon[-1] = 1e-2
        for i in range(self.Nlines):
            epsilon[1+i*3] = params[0]/1e5
            epsilon[2+i*3] = 1e-3
            epsilon[3+i*3] *= 1e-5
        return epsilon


class GaussianFitter(LineFitter):
    """Fit a single Gaussian line, with a low-E tail.

    Parameters are:
    0 - Gaussian resolution (FWHM)
    1 - Pulse height (x-value) of the line peak
    2 - Amplitude (y-value) of the line peak
    3 - Mean background counts per bin
    4 - Background slope (counts per bin per bin)
    5 - Low-energy tail fraction (0 <= f <= 1)
    6 - Low-energy tail scale length

    The units of 0, 1, 2, and 6 are all whatsoever units are used for pulse heights.
    """

    param_meaning = {
        "resolution": 0,
        "peak_ph": 1,
        "amplitude": 2,
        "background": 3,
        "bg_slope": 4,
        "tail_frac": 5,
        "tail_length": 6
    }
    nparam = 7

    def __init__(self):
        super(GaussianFitter, self).__init__()

    def feature_scale(self, params):
        return params[self.param_meaning["resolution"]]

    def guess_starting_params(self, data, binctrs, tailf=0.0, tailt=25.0):
        order_stat = np.array(data.cumsum(), dtype=np.float) / data.sum()

        def percentiles(p):
            return binctrs[(order_stat > p).argmax()]

        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75) - percentiles(0.25))
        res = iqr * 0.95
        # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
        baseline = max(data[0:10].mean(), 1.0/len(data))
        baseline_slope = (data[-10:].mean() - baseline) / len(data)
        ampl = (data.max() - baseline) * np.pi
        return [res, peak_loc, ampl, baseline, baseline_slope, tailf, tailt]

    def fitfunc(self, params, x):
        """Return the smeared line complex.

        Args:
            <params>  The 7 parameters of the fit (see self.__doc__ for details).
            <x>       An array of pulse heights (params will scale them to energy).
        Returns:
            The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params
        sigma = P_gaussfwhm / (8 * np.log(2))**0.5

        def cleanspectrum_fn(x):
            return np.exp(-0.5*(x-P_phpeak)**2/(sigma**2))

        spectrum = _smear_lowEtail(cleanspectrum_fn, x, 0, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def setbounds(self, params, ph):
        # bin-0 background should be bounded IF the BG slope is held
        minBG0 = self._minBG0(params, ph)

        self.bounds = []
        self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None))  # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, None))       # Amplitude
        self.bounds.append((minBG0, None))  # Background level (bin 0)
        self.bounds.append((None, None))    # Background slope (counts/bin)
        self.bounds.append((0, 1))          # Tail fraction
        self.bounds.append((0, None))       # Tail scale length

    def stepsize(self, params):
        eps = np.ones(self.nparam, dtype=float)
        eps = np.array((1e-3, params[0]/1e5, 1e-3, 1e-3, 1e-3, 1e-4, 1e-2))
        return eps


class MultiLorentzianComplexFitter(LineFitter):
    """Abstract base class for objects that can fit a spectral line complex.

    Provides methods fitfunc() and fit().  The child classes must provide:
    * a self.spect function object returning the spectrum at a given energy, and
    * a self.guess_starting_params method to return fit parameter guesses given a histogram.
    """

    param_meaning = {
        "resolution": 0,
        "peak_ph": 1,
        "dP_dE": 2,
        "amplitude": 3,
        "background": 4,
        "bg_slope": 5,
        "tail_frac": 6,
        "tail_length": 7
    }
    nparam = 8

    def __init__(self):
        super(MultiLorentzianComplexFitter, self).__init__()

    def feature_scale(self, params):
        return params[self.param_meaning["resolution"]]

    def fitfunc(self, params, x):
        """Return the smeared line complex.

        Args:
            <params>  The 8 parameters of the fit (see self.fit for details).
            <x>       An array of pulse heights (params will scale them to energy).
        Returns:
            The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_dphde, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params

        energy = (x - P_phpeak) / P_dphde + self.spect.peak_energy
        self.spect.set_gauss_fwhm(P_gaussfwhm)
        cleanspectrum_fn = self.spect.pdf
        spectrum = _smear_lowEtail(cleanspectrum_fn, energy, P_gaussfwhm, P_tailfrac, P_tailtau)
        retval = _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)
        if any(np.isnan(retval)) or any(retval < 0):
            raise ValueError
        return retval

    def stepsize(self, params):
        """Vector of the parameter step sizes for finding discrete gradient."""
        eps = np.array((1e-3, 1e-3, 1e-4, params[3]/1e4, 1e-3, 1e-3, 1e-3, 1e-1))
        return eps

    def setbounds(self, params, ph):
        # bin-0 background should be bounded IF the BG slope is held
        minBG0 = self._minBG0(params, ph)

        self.bounds = []
        self.bounds.append((0, 5*(np.max(ph)-np.min(ph))))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None))  # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, None))       # dPH/dE > 0
        self.bounds.append((0, None))       # Amplitude
        self.bounds.append((minBG0, None))  # Background level (bin 0)
        self.bounds.append((None, None))    # Background slope (counts/bin)
        self.bounds.append((0, 1))          # Tail fraction
        self.bounds.append((0, None))       # Tail scale length

    def plotFit(self, color=None, axis=None, label=""):
        """Plot the last fit and the data to which it was fit."""
        if color is None:
            color = 'blue'
        bins = self.last_fit_bins
        data = self.last_fit_contents
        ph_binsize = bins[1] - bins[0]
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            plt.xlabel('pulseheight (arb) - %s' % self.spect.name)
            plt.ylabel('counts per %.3f unit bin' % ph_binsize)
            plt.title(('resolution %.3f, amplitude %.3f, dph/de %.3f\n amp %.3f, '
                       'bg %.3f, bg_slope %.3f. T=%.3f $\\tau$=%.3f') %
                      tuple(self.last_fit_params))
        plot_as_stepped_hist(axis, data, bins, color=color)
        axis.set_xlim([bins[0] - 0.5 * ph_binsize, bins[-1] + 0.5 * ph_binsize])
        de = np.sqrt(self.last_fit_cov[0, 0])
        axis.plot(bins, self.last_fit_result, color='#666666',
                  label="%.2f +- %.2f eV %s" % (self.last_fit_params[0], de, label))
        axis.legend(loc='upper left')


class GenericKAlphaFitter(MultiLorentzianComplexFitter):
    """Fits a generic K alpha spectrum for energy shift and scale, amplitude, and resolution.
    Background level (including a fixed slope) and low-E tailing are also included.

    Note that self.tailfrac and self.tailtau are attributes that determine the starting
    guess for the fraction of events in an exponential low-energy tail and for that tail's
    exponential scale length (in eV). Change if desired.

    Need to add the self.spect instance to subclasses before initializing.
    """

    def __init__(self):
        """Set up a fitter for a K-alpha line complex

        spectrumDef -- should be mass.fluorescence_lines.MnKAlpha, or similar
            subclasses of SpectralLine.
        """
        super(GenericKAlphaFitter, self).__init__()

    def guess_starting_params(self, data, binctrs):
        """A decent heuristic for guessing the inital values, though your informed
        starting point is likely to be better than this.
        """
        if data.sum() <= 0:
            raise ValueError("This histogram has no contents")

        # Heuristic: find the Ka1 line as the peak bin, and then make
        # assumptions about the full width (from 1/4-peak to 1/4-peak) and
        # how that relates to the PH spacing between Ka1 and Ka2
        peak_val = data.max()
        peak_ph = binctrs[data.argmax()]
        lowqtr = binctrs[(data > peak_val * 0.25).argmax()]
        N = len(data)
        topqtr = binctrs[N - 1 - (data > peak_val * 0.25)[::-1].argmax()]

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
        return [res, ph_ka1, dph / dE, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]


class GenericKBetaFitter(MultiLorentzianComplexFitter):

    def __init__(self):
        """Subclasses must define a SpectralLine in self.spect
        """
        super(GenericKBetaFitter, self).__init__()

    def guess_starting_params(self, data, binctrs):
        """Hard to estimate dph/de from a K-beta line. Have to guess scale=1 and
        hope it's close enough to get convergence. Ugh!"""
        peak_ph = binctrs[data.argmax()]
        ampl = data.max() * 9.4
        res = 4.0
        if len(data) > 20:
            # Ensure baseline guess > 0 (see Issue #152). Guess at least 1 background across all bins
            baseline = max(data[0:10].mean(), 1.0/len(data))
        else:
            baseline = 0.1
        baseline_slope = 0.0
        return [res, peak_ph, 1.0, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]


class _lowZ_KAlphaFitter(GenericKAlphaFitter):
    """Overrides the starting parameter guesses, more appropriate
    for low Z where the Ka1,2 peaks can't be resolved."""

    def guess_starting_params(self, data, binctrs):
        n = data.sum()
        if n <= 0:
            raise ValueError("This histogram has no contents")
        cumdata = np.cumsum(data)
        ph_ka1 = binctrs[(cumdata * 2 > n).argmax()]
        res = 2.0
        dph_de = 1.0
        baseline, baseline_slope = 1.0, 0.0
        ampl = 4 * np.max(data)
        return [res, ph_ka1, dph_de, ampl, baseline, baseline_slope, 0.1, 25]
