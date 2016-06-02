"""
line_fits.py

Addded Joe Fowler 5 May, 2016

Separated line fits (here) from the line shapes (still in fluorescence_lines.py)
"""

import numpy as np
import pylab as plt

from mass.mathstat.fitting import MaximumLikelihoodHistogramFitter
from mass.mathstat.utilities import plot_as_stepped_hist
from mass.mathstat.special import voigt
from . import fluorescence_lines as lines


def _smear_lowEtail(cleanspectrum_fn, x, P_resolution, P_tailfrac, P_tailtau ):
    """Evaluate cleanspectrum_fn(x), but padded and smeared to add a low-E tail."""
    if P_tailfrac <= 1e-5:
        return cleanspectrum_fn(x)

    # Compute the low-E-tailed spectrum. This is done by
    # convolution, which is computed using DFT methods.
    # A wider energy range must be used, or wrap-around effects of
    # tails will corrupt the model.
    dx = x[1] - x[0]
    nlow = int(min(P_tailtau, 100) * 6 / dx + 0.5)
    nhi = int((P_resolution + min(P_tailtau, 50)) / dx + 0.5)
    nhi = min(3000, nhi)  # A practical limit
    nlow = max(nlow, nhi)
    lowx = np.arange(-nlow, 0) * dx + x[0]
    highx = np.arange(1, nhi + 1) * dx + x[-1]
    x_wide = np.hstack([lowx, x, highx])
    freq = np.fft.rfftfreq(len(x_wide), d=dx)
    rawspectrum = cleanspectrum_fn(x_wide)
    ft = np.fft.rfft(rawspectrum)
    ft += ft * P_tailfrac * (1.0 / (1 - 2j * np.pi * freq * P_tailtau) - 1)
    smoothspectrum = np.fft.irfft(ft, n=len(x_wide))
    return smoothspectrum[nlow:nlow + len(x)]


def _scale_add_bg(spectrum, P_amplitude, P_bg=0, P_bgslope=0):
    "Scale a spectrum and add a constant+slope background."
    spectrum = spectrum * P_amplitude + P_bg
    if P_bgslope != 0:
        spectrum += P_bgslope * np.arange(len(spectrum))
    return spectrum


class LineFitter(object):
    """Abstract base class for line fitting objects.
    """
    def __init__(self):
        """ """
        # Parameters+covariance from last successful fit
        self.last_fit_params = None
        self.last_fit_cov = None
        # Fit function samples from last successful fit
        self.last_fit_result = None
        self.tailfrac = 0.0
        self.tailtau = 25
        # Whether pulse heights are necessarily non-negative.
        self.phscale_positive = True

    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None,
            color=None, label=True, vary_resolution=True, vary_bg=True,
            vary_bg_slope=False, vary_tail=False, hold=None, verbose=False):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the
        set of histogram bins <pulseheights>.

        pulseheights -- the histogram bin centers or bin edges.

        params: see self.__doc__, because the group of parameters and their numbering
                depends on the exact line shape being fit.

        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)

        vary_resolution Whether to let the Gaussian resolution vary in the fit
        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        vary_tail --   Whether to let a low-energy exponential tail to vary.
        hold:          A sequence of parameter numbers to keep fixed.  Resolution, BG
                       BG slope, or tail will be held if relevant parameter number
                       appears in the hold sequence OR if relevant boolean vary_* tests False.

        returns (fitparams, covariance)
        fitparams has same format as input variable params
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

        if hold is None:
            hold = set([])
        else:
            hold = set(hold)
        if not vary_resolution:
            hold.add(self.param_meaning["resolution"])
        if not vary_bg:
            hold.add(self.param_meaning["background"])
        if not vary_bg_slope:
            hold.add(self.param_meaning["bg_slope"])
        if not vary_tail:
            hold.add(self.param_meaning["tail_frac"])
            hold.add(self.param_meaning["tail_length"])

        try:
            len(params) == self.nparam
        except:
            params = self.guess_starting_params(data, pulseheights)

        if plot:
            if color is None:
                color = 'blue'
            if axis is None:
                plt.clf()
                axis = plt.subplot(111)

            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            ph_binsize = pulseheights[1] - pulseheights[0]
            axis.set_xlim([pulseheights[0] - 0.5 * ph_binsize, pulseheights[-1] + 0.5 * ph_binsize])

        # Max-likelihood histogram fitter
        epsilon = self.stepsize(params)
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params,
                                                  self.fitfunc, TOL=1e-4, epsilon=epsilon)
        self.setbounds(params, pulseheights)
        for i, b in enumerate(self.bounds):
            fitter.setbounds(i, b[0], b[1])

        for h in hold:
            fitter.hold(h)

        fitparams, covariance = fitter.fit(verbose=verbose)

        self.last_fit_params = fitparams
        self.last_fit_cov = covariance
        self.last_fit_chisq = fitter.chisq
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        self.last_fit_bins = pulseheights.copy()
        self.last_fit_contents = data.copy()

        if plot:
            pnum_res = self.param_meaning["resolution"]
            slabel = ""
            if label and pnum_res not in hold:
                pnum_tf = self.param_meaning["tail_frac"]
                res = fitparams[pnum_res]
                err = covariance[pnum_res,pnum_res]**0.5
                tf = fitparams[pnum_tf]
                slabel = "FWHM: %.2f +- %.2f" % (res, err)
                if tf > 0.001:
                    slabel += "\nf$_\\mathrm{tail}$: %.1f%%"%(tf*100)
            axis.plot(pulseheights, self.last_fit_result, color='#666666',
                      label=slabel)
            if len(slabel) > 0:
                axis.legend(loc='upper left')
        return fitparams, covariance


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
        "amplitude":3,
        "background": 4,
        "bg_slope": 5,
        "tail_frac": 6,
        "tail_length": 7
    }
    nparam = 8

    def __init__(self):
        super( VoigtFitter, self ).__init__()

    def guess_starting_params(self, data, binctrs, tailf=0.0, tailt=25.0):
        order_stat = np.array(data.cumsum(), dtype=np.float) / data.sum()
        percentiles = lambda p: binctrs[(order_stat > p).argmax()]
        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75) - percentiles(0.25))
        res = iqr * 0.7
        lor_fwhm = res
        baseline = data[0:10].mean()
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
        Returns:  The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_lorenzfwhm, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params
        sigma = P_gaussfwhm / (8 * np.log(2))**0.5
        lorentz_hwhm = P_lorenzfwhm*0.5
        cleanspectrum_fn = lambda x: voigt(x, params[1], lorentz_hwhm, sigma)
        spectrum = _smear_lowEtail(cleanspectrum_fn, x, P_gaussfwhm, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def setbounds(self, params, ph):
        self.bounds = []
        DE = 10*(np.max(ph)-np.min(ph))
        self.bounds.append((0,10*DE))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None)) # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, 10*DE)) # Lorentz FWHM
        self.bounds.append((0, None))   # Amplitude
        self.bounds.append((0, None))   # Background level
        self.bounds.append((None, None))# Background slope
        self.bounds.append((0, 1))      # Tail fraction
        self.bounds.append((0, None))   # Tail scale length

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
        super( NVoigtFitter, self ).__init__()
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
            self.param_meaning["peak_ph%d"%j] = i*3+1
            self.param_meaning["lorentz_fwhm%d"%j] = i*3+2
            self.param_meaning["amplitude%d"%j] = i*3+3

    def guess_starting_params(self, data, binctrs):
        raise NotImplementedError("I don't know how to guess starting parameters for a %d-peak Voigt."%self.Nlines)

    # Compute the smeared line value.
    #
    # @param params  The parameters of the fit (see self.fit for details).
    # @param x       An array of pulse heights (params will scale them to energy).
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 3N+4 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
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
        self.bounds = []
        DE = 10*(np.max(ph)-np.min(ph))
        self.bounds.append((0,10*DE))  # Gauss FWHM
        for _ in range(self.Nlines):
            self.bounds.append((np.min(ph), np.max(ph)))
            self.bounds.append((0, 10*DE)) # Lorentz FWHM
            self.bounds.append((0, None))   # Amplitude
        self.bounds.append((0, None))   # Background level
        self.bounds.append((None, None))# Background slope
        self.bounds.append((0, 1))      # Tail fraction
        self.bounds.append((0, None))   # Tail scale length

    def stepsize(self, params):
        epsilon = np.copy(params)
        epsilon[0] = 1e-3
        epsilon[-4] = 1e-3 # BG level
        epsilon[-3] = 1e-3 # BG slope
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
        "amplitude":2,
        "background": 3,
        "bg_slope": 4,
        "tail_frac": 5,
        "tail_length": 6
    }
    nparam = 7

    def __init__(self):
        super( GaussianFitter, self ).__init__()

    def guess_starting_params(self, data, binctrs, tailf=0.0, tailt=25.0):
        order_stat = np.array(data.cumsum(), dtype=np.float) / data.sum()
        percentiles = lambda p: binctrs[(order_stat > p).argmax()]
        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75) - percentiles(0.25))
        res = iqr * 0.95
        baseline = data[0:10].mean()
        baseline_slope = (data[-10:].mean() - baseline) / len(data)
        ampl = (data.max() - baseline) * np.pi
        return [res, peak_loc, ampl, baseline, baseline_slope, tailf, tailt]

    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 7 parameters of the fit (see self.__doc__ for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params
        sigma = P_gaussfwhm / (8 * np.log(2))**0.5
        lorentz_hwhm = 0
        cleanspectrum_fn = lambda x: np.exp(-0.5*(x-params[1])**2/(sigma**2))
        spectrum = _smear_lowEtail(cleanspectrum_fn, x, P_gaussfwhm, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def setbounds(self, params, ph):
        self.bounds = []
        DE = 10*(np.max(ph)-np.min(ph))
        self.bounds.append((0,10*DE))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None)) # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, None))   # Amplitude
        self.bounds.append((0, None))   # Background level
        self.bounds.append((None, None))# Background slope
        self.bounds.append((0, 1))      # Tail fraction
        self.bounds.append((0, None))   # Tail scale length

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
        "dP_dE":2,
        "amplitude":3,
        "background": 4,
        "bg_slope": 5,
        "tail_frac": 6,
        "tail_length": 7
    }
    nparam = 8

    def __init__(self):
        super( MultiLorentzianComplexFitter, self ).__init__()

    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 8 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_dphde, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params

        energy = (x - P_phpeak) / P_dphde + self.spect.peak_energy
        self.spect.set_gauss_fwhm(P_gaussfwhm)
        cleanspectrum_fn = self.spect.pdf
        spectrum = _smear_lowEtail(cleanspectrum_fn, energy, P_gaussfwhm, P_tailfrac, P_tailtau)
        return _scale_add_bg(spectrum, P_amplitude, P_bg, P_bgslope)

    def stepsize(self, params):
        """Vector of the parameter step sizes for finding discrete gradient."""
        eps = np.array((1e-3, 1e-3, 1e-4, params[3]/1e4, 1e-3, 1e-3, 1e-3, 1e-1))
        return eps

    def setbounds(self, params, ph):
        self.bounds = []
        DE = 10*(np.max(ph)-np.min(ph))
        self.bounds.append((0,10*DE))  # Gauss FWHM
        if self.phscale_positive:
            self.bounds.append((0, None)) # PH Center
        else:
            self.bounds.append((None, None))
        self.bounds.append((0, None)) # dPH/dE > 0
        self.bounds.append((0, None))   # Amplitude
        self.bounds.append((0, None))   # Background level
        self.bounds.append((None, None))# Background slope
        self.bounds.append((0, 1))      # Tail fraction
        self.bounds.append((0, 10*DE))   # Tail scale length

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
    """
    Fits a generic K alpha spectrum for energy shift and scale, amplitude, and resolution.
    Background level (including a fixed slope) and low-E tailing are also included.

    Note that self.tailfrac and self.tailtau are attributes that determine the starting
    guess for the fraction of events in an exponential low-energy tail and for that tail's
    exponential scale length (in eV). Change if desired.
    """

    def __init__(self, spectrumDef):
        """Set up a fitter for a K-alpha line complex

        spectrumDef -- should be mass.fluorescence_lines.MnKAlpha, or similar
            subclasses of SpectralLine.
        """
        self.spect = spectrumDef
        super( GenericKAlphaFitter, self ).__init__()

    def guess_starting_params(self, data, binctrs):
        """
        A decent heuristic for guessing the inital values, though your informed
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
            baseline = data[0:10].mean() + 1e-6
        else:
            baseline = 0.1
        baseline_slope = 0.0
        return [res, ph_ka1, dph / dE, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]


class GenericKBetaFitter(MultiLorentzianComplexFitter):

    def __init__(self, spectrumDef):
        """
        Constructor argument spectrumDef should be mass.fluorescence_lines.MnKBeta, or similar
        subclasses of SpectralLine.
        """
        self.spect = spectrumDef
        super(GenericKBetaFitter, self).__init__()

    def guess_starting_params(self, data, binctrs):
        """Hard to estimate dph/de from a K-beta line. Have to guess scale=1 and
        hope it's close enough to get convergence. Ugh!"""
        peak_ph = binctrs[data.argmax()]
        ampl = data.max() * 9.4
        res = 4.0
        if len(data) > 20:
            baseline = data[0:10].mean()
        else:
            baseline = 0.1
        baseline_slope = 0.0
        return [res, peak_ph, 1.0, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]


# create specific KAlpha Fitters
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


class AlKAlphaFitter(_lowZ_KAlphaFitter):

    def __init__(self):
        super( AlKAlphaFitter, self ).__init__(self, lines.AlKAlpha())


class MgKAlphaFitter(_lowZ_KAlphaFitter):

    def __init__(self):
        super( MgKAlphaFitter, self ).__init__(self, lines.MgKAlpha())


class ScKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.ScKAlpha())


class TiKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.TiKAlpha())


class VKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.VKAlpha())


class CrKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.CrKAlpha())


class MnKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.MnKAlpha())


class FeKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.FeKAlpha())


class CoKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.CoKAlpha())


class NiKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.NiKAlpha())


class CuKAlphaFitter(GenericKAlphaFitter):

    def __init__(self):
        GenericKAlphaFitter.__init__(self, lines.CuKAlpha())


class TiKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.TiKBeta())


class VKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.VKBeta())


class CrKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.CrKBeta())


class MnKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.MnKBeta())


class FeKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.FeKBeta())


class CoKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.CoKBeta())


class NiKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.NiKBeta())


class CuKBetaFitter(GenericKBetaFitter):

    def __init__(self):
        GenericKBetaFitter.__init__(self, lines.CuKBeta())
