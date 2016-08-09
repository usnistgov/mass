import itertools
import operator

import numpy as np
import pylab as plt

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator

try:
    import statsmodels.api as sm
except ImportError:  # On linux the name was as follows: (I guess the name is different in Anaconda python.)
    import scikits.statsmodels.api as sm
    sm.nonparametric.KDEUnivariate = sm.nonparametric.KDE

from mass.calibration.energy_calibration import STANDARD_FEATURES
import mass.calibration
from .energy_calibration import EnergyCalibration

# this file is intended to include algorithms that could be generally useful
# for calibration
# mostly they are pulled out of mass.calibration.young module


def __line_names(line_names):
    """
    takes a list of line_names, return name, energy in eV
    can also accept energies in eV directly
    return names, energies
    """
    if not len(line_names):
        return [], []

    energies = [STANDARD_FEATURES.get(name_or_energy, name_or_energy) for name_or_energy in line_names]
    names = [str(name_or_energy) for name_or_energy in line_names]
    return zip(*sorted(zip(names, energies), key=operator.itemgetter(1)))


def find_local_maxima(pulse_heights, gaussian_fwhm):
    """
    find_local_maxima(pulse_heights, gaussian_fwhm)
    pulse_heights = list of pulse heights (eg p_filt_value)
    gaussian_fwhm = fwhm of a gaussian that each pulse is smeared with, in same units as pulse heights
    smeares each pulse by a gaussian of gaussian_fhwm and finds local maxima, returns a list of
    their locations in pulse_height units, sorted by number of pulses in peak
    """
    kde = sm.nonparametric.KDEUnivariate(np.array(pulse_heights, dtype="double"))
    kde.fit(bw=gaussian_fwhm)
    x = kde.support
    y = kde.density
    flag = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    lm = np.arange(1, len(x)-1)[flag]
    lm = lm[np.argsort(-y[lm])]
    return np.array(x[lm])


def find_opt_assignment(peak_positions, line_names, nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
    """
    find_opt_assignment(peak_positions, line_names, nextra=2, nextramax=6, maxacc=0.015, nincrement=3)
    peak_positions = a list of peak locations in arb units, eg p_filt_value units
    line_names = of energies in eV, either as number, or names to be looked up in STANDARD_FEATURES
    nextra = the algorithm starts with the first len(line_names)+nextra peak_positions
    nincrement = each the algorithm fails to find a satisfatory peak assignment, it uses nincrement more lines
    nextramax = the algorithm fails with an error if the algorithm tries to use more than this many lines
    maxacc = an emprical number that determines if an assignment is good enough. the default number works reasonably well for tupac data
    takes a list of peak_positions in arb units and line names that correspond to energies
    tries to find an assignment of peaks to line names that is reasonably self consistent and smooth
    returns (energies, opt_assigned_pulseheights)
    """
    name_e, e_e = __line_names(line_names)

    n_sel_pp = len(line_names)+nextra  # number of peak_positions to use to line up to line_names
    nmax = len(line_names)+nextramax

    while n_sel_pp < nmax:
        sel_positions = np.array(peak_positions[:n_sel_pp], dtype="float")
        energies = np.array(e_e, dtype="float")
        assign = np.array(list(itertools.combinations(sel_positions, len(line_names))))
        assign.sort(axis=1)
        fracs = (energies[1:-1] - energies[:-2])/(energies[2:] - energies[:-2])
        est_pos = assign[:, :-2]*(1 - fracs) + assign[:, 2:]*fracs
        acc_est = np.linalg.norm((est_pos - assign[:, 1:-1]) /
                                 (assign[:, 2:] - assign[:, :-2]), axis=1)

        opt_assign_i = np.argmin(acc_est)
        acc = acc_est[opt_assign_i]
        opt_assign = assign[opt_assign_i]

        if acc > maxacc * np.sqrt(len(energies)):
            n_sel_pp += nincrement
            continue
        else:
            break
    else:  # if break does not occur
        raise ValueError("no succesful peak assignment found: acc %g, maxacc*sqrt(len(energies)) %g" %
                         (acc, maxacc * np.sqrt(len(energies))))

    return energies, list(opt_assign)


def build_fit_ranges_ph(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """call build_fit_ranges then convert to ph using approx_ecal
    """
    e_e, fit_lo_hi, slopes_de_dph = build_fit_ranges(line_names, excluded_line_names, approx_ecal, fit_width_ev)
    fit_lo_hi_ph = []
    for (lo, hi) in fit_lo_hi:
        lo_ph = approx_ecal.energy2ph(lo)
        hi_ph = approx_ecal.energy2ph(hi)
        fit_lo_hi_ph.append((lo_ph, hi_ph))
    return e_e, fit_lo_hi_ph, slopes_de_dph


def build_fit_ranges(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """line_names - list or line names or energies
    excluded_line_names - list of line_names or energies to avoid when making fit ranges
    approx_cal - an EnergyCalibration object containing an approximate calibration
    fit_width_ev - full size in eV of fit ranges
    returns a list of (lo,hi) where lo and hi have units of pulseheights of ranges to fit in for eacn energy in line_names
    """
    name_e, e_e = __line_names(line_names)
    excl_name_e, excl_e_e = __line_names(excluded_line_names)
    half_width_ev = fit_width_ev/2.0
    all_e = np.sort(np.hstack((e_e, excl_e_e)))
    assert(len(all_e) == len(np.unique(all_e)))
    fit_lo_hi = []
    slopes_de_dph = []

    for i in range(len(e_e)):
        e = e_e[i]
        slope_de_dph = approx_ecal.energy2dedph(e)
        half_width_ph = half_width_ev/slope_de_dph
        if any(all_e < e):
            nearest_below = all_e[all_e < e][-1]
        else:
            nearest_below = -np.inf
        if any(all_e > e):
            nearest_above = all_e[all_e > e][0]
        else:
            nearest_above = np.inf
        lo = max(e - half_width_ph, (e + nearest_below) / 2.0)
        hi = min(e + half_width_ph, (e + nearest_above) / 2.0)
        fit_lo_hi.append((lo, hi))
        slopes_de_dph.append(slope_de_dph)
    return e_e, fit_lo_hi, slopes_de_dph


class FailedFitter(object):
    def __init__(self, hist, bins):
        self.hist = hist
        self.bins = bins
        self.last_fit_params = [-1, np.sum(self.hist * bins[:-1]) / np.sum(self.hist)] + [None] * 4

    def fitfunc(self, param, x):
        self.last_fit_params = param
        return np.zeros_like(x)


def getfitter(name):
    """
    name - a name like "MnKAlpha" or "1150"
    "MnKAlpha" will return a MnKAlphaFitter
    "1150" will return a GaussianFitter
    """
    try:
        class_name = name+"Fitter"
        fitter = getattr(mass.calibration.line_fits, class_name)()
    except AttributeError:
        fitter = mass.calibration.line_fits.GaussianFitter()
    return fitter


def multifit(ph, line_names, fit_lo_hi, binsize_ev, slopes_de_dph):
    """
    ph - list of pulseheights
    line_names - as with others
    fit_lo_hi - a list of (lo,hi) with units of ph, used as edges of histograms for fitting
    binsize - binsize in eV
    slopes_de_dph - list of slopes de_dph (e in eV)
    """
    name_e, e_e = __line_names(line_names)
    fitters = []
    peak_ph = []
    eres = []

    for i in range(len(name_e)):
        lo, hi = fit_lo_hi[i]
        dP_dE = 1/slopes_de_dph[i]
        binsize_ph = binsize_ev[i]*dP_dE
        fitter = singlefit(ph, name_e[i], e_e[i], lo, hi, binsize_ph, dP_dE)
        fitters.append(fitter)
        peak_ph.append(fitter.last_fit_params[fitter.param_meaning["peak_ph"]])
        if isinstance(fitter, mass.calibration.line_fits.GaussianFitter):
            eres.append(fitter.last_fit_params[fitter.param_meaning["resolution"]])
            eres[-1] /= dP_dE  # gaussian fitter reports resolution in ph units
        else:
            eres.append(fitter.last_fit_params[fitter.param_meaning["resolution"]])
    return {"fitters": fitters, "peak_ph": peak_ph,
            "eres": eres, "line_names": name_e, "energies": e_e}


def singlefit(ph, name, energy, lo, hi, binsize_ph, approx_dP_dE):
    counts, bin_edges = np.histogram(ph, np.arange(lo, hi, binsize_ph))
    fitter = getfitter(name)
    guess_params = fitter.guess_starting_params(counts, bin_edges)
    if not isinstance(fitter, mass.calibration.line_fits.GaussianFitter):
        guess_params[fitter.param_meaning["dP_dE"]] = approx_dP_dE
        hold = [fitter.param_meaning["dP_dE"]]
    else:
        hold = []
    fitter.fit(counts, bin_edges, guess_params, plot=False, hold=hold)
    return fitter


class EnergyCalibrationAutocal(EnergyCalibration):
    def __init__(self):
        super(EnergyCalibrationAutocal, self).__init__()
        self.fitters = None
        self.energy_resolutions = None
        self.line_names = None

        self.energies_opt = None
        self.ph_opt = None
        self.fit_lo_hi = None
        self.slopes_de_dph = None

    def guess_fit_params(self, ph, line_names, smoothing_res_ph=20, binsize_ev=1.0,
                nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
        lm = find_local_maxima(ph, smoothing_res_ph)
        self.energies_opt, self.ph_opt = find_opt_assignment(lm, line_names, maxacc=maxacc)
        self.binsize_ev = [binsize_ev] * len(self.energies_opt)
        approxcal = mass.energy_calibration.EnergyCalibration(1, approximate=False)
        for (ee, phph) in zip(self.energies_opt, self.ph_opt):
            approxcal.add_cal_point(phph, ee)
        energies, self.fit_lo_hi, self.slopes_de_dph = build_fit_ranges_ph(self.energies_opt, [], approxcal, 100)

    def fit_lines(self, ph, line_names, smoothing_res_ph=20,
                nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
        mresult = multifit(ph, line_names, self.fit_lo_hi, self.binsize_ev, self.slopes_de_dph)
        for (ee, phph) in zip(mresult["energies"], mresult["peak_ph"]):
            self.add_cal_point(phph, ee)
        self.fitters = mresult["fitters"]
        self.energy_resolutions = mresult["eres"]
        self.line_names = mresult["line_names"]

    def autocal(self, *args, **kargs):
        self.guess_fit_params(*args, **kargs)
        self.fit_lines(*args, **kargs)

    def diagnose(self):
        fig = plt.figure(figsize=(16, 9))

        n = int(np.ceil(np.sqrt(len(self.line_names))))

        w, h, lm, bm, hs, vs = 0.6, 0.9, 0.05, 0.08, 0.1, 0.1
        for i, (el, fitter, eres) in enumerate(zip(self.line_names, self.fitters, self.energy_resolutions)):
            ax = fig.add_axes([w * (i % n) / n + lm,
                               h * (i // n) / n + bm,
                               (w - hs) / n,
                               (h - vs) / n])
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

            binsize = fitter.last_fit_bins[1]-fitter.last_fit_bins[0]
            bin_edges = np.linspace(fitter.last_fit_bins[0]-binsize/2.0, fitter.last_fit_bins[-1]+binsize/2.0, len(fitter.last_fit_bins)+1)
            ax.fill(np.repeat(bin_edges, 2), np.hstack([[0], np.repeat(fitter.last_fit_contents, 2), [0]]),
                    lw=1, fc=(0.3, 0.3, 0.9), ec=(0.1, 0.1, 1.0), alpha=0.8)

            x = np.linspace(fitter.last_fit_bins[0], fitter.last_fit_bins[-1], 201)
            if isinstance(fitter, mass.calibration.line_fits.GaussianFitter):
                ax.text(0.05, 0.97, str(el) +
                        ' (eV)\n' + "Resolution: {0:.1f} (eV)".format(eres),
                        transform=ax.transAxes, ha='left', va='top')
                # y = [np.median(fitter.theory_function(fitter.params, a)) for a in x]
            else:
                ax.text(0.05, 0.97, el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$') +
                        '\n' + "Resolution: {0:.1f} (eV)".format(eres),
                        transform=ax.transAxes, ha='left', va='top')
            y = fitter.fitfunc(fitter.last_fit_params, x)
            ax.plot(x, y, '-', color=(0.9, 0.1, 0.1), lw=2)
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(0, np.max(fitter.last_fit_contents) * 1.3)

        ax = fig.add_axes([lm + w, bm, (1.0 - lm - w) - 0.06, h - 0.05])

        for el, pht, fitter, energy in zip(self.line_names, self._ph,
                                           self.fitters, self._energies):
            peak_name = 'Unknown'
            if isinstance(el, str):
                peak_name = el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$')
            elif isinstance(el, int) or isinstance(el, float):
                peak_name = "{0:.1f} (eV)".format(energy)
            ax.text(pht, energy,
                    peak_name,
                    ha='left', va='top',
                    transform=ax.transData + mtrans.ScaledTranslation(5.0 / 72, -12.0 / 72, fig.dpi_scale_trans))

        ax.scatter(self._ph,
                   self._energies, s=36, c=(0.2, 0.2, 0.8))

        lb = np.amin(self._ph)
        ub = np.amax(self._ph)
        
        width = ub - lb
        x = np.linspace(lb - width / 10, ub + width / 10, 101)
        y = self(x)
        ax.plot(x, y, '--', color='orange', lw=2, zorder=-2)

        ax.yaxis.set_tick_params(labelleft=False, labelright=True)
        ax.yaxis.set_label_position('right')

        ax.set_xlabel('Pulse height')
        ax.set_ylabel('Energy (eV)')

        ax.set_xlim(lb - width / 10, ub + width / 10)

        fig.show()
