"""
This file is intended to include algorithms that could be generally useful
for calibration. Mostly they are pulled out of the former
mass.calibration.young module.
"""

import itertools
import operator
import numpy as np

from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator

from ..common import isstr
from mass.calibration.energy_calibration import STANDARD_FEATURES
import mass.calibration


def line_names_and_energies(line_names):
    """Given a list of line_names, return (names, energies) in eV.

    Can also accept energies in eV directly and return (names, energies).
    """
    if len(line_names) <= 0:
        return [], []

    energies = [STANDARD_FEATURES.get(name_or_energy, name_or_energy)
                for name_or_energy in line_names]
    # names = [str(name_or_energy) for name_or_energy in line_names]
    return zip(*sorted(zip(line_names, energies), key=operator.itemgetter(1)))


def find_local_maxima(pulse_heights, gaussian_fwhm):
    """Smears each pulse by a gaussian of gaussian_fhwm and finds local maxima,
    returns a list of their locations in pulse_height units (sorted by number of
    pulses in peak) AND their peak values as: (peak_locations, peak_intensities)

    Args:
        pulse_heights (np.array(dtype=float)): a list of pulse heights (eg p_filt_value)
        gaussian_fwhm = fwhm of a gaussian that each pulse is smeared with, in same units as pulse heights
    """
    # kernel density estimation (with a gaussian kernel)
    n = 128 * 1024
    gaussian_fwhm = float(gaussian_fwhm)
    # The above ensures that lo & hi are floats, so that (lo-hi)/n is always a float in python2
    sigma = gaussian_fwhm / (np.sqrt(np.log(2) * 2) * 2)
    tbw = 1.0 / sigma / (np.pi * 2)
    lo = np.min(pulse_heights) - 3 * gaussian_fwhm
    hi = np.max(pulse_heights) + 3 * gaussian_fwhm
    hist, bins = np.histogram(pulse_heights, np.linspace(lo, hi, n + 1))
    tx = np.fft.rfftfreq(n, (lo - hi) / n)
    ty = np.exp(-tx**2 / 2 / tbw**2)
    x = (bins[1:] + bins[:-1]) / 2
    y = np.fft.irfft(np.fft.rfft(hist) * ty)

    flag = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    lm = np.arange(1, n - 1)[flag]
    lm = lm[np.argsort(-y[lm])]

    return np.array(x[lm]), np.array(y[lm])


def find_opt_assignment(peak_positions, line_names, nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
    """Tries to find an assignment of peaks to line names that is reasonably self consistent and smooth

    Args:
        peak_positions (np.array(dtype=float)): a list of peak locations in arb units,
            e.g. p_filt_value units
        line_names (list[str or float)]): a list of calibration lines either as number (which is
            energies in eV), or name to be looked up in STANDARD_FEATURES
        nextra (int): the algorithm starts with the first len(line_names) + nextra peak_positions
        nincrement (int): each the algorithm fails to find a satisfactory peak assignment, it uses
            nincrement more lines
        nextramax (int): the algorithm stops incrementint nextra past this value, instead
            failing with a ValueError saying "no peak assignment succeeded"
        maxacc (float): an empirical number that determines if an assignment is good enough.
            The default number works reasonably well for tupac data
    """
    name_e, e_e = line_names_and_energies(line_names)

    n_sel_pp = len(line_names) + nextra  # number of peak_positions to use to line up to line_names
    nmax = len(line_names) + nextramax

    while True:
        sel_positions = np.asarray(peak_positions[:n_sel_pp], dtype="float")
        energies = np.asarray(e_e, dtype="float")
        assign = np.array(list(itertools.combinations(sel_positions, len(line_names))))
        assign.sort(axis=1)
        fracs = np.divide(energies[1:-1] - energies[:-2], energies[2:] - energies[:-2])
        est_pos = assign[:, :-2] * (1 - fracs) + assign[:, 2:] * fracs
        acc_est = np.linalg.norm(np.divide(est_pos - assign[:, 1:-1],
                                           assign[:, 2:] - assign[:, :-2]), axis=1)

        opt_assign_i = np.argmin(acc_est)
        acc = acc_est[opt_assign_i]
        opt_assign = assign[opt_assign_i]

        if acc > maxacc * np.sqrt(len(energies)):
            n_sel_pp += nincrement
            if n_sel_pp > nmax:
                msg = f"no peak assignment succeeded: acc {acc:g}, maxacc*sqrt(len(energies)) "\
                    f"{maxacc * np.sqrt(len(energies)):g}"
                raise ValueError(msg)
            else:
                continue
        else:
            return name_e, energies, list(opt_assign)


def build_fit_ranges_ph(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """Call build_fit_ranges() to get (lo,hi) for fitranges in energy units,
    then convert to ph using approx_ecal"""
    e_e, fit_lo_hi_energy, slopes_de_dph = build_fit_ranges(
        line_names, excluded_line_names, approx_ecal, fit_width_ev)
    fit_lo_hi_ph = []
    for lo, hi in fit_lo_hi_energy:
        lo_ph = approx_ecal.energy2ph(lo)
        hi_ph = approx_ecal.energy2ph(hi)
        fit_lo_hi_ph.append((lo_ph, hi_ph))

    return e_e, fit_lo_hi_ph, slopes_de_dph


def build_fit_ranges(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """Returns a list of (lo,hi) where lo and hi have units of energy of
    ranges to fit in for each energy in line_names.

    Args:
        line_names (list[str or float]): list or line names or energies
        excluded_line_names (list[str or float]): list of line_names or energies to
            avoid when making fit ranges
        approx_ecal: an EnergyCalibration object containing an approximate calibration
        fit_width_ev (float): full size in eV of fit ranges
    """
    _names, e_e = line_names_and_energies(line_names)
    _excl_names, excl_e_e = line_names_and_energies(excluded_line_names)
    half_width_ev = fit_width_ev / 2.0
    all_e = np.sort(np.hstack((e_e, excl_e_e)))
    assert (len(all_e) == len(np.unique(all_e)))
    fit_lo_hi_energy = []
    slopes_de_dph = []

    for e in e_e:
        slope_de_dph = approx_ecal.energy2dedph(e)
        if any(all_e < e):
            nearest_below = all_e[all_e < e][-1]
        else:
            nearest_below = -np.inf
        if any(all_e > e):
            nearest_above = all_e[all_e > e][0]
        else:
            nearest_above = np.inf
        lo = max(e - half_width_ev, (e + nearest_below) / 2.0)
        hi = min(e + half_width_ev, (e + nearest_above) / 2.0)
        fit_lo_hi_energy.append((lo, hi))
        slopes_de_dph.append(slope_de_dph)

    return e_e, fit_lo_hi_energy, slopes_de_dph


class FailedFit:
    def __init__(self, hist, bins):
        self.hist = hist
        self.bins = bins


class FailedToGetModelException(Exception):
    pass


def get_model(lineNameOrEnergy, has_linear_background=True, has_tails=False):
    if isinstance(lineNameOrEnergy, mass.GenericLineModel):
        line = lineNameOrEnergy.spect
    elif isinstance(lineNameOrEnergy, mass.SpectralLine):
        line = lineNameOrEnergy
    elif isinstance(lineNameOrEnergy, str):
        if lineNameOrEnergy in mass.spectra:
            line = mass.spectra[lineNameOrEnergy]
        elif lineNameOrEnergy in mass.STANDARD_FEATURES:
            energy = mass.STANDARD_FEATURES[lineNameOrEnergy]
            line = mass.SpectralLine.quick_monochromatic_line(lineNameOrEnergy, energy, 0.001, 0)
        else:
            raise FailedToGetModelException(
                f"failed to get line from lineNameOrEnergy={lineNameOrEnergy}")
    else:
        try:
            energy = float(lineNameOrEnergy)
        except Exception:
            raise FailedToGetModelException(
                f"lineNameOrEnergy = {lineNameOrEnergy} is not convertable to float or "
                "a str in mass.spectra or mass.STANDARD_FEATURES")
        line = mass.SpectralLine.quick_monochromatic_line(
            f"{lineNameOrEnergy}eV", float(lineNameOrEnergy), 0.001, 0)
    return line.model(has_linear_background=has_linear_background, has_tails=has_tails)


# support both names as they were both used historically
getmodel = get_model


def multifit(ph, line_names, fit_lo_hi, binsize_ev, slopes_de_dph, hide_deprecation=False):
    """
    Args:
        ph (np.array(dtype=float)): list of pulse heights
        line_names: names of calibration  lines
        fit_lo_hi (list[list[float]]): a list of (lo,hi) with units of ph, used as
            edges of histograms for fitting
        binsize_ev (list[float]): list of binsizes in eV for calibration lines
        slopes_de_dph (list[float]): - list of slopes de_dph (e in eV)
        hide_deprecation: whether to suppress deprecation warnings
    """
    name_e, e_e = line_names_and_energies(line_names)
    results = []
    peak_ph = []
    eres = []

    for i, name in enumerate(name_e):
        lo, hi = fit_lo_hi[i]
        dP_dE = 1 / slopes_de_dph[i]
        binsize_ph = binsize_ev[i] * dP_dE
        result = singlefit(ph, name, lo, hi, binsize_ph, dP_dE)
        results.append(result)
        peak_ph.append(result.best_values["peak_ph"])
        eres.append(result.best_values["fwhm"])
    return {"results": results, "peak_ph": peak_ph,
            "eres": eres, "line_names": name_e, "energies": e_e}


def singlefit(ph, name, lo, hi, binsize_ph, approx_dP_dE):
    nbins = (hi - lo) / binsize_ph
    if nbins > 5000:
        raise Exception("too damn many bins, dont like running out of memory")
    counts, bin_edges = np.histogram(ph, np.arange(lo, hi, binsize_ph))
    e = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    model = getmodel(name)
    guess_params = model.guess(counts, bin_centers=e, dph_de=approx_dP_dE)
    if "Gaussian" not in model.name:
        guess_params["dph_de"].set(approx_dP_dE, vary=False)
    result = model.fit(counts, guess_params, bin_centers=e, minimum_bins_per_fwhm=1.5)
    result.energies = e
    return result


class EnergyCalibrationAutocal:
    def __init__(self, ph=None, line_names=None):
        """
        Args:
            line_names (list[str]): names of calibration lines. Names doesn't need to be
            ordered in their energies.
        """
        self.cal_factory = mass.EnergyCalibrationMaker.init()
        self.results = None
        self.energy_resolutions = None
        self.line_names = line_names

        self.energies_opt = None
        self.ph_opt = None
        self.fit_range_ev = None
        self.fit_lo_hi = None
        self.slopes_de_dph = None

        self.binsize_ev = None
        self.ph = ph
        self._hide_deprecation = False

    def guess_fit_params(self, smoothing_res_ph=20, fit_range_ev=200.0, binsize_ev=1.0,
                         nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
        """Calculate reasonable parameters for complex models or Gaussian models.

         Args:
             binsize_ev (float or list[float]): bin sizes of the histograms of given calibration lines.
                 If a single number is given, this same number will be used for all calibration lines.
        """
        lm, _ = find_local_maxima(self.ph, smoothing_res_ph)

        # Note that find_opt_assignment does not require line_names be sorted by energies.
        self.line_names, self.energies_opt, self.ph_opt = find_opt_assignment(
            lm, self.line_names, nextra, nincrement, nextramax, maxacc=maxacc)

        if isinstance(binsize_ev, Iterable):
            self.binsize_ev = binsize_ev
        else:
            self.binsize_ev = [binsize_ev] * len(self.energies_opt)

        approx_cal = mass.EnergyCalibrationMaker.init(self.ph_opt, self.energies_opt).make_calibration()
        self.fit_range_ev = fit_range_ev

        #  Default fit range width is 100 eV for each line.
        #  But you can customize these numbers after self.guess_fit_params is finished.
        #  New self.fit_lo_hi values will be in self.fit_lines in the next step.
        _, self.fit_lo_hi, self.slopes_de_dph = build_fit_ranges_ph(self.energies_opt, [],
                                                                    approx_cal, self.fit_range_ev)

    def fit_lines(self):
        """All calibration emission lines are fitted with appropriate model.
        self.line_names will be sored by energy after this method is finished.
        """
        mresult = multifit(self.ph, self.line_names, self.fit_lo_hi,
                           self.binsize_ev, self.slopes_de_dph, hide_deprecation=self._hide_deprecation)

        self.cal_factory = mass.EnergyCalibrationMaker.init(
            mresult["peak_ph"], mresult["energies"], names=mresult['line_names'])
        self.results = mresult["results"]
        self.energy_resolutions = mresult["eres"]
        self.line_names = mresult["line_names"]

        return self.cal_factory

    def autocal(self, smoothing_res_ph=20, fit_range_ev=200.0, binsize_ev=1.0,
                nextra=2, nincrement=3, nextramax=8, maxacc=0.015):
        self.guess_fit_params(smoothing_res_ph, fit_range_ev, binsize_ev, nextra, nincrement,
                              nextramax, maxacc)
        return self.fit_lines()

    @property
    def anyfailed(self):
        return any([isinstance(cf, FailedFit) for cf in self.results])

    def diagnose(self):
        fig = plt.figure(figsize=(16, 9))

        n = int(np.ceil(np.sqrt(len(self.line_names))))

        w, h, lm, bm, hs, vs = 0.6, 0.9, 0.05, 0.08, 0.1, 0.1
        for i, (el, result, eres) in enumerate(zip(self.line_names, self.results, self.energy_resolutions)):
            ax = fig.add_axes([w * (i % n) / n + lm,
                               h * (i // n) / n + bm,
                               (w - hs) / n,
                               (h - vs) / n])
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

            binsize = result.energies[1] - result.energies[0]
            bin_edges = np.linspace(result.energies[0] - binsize / 2.0,
                                    result.energies[-1] + binsize / 2.0, len(result.energies) + 1)
            ax.fill(np.repeat(bin_edges, 2), np.hstack([[0], np.repeat(result.data, 2), [0]]),
                    lw=1, fc="#4c4ce6", ec="#1a1aff", alpha=0.8)

            x = np.linspace(result.energies[0], result.energies[-1], 201)
            if "Gaussian" in result.model.name:
                ax.text(0.05, 0.97, str(el)
                        + ' (eV)\n' + f"Resolution: {eres:.1f} (eV)",
                        transform=ax.transAxes, ha='left', va='top')
            else:
                ax.text(0.05, 0.97, el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$')
                        + '\n' + f"Resolution: {eres:.1f} (eV)",
                        transform=ax.transAxes, ha='left', va='top')
            y = result.model.eval(result.params, bin_centers=x)
            ax.plot(x, y, '-', color=(0.9, 0.1, 0.1), lw=2)
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(0, np.max(result.data) * 1.3)

        ax = fig.add_axes([lm + w, bm, (1.0 - lm - w) - 0.06, h - 0.05])

        for el, pht, energy in zip(self.line_names, self.cal_factory.ph,
                                   self.cal_factory.energy):
            peak_name = 'Unknown'
            if isstr(el):
                peak_name = el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$')
            elif isinstance(el, (int, float)):
                peak_name = f"{energy:.1f} (eV)"
            ax.text(pht, energy,
                    peak_name,
                    ha='left', va='top',
                    transform=ax.transData + mtrans.ScaledTranslation(5.0 / 72, -12.0 / 72,
                                                                      fig.dpi_scale_trans))

        ax.scatter(self.cal_factory.ph,
                   self.cal_factory.energy, s=36, c="#3333cc")

        lb = np.amin(self.cal_factory.ph)
        ub = np.amax(self.cal_factory.ph)

        width = ub - lb
        x = np.linspace(lb - width / 10, ub + width / 10, 101)
        cal = self.cal_factory.make_calibration()
        y = cal(x)
        ax.plot(x, y, '--', color='orange', lw=2, zorder=-2)

        ax.yaxis.set_tick_params(labelleft=False, labelright=True)
        ax.yaxis.set_label_position('right')

        ax.set_xlabel('Pulse height')
        ax.set_ylabel('Energy (eV)')

        ax.set_xlim(lb - width / 10, ub + width / 10)
