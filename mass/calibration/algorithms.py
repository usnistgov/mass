import scipy
import numpy as np
import pylab as plt
try:
    import statsmodels.api as sm
except ImportError:  # On linux the name was as follows: (I guess the name is different in Anaconda python.)
    import scikits.statsmodels.api as sm
    sm.nonparametric.KDEUnivariate = sm.nonparametric.KDE
from mass.calibration.energy_calibration import STANDARD_FEATURES
import mass.calibration
import operator
import itertools
# this file is intended to include algorithms that could be generally useful
# for calibration
# mostly they are pulled out of young.py


def __line_names(line_names):
    """
    takes a list of line_names, return name, energy in eV
    can also accept energies in eV directly
    return names, energies
    """
    energies = [STANDARD_FEATURES.get(name_or_energy, name_or_energy) for name_or_energy in line_names]
    names = [str(name_or_energy) for name_or_energy in line_names]
    sort_ind = np.argsort(energies)
    energies = [energies[i] for i in sort_ind]
    names = [names[i] for i in sort_ind]
    return names, energies

def find_local_maxima(pulse_heights, gaussian_fwhm):
    """
    find_local_maxima(pulse_heights, gaussian_fwhm)
    pulse_heights = list of pulse heights (eg p_filt_value)
    gaussian_fwhm = fwhm of a gaussian that each pulse is smeared with, in same units as pulse heights
    smeares each pulse by a gaussian of gaussian_fhwm and finds local maxima, returns a list of
    their locations in pulse_height units, sorted by number of pulses in peak
    """
    kde = sm.nonparametric.KDEUnivariate(np.array(pulse_heights,dtype="double"))
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

    n_sel_pp = len(line_names)+nextra # number of peak_positions to use to line up to line_names
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
    else: # if break does not occur
        raise ValueError("no succesful peak assignment found: acc %g, maxacc*sqrt(len(energies)) %g"%(acc, maxacc * np.sqrt(len(energies))))

    return energies, list(opt_assign)

def build_fit_ranges(line_names, excluded_line_names, approx_ecal, fit_width_ev):
    """
    line_names - list or line names or energies
    excluded_line_names - list of line_names or energies to avoid when making fit ranges
    approx_cal - an EnergyCalibration object containing an approximate calibration
    fit_width_ev - full size in eV of fit ranges
    returns a list of (lo,hi) where lo and hi have units of pulseheights of ranges to fit in for eacn energy in line_names
    """
    name_e, e_e = __line_names(line_names)
    excl_name_e, excl_e_e = __line_names(excluded_line_names)
    half_width_ev = fit_width_ev/2.0
    all_e = np.sort(np.hstack((e_e, excl_e_e)))
    assert(len(all_e)==len(np.unique(all_e)))
    fit_lo_hi = []
    for i in xrange(len(e_e)):
        e = e_e[i]
        slope_de_dph = approx_ecal.energy2dedph(e)
        half_width_ph = half_width_ev/slope_de_dph
        if any(all_e<e):
            nearest_below = all_e[all_e<e][-1]
        else:
            nearest_below = -np.inf
        if any(all_e>e):
            nearest_above = all_e[all_e>e][0]
        else:
            nearest_above = np.inf
        lo = max(e-half_width_ph, (e+nearest_below)/2.0)
        hi = min(e+half_width_ph, (e+nearest_above)/2.0)
        fit_lo_hi.append((lo,hi))
    return e_e, fit_lo_hi
