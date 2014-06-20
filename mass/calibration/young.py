from collections import Counter
import itertools
import operator
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.patheffects as patheffects
from sklearn.cluster import DBSCAN
from mass.calibration.energy_calibration import STANDARD_FEATURES


def separate_spectrum(data, elements, eps=5, mcs=100, hw=200, excl=()):
    """
    Identify the emission lines in a spectrum.

    Given the set of numbers representing energies of pulses and the list of names of emission lines.

    Parameters
    ----------
    data : array_like
        Array of floating numbers representing energies of pulses. These numbers are expected in an approximately
        locally linear but arbitrary scale against the energies.
    elements : list
        Names of emission lines to be identified for the calibration.
    eps : float
        Cluster size parameter for the DBSCAN algorithm
    mcs : integer
        Minimum number of pulses of a cluster to be included in the calculation
    hw : integer
        The width of the histogram for each element in the eV unit
    excl : list
        Names of emission lines that possibly be present in the spectrum but should be excluded from histograms

    Returns
    -------
        Dict which has names of emission lines as keys and tuples of histograms and estimated slopes as values
        Histograms are tuples of counts (length of 100) and bins (length of 101).
    """

    dbs = DBSCAN(eps=eps)
    dbs.fit(data[:, np.newaxis])

    count = Counter(dbs.labels_)
    peak_positions = []

    for l, c in count.most_common():
        # Outlier cluster won't count.
        if l < -0.5:
            continue
        # Not bright enough cluster won't count either.
        if c < mcs:
            break

        peak_positions.append(np.average(data[dbs.labels_ == l]))

    if len(peak_positions) < len(elements):
        print 'Not enough clusters.'
        return None

    peak_positions = np.array(peak_positions)

    name_e, e_e = zip(*sorted([[element, STANDARD_FEATURES[element]] for element in elements],
                              key=operator.itemgetter(1)))

    # Exhaustive search for the best assignment.
    lh_results = []

    for assign in itertools.combinations(peak_positions, len(elements)):
        assign = sorted(assign)

        acc_est = 0.0
        for i in xrange(len(assign) - 2):
            est = assign[i] + (assign[i + 2] - assign[i]) * (e_e[i + 1] - e_e[i]) / (e_e[i+2] - e_e[i])
            acc_est += ((est - assign[i + 1]) / (assign[i + 2] - assign[i]))**2

        lh_results.append([assign, acc_est])

    lh_results = sorted(lh_results, key=operator.itemgetter(1))
    opt_assignment = lh_results[0][0]

    # In order to estimate a slope of the DE/DV curve, b-spline is used.
    ve_spl = splrep(opt_assignment, e_e)
    app_slope = splev(opt_assignment, ve_spl, der=1)

    if len(excl) > 0:
        ev_spl = splrep(e_e, opt_assignment)
        excl_positions = [splev(STANDARD_FEATURES[element], ev_spl) for element in excl]
        peak_positions = np.hstack([peak_positions, excl_positions])

    # Calculate a histograms for each element.
    histograms = []
    for pp in opt_assignment:
        try:
            lnp = np.max(peak_positions[peak_positions < pp])
        except ValueError:
            lnp = -np.inf
        try:
            rnp = np.min(peak_positions[peak_positions > pp])
        except ValueError:
            rnp = np.inf

        width = hw / splev(pp, ve_spl, der=1)
        histograms.append(np.histogram(data, bins=np.linspace(np.max([pp - width / 2, (pp + lnp)/2]),
                                                              np.min([pp + width / 2, (pp + rnp)/2]), 101)))

    return {el: [hist, slope] for el, hist, slope in zip(name_e, histograms, app_slope)}