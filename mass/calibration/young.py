from collections import Counter
import itertools
import operator
import inspect
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from sklearn.cluster import DBSCAN
from mass.calibration.energy_calibration import STANDARD_FEATURES
import mass.calibration.fluorescence_lines
import mass.mathstat.interpolate


class EnergyCalibration(object):

    def __init__(self, eps=5, mcs=100, hw=200, excl=()):
        self.eps = eps
        self.mcs = mcs
        self.hw = hw
        self.excl = excl
        self.elements = None
        self.energy_resolutions = None
        self.peak_positions = None
        self.ph2energy = None

    def fit(self, data, elements):
        dbs = DBSCAN(eps=self.eps)
        dbs.fit(data[:, np.newaxis])

        count = Counter(dbs.labels_)
        peak_positions = []

        for l, c in count.most_common():
            # Outlier cluster won't count.
            if l < -0.5:
                continue
            # Not bright enough cluster won't count either.
            if c < self.mcs:
                break

            peak_positions.append(np.average(data[dbs.labels_ == l]))

        if len(peak_positions) < len(elements):
            raise ValueError('Not enough clusters are identified in data.')

        peak_positions = np.array(peak_positions)

        name_e, e_e = zip(*sorted([[element, STANDARD_FEATURES[element]] for element in elements],
                                  key=operator.itemgetter(1)))
        self.elements = name_e

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

        if len(self.excl) > 0:
            ev_spl = splrep(e_e, opt_assignment)
            excl_positions = [splev(STANDARD_FEATURES[element], ev_spl) for element in self.excl]
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

            width = self.hw / splev(pp, ve_spl, der=1)
            histograms.append(np.histogram(data, bins=np.linspace(np.max([pp - width / 2, (pp + lnp)/2]),
                                                                  np.min([pp + width / 2, (pp + rnp)/2]), 201)))

        #return {el: [hist, slope] for el, hist, slope in zip(name_e, histograms, app_slope)}

        refined_peak_positions = []
        energy_resolutions = []

        for el, hist, slope in zip(name_e, histograms, app_slope):
            flu_members = {name: obj for name, obj in inspect.getmembers(mass.calibration.fluorescence_lines)}

            try:
                fitter = flu_members[el + 'Fitter']()
            except KeyError as e:
                raise KeyError("Corresponding fitter is not found.")

            params, cov = fitter.fit(hist[0], hist[1], plot=False)
            refined_peak_positions.append(params[1])
            energy_resolutions.append(params[0])

        self.energy_resolutions = np.array(energy_resolutions)
        self.peak_positions = np.array(refined_peak_positions)

        if len(refined_peak_positions) > 3:
            self.ph2energy = mass.mathstat.interpolate.CubicSpline(refined_peak_positions, e_e)
        else:
            self.ph2energy = interp1d(refined_peak_positions, e_e, kind='linear', bounds_error=True)

        return self

    def __call__(self, ph):
        if self.ph2energy is None:
            raise ValueError('Has not been calibrated yet.')

        return self.ph2energy(ph)


import matplotlib.pyplot as plt


def test_young(ch):
    data = np.loadtxt('C:/Users/ynj5/Google Drive/projects/CALIBRONIUM_ANALYSIS/chan{0}_calibronium'.format(ch))
    names = ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha']

    cal = EnergyCalibration()
    cal.fit(data, names)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    te = [STANDARD_FEATURES[n] for n in cal.elements]

    ax.hist(data, bins=1001)

    print cal.peak_positions
    ax2.plot(cal.peak_positions, te, 'o', markersize=7, mfc=(0.8, 0.2, 0.2), mew=2)

    x = np.linspace(np.min(cal.peak_positions), np.max(cal.peak_positions), 1001)
    ax2.plot(x, cal(x), '--', color='orange', lw=4, zorder=-2)

    plt.show()