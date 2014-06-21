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
        self.histograms = None
        self.complex_fitters = None
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
                                                                  np.min([pp + width / 2, (pp + rnp)/2]), 101)))

        #return {el: [hist, slope] for el, hist, slope in zip(name_e, histograms, app_slope)}
        self.histograms = histograms

        complex_fitters = []
        refined_peak_positions = []

        for el, hist, slope in zip(name_e, histograms, app_slope):
            flu_members = {name: obj for name, obj in inspect.getmembers(mass.calibration.fluorescence_lines)}

            try:
                fitter = flu_members[el + 'Fitter']()
            except KeyError:
                raise KeyError("Corresponding fitter is not found.")

            params, _ = fitter.fit(hist[0], hist[1], plot=False)
            complex_fitters.append(fitter)
            refined_peak_positions.append(params[1])
            #energy_resolutions.append(params[0])

        self.complex_fitters = complex_fitters

        if len(refined_peak_positions) > 3:
            self.ph2energy = mass.mathstat.interpolate.CubicSpline(refined_peak_positions, e_e)
        else:
            self.ph2energy = interp1d(refined_peak_positions, e_e, kind='linear', bounds_error=True)

        return self

    def __call__(self, ph):
        if self.ph2energy is None:
            raise ValueError('Has not been calibrated yet.')

        return self.ph2energy(ph)


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator
mpl.rcParams['font.sans-serif'] = 'Arial'


def diagnose_calibration(cal):
    fig = plt.figure(figsize=(16, 9))

    n = int(np.ceil(np.sqrt(len(cal.elements))))

    w, h, lm, bm, hs, vs = 0.6, 0.9, 0.05, 0.08, 0.1, 0.1
    for i, (el, hist, fitter) in enumerate(zip(cal.elements, cal.histograms, cal.complex_fitters)):
        ax = fig.add_axes([w * (i % n) / n + lm,
                           h * (i / n) / n + bm,
                           (w - hs) / n,
                           (h - vs) / n])
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

        ax.step(hist[1][:-1], hist[0], where='mid', color='grey', lw=1)
        ax.text(0.05, 0.95, el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$'),
                transform=ax.transAxes, ha='left', va='top')
        x = np.linspace(hist[1][0], hist[1][-1], 201)
        y = fitter.fitfunc(fitter.last_fit_params, x)
        ax.plot(x, y, color='k', lw=1)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, np.max(hist[0]) * 1.1)

    ax = fig.add_axes([lm + w, bm, (1.0 - lm - w) - 0.06, h - 0.05])
    for el, fitter in zip(cal.elements, cal.complex_fitters):
        ax.text(fitter.last_fit_params[1], STANDARD_FEATURES[el],
                el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$'), ha='left', va='top',
                transform=mtrans.ScaledTranslation(5.0 / 72, -64.0 / 72, fig.dpi_scale_trans) + ax.transData)

    ax.scatter([fitter.last_fit_params[1] for fitter in cal.complex_fitters],
            [STANDARD_FEATURES[el] for el in cal.elements], s=36, c=(0.2, 0.2, 0.8))

    lb, ub = cal.complex_fitters[0].last_fit_params[1], cal.complex_fitters[-1].last_fit_params[1]
    width = ub - lb
    x = np.linspace(lb - width / 10, ub + width / 10, 101)
    y = cal(x)
    ax.plot(x, y, '--', color='orange', lw=2, zorder=-2)

    ax.yaxis.set_tick_params(labelleft=False, labelright=True)
    ax.yaxis.set_label_position('right')

    ax.set_xlabel('Pulse height')
    ax.set_ylabel('Energy (eV)')

    ax.set_xlim(lb - width / 10, ub + width / 10)

    return fig

def test_young(ch):
    data = np.loadtxt('C:/Users/ynj5/Google Drive/projects/CALIBRONIUM_ANALYSIS/chan{0}_calibronium'.format(ch))
    names = ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha']

    cal = EnergyCalibration()
    cal.fit(data, names)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    te = [STANDARD_FEATURES[n] for n in cal.elements]

    ax.hist(data, bins=1001)

    print cal.peak_positions
    ax2.plot(cal.peak_positions, te, 'o', markersize=7, mfc=(0.8, 0.2, 0.2), mew=2)

    x = np.linspace(np.min(cal.peak_positions), np.max(cal.peak_positions), 1001)
    ax2.plot(x, cal(x), '--', color='orange', lw=4, zorder=-2)

    plt.show()