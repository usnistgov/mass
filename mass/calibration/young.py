from collections import Counter
import itertools
import operator
import inspect
import numpy as np
from scipy.linalg import LinAlgError
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import gaussian_kde
from scipy.optimize import brentq

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import MaxNLocator

mpl.rcParams['font.sans-serif'] = 'Arial'
from sklearn.cluster import DBSCAN
from mass.calibration.energy_calibration import STANDARD_FEATURES
import mass.calibration.fluorescence_lines
import mass.mathstat.interpolate


class FailedFitter(object):
    def __init__(self, hist, bins):
        self.hist = hist
        self.bins = bins

        self.last_fit_params = [None, np.sum(self.hist * bins[:-1]) / np.sum(self.hist)] + [None] * 4

    def fitfunc(self, param, x):
        self.last_fit_params = param
        return np.zeros_like(x)


class EnergyCalibration(object):
    def __init__(self, eps=5, mcs=100, hw=200, excl=(), plot_on_fail=False, use_00=True):
        self.dbs = DBSCAN(eps=eps)
        self.data = np.zeros(0)
        self.mcs = mcs
        self.hw = hw
        self.excl = excl
        self.elements = None
        self.histograms = None
        self.complex_fitters = None
        self.ph2energy = None
        self.plot_on_fail = plot_on_fail
        self.use_00 = use_00

    def __identify_clusters(self, pulse_heights):
        self.data = np.hstack([self.data, pulse_heights])
        self.dbs.fit(self.data[:, np.newaxis])

        count = Counter(self.dbs.labels_)
        peak_positions = []

        for l, c in count.most_common():
            # Outlier cluster won't count.
            if l < -0.5:
                continue
            # Not bright enough cluster won't count either.
            if c < self.mcs:
                break

            peak_positions.append(np.average(self.data[self.dbs.labels_ == l]))

        return np.array(peak_positions)

    def __find_opt_assignment(self, peak_positions, line_names):
        name_e, e_e = zip(*sorted([[element, STANDARD_FEATURES[element]] for element in line_names],
                          key=operator.itemgetter(1)))
        self.elements = name_e

        lh_results = []

        for assign in itertools.combinations(peak_positions, len(line_names)):
            assign = sorted(assign)

            acc_est = 0.0
            for i in xrange(len(assign) - 2):
                est = assign[i] + (assign[i + 2] - assign[i]) * (e_e[i + 1] - e_e[i]) / (e_e[i+2] - e_e[i])
                acc_est += ((est - assign[i + 1]) / (assign[i + 2] - assign[i]))**2

            lh_results.append([assign, acc_est])

        lh_results = sorted(lh_results, key=operator.itemgetter(1))

        if lh_results[0][-1] > 0.001:
            raise ValueError('Could not match a pattern')

        return lh_results[0][0]

    def fit(self, pulse_heights, line_names):
        # Identify unlabeled clusters in a given spectrum
        peak_positions = self.__identify_clusters(pulse_heights)

        if len(peak_positions) < len(line_names):
            raise ValueError('Not enough clusters are identified in data.')

        # Exhaustive search for the best assignment.
        opt_assignment = self.__find_opt_assignment(peak_positions, line_names)
        e_e = [STANDARD_FEATURES[element] for element in self.elements]

        # In order to estimate a slope of the DE/DV curve, b-spline is used.
        # Do I need approximate DV/DEs for complex fittings?
        ev_spl = splrep(e_e, opt_assignment)
        app_slope = splev(e_e, ev_spl, der=1)

        if len(self.excl) > 0:
            excl_positions = [splev(STANDARD_FEATURES[element], ev_spl) for element in self.excl]
            peak_positions = np.hstack([peak_positions, excl_positions])

        histograms = []
        complex_fitters = []

        for pp, el, slope in zip(opt_assignment, self.elements, app_slope):
            flu_members = {name: obj for name, obj in inspect.getmembers(mass.calibration.fluorescence_lines)}

            try:
                lnp = np.max(peak_positions[peak_positions < pp])
            except ValueError:
                lnp = -np.inf
            try:
                rnp = np.min(peak_positions[peak_positions > pp])
            except ValueError:
                rnp = np.inf

            # width is the histrogram width in pulseheight units, calculate from self.hw which is in eV and
            # an evaluation of the spline which gives the derivative
            slope_dpulseheight_denergy = slope  # splev(STANDARD_FEATURES[el], ev_spl, der=1)
            width = self.hw * slope_dpulseheight_denergy
            if width <= 0:
                print("width below zero")
            binmin, binmax = np.max([pp - width / 2, (pp + lnp)/2]), np.min([pp + width / 2, (pp + rnp)/2])
            bin_size_ev = 2
            nbins = int(np.ceil((binmax-binmin)/(slope_dpulseheight_denergy*bin_size_ev)))

            bins = np.linspace(binmin, binmax, nbins + 1)
            hist, bins = np.histogram(pulse_heights, bins)

            # If a corresponding fitter could not be found then create a FailedFitter object.
            try:
                fitter_cls = flu_members[el + 'Fitter']
                fitter = fitter_cls()
            except KeyError:
                fitter = FailedFitter(hist, bins)
                histograms.append((hist, bins))
                complex_fitters.append(fitter)
                continue

            # Trying to fit histograms with different number of bins
            # with a corresponding complex fitter.
            while nbins > 32:
                params_guess = [None] * 6
                # resolution guess parameter should be something you can pass
                params_guess[0] = 10 * slope_dpulseheight_denergy  # resolution in pulse height units
                params_guess[1] = pp  # Approximate peak position
                params_guess[2] = slope_dpulseheight_denergy  # energy scale factor (pulseheight/eV)
                #hold = [2]  #hold the slope_dpulseheight_denergy constant while fitting

                try:
                    fitter.fit(hist, bins, params_guess, plot=False)
                    break
                except (ValueError, LinAlgError, RuntimeError):
                    if self.plot_on_fail:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.set_xlabel("pulse height (arbs)")
                        ax.set_ylabel("counts per %0.2f arb bin" % (bins[1]-bins[0]))
                        ax.set_title("%s, %s" % (el, str(params_guess)))

                        #ax.step(hist[1][:-1], hist[0])
                        ax.fill(np.repeat(bins, 2), np.hstack([[0], np.repeat(hist, 2), [0]]),
                                fc=(0.1, 0.1, 1.0), ec='b')
                        plt.show()
                    nbins /= 2
                    bins = np.linspace(binmin, binmax, nbins + 1)
                    hist, bins = np.histogram(pulse_heights, bins)
                    continue
            else:
                # If every attempt fails, a FailedFitter object is created.
                fitter = FailedFitter(hist, bins)
            histograms.append((hist, bins))
            complex_fitters.append(fitter)

        self.histograms = histograms
        self.complex_fitters = complex_fitters

        interp_peak_positions = self.refined_peak_positions
        if self.use_00:
            interp_peak_positions = [0] + self.refined_peak_positions
            e_e = [0]+e_e
        if len(e_e) > 3:
            self.ph2energy = mass.mathstat.interpolate.CubicSpline(interp_peak_positions, e_e)
        else:
            self.ph2energy = interp1d(interp_peak_positions, e_e, kind='linear', bounds_error=True)

        return self

    def __call__(self, ph):
        if self.ph2energy is None:
            raise ValueError('Has not been calibrated yet.')

        return self.ph2energy(ph)

    def energy2ph(self, energy):
        max_ph = self.complex_fitters[-1].last_fit_params[1] * 2  # twice the pulseheight of the largest pulseheight
        # in the calibration
        return brentq(lambda ph: self.ph2energy(ph)-energy, 0., max_ph)  # brentq is finds zeros

    def name2ph(self, feature_name):
        return self.energy2ph(mass.calibration.energy_calibration.STANDARD_FEATURES[feature_name])

    @property
    def refined_peak_positions(self):
        if self.complex_fitters is not None:
            return [fitter.last_fit_params[1] for fitter in self.complex_fitters]

        return None

    @property
    def energy_resolutions(self):
        if self.complex_fitters is not None:
            return [fitter.last_fit_params[0] for fitter in self.complex_fitters]

        return None

    @property
    def npts(self):
        if self.complex_fitters is not None:
            return len(self.complex_fitters)

        return 0

    def __repr__(self):
        return "EnergyCalibration with %d features" % (0 if self.complex_fitters is None else len(self.complex_fitters))


def diagnose_calibration(cal, hist_plot=False):
    #if cal.complex_fitters is None:
    if hist_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bmap = plt.get_cmap("spectral", 11)

        kde = gaussian_kde(cal.data, bw_method=0.002)
        counter = Counter(cal.dbs.labels_)
        peaks = list([[np.min(cal.data[cal.dbs.labels_ == x[0]]),
                      np.max(cal.data[cal.dbs.labels_ == x[0]])]
                     for x in counter.most_common() if (x[1] > cal.mcs) and (x[0] > -0.5)])
        peaks = sorted(peaks, key=operator.itemgetter(0))

        colors = bmap(np.linspace(0, 1, len(peaks)))

        x = np.linspace(np.min(cal.data), np.max(cal.data), 2001)
        y = kde(x)

        ax.fill_between(x, y, facecolor='k')

        for i, (lb, ub) in enumerate(peaks):
            ax.fill_between(x[(x > lb) & (x < ub)],
                            y[(x > lb) & (x < ub)], facecolor=colors[i])
        fig.show()

        #return fig

    fig = plt.figure(figsize=(16, 9))

    n = int(np.ceil(np.sqrt(len(cal.elements))))

    w, h, lm, bm, hs, vs = 0.6, 0.9, 0.05, 0.08, 0.1, 0.1
    for i, (el, hist, fitter) in enumerate(zip(cal.elements, cal.histograms, cal.complex_fitters)):
        ax = fig.add_axes([w * (i % n) / n + lm,
                           h * (i / n) / n + bm,
                           (w - hs) / n,
                           (h - vs) / n])
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

        #ax.step(hist[1][:-1], hist[0], where='mid', color='grey', lw=1)
        ax.fill(np.repeat(hist[1], 2), np.hstack([[0], np.repeat(hist[0], 2), [0]]),
                lw=1, fc=(0.3, 0.3, 0.9), ec=(0.1, 0.1, 1.0), alpha=0.8)
        ax.text(0.05, 0.97, el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$') +
                '\n' + "Resolution: {0:.1f} eV".format(fitter.last_fit_params[0]),
                transform=ax.transAxes, ha='left', va='top')
        x = np.linspace(hist[1][0], hist[1][-1], 201)
        y = fitter.fitfunc(fitter.last_fit_params, x)
        ax.plot(x, y, color=(0.9, 0.1, 0.1), lw=2)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0, np.max(hist[0]) * 1.3)

    ax = fig.add_axes([lm + w, bm, (1.0 - lm - w) - 0.06, h - 0.05])
    for el, fitter in zip(cal.elements, cal.complex_fitters):
        ax.text(fitter.last_fit_params[1], STANDARD_FEATURES[el],
                el.replace('Alpha', r'$_{\alpha}$').replace('Beta', r'$_{\beta}$'),
                ha='left', va='top',
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

    fig.show()

    return fig


def test_young(ch):
    data = np.loadtxt('C:/Users/YoungIl/Google Drive/projects/CALIBRONIUM_ANALYSIS/chan{0}_calibronium'.format(ch))
    names = ['VKAlpha', 'MnKAlpha', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', 'CuKBeta']

    cal = EnergyCalibration()
    cal.fit(data, names)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    te = [STANDARD_FEATURES[n] for n in cal.elements]

    ax.hist(data, bins=1001)

    peak_positions = [fitter.last_fit_params[1] for fitter in cal.complex_fitters]
    ax2.plot(peak_positions, te, 'o', markersize=7, mfc=(0.8, 0.2, 0.2), mew=2)

    x = np.linspace(np.min(peak_positions), np.max(peak_positions), 1001)
    ax2.plot(x, cal(x), '--', color='orange', lw=4, zorder=-2)

    plt.show()


def is_calibrated(cal):
    if hasattr(cal, "npts"):  # checks for Joe style calibration
        return False
    if cal.elements is None:  # then checks for now many elements are fitted for
        return False
    return True
