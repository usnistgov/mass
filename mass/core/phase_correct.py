import numpy as np
import scipy as sp
import scipy.signal

import mass.mathstat.entropy
from mass.mathstat.interpolate import CubicSpline
from mass.common import tostr
import logging
LOG = logging.getLogger("mass")


class PhaseCorrector:
    version = 1

    def __init__(self, phase_uniformifier_x, phase_uniformifier_y, corrections, indicatorName, uncorrectedName):
        self.corrections = corrections
        self.phase_uniformifier_x = np.array(phase_uniformifier_x)
        self.phase_uniformifier_y = np.array(phase_uniformifier_y)
        self.indicatorName = tostr(indicatorName)
        self.uncorrectedName = tostr(uncorrectedName)
        self.phase_uniformifier = CubicSpline(
            self.phase_uniformifier_x, self.phase_uniformifier_y)

    def toHDF5(self, hdf5_group, name="phase_correction", overwrite=False):
        """Write to the given HDF5 group for later recovery from disk (by fromHDF5 class method)."""
        group = hdf5_group.require_group(name)

        def h5group_update(name, vector):
            if name in group:
                if overwrite:
                    del group[name]
                else:
                    raise AttributeError("Cannot overwrite phase correction dataset '%s'" % name)
            group[name] = vector
        h5group_update("phase_uniformifier_x", self.phase_uniformifier_x)
        h5group_update("phase_uniformifier_y", self.phase_uniformifier_y)
        h5group_update("uncorrected_name", self.uncorrectedName)
        h5group_update("indicator_name", self.indicatorName)
        h5group_update("version", self.version)
        for (i, correction) in enumerate(self.corrections):
            h5group_update(f"correction_{i}_x", correction._x)
            h5group_update(f"correction_{i}_y", correction._y)

    def correct(self, phase, ph):
        # attempt to force phases to fall between X and X
        phase_uniformified = phase - self.phase_uniformifier(ph)
        # Compute a correction for each pulse for each correction-line energy
        # For the actual correction, don't let |ph| > 0.6 sample
        phase_clipped = np.clip(phase_uniformified, -0.6, 0.6)
        pheight_corrected = _phase_corrected_filtvals(phase_clipped, ph, self.corrections)
        return pheight_corrected

    def __call__(self, phase_indicator, ph):
        return self.correct(phase_indicator, ph)

    @classmethod
    def fromHDF5(cls, hdf5_group, name="phase_correction"):
        x = hdf5_group[f"{name}/phase_uniformifier_x"][()]
        y = hdf5_group[f"{name}/phase_uniformifier_y"][()]
        uncorrectedName = tostr(hdf5_group[f"{name}/uncorrected_name"][()])
        indicatorName = tostr(hdf5_group[f"{name}/indicator_name"][()])
        version = hdf5_group[f"{name}/version"][()]
        i = 0
        corrections = []
        while f"{name}/correction_{i}_x" in hdf5_group:
            _x = hdf5_group[f"{name}/correction_{i}_x"][()]
            _y = hdf5_group[f"{name}/correction_{i}_y"][()]
            corrections.append(CubicSpline(_x, _y))
            i += 1
        assert (version == cls.version)
        return cls(x, y, corrections, indicatorName, uncorrectedName)

    def __repr__(self):
        s = f"""PhaseCorrector with
        splines at this many levels: {len(self.corrections)}
        phase_uniformifier_x: {self.phase_uniformifier_x}
        phase_uniformifier_y: {self.phase_uniformifier_y}
        uncorrectedName: {self.uncorrectedName}
        """
        return s


def phase_correct(phase, pheight, ph_peaks=None, method2017=True, kernel_width=None,
                  indicatorName="", uncorrectedName=""):
    if ph_peaks is None:
        ph_peaks = _find_peaks_heuristic(pheight)
    if len(ph_peaks) <= 0:
        raise ValueError("Could not phase_correct because no peaks found")
    ph_peaks = np.asarray(ph_peaks)
    ph_peaks.sort()

    # Compute a correction function at each line in ph_peaks
    corrections = []
    median_phase = []
    if kernel_width is None:
        kernel_width = np.max(ph_peaks) / 1000.0
    for pk in ph_peaks:
        nextcorr, mphase = _phasecorr_find_alignment(
            phase, pheight, pk, .012 * np.mean(ph_peaks),
            method2017=method2017, kernel_width=kernel_width)
        corrections.append(nextcorr)
        median_phase.append(mphase)
    median_phase = np.array(median_phase)

    NC = len(corrections)
    if NC > 3:
        phase_uniformifier_x = ph_peaks
        phase_uniformifier_y = median_phase
    else:
        # Too few peaks to spline, so just bin and take the median per bin, then
        # interpolated (approximating) spline through/near these points.
        NBINS = 10
        top = min(pheight.max(), 1.2 * np.percentile(pheight, 98))
        bin = np.digitize(pheight, np.linspace(0, top, 1 + NBINS)) - 1
        x = np.zeros(NBINS, dtype=float)
        y = np.zeros(NBINS, dtype=float)
        w = np.zeros(NBINS, dtype=float)
        for i in range(NBINS):
            w[i] = (bin == i).sum()
            if w[i] == 0:
                continue
            x[i] = np.median(pheight[bin == i])
            y[i] = np.median(phase[bin == i])

        nonempty = (w > 0)
        # Use sp.interpolate.UnivariateSpline because it can make an approximating
        # spline. But then use its x/y data and knots to create a Mass CubicSpline,
        # because that one can have natural boundary conditions instead of insane
        # cubic functions in the extrapolation.
        if nonempty.sum() > 1:
            spline_order = min(3, nonempty.sum() - 1)
            crazy_spline = sp.interpolate.UnivariateSpline(
                x[nonempty], y[nonempty], w=w[nonempty] * (12**-0.5),
                k=spline_order)
            phase_uniformifier_x = crazy_spline._data[0]
            phase_uniformifier_y = crazy_spline._data[1]
        else:
            phase_uniformifier_x = np.array([0, 0, 0, 0])
            phase_uniformifier_y = np.array([0, 0, 0, 0])

    return PhaseCorrector(phase_uniformifier_x, phase_uniformifier_y, corrections,
                          indicatorName, uncorrectedName)


def _phasecorr_find_alignment(phase_indicator, pulse_heights, peak, delta_ph,
                              method2017=False, nf=10, kernel_width=2.0):
    """Find the way to align (flatten) `pulse_heights` as a function of `phase_indicator`
    working only within the range [peak-delta_ph, peak+delta_ph].

    If `method2017`, then use a scipy LSQUnivariateSpline with a reasonable (?)
    number of knots. Otherwise, use `nf` bins in `phase_indicator`, shifting each
    such that its `pulse_heights` histogram best aligns with the overall histogram.
    `method2017==False` (the 2015 way) is subject to particular problems when
    there are not a lot of counts in the peak.
    """
    phrange = np.array([-delta_ph, delta_ph]) + peak
    use = np.logical_and(np.abs(pulse_heights[:] - peak) < delta_ph,
                         np.abs(phase_indicator) < 2)
    low_phase, median_phase, high_phase = \
        np.percentile(phase_indicator[use], [3, 50, 97])

    if method2017:
        x = phase_indicator[use]
        y = pulse_heights[use]
        NBINS = len(x) // 300
        NBINS = max(3, NBINS)
        NBINS = min(12, NBINS)

        bin_edge = np.linspace(low_phase, high_phase, NBINS + 1)
        dx = high_phase - low_phase
        bin_edge[0] -= dx
        bin_edge[-1] += dx
        bins = np.digitize(x, bin_edge) - 1

        knots = np.zeros(NBINS, dtype=float)
        yknot = np.zeros(NBINS, dtype=float)
        iter1 = 0
        for i in range(NBINS):
            knots[i] = np.median(x[bins == i])

            def target(shift):
                yadj = y.copy()
                yadj[bins == i] += shift
                return mass.mathstat.entropy.laplace_entropy(yadj, kernel_width)
            brack = 0.003 * np.array([-1, 1], dtype=float)
            sbest, _KLbest, niter, _ = sp.optimize.brent(
                target, (), brack=brack, full_output=True, tol=3e-4)
            iter1 += niter
            yknot[i] = sbest

        yknot -= yknot.mean()
        correction1 = CubicSpline(knots, yknot)
        ycorr = y + correction1(x)

        iter2 = 0
        yknot2 = np.zeros(NBINS, dtype=float)
        for i in range(NBINS):
            def target(shift):
                yadj = ycorr.copy()
                yadj[bins == i] += shift
                return mass.mathstat.entropy.laplace_entropy(yadj, kernel_width)
            brack = 0.002 * np.array([-1, 1], dtype=float)
            sbest, _KLbest, niter, _ = sp.optimize.brent(
                target, (), brack=brack, full_output=True, tol=1e-4)
            iter2 += niter
            yknot2[i] = sbest
        correction = CubicSpline(knots, yknot + yknot2)
        H0 = mass.mathstat.entropy.laplace_entropy(y, kernel_width)
        H1 = mass.mathstat.entropy.laplace_entropy(ycorr, kernel_width)
        H2 = mass.mathstat.entropy.laplace_entropy(y + correction(x), kernel_width)
        LOG.info("Laplace entropy before/middle/after: %.4f, %.4f %.4f (%d+%d iterations, %d phase groups)",
                 H0, H1, H2, iter1, iter2, NBINS)

        curve = CubicSpline(knots - median_phase, peak - (yknot + yknot2))
        return curve, median_phase

    # Below here is "method2015", in which we perform correlations and fit to quadratics.
    # It is basically unsuitable for small statistics, so it is no longer preferred.
    Pedges = np.linspace(low_phase, high_phase, nf + 1)
    Pctrs = 0.5 * (Pedges[1:] + Pedges[:-1])
    Pbin = np.digitize(phase_indicator, Pedges) - 1

    NBINS = 200
    hists = np.zeros((nf, NBINS), dtype=float)
    for i, P in enumerate(Pctrs):
        use = (Pbin == i)
        c, b = np.histogram(pulse_heights[use], NBINS, phrange)
        hists[i] = c
    bctr = 0.5 * (b[1] - b[0]) + b[:-1]

    kernel = np.mean(hists, axis=0)[::-1]
    peaks = np.zeros(nf, dtype=float)
    for i in range(nf):
        # Find the PH of this ridge by fitting quadratic to the correlation
        # of histogram #i and the mean histogram, then finding its local max.
        conv = sp.signal.fftconvolve(kernel, hists[i], 'same')
        m = conv.argmax()
        if conv[m] <= 0:
            continue
        p = np.poly1d(np.polyfit(bctr[m - 2:m + 3], conv[m - 2:m + 3], 2))
        # p = np.poly1d(np.polyfit(b[m-2:m+3], conv[m-2:m+3], 2))
        peak = p.deriv(m=1).r[0]
        # if peak < bctr[m-2]: peak = bctr[m]
        # if peak > bctr[m+2]: peak = bctr[m]
        peaks[i] = peak
    # use = peaks>0
    # if use.sum() >= 2:
    #     curve = CubicSpline(Pctrs[use]-median_phase, peaks[use])
    # else:
    #     curve = CubicSpline(Pctrs-median_phase, np.mean(phrange)+np.zeros_like(Pctrs))
    curve = CubicSpline(Pctrs - median_phase, peaks)
    return curve, median_phase


def _phase_corrected_filtvals(phase, uncorrected, corrections):
    """Apply phase correction to `uncorrected`.

    Returns:
        the corrected vector.
    """
    NC = len(corrections)
    NP = len(phase)
    assert NP == len(uncorrected)
    phase = np.asarray(phase)
    uncorrected = np.asarray(uncorrected)

    ph = np.hstack([0] + [c(0) for c in corrections])
    assert (ph[1:] > ph[:-1]).all()  # corrections should be sorted by PH
    corr = np.zeros((NC + 1, NP), dtype=float)
    for i, c in enumerate(corrections):
        corr[i + 1] = c(0) - c(phase)

    # Now apply the appropriate correction (a linear interp between 2 neighboring values)
    corrected = uncorrected.copy()
    binnum = np.digitize(uncorrected, ph)
    for b in range(NC):
        # Don't correct binnum=0, which would be negative PH
        use = (binnum == 1 + b)
        if b + 1 == NC:  # For the last bin, extrapolate
            use = (binnum >= 1 + b)
        if use.sum() == 0:
            continue
        frac = (uncorrected[use] - ph[b]) / (ph[b + 1] - ph[b])
        corrected[use] += frac * corr[b + 1, use] + (1 - frac) * corr[b, use]
    return corrected


def _find_peaks_heuristic(phnorm):
    """A heuristic method to identify the peaks in a spectrum.

    This can be used to design the arrival-time-bias correction. Of course,
    you might have better luck finding peaks by an experiment-specific
    method, but this will stand in if you cannot or do not want to find
    peaks another way.

    Args:
        phnorm: a vector of pulse heights, found by whatever means you like.
            Normally it will be the self.p_filt_value_dc AFTER CUTS.

    Returns:
        ndarray of the various peaks found in the input vector.
    """
    median_scale = np.median(phnorm)

    # First make histogram with bins = 0.2% of median PH
    hist, bins = np.histogram(phnorm, 1000, [0, 2 * median_scale])
    binctr = bins[1:] - 0.5 * (bins[1] - bins[0])

    # Scipy continuous wavelet transform
    pk1 = np.array(sp.signal.find_peaks_cwt(hist, np.array([2, 4, 8, 12])))

    # A peak must contain 0.5% of the data or 500 events, whichever is more,
    # but the requirement is not more than 5% of data (for meager data sets)
    Ntotal = len(phnorm)
    MinCountsInPeak = min(max(500, Ntotal // 200), Ntotal // 20)
    pk2 = pk1[hist[pk1] > MinCountsInPeak]

    # Now take peaks from highest to lowest, provided they are at least 40 bins from any neighbor
    ordering = hist[pk2].argsort()
    pk2 = pk2[ordering]
    peaks = [pk2[0]]

    for pk in pk2[1:]:
        if (np.abs(peaks - pk) > 10).all():
            peaks.append(pk)
    peaks.sort()
    return np.array(binctr[peaks])
