"""
Classes to create time-domain and Fourier-domain optimal filters.
"""

import numpy as np
import matplotlib.pylab as plt
import numpy.typing as npt
from typing import Optional, Callable
from dataclasses import dataclass

from mass.mathstat.toeplitz import ToeplitzSolver


def band_limit(modelmatrix: np.ndarray, sample_time: float, fmax: Optional[float], f_3db: Optional[float]):
    """Band-limit the column-vectors in a model matrix with a hard and/or
    1-pole low-pass filter. Change the input `modelmatrix` in-place.

    No effect if both `fmax` and `f_3db` are `None`.

    Parameters
    ----------
    modelmatrix : np.ndarray
        The 1D or 2D array to band-limit. (If a 2D array, columns are independently band-limited.)
    sample_time : float
        The sampling period
    fmax : Optional[float]
        The hard maximum frequency (units are inverse of `sample_time` units)
    f_3db : Optional[float]
        The 1-pole low-pass filter's 3 dB point (same units as `fmax`)
    """
    if fmax is None and f_3db is None:
        return

    # Handle the 2D case by calling this function once per column.
    assert len(modelmatrix.shape) <= 2
    if len(modelmatrix.shape) == 2:
        for i in range(modelmatrix.shape[1]):
            band_limit(modelmatrix[:, i], sample_time, fmax, f_3db)
        return

    vector = modelmatrix
    filt_length = len(vector)
    sig_ft = np.fft.rfft(vector)
    freq = np.fft.fftfreq(filt_length, d=sample_time)
    freq = np.abs(freq[:len(sig_ft)])
    if fmax is not None:
        sig_ft[freq > fmax] = 0.0
    if f_3db is not None:
        sig_ft /= (1. + (1.0 * freq / f_3db)**2)

    # n=filt_length is needed when filt_length is ODD
    vector[:] = np.fft.irfft(sig_ft, n=filt_length)


@dataclass(frozen=True)
class Filter:
    values: np.ndarray
    variance: float
    predicted_v_over_dv: float
    dt_values: Optional[np.ndarray]
    convolution_lags: int = 1
    fmax: float = None
    f_3db: float = None
    cut_pre: int = 0
    cut_post: int = 0
    _filter_type: str = ""

    def __post_init__(self):
        if self.cut_pre < 0 or self.cut_post < 0:
            raise ValueError(f"(cut_pre,cut_post)=({self.cut_pre},{self.cut_post}), but neither can be negative")
        ns = len(self.values)
        if self.cut_pre + self.cut_post >= ns:
            raise ValueError(f"cut_pre+cut_post = {self.cut_pre + self.cut_post} but should be < {ns}")

    @property
    def is_arrival_time_safe(self):
        return self.dt_values is not None and self.convolution_lags == 1

    def plot(self, axis=None, **kwargs):
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        axis.plot(self.values, **kwargs)

    def report(self, std_energy=5898.8):
        """Report on V/dV for the filter

        Args:
            <std_energy> Energy (in eV) of a "standard" pulse.  Resolution will be given in eV at this energy,
                assuming linear devices.
        """
        var = self.variance
        if var < 0:
            v_dv = np.nan  # don't want to take a sqrt of negative number
            # avoid printing warnings
        else:
            v_dv = var**(-.5) / np.sqrt(8 * np.log(2))
        fwhm_eV = std_energy / v_dv
        print(f"{v_dv=:.2f} {var=:.2f} {fwhm_eV=:.2f} at {std_energy=:.2f} eV")


@dataclass(frozen=True)
class FilterMaker:
    """An object capable of creating optimal filter based on a single signal and noise set.

    Arguments:

    <signal_model> The average signal shape.  Filters will be rescaled so that the output
            upon putting this signal into the filter equals the *peak value* of this
            filter (that is, peak value relative to the baseline level).
    <n_pretrigger>   The number of leading samples in the average signal that are considered
            to be pre-trigger samples.  The avg_signal in this section is replaced by
            its constant averaged value before creating filters.  Also, one filter
            (filt_baseline_pretrig) is designed to infer the baseline using only
            <n_pretrigger> samples at the start of a record.
    <noise_autocorr> The autocorrelation function of the noise, where the lag spacing is
            assumed to be the same as the sample period of <avg_signal>.  If None,
            then several filters won't be computed.  (One of <noise_psd> or
            <noise_autocorr> must be a valid array, or <whitener> must be given.)
    <noise_psd>      The noise power spectral density.  If None, then filt_fourier won't be
            computed.  If not None, then it must be of length (2N+1), where N is the
            length of <avg_signal>, and its values are assumed to cover the non-negative
            frequencies from 0, 1/Delta, 2/Delta,.... up to the Nyquist frequency.
    <whitener>       An optional function object which, when called, whitens a vector or the
            columns of a matrix. Supersedes <noise_autocorr> if both are given.
    <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
            This must be given if <fmax> or <f_3db> are ever to be used.
    """

    signal_model: npt.ArrayLike
    n_pretrigger: int
    noise_autocorr: npt.ArrayLike = None
    noise_psd: npt.ArrayLike = None
    dt_model: npt.ArrayLike = None
    whitener: Callable = None
    sample_time: float = 0.0
    peak: float = 0.0

    def compute_5lag(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        """Compute a single filter.

        <fmax>  The strict maximum frequency to be passed in all filters.
        <f_3db> The 3 dB point for a one-pole low-pass filter to be applied to all filters.

        Either or both of <fmax> and <f_3db> are allowed.
        """
        if self.noise_autocorr is None and self.whitener is None:
            raise ValueError("Filter must have noise_autocorr or whitener arguments to generate 5-lag filters")
        noise_autocorr = self._compute_autocorr(cut_pre, cut_post)
        avg_signal, peak, _ = self._normalize_signal(cut_pre, cut_post)

        if self.sample_time <= 0 and not (fmax is None and f_3db is None):
            raise ValueError(
                "Filter must have a sample_time if it's to be smoothed with fmax or f_3db")
        if cut_pre < 0 or cut_post < 0:
            raise ValueError(f"(cut_pre,cut_post)=({cut_pre},{cut_post}), but neither can be negative")

        # Time domain filters
        shorten = 2
        avg_signal = avg_signal[shorten:-shorten]
        n = len(avg_signal)
        assert len(noise_autocorr) >= n

        noise_corr = noise_autocorr[:n]
        TS = ToeplitzSolver(noise_corr, symmetric=True)
        Rinv_sig = TS(avg_signal)
        Rinv_1 = TS(np.ones(n))
        filt_noconst = Rinv_1.sum() * Rinv_sig - Rinv_sig.sum() * Rinv_1

        band_limit(filt_noconst, self.sample_time, fmax, f_3db)

        self.normalize_filter(filt_noconst, avg_signal)
        variance = bracketR(filt_noconst, noise_corr)

        # Set weights in the cut_pre and cut_post windows to 0
        if cut_pre > 0 or cut_post > 0:
            filt_noconst = np.hstack([np.zeros(cut_pre), filt_noconst, np.zeros(cut_post)])

        vdv = peak / (8 * np.log(2) * variance)**0.5
        return Filter(filt_noconst, variance, vdv, None, 1 + 2 * shorten,
                      fmax, f_3db, cut_pre, cut_post, "5lag")

    def compute_fourier(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        # Make sure we have either a noise PSD or an autocorrelation or a whitener
        if self.noise_psd is None:
            raise ValueError("Filter must have noise_psd to generate a Fourier filter")
        if cut_pre > 0 or cut_post > 0:
            raise NotImplementedError("Haven't implemented sample non-weighting for Fourier filters")
        if cut_pre < 0 or cut_post < 0:
            raise ValueError(f"(cut_pre,cut_post)=({cut_pre},{cut_post}), but neither can be negative")

        avg_signal, peak, _ = self._normalize_signal(cut_pre, cut_post)

        shorten = 2  # to use in 5-lag style
        avg_signal = avg_signal[shorten:-shorten]
        n = len(self.noise_psd)
        window = 1.0
        sig_ft = np.fft.rfft(avg_signal * window)

        if len(sig_ft) != n - shorten:
            raise ValueError(f"signal real DFT and noise PSD are not the same length ({len(sig_ft)} and {n})")

        # Careful with PSD: "shorten" it by converting into a real space autocorrelation,
        # truncating the middle, and going back to Fourier space
        if shorten > 0:
            noise_autocorr = np.fft.irfft(self.noise_psd)
            noise_autocorr = np.hstack((noise_autocorr[:n - shorten - 1],
                                        noise_autocorr[-n + shorten:]))
            noise_psd = np.fft.rfft(noise_autocorr)
        else:
            noise_psd = self.noise_psd
        sig_ft_weighted = sig_ft / noise_psd

        # Band-limit
        if fmax is not None or f_3db is not None:
            freq = np.arange(0, n - shorten, dtype=float) * \
                0.5 / ((n - 1) * self.sample_time)
            if fmax is not None:
                sig_ft_weighted[freq > fmax] = 0.0
            if f_3db is not None:
                sig_ft_weighted /= (1 + (freq * 1.0 / f_3db)**2)

        sig_ft_weighted[0] = 0.0
        filt_fourier = np.fft.irfft(sig_ft_weighted) / window
        self.normalize_filter(filt_fourier, avg_signal)

        # How we compute the uncertainty depends on whether there's a noise autocorrelation result
        if self.noise_autocorr is None:
            noise_ft_squared = (len(self.noise_psd) - 1) / self.sample_time * self.noise_psd
            kappa = (np.abs(sig_ft * self.peak)**2 / noise_ft_squared)[1:].sum()
            variance_fourier = 1. / kappa
        else:
            ac = self.noise_autocorr[:len(filt_fourier)].copy()
            variance_fourier = bracketR(filt_fourier, ac) / self.peak**2
        vdv = peak / (8 * np.log(2) * variance_fourier)**0.5
        return Filter(filt_fourier, variance_fourier, vdv, None, 1 + 2 * shorten,
                      fmax, f_3db, cut_pre, cut_post, "fourier")

    def compute_ats(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        """Compute a single filter.

        <fmax>   The strict maximum frequency to be passed in all filters.
        <f_3db>  The 3 dB point for a one-pole low-pass filter to be applied to all filters.
        <cut_pre> Cut this many samples from the start of the filter, giving them 0 weight.
        <cut_post> Cut this many samples from the end of the filter, giving them 0 weight.

        Either or both of <fmax> and <f_3db> are allowed.
        """
        if self.noise_autocorr is None and self.whitener is None:
            raise ValueError("Filter must have noise_autocorr or whitener arguments to generate ATS filters")
        if self.dt_model is None:
            raise ValueError("Filter must have dt_model to generate ATS filters")
        if self.sample_time is None and not (fmax is None and f_3db is None):
            raise ValueError(
                "Filter must have a sample_time if it's to be smoothed with fmax or f_3db")

        noise_autocorr = self._compute_autocorr(cut_pre, cut_post)
        avg_signal, peak, dt_model = self._normalize_signal(cut_pre, cut_post)

        ns = len(avg_signal)
        assert ns == len(dt_model)
        if cut_pre < 0 or cut_post < 0:
            raise ValueError(f"(cut_pre,cut_post)=({cut_pre},{cut_post}), but neither can be negative")
        if cut_pre + cut_post >= ns:
            raise ValueError(f"cut_pre+cut_post = {cut_pre + cut_post} but should be < {ns}")

        MT = np.vstack((avg_signal, dt_model, np.ones(ns)))

        if self.whitener is not None:
            WM = self.whitener(MT.T)
            A = np.dot(WM.T, WM)
            Ainv = np.linalg.inv(A)
            WtWM = self.whitener.applyWT(WM)
            filt = np.dot(Ainv, WtWM.T)

        else:
            assert len(noise_autocorr) >= ns
            noise_corr = noise_autocorr[:ns]
            TS = ToeplitzSolver(noise_corr, symmetric=True)

            RinvM = np.vstack([TS(r) for r in MT]).T
            A = np.dot(MT, RinvM)
            Ainv = np.linalg.inv(A)
            filt = np.dot(Ainv, RinvM.T)

        band_limit(filt.T, self.sample_time, fmax, f_3db)
        scale = np.max(avg_signal) / np.dot(filt[0], avg_signal)

        if cut_pre > 0 or cut_post > 0:
            nfilt = filt.shape[0]
            filt = np.hstack([np.zeros((nfilt, cut_pre), dtype=float),
                              filt,
                              np.zeros((nfilt, cut_post), dtype=float)])

        filt_noconst = filt[0]
        filt_dt = filt[1]
        # filt_baseline = filt[2]

        filt_noconst *= scale
        filt_dt *= scale
        variance = bracketR(filt_noconst, self.noise_autocorr)
        vdv = peak / (np.log(2) * 8 * variance)**0.5
        return Filter(filt_noconst, variance, vdv, filt_dt, 1,
                      fmax, f_3db, cut_pre, cut_post, "ats")

    def _compute_autocorr(self, cut_pre, cut_post):
        # If there's an autocorrelation, cut it down to length.
        if self.noise_autocorr is None:
            return None
        return self.noise_autocorr[:len(self.signal_model) - (cut_pre + cut_post)]

    def _normalize_signal(self, cut_pre, cut_post):
        avg_signal = self.signal_model.copy()
        ns = len(avg_signal)
        pre_avg = avg_signal[cut_pre:self.n_pretrigger - 1].mean()

        # Find signal's peak value. This is normally peak=(max-pretrigger).
        # If signal is negative-going, however, then peak=(pretrigger-min).
        if self.peak > 0.0:
            peak_signal = self.peak
        else:
            a = avg_signal[cut_pre:ns - cut_post].min()
            b = avg_signal[cut_pre:ns - cut_post].max()
            is_negative = pre_avg - a > b - pre_avg
            if is_negative:
                peak_signal = a - pre_avg
            else:
                peak_signal = b - pre_avg

        # avg_signal: normalize to have unit peak
        avg_signal = (avg_signal - pre_avg) / peak_signal
        avg_signal[:self.n_pretrigger] = 0.0
        avg_signal = avg_signal[cut_pre:ns - cut_post]
        if self.dt_model is None:
            dt_model = None
        else:
            dt_model = self.dt_model / peak_signal
            dt_model = dt_model[cut_pre:ns - cut_post]
        return avg_signal, peak_signal, dt_model

    def normalize_filter(self, f, avg_signal):
        """Rescale filter `f` in-place so that it gives unit response to avg_signal"""
        if len(f) == len(avg_signal):
            f *= 1 / np.dot(f, avg_signal)
        elif self.shorten >= 2:
            conv = np.zeros(5, dtype=float)
            for i in range(5):
                conv[i] = np.dot(f, avg_signal[i:i + len(f)])
            x = np.arange(-2, 2.1)
            fit = np.polyfit(x, conv, 2)
            fit_ctr = -0.5 * fit[1] / fit[0]
            fit_peak = np.polyval(fit, fit_ctr)
            f *= 1.0 / fit_peak
        else:
            f *= 1.0 / np.dot(f, avg_signal[self.shorten:-self.shorten])


def bracketR(q, noise):
    """Return the dot product (q^T R q) for vector <q> and matrix R constructed from
    the vector <noise> by R_ij = noise_|i-j|.  We don't want to construct the full matrix
    R because for records as long as 10,000 samples, the matrix will consist of 10^8 floats
    (800 MB of memory).
    """

    if len(noise) < len(q):
        raise ValueError(f"Vector q (length {len(q)}) cannot be longer than the noise (length {len(noise)})")
    n = len(q)
    r = np.zeros(2 * n - 1, dtype=float)
    r[n - 1:] = noise[:n]
    r[n - 1::-1] = noise[:n]
    dot = 0.0
    for i in range(n):
        dot += q[i] * r[n - i - 1:2 * n - i - 1].dot(q)
    return dot


class ToeplitzWhitener:
    """An object that can perform approximate noise whitening.

    For an ARMA(p,q) noise model, mutliply by (or solve) the matrix W (or its
    transpose), where W is the Toeplitz approximation to the whitening matrix V.
    A whitening matrix V means that if R is the ARMA noise covariance matrix,
    then VRV' = I. While W only approximately satisfies this, it has some handy
    properties that make it a useful replacement. (In particular, it has the
    time-transpose property that if you zero-pad the beginning of vector v and
    shift the remaining elements, then the same is done to Wv.)

    The callable function object returns Wv or WM if called with
    vector v or matrix M. Other methods:

    * `tw.whiten(v)` returns Wv; it is equivalent to `tw(v)`
    * `tw.solveWT(v)` returns inv(W')*v
    * `tw.applyWT(v)` returns W'v
    * `tw.solveW(v)` returns inv(W)*v
    """

    def __init__(self, thetacoef: npt.ArrayLike, phicoef: npt.ArrayLike):
        """Initialize using the coefficients `thetacoef` of the MA process
        and `phicoef` of the AR process.

        Parameters
        ----------
        thetacoef : npt.ArrayLike
            The moving-average (MA) process coefficients
        phicoef : npt.ArrayLike
            The autoregressive (AR) process coefficients
        """
        self.theta = np.array(thetacoef)
        self.phi = np.array(phicoef)
        self.p = len(self.phi) - 1
        self.q = len(self.theta) - 1

    def whiten(self, v: npt.ArrayLike) -> np.ndarray:
        "Return whitened vector (or matrix of column vectors) Wv"
        return self(v)

    def __call__(self, v: npt.ArrayLike) -> np.ndarray:
        "Return whitened vector (or matrix of column vectors) Wv"
        v = np.asarray(v)
        if v.ndim > 3:
            raise ValueError("v must be an array of dimension 1 or 2")
        elif v.ndim == 2:
            w = np.zeros_like(v)
            for i in range(v.shape[1]):
                w[:, i] = self(v[:, i])
            return w

        # Multiply by the Toeplitz AR matrix to make the MA*w vector.
        N = len(v)
        y = self.phi[0] * v
        for i in range(1, 1 + self.p):
            y[i:] += self.phi[i] * v[:-i]

        # Second, solve the MA matrix (a banded, lower-triangular Toeplitz matrix with
        # q non-zero subdiagonals.)
        y[0] /= self.theta[0]
        if N == 1:
            return y
        for i in range(1, min(self.q, N)):
            for j in range(i):
                y[i] -= y[j] * self.theta[i - j]
            y[i] /= self.theta[0]
        for i in range(self.q, N):
            for j in range(i - self.q, i):
                y[i] -= y[j] * self.theta[i - j]
            y[i] /= self.theta[0]
        return y

    def solveW(self, v: npt.ArrayLike) -> np.ndarray:
        "Return unwhitened vector (or matrix of column vectors) inv(W)*v"
        v = np.asarray(v)
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.solveW(v[:, i])
            return r

        # Multiply by the Toeplitz MA matrix to make the AR*w vector.
        N = len(v)
        y = self.theta[0] * v
        for i in range(1, 1 + self.q):
            y[i:] += self.theta[i] * v[:-i]

        # Second, solve the AR matrix (a banded, lower-triangular Toeplitz matrix with
        # p non-zero subdiagonals.)
        y[0] /= self.phi[0]
        if N == 1:
            return y
        for i in range(1, min(self.p, N)):
            for j in range(i):
                y[i] -= y[j] * self.phi[i - j]
            y[i] /= self.phi[0]
        for i in range(self.p, N):
            for j in range(i - self.p, i):
                y[i] -= y[j] * self.phi[i - j]
            y[i] /= self.phi[0]
        return y

    def solveWT(self, v: npt.ArrayLike) -> np.ndarray:
        "Return vector (or matrix of column vectors) inv(W')*v"
        v = np.asarray(v)
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.solveWT(v[:, i])
            return r

        N = len(v)
        y = np.array(v)
        y[N - 1] /= self.phi[0]
        for i in range(N - 2, -1, -1):
            f = min(self.p + 1, N - i)
            y[i] -= np.dot(y[i + 1:i + f], self.phi[1:f])
            y[i] /= self.phi[0]
        return np.correlate(y, self.theta, "full")[self.q:]

    def applyWT(self, v: npt.ArrayLike) -> np.ndarray:
        """Return vector (or matrix of column vectors) W'v"""
        v = np.asarray(v)
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.applyWT(v[:, i])
            return r

        N = len(v)
        y = np.array(v)
        y[N - 1] /= self.theta[0]
        for i in range(N - 2, -1, -1):
            f = min(self.q + 1, N - i)
            y[i] -= np.dot(y[i + 1:i + f], self.theta[1:f])
            y[i] /= self.theta[0]
        return np.correlate(y, self.phi, "full")[self.p:]

    def W(self, N: int) -> np.ndarray:
        """Return the full, approximate whitening matrix.

        Normally the full W is large and slow to use. But it's here so you can
        easily test that W(len(v))*v == whiten(v), and similar.
        """
        AR = np.zeros((N, N), dtype=float)
        MA = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(max(0, i - self.p), i + 1):
                AR[i, j] = self.phi[i - j]
            for j in range(max(0, i - self.q), i + 1):
                MA[i, j] = self.theta[i - j]
        return np.linalg.solve(MA, AR)
