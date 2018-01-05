"""
Contains classes to do time-domain optimal filtering.
"""

import numpy as np
import scipy as sp
import matplotlib.pylab as plt

import mass
import mass.mathstat.toeplitz


def band_limit(modelmatrix, sample_time, fmax, f_3db):
    """Band-limit the column-vectors in a model matrix with a hard and/or
    1-pole low-pass filter."""
    assert len(modelmatrix.shape) <= 2
    if len(modelmatrix.shape) == 2:
        for i in range(modelmatrix.shape[1]):
            band_limit(modelmatrix[:, i], sample_time, fmax, f_3db)
    else:
        vector = modelmatrix
        filt_length = len(vector)
        sig_ft = np.fft.rfft(vector)
        freq = np.fft.fftfreq(filt_length, d=sample_time)
        freq = np.abs(freq[:len(sig_ft)])
        if fmax is not None:
            sig_ft[freq > fmax] = 0.0
        if f_3db is not None:
            sig_ft /= (1. + (1.0 * freq / f_3db)**2)
        vector[:] = np.fft.irfft(sig_ft, n=filt_length)
        # n=filt_length is needed when filt_length is ODD


class Filter(object):
    """A set of optimal filters based on a single signal and noise set."""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None,
                 whitener=None, sample_time=None, shorten=0):
        """Create a set of filters under various assumptions and for various purposes.

        Note that you now have to call Filter.compute() yourself to compute the filters.

        Args:
            <avg_signal>     The average signal shape.  Filters will be rescaled so that the output
                 upon putting this signal into the filter equals the *peak value* of this
                 filter (that is, peak value relative to the baseline level).
            <n_pretrigger>   The number of leading samples in the average signal that are considered
                 to be pre-trigger samples.  The avg_signal in this section is replaced by
                 its constant averaged value before creating filters.  Also, one filter
                 (filt_baseline_pretrig) is designed to infer the baseline using only
                 <n_pretrigger> samples at the start of a record.
            <noise_psd>      The noise power spectral density.  If None, then filt_fourier won't be
                 computed.  If not None, then it must be of length (2N+1), where N is the
                 length of <avg_signal>, and its values are assumed to cover the non-negative
                 frequencies from 0, 1/Delta, 2/Delta,.... up to the Nyquist frequency.
            <noise_autocorr> The autocorrelation function of the noise, where the lag spacing is
                 assumed to be the same as the sample period of <avg_signal>.  If None,
                 then several filters won't be computed.  (One of <noise_psd> or
                 <noise_autocorr> must be a valid array, or <whitener> must be given.)
            <whitener>       An optional function object which, when called, whitens a vector or the
                 columns of a matrix. Supersedes <noise_autocorr> if both are given.
            <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                 This must be given if <fmax> or <f_3db> are ever to be used.
            <shorten>        The time-domain filters should be shortened by removing this many
                 samples from each end.  (Do this for convenience of convolution over
                 multiple lags.)
        """
        self.sample_time = sample_time
        self.shorten = shorten
        pre_avg = avg_signal[:n_pretrigger].mean()

        # If signal is negative-going,
        is_negative = (avg_signal.min() - pre_avg) / (avg_signal.max() - pre_avg) < -1
        if is_negative:
            self.peak_signal = avg_signal.min() - pre_avg
        else:
            self.peak_signal = avg_signal.max() - pre_avg

        # self.avg_signal is normalized to have unit peak
        self.avg_signal = (avg_signal - pre_avg) / self.peak_signal
        self.avg_signal[:n_pretrigger] = 0.0
        
        self.ns = len(self.avg_signal)

        self.n_pretrigger = n_pretrigger
        if noise_psd is None:
            self.noise_psd = None
        else:
            self.noise_psd = np.asarray(noise_psd)
        if noise_autocorr is None:
            self.noise_autocorr = None
        else:
            self.noise_autocorr = np.asarray(noise_autocorr)
        self.whitener = whitener
        if noise_psd is None and noise_autocorr is None and whitener is None:
            raise ValueError("Filter must have noise_psd, noise_autocorr, or whitener arguments")

        self.filt_fourier = None
        self.filt_fourierfull = None
        self.filt_noconst = None
        self.filt_baseline = None
        self.filt_baseline_pretrig = None

        self.variances = {}
        self.predicted_v_over_dv = {}

    def normalize_filter(self, q):
        """Rescale filter <q> in-place so that it gives unit response to self.avg_signal"""
        if len(q) == len(self.avg_signal):
            q *= 1 / np.dot(q, self.avg_signal)
        elif self.shorten >= 2:
            conv = np.zeros(5, dtype=np.float)
            for i in range(5):
                conv[i] = np.dot(q, self.avg_signal[i:i + len(q)])
            x = np.arange(-2, 2.1)
            fit = np.polyfit(x, conv, 2)
            fit_ctr = -0.5 * fit[1] / fit[0]
            fit_peak = np.polyval(fit, fit_ctr)
            q *= 1.0 / fit_peak
        else:
            q *= 1.0 / np.dot(q, self.avg_signal[self.shorten:-self.shorten])

    def _compute_fourier_filter(self):
        """Compute the Fourier-domain filter."""
        if self.noise_psd is None:
            return

        # Careful: let's be sure that the Fourier domain filter is done consistently in Filter and
        # its child classes.

        n = len(self.noise_psd)
        # window = power_spectrum.hamming(2*(n-1-self.shorten))
        window = 1.0

        sig_ft = np.fft.rfft(self.avg_signal[self.shorten:self.ns-self.shorten] * window)
        #if self.shorten > 0:
            #sig_ft = np.fft.rfft(self.avg_signal[self.shorten:-self.shorten] * window)
        #else:
            #sig_ft = np.fft.rfft(self.avg_signal * window)

        if len(sig_ft) != n - self.shorten:
            raise ValueError("signal real DFT and noise PSD are not the same length (%d and %d)" %
                             (len(sig_ft), n))

        # Careful with PSD: "shorten" it by converting into a real space autocorrelation,
        # truncating the middle, and going back to Fourier space
        if self.shorten > 0:
            noise_autocorr = np.fft.irfft(self.noise_psd)
            noise_autocorr = np.hstack((noise_autocorr[:n - self.shorten - 1],
                                        noise_autocorr[-n + self.shorten:]))
            noise_psd = np.fft.rfft(noise_autocorr)
        else:
            noise_psd = self.noise_psd
        sig_ft_weighted = sig_ft / noise_psd

        # Band-limit
        if self.fmax is not None or self.f_3db is not None:
            freq = np.arange(0, n - self.shorten, dtype=np.float) * 0.5 / ((n - 1) * self.sample_time)
            if self.fmax is not None:
                sig_ft_weighted[freq > self.fmax] = 0.0
            if self.f_3db is not None:
                sig_ft_weighted /= (1 + (freq * 1.0 / self.f_3db)**2)

        # Compute both the normal (DC-free) and the full (with DC) filters.
        self.filt_fourierfull = np.fft.irfft(sig_ft_weighted) / window
        sig_ft_weighted[0] = 0.0
        self.filt_fourier = np.fft.irfft(sig_ft_weighted) / window
        self.normalize_filter(self.filt_fourierfull)
        self.normalize_filter(self.filt_fourier)

        # How we compute the uncertainty depends on whether there's a noise autocorrelation result
        if self.noise_autocorr is None:
            noise_ft_squared = (len(self.noise_psd) - 1) / self.sample_time * self.noise_psd
            kappa = (np.abs(sig_ft * self.peak_signal)**2 / noise_ft_squared)[:].sum()
            self.variances['fourierfull'] = 1. / kappa

            kappa = (np.abs(sig_ft * self.peak_signal)**2 / noise_ft_squared)[1:].sum()
            self.variances['fourier'] = 1. / kappa
        else:
            ac = self.noise_autocorr[:len(self.filt_fourier)].copy()
            self.variances['fourier'] = self.bracketR(self.filt_fourier, ac) / self.peak_signal**2
            self.variances['fourierfull'] = self.bracketR(self.filt_fourierfull, ac) / self.peak_signal**2
            # print 'Fourier filter done.  Variance: ',self.variances['fourier'],
            # 'V/dV: ',self.variances['fourier']**(-0.5)/2.35482

    def compute(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        """Compute a single filter.

        <fmax>   The strict maximum frequency to be passed in all filters.
        <f_3db>  The 3 dB point for a one-pole low-pass filter to be applied to all filters.
             Either or both of <fmax> and <f_3db> are allowed.
        <cut_pre> Cut this many samples from the start of the filter, giving them 0 weight.
        <cut_post> Cut this many samples from the end of the filter, giving them 0 weight.
        """
        if self.sample_time is None and not (fmax is None and f_3db is None):
            raise ValueError("Filter must have a sample_time if it's to be smoothed with fmax or f_3db")            
        if cut_pre < 0 or cut_post < 0:
            raise ValueError("(cut_pre,cut_post)=(%d,%d), but neither can be negative"%
                             (cut_pre,cut_post))
        if cut_pre+cut_post >= self.ns-2*self.shorten:
            raise ValueError("cut_pre+cut_post = %d but should be < %d"%(
                             cut_pre+cut_post, self.ns-2*self.shorten))
            
        self.fmax = fmax
        self.f_3db = f_3db
        self.variances = {}

        self._compute_fourier_filter()

        # Time domain filters
        assert self.noise_autocorr is not None
        n = len(self.avg_signal) - 2 * self.shorten
        assert len(self.noise_autocorr) >= n
        
        avg_signal = self.avg_signal[self.shorten+cut_pre:self.ns-(self.shorten+cut_post)]

        noise_corr = self.noise_autocorr[:n-(cut_pre+cut_post)] / self.peak_signal**2
        TS = mass.mathstat.toeplitz.ToeplitzSolver(noise_corr, symmetric=True)
        Rinv_sig = TS(avg_signal)
        Rinv_1 = TS(np.ones(n-(cut_pre+cut_post)))
        self.filt_noconst = Rinv_1.sum() * Rinv_sig - Rinv_sig.sum() * Rinv_1

        # Band-limit
        if self.fmax is not None or self.f_3db is not None:
            band_limit(self.filt_noconst, self.sample_time, self.fmax, self.f_3db)

        self.normalize_filter(self.filt_noconst)
        self.variances['noconst'] = self.bracketR(self.filt_noconst, noise_corr)

        self.filt_baseline = np.dot(avg_signal, Rinv_sig) * Rinv_1 - Rinv_sig.sum() * Rinv_sig
        self.filt_baseline /= self.filt_baseline.sum()
        self.variances['baseline'] = self.bracketR(self.filt_baseline, noise_corr)

        try:
            Rpretrig = sp.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger-cut_pre] / self.peak_signal**2)
            self.filt_baseline_pretrig = np.linalg.solve(Rpretrig, np.ones(self.n_pretrigger-cut_pre))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig, Rpretrig[0, :])
        except sp.linalg.LinAlgError:
            pass

        # Set weights in the pre_cut and post_cut windows to 0
        self.filt_noconst = np.hstack([np.zeros(cut_pre), self.filt_noconst, np.zeros(cut_post)])
        self.filt_baseline = np.hstack([np.zeros(cut_pre), self.filt_baseline, np.zeros(cut_post)])
        self.filt_baseline_pretrig = np.hstack([np.zeros(cut_pre), self.filt_baseline_pretrig])

        for key in self.variances.keys():
            self.predicted_v_over_dv[key] = 1 / (np.sqrt(np.log(2) * 8) * self.variances[key]**0.5)

    def bracketR(self, q, noise):
        """Return the dot product (q^T R q) for vector <q> and matrix R constructed from
        the vector <noise> by R_ij = noise_|i-j|.  We don't want to construct the full matrix
        R because for records as long as 10,000 samples, the matrix will consist of 10^8 floats
        (800 MB of memory)."""

        if len(noise) < len(q):
            raise ValueError("Vector q (length %d) cannot be longer than the noise (length %d)" %
                             (len(q), len(noise)))
        n = len(q)
        r = np.zeros(2 * n - 1, dtype=np.float)
        r[n - 1:] = noise[:n]
        r[n - 1::-1] = noise[:n]
        dot = 0.0
        for i in range(n):
            dot += q[i] * np.dot(r[n - i - 1:2 * n - i - 1], q)
        return dot

    def plot(self, axis=None, filtname="filt_noconst"):
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        try:
            axis.plot(self.__dict__[filtname], color='red')
        except AttributeError:
            pass

    def report(self, filters=None, std_energy=5898.8):
        """Report on V/dV for all filters

        Args:
            <filters>   Either the name of one filter or a sequence of names.  If not given, then all filters
                not starting with "baseline" will be reported.
            <std_energy> Energy (in eV) of a "standard" pulse.  Resolution will be given in eV at this energy,
                assuming linear devices.
        """

        # Handle <filters> is a single string --> convert to tuple of 1 string
        if isinstance(filters, basestring):
            filters = (filters,)

        # Handle default <filters> not given.
        if filters is None:
            filters = list(self.variances.keys())
            for f in self.variances:
                if f.startswith("baseline"):
                    filters.remove(f)
            filters.sort()

        for f in filters:
            try:
                var = self.variances[f]
                v_dv = var**(-.5) / np.sqrt(8 * np.log(2))
                print("%-20s  %10.3f  %10.4e" % (f, v_dv, var))
            except KeyError:
                print("%-20s not known" % f)


class ArrivalTimeSafeFilter(Filter):
    """Compute a filter for pulses given a pulse model expressed as a
    polynomial in "arrival time". The filter will be insensitive to the
    linear (and any higher-order) terms."""

    def __init__(self, pulsemodel, n_pretrigger, noise_autocorr=None,
                 whitener=None, sample_time=None, peak=1.0):
        if noise_autocorr is None and whitener is None:
            raise ValueError("%s requires either noise_autocorr or whitener to be set" %
                             (self.__class__.__name__))
        noise_psd = None
        sample_time = sample_time

        avg_signal = pulsemodel[:, 0]
        self.pulsemodel = pulsemodel
        self.peak = peak
        super(self.__class__, self).__init__(
            avg_signal, n_pretrigger, noise_psd, noise_autocorr=noise_autocorr,
            whitener=whitener, sample_time=sample_time, shorten=0)

    def compute(self, fmax=None, f_3db=None, cut_pre=0, cut_post=0):
        """Compute a single filter.

        <fmax>   The strict maximum frequency to be passed in all filters.
        <f_3db>  The 3 dB point for a one-pole low-pass filter to be applied to all filters.
             Either or both of <fmax> and <f_3db> are allowed.
        <cut_pre> Cut this many samples from the start of the filter, giving them 0 weight.
        <cut_post> Cut this many samples from the end of the filter, giving them 0 weight.
        """
        if self.sample_time is None and not (fmax is None and f_3db is None):
            raise ValueError("Filter must have a sample_time if it's to be smoothed with fmax or f_3db")
        if cut_pre < 0 or cut_post < 0:
            raise ValueError("(cut_pre,cut_post)=(%d,%d), but neither can be negative"%
                             (cut_pre,cut_post))
        ns = self.pulsemodel.shape[0]
        if cut_pre+cut_post >= ns:
            raise ValueError("cut_pre+cut_post = %d but should be < %d"%(
                             cut_pre+cut_post, ns))

        self.fmax = fmax
        self.f_3db = f_3db
        self.variances = {}

        n = len(self.avg_signal) - 2 * self.shorten
        unit = np.ones(n)
        MT = np.vstack((self.pulsemodel.T, unit))
        MT = MT[:, cut_pre:n-cut_post]
        n = n - (cut_pre + cut_post)

        if self.whitener is not None:
            WM = self.whitener(MT.T)
            if fmax is not None or f_3db is not None:
                band_limit(WM, self.sample_time, fmax, f_3db)
            A = np.dot(WM.T, WM)
            Ainv = np.linalg.inv(A)
            WtWM = self.whitener.applyWT(WM)
            filt = np.dot(Ainv, WtWM.T)

        else:
            assert self.noise_autocorr is not None
            assert len(self.noise_autocorr) >= n
            noise_corr = self.noise_autocorr[:n] / self.peak_signal**2
            TS = mass.mathstat.toeplitz.ToeplitzSolver(noise_corr, symmetric=True)

            RinvM = np.vstack([TS(r) for r in MT]).T
            if fmax is not None or f_3db is not None:
                band_limit(RinvM, self.sample_time, fmax, f_3db)
            A = np.dot(MT, RinvM)
            Ainv = np.linalg.inv(A)
            filt = np.dot(Ainv, RinvM.T)


        if cut_pre > 0 or cut_post > 0:
            nfilt = filt.shape[0]
            filt = np.hstack([np.zeros((nfilt, cut_pre), dtype=float),
                              filt,
                              np.zeros((nfilt, cut_post), dtype=float)])

        self.filt_noconst = filt[0]
        self.filt_aterms = filt[1:-1]
        self.filt_baseline = filt[-1]
        scale = np.max(self.avg_signal) / np.dot(filt[0], self.avg_signal)
        self.filt_noconst *= scale
        self.filt_aterms *= scale
        Ainv *= self.peak**-2

        self.variances['noconst'] = Ainv[0, 0]
        self.variances['baseline'] = Ainv[-1, -1]

        for key in self.variances.keys():
            self.predicted_v_over_dv[key] = 1 / (np.sqrt(np.log(2) * 8) * self.variances[key]**0.5)


class ExperimentalFilter(Filter):
    """Compute and all filters for pulses given an <avgpulse>, the
    <noise_autocorr>, and an expected time constant <tau> for decaying exponentials.
    Shorten the filters w.r.t. the avgpulse function by <shorten> samples on each end.

    CAUTION: THESE ARE EXPERIMENTAL!  Don't use yet if you don't know what you're doing!"""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None,
                 sample_time=None, shorten=0, tau=2.0):
        """
        Create a set of filters under various assumptions and for various purposes.

        <avg_signal>     The average signal shape.  Filters will be rescaled so that the output
                         upon putting this signal into the filter equals the *peak value* of this
                         filter (that is, peak value relative to the baseline level).
        <n_pretrigger>   The number of leading samples in the average signal that are considered
                         to be pre-trigger samples.  The avg_signal in this section is replaced by
                         its constant averaged value before creating filters.  Also, one filter
                         (filt_baseline_pretrig) is designed to infer the baseline using only
                         <n_pretrigger> samples at the start of a record.
        <noise_psd>      The noise power spectral density.  If None, then filt_fourier won't be
                         computed.  If not None, then it must be of length (2N+1), where N is the
                         length of <avg_signal>, and its values are assumed to cover the non-negative
                         frequencies from 0, 1/Delta, 2/Delta,.... up to the Nyquist frequency.
        <noise_autocorr> The autocorrelation function of the noise, where the lag spacing is
                         assumed to be the same as the sample period of <avg_signal>.  If None,
                         then several filters won't be computed.  (One of <noise_psd> or
                         <noise_autocorr> must be a valid array.)
        <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                         This must be given if <fmax> or <f_3db> are ever to be used.
        <shorten>        The time-domain filters should be shortened by removing this many
                         samples from each end.  (Do this for convenience of convolution over
                         multiple lags.)
        <tau>            Time constant of exponential to filter out (in milliseconds)
        """

        if isinstance(tau, (int, float)):
            tau = [tau]
        self.tau = tau  # in milliseconds; can be a sequence of taus
        super(self.__class__, self).__init__(avg_signal, n_pretrigger, noise_psd,
                                             noise_autocorr, fmax, f_3db, sample_time, shorten)

    def compute(self, fmax=None, f_3db=None):
        """
        Compute a set of filters.  This is called once on construction, but you can call it
        again if you want to change the frequency cutoff or rolloff points.

        Set is:
        filt_fourier    Fourier filter for signals
        filt_full       Alpert basic filter
        filt_noconst    Alpert filter insensitive to constants
        filt_noexp      Alpert filter insensitive to exp(-t/tau)
        filt_noexpcon   Alpert filter insensitive to exp(-t/tau) and to constants
        filt_noslope    Alpert filter insensitive to slopes
        filt_nopoly1    Alpert filter insensitive to Chebyshev polynomials order 0 to 1
        filt_nopoly2    Alpert filter insensitive to Chebyshev polynomials order 0 to 2
        filt_nopoly3    Alpert filter insensitive to Chebyshev polynomials order 0 to 3
        """

        if self.sample_time is None and not (fmax is None and f_3db is None):
            raise ValueError("Filter must have a sample_time if it's to be smoothed with fmax or f_3db")

        self.fmax = fmax
        self.f_3db = f_3db
        self.variances = {}

        self._compute_fourier_filter()

        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2 * self.shorten
            if self.shorten > 0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            assert len(self.noise_autocorr) >= n

            expx = np.arange(n, dtype=np.float) * self.sample_time * 1e3  # in ms
            chebyx = np.linspace(-1, 1, n)

            R = self.noise_autocorr[:n] / self.peak_signal**2  # A *vector*, not a matrix
            ts = mass.mathstat.toeplitz.ToeplitzSolver(R, symmetric=True)

            unit = np.ones(n)
            exps = [np.exp(-expx / tau) for tau in self.tau]
            cht1 = sp.special.chebyt(1)(chebyx)
            cht2 = sp.special.chebyt(2)(chebyx)
            cht3 = sp.special.chebyt(3)(chebyx)
            deriv = avg_signal - np.roll(avg_signal, 1)
            deriv[0] = 0

            Rinv_sig = ts(avg_signal)
            Rinv_unit = ts(unit)
            Rinv_exps = [ts(e) for e in exps]
            Rinv_cht1 = ts(cht1)
            Rinv_cht2 = ts(cht2)
            Rinv_cht3 = ts(cht3)
            Rinv_deriv = ts(deriv)

            # Band-limit
            def band_limit(vector, fmax, f_3db):
                filt_length = len(vector)
                sig_ft = np.fft.rfft(vector)
                freq = np.fft.fftfreq(filt_length, d=self.sample_time)
                freq = np.abs(freq[:len(sig_ft)])
                if fmax is not None:
                    sig_ft[freq > fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1. + (1.0 * freq / f_3db)**2)
                vector[:] = np.fft.irfft(sig_ft, n=filt_length)  # n= is needed when filt_length is ODD

            if fmax is not None or f_3db is not None:
                for vector in Rinv_sig, Rinv_unit, Rinv_cht1, Rinv_cht2, Rinv_cht3, Rinv_deriv:
                    band_limit(vector, fmax, f_3db)
                for vector in Rinv_exps:
                    band_limit(vector, fmax, f_3db)

            exp_orthogs = ['exps[%d]' % i for i in range(len(self.tau))]
            orthogonalities = {
                'filt_full': (),
                'filt_noconst': ('unit',),
                'filt_noexp': exp_orthogs,
                'filt_noexpcon': ['unit'] + exp_orthogs,
                'filt_noslope': ('cht1',),
                'filt_nopoly1': ('unit', 'cht1'),
                'filt_nopoly2': ('unit', 'cht1', 'cht2'),
                'filt_nopoly3': ('unit', 'cht1', 'cht2', 'cht3'),
                'filt_noderivcon': ('unit', 'deriv'),
            }

            for shortname in ('full', 'noexp', 'noconst', 'noexpcon', 'nopoly1', 'noderivcon'):
                name = 'filt_%s' % shortname
                orthnames = orthogonalities[name]
                filt = Rinv_sig

                N_orth = len(orthnames)  # To how many vectors are we orthgonal?
                if N_orth > 0:
                    u = np.vstack((Rinv_sig, [eval('Rinv_%s' % v) for v in orthnames]))
                else:
                    u = Rinv_sig.reshape((1, n))
                M = np.zeros((1 + N_orth, 1 + N_orth), dtype=np.float)
                for i in range(1 + N_orth):
                    M[0, i] = np.dot(avg_signal, u[i, :])
                    for j in range(1, 1 + N_orth):
                        M[j, i] = np.dot(eval(orthnames[j - 1]), u[i, :])
                Minv = np.linalg.inv(M)
                weights = Minv[:, 0]

                filt = np.dot(weights, u)
                filt = u[0, :] * weights[0]
                for i in range(1, 1 + N_orth):
                    filt += u[i, :] * weights[i]

                self.normalize_filter(filt)
                self.__dict__[name] = filt

                print('%15s' % name),
                for v in (avg_signal, np.ones(n), np.exp(-expx / self.tau[0]), sp.special.chebyt(1)(chebyx),
                          sp.special.chebyt(2)(chebyx)):
                    print('%10.5f ' % np.dot(v, filt)),

                self.variances[shortname] = self.bracketR(filt, R)
                fw = np.sqrt(8 * np.log(2))
                print('Res=%6.3f eV = %.5f' % (5898.801 * fw * self.variances[shortname]**.5,
                                               (self.variances[shortname] / self.variances['full'])**.5))

            self.filt_baseline = np.dot(avg_signal, Rinv_sig) * Rinv_unit - Rinv_sig.sum() * Rinv_sig
            self.filt_baseline /= self.filt_baseline.sum()
            self.variances['baseline'] = self.bracketR(self.filt_baseline, R)

            Rpretrig = sp.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger] / self.peak_signal**2)
            self.filt_baseline_pretrig = np.linalg.solve(Rpretrig, np.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig,
                                                               R[:self.n_pretrigger])

            if self.noise_psd is not None:
                r = self.noise_autocorr[:len(self.filt_fourier)] / self.peak_signal**2
                self.variances['fourier'] = self.bracketR(self.filt_fourier, r)

    def plot(self, axes=None):
        if axes is None:
            plt.clf()
            axis1 = plt.subplot(211)
            axis2 = plt.subplot(212)
        else:
            axis1, axis2 = axes
        try:
            axis1.plot(self.filt_noconst, color='red')
            axis2.plot(self.filt_baseline, color='purple')
            axis2.plot(self.filt_baseline_pretrig, color='blue')
        except AttributeError:
            pass
        try:
            axis1.plot(self.filt_fourier, color='gold')
        except AttributeError:
            pass


class ToeplitzWhitener(object):
    """An object that can perform approximate noise whitening.

    For an ARMA(p,q) noise model, mutliply by (or solve) the matrix W (or its
    transpose), where W is the Toeplitz approximation to the whitening matrix V.
    A whitening matrix V means that if R is the ARMA noise covariance matrix,
    then VRV' = I. While W is only approximately equal to V, it has some handy
    properties that make it a useful replacement. (In particular, it has the
    time-transpose property that if you zero-pad the beginning of vector v and
    shift the remaining elements, then the same is doen to Wv.)

    The callable function object returns Wv or WM if called with
    vector v or matrix M. Other methods:

    * `tw.whiten(v)` returns Wv, equivalent to `tw(v)`
    * `tw.solveWT(v)` returns inv(W')*v
    * `tw.applyWT(v)` returns W'*v
    * `tw.solveW(v)` returns inv(W)*v (_not implemented yet_)"""

    def __init__(self, thetacoef, phicoef):
        """Initialize using the coefficients `thetacoef` of the MA process
        and `phicoef` of the AR process."""
        self.theta = np.array(thetacoef)
        self.phi = np.array(phicoef)
        self.p = len(phicoef)-1
        self.q = len(thetacoef)-1

    def whiten(self, v):
        "Return whitened vector (or matrix of column vectors) Wv"
        return self(v)

    def __call__(self, v):
        "Return whitened vector (or matrix of column vectors) Wv"
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            w = np.zeros_like(v)
            for i in range(v.shape[1]):
                w[:, i] = self(v[:, i])
            return w

        N = len(v)
        # Multiply by the Toeplitz AR matrix to make the MA*w vector.
        y = self.phi[0] * v
        for i in range(1, 1+self.p):
            y[i:] += self.phi[i]*v[:-i]
        # Second, solve the MA matrix (also a banded Toeplitz matrix with
        # q non-zero subdiagonals.)
        y[0] /= self.theta[0]
        if N == 1:
            return y
        for i in range(1, min(self.q, N)):
            for j in range(i):
                y[i] -= y[j]*self.theta[i-j]
            y[i] /= self.theta[0]
        for i in range(self.q, N):
            for j in range(i-self.q, i):
                y[i] -= y[j]*self.theta[i-j]
            y[i] /= self.theta[0]
        return y

    def solveW(self, v):
        "Return unwhitened vector (or matrix of column vectors) inv(W)*v"
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.solveW(v[:, i])
            return r

        N = len(v)
        # Multiply by the Toeplitz MA matrix to make the AR*w vector.
        y = self.theta[0] * v
        for i in range(1, 1+self.q):
            y[i:] += self.theta[i]*v[:-i]
        # Second, solve the AR matrix (also a banded Toeplitz matrix with
        # q non-zero subdiagonals.)
        y[0] /= self.phi[0]
        if N == 1:
            return y
        for i in range(1, min(self.p, N)):
            for j in range(i):
                y[i] -= y[j]*self.phi[i-j]
            y[i] /= self.phi[0]
        for i in range(self.p, N):
            for j in range(i-self.p, i):
                y[i] -= y[j]*self.phi[i-j]
            y[i] /= self.phi[0]
        return y

    def solveWT(self, v):
        "Return vector (or matrix of column vectors) inv(W')*v"
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.solveWT(v[:, i])
            return r

        N = len(v)
        y = np.array(v)
        y[N-1] /= self.phi[0]
        for i in range(N-2, -1, -1):
            f = min(self.p+1, N-i)
            y[i] -= np.dot(y[i+1:i+f], self.phi[1:f])
            y[i] /= self.phi[0]
        return np.correlate(y, self.theta, "full")[self.q:]

    def applyWT(self, v):
        "Return vector (or matrix of column vectors) W'*v"
        if v.ndim > 3:
            raise ValueError("v must be dimension 1 or 2")
        elif v.ndim == 2:
            r = np.zeros_like(v)
            for i in range(v.shape[1]):
                r[:, i] = self.applyWT(v[:, i])
            return r

        N = len(v)
        y = np.array(v)
        y[N-1] /= self.theta[0]
        for i in range(N-2, -1, -1):
            f = min(self.q+1, N-i)
            y[i] -= np.dot(y[i+1:i+f], self.theta[1:f])
            y[i] /= self.theta[0]
        return np.correlate(y, self.phi, "full")[self.p:]

    def W(self, N):
        """Return the full whitening matrix.

        Normally the full W is large and slow to use. But it's here so you can
        easily test that W(len(v))*v == whiten(v), and similar."""
        AR = np.zeros((N, N), dtype=float)
        MA = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(max(0, i-self.p), i+1):
                AR[i, j] = self.phi[i-j]
            for j in range(max(0, i-self.q), i+1):
                MA[i, j] = self.theta[i-j]
        return np.linalg.solve(MA, AR)
