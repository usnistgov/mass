'''
Contains classes to do classic (Fourier) and time-domain optimal filtering

Created on Nov 11, 2011 from code that was in channel.py

@author: fowlerj
'''

__all__ = ['Filter', 'ExperimentalFilter']

import numpy
import pylab
import scipy.linalg, scipy.special

import mass


class Filter(object):
    """A set of optimal filters based on a single signal and noise set."""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None, 
                 fmax=None, f_3db=None, sample_time=None, shorten=0):
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
        <fmax>           The strict maximum frequency to be passed in all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
        <f_3db>          The 3 dB point for a one-pole low-pass filter to be applied to all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
                         Either or both of <fmax> and <f_3db> are allowed.
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
        is_negative =  (avg_signal.min()-pre_avg)/(avg_signal.max()-pre_avg) < -1
        if is_negative:
            self.peak_signal = avg_signal.min() - pre_avg
        else:
            self.peak_signal = avg_signal.max() - pre_avg

        # self.avg_signal is normalized to have unit peak
        self.avg_signal = (avg_signal - pre_avg) / self.peak_signal
        self.avg_signal[:n_pretrigger] = 0.0
        
        self.n_pretrigger = n_pretrigger
        if noise_psd is None:
            self.noise_psd = None
        else:
            self.noise_psd = numpy.array(noise_psd)
        if noise_autocorr is None:
            self.noise_autocorr = None
        else:
            self.noise_autocorr = numpy.array(noise_autocorr)
        if noise_psd is None and noise_autocorr is None:
            raise ValueError("Filter must have noise_psd or noise_autocorr arguments (or both)")
        
        self.compute(fmax=fmax, f_3db=f_3db)


    def normalize_filter(self, q): 
        "Rescale filter <q> so that it gives unit response to self.avg_signal"
        if len(q) == len(self.avg_signal):
            q *= 1 / numpy.dot(q, self.avg_signal)
        else:  
#                print "scaling by 1/%f"%numpy.dot(q, self.avg_signal[self.shorten:-self.shorten])
#                print self.peak_signal, q
            q *= 1 / numpy.dot(q, self.avg_signal[self.shorten:-self.shorten]) 


    def _compute_fourier_filter(self, fmax=None, f_3db=None):
        "Compute the Fourier-domain filter"
        if self.noise_psd is None: return
        
        # Careful: let's be sure that the Fourier domain filter is done consistently in Filter and
        # its child classes.
        
        n = len(self.noise_psd)
#        window = power_spectrum.hamming(2*(n-1-self.shorten))
        window = 1.0

        if self.shorten>0:
            sig_ft = numpy.fft.rfft(self.avg_signal[self.shorten:-self.shorten]*window)
        else:
            sig_ft = numpy.fft.rfft(self.avg_signal * window)

        if len(sig_ft) != n-self.shorten:
            raise ValueError("signal real DFT and noise PSD are not the same length (%d and %d)"
                             %(len(sig_ft), n))
            
        # Careful with PSD: "shorten" it by converting into a real space autocorrelation, 
        # truncating the middle, and going back to Fourier space
        if self.shorten>0:
            noise_autocorr = numpy.fft.irfft(self.noise_psd)
            noise_autocorr = numpy.hstack((noise_autocorr[:n-self.shorten-1], noise_autocorr[-n+self.shorten:]))
            noise_psd = numpy.fft.rfft(noise_autocorr)
        else:
            noise_psd = self.noise_psd
        sig_ft_weighted = sig_ft/noise_psd
        
        # Band-limit
        if fmax is not None or f_3db is not None:
            freq = numpy.arange(0,n-self.shorten,dtype=numpy.float)*0.5/((n-1)*self.sample_time)
            if fmax is not None:
                sig_ft_weighted[freq>fmax] = 0.0
            if f_3db is not None:
                sig_ft_weighted /= (1+(freq*1.0/f_3db)**2)

        # Compute both the normal (DC-free) and the full (with DC) filters.
        self.filt_fourierfull = numpy.fft.irfft(sig_ft_weighted)/window
        sig_ft_weighted[0] = 0.0
        self.filt_fourier = numpy.fft.irfft(sig_ft_weighted)/window
        self.normalize_filter(self.filt_fourierfull)
        self.normalize_filter(self.filt_fourier)
        
        # How we compute the uncertainty depends on whether there's a noise autocorrelation result
        if self.noise_autocorr is None:
            noise_ft_squared = (len(self.noise_psd)-1)/self.sample_time * self.noise_psd
            kappa = (numpy.abs(sig_ft*self.peak_signal)**2/noise_ft_squared)[:].sum()
            self.variances['fourierfull'] = 1./kappa
            
            kappa = (numpy.abs(sig_ft*self.peak_signal)**2/noise_ft_squared)[1:].sum()
            self.variances['fourier'] = 1./kappa
        else:
            ac = self.noise_autocorr[:len(self.filt_fourier)].copy()
            self.variances['fourier'] = self.bracketR(self.filt_fourier, ac)/self.peak_signal**2
            self.variances['fourierfull'] = self.bracketR(self.filt_fourierfull, ac)/self.peak_signal**2
#        print 'Fourier filter done.  Variance: ',self.variances['fourier'], 'V/dV: ',self.variances['fourier']**(-0.5)/2.35482


    def compute(self, fmax=None, f_3db=None, use_toeplitz_solver=True):
        """
        Compute a set of filters.  This is called once on construction, but you can call it
        again if you want to change the frequency cutoff or f_3db rolloff point.
        """

        self.fmax=fmax
        self.f_3db=f_3db
        self.variances={}
        self._compute_fourier_filter(fmax=fmax, f_3db=f_3db)

        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2*self.shorten
            assert len(self.noise_autocorr) >= n
            if self.shorten>0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            
            noise_corr = self.noise_autocorr[:n]/self.peak_signal**2
            if use_toeplitz_solver:
                ts = mass.mathstat.toeplitz.ToeplitzSolver(noise_corr, symmetric=True)
                Rinv_sig = ts(avg_signal)
                Rinv_1 = ts(numpy.ones(n))
            else:
                if n>6000: raise ValueError("Not allowed to use generic solver for vectors longer than 6000, because it's slow-ass.")
                R =  scipy.linalg.toeplitz(noise_corr)
                Rinv_sig = numpy.linalg.solve(R, avg_signal)
                Rinv_1 = numpy.linalg.solve(R, numpy.ones(n))
            
            self.filt_noconst = Rinv_1.sum()*Rinv_sig - Rinv_sig.sum()*Rinv_1

            # Band-limit
            if fmax is not None or f_3db is not None:
                sig_ft = numpy.fft.rfft(self.filt_noconst)
                freq = numpy.arange(0,n/2+1,dtype=numpy.float)*0.5/self.sample_time/(n/2)
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1.+(1.0*freq/f_3db)**2)
                self.filt_noconst = numpy.fft.irfft(sig_ft)

            self.normalize_filter(self.filt_noconst)

            self.filt_baseline = numpy.dot(avg_signal, Rinv_sig)*Rinv_1 - Rinv_sig.sum()*Rinv_sig
            self.filt_baseline /=  self.filt_baseline.sum()
            
            Rpretrig = scipy.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger]/self.peak_signal**2)
            self.filt_baseline_pretrig = numpy.linalg.solve(Rpretrig, numpy.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()

            self.variances['noconst'] = self.bracketR(self.filt_noconst, noise_corr) 
            self.variances['baseline'] = self.bracketR(self.filt_baseline, noise_corr)
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig, Rpretrig[0,:])

                
    def bracketR(self, q, noise):
        """Return the dot product (q^T R q) for vector <q> and matrix R constructed from
        the vector <noise> by R_ij = noise_|i-j|.  We don't want to construct the full matrix
        R because for records as long as 10,000 samples, the matrix will consist of 10^8 floats
        (800 MB of memory)."""
        
        if len(noise) < len(q):
            raise ValueError("Vector q (length %d) cannot be longer than the noise (length %d)"%
                             (len(q),len(noise)))
        n=len(q)
        r = numpy.zeros(2*n-1, dtype=numpy.float)
        r[n-1:] = noise[:n]
        r[n-1::-1] = noise[:n]
        dot = 0.0
        for i in range(n):
            dot += q[i]*numpy.dot(r[n-i-1:2*n-i-1], q)
        return dot
    
            
    def plot(self, axes=None):
        if axes is None:
            pylab.clf()
            axis1 = pylab.subplot(211)
            axis2 = pylab.subplot(212)
        else:
            axis1,axis2 = axes
        try:
            axis1.plot(self.filt_noconst,color='red')
            axis2.plot(self.filt_baseline,color='purple')
            axis2.plot(self.filt_baseline_pretrig,color='blue')
        except AttributeError: pass
        try:
            axis1.plot(self.filt_fourier,color='gold')
        except AttributeError: pass


    def report(self, filters=None):
        """Report on V/dV for all filters
        
        <filters>   Either the name of one filter or a sequence of names.  If not given, then all filters
                    not starting with "baseline" will be reported
        """
        
        # Handle <filters> is a single string --> convert to tuple of 1 string
        if isinstance(filters,str):
            filters=(filters,)
            
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
                v_dv = var**(-.5) / numpy.sqrt(8*numpy.log(2))
                print "%-20s  %10.3f  %10.4e"%(f, v_dv, var)
            except KeyError:
                print "%-20s not known"%f



class ExperimentalFilter(Filter):
    """Compute and all filters for pulses given an <avgpulse>, the
    <noise_autocorr>, and an expected time constant <tau> for decaying exponentials.
    Shorten the filters w.r.t. the avgpulse function by <shorten> samples on each end.
    
    CAUTION: THESE ARE EXPERIMENTAL!  Don't use yet if you don't know what you're doing!"""

    def __init__(self, avg_signal, n_pretrigger, noise_psd=None, noise_autocorr=None, 
                 fmax=None, f_3db=None, sample_time=None, shorten=0, tau=2.0):
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
        <fmax>           The strict maximum frequency to be passed in all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
        <f_3db>          The 3 dB point for a one-pole low-pass filter to be applied to all filters.
                         If supplied, then it is passed on to the compute() method for the *first*
                         filter calculation only.  (Future calls to compute() can override).
                         Either or both of <fmax> and <f_3db> are allowed.
        <sample_time>    The time step between samples in <avg_signal> and <noise_autocorr>
                         This must be given if <fmax> or <f_3db> are ever to be used.
        <shorten>        The time-domain filters should be shortened by removing this many
                         samples from each end.  (Do this for convenience of convolution over
                         multiple lags.)
        <tau>            Time constant of exponential to filter out (in milliseconds)
        """
        
        if isinstance(tau, (int, float)): tau = [tau]
        self.tau = tau # in milliseconds; can be a sequence of taus
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

        self.fmax=fmax
        self.f_3db=f_3db
        self.variances={}
        
        self._compute_fourier_filter(fmax=fmax, f_3db=f_3db)
        
        # Time domain filters
        if self.noise_autocorr is not None:
            n = len(self.avg_signal) - 2*self.shorten
            if self.shorten>0:
                avg_signal = self.avg_signal[self.shorten:-self.shorten]
            else:
                avg_signal = self.avg_signal
            assert len(self.noise_autocorr) >= n
            
            expx = numpy.arange(n, dtype=numpy.float)*self.sample_time*1e3 # in ms
            chebyx = numpy.linspace(-1, 1, n)
            
            R = self.noise_autocorr[:n]/self.peak_signal**2 # A *vector*, not a matrix
            ts = mass.mathstat.toeplitz.ToeplitzSolver(R, symmetric=True)
            
            unit = numpy.ones(n)
            exps  = [numpy.exp(-expx/tau) for tau in self.tau]
            cht1 = scipy.special.chebyt(1)(chebyx)
            cht2 = scipy.special.chebyt(2)(chebyx)
            cht3 = scipy.special.chebyt(3)(chebyx)
            
            Rinv_sig  = ts(avg_signal)
            Rinv_unit = ts(unit)
            Rinv_exps = [ts(e) for e in exps]
            Rinv_cht1 = ts(cht1)
            Rinv_cht2 = ts(cht2)
            Rinv_cht3 = ts(cht3)
            
            # Band-limit
            def band_limit(vector, fmax, f_3db):
                sig_ft = numpy.fft.rfft(vector)
                freq = numpy.fft.fftfreq(n, d=self.sample_time) 
                freq=freq[:n/2+1]
                freq[-1] *= -1
                if fmax is not None:
                    sig_ft[freq>fmax] = 0.0
                if f_3db is not None:
                    sig_ft /= (1+(freq/f_3db)**2)
                vector[:] = numpy.fft.irfft(sig_ft)
                
            if fmax is not None or f_3db is not None:
                for vector in Rinv_sig, Rinv_unit, Rinv_cht1, Rinv_cht2, Rinv_cht3:
                    band_limit(vector, fmax, f_3db)
                for vector in Rinv_exps:
                    band_limit(vector, fmax, f_3db)

            exp_orthogs = ['exps[%d]'%i for i in range(len(self.tau))]
            orthogonalities={
                'filt_full':(),
                'filt_noconst' :('unit',),
                'filt_noexp'   :exp_orthogs,
                'filt_noexpcon':['unit']+exp_orthogs,
                'filt_noslope' :('cht1',),
                'filt_nopoly1' :('unit', 'cht1'),
                'filt_nopoly2' :('unit', 'cht1', 'cht2'),
                'filt_nopoly3' :('unit', 'cht1', 'cht2', 'cht3'),
                }
            
#            pylab.clf()
#            pylab.plot(self.filt_fourier, color='gold',label='Fourier')
#            for shortname in ('full','noconst','noexpcon','nopoly1'):
#            for shortname in ('full','noconst','noexp','noexpcon','noslope','nopoly1','nopoly2','nopoly3'):
            for shortname in ('full','noexp','noconst','noexpcon','nopoly1'):
                name = 'filt_%s'%shortname
                orthnames = orthogonalities[name]
                filt = Rinv_sig
                
                N_orth = len(orthnames) # To how many vectors are we orthgonal?
                if N_orth > 0:
                    u = numpy.vstack((Rinv_sig, [eval('Rinv_%s'%v) for v in orthnames]))
                else:
                    u = Rinv_sig.reshape((1,n))
                M = numpy.zeros((1+N_orth,1+N_orth), dtype=numpy.float)
                for i in range(1+N_orth):
                    M[0,i] = numpy.dot(avg_signal, u[i,:])
                    for j in range(1,1+N_orth):
                        M[j,i] = numpy.dot(eval(orthnames[j-1]), u[i,:])
                Minv = numpy.linalg.inv(M)
                weights = Minv[:,0]

                filt = numpy.dot(weights, u)
                filt = u[0,:]*weights[0]
                for i in range(1,1+N_orth):
                    filt += u[i,:]*weights[i]
                
                
                self.normalize_filter(filt)
                self.__dict__[name] = filt
                
                print '%15s'%name,
#                pylab.plot(filt, label=name)
                for v in (avg_signal,numpy.ones(n),numpy.exp(-expx/self.tau[0]),scipy.special.chebyt(1)(chebyx),
                          scipy.special.chebyt(2)(chebyx)):
                    print '%10.5f '%numpy.dot(v,filt),
                    
                self.variances[shortname] = self.bracketR(filt, R)
                print 'Res=%6.3f eV = %.5f'%(5898.801*numpy.sqrt(8*numpy.log(2))*self.variances[shortname]**(.5), (self.variances[shortname]/self.variances['full'])**.5)
#            pylab.legend()

            self.filt_baseline = numpy.dot(avg_signal, Rinv_sig)*Rinv_unit - Rinv_sig.sum()*Rinv_sig
            self.filt_baseline /=  self.filt_baseline.sum()
            self.variances['baseline'] = self.bracketR(self.filt_baseline, R)
            
            Rpretrig = scipy.linalg.toeplitz(self.noise_autocorr[:self.n_pretrigger]/self.peak_signal**2)
            self.filt_baseline_pretrig = numpy.linalg.solve(Rpretrig, numpy.ones(self.n_pretrigger))
            self.filt_baseline_pretrig /= self.filt_baseline_pretrig.sum()
            self.variances['baseline_pretrig'] = self.bracketR(self.filt_baseline_pretrig, R[:self.n_pretrigger])

            if self.noise_psd is not None:
                r =  self.noise_autocorr[:len(self.filt_fourier)]/self.peak_signal**2
                self.variances['fourier'] = self.bracketR(self.filt_fourier, r)


            
    def plot(self, axes=None):
        if axes is None:
            pylab.clf()
            axis1 = pylab.subplot(211)
            axis2 = pylab.subplot(212)
        else:
            axis1,axis2 = axes
        try:
            axis1.plot(self.filt_noconst,color='red')
            axis2.plot(self.filt_baseline,color='purple')
            axis2.plot(self.filt_baseline_pretrig,color='blue')
        except AttributeError: pass
        try:
            axis1.plot(self.filt_fourier,color='gold')
        except AttributeError: pass
