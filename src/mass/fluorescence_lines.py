"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.

Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
Phys Rev A56 (#6) pages 4554ff (1997 December).  See online at
http://pra.aps.org/pdf/PRA/v56/i6/p4554_1

Joe Fowler, NIST

March 9, 2011
November 24, 2010 : started as mn_kalpha.py
"""

import numpy
import pylab
import scipy.stats, scipy.interpolate, scipy.special
import mass

def lorentzian(x, fwhm):
    """Return the value of Lorentzian prob distribution function at <x> (may be a numpy array)
    centered at x=0 with the given FWHM."""
    return 2./(fwhm*numpy.pi)/(1.+4*(x/fwhm)**2)

def lorentzian_cdf(x, fwhm):
    """Return the value of Lorentzian cumulative distribution function at <x> (may be a numpy array)
    centered at x=0 with the given FWHM."""
    return 0.5 + numpy.arctan(2*x/fwhm)/numpy.pi


class SpectralLine(object):
    """An abstract base class for modeling spectral lines as a sum
    of Lorentzians.
    
    Instantiate one of its subclasses, which will have to define
    self.energies, self.fwhm, self.amplitudes.  Each must be a sequence
    of the same length.
    """
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = numpy.asarray(x, dtype=numpy.float)
        result = numpy.zeros_like(x)
        for e,f,a in zip(self.energies, self.fwhm, self.amplitudes):
            result += a*lorentzian(x-e, f)
        return result
    
    def cdf(self, x):
        """Cumulative distribution function where <x> = set of energies."""
        x = numpy.asarray(x, dtype=numpy.float)
        result = numpy.zeros_like(x)
        for e,f,a in zip(self.energies, self.fwhm, self.amplitudes):
            result += a*lorentzian_cdf(x-e, f)
        return result



class MnKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    name = 'Manganese K-alpha'    
    
    # The approximation is as a series of 8 Lorentzians (6 for KA1,2 for KA2)
    energies = 5800+numpy.array((98.853,97.867,94.829,96.532,99.417,102.712,87.743,86.495))
    fwhm = numpy.array((1.715,2.043,4.499,2.663,0.969,1.553,2.361,4.216))
    peak_heights = numpy.array((790,264,68,96,71,10,372,100),dtype=numpy.float)/1e3
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    peak_energy = 5898.802 # eV        


    
class MnKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    name = 'Manganese K-beta'    
    
    # The approximation is as a series of 4 Lorentzians 
    energies = 6400+numpy.array((90.89,86.31, 77.73, 90.06, 88.83))
    fwhm = numpy.array((1.83, 9.40, 13.22, 1.81, 2.81))
    peak_heights = numpy.array((608,109,77,397,176),dtype=numpy.float)/1e3
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    peak_energy = 6490.18 # eV        


    
class CuKAlpha(SpectralLine):
    """Function object to approximate the copper K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    name = 'Copper K-alpha'
            
    # The approximation is 4 of Lorentzians (2 for KA1 for KA2)
    energies = numpy.array((8047.8372, 8045.3672, 8027.9935, 8026.5041))
    fwhm = numpy.array((2.285, 3.358, 2.667, 3.571))
    peak_heights = numpy.array((957,90,334,111), dtype=numpy.float)/1e3
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    peak_energy = 8047.83 # eV        
    


class GaussianLine(object):
    """An abstract base class for modeling spectral lines as a 
    single Gaussian of known energy.
    
    Instantiate one of its subclasses, which will have to define
    self.energy.
    """
    
    def __init__(self):
        self.sigma = self.fwhm/numpy.sqrt(8*numpy.log(2))
        self.amplitude = (2*numpy.pi)**(-0.5)/self.sigma
    
    def __call__(self, x, fwhm=None):
        return self.pdf(x, fwhm=fwhm)
    
    def pdf(self, x, fwhm=None):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = numpy.asarray(x, dtype=numpy.float)
        if fwhm is None:
            sigma, ampl = self.sigma, self.amplitude
        else:
            sigma = fwhm/numpy.sqrt(8*numpy.log(2))
#            ampl = (2*numpy.pi)**(-0.5)/sigma
        result = numpy.exp(-0.5*(x-self.energy)**2/sigma**2)# * ampl
        return result
    
    def cdf(self, x, fwhm=None):
        """Cumulative distribution function where <x> = set of energies."""
        x = numpy.asarray(x, dtype=numpy.float)
        if fwhm is None:
            arg = (x-self.energy)/self.sigma/numpy.sqrt(2)
        else:
            sigma = fwhm/numpy.sqrt(8*numpy.log(2))
            arg = (x-self.energy)/sigma/numpy.sqrt(2)
        return (scipy.special.erf(arg)+1)*.5
    
class Gd97(GaussianLine):
    energy = 97431.0
    fwhm = 50.0
    
class Gd103(GaussianLine):
    energy = 103180.0
    fwhm = 50.0


    
class MultiLorentzian_distribution(scipy.stats.rv_continuous):
    """For producing random variates of the an energy distribution having the form
    of several Lorentzians summed together."""
    
    def __init__(self, epoints, *args, **kwargs):
        """<epoints> is a vector of energy points, densely collected at places
        where the distribution changes rapidly.
        <args> and <kwargs> are passed on to scipy.stats.rv_continuous"""

        scipy.stats.rv_continuous.__init__(self, *args, **kwargs)
        epoints = epoints[numpy.logical_and(epoints>=self.a, epoints<=self.b)]
        epoints = numpy.hstack((self.a, epoints, self.b))
        cdf = self.distribution.cdf(epoints)
        a,b = cdf[0], cdf[-1] # would be 0,1 if we covered all of x-axis
        cdf = (cdf-a)/(b-a) # Rescale so that it *does* run from [0,1]
        self.minCDF, self.maxCDF = a,b

        # Redefine the percentile point function (maps [0,1] to energies),
        # the prob distrib function and the cumulative distrib function
        self._ppf = scipy.interpolate.interp1d(cdf, epoints, kind='linear')
        self._pdf = self.distribution
        self._cdf = lambda x: (self.distribution.cdf(x)-self.minCDF)/(self.maxCDF-self.minCDF)
        


class MnKAlpha_distribution(MultiLorentzian_distribution):
    """For producing random variates of the manganese K Alpha energy distribution"""
    
    def __init__(self, *args, **kwargs):
        self.distribution = MnKAlpha()
        epoints = numpy.hstack(((0,3000,5000,5500),
                                numpy.arange(5800,5880.5),
                                numpy.arange(5880,5920.-.025,.05),
                                numpy.arange(5920,6001),
                                (6100, 6300, 6600,7000,8000,12000)))*1.0
        MultiLorentzian_distribution.__init__(self, epoints, *args, **kwargs)
        


class CuKAlpha_distribution(MultiLorentzian_distribution):
    """For producing random variates of the copper K Alpha energy distribution"""
    
    def __init__(self, *args, **kwargs):
        self.distribution = CuKAlpha()
        epoints = numpy.hstack(((0,3000,6000,7000,7500),
                                numpy.arange(7800,8010.5),
                                numpy.arange(8011,8060.-.025,.05),
                                numpy.arange(8060,8201),
                                (8300, 8500, 8800, 9300, 12000)))*1.0
        MultiLorentzian_distribution.__init__(self, epoints, *args, **kwargs)
        


class SpectralLineFitter(object):
    def __init__(self): pass
    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        params: a 5-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                energy scale factor (counts/eV), amplitude, background level (per bin) ]
        
        If pulseheights is None, then the parameters having pulseheight units will be returned as bin numbers.
        
        If params is None or does not have 5 elements, then they will be guessed."""
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        try:
            _,_,_,_,_ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
#            print 'Guessed parameters: ',params
#            print 'PH range: ',pulseheights[0],pulseheights[-1]
        
        def fitfunc(params, x):
            E_peak = self.spect.peak_energy
            
            energy = (x-params[1])/abs(params[2]) + E_peak
            spectrum = self.spect(energy)
#            if numpy.isnan(energy[1]-energy[0]):
#                print params, 'yikes!'
#                raise ValueError("NaN energies")
            smeared = smear(spectrum, abs(params[0]), stepsize = energy[1]-energy[0])
            return smeared * abs(params[3]) + abs(params[4])
        
        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2))
        fitter = mass.utilities.MaximumLikelihoodHistogramFitter(pulseheights, data, params, fitfunc, TOL=1e-4, epsilon=epsilon)
        if hold is not None:
            for h in hold: fitter.hold(h)
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.lastFitParams = fitparams
        self.lastFitResult = fitfunc(fitparams, pulseheights)
        
#        if iflag not in (1,2,3,4): 
        if iflag not in (0,2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            if color is None: color='blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            de = numpy.sqrt(covariance[0,0])
            mass.utilities.plot_as_stepped_hist(axis, pulseheights, data, color=color, label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.plot(pulseheights, self.lastFitResult, color='black')
            axis.legend(loc='upper left')
            dp = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*dp, pulseheights[-1]+0.5*dp])
        return fitparams, covariance



class MnKAlphaFitter(SpectralLineFitter):
    "Fits a Mn K alpha spectrum for energy shift and scale, amplitude, and resolution"
    
    def __init__(self):
        self.spect = MnKAlpha()
        super(self.__class__, self).__init__()
        # At first, I was pre-computing lots of stuff, but now I don't think it's needed.
        
    def guess_starting_params(self, data, binctrs):
        """If the cuts are tight enough, then we can estimate the locations of the
        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
        
        n = data.sum()
        if n<=0:
            raise ValueError("This histogram has no contents")
        sum_d = (data*binctrs).sum()
        sum_d2 = (data*binctrs*binctrs).sum()
        mean_d = sum_d/n
        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
#        print n, sum_d, sum_d2, mean_d, rms_d
        
        ph_ka1 = mean_d + rms_d*.65
        ph_ka2 = mean_d - rms_d

        dph = ph_ka1-ph_ka2
        dE = 11.1 # eV difference between KAlpha peaks
        ampl = data.max() *9.4
        res = 4.0
        baseline = 0.1
        return [res, ph_ka1, dph/dE, ampl, baseline]



class MnKBetaFitter(SpectralLineFitter):
    "Fits a Mn K beta spectrum for energy shift and scale, amplitude, and resolution"
    
    def __init__(self):
        self.spect = MnKBeta()
        super(self.__class__, self).__init__()
        
    def guess_starting_params(self, data, binctrs):
        """If the cuts are tight enough, then we can estimate the locations of the
        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
        
        n = data.sum()
        sum_d = (data*binctrs).sum()
#        sum_d2 = (data*binctrs*binctrs).sum()
        mean_d = sum_d/n
#        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
#        print n, sum_d, sum_d2, mean_d, rms_d
        
        ph_peak = mean_d

        ampl = data.max() *9.4
        res = 4.0
        baseline = 0.1
        return [res, ph_peak, 1.0, ampl, baseline]



class CuKAlphaFitter(SpectralLineFitter):
    "Fits a Cu K alpha spectrum for energy shift and scale, amplitude, and resolution"
    
    def __init__(self):
        self.spect = CuKAlpha()
        super(self.__class__, self).__init__()
        
    def guess_starting_params(self, data, binctrs):
        """If the cuts are tight enough, then we can estimate the locations of the
        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
        
        n = data.sum()
        sum_d = (data*binctrs).sum()
        sum_d2 = (data*binctrs*binctrs).sum()
        mean_d = sum_d/n
        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
#        print n, sum_d, sum_d2, mean_d, rms_d
        
        ph_ka1 = mean_d + rms_d*.65
        ph_ka2 = mean_d - rms_d

        dph = ph_ka1-ph_ka2
        dE = 19.94 # eV difference between KAlpha peaks
        ampl = data.max() *9.4
        res = 3.7
        baseline = 0.1
        return [res, ph_ka1, dph/dE, ampl, baseline]
    


def smear(f, fwhm, stepsize=1.0):
    """Convolve a sampled function <f> with a Gaussian of the given
    <fwhm>, where the samples <f> are spaced evenly by <stepsize>.
    Function is padded at each end by enough samples to cover 5 FWHMs.
    The padding equals the values of <f> at its two endpoints. """
    N = len(f)
    assert N%2==0
    
    fwhm = numpy.abs(fwhm)
    padwidth = 5.0*fwhm
    if padwidth > 100.0:
        padwidth = 100.0
    Npad = int(padwidth/stepsize+0.5)
    if Npad > N*7: Npad = N*7 
    
    # Make sure that total FFT size is a power of 2
    Ntotal = N+2*Npad
    for j in range(2,25):
        if Ntotal <= 2**j:
            Npad = (2**j-N)/2
            break
    
    fpadded = numpy.hstack((f, numpy.zeros(2*Npad)))
    fpadded[N:N+Npad] = f[-1]
    fpadded[N+Npad:] = f[0]
    Nfull = len(fpadded)
    
    ft = numpy.fft.rfft(fpadded)
    freq = numpy.fft.fftfreq(Nfull, d=stepsize)[:Nfull/2+1]
    freq[-1] = numpy.abs(freq[-1]) # convention is that the f_crit is negative.  Fix it.
    
    # Filter the function in Fourier space
    sigma = fwhm / numpy.sqrt(8*numpy.log(2))
    sigmaConjugate = 1.0/(2 * numpy.pi * sigma)
    ft *= numpy.exp(-0.5*(freq/sigmaConjugate)**2)
    return numpy.fft.irfft(ft)[0:N]


class GaussianFitter(object):
    def __init__(self, spect):
        self.spect = spect
        
    def guess_starting_params(self, data, binctrs):
        """doc"""
        
        n = data.sum()
        sum_d = (data*binctrs).sum()
        sum_d2 = (data*binctrs*binctrs).sum()
        mean_d = sum_d/n
        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
        res = rms_d * 2.3548
        ph_peak = mean_d
        ampl = data.max()
        baseline = 0.1
        return [res, ph_peak, ampl, baseline]
    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        params: a 4-element sequence of [Resolution (fwhm), center of the peak,
                amplitude, background level (per bin) ]
        
        If pulseheights is None, then the parameters having pulseheight units will be returned as bin numbers.
        
        If params is None or does not have 4 elements, then they will be guessed."""
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        try:
            _,_,_,_ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
        
        def fitfunc(params, x):
            E_peak = self.spect.energy
            
            energy = (x-params[1]) + E_peak
            spectrum = self.spect(energy, fwhm=params[0]*params[1]/E_peak)
            return spectrum * abs(params[2]) + abs(params[3])
        
        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, params[2]/1e5, params[3]/1e2))
        fitter = mass.utilities.MaximumLikelihoodHistogramFitter(pulseheights, data, params, fitfunc, TOL=1e-4, epsilon=epsilon)
        if hold is not None:
            for h in hold: fitter.hold(h)
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.lastFitParams = fitparams
        self.lastFitResult = fitfunc(fitparams, pulseheights)
        
#        if iflag not in (1,2,3,4): 
        if iflag not in (0,2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            if color is None: color='blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            de = numpy.sqrt(covariance[0,0])
            mass.utilities.plot_as_stepped_hist(axis, pulseheights, data, color=color, label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.plot(pulseheights, self.lastFitResult, color='black')
            axis.legend(loc='upper left')
            dp = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*dp, pulseheights[-1]+0.5*dp])
        return fitparams, covariance



def plot_spectrum(spectrumf=MnKAlpha(), resolutions=(2,3,4,5,6,7,8,10,12), 
                  energy_range=(5870,5920), stepsize=0.05):
    """Plot a spectrum at several different resolutions.
    
    <spectrum>    A callable that accepts a vector of energies and returns
                  the matching probability distribution function.
    <resolutions> A sequence of energy resolution (FWHM) to be stepped through.
    <energy_range> The (min,max) energy to be plotted.
    <stepsize>    The plotting step size in energy units.  
    """
    if resolutions is None:
        resolutions = [2,3,4,5,6,7,8,10,12]
    e = numpy.arange(energy_range[0]-2.5*resolutions[-1],energy_range[1]+2.5*resolutions[-1],stepsize)
    spectrum = spectrumf(e)
    spectrum /= spectrum.max()

    pylab.clf()
    ax = pylab.subplot(111)
    pylab.plot(e,spectrum, color='black',lw=2, label=' 0 eV')
    ax.set_color_cycle(('red','orange','#bbbb00','green','cyan','blue','indigo','purple','brown'))
    for r in resolutions:
        sp = smear(spectrum, r, stepsize = stepsize)
        sp /= sp.max()
        sp *= (1+r*.01)
        pylab.plot(e,sp, label="%2d eV"%r, lw=2)
        
        # Find the peak, valley, peak
        if spectrumf.name == 'Manganese K-alpha':
            e2,ev,e1 = 5887.7,5892, 5898.801
        elif spectrumf.name == 'Copper K-alpha':
            e2,ev,e1 = 8027.89,8036.6,8047.83
            
        p1 = sp[numpy.abs(e-e1)<2].max()
        if r < 8.12:
            p2 = sp[numpy.abs(e-e2)<2].max()
            pv = sp[numpy.abs(e-ev)<3].min()
            print "Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f"%(r,p2/p1,pv/p2) 
        
    pylab.xlim(energy_range)
    pylab.ylim([0,1.13])
    pylab.legend(loc='upper left')
    
    pylab.title("%s lines at various resolutions (FWHM of Gaussian)"%spectrumf.name)
    pylab.xlabel("Energy (eV)")
    pylab.ylabel("Intensity (arb.)")
