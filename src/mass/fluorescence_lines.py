"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.

Joe Fowler, NIST

March 9, 2011
November 24, 2010 : started as mn_kalpha.py
"""

import numpy
from matplotlib import pylab
import scipy.optimize, scipy.stats, scipy.interpolate

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
        x = numpy.asarray(x)
        result = numpy.zeros_like(x)
        for e,f,a in zip(self.energies, self.fwhm, self.amplitudes):
            result += a*lorentzian(x-e, f)
        return result
    
    def cdf(self, x):
        """Cumulative distribution function where <x> = set of energies."""
        x = numpy.asarray(x)
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
    
    # The approximation is as a series of 8 Lorentzians (6 for KA1,2 for KA2)
    energies = 5800+numpy.array((98.853,97.867,94.829,96.532,99.417,102.712,87.743,86.495))
    fwhm = numpy.array((1.715,2.043,4.499,2.663,0.969,1.553,2.361,4.216))
    peak_heights = numpy.array((790,264,68,96,71,10,372,100),dtype=numpy.float)/1e3
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()

        
    
class MnKAlpha_distribution(scipy.stats.rv_continuous):
    """For producing random variates of the Mn K Alpha energy distribution"""
    
    def __init__(self, *args, **kwargs):
        scipy.stats.rv_continuous.__init__(self, *args, **kwargs)
        self.distribution = MnKAlpha()

        epoints = numpy.hstack(((0,3000,5000,5500),
                                numpy.arange(5800,5880.5),
                                numpy.arange(5880,5920.-.025,.05),
                                numpy.arange(5920,6001),
                                (6100, 6300, 6600,7000,8000,12000)))*1.0
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
        self._cdf = lambda x: (self.distribution.cdf()-self.minCDF)/(self.maxCDF-self.minCDF)
        


class MnKAlphaFitter(object):
    "Fits a Mn K alpha spectrum for energy shift and scale, amplitude, and resolution"
    
    def __init__(self):
        self.spect = MnKAlpha()
        # At first, I was pre-computing lots of stuff, but now I don't think it's needed.
        
    def __guess_starting_params(self, data, binctrs):
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
        
#        bin_ka1 = data.argmax()
#        ph_ka1 = binctrs[bin_ka1]
#        ph_ka2 = binctrs[data[:bin_ka1-7].argmax()]
        dph = ph_ka1-ph_ka2
        dE = 11.1 # eV difference between KAlpha peaks
        ampl = data.max() *9.4
        res = 3.4
        baseline = 0.0
        return [res, ph_ka1, dph/dE, ampl, baseline]
    
    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label=""):
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
            params = self.__guess_starting_params(data, pulseheights)
#            print 'Guessed parameters: ',params
#            print 'PH range: ',pulseheights[0],pulseheights[-1]
        
        def fitfunc(params, x):
            E_kalpha1 = 5898.802 # eV
            
            energy = (x-params[1])/params[2] + E_kalpha1
            spectrum = self.spect(energy)
            smeared = smear(spectrum, params[0], stepsize = energy[1]-energy[0])
            return smeared * params[3] + numpy.abs(params[4])
            
#        def errfunc(p,x,y):
#            d = fitfunc(p,x) - y
#            return d/numpy.sqrt(y+5.0)
        errfunc = lambda p, x, y: (fitfunc(p, x) - y)/numpy.sqrt(y+.5)

        # Do the fit and store the parameters and the
        fitparams, covariance, _infodict, _mesg, iflag = \
           scipy.optimize.leastsq(errfunc, params, args = (pulseheights, data), full_output=True )
        fitparams[0] = abs(fitparams[0])
        
        self.lastFitParams = fitparams
        self.lastFitResult = fitfunc(fitparams, pulseheights)
        
        if iflag not in (1,2,3,4): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            if color is None: color='blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            # plot in step-histogram format
            def plot_as_stepped_hist(axis, bin_ctrs, data, **kwargs):
                x = numpy.zeros(2+2*len(bin_ctrs), dtype=numpy.float)
                y = numpy.zeros_like(x)
                dx = bin_ctrs[1]-bin_ctrs[0]
                x[0:-2:2] = bin_ctrs-dx
                x[1:-2:2] = bin_ctrs-dx
                x[-2:] = bin_ctrs[-1]+dx
                y[1:-1:2] = data
                y[2:-1:2] = data
                axis.plot(x, y, **kwargs)
                axis.set_xlim([x[0],x[-1]])
            
            de = numpy.sqrt(covariance[0,0])
            plot_as_stepped_hist(axis, pulseheights, data, color=color, label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.plot(pulseheights, self.lastFitResult, color='black')
            axis.legend(loc='upper left')
        return fitparams, covariance



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
        e2,ev,e1 = 5887.7,5892, 5898.801
        p1 = sp[numpy.abs(e-e1)<2].max()
        if r < 8.12:
            p2 = sp[numpy.abs(e-e2)<2].max()
            pv = sp[numpy.abs(e-ev)<3].min()
            print "Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f"%(r,p2/p1,pv/p2) 
        
    pylab.xlim(energy_range)
    pylab.ylim([0,1.13])
    pylab.legend(loc='upper left')
    
    pylab.title("Manganese K-alpha lines at various resolutions (FWHM of Gaussian)")
    pylab.xlabel("Energy (eV)")
    pylab.ylabel("Intensity (arb.)")
