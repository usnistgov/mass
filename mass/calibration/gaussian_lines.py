'''
Created on Apr 26, 2012

@author: fowlerj
'''

import numpy
import pylab
import scipy.special

from mass.calibration import energy_calibration
from mass.mathstat.fitting import MaximumLikelihoodGaussianFitter
from mass.mathstat.utilities import plot_as_stepped_hist

__all__=['GaussianFitter','GaussianLine','Gd97','Gd103','AlKalpha','SiKalpha']

class GaussianFitter(object):
    """Abstract base class for objects that can fit a single Gaussian line.
    
    Provides methods fitfunc() and fit().  The child classes must provide:
    * a self.spect function object returning the spectrum at a given energy, and
    * a self.guess_starting_params method to return fit parameter guesses given a histogram.
    """

    def __init__(self):
        """ """
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None
        ## Last chi-square from last successful fit
        self.last_chisq = None
        
    def guess_starting_params(self, data, binctrs):
        """Guess the best Gaussian line location/width/amplitude/background given the spectrum."""
        
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
    
    
    
    def fit(self, data, pulseheights=None, params=None, plot=True, 
            axis=None, color=None, label="", hold=None):
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
            _, _, _, _ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
        
        # Joe's new max-likelihood fitter
        fitter = MaximumLikelihoodGaussianFitter(pulseheights, data, params, 
                                                 TOL=1e-4)
        if hold is not None:
            for hnum in hold: 
                fitter.hold(hnum)
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = fitter.theory_function(fitparams, pulseheights)
        self.last_chisq = fitter.chisq
        
#        if iflag not in (1,2,3,4): 
        if iflag not in (0, 2): 
            print "Oh no! iflag=%d" % iflag
        elif plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            de = numpy.sqrt(covariance[0, 0])
            plot_as_stepped_hist(axis, data, pulseheights, color=color, 
                                                label="FWHM: %.2f +- %.2f %s"%(fitparams[0], de, label))
            axis.plot(pulseheights, self.last_fit_result, color='black')
            axis.legend(loc='upper left')
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])
        return fitparams, covariance



class GaussianLine(object):
    """An abstract base class for modeling spectral lines as a 
    single Gaussian of known energy.
    
    Instantiate one of its subclasses, which will have to define
    self.energy.
    """
    
    def __init__(self, energy=10.0, fwhm=1.0):
        """ """
        ## Line center energy
        self.energy = energy
        ## Full width at half-maximum
        self.fwhm = fwhm
        ## Gaussian parameter sigma, computed from self.fwhm
        self.sigma = self.fwhm/numpy.sqrt(8*numpy.log(2))
        ## Gaussian amplitude, chosen to normalize the distribution.
        self.amplitude = (2*numpy.pi)**(-0.5)/self.sigma
    
    def __call__(self, x, fwhm=None):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x, fwhm=fwhm)
    
    def pdf(self, x, fwhm=None):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = numpy.asarray(x, dtype=numpy.float)
        if fwhm is None:
            sigma, _ampl = self.sigma, self.amplitude
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
    """The 97 keV line of 153Gd."""
    def __init__(self):
        super(self.__class__, self).__init__(energy=97431.0, fwhm=50.0)
    

class Gd103(GaussianLine):
    """The 103 keV line of 153Gd."""
    def __init__(self):
        super(self.__class__, self).__init__(energy=103180.0, fwhm=50.0)

