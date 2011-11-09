## @file fluorescence_lines.py
#
# @brief Tools for fitting and simulating X-ray fluorescence lines.
# 
# Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
# Phys Rev A56 (#6) pages 4554ff (1997 December).  See online at
# http://pra.aps.org/pdf/PRA/v56/i6/p4554_1

__all__ = ['MnKAlpha', 'MnKBeta', 'CuKAlpha',
           'GaussianLine', 'Gd97', 'Gd103', 'AlKalpha', 'SiKalpha',
           'MultiLorentzianDistribution', 'MnKAlphaDistribution',
           'CuKAlphaDistribution', 'MnKAlphaFitter', 'MnKBetaFitter',
           'CuKAlphaFitter', 'GaussianFitter', 'plot_spectrum']
 
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

from mass.calibration import energy_calibration
from mass.mathstat.utilities import MaximumLikelihoodHistogramFitter, plot_as_stepped_hist

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
    def __init__(self):
        """Dummy constructor"""
        pass
    
    def __call__(self, x):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x)
    
    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = numpy.asarray(x, dtype=numpy.float)
        result = numpy.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.fwhm, self.amplitudes):
            result += ampl*lorentzian(x-energy, fwhm)
        return result
    
    def cdf(self, x):
        """Cumulative distribution function where <x> = set of energies."""
        x = numpy.asarray(x, dtype=numpy.float)
        result = numpy.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.fwhm, self.amplitudes):
            result += ampl*lorentzian_cdf(x-energy, fwhm)
        return result



class MnKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    
    ## Spectral complex name.
    name = 'Manganese K-alpha'    
    
    # The approximation is as a series of 8 Lorentzians (6 for KA1,2 for KA2)
    
    ## The Lorentzian energies
    energies = 5800+numpy.array((98.853, 97.867, 94.829, 96.532, 
                                 99.417, 102.712, 87.743, 86.495))
    ## The Lorentzian widths
    fwhm = numpy.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216))
    ## The Lorentzian peak height
    peak_heights = numpy.array((790, 264, 68, 96, 71, 10, 372, 100), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak
    peak_energy = 5898.802 # eV        


    
class MnKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Manganese K-beta'    
    
    # The approximation is as a series of 4 Lorentzians 
    ## The Lorentzian energies
    energies = 6400+numpy.array((90.89, 86.31, 77.73, 90.06, 88.83))
    ## The Lorentzian widths
    fwhm = numpy.array((1.83, 9.40, 13.22, 1.81, 2.81))
    ## The Lorentzian peak height
    peak_heights = numpy.array((608, 109, 77, 397, 176), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak
    peak_energy = 6490.18 # eV        


    
class CuKAlpha(SpectralLine):
    """Function object to approximate the copper K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Copper K-alpha'
            
    # The approximation is 4 of Lorentzians (2 for Ka1, 2 for Ka2)

    ## The Lorentzian energies
    energies = numpy.array((8047.8372, 8045.3672, 8027.9935, 8026.5041))
    ## The Lorentzian widths
    fwhm = numpy.array((2.285, 3.358, 2.667, 3.571))
    ## The Lorentzian peak height
    peak_heights = numpy.array((957, 90, 334, 111), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak
    peak_energy = 8047.83 # eV        
    


class GaussianLine(object):
    """An abstract base class for modeling spectral lines as a 
    single Gaussian of known energy.
    
    Instantiate one of its subclasses, which will have to define
    self.energy.
    """
    
    def __init__(self, energy=10.0, fwhm=1.0):
        """"""
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
        super(self.__class__, self).__init__(self, energy=97431.0, fwhm=50.0)
    

class Gd103(GaussianLine):
    """The 103 keV line of 153Gd."""
    def __init__(self):
        super(self.__class__, self).__init__(self, energy=103180.0, fwhm=50.0)


class AlKalpha(GaussianLine):
    """The K-alpha fluorescence lines of aluminum.
    WARNING: Not correct shape!"""
    def __init__(self):
        energy = energy_calibration.STANDARD_FEATURES['Al Ka']
        super(self.__class__, self).__init__(self, energy=energy, fwhm=3.0)

class SiKalpha(GaussianLine):
    """The K-alpha fluorescence lines of silicon.
    WARNING: Not correct shape!"""
    def __init__(self):
        energy = energy_calibration.STANDARD_FEATURES['Si Ka']
        super(self.__class__, self).__init__(self, energy=energy, fwhm=3.0)

    
class MultiLorentzianDistribution(scipy.stats.rv_continuous):
    """For producing random variates of the an energy distribution having the form
    of several Lorentzians summed together."""
    
    ## Approximates the random variate defined by multiple Lorentzian components.
    #  @param epoints The points at which the Lorentzian will be sampled for linear interpolation.
    #  (should be dense where distribution is changing rapidly)
    #  @param args  Pass all other parameters to parent class.
    #  @param kwargs  Pass all other parameters to parent class. 
    def __init__(self, epoints, distribution, *args, **kwargs):
        """<epoints> is a vector of energy points, densely collected at places
        where the distribution changes rapidly.
        <args> and <kwargs> are passed on to scipy.stats.rv_continuous"""

        ## The probability distribution function
        self.distribution = distribution
        
        scipy.stats.rv_continuous.__init__(self, *args, **kwargs)

        epoints = epoints[numpy.logical_and(epoints>=self.a, epoints<=self.b)]
        epoints = numpy.hstack((self.a, epoints, self.b))
        cdf = self.distribution.cdf(epoints)
        
        ## The minimum and maximum values that the CDF would have returned if we weren't careful!
        self.minCDF, self.maxCDF = cdf[0], cdf[-1] # would be 0,1 if we covered all of x-axis
        cdf = (cdf-cdf[0])/(cdf[-1]-cdf[0]) # Rescale so that it *does* run from [0,1]

        # Redefine the percentile point function (maps [0,1] to energies),
        # the prob distrib function and the cumulative distrib function
        ## Reimplements percentile point function.
        self._ppf = scipy.interpolate.interp1d(cdf, epoints, kind='linear')
        ## Reimplements probability distribution function.
        self._pdf = self.distribution
        ## Reimplements cumulative distribution function.
        self._cdf = lambda x: (self.distribution.cdf(x)-self.minCDF)/(self.maxCDF-self.minCDF)
        


class MnKAlphaDistribution(MultiLorentzianDistribution):
    """For producing random variates of the manganese K Alpha energy distribution"""
    
    def __init__(self, *args, **kwargs):
        """"""
        epoints = numpy.hstack(((0, 3000, 5000, 5500),
                                numpy.arange(5800, 5880.5),
                                numpy.arange(5880, 5920.-.025, .05),
                                numpy.arange(5920, 6001),
                                (6100, 6300, 6600, 7000, 8000, 12000)))*1.0
        MultiLorentzianDistribution.__init__(self, epoints, distribution = MnKAlpha(), *args, **kwargs)
        


class CuKAlphaDistribution(MultiLorentzianDistribution):
    """For producing random variates of the copper K Alpha energy distribution"""
    
    def __init__(self, *args, **kwargs):
        """"""
        epoints = numpy.hstack(((0, 3000, 6000, 7000, 7500),
                                numpy.arange(7800, 8010.5),
                                numpy.arange(8011, 8060.-.025, .05),
                                numpy.arange(8060, 8201),
                                (8300, 8500, 8800, 9300, 12000)))*1.0
        MultiLorentzianDistribution.__init__(self, epoints, distribution = CuKAlpha(), *args, **kwargs)
        


class MultiLorentzianComplexFitter(object):
    """Abstract base class for objects that can fit a spectral line complex.
    
    Provides methods fitfunc() and fit().  The child classes must provide:
    * a self.spect function object returning the spectrum at a given energy, and
    * a self.guess_starting_params method to return fit parameter guesses given a histogram.
    """
    def __init__(self):
        """"""
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None
        
    
    ## Compute the smeared line complex.
    #
    # @param params  The 6 parameters of the fit (see self.fit for details).
    # @param x       An array of pulse heights (params will scale them to energy).
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.
        
        <params>  The 6 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        E_peak = self.spect.peak_energy
        
        energy = (x-params[1])/abs(params[2]) + E_peak
        spectrum = self.spect(energy)
        smeared = smear(spectrum, abs(params[0]), stepsize = energy[1]-energy[0])
        nbins = len(x)
        return smeared * abs(params[3]) + abs(params[4]) + params[5]*numpy.arange(nbins)
    

    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_bg=True, vary_bg_slope=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                energy scale factor (counts/eV), amplitude, background level (per bin),
                and background slope (in counts per bin per bin) ]
                If params is None or does not have 6 elements, then they will be guessed.
        
        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the 
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)
        
        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        hold:          A sequence of parameter numbers (0 to 5, inclusive) to hold.  BG and BG slope will
                       be held if 4 or 5 appears in the hold sequence OR if the relevant boolean
                       vary_* tests False.
        """
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        try:
            _, _, _, _, _, _ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
#            print 'Guessed parameters: ',params
#            print 'PH range: ',pulseheights[0],pulseheights[-1]
        
        if plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            plot_as_stepped_hist(axis, pulseheights, data, color=color)
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01))
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params, 
                                                                 self.fitfunc, TOL=1e-4, epsilon=epsilon)
        
        if hold is not None:
            for h in hold:
                fitter.hold(h)
        if not vary_bg: fitter.hold(4)
        if not vary_bg_slope: fitter.hold(5)
            
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        
#        if iflag not in (1,2,3,4): 
        if iflag not in (0, 2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            de = numpy.sqrt(covariance[0, 0])
            axis.plot(pulseheights, self.last_fit_result, color='#666666', 
                      label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.legend(loc='upper left')
        return fitparams, covariance



class MnKAlphaFitter(MultiLorentzianComplexFitter):
    """Fits a Mn K alpha spectrum for energy shift and scale, amplitude, and resolution"""
    
    def __init__(self):
        """"""
        ## Spectrum function object
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
        baseline_slope = 0.0
        return [res, ph_ka1, dph/dE, ampl, baseline, baseline_slope]



class MnKBetaFitter(MultiLorentzianComplexFitter):
    """Fits a Mn K beta spectrum for energy shift and scale, amplitude, and resolution"""
    
    def __init__(self):
        """"""
        ## Spectrum function object
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
        baseline_slope = 0.0
        return [res, ph_peak, 1.0, ampl, baseline, baseline_slope]



class CuKAlphaFitter(MultiLorentzianComplexFitter):
    """Fits a Cu K alpha spectrum for energy shift and scale, amplitude, and resolution"""
    
    def __init__(self):
        """"""
        ## Spectrum function object
        self.spect = CuKAlpha()
        super(self.__class__, self).__init__()
        
    def guess_starting_params(self, data, binctrs):
        """If the cuts are tight enough, then we can estimate the locations of the
        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
        
        
        ph_ka1 = binctrs[data.argmax()]
        
        res = 5
        baseline = data[0:10].mean()
        baseline_slope = (data[-10:].mean()-baseline)/len(data)
        ampl = data.max()-data.mean()
        return [res, ph_ka1, 0.6, ampl, baseline, baseline_slope]
    


def smear(f, fwhm, stepsize=1.0):
    """Convolve a sampled function <f> with a Gaussian of the given
    <fwhm>, where the samples <f> are spaced evenly by <stepsize>.
    Function is padded at each end by enough samples to cover 5 FWHMs.
    The padding equals the values of <f> at its two endpoints. """
    nsamp = len(f)
    assert nsamp%2 == 0
    
    fwhm = numpy.abs(fwhm)
    padwidth = 5.0*fwhm
    if padwidth > 100.0:
        padwidth = 100.0
    npad = int(padwidth/stepsize+0.5)
    if npad > nsamp*7: npad = nsamp*7 
    
    # Make sure that total FFT size is a power of 2
    ntotal = nsamp+2*npad
    for j in range(2, 25):
        if ntotal <= 2**j:
            npad = (2**j-nsamp)/2
            break
    
    fpadded = numpy.hstack((f, numpy.zeros(2*npad)))
    fpadded[nsamp:nsamp+npad] = f[-1]
    fpadded[nsamp+npad:] = f[0]
    nfull = len(fpadded)
    
    ft = numpy.fft.rfft(fpadded)
    freq = numpy.fft.fftfreq(nfull, d=stepsize)[:nfull/2+1]
    freq[-1] = numpy.abs(freq[-1]) # convention is that the f_crit is negative.  Fix it.
    
    # Filter the function in Fourier space
    sigma = fwhm / numpy.sqrt(8*numpy.log(2))
    sigmaConjugate = 1.0/(2 * numpy.pi * sigma)
    ft *= numpy.exp(-0.5*(freq/sigmaConjugate)**2)
    return numpy.fft.irfft(ft)[0:nsamp]


class GaussianFitter(object):
    """Abstract base class for objects that can fit a single Gaussian line.
    
    Provides methods fitfunc() and fit().  The child classes must provide:
    * a self.spect function object returning the spectrum at a given energy, and
    * a self.guess_starting_params method to return fit parameter guesses given a histogram.
    """

    def __init__(self, spect):
        """"""
        ## Spectrum function object
        self.spect = spect 
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None
        
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
        
        def fitfunc(params, x):
            """Fitting function.  <x> is pulse height (arb units).  <params> are model parameters."""
            e_peak = self.spect.energy
            
            energy = (x-params[1]) + e_peak
            spectrum = self.spect(energy, fwhm=params[0]*params[1]/e_peak)
            return spectrum * abs(params[2]) + abs(params[3])
        
        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, params[2]/1e5, params[3]/1e2))
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params, 
                                                                 fitfunc, TOL=1e-4, epsilon=epsilon)
        if hold is not None:
            for hnum in hold: 
                fitter.hold(hnum)
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = fitfunc(fitparams, pulseheights)
        
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
            plot_as_stepped_hist(axis, pulseheights, data, color=color, 
                                                label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.plot(pulseheights, self.last_fit_result, color='black')
            axis.legend(loc='upper left')
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])
        return fitparams, covariance



def plot_spectrum(spectrumf=MnKAlpha(), 
                  resolutions=(2, 3, 4, 5, 6, 7, 8, 10, 12), 
                  energy_range=(5870, 5920), stepsize=0.05):
    """Plot a spectrum at several different resolutions.
    
    <spectrum>    A callable that accepts a vector of energies and returns
                  the matching probability distribution function.
    <resolutions> A sequence of energy resolution (FWHM) to be stepped through.
    <energy_range> The (min,max) energy to be plotted.
    <stepsize>    The plotting step size in energy units.  
    """
    if resolutions is None:
        resolutions = (2, 3, 4, 5, 6, 7, 8, 10, 12)
    e = numpy.arange(energy_range[0]-2.5*resolutions[-1],
                     energy_range[1]+2.5*resolutions[-1], stepsize)
    spectrum = spectrumf(e)
    spectrum /= spectrum.max()

    pylab.clf()
    axis = pylab.subplot(111)
    pylab.plot(e, spectrum, color='black', lw=2, label=' 0 eV')
    axis.set_color_cycle(('red', 'orange', '#bbbb00', 'green', 'cyan',
                          'blue', 'indigo', 'purple', 'brown'))
    for res in resolutions:
        smeared_spectrum = smear(spectrum, res, stepsize = stepsize)
        smeared_spectrum /= smeared_spectrum.max()
        smeared_spectrum *= (1+res*.01)
        pylab.plot(e, smeared_spectrum, label="%2d eV"%res, lw=2)
        
        # Find the peak, valley, peak
        if spectrumf.name == 'Manganese K-alpha':
            epk2, evalley, epk1 = 5887.70, 5892.0, 5898.801
        elif spectrumf.name == 'Copper K-alpha':
            epk2, evalley, epk1 = 8027.89, 8036.6, 8047.83
            
        p1 = smeared_spectrum[numpy.abs(e-epk1)<2].max()
        if res < 8.12:
            pk2 = smeared_spectrum[numpy.abs(e-epk2)<2].max()
            pval = smeared_spectrum[numpy.abs(e-evalley)<3].min()
            print "Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f" % (res, pk2/p1, pval/pk2) 
        
    pylab.xlim(energy_range)
    pylab.ylim([0, 1.13])
    pylab.legend(loc='upper left')
    
    pylab.title("%s lines at various resolutions (FWHM of Gaussian)" % spectrumf.name)
    pylab.xlabel("Energy (eV)")
    pylab.ylabel("Intensity (arb.)")
