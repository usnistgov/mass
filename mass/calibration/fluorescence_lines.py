"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.

Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
Phys Rev A56 (#6) pages 4554ff (1997 December).  See online at
http://pra.aps.org/pdf/PRA/v56/i6/p4554_1

Joe Fowler, NIST

July 12, 2012  : added fitting of Voigt and Lorentzians
March 9, 2011
November 24, 2010 : started as mn_kalpha.py
"""

__all__ = ['MnKAlpha', 'MnKBeta', 'CuKAlpha', 
           'VoigtFitter', 'LorentzianFitter',
           'MultiLorentzianDistribution_gen', 'MultiLorentzianComplexFitter', 'MnKAlphaDistribution',
           'CuKAlphaDistribution',
           'ScKAlphaFitter', 'TiKAlphaFitter', 'VKAlphaFitter', 'CrKAlphaFitter',
           'MnKAlphaFitter', 'FeKAlphaFitter', 'CoKAlphaFitter', 'NiKAlphaFitter', 'CuKAlphaFitter',
           'TiKBetaFitter', 'CrKBetaFitter', 'MnKBetaFitter', 'FeKBetaFitter', 'CoKBetaFitter', 'NiKBetaFitter', 'CuKBetaFitter',
           'plot_spectrum']
 
import numpy
import pylab
import scipy.stats

from mass.mathstat import MaximumLikelihoodHistogramFitter, \
    plot_as_stepped_hist, voigt #@UnresolvedImport


class SpectralLine(object):
    """An abstract base class for modeling spectral lines as a sum
    of Voigt profiles (i.e., Gaussian-convolved Lorentzians).
    
    Instantiate one of its subclasses, which will have to define
    self.energies, self.fwhm, self.amplitudes.  Each must be a sequence
    of the same length.
    """
    def __init__(self):
        """Set up a default Gaussian smearing of 0"""
        self.gauss_sigma = 0.0

    def set_gauss_fwhm(self, fwhm):
        """Update the Gaussian smearing to have <fwhm> as the full-width at half-maximum"""
        self.gauss_sigma = fwhm/(8*numpy.log(2))**0.5
    
    def __call__(self, x):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x)
    
    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = numpy.asarray(x, dtype=numpy.float)
        result = numpy.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.fwhm, self.amplitudes):
            result += ampl*voigt(x, energy, hwhm=fwhm*0.5, sigma=self.gauss_sigma)
        return result

class ScKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006). Characterization of K spectral profiles for vanadium, component redetermination for scandium, titanium, chromium, and manganese, and development of satellite structure for Z=21 to Z=25. Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Scandium K-alpha'    
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    ## The Lorentzian energies (Table I C_i)
    energies = numpy.array((4090.595, 4089.308, 4087.666, 4093.428, 4085.773, 4083.697))
    ## The Lorentzian widths (Table I W_i)
    fwhm = numpy.array((1.13, 2.46, 1.58, 2.04, 1.94, 3.42))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = numpy.array((8203, 818, 257, 381, 4299, 105), dtype=numpy.float)
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    peak_energy = 4090.735 # eV  


### Ti and V and KAlphas are commented out de
class TiKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006). Characterization of K spectral profiles for vanadium, component redetermination for scandium, titanium, chromium, and manganese, and development of satellite structure for Z=21 to Z=25. Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    Note that to reproduce the plots in the reference paper, you must include the Gaussian 
    broadening of their instrument, which was 0.082eV FWHM
    The underlying line profile has zero fundamental broadening.
    """
    ## Spectral complex name.
    name = 'Titanium K-alpha'    
    # the paper has two sets of Ti data, I used the set Refit of [21] Kawai et al 1994
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    ## The Lorentzian energies (Table I C_i)
    energies = numpy.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088))
    ## The Lorentzian widths (Table I W_i)
    fwhm = numpy.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = numpy.array((4549, 626, 236, 143, 2034, 54), dtype=numpy.float)
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    peak_energy = 4510.903 # eV 
    
class TiKBeta(SpectralLine):
    """the data in this are made up based on the CrKBeta line and the tabulated TiKBeta1 energy 
    I just scaled the energies from Cr by the ratio of the peak energies
    """
    name = 'Titanium K-beta'    
    

    energies = numpy.array([ 4931.9592774 ,  4922.26453989,  4931.32899506,  4927.84585584,
        4930.24258735])
    fwhm = numpy.array((1.70, 15.98, 1.90, 6.69, 3.37))
    peak_heights = numpy.array((670, 55, 337, 82, 151), dtype=numpy.float)/1e3
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    peak_energy = 4931.81 # eV     

class VKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006). Characterization of K spectral profiles for vanadium, component redetermination for scandium, titanium, chromium, and manganese, and development of satellite structure for Z=21 to Z=25. Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    Note that to reproduce the plots in the reference paper, you must include the Gaussian 
    broadening of their instrument, which was 1.99eV FWHM
    The underlying line profile has zero fundamental broadening.
    """
    ## Spectral complex name.
    name = 'Vanadium K-alpha'    
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    ## The Lorentzian energies (Table I C_i)
    energies = numpy.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014))
    ## The Lorentzian widths (Table I W_i)
    fwhm = numpy.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = numpy.array((25832, 5410, 1536, 956, 12971, 603), dtype=numpy.float)
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    peak_energy = 4952.216 # eV   
    
class CrKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    
    ## Spectral complex name.
    name = 'Chromium K-alpha'    
    
    # The approximation is as a series of 7 Lorentzians (5 for KA1,2 for KA2)
    
    ## The Lorentzian energies (Table II E_i)
    energies = 5400+numpy.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986))
    ## The Lorentzian widths (Table II W_i)
    fwhm = numpy.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.4740))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = numpy.array((882, 237, 85, 45, 15, 386, 36), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV alpha_1)
    peak_energy = 5414.81 # eV   
    
class CrKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Chromium K-beta'    
    
    # The approximation is as a series of 5 Lorentzians 
    ## The Lorentzian energies (Table III E_i)
    energies = 5900+numpy.array((47.00, 35.31, 46.24, 42.04, 44.93))
    ## The Lorentzian widths (Table III W_i)
    fwhm = numpy.array((1.70, 15.98, 1.90, 6.69, 3.37))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = numpy.array((670, 55, 337, 82, 151), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    peak_energy = 5946.82 # eV     

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
    ## the 102.712 line doesn't appear in the reference paper, apparently it was added in Scott Porter's refit of the complex, also one of the intensities went from 0.005 to 0.018
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
    
class FeKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Iron K-alpha'    
    # The approximation is as a series of 7 Lorentzians (4 for KA1,3 for KA2)
    ## The Lorentzian energies (Table II E_i)
    energies = numpy.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275))
    ## The Lorentzian widths (Table II W_i)
    fwhm = numpy.array((1.613, 1.965, 4.833, 2.803, 2.487, 2.339, 4.433))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = numpy.array((697, 376, 88, 136, 339, 60, 102), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV alpha_1)
    peak_energy = 6404.01 # eV   
    
class FeKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Iron K-beta'    
    # The approximation is as a series of 4 Lorentzians 
    ## The Lorentzian energies (Table III E_i)
    energies = numpy.array((7046.90, 7057.21, 7058.36, 7054.75))
    ## The Lorentzian widths (Table III W_i)
    fwhm = numpy.array((14.17, 3.12, 1.97, 6.38))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = numpy.array((107, 448, 615, 141), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    peak_energy = 7058.18 # eV      
    
class CoKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Cobalt K-alpha'    
    # The approximation is as a series of 7 Lorentzians (4 for KA1,3 for KA2)
    ## The Lorentzian energies (Table II E_i)
    energies = numpy.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078))
    ## The Lorentzian widths (Table II W_i)
    # the calculated amplitude for the 4th entry 0.808 differs from the paper, but the other numbers appear to
    # be correct, so I think they may have a typo
    fwhm = numpy.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = numpy.array((809, 205, 107, 41, 314, 131, 43), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV alpha_1)
    peak_energy = 6930.38 # eV   
    
class CoKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Cobalt K-beta'    
    # The approximation is as a series of 6 Lorentzians 
    ## The Lorentzian energies (Table III E_i)
    energies = numpy.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13))
    ## The Lorentzian widths (Table III W_i)
    fwhm = numpy.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = numpy.array((798, 286, 85, 114, 33, 35), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    peak_energy = 7649.45 # eV  
    
class NiKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Nickle K-alpha'    
    # The approximation is as a series of 5 Lorentzians (2 for KA1,3 for KA2)
    ## The Lorentzian energies (Table II E_i)
    energies = numpy.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029))
    ## The Lorentzian widths (Table II W_i)
    fwhm = numpy.array((2.013, 4.711, 2.674, 3.039, 4.476))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = numpy.array((909, 136, 351, 79, 24), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV alpha_1)
    peak_energy = 7478.26 # eV   
    
class NiKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Nickle K-beta'    
    # The approximation is as a series of 4 Lorentzians 
    ## The Lorentzian energies (Table III E_i)
    energies = numpy.array((8265.01, 8263.01, 8256.67, 8268.70))
    ## The Lorentzian widths (Table III W_i)
    fwhm = numpy.array((3.76, 4.34, 13.70, 5.18))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = numpy.array((722, 358, 89, 104), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    peak_energy = 8264.78 # eV  
    
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
    
class CuKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Copper K-beta'    
    
    # The approximation is as a series of 5 Lorentzians 
    ## The Lorentzian energies (Table III E_i)
    energies = numpy.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393))
    ## The Lorentzian widths (Table III W_i)
    fwhm = numpy.array((3.52, 3.52, 3.55, 8.08, 5.31))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = numpy.array((757, 388, 171, 68, 55), dtype=numpy.float)/1e3
    ## Amplitude of the Lorentzians
    amplitudes = (0.5*numpy.pi*fwhm) * peak_heights
    amplitudes /= amplitudes.sum()
    ## The energy at the main peak (from table IV beta1,3)
    peak_energy = 8905.42 # eV      


    

class MultiLorentzianDistribution_gen(scipy.stats.rv_continuous):
    """For producing random variates of the an energy distribution having the form
    of several Lorentzians summed together."""
    
    ## Approximates the random variate defined by multiple Lorentzian components.
    #  @param args  Pass all other parameters to parent class.
    #  @param kwargs  Pass all other parameters to parent class. 
    def __init__(self, distribution, *args, **kwargs):
        """<args> and <kwargs> are passed on to scipy.stats.rv_continuous"""

        scipy.stats.rv_continuous.__init__(self, *args, **kwargs)
        self.distribution = distribution
        self.cumulative_amplitudes = self.distribution.amplitudes.cumsum()
        self.name = distribution.name
        self.set_gauss_fwhm = self.distribution.set_gauss_fwhm

        ## Reimplements probability distribution function.
        self._pdf = self.distribution.pdf

    def _rvs(self, *args):
        """The CDF and PPF (cumulative distribution and percentile point functions) are hard to
        compute.  But it's easy enough to generate the random variates themselves, so we 
        override that method.  Don't call this directly!  Instead call .rvs(), which wraps this."""
        # Choose from among the N Lorentzian lines in proportion to the line amplitudes
        iline = self.cumulative_amplitudes.searchsorted(
                            numpy.random.uniform(0, self.cumulative_amplitudes[-1], size=self._size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = numpy.random.standard_cauchy(size=self._size)*self.distribution.fwhm[iline]*0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if self.distribution.gauss_sigma > 0.0:
            lor += numpy.random.standard_normal(size=self._size)*self.distribution.gauss_sigma
        # Finally, add the line centers.
        return lor + self.distribution.energies[iline]

# Some specific fluorescence lines
MnKAlphaDistribution = MultiLorentzianDistribution_gen(distribution=MnKAlpha(), name="Mn Kalpha fluorescence")
MnKBetaDistribution  = MultiLorentzianDistribution_gen(distribution=MnKBeta(), name="Mn Kbeta fluorescence")
CuKAlphaDistribution = MultiLorentzianDistribution_gen(distribution=CuKAlpha(), name="Cu Kalpha fluorescence")



class VoigtFitter(object):
    """Fit a single Lorentzian line, with Gaussian smearing."""
    def __init__(self):
        """ """
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None
    
    
    def guess_starting_params(self, data, binctrs):
        order_stat = numpy.array(data.cumsum(), dtype=numpy.float)/data.sum()
        percentiles = lambda p: binctrs[(order_stat>p).argmax()]
        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75)-percentiles(0.25))
        res = iqr*0.7
        lor_hwhm = res*0.5
        baseline = data[0:10].mean()
        baseline_slope = (data[-10:].mean()-baseline)/len(data)
        ampl = (data.max()-baseline)*numpy.pi
        return [res, peak_loc, lor_hwhm, ampl, baseline, baseline_slope]
        
    
    ## Compute the smeared line value.
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
        sigma = params[0]/(8*numpy.log(2))**0.5
        spectrum = voigt(x, params[1], params[2], sigma)
        nbins = len(x)
        return spectrum * abs(params[3]) + abs(params[4]) + params[5]*numpy.arange(nbins)
    

    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_resolution=True, vary_bg=True, vary_bg_slope=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 6-element sequence of [Gaussian resolution (fwhm), Pulseheight of the line peak,
                Lorenztian HALF-width at half-max, amplitude, background level (per bin),
                and background slope (in counts per bin per bin) ]
                If params is None or does not have 6 elements, then they will be guessed.
        
        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the 
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)
        
        vary_resolution Whether to let the Gaussian resolution vary in the fit
        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        hold:          A sequence of parameter numbers (0 to 5, inclusive) to hold.  Resolution, BG
                       or BG slope will be held if 0, 4 or 5 appears in the hold sequence OR 
                       if the relevant boolean vary_* tests False.
        
        The interaction between <hold> (or its vary_* aliases) and <params> is simple if <params> is given
        as a 6-element sequence.  Otherwise, for i in [0,4,5], params[i] will be forced to 0 if the given
        parameter i is in the <hold> list.  So you can fix the resolution at 0 by vary_resolution=False.
        If you want to fix it at 2.5, then you have to give params=[2.5, u,v,w,x,y].
        
        """
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        if hold is None:
            hold = []
        else:
            hold = list(hold)
        if not vary_resolution:
            hold.append(0)
        if not vary_bg:
            hold.append(4)
        if not vary_bg_slope:
            hold.append(5)
        print 'Params is: ', params
        try:
            _, _, _, _, _, _ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
            if 0 in hold:
                params[0] = 0
                params[2] *= 1.4
            if 4 in hold:
                params[4] = 0
            if 5 in hold:
                params[5] = 0
        
        if plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01))
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params, 
                                                                 self.fitfunc, TOL=1e-4, epsilon=epsilon)
        
        for h in hold:
            fitter.hold(h)
            
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        
        if iflag not in (0, 2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            de = numpy.sqrt(covariance[2, 2])
            label = "Lorentz HWHM: %.2f +- %.2f eV %s"%(fitparams[2], de, label)
            if 0 not in hold:
                de = numpy.sqrt(covariance[0, 0])
                label += "\nGauss FWHM: %.2f +- %.2f eV"%(fitparams[0], de)
            axis.plot(pulseheights, self.last_fit_result, color='#666666', 
                      label=label)
            axis.legend(loc='upper left')
        return fitparams, covariance



class TwoVoigtFitter(object):
    """Fit a single Lorentzian line, with Gaussian smearing.
    
    So far, I don't know how to guess the starting parameters, so you have to supply all 8.
    (See method fit() for explanation).
    """
    def __init__(self):
        """ """
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None
    
    
    def guess_starting_params(self, data, binctrs):
#        order_stat = numpy.array(data.cumsum(), dtype=numpy.float)/data.sum()
#        percentiles = lambda p: binctrs[(order_stat>p).argmax()]
#        peak_loc = percentiles(0.5)
#        iqr = (percentiles(0.75)-percentiles(0.25))
#        res = iqr*0.7
#        lor_hwhm = res*0.5
#        baseline = data[0:10].mean()
#        baseline_slope = (data[-10:].mean()-baseline)/len(data)
#        ampl = (data.max()-baseline)*numpy.pi
#        return [res, peak_loc, lor_hwhm, ampl, baseline, baseline_slope]
        raise NotImplementedError("I don't know how to guess starting parameters for a 2-peak Voigt.")
        
    
    ## Compute the smeared line value.
    #
    # @param params  The 6 parameters of the fit (see self.fit for details).
    # @param x       An array of pulse heights (params will scale them to energy).
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.
        
        <params>  The 8 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        sigma = params[0]/(8*numpy.log(2))**0.5
        spectrum = voigt(x, params[1], params[2], sigma) * abs(params[3]) +\
             voigt(x, params[4], params[5], sigma) * abs(params[6])
        return spectrum  + abs(params[7])
    

    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_resolution=True, vary_bg=True, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 8-element sequence of [Gaussian resolution (fwhm), Pulseheight of the line 1 peak,
                Lorenztian HALF-width at half-max of line 1, amplitude of line 1, PH of line 2 peak,
                Lorentzian HALF-width at half-max of line 2, amplitude of line 2, and
                constant background level (per bin) ]
                If params is None or does not have 8 elements, then they will be guessed.
        
        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the 
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)
        
        vary_resolution Whether to let the Gaussian resolution vary in the fit
        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        hold:          A sequence of parameter numbers (0 to 5, inclusive) to hold.  Resolution, BG
                       or BG slope will be held if 0, 4 or 5 appears in the hold sequence OR 
                       if the relevant boolean vary_* tests False.
        
        """
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        if hold is None:
            hold = []
        else:
            hold = list(hold)
        if not vary_resolution:
            hold.append(0)
        if not vary_bg:
            hold.append(7)
        print 'Params is: ', params
        try:
            _, _, _, _, _, _, _, _ = params
        except:
            params = self.guess_starting_params(data, pulseheights)
            if 0 in hold:
                params[0] = 0
                params[2] *= 1.4
            if 4 in hold:
                params[4] = 0
            if 5 in hold:
                params[5] = 0
        
        if plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                
            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

        # Joe's new max-likelihood fitter
        epsilon = numpy.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, 
                               params[4]/1e5, 1e-3, params[6]/1e5, params[7]/1e2))
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params, 
                                                                 self.fitfunc, TOL=1e-4, epsilon=epsilon)
        
        for h in hold:
            fitter.hold(h)
            
        fitparams, covariance = fitter.fit()
        iflag = 0

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        
        if iflag not in (0, 2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            de1 = numpy.sqrt(covariance[2, 2])
            label = "Lorentz HWHM 1: %.2f +- %.2f eV %s"%(fitparams[2], de1, label)
            de2 = numpy.sqrt(covariance[5, 5])
            label += "\nLorentz HWHM 2: %.2f +- %.2f eV"%(fitparams[5], de2)
            if 0 not in hold:
                de = numpy.sqrt(covariance[0, 0])
                label += "\nGauss FWHM: %.2f +- %.2f eV"%(fitparams[0], de)
            axis.plot(pulseheights, self.last_fit_result, color='#666666', 
                      label=label)
            axis.legend(loc='upper left')
        return fitparams, covariance




class LorentzianFitter(VoigtFitter):
    """Fit a single Lorentzian line, without Gaussian smearing.
    To allow Gaussian smearing, too, use VoigtFitter instead."""

    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_bg=True, vary_bg_slope=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 5-element sequence of [Pulseheight of the line peak,
                Lorenztian HALF-width at half-max, amplitude, background level (per bin),
                and background slope (in counts per bin per bin) ]
                If params is None or does not have 5 elements, then they will be guessed.
        
        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the 
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)
        
        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        hold:          A sequence of parameter numbers (0 to 4, inclusive) to hold.  BG
                       or BG slope will be held if 3 or 4 appears in the hold sequence OR 
                       if the relevant boolean vary_* tests False.
        
        The interaction between <hold> (or its vary_* aliases) and <params> is simple if <params> is given
        as a 6-element sequence.  Otherwise, for i in [0,4,5], params[i] will be forced to 0 if the given
        parameter i is in the <hold> list.  So you can fix the resolution at 0 by vary_resolution=False.
        If you want to fix it at 2.5, then you have to give params=[2.5, u,v,w,x,y].
        
        """
        if params is not None:
            params = [0] + list(params)
        if hold is not None:
            hold = [1+h for h in hold]
        p,c = VoigtFitter.fit(self, data, pulseheights=pulseheights, params=params, 
                              plot=plot, axis=axis, color=color, label=label, 
                              vary_bg=vary_bg, vary_bg_slope=vary_bg_slope, 
                              hold=hold, vary_resolution=False)
        # Remove the meaningless parameter 0 (and row/cols 0 of covariance) 
        return p[1:], c[1:,1:]


class SimultaneousMultiLorentzianComplexFitter(object):
    ''' doesn't seem to work very well yet '''
    def __init__(self, spectraDefs = (MnKAlpha(), CrKBeta())):
        self.spectraDefs = spectraDefs
        self.last_fit_params = None
        self.last_fit_result = None
        
    def fitfunc(self, params, ph):
        """Return the smeared line complex.
        <params>  The 6+ parameters of the fit (see self.fit for details).
        <ph>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """

        E_peak = self.spectraDefs[0].peak_energy
        energy = (ph-params[1])/abs(params[2]) + E_peak
        outSpectrum = numpy.zeros_like(energy)
        for i,spectrum  in enumerate(self.spectraDefs):
            # if it crashes here you probably didnt instantiate your spectra defs... ie use MnKAlpha() not MnKAlpha
            spectrum.set_gauss_fwhm(params[0])
            if i == 0: 
                ampIndex = 3
            else:
                ampIndex = 5+i
            outSpectrum += params[ampIndex]*spectrum.pdf(energy) # probability density function
        background = abs(params[4])+params[5]*numpy.arange(len(energy))
        return outSpectrum + background     
    
    def guess_starting_params(self, data, binctrs):
        """We're going to hope that the difference between the two farther features is roughly 2 standard deviations
        of pulse energies"""
        n = data.sum()
        if n<=0:
            raise ValueError("This histogram has no contents")
        sum_d = (data*binctrs).sum()
        sum_d2 = (data*binctrs*binctrs).sum()
        mean_d = sum_d/n
        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
#        print n, sum_d, sum_d2, mean_d, rms_d

        ph_ka1 = binctrs[numpy.argmax(data)]
        dph = 2*rms_d

        if len(self.spectraDefs) >= 2:
            linecenters = []
            for spectrum in self.spectraDefs:
                linecenters.append(spectrum.peak_energy)
            dE = numpy.max(linecenters)-numpy.min(linecenters)
        else:
            dE = numpy.max(self.spectraDefs[0].energies)-numpy.min(self.spectraDefs[0].energies) # eV difference between KAlpha peaks
        # this should be caluclated from data in the spectrumDef, but currently
        # the KAlpha object don't include the KAlpha2 energy.
        ampl = data.max()*9.4 
        res = 4.0
        if len(data) > 40:
            baseline = data[0:10].mean()
            baseline_slope = (data[-10:].mean()-baseline)/len(data)
        else:
            baseline = 0.1
            baseline_slope = 0.0
        param = [res, ph_ka1, dph/dE, ampl, baseline, baseline_slope]  
        for i in range(len(self.spectraDefs)-1): # add elements if neccesary for relative amplitudes of each spectrumDef
            param.append(data.max())
        return param
            
        
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_bg=True, vary_bg_slope=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 6+ element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                energy scale factor (pulseheight/eV), amplitude spectra 0, background level (per bin at left edge),
                and background slope (in counts per bin per bin), amplitude spectra 1 (if it exists), 
                amplitude spectra 2 (if it exists), amplitude spectra 3... ]
                params = [res, ph_ka1_spec0, dph, amp0, bg_level, bg_slope, amp1, amp2,...] 
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
        ====================================================================
        returns fitparams, covariance
        fitparams has same format as input variable params
        """
        try:
            assert len(pulseheights) == len(data)
        except:
            pulseheights = numpy.arange(len(data), dtype=numpy.float)
        if params == None:
            params = self.guess_starting_params(data, pulseheights)
        if params[4]==0: params[4]=1e-7 # the fitter crashes if params[4] bg_level is zero
        print 'start params'
        print params
        assert len(params) == 5+len(self.spectraDefs)
            
#            print 'Guessed parameters: ',params
#            print 'PH range: ',pulseheights[0],pulseheights[-1]
        ph_binsize = pulseheights[1]-pulseheights[0]


        # Joe's new max-likelihood fitter
        epsilon = numpy.ones_like(params)*params[1]/1e4
        epsilon[:6] = numpy.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01))
#        print 'epsilon', epsilon
        fitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params, 
                                                                 self.fitfunc, TOL=1e-4, epsilon=epsilon)
        
        if hold is not None:
            for h in hold:
                fitter.hold(h)
        if not vary_bg: fitter.hold(4)
        if not vary_bg_slope: fitter.hold(5)
#        print 'held'
#        print fitter.param_free
            
        fitparams, covariance = fitter.fit(verbose=False)

        fitparams[0] = abs(fitparams[0])
        
        self.last_fit_params = fitparams
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        
        ## all this plot stuff should go into a seperate function then we have 
        ## if plot: self.plotFit(self.last_fit_result, self.last_fit_params)
        if plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                pylab.xlabel('pulseheight (arb)')
                pylab.ylabel('counts per %.3f unit bin'%ph_binsize)
                pylab.title('resolution %.3f, Ka1_ph %.3f, dph/de %.3f\n amp %.3f, bg %.3f, bg_slope %.3f'%tuple(fitparams[:6]))        
                plot_as_stepped_hist(axis, data, pulseheights, color=color)
                axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

            de = numpy.sqrt(covariance[0, 0])
            axis.plot(pulseheights, self.last_fit_result, color='#666666', 
                      label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.legend(loc='upper left')
        return fitparams, covariance

        
# Galen 20130208: I think this could be completly replaced by SimultaneousMultiLorentzianCompleFitter
class MultiLorentzianComplexFitter(object):
    """Abstract base class for objects that can fit a spectral line complex.
    
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
        self.spect.set_gauss_fwhm(abs(params[0]))
        spectrum = self.spect(energy) # this is the same as self.spec.pdf(), and I found it very confusing
        nbins = len(x)
        return spectrum * abs(params[3]) + abs(params[4]) + params[5]*numpy.arange(nbins)
    

    
    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None, label="", 
            vary_bg=True, vary_bg_slope=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the 
        set of histogram bins <pulseheights>.
        
        pulseheights: the histogram bin centers.  If pulseheights is None, then the parameters 
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                energy scale factor (pulseheight/eV), amplitude, background level (per bin),
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
        ====================================================================
        returns fitparams, covariance
        fitparams has same format as input variable params
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
        ph_binsize = pulseheights[1]-pulseheights[0]


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
        
        ## all this plot stuff should go into a seperate function then we have 
        ## if plot: self.plotFit(self.last_fit_result, self.last_fit_params)
        if plot:
            if color is None: 
                color = 'blue'
            if axis is None:
                pylab.clf()
                axis = pylab.subplot(111)
                pylab.xlabel('pulseheight (arb) - %s'%self.spect.name)
                pylab.ylabel('counts per %.3f unit bin'%ph_binsize)
                pylab.title('resolution %.3f, amplitude %.3f, dph/de %.3f\n amp %.3f, bg %.3f, bg_slope %.3f'%tuple(fitparams))        
            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

#        if iflag not in (1,2,3,4): 
        if iflag not in (0, 2): 
            print "Oh no! iflag=%d"%iflag
        elif plot:
            de = numpy.sqrt(covariance[0, 0])
            axis.plot(pulseheights, self.last_fit_result, color='#666666', 
                      label="%.2f +- %.2f eV %s"%(fitparams[0], de, label))
            axis.legend(loc='upper left')
        return fitparams, covariance


class GenericKAlphaFitter(MultiLorentzianComplexFitter):
    """Fits a generic K alpha spectrum for energy shift and scale, amplitude, and resolution"""
    def __init__(self, spectrumDef = MnKAlpha):
        """ """
        ## Spectrum function object
        self.spect = spectrumDef
        MultiLorentzianComplexFitter.__init__(self)
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
        # this should be caluclated from data in the spectrumDef, but currently
        # the KAlpha object don't include the KAlpha2 energy.
        ampl = data.max() *9.4 
        res = 4.0
        if len(data) > 20:
            baseline = data[0:10].mean()
            baseline_slope = (data[-10:].mean()-baseline)/len(data)
        else:
            baseline = 0.1
            baseline_slope = 0.0
        return [res, ph_ka1, dph/dE, ampl, baseline, baseline_slope]
    
    
class GenericKBetaFitter(MultiLorentzianComplexFitter):
    def __init__(self, spectrumDef=MnKBeta):
        """ """
        ## Spectrum function object
        self.spect = spectrumDef
        MultiLorentzianComplexFitter.__init__(self) 
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
        if len(data) > 20:
            baseline = data[0:10].mean()
            baseline_slope = (data[-10:].mean()-baseline)/len(data)
        else:
            baseline = 0.1
            baseline_slope = 0.0
        return [res, ph_peak, 1.0, ampl, baseline, baseline_slope]
    
## create specific KAlpha Fitters
class ScKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, ScKAlpha())
class TiKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, TiKAlpha())
class VKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, VKAlpha())
class CrKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, CrKAlpha())
class MnKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, MnKAlpha())
class FeKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, FeKAlpha())
class CoKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, CoKAlpha())
class NiKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, NiKAlpha())
class CuKAlphaFitter(GenericKAlphaFitter):
    def __init__(self):
        GenericKAlphaFitter.__init__(self, CuKAlpha())
        
## create specific KBeta Fitters
class TiKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        print('warning using simple guess at TiKBeta lineshape, havent found good fit data')
        GenericKBetaFitter.__init__(self, TiKBeta())
class CrKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, CrKBeta())
class MnKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, MnKBeta())
class FeKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, FeKBeta())
class CoKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, CoKBeta())
class NiKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, NiKBeta())
class CuKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, CuKBeta())
# previous method of MnKAlphaFitter, redundant since GenericKAlphaFitter exists now
#class MnKAlphaFitter(MultiLorentzianComplexFitter):
#    """Fits a Mn K alpha spectrum for energy shift and scale, amplitude, and resolution"""
#    def __init__(self):
#        """ """
#        ## Spectrum function object
#        self.spect = MnKAlpha()
#        super(self.__class__, self).__init__()
#        # At first, I was pre-computing lots of stuff, but now I don't think it's needed.
#    def guess_starting_params(self, data, binctrs):
#        """If the cuts are tight enough, then we can estimate the locations of the
#        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
#        n = data.sum()
#        if n<=0:
#            raise ValueError("This histogram has no contents")
#        sum_d = (data*binctrs).sum()
#        sum_d2 = (data*binctrs*binctrs).sum()
#        mean_d = sum_d/n
#        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
##        print n, sum_d, sum_d2, mean_d, rms_d
#        ph_ka1 = mean_d + rms_d*.65
#        ph_ka2 = mean_d - rms_d
#        dph = ph_ka1-ph_ka2
#        dE = 11.1 # eV difference between KAlpha peaks
#        ampl = data.max() *9.4
#        res = 4.0
#        baseline = 0.1
#        baseline_slope = 0.0
#        return [res, ph_ka1, dph/dE, ampl, baseline, baseline_slope]


#replaced with GenericKBetaFitter
#class MnKBetaFitter(MultiLorentzianComplexFitter):
#    """Fits a Mn K beta spectrum for energy shift and scale, amplitude, and resolution"""
#    
#    def __init__(self):
#        """ """
#        ## Spectrum function object
#        self.spect = MnKBeta()
#        super(self.__class__, self).__init__()
#        
#    def guess_starting_params(self, data, binctrs):
#        """If the cuts are tight enough, then we can estimate the locations of the
#        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
#        
#        n = data.sum()
#        sum_d = (data*binctrs).sum()
##        sum_d2 = (data*binctrs*binctrs).sum()
#        mean_d = sum_d/n
##        rms_d = numpy.sqrt(sum_d2/n - mean_d**2)
##        print n, sum_d, sum_d2, mean_d, rms_d
#        
#        ph_peak = mean_d
#
#        ampl = data.max() *9.4
#        res = 4.0
#        baseline = 0.1
#        baseline_slope = 0.0
#        return [res, ph_peak, 1.0, ampl, baseline, baseline_slope]


#### replaced using GenericKAlphaFitter
#class CuKAlphaFitter(MultiLorentzianComplexFitter):
#    """Fits a Cu K alpha spectrum for energy shift and scale, amplitude, and resolution"""
#    
#    def __init__(self):
#        """ """
#        ## Spectrum function object
#        self.spect = CuKAlpha()
#        super(self.__class__, self).__init__()
#        
#    def guess_starting_params(self, data, binctrs):
#        """If the cuts are tight enough, then we can estimate the locations of the
#        K alpha-1 and -2 peaks as the (mean + 2/3 sigma) and (mean-sigma)."""
#        
#        
#        ph_ka1 = binctrs[data.argmax()]
#        
#        res = 5
#        baseline = data[0:10].mean()
#        baseline_slope = (data[-10:].mean()-baseline)/len(data)
#        ampl = data.max()-data.mean()
#        return [res, ph_ka1, 0.6, ampl, baseline, baseline_slope]
    


def plot_allMultiLorentzianLineComplexs():
    """ makes a bunch of plots showing the line shape and component parts for the KAlpha and KBeta complexes defined in here,
    intended to nearly replicate plots in papers giving spectral lineshapes"""
    plot_multiLorentzianLineComplex(ScKAlpha)
    plot_multiLorentzianLineComplex(TiKAlpha, instrumentGaussianSigma=0.68/2.354)
    plot_multiLorentzianLineComplex(VKAlpha, instrumentGaussianSigma=1.99/2.354) # must include instrument broadening from Table 1, Source to recreate plots
    plot_multiLorentzianLineComplex(CrKAlpha)
    plot_multiLorentzianLineComplex(MnKAlpha)
    plot_multiLorentzianLineComplex(FeKAlpha)
    plot_multiLorentzianLineComplex(CoKAlpha)
    plot_multiLorentzianLineComplex(NiKAlpha)
    plot_multiLorentzianLineComplex(CuKAlpha)
    plot_multiLorentzianLineComplex(CrKBeta)
    plot_multiLorentzianLineComplex(MnKBeta)
    plot_multiLorentzianLineComplex(FeKBeta)
    plot_multiLorentzianLineComplex(CoKBeta)
    plot_multiLorentzianLineComplex(NiKBeta)
    plot_multiLorentzianLineComplex(CuKBeta)
    

def plot_multiLorentzianLineComplex(spectrumDef = CrKAlpha, instrumentGaussianSigma = 0):
    """Makes a single plot showing the lineshape and component parts for a SpectalLine object"""
    plotEnergies = numpy.arange(numpy.round(0.995*spectrumDef.peak_energy),numpy.round(1.008*spectrumDef.peak_energy),0.25)
    
    pylab.figure()
    result = numpy.zeros_like(plotEnergies)
    for energy, fwhm, ampl in zip(spectrumDef.energies, spectrumDef.fwhm, spectrumDef.amplitudes):
        pylab.plot(plotEnergies,ampl*voigt(plotEnergies, energy, hwhm=fwhm*0.5, sigma=instrumentGaussianSigma), label='%.3f, %.3f, %.3f'%(energy,fwhm, ampl))
        result += ampl*voigt(plotEnergies, energy, hwhm=fwhm*0.5, sigma=instrumentGaussianSigma)
    pylab.plot(plotEnergies, result, label='combined', linewidth=2)
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('Fit Counts (arb)')
    pylab.title(spectrumDef.name)
    pylab.legend()
    pylab.xlim((plotEnergies[0], plotEnergies[-1]))
    pylab.ylim((0,numpy.max(result)))
    pylab.show()

def plot_spectrum(spectrum=MnKAlpha(), 
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

    pylab.clf()
    axis = pylab.subplot(111)
    spectrum.set_gauss_fwhm(0.0)
    yvalue = spectrum(e)
    yvalue /= yvalue.max()
    pylab.plot(e, yvalue, color='black', lw=2, label=' 0 eV')
    axis.set_color_cycle(('red', 'orange', '#bbbb00', 'green', 'cyan',
                          'blue', 'indigo', 'purple', 'brown'))
    for res in resolutions:
        spectrum.set_gauss_fwhm(res)
        smeared_spectrum = spectrum(e)
        smeared_spectrum /= smeared_spectrum.max()
        smeared_spectrum *= (1+res*.01)
        pylab.plot(e, smeared_spectrum, label="%2d eV"%res, lw=2)
        
        # Find the peak, valley, peak
        if spectrum.name == 'Manganese K-alpha':
            epk2, evalley, epk1 = 5887.70, 5892.0, 5898.801
        elif spectrum.name == 'Copper K-alpha':
            epk2, evalley, epk1 = 8027.89, 8036.6, 8047.83
            
        p1 = smeared_spectrum[numpy.abs(e-epk1)<2].max()
        if res < 8.12:
            pk2 = smeared_spectrum[numpy.abs(e-epk2)<2].max()
            pval = smeared_spectrum[numpy.abs(e-evalley)<3].min()
            print "Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f" % (res, pk2/p1, pval/pk2) 
        
    pylab.xlim(energy_range)
    pylab.ylim([0, 1.13])
    pylab.legend(loc='upper left')
    
    pylab.title("%s lines at various resolutions (FWHM of Gaussian)" % spectrum.name)
    pylab.xlabel("Energy (eV)")
    pylab.ylabel("Intensity (arb.)")
