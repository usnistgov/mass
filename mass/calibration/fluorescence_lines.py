"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.

Many data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
Phys Rev A56 (#6) pages 4554ff (1997 December).  See online at
http://pra.aps.org/pdf/PRA/v56/i6/p4554_1

Joe Fowler, NIST

Feb 2014       : added aluminum in metal and oxide form
July 12, 2012  : added fitting of Voigt and Lorentzians
November 24, 2010 : started as mn_kalpha.py
"""

__all__ = ['VoigtFitter', 'LorentzianFitter',
           'MultiLorentzianDistribution_gen', 'MultiLorentzianComplexFitter',
           'MnKAlphaDistribution', 'CuKAlphaDistribution',
           'MgKAlphaFitter', 'AlKAlphaFitter',
           'ScKAlphaFitter', 'TiKAlphaFitter', 'VKAlphaFitter',
           'CrKAlphaFitter', 'MnKAlphaFitter', 'FeKAlphaFitter', 'CoKAlphaFitter',
           'NiKAlphaFitter', 'CuKAlphaFitter','TiKBetaFitter', 'CrKBetaFitter',
           'MnKBetaFitter', 'FeKBetaFitter', 'CoKBetaFitter', 'NiKBetaFitter',
           'CuKBetaFitter', 'plot_spectrum']

import numpy as np
import scipy as sp
import pylab as plt

from mass.mathstat import MaximumLikelihoodHistogramFitter, \
    plot_as_stepped_hist, voigt


class SpectralLine(object):
    """An abstract base class for modeling spectral lines as a sum
    of Voigt profiles (i.e., Gaussian-convolved Lorentzians).

    Instantiate one of its subclasses, which will have to define
    self.energies, self.fwhm, self.integral_intensity.  Each must be a sequence
    of the same length.
    """
    def __init__(self):
        """Set up a default Gaussian smearing of 0"""
        self.gauss_sigma = 0.0
        neg_pdf = lambda x: -self.pdf(x)
        self.peak_energy = sp.optimize.brent(neg_pdf, brack=np.array((0.5,1,1.5))*self.nominal_peak_energy)

    def set_gauss_fwhm(self, fwhm):
        """Update the Gaussian smearing to have <fwhm> as the full-width at half-maximum"""
        self.gauss_sigma = fwhm/(8*np.log(2))**0.5

    def __call__(self, x):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x)

    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = np.asarray(x, dtype=np.float)
        result = np.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.fwhm, self.integral_intensity):
            result += ampl*voigt(x, energy, hwhm=fwhm*0.5, sigma=self.gauss_sigma)
            # Note that voigt is normalized to have unit integrated intensity
        return result



class MgKAlpha(SpectralLine):
    """This is the fluorescence line complex of **metallic** magnesium.
    Data are from C. Klauber, Applied Surface Science 70/71 (1993) pages 35-39.
    "Magnesium Kalpha X-ray line structure revisited".  Also discussed in more
    detail in C. Klauber, Surface & Interface Analysis 20 (1993), 703-715.
    """

    ## Spectral complex name.
    name = 'Magnesium K-alpha'
    # The approximation is as a series of 7 Lorentzians
    energies = np.array((-.265, 0, 4.740, 8.210, 8.487, 10.095, 17.404, 20.430)) + 1253.60
    ## The Lorentzian widths (FWHM)
    fwhm = np.array((.541, .541, 1.1056, .6264, .7349, 1.0007, 1.4311, .8656))
    ## The Lorentzian amplitude, in relative integrated intensity
    integral_intensity = np.array((0.5, 1, .02099, .07868, .04712, .09071, .01129, .00538))
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 1253.5587
    ka12_energy_diff = 2.2 # eV (not real, but scales the parameter initial guesses)


class AlKAlpha(SpectralLine):
    """This is the fluorescence line complex of **metallic** aluminum.
    Data are from Joel Ullom, based on email to him from Caroline Kilbourne (NASA
    GSFC) dated 28 Sept 2010.
    """

    ## Spectral complex name.
    name = 'Aluminum K-alpha'
    # The approximation is as a series of 5 Lorentzians
    energies = np.array((1486.9, 1486.5, 1492.3, 1496.4, 1498.4))
    ## The Lorentzian widths (FWHM)
    fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.255))
    ## The Lorentzian peak height, in relative intensity
    # The numbers from Caroline are (1, .5, .02, .12, .06)
    # Steve Smith email to Joe/Joel on 6 March 2014 verifies that these
    # are relative *integral intensities*, not peak heights
    integral_intensity = np.array((1, .5, .02, .12, .06))
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 1486.88931733
    ka12_energy_diff = 3. # eV (not real, but scales the parameter initial guesses)


class AlOxKAlpha(SpectralLine):
    """The K-alpha complex of aluminum **when in oxide form**.
    Data are from Wollman, Nam, Newbury, Hilton, Irwin, Berfren, Deiker, Rudman,
    and Martinis, NIM A 444 (2000) page 145. They come from combining 8 earlier
    references dated 1965 - 1993.
    """

    ## Spectral complex name.
    name = 'Aluminum (oxide) K-alpha'
    # The approximation is as a series of 7 Lorentzians
    energies = np.array((1486.94, 1486.52, 1492.94, 1496.85, 1498.70, 1507.4, 1510.9))
    ## The Lorentzian widths (FWHM)
    fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.25, 1.5, 0.9))
    ## The Lorentzian peak height, in relative intensity
    peak_heights = np.array((1.0, 0.5, 0.033, 0.12, 0.11, 0.07, 0.05), dtype=np.float)
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 1486.930456 # eV
    ka12_energy_diff = 3. # eV (not real, but scales the parameter initial guesses)


class ScKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006).
    "Characterization of K spectral profiles for vanadium, component redetermination for
    scandium, titanium, chromium, and manganese, and development of satellite structure
    for Z=21 to Z=25." Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    ## Spectral complex name.
    name = 'Scandium K-alpha'
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    ## The Lorentzian energies (Table I C_i)
    energies = np.array((4090.595, 4089.308, 4087.666, 4093.428, 4085.773, 4083.697))
    ## The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.13, 2.46, 1.58, 2.04, 1.94, 3.42))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((8203, 818, 257, 381, 4299, 105), dtype=np.float)
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4090.735 # eV
    ka12_energy_diff = 5.1 # eV



class TiKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006).
    "Characterization of K spectral profiles for vanadium, component redetermination for
    scandium, titanium, chromium, and manganese, and development of satellite structure
    for Z=21 to Z=25." Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
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
    energies = np.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088))
    ## The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((4549, 626, 236, 143, 2034, 54), dtype=np.float)
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4510.903 # eV
    ka12_energy_diff = 6.0 # eV

class TiKBeta(SpectralLine):
    """From C Chantler, L Smale, J Kimpton, et al., J Phys B 46, 145601 (2013).
    http://iopscience.iop.org/0953-4075/46/14/145601
    """
    name = 'Titanium K-beta'
    energies = np.array((25.37, 30.096, 31.967, 35.59))+4900
    fwhm = np.array((16.3, 4.25, 0.42, 0.47))
    integral_intensity = np.array((199, 455, 326, 19.2), dtype=np.float)/1e3
    ## The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 4931.966 # eV



class VKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006).
    "Characterization of K spectral profiles for vanadium, component redetermination for
    scandium, titanium, chromium, and manganese, and development of satellite structure
    for Z=21 to Z=25." Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
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
    energies = np.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014))
    ## The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09))
    ## The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((25832, 5410, 1536, 956, 12971, 603), dtype=np.float)
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4952.216 # eV
    ka12_energy_diff = 7.5 # eV

class VKBeta(SpectralLine):
    """From L Smale, C Chantler, M Kinnane, J Kimpton, et al., Phys Rev A 87 022512 (2013).
    http://pra.aps.org/abstract/PRA/v87/i2/e022512
    """
    name = 'Vanadium K-beta'
    energies = np.array((18.20, 24.50, 26.998))+5400
    fwhm = np.array((18.86, 5.48, 2.498))
    integral_intensity = np.array((258, 236, 507), dtype=np.float)/1e3
    nominal_peak_energy = 5426.962 # eV



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
    energies = 5400+np.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986))
    ## The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.4740))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((882, 237, 85, 45, 15, 386, 36), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 5414.81 # eV
    ka12_energy_diff = 9.2 # eV

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
    energies = 5900+np.array((47.00, 35.31, 46.24, 42.04, 44.93))
    ## The Lorentzian widths (Table III W_i)
    fwhm = np.array((1.70, 15.98, 1.90, 6.69, 3.37))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((670, 55, 337, 82, 151), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 5946.82 # eV

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
    ## the 102.712 line doesn't appear in the reference paper, apparently it was added in Scott
    # Porter's refit of the complex. Also, one of the intensities went from 0.005 to 0.018
    energies = 5800+np.array((98.853, 97.867, 94.829, 96.532,
                              99.417, 102.712, 87.743, 86.495))
    ## The Lorentzian widths
    fwhm = np.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216))
    ## The Lorentzian peak height
    peak_heights = np.array((790, 264, 68, 96, 71, 10, 372, 100), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 5898.802 # eV
    ka12_energy_diff = 11.1 # eV


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
    energies = 6400+np.array((90.89, 86.31, 77.73, 90.06, 88.83))
    ## The Lorentzian widths
    fwhm = np.array((1.83, 9.40, 13.22, 1.81, 2.81))
    ## The Lorentzian peak height
    peak_heights = np.array((608, 109, 77, 397, 176), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 6490.18 # eV



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
    energies = np.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275))
    ## The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.613, 1.965, 4.833, 2.803, 2.487, 2.339, 4.433))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((697, 376, 88, 136, 339, 60, 102), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 6404.01 # eV
    ka12_energy_diff = 13.0 # eV

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
    energies = np.array((7046.90, 7057.21, 7058.36, 7054.75))
    ## The Lorentzian widths (Table III W_i)
    fwhm = np.array((14.17, 3.12, 1.97, 6.38))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((107, 448, 615, 141), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 7058.18 # eV



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
    energies = np.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078))
    ## The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((809, 205, 107, 41, 314, 131, 43), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    # Note that the calculated amplitude for the 4th entry 0.808 differs from the paper,
    # but the other numbers appear to be correct, so I think they may have a typo.
    # Also supporting this: the integral intensities
    # in the paper do not add to 1.0 as they are supposed to.
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 6930.38 # eV
    ka12_energy_diff = 15.0 # eV

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
    energies = np.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13))
    ## The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((798, 286, 85, 114, 33, 35), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 7649.45 # eV



class NiKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    ## Spectral complex name.
    name = 'Nickel K-alpha'
    # The approximation is as a series of 5 Lorentzians (2 for KA1,3 for KA2)
    ## The Lorentzian energies (Table II E_i)
    energies = np.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029))
    ## The Lorentzian widths (Table II W_i)
    fwhm = np.array((2.013, 4.711, 2.674, 3.039, 4.476))
    ## The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((909, 136, 351, 79, 24), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 7478.26 # eV
    ka12_energy_diff = 17.2 # eV

class NiKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """
    ## Spectral complex name.
    name = 'Nickel K-beta'
    # The approximation is as a series of 4 Lorentzians
    ## The Lorentzian energies (Table III E_i)
    energies = np.array((8265.01, 8263.01, 8256.67, 8268.70))
    ## The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.76, 4.34, 13.70, 5.18))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((722, 358, 89, 104), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 8264.78 # eV



class CuKAlpha(SpectralLine):
    """Function object to approximate the copper K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """
    ## Spectral complex name.
    name = 'Copper K-alpha'
    # The approximation is 4 of Lorentzians (2 for Ka1, 2 for Ka2)
    ## The Lorentzian energies
    energies = np.array((8047.8372, 8045.3672, 8027.9935, 8026.5041))
    ## The Lorentzian widths
    fwhm = np.array((2.285, 3.358, 2.667, 3.571))
    ## The Lorentzian peak height
    peak_heights = np.array((957, 90, 334, 111), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak
    nominal_peak_energy = 8047.83 # eV
    ka12_energy_diff = 20.0 # eV

class CuKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """

    ## Spectral complex name.
    name = 'Copper K-beta'

    # The approximation is as a series of 5 Lorentzians
    ## The Lorentzian energies (Table III E_i)
    energies = np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393))
    ## The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.52, 3.52, 3.55, 8.08, 5.31))
    ## The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((757, 388, 171, 68, 55), dtype=np.float)/1e3
    ## Amplitude of the Lorentzians
    integral_intensity = (0.5*np.pi*fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    ## The energy at the main peak (from table IV beta1,3)
    nominal_peak_energy = 8905.42 # eV




class MultiLorentzianDistribution_gen(sp.stats.rv_continuous):
    """For producing random variates of the an energy distribution having the form
    of several Lorentzians summed together."""

    ## Approximates the random variate defined by multiple Lorentzian components.
    #  @param args  Pass all other parameters to parent class.
    #  @param kwargs  Pass all other parameters to parent class.
    def __init__(self, distribution, *args, **kwargs):
        """<args> and <kwargs> are passed on to sp.stats.rv_continuous"""

        sp.stats.rv_continuous.__init__(self, *args, **kwargs)
        self.distribution = distribution
        self.cumulative_amplitudes = self.distribution.integral_intensity.cumsum()
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
                            np.random.uniform(0, self.cumulative_amplitudes[-1], size=self._size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = np.random.standard_cauchy(size=self._size)*self.distribution.fwhm[iline]*0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if self.distribution.gauss_sigma > 0.0:
            lor += np.random.standard_normal(size=self._size)*self.distribution.gauss_sigma
        # Finally, add the line centers.
        return lor + self.distribution.energies[iline]



# Some specific fluorescence lines
# You can see how to make more if you like.
MnKAlphaDistribution = MultiLorentzianDistribution_gen(distribution=MnKAlpha(),
                                                       name="Mn Kalpha fluorescence")
MnKBetaDistribution  = MultiLorentzianDistribution_gen(distribution=MnKBeta(),
                                                       name="Mn Kbeta fluorescence")
CuKAlphaDistribution = MultiLorentzianDistribution_gen(distribution=CuKAlpha(),
                                                       name="Cu Kalpha fluorescence")



class VoigtFitter(object):
    """Fit a single Lorentzian line, with Gaussian smearing."""
    def __init__(self):
        """ """
        ## Parameters from last successful fit
        self.last_fit_params = None
        ## Fit function samples from last successful fit
        self.last_fit_result = None


    def guess_starting_params(self, data, binctrs):
        order_stat = np.array(data.cumsum(), dtype=np.float)/data.sum()
        percentiles = lambda p: binctrs[(order_stat>p).argmax()]
        peak_loc = percentiles(0.5)
        iqr = (percentiles(0.75)-percentiles(0.25))
        res = iqr*0.7
        lor_hwhm = res*0.5
        baseline = data[0:10].mean()
        baseline_slope = (data[-10:].mean()-baseline)/len(data)
        ampl = (data.max()-baseline)*np.pi
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
        sigma = params[0]/(8*np.log(2))**0.5
        spectrum = voigt(x, params[1], params[2], sigma)
        nbins = len(x)
        return spectrum * abs(params[3]) + abs(params[4]) + params[5]*np.arange(nbins)



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
        # Work with bin edges
        if len(pulseheights) == len(data) + 1:
            dp = pulseheights[1]-pulseheights[0]
            pulseheights = 0.5*dp + pulseheights[:-1]

        # Pulseheights doesn't make sense as bin centers, either.
        # So just use the integers starting at zero.
        elif len(pulseheights) != len(data):
            pulseheights = np.arange(len(data), dtype=np.float)

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
                plt.clf()
                axis = plt.subplot(111)

            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

        # Joe's new max-likelihood fitter
        epsilon = np.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01))
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
            de = np.sqrt(covariance[2, 2])
            label = "Lorentz HWHM: %.2f +- %.2f eV %s"%(fitparams[2], de, label)
            if 0 not in hold:
                de = np.sqrt(covariance[0, 0])
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
#        order_stat = np.array(data.cumsum(), dtype=np.float)/data.sum()
#        percentiles = lambda p: binctrs[(order_stat>p).argmax()]
#        peak_loc = percentiles(0.5)
#        iqr = (percentiles(0.75)-percentiles(0.25))
#        res = iqr*0.7
#        lor_hwhm = res*0.5
#        baseline = data[0:10].mean()
#        baseline_slope = (data[-10:].mean()-baseline)/len(data)
#        ampl = (data.max()-baseline)*np.pi
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
        sigma = params[0]/(8*np.log(2))**0.5
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
        # Work with bin edges
        if len(pulseheights) == len(data) + 1:
            dp = pulseheights[1]-pulseheights[0]
            pulseheights = 0.5*dp + pulseheights[:-1]

        # Pulseheights doesn't make sense as bin centers, either.
        # So just use the integers starting at zero.
        elif len(pulseheights) != len(data):
            pulseheights = np.arange(len(data), dtype=np.float)

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
                plt.clf()
                axis = plt.subplot(111)

            plot_as_stepped_hist(axis, data, pulseheights, color=color)
            ph_binsize = pulseheights[1]-pulseheights[0]
            axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

        # Joe's new max-likelihood fitter
        epsilon = np.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5,
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
            de1 = np.sqrt(covariance[2, 2])
            label = "Lorentz HWHM 1: %.2f +- %.2f eV %s"%(fitparams[2], de1, label)
            de2 = np.sqrt(covariance[5, 5])
            label += "\nLorentz HWHM 2: %.2f +- %.2f eV"%(fitparams[5], de2)
            if 0 not in hold:
                de = np.sqrt(covariance[0, 0])
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
        outSpectrum = np.zeros_like(energy)
        for i,spectrum  in enumerate(self.spectraDefs):
            # if it crashes here you probably didnt instantiate your spectra defs... ie use MnKAlpha() not MnKAlpha
            spectrum.set_gauss_fwhm(params[0])
            if i == 0:
                ampIndex = 3
            else:
                ampIndex = 5+i
            outSpectrum += params[ampIndex]*spectrum.pdf(energy) # probability density function
        background = abs(params[4])+params[5]*np.arange(len(energy))
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
        rms_d = np.sqrt(sum_d2/n - mean_d**2)
#        print n, sum_d, sum_d2, mean_d, rms_d

        ph_ka1 = binctrs[np.argmax(data)]
        dph = 2*rms_d

        if len(self.spectraDefs) >= 2:
            linecenters = []
            for spectrum in self.spectraDefs:
                linecenters.append(spectrum.peak_energy)
            dE = np.max(linecenters)-np.min(linecenters)
        else:
            dE = np.max(self.spectraDefs[0].energies)-np.min(self.spectraDefs[0].energies) # eV difference between KAlpha peaks
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
        for _ in range(len(self.spectraDefs)-1): # add elements if neccesary for relative amplitudes of each spectrumDef
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
        # Work with bin edges
        if len(pulseheights) == len(data) + 1:
            dp = pulseheights[1]-pulseheights[0]
            pulseheights = 0.5*dp + pulseheights[:-1]

        # Pulseheights doesn't make sense as bin centers, either.
        # So just use the integers starting at zero.
        elif len(pulseheights) != len(data):
            pulseheights = np.arange(len(data), dtype=np.float)

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
        epsilon = np.ones_like(params)*params[1]/1e4
        epsilon[:6] = np.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01))
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
                plt.clf()
                axis = plt.subplot(111)
                plt.xlabel('pulseheight (arb)')
                plt.ylabel('counts per %.3f unit bin'%ph_binsize)
                plt.title('resolution %.3f, Ka1_ph %.3f, dph/de %.3f\n amp %.3f, bg %.3f, bg_slope %.3f'%tuple(fitparams[:6]))
                plot_as_stepped_hist(axis, data, pulseheights, color=color)
                axis.set_xlim([pulseheights[0]-0.5*ph_binsize, pulseheights[-1]+0.5*ph_binsize])

            de = np.sqrt(covariance[0, 0])
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
        ## Parameters and fit function from last successful fit
        self.last_fit_params = None
        self.last_fit_cov = None
        self.last_fit_result = None


    ## Compute the smeared line complex.
    #
    # @param params  The 8 parameters of the fit (see self.fit for details).
    # @param x       An array of pulse heights (params will scale them to energy).
    # @return:       The line complex intensity, including resolution smearing.
    def fitfunc(self, params, x):
        """Return the smeared line complex.

        <params>  The 8 parameters of the fit (see self.fit for details).
        <x>       An array of pulse heights (params will scale them to energy).
        Returns:  The line complex intensity, including resolution smearing.
        """
        (P_gaussfwhm, P_phpeak, P_dphde, P_amplitude,
         P_bg, P_bgslope, P_tailfrac, P_tailtau) = params

        energy = (x-P_phpeak)/P_dphde + self.spect.peak_energy
        self.spect.set_gauss_fwhm(P_gaussfwhm)

        # Now either return the pure sum-of-Voigt function spectrum, if no low-E tail
        if P_tailfrac <= 1e-5:
            spectrum = self.spect.pdf(energy)

        # Or compute the low-E-tailed spectrum. This is done by
        # convolution, which is computed using DFT methods.
        # A wider energy range must be used, or wrap-around effects of
        # tails will corrupt the model.
        else:
            de = energy[1]-energy[0]
            nlow = int(min(P_tailtau, 100)*6/de + 0.5)
            nhi = int((abs(params[0])+min(P_tailtau, 50))/de + 0.5)
            nhi = min(3000, nhi) # A practical limit
            nlow = max(nlow, nhi)
            lowe = np.arange(-nlow,0)*de + energy[0]
            highe = np.arange(1,nhi+1)*de + energy[-1]
            energy_wide = np.hstack([lowe, energy, highe])
            freq = np.fft.rfftfreq(len(energy_wide), d=de)
            rawspectrum = self.spect.pdf(energy_wide)
            ft = np.fft.rfft(rawspectrum)
            # Joe writes: Not sure why the following fails. Hmm.
#             ft = np.zeros(len(freq), dtype=complex)
#             for line_e, line_fwhm, line_intens in zip(self.spect.energies, self.spect.fwhm,
#                                                        self.spect.integral_intensity):
#                 ft += line_intens*np.exp(-np.pi*(line_fwhm*np.abs(freq)+2j*freq*line_e))
#             ft *= np.exp(-2*(np.pi*params[0]/2.3548)**2)
            ft += ft*P_tailfrac*(1.0/(1-2j*np.pi*freq*P_tailtau)-1)
            smoothspectrum = np.fft.irfft(ft, n=len(energy_wide))
            spectrum = smoothspectrum[nlow:nlow+len(energy)]
        nbins = len(x)
        return spectrum * P_amplitude + P_bg + P_bgslope*np.arange(nbins)



    def fit(self, data, pulseheights=None, params=None, plot=True, axis=None, color=None,
            label="", vary_bg=True, vary_bg_slope=False, vary_tail=False, hold=None):
        """Attempt a fit to the spectrum <data>, a histogram of X-ray counts parameterized as the
        set of histogram bins <pulseheights>.

        pulseheights: the histogram bin centers or edges.  This will be inferred by whether
                      its length is equal to or one more than the length of data.
                      If pulseheights is None, then the parameters
                      normally having pulseheight units will be returned as bin numbers instead.

        params: a 8-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                energy scale factor (pulseheight/eV), amplitude, background level (per bin),
                background slope (in counts per bin per bin), fraction in low-E-tail, and
                exponential scale length (eV) of the tail. ]
                If params is None, all params will be guessed.
                If params is a 8 element list, all elements with value None will be guessed.

        plot:   Whether to make a plot.  If not, then the next few args are ignored
        axis:   If given, and if plot is True, then make the plot on this matplotlib.Axes rather than on the
                current figure.
        color:  Color for drawing the histogram contents behind the fit.
        label:  Label for the fit line to go into the plot (usually used for resolution and uncertainty)

        vary_bg:       Whether to let a constant background level vary in the fit
        vary_bg_slope: Whether to let a slope on the background level vary in the fit
        vary_tail:     Whether to let an exponential tail vary in the fit
        hold:          A sequence of parameter numbers (0 to 7, inclusive) to hold, or None.
                       BG, BG slope, and tail will be held if 4, 5, or 6 appears in the hold
                       sequence OR if the relevant boolean vary_* tests False.
        ====================================================================
        returns fitparams, covariance
        fitparams has same format as input variable params
        """

        # Were we given bin edges? Recognize by having 1 extra value.
        if len(pulseheights) == len(data) + 1:
            dp = pulseheights[1]-pulseheights[0]
            pulseheights = 0.5*dp + pulseheights[:-1]

        # Pulseheights is the wrong length to be either bin edges OR centers,
        # so just use the integers starting at zero and ignore the input
        elif len(pulseheights) != len(data):
            print "Warning: len(pulseheights)=%d makes no sense given len(data)=%d"%(
                len(pulseheights), len(data))
            pulseheights = np.arange(len(data), dtype=np.float)

        # If params is None, use guesses for all parameters. If it's a sequence with some
        # None values, use guesses just for those.
        guess_params = self.guess_starting_params(data, pulseheights)
        if params is None:
            params = guess_params
        for j in xrange(len(params)):
            if params[j] is not None:
                guess_params[j] = params[j]

        # Set held parameters to be held.
        # Note that some can be held in either of two different ways.
        if hold is None:
            hold = set()
        else:
            hold = set(hold)
        if not vary_bg: hold.add(4)
        if not vary_bg_slope: hold.add(5)
        if not vary_tail: hold.add(6)
        if 6 in hold and params[6] <= 0.0: hold.add(7)
        if 6 not in hold and params[6] <= 0.0: params[6] = .01
        if params[4] <=0: params[4] = 1e-6

        # Set up the maximum-likelihood fitter
        epsilon = np.array((1e-3, params[1]/1e5, 1e-3, params[3]/1e5, params[4]/1e2, .01, .01, 1))
        MLfitter = MaximumLikelihoodHistogramFitter(pulseheights, data, params,
                                                    self.fitfunc, TOL=1e-4, epsilon=epsilon)
        MLfitter.setbounds(0, 0, 100)  # 100 > FWHM > 0
        MLfitter.setbounds(1, 0, None) # PH at peak > 0
        MLfitter.setbounds(2, 0, None) # dPH/dE > 0
        MLfitter.setbounds(3, 0, None) # amplitude > 0
        MLfitter.setbounds(4, 0, None) # BG level > 0
        MLfitter.setbounds(6, 0, 1)    # 0 < tailamplitude < 1
        maxtail = 200 * params[2]
        MLfitter.setbounds(7, 0, maxtail) # 0 < tailtau < 200 eV

        for h in hold:
            MLfitter.hold(h)

        fitparams, covariance = MLfitter.fit()

        self.last_fit_params = fitparams
        self.last_fit_cov = covariance
        self.last_fit_chisq = MLfitter.chisq
        self.last_fit_result = self.fitfunc(fitparams, pulseheights)
        self.last_fit_bins = pulseheights.copy()
        self.last_fit_contents = data.copy()
        if plot: self.plotFit(color, axis, label)

        return fitparams, covariance


    def plotFit(self, color=None, axis=None, label=""):
        """Plot the last fit and the data to which it was fit."""
        if color is None: color = 'blue'
        bins = self.last_fit_bins
        data = self.last_fit_contents
        ph_binsize = bins[1]-bins[0]
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            plt.xlabel('pulseheight (arb) - %s'%self.spect.name)
            plt.ylabel('counts per %.3f unit bin'%ph_binsize)
            plt.title(('resolution %.3f, amplitude %.3f, dph/de %.3f\n amp %.3f, '
                      'bg %.3f, bg_slope %.3f. T=%.3f $\\tau$=%.3f')%
                      tuple(self.last_fit_params))
        plot_as_stepped_hist(axis, data, bins, color=color)
        axis.set_xlim([bins[0]-0.5*ph_binsize, bins[-1]+0.5*ph_binsize])
        de = np.sqrt(self.last_fit_cov[0, 0])
        axis.plot(bins, self.last_fit_result, color='#666666',
                  label="%.2f +- %.2f eV %s"%(self.last_fit_params[0], de, label))
        axis.legend(loc='upper left')



class GenericKAlphaFitter(MultiLorentzianComplexFitter):
    """
    Fits a generic K alpha spectrum for energy shift and scale, amplitude, and resolution.
    Background level (including a fixed slope) and low-E tailing are also included.

    Note that self.tailfrac and self.tailtau are attributes that determine the starting
    guess for the fraction of events in an exponential low-energy tail and for that tail's
    exponential scale length (in eV). Change if desired.
    """

    def __init__(self, spectrumDef):
        """
        Constructor argument spectrumDef should be mass.fluorescence_lines.MnKAlpha, or similar
        subclasses of SpectralLine.
        """
        self.spect = spectrumDef
        MultiLorentzianComplexFitter.__init__(self)
        self.tailfrac = 0.0
        self.tailtau = 25

    def guess_starting_params(self, data, binctrs):
        """
        A decent heuristic for guessing the inital values, though your informed
        starting point is likely to be better than this.
        """
        if data.sum()<=0: raise ValueError("This histogram has no contents")

        # Heuristic: find the Ka1 line as the peak bin, and then make
        # assumptions about the full width (from 1/4-peak to 1/4-peak) and
        # how that relates to the PH spacing between Ka1 and Ka2
        peak_val = data.max()
        peak_ph = binctrs[data.argmax()]
        lowqtr = binctrs[(data>peak_val*0.25).argmax()]
        N = len(data)
        topqtr = binctrs[N-(data>peak_val*0.25)[::-1].argmax()]

        ph_ka1 = peak_ph
        dph = 0.66*(topqtr-lowqtr)
        dE = self.spect.ka12_energy_diff # eV difference between KAlpha peaks
        ampl = data.max() * 9.4
        res = 4.0
        if len(data) > 20:
            baseline = data[0:10].mean()+1e-6
        else:
            baseline = 0.1
        baseline_slope = 0.0
        return [res, ph_ka1, dph/dE, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]



class GenericKBetaFitter(MultiLorentzianComplexFitter):
    def __init__(self, spectrumDef=MnKBeta):
        """
        Constructor argument spectrumDef should be mass.fluorescence_lines.MnKAlpha, or similar
        subclasses of SpectralLine.
        """
        self.spect = spectrumDef
        MultiLorentzianComplexFitter.__init__(self)
        self.tailfrac = 0.0
        self.tailtau = 25

    def guess_starting_params(self, data, binctrs):
        """Hard to estimate dph/de from a K-beta line. Have to guess scale=1 and
        hope it's close enough to get convergence. Ugh!"""
        peak_ph = binctrs[data.argmax()]
        ampl = data.max() * 9.4
        res = 4.0
        if len(data) > 20:
            baseline = data[0:10].mean()
        else:
            baseline = 0.1
        baseline_slope = 0.0
        return [res, peak_ph, 1.0, ampl, baseline, baseline_slope,
                self.tailfrac, self.tailtau]


## create specific KAlpha Fitters
class _lowZ_KAlphaFitter(GenericKAlphaFitter):
    """Overrides the starting parameter guesses, more appropriate
    for low Z where the Ka1,2 peaks can't be resolved."""
    def guess_starting_params(self, data, binctrs):
        n = data.sum()
        if n<=0:
            raise ValueError("This histogram has no contents")
        cumdata = np.cumsum(data)
        ph_ka1 = binctrs[(cumdata*2>n).argmax()]
        res = 2.0
        dph_de = 1.0
        baseline, baseline_slope = 1.0, 0.0
        ampl = 4*np.max(data)
        return [res, ph_ka1, dph_de, ampl, baseline, baseline_slope]



class AlKAlphaFitter(_lowZ_KAlphaFitter):
    def __init__(self):
        _lowZ_KAlphaFitter.__init__(self, AlKAlpha())
class MgKAlphaFitter(_lowZ_KAlphaFitter):
    def __init__(self):
        _lowZ_KAlphaFitter.__init__(self, MgKAlpha())
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
        GenericKBetaFitter.__init__(self, TiKBeta())
class VKBetaFitter(GenericKBetaFitter):
    def __init__(self):
        GenericKBetaFitter.__init__(self, VKBeta())
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

    plot_multiLorentzianLineComplex(TiKBeta)
    plot_multiLorentzianLineComplex(VKBeta)
    plot_multiLorentzianLineComplex(CrKBeta)
    plot_multiLorentzianLineComplex(MnKBeta)
    plot_multiLorentzianLineComplex(FeKBeta)
    plot_multiLorentzianLineComplex(CoKBeta)
    plot_multiLorentzianLineComplex(NiKBeta)
    plot_multiLorentzianLineComplex(CuKBeta)


def plot_multiLorentzianLineComplex(spectrumDef = CrKAlpha, instrumentGaussianSigma = 0):
    """Makes a single plot showing the lineshape and component parts for a SpectalLine object"""
    peak = spectrumDef().peak_energy
    plotEnergies = np.arange(np.round(0.995*peak),np.round(1.008*peak),0.25)

    plt.figure()
    result = np.zeros_like(plotEnergies)
    for energy, fwhm, ampl in zip(spectrumDef.energies, spectrumDef.fwhm, spectrumDef.integral_intensity):
        plt.plot(plotEnergies,ampl*voigt(plotEnergies, energy, hwhm=fwhm*0.5, sigma=instrumentGaussianSigma), label='%.3f, %.3f, %.3f'%(energy,fwhm, ampl))
        result += ampl*voigt(plotEnergies, energy, hwhm=fwhm*0.5, sigma=instrumentGaussianSigma)
    plt.plot(plotEnergies, result, label='combined', linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Fit Counts (arb)')
    plt.title(spectrumDef.name)
    plt.legend()
    plt.xlim((plotEnergies[0], plotEnergies[-1]))
    plt.ylim((0,np.max(result)))
    plt.show()

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
    e = np.arange(energy_range[0]-2.5*resolutions[-1],
                     energy_range[1]+2.5*resolutions[-1], stepsize)

    plt.clf()
    axis = plt.subplot(111)
    spectrum.set_gauss_fwhm(0.0)
    yvalue = spectrum(e)
    yvalue /= yvalue.max()
    plt.plot(e, yvalue, color='black', lw=2, label=' 0 eV')
    axis.set_color_cycle(('red', 'orange', '#bbbb00', 'green', 'cyan',
                          'blue', 'indigo', 'purple', 'brown'))
    for res in resolutions:
        spectrum.set_gauss_fwhm(res)
        smeared_spectrum = spectrum(e)
        smeared_spectrum /= smeared_spectrum.max()
        smeared_spectrum *= (1+res*.01)
        plt.plot(e, smeared_spectrum, label="%2d eV"%res, lw=2)

        # Find the peak, valley, peak
        if spectrum.name == 'Titanium K-alpha':
            epk2, evalley, epk1 = 4504.91, 4507.32, 4510.90
        elif spectrum.name == 'Chromium K-alpha':
            epk2, evalley, epk1 = 5405.55, 5408.87, 5414.81
        elif spectrum.name == 'Manganese K-alpha':
            epk2, evalley, epk1 = 5887.70, 5892.0, 5898.801
        elif spectrum.name == 'Iron K-alpha':
            epk2, evalley, epk1 = 6391.06, 6396.13, 6404.01
        elif spectrum.name == 'Cobalt K-alpha':
            epk2, evalley, epk1 = 6915.55, 6921.40, 6930.39
        elif spectrum.name == 'Copper K-alpha':
            epk2, evalley, epk1 = 8027.89, 8036.6, 8047.83
        else:
            continue

        p1 = smeared_spectrum[np.abs(e-epk1)<2].max()
        if res < 8.12:
            pk2 = smeared_spectrum[np.abs(e-epk2)<2].max()
            pval = smeared_spectrum[np.abs(e-evalley)<3].min()
            print "Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f" % (res, pk2/p1, pval/pk2)

    plt.xlim(energy_range)
    plt.ylim([0, 1.13])
    plt.legend(loc='upper left')

    plt.title("%s lines at various resolutions (FWHM of Gaussian)" % spectrum.name)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (arb.)")


# code for exporting lineshapes
# for key in mass.fluorescence_lines.__dict__:
#     try:
#         obj = mass.fluorescence_lines.__dict__[key]
#         print(key+'=MultiLorentzianComplex('+str(obj.energies)+','+str(obj.fwhm)+','+str(obj.integral_intensity)+','+str(obj.nominal_peak_energy)+')')
#     except:
#         pass
