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

# __all__ = ['VoigtFitter', 'LorentzianFitter',
#            'MultiLorentzianDistribution_gen', 'MultiLorentzianComplexFitter',
#            'MnKAlphaDistribution', 'CuKAlphaDistribution',
#            'MgKAlphaFitter', 'AlKAlphaFitter',
#            'ScKAlphaFitter', 'TiKAlphaFitter', 'VKAlphaFitter',
#            'CrKAlphaFitter', 'MnKAlphaFitter', 'FeKAlphaFitter', 'CoKAlphaFitter',
#            'NiKAlphaFitter', 'CuKAlphaFitter','TiKBetaFitter', 'CrKBetaFitter',
#            'MnKBetaFitter', 'FeKBetaFitter', 'CoKBetaFitter', 'NiKBetaFitter',
#            'CuKBetaFitter', 'plot_spectrum']

import numpy as np
import scipy as sp
import pylab as plt

from mass.mathstat.fitting import MaximumLikelihoodHistogramFitter
from mass.mathstat.utilities import plot_as_stepped_hist
from mass.mathstat.special import voigt


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
        self.peak_energy = sp.optimize.brent(neg_pdf, brack=np.array((0.5, 1, 1.5)) * self.nominal_peak_energy)

    def set_gauss_fwhm(self, fwhm):
        """Update the Gaussian smearing to have <fwhm> as the full-width at half-maximum"""
        self.gauss_sigma = fwhm / (8 * np.log(2))**0.5

    def __call__(self, x):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x)

    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = np.asarray(x, dtype=np.float)
        result = np.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.fwhm, self.integral_intensity):
            result += ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=self.gauss_sigma)
            # Note that voigt is normalized to have unit integrated intensity
        return result


class MgKAlpha(SpectralLine):
    """This is the fluorescence line complex of **metallic** magnesium.
    Data are from C. Klauber, Applied Surface Science 70/71 (1993) pages 35-39.
    "Magnesium Kalpha X-ray line structure revisited".  Also discussed in more
    detail in C. Klauber, Surface & Interface Analysis 20 (1993), 703-715.
    """

    # Spectral complex name.
    name = 'Magnesium K-alpha'
    # The approximation is as a series of 7 Lorentzians
    energies = np.array((-.265, 0, 4.740, 8.210, 8.487, 10.095, 17.404, 20.430)) + 1253.60
    # The Lorentzian widths (FWHM)
    fwhm = np.array((.541, .541, 1.1056, .6264, .7349, 1.0007, 1.4311, .8656))
    # The Lorentzian amplitude, in relative integrated intensity
    integral_intensity = np.array((0.5, 1, .02099, .07868, .04712, .09071, .01129, .00538))
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 1253.5587
    ka12_energy_diff = 2.2  # eV (not real, but scales the parameter initial guesses)


class AlKAlpha(SpectralLine):
    """This is the fluorescence line complex of **metallic** aluminum.
    Data are from Joel Ullom, based on email to him from Caroline Kilbourne (NASA
    GSFC) dated 28 Sept 2010.
    """

    # Spectral complex name.
    name = 'Aluminum K-alpha'
    # The approximation is as a series of 5 Lorentzians
    energies = np.array((1486.9, 1486.5, 1492.3, 1496.4, 1498.4))
    # The Lorentzian widths (FWHM)
    fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.255))
    # The Lorentzian peak height, in relative intensity
    # The numbers from Caroline are (1, .5, .02, .12, .06)
    # Steve Smith email to Joe/Joel on 6 March 2014 verifies that these
    # are relative *integral intensities*, not peak heights
    integral_intensity = np.array((1, .5, .02, .12, .06))
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 1486.88931733
    ka12_energy_diff = 3.  # eV (not real, but scales the parameter initial guesses)


class AlOxKAlpha(SpectralLine):
    """The K-alpha complex of aluminum **when in oxide form**.
    Data are from Wollman, Nam, Newbury, Hilton, Irwin, Berfren, Deiker, Rudman,
    and Martinis, NIM A 444 (2000) page 145. They come from combining 8 earlier
    references dated 1965 - 1993.
    """

    # Spectral complex name.
    name = 'Aluminum (oxide) K-alpha'
    # The approximation is as a series of 7 Lorentzians
    energies = np.array((1486.94, 1486.52, 1492.94, 1496.85, 1498.70, 1507.4, 1510.9))
    # The Lorentzian widths (FWHM)
    fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.25, 1.5, 0.9))
    # The Lorentzian peak height, in relative intensity
    peak_heights = np.array((1.0, 0.5, 0.033, 0.12, 0.11, 0.07, 0.05), dtype=np.float)
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 1486.930456  # eV
    ka12_energy_diff = 3.  # eV (not real, but scales the parameter initial guesses)


class ScKAlpha(SpectralLine):
    """Data are from Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006).
    "Characterization of K spectral profiles for vanadium, component redetermination for
    scandium, titanium, chromium, and manganese, and development of satellite structure
    for Z=21 to Z=25." Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    # Spectral complex name.
    name = 'Scandium K-alpha'
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    # The Lorentzian energies (Table I C_i)
    energies = np.array((4090.595, 4089.308, 4087.666, 4093.428, 4085.773, 4083.697))
    # The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.13, 2.46, 1.58, 2.04, 1.94, 3.42))
    # The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((8203, 818, 257, 381, 4299, 105), dtype=np.float)
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4090.735  # eV
    ka12_energy_diff = 5.1  # eV


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
    # Spectral complex name.
    name = 'Titanium K-alpha'
    # the paper has two sets of Ti data, I used the set Refit of [21] Kawai et al 1994
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    # The Lorentzian energies (Table I C_i)
    energies = np.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088))
    # The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49))
    # The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((4549, 626, 236, 143, 2034, 54), dtype=np.float)
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4510.903  # eV
    ka12_energy_diff = 6.0  # eV


class TiKBeta(SpectralLine):
    """From C Chantler, L Smale, J Kimpton, et al., J Phys B 46, 145601 (2013).
    http://iopscience.iop.org/0953-4075/46/14/145601

    Careful! Note that Chantler's instrument had a beam sigma of 1.244+-.041 eV,
    or so this result cannot be taken very seriously below 3 eV (FWHM) of Gaussian
    smearing.
    """
    name = 'Titanium K-beta'
    energies = np.array((25.37, 30.096, 31.967, 35.59)) + 4900
    fwhm = np.array((16.3, 4.25, 0.42, 0.47))
    integral_intensity = np.array((199, 455, 326, 19.2), dtype=np.float) / 1e3
    # The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 4931.966  # eV


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
    # Spectral complex name.
    name = 'Vanadium K-alpha'
    # The approximation is as a series of 6 Lorentzians (4 for KA1,2 for KA2)
    # The Lorentzian energies (Table I C_i)
    energies = np.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014))
    # The Lorentzian widths (Table I W_i)
    fwhm = np.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09))
    # The Lorentzian peak height (Table I A_i)
    peak_heights = np.array((25832, 5410, 1536, 956, 12971, 603), dtype=np.float)
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table III Kalpha_1^0)
    nominal_peak_energy = 4952.216  # eV
    ka12_energy_diff = 7.5  # eV


class VKBeta(SpectralLine):
    """We were using L Smale, C Chantler, M Kinnane, J Kimpton, et al., Phys Rev A 87 022512 (2013).
    http://pra.aps.org/abstract/PRA/v87/i2/e022512

    BUT these were adjusted in C Chantler, L Smale, J Kimpton, et al., J Phys B 46, 145601 (2013).
    http://iopscience.iop.org/0953-4075/46/14/145601  (see Section 5 "Redefinition of
    vanadium K-beta standard")  Both papers are by the same group, of course.
    """
    name = 'Vanadium K-beta'
    energies = np.array((18.19, 24.50, 26.992)) + 5400
    fwhm = np.array((18.86, 5.48, 2.499))
    integral_intensity = np.array((258, 236, 507), dtype=np.float) / 1e3
    nominal_peak_energy = 5426.956  # eV


class CrKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).

    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Chromium K-alpha'
    # The approximation is as a series of 7 Lorentzians (5 for KA1,2 for KA2)
    # The Lorentzian energies (Table II E_i)
    energies = 5400 + np.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986))
    # The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.4740))
    # The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((882, 237, 85, 45, 15, 386, 36), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 5414.81  # eV
    ka12_energy_diff = 9.2  # eV


class CrKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).

    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    # Spectral complex name.
    name = 'Chromium K-beta'

    # The approximation is as a series of 5 Lorentzians
    # The Lorentzian energies (Table III E_i)
    energies = 5900 + np.array((47.00, 35.31, 46.24, 42.04, 44.93))
    # The Lorentzian widths (Table III W_i)
    fwhm = np.array((1.70, 15.98, 1.90, 6.69, 3.37))
    # The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((670, 55, 337, 82, 151), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 5946.82  # eV


class MnKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).

    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Manganese K-alpha'

    # The approximation is as a series of 8 Lorentzians (6 for KA1,2 for KA2)

    # The Lorentzian energies
    # the 102.712 line doesn't appear in the reference paper, apparently it was added in Scott
    # Porter's refit of the complex. Also, one of the intensities went from 0.005 to 0.018
    energies = 5800 + np.array((98.853, 97.867, 94.829, 96.532,
                                99.417, 102.712, 87.743, 86.495))
    # The Lorentzian widths
    fwhm = np.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216))
    # The Lorentzian peak height
    peak_heights = np.array((790, 264, 68, 96, 71, 10, 372, 100), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 5898.802  # eV
    ka12_energy_diff = 11.1  # eV


class MnKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).

    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """

    # Spectral complex name.
    name = 'Manganese K-beta'

    # The approximation is as a series of 4 Lorentzians
    # The Lorentzian energies
    energies = 6400 + np.array((90.89, 86.31, 77.73, 90.06, 88.83))
    # The Lorentzian widths
    fwhm = np.array((1.83, 9.40, 13.22, 1.81, 2.81))
    # The Lorentzian peak height
    peak_heights = np.array((608, 109, 77, 397, 176), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 6490.18  # eV


class FeKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Iron K-alpha'
    # The approximation is as a series of 7 Lorentzians (4 for KA1,3 for KA2)
    # The Lorentzian energies (Table II E_i)
    energies = np.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275))
    # The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.613, 1.965, 4.833, 2.803, 2.487, 2.339, 4.433))
    # The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((697, 376, 88, 136, 339, 60, 102), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 6404.01  # eV
    ka12_energy_diff = 13.0  # eV


class FeKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Iron K-beta'
    # The approximation is as a series of 4 Lorentzians
    # The Lorentzian energies (Table III E_i)
    energies = np.array((7046.90, 7057.21, 7058.36, 7054.75))
    # The Lorentzian widths (Table III W_i)
    fwhm = np.array((14.17, 3.12, 1.97, 6.38))
    # The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((107, 448, 615, 141), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 7058.18  # eV


class CoKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Cobalt K-alpha'
    # The approximation is as a series of 7 Lorentzians (4 for KA1,3 for KA2)
    # The Lorentzian energies (Table II E_i)
    energies = np.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078))
    # The Lorentzian widths (Table II W_i)
    fwhm = np.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463))
    # The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((809, 205, 107, 41, 314, 131, 43), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    # Note that the calculated amplitude for the 4th entry 0.808 differs from the paper,
    # but the other numbers appear to be correct, so I think they may have a typo.
    # Also supporting this: the integral intensities
    # in the paper do not add to 1.0 as they are supposed to.
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 6930.38  # eV
    ka12_energy_diff = 15.0  # eV


class CoKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Cobalt K-beta'
    # The approximation is as a series of 6 Lorentzians
    # The Lorentzian energies (Table III E_i)
    energies = np.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13))
    # The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79))
    # The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((798, 286, 85, 114, 33, 35), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 7649.45  # eV


class NiKAlpha(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code).
    Note that the subclass holds all the data (as class attributes), while
    the parent class SpectralLine holds all the code.
    """
    # Spectral complex name.
    name = 'Nickel K-alpha'
    # The approximation is as a series of 5 Lorentzians (2 for KA1,3 for KA2)
    # The Lorentzian energies (Table II E_i)
    energies = np.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029))
    # The Lorentzian widths (Table II W_i)
    fwhm = np.array((2.013, 4.711, 2.674, 3.039, 4.476))
    # The Lorentzian peak height (Table II I_i)
    peak_heights = np.array((909, 136, 351, 79, 24), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV alpha_1)
    nominal_peak_energy = 7478.26  # eV
    ka12_energy_diff = 17.2  # eV


class NiKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """
    # Spectral complex name.
    name = 'Nickel K-beta'
    # The approximation is as a series of 4 Lorentzians
    # The Lorentzian energies (Table III E_i)
    energies = np.array((8265.01, 8263.01, 8256.67, 8268.70))
    # The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.76, 4.34, 13.70, 5.18))
    # The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((722, 358, 89, 104), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV beta_1,3)
    nominal_peak_energy = 8264.78  # eV


class CuKAlpha(SpectralLine):
    """Function object to approximate the copper K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """
    # Spectral complex name.
    name = 'Copper K-alpha'
    # The approximation is 4 of Lorentzians (2 for Ka1, 2 for Ka2)
    # The Lorentzian energies
    energies = np.array((8047.8372, 8045.3672, 8027.9935, 8026.5041))
    # The Lorentzian widths
    fwhm = np.array((2.285, 3.358, 2.667, 3.571))
    # The Lorentzian peak height
    peak_heights = np.array((957, 90, 334, 111), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak
    nominal_peak_energy = 8047.83  # eV
    ka12_energy_diff = 20.0  # eV


class CuKBeta(SpectralLine):
    """Function object to approximate the manganese K-alpha complex
    Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December).
    """

    # Spectral complex name.
    name = 'Copper K-beta'

    # The approximation is as a series of 5 Lorentzians
    # The Lorentzian energies (Table III E_i)
    energies = np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393))
    # The Lorentzian widths (Table III W_i)
    fwhm = np.array((3.52, 3.52, 3.55, 8.08, 5.31))
    # The Lorentzian peak height (Table III I_i)
    peak_heights = np.array((757, 388, 171, 68, 55), dtype=np.float) / 1e3
    # Amplitude of the Lorentzians
    integral_intensity = (0.5 * np.pi * fwhm) * peak_heights
    integral_intensity /= integral_intensity.sum()
    # The energy at the main peak (from table IV beta1,3)
    nominal_peak_energy = 8905.42  # eV


# the API for this is terrible, you have to create a class, you cant just pass in a distirubtion
# it should be changed, and the associated tests failures should be fixed
class MultiLorentzianDistribution_gen(sp.stats.rv_continuous):
    """For producing random variates of the an energy distribution having the form
    of several Lorentzians summed together."""

    # Approximates the random variate defined by multiple Lorentzian components.
    #  @param args  Pass all other parameters to parent class.
    #  @param kwargs  Pass all other parameters to parent class.
    def __init__(self, *args, **kwargs):
        """<args> and <kwargs> are passed on to sp.stats.rv_continuous"""
        sp.stats.rv_continuous.__init__(self, *args, **kwargs)
        self.cumulative_amplitudes = self.distribution.integral_intensity.cumsum()
        self.set_gauss_fwhm = self.distribution.set_gauss_fwhm
        # Reimplements probability distribution function.
        self._pdf = self.distribution.pdf

    def _rvs(self, *args):
        """The CDF and PPF (cumulative distribution and percentile point functions) are hard to
        compute.  But it's easy enough to generate the random variates themselves, so we
        override that method.  Don't call this directly!  Instead call .rvs(), which wraps this."""
        # Choose from among the N Lorentzian lines in proportion to the line amplitudes
        iline = self.cumulative_amplitudes.searchsorted(
            np.random.uniform(0, self.cumulative_amplitudes[-1], size=self._size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = np.random.standard_cauchy(size=self._size) * self.distribution.fwhm[iline] * 0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if self.distribution.gauss_sigma > 0.0:
            lor += np.random.standard_normal(size=self._size) * self.distribution.gauss_sigma
        # Finally, add the line centers.
        return lor + self.distribution.energies[iline]


# Some specific fluorescence lines
# You can see how to make more if you like.
class MnKAlphaDistribution(MultiLorentzianDistribution_gen):
    name = "Mn KAlpha fluorescence"
    distribution = MnKAlpha()

class MnKBetaDistribution(MultiLorentzianDistribution_gen):
    name = "Mn KBeta fluorescence"
    distribution = MnKBeta()

class CuKAlphaDistribution(MultiLorentzianDistribution_gen):
    name = "Cu KAlpha fluorescence"
    distribution = CuKAlpha()

class TiKAlphaDistribution(MultiLorentzianDistribution_gen):
    name = "Ti KAlpha fluorescence"
    distribution = TiKAlpha()

class FeKAlphaDistribution(MultiLorentzianDistribution_gen):
    name = "Fe KAlpha fluorescence"
    distribution = FeKAlpha()



def plot_allMultiLorentzianLineComplexes():
    """ makes a bunch of plots showing the line shape and component parts for the KAlpha and KBeta complexes defined in here,
    intended to nearly replicate plots in papers giving spectral lineshapes"""
    plot_multiLorentzianLineComplex(ScKAlpha)
    plot_multiLorentzianLineComplex(TiKAlpha, instrumentGaussianSigma=0.68 / 2.354)
    plot_multiLorentzianLineComplex(VKAlpha, instrumentGaussianSigma=1.99 / 2.354)  # must include instrument broadening from Table 1, Source to recreate plots
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


def plot_multiLorentzianLineComplex(spectrumDef=CrKAlpha, instrumentGaussianSigma=0):
    """Makes a single plot showing the lineshape and component parts for a SpectalLine object"""
    peak = spectrumDef().peak_energy
    plotEnergies = np.arange(np.round(0.995 * peak), np.round(1.008 * peak), 0.25)

    plt.figure()
    result = np.zeros_like(plotEnergies)
    for energy, fwhm, ampl in zip(spectrumDef.energies, spectrumDef.fwhm, spectrumDef.integral_intensity):
        plt.plot(plotEnergies, ampl * voigt(plotEnergies, energy, hwhm=fwhm * 0.5, sigma=instrumentGaussianSigma), label='%.3f, %.3f, %.3f' % (energy, fwhm, ampl))
        result += ampl * voigt(plotEnergies, energy, hwhm=fwhm * 0.5, sigma=instrumentGaussianSigma)
    plt.plot(plotEnergies, result, label='combined', linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Fit Counts (arb)')
    plt.title(spectrumDef.name)
    plt.legend()
    plt.xlim((plotEnergies[0], plotEnergies[-1]))
    plt.ylim((0, np.max(result)))
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
    e = np.arange(energy_range[0] - 2.5 * resolutions[-1],
                  energy_range[1] + 2.5 * resolutions[-1], stepsize)

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
        smeared_spectrum *= (1 + res * .01)
        plt.plot(e, smeared_spectrum, label="%2d eV" % res, lw=2)

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

        p1 = smeared_spectrum[np.abs(e - epk1) < 2].max()
        if res < 8.12:
            pk2 = smeared_spectrum[np.abs(e - epk2) < 2].max()
            pval = smeared_spectrum[np.abs(e - evalley) < 3].min()
            print("Resolution: %5.2f pk ratio: %.6f   PV ratio: %.6f" % (res, pk2 / p1, pval / pk2))

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
