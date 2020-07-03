# -*- coding: UTF-8 -*-
"""
fluorescence_lines.py

Tools for fitting and simulating X-ray fluorescence lines.
"""

import numpy as np
import scipy as sp
import pylab as plt
from . import line_fits
from . import line_models
from collections import OrderedDict

from mass.mathstat.special import voigt
import logging
LOG = logging.getLogger("mass")

FWHM_OVER_SIGMA = (8 * np.log(2))**0.5


class SpectralLine(sp.stats.rv_continuous):
    """An abstract base class for modeling spectral lines as a sum
    of Voigt profiles (i.e., Gaussian-convolved Lorentzians).

    Call addline to create a new subclass properly.

    The API follows scipy.stats.stats.rv_continuous and is kind of like rv_frozen.
    Calling this object with an argument evalutes the pdf at the argument, it does not
    return an rv_frozen.

    But so far we ony define `rvs` and `pdf`.

    """

    def __init__(self, element, material, linetype, energies, lorentzian_fwhm, intrinsic_sigma,
                 reference_plot_instrument_gaussian_fwhm, reference_short, reference_amplitude, reference_amplitude_type,
                 normalized_lorentzian_integral_intensity, nominal_peak_energy, fitter_type, position_uncertainty,
                 reference_measurement_type, is_default_material):
        """Constructor needs two Gaussian widths (both default to zero):
        `intrinsic_sigma` is the width (sigma) of any 'intrinsic Gaussian', as found (for example) in
            the Fowler et al 2020 metrology shape estimation for the lanthanide L lines. Normally zero.
        """
        self.element = element
        self.material = material
        self.linetype = linetype
        self.energies = energies
        self.lorentzian_fwhm = lorentzian_fwhm
        self.intrinsic_sigma = intrinsic_sigma
        self.reference_plot_instrument_gaussian_fwhm = reference_plot_instrument_gaussian_fwhm
        self.reference_short = reference_short
        self.reference_amplitude = reference_amplitude
        self.reference_amplitude_type = reference_amplitude_type
        self.reference_measurement_type = reference_measurement_type
        self.normalized_lorentzian_integral_intensity = normalized_lorentzian_integral_intensity
        self.nominal_peak_energy = nominal_peak_energy
        self.fitter_type = fitter_type
        self.position_uncertainty = position_uncertainty
        self.reference_measurement_type = reference_measurement_type
        self.is_default_material = is_default_material
        self.has_peak_energy = False
        self.cumulative_amplitudes = self.normalized_lorentzian_integral_intensity.cumsum()

    @property
    def peak_energy(self):
        # lazily calculate peak energy
        if not self.has_peak_energy:
            self._peak_energy = sp.optimize.brent(lambda x: -self.pdf(x, instrument_gaussian_fwhm=0),
                                                  brack=np.array((0.5, 1, 1.5))*self.nominal_peak_energy)
            self.has_peak_energy = True
        return self._peak_energy

    def __call__(self, x, instrument_gaussian_fwhm):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x, instrument_gaussian_fwhm)

    def pdf(self, x, instrument_gaussian_fwhm):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        x = np.asarray(x, dtype=np.float)
        result = np.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.normalized_lorentzian_integral_intensity):
            result += ampl * voigt(x, energy, hwhm=fwhm*0.5, sigma=gaussian_sigma)
            # mass.voigt() is normalized to have unit integrated intensity
        return result

    def components(self, x, instrument_gaussian_fwhm):
        """List of spectrum components as a function of <x>, the energy in eV"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        x = np.asarray(x, dtype=np.float)
        components = []
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.normalized_lorentzian_integral_intensity):
            components.append(ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=gaussian_sigma))
        return components

    def plot(self, x=None, instrument_gaussian_fwhm=0, axis=None, components=True, label=None, setylim=True):
        """Plot the spectrum.
        x - np array of energy in eV to plot at (sensible default)
        axis - axis to plot on (default creates new figure)
        components - True plots each voigt component in addition to the spectrum
        label - a string to label the plot with (optional)"""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        if x is None:
            width = max(2*gaussian_sigma, 3*np.amax(self.lorentzian_fwhm))
            lo = np.amin(self.energies)-width
            hi = np.amax(self.energies)+width
            x = np.linspace(lo, hi, 500)
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if components:
            for component in self.components(x, instrument_gaussian_fwhm):
                axis.plot(x, component, "--")
        pdf = self.pdf(x, instrument_gaussian_fwhm)
        axis.plot(x, pdf, "k", lw=2, label=label)
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Counts per {:.2} eV bin".format(x[1]-x[0]))
        axis.set_xlim(x[0], x[-1])
        if setylim:
            axis.set_ylim(np.amin(pdf)*0.1, np.amax(pdf))
        axis.set_title("{} with resolution {:.2f} eV FWHM".format(self.shortname,
                                                                  instrument_gaussian_fwhm))
        return axis

    def plot_like_reference(self, axis=None):
        axis = self.plot(
            axis=axis, instrument_gaussian_fwhm=self.reference_plot_instrument_gaussian_fwhm)
        return axis

    def rvs(self, size, instrument_gaussian_fwhm):
        """The CDF and PPF (cumulative distribution and percentile point functions) are hard to
        compute.  But it's easy enough to generate the random variates themselves, so we
        override that method.  Don't call this directly!  Instead call .rvs(), which wraps this.
        Takes gaussian_fwhm as a keyword argument."""
        gaussian_sigma = self._gaussian_sigma(instrument_gaussian_fwhm)
        # Choose from among the N Lorentzian lines in proportion to the line amplitudes
        iline = self.cumulative_amplitudes.searchsorted(
            np.random.uniform(0, self.cumulative_amplitudes[-1], size=size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = np.random.standard_cauchy(size=size) * self.lorentzian_fwhm[iline] * 0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if gaussian_sigma > 0.0:
            lor += np.random.standard_normal(size=size) * gaussian_sigma
        # Finally, add the line centers.
        results = lor + self.energies[iline]
        # We must check for non-positive results and replace them by recursive call
        # to self.rvs().
        not_positive = results <= 0.0
        if np.any(not_positive):
            Nbad = not_positive.sum()
            results[not_positive] = self.rvs(
                size=Nbad, instrument_gaussian_fwhm=instrument_gaussian_fwhm)
        return results

    @property
    def shortname(self):
        if self.is_default_material:
            return "{}{}".format(self.element, self.linetype)
        else:
            return "{}{}_{}".format(self.element, self.linetype, self.material)

    @property
    def reference(self):
        return lineshape_references[self.reference_short]

    def _gaussian_sigma(self, instrument_gaussian_fwhm):
        """combined intrinstic_sigma and insturment_gaussian_fwhm in quadrature and return the result
        """
        assert instrument_gaussian_fwhm >= 0
        return ((instrument_gaussian_fwhm/FWHM_OVER_SIGMA)**2 + self.intrinsic_sigma**2)**0.5

    def __repr__(self):
        return "SpectralLine: {}".format(self.shortname)

    def model(self, has_linear_background=True, has_tails=False, prefix=""):
        """Generate a LineModel instance from a SpectralLine"""
        if self.linetype == "KAlpha":
            model_class = line_models.GenericKAlphaModel
        else:
            model_class = line_models.GenericLineModel
        name = self.element+self.linetype
        m = model_class(name=name, spect=self, has_linear_background=has_linear_background,
                        has_tails=has_tails, prefix=prefix)
        return m

    def fitter(self):
        return make_line_fitter(self)

    def minimum_fwhm(self, instrument_gaussian_fwhm):
        """for the narrowest lorentzian in the line model, calculate the combined fwhm including
        the lorentzian, intrinstic_sigma, and instrument_gaussian_fwhm"""
        fwhm2 = np.amin(self.lorentzian_fwhm)**2+instrument_gaussian_fwhm**2 + \
            (self.intrinsic_sigma*FWHM_OVER_SIGMA)**2
        return np.sqrt(fwhm2)

    @classmethod
    def quick_monochromatic_line(cls, name, energy, lorentzian_fwhm, intrinsic_sigma):
        """
        Create a quick monochromatic line. Intended for use in calibration when we know a line energy, but not a lineshape model.
        Returns and instrance of SpectralLine with most fields having contents like "unknown: quick_line". The line will have
        a single lorentzian element with the given energy, fwhm, and intrinsic_sigma values.
        """
        element = name
        material = "unknown: quick_line"
        energies = np.array([energy])
        lorentzian_fwhm = np.array([lorentzian_fwhm])
        intrinsic_sigma = intrinsic_sigma
        linetype = "quick_line"
        reference_plot_instrument_gaussian_fwhm = "unkown: quick_line"
        reference_short = "unkown: quick_line"
        reference_amplitude = "unkown: quick_line"
        reference_amplitude_type = "unkown: quick_line"
        normalized_lorentzian_integral_intensity = np.array([1])
        nominal_peak_energy = energy
        fitter_type = line_fits.GenericKBetaFitter  # dont float dph_de
        position_uncertainty = "unknown: quick_line"
        reference_measurement_type = "unkown: quick_line"
        is_default_material = True
        return cls(element, material, linetype, energies, lorentzian_fwhm, intrinsic_sigma,
                   reference_plot_instrument_gaussian_fwhm, reference_short, reference_amplitude, reference_amplitude_type,
                   normalized_lorentzian_integral_intensity, nominal_peak_energy, fitter_type, position_uncertainty,
                   reference_measurement_type, is_default_material)


lineshape_references = OrderedDict()
lineshape_references["Klauber 1993"] = """Data are from C. Klauber, Applied Surface Science 70/71 (1993) pages 35-39.
    "Magnesium Kalpha X-ray line structure revisited".  Also discussed in more
    detail in C. Klauber, Surface & Interface Analysis 20 (1993), 703-715.

    Klauber offers only an energy shift relative to some absolute standard. For
    an absolute standard, we use the value 1253.687 as the Ka1 peak as found by
    J. Schweppe, R. D. Deslattes, T. Mooney, and C. J. Powell in J. Electron
    Spectroscopy and Related Phenomena 67 (1994) 463-478 titled "Accurate measurement
    of Mg and Al Kalpha_{1,2} X-ray energy profiles". See Table 5 "Average" column.
    """
lineshape_references["Wollman 2000"] = """Data are from Wollman, Nam, Newbury, Hilton, Irwin, Berfren, Deiker, Rudman,
    and Martinis, NIM A 444 (2000) page 145. They come from combining 8 earlier
    references dated 1965 - 1993."""
lineshape_references["Chantler 2006"] = """Chantler, C., Kinnane, M., Su, C.-H., & Kimpton, J. (2006).
    "Characterization of K spectral profiles for vanadium, component redetermination for
    scandium, titanium, chromium, and manganese, and development of satellite structure
    for Z=21 to Z=25." Physical Review A, 73(1), 012508. doi:10.1103/PhysRevA.73.012508
    url: http://link.aps.org/doi/10.1103/PhysRevA.73.012508

    Be sure to look at Table I, not the very similar Table II which lists parameters in different parameterization."""
lineshape_references["Chantler 2013"] = """C Chantler, L Smale, J Kimpton, et al., J Phys B 46, 145601 (2013).
http://iopscience.iop.org/0953-4075/46/14/145601"""
lineshape_references["Chantler 2013, Section 5"] = """C Chantler, L Smale, J Kimpton, et al., J Phys B 46, 145601 (2013).
http://iopscience.iop.org/0953-4075/46/14/145601
We were originally using L Smale, C Chantler, M Kinnane, J Kimpton, et al., Phys
Rev A 87 022512 (2013). http://pra.aps.org/abstract/PRA/v87/i2/e022512

BUT these were adjusted in C Chantler, L Smale, J Kimpton, et al., J Phys B
46, 145601 (2013).  http://iopscience.iop.org/0953-4075/46/14/145601
(see Section 5 "Redefinition of vanadium K-beta standard")  Both papers are
by the same group, of course.
"""
lineshape_references["Hoelzer 1997, NISTfits.ipf"] = """Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code)."""
lineshape_references["Hoelzer 1997"] = """Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December)."""
lineshape_references["Zn Hack"] = """This is a hack, a copy of the Hoelzer, Fritsch, Deutsch, Haertwig, Foerster
    Phys Rev A56 (#6) pages 4554ff (1997 December) model, with the numbers
    adjusted to get line energies of 8615.823, 8638.91 eV and widths 10% wider
    than for Cu. Those widths are based on Zschornack's book.

    The KBeta also appears to be a hack with scaled values."""
lineshape_references["Steve Smith"] = """This is what Steve Smith at NASA GSFC uses for Br K-alpha."""
lineshape_references["Joe Fowler"] = """This is what Joe Fowler measured for tungsten L-lines in 2018."""
lineshape_references["NIST ASD"] = """NIST Atomic Spectra Database
Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2018). NIST Atomic Spectra Database (ver. 5.6.1),
[Online]. Available: https://physics.nist.gov/asd [2018, December 12]. National Institute of Standards and
Technology, Gaithersburg, MD. DOI: https://doi.org/10.18434/T4W30F"""
lineshape_references["Clementson 2010"] = """J. Clementson, P. Beiersdorfer, G. V. Brown, and M. F. Gu,
    "Spectroscopy of M-shell x-ray transitions in Zn-like through Co-like W,"
    Physica Scripta 81, 015301 (2010). https://iopscience.iop.org/article/10.1088/0031-8949/81/01/015301/meta"""
lineshape_references["Steve Smith"] = """This is what Steve Smith at NASA GSFC uses for Br K-alpha."""
lineshape_references["Nilsen 1995"] = """Elliott, S. R., Beiersdorfer, P., Macgowan, B. J., & Nilsen, J. (1995).
Measurements of line overlap for resonant spoiling of x-ray lasing transitions in nickel-like tungsten, 52(4),
2689–2692. https://doi.org/10.1103/PhysRevA.52.2689"""
lineshape_references["Deslattes Notebook Si"] = """Scanned pages from Deslattes/Mooney's notebook provided by Csilla Szabo-Foster.
Added by GCO Oct 7 2019. Used the postion and width values from the from the lowest listed fit, the one in energy units.
Used the intensities from the Second lowest fit, the one labeled PLUS-POSITION SCAN (best-fit Voight profile).
Also the notebook only included the Ka1 and Ka2, not the higher energy satellites, so I made up numbers for the
small feature at higher energy"""
lineshape_references["Schweppe 1992 Al"] = """J. Schweppe, R. D. Deslattes, T. Mooney, and C. J. Powell
in J. Electron Spectroscopy and Related Phenomena 67
(1994) 463-478 titled "Accurate measurement of Mg and Al Kalpha_{1,2} X-ray energy profiles". See Table 5
"Average" column. They do not provide a full lineshape, GCO interperpreted the paper as follows: Ka1 and Ka2
positions are taken from Table 6 "This work" column Ka1 and Ka2 widths were fixed as equal, and we taken the
value 0.43 eV from the 2nd to last paragraph The higher energy satellite features are not measured by
Schweppe, and instead taken from an email from Caroline Kilbourne to Joel Ullom dated 28 Sept 2010 We expect
these higher energy satellites do not affect the fitting of the peak location very much.
"""
lineshape_references["Mendenhall 2019"] = """Marcus H. Mendenhall et al., J. Phys B in press (2019).
    https://doi.org/10.1088/1361-6455/ab45d6"""
lineshape_references["Deslattes Notebook S, Cl, K"] = """Scanned pages from Deslattes/Mooney's notebook
provided by Csilla Szabo-Foster. Added by GCO Oct 30 2019. Used
the postion and width values from the from the lowest listed fit, the one in energy units. Used the
intensities from the Second lowest fit, the one labeled PLUS-POSITION SCAN (best-fit Voight profile). The
detector resolution ("width of Gauss. res. func." in MINUS-POSITON scan) is less than the Gaussian Width
("Gaussian width" in PLUS-POSITON scan) I haven't accounted for that, so our models still don't match
Deslattes. We would need a gaussian_atomic_physics component added to our models. Also the notebook only
included the Ka1 and Ka2, not the higher energy satellites, so I made up numbers for the small feature at
higher energy or estimated them from data in Mauron, O., Dousse, J. C., Hoszowska, J., Marques, J. P.,
Parente, F., & Polasik, M. (2000). L-shell shake processes resulting from 1s photoionization in elements
11≤Z≤17. Physical Review A - Atomic, Molecular, and Optical Physics, 62(6), 062508–062501.
https://doi.org/10.1103/PhysRevA.62.062508"""
lineshape_references["Ravel 2018"] = """Bruce Ravel et al., Phys. Rev. B 97 (2018) 125139
    https://doi.org/10.1103/PhysRevB.97.125139"""

spectra = OrderedDict()
spectrum_classes = OrderedDict()  # for backwards compatability

LORENTZIAN_PEAK_HEIGHT = 999
LORENTZIAN_INTEGRAL_INTENSITY = 9999
VOIGT_PEAK_HEIGHT = 99999


def addline(element, linetype, material, reference_short, reference_plot_instrument_gaussian_fwhm,
            nominal_peak_energy, energies, lorentzian_fwhm, reference_amplitude,
            reference_amplitude_type, ka12_energy_diff=None, fitter_type=None,
            position_uncertainty=np.NaN, intrinsic_sigma=0, reference_measurement_type=None, is_default_material=True):

    # require exactly one method of specifying the amplitude of each component
    assert reference_amplitude_type in [LORENTZIAN_PEAK_HEIGHT,
                                        LORENTZIAN_INTEGRAL_INTENSITY, VOIGT_PEAK_HEIGHT]
    # require the reference exists in lineshape_references
    assert reference_short in lineshape_references

    # require kalpha lines to have ka12_energy_diff
    if linetype.startswith("KAlpha"):
        ka12_energy_diff = float(ka12_energy_diff)
    # require reference_plot_instrument_gaussian_fwhm to be a float or None
    assert reference_plot_instrument_gaussian_fwhm is None or isinstance(
        reference_plot_instrument_gaussian_fwhm, float)

    # calculate normalized lorentzian_integral_intensity
    if reference_amplitude_type == VOIGT_PEAK_HEIGHT:
        reference_instrument_gaussian_sigma = reference_plot_instrument_gaussian_fwhm/FWHM_OVER_SIGMA
        lorentzian_integral_intensity = [ph/voigt(0, 0, lw/2.0, reference_instrument_gaussian_sigma)
                                         for ph, lw in zip(reference_amplitude, lorentzian_fwhm)]
    elif reference_amplitude_type == LORENTZIAN_PEAK_HEIGHT:
        lorentzian_integral_intensity = (
            0.5 * np.pi * lorentzian_fwhm) * np.array(reference_amplitude)
    elif reference_amplitude_type == LORENTZIAN_INTEGRAL_INTENSITY is not None:
        lorentzian_integral_intensity = reference_amplitude
    normalized_lorentzian_integral_intensity = np.array(lorentzian_integral_intensity) / \
        float(np.sum(lorentzian_integral_intensity))

    line = SpectralLine(
        element=element,
        material=material,
        linetype=linetype,
        energies=np.array(energies),
        lorentzian_fwhm=np.array(lorentzian_fwhm),
        intrinsic_sigma=intrinsic_sigma,
        reference_plot_instrument_gaussian_fwhm=reference_plot_instrument_gaussian_fwhm,
        reference_short=reference_short,
        reference_amplitude=reference_amplitude,
        reference_amplitude_type=reference_amplitude_type,
        normalized_lorentzian_integral_intensity=np.array(normalized_lorentzian_integral_intensity),
        nominal_peak_energy=float(nominal_peak_energy),
        fitter_type=fitter_type,
        position_uncertainty=float(position_uncertainty),
        reference_measurement_type=reference_measurement_type,
        is_default_material=is_default_material,
    )
    if linetype.startswith("KAlpha"):
        line.ka12_energy_diff = ka12_energy_diff
    name = line.shortname
    if name in spectra.keys():
        raise ValueError("spectrum {} already exists".format(name))

    # Add this SpectralLine to spectra dict AND make it be a variable in the module
    spectra[name] = line
    spectrum_classes[name] = lambda: line
    globals()[name] = line

    def mlf():
        return make_line_fitter(line)
    globals()[name+"Fitter"] = mlf
    return line


def make_line_fitter(line):
    """Generate a LineFitter instance from a SpectralLine (deprecated)"""
    if line.fitter_type is not None:
        fitter_class = line.fitter_type
    elif line.linetype == "KAlpha" or line.linetype == "LAlpha":
        if line.element in ["Al", "Mg"]:
            fitter_class = line_fits._lowZ_KAlphaFitter
        else:
            fitter_class = line_fits.GenericKAlphaFitter
    elif line.linetype.startswith("KBeta") or "LBeta" in line.linetype:
        fitter_class = line_fits.GenericKBetaFitter
    else:
        raise ValueError("no generic fitter for {}".format(line))
    f = fitter_class()
    f.spect = line
    f.name = line.element+line.linetype
    return f


addline(
    element="Mg",
    material="metal",
    linetype="KAlpha",
    reference_short="Klauber 1993",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1253.687,
    energies=np.array((-.265, 0, 4.740, 8.210, 8.487, 10.095, 17.404, 20.430)) + 1253.687,
    lorentzian_fwhm=np.array((.541, .541, 1.1056, .6264, .7349, 1.0007, 1.4311, .8656)),
    reference_amplitude=np.array((0.5, 1, .02099, .07868, .04712, .09071, .01129, .00538)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=2.2,
)

addline(
    element="Al",
    material="metal",
    linetype="KAlpha",
    reference_short="Schweppe 1992 Al",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1486.88931733,
    energies=np.array((1486.706, 1486.293, 1492.3, 1496.4, 1498.4)),
    lorentzian_fwhm=np.array((0.43, 0.43, 1.34, 0.96, 1.255)),
    reference_amplitude=np.array((1, .5, .02, .12, .06)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=3.0,
    position_uncertainty=0.010,
)

addline(
    element="Al",
    material="AlO",
    linetype="KAlpha",
    reference_short="Wollman 2000",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=1486.930456,
    energies=np.array((1486.94, 1486.52, 1492.94, 1496.85, 1498.70, 1507.4, 1510.9)),
    lorentzian_fwhm=np.array((0.43, 0.43, 1.34, 0.96, 1.25, 1.5, 0.9)),
    reference_amplitude=np.array((1.0, 0.5, 0.033, 0.12, 0.11, 0.07, 0.05)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=3.,
    is_default_material=False
)

addline(
    element="Si",
    material="Si crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook Si",
    reference_plot_instrument_gaussian_fwhm=0.245,
    nominal_peak_energy=1739.986,
    energies=np.array((1739.39, 1739.986, 1752.0)),
    lorentzian_fwhm=np.array((0.539, 0.524, 5)),
    reference_amplitude=np.array((3.134e2, 6.121e3, 8e2)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=.6,
    position_uncertainty=0.040
)

addline(
    element="S",
    material="MoS2 spray",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.2414,
    nominal_peak_energy=2307.89,
    energies=np.array((2307.89, 2306.70)),
    lorentzian_fwhm=np.array((0.769, 0.722)),
    reference_amplitude=np.array((0.11951e5, 0.61114e4)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=1.19,
    position_uncertainty=0.040
)

addline(
    element="Cl",
    material="KCl crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.266,
    nominal_peak_energy=2622.44,
    energies=np.array((2622.44, 2620.85, 2640)),
    lorentzian_fwhm=np.array((0.925, 0.945, 5)),
    reference_amplitude=np.array((0.15153e5, 0.82429e4, 0.15153e5/8.0)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=1.6,
    position_uncertainty=0.040
)

addline(
    element="K",
    material="KCl crystal",
    linetype="KAlpha",
    reference_short="Deslattes Notebook S, Cl, K",
    reference_plot_instrument_gaussian_fwhm=0.0896,
    nominal_peak_energy=3313.93,
    energies=np.array((3313.93, 3311.17)),
    lorentzian_fwhm=np.array((0.948, 0.939)),
    reference_amplitude=np.array((0.15153e5, 0.82429e4)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
    ka12_energy_diff=2.75,
    position_uncertainty=0.020
)

addline(
    element="Sc",
    material="metal",
    linetype="KAlpha",
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=0.52,
    nominal_peak_energy=4090.735,
    energies=np.array((4090.745, 4089.452, 4087.782, 4093.547, 4085.941, 4083.976)),  # Table I C_i
    lorentzian_fwhm=np.array((1.17, 2.65, 1.41, 2.09, 1.53, 3.49)),  # Table I W_i
    reference_amplitude=np.array((8175, 878, 232, 287, 4290, 119)),  # Table I A_i
    reference_amplitude_type=VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=5.1,
)

addline(
    # The paper has two sets of TiKAlpha data, I used the set Refit of [21] Kawai et al 1994
    element="Ti",
    material="metal",
    linetype="KAlpha",
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=0.11,
    nominal_peak_energy=4510.903,
    energies=np.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088)),  # Table I C_i
    lorentzian_fwhm=np.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49)),  # Table I W_i
    reference_amplitude=np.array((4549, 626, 236, 143, 2034, 54)),  # Table I A_i
    reference_amplitude_type=VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=6.0,
)

addline(
    element="Ti",
    material="metal",
    linetype="KBeta",
    reference_short="Chantler 2013",
    reference_plot_instrument_gaussian_fwhm=1.244*2.3548,
    nominal_peak_energy=4931.966,
    energies=np.array((25.37, 30.096, 31.967, 35.59)) + 4900,
    lorentzian_fwhm=np.array((16.3, 4.25, 0.42, 0.47)),
    reference_amplitude=np.array((199, 455, 326, 19.2)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)

addline(
    element="V",
    material="metal",
    linetype="KAlpha",
    reference_short="Chantler 2006",
    reference_plot_instrument_gaussian_fwhm=1.99,  # Table I, other parameters
    nominal_peak_energy=4952.216,
    energies=np.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014)),  # Table I C_i
    lorentzian_fwhm=np.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09)),  # Table I W_i
    reference_amplitude=np.array((25832, 5410, 1536, 956, 12971, 603)),  # Table I A_i
    reference_amplitude_type=VOIGT_PEAK_HEIGHT,
    ka12_energy_diff=7.5,
)

addline(
    element="V",
    material="metal",
    linetype="KBeta",
    reference_short="Chantler 2013, Section 5",
    reference_plot_instrument_gaussian_fwhm=0.805*2.3548,
    nominal_peak_energy=5426.956,
    energies=np.array((18.19, 24.50, 26.992)) + 5400,
    lorentzian_fwhm=np.array((18.86, 5.48, 2.499)),
    reference_amplitude=np.array((258, 236, 507)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)

addline(
    element="Cr",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5414.81,
    energies=5400 + np.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986)),
    lorentzian_fwhm=np.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.740)),
    reference_amplitude=np.array((822, 237, 85, 45, 15, 386, 36)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=9.2,
)

addline(
    element="Cr",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5946.82,
    energies=5900 + np.array((47.00, 35.31, 46.24, 42.04, 44.93)),  # Table III E_i
    lorentzian_fwhm=np.array((1.70, 15.98, 1.90, 6.69, 3.37)),  # Table III W_i
    reference_amplitude=np.array((670, 55, 337, 82, 151)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Mn",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=5898.802,
    energies=5800 + np.array((98.853, 97.867, 94.829, 96.532, 99.417, 102.712, 87.743, 86.495)),
    lorentzian_fwhm=np.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216)),
    reference_amplitude=np.array((790, 264, 68, 96, 71, 10, 372, 100)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=11.1,
)

addline(
    element="Mn",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6490.18,
    energies=6400 + np.array((90.89, 86.31, 77.73, 90.06, 88.83)),  # Table III E_i
    lorentzian_fwhm=np.array((1.83, 9.40, 13.22, 1.81, 2.81)),  # Table III W_i
    reference_amplitude=np.array((608, 109, 77, 397, 176)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Fe",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6404.01,
    energies=np.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275)),
    lorentzian_fwhm=np.array((1.613, 1.965, 4.833, 2.803, 2.487, 4.339, 2.57)),
    reference_amplitude=np.array((697, 376, 88, 136, 339, 60, 102)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=13.0,
)
# ERROR IN HOLZER PAPER:
# The FWHM in the Table II of Holzer have the Kalpha_22 and _23 widths as 2.339 and 4.433, but
# these disagree with their Figure 1c. Swapping the widths (as you see above) makes the curve
# match the figure, though the exact I_int values still don't match those in Table II.
# To get the I_int values close, we choose width 4.339 and 2.57. These are still just a guess,
# but that's where the above values come from. See conversation with Richard Gnewkow of the
# Technische Universität Berlin on April 23-24, 2020.

addline(
    element="Fe",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7058.18,
    energies=np.array((7046.90, 7057.21, 7058.36, 7054.75)),  # Table III E_i
    lorentzian_fwhm=np.array((14.17, 3.12, 1.97, 6.38)),  # Table III W_i
    reference_amplitude=np.array((107, 448, 615, 141)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Co",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=6930.38,
    energies=np.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078)),
    lorentzian_fwhm=np.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463)),
    reference_amplitude=np.array((809, 205, 107, 41, 314, 131, 43)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=15.0,
)
# Notice that Co KAlpha in Holzer shows the 4th line as having integrated intensity of 0.088, but
# this is probably a typo (should read 0.008). No effect on the data above, though, because it's a
# derived quantity.

addline(
    element="Co",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7649.45,
    energies=np.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13)),  # Table III E_i
    lorentzian_fwhm=np.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79)),  # Table III W_i
    reference_amplitude=np.array((798, 286, 85, 114, 33, 35)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Ni",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997, NISTfits.ipf",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=7478.26,
    energies=np.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029)),
    lorentzian_fwhm=np.array((2.013, 4.711, 2.674, 3.039, 4.476)),
    reference_amplitude=np.array((909, 136, 351, 79, 24)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=17.2,
)

addline(
    element="Ni",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8264.78,
    energies=np.array((8265.01, 8263.01, 8256.67, 8268.70)),  # Table III E_i
    lorentzian_fwhm=np.array((3.76, 4.34, 13.70, 5.18)),  # Table III W_i
    reference_amplitude=np.array((722, 358, 89, 104)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Cu",
    material="metal",
    linetype="KAlpha",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8047.83,
    energies=np.array((8047.8372, 8045.3672, 8027.9935, 8026.5041)),
    lorentzian_fwhm=np.array((2.285, 3.358, 2.667, 3.571)),
    reference_amplitude=np.array((957, 90, 334, 111)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=20.0,
)

addline(
    element="Cu",
    material="metal",
    linetype="KBeta",
    reference_short="Hoelzer 1997",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8905.42,
    energies=np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393)),  # Table III E_i
    lorentzian_fwhm=np.array((3.52, 3.52, 3.55, 8.08, 5.31)),  # Table III W_i
    reference_amplitude=np.array((757, 388, 171, 68, 55)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Zn",
    material="metal",
    linetype="KAlpha",
    reference_short="Zn Hack",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8638.91,
    energies=[8638.8872, 8636.4172, 8615.9835, 8614.4941],
    lorentzian_fwhm=np.array((2.285, 3.358, 2.667, 3.571)) * 1.1,
    reference_amplitude=np.array((957, 90, 334, 111)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=23.0,
)

addline(
    element="Zn",
    material="metal",
    linetype="KBeta",
    reference_short="Zn Hack",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9572.03,
    energies=np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393))*1.06 + 133.85,
    lorentzian_fwhm=np.array((3.52, 3.52, 3.55, 8.08, 5.31))*1.06,
    reference_amplitude=np.array((757, 388, 171, 68, 55)),  # Table III I_I
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addline(
    element="Br",
    material="metal",
    linetype="KAlpha",
    reference_short="Steve Smith",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=11924.36,
    energies=np.array((11924.2, 11877.6)),
    lorentzian_fwhm=np.array((3.60, 3.73)),
    reference_amplitude=np.array((2, 1)),
    ka12_energy_diff=46.6,
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="W",
    material="metal",
    linetype="LAlpha",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=8398.24,
    energies=np.array((8335.34, 8398.24)),
    lorentzian_fwhm=np.array((6.97, 7.01)),
    reference_amplitude=np.array((.1020, .8980)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="W",
    material="metal",
    linetype="LBeta1",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9672.58,
    energies=np.array((9672.58,)),
    lorentzian_fwhm=np.array((7.71,)),
    reference_amplitude=np.array((1,)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="W",
    material="metal",
    linetype="LBeta2",
    reference_short="Joe Fowler",
    reference_plot_instrument_gaussian_fwhm=None,
    nominal_peak_energy=9964.13,
    energies=np.array((9950.82, 9962.93, 9967.53))+1,
    lorentzian_fwhm=np.array((9.16, 9.82, 9.90)),
    reference_amplitude=np.array((.0847, .7726, .1426)),
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)


addline(
    element="Nb",
    linetype="KBeta",
    material="Nb2O5",
    reference_short="Ravel 2018",
    reference_plot_instrument_gaussian_fwhm=1.2,
    nominal_peak_energy=18625.4,
    energies=np.array((18625.4, 18609.9)),
    lorentzian_fwhm=np.array((6.7, 6.7)),
    reference_amplitude=np.array((1, 0.5)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Nb",
    linetype="KBeta24",
    material="Nb2O5",
    reference_short="Ravel 2018",
    reference_plot_instrument_gaussian_fwhm=1.2,
    nominal_peak_energy=18952.79,
    energies=np.array((18952.79, 18968.0, 18982.7)),
    lorentzian_fwhm=np.array((8.67, 1.9, 5.2)),
    reference_amplitude=np.array((14.07, 0.066, 0.359)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Mo",
    material="metal",
    linetype="KAlpha",
    reference_short="Mendenhall 2019",
    reference_plot_instrument_gaussian_fwhm=0.02,
    nominal_peak_energy=17479.389,
    energies=np.array((17479.389, 17374.577)),
    lorentzian_fwhm=np.array((6.389, 6.3876)),
    reference_amplitude=np.array((3331.119, 1684.988)),
    ka12_energy_diff=104.812,
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)


addline(
    element="Mo",
    material="metal",
    linetype="KBeta",
    reference_short="Mendenhall 2019",
    reference_plot_instrument_gaussian_fwhm=0.02,
    nominal_peak_energy=19606.734,
    energies=np.array((19606.733, 19589.251, 19623.217)),
    lorentzian_fwhm=np.array((6.88, 6.88, 6.88)),
    reference_amplitude=np.array((958.08, 488.67, 29.14)),
    reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)


def plot_all_spectra(maxplots=10):
    """Makes plots showing the line shape and component parts for some lines.
    Intended to replicate plots in the literature giving spectral lineshapes."""
    keys = list(spectra.keys())[:maxplots]
    for name in keys:
        spectrum = spectra[name]()
        spectrum.plot_like_reference()
