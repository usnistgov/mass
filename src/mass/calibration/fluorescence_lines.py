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

import numpy as np
import scipy as sp
import pylab as plt
import palettable
import line_fits
from cycler import cycler
from collections import OrderedDict

from mass.mathstat.special import voigt
import logging
LOG = logging.getLogger("mass")

FWHM_OVER_SIGMA = (8 * np.log(2))**0.5

class SpectralLine(sp.stats.rv_continuous):
    """An abstract base class for modeling spectral lines as a sum
    of Voigt profiles (i.e., Gaussian-convolved Lorentzians).

    Call addfitter to create a new subclass properly.

    This subclasses scipy.stats.stats.rv_continuous, but acts more like an rv_frozen.
    Calling this object with an argument evalutes the pdf at the argument, it does not
    return an rv_frozen.
    """

    def __init__(self, pdf_gaussian_fwhm=0.0):
        """Set up a default Gaussian smearing of 0"""
        self.pdf_gaussian_fwhm = pdf_gaussian_fwhm
        self.peak_energy = sp.optimize.brent(lambda x: -self.pdf(x),
                                             brack=np.array((0.5, 1, 1.5))*self.nominal_peak_energy)
        # make it a subclassing of rv_continuous work
        sp.stats.rv_continuous.__init__(self)
        self.cumulative_amplitudes = self.normalized_lorentzian_integral_intensity.cumsum()
        self._pdf = self.pdf

    def __call__(self, x):
        """Make the class callable, returning the same value as the self.pdf method."""
        return self.pdf(x)

    def pdf(self, x):
        """Spectrum (arb units) as a function of <x>, the energy in eV"""
        x = np.asarray(x, dtype=np.float)
        result = np.zeros_like(x)
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.normalized_lorentzian_integral_intensity):
            result += ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=self.gaussian_sigma)
            # Note that voigt is normalized to have unit integrated intensity
        return result

    def components(self, x):
        """List of spectrum components as a function of <x>, the energy in eV"""
        x = np.asarray(x, dtype=np.float)
        components = []
        for energy, fwhm, ampl in zip(self.energies, self.lorentzian_fwhm,
                                      self.normalized_lorentzian_integral_intensity):
            components.append(ampl * voigt(x, energy, hwhm=fwhm * 0.5, sigma=self.gaussian_sigma))
        return components

    def plot(self,x=None,axis=None,components=True,label=None,setylim=True):
        """Plot the spectrum.
        x - np array of energy in eV to plot at (sensible default)
        axis - axis to plot on (default creates new figure)
        components - True plots each voigt component in addition to the spectrum
        label - a string to label the plot with (optional)"""
        if x is None:
            width = 3*np.amax(self.lorentzian_fwhm)
            lo = np.amin(self.energies)-width
            hi = np.amax(self.energies)+width
            x = np.arange(lo,hi,0.1)
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if components:
            for component in self.components(x):
                axis.plot(x,component,"--")
        pdf = self.pdf(x)
        axis.plot(x, self.pdf(x),"k",lw=2, label=label)
        axis.set_xlabel("energy (eV)")
        axis.set_ylabel("counts (arb)")
        axis.set_xlim(x[0],x[-1])
        if setylim:
            axis.set_ylim(np.amin(pdf)*0.1,np.amax(pdf))
        axis.set_title("{} with resolution {:.2f} eV FWHM".format(self.shortname,self.pdf_gaussian_fwhm))
        return axis

    def plot_like_reference(self,axis=None):
        lastresolution = self.pdf_gaussian_fwhm
        if self.reference_plot_gaussian_fwhm is not None:
            self.pdf_gaussian_fwhm=self.reference_plot_gaussian_fwhm
        axis = self.plot(axis)
        self.pdf_gaussian_fwhm=lastresolution
        return axis

    def _rvs(self, *args, **kwargs):
        """The CDF and PPF (cumulative distribution and percentile point functions) are hard to
        compute.  But it's easy enough to generate the random variates themselves, so we
        override that method.  Don't call this directly!  Instead call .rvs(), which wraps this.
        Takes gaussian_fwhm as a keyword argument."""
        # Choose from among the N Lorentzian lines in proportion to the line amplitudes
        iline = self.cumulative_amplitudes.searchsorted(
            np.random.uniform(0, self.cumulative_amplitudes[-1], size=self._size))
        # Choose Lorentzian variates of the appropriate width (but centered on 0)
        lor = np.random.standard_cauchy(size=self._size) * self.lorentzian_fwhm[iline] * 0.5
        # If necessary, add a Gaussian variate to mimic finite resolution
        if self.gaussian_sigma> 0.0:
            lor += np.random.standard_normal(size=self._size) * self.gaussian_sigma
        # Finally, add the line centers.
        results = lor + self.energies[iline]
        # We must check for non-positive results and replace them by recursive call
        # to self.rvs().
        not_positive = results <= 0.0
        if np.any(not_positive):
            Nbad = not_positive.sum()
            results[not_positive] = self.rvs(size=Nbad)
        return results

    @property
    def gaussian_sigma(self):
        return self.pdf_gaussian_fwhm/FWHM_OVER_SIGMA

    @property
    def shortname(self):
        return self.element+self.linetype

    def set_gauss_fwhm(self,fwhm):
        self.pdf_gaussian_fwhm = fwhm

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
lineshape_references["Ullom Email 2010"]="""Data are from Joel Ullom, based on email to him from Caroline Kilbourne (NASA
    GSFC) dated 28 Sept 2010."""
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
lineshape_references["Chantler 2013, Section 5"]="""We were using L Smale, C Chantler, M Kinnane, J Kimpton, et al., Phys
    Rev A 87 022512 (2013). http://pra.aps.org/abstract/PRA/v87/i2/e022512

    BUT these were adjusted in C Chantler, L Smale, J Kimpton, et al., J Phys B
    46, 145601 (2013).  http://iopscience.iop.org/0953-4075/46/14/145601
    (see Section 5 "Redefinition of vanadium K-beta standard")  Both papers are
    by the same group, of course.
    """
lineshape_references["Hoelzer 1997, NISTfits.ipf"]="""Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December), ***as corrected***
    by someone at LANL: see 11/30/2004 corrections in NISTfits.ipf (Igor code)."""
lineshape_references["Hoelzer 1997"]="""Data are from Hoelzer, Fritsch, Deutsch, Haertwig, Foerster in
    Phys Rev A56 (#6) pages 4554ff (1997 December)."""
lineshape_references["Zn Hack"]="""This is a hack, a copy of the Hoelzer, Fritsch, Deutsch, Haertwig, Foerster
    Phys Rev A56 (#6) pages 4554ff (1997 December) model, with the numbers
    adjusted to get line energies of 8615.823, 8638.91 eV and widths 10% wider
    than for Cu. Those widths are based on Zschornack's book.

    The KBeta also appears to be a hack with scaled values."""

spectrum_classes = OrderedDict()
fitter_classes = OrderedDict()

LORENTZIAN_PEAK_HEIGHT = 999
LORENTZIAN_INTEGRAL_INTENSITY = 9999
VOIGT_PEAK_HEIGHT = 99999

def addfitter(element, linetype, reference_short, reference_plot_gaussian_fwhm,
    nominal_peak_energy, energies, lorentzian_fwhm, reference_amplitude, reference_amplitude_type,
    ka12_energy_diff=None,):

    # require exactly one method of specifying the amplitude of each component
    assert reference_amplitude_type in [LORENTZIAN_PEAK_HEIGHT, LORENTZIAN_INTEGRAL_INTENSITY, VOIGT_PEAK_HEIGHT]
    # require the reference exists in lineshape_references
    assert lineshape_references.has_key(reference_short)
    # require that linetype is supported
    assert linetype in ["KBeta","KAlpha"]
    # require kalpha lines to have ka12_energy_diff
    if linetype == "KAlpha":
        ka12_energy_diff = float(ka12_energy_diff)
    # require reference_plot_gaussian_fwhm to be a float or None
    assert reference_plot_gaussian_fwhm is None or isinstance(reference_plot_gaussian_fwhm,float)


    # calculate normalized lorentzian_integral_intensity
    if reference_amplitude_type == VOIGT_PEAK_HEIGHT:
        reference_instrument_gaussian_sigma = reference_plot_gaussian_fwhm/FWHM_OVER_SIGMA
        lorentzian_integral_intensity = [ph/voigt(0,0,lw/2.0,reference_instrument_gaussian_sigma)
                                         for ph,lw in zip(reference_amplitude,lorentzian_fwhm)]
    elif reference_amplitude_type == LORENTZIAN_PEAK_HEIGHT:
        lorentzian_integral_intensity = (0.5 * np.pi * lorentzian_fwhm) * np.array(reference_amplitude)
    elif reference_amplitude_type == LORENTZIAN_INTEGRAL_INTENSITY is not None:
        lorentzian_integral_intensity = reference_amplitude
    normalized_lorentzian_integral_intensity = np.array(lorentzian_integral_intensity)/float(np.sum(lorentzian_integral_intensity))

    dict = {
    "element":element,
    "linetype":linetype,
    "energies":np.array(energies),
    "lorentzian_fwhm":np.array(lorentzian_fwhm),
    "reference_plot_gaussian_fwhm":reference_plot_gaussian_fwhm,
    "reference_short":reference_short,
    "reference_amplitude":reference_amplitude,
    "reference_amplitude_type":reference_amplitude_type,
    "normalized_lorentzian_integral_intensity":np.array(normalized_lorentzian_integral_intensity),
    "nominal_peak_energy":float(nominal_peak_energy)
    }
    if linetype == "KAlpha":
        dict["ka12_energy_diff"] = ka12_energy_diff
    classname = element+linetype
    cls = type(classname, (SpectralLine,), dict)

    ### The above is nearly equivalent to the below
    ### but the below doesn't errors because it doesn't like the use of the same
    ### name in both the class and the function arguments
    ### eg energies and energies
    # class cls(SpectralLine):
    #     __name__ = element+linetype
    #     energies = np.array(energies)
    #     lorentzian_fwhm = np.array(lorentzian_fwhm)
    #     reference_plot_gaussian_fwhm = float(reference_plot_gaussian_fwhm)
    #     reference_short = reference_short
    #     normalized_lorentzian_integral_intensity = np.array(normalized_lorentzian_integral_intensity)
    #     nominal_peak_energy = float(nominal_peak_energy)


    # add fitter to spectrum_classes dict
    spectrum_classes[cls.__name__]=cls
    # make the fitter be a variable in the module
    globals()[cls.__name__]=cls
    # create fitter as well
    spectrum = cls()
    if spectrum.element in ["Al","Mg"]:
        superclass = line_fits._lowZ_KAlphaFitter
    elif spectrum.linetype == "KAlpha":
        superclass = line_fits.GenericKAlphaFitter
    elif spectrum.linetype == "KBeta":
        superclass = line_fits.GenericKBetaFitter
    else:
        raise ValueError("no generic fitter for {}".format(spectrum))
    dict = {"spect":spectrum}
    fitter_class = type(cls.__name__+"Fitter",(superclass,),dict)
    globals()[cls.__name__+"Fitter"] = fitter_class
    fitter_classes[cls.__name__] = fitter_class

    return cls

addfitter(
element="Mg",
linetype="KAlpha",
reference_short = "Klauber 1993",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=1253.687,
energies = np.array((-.265, 0, 4.740, 8.210, 8.487, 10.095, 17.404, 20.430)) + 1253.687,
lorentzian_fwhm = np.array((.541, .541, 1.1056, .6264, .7349, 1.0007, 1.4311, .8656)),
reference_amplitude=np.array((0.5, 1, .02099, .07868, .04712, .09071, .01129, .00538)),
reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
ka12_energy_diff = 2.2,
)

addfitter(
element="Al",
linetype="KAlpha",
reference_short = "Ullom Email 2010",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=1486.88931733,
energies = np.array((1486.9, 1486.5, 1492.3, 1496.4, 1498.4)),
lorentzian_fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.255)),
reference_amplitude=np.array((1, .5, .02, .12, .06)),
reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
ka12_energy_diff = 3.0,
)

addfitter(
element="AlOx",
linetype="KAlpha",
reference_short = "Wollman 2000",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=1486.930456,
energies = np.array((1486.94, 1486.52, 1492.94, 1496.85, 1498.70, 1507.4, 1510.9)),
lorentzian_fwhm = np.array((0.43, 0.43, 1.34, 0.96, 1.25, 1.5, 0.9)),
reference_amplitude=np.array((1.0, 0.5, 0.033, 0.12, 0.11, 0.07, 0.05)),
reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
ka12_energy_diff = 3.,
)

addfitter(
element="Sc",
linetype="KAlpha",
reference_short = "Chantler 2006",
reference_plot_gaussian_fwhm=0.52,
nominal_peak_energy=4090.735,
energies = np.array((4090.745, 4089.452, 4087.782, 4093.547, 4085.941, 4083.976)),#Table I C_i
lorentzian_fwhm = np.array((1.17, 2.65, 1.41, 2.09, 1.53, 3.49)),#Table I W_i
reference_amplitude=np.array((8175, 878, 232, 287, 4290, 119)),#Table I A_i
reference_amplitude_type=VOIGT_PEAK_HEIGHT,
ka12_energy_diff = 5.1,
)

addfitter(
element="Ti",# the paper has two sets of TiKAlpha data, I used the set Refit of [21] Kawai et al 1994
linetype="KAlpha",
reference_short = "Chantler 2006",
reference_plot_gaussian_fwhm=0.11,
nominal_peak_energy=4510.903,
energies = np.array((4510.918, 4509.954, 4507.763, 4514.002, 4504.910, 4503.088)),#Table I C_i
lorentzian_fwhm = np.array((1.37, 2.22, 3.75, 1.70, 1.88, 4.49)),#Table I W_i
reference_amplitude=np.array((4549, 626, 236, 143, 2034, 54)),#Table I A_i
reference_amplitude_type=VOIGT_PEAK_HEIGHT,
ka12_energy_diff = 6.0,
)

addfitter(
element="Ti",
linetype="KBeta",
reference_short = "Chantler 2013",
reference_plot_gaussian_fwhm=1.244,
nominal_peak_energy=4931.966,
energies = np.array((25.37, 30.096, 31.967, 35.59)) + 4900,
lorentzian_fwhm = np.array((16.3, 4.25, 0.42, 0.47)),
reference_amplitude=np.array((199, 455, 326, 19.2)),
reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)

addfitter(
element="V",
linetype="KAlpha",
reference_short = "Chantler 2006",
reference_plot_gaussian_fwhm=1.99,#Table I, other parameters
nominal_peak_energy=4952.216,
energies = np.array((4952.237, 4950.656, 4948.266, 4955.269, 4944.672, 4943.014)),#Table I C_i
lorentzian_fwhm = np.array((1.45, 2.00, 1.81, 1.76, 2.94, 3.09)),#Table I W_i
reference_amplitude=np.array((25832, 5410, 1536, 956, 12971, 603)),#Table I A_i
reference_amplitude_type=VOIGT_PEAK_HEIGHT,
ka12_energy_diff = 7.5,
)

addfitter(
element="V",
linetype="KBeta",
reference_short = "Chantler 2013, Section 5",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=5426.956,
energies = np.array((25.37, 30.096, 31.967, 35.59)) + 4900,
lorentzian_fwhm = np.array((18.86, 5.48, 2.499)),
reference_amplitude=np.array((258, 236, 507)),
reference_amplitude_type=LORENTZIAN_INTEGRAL_INTENSITY,
)

addfitter(
element="Cr",
linetype="KAlpha",
reference_short = "Hoelzer 1997, NISTfits.ipf",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=5414.81 ,
energies = 5400 + np.array((14.874, 14.099, 12.745, 10.583, 18.304, 5.551, 3.986)),
lorentzian_fwhm = np.array((1.457, 1.760, 3.138, 5.149, 1.988, 2.224, 4.4740)),
reference_amplitude=np.array((882, 237, 85, 45, 15, 386, 36)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 9.2,
)

addfitter(
element="Cr",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=5946.82,
energies = 5900 + np.array((47.00, 35.31, 46.24, 42.04, 44.93)),#Table III E_i
lorentzian_fwhm = np.array((1.70, 15.98, 1.90, 6.69, 3.37)),#Table III W_i
reference_amplitude=np.array((670, 55, 337, 82, 151)),#Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Mn",
linetype="KAlpha",
reference_short = "Hoelzer 1997, NISTfits.ipf",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=5898.802 ,
energies = 5800 + np.array((98.853, 97.867, 94.829, 96.532, 99.417, 102.712, 87.743, 86.495)),
lorentzian_fwhm = np.array((1.715, 2.043, 4.499, 2.663, 0.969, 1.553, 2.361, 4.216)),
reference_amplitude=np.array((790, 264, 68, 96, 71, 10, 372, 100)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 11.1,
)

addfitter(
element="Mn",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=6490.18,
energies = 6400 + np.array((90.89, 86.31, 77.73, 90.06, 88.83)),#Table III E_i
lorentzian_fwhm = np.array((1.83, 9.40, 13.22, 1.81, 2.81)),#Table III W_i
reference_amplitude=np.array((608, 109, 77, 397, 176)),#Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Fe",
linetype="KAlpha",
reference_short = "Hoelzer 1997, NISTfits.ipf",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=6404.01 ,
energies = np.array((6404.148, 6403.295, 6400.653, 6402.077, 6391.190, 6389.106, 6390.275)),
lorentzian_fwhm = np.array((1.613, 1.965, 4.833, 2.803, 2.487, 2.339, 4.433)),
reference_amplitude=np.array((697, 376, 88, 136, 339, 60, 102)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 13.0,
)

addfitter(
element="Fe",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=7058.18,
energies = np.array((7046.90, 7057.21, 7058.36, 7054.75)),#Table III E_i
lorentzian_fwhm = np.array((14.17, 3.12, 1.97, 6.38)),#Table III W_i
reference_amplitude=np.array((107, 448, 615, 141)),#Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Co",
linetype="KAlpha",
reference_short = "Hoelzer 1997, NISTfits.ipf",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=6930.38,
energies = np.array((6930.425, 6929.388, 6927.676, 6930.941, 6915.713, 6914.659, 6913.078)),
lorentzian_fwhm = np.array((1.795, 2.695, 4.555, 0.808, 2.406, 2.773, 4.463)),
reference_amplitude=np.array((809, 205, 107, 41, 314, 131, 43)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 15.0,
)

addfitter(
element="Co",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=7649.45,
energies = np.array((7649.60, 7647.83, 7639.87, 7645.49, 7636.21, 7654.13)), #Table III E_i
lorentzian_fwhm = np.array((3.05, 3.58, 9.78, 4.89, 13.59, 3.79)), #Table III W_i
reference_amplitude=np.array((798, 286, 85, 114, 33, 35)), #Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Ni",
linetype="KAlpha",
reference_short = "Hoelzer 1997, NISTfits.ipf",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=7478.26 ,
energies = np.array((7478.281, 7476.529, 7461.131, 7459.874, 7458.029)),
lorentzian_fwhm = np.array((2.013, 4.711, 2.674, 3.039, 4.476)),
reference_amplitude=np.array((909, 136, 351, 79, 24)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 17.2,
)

addfitter(
element="Ni",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=8264.78,
energies = np.array((8265.01, 8263.01, 8256.67, 8268.70)), #Table III E_i
lorentzian_fwhm = np.array((3.76, 4.34, 13.70, 5.18)), #Table III W_i
reference_amplitude=np.array((722, 358, 89, 104)), #Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Cu",
linetype="KAlpha",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=8047.83 ,
energies = np.array((8047.8372, 8045.3672, 8027.9935, 8026.5041)),
lorentzian_fwhm = np.array((2.285, 3.358, 2.667, 3.571)),
reference_amplitude=np.array((957, 90, 334, 111)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 20.0,
)

addfitter(
element="Cu",
linetype="KBeta",
reference_short = "Hoelzer 1997",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=8905.42,
energies = np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393)), #Table III E_i
lorentzian_fwhm = np.array((3.52, 3.52, 3.55, 8.08, 5.31)), #Table III W_i
reference_amplitude=np.array((757, 388, 171, 68, 55)), #Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

addfitter(
element="Zn",
linetype="KAlpha",
reference_short = "Zn Hack",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=8638.91 ,
energies = [8638.8872, 8636.4172, 8615.9835, 8614.4941],
lorentzian_fwhm = np.array((2.285, 3.358, 2.667, 3.571)) * 1.1,
reference_amplitude=np.array((957, 90, 334, 111)),
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff = 23.0,
)

addfitter(
element="Zn",
linetype="KBeta",
reference_short = "Zn Hack",
reference_plot_gaussian_fwhm=None,
nominal_peak_energy=9573.6,
energies = np.array((8905.532, 8903.109, 8908.462, 8897.387, 8911.393))*1.06 + 133.85,
lorentzian_fwhm = np.array((3.52, 3.52, 3.55, 8.08, 5.31))*1.06,
reference_amplitude=np.array((757, 388, 171, 68, 55)), #Table III I_I
reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
)

def plot_all_spectra():
    """Makes a bunch of plots showing the line shape and component parts for the KAlpha
    and KBeta complexes defined in here.
    Intended to nearly replicate plots in references giving spectral lineshapes"""
    for name,spectrum_class in spectrum_classes.items():
        spectrum = spectrum_class()
        spectrum.plot_like_reference()

if __name__ == "__main__":
    spectrum = MgKAlpha()
    spectrum.rvs(100)
    spectrum.gaussian_fwhm=1
    spectrum.rvs(100)
    spectrum.plot()
    plt.close("all")

    plot_all_spectra()
    plt.show()
