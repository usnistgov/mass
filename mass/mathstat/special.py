"""
special.py

Module mass.mathstat.special

Contains special functions not readily available elsewhere.

voigt               - The Voigt profile, a convolution of Lorentzian and Gaussians.
voigt_fwhm_approx   - A ~0.02% accurate approximation for the FWHM of a Voigt function.

Joe Fowler, NIST

February 3, 2012
"""

__all__=['voigt', 'voigt_approx_fwhm']

import numpy, scipy.special

_sqrt2 = numpy.sqrt(2.0)
_sqrt2pi = numpy.sqrt(2.0*numpy.pi)


def voigt(x, x0, hwhm, sigma):
    """
    Compute and return the Voigt function V(x; x0,hwhm,sigma) for a sequence of points <x>.
    V is the convolution of a Lorentzian centered at x0 with half-width at half-max of hwhm
    with a Gaussian having standard width sigma.

    This is the lineshape of a Lorentzian (a.k.a. Breit-Wigner and Cauchy) distributed emission 
    line with Gaussian broadening due either to finite measurement resolution or to physical
    effects like Doppler shifts in molecules having a Maxwellian velocity distribution.

    L(x) = (hwhm/pi) / ((x-x0)**2 + hwhm**2)
    G(x) = 1/(sigma sqrt(2pi)) * exp(-0.5*(x/sigma)**2
    V(x) = integral (-inf to +inf) G(x') L(x-x') dx'

    Scalar parameters are:
    x0      Center of Lorentzian line
    hwhm    Half-width at half-maximum of the Lorentzian line
    sigma   Square root of the Gaussian variance

    The construction uses:
    V(x) = Re[w(z)]/(sigma * sqrt(2pi)), where
    w(z) = exp(-z*z) * erfc(-iz) is the Faddeeva or complex error function, and
    z = (x+i*fwhm)/(sigma sqrt(2))
    """

    if not isinstance(x, numpy.ndarray):
        return voigt( numpy.array(x), x0, hwhm, sigma)
    # Pure Gaussian limit
    if hwhm == 0.0:
        return numpy.exp(-0.5*((x-x0)/sigma)**2) / (sigma*_sqrt2pi)
    # Pure Lorentzian limit
    if sigma == 0.0:
        return (hwhm/numpy.pi) / ((x-x0)**2 + hwhm**2)

    # General Voigt function
    z = (x-x0 + 1j*hwhm)/(sigma * _sqrt2)
    w = scipy.special.wofz(z)
    return (w.real)/(sigma * _sqrt2pi)



def voigt_approx_fwhm(fwhm_lorentzian, fwhm_gaussian):
    """Return the Olivero & Longbothum 1977 approximation to the Voigt full-width at half-maximum,
    found in doi:10.1016/0022-4073(77)90161-3 and also in Wikipedia.

    This ought to be accurate to 0.02% at typical values up to 0.033% for a nearly pure
    Lorentzian."""
    if fwhm_lorentzian == 0.0:
        return fwhm_gaussian
    if fwhm_gaussian == 0.0:
        return fwhm_lorentzian
    return 0.5346*fwhm_lorentzian + numpy.sqrt(0.2166*(fwhm_lorentzian**2) + fwhm_gaussian**2)