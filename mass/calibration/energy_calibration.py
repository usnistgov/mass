"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011
"""
import numpy as np
from scipy.optimize import brentq
import numpy.typing as npt
from typing import Optional
from collections.abc import Callable
from dataclasses import dataclass

from mass.mathstat.interpolate import CubicSplineFunction, SmoothingSplineFunction, GPRSplineFunction
from mass.mathstat.derivative import ExponentialFunction, Identity, LogFunction
from mass.calibration.nist_xray_database import NISTXrayDBFile
from ..common import isstr


def LineEnergies():
    """
    A dictionary to know a lot of x-ray fluorescence line energies, based on Deslattes' database.

    It is built on facts from mass.calibration.nist_xray_database module.

    It is a dictionary from peak name to energy, with several alternate names
    for the lines:

    E = Energies()
    print E["MnKAlpha"]
    print E["MnKAlpha"], E["MnKA"], E["MnKA1"], E["MnKL3"]
    """
    db = NISTXrayDBFile()
    alternate_line_names = {v: k for (k, v) in db.LINE_NICKNAMES.items()}
    data = {}

    for fullname, L in db.lines.items():
        element, linename = fullname.split(" ", 1)
        allnames = [linename]
        if linename in alternate_line_names:
            siegbahn_linename = alternate_line_names[linename]
            long_linename = siegbahn_linename.replace("A", "Alpha"). \
                replace("B", "Beta").replace("G", "Gamma")

            allnames.append(siegbahn_linename)
            allnames.append(long_linename)

            if siegbahn_linename.endswith("1"):
                allnames.append(siegbahn_linename[:-1])
                allnames.append(long_linename[:-1])

        for name in allnames:
            key = "".join((element, name))
            data[key] = L.peak

    return data


# Some commonly-used standard energy features.
STANDARD_FEATURES = LineEnergies()


@dataclass(frozen=True)
class EnergyCalibrationMaker:
    """An object that can make energy calibration curves under various assumptions,
    but using a single set of calibration anchor points and uncertainties on them.

    Returns
    -------
    EnergyCalibrationMaker
        A factory for making various `EnergyCalibration` objects from the same anchor points.

    Raises
    ------
    ValueError
        When calibration data arrays have unequal length, or `ph` is not monotone in `energy`.
    """
    ph: np.ndarray[np.float64]
    energy: np.ndarray[np.float64]
    dph: np.ndarray[np.float64]
    de: np.ndarray[np.float64]
    names: list[str]

    def __post_init__(self):
        """Check for inputs of unequal length. Check for monotone anchor points.
        Sort the input data by energy."""
        N = len(self.ph)
        assert N == len(self.energy)
        assert N == len(self.dph)
        assert N == len(self.de)
        assert N == len(self.names)

        # First sort according to energy of the calibration point
        if not np.all(np.diff(self.energy) > 0):
            idx = np.argsort(self.energy)
            self.ph[:] = self.ph[idx]
            self.energy[:] = self.energy[idx]
            self.dph[:] = self.dph[idx]
            self.de[:] = self.de[idx]
            self.names[:] = [self.names[i] for i in idx]

        # Then confirm that the pulse heights are also in order
        order_ph = self.ph.argsort()
        order_en = self.energy.argsort()
        if not np.all(order_ph == order_en):
            a = f"PH:     {self.ph[order_ph]}"
            b = f"Energy: {self.energy[order_ph]}"
            raise ValueError(f"Calibration points are not monotone:\n{a}\n{b}")

    def _remove_cal_point_idx(self, idx):
        """Remove calibration point number `idx` from the calibration. Return a new maker."""
        ph = np.hstack((self.ph[:idx], self.ph[idx + 1:]))
        energy = np.hstack((self.energy[:idx], self.energy[idx + 1:]))
        dph = np.hstack((self.dph[:idx], self.dph[idx + 1:]))
        de = np.hstack((self.de[:idx], self.de[idx + 1:]))
        names = self.names.copy().pop(idx)
        return EnergyCalibrationMaker(ph, energy, dph, de, names)

    def remove_cal_point_name(self, name):
        """Remove calibration point named `name`. Return a new maker."""
        idx = self.names.index(name)
        return self._remove_cal_point_idx(idx)

    def remove_cal_point_prefix(self, prefix):
        """This removes all cal points whose name starts with `prefix`.  Return a new maker."""
        # Work recursively: remove the first match and make a new Maker, and repeat until none match.
        # This is clearly less efficient when removing N matches, as N copies are made. So what?
        # This feature is likely to be rarely used, and we favor clarity over performance here.
        for name in tuple(self.names):
            if name.startswith(prefix):
                return self.remove_cal_point_name(name).remove_cal_point_prefix(prefix)
        return self

    def remove_cal_point_energy(self, energy, de):
        """Remove cal points at energies within ±`de` of `energy`. Return a new maker."""
        idxs = np.nonzero(np.abs(self.energy - energy) < de)[0]
        if len(idxs) == 0:
            return self
        # Also recursive and less efficient. See previous method's comment.
        return self._remove_cal_point_idx(idxs[0]).remove_cal_point_energy(energy, de)

    def add_cal_point(self, ph, energy, name="", ph_error=None, e_error=None, replace=True):
        """Add a single energy calibration point.

        Can call as .add_cal_point(ph, energy, name) or if the "energy" is a line name, then
        .add_cal_point(ph, name) will find energy as `energy=mass.STANDARD_FEATURES[name]`.
        Thus the following are equivalent:

            cal = cal.add_cal_point(12345.6, 5898.801, "Mn Ka1")
            cal = cal.add_cal_point(12456.6, "Mn Ka1")

        `ph` must be in units of the self.ph_field and `energy` is in eV.
        `ph_error` is the 1-sigma uncertainty on the pulse height.  If None
        (the default), then assign ph_error = `ph`/1000. `e_error` is the
        1-sigma uncertainty on the energy itself. If None (the default), then
        assign e_error=0.01 eV.

        Careful!  If you give a name that's already in the list, or you add an equivalent
        energy but do NOT give a name, then this value replaces the previous one.
        You can prevent overwriting (and instead raise an error) by setting `replace`=False.
        """

        # If <energy> is a string and a known spectral feature's name, use it as the name instead
        # Otherwise, it needs to be a numeric type convertible to float.
        try:
            energy = float(energy)
        except ValueError:
            try:
                name = energy
                energy = STANDARD_FEATURES[name]
            except Exception:
                raise ValueError("2nd argument must be an energy or a known name"
                                 + " from mass.energy_calibration.STANDARD_FEATURES")

        if ph_error is None:
            ph_error = ph * 0.001
        if e_error is None:
            e_error = 0.01  # Assume 0.01 eV error if none given

        update_index = None
        if name and name in self.names:  # Update an existing point by name
            if not replace:
                raise ValueError(
                    f"Calibration point '{name}' is already known and overwrite is False")
            update_index = self.names.index(name)

        elif np.abs(energy - self.energy).min() <= e_error:  # Update existing point
            if not replace:
                raise ValueError(
                    f"Calibration point at energy {energy:.2f} eV is already known and overwrite is False")
            update_index = np.abs(energy - self.energy).argmin()

        if update_index is None:   # Add a new calibration anchor point
            new_ph = np.hstack((self.ph, ph))
            new_energy = np.hstack((self.energy, energy))
            new_dph = np.hstack((self.dph, ph_error))
            new_de = np.hstack((self.de, e_error))
            new_names = self.names + [name]
        else:  # Replace an existing calibration anchor point.
            new_ph = self.ph.copy()
            new_energy = self.energy.copy()
            new_dph = self.dph.copy()
            new_de = self.de.copy()
            new_names = self.names.copy()
            new_ph[update_index] = ph
            new_energy[update_index] = energy
            new_dph[update_index] = ph_error
            new_de[update_index] = e_error
            new_names[update_index] = name
        return EnergyCalibrationMaker(new_ph, new_energy, new_dph, new_de, new_names)


class EnergyCalibration:  # noqa: PLR0904
    """Object to store information relevant to one detector's absolute energy
    calibration and to offer conversions between pulse height and energy.

    The behavior is governed by the constructor arguments `loglog`, `approximate`,
    and `zerozero` and by the number of data points. The construction-time arguments
    can be changed by calling EnergyCalibration.set_use_approximation() and
    EnergyCalibration.set_curvetype().

    curvetype -- Either a code number in the range [0,len(self.CURVETYPE)) or a
        string from the tuple self.CURVETYPE.

    approximate -- Whether to construct a smoothing spline (minimal curvature
        subject to a condition that chi-squared not be too large). If not,
        curve will be an exact spline in E vs PH, in log(E) vs log(PH), or
        as appropriate to the curvetype.

    The forward conversion from PH to E uses the callable __call__ method or its synonym,
    the method ph2energy.

    The inverse conversion method energy2ph calls Brent's method of root-finding.
    It's probably quite slow compared to a self.ph2energy for an array of equal length.

    All of __call__, ph2energy, and energy2ph should return a scalar when given a
    scalar input, or a matching numpy array when given any sequence as an input.
    """

    CURVETYPE = (
        "loglog",
        "linear",
        "linear+0",
        "gain",
        "invgain",
        "loggain",
    )

    def __init__(self, nonlinearity=1.1, curvetype="loglog", approximate=False,
                 useGPR=True):
        """Create an EnergyCalibration object for pulse-height-related field.

        Args:
            nonlinearity (float): the exponent N in the default, low-energy limit of
                E propto (PH)^N.  Typically 1.0 to 1.3 are reasonable.
            curvetype (str or int): one of EnergyCalibration.CURVETYPE.
            approximate (boolean):  Whether to use approximate "smoothing splines". (If not, use splines
                that go exactly through the data.) Default = False, because this works
                poorly unless the user calls with sensible PH and Energy uncertainties.
            useGPR (boolean): whether to use the new GPR-style choice for smoothing splines.
                Default True, but set False to use the pre-2021 approximation method.
        """
        self._curvetype = 0
        self.set_curvetype(curvetype)
        self._ph2e = self.__default_ph2e
        self._e2ph = self.__default_ph2e
        self._ph = np.zeros(0, dtype=float)
        self._energies = np.zeros(0, dtype=float)
        self._dph = np.zeros(0, dtype=float)
        self._de = np.zeros(0, dtype=float)
        self._names = []
        self.npts = 0
        self._use_approximation = approximate
        self._use_GPR = useGPR
        self._model_is_stale = False
        self._e2phwarned = False
        self.nonlinearity = nonlinearity
        self.set_nonlinearity()

    @staticmethod
    def __default_ph2e(x, der=0):
        if der > 0:
            return np.zeros_like(x)
        return x

    def __call__(self, pulse_ht, der=0):
        return self.ph2energy(pulse_ht, der=der)

    def ph2energy(self, pulse_ht, der=0):
        """Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV).
        Should return a scalar if passed a scalar, and a numpy array if passed a list or array

        Args:
            pulse_ht (float or np.array(dtype=float)): pulse heights in an arbitrary unit.
            der (int): the order of derivative. `der` should be >= 0.
        """
        if self._model_is_stale:
            self._update_converters()
        result = self._ph2e(pulse_ht, der=der)

        if np.isscalar(pulse_ht):
            if np.isscalar(result):
                return result
            return result.item()

        # Change any inf or NaN results to 0.0 energy.
        if any(np.isnan(result)):
            result[np.isnan(result)] = 0.0
        if any(np.isinf(result)):
            result[np.isinf(result)] = 0.0
        return np.array(result)

    def energy2ph(self, energy):
        """Convert energy (or array of energies) `energy` to pulse height in arbs.

        Should return a scalar if passed a scalar, and a numpy array if passed a list or array
        Uses a spline with steps no greater than ~1% in pulse height space. For a Brent's
        method root finding (i.e., an actual inversion of the ph->energy function), use
        method `energy2ph_exact`.
        """
        if self._model_is_stale:
            self._update_converters()
        result = self._e2ph(energy)

        if np.isscalar(energy):
            if np.isscalar(result):
                return result
            return result.item()
        return np.array(result)

    def energy2ph_exact(self, energy):
        """Convert energy (or array of energies) `energy` to pulse height in arbs.

        Inverts the _ph2e function by Brent's method for root finding. Can be fragile! Use
        method `energy2ph` for less precise but more generally error-free computation.
        Should return a scalar if passed a scalar, and a numpy array if passed a list or array.
        """
        if self._model_is_stale:
            self._update_converters()

        def energy_residual(ph, etarget):
            return self._ph2e(ph) - etarget

        if np.isscalar(energy):
            return brentq(energy_residual, 1e-6, self._max_ph, args=(energy,))
        elif len(energy) > 10 and not self._e2phwarned:
            print("WARNING: EnergyCalibration.energy2ph can be slow for long inputs.")
            self._e2phwarned = True

        if len(energy) > 1024:
            phs = np.array(energy)
            # Newton methods with a fixed number of iterations.
            for _ in range(5):
                phs -= (self(phs) - energy) / self(phs, der=1)

            return phs

        result = [brentq(energy_residual, 1e-6, self._max_ph, args=(e,)) for e in energy]
        return np.array(result)

    def ph2dedph(self, ph):
        """Calculate the slope at pulse heights `ph`."""
        if self._model_is_stale:
            self._update_converters()
        return self(ph, der=1)

    def energy2dedph(self, energy):
        """Calculate the slope at energy."""
        return self.ph2dedph(self.energy2ph(energy))

    def ph2uncertainty(self, ph):
        """Cal uncertainty in eV at the given pulse heights."""
        if not self._use_approximation:
            raise ValueError(
                "Cannot estimate uncertainty. First use cal.set_use_approximation(True)")
        if self._model_is_stale:
            self._update_converters()
        return self._uncertainty(ph)

    def energy2uncertainty(self, energies):
        """Cal uncertainty in eV at the given pulse heights."""
        ph = self.energy2ph(energies)
        return self.ph2uncertainty(ph)

    def e2ph(self, energy): return self.energy2ph(energy)

    def e2dedph(self, energy): return self.energy2dedph(energy)

    def e2uncertainty(self, energy): return self.energy2uncertainty(energy)

    def __str__(self):
        self._update_converters()  # To sort the points
        seq = ["EnergyCalibration()"]
        for name, pulse_ht, energy in zip(self._names, self._ph, self._energies):
            seq.append(f"  energy(ph={pulse_ht:7.2f}) --> {energy:9.2f} eV ({name})")
        return "\n".join(seq)

    def set_nonlinearity(self, powerlaw=1.15):
        """Update the power law index assumed when there's 1 data point and a loglog curve type."""
        if self.curvename() == "loglog" and powerlaw != self.nonlinearity:
            self._model_is_stale = True
        self.nonlinearity = powerlaw

    def set_use_approximation(self, useit):
        """Switch to using (or to NOT using) approximating splines with
        reduced knot count. You can interchange this with adding points, because
        the actual model computation isn't done until the cal curve is called."""
        if useit != self._use_approximation:
            self._use_approximation = useit
            self._model_is_stale = True

    def set_GPR(self, useit):
        if useit != self._use_GPR:
            self._use_GPR = useit
            self._model_is_stale = True

    def set_curvetype(self, curvetype):
        if isstr(curvetype):
            # Fix a behavior of h5py for writing in py2, reading in py3.
            if isinstance(curvetype, bytes):
                curvetype = curvetype.decode("utf-8")
            try:
                curvetype = self.CURVETYPE.index(curvetype.lower())
            except ValueError:
                raise ValueError(f"EnergyCalibration.CURVETYPE does not contain '{curvetype}'")
        assert 0 <= curvetype < len(self.CURVETYPE)

        if curvetype != self._curvetype:
            self._curvetype = curvetype
            self._model_is_stale = True

    def curvename(self):
        return self.CURVETYPE[self._curvetype]

    def copy(self):
        """Return a deep copy."""
        ecal = EnergyCalibration()
        ecal.__dict__.update(self.__dict__)
        ecal._names = list(self._names)
        ecal._ph = self._ph.copy()
        ecal._energies = self._energies.copy()
        ecal._dph = self._dph.copy()
        ecal._de = self._de.copy()
        ecal._use_approximation = self._use_approximation
        ecal._use_GPR = self._use_GPR
        ecal._model_is_stale = True
        ecal._curvetype = self._curvetype
        return ecal

    def _remove_cal_point_idx(self, idx):
        """Remove calibration point number `idx` from the calibration."""
        self._names.pop(idx)
        self._ph = np.hstack((self._ph[:idx], self._ph[idx + 1:]))
        self._energies = np.hstack((self._energies[:idx], self._energies[idx + 1:]))
        self._dph = np.hstack((self._dph[:idx], self._dph[idx + 1:]))
        self._de = np.hstack((self._de[:idx], self._de[idx + 1:]))
        self.npts -= 1
        self._model_is_stale = True

    def remove_cal_point_name(self, name):
        """If you don't like calibration point named <name>, this removes it."""
        idx = self._names.index(name)
        self._remove_cal_point_idx(idx)

    def remove_cal_point_prefix(self, prefix):
        """This removes all cal points whose name starts with <prefix>.  Return number removed."""
        for name in tuple(self._names):
            if name.startswith(prefix):
                self.remove_cal_point_name(name)

    def remove_cal_point_energy(self, energy, de):
        """Remove cal points at energies with <de> of <energy>"""
        idxs = np.nonzero(np.abs(self._energies - energy) < de)[0]

        for idx in idxs:
            self._remove_cal_point_idx(idx)

    def add_cal_point(self, pht, energy, name="", pht_error=None, e_error=None, overwrite=True):
        """Add a single energy calibration point <pht>, <energy>,

        <pht> must be in units of the self.ph_field and <energy> is in eV.
        <pht_error> is the 1-sigma uncertainty on the pulse height.  If None
        (the default), then assign pht_error = <pht>/1000. <e_error> is the
        1-sigma uncertainty on the energy itself. If None (the default), then
        assign e_error=<energy>/10^5 (typically 0.05 eV).

        Also, you can call it with <energy> as a string, provided it's the name
        of a known feature appearing in the dictionary
        mass.energy_calibration.STANDARD_FEATURES.  Thus the following are
        equivalent:

        cal.add_cal_point(12345.6, 5898.801, "Mn Ka1")
        cal.add_cal_point(12456.6, "Mn Ka1")

        Careful!  If you give a name that's already in the list, then this value
        replaces the previous one.  If you do NOT give a name, though, then this
        will NOT replace but will add to any existing points at the same energy.
        You can prevent overwriting by setting <overwrite>=False.
        """
        self._model_is_stale = True

        # If <energy> is a string and a known spectral feature's name, use it as the name instead
        # Otherwise, it needs to be a numeric type convertible to float.
        # if energy in STANDARD_FEATURES:
        #     name = energy
        #     energy = STANDARD_FEATURES[name]
        # else:
        try:
            energy = float(energy)
        except ValueError:
            try:
                name = energy
                energy = STANDARD_FEATURES[name]
            except Exception:
                raise ValueError("2nd argument must be an energy or a known name"
                                 + " from mass.energy_calibration.STANDARD_FEATURES")

        if pht_error is None:
            pht_error = pht * 0.001
        if e_error is None:
            e_error = 0.01  # Assume 0.01 eV error if none given

        update_index = None
        if name and name in self._names:  # Update an existing point by name
            if not overwrite:
                raise ValueError(
                    f"Calibration point '{name}' is already known and overwrite is False")
            update_index = self._names.index(name)

        elif self.npts > 0 and np.abs(energy - self._energies).min() <= e_error:  # Update existing point
            if not overwrite:
                raise ValueError(
                    f"Calibration point at energy {energy:.2f} eV is already known and overwrite is False")
            update_index = np.abs(energy - self._energies).argmin()

        if update_index is None:   # Add a new point
            self._ph = np.hstack((self._ph, pht))
            self._energies = np.hstack((self._energies, energy))
            self._dph = np.hstack((self._dph, pht_error))
            self._de = np.hstack((self._de, e_error))
            self._names.append(name)
        else:
            self._ph[update_index] = pht
            self._energies[update_index] = energy
            self._dph[update_index] = pht_error
            self._de[update_index] = e_error
        self.npts = len(self._ph)

        # Validate: ph and energies must be monotone increasing
        order_ph = self._ph.argsort()
        order_en = self._energies.argsort()
        if not np.all(order_ph == order_en):
            a = f"PH:     {self._ph[order_ph]}"
            b = f"Energy: {self._energies[order_ph]}"
            raise Exception(f"Calibration points are not monotone:\n{a}\n{b}")

    @property
    def cal_point_phs(self):
        return self._ph

    @property
    def cal_point_energies(self):
        return self._energies

    @property
    def cal_point_names(self):
        return self._names

    @property
    def ismonotonic(self):
        "Is the curve monotonic from 0 to 1.05 times the max anchor point's pulse height?"
        npoints = 1001
        ph = np.linspace(0, 1.05 * self._ph.max(), npoints)
        e = self(ph)
        return np.all(np.diff(e) > 0)

    def _update_converters(self):
        """There is now one (or more) new data points. All the math goes on in this method."""
        # Sort in ascending energy order
        sortkeys = np.argsort(self._ph)
        self._ph = self._ph[sortkeys]
        self._energies = self._energies[sortkeys]
        self._dph = self._dph[sortkeys]
        self._de = self._de[sortkeys]
        self._names = [self._names[s] for s in sortkeys]

        assert self.npts == len(self._ph)
        assert self.npts == len(self._dph)
        assert self.npts == len(self._energies)
        assert self.npts == len(self._de)

        self._max_ph = 2 * np.max(self._ph)
        # Compute cal curve inverse and (if GPRSplined, the variance) at these points
        ph = self._ph
        ph_pts = [np.linspace(0, ph[0], 51)[1:]]
        for i in range(len(ph) - 1):
            npts = 2 + int(0.5 + (ph[i + 1] / ph[i] - 1) * 100)
            ph_pts.append(np.linspace(ph[i], ph[i + 1], npts)[1:])
        ph_pts.append(np.linspace(ph[-1], 2 * ph[-1], 101)[1:])
        ph_pts = np.hstack(ph_pts)

        if self._use_approximation and self.npts >= 3:
            self._update_approximators(ph_pts)
        else:
            self._update_exactcurves()
        self._e2ph = CubicSplineFunction(self._ph2e(ph_pts), ph_pts)
        self._model_is_stale = False

    def _update_approximators(self, ph_pts):
        "Update approximating spline. Find and spline variance at points `ph_pts`"
        PreferredSpline = GPRSplineFunction
        if not self._use_GPR:
            PreferredSpline = SmoothingSplineFunction

        # Make sure the errors in both dimensions are reasonable (positive)
        if (self._dph <= 0.0).any():
            if (self._dph > 0).any():
                self._dph[self._dph <= 0.0] = self._dph[self._dph > 0].min()
            else:
                self._dph = np.zeros_like(self._dph)
        if (self._de <= 0.0).any():
            if (self._de > 0).any():
                self._de[self._de <= 0.0] = self._de[self._de > 0].min()
            else:
                self._de = np.zeros_like(self._de)

        # Find transformed data. For dy, assume that E and PH errors are uncorrelated.
        ph, dph, e, de = self._ph, self._dph, self._energies, self._de

        if self.curvename() == "loglog":
            underlying_spline = PreferredSpline(np.log(ph), np.log(e), de / e, dph / ph)
            self._ph2e = ExponentialFunction() << underlying_spline << LogFunction()
            cal_uncert = underlying_spline(ph_pts) * underlying_spline.variance(np.log(ph_pts))**0.5

        elif self.curvename().startswith("linear"):
            if ("+0" in self.curvename()) and (0.0 not in ph):
                ph = np.hstack([[0], ph])
                e = np.hstack([[0], e])
                de = np.hstack([[de.min() * 0.1], de])
                dph = np.hstack([[dph.min() * 0.1], dph])
            underlying_spline = PreferredSpline(ph, e, de, dph)
            self._ph2e = underlying_spline
            cal_uncert = underlying_spline.variance(ph_pts)**0.5

        elif self.curvename() == "gain":
            g = ph / e
            m = np.polyfit(ph, g, 1)[0]
            dg = g * (((m * ph / g - 1) * dph / ph)**2 + (de / e)**2)**0.5
            underlying_spline = PreferredSpline(ph, g, dg)
            p = Identity()
            self._ph2e = p / (underlying_spline << p)
            est_g = underlying_spline(ph_pts)
            est_e = ph_pts / est_g
            cal_uncert = underlying_spline.variance(ph_pts)**0.5 * est_e / est_g

            # Gain curves have a problem: gain<0 screws it all up. Avoid that region.
            trial_phmax = 10 * self._ph.max()
            if underlying_spline(trial_phmax) > 0:
                self._max_ph = trial_phmax

        elif self.curvename() == "invgain":
            ig = e / ph
            m = np.polyfit(ph, ig, 1)[0]
            d_ig = ig * (((1 + m * ph / ig) * dph / ph)**2 + (de / e)**2)**0.5
            p = Identity()
            underlying_spline = PreferredSpline(ph, ig, d_ig)
            self._ph2e = p * (underlying_spline << p)
            cal_uncert = underlying_spline.variance(ph_pts)**0.5 * ph_pts

        elif self.curvename() == "loggain":
            lg = np.log(ph / e)
            m = np.polyfit(ph, lg, 1)[0]
            d_lg = (((m * ph - 1) * dph / ph)**2 + (de / e)**2)**0.5
            p = Identity()
            underlying_spline = PreferredSpline(ph, lg, d_lg)
            self._ph2e = p * (ExponentialFunction() << (-underlying_spline << p))
            e_pts = self._ph2e(ph_pts)
            dfdp = underlying_spline(ph_pts, der=1)
            cal_uncert = underlying_spline.variance(ph_pts)**0.5 * e_pts * np.abs(dfdp)

        self._underlying_spline = underlying_spline
        if self._use_GPR:
            self._uncertainty = CubicSplineFunction(ph_pts, cal_uncert)
        else:
            self._uncertainty = np.zeros_like

    def _update_exactcurves(self):
        """Update the E(P) curve; assume exact interpolation of calibration data."""
        # Choose proper curve/interpolating function object
        # For N=0 points, in the absence of any information at all, we just let E = PH.
        # For N=1 points, use E proportional to PH (or if loglog curve, then a power law of
        #    the assumed nonlinearity).
        # For N>1 points, use the chosen curve type (but for N=2, recall that the spline will be a line).
        if self.npts <= 0:
            # self._ph2e = lambda p: p
            self._ph2e = Identity()

        elif self.npts == 1:
            p1 = self._ph[0]
            e1 = self._energies[0]
            if self.curvename() == "loglog":
                self._ph2e = e1 * (Identity() / p1)**self.nonlinearity
            elif self.curvename() in {"gain", "invgain"}:
                self._ph2e = (e1 / p1) * Identity()
            else:
                raise Exception(f"curvename={self.curvename()} not implemented for npts=1")

        elif self.curvename() == "loglog":
            x = np.log(self._ph)
            y = np.log(self._energies)
            # self._x2yfun = CubicSpline(x, y)
            # self._ph2e = lambda p: np.exp(self._x2yfun(np.log(p)))
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2e = ExponentialFunction() << underlying_spline << LogFunction()

        elif self.curvename().startswith("linear"):
            x = self._ph
            y = self._energies
            if ("+0" in self.curvename()) and (0.0 not in x):
                x = np.hstack(([0], x))
                y = np.hstack(([0], y))
            # self._ph2e = CubicSpline(x, y)
            self._ph2e = CubicSplineFunction(x, y)

        elif self.curvename() == "gain":
            x = self._ph
            y = x / self._energies
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2e = lambda p: p/self._underlying_spline(p)
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2e = Identity() / underlying_spline

            # Gain curves have a problem: gain<0 screws it all up. Avoid that region.
            trial_phmax = 10 * self._ph.max()
            if underlying_spline(trial_phmax) > 0:
                self._max_ph = trial_phmax
            else:
                self._max_ph = 0.99 * brentq(underlying_spline, 0, trial_phmax)

        elif self.curvename() == "invgain":
            x = self._ph
            y = self._energies / x
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2e = lambda p: p*self._underlying_spline(p)
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2e = Identity() * underlying_spline

        elif self.curvename() == "loggain":
            x = self._ph
            y = np.log(x / self._energies)
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2e = lambda p: p*np.exp(-self._underlying_spline(p))
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2e = Identity() * (ExponentialFunction() << -underlying_spline)

        ph = self._ph
        ph_pts = [np.linspace(0, ph[0], 51)[1:]]
        for i in range(len(ph) - 1):
            npts = 2 + int(0.5 + (ph[i + 1] / ph[i] - 1) * 100)
            ph_pts.append(np.linspace(ph[i], ph[i + 1], npts)[1:])
        ph_pts.append(np.linspace(ph[-1], 2 * ph[-1], 101)[1:])
        ph_pts = np.hstack(ph_pts)
        self._e2ph = CubicSplineFunction(self._ph2e(ph_pts), ph_pts)

    def name2ph(self, name):
        """Convert a named energy feature to pulse height. `name` need not be a calibration point."""
        energy = STANDARD_FEATURES[name]
        return self.energy2ph(energy)

    def plotgain(self, **kwargs):
        kwargs["plottype"] = "gain"
        self.plot(**kwargs)

    def plotinvgain(self, **kwargs):
        kwargs["plottype"] = "invgain"
        self.plot(**kwargs)

    def plotloggain(self, **kwargs):
        kwargs["plottype"] = "loggain"
        self.plot(**kwargs)

    def plot(self, axis=None, color="blue", markercolor="red", plottype="linear", ph_rescale_power=0.0,  # noqa: PLR0917
             removeslope=False, energy_x=False, showtext=True, showerrors=True, min_energy=None, max_energy=None):
        # Plot smooth curve
        minph, maxph = self._ph.max() * .001, self._ph.max() * 1.1
        if min_energy is not None:
            minph = self.e2ph(min_energy)
        if max_energy is not None:
            maxph = self.e2ph(max_energy)
        phplot = np.linspace(minph, maxph, 1000)
        eplot = self(phplot)
        gplot = phplot / eplot
        dyplot = None
        gains = self._ph / self._energies
        slope = 0.0
        xplot = phplot
        x = self._ph
        xerr = self._dph
        if energy_x:
            xplot = eplot
            x = self._energies
            xerr = self._de

        import pylab as plt  # noqa: PLC0415
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
            # axis.set_xlim([x[0], x[-1]*1.1])
        if energy_x:
            axis.set_xlabel("Energy (eV)")
        else:
            axis.set_xlabel("Pulse height")

        if plottype == "linear":
            yplot = self(phplot) / (phplot**ph_rescale_power)
            if self._use_approximation:
                dyplot = self.ph2uncertainty(phplot) / (phplot**ph_rescale_power)
            y = self._energies / (self._ph**ph_rescale_power)
            if ph_rescale_power == 0.0:
                ylabel = "Energy (eV)"
                axis.set_title("Energy calibration curve")
            else:
                ylabel = f"Energy (eV) / PH^{ph_rescale_power:.4f}"
                axis.set_title(f"Energy calibration curve, scaled by {ph_rescale_power:.4f} power of PH")
        elif plottype == "gain":
            yplot = gplot
            if self._use_approximation:
                dyplot = self.ph2uncertainty(phplot) / eplot * gplot
            y = gains
            ylabel = "Gain (PH/eV)"
            axis.set_title("Energy calibration curve, gain")
        elif plottype == "invgain":
            yplot = 1.0 / gplot
            if self._use_approximation:
                dyplot = self.ph2uncertainty(phplot) / phplot
            y = 1.0 / gains
            ylabel = "Inverse Gain (eV/PH)"
            axis.set_title("Energy calibration curve, inverse gain")
        elif plottype == "loggain":
            yplot = np.log(gplot)
            if self._use_approximation:
                dyplot = self.ph2uncertainty(phplot) / eplot
            y = np.log(gains)
            ylabel = "Log Gain: log(eV/PH)"
            axis.set_title("Energy calibration curve, log gain")
        elif plottype == "loglog":
            yplot = np.log(eplot)
            xplot = np.log(phplot)
            if self._use_approximation:
                dyplot = self.ph2uncertainty(phplot) / eplot
            y = np.log(self._energies)
            x = np.log(self._ph)
            xerr = (self._dph / self._ph)
            ylabel = "Log energy/1 eV"
            axis.set_xlabel("log(Pulse height/arbs)")
            axis.set_title("Energy calibration curve, log gain")
        else:
            raise ValueError("plottype must be one of ('linear', 'gain','loggain','invgain').")

        if removeslope:
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            yplot -= slope * xplot

        axis.plot(xplot, yplot, color=color)
        if dyplot is not None and showerrors:
            axis.plot(xplot, yplot + dyplot, color=color, alpha=0.35)
            axis.plot(xplot, yplot - dyplot, color=color, alpha=0.35)

        # Plot and label cal points
        dy = ((self._de / self._energies)**2 + (self._dph / self._ph)**2)**0.5 * y
        axis.errorbar(x, y - slope * x, yerr=dy, xerr=xerr, fmt='o',
                      mec='black', mfc=markercolor, capsize=0)
        axis.grid(True)
        if removeslope:
            ylabel = f"{ylabel} slope removed"
        axis.set_ylabel(ylabel)
        if showtext:
            for xval, name, yval in zip(x, self._names, y):
                axis.text(xval, yval - slope * xval, name + '  ', ha='right')

    def drop_one_errors(self):
        """For each calibration point, calculate the difference between the 'correct' energy
        and the energy predicted by creating a calibration without that point and using
        ph2energy to calculate the predicted energy, return energies, drop_one_energy_diff"""
        drop_one_energy_diff = np.zeros(self.npts)
        for i in range(self.npts):
            drop_one_energy, drop_one_pulseheight = self._energies[i], self._ph[i]
            cal2 = self.copy()
            cal2._remove_cal_point_idx(i)
            predicted_energy = cal2.ph2energy(drop_one_pulseheight)
            drop_one_energy_diff[i] = predicted_energy - drop_one_energy
        perm = np.argsort(self._energies)
        return self._energies[perm], drop_one_energy_diff[perm]

    def save_to_hdf5(self, hdf5_group, name):
        if name in hdf5_group:
            del hdf5_group[name]

        cal_group = hdf5_group.create_group(name)
        cal_group["name"] = [_name.encode() for _name in self._names]
        cal_group["ph"] = self._ph
        cal_group["energy"] = self._energies
        cal_group["dph"] = self._dph
        cal_group["de"] = self._de
        cal_group.attrs['nonlinearity'] = self.nonlinearity
        cal_group.attrs['curvetype'] = self.CURVETYPE[self._curvetype]
        cal_group.attrs['approximate'] = self._use_approximation

    @classmethod
    def load_from_hdf5(cls, hdf5_group, name):
        cal_group = hdf5_group[name]
        cal = cls(cal_group.attrs['nonlinearity'],
                  cal_group.attrs['curvetype'],
                  cal_group.attrs['approximate'])

        _names = cal_group["name"][:]
        _ph = cal_group["ph"][:]
        _energies = cal_group["energy"][:]
        _dph = cal_group["dph"][:]
        _de = cal_group["de"][:]

        for thisname, ph, e, dph, de in zip(_names, _ph, _energies, _dph, _de):
            cal.add_cal_point(ph, e, thisname.decode(), dph, de)

        return cal

    def __repr__(self):
        s = f"""mass.EnergyCalibration with {len(self._names)} entries
        _ph: {self._ph}
        _energies: {self._energies}
        _names: {self._names}
        _curvetype: {self._curvetype}
        _use_approximation: {self._use_approximation}"""
        return s
