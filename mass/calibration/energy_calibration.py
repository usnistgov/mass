"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011
Completely redesigned January 2025
"""
import numpy as np
import pylab as plt
import numpy.typing as npt
from typing import Optional
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass

from mass.mathstat.interpolate import CubicSplineFunction, GPRSplineFunction
from mass.calibration.nist_xray_database import NISTXrayDBFile


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

    @classmethod
    def init(cls,
             ph: Optional[npt.ArrayLike] = None,
             energy: Optional[npt.ArrayLike] = None,
             dph: Optional[npt.ArrayLike] = None,
             de: Optional[npt.ArrayLike] = None,
             names: Optional[list] = None,
             ):
        if ph is None:
            ph = np.array([], dtype=float)
        else:
            ph = np.asarray(ph)
        if energy is None:
            energy = np.array([], dtype=float)
        else:
            energy = np.asarray(energy)
        if dph is None:
            dph = 1e-3 * ph
        else:
            dph = np.asarray(dph)
        if de is None:
            de = 1e-3 * energy
        else:
            de = np.asarray(de)
        if names is None:
            names = ["dummy"] * len(dph)
        return cls(ph, energy, dph, de, names)

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
            sortkeys = np.argsort(self.energy)
            self.ph[:] = self.ph[sortkeys]
            self.energy[:] = self.energy[sortkeys]
            self.dph[:] = self.dph[sortkeys]
            self.de[:] = self.de[sortkeys]
            self.names[:] = [self.names[i] for i in sortkeys]

        # Then confirm that the pulse heights are also in order
        order_ph = self.ph.argsort()
        order_en = self.energy.argsort()
        if not np.all(order_ph == order_en):
            a = f"PH:     {self.ph[order_ph]}"
            b = f"Energy: {self.energy[order_ph]}"
            raise ValueError(f"Calibration points are not monotone:\n{a}\n{b}")

    @property
    def npts(self):
        return len(self.ph)

    def _remove_cal_point_idx(self, idx):
        """Remove calibration point number `idx` from the calibration. Return a new maker."""
        ph = np.delete(self.ph, idx)
        energy = np.delete(self.energy, idx)
        dph = np.delete(self.dph, idx)
        de = np.delete(self.de, idx)
        names = self.names.copy()
        names.pop(idx)
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
        if self.npts > 0:
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

    ALLOWED_CURVENAMES = {
        "linear", "linear+0", "loglog",
        "gain", "invgain", "loggain",
    }

    @staticmethod
    def heuristic_samplepoints(anchors: npt.ArrayLike) -> np.ndarray:
        """Given a set of calibration anchor points, return a few hundred
        sample points, reasonably spaced below, between, and above the anchor points.

        Parameters
        ----------
        anchors : ArrayLike
            The anchor points (in pulse height space)

        Returns
        -------
        np.ndarray
            _description_
        """
        anchors = np.asarray(anchors)
        # Prescription is 50 points up to lowest anchor (but exclude 0):
        x = [np.linspace(0, anchors.min(), 51)[1:]]
        # Then one points, plus one extra per 1% spacing between (and at) each anchor
        for i in range(len(anchors) - 1):
            low, high = anchors[i:i + 2]
            n = 1 + int(100 * (high / low - 1) + 0.5)
            x.append(np.linspace(low, high, n + 1)[1:])
        # Finally, 100 more points between the highest anchor and 2x that.
        x.append(anchors.max() * np.linspace(1, 2, 101)[1:])
        return np.hstack(x)

    def make_calibration(self, curvename="loglog", approximate=False, powerlaw=1.15, allow_attributes=False):
        if approximate and self.npts < 3:
            raise ValueError(f"approximating curves require 3 or more cal anchor points, have {self.npts}")
        if curvename not in self.ALLOWED_CURVENAMES:
            raise ValueError(f"curvename='{curvename}', must be in {self.ALLOWED_CURVENAMES}")

        # Use a heuristic to repair negative uncertainties.
        def regularize_uncertainties(x):
            if not np.any(x < 0):
                return x
            target = max(0.0, x.min())
            x = x.copy()
            x[x < 0] = target
            return x
        dph = regularize_uncertainties(self.dph)
        de = regularize_uncertainties(self.de)

        if curvename == "loglog":
            input_transform = EnergyCalibration._ecal_input_log
            output_transform = EnergyCalibration._ecal_output_log
            x = np.log(self.ph)
            y = np.log(self.energy)
            # When there's only one point, enhance it by a fake point to enforce power-law behavior
            if self.npts == 1:
                arboffset = 1.0
                x = np.hstack([x, x + arboffset])
                y = np.hstack([y, y + arboffset / powerlaw])
            dx = dph / self.ph
            dy = de / self.energy

        elif curvename == "gain":
            input_transform = EnergyCalibration._ecal_input_identity
            output_transform = EnergyCalibration._ecal_output_gain
            x = self.ph
            y = self.ph / self.energy
            # Estimate spline uncertainties using slope of best-fit line
            slope = np.polyfit(x, y, 1)[0]
            dy = y * (((slope * self.energy - 1) * dph / x)**2 + (de / self.energy)**2)**0.5
            dx = dph

        elif curvename == "invgain":
            input_transform = EnergyCalibration._ecal_input_identity
            output_transform = EnergyCalibration._ecal_output_invgain
            x = self.ph
            y = self.energy / self.ph
            # Estimate spline uncertainties using slope of best-fit line
            slope = np.polyfit(x, y, 1)[0]
            dy = y * (((slope * self.ph / y + 1) * dph / x)**2 + (de / self.energy)**2)**0.5
            dx = dph

        elif curvename in {"linear", "linear+0"}:
            input_transform = EnergyCalibration._ecal_input_identity
            output_transform = EnergyCalibration._ecal_output_identity
            x = self.ph
            y = self.energy
            dx = dph
            dy = de
            if ("+0" in curvename) and (0.0 not in x):
                # Add a "zero"-energy and -PH point. But to avoid numerical problems, actually just use
                # 1e-3 times the lowest value, giving ±100% uncertainty on the values.
                x = np.hstack(([x.min() * 1e-3], x))
                y = np.hstack(([y.min() * 1e-3], y))
                dx = np.hstack(([x[0] * 1e-3], dx))
                dy = np.hstack((y[0] * 1e-3, dy))

        elif curvename == "loggain":
            input_transform = EnergyCalibration._ecal_input_identity
            output_transform = EnergyCalibration._ecal_output_loggain
            x = self.ph
            y = np.log(self.ph / self.energy)
            # Estimate spline uncertainties using slope of best-fit line
            slope = np.polyfit(x, y, 1)[0]
            dy = y * (((slope * x - 1) * dph / x)**2 + (de / self.energy)**2)**0.5
            dx = dph

        else:
            raise ValueError(f"curvename='{curvename}' not recognized")

        if approximate:
            internal_spline = GPRSplineFunction(x, y, dy, dx)
        elif len(x) > 1:
            internal_spline = CubicSplineFunction(x, y)
        else:
            internal_spline = CubicSplineFunction(x * [1, 2], y * [1, 2])

        ph_samplepoints = EnergyCalibrationMaker.heuristic_samplepoints(self.ph)
        E_samplepoints = output_transform(
            ph_samplepoints,
            internal_spline(input_transform(ph_samplepoints))
        )
        energy2ph = CubicSplineFunction(E_samplepoints, ph_samplepoints)

        if approximate:
            dspline = internal_spline.variance(ph_samplepoints)**0.5
            if curvename == "loglog":
                de_samplepoints = dspline * internal_spline(input_transform(ph_samplepoints))
            elif curvename == "gain":
                de_samplepoints = dspline * E_samplepoints**2 / ph_samplepoints
            elif curvename == "invgain":
                de_samplepoints = dspline * ph_samplepoints
            elif curvename in {"linear", "linear+0"}:
                de_samplepoints = dspline
            elif curvename == "loggain":
                abs_dfdp = np.abs(internal_spline(ph_samplepoints, der=1))
                de_samplepoints = dspline * E_samplepoints * abs_dfdp
            else:
                raise ValueError(f"curvename='{curvename}' not recognized")

            uncertainty_spline = CubicSplineFunction(ph_samplepoints, de_samplepoints)
        else:
            uncertainty_spline = np.zeros_like

        ECalContructor = EnergyCalibration
        if allow_attributes:
            ECalContructor = EnergyCalibrationWithAttributes

        return ECalContructor(
            self.ph, self.energy, self.dph, self.de, self.names,
            curvename=curvename,
            approximating=approximate,
            spline=internal_spline,
            ph2uncertainty=uncertainty_spline,
            input_transform=input_transform,
            output_transform=output_transform,
            energy2ph=energy2ph
        )

    def drop_one_errors(self, curvename: str = "loglog",
                        approximate: bool = False,
                        powerlaw: float = 1.15) -> tuple[np.ndarray, np.ndarray]:
        # """For each calibration point, calculate the difference between the 'correct' energy
        # and the energy predicted by creating a calibration without that point and using
        # ph2energy to calculate the predicted energy, return (energies, drop_one_energy_diff)"""
        drop_one_energy_diff = np.zeros(self.npts)
        for i in range(self.npts):
            dropped_pulseheight = self.ph[i]
            dropped_energy = self.energy[i]
            drop_one_maker = self._remove_cal_point_idx(i)
            drop_one_cal = drop_one_maker.make_calibration(curvename=curvename, approximate=approximate, powerlaw=powerlaw)
            predicted_energy = drop_one_cal.ph2energy(dropped_pulseheight)
            drop_one_energy_diff[i] = predicted_energy - dropped_energy
        return self.energy, drop_one_energy_diff


@dataclass(frozen=True)
class EnergyCalibration:
    """An energy calibration object that can convert pulse heights to (estimated) energies.

    Subclasses implement the math of either exact or approximating calibration curves.
    Methods allow you to convert between pulse heights and energies, estimate energy uncertainties,
    and estimate pulse heights for lines whose names are know, or estimate the cal curve slope.
    Methods allow you to plot the calibration curve with its anchor points.

    Returns
    -------
    EnergyCalibration

    Raises
    ------
    ValueError
        _description_
    """
    ph: np.ndarray[np.float64]
    energy: np.ndarray[np.float64]
    dph: np.ndarray[np.float64]
    de: np.ndarray[np.float64]
    names: list[str]
    curvename: str
    approximating: bool
    spline: Callable
    ph2uncertainty: Callable
    input_transform: Callable
    output_transform: Callable | None = None
    energy2ph: Callable | None = None

    def __post_init__(self):
        assert self.npts > 0

    def copy(self, **changes):
        return dataclasses.replace(self, **changes)

    @property
    def npts(self):
        return len(self.ph)

    @staticmethod
    def _ecal_input_identity(ph, der=0):
        "Use ph as the argument to the spline"
        assert der >= 0
        if der == 0:
            return ph
        elif der == 1:
            return np.ones_like(ph)
        return np.zeros_like(ph)

    @staticmethod
    def _ecal_input_log(ph, der=0):
        "Use log(ph) as the argument to the spline"
        assert der >= 0
        if der == 0:
            return np.log(ph)
        elif der == 1:
            return 1.0 / ph
        raise ValueError(f"der={der}, should be one of (0,1)")

    @staticmethod
    def _ecal_output_identity(ph, yspline, der=0, dery=0):
        "Use the spline result as E itself"
        assert der >= 0 and dery >= 0
        if der > 0:
            return np.zeros_like(ph)
        if dery == 0:
            return yspline
        elif dery == 1:
            return np.ones_like(ph)
        else:
            return np.zeros_like(ph)

    @staticmethod
    def _ecal_output_log(ph, yspline, der=0, dery=0):
        "Use the spline result as log(E)"
        assert der >= 0 and dery >= 0
        if der == 0:
            # Any order of d/dy equals E(y) itself, or exp(y).
            return np.exp(yspline)
        else:
            return np.zeros_like(ph)

    @staticmethod
    def _ecal_output_gain(ph, yspline, der=0, dery=0):
        "Use the spline result as gain = ph/E"
        assert der >= 0 and dery >= 0
        if dery == 0:
            if der == 0:
                return ph / yspline
            elif der == 1:
                return 1.0 / yspline
            else:
                return np.zeros_like(ph)
        assert dery == 1
        return -ph / yspline**2

    @staticmethod
    def _ecal_output_invgain(ph, yspline, der=0, dery=0):
        "Use the spline result as the inverse gain = E/ph"
        assert der >= 0 and dery >= 0
        if dery == 0:
            if der == 0:
                return ph * yspline
            elif der == 1:
                return yspline
            else:
                return np.zeros_like(ph)
        assert dery == 1
        return ph

    @staticmethod
    def _ecal_output_loggain(ph, yspline, der=0, dery=0):
        "Use the spline result as the log of the gain, or log(ph/E)"
        assert der >= 0 and dery >= 0
        if dery == 0:
            if der == 0:
                return ph * np.exp(-yspline)
            elif der == 1:
                return np.exp(-yspline)
            else:
                return np.zeros_like(ph)
        assert dery == 1
        return -ph * np.exp(-yspline)

    @property
    def ismonotonic(self):
        """Is the curve monotonic from 0 to 1.05 times the max anchor point's pulse height?
        Test at 1001 points, equally spaced in pulse height."""
        nsamples = 1001
        ph = np.linspace(0, 1.05 * self.ph.max(), nsamples)
        e = self(ph)
        return np.all(np.diff(e) > 0)

    def name2ph(self, name):
        """Convert a named energy feature to pulse height. `name` need not be a calibration point."""
        energy = STANDARD_FEATURES[name]
        return self.energy2ph(energy)

    def energy2dedph(self, energy):
        """Calculate the slope at energy."""
        return self.ph2dedph(self.energy2ph(energy))

    def energy2uncertainty(self, energies):
        """Cal uncertainty in eV at the given pulse heights."""
        ph = self.energy2ph(energies)
        return self.ph2uncertainty(ph)

    def __str__(self):
        seq = [f"EnergyCalibration({self.curvename})"]
        for name, pulse_ht, energy in zip(self.names, self.ph, self.energy):
            seq.append(f"  energy(ph={pulse_ht:7.2f}) --> {energy:9.2f} eV ({name})")
        return "\n".join(seq)

    def ph2energy(self, ph, exact=False):
        x = self.input_transform(ph)
        y = self.spline(x)
        E = self.output_transform(ph, y)
        return E

    __call__ = ph2energy

    def ph2dedph(self, ph):
        """Calculate the slope at pulse heights `ph`."""
        x = self.input_transform(ph)
        dgdP = self.input_transform(ph, der=1)
        dydx = self.spline(x, der=1)
        dEdP = dydx * dgdP
        if self.output_transform is not None:
            y = self.spline(x)
            dfdP = self.output_transform(ph, y, der=1)
            dfdy = self.output_transform(ph, y, dery=1)
            dEdP = dfdP + dfdy * dydx * dgdP
        return dEdP

    def energy2ph_exact(self, E):
        # TODO use the spline as a starting point for Brent's method
        return self.energy2ph(E)

    def save_to_hdf5(self, hdf5_group, name):
        if name in hdf5_group:
            del hdf5_group[name]

        cal_group = hdf5_group.create_group(name)
        cal_group["name"] = [str(n).encode() for n in self.names]
        cal_group["ph"] = self.ph
        cal_group["energy"] = self.energy
        cal_group["dph"] = self.dph
        cal_group["de"] = self.de
        cal_group.attrs['curvetype'] = self.curvename
        cal_group.attrs['approximate'] = self.approximating

    @staticmethod
    def load_from_hdf5(hdf5_group, name):
        cal_group = hdf5_group[name]

        # Fix a behavior of h5py for writing in py2, reading in py3.
        curvetype = cal_group.attrs['curvetype']
        if isinstance(curvetype, bytes):
            curvetype = curvetype.decode("utf-8")

        maker = EnergyCalibrationMaker(
            cal_group["ph"][:],
            cal_group["energy"][:],
            cal_group["dph"][:],
            cal_group["de"][:],
            cal_group["name"][:]
        )
        approximate = cal_group.attrs['approximate']
        return maker.make_calibration(curvetype, approximate=approximate)

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
        minph, maxph = self.ph.min() * .9, self.ph.max() * 1.1
        if min_energy is not None:
            minph = self.e2ph(min_energy)
        if max_energy is not None:
            maxph = self.e2ph(max_energy)
        phplot = np.linspace(minph, maxph, 1000)
        eplot = self(phplot)
        gplot = phplot / eplot
        dyplot = None
        gains = self.ph / self.energy
        slope = 0.0
        xplot = phplot
        x = self.ph
        xerr = self.dph
        if energy_x:
            xplot = eplot
            x = self.energy
            xerr = self.de

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
            if self.approximating:
                dyplot = self.ph2uncertainty(phplot) / (phplot**ph_rescale_power)
            y = self.energy / (self.ph**ph_rescale_power)
            if ph_rescale_power == 0.0:
                ylabel = "Energy (eV)"
                axis.set_title("Energy calibration curve")
            else:
                ylabel = f"Energy (eV) / PH^{ph_rescale_power:.4f}"
                axis.set_title(f"Energy calibration curve, scaled by {ph_rescale_power:.4f} power of PH")
        elif plottype == "gain":
            yplot = gplot
            if self.approximating:
                dyplot = self.ph2uncertainty(phplot) / eplot * gplot
            y = gains
            ylabel = "Gain (PH/eV)"
            axis.set_title("Energy calibration curve, gain")
        elif plottype == "invgain":
            yplot = 1.0 / gplot
            if self.approximating:
                dyplot = self.ph2uncertainty(phplot) / phplot
            y = 1.0 / gains
            ylabel = "Inverse Gain (eV/PH)"
            axis.set_title("Energy calibration curve, inverse gain")
        elif plottype == "loggain":
            yplot = np.log(gplot)
            if self.approximating:
                dyplot = self.ph2uncertainty(phplot) / eplot
            y = np.log(gains)
            ylabel = "Log Gain: log(eV/PH)"
            axis.set_title("Energy calibration curve, log gain")
        elif plottype == "loglog":
            yplot = np.log(eplot)
            xplot = np.log(phplot)
            if self.approximating:
                dyplot = self.ph2uncertainty(phplot) / eplot
            y = np.log(self.energy)
            x = np.log(self.ph)
            xerr = (self.dph / self.ph)
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
        dy = ((self.de / self.energy)**2 + (self.dph / self.ph)**2)**0.5 * y
        axis.errorbar(x, y - slope * x, yerr=dy, xerr=xerr, fmt='o',
                      mec='black', mfc=markercolor, capsize=0)
        axis.grid(True)
        if removeslope:
            ylabel = f"{ylabel} slope removed"
        axis.set_ylabel(ylabel)
        if showtext:
            for xval, name, yval in zip(x, self.names, y):
                axis.text(xval, yval - slope * xval, name + '  ', ha='right')


# Now a class like EnergyCalibration, but it can have attributes added for later use.
# This is maybe not a great interface by some measures, but the OFF analysis system assumes
# arbitrary attributes can be attached to a calibration, so here it is:
class EnergyCalibrationWithAttributes(EnergyCalibration):
    pass
