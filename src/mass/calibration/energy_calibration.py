"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011
"""
import six

import numpy as np
from scipy.optimize import brentq

from mass.mathstat.interpolate import *
from mass.mathstat.derivative import *
from mass.calibration.nist_xray_database import NISTXrayDBFile


def LineEnergies():
    """
    A dictionary to know a lot of x-ray fluorescence line energies, based on Deslattes' database.

    It is built on facts from mass.calibration.nist_xray_database module.

    It is a dictionary from peak name to energy, with several alternate names
    for the lines:

    >>> E = Energies()
    >>> print E["MnKAlpha"]
    >>> print E["MnKAlpha"], E["MnKA"], E["MnKA1"], E["MnKL3"]
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


class EnergyCalibration(object):
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

    def __init__(self, nonlinearity=1.1, curvetype="loglog", approximate=False):
        """Create an EnergyCalibration object for pulse-height-related field.

        Args:
            nonlinearity (float): the exponent N in the default, low-energy limit of
                E propto (PH)^N.  Typically 1.0 to 1.3 are reasonable.
            curvetype (str or int): one of EnergyCalibration.CURVETYPE.
            approximate (boolean):  Whether to use approximate "smoothing splines". (If not, use splines
                that go exactly through the data.) Default = False, because this works
                poorly unless the user calls with sensible PH and Energy uncertainties.
        """
        self._curvetype = 0
        self.set_curvetype(curvetype)
        self._ph2energy_anon = self.__default_ph2energy_anon
        self._ph = np.zeros(0, dtype=np.float)
        self._energies = np.zeros(0, dtype=np.float)
        self._dph = np.zeros(0, dtype=np.float)
        self._de = np.zeros(0, dtype=np.float)
        self._names = []
        self.npts = 0
        self._use_approximation = approximate
        self._model_is_stale = False
        self._e2phwarned = False
        self.nonlinearity = nonlinearity
        self.set_nonlinearity()

    @staticmethod
    def __default_ph2energy_anon(x, der=0):
        if der > 0:
            return np.zeros_like(x)
        return x

    def __call__(self, pulse_ht, der=0):
        return self.ph2energy(pulse_ht, der=der)

    def ph2energy(self, pulse_ht, der=0):
        """Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV).
        Should return a scalar if passed a scalar, and a numpy array if passed a list or array

        Args:
            pulse_ht (float or numpy.array(dtype=numpy.float)): pulse heights in an arbitrary unit.
            der (int): the order of derivative. `der` should be >= 0.
        """
        if self._model_is_stale:
            self._update_converters()
        result = self._ph2energy_anon(pulse_ht, der=der)

        if np.isscalar(pulse_ht):
            if np.isscalar(result):
                return result
            return np.asscalar(result)

        # Change any inf or NaN results to 0.0 energy.
        if any(np.isnan(result)):
            result[np.isnan(result)] = 0.0
        if any(np.isinf(result)):
            result[np.isinf(result)] = 0.0
        return np.array(result)

    def energy2ph(self, energy):
        """Convert a single energy `energy` in eV to a pulse height.
        Inverts the _ph2energy_anon function by Brent's method for root finding.
        Should return a scalar if passed a scalar, and a numpy array if passed a list or array.
        """
        if self._model_is_stale:
            self._update_converters()

        def energy_residual(ph, etarget):
            return self._ph2energy_anon(ph) - etarget

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

    def energy2dedph(self, energy):
        """Calculate the slope at energy."""
        ph = self.energy2ph(energy)
        return self(ph, der=1)

    def __str__(self):
        self._update_converters()  # To sort the points
        seq = ["EnergyCalibration()"]
        for name, pulse_ht, energy in zip(self._names, self._ph, self._energies):
            seq.append("  energy(ph=%7.2f) --> %9.2f eV (%s)" % (pulse_ht, energy, name))
        return "\n".join(seq)

    def set_nonlinearity(self, powerlaw=1.15):
        """Update the power law index assumed when there's 1 data point and a loglog curve type."""
        if self.curvename() == "loglog" and powerlaw != self.nonlinearity:
            self.nonlinearity = powerlaw
            self._model_is_stale = True

    def set_use_approximation(self, useit):
        """Switch to using (or to NOT using) approximating splines with
        reduced knot count. You can interchange this with adding points, because
        the actual model computation isn't done until the cal curve is called."""
        if useit != self._use_approximation:
            self._use_approximation = useit
            self._model_is_stale = True

    def set_curvetype(self, curvetype):
        if isinstance(curvetype, six.string_types):
            try:
                curvetype = self.CURVETYPE.index(curvetype.lower())
            except ValueError:
                raise ValueError("EnergyCalibration.CURVETYPE does not contain '%s'" % curvetype)
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
        ecal._model_is_stale = True
        ecal._curvetype = self._curvetype
        return ecal

    def _remove_cal_point_idx(self, idx):
        """Remove calibration point number `idx` from the calibration."""
        self._names.pop(idx)
        self._ph = np.hstack((self._ph[:idx], self._ph[idx+1:]))
        self._energies = np.hstack((self._energies[:idx], self._energies[idx+1:]))
        self._dph = np.hstack((self._dph[:idx], self._dph[idx+1:]))
        self._de = np.hstack((self._de[:idx], self._de[idx+1:]))
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
        idxs = np.nonzero(np.abs(self._energies-energy) < de)[0]

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
                raise ValueError("2nd argument must be an energy or a known name" +
                                 " from mass.energy_calibration.STANDARD_FEATURES")

        if pht_error is None:
            pht_error = pht*0.001
        if e_error is None:
            e_error = 0.01  # Assume 0.01 eV error if none given

        update_index = None
        if name != "" and name in self._names:  # Update an existing point by name
            if not overwrite:
                raise ValueError(
                    "Calibration point '%s' is already known and overwrite is False" % name)
            update_index = self._names.index(name)

        elif self.npts > 0 and np.abs(energy-self._energies).min() <= e_error:  # Update existing point
            if not overwrite:
                raise ValueError(
                    "Calibration point at energy %.2f eV is already known and overwrite is False" % energy)
            update_index = np.abs(energy-self._energies).argmin()

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

    @property
    def cal_point_phs(self):
        return self._ph

    @property
    def cal_point_energies(self):
        return self._energies

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

        self._max_ph = 2*np.max(self._ph)
        if self._use_approximation and self.npts >= 3:
            self._update_approximators()
        else:
            self._update_exactcurves()
        self._model_is_stale = False

    def _update_approximators(self):
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
            cubic_spline = SmoothingSplineFunction(np.log(ph), np.log(e), de/e, dph/ph)
            self._ph2energy_anon = ExponentialFunction() << cubic_spline << LogFunction()

        elif self.curvename().startswith("linear"):
            if ("+0" in self.curvename()) and (0.0 not in ph):
                ph = np.hstack([[0], ph])
                e = np.hstack([[0], e])
                de = np.hstack([[de.min()*0.1], de])
                dph = np.hstack([[dph.min()*0.1], dph])
            self._ph2energy_anon = SmoothingSplineFunction(ph, e, de, dph)

        elif self.curvename() == "gain":
            g = ph/e
            scale = ph.mean()
            dg = g * ((dph/ph)**2+(de/e)**2)**0.5
            # self._underlying_spline = SmoothingSpline(ph/scale, g, dg, dph/scale)
            # self._ph2energy_anon = lambda p: p/self._underlying_spline(p/scale)
            p = Identity()
            underlying_spline = SmoothingSplineFunction(ph / scale, g, dg, dph / scale)
            self._ph2energy_anon = p / (underlying_spline << (p / scale))

            # Gain curves have a problem: gain<0 screws it all up. Avoid that region.
            trial_phmax = 10 * self._ph.max()
            if underlying_spline(trial_phmax) > 0:
                self._max_ph = trial_phmax
            else:
                self._max_ph = scale*0.99*brentq(underlying_spline, self._ph.max()/scale, trial_phmax/scale)

        elif self.curvename() == "invgain":
            ig = e/ph
            scale = ph.mean()
            dg = ig * ((dph/ph)**2+(de/e)**2)**0.5
            # self._underlying_spline = SmoothingSpline(ph/scale, ig, dg, dph/scale)
            # self._ph2energy_anon = lambda p: p*self._underlying_spline(p/scale)
            p = Identity()
            underlying_spline = SmoothingSplineFunction(ph / scale, ig, dg, dph / scale)
            self._ph2energy_anon = p * (underlying_spline << (p / scale))

        elif self.curvename() == "loggain":
            lg = np.log(ph/e)
            dlg = ((dph/ph)**2+(de/e)**2)**0.5
            scale = ph.mean()
            # self._underlying_spline = SmoothingSpline(ph/scale, lg, dlg, dph/scale)
            # self._ph2energy_anon = lambda p: p*np.exp(-self._underlying_spline(p/scale))
            p = Identity()
            underlying_spline = SmoothingSplineFunction(ph / scale, lg, dlg, dph / scale)
            self._ph2energy_anon = p * (ExponentialFunction() << (-underlying_spline << (p / scale)))

    def _update_exactcurves(self):
        """Update the E(P) curve assume exact interpolation of calibration data."""
        # Choose proper curve/interpolating function object
        # For N=0 points, in the absence of any information at all, we just let E = PH.
        # For N=1 points, use E proportional to PH (or if loglog curve, then a power law of
        #    the assumed nonlinearity).
        # For N>1 points, use the chosen curve type (but for N=2, recall that the spline will be a line).
        if self.npts <= 0:
            # self._ph2energy_anon = lambda p: p
            self._ph2energy_anon = Identity()

        elif self.npts == 1:
            p1 = self._ph[0]
            e1 = self._energies[0]
            # self._ph2energy_anon = lambda p: e1*(p/p1)**self.nonlinearity
            # p_over_scale = Multiplication(ConstantFunction(1.0 / p1), Identity())
            # power_p = Composition(PowerFunction(self.nonlinearity), p_over_scale)
            self._ph2energy_anon = e1 * (Identity() / p1)**self.nonlinearity

        elif self.npts == 2:
            p1, p2 = self._ph
            e1, e2 = self._energies
            self.nonlinearity = np.log(e2/e1) / np.log(p2/p1)

            # self._ph2energy_anon = lambda p: e1*(p/p1)**self.nonlinearity
            # p_over_scale = Multiplication(ConstantFunction(1.0 / p1), Identity())
            # power_p = Composition(PowerFunction(self.nonlinearity), p_over_scale)
            self._ph2energy_anon = e1 * (Identity() / p1)**self.nonlinearity

        elif self.curvename() == "loglog":
            x = np.log(self._ph)
            y = np.log(self._energies)
            # self._x2yfun = CubicSpline(x, y)
            # self._ph2energy_anon = lambda p: np.exp(self._x2yfun(np.log(p)))
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2energy_anon = ExponentialFunction() << underlying_spline << LogFunction()

        elif self.curvename().startswith("linear"):
            x = self._ph
            y = self._energies
            if ("+0" in self.curvename()) and (0.0 not in x):
                x = np.hstack(([0], x))
                y = np.hstack(([0], y))
            # self._ph2energy_anon = CubicSpline(x, y)
            self._ph2energy_anon = CubicSplineFunction(x, y)

        elif self.curvename() == "gain":
            x = self._ph
            y = x / self._energies
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2energy_anon = lambda p: p/self._underlying_spline(p)
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2energy_anon = Identity() / underlying_spline

            # Gain curves have a problem: gain<0 screws it all up. Avoid that region.
            trial_phmax = 10*self._ph.max()
            if underlying_spline(trial_phmax) > 0:
                self._max_ph = trial_phmax
            else:
                self._max_ph = 0.99 * brentq(underlying_spline, 0, trial_phmax)

        elif self.curvename() == "invgain":
            x = self._ph
            y = self._energies/x
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2energy_anon = lambda p: p*self._underlying_spline(p)
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2energy_anon = Identity() * underlying_spline

        elif self.curvename() == "loggain":
            x = self._ph
            y = np.log(x / self._energies)
            # self._underlying_spline = CubicSpline(x, y)
            # self._ph2energy_anon = lambda p: p*np.exp(-self._underlying_spline(p))
            underlying_spline = CubicSplineFunction(x, y)
            self._ph2energy_anon = Identity() * (ExponentialFunction() << -underlying_spline)

    def name2ph(self, name):
        """Convert a named energy feature to pulse height"""
        energy = STANDARD_FEATURES[name]
        return self.energy2ph(energy)

    def plot(self, axis=None, ph_rescale_power=0.0, color="blue", markercolor="red"):
        self._plot(axis, color, markercolor, plottype="linear", ph_rescale_power=ph_rescale_power)

    def plotgain(self, axis=None, color="blue", markercolor="red"):
        self._plot(axis, color, markercolor, plottype="gain")

    def plotinvgain(self, axis=None, color="blue", markercolor="red"):
        self._plot(axis, color, markercolor, plottype="invgain")

    def plotloggain(self, axis=None, color="blue", markercolor="red"):
        self._plot(axis, color, markercolor, plottype="loggain")

    def _plot(self, axis=None, color="blue", markercolor="red", plottype="gain", ph_rescale_power=0.0):
        import pylab
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            axis.set_xlim([0, self._ph.max()*1.1])

        # Plot smooth curve
        pht = np.linspace(self._ph.max()*.001, self._ph.max()*1.1, 1000)
        smoothgain = pht / self(pht)
        gains = self._ph / self._energies
        if plottype == "linear":
            if ph_rescale_power == 0.0:
                axis.plot(pht, self(pht), color=color)
                y = self._energies
                axis.set_ylabel("Energy (eV)")
                axis.set_title("Energy calibration curve")
            else:
                axis.plot(pht, self(pht) / (pht**ph_rescale_power), color=color)
                y = self._energies / (self._ph**ph_rescale_power)
                axis.set_ylabel("Energy (eV) / PH^%.4f" % ph_rescale_power)
                axis.set_title("Energy calibration curve, scaled by %.4f power of PH" % ph_rescale_power)
        elif plottype == "gain":
            axis.plot(pht, smoothgain, color=color)
            y = gains
            axis.set_ylabel("Gain (PH/eV)")
            axis.set_title("Energy calibration curve, gain")
        elif plottype == "invgain":
            axis.plot(pht, 1.0/smoothgain, color=color)
            y = 1.0/gains
            axis.set_ylabel("Inverse Gain (eV/PH)")
            axis.set_title("Energy calibration curve, inverse gain")
        elif plottype == "loggain":
            axis.plot(pht, np.log(smoothgain), color=color)
            y = np.log(gains)
            axis.set_ylabel("Log Gain: log(eV/PH)")
            axis.set_title("Energy calibration curve, log gain")
        else:
            raise ValueError("plottype must be one of ('linear', 'gain','loggain','invgain').")
        dy = ((self._de/self._energies)**2 + (self._dph/self._ph)**2)**0.5 * y

        # Plot and label cal points
        axis.errorbar(self._ph, y, yerr=dy, xerr=self._dph, fmt='o',
                      mec='black', mfc=markercolor, capsize=0)
        axis.grid(True)
        axis.set_xlabel("Pulse height")
        for pht, name, yval in zip(self._ph, self._names, y):
            axis.text(pht, yval, name+'  ', ha='right')


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
            drop_one_energy_diff[i] = predicted_energy-drop_one_energy
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

        _names = cal_group["name"].value
        _ph = cal_group["ph"].value
        _energies = cal_group["energy"].value
        _dph = cal_group["dph"].value
        _de = cal_group["de"].value

        for thisname, ph, e, dph, de in zip(_names, _ph, _energies, _dph, _de):
            cal.add_cal_point(ph, e, thisname.decode(), dph, de)

        return cal
