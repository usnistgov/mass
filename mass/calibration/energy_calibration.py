"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011

@author: fowlerj
"""

__all__ = ['EnergyCalibration']

import numpy as np
import scipy as sp
from ..mathstat.interpolate import *

# Some commonly-used standard energy features.
STANDARD_FEATURES = {
   'Gd1' :  97431.0,
   'Gd97':  97431.0,
   'Gd2':  103180.0,
   'Gd103':103180.0,
   'zero': 0.0,

   # The following Kalpha (alpha 1) and Kbeta (beta 1,3) line positions were
   # cross-checked 4 Feb 2014 against Deslattes.
   # Each is named to agree with the name in fluorescence_lines.
   'AlKAlpha': 1486.71, # __KAlpha refers to K Alpha 1
   'AlKBeta':  1557.6,
   'SiKAlpha': 1739.99,
   'SiKBeta':  1836.0,
   'CaKAlpha': 3691.72,
   'CaKBeta':  4012.76,
   'ScKAlpha': 4090.74,
   'TiKAlpha': 4510.90,
   'TiKBeta':  4931.83, #http://www.orau.org/ptp/PTP%20Library/library/ptp/x.pdf
   'VKAlpha':  4952.22,
   'VKBeta':   5426.962, # From L Smale, C Chantler, M Kinnane, J Kimpton, et al., Phys Rev A 87 022512 (2013). http://pra.aps.org/abstract/PRA/v87/i2/e022512
   'CrKAlpha' :5414.805,
   'CrKBeta':  5946.82,
   'MnKAlpha': 5898.801,
   'MnKBeta':  6490.59,
   'FeKAlpha': 6404.01,
   'FeKBeta':  7058.18,
   'CoKAlpha': 6930.378,
   'CoKBeta':  7649.45,
   'NiKAlpha': 7478.252,
   'NiKBeta':  8264.78,
   'CuKAlpha': 8047.823,
   'CuKBeta':  8905.41,

   'TiKEdge': 4966.0,
   'VKEdge':  5465.0, # defined as peak of derivative from exafs materials.com
   'CrKEdge': 5989.0,
   'MnKEdge': 6539.0,
   'FeKEdge': 7112.0,
   'CoKEdge': 7709.0,
   'NiKEdge': 8333.0,
   'CuKEdge': 8979.0,
   'ZnKEdge': 9659.0,
   # Randy's rare earth metals from Deslattes (Rev Mod Phys vol 75, 2003)
   'RhLl':     2376.55,
   'RhLAlpha2':2692.08,
   'RhLAlpha1':2696.76,
   'RhLBeta1': 2834.44,
   'RhLBeta3': 2915.7,
   'RhLBeta2': 3001.27,
   'RhLGamma1':3143.81,
   'NdLl':     4631.85,
   'NdLAlpha2':5207.7,
   'NdLAlpha1':5230.24,
   'NdLBeta1': 5721.45,
   'NdLBeta3': 5827.80,
   'NdLBeta2': 6091.25,
   'NdLGamma1':6601.16,
   'SmLl':     4990.43,
   'SmLAlpha2':5609.05,
   'SmLAlpha1':5635.97,
   'SmLBeta1': 6204.07,
   'SmLBeta3': 6316.36,
   'SmLBeta2': 6587.17,
   'SmLGamma1':7178.09,
   'TbLl':     5546.81,
   'TbLAlpha2':6238.10,
   'TbLAlpha1':6272.82,
   'TbLBeta1': 6977.80,
   'TbLBeta3': 7096.10,
   'TbLBeta2': 7366.70,
   'TbLGamma1':8101.80,
   'HoLl':     5939.96,
   'HoLAlpha2':6678.48,
   'HoLAlpha1':6719.68,
   'HoLBeta1': 7525.67,
   'HoLBeta3': 7651.8,
   'HoLBeta2': 7911.35,
   'HoLGamma1':8747.2,
}


class EnergyCalibration(object):
    """
    Object to store information relevant to one detector's absolute energy
    calibration and to offer conversions between pulse height and energy.

    The behavior is goverened by the constructor arguments `loglog`, `approximate`,
    and `zerozero` and by the number of data points. The construction-time arguments
    can be changed by calling EnergyCalibration.use_loglog() and similar.

    loglog -- Whether to spline in log(E) vs log(PH) space. Default: yes. If not,
              then splines are in E vs PH.
    approximate -- Whether to construct a smoothing spline (minimal curvature
            subject to a condition that chi-squared not be too large). If not,
            curve will be an exact spline in E vs PH or in log(E) vs log(PH).
    zerozero -- Only used when loglog==False. Whether to include the implicit
            points PH=0 and E=0 in the curve.

    The forward conversion from PH to E uses the callable __call__ method or its synonym,
    the method ph2energy.

    The inverse conversion method energy2ph calls Brent's method of root-finding.
    It's probably quite slow compared to a self.ph2energy for an array of equal length.

    All of __call__, ph2energy, and energy2ph should return a scalar when given a
    scalar input, or a matching numpy array when given any sequence as an imput.
    """

    def __init__(self, nonlinearity=1.1, loglog=True, approximate=True, zerozero=True):
        """
        Create an EnergyCalibration object for pulse-height-related field.

        `nonlinearity` is the exponent N in the default, low-energy limit of
        E \propto (PH)^N.  Typically 1.0 to 1.3 are reasonable.
        `loglog`  Whether to spline in log(E) vs log(PH) space. (If not, spline E vs PH.)
        `approximate`  Whether to use approximate "smoothing splines". (If not, use splines
                        that go exactly through the data.)
        `zerozero` Whether to force the cal curve to go through (0,0).
        """

        self._ph2energy_anon = lambda x: x
        self._ph = np.zeros(0, dtype=np.float)
        self._energies = np.zeros(0, dtype=np.float)
        self._dph = np.zeros(0, dtype=np.float)
        self._de = np.zeros(0, dtype=np.float)
        self._names = []
        self.npts = 0
        self.nonlinearity = nonlinearity
        self._use_approximation = approximate
        self._use_loglog = loglog
        self._use_zerozero = zerozero
        self._model_is_stale = False
        self._e2phwarned = False

    def __call__(self, pulse_ht):
        "Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV)."
        if self._model_is_stale:
            self._update_converters()
        return self._ph2energy_anon(pulse_ht)

    def ph2energy(self, pulse_ht):
        """Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV).
        This is a synonym for self.__call__(...). """
        if self._model_is_stale:
            self._update_converters()
        return self._ph2energy_anon(pulse_ht)

    def energy2ph(self, energy):
        """Convert a single energy `energy` in eV to a pulse height.
        Inverts the _ph2energy_anon function by Brent's method for root finding."""
        if self._model_is_stale:
            self._update_converters()
        energy_residual = lambda ph, etarget: self._ph2energy_anon(ph)-etarget
        if np.isscalar(energy):
            return sp.optimize.brentq(energy_residual, 1e-6, self._max_ph, args=(energy,))
        elif len(energy) > 10 and not self._e2phwarned:
            print("WARNING: EnergyCalibration.energy2ph can be slow for long inputs.")
            self._e2phwarned = False
        result = [sp.optimize.brentq(energy_residual, 1e-6, self._max_ph, args=(e,)) for e in energy]
        return np.array(result)

    def energy2dedph(self, energy, denergy=1):
        """Calculate the slope between <energy-denergy> and <energy>+<denergy> with two points.
        """
        loe, hie = energy-denergy, energy+denergy
        loph, hiph = self.energy2ph(loe), self.energy2ph(hie)
        return (hie-loe)/(hiph-loph)

    def __str__(self):
        seq = ["EnergyCalibration()"]
        for name, pulse_ht, energy in zip(self._names, self._ph, self._energies):
            seq.append("  energy(ph=%7.2f) --> %9.2f eV (%s)" % (pulse_ht, energy, name))
        return "\n".join(seq)

    def set_use_approximation(self, useit):
        """Switch to using (or to NOT using) approximating splines with
        reduced knot count. You can interchange this with adding points, because
        the actual model computation isn't done until the cal curve is called."""
        if useit != self._use_approximation:
            self._use_approximation = useit
            self._model_is_stale = True

    def set_use_loglog(self, useit):
        """Switch to using (or to NOT using) splines in log(PH) vs log(E) space."""
        if useit != self._use_loglog:
            self._use_loglog = useit
            self._model_is_stale = True

    def set_use_zerozero(self, useit):
        """Switch to using (or to NOT using) (PH,E)=(0,0) as an implied cal point."""
        if useit != self._use_zerozero:
            self._use_zerozero = useit
            self._model_is_stale = True

    def copy(self, new_ph_field=None):
        """Return a deep copy"""
        ecal = EnergyCalibration()
        ecal.__dict__.update(self.__dict__)
        ecal._names = list(self._names)
        ecal._ph = self._ph.copy()
        ecal._energies = self._energies.copy()
        ecal._dph = self._dph.copy()
        ecal._de = self._de.copy()
        ecal._use_loglog = self._use_loglog
        ecal._use_approximation = self._use_approximation
        ecal._use_zerozero = self._use_zerozero
        ecal._model_is_stale = True
        return ecal

    def _remove_cal_point_idx(self, idx):
        "Remove calibration point number `idx` from the calibration."
        self._names.pop(idx)
        self._ph = np.hstack((self._ph[:idx], self._ph[idx+1:]))
        self._energies = np.hstack((self._energies[:idx], self._energies[idx+1:]))
        self._dph = np.hstack((self._dph[:idx], self._dph[idx+1:]))
        self._de = np.hstack((self._de[:idx], self._de[idx+1:]))
        self.npts -= 1
        self._model_is_stale = True

    def remove_cal_point_name(self, name):
        "If you don't like calibration point named <name>, this removes it"
        idx = self._names.index(name)
        self.__remove_cal_point_idx(idx)

    def remove_cal_point_prefix(self, prefix):
        """This removes all cal points whose name starts with <prefix>.  Return number removed."""
        for name in tuple(self._names):
            if name.startswith(prefix):
                self.remove_cal_point_name(name)

    def remove_cal_point_energy(self, energy, de):
        "Remove cal points at energies with <de> of <energy>"
        idxs = np.nonzero(np.abs(self._energies-energy)<de)[0]
        for idx in idxs:
            self.__remove_cal_point_idx(idx)

    def add_cal_point(self, pht, energy, name="", pht_error=None, e_error=None, overwrite=True):
        """
        Add a single energy calibration point <pht>, <energy>, where <pht> must be in units
        of the self.ph_field and <energy> is in eV.  <pht_error> is the 1-sigma uncertainty
        on the pulse height.  If None (the default), then assign pht_error = <pht>/1000.
        <e_error> is the 1-sigma uncertainty on the energy itself. If None (the default),
        then assign e_error=<energy>/10^5 (typically 0.05 eV).

        Also, you can call it with <energy> as a string, provided it's the name of a known
        feature appearing in the dictionary mass.energy_calibration.STANDARD_FEATURES.  Thus
        the following are equivalent:

        cal.add_cal_point(12345.6, 5898.801, "Mn Ka1")
        cal.add_cal_point(12456.6, "Mn Ka1")

        Careful!  If you give a name that's already in the list, then this value replaces
        the previous one.  If you do NOT give a name, though, then this will NOT replace
        but will add to any existing points at the same energy.  You can prevent overwriting
        by setting <overwrite>=False.
        """
        self._model_is_stale = True

        # If <energy> is a string and a known spectral feature's name, use it as the name instead
        # Otherwise, it needs to be a numeric type convertable to float.
        if energy in STANDARD_FEATURES:
            name = energy
            energy = STANDARD_FEATURES[name]
        else:
            try:
                energy = float(energy)
            except ValueError:
                raise ValueError("2nd argument must be an energy or a known name"+
                                 " from mass.energy_calibration.STANDARD_FEATURES")

        if pht_error is None:
            pht_error = pht*0.001
        if e_error is None:
            e_error = 0.01 # Assume 0.01 eV error if none given

        if name != "" and name in self._names:  # Update an existing point
            if not overwrite:
                raise ValueError("Calibration point '%s' is already known and overwrite is False" % name)
            index = self._names.index(name)
            self._ph[index] = pht
            self._energies[index] = energy
            self._dph[index] = pht_error
            self._de[index] = e_error

        else:   # Add a new point
            self._ph = np.hstack((self._ph, pht))
            self._energies = np.hstack((self._energies, energy))
            self._dph = np.hstack((self._dph, pht_error))
            self._de = np.hstack((self._de, e_error))
            self._names.append(name)

        # Sort in ascending energy order
        sortkeys = np.argsort(self._ph)
        self._ph = self._ph[sortkeys]
        self._energies = self._energies[sortkeys]
        self._dph = self._dph[sortkeys]
        self._de = self._de[sortkeys]
        self._names = [self._names[s] for s in sortkeys]
        self.npts = len(self._names)
        assert self.npts == len(self._ph)
        assert self.npts == len(self._dph)
        assert self.npts == len(self._energies)
        assert self.npts == len(self._de)

    def _update_converters(self):
        """There is now one (or more) new data points. All the math goes on in this method."""
        assert len(self._ph)==len(self._energies)
        assert len(self._ph)==self.npts
        self._max_ph = 20*np.max(self._ph)
        if self._use_approximation and self.npts > 3:
            self._update_approximators()
        else:
            self._update_exactcurves()

        self._model_is_stale = False

    def _update_approximators(self):
        # Make sure the errors in both dimensions are reasonable (positive)
        if (self._dph <= 0.0).any():
            if (self._dph > 0).any():
                self._dph[self._dph<=0.0] = self._dph[self._dph>0].min()
            else:
                self._dph = np.zeros_like(self._dph)
        if (self._de <= 0.0).any():
            if (self._de > 0).any():
                self._de[self._de<=0.0] = self._de[self._de>0].min()
            else:
                self._de = np.zeros_like(self._de)

        # Find transformed data. For dy, assume that E and PH errors are uncorrelated.
        ph, dph, e, de = self._ph, self._dph, self._energies, self._de

        if self._use_loglog:
            self._ph2energy_anon = SmoothingSplineLog(ph, e, de, dph)
        else:
            if self._use_zerozero and (0.0 not in ph):
                ph = np.hstack([[0],ph])
                e  = np.hstack([[0],e])
                de = np.hstack([[de.min()*0.1],de])
                dph= np.hstack([[dph.min()*0.1],dph])
            self._ph2energy_anon = SmoothingSpline(ph, e, de, dph)

    def _update_exactcurves(self):
        """Update the E(P) curve assume exact interpolation of calibration data."""
        # Choose proper curve/interpolating function object
        # For N=0 points, we just let E = PH
        # For N=1 points, use a power law of the assumed nonlinearity
        # For N=2 points, use a power law, updating nonlinearity to be the actual value.
        if self.npts <= 0:
            self._ph2energy_anon = lambda p: p
        elif self.npts == 1:
            p1 = self._ph[0]
            e1 = self._energies[0]
            self._ph2energy_anon = lambda p: e1*(p/p1)**self.nonlinearity
        elif self.npts == 2:
            p1,p2 = self._ph
            e1,e2 = self._energies
            self.nonlinearity = np.log(e2/e1) / np.log(p2/p1)
            self._ph2energy_anon = lambda p: e1*(p/p1)**self.nonlinearity
        else:
            if self._use_loglog:
                x = np.log(self._ph)
                y = np.log(self._energies)
                self._x2yfun = CubicSpline(x, y)
                self._ph2energy_anon = lambda p: np.exp(self._x2yfun(np.log(p)))
            elif self._use_zerozero:
                x = np.hstack(([0], self._ph))
                y = np.hstack(([0], self._energies))
                self._ph2energy_anon = CubicSpline(x, y)
            else:
                x = self._ph
                y = self._energies
                self._ph2energy_anon = CubicSpline(x, y)

    def name2ph(self, name):
        """Convert a named energy feature to pulse height"""
        energy = STANDARD_FEATURES[name]
        return self.energy2ph(energy)


    def plot(self, axis=None, ph_rescale_power=0.0, color='blue', markercolor='red'):
        """Plot the energy calibration function using pylab.  If <axis> is None,
        a new pylab.subplot(111) will be used.  Otherwise, axis should be a
        pylab.Axes object to plot onto.

        <ph_rescale_power>   Plot E/PH**ph_rescale_power vs PH.  Default is 0, so plot E vs PH.
        """
        import pylab
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
            axis.set_ylim([0, self._energies.max()*1.05/self._ph.max()**ph_rescale_power])
            axis.set_xlim([0, self._ph.max()*1.1])

        # Plot smooth curve
        pht = np.arange(0, self._ph.max()*1.1)
        y = self(pht) / pht**ph_rescale_power
        axis.plot(pht, y, color=color)

        # Plot and label cal points
        if ph_rescale_power==0.0:
            axis.errorbar(self._ph, self._energies, yerr=self._de, xerr=self._dph, fmt='o',
                          mec='black', mfc=markercolor, capsize=0)
        else:
            yscale = 1.0/(self._ph**ph_rescale_power)
            dy = np.sqrt(self._de**2 + (self._dph*ph_rescale_power)**2)
            axis.errorbar(self._ph, self._energies*yscale, yerr=dy*yscale,
                        xerr=self._dph, fmt='or', capsize=0)
        for pht, name in zip(self._ph, self._names):
            axis.text(pht, self(pht)/pht**ph_rescale_power, name+'  ', ha='right')

        axis.grid(True)
        axis.set_xlabel("Pulse height")
        if ph_rescale_power == 0.0:
            axis.set_ylabel("Energy (eV)")
            axis.set_title("Energy calibration curve")
        else:
            axis.set_ylabel("Energy (eV) / PH^%.4f"%ph_rescale_power)
            axis.set_title("Energy calibration curve, scaled by %.4f power of PH"%ph_rescale_power)

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
