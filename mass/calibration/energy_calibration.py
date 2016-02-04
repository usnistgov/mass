"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011

@author: fowlerj
"""

__all__ = ['EnergyCalibration']

import numpy as np
import scipy as sp
from mass.mathstat.interpolate import *

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

    The conversion function depends on the number of calibration points known so
    far.  The point (ph=0, energy=0) is always known.  If there is one additional
    non-trivial point known, then the conversion is linear.  If there are two
    non-trivial points known, then the conversion is quadratic.  If there are at
    least three non-trivial points known, then the conversion is a smoothing
    cubic spline.  (Adjust self.smooth if you don't like the default value of
    1 eV smoothing).

    The inverse conversion self.energy2ph **for now** works only on a scalar energy.
    It calls Brent's method of root-finding.  Fix this method if you find that you
    need to solve for vectors of energy->PH conversions.
    """

    def __init__(self, nonlinearity=1.1):
        """
        Create an EnergyCalibration object for pulse-height-related field.

        <nonlinearity> is the exponent N in the default, low-energy limit of
        E \propto (PH)^N.  Typically 1.0 to 1.3 are reasonable.
        """

        self.ph2energy = lambda x: x
        self.energy2ph = lambda x: x
        self.info = [{}]
        self._ph = np.zeros(0, dtype=np.float)
        self._energies = np.zeros(0, dtype=np.float)
        self._dph = np.zeros(0, dtype=np.float)
        self._de = np.zeros(0, dtype=np.float)
        self._names = []
        self.npts = 0
        self.nonlinearity = nonlinearity
        self._use_approximation = True
        self._model_is_stale = False
        self._use_loglog = True

    def __call__(self, pulse_ht):
        "Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV)."
        if self._model_is_stale:
            self._update_converters()
        return self.ph2energy(pulse_ht)

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

    def copy(self, new_ph_field=None):
        """Return a deep copy"""
        ecal = EnergyCalibration()
        ecal.__dict__.update(self.__dict__)
        ecal._names = list(self._names)
        ecal._ph = self._ph.copy()
        ecal._energies = self._energies.copy()
        ecal._dph = self._dph.copy()
        ecal._de = self._de.copy()
        ecal._model_is_stale = True
        return ecal

    def remove_cal_point_name(self, name):
        "If you don't like calibration point named <name>, this removes it"
        idx = self._names.index(name)
        self._names.pop(idx)
        self._ph = np.hstack((self._ph[:idx], self._ph[idx+1:]))
        self._energies = np.hstack((self._energies[:idx], self._energies[idx+1:]))
        self._dph = np.hstack((self._dph[:idx], self._dph[idx+1:]))
        self._de = np.hstack((self._de[:idx], self._de[idx+1:]))
        self.npts -= 1
        self._model_is_stale = True

    def remove_cal_point_prefix(self, prefix):
        """This removes all cal points whose name starts with <prefix>.  Return number removed."""
        for name in tuple(self._names):
            if name.startswith(prefix):
                self.remove_cal_point_name(name)

    def add_cal_point(self, pht, energy, name="", info={}, pht_error=None, e_error=None, overwrite=True):
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
        info['name']=name
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
            self.info[index] = info.copy()

        else:   # Add a new point
            self._ph = np.hstack((self._ph, pht))
            self._energies = np.hstack((self._energies, energy))
            self._dph = np.hstack((self._dph, pht_error))
            self._de = np.hstack((self._de, e_error))
            self._names.append(name)
            self.info.append(info.copy())

        # Sort in ascending energy order
        sortkeys = np.argsort(self._ph)
        self._ph = self._ph[sortkeys]
        self._energies = self._energies[sortkeys]
        self._dph = self._dph[sortkeys]
        self._de = self._de[sortkeys]
        self._names = [self._names[s] for s in sortkeys]
        self.info = [self.info[s] for s in sortkeys]
        self.npts = len(self._names)
        assert self.npts == len(self._ph)
        assert self.npts == len(self._dph)
        assert self.npts == len(self._energies)
        assert self.npts == len(self._de)
        assert self.npts == len(self.info)


    def _update_converters(self):
        """There is now one (or more) new data points. All the math goes on in this method."""
        assert len(self._ph)==len(self._energies)
        assert len(self._ph)==self.npts
        self._max_ph = 20*np.max(self._ph)
        if self._use_approximation and self.npts > 3:
            self._update_approximators()
        else:
            self._update_exactcurves()

        # The inverse function is found numerically, with a root-finder.
        energy_residual = lambda ph, etarget: self.ph2energy(ph)-etarget
        self.energy2ph = lambda e: sp.optimize.brentq(energy_residual, 1e-6, self._max_ph, args=(e,))
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
            self.ph2energy = SmoothingSplineLog(ph, e, de, dph)
        else:
            if 0.0 not in ph:
                ph = np.hstack([[0],ph])
                e  = np.hstack([[0],e])
                de = np.hstack([[1e-2],de])
                dph= np.hstack([[1e-2],dph])
            self.ph2energy = SmoothingSpline(ph, e, de, dph)


    def _update_exactcurves(self):
        """Update the E(P) curve assume exact interpolation of calibration data."""
        # Choose proper curve/interpolating function object
        # For N=0 points, we just let E = PH
        # For N=1 points, use a power law of the assumed nonlinearity
        # For N=2 points, use a power law, updating nonlinearity to be the actual value.
        if self.npts <= 0:
            self.ph2energy = lambda p: p
        elif self.npts == 1:
            p1 = self._ph[0]
            e1 = self._energies[0]
            self.ph2energy = lambda p: e1*(p/p1)**self.nonlinearity
        elif self.npts == 2:
            p1,p2 = self._ph
            e1,e2 = self._energies
            self.nonlinearity = np.log(e2/e1) / np.log(p2/p1)
            self.ph2energy = lambda p: e1*(p/p1)**self.nonlinearity
        else:
            if self._use_loglog:
                x = np.log(self._ph)
                y = np.log(self._energies)
                self._x2yfun = CubicSpline(x, y)
                self.ph2energy = lambda p: np.exp(self._x2yfun(np.log(p)))
            else:
                x = np.hstack(([0], self._ph))
                y = np.hstack(([0], self._energies))
                self.ph2energy = CubicSpline(x, y)


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



class EnergyFeature(object):
    """
    Honestly, I don't know what this is or whether it's used.  It
    appears to be unfinished, so I'll raise an error....
    """

    def __init__(self, name, energy, **kwargs):
        self.name = name
        self.energy = energy
        self.phs = {}
        self.phs.update(kwargs)
        raise NotImplementedError("EnergyFeature looks incomplete to me (Joe 11/8/11)")

    def set_val(self, feature_name, value):
        """Associate <value> with feature name <feature_name>.  ???"""
        if feature_name in self.phs:
            self.phs['%s_prev' % feature_name] = self.phs[feature_name]
        self.phs[feature_name] = value

    def copy(self):
        """Return a deep copy"""
        feature = EnergyFeature(self.name, self.energy)
        feature.__dict__.update(self.__dict__)
        return feature

    def __str__(self):
        s = ['Energy feature %s at %8.3f eV' % (self.name, self.energy)]
        for k, v in self.phs.iteritems():
            s.append("    '%-20s': %9.3f" % (k, v))
        return "\n".join(s)

    def __repr__(self):
        s = ["EnergyFeature('%s',%f," % (self.name, self.energy)]
        for k, v in self.phs.iteritems():
            s.append("%s=%f" % (k, v))
        s.append(")")
        return " ".join(s)



def make_energy_feature(name):
    """
    Factory function to make EnergyFeature objects of known
    energy (and standard names).
    """

    if name in STANDARD_FEATURES:
        return EnergyFeature(name, STANDARD_FEATURES[name])

    raise ValueError("Known energy features are:" % STANDARD_FEATURES.keys())
