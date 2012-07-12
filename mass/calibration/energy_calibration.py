"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011

@author: fowlerj
"""

__all__ = ['EnergyCalibration']

import numpy
import scipy.interpolate
import scipy.optimize

# Some commonly-used standard energy features.
STANDARD_FEATURES = {
   'Al Ka': 1486.35,
   'Al Kb': 1557.,
   'Si Ka': 1739.6,
   'Si Kb': 1837.,
   'Mn Ka1': 5898.802,
   'Mn Ka2': 5887.592,
   'Cr Kedge': 5989.0,
   'Mn Kb':  6490.18,
   'Fe Kedge': 7112.0,
   'Cu Ka': 8047.83,
   'Cu Kedge': 8979.0,
   'Gd1' :  97431.0,
   'Gd97':  97431.0,
   'Gd2':  103180.0,
   'Gd103':103180.0,
   'zero': 0.0,
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
    
    def __init__(self, ph_field, spline=True):
        """Create an EnergyCalibration object for pulse-height-related field named <ph_field>.
        <spline>=True uses quadratic for 3 points and approximating splines for 4+ points.
        <spline>=False uses exact linear interpolation between points."""
        self.ph_field = ph_field
        self.ph2energy = lambda x: x
        self.energy2ph = lambda x: x
        self._ph = numpy.zeros(1, dtype=numpy.float)
        self._energies = numpy.zeros(1, dtype=numpy.float)
        self._stddev = numpy.zeros(1, dtype=numpy.float)
        self._names = ['null']
        self.npts = 1
        self.smooth = 1.0  # This ought to make the curve stay within ~1-sigma of each point.
        self.use_spline = spline
        
    def __call__(self, pulse_ht):
        "Convert pulse height (or array of pulse heights) <pulse_ht> to energy (in eV)."
        return self.ph2energy(pulse_ht)
    
    def __repr__(self):
        return "EnergyCalibration('%s')" % self.ph_field
    
    def __str__(self):
        seq = ["EnergyCalibration('%s')" % self.ph_field]
        for name, pulse_ht, energy in zip(self._names, self._ph, self._energies):
            seq.append("  energy(ph=%7.2f) --> %9.2f eV (%s)" % (pulse_ht, energy, name))
        return "\n".join(seq)
    
    def __getstate__(self):
        """Pickle will use the return value of this in place of self.__dict__ to pickle the
        objects.  Since energy2ph and ph2energy are functions generated at runtime, they don't
        pickle.  We remove them before pickling and reconstruct them on load."""
        d = self.__dict__.copy()
        d.pop('energy2ph', None)
        d.pop('ph2energy', None)
        return d
    
    def __setstate__(self, d):
        """Pickle will pass d to this instead of just setting self.__dict__ to unpickle the
        objects.  Since energy2ph and ph2energy are functions generated at runtime, they 
        aren't pickled.  We remove them before pickling and reconstruct them on load."""
        self.__dict__.update(d)
        self._update_converters()
    
    def set_use_spline(self, spline):
        self.use_spline = spline
        self._update_converters()
    
    def copy(self, new_ph_field=None):
        """Return a deep copy"""
        ecal = EnergyCalibration(self.ph_field)
        ecal.__dict__.update(self.__dict__)
        ecal._names = list(self._names)
        ecal._ph = self._ph.copy()
        ecal._energies = self._energies.copy()
        ecal._stddev = self._stddev.copy()
        ecal.use_spline = self.use_spline
        if new_ph_field is not None:
            ecal.ph_field = new_ph_field
        return ecal
    
    def remove_cal_point_name(self, name):
        "If you don't like calibration point named <name>, this removes it"
        idx = self._names.index(name)
        self._names.pop(idx)
        self._ph = numpy.hstack((self._ph[:idx], self._ph[idx+1:]))
        self._energies = numpy.hstack((self._energies[:idx], self._energies[idx+1:]))
        self._stddev = numpy.hstack((self._stddev[:idx], self._stddev[idx+1:]))
        self.npts -= 1
        self._update_converters()
        
    def remove_cal_point_prefix(self, prefix):
        """This removes all cal points whose name starts with <prefix>.  Return number removed."""
        for name in tuple(self._names):
            if name.startswith(prefix):
                self.remove_cal_point_name(name)
        
    def add_cal_point(self, pht, energy, name="", pht_error=None, overwrite=True):
        """
        Add a single energy calibration point <pht>, <energy>, where <pht> must be in units
        of the self.ph_field and <energy> is in eV.  <pht_error> is the 1-sigma uncertainty
        on the pulse height.  If None (the default), then assign pht_error = <pht>/1000.
        
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
        
        if name in self._names:  # Update an existing point
            if not overwrite:
                raise ValueError("Calibration point '%s' is already known and overwrite is False" % name)
            index = self._names.index(name)
            self._ph[index] = pht
            self._energies[index] = energy
            self._stddev[index] = pht_error
            
        else:   # Add a new point
            self._ph = numpy.hstack((self._ph, pht))
            self._energies = numpy.hstack((self._energies, energy))
            self._stddev = numpy.hstack((self._stddev, pht_error))
            self._names.append(name)
            
            # Sort in ascending energy order
            sortkeys = numpy.argsort(self._energies)
            self._ph = self._ph[sortkeys]
            self._energies = self._energies[sortkeys]
            self._stddev = self._stddev[sortkeys]
            self._names = [self._names[s] for s in sortkeys]
            self.npts += 1
            assert len(self._names)==len(self._ph)
            assert len(self._names)==len(self._stddev)
            assert len(self._names)==len(self._energies)

        self._update_converters()
        
    def _update_converters(self):
        """There is now a new data point."""
        assert len(self._ph)==len(self._energies)
        assert len(self._ph)==self.npts
        assert self.npts>1
        
        
        if (self._stddev <= 0.0).any():
            self._stddev[self._stddev<=0.0] = self._stddev[self._stddev>0].min()
        
        if (not self.use_spline) and self.npts >= 2:
            highest_slope = (self._energies[-1]-self._energies[-2])/(self._ph[-1]-self._ph[-2])
            ph = numpy.hstack((self._ph, [1e6]))
            energy = numpy.hstack((self._energies, [highest_slope*(ph[-1]-ph[-2])+self._energies[-1]]))
            self.ph2energy = scipy.interpolate.interp1d(ph, energy, kind='linear', bounds_error = True)
        elif self.npts > 3:
            weight = 1/numpy.array(self._stddev)
            weight[self._stddev <= 0.0] = 1/self._stddev.min()
            self.ph2energy = scipy.interpolate.UnivariateSpline(self._ph, self._energies, w=weight, k=3, 
                                                                bbox=[0, 1.6*self._ph.max()], s=self.smooth*self.npts)
        elif self.npts == 3:
            self.ph2energy = numpy.poly1d(numpy.polyfit(self._ph, self._energies, 2))
        elif self.npts == 2:
            self.ph2energy = numpy.poly1d(numpy.polyfit(self._ph, self._energies, 1))
        else:
            raise ValueError("Not enough good samples")
        max_ph = 1.3*self._ph[-1]
        ph2offset_energy = lambda ph, eoffset: self.ph2energy(ph)-eoffset 
        self.energy2ph = lambda e: scipy.optimize.brentq(ph2offset_energy, 0., max_ph, args=(e,))
        
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
        pht = numpy.arange(0, self._ph.max()*1.1)
        y = self(pht) / pht**ph_rescale_power
        axis.plot(pht, y, color=color)
        
        # Plot and label cal points
        if ph_rescale_power==0.0:
            axis.errorbar(self._ph, self._energies, yerr=self._stddev, fmt='o', 
                          mec='black', mfc=markercolor, capsize=0)
        else:
            axis.errorbar(self._ph, self._energies/(self._ph**ph_rescale_power), yerr=self._stddev/(self._ph**ph_rescale_power), fmt='or', capsize=0)
        for pht, name in zip(self._ph[1:], self._names[1:]):  
            axis.text(pht, self(pht)/pht**ph_rescale_power, name+'  ', ha='right')        

        axis.grid(True)
        axis.set_xlabel("Pulse height ('%s')" % self.ph_field)
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