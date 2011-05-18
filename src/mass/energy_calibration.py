"""
Objects to assist with calibration from pulse heights to absolute energies.

Created on May 16, 2011

@author: fowlerj
"""

import pylab
import numpy
import scipy.interpolate
import scipy.optimize

# Some commonly-used standard energy features.
STANDARD_FEATURES={
   'Mn Ka1': 5898.802,
   'Mn Ka2': 5887.592,
   'Mn Kb':  6490.18,
   'Mn Kbeta': 6490.18,
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
    
    def __init__(self, ph_field):
        "Create an EnergyCalibration object for pulse-height-related field named <ph_field>."
        self.ph_field = ph_field
        self.ph2energy = lambda x: x
        self.energy2ph = lambda x: x
        self._ph = numpy.zeros(1, dtype=numpy.float)
        self._energies = numpy.zeros(1, dtype=numpy.float)
        self._names=['null']
        self.npts=1
        self.smooth = 1.0
        
    def __call__(self, ph):
        "Convert pulse height (or array of pulse heights) <ph> to energy (in eV)."
        return self.ph2energy(ph)
    
    def copy(self):
        ec = EnergyCalibration(self.ph_field)
        ec.__dict__.update(self.__dict__)
        ec._names = list(self._names)
        ec._ph = self._ph.copy()
        ec._energies = self._energies.copy()
        return ec
        
    def add_cal_point(self, ph, energy, name=""):
        """
        Add a single energy calibration point <ph>, <energy>, where <ph> must be in units
        of the self.ph_field and <energy> is in eV.
        
        Also, you can call it with <energy> as a string, provided it's the name of a known
        feature appearing in the dictionary mass.energy_calibration.STANDARD_FEATURES.  Thus
        the following are equivalent:
        
        cal.add_cal_point(12345.6, 5898.801, "Mn Ka1") 
        cal.add_cal_point(12456.6, "Mn Ka1")
        
        Careful!  If you give a name that's already in the list, then this value replaces
        the previous one.  If you do NOT give a name, though, then this will NOT replace
        but will add to any existing points at the same energy.
        """
        
        # If <energy> is a string and a known spectral feature's name, use it as the name instead
        if energy in STANDARD_FEATURES:
            name = energy
            energy = STANDARD_FEATURES[name]
        
        if name in self._names:  # Update an existing point
            index = self._names.index(name)
            self._ph[index] = ph
            self._energies[index] = energy
            
        else:   # Add a new point
            self._ph = numpy.hstack((self._ph,ph))
            self._energies = numpy.hstack((self._energies,energy))
            self._names.append(name)
            
            # Sort in ascending energy order
            sortkeys = numpy.argsort(self._energies)
            self._ph = self._ph[sortkeys]
            self._energies = self._energies[sortkeys]
            self._names = [self._names[s] for s in sortkeys]
            self.npts += 1
            assert len(self._names)==len(self._ph)
            assert len(self._names)==len(self._energies)

        self._update_converters()
        
    def _update_converters(self):
        """There is now a new data point."""
        assert(len(self._ph)==len(self._energies))
        assert(len(self._ph)==self.npts)
        assert(self.npts>1)
        
        if self.npts>3:
            self.ph2energy = scipy.interpolate.UnivariateSpline(self._ph, self._energies, k=3, s=self.smooth)
        elif self.npts==3:
            self.ph2energy = numpy.poly1d(numpy.polyfit(self._ph, self._energies, 2))
        elif self.npts==2:
            self.ph2energy = numpy.poly1d(numpy.polyfit(self._ph, self._energies,1))
        else:
            raise ValueError("Not enough good samples")
        ph2offset_energy = lambda ph, eoffset: self.ph2energy(ph)-eoffset 
        self.energy2ph = lambda e: scipy.optimize.brentq(ph2offset_energy, 0., 1e5, args=(e,))
        
    def name2ph(self, name):
        """Convert a named energy feature to pulse height"""
        e = STANDARD_FEATURES[name]
        return self.energy2ph(e)

    def plot(self, axis=None):
        if axis is None:
            pylab.clf()
            axis = pylab.subplot(111)
        axis.plot(self._ph, self._energies,'or')
        ph = numpy.arange(0,self._ph.max()*1.1)
        axis.plot(ph, self(ph),color='green')
        for ph,name in zip(self._ph[1:], self._names[1:]):  
            axis.text(ph-.01*self._ph.max(), self(ph), name, ha='right')        
        axis.grid(True)
        axis.set_xlabel("Pulse height ('%s')"%self.ph_field)
        axis.set_ylabel("Energy (eV)")
        axis.set_title("Energy calibration curve")
        

class EnergyCalibrationCrap(object):
    """
    Object to store all information relevant to one detector's absolute
    energy calibration.
    """


    def __init__(self):
        """
        Constructor (duh)
        """
        self.features = {}
        
    def __str__(self):
        channames=set()
        featurenames=[]
        for f in self.features.values():
            featurenames.append(f.name)
            for k in f.ph.keys():
                if not k.endswith("_prev"):
                    channames.add(k)
            
        s = [""]
        featurenames = tuple(featurenames)
        channames = tuple(channames)
        s.append(16*' '+"".join(["%15s"%cn for cn in channames]))
        s.append(16*' '+" ".join([15*'-' for cn in channames]))
        for f in featurenames:
            words=["%14s: "%f]
            for k in channames:
                try:
                    words.append("%15.3f"%self.features[f].ph[k])
                except KeyError:
                    words.append(12*" "+"n/a")    
            s.append("".join(words))
        return "\n".join(s)
        
    def add_feature(self, feature):
        self.features[feature.name] = feature
        
    def copy(self):
        ec = EnergyCalibration()
        ec.__dict__.update(self.__dict__)
        ec.features = {}
        for k,v in self.features.iteritems():
            try:
                ec.features[k] = v.copy()
            except AttributeError, e:
                print e
        return ec


    def fit_mn_kalpha(self, dataset, range=150, type='filt'):
        channame={'filt':'p_filt_value',
                  'dc': 'p_filt_value_dc'}[type]
        range = numpy.array((-range,range)) + self.features['Mn Ka1'].ph[channame]
        params, _covar = dataset.fit_mn_kalpha(range, type=type)
        ph_ka1 = params[1]
        ph_ka2 = params[1] - 11.1*params[2]
        self.features['Mn Ka1'].set_val(channame, ph_ka1)
        if 'Mn Ka2' not in self.features:
            self.features['Mn Ka2'] = make_energy_feature('Mn Ka2')
        self.features['Mn Ka2'].set_val(channame, ph_ka2)


    def make_calibrator(self, channame='p_filt_value', smooth=1.0):
        fs = self.features.values()
        ep=[(0.,0.)]
        for f in fs:
            try:
                ep.append( (f.energy, f.ph[channame]) )
            except KeyError:
                continue
        ep.sort()
        ph = [x[1] for x in ep]
        energy = [x[0] for x in ep]
        print ph
        print energy
        if len(ph)>3:
            calibrator = scipy.interpolate.UnivariateSpline(ph, energy, k=3, s=smooth)
        elif len(ph)==3:
            calibrator = numpy.poly1d(numpy.polyfit(ph, energy, 2))
        elif len(ph==2):
            calibrator = numpy.poly1d(numpy.polyfit(ph,energy,1))
        else:
            raise ValueError("Not enough good samples")
        return calibrator

class EnergyFeature(object):
    """
    """
    
    def __init__(self, name, energy, **kwargs):
        self.name = name
        self.energy = energy
        self.ph = {}
        self.ph.update(kwargs)

    def set_val(self, type, value):
        "asdfadsf"
        if type in self.ph:
            self.ph['%s_prev'%type] = self.ph[type]
        self.ph[type] = value
        
    def copy(self):
        ef = EnergyFeature(self.name, self.energy)
        ef.__dict__.update(self.__dict__)
        return ef
        
    def __str__(self):
        s=['Energy feature %s at %8.3f eV'%(self.name, self.energy)]
        for k,v in self.ph.iteritems():
            s.append("    '%-20s': %9.3f"%(k,v))
        return "\n".join(s)

    def __repr__(self):
        s=["EnergyFeature('%s',%f,"%(self.name, self.energy)]
        for k,v in self.ph.iteritems():
            s.append("%s=%f"%(k,v))
        s.append(")")
        return " ".join(s)

def make_energy_feature(name):
    """
    Factory function to make EnergyFeature objects of known
    energy (and standard names).
    """
    
    if name in STANDARD_FEATURES:
        return EnergyFeature(name, STANDARD_FEATURES[name])
    
    raise ValueError("Known energy features are:"%STANDARD_FEATURES.keys())