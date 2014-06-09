'''
Module for spectrum analysis classes.

Created on Dec 12, 2011

@author: fowlerj
'''

import numpy as np
import scipy as sp
import pylab as plt
import cPickle as pickle

from mass.mathstat.utilities import plot_as_stepped_hist
import mass.calibration

__all__ = ['SpectrumGroup']

class RawSpectrum(object):
    """
    Object to contain a single detector's voltage spectrum and its calibration
    to energy units.
    """
    
    def __init__(self, pulses):
        """
        <pulses> a sequence of the pulse sizes (volts or similar instrument-referenced quantity).
                 Will be copied and stored internally as a np.ndarray.
        """
        self.pulses = np.array(pulses, dtype=np.float)
        self.pulses.sort()  # This might not be a good idea?
        self.energies = self.pulses.copy()
        self.npulses = len(self.pulses)
        self.brightest_lines = []
        self.calibration = mass.calibration.energy_calibration.EnergyCalibration('volts')
        self.calibration_valid = True
    
    
    def copy(self):
        rs = RawSpectrum(self.pulses)
        rs.brightest_lines = self.brightest_lines.copy()
        rs.energies = self.energies.copy()
        rs.calibration = self.calibration.copy()
        rs.calibration_valid = self.calibration_valid
        return rs
    

    def max(self): #@ReservedAssignment
        return self.pulses.max()


    def calibrate_brightest_lines(self, line_energies, nbins, vmax, dv_smear,
                                  min_bin_sep=8, line_names=None):
        """"""
        lines = self.find_brightest_lines(nbins, vmax, dv_smear, min_bin_sep=min_bin_sep,
                                          max_lines=len(line_energies))
        if line_names is None:
            line_names = ['line%d'%i for i in range(len(lines))]
        if len(lines)==0:
            self.calibration_valid = False
            return
        
        # So far, line_energies and lines are both sorted by peak contents.
        # But we need to match up these lines by energy order, even if contents happen to be out of order
        volts = lines[:,1]
        volts.sort()
        line_energies = line_energies[:len(volts)]
        line_energies.sort()
        
        for (result, energy, name) in zip(volts, line_energies, line_names):
            self.calibration.add_cal_point(result, energy, name)
        self.recompute_energies()
        

    def recompute_energies(self):
        self.energies = self.calibration(self.pulses)

        
    def find_brightest_lines(self, nbins, vmax, dv_smear,
                             min_bin_sep=8, max_lines=None):
        
        if max_lines is None:
            max_lines = 999999999
        
        # Step 1: find lines by histogramming the data and smearing.
        cont, bins = np.histogram(self.pulses, nbins, [0,vmax])
        bin_ctr = 0.5*(bins[1:]+bins[:-1])

        # Assume 40 counts = 1000 eV and smear by resolution (typically 60 eV FWHM, or 25 eV rms)
        cont_smear = mass.calibration.fluorescence_lines.smear(cont, fwhm=dv_smear, stepsize=1.0)
    
        smallest_peak = 0.02*cont_smear.max()
        if smallest_peak < 40:
            smallest_peak = 40.
    
        # Find peaks.  First, sort all bins by their contents:
        c_order = np.argsort(cont_smear)[::-1]
        peak_bins = []
    
        for c in c_order:
            if cont_smear[c] < smallest_peak:
                break
            nearby_peak = False
            for peak in peak_bins:
                if np.abs(c-peak) < min_bin_sep:
                    nearby_peak = True
                    break
            if not nearby_peak:
                peak_bins.append(c)
    
        spectral_line = mass.calibration.GaussianLine()
        fitter = mass.calibration.GaussianFitter(spectral_line)
    
        lines=[]
        for peak in peak_bins:
            spectral_line.energy = peak
            try:
                result, _covar = fitter.fit(
                    cont[peak-min_bin_sep:peak+min_bin_sep+1],
                    bin_ctr[peak-min_bin_sep:peak+min_bin_sep+1],
                    params=(3.0, bin_ctr[peak], cont[peak], .1),
                    plot=False)
                fwhm, centroid, peak, _bg = result
                area = fwhm/2.35482*np.sqrt(2*np.pi)*peak
    
                lines.append((area, centroid))
                if len(lines) >= max_lines:
                    break
                
            except RuntimeError, e:
                print e
            except ValueError, e:
                print e
    
        if len(lines)==0:
            return []
        lines = np.array(lines)
        line_order = lines[:,0].argsort()[::-1]
        return lines[line_order]
    
    

class SpectrumGroup(object):
    '''
    Object to contain voltage spectra from multiple detectors and to manage their calibration both
    to each other and to known energy features.
    '''


    def __init__(self, spectrum_iter=None):
        '''
        Construct a SpectrumGroup, optionally with initial voltage (uncalibrated) spectra.
        
        <spectrum_iter> is an iterator that yields one or more np.ndarray objects.  Each
                        is assumed to be an unsorted array of pulse sizes (presumably you want
                        them to be optimally filtered pulse heights).
        '''
        self.raw_spectra = []
        self.plot_ordering = []
        self.nchan = 0
        self.npulses = 0
        
        if spectrum_iter is not None:
            for sp in spectrum_iter:
                self.add_spectrum(sp)
        
    
    def copy(self):
        """Return a deep copy of self (useful when code changes)"""
        sg = SpectrumGroup(self.raw_spectra)
        sg.plot_ordering = self.plot_ordering
        return sg
        
    
    def add_spectrum(self, sp):
        """Add <sp> to the list of spectra, where <sp> is an ndarray containing uncalibrated pulse sizes
        or an instance of RawSpectrum."""
        try:
            sp = RawSpectrum(sp)
        except TypeError:
            pass
        self.raw_spectra.append(sp)
        self.plot_ordering.append(self.nchan)
        self.nchan += 1
        self.npulses += sp.npulses
    
    
    def store(self, filename):
        """Store the complete data state in file named <filename>.
        Use mass.calibration.energy_calibration.load_spectrum_group() to restore."""
        fp = open(filename, "wb")
        pickler = pickle.Pickler(fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(self.nchan)
        for sp in self.raw_spectra:
            pickler.dump(sp)
        fp.close()
    
    
    def plot_all(self, nbins=2000, binrange=None, yoffset=50, axis=None, color=None, raw=False):
        if binrange is None:
            m = max((rs.max() for rs in self.raw_spectra))
            binrange = [0,m]
        if axis is None:
            plt.clf()
            axis = plt.subplot(111)
        if color is None:
            color = plt.cm.get_cmap('spectral')
        
        bin_centers = np.arange(0.5, nbins)*(binrange[1]-binrange[0])/nbins + binrange[0]
        
        for i,spect_number in enumerate(self.plot_ordering):
            rs = self.raw_spectra[spect_number]
            if raw:
                cont, _bins = np.histogram(rs.pulses, nbins, binrange)
            else:
                cont, _bins = np.histogram(rs.energies, nbins, binrange)
            plot_as_stepped_hist(axis, cont+i*yoffset, bin_centers, color=color(float(i)/self.nchan))


    def calibrate_brightest_lines(self, line_energies, nbins, vmax, dv_smear,
                                  min_bin_sep=8, line_names=None):
        for rs in self.raw_spectra:
            rs.calibrate_brightest_lines(line_energies, nbins=nbins, vmax=vmax, dv_smear=dv_smear,
                                         min_bin_sep = min_bin_sep, line_names=line_names)


def load_spectrum_group(filename):
    """Return a SpectrumGroup stored with SpectrumGroup.store(filename)."""
    fp = open(filename, "rb")
    up = pickle.Unpickler(fp)
    group = SpectrumGroup()
    nchan =  up.load()
    print "Loading %d channels from %s"%(nchan, filename)
    while True:
        try:
            group.add_spectrum(up.load())
        except EOFError:
            break
    print nchan, group.nchan
    fp.close()
    return group


def minimize_ks_prob(s1, s2, ex, ey, search_range=None, tol=1e-6, print_output=False):
    """Given two RawSpectrum objects <s1> and <s2>, as well as a "calibration energy" <ex>,
    Find the calibration voltages to energy ex such that the spectra match best from energy
    <ex> to <ey>  (ex may be greater or less than ey).  
    
    Specifically, find the voltages v1 and v2 such that cal1(v1) = cal2(v2) = ex, and then find
    the scale factor G such that setting cal1(v1*G) = cal2(v2/G) = ex gives the best improvement
    in the matching of the spectra from energy in [ex, ey] as measured by the
    Kolmagorov-Smirnov test for comparing two sampled distributions.  (See sp.stats.ks_2samp).
    
    Returns: the best scale factor G as defined above.  If G>1, it means that spectrum s2 converts
    a higher voltage to e than s1 does."""
    
    if search_range is None:    
        search_range = [0.92, 1.08]
    
    def ks_statistic(scale, s1, s2, ex, ey):
        c1 = s1.calibration
        c2 = s2.calibration
        v1 = c1.energy2ph(ex)
        v2 = c2.energy2ph(ex)
        c1.add_cal_point(v1*scale, ex, "tmp")
        c2.add_cal_point(v2/scale, ex, "tmp")
        e1 = c1(s1.pulses)
        e2 = c2(s2.pulses)
        c1.remove_cal_point_name("tmp")
        c2.remove_cal_point_name("tmp")
        emin = min(ex,ey)
        emax = max(ex,ey)
        use1 = np.logical_and(e1>emin, e1<emax)
        use2 = np.logical_and(e2>emin, e2<emax)
        if (use1).sum() == 0 or (use2).sum() == 0:
            return 1.0, v1, v2
        return sp.stats.ks_2samp(e1[use1], e2[use2])[0]
    
    ex = float(ex)
    ey = float(ey)
#    a = ks_statistic(search_range[0], s1, s2, ex, ey)
#    b = ks_statistic(1.0, s1, s2, ex, ey)
#    c = ks_statistic(search_range[1], s1, s2, ex, ey)
#    print 'a,b,c=', a, b, c
    
    best_scale, best_ks_stat, _iter, funcalls = \
        sp.optimize.brent(ks_statistic, args=(s1, s2, ex, ey), 
                             brack=search_range, tol=tol, full_output=True)
    if print_output:
        print "Brent's method scale=%.6f Best KS-stat %.6f.  %d function calls."%(
                best_scale, best_ks_stat, funcalls)
    v1 = s1.calibration.energy2ph(ex)
    v2 = s2.calibration.energy2ph(ex)
    return best_scale, v1*best_scale, v2/best_scale


def match_two_spectra(s1, s2, initial_energies, min_scale=0.9, max_scale=1.1):
    c1 = s1.calibration
    c2 = s2.calibration
    for cal in (c1,c2):
        cal.remove_cal_point_prefix("tmp")
        cal.set_use_spline(False)
    print c1
    for i,erange in enumerate(initial_energies):
        ematch, emax = erange
        best_scale, v1, v2 = minimize_ks_prob(s1, s2, ematch, emax, tol=1e-4, search_range=[.96,1.04])
        if best_scale > min_scale and best_scale<max_scale:
            print "Range %f to %f has best_scale, v1, v2=%f %f %f"%(ematch, emax, best_scale, v1, v2)
            c1.add_cal_point(v1, ematch, "tmp%d"%i)
            c2.add_cal_point(v2, ematch, "tmp%d"%i)
    s1.recompute_energies()
    s2.recompute_energies()
