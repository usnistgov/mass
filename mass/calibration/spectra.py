'''
Module for spectrum analysis classes.

Created on Dec 12, 2011

@author: fowlerj
'''

import numpy
import cPickle as pickle

__all__=['SpectrumGroup']

class SpectrumGroup(object):
    '''
    Object to contain voltage spectra from multiple detectors and to manage their calibration both
    to each other and to known energy features.
    '''


    def __init__(self, spectrum_iter=None):
        '''
        Construct a SpectrumGroup, optionally with initial voltage (uncalibrated) spectra.
        
        <spectrum_iter> is an iterator that yields one or more numpy.ndarray objects.  Each
                        is assumed to be an unsorted array of pulse sizes (presumably you want
                        them to be optimally filtered pulse heights).
        '''
        self.raw_spectra = []
        self.nchan = 0
        self.npulses = 0
        
        if spectrum_iter is not None:
            for sp in spectrum_iter:
                self.add_spectrum(sp)
        
    
    def add_spectrum(self, sp):
        "Add <sp> to the list of spectra, where <sp> is an ndarray containing uncalibrated pulse sizes."
        self.raw_spectra.append(sp)
        self.nchan += 1
        self.npulses += len(sp)
    
    
    def store(self, filename):
        """Store the complete data state in file named <filename>.
        Use mass.calibration.energy_calibration.load_spectrum_group() to restore."""
        data = dict(raw_spectra=self.raw_spectra,
                    nchan = self.nchan,
                    npulses = self.npulses)
        fp = open(filename, "wb")
        pickle.dump(data, fp, protocol=2)
        fp.close()


def load_spectrum_group(filename):
    """Return a SpectrumGroup stored with SpectrumGroup.store(filename)."""
    group = SpectrumGroup()
    fp = open(filename, "rb")
    group.__dict__ = pickle.load(fp)
    fp.close()
    return group
