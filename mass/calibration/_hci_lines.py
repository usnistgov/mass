'''
_hci_lines.py

Uses pickle file containing NIST ASD levels data to generate some commonly used HCI lines in mass.
Meant to be a replacement for _highly_charged_ion_lines.py, which hard codes in line parameters.

February 2020
Paul Szypryt
'''

import numpy as np
import pickle
import scipy.constants as sp_const
import os
import mass
from . import fluorescence_lines
from . import line_fits
from . import LORENTZIAN_PEAK_HEIGHT
try:
    import xraylib
except ImportError:
    raise ImportError('This module requires the xraylib python package. Please see https://github.com/tschoonj/xraylib/wiki for installation instructions.')

INVCM_TO_EV = sp_const.c * sp_const.physical_constants['Planck constant in eV s'][0] * 100.0
DEFAULT_PICKLE_NAME = 'nist_asd.pickle'

class NIST_ASD():
    '''Class for working with a pickled atomic spectra database'''

    def __init__(self, pickleFilename=None):
        '''Loads ASD pickle file

        Args:
            pickleFilename: (default None) ASD pickle file name, as str, if not using default
        '''

        if pickleFilename is None:
            pickleFilename = os.path.join(os.path.split(__file__)[0], DEFAULT_PICKLE_NAME)
        with open(pickleFilename, 'rb') as handle:            
            self.NIST_ASD_Dict = pickle.load(handle)

    def getAvailableElements(self):
        '''Returns a list of all available elements from the ASD pickle file'''

        return list(self.NIST_ASD_Dict.keys())

    def getAvailableSpectralCharges(self, element):
        '''For a given element, returns a list of all available charge states from the ASD pickle file
        
        Args:
            element: str representing atomic symbol of element, e.g. 'Ne'
        '''

        return list(self.NIST_ASD_Dict[element].keys())

    def getAvailableLevels(self, element, spectralCharge, requiredConf=None, requiredTerm=None, requiredJVal=None, maxLevels=None, units='eV',
    getUncertainty=True):
        '''For a given element and spectral charge state, return a dict of all known levels from the ASD pickle file

        Args:
            element: str representing atomic symbol of element, e.g. 'Ne'
            spectralCharge: int representing spectral charge state, e.g. 1 for neutral atoms, 10 for H-like Ne
            requiredConf: (default None) filters results to those with ``conf == requiredConf``
            requiredTerm: (default None) filters results to those with ``term == requiredTerm``
            requiredJVal: (default None) filters results to those with ``JVal == requiredJVal``
            maxLevels: (default None) the maximum number of levels (sorted by energy) to return
            units: (default 'eV') 'cm-1' or 'eV' for returned line position. If 'eV', converts from database 'cm-1' values
        '''

        spectralCharge=int(spectralCharge) 
        levelsDict={}
        numLevels=0
        for iLevel in list(self.NIST_ASD_Dict[element][spectralCharge].keys()):
            try:
                # Check to see if we reached maximum number of levels to return
                if maxLevels is not None:
                    if numLevels == maxLevels:
                        return levelsDict
                # If required, check to see if level matches search conf, term, JVal
                includeConf = False
                includeTerm = False
                includeJVal = False            
                conf, term, j_str = iLevel.split()
                JVal= j_str.split('=')[1]            
                if requiredConf is None:
                    includeConf = True
                else:
                    if conf == requiredConf:
                        includeConf = True
                if requiredTerm is None:
                    includeTerm = True
                else:
                    if term == requiredTerm:
                        includeTerm = True
                if requiredJVal is None:
                    includeJVal = True
                else:
                    if JVal == requiredJVal:
                        includeJVal = True
                # Include levels that match, in either cm-1 or eV
                if includeConf and includeTerm and includeJVal:
                    numLevels+=1
                    if units == 'cm-1':
                        if getUncertainty:
                            levelsDict[iLevel] = self.NIST_ASD_Dict[element][spectralCharge][iLevel]
                        else:
                            levelsDict[iLevel] = self.NIST_ASD_Dict[element][spectralCharge][iLevel][0]
                    elif units == 'eV':
                        if getUncertainty:
                            levelsDict[iLevel] = [iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge][iLevel]]
                        else:
                            levelsDict[iLevel] = INVCM_TO_EV * self.NIST_ASD_Dict[element][spectralCharge][iLevel][0]
                    else:
                        levelsDict = None
                        print('Unit type not supported, please use eV or cm-1')
            except ValueError:
                'Warning: cannot parse level: {}'.format(iLevel)
        return levelsDict

    def getSingleLevel(self, element, spectralCharge, conf, term, JVal, units='eV', getUncertainty=True):
        '''Return the level data for a fully defined element, charge state, conf, term, and JVal.

        Args:
            element: str representing atomic symbol of element, e.g. 'Ne'
            spectralCharge: int representing spectral charge state, e.g. 1 for neutral atoms, 10 for H-like Ne
            conf: str representing nuclear configuration, e.g. '2p'
            term: str representing nuclear term, e.g. '2P*'
            JVal: str representing total angular momentum J, e.g. '3/2'
            units: (default 'eV') 'cm-1' or 'eV' for returned line position. If 'eV', converts from database 'cm-1' values
            getUncertainty: (default True) if True, includes uncertainties in list of levels
        '''

        if units == 'cm-1':
            if getUncertainty:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, JVal)]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, JVal)][0]
        elif units == 'eV':
            if getUncertainty:
                levelEnergy = [iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, JVal)]]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, JVal)][0] * INVCM_TO_EV
        else:
            levelEnergy = None
            print('Unit type not supported, please use eV or cm-1')
        return levelEnergy


# Some non-class functions useful for integration with mass
def add_hci_line(element, spectr_ch, line_identifier, energies, widths, ratios, nominal_peak_energy=None):
    energies=np.array(energies)
    widths=np.array(widths)
    ratios=np.array(ratios)
    if nominal_peak_energy == None:
        nominal_peak_energy = np.dot(energies, ratios)/np.sum(ratios)
    linetype = "{} {}".format(int(spectr_ch), line_identifier)

    spectrum_class = fluorescence_lines.addline(
    element=element,
    material="Highly Charged Ion",
    linetype=linetype,
    reference_short='NIST ASD',
    fitter_type=line_fits.GenericKBetaFitter,
    reference_plot_instrument_gaussian_fwhm=0.5,
    nominal_peak_energy=nominal_peak_energy,
    energies=energies,
    lorentzian_fwhm=widths,
    reference_amplitude=ratios,
    reference_amplitude_type=LORENTZIAN_PEAK_HEIGHT,
    ka12_energy_diff=None
    )    
    return spectrum_class

def add_H_like_lines_from_asd(asd, element, maxLevels=None):
    spectr_ch = int(xraylib.SymbolToAtomicNumber(element))
    added_lines=[]
    if maxLevels is not None:
        levelsDict=asd.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels+1)
    else:
        levelsDict=asd.getAvailableLevels(element, spectralCharge=spectr_ch)
    for iLevel in list(levelsDict.keys()):
        lineEnergy = levelsDict[iLevel][0]
        if lineEnergy != 0.0:
            iLine=add_hci_line(element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=[lineEnergy], widths=[0.1], ratios=[1.0])
            added_lines.append(iLine)
    return added_lines

def add_He_like_lines_from_asd(asd, element, maxLevels=None):
    spectr_ch = int(xraylib.SymbolToAtomicNumber(element)-1)
    added_lines=[]
    if maxLevels is not None:
        levelsDict=asd.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels+1)
    else:
        levelsDict=asd.getAvailableLevels(element, spectralCharge=spectr_ch)
    for iLevel in list(levelsDict.keys()):
        lineEnergy = levelsDict[iLevel][0]
        if lineEnergy != 0.0:
            iLine = add_hci_line(element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=[lineEnergy], widths=[0.1], ratios=[1.0])
            added_lines.append(iLine)
    return added_lines


# Script for adding some lines for elements commonly used at the EBIT
asd = NIST_ASD()
elementList = ['N', 'O', 'Ne', 'Ar']
# Add all known H- and He-like lines for these elements
for iElement in elementList:
    add_H_like_lines_from_asd(asd=asd, element=iElement, maxLevels=None)
    add_He_like_lines_from_asd(asd=asd, element=iElement, maxLevels=None)
