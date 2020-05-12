'''
_hci_lines.py

Uses pickle file containing NIST ASD levels data to generate some commonly used HCI lines in mass.
Meant to be a replacement for _highly_charged_ion_lines.py, which hard codes in line parameters.

February 2020
Paul Szypryt
'''

import numpy as np
import xraylib
import pickle
import scipy.constants as sp_const
import os
import mass
from . import fluorescence_lines
from . import line_fits
from . import LORENTZIAN_PEAK_HEIGHT

INVCM_TO_EV = sp_const.c * sp_const.physical_constants['Planck constant in eV s'][0] * 100.0
DEFAULT_PICKLE_NAME = 'nist_asd.pickle'

class NIST_ASD():
    def __init__(self, pickleFilename=None):
        if pickleFilename is None:
            pickleFilename = os.path.join(os.path.split(__file__)[0], DEFAULT_PICKLE_NAME)
        with open(pickleFilename, 'rb') as handle:            
            self.NIST_ASD_Dict = pickle.load(handle)

    def getAvailableElements(self):
        return list(self.NIST_ASD_Dict.keys())

    def getAvailableSpectralCharges(self, element):
        return list(self.NIST_ASD_Dict[element].keys())

    def getAvailableLevels(self, element, spectralCharge, requiredConf=None, requiredTerm=None, requiredJVal=None, maxLevels=None, units='eV'):
        spectralCharge=int(spectralCharge) 
        levelsDict={}
        numLevels=0
        for iLevel in list(self.NIST_ASD_Dict[element][spectralCharge].keys()):
            try:
                # Check to see if we reached maximum number of levels to return
                if maxLevels is not None:
                    if numLevels == maxLevels:
                        return levelsDict
                # If required, check to see if level matches search conf, term, j_val
                includeConf = False
                includeTerm = False
                includeJVal = False            
                conf, term, j_str = iLevel.split()
                j_val = j_str.split('=')[1]            
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
                    if j_val == requiredJVal:
                        includeJVal = True
                # Include levels that match, in either cm-1 or eV
                if includeConf and includeTerm and includeJVal:
                    numLevels+=1
                    if units == 'cm-1':
                        levelsDict[iLevel] = self.NIST_ASD_Dict[element][spectralCharge][iLevel]
                    elif units == 'eV':
                        levelsDict[iLevel] = [iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge][iLevel]]
                    else:
                        levelsDict = None
                        print('Unit type not supported, please use eV or cm-1')
            except ValueError:
                'Warning: cannot parse level: {}'.format(iLevel)
        return levelsDict

    def getSingleLevel(self, element, spectralCharge, conf, term, j_val, units='eV', getUncertainty=True):
        if units == 'cm-1':
            if getUncertainty:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, j_val)]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, j_val)][0]
        elif units == 'eV':
            if getUncertainty:
                levelEnergy = [iValue * INVCM_TO_EV for iValue in self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, j_val)]]
            else:
                levelEnergy = self.NIST_ASD_Dict[element][spectralCharge]['{} {} J={}'.format(conf, term, j_val)][0] * INVCM_TO_EV
        else:
            levelEnergy = None
            print('Unit type not supported, please use eV or cm-1')
        return levelEnergy

    def add_H_like_pairs(self, element, maxPairs=None):
        # Grab term = 2P* levels for a given element and charge state
        spectr_ch = xraylib.SymbolToAtomicNumber(element)
        requiredTerm = '2P*'
        if maxPairs is not None:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch, requiredTerm=requiredTerm, maxLevels=maxPairs*2)
        else:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch, requiredTerm=requiredTerm)    
        # Check for 2 levels per term (J=1/2 and J=3/2)
        configurations = []
        for iLevel in list(levelsDict.keys()):
            conf = iLevel.split(' ')[0]
            configurations.append(conf)
        validConfigurations = []
        for iConfiguration in np.unique(configurations):
            if configurations.count(iConfiguration)==2:
                validConfigurations.append(iConfiguration)
        # Create lines for each valid doublet
        for iValidConf in validConfigurations:
            confDict=self.getAvailableLevels(element, spectralCharge=spectr_ch, requiredConf=iValidConf, requiredTerm=requiredTerm)
            energies = np.array(list(confDict.values()))[:,0]
            line_identifier = '{}'.format(iValidConf)
            add_hci_line(element=element, spectr_ch=spectr_ch, line_identifier=line_identifier, energies=energies, widths=[0.1, 0.1], ratios=[1.0, 2.0])

    def add_He_like_lines(self, element, maxLevels=None):
        spectr_ch = int(xraylib.SymbolToAtomicNumber(element)-1)
        if maxLevels is not None:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels)
        else:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch)
        for iLevel in list(levelsDict.keys()):
            lineEnergy = [levelsDict[iLevel][0]]
            add_hci_line(element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=lineEnergy, widths=[0.1], ratios=[1.0])

    def add_H_like_lines(self, element, maxLevels=None):
        spectr_ch = int(xraylib.SymbolToAtomicNumber(element))
        if maxLevels is not None:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch, maxLevels=maxLevels)
        else:
            levelsDict=self.getAvailableLevels(element, spectralCharge=spectr_ch)
        for iLevel in list(levelsDict.keys()):
            lineEnergy = [levelsDict[iLevel][0]]
            add_hci_line(element=element, spectr_ch=spectr_ch, line_identifier=iLevel, energies=lineEnergy, widths=[0.1], ratios=[1.0])

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


asd = NIST_ASD()
elementList = ['N', 'O', 'Ne', 'Ar']
for iElement in elementList:
    asd.add_H_like_lines(iElement, maxLevels=None)
    asd.add_He_like_lines(iElement, maxLevels=None)
    asd.add_H_like_pairs(iElement, maxPairs=None) # eventually want to just do this with lmfit models


# Eventually create these as lmfit models rather than mass spectra classes
# Create some He-like 1s2s,2p line complexes
# Ne
Ne9_1s2s_3S = asd.getSingleLevel('Ne', 9, conf='1s.2s', term='3S', j_val='1') # less dominant line
Ne8_contam_line = asd.getSingleLevel('Ne', 8,  conf='1s.(2S).2s.2p.(3P*)', term='2P*', j_val='')
Ne9_1s2p_triplet = list(asd.getAvailableLevels('Ne', 9, requiredConf='1s.2p', requiredTerm='3P*').values())
#Ne_1s2s_1S = asd.getSingleLevel('Ne', 9, conf='1s.2s', term='1S', j_val='0') # weak, probably do not include
Ne9_1s2p_singlet = asd.getSingleLevel('Ne', 9, conf='1s.2p', term='1P*', j_val='1') # dominant line
Ne9_complex_energies = np.hstack([Ne9_1s2s_3S[0], Ne8_contam_line[0], list(np.array(Ne9_1s2p_triplet)[:,0]), Ne9_1s2p_singlet[0]])
Ne9_complex_widths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
Ne9_complex_ratios = [12, 4, 1, 1, 1, 42]
add_hci_line(
    element='Ne',
    spectr_ch = 9,
    line_identifier = '1s2s,2p Complex', 
    energies = Ne9_complex_energies,
    widths = Ne9_complex_widths,
    ratios = Ne9_complex_ratios, 
    nominal_peak_energy=Ne9_1s2p_singlet[0]
    )

# Combined N7 2p + N6 1s3p 1P*
N6_1s3p_1P = asd.getSingleLevel('N', 6, conf='1s.3p', term='1P*', j_val='1')
N7_2p = list(asd.getAvailableLevels('N', 7, requiredConf='2p', requiredTerm='2P*').values())
N7_2p_combined_energies = np.hstack([N6_1s3p_1P[0], list(np.array(N7_2p)[:,0])])
N7_2p_combined_widths = [0.1, 0.1, 0.1]
N7_2p_combined_ratios = [2, 1, 2]
add_hci_line(
    element='N',
    spectr_ch = 7,
    line_identifier = '2p + N6 1s3p', 
    energies = N7_2p_combined_energies,
    widths = N7_2p_combined_widths,
    ratios = N7_2p_combined_ratios
    )