"""
import_asd.py

Tool for converting a NIST ASD levels sql dump into a pickle file

February 2020
Paul Szypryt
"""

import re
import ast
import numpy as np
import h5py
import pickle
import argparse


def write_asd_pickle(inputFilename, outputFilename):
    createTableString = 'CREATE TABLE'
    valueSearchString = r'\`([^\`]*)\`'
    tableName = ''
    fieldNamesDict = {}
    energyLevelsDict = {}
    with open(inputFilename, 'r') as ASD_file:
        for line_number, line in enumerate(ASD_file):
            # Create dictionary of field names for various tables
            if line.startswith(createTableString):
                tableName = re.search(valueSearchString, line).groups()[0]
                fieldNamesDict[tableName] = []
            elif tableName is not '' and line.strip().startswith('`'):
                fieldName = re.search(valueSearchString, line).groups()[0]
                fieldNamesDict[tableName].append(fieldName)
            # Parse Levels portion
            elif line.startswith('INSERT INTO `ASD_Levels` VALUES'):
                partitionedLine = line.partition(' VALUES ')[-1].strip()
                nullReplacedLine = partitionedLine.replace('NULL', "''")
                if nullReplacedLine[-1] == ';':
                    formattedLine = nullReplacedLine[:-1]
                lineAsArray = np.array(ast.literal_eval(formattedLine))
                for iEntry in lineAsArray:
                    element = iEntry[fieldNamesDict['ASD_Levels'].index('element')]
                    spectr_charge = int(iEntry[fieldNamesDict['ASD_Levels'].index('spectr_charge')])
                    # Pull information that will be used to name dictionary keys
                    conf = iEntry[fieldNamesDict['ASD_Levels'].index('conf')]
                    term = iEntry[fieldNamesDict['ASD_Levels'].index('term')]
                    j_val = iEntry[fieldNamesDict['ASD_Levels'].index('j_val')]
                    # Pull energy and uncertainty
                    energy = iEntry[fieldNamesDict['ASD_Levels'].index('energy')]  # cm^-1, str
                    unc = iEntry[fieldNamesDict['ASD_Levels'].index('unc')]  # cm^-1, str
                    try:
                        energy_inv_cm = float(energy)  # cm^-1
                    except ValueError:
                        energy_inv_cm = np.nan
                    try:
                        unc_inv_cm = float(unc)  # cm^-1
                    except ValueError:
                        unc_inv_cm = np.nan
                    if (conf != '') and (term != '' and term != '*'):
                        # Set up upper level dictionary
                        if element not in energyLevelsDict.keys():
                            energyLevelsDict[element] = {}
                        if spectr_charge not in energyLevelsDict[element].keys():
                            energyLevelsDict[element][spectr_charge] = {}
                        levelName = '{} {} J={}'.format(conf, term, j_val)
                        energyLevelsDict[element][spectr_charge][levelName] = [energy_inv_cm, unc_inv_cm]
    # Sort levels within an element/charge state by energy
    outputDict = {}
    for iElement in energyLevelsDict.keys():
        for iCharge in energyLevelsDict[iElement].keys():
            energyOrder = np.argsort(np.array(list(energyLevelsDict[iElement][iCharge].values()))[:, 0])
            orderedKeys = np.array(list(energyLevelsDict[iElement][iCharge].keys()))[energyOrder]
            orderedValues = np.array(list(energyLevelsDict[iElement][iCharge].values()))[energyOrder]
            for i, iKey in enumerate(list(orderedKeys)):
                if iElement not in outputDict.keys():
                    outputDict[iElement] = {}
                if iCharge not in outputDict[iElement].keys():
                    outputDict[iElement][iCharge] = {}
                outputDict[iElement][iCharge][str(iKey)] = orderedValues[i].tolist()
    # Write dict to pickle file
    with open(outputFilename, 'wb') as handle:
        pickle.dump(outputDict, handle, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', required=True, help='Input sql dump file name')
    requiredNamed.add_argument('-o', '--output', required=True, help='Output pickle file name')
    args = parser.parse_args()
    print('Reading from file {}'.format(args.input))
    print('Writing to file {}'.format(args.output))
    write_asd_pickle(inputFilename=args.input, outputFilename=args.output)
