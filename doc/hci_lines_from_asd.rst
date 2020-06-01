Highly Charged Ion (HCI) Lines from NIST ASD
============================================

.. warning:: This module requires the xraylib python package. Please see https://github.com/tschoonj/xraylib/wiki for installation instructions.


Motivation
----------
We often find ourselves hard coding line center positions into mass, 
which is prone to errors and can be tedious when there are many lines of interest to insert.
In addition, the line positions would need to be manually updated for any changes in established results.
In the case of highly charged ions, such as those produced in an electron beam ion trap (EBIT),
there is a vast number of potential lines coming from almost any charge state of almost any element.
Luckily, these lines are well documented through the NIST Atomic Spectral Database (ASD). 
Here, we have parsed a NIST ASD SQL dump and converted it into an easily Python readable pickle file.
The ``_hci_lines.py`` module implements the ``NIST_ASD`` class, 
which loads that pickle file and contains useful functions for working with the ASD data.
It also automatically adds in some of the more common HCI lines that we commonly use in our EBIT data analyses.


Exploring the methods of class ``NIST_ASD``
-------------------------------------------
The class ``NIST_ASD`` can be initialized without arguments if the user wants to use the default ASD pickle file.
This file is located at mass/calibration/nist_asd.pickle.
A custom pickle file can be used by passing in the ``pickleFilename`` argument during initialization.
The methods of the ``NIST_ASD`` class are described below:

.. autoclass:: mass.calibration._hci_lines.NIST_ASD
  :members:
  :undoc-members:

Next, we will demonstrate usage of these methods with the example of Ne,
a commonly injected gas at the NIST EBIT.

.. testcode::

  import mass.calibration._hci_lines
  import numpy

  test_asd = mass.calibration._hci_lines.NIST_ASD()
  availableElements = test_asd.getAvailableElements()
  assert 'Ne' in availableElements
  availableNeCharges = test_asd.getAvailableSpectralCharges(element='Ne')
  assert 10 in availableNeCharges
  subsetNe10Levels = test_asd.getAvailableLevels(element='Ne', spectralCharge=10, maxLevels=6, getUncertainty=False)
  assert '2p 2P* J=1/2' in list(subsetNe10Levels.keys())
  exampleNeLevel = test_asd.getSingleLevel(element='Ne', spectralCharge=10, conf='2p', term='2P*', JVal='1/2', getUncertainty=False)

  print(availableElements[:10])
  print(availableNeCharges)
  for k, v in subsetNe10Levels.items():
    subsetNe10Levels[k] = round(v, 1)
  print(subsetNe10Levels)
  print('{:.1f}'.format(exampleNeLevel))

.. testoutput::

  ['Sn', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Se', 'Cl', 'Br']
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  {'1s 2S J=1/2': 0.0, '2p 2P* J=1/2': 1021.5, '2s 2S J=1/2': 1021.5, '2p 2P* J=3/2': 1022.0, '3p 2P* J=1/2': 1210.8, '3s 2S J=1/2': 1210.8}
  1021.5
