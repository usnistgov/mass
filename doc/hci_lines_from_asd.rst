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

``getAvailableElements()``

``getAvailableSpectralCharges(element)``

``getAvailableLevels(element, spectralCharge, requiredConf=None, requiredTerm=None, requiredJVal=None, maxLevels=None, units='eV')``

``getSingleLevel(self, element, spectralCharge, conf, term, j_val, units='eV', getUncertainty=True)``


.. testcode::

  import mass.calibration._hci_lines
