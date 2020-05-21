Detector X-ray Efficiency Models
=======================

Motivation
----------
For many analyses, it is important to estimate a x-ray spectrum as it would be seen from the source rather than as it would be measured with a set of detectors.
This can be important, for example, when trying to determine line intensity ratios of two lines separated in energy space.
Here, we attempt to model the effects that would cause the measured spectrum to be different from the true spectrum, 
such as energy dependent losses in transmission due to IR blocking filters and vacuum windows.
Energy dependent absorber efficiency can also be modeled.

Exploring model class functions with premade efficiency models
---------------------
Here, we import the mass.efficiency_models module and demonstrate the functionality with some of the premade efficiency models.
Generally, these premade models are put in place for TES instruments with well known absorber and filter stack compositions.

.. testcode::

  import mass.efficiency_models
  EBIT_model = mass.efficiency_models.models['EBIT 2018']