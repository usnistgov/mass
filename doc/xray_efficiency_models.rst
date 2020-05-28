Detector X-ray Efficiency Models
=======================

Motivation
----------
For many analyses, it is important to estimate a x-ray spectrum as it would be seen from the source rather than as it would be measured with a set of detectors.
This can be important, for example, when trying to determine line intensity ratios of two lines separated in energy space.
Here, we attempt to model the effects that would cause the measured spectrum to be different from the true spectrum, 
such as energy dependent losses in transmission due to IR blocking filters and vacuum windows.
Energy dependent absorber efficiency can also be modeled.

Exploring ``FilterStack`` class and subclass functions with premade efficiency models
---------------------
Here, we import the mass.efficiency_models module and demonstrate the functionality with some of the premade efficiency models.
Generally, these premade models are put in place for TES instruments with well known absorber and filter stack compositions.
To demonstrate, we work with the 'EBIT 2018' model, which models the TES spectrometer setup at the NIST EBIT, as it was commissioned in 2018.
This model includes a ~1um thick absorber, 3 ~100nm thick Al IR blocking filters, and LEX HT vacuum windows for both the TES and EBIT vacuums.
We begin by importing ``efficiency_models`` and examining the EBIT efficiency model components.

.. testcode::

  import mass.efficiency_models
  import numpy as np
  import pylab as plt

  EBIT_model = mass.efficiency_models.models['EBIT 2018']
  print('{} components:'.format(EBIT_model.name), EBIT_model.components)

.. testoutput::

  EBIT 2018 components: [Film(name=Electroplated Au Absorber), AlFilmWithOxide(name=50mK Filter), AlFilmWithOxide(name=3K Filter), FilterStack(name=50K Filter), LEX_HT(name=Luxel Window TES), LEX_HT(name=Luxel Window EBIT)]

In this case, the components represented the various filters and absorbers within the filter stack. 
More complicated filters can be built up with components of an arbitrary number of layers. 
For example, a filter can consist of both a film and a support mesh backing the film.

.. testcode::

  for iComponent in EBIT_model.components:
    if len(iComponent.components) > 0:
      print('{} components:'.format(iComponent.name), iComponent.components)

.. testoutput::

  50K Filter components: [AlFilmWithOxide(name=Al Film), Mesh(name=Ni Mesh)]
  Luxel Window TES components: [Film(name=LEX_HT Film), Mesh(name=LEX_HT Mesh)]
  Luxel Window EBIT components: [Film(name=LEX_HT Film), Mesh(name=LEX_HT Mesh)]

Next, we examine the function ``get_efficiency(xray_energies_eV)``, which is an attribute of ``FilterStack``. 
This can be called for the entire filter stack or for individual components in the filter stack. 
As an example, we look at the efficiency of the EBIT 2018 filter stack and the 50K filter component between 
2,000 eV and 10,000 eV, at 1,000 eV steps.

.. testcode::

  sparse_xray_energies_eV = np.arange(2000, 10000, 1000)
  stack_efficiency = EBIT_model.get_efficiency(sparse_xray_energies_eV)
  filter50K_component = EBIT_model.components[[iComponent.name for iComponent in EBIT_model.components].index('50K Filter')]
  filter50K_efficiency = filter50K_component.get_efficiency(sparse_xray_energies_eV)

  print(stack_efficiency.round(decimals=2))
  print(filter50K_efficiency.round(decimals=2))

.. testoutput::

  [0.34 0.47 0.46 0.38 0.31 0.24 0.19 0.14]
  [0.78 0.81 0.82 0.84 0.87 0.89 0.92 0.83]

Instead of getting an array with efficiencies, we can create a plot of the efficiencies.
Here, we use the function ``plot_efficiency(xray_energies_eV, ax)``.
``ax`` defaults to None, but can be used to plot the efficiencies on a user provided axis.
Just like ``get_efficiency``, ``plot_efficiency`` works with FilterStack and its subclasses.
Testing with energy range 100 to 20,000 eV, 1 eV steps.

.. testcode::

  xray_energies_eV = np.arange(100,20000,1)
  EBIT_model.plot_efficiency(xray_energies_eV)
  filter50K_component.plot_efficiency(xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/filter_50K_efficiency.png);plt.close()
  plt.savefig("img/EBIT_efficiency.png");plt.close()

.. image:: img/EBIT_efficiency.png
  :width: 45%

.. image:: img/filter_50K_efficiency.png
  :width: 45%

Alternatively, you could plot the individual component efficiencies of a filter.
Here, we plot the efficiencies of the 6 components that make up the EBIT system's filter stack.

.. testcode::

  EBIT_model.plot_component_efficiencies(xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/component_EBIT_window.png");plt.close()
  plt.savefig("img/component_TES_window.png");plt.close()
  plt.savefig("img/component_50K.png");plt.close()
  plt.savefig("img/component_3K.png");plt.close()
  plt.savefig("img/component_50mK.png");plt.close()
  plt.savefig("img/component_absorber.png");plt.close()

.. image:: img/component_absorber.png
  :width: 30%

.. image:: img/component_50mK.png
  :width: 30%

.. image:: img/component_3K.png
  :width: 30%

.. image:: img/component_50K.png
  :width: 30%

.. image:: img/component_TES_window.png
  :width: 30%

.. image:: img/component_EBIT_window.png
  :width: 30%
