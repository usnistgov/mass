Detector X-ray Efficiency Models
================================

.. warning:: This module requires the xraylib python package. Please see https://github.com/tschoonj/xraylib/wiki for installation instructions.


Motivation
----------
For many analyses, it is important to estimate a x-ray spectrum as it would be seen from the source rather than as it would be measured with a set of detectors.
This can be important, for example, when trying to determine line intensity ratios of two lines separated in energy space.
Here, we attempt to model the effects that would cause the measured spectrum to be different from the true spectrum, 
such as energy dependent losses in transmission due to IR blocking filters and vacuum windows.
Energy dependent absorber efficiency can also be modeled.

Exploring ``FilterStack`` class and subclass functions with premade efficiency models
-------------------------------------------------------------------------------------
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

  EBIT 2018 components: {'Electroplated Au Absorber': Film, '50mK Filter': AlFilmWithOxide, '3K Filter': AlFilmWithOxide, '50K Filter': FilterStack, 'Luxel Window TES': LEX_HT, 'Luxel Window EBIT': LEX_HT}

In this case, the components represented the various filters and absorbers within the filter stack. 
More complicated filters can be built up with components of an arbitrary number of layers. 
For example, a filter can consist of both a film and a support mesh backing the film.

.. testcode::

  for iComponent in list(EBIT_model.components.values()):
    if iComponent.components != {}:
      print('{} components:'.format(iComponent.name), iComponent.components)

.. testoutput::

  50K Filter components: {'Al Film': AlFilmWithOxide, 'Ni Mesh': Film}
  Luxel Window TES components: {'LEX_HT Film': Film, 'LEX_HT Mesh': Film}
  Luxel Window EBIT components: {'LEX_HT Film': Film, 'LEX_HT Mesh': Film}

Next, we examine the function ``get_efficiency(xray_energies_eV)``, which is an attribute of ``FilterStack``. 
This can be called for the entire filter stack or for individual components in the filter stack. 
As an example, we look at the efficiency of the EBIT 2018 filter stack and the 50K filter component between 
2,000 eV and 10,000 eV, at 1,000 eV steps.

.. testcode::

  sparse_xray_energies_eV = np.arange(2000, 10000, 1000)
  stack_efficiency = EBIT_model.get_efficiency(sparse_xray_energies_eV)
  filter50K_efficiency = EBIT_model.components['50K Filter'].get_efficiency(sparse_xray_energies_eV)

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
  EBIT_model.components['50K Filter'].plot_efficiency(xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/filter_50K_efficiency.png");plt.close()
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


Creating your own custom filter stack model using ``FilterStack`` objects
-------------------------------------------------------------------------
Now we will explore creating custom ``FilterStack`` objects and building up your very own filter stack model.
First, we will create a general ``FilterStack`` object, representing a stack of filters.
We will then populate this object with filters, which take the form of the various ``FilterStack`` object subclasses, such as ``Film``,
or even other ``FilterStack`` objects to create more complicated filters with multiple components.
The ``add`` argument can be used to add a premade ``FilterStack`` object as a component of a different ``FilterStack`` object.
We will start by adding some simple ``Film`` objects to the filter stack.
This class requires a the ``name`` and ``material`` arguments, and the optical depth can be specified by passing in either
``area_density_g_per_cm2`` or ``thickness_nm`` (but not both). 
By default, most ``FilterStack`` objects use the bulk density of a material to calculate the optical depth when the ``thickness_nm`` is used,
but a custom density can be specified with the ``density_g_per_cm3`` argument. 
In addition, a meshed style filter can be modelled using the ``fill_fraction`` argument.
Finally, most ``FilterStack`` subclasses can use the ``absorber`` argument (default False), which will cause the object to return absorption,
instead of transmittance, as the efficiency.

.. testcode::

  custom_model = mass.efficiency_models.FilterStack(name='My Filter Stack')
  custom_model.add_Film(name='My Bi Absorber', material='Bi', thickness_nm=4.0e3, absorber=True)
  custom_model.add_Film(name='My Al 50mK Filter', material='Al', thickness_nm=100.0)
  custom_model.add_Film(name='My Si 3K Filter', material='Si', thickness_nm=500.0)
  custom_filter = mass.efficiency_models.FilterStack(name='My meshed 50K Filter')
  custom_filter.add_Film(name='Al Film', material='Al', thickness_nm=100.0)
  custom_filter.add_Film(name='Ni Mesh', material='Ni', thickness_nm=10.0e3, fill_fraction=0.2)
  custom_model.add(custom_filter)

Let us look at the efficiency curves of the filter stack and its components.

.. testcode::

  custom_model.plot_efficiency(xray_energies_eV)
  custom_model.plot_component_efficiencies(xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/custom_50K.png");plt.close()
  plt.savefig("img/custom_3K.png");plt.close()
  plt.savefig("img/custom_50mK.png");plt.close()
  plt.savefig("img/custom_absorber.png");plt.close()
  plt.savefig("img/custom_filter_stack.png");plt.close()

.. image:: img/custom_filter_stack.png
  :width: 30%  

.. image:: img/custom_absorber.png
  :width: 30%  

.. image:: img/custom_50mK.png
  :width: 30%  

.. image:: img/custom_3K.png
  :width: 30%  

.. image:: img/custom_50K.png
  :width: 30%  

We can also look more in depth at 50K filter component efficiencies.

.. testcode::

  custom_filter.plot_component_efficiencies(xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/custom_Ni_mesh.png");plt.close()
  plt.savefig("img/custom_Al_film.png");plt.close()

.. image:: img/custom_Al_film.png
  :width: 30%  

.. image:: img/custom_Ni_mesh.png
  :width: 30%  

There are also some premade filter classes for filters that commonly show up in our instrument filter stacks.
At the moment, the FilterStack subclasses listed below are implemented:
- ``AlFilmWithOxide`` - models a typical IR blocking filter with native oxide layers, which can be important for thin filters.
- ``AlFilmWithPolymer`` - models a similar IR blocking filter, but with increased structural support from a polymer backing.
- ``LEX_HT`` - models LEX_HT vacuum windows, which contain a polymer backed Al film and stainless steel mesh.
Usage examples and efficiency curves of these classes are shown below.

.. testcode::

  premade_filter_stack = mass.efficiency_models.FilterStack(name='A Stack of Premade Filters')
  premade_filter_stack.add_AlFilmWithOxide(name='My Oxidized Al Filter', Al_thickness_nm=50.0)
  premade_filter_stack.add_AlFilmWithPolymer(name='My Polymer Backed Al Filter', Al_thickness_nm=100.0, polymer_thickness_nm=200.0)
  premade_filter_stack.add_LEX_HT(name='My LEX HT Filter')
  low_xray_energies_eV = np.arange(100,3000,1)
  premade_filter_stack.plot_component_efficiencies(low_xray_energies_eV)

.. testcode::
  :hide:

  plt.savefig("img/premade_LEX_HT.png");plt.close()
  plt.savefig("img/premade_Al_polymer.png");plt.close()
  plt.savefig("img/premade_Al_oxide.png");plt.close()

.. image:: img/premade_Al_oxide.png
  :width: 30%  

.. image:: img/premade_Al_polymer.png
  :width: 30%  

.. image:: img/premade_LEX_HT.png
  :width: 30%
