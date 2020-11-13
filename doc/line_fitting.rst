Fitting spectral line models to data in MASS
============================================

Designed for MASS version 0.7.3.

Joe Fowler, January 2020.

Previously, we wrote our own modification to the Levenberg-Marquardt optimizer in order to reach maximum-likelihood (not least-squares) fits. This appears in mass as the ``LineFitter`` class and its subclasses. In its place, we want to use the `LMFIT package <https://lmfit.github.io/lmfit-py/>`_ and the new MASS class ``GenericLineModel`` and its subclasses.

LMFit vs Scipy vs our homemade method
-------------------------------------

LMFIT has numerous advantages over the basic ``scipy.optimize`` module. Quoting from the LMFIT documentation, the user can:

* forget about the order of variables and refer to Parameters by meaningful names.
* place bounds on Parameters as attributes, without worrying about preserving the order of arrays for variables and boundaries.
* fix Parameters, without having to rewrite the objective function.
* place algebraic constraints on Parameters.

Only some of these are implemented in the original MASS fitters, and even they are not all implemented in the most robust possible way. The one disadvantage of the core LMFIT package is that it minimizes the sum of squares of a vector instead of maximizing the Poisson likelihood. This is easily remedied, however, by replacing the usual computation of residuals with one that computes the square root of the Poisson likelihood contribution from each bin. Voilá! A maximum likelihood fitter for histograms.

Advantages of LMFIT over the homemade method of the ``LineFitter`` include:

* Users can forget about the order of variables and refer to Parameters by meaningful names.
* Users can place algebraic constraints on Parameters.
* The interface for setting upper/lower bounds on parameters and for varying or fixing them is much more elegant, memorable, and simple than our homemade version.
* It ultimately wraps the ``scipy.optimize`` package and therefore inherits all of its advantages:

  * a choice of over a dozen optimizers with highly technical documentation,
  * some optimizers that aim for true global (not just local) optimization, and
  * the countless expert-years that have been invested in perfecting it.
* LMFIT automatically computes numerous statistics of each fit including estimated uncertainties, correlations, and multiple quality-of-fit statistics (information criteria as well as chi-square) and offers user-friendly fit reports. See the `MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#minimizerresult-the-optimization-result>`_ object.
* It's the work of Matt Newville, an x-ray scientist responsible for the excellent `ifeffit <http://cars9.uchicago.edu/ifeffit/>`_ and its successor `Larch <https://xraypy.github.io/xraylarch/>`_.
* Above all, *its documentation is complete, already written, and maintained by not-us.*

Usage guide
-----------

This overview is hardly complete, but we hope it can be a quick-start guide and also hint at how you can convert your own analysis work from the old to the new, preferred fitting methods.

The underlying spectral line shape models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Objects of the type ``SpectralLine`` encode the line shape of a fluorescence line, as a sum of Voigt or Lorentzian distributions. Because they inherit from ``scipy.stats.rv_continuous``, they allow computation of cumulative distribution functions and the simulation of data drawn from the distribution. An example of the creation and usage is:

.. testsetup::

  import mass
  import numpy as np
  import pylab as plt
  import mass.materials

.. testcode::

  # In general, known lines are accessed by:
  line = mass.spectra["MnKAlpha"]

  # But the following is a shortcut for many lines:
  line = mass.MnKAlpha

  np.random.seed(1066)
  N = 100000
  energies = line.rvs(size=N, instrument_gaussian_fwhm=2.2)  # draw from the distribution
  plt.clf()
  sim, bin_edges, _ = plt.hist(energies, 120, [5865, 5925], histtype="step");
  binsize = bin_edges[1] - bin_edges[0]
  e = bin_edges[:-1] + 0.5*binsize
  plt.plot(e, line(e, instrument_gaussian_fwhm=2.2)*N*binsize, "k")
  plt.xlabel("Energy (eV)")
  plt.title("Mn K$\\alpha$ random deviates and theory curve")

.. testcode::
  :hide:

  plt.savefig("img/distribution_plus_theory.png"); plt.close()

.. image:: img/distribution_plus_theory.png
  :width: 40%


The ``SpectralLine`` object is useful to you if you need to generate simulated data, or to plot a line shape, as shown above. Both the new fitting "model" objects and the old "fitter" objects use the ``SpectralLine`` object to hold line shape information. You don't need to create a ``SpectralLine`` object for fitting, though; it will be done automatically.


How to use the new, LMFIT-based models for fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest case of line fitting requires only 3 steps: create a model instance from a ``SpectralLine``, guess its parameters from the data, and perform a fit with this guess. Unlike the old fitters, plotting is not done as part of the fit--you have to do that separately.

.. testcode::

  model = line.model()
  params = model.guess(sim, bin_centers=e)
  resultA = model.fit(sim, params, bin_centers=e)

  # Fit again but with dPH/dE held at 1.
  params = resultA.params.copy()
  params["dph_de"].set(1.0, vary=False)
  resultB = model.fit(sim, params, bin_centers=e)
  resultB.plot()
  # The best-fit params are found in resultB.params
  # and a dictionary of their values is resultB.best_values.
  # The parameters given as an argument to fit are unchanged.

.. testcode::
  :hide:

  plt.savefig("img/mnka_fit1.png"); plt.close()

.. image:: img/mnka_fit1.png
  :width: 40%

You can print a nicely formatted fit report with ``fit_report()``:

.. code-block:: python

  print(resultB.fit_report())

.. code-block:: none

  [[Model]]
      GenericKAlphaModel(MnKAlpha)
  [[Fit Statistics]]
      # fitting method   = least_squares
      # function evals   = 4
      # data points      = 120
      # variables        = 4
      chi-square         = 107.219686
      reduced chi-square = 0.92430764
      Akaike info crit   = -5.51342425
      Bayesian info crit = 5.63654272
  [[Variables]]
      fwhm:        2.22986459 +/- 0.02771088 (1.24%) (init = 2.219625)
      peak_ph:     5898.80222 +/- 0.00816914 (0.00%) (init = 5898.807)
      dph_de:      1 (fixed)
      integral:    100091.321 +/- 324.744927 (0.32%) (init = 100096)
      background:  6.4245e-19 +/- 0.82673575 (128685082661343789056.00%) (init = 2.052403e-13)
      bg_slope:    0 (fixed)
  [[Correlations]] (unreported correlations are < 0.100)
      C(integral, background) = -0.314
      C(fwhm, peak_ph)        = -0.111


Fitting with exponential tails (to low or high energy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Notice when you report the fit (or check the contents of the ``params`` or ``resultB.params`` objects), there are no parameters referring to exponential tails of a Bortels response. That's because the default fitter assumes a *Gaussian* response. If you want tails, that's a constructor argument:

.. testcode::

  model = line.model(has_tails=True)
  params = model.guess(sim, bin_centers=e)
  params["dph_de"].set(1.0, vary=False)
  resultC = model.fit(sim, params, bin_centers=e)
  resultC.plot()
  # print(resultC.fit_report())

.. testcode::
  :hide:

  plt.savefig("img/mnka_fit2.png"); plt.close()

.. image:: img/mnka_fit2.png
  :width: 40%


By default, the ``has_tails=True`` will set up a non-zero low-energy tail and allow it to vary, while the high-energy tail is set to zero amplitude and doesn't vary. Use these numbered examples if you want to fit for a high-energy tail (1), to fix the low-E tail at some non-zero level (2) or to turn off the low-E tail completely (3):

.. testcode::

  # 1. To let the high-E tail vary
  params["tail_frac_hi"].set(.1, vary=True)
  params["tail_tau_hi"].set(30, vary=True)

  # 2. To fix the low-E tail at a 10% level, tau=30 eV
  params["tail_frac"].set(.1, vary=False)
  params["tail_tau"].set(30, vary=False)

  # 3. To turn off low-E tail
  params["tail_frac"].set(0, vary=False)
  params["tail_tau"].set(vary=False)


Adding or removing the ``_hi`` suffix to/from the parameter names in the examples above will allow you to fix the high-E tail (examples 2 or 3) or to re-enable fitting of the low-E tail (example 1).

Fitting with a quantum efficiency model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to multiply the line models by a model of the quantum efficiency, you can do that. You need a ``qemodel`` function or callable function object that takes an energy (scalar or vector) and returns the corresponding QE. For example, you can use the "Raven1 2019" QE model from `mass.materials`. The filter-stack models are not terribly fast to run, so it's best to compute once, spline the results, and pass that spline as the ``qemodel`` to ``line.model(qemodel=qemodel)``.

.. testcode::

  raven_filters = mass.materials.efficiency_models.filterstack_models["RAVEN1 2019"]
  eknots = np.linspace(100, 20000, 1991)
  qevalues = raven_filters(eknots)
  qemodel = mass.mathstat.interpolate.CubicSpline(eknots, qevalues)

  model = line.model(qemodel=qemodel)
  resultD = model.fit(sim, params, bin_centers=e)
  resultD.plotm()
  # print(resultD.fit_report())

  fit_counts = resultD.params["integral"].value
  localqe= qemodel(mass.STANDARD_FEATURES["MnKAlpha"])[0]
  fit_observed = fit_counts*localqe
  fit_err = resultD.params["integral"].stderr
  if fit_err is None:
      fit_err = fit_counts / N**0.5
  print("Fit finds {:.0f}±{:.0f} counts before QE or {:.0f}±{:.0f} observed. True value {:d}".format(
      fit_counts, fit_err, fit_observed, fit_err*localqe, N))

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Fit finds 173065±547 counts before QE or 102670±325 observed. True value 100000

.. testcode::
  :hide:

  plt.savefig("img/mnka_fit3.png"); plt.close()

.. image:: img/mnka_fit3.png
  :width: 40%


When you fit with a non-trivial QE model, the fit parameters that refer to signal and background intensity all refer to a sensor with an ideal QE=1. These include:

* ``integral``
* ``background``
* ``bg_slope``

That is, the fit values must be multiplied by the local QE to give the number of _observed_ signal counts, background counts per bin, or background slope.
With or without a QE model, "integral" refers to the number of photons that would be seen across all energies (not just in the range being fit).

Fitting a simple Gaussian, Lorentzian, or Voigt function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode::

  e_ctr = 1000.0
  Nsig = 10000
  Nbg = 1000

  sigma = 1.0
  x_gauss = np.random.standard_normal(Nsig)*sigma + e_ctr
  hwhm = 1.0
  x_lorentz = np.random.standard_cauchy(Nsig)*hwhm + e_ctr
  x_voigt = np.random.standard_cauchy(Nsig)*hwhm + np.random.standard_normal(Nsig)*sigma + e_ctr
  bg = np.random.uniform(e_ctr-5, e_ctr+5, size=Nbg)

  # Gaussian fit
  c, b = np.histogram(np.hstack([x_gauss, bg]), 50, [e_ctr-5, e_ctr+5])
  bin_ctr = b[:-1] + (b[1]-b[0]) * 0.5
  line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", e_ctr, 0, 0)
  line.linetype = "Gaussian"
  model = line.model()
  params = model.guess(c, bin_centers=bin_ctr)
  params["fwhm"].set(2.3548*sigma)
  params["background"].set(Nbg/len(c))
  resultG = model.fit(c, params, bin_centers=bin_ctr)
  resultG.plotm()
  # print(resultG.fit_report())

  # Lorentzian fit
  c, b = np.histogram(np.hstack([x_lorentz, bg]), 50, [e_ctr-5, e_ctr+5])
  bin_ctr = b[:-1] + (b[1]-b[0]) * 0.5
  line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", e_ctr, hwhm*2, 0)
  line.linetype = "Lorentzian"
  model = line.model()
  params = model.guess(c, bin_centers=bin_ctr)
  params["fwhm"].set(2.3548*sigma)
  params["background"].set(Nbg/len(c))
  resultL = model.fit(c, params, bin_centers=bin_ctr)
  resultL.plotm()
  # print(resultL.fit_report())

  # Voigt fit
  c, b = np.histogram(np.hstack([x_voigt, bg]), 50, [e_ctr-5, e_ctr+5])
  bin_ctr = b[:-1] + (b[1]-b[0]) * 0.5
  line = mass.fluorescence_lines.SpectralLine.quick_monochromatic_line("testline", e_ctr, hwhm*2, sigma)
  line.linetype = "Voigt"
  model = line.model()
  params = model.guess(c, bin_centers=bin_ctr)
  params["fwhm"].set(2.3548*sigma)
  params["background"].set(Nbg/len(c))
  resultV = model.fit(c, params, bin_centers=bin_ctr)
  resultV.plotm()
  # print(resultV.fit_report())

.. testcode::
  :hide:

  plt.savefig("img/mnka_fitV.png"); plt.close()
  plt.savefig("img/mnka_fitL.png"); plt.close()
  plt.savefig("img/mnka_fitG.png"); plt.close()

.. image:: img/mnka_fitG.png
  :width: 40%

.. image:: img/mnka_fitL.png
  :width: 40%

.. image:: img/mnka_fitV.png
  :width: 40%


How you can use the old, homemade fitters (but don't!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep in mind that the code in this section is considered `deprecated`. You should replace it (see the next section for how) in your own scripts. This explanation is here simply for reference and to help you replace.

.. testcode::

  # Fitters for known lines are instantiated by:
  fitter = mass.MnKAlpha.fitter()
  fitter._have_warned = True  # hide the deprecation warning
  paramA, covar = fitter.fit(sim, e)
  with np.printoptions(3, suppress=True): print(paramA)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [    2.22   5898.807     1.002 49975.02      0.        0.        0.
      25.   ]

.. testcode::
  :hide:

  plt.savefig("img/mnka_fitdeprecated1.png"); plt.close()

.. image:: img/mnka_fitdeprecated1.png
  :width: 40%


Notice that it's on you to remember that the ordering of the ``param`` vector (and rows and columns of the ``covar`` matrix) is:

0. Energy resolution (gaussian FWHM)
1. Energy where the nominal peak is found
2. dPH/dE input-to-energy stretch factor
3. Amplitude (= integrated number of photons times bin width)
4. Mean BG level (counts per bin)
5. BG slope (counts per bin per bin)
6. Tail fraction (0-1, but by default doesn't vary)
7. Tail length (in bins)

To hold a parameter fixed, say the dPH/dE, you need provide a parameter guess, and also remember its code number:

.. testcode::

  paramA[2] = 1.0
  paramB, covarB = fitter.fit(sim, e, paramA, hold=[2])
  with np.printoptions(3, suppress=True): print(paramB)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [    2.204  5898.802     1.    50047.184     0.        0.        0.
      25.   ]

.. testcode::
  :hide:

  plt.savefig("img/mnka_fitdeprecated2.png"); plt.close()

.. image:: img/mnka_fitdeprecated2.png
  :width: 40%



You can allow low-energy tail to exist by setting the last two guess parameters to nonzero values. You can allow it to vary with the `vary_tail` optional argument:

.. testcode::

  paramB[-2:] = 0.1, 30
  paramC, covarC = fitter.fit(sim, e, paramB, hold=[2], plot=False)
  paramD, covarD = fitter.fit(sim, e, paramC, hold=[2], vary_tail=True, vary_bg_slope=True)
  with np.printoptions(3, suppress=True):
      print(paramC)
      print(paramD)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [    2.204  5898.802     1.    50047.184     0.        0.        0.1
      30.   ]
  [    2.194  5898.805     1.    50048.18    -76.213     0.274     0.003
       6.367]

.. testcode::
 :hide:

 plt.savefig("img/mnka_fitdeprecated3.png"); plt.close()

.. image:: img/mnka_fitdeprecated3.png
 :width: 40%


Did you get a ``LinAlgWarning`` when you performed that last fit? I did! This is part of what we're trying to avoid with the new fitters.

How to convert your personal analysis code from the old to the new method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Notice how the old "Fitter" methods are very simple to use in the usual case, but increasingly klunky if you want to vary what usually doesn't vary, to hold what usually isn't held, and to skip plotting, etc etc?

An overview of how to convert is:

#. Get a Model object instead of a Fitter object.
#. Use ``p=model.guess(data, bin_centers=e)`` to create a heuristic for the starting parameters.
#. Change starting values and toggle the ``vary`` attribute on parameters, as needed. For example: ``p["dph_de"].set(1.0, vary=False)``
#. Use ``result=model.fit(data, p, bin_centers=e)`` to perform the fit and store the result.
#. The result holds many attributes and methods (see `MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#minimizerresult-the-optimization-result>`_ for full documentation). These include:

  * ``result.params`` = the model's best-fit parameters object
  * ``result.best_values`` = a dictionary of the best-fit parameter values
  * ``result.best_fit`` = the model's y-values at the best-fit parameter values
  * ``result.chisqr`` = the chi-squared statistic of the fit (here, -2log(L))
  * ``result.covar`` = the computed covariance
  * ``result.fit_report()`` = return a pretty-printed string reporting on the fit
  * ``result.plot_fit()`` = make a plot of the data and fit
  * ``result.plot_residuals()`` = make a plot of the residuals (fit-data)
  * ``result.plot()`` = make a plot of the data, fit, and residuals


One detail that's changed: the new models parameterize the tau values (scale lengths of exponential tails) in eV units. The old fitters assumed tau were given in units of bins. Another is that the parameter "integral" refers to the integrated number of counts across all energies; the old parameter "amplitude" was the same but scaled by the bin width in eV. The old way didn't make sense, but that's how it was.

To do
^^^^^

* [x] We probably should restructure the ``SpectralLine``, ``GenericLineModel``, and perhaps also the older ``LineFitter`` objects such that the specific versions for (say) Mn Kα become not subclasses but instances of them. See `issue 182 <https://bitbucket.org/joe_fowler/mass/issues/182/does-creation-of-3-classes-per-spectral>`_ on the question of whether this change might speed up loading of MASS. Done by PR#120.
* [x] Add to ``GenericLineModel`` one or more methods to make plots comparing data and fit with parameter values printed on the plot.
* [x] The LMFIT view of models is such that we would probably find it easy to fit one histogram for the sum of (say) a Mn Kα and a Cr Kβ line simultaneously. Add features to our object, as needed, and document the procedure here.
* [ ] We could implement convolution between two models (see just below `CompositeModel <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_ in the docs for how to do this).
* [ ] At some point, we ought to remove the deprecated ``LineFitter`` object and subclasses thereof.
