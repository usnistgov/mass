# Fitting spectral line models to data in MASS

Designed for MASS version 0.7.3.

Joe Fowler, January 2020.

Beginning in January 2020, we are trying to stop using our previous method, in which we wrote our own modification to the Levenberg-Marquardt optimizer in order to reach maximum-likelihood (not least-squares) fits. This appears in mass as the `LineFitter` class and its subclasses. In its place, we want to use the [LMFIT package](https://lmfit.github.io/lmfit-py/) and the new MASS class `GenericLineModel` and its subclasses.

## LMFit vs Scipy vs our homemade method

LMFIT has numerous advantages over the basic `scipy.optimize` module. Quoting from the LMFIT documentation, the user can:

* forget about the order of variables and refer to Parameters by meaningful names.
* place bounds on Parameters as attributes, without worrying about preserving the order of arrays for variables and boundaries.
* fix Parameters, without having to rewrite the objective function.
* place algebraic constraints on Parameters.

Only some of these are implemented in the original MASS fitters, and even they are not all implemented in the most robust possible way. The one disadvantage of the core LMFIT package is that it minimizes the sum of squares of a vector instead of maximizing the Poisson likelihood. This is easily remedied, however, by replacing the usual computation of residuals with one that computes the square root of the Poisson likelihood contribution from each bin. Voilá! A maximum likelihood fitter for histograms.

Advantages of LMFIT over the homemade method of the `LineFitter` include:

* Users can forget about the order of variables and refer to Parameters by meaningful names.
* Users can place algebraic constraints on Parameters.
* The interface for setting upper/lower bounds on parameters and for varying or fixing them is much more elegant, memorable, and simple than our homemade version.
* It ultimately wraps the `scipy.optimize` package and therefore inherits all of its advantages:
  * a choice of over a dozen optimizers with highly technical documentation,
  * some optimizers that aim for true global (not just local) optimization, and
  * the countless expert-years that have been invested in perfecting it.
* LMFIT automatically computes numerous statistics of each fit including estimated uncertainties, correlations, and multiple quality-of-fit statistics (information criteria as well as chi-square) and offers user-friendly fit reports. See the [`MinimizerResult`](https://lmfit.github.io/lmfit-py/fitting.html#minimizerresult-the-optimization-result) object.
* It's the work of Matt Newville, an x-ray scientist responsible for the excellent [ifeffit](http://cars9.uchicago.edu/ifeffit/) and its successor [Larch](https://xraypy.github.io/xraylarch/).
* Above all, *its documentation is complete, already written, and maintained by not-us.*

## Usage guide

This overview is hardly complete, but we hope it can be a quick-start guide and also hint at how you can convert your own analysis work from the old to the new, preferred fitting methods.

### The underlying spectral line shape models

Objects of the type `SpectralLine` encode the line shape of a fluorescence line, as a sum of Voigt or Lorentzian distributions. Because they inherit from `scipy.stats.rv_continuous`, they allow computation of cumulative distribution functions and the simulation of data drawn from the distribution. An example of the creation and usage is:

```python
import mass
import pylab as plt

# In general, known lines are instantiated by:
line = mass.spectrum_classes["MnKAlpha"]()
# But the following is a shortcut for many lines:
line = mass.MnKAlpha()
N = 100000
energies = line.rvs(size=N, instrument_gaussian_fwhm=2.2)  # draw from the distribution
plt.clf()
sim, bin_edges, _ = plt.hist(energies, 120, [5865, 5925], histtype="step");
binsize = bin_edges[1] - bin_edges[0]
e = bin_edges[:-1] + 0.5*binsize
plt.plot(e, line(e)*N*binsize, "k")
```

The `SpectralLine` object is useful to you if you need to generate simulated data, or to plot a line shape, as shown above. Both the new fitting "model" objects and the old "fitter" objects use the `SpectralLine` object to hold line shape information. You don't need to create a `SpectralLine` object for fitting, though; it will be done automatically.


### How to use the new, LMFIT-based models for fitting

The simplest case requires only 3 steps: create a model instance from a `SpectralLine`, guess its parameters from the data, and perform a fit with this guess. Unlike the old fitters, plotting is not done as part of the fit--you have to do that separately.

```python
model = mass.make_line_model(line)
params = model.guess(sim, bin_centers=e)
resultA = model.fit(sim, params, bin_centers=e)

# Repeat but with dPH/dE held at 1.
params = resultA.params.copy()
params["dph_de"].set(1.0, vary=False)
resultB = model.fit(sim, params, bin_centers=e)
print(resultB.fit_report())
resultB.plot()
# The best-fit params are found in resultB.params
# and a dictionary of their values is resultB.best_values.
# The parameters given as an argument to fit are unchanged.
```

Notice when you report the fit (or check the contents of the `params` or `resultB.params` objects), there are no parameters referring to exponential tails of a Bortels response. That's because the default fitter assumes a Gaussian response. If you want tails, that's a constructor argument:

```python
model = mass.make_line_model(line, has_tails=True)
params = model.guess(sim, bin_centers=e)
params["dph_de"].set(1.0, vary=False)
resultC = model.fit(sim, params, bin_centers=e)
print(resultC.fit_report())
resultC.plot()
```

By default, the `has_tails=True` will set up a non-zero low-energy tail and allow it to vary, while the high-energy tail is set to zero amplitude and doesn't vary. Use these numbered examples if you want to fit for a high-energy tail (1), to fix the low-E tail at some non-zero level (2) or to turn off the low-E tail completely (3):

```python
# 1. To let the high-E tail vary
params["tail_frac_hi"].set(.1, vary=True)
params["tail_tau_hi"].set(30, vary=True)

# 2. To fix the low-E tail at a 10% level, tau=30 eV
params["tail_frac"].set(.1, vary=False)
params["tail_tau"].set(30, vary=False)

# 3. To turn off low-E tail
params["tail_frac"].set(0, vary=False)
params["tail_tau"].set(vary=False)
```

Adding or removing the `_hi` suffix to/from the parameter names in the examples above will allow you to fix the high-E tail (2 or 3) or to re-enable fitting of the low-E tail (1).


### How you can use the old, homemade fitters (but don't!)

Keep in mind that the code in this section is considered deprecated. You should replace it (see the next section for how) in your own scripts. This explanation is here simply for reference and to help you replace.

```python
# Fitters for known lines are instantiated by:
fitter = mass.make_line_fitter(line)
paramA, covar = fitter.fit(sim, e)
print(paramA)
```

Notice that it's on you to remember that the ordering of the `param` vector (and rows and columns of the `covar` matrix) is:
0. Energy resolution (gaussian FWHM)
1. Energy where the nominal peak is found
1. dPH/dE input-to-energy stretch factor
1. Amplitude (= integrated number of photons times bin width)
1. Mean BG level (counts per bin)
1. BG slope (counts per bin per bin)
1. Tail fraction (0-1, but by default doesn't vary)
1. Tail length (in bins)

To hold a parameter fixed, say the dPH/dE, you need provide a parameter guess, and also remember its code number:

```python
paramA[2] = 1.0
paramB, covarB = fitter.fit(sim, e, paramA, hold=[2])
print(paramB)
```

You can allow low-energy tail to exist by setting the last two guess parameters to nonzero values. You can allow it to vary with the `vary_tail` optional argument:

```python
paramB[-2:] = 0.1, 30
paramC, covarC = fitter.fit(sim, e, paramB, hold=[2], plot=False)
print(paramC)
paramD, covarD = fitter.fit(sim, e, paramC, hold=[2],
                            vary_tail=True, vary_bg_slope=True)
print(paramD)
```

Did you get a `LinAlgWarning` when you performed that last fit? I did! This is part of what we're trying to avoid with the new fitters.

### How to convert your personal analysis code from the old to the new method

Notice how the old "Fitter" methods are very simple to use in the usual case, but increasingly klunky if you want to vary what usually doesn't vary, to hold what usually isn't held, and to skip plotting, etc etc?

An overview of how to convert is:

1. Get a Model object instead of a fitter object.
1. Use `p=model.guess(data, bin_centers=e)` to create a heuristic for the starting parameters.
1. Change starting values and toggle the `vary` attribute on parameters, as needed.
1. Use `result=model.fit(data, p, bin_centers=e)` to perform the fit and store the result.
1. The result holds many attributes and methods (see [MinimizerResult](https://lmfit.github.io/lmfit-py/fitting.html#minimizerresult-the-optimization-result) for full documentation). These include:
  * `result.params` = the model's best-fit parameters object
  * `result.best_values` = a dictionary of the best-fit parameter values
  * `result.best_fit` = the model's y-values at the best-fit parameter values
  * `result.chisqr` = the chi-squared statistic of the fit (here, -2log(L))
  * `result.covar` = the computed covariance
  * `result.fit_report()` = return a pretty-printed string reporting on the fit
  * `result.plot_fit()` = make a plot of the data and fit
  * `result.plot_residuals()` = make a plot of the residuals (fit-data)
  * `result.plot()` = make a plot of the data, fit, and residuals


One detail that's changed: the new models parameterize the tau values (scale lengths of exponential tails) in eV units. The old fitters assumed tau were given in units of bins.

## To do

* [ ] We probably should restructure the `SpectralLine`, `GenericLineModel`, and perhaps also the older `LineFitter` objects such that the specific versions for (say) Mn Kα become not subclasses but instances of them. See [issue 182](https://bitbucket.org/joe_fowler/mass/issues/182/does-creation-of-3-classes-per-spectral) on the question of whether this change might speed up loading of MASS.
* [ ] Add to `GenericLineModel` one or more methods to make plots comparing data and fit with parameter values printed on the plot.
* [ ] The LMFIT view of models is such that we would probably find it easy to fit one histogram for the sum of (say) a Mn Kα and a Cr Kβ line simultaneously. Add features to our object, as needed, and document the procedure here.
* [ ] We could implement convolution between two models (see just below [CompositeModel](https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel) in the docs for how to do this).
* [ ] At some point, we ought to remove the deprecated `LineFitter` object and subclasses thereof.
