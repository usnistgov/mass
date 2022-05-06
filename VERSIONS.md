## Note on version numbers of Mass

**0.7.8** November 29, 2021 - ?
* Fix issue 217: problem with rounding error giving negative values in fit model.
* Fix issue 218: `distutils.version` is deprecated.
* Fix issue 219: should raise Exception when `MLEModel.fit(...weights=w)` for non-None weights.
* Put `mass.materials` in default sub-packages for pip installation.


**0.7.7** November 29, 2021

* Fix issue 205: add line shapes of Se, Y, Zr K lines from Ito et al 2020.
* Fix issue 206: update calibration curves: uncertainty and better smoothing spline theory.
* Fix issue 207: update usage of numpy/Cython; standardize imports of numpy/scipy/pylab as np/sp/plt.
* Fix issue 208: allow save/restore of filters to HDF5 even when too long for HDF5 attributes.
* Fix issue 209: replace dependency xraylib with xraydb.
* Fix issue 210: add line shapes of Ir, Pt, Au, Pb, Bi L lines.
* Fix issue 211: hide math warnings during pytest testing.


**0.7.6** November 24, 2020

* Fix issue 189: clean up top-level directory and pytest configuration.
* Fix issue 191: typos in our CrKAlpha line and apparent typo in Hölzer on the FeKAlpha line.
* Fix issue 192: some problem with using `MLEModel` for fits that aren't spectral lines.
* Fix issue 193: problem in fitting to histograms with `dtype=np.float32` for the bin edges.
* Fix issue 194: triggers the too-narrow-bins fit error when it should not, if dPH/dE >> 1.
* Fix issue 196: reorganize x-ray filter code; add Horton 2018 design.
* Fix issue 197: work around a problem opening certain noise HDF5 files.
* Fix issue 199: remove Qt4 (a Python 2-only package) and GUIs based on it.
* Fix issue 200: work with h5py version 3 (reads strings as bytes type).
* Fix issue 201: add examples in line_fitting.rst to show how to fit Gauss, Lorentz, or Voigt
* Fix issue 202: fluorescence models use parameter "integral" instead of ill-defined "amplitude".
* Fix issue 203: autocal uses new-style fits.
* Fix issue 204: add optional quantum efficiency model to line fits: now can fit line times QE.
* Replace line fitting documentation with a doctest document.


**0.7.5** March 31, 2020

* This is the last version that supports Python 2.7.
* `mass.spectra` contains dictionary of class instances instead of class objects.
* Fix long lines and other pep8 warnings. Fix the Make targets pep8 and lint for Python3.

**0.7.4** March 26, 2020

* Make `mass` installable by pip.
* Add entry points so some scripts are automatically installed.
* Fix minor annoyances and pep-8 violations.
* Fix issue 187: cannot plot_average_pulses or plot_filters.
* Fix issue 188: annoying log messages upon reading older LJH files.

**0.7.3** January 2020-March 2020

* Reorganize code to live in mass/... instead of src/mass/...; always test installed Mass.
* Can now use "setup.py develop" to link `site-packages` back to local mass, so "installed Mass" can be your local.
* Fix issue 172: add an intrinsic Gaussian in SpectralLine: components can now be Voigts, not Lorentzians.
* Spectral lines can have a 2-sided exponential tail, not only a low-energy tail.
* Instantiating a `GenericLineModel` with the default `has_tails=False` forbids exponential tails, taking a shortcut around that (slow) code.
* Add doc/LineFitting.md to document use of the new fitting Models.
* Fix issue 165: use of LMFIT for fitting is now the default. Deprecate the old `*Fitter` objects.
* Fix issue 163: problems when the scaling of parameters (or their uncertainty) varies widely between parameters.
* Start using [pytest](https://docs.pytest.org/en/latest/) for package testing instead of our own `runtests.py` script.
* Fix issue 181: deprecated `import imp` was removed; no longer needed with pytest.
* Remove global dictionaries `fitter_classes` and `model_classes`. Replace with functions `make_line_fitter` and `make_line_model` that generate instances from a `SpectralLine` object. Fixes issue 182.
* Fix issues 183-184: problems in `mass.off`.

**0.7.2** December 2019-January 2020

* Allow ArrivalTimeSafeFilter to be full length, skipping the "shift-1" possibility that DASTARD doesn't need.
* Refactor calling `MicrocalDataSet` method `compute_filter()` into `compute_5lag_filter()` and `compute_ats_filter()`.
* Add scripts/ljh2off.py and `MicrocalDataSet.projectors_to_hdf5()`
* Fix issue 177: read new and old stored filters.
* projectors include v_dv, noise_rms_median, noise_rms_sigma
* refactor projectors to `pulse_model.PulseModel`
* `ExperimentStateFile` stores unique states as slices
* faster `getOffAttr` (OFF file performance) for lists of 100s of slices
* `data.refreshFromFiles` added, this parses the experiment state file from the last point it was parsed to, then updates the off file mmaps, then recalculates all `ds._statesDict` based on the new states and info in the OFF file.
* Has tests on the internals of `ExperimentStatefile` and `OffFile`.
* Fix reading in Py3 of HDF5 calbrations written by Py2, and other Py3 problems.
* Silent testing.
* Support OFF files version 0.3.0.

**0.7.1** November 2019-December 2019

* More neon HCI lines.
* compute_newfilter(...) takes option to turn off shift-by-1 (useful when working on Dastard data).
* Fix issue 175: use of basestring isn't Python 3 compatible.
* Fix issue 176: band limited ArrivalTimeSafeFilter were not DC-insensitive.
* In the above, we fundamentally changed the exact computation of a filter so filtered data _will_ be changed by this version of MASS.

**0.7.0** December 2018-October 2019

* Add partial use of [LMFit](https://lmfit.github.io/lmfit-py/index.html) for fitting.
* Add code for analysis of OFF files.
* Add K line shapes for elements K, S, Cl. Also Nb (K-beta only).
* Add fitters for certain highly charged ions from NIST Atomic Spectra Database.
* Many other small changes, particularly for use at the EBIT.

**0.6.6** August-October 2019  

* Fix issue 162: overcome biased fits when bins are too wide by allowing numerical integration in bin. How densely to sample can be chosen by user or a heuristic.  
* Fix issue 164: silently "correct" the off-by-3 error on `nPresamples` in MATTER-produced LJH files.  
* Fix issue 167: add molybdenum L line data from Mendenhall 2019.  
* Fix issue 147: auto_cuts fails if other cuts already exists.  
* Fix issue 140: pass tests on Python version 3.  
* Fix issue 166: could not run phase_correct twice on same data (HDF5 error).

**0.6.5** July 2018-June 2019  

* Factor phase correction into its own class and source file.
* Add lineshape models for bromine K and tungsten L lines.
* Allow read-only opening of HDF5 files (don't try to update attributes in that case).
* Fix bugs in `EnergyCalibration.plot()`

**0.6.4** May-July 2018  

* Fix issue 156: phase correction with low statistics.
* Fix issue 157: noise fails if noise records aren't continuous.
* Fix issue 158: certain test failures.
* Make phase_correct method2017=True the default.
* Pep-8 fixes.

**0.6.3** May 2018  

* Refactor Fluorescence Line and Fitter Code.  
* Fix incorrect fitters where Chantler reported Voigt peak height instead of area, such as `VKalpha`.  
* Add `fluorescence.md` doc file.

**0.6.2** November 2017 - April 2018  

* Fixed `setup.py` to be compatible with Pip 10.0.0+.  
* Fixed some minor bugs and redundancies.  
* Added `ToeplitzWhitener.solveW` method and tests for `ToeplitzWhitener`.    
* New filtering API: `f_3db` and `fmax` are set only at `filter.compute()` time.    
* Added arguments to `ArrivalTimeSafeFilter.compute()` so you can emulate shorter records.  
* Change `plot_average_pulses` and `plot_noise` to leave out bad channels by default.  

**0.6.1** September-November 2017  

* Added some features to support analysis of microwave MUX data.  
* Added some random-matrix techniques to `mass.mathstat.utilities`.  
* Used a decorator to add methods to `TESGroup` that loop over channel objects.  
* Galen added configuration of Sphinx to auto-generate HTML documentation from the docstrings.  

**0.6.0** September 2017  

* Fixing a number of outstanding problems (issues 88-100 at least).  
* Fixed problems in new-style filtering.  
* Removed requirement that user specify a peak time.  
* Lots of cleaning up: Remove code that is not still used or correct, or move it to nonstandard. Remove julia subdirectory and workarounds.py.

**0.5.3** March-April 2017  

* Joe added entropy, cross-entropy, and Kullback-Leibler divergence computation on distributions assuming a Laplace kernel-density estimator.  
* Added a new, tentative "method2017" for phase correction.  
* Also, bug fixes.

**0.5.2** February 2017  

* Young is working on some things.  
* Joe added LJH modifier.  
* Deprecated ROOT file format handler.  

**0.5.1** November 2016  

* Fix a bug in how filters are restored from disk (issue 82).

**0.5.0** October 2016  

* Galen reorganized the source tree. That alone seems like worth a minor version number.  
* Also created some 69 regression and other unit tests, which all pass now. Several bugs were fixed in the process.  
* Added file pattern-matching to TESGroup constructor.  
* New-style optimal filters became the default.
* Fixed issues 60, 62, and most of 70-81.

**0.4.4** August 2016  

* Young changed the version number, but I (JF) do not know why.

**0.4.3** May 2016  

* Reorganized code that fits spectral line models (Voigt, Gaussian, and specific K&alpha; or K&beta; lines).  
* Added low-E tails to the Voigt and Gaussian fitters.  
* Fixed issues #45-51, except for 48.

**0.4.2** October 2015  

* Return the main objects to pure Python, with Cython subclasses. Also not sure this is a good idea, but we'll experiment with it.

**0.4.1** October 1, 2015   

* Uses pure Cython for channel.pyx. We decided that this experiment was not a positive development, but it was worth trying.

**0.4.0** September 2015   

* New filtering methods available from Joe. Still experimental! Don't set ```MicrocalDataSet._use_new_filters = True``` unless you know what you're doing!
