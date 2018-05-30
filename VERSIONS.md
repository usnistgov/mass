## Note on version numbers of Mass
* **0.6.4** May- 2018
Fix issue 156 (phase correction with low statistics)

* **0.6.3** May 2018
Refactor Fluorescence Line and Fitter Code
Fix incorrect fitters where Chantler reported voigt peak height, such as VKalpha
Add fluorescence.md doc file

* **0.6.2** November 2017 - April 2018
Fixed setup.py to be compatible with Pip 10.0.0+.
Fixed some minor bugs and redundancies.
Added ToeplitzWhitener.solveW method and tests for ToeplitzWhitener.  
New filtering API: f_3db and fmax are set only at filter.compute() time.  
Added arguments to ArrivalTimeSafeFilter.compute() so you can emulate shorter records.  
Change plot_average_pulses and plot_noise to leave out bad channels by default.  

* **0.6.1** September-November 2017    
Added some features to support analysis of microwave MUX data.  
Added some random-matrix techniques to `mass.mathstat.utilities`.  
Used a decorator to add methods to TESGroup that loop over channel objects.  
Galen added configuration of Sphinx to auto-generate HTML documentation from the docstrings.  

* **0.6.0** September 2017  
Fixing a number of outstanding problems (issues 88-100 at least).  
Fixed problems in new-style filtering.  
Removed requirement that user specify a peak time.  
Lots of cleaning up: Remove code that is not still used or correct, or move it to nonstandard. Remove julia subdirectory and workarounds.py.

* **0.5.3** March-April 2017  
Joe added entropy, cross-entropy, and Kullback-Leibler divergence computation on distributions assuming a Laplace kernel-density estimator.  
Added a new, tentative "method2017" for phase correction.  
Also, bug fixes.

* **0.5.2** February 2017  
Young is working on some things.  
Joe added LJH modifier.  
Deprecated ROOT file format handler.  

* **0.5.1** November 2016  
Fix a bug in how filters are restored from disk (issue 82).

* **0.5.0** October 2016  
Galen reorganized the source tree. That alone seems like worth a minor version number.  
Also created some 69 regression and other unit tests, which all pass now. Several bugs were fixed in the process.  
Added file pattern-matching to TESGroup constructor.  
New-style optimal filters became the default. Fixed issues 60, 62, and most of 70-81.

* **0.4.4** August 2016  
Young changed the version number, but I (JF) do not know why.

* **0.4.3** May 2016  
Reorganized code that fits spectral line models (Voigt, Gaussian, and specific K&alpha; or K&beta; lines).  
Added low-E tails to the Voigt and Gaussian fitters.  
Fixed issues #45-51, except for 48.

* **0.4.2** October 2015  
Return the main objects to pure Python, with Cython subclasses. Also not sure this is a good idea, but we'll experiment with it.

* **0.4.1** October 1, 2015   
Uses pure Cython for channel.pyx. We decided that this experiment was not a positive development, but it was worth trying.

* **0.4.0** September 2015  
New filtering methods available from Joe. Still experimental! Don't set ```MicrocalDataSet._use_new_filters = True``` unless you know what you're doing!
