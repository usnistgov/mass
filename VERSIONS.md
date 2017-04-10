## Note on version numbers of Mass

* **0.5.3** March-April 2017
Joe added entropy, cross-entropy, and Kullback-Leibler divergence computation
on distributions assuming a Laplace kernel-density estimator. Also, bug fixes.
Added a new, tentative "method2017" for phase correction.

* **0.5.2** February 2017
Young is working on some things.
Joe added LJH modifier. Deprecated ROOT file format handler.

* **0.5.1** November 2016
Fix a bug in how filters are restored from disk (issue 82).

* **0.5.0** October 2016
Galen reorganized the source tree. That alone seems like worth a minor version number.
Also created some 69 regression and other unit tests, which all pass now. Several
bugs were fixed in the process.  Added file pattern-matching to TESGroup constructor.
New-style optimal filters became the default. Fixed issues 60, 62, and most of 70-81.

* **0.4.4** August 2016
Young changed the version number, but I do not know why. (JF)

* **0.4.3** May 2016
Reorganized code that fits spectral line models (Voigt, Gaussian, and specific
    K&alpha; or K&beta; lines). Added low-E tails to the Voigt and Gaussian fitters.
    Fixed issues #45-51, except for 48.

* **0.4.2** October 2015
Return the main objects to pure Python, with Cython subclasses. Also not sure
    this is a good idea, but we'll experiment with it.

* **0.4.1** October 1, 2015
Uses pure Cython for channel.pyx. We decided that this experiment was not a
    positive development, but it was worth trying.

* **0.4.0** September 2015
New filtering methods available from Joe. Still experimental! Don't set
    ```MicrocalDataSet._use_new_filters = True``` unless you know what you're
    doing!
