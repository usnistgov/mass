## Note on version numbers of Mass

* **0.5.0** October 2016
Galen reorganized the source tree. That alone seems like worth a minor version number.

* **0.4.4** August 2016
Young changed the version number, but I do not know why. (JF)

* **0.4.3** May 2016
Reorganized code that fits spectral line models (Voigt, Gaussian, and specific
    Kalpha or KBeta lines). Added low-E tails to the Voigt and Gaussian fitters.
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
