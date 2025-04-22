Optimal filtering
=======================

The `FilterMaker` interface
------------------------------

As of MASS version 0.8.6 (December 2024), there is an entirely new API for optimal filtering. It is based on `data classes <https://docs.python.org/3/library/dataclasses.html>`_ (object ``dataclasses.dataclass``), a modern approach to Python objects based on having set of attributes fixed at creation time. When the dataclass is "frozen" (as in this case), it also does not allow changing the _values_ of these attributes. Our intention with the new API is offer a range of objects that can perform optimal filtering, or create objects that do, but rejecting the proliferation of incompatible features that used to appear in slightly different flavors of filters.

This API consists of two key objects:
1. The ``Filter`` is a specific implementation of an optimal filter, designed to be used in one-lag or five-lag mode, and with fixed choices about low-pass filtering of the filter's values, or about giving zero weight to a number of initial or final samples in a record. Offers a `filter_records(r)` method to apply its optimal filter to one or more pulse records `r`.
1. The ``FilterMaker`` contains information about one channel's expected signal and noise, and is able to create various objects of the type ``Filter`` (or subtypes thereof).

The user first creates a ``FilterMaker`` from the analyzed noise and signal, the uses it to generate an optimal filter (from a subclass of ``Filter``) with the desired properties. That object has a `filter_records(...)` method.

.. testcode::

    import numpy as np
    import mass

    n = 500
    Maxsignal = 1000.0
    sigma_noise = 1.0
    tau = [.05, .25]
    t = np.linspace(-1, 1, n+4)
    npre = (t < 0).sum()
    signal = (np.exp(-t/tau[1]) - np.exp(-t/tau[0]) )
    signal[t <= 0] = 0
    signal *= Maxsignal / signal.max()
    truncated_signal = signal[2:-2]

    noise_covar = np.zeros(n)
    noise_covar[0] = sigma_noise**2

    maker = mass.FilterMaker(signal, npre, noise_covar, peak=Maxsignal)
    F5 = maker.compute_5lag()

    print(f"Filter peak value:            {F5.nominal_peak:.1f}")
    print(f"Filter rms value:             {F5.variance**0.5:.4f}")
    print(f"Filter predicted V/dV (FWHM): {F5.predicted_v_over_dv:.4f}")

This code should produce an optimal filter ``F5`` and a filter maker ``maker`` and generate the following output

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

    Filter peak value:            1000.0
    Filter rms value:             0.1549
    Filter predicted V/dV (FWHM): 2741.6517



A test of normalization and filter variance
------------------------------

.. testcode::

    import numpy as np
    import mass

    def verify_close(x, y, rtol=1e-5, topic=None):
        if topic is not None:
            print(f"Checking {topic:20s}: ", end="")
        isclose = np.isclose(x, y, rtol=rtol)
        print(f"x={x:.4e}, y={y:.4e} are close to each other? {isclose}")
        assert isclose

    def test_mass_5lag_filters(Maxsignal=100.0, sigma_noise=1.0, n=500):
        tau = [.05, .25]
        t = np.linspace(-1, 1, n+4)
        npre = (t < 0).sum()
        signal = (np.exp(-t/tau[1]) - np.exp(-t/tau[0]) )
        signal[t <= 0] = 0
        signal *= Maxsignal / signal.max()
        truncated_signal = signal[2:-2]

        noise_covar = np.zeros(n)
        noise_covar[0] = sigma_noise**2
        maker = mass.FilterMaker(signal, npre, noise_covar, peak=Maxsignal)
        F5 = maker.compute_5lag()

        # Check filter's normalization
        f = F5.values
        verify_close(Maxsignal, f.dot(truncated_signal), rtol=1e-5, topic = "Filter normalization")

        # Check filter's variance 
        expected_dV = sigma_noise / n**0.5 * signal.max()/truncated_signal.std()
        verify_close(expected_dV, F5.variance**0.5, rtol=1e-5, topic="Expected variance")

        # Check filter's V/dV calculation
        fwhm_sigma_ratio = np.sqrt(8*np.log(2))
        expected_V_dV = Maxsignal / (expected_dV * fwhm_sigma_ratio)
        verify_close(expected_V_dV, F5.predicted_v_over_dv, rtol=1e-5, topic="Expected V/\u03b4v")
        print()

    test_mass_5lag_filters(100, 1.0, 500)
    test_mass_5lag_filters(400, 1.0, 500)
    test_mass_5lag_filters(100, 1.0, 1000)
    test_mass_5lag_filters(100, 2.0, 500)

These four tests should yield the following output:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

    Checking Filter normalization: x=1.0000e+02, y=1.0000e+02 are close to each other? True
    Checking Expected variance   : x=1.5489e-01, y=1.5489e-01 are close to each other? True
    Checking Expected V/δv       : x=2.7417e+02, y=2.7417e+02 are close to each other? True

    Checking Filter normalization: x=4.0000e+02, y=4.0000e+02 are close to each other? True
    Checking Expected variance   : x=1.5489e-01, y=1.5489e-01 are close to each other? True
    Checking Expected V/δv       : x=1.0967e+03, y=1.0967e+03 are close to each other? True

    Checking Filter normalization: x=1.0000e+02, y=1.0000e+02 are close to each other? True
    Checking Expected variance   : x=1.0963e-01, y=1.0963e-01 are close to each other? True
    Checking Expected V/δv       : x=3.8734e+02, y=3.8734e+02 are close to each other? True

    Checking Filter normalization: x=1.0000e+02, y=1.0000e+02 are close to each other? True
    Checking Expected variance   : x=3.0978e-01, y=3.0978e-01 are close to each other? True
    Checking Expected V/δv       : x=1.3708e+02, y=1.3708e+02 are close to each other? True
