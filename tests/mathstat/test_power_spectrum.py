import pytest
import numpy as np
import pylab as plt
import mass

from mass.mathstat.power_spectrum import computeSpectrum

rng = np.random.default_rng(34234)


def test_basic():
    for n in [1000, 2000, 2500, 33333, 10000]:
        for segfactor in [1, 4, 10, 33]:
            data = np.zeros(n)
            _psd = computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
            _f, _psd = computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)
            _psd = computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
            _f, _psd = computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)


def test_values():
    f, psd = mass.mathstat.power_spectrum.computeSpectrum(
        np.arange(10), segfactor=1, dt=1)
    expected = [405., 52.36067977, 14.47213595, 7.63932023, 5.52786405, 5.]
    for a, b in zip(psd, expected):
        assert a == pytest.approx(b)
    expected = np.linspace(0, 0.5, 6)
    for a, b in zip(f, expected):
        assert a == pytest.approx(b)


# test functions used in the oct23 true bq analysis
# where the pulses did not live in ljh files, instead the pulses needed to be passed
# explicitly as arguments to the psd and autocorr calculation
#
# This documents how power spectrum and noise autocorrelation can be computed,
# and how to use them to create a `FilterMaker` and in turn a `Filter`.
def test_creating_filter_from_non_LJH_data():
    record_len = 100
    npre = 50
    n_noise_records = 50
    frametime_s = 1e-5
    avg_pulse_values = 1000 * np.arange(record_len, dtype=float)
    # ^ we're just testing math here, this "average pulse" is easier than making an actual pulse shape

    noise_traces = rng.standard_normal((record_len, n_noise_records))
    noise_autocorr = mass.mathstat.power_spectrum.autocorrelation_broken_from_pulses(noise_traces)
    noise_psd_calculator = mass.mathstat.power_spectrum.PowerSpectrum(record_len // 2, dt=frametime_s)
    window = np.ones(record_len)
    for i in range(n_noise_records):
        noise_psd_calculator.addDataSegment(noise_traces[:, i], window=window)
    # test that we can plot the noise spectrum
    noise_psd_calculator.plot()
    plt.close()
    noise_psd = noise_psd_calculator.spectrum()

    maker = mass.FilterMaker(avg_pulse_values, npre, noise_autocorr=noise_autocorr,
                             noise_psd=noise_psd, sample_time_sec=frametime_s)
    filter_obj = maker.compute_5lag()
    print("predicted resolutions")
    filter_obj.report(std_energy=1000)
    chosen_filter = filter_obj.values
    np.dot(chosen_filter, avg_pulse_values[2:-2])
