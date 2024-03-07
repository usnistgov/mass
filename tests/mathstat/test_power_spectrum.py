import pytest
import numpy as np
import mass
import pylab as plt

from mass.mathstat.power_spectrum import computeSpectrum


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
def test_orthognal_to_exponential_filter():
        record_len = 100
        npre = 50
        n_noise_records = 50
        frametime_s = 1e-5
        filter_orthogonal_to_exponential_time_constant_ms = 5
        avg_pulse_values = 1000*np.arange(record_len) # we're just testing math here, this is easier than making an actual pulse shape
        noise_traces = np.random.randn(record_len, n_noise_records)
        noise_autocorr = mass.mathstat.power_spectrum.autocorrelation_broken_from_pulses(noise_traces)
        noise_psd_calculator = mass.mathstat.power_spectrum.PowerSpectrum(record_len // 2, dt=frametime_s)
        window = np.ones(record_len)
        for i in range(n_noise_records):
            noise_psd_calculator.addDataSegment(noise_traces[:,i], window=window)
        # test that we can plot the noise spectrum
        noise_psd_calculator.plot()
        plt.close()
        noise_psd = noise_psd_calculator.spectrum()
        filter_obj = mass.ExperimentalFilter(avg_pulse_values, npre,
                                 noise_psd, sample_time=frametime_s, 
                                 noise_autocorr=noise_autocorr,
                                 tau=filter_orthogonal_to_exponential_time_constant_ms)
        filter_obj.compute()
        print("predicted resolutions")
        filter_obj.report(std_energy=1000)
        chosen_filter = filter_obj.filt_noexpcon
        np.dot(chosen_filter, avg_pulse_values)
