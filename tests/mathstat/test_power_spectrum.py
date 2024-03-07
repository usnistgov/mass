import pytest
import numpy as np
import mass

from mass.mathstat.power_spectrum import computeSpectrum


class Test_PowerSpectrum:
    @staticmethod
    def test_basic():
        for n in [1000, 2000, 2500, 33333, 10000]:
            for segfactor in [1, 4, 10, 33]:
                data = np.zeros(n)
                _psd = computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
                _f, _psd = computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)
                _psd = computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
                _f, _psd = computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)

    @staticmethod
    def test_values():
        f, psd = mass.mathstat.power_spectrum.computeSpectrum(
            np.arange(10), segfactor=1, dt=1)
        expected = [405., 52.36067977, 14.47213595, 7.63932023, 5.52786405, 5.]
        for a, b in zip(psd, expected):
            assert a == pytest.approx(b)
        expected = np.linspace(0, 0.5, 6)
        for a, b in zip(f, expected):
            assert a == pytest.approx(b)
