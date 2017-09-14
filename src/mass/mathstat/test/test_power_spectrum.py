import unittest
import numpy as np
import mass


class Test_PowerSpectrum(unittest.TestCase):
    def test_basic(self):
        for n in [1000,2000,2500,33333,10000]:
            for segfactor in [1,4,10,33]:
                data = np.zeros(n)
                psd = mass.mathstat.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
                f,psd = mass.mathstat.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)
                psd = mass.mathstat.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=None, window=None)
                f,psd = mass.mathstat.power_spectrum.computeSpectrum(data, segfactor=segfactor, dt=1e-6, window=None)


if __name__ == "__main__":
    unittest.main()
