import tempfile
import os.path

import numpy as np
import scipy as sp
import os
import shutil
import unittest as ut

import mass
import mass.core.phase_correct as phase_correct

import logging
LOG = logging.getLogger("mass")


class TestPhaseCorrect(ut.TestCase):
    """Test various functions that handle LJH filenames."""

    def test_phase_correct(self):
        # make some fake data
        energies = np.arange(3000)
        ph_peaks = []
        for i,name in enumerate(["MnKAlpha","FeKAlpha","CuKAlpha"]):
            spect = mass.spectrum_classes[name]()
            energies[i*1000:(i+1)*1000]=spect.rvs(size=1000)
            ph_peaks.append(spect.nominal_peak_energy)
        np.random.shuffle(energies)
        phase = np.linspace(-0.6,0.6,len(energies))
        ph = energies+phase*10
        phaseCorrector = phase_correct.phase_correct(phase, ph, use=None, ph_peaks = ph_peaks)
        corrected = phaseCorrector(phase, ph)
        print(corrected-ph)
        for i in xrange(1000):
            self.assertAlmostEqual(corrected[0],corrected[i])
        print(corrected[0])

if __name__ == '__main__':
    ut.main()
