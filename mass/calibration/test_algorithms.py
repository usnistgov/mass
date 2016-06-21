import unittest
import numpy as np
import pylab as plt
import mass
from mass.calibration.algorithms import *
import itertools

class TestAlgorithms(unittest.TestCase):

    def test_find_opt_assignment(self):
        known_energies = np.array([3100,3200,3300,3600,4000,4500,5200,5800,6500,8300,9200,10200])
        ph = known_energies**0.95
        combos = itertools.combinations(range(len(ph)),8)
        tries = 0
        passes = 0
        for combo in combos:
            tries +=1
            inds = np.array(combo)
            energies_out, opt_assignments = find_opt_assignment(ph, known_energies[inds])
            if all(energies_out==known_energies[inds]) and all(opt_assignments==ph[inds]):
                passes +=1
        self.assertTrue(passes>tries*0.9)

    def test_find_local_maxima(self):
        np.random.seed(100)
        ph = np.random.randn(10000)+7000
        ph = np.hstack((ph, np.random.randn(5000)+4000))
        ph = np.hstack((ph, np.random.randn(1000)+1000))
        local_maxima = find_local_maxima(ph,10)
        rounded = np.round(local_maxima)
        self.assertTrue(all(rounded[:3]==np.array([7000, 4000, 1000])))
if __name__ == "__main__":
    unittest.main()
