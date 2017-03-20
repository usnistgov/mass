"""
test_entropy.py

Test that Laplace KDE entropy code works

20 March 2017
Joe Fowler
"""

import unittest
import numpy as np
import mass
from mass.mathstat.entropy import laplace_entropy, _merge_orderedlists, laplace_KL_divergence

class Test_LaplaceEntropy(unittest.TestCase):

    def test_entropy1(self):
        """Entropy on a distribution with 1 value."""
        for w in [.1, .2, .5, 1, 2, 5]:
            expected = 1+np.log(2*w)
            for i in [1.1, 0.0, -39]:
                d = np.array([i], dtype=float)
                self.assertAlmostEqual(laplace_entropy(d, w), expected, 4)

    def test_specific_values_entropy(self):
        """Entropy on some specific values tested in Julia already."""
        e = laplace_entropy([1,2,3.0], .2)
        self.assertAlmostEqual(e, 1.0188440534191967)
        e = laplace_entropy([1,2,3.0], 1.0)
        self.assertAlmostEqual(e, 1.8865648057292637)
        e = laplace_entropy([1,2,3.0], 5.0)
        self.assertAlmostEqual(e, 3.3145184966015533)
        e = laplace_entropy([1,2,3,.1,.5,.3,.9,-5,3], 0.2)
        self.assertAlmostEqual(e, 1.4343995252696242)
        e = laplace_entropy([1,2,3,.1,.5,.3,.9,-5,3], 1.0)
        self.assertAlmostEqual(e, 2.2473133161879617)
        e = laplace_entropy([1,2,3,.1,.5,.3,.9,-5,3], 5.0)
        self.assertAlmostEqual(e, 3.3724487611280987)

    def test_merge(self):
        """Test that _merge_orderedlists(x,y) works as intended."""
        x = [1,2,4,6,7,8]
        y = [3,5]
        m,is1 = _merge_orderedlists(x, y)
        expect_m = np.sort(np.hstack([x,y]))
        expect_i = np.array([1,1,0,1,0,1,1,1], dtype=np.bool)

        for a,b in zip(m, expect_m):
            self.assertEquals(a,b)
        for a,b in zip(is1, expect_i):
            self.assertEquals(a,b)

    def test_specific_values_KL(self):
        """KL-divergence on some specific values tested in Julia already."""
        e = laplace_KL_divergence(np.linspace(1,3,30), [1,2,3.], .2)
        self.assertAlmostEqual(e, 2.8911307675343667)
        e = laplace_KL_divergence(np.linspace(1,3,30), [1,2,3.], 1)
        self.assertAlmostEqual(e, 1.8227097220918818)
        e = laplace_KL_divergence(np.linspace(1,3,30), [1,2,3.], 5)
        self.assertAlmostEqual(e, 1.6998161145811683)
        e = laplace_KL_divergence(np.linspace(1,3,30), np.linspace(1,3,30), .2)
        self.assertAlmostEqual(e, 2.582370375922301)
        e = laplace_KL_divergence(np.linspace(1,3,30), np.linspace(1,3,30), 1)
        self.assertAlmostEqual(e, 1.8125913537170852)
        e = laplace_KL_divergence(np.linspace(1,3,30), np.linspace(1,3,30), 5)
        self.assertAlmostEqual(e, 1.699711039955758)

    def test_empty(self):
        self.assertRaises(ValueError, laplace_entropy, [], 1.0)
        self.assertRaises(ValueError, laplace_KL_divergence, [], [], 1.0)
        self.assertRaises(ValueError, laplace_KL_divergence, [], [0.0], 1.0)
        self.assertRaises(ValueError, laplace_KL_divergence, [0.0], [], 1.0)

    def test_negative_widths(self):
        self.assertRaises(ValueError, laplace_entropy, [1,2], 0.0)
        self.assertRaises(ValueError, laplace_entropy, [1,2], -1.0)
        self.assertRaises(ValueError, laplace_KL_divergence, [1,2], [1,2], 0.0)
        self.assertRaises(ValueError, laplace_KL_divergence, [1,2], [1,2], -1.0)


if __name__ == "__main__":
    unittest.main()
