"""
test_entropy.py

Test that Laplace KDE entropy code works

20 March 2017
Joe Fowler
"""

import unittest
import numpy as np
import mass
from mass.mathstat.entropy import laplace_entropy, _merge_orderedlists,  \
    laplace_cross_entropy, laplace_KL_divergence

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

    def test_specific_values_crossH(self):
        """Cross-entropy on some specific values tested in Julia already."""
        e = laplace_cross_entropy(np.linspace(1,3,30), [1,2,3.], .2)
        self.assertAlmostEqual(e, 1.2816928551002664)
        e = laplace_cross_entropy(np.linspace(1,3,30), [1,2,3.], 1)
        self.assertAlmostEqual(e, 1.8227097220918833)
        e = laplace_cross_entropy(np.linspace(1,3,30), [1,2,3.], 5)
        self.assertAlmostEqual(e, 3.3092540270152693)
        e = laplace_cross_entropy(np.linspace(1,3,30), np.linspace(1,3,30), .2)
        self.assertAlmostEqual(e, 0.9729324634882015)
        e = laplace_cross_entropy(np.linspace(1,3,30), np.linspace(1,3,30), 1)
        self.assertAlmostEqual(e, 1.8125913537170846)
        e = laplace_cross_entropy(np.linspace(1,3,30), np.linspace(1,3,30), 5)
        self.assertAlmostEqual(e, 3.309148952389859)

    def test_exact_approx_entropy(self):
        """Test the exact vs approximated modes of laplace_entropy."""
        x = np.linspace(-1,1,1001)
        z = np.hstack([x-.001, x-.0005, x, x+.0002, x+.0008])
        # Because these are size 5005 vectors, they should default to "exact" mode.
        e = laplace_entropy(z, 1, "exact")
        self.assertAlmostEqual(e, 1.8064846705587594)
        e = laplace_entropy(z, 1)
        self.assertAlmostEqual(e, 1.8064846705587594)
        e = laplace_entropy(z, 1, "approx")
        self.assertAlmostEqual(e, 1.7862710795706667)

        e = laplace_entropy(z, .1, "exact")
        self.assertAlmostEqual(e, 0.8215581872670996)
        e = laplace_entropy(z, .1)
        self.assertAlmostEqual(e, 0.8215581872670996)
        e = laplace_entropy(z, .1, "approx")
        self.assertAlmostEqual(e, 0.8189686027988935)

    def test_exact_approx_cross_entropy(self):
        """Test the exact vs approximated modes of laplace_cross_entropy."""
        x = np.linspace(-1,1,1001)
        z = np.hstack([x-.001, x-.0005, x-.0001, x+.0002, x+.0008])
        # Because these are size 5005 vectors, they should default to "exact" mode.
        e = laplace_cross_entropy(z, x, 1, "exact")
        self.assertAlmostEqual(e, 1.8064846732634803)
        e = laplace_cross_entropy(x, z, 1, "exact")
        self.assertAlmostEqual(e, 1.8064845715698967)
        e = laplace_cross_entropy(z, x, 1)
        self.assertAlmostEqual(e, 1.8064846732634803)
        e = laplace_cross_entropy(x, z, 1)
        self.assertAlmostEqual(e, 1.8064845715698967)
        e = laplace_cross_entropy(z, x, 1, "approx")
        self.assertAlmostEqual(e, 1.786246409970764)
        e = laplace_cross_entropy(x, z, 1, "approx")
        self.assertAlmostEqual(e, 1.7861438931128626)
        #
        e = laplace_cross_entropy(z, x, .1, "exact")
        self.assertAlmostEqual(e, 0.821558228945916)
        e = laplace_cross_entropy(x, z, .1, "exact")
        self.assertAlmostEqual(e, 0.821556935306746)
        e = laplace_cross_entropy(z, x, .1)
        self.assertAlmostEqual(e, 0.821558228945916)
        e = laplace_cross_entropy(x, z, .1)
        self.assertAlmostEqual(e, 0.821556935306746)
        e = laplace_cross_entropy(z, x, .1, "approx")
        self.assertAlmostEqual(e, 0.8189687816571245)
        e = laplace_cross_entropy(x, z, .1, "approx")
        self.assertAlmostEqual(e, 0.8189547106424491)

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

    def test_types(self):
        for t2 in (np.int, np.float, np.float32):
            e = laplace_entropy(np.array([1,2,3], dtype=t2), 1.0)
            self.assertAlmostEqual(e, 1.8865648057292637)

            for t1 in (np.float, np.float32):
                e = laplace_cross_entropy(np.linspace(1,3,30, dtype=t1), np.array([1,2,3], dtype=t2), .2)
                self.assertAlmostEqual(e, 1.2816928551002664)
                e = laplace_KL_divergence(np.linspace(1,3,30, dtype=t1), np.array([1,2,3], dtype=t2), .2)
                self.assertAlmostEqual(e, 0.30876039161206514)


if __name__ == "__main__":
    unittest.main()