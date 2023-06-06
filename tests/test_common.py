import unittest as ut
import numpy as np

from mass.common import isstr, tostr


class TestCommon(ut.TestCase):
    def test_isstr(self):
        truestrings = ("text", b"bytes", "unicode", r"rawstrin\g")
        notstrings = ((), [], 3, 3.4, np.array([3, 5]), None, True, False, {3: 4})

        for s in truestrings:
            self.assertTrue(isstr(s), msg="isstr({}) returns False, want True".format(s))
        for n in notstrings:
            self.assertFalse(isstr(n), msg="isstr({}) returns True, want False".format(n))

    def test_tostr(self):
        inputs = ("text", b"text", "text", r"text")
        for inp in inputs:
            t = tostr(inp)
            self.assertTrue(t == "text", msg="tostr({}) returns {}, want 'text'".format(inp, t))


if __name__ == '__main__':
    ut.main()
