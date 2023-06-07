import unittest as ut
import numpy as np

from mass.common import isstr, tostr


class TestCommon(ut.TestCase):
    def test_isstr(self):
        truestrings = ("text", b"bytes", "unicode", r"rawstrin\g")
        notstrings = ((), [], 3, 3.4, np.array([3, 5]), None, True, False, {3: 4})

        for s in truestrings:
            self.assertTrue(isstr(s), msg=f"isstr({s}) returns False, want True")
        for n in notstrings:
            self.assertFalse(isstr(n), msg=f"isstr({n}) returns True, want False")

    def test_tostr(self):
        inputs = ("text", b"text", "text", r"text")
        for inp in inputs:
            t = tostr(inp)
            self.assertTrue(t == "text", msg=f"tostr({inp}) returns {t}, want 'text'")


if __name__ == '__main__':
    ut.main()
