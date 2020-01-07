import unittest as ut
import numpy as np

from common import isstr


class TestCommon(ut.TestCase):
    def test_isstr(self):
        truestrings = ("text", b"bytes", u"unicode", r"rawstrin\g")
        notstrings = ((), [], 3, 3.4, np.array([3, 5]), None, True, False, {3: 4})

        for s in truestrings:
            self.assertTrue(isstr(s), msg="isstr({}) returns False, want True".format(s))
        for n in notstrings:
            self.assertFalse(isstr(n), msg="isstr({}) returns True, want False".format(n))


if __name__ == '__main__':
    ut.main()
