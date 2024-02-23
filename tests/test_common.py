import numpy as np

from mass.common import isstr, tostr


def test_isstr():
    truestrings = ("text", b"bytes", "unicode", r"rawstrin\g")
    notstrings = ((), [], 3, 3.4, np.array([3, 5]), None, True, False, {3: 4})

    for s in truestrings:
        assert isstr(s), f"isstr({s}) returns False, want True"
    for n in notstrings:
        assert not isstr(n), f"isstr({n}) returns True, want False"


def test_tostr():
    inputs = ("text", b"text", "text", r"text")
    for inp in inputs:
        t = tostr(inp)
        assert t == "text", f"tostr({inp}) returns {t}, want 'text'"
