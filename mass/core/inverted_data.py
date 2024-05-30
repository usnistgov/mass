import numpy as np


class InvertedData:
    """Wrap a numpy.memmap so that it always reads as the bitwise inverse of the raw data.

    Use this object when a memmap needs to be bit-wise inverted but only when read.
    That is, we don't want to read in and store the entire underlying data file.

    WARNING: this object will not work on the right-hand side of a numpy operation like
    `==` or `+` without either using `~obj` (which calls the `__invert__` method) or
    by indexing it like `obj[:]` or `obj[0::10]` (which calls `__getitem__`).

    Thanks to methods `__eq__`, `__ne__` and `__add__`, it will work on the left side.

    Sorry.
    """

    def __init__(self, mmap):
        self._mm = mmap

    def __getitem__(self, key):
        return ~np.memmap.__getitem__(self._mm, key)

    def __invert__(self):
        return self._mm

    def __eq__(self, other):
        return ~self._mm == other

    def __ne__(self, other):
        return ~self._mm != other

    def __add__(self, other):
        return ~self._mm + other

    def __getattr__(self, name):
        if name.startswith("__array") or name in {}:
            return getattr(self._mm, name)
        print("Getting ", name)
        return getattr(self, "__dict__", name)

    def __hash__(self):
        return hash(self._mm)
