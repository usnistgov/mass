import numpy as np


class InvertedData1(np.memmap):
    """Wrap a numpy.memmap so that it always reads as the bitwise inverse of the raw data."""

    def __init__(self, filename, dtype=np.uint16, mode="readonly", offset=0, shape=None, order="C"):
        np.memmap.__init__(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)

    def __invert__(self):
        return np.memmap.__invert__(self)

    def __getitem__(self, key):
        return ~np.memmap.__getitem__(self, key)

    def __eq__(self, other):
        return np.memmap.__eq__(~self, other)

    def __ne__(self, other):
        return np.memmap.__ne__(~self, other)

    def __hash__(self):
        return np.memmap.__hash__(self)


class InvertedData2:
    def __init__(self, mmap):
        self._mm = mmap

    @indexedproperty
    def data(self, key=None):
        if key is None:
            return ~self._mm
        return ~(self._mm[key])

    @property
    def alldata(self):
        return ~self._mm
