import numpy as np


class InvertedData:
    """Wrap a numpy.memmap so that it always reads as the bitwise inverse of the raw data."""
    def __init__(self, memmap):
        self._mm = memmap

    # def __index__
