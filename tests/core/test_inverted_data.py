import numpy as np
import tempfile

from mass.core import InvertedData


def test_inverted_data():
    desired_data = np.arange(16, dtype=np.uint16)
    inverted_data = ~desired_data
    assert np.all(~desired_data == inverted_data)

    with tempfile.NamedTemporaryFile() as dfile:
        inverted_data.astype(np.uint16).tofile(dfile.name)
        inverted_memmap = np.memmap(dfile.name, dtype=np.uint16, mode="readonly")
        corrected = InvertedData(inverted_memmap)

        assert np.all(desired_data == ~inverted_memmap)
        assert np.all(inverted_data == inverted_memmap)

        assert np.all(inverted_data == ~corrected)
        assert np.all(~desired_data == ~corrected)
        assert np.all(corrected == ~inverted_data)
        assert np.all(corrected == desired_data)
        # The universal index of [:] in what follows is not ideal, but it's
        # a known, current limitation of the `InvertedData` object.
        assert np.all(~inverted_data == corrected[:])
        assert np.all(desired_data == corrected[:])
        assert np.all(desired_data + ~corrected == 0xffff)
        assert np.all(~desired_data + corrected[:] == 0xffff)
        assert np.all(corrected + ~desired_data == 0xffff)
        assert desired_data[2] == corrected[2]
        assert inverted_data[2] == ~corrected[2]
        assert np.all(inverted_data[2:10] == ~corrected[2:10])
        assert np.all(desired_data[2:10] == corrected[2:10])
        assert np.all(desired_data[2:10] + ~corrected[2:10] == 0xffff)
        assert np.all(~desired_data[2:10] + corrected[2:10] == 0xffff)
