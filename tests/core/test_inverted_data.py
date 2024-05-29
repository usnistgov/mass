import numpy as np
import tempfile

# from mass.core import InvertedData


def test_inverted_data():
    data_array = np.arange(16, dtype=np.uint16)
    with tempfile.NamedTemporaryFile() as dfile:
        data_array.astype(np.uint16).tofile(dfile.name)
        data_invert = ~data_array
        data_memmap = np.memmap(dfile.name, dtype=np.uint16, mode="readonly")
        # data_ID = InvertedData(data_memmap)

        assert np.all(~data_array == data_invert)
        assert np.all(data_array == data_memmap)

        # assert np.all(data_invert == data_ID)
        # assert np.all(data_array == ~data_ID)
        # assert np.all(data_array + ~data_ID == 0xffff)
        # assert np.all(~data_array + data_ID == 0xffff)
        # assert np.all(data_invert[2:10] == data_ID[2:10])
        # assert np.all(data_array[2:10] == ~data_ID[2:10])
        # assert np.all(data_array[2:10] + ~data_ID[2:10] == 0xffff)
        # assert np.all(~data_array[2:10] + data_ID[2:10] == 0xffff)
