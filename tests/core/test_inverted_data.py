import numpy as np
import os
import mass

_PATH = os.path.dirname(os.path.realpath(__file__))
ljh_root = os.path.normpath(os.path.join(_PATH, "..", "ljh_files"))


def test_inverted_data(tmp_path):
    """Read the same file normally and inverted. Be sure that all means of
    accssing the raw data give bitwise inverses of each other."""
    pulse_str = os.path.join(ljh_root, "20230626", "0001", "20230626_run0001_chan*.ljh")
    # noise_str = os.path.join(ljh_root, "20230626", "0000", "20230626_run0000_chan*.ljh")

    hf1 = tmp_path / "normal.hdf5"
    hf2 = tmp_path / "inverted.hdf5"
    data1 = mass.TESGroup(pulse_str, hdf5_filename=hf1, invert_data=False)
    data2 = mass.TESGroup(pulse_str, hdf5_filename=hf2, invert_data=True)
    ds1 = data1.channel[4102]
    ds2 = data2.channel[4102]

    # These 3 tests might seem redundant, but they might not be, depending on implementation:
    assert np.all(~ds1.alldata == ds2.alldata)
    assert np.all(ds1.alldata == ~ds2.alldata)
    assert np.all(ds2.alldata == ~ds1.alldata)

    # Compare alldata to data[:]
    assert np.all(ds1.alldata == ds1.data[:])
    assert np.all(ds2.alldata == ds2.data[:])

    # Try indexing all
    assert np.all(~ds1.data[:] == ds2.data[:])
    assert np.all(ds1.data[:] == ~ds2.data[:])

    # Try slicing in each dimension and both dimensions
    assert ds1.nPulses >= 60
    assert ds1.nSamples >= 300
    assert np.all(~ds1.data[6:60] == ds2.data[6:60])
    assert np.all(ds1.data[6:60] == ~ds2.data[6:60])
    assert np.all(~ds1.data[:, 200:300] == ds2.data[:, 200:300])
    assert np.all(ds1.data[:, 200:300] == ~ds2.data[:, 200:300])
    assert np.all(~ds1.data[6:60, 200:300] == ds2.data[6:60, 200:300])
    assert np.all(ds1.data[6:60, 200:300] == ~ds2.data[6:60, 200:300])
