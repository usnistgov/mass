import os
import numpy as np
from os import path
import glob
import pytest

import mass

ljhdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/regression_test"


class TestExternalTrigger:

    @classmethod
    def setup_class(cls):
        prefix = "regress"
        pulse_files = path.join(ljhdir, f"{prefix}_chan*.ljh")
        noise_files = path.join(ljhdir, f"{prefix}_noise_chan*.ljh")

        # Start from clean slate by removing any hdf5 files
        for fl in glob.glob(path.join(ljhdir, f"{prefix}_mass.hdf5")):
            os.remove(fl)
        for fl in glob.glob(path.join(ljhdir, f"{prefix}_noise_mass.hdf5")):
            os.remove(fl)

        data = mass.TESGroup(pulse_files, noise_files)
        data.summarize_data(forceNew=True)
        cls.data = data

    @classmethod
    def teardown_class(cls):
        cls.data.hdf5_file.close()
        cls.data.hdf5_noisefile.close()

    def test_external_trigger(self):
        ds = self.data.channel[1]
        assert np.all(ds.p_subframecount[:] > 0)
        with pytest.raises(ValueError):
            _ = ds.subframes_after_last_external_trigger[:]
        self.data.calc_external_trigger_timing()
        assert np.all(ds.subframes_after_last_external_trigger[:] > 0)
