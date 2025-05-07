import numpy as np
import os
from os import path
import glob
import pytest

import mass

ljhdir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))+"/regression_test"

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
    
    def test_external_trigger(self):
        self.data.calc_external_trigger_timing()
