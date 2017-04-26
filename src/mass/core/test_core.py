import tempfile
import os.path

import numpy as np
import os
import shutil
import unittest as ut

import mass
from mass.core.ljh_modify import *

class TestFiles(ut.TestCase):

    def test_ljh_copy_and_append_traces(self):
        """Test copying and appending traces to LJH files."""
        src_name = os.path.join('src', 'mass', 'regression_test', 'regress_chan1.ljh')
        dest_name = os.path.join(tempfile.gettempdir(), 'foo_chan1.ljh')
        src = LJHFile(src_name)

        source_traces = [20]
        ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        source_traces = [0, 30, 20, 10]
        ljh_copy_traces(src_name, dest_name, source_traces, overwrite=True)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        source_traces.append(5)
        ljh_append_traces(src_name, dest_name, [5])
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))

        new_traces = [15, 25, 3]
        source_traces.extend(new_traces)
        ljh_append_traces(src_name, dest_name, new_traces)
        dest = LJHFile(dest_name)
        for i, st in enumerate(source_traces):
            self.assertTrue(np.all(src.read_trace(st) == dest.read_trace(i)))


class TestTESGroup(ut.TestCase):
    """Basic tests of the TESGroup object."""

    def load_data(self, clear_hdf5=True):
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        if clear_hdf5:
            for name in ['src/mass/regression_test/regress_mass.hdf5',
                         'src/mass/regression_test/regress_noise_mass.hdf5']:
                if os.path.isfile(name):
                    os.remove(name)
        return mass.TESGroup([src_name], [noi_name])

    def test_all_channels_bad(self):
        """Make sure it isn't an error to load a data set where all channels are marked bad"""
        data = self.load_data()
        data.set_chan_bad(1, "testing all channels bad")
        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data

        try:
            data = self.load_data(clear_hdf5=False)
        except:
            self.fail("Opening a file with all channels bad raises and Exception.")
        self.assertNotIn(1, data.good_channels)
        data.set_chan_good(1)
        print("Testing printing of a TESGroup")
        print(data)

    def test_save_hdf5_calibration_storage(self):
        "calibrate a dataset, make sure it saves to hdf5"
        data = self.load_data()
        data.summarize_data()
        data.calibrate("p_pulse_rms", [10000.])
        data.calibrate("p_pulse_rms", [10000.], name_ext="abc")
        ds = data.first_good_dataset

        data2 = self.load_data(clear_hdf5=False)
        ds2 = data2.first_good_dataset
        self.assertTrue(all([k in ds.calibration.keys() for k in ds2.calibration.keys()]))
        self.assertEqual(len(ds.calibration.keys()), 2)

        # These 2 checks test issue #102.
        self.assertIsNotNone(ds2.peak_samplenumber)
        self.assertEqual(ds2.peak_samplenumber, ds.peak_samplenumber)

    def test_make_auto_cuts(self):
        """Make sure that non-trivial auto-cuts are generated."""
        data = self.load_data()
        data.summarize_data()
        data.auto_cuts(forceNew=True, clearCuts=True)
        ds = data.first_good_dataset
        self.assertLess(ds.cuts.good().sum(), ds.nPulses)

    def test_plot_filters(self):
        "Check that issue 105 is fixed: data.plot_filters() doesn't fail on 1 channel."
        data = self.load_data()
        data.set_chan_good(1)
        data.summarize_data()
        data.channel[1]._use_new_filters = False  # Not enough pulses for new filters.
        data.avg_pulses_auto_masks()
        data.compute_noise_spectra()
        data.compute_filters()
        data.plot_filters()


class TestTESHDF5Only(ut.TestCase):
    """Basic tests of the TESGroup object when we use the HDF5-only variant."""

    def test_all_channels_bad(self):
        """Make sure it mass can open a mass generated file in HDF5 Only mode."""
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        for name in ['src/mass/regression_test/regress_mass.hdf5',
                     'src/mass/regression_test/regress_noise_mass.hdf5']:
            if os.path.isfile(name):
                os.remove(name)
        data = mass.TESGroup([src_name], [noi_name])
        h5filename = data.hdf5_file.filename
        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data

        data2 = mass.TESGroupHDF5(h5filename)
        print("Testing printing of a TESGroupHDF5")
        print(data2)

    def test_ordering_hdf5only(self):
        src_name = "src/mass/regression_test/regress_chan1.ljh"
        dir = tempfile.mkdtemp()
        dest_name = "%s/temporary_chan%d.ljh"
        chan1_dest = dest_name % (dir, 1)
        shutil.copy(src_name, chan1_dest)
        cnums = (1, 3, 5, 11, 13, 15)
        for c in cnums[1:]:
            os.link(chan1_dest, dest_name % (dir, c))

        data1 = mass.TESGroup("%s/temporary_chan*.ljh" % dir)
        # Make sure the usual TESGroup is in the right order
        for i, ds in enumerate(data1):
            self.assertEqual(ds.channum, cnums[i])
        fname = data1.hdf5_file.filename
        del data1

        # Make sure the usual TESGroup is in the right order
        data = mass.TESGroupHDF5(fname)
        for i, ds in enumerate(data):
            self.assertEqual(ds.channum, cnums[i])


if __name__ == '__main__':
    ut.main()
