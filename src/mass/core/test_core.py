import tempfile
import os.path

import numpy as np
import pylab as pl
import glob, os
import mass
import unittest as ut
import mass.core.channel_group as mcg
from mass.core.files import *


class TestFilenameHandling(ut.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     pass

    def test_glob(self):
        self.assertIn(os.path.join("src", "mass", "regression_test", "regress_chan1.ljh"),
                      mcg._glob_expand(os.path.join("src", "mass", "regression_test", "regress_chan*.ljh")))
        self.assertIn(os.path.join("src", "mass", "regression_test", "regress_chan1.noi"),
                      mcg._glob_expand(os.path.join("src", "mass", "regression_test", "regress_chan*.noi")))

    def test_extract_channum(self):
        self.assertEqual(1, mcg._extract_channum("dummy_chan1.ljh"))
        self.assertEqual(101, mcg._extract_channum("dummy_chan101.ljh"))
        self.assertEqual(101, mcg._extract_channum("path/to/file/dummy_chan101.ljh"))
        self.assertEqual(101, mcg._extract_channum("path/to/file/dummy_chan101.ljh.saved/pointless_subdir"))

    def test_remove_unmatched_channums(self):
        fnames1 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,11,13)]
        fnames2 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,9,15)]
        validns = ["dummy_chan%d.ljh"%d for d in (1,3,5,7)]
        mcg._remove_unmatched_channums(fnames1, fnames2)
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v,f1,f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_remove_unmatched_channums_with_neveruse(self):
        fnames1 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,11,13)]
        fnames2 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,9,15)]
        validns = ["dummy_chan%d.ljh"%d for d in (1,3)]
        mcg._remove_unmatched_channums(fnames1, fnames2, never_use=(5,7))
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v,f1,f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_remove_unmatched_channums_with_useonly(self):
        fnames1 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,11,13)]
        fnames2 = ["dummy_chan%d.ljh"%d for d in (1,3,5,7,9,15)]
        validns = ["dummy_chan%d.ljh"%d for d in (1,3)]
        mcg._remove_unmatched_channums(fnames1, fnames2, use_only=(1,3))
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v,f1,f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_sort_filenames_numerically(self):
        cnums = [1,11,13,3,5,7,9,99]
        fnames = ["d_chan%d.ljh"%d for d in cnums]
        fnames.sort()
        sorted_names = mcg._sort_filenames_numerically(fnames)
        cnums.sort()
        correct_order = ["d_chan%d.ljh"%d for d in cnums]
        self.assertEqual(len(sorted_names), len(correct_order))
        for s,c in zip(sorted_names, correct_order):
            self.assertEqual(s, c)


class TestFiles(ut.TestCase):

    def test_ljh_copy_and_append_traces(self):
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

    def test_all_channels_bad(self):
        """Make sure it isn't an error to load a data set where all channels are marked bad"""
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        for name in ['src/mass/regression_test/regress_mass.hdf5', 'src/mass/regression_test/regress_noise_mass.hdf5']:
            if os.path.isfile(name): os.remove(name)
        data = mass.TESGroup([src_name], [noi_name])
        data.set_chan_bad(1, "testing all channels bad")

        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data
        try:
            data = mass.TESGroup([src_name], [noi_name])
        except:
            self.fail("Opening a file with all channels bad raises and Exception.")
        self.assertNotIn(1, data.good_channels)


class TestTESHDF5Only(ut.TestCase):

    def test_all_channels_bad(self):
        """Make sure it mass can open a mass generated file in HDF5 Only mode."""
        src_name = 'src/mass/regression_test/regress_chan1.ljh'
        noi_name = 'src/mass/regression_test/regress_chan1.noi'
        for name in ['src/mass/regression_test/regress_mass.hdf5', 'src/mass/regression_test/regress_noise_mass.hdf5']:
            if os.path.isfile(name): os.remove(name)
        data = mass.TESGroup([src_name], [noi_name])
        h5filename = data.hdf5_file.filename
        data.hdf5_file.close()
        data.hdf5_noisefile.close()
        del data


        data2 = mass.TESGroupHDF5(h5filename)

    def test_ordering_hdf5only(self):
        src_name = "src/mass/regression_test/regress_chan1.ljh"
        dir = tempfile.mkdtemp()
        dest_name = "%s/temporary_chan%d.ljh"
        cnums = (1,3,5,11,13,15)
        for c in cnums:
            os.link(src_name, dest_name%(dir,c))

        data1 = mass.TESGroup("%s/temporary_chan*.ljh"%dir)
        # Make sure the usual TESGroup is in the right order
        for i,ds in enumerate(data1):
            self.assertEqual(ds.channum, cnums[i])
        fname = data1.hdf5_file.filename
        del data1

        # Make sure the usual TESGroup is in the right order
        data = mass.TESGroupHDF5(fname)
        for i,ds in enumerate(data):
            self.assertEqual(ds.channum, cnums[i])



if __name__ == '__main__':
    ut.main()
