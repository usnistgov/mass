import numpy as np
import pylab as pl
import glob, os
import mass
import unittest as ut
import mass.core.channel_group as mcg

class TestFilenameHandling(ut.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     pass

    def test_glob(self):
        self.assertIn("src/mass/regression_test/regress_chan1.ljh", mcg._glob_expand("src/mass/regression_test/regress_chan*.ljh"))
        self.assertIn("src/mass/regression_test/regress_chan1.noi", mcg._glob_expand("src/mass/regression_test/regress_chan*.noi"))

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

if __name__ == '__main__':
    ut.main()
