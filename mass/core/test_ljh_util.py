import os
import os.path
import unittest as ut

from mass.core.ljh_util import ljh_channum, filename_glob_expand, \
    remove_unpaired_channel_files, ljh_sort_filenames_numerically, \
    ljh_chan_names, ljh_basename_channum


class TestFilenameHandling(ut.TestCase):
    """Test various functions that handle LJH filenames."""

    def test_glob(self):
        self.assertIn(os.path.join("mass", "regression_test", "regress_chan1.ljh"),
                      filename_glob_expand(os.path.join("mass", "regression_test", "regress_chan*.ljh")))
        self.assertIn(os.path.join("mass", "regression_test", "regress_chan1.noi"),
                      filename_glob_expand(os.path.join("mass", "regression_test", "regress_chan*.noi")))

    def test_extract_channum(self):
        self.assertEqual(1, ljh_channum("dummy_chan1.ljh"))
        self.assertEqual(101, ljh_channum("dummy_chan101.ljh"))
        self.assertEqual(101, ljh_channum("path/to/file/dummy_chan101.ljh"))
        self.assertEqual(101, ljh_channum("path/to/file/dummy_chan101.other_suffix"))

    def test_remove_unmatched_channums(self):
        fnames1 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 9, 15)]
        validns = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7)]
        remove_unpaired_channel_files(fnames1, fnames2)
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_remove_unmatched_channums_with_neveruse(self):
        fnames1 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 9, 15)]
        validns = ["dummy_chan%d.ljh" % d for d in (1, 3)]
        remove_unpaired_channel_files(fnames1, fnames2, never_use=(5, 7))
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_remove_unmatched_channums_with_useonly(self):
        fnames1 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = ["dummy_chan%d.ljh" % d for d in (1, 3, 5, 7, 9, 15)]
        validns = ["dummy_chan%d.ljh" % d for d in (1, 3)]
        remove_unpaired_channel_files(fnames1, fnames2, use_only=(1, 3))
        self.assertEqual(len(validns), len(fnames1))
        self.assertEqual(len(validns), len(fnames2))
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            self.assertEqual(v, f1)
            self.assertEqual(v, f2)

    def test_sort_filenames_numerically(self):
        cnums = [1, 11, 13, 3, 5, 7, 9, 99]
        fnames = ["d_chan%d.ljh" % d for d in cnums]
        fnames.sort()
        sorted_names = ljh_sort_filenames_numerically(fnames)
        cnums.sort()
        correct_order = ["d_chan%d.ljh" % d for d in cnums]
        self.assertEqual(len(sorted_names), len(correct_order))
        for s, c in zip(sorted_names, correct_order):
            self.assertEqual(s, c)

    def test_ljh_basename(self):
        bname = "/a/b/c/d_chan1.ljh"
        bnamenoi = "/a/b/c/d_chan1.noi"
        out = ljh_chan_names(bname, [3])
        outnoi = ljh_chan_names(bnamenoi, [3])
        self.assertTrue("/a/b/c/d_chan3.ljh" in out)
        self.assertTrue("/a/b/c/d_chan3.noi" in outnoi)

    def test_ljh_basename_channum(self):
        basename = "/a/b/c/d"
        bname = basename+"_chan%d.ljh"
        for cnum in [1, 3, 5, 100, 200, 94932]:
            b, c = ljh_basename_channum(bname % cnum)
            self.assertEqual(c, cnum)
            self.assertEqual(b, basename)

    def test_ljh_channum(self):
        bname = "/a/b/c/d_chan%d.ljh"
        for cnum in [1, 3, 5, 100, 200, 94932]:
            self.assertEqual(ljh_channum(bname % cnum), cnum)

    def test_ljh_sort(self):
        """Make sure we can sort LJH filenames by channel number."""
        bname = "/a/b/c/d_chan%d.ljh"
        channels = (9, 4, 1, 3, 5, 100, 200, 94932)
        schannels = sorted(channels)
        snames = [bname % c for c in schannels]
        rnames = ljh_sort_filenames_numerically([bname % c for c in channels])
        for x, y in zip(rnames, snames):
            self.assertEqual(x, y)


if __name__ == '__main__':
    ut.main()
