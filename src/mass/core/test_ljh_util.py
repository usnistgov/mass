import tempfile
import os.path

import numpy as np
import os
import shutil
import unittest as ut

import mass
import ljh_util

class TestFilenameHandling(ut.TestCase):
    """Test the LJH filename functions."""

    def test_ljh_basename(self):
        bname = "/a/b/c/d_chan1.ljh"
        bnamenoi = "/a/b/c/d_chan1.noi"
        out = ljh_util.ljh_chan_names(bname,[3])
        outnoi = ljh_util.ljh_chan_names(bnamenoi,[3])
        self.assertTrue("/a/b/c/d_chan3.ljh" in out)
        self.assertTrue("/a/b/c/d_chan3.noi" in outnoi)

    def test_ljh_basename_channum(self):
        basename = "/a/b/c/d"
        bname = basename+"_chan%d.ljh"
        for cnum in [1,3,5,100,200,94932]:
            b, c = ljh_util.ljh_basename_channum(bname % cnum)
            self.assertEqual(c, cnum)
            self.assertEqual(b, basename)

    def test_ljh_channum(self):
        bname = "/a/b/c/d_chan%d.ljh"
        for cnum in [1,3,5,100,200,94932]:
            self.assertEqual(ljh_util.ljh_channum(bname % cnum), cnum)

    def test_ljh_sort(self):
        """Make sure we can sort LJH filenames by channel number."""
        bname = "/a/b/c/d_chan%d.ljh"
        channels = (9,4,1,3,5,100,200,94932)
        schannels = sorted(channels)
        snames = [bname % c for c in schannels]
        rnames = ljh_util.ljh_sort_filenames_numerically([bname % c for c in channels])
        for x,y in zip(rnames, snames):
            self.assertEqual(x,y)

if __name__ == '__main__':
    ut.main()
