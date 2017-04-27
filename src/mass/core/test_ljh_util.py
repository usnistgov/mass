import tempfile
import os.path

import numpy as np
import os
import shutil
import unittest as ut

import mass
import ljh_util

class TestFilenameHandling(ut.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     pass

    def test_ljh_basename(self):
        bname = "/a/b/c/d_chan1.ljh"
        bnamenoi = "/a/b/c/d_chan1.noi"
        out = ljh_util.ljh_chan_names(bname,[3])
        outnoi = ljh_util.ljh_chan_names(bnamenoi,[3])
        self.assertTrue("/a/b/c/d_chan3.ljh" in out)
        self.assertTrue("/a/b/c/d_chan3.noi" in outnoi)

if __name__ == '__main__':
    ut.main()
