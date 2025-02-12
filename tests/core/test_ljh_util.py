import os
import os.path
import shutil
import subprocess
import numpy as np

from mass.core.ljh_util import ljh_channum, filename_glob_expand, \
    remove_unpaired_channel_files, ljh_sort_filenames_numerically, \
    ljh_chan_names, ljh_basename_channum
from mass.core.ljh_modify import LJHFile


class TestFilenameHandling:
    """Test various functions that handle LJH filenames."""

    @staticmethod
    def test_glob():
        assert os.path.join("tests", "regression_test", "regress_chan1.ljh") in \
            filename_glob_expand(os.path.join("tests", "regression_test", "regress_chan*.ljh"))
        assert os.path.join("tests", "regression_test", "regress_noise_chan1.ljh") in \
            filename_glob_expand(os.path.join("tests", "regression_test", "regress_noise_chan*.ljh"))

    @staticmethod
    def test_extract_channum():
        assert 1 == ljh_channum("dummy_chan1.ljh")
        assert 101 == ljh_channum("dummy_chan101.ljh")
        assert 101 == ljh_channum("path/to/file/dummy_chan101.ljh")
        assert 101 == ljh_channum("path/to/file/dummy_chan101.other_suffix")

    @staticmethod
    def test_remove_unmatched_channums():
        fnames1 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 9, 15)]
        validns = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7)]
        remove_unpaired_channel_files(fnames1, fnames2)
        assert len(validns) == len(fnames1)
        assert len(validns) == len(fnames2)
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            assert v == f1
            assert v == f2

    @staticmethod
    def test_remove_unmatched_channums_with_neveruse():
        fnames1 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 9, 15)]
        validns = [f"dummy_chan{d}.ljh" for d in (1, 3)]
        remove_unpaired_channel_files(fnames1, fnames2, never_use=(5, 7))
        assert len(validns) == len(fnames1)
        assert len(validns) == len(fnames2)
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            assert v == f1
            assert v == f2

    @staticmethod
    def test_remove_unmatched_channums_with_useonly():
        fnames1 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 9, 15)]
        validns = [f"dummy_chan{d}.ljh" for d in (1, 3)]
        remove_unpaired_channel_files(fnames1, fnames2, use_only=(1, 3))
        assert len(validns) == len(fnames1)
        assert len(validns) == len(fnames2)
        for v, f1, f2 in zip(validns, fnames1, fnames2):
            assert v == f1
            assert v == f2

    @staticmethod
    def test_remove_unmatched_channums_with_neveruse_nosecondlist():
        "remove_unpaired_channel_files needs to work if 2nd list is empty"
        fnames1 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = []
        validns = [f"dummy_chan{d}.ljh" for d in (1, 3, 11, 13)]
        remove_unpaired_channel_files(fnames1, fnames2, never_use=(5, 7))
        assert len(validns) == len(fnames1)
        for v, f1 in zip(validns, fnames1):
            assert v == f1

    @staticmethod
    def test_remove_unmatched_channums_with_useonly_nosecondlist():
        "remove_unpaired_channel_files needs to work if 2nd list is empty"
        fnames1 = [f"dummy_chan{d}.ljh" for d in (1, 3, 5, 7, 11, 13)]
        fnames2 = []
        validns = [f"dummy_chan{d}.ljh" for d in (1, 3, 7)]
        remove_unpaired_channel_files(fnames1, fnames2, use_only=(1, 3, 7))
        assert len(validns) == len(fnames1)
        for v, f1 in zip(validns, fnames1):
            assert v == f1

    @staticmethod
    def test_sort_filenames_numerically():
        cnums = [1, 11, 13, 3, 5, 7, 9, 99]
        fnames = [f"d_chan{d}.ljh" for d in cnums]
        fnames.sort()
        sorted_names = ljh_sort_filenames_numerically(fnames)
        cnums.sort()
        correct_order = [f"d_chan{d}.ljh" for d in cnums]
        assert len(sorted_names) == len(correct_order)
        for s, c in zip(sorted_names, correct_order):
            assert s == c

    @staticmethod
    def test_ljh_basename():
        bname = "/a/b/c/d_chan1.ljh"
        bnamenoi = "/a/b/c/d_noise_chan1.ljh"
        out = ljh_chan_names(bname, [3])
        outnoi = ljh_chan_names(bnamenoi, [3])
        assert "/a/b/c/d_chan3.ljh" in out
        assert "/a/b/c/d_noise_chan3.ljh" in outnoi

    @staticmethod
    def test_ljh_basename_channum():
        basename = "/a/b/c/d"
        bname = basename + "_chan%d.ljh"
        for cnum in [1, 3, 5, 100, 200, 94932]:
            b, c = ljh_basename_channum(bname % cnum)
            assert c == cnum
            assert b == basename

    @staticmethod
    def test_ljh_channum():
        bname = "/a/b/c/d_chan%d.ljh"
        for cnum in [1, 3, 5, 100, 200, 94932]:
            assert ljh_channum(bname % cnum) == cnum

    @staticmethod
    def test_ljh_sort():
        """Make sure we can sort LJH filenames by channel number."""
        bname = "/a/b/c/d_chan%d.ljh"
        channels = (9, 4, 1, 3, 5, 100, 200, 94932)
        schannels = sorted(channels)
        snames = [bname % c for c in schannels]
        rnames = ljh_sort_filenames_numerically([bname % c for c in channels])
        for x, y in zip(rnames, snames):
            assert x == y


def test_ljh_merge(tmp_path):
    """Make sure the LJH merge script works."""
    dest1_name = tmp_path / "test1_chan3.ljh"
    dest2_name = tmp_path / "test2_chan3.ljh"
    src_name = os.path.join('tests', 'regression_test', 'regress_chan3.ljh')
    src = LJHFile.open(src_name)
    Npulses = min(20, src.nPulses)
    wordsize = 2
    timing_size = 16
    truncated_length = src.header_size + Npulses * (src.nSamples * wordsize + timing_size)

    shutil.copy(src_name, dest1_name)
    os.truncate(dest1_name, truncated_length)
    shutil.copy(dest1_name, dest2_name)

    cmd = ["bin/ljh_merge", tmp_path / "test?_chan3.ljh"]
    ps = subprocess.run(cmd, capture_output=True, check=True)
    assert ps.returncode == 0

    result_name = tmp_path / "merged_chan3.ljh"
    result = LJHFile.open(result_name)
    assert 2 * Npulses == result.nPulses
    assert src.nSamples == result.nSamples
    assert np.all(result.datatimes_raw >= src.datatimes_raw[0])
    assert np.all(result.subframecount >= src.subframecount[0])

    # Make sure we can't run another merge w/o the --force flag
    ps = subprocess.run(cmd, capture_output=True, check=False)
    assert ps.returncode != 0

    # Make sure we CAN run another merge with the --force flag
    cmdF = ["bin/ljh_merge", "--force", tmp_path / "test?_chan3.ljh"]
    ps = subprocess.run(cmdF, capture_output=True, check=True)
    assert ps.returncode == 0
