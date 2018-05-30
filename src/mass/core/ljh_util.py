"""
Various functions for manipulating LJH files' filenames (extracting the channel
number from the name, sorting names by channel number, and so on).
"""

import glob
from os import path
import os
import re

import six

import numpy as np

__all__ = ["ljh_basename_channum", "ljh_chan_names", "ljh_get_channels",
           "output_basename_from_ljh_fname",
           "ljh_channum", "filename_glob_expand", "remove_unpaired_channel_files",
           "ljh_sort_filenames_numerically", "ljh_get_channels_both"]


def ljh_get_channels(fname):
    basename, chan = ljh_basename_channum(fname)
    dir, ljhname = path.split(basename)
    chans = []
    for f in os.listdir(dir):
        if not os.path.isfile(os.path.join(dir, f)):
            continue
        bname, chan = ljh_basename_channum(f)
        if bname == ljhname and isinstance(chan, int):
            chans.append(chan)
    return sorted(chans)


def ljh_get_channels_both(fname, nfname):
    return sorted(set(ljh_get_channels(fname)).intersection(ljh_get_channels(nfname)))


def ljh_basename_channum(fname):
    """Returns the base LJH file name and the channel number parsed from the name.

    Finds the channel number by the pattern in the file's base name, such as
    'blahblah_chan15.suffix'

    Returns:
        (basename, channum) where basename is the full file path, up to the last
        occurance of '_chan', and channum is an int (or None, if not found).
    """
    if path.isdir(fname):
        # assume it is a directory containing ljh files of the same name as the directory
        while fname[-1] == '/':
            fname = fname[:-1]
        base_dir, ljh_dir = path.split(fname)
        fname = path.join(base_dir, ljh_dir, ljh_dir)
    chanmatches = re.finditer(r"_chan\d+", fname)
    last_chan_match = None
    for last_chan_match in chanmatches:
        pass
    if last_chan_match is None:
        basename, ext = path.splitext(fname)
        chan = None
    else:
        basename = fname[:last_chan_match.start()]
        chan = int(last_chan_match.group()[5:])
    return basename, chan


def ljh_channum(name):
    """Return the channel number found in the filename, as an int."""
    return ljh_basename_channum(name)[1]


def ljh_chan_names(fname, chans):
    basename, chan = ljh_basename_channum(fname)
    ext = path.splitext(fname)[1]
    return [basename+"_chan%d%s" % (chan, ext) for chan in chans]


def ljh_get_extern_trig_fname(fname):
    basename, chan = ljh_basename_channum(fname)
    return basename+"_extern_trig.hdf5"


def output_basename_from_ljh_fname(ljh):
    basename, chan = ljh_basename_channum(ljh)
    dir, fname = path.split(basename)
    if not path.isdir(dir):
        raise("ValueError %s is not valid directory" % dir)
    outputdir = path.join(dir, "mass_output")
    if not path.isdir(outputdir):
        os.mkdir(outputdir)
    output_basefname = path.join(outputdir, fname)
    return output_basefname


def mass_folder_from_ljh_fname(ljh, filename=""):
    basename, chan = ljh_basename_channum(ljh)
    dir, fname = path.split(basename)
    if not path.isdir(dir):
        raise ValueError("%s is not valid directory" % dir)
    outputdir = path.join(dir, "mass")
    if not path.isdir(outputdir):
        os.mkdir(outputdir)
    return path.join(outputdir, filename)


def remove_unpaired_channel_files(filenames1, filenames2, never_use=None, use_only=None):
    """Extract the channel number in the filenames appearing in both lists.

    Remove from each list any file whose channel number doesn't appear on both lists.
    Also remove any file whose channel number is in the `never_use` list.

    If either `filenames1` or `filenames2` is empty, do nothing.

    Args:
        filenames1: a list of filenames containing channel #s in the form "blah_chan15".
            Will be modified.
        filenames2: a list of filenames containing channel #s in the form "blah_chan15".
            Will be modified.
        never_use: a sequence of channel numbers to exclude even if found in both lists (default None)
        use_only: if a sequence of channel numbers, exclude any channels not in it (default None)
    """
    # If one list is empty, then matching is not required or expected.
    if filenames1 is None or len(filenames1) == 0 \
            or filenames2 is None or len(filenames2) == 0:
        return
    assert isinstance(filenames1, list)
    assert isinstance(filenames2, list)

    # Now make a mapping of channel numbers to names.
    names1 = {ljh_channum(f): f for f in filenames1}
    names2 = {ljh_channum(f): f for f in filenames2}
    cnum1 = set(names1.keys())
    cnum2 = set(names2.keys())

    # Find the set of valid channel numbers.
    valid_cnum = cnum1.intersection(cnum2)
    if never_use is not None:
        valid_cnum -= set(never_use)
    if use_only is not None:
        valid_cnum = valid_cnum.intersection(set(use_only))

    # Remove invalid channel numbers
    for c in (cnum1-valid_cnum):
        filenames1.remove(names1[c])
    for c in (cnum2-valid_cnum):
        filenames2.remove(names2[c])


def ljh_sort_filenames_numerically(fnames, inclusion_list=None):
    """Return a sorted sequence of filenames of the form '*_chanXXX.*',
    sorted according to the numerical value of channel number XXX.

    Args:
        fnames: A sequence of filenames of the form '*_chan*.*'
        inclusion_list: If not None, a container with channel numbers. All files
            whose channel numbers are not on this list will be omitted from the
            output (default None).

    Returns:
        A list containg the same filenames, sorted
        according to the numerical value of channel number.
    """
    if fnames is None or len(fnames) == 0:
        return None

    if inclusion_list is not None:
        fnames = filter(lambda n: ljh_channum(n) in inclusion_list, fnames)

    return sorted(fnames, key=ljh_channum)


def filename_glob_expand(pattern):
    """Return the result of glob-expansion on the input pattern.

    Args:
        pattern: If a string, treat it as a glob pattern and return the glob-result
            as a list. If it isn't a string, return it unchanged (presumably then
            it's already a sequence).

    Returns:
        A list of 0 or more filenames. The result is sorted by
        ljh_sort_filenames_numerically().
    """
    if not isinstance(pattern, six.string_types):
        return pattern

    result = glob.glob(pattern)
    return ljh_sort_filenames_numerically(result)


# Below here are old code for handling 2 types of files that are not used in new
# data: the timing-aux file (for translating Posix time and row counts into each
# other for LJH 2.1 and earlier files); and the microphone file, in which we were
# briefly recording external triggers, before the modern extrnal-trigger system
# was created in late 2014.

def ljh_get_aux_fname(fname):
    basename, chan = ljh_basename_channum(fname)
    return basename+".timing_aux"


def ljh_get_mic_fname(fname):
    return path.join(path.dirname(fname), "microphone_timestamps")


def load_aux_file(fname):
    fname = ljh_get_aux_fname(fname)
    raw = np.fromfile(fname, dtype=np.uint64)
    raw.shape = (len(raw)/2, 2)
    crate_epoch_usec = raw[:, 1]
    crate_frame = raw[:, 0]
    return crate_epoch_usec, crate_frame


def load_mic_file(fname):
    fname = ljh_get_mic_fname(fname)
    return np.array(np.loadtxt(fname)*1e6, dtype=np.int64)
