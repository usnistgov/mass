from os import path
import os
import re
import numpy as np

__all__ = ['ljh_basename', 'ljh_chan_names', 'ljh_get_aux_fname', 'ljh_get_mic_fname', 'load_aux_file',
           'load_mic_file', 'ljh_get_channels',
           "ljh_get_channels_both", "output_basename_from_ljh_fname"]


def ljh_basename(fname):
    if path.isdir(fname):
        # assume it is a directory containing ljh files of the same name as the directory
        while fname[-1] == '/':
            fname = fname[:-1]
        base_dir, ljh_dir = path.split(fname)
        fname = path.join(base_dir, ljh_dir, ljh_dir)
    chanmatches = re.finditer("_chan\d+", fname)
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


def ljh_chan_names(fname, chans):
    basename, chan = ljh_basename(fname)
    ext = path.splitext(fname)[1]
    return [basename+"_chan%d%s"%(chan,ext) for chan in chans]


def ljh_get_aux_fname(fname):
    basename, chan = ljh_basename(fname)
    return basename+".timing_aux"


def ljh_get_mic_fname(fname):
    return path.join(path.dirname(fname), "microphone_timestamps")


def ljh_get_extern_trig_fname(fname):
    basename, chan = ljh_basename(fname)
    return basename+"_extern_trig.hdf5"


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


def ljh_get_channels(fname):
    basename, chan = ljh_basename(fname)
    dir, ljhname = path.split(basename)
    chans = []
    for f in os.listdir(dir):
        bname, chan = ljh_basename(f)
        if bname == ljhname and isinstance(chan, int):
            chans.append(chan)
    return sorted(chans)


def ljh_get_channels_both(fname, nfname):
    return sorted(set(ljh_get_channels(fname)).intersection(ljh_get_channels(nfname)))


def output_basename_from_ljh_fname(ljh):
    basename, chan = ljh_basename(ljh)
    dir, fname = path.split(basename)
    if not path.isdir(dir):
        raise("ValueError %s is not valid directory" % dir)
    outputdir = path.join(dir, "mass_output")
    if not path.isdir(outputdir):
        os.mkdir(outputdir)
    output_basefname = path.join(outputdir, fname)
    return output_basefname


def mass_folder_from_ljh_fname(ljh, filename=""):
    basename, chan = ljh_basename(ljh)
    dir, fname = path.split(basename)
    if not path.isdir(dir):
        raise ValueError("%s is not valid directory" % dir)
    outputdir = path.join(dir, "mass")
    if not path.isdir(outputdir):
        os.mkdir(outputdir)
    return path.join(outputdir, filename)
