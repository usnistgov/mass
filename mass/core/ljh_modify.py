"""
Functions ljh_copy_traces and ljh_append_traces and class LJHModify, all of
which can modify existing LJH files.
"""

import os
import struct
import time
import numpy as np
from packaging.version import Version

from mass.core.files import LJHFile, make_ljh_header
from mass.core.utilities import InlineUpdater


def LJHModify(input_filename, output_filename, callback, overwrite=False):
    """Copy an LJH file `input_filename` to a new LJH file `output_filename`
    with the identical header, but with the raw data records transformed in-place
    by the function (or other callable object) `callback`.

    The function `callback` should be of the form `modified=callback(record)`, where
    `record` is an array of raw data records of shape (Nsamples, ).
    The callback might take the following form, if you need it to loop over records:

    def mycallback(record):
        return 1000 + (record/2)   # or whatever operations you need.

    NOT IMPLEMENTED: this version of LJHModify does *not* allow the caller to
    modify the per-pulse row counter or posix time. Please file an issue if this
    becomes a problem.
    """

    # Check for file problems, then open the input and output LJH files.
    if os.path.exists(output_filename):
        if os.path.samefile(input_filename, output_filename):
            raise ValueError(f"Input '{input_filename}' and output '{output_filename}' "
                             "are the same file, which is not allowed.")
        if overwrite:
            print(f"WARNING: overwriting output file '{output_filename}'")
        else:
            raise ValueError(f"Output file '{output_filename}' exists. "
                             "Call with overwrite=True to proceed anyway.")

    infile = LJHFile.open(input_filename)
    with open(output_filename, "wb") as outfile:
        # Copy the header as a single string.
        outfile.write("".join(infile.header_lines))

        # For now, we are not modifying the times and row #s
        # If we wanted to, that would require a fancier callback, I guess.

        # Write the modified segdata (and the unmodified row count and timestamps).
        updater = InlineUpdater("LJHModify")
        if Version(infile.version_str.decode()) >= Version("2.2.0"):
            for i in range(infile.nPulses):
                data = callback(infile.alldata[i])
                x = np.zeros((1,), dtype=infile.dtype)
                x["rowcount"] = infile.rowcount
                x["posix_usec"] = infile.datatimes_float*1e6
                x["data"] = data
                x.tofile(outfile)
                if i % 100 == 0:
                    updater.update(float(i+1)/infile.nPulses)

        else:
            for i in range(infile.nPulses):
                data = callback(infile.alldata[i])
                x = np.zeros((1, 3+infile.nSamples), dtype=np.uint16)
                x[:, 3:] = data
                x.tofile(outfile)
                if i % 100 == 0:
                    updater.update(float(i+1)/infile.nPulses)

        updater.update(1.0)


# A callback that does nothing
def dummy_callback(data):
    return data

# Here's how you supply a simple callback without any free parameters.
# This function will invert every data value. For an unsigned int, it might
# not be clear what "invert" means. I mean that we replace every 0 with 0xffff,
# ever 1 with 0xfffe, and so on.


def callback_invert(record):
    assert record.dtype == np.uint16
    return 0xffff-record


# Here's how to supply a callback with a free parameter (some kind of "state").
# This creates a "function object", which is callable but also stores internally
# the number that you wanted to add to every raw data value.

class callback_shift():
    def __init__(self, shiftby):
        self.shift = shiftby

    def __call__(self, segdata):
        return segdata + self.shift


def helper_write_pulse(dest, src, i):
    rowcount, timestamp_usec, trace = src.read_trace_with_timing(i)
    prefix = struct.pack('<Q', int(rowcount))
    dest.write(prefix)
    prefix = struct.pack('<Q', int(timestamp_usec))
    dest.write(prefix)
    trace.tofile(dest, sep="")


def ljh_copy_traces(src_name, dest_name, pulses, overwrite=False):
    """
    Copy traces from one ljh file to another. The destination file will be of
    LJH version 2.2.0.

    Can be used to grab specific traces from some other ljh file, and put them into a new file

    Args:
        src_name: the name of the source file
        dest_name: the name of the destination file
        pulses: indices of the pulses to copy
        overwrite: If the destination file exists and overwrite is not True,
            then the copy fails (default False).
    """

    if os.path.exists(dest_name) and not overwrite:
        raise OSError(f"The ljhfile '{dest_name}' exists and overwrite was not set to True")

    src = LJHFile.open(src_name)

    header_dict = src.__dict__.copy()
    header_dict['asctime'] = time.asctime(time.gmtime())
    header_dict['version_str'] = '2.2.0'
    ljh_header = make_ljh_header(header_dict)

    with open(dest_name, "wb") as dest_fp:
        dest_fp.write(ljh_header)
        for i in pulses:
            helper_write_pulse(dest_fp, src, i)


def ljh_append_traces(src_name, dest_name, pulses=None):
    """Append traces from one LJH file onto another. The destination file is
    assumed to be version 2.2.0.

    Can be used to grab specific traces from some other ljh file, and append them onto an existing ljh file.

    Args:
        src_name: the name of the source file
        dest_name: the name of the destination file
        pulses: indices of the pulses to copy (default: None, meaning copy all)
    """

    src = LJHFile.open(src_name)
    if pulses is None:
        pulses = range(src.nPulses)
    with open(dest_name, "ab") as dest_fp:
        for i in pulses:
            helper_write_pulse(dest_fp, src, i)


def ljh_truncate(input_filename, output_filename, n_pulses=None, timestamp=None, segmentsize=None):
    """Truncate an LJH file.

    Writes a new copy of an LJH file, with
    with the identical header, but with a smaller number of raw data pulses.

    Arguments:
    input_filename  -- name of file to truncate
    output_filename -- filename for truncated file
    n_pulses        -- truncate to include only this many pulses (default None)
    timestamp       -- truncate to include only pulses with timestamp earlier
                       than this number (default None)
    segmentsize     -- number of bytes per segment; this is primarily here to
                       facilitate testing (defaults to same value as in LJHFile)

    Exactly one of n_pulses and timestamp must be specified.
    """

    if (n_pulses is None and timestamp is None) or \
            (n_pulses is not None and timestamp is not None):
        msg = "Must specify exactly one of n_pulses, timestamp."
        msg = msg+f" Values were {str(n_pulses)}, {str(timestamp)}"
        raise Exception(msg)

    # Check for file problems, then open the input and output LJH files.
    if os.path.exists(output_filename):
        if os.path.samefile(input_filename, output_filename):
            msg = f"Input '{input_filename}' and output '{output_filename}' are the same file, which is not allowed"
            raise ValueError(msg)

    infile = LJHFile.open(input_filename)
    if segmentsize is not None:
        infile.set_segment_size(segmentsize)

    if Version(infile.version_str.decode()) < Version("2.2.0"):
        raise Exception(f"Don't know how to truncate this LJH version [{infile.version_str}]")

    with open(output_filename, "wb") as outfile:
        # write the header as a single string.
        for (k, v) in infile.header_dict.items():
            outfile.write(k+b": "+v+b"\r\n")
        outfile.write(b"#End of Header\r\n")

        # Write pulses. Stop reading segments from the original file as soon as possible.
        if n_pulses is None:
            n_pulses = infile.nPulses
        for i in range(n_pulses):
            if (timestamp is not None and infile.datatimes_float[i] > timestamp):
                break
            prefix = struct.pack('<Q', np.uint64(infile.rowcount[i]))
            outfile.write(prefix)
            prefix = struct.pack('<Q', np.uint64(infile.datatimes_raw[i]))
            outfile.write(prefix)
            trace = infile.alldata[i, :]
            trace.tofile(outfile, sep="")
