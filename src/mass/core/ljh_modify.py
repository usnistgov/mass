import numpy as np
import os
import mass.core.files
from mass.core.utilities import InlineUpdater
from distutils.version import StrictVersion

def LJHModify(input_filename, output_filename, callback, overwrite=False):
    """Copy an LJH file `input_filename` to a new LJH file `output_filename`
    with the identical header, but with the raw data records transformed by
    the function (or other callable object) `callback`.

    The function `callback` should be of the form `callback(pulsearray)`, where
    `pulsearray` is an array of raw data records of shape (Nrecords, Nsamples).
    The callback might take the following form, if you need it to loop over records:

    def mycallback(pulsearray):
        for record in pulsearray:
             record[:] = 1000 + (record/2)   # or whatever operations you need.

    In the above example, the index `[:]` is required. It instructs the array `record`
    to change the values it contains *in place*. If you omit the `[:]`, then you'd be
    asking the name `record` to be re-used for some other purpose, and thus `pulsearray`
    would not be changed.

    NOT IMPLEMENTED: this version of LJHModify does *not* allow the caller to modify the
    per-pulse row counter or posix time. Please file an issue if this becomes a problem.
    """

    # Check for file problems, then open the input and output LJH files.
    if os.path.exists(output_filename):
        if os.path.samefile(input_filename, output_filename):
            raise ValueError("Input '%s' and output '%s' are the same file, which is not allowed." %
                (input_filename, output_filename))
        if overwrite:
            print("WARNING: overwriting output file '%s'"%output_filename)
        else:
            raise ValueError("Output file '%s' exists. Call with overwrite=True to proceed anyway."
                %output_filename)

    infile = mass.core.files.LJHFile(input_filename)
    outfile = open(output_filename, "wb")

    # Copy the header as a single string.
    outfile.write("".join(infile.header_lines))
    updater = InlineUpdater("LJHModify")

    # Loop over data in segments
    for (first, last, segnum, segdata) in infile.iter_segments():
        # For now, we are not modifying the times and row #s
        # If we wanted to, that would require a fancier callback, I guess.
        callback(segdata)

        # Write the modified segdata (and the unmodified row count and timestamps).
        if StrictVersion(infile.version_str.decode()) >= StrictVersion("2.2.0"):
            x = np.zeros((last-first,), dtype=infile.post22_data_dtype)
            x["rowcount"] = infile.rowcount
            x["posix_usec"] = infile.datatimes_float*1e6
            x["data"] = segdata
            x.tofile(outfile)
        else:
            x = np.zeros((last-first, 3+infile.nSamples), dtype=np.uint16)
            x[:, 3:] = segdata
            x.tofile(outfile)
        updater.update(float(segnum+1)/infile.n_segments)


    outfile.close()

# A callback that does nothing
def dummy_callback(segdata): pass

# Here's how you supply a simple callback without any free parameters.
# This function will invert every data value. For an unsigned int, it might
# not be clear what "invert" means. I mean that we replace every 0 with 0xffff,
# ever 1 with 0xfffe, and so on.

def callback_invert(segdata):
    assert segdata.dtype == np.uint16
    segdata = 0xffff-segdata


# Here's how to supply a callback with a free parameter (some kind of "state").
# This creates a "function object", which is callable but also stores internally
# the number that you wanted to add to every raw data value.

class callback_shift(object):
    def __init__(self, shiftby):
        self.shift=shiftby
    def __call__(self, segdata):
        segdata += self.shift