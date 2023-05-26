"""
The mass.files module contains classes required for handling the various types
of pulse data files.  In principle, there are several data file types:
* LJH files
* PLS files
* LANL files

...but in practice, we are not ever using PLS files, and LANLFile is deprecated.
Therefore, this module contains only three concrete classes, the VirtualFile,
LJHFile, and the LANLFile (along with the abstract base class MicrocalFile).
VirtualFile is for treating an array of data as if it were a file.

If you find yourself wanting to read PLS (or other?) file types,
then make a new class that inherits from MicrocalFile and calls
MicrocalFile.__init__ to verify that it has the required interface:
* read_segment(segment_num)
* read_trace(trace_num)
* copy()

Created on Feb 16, 2011
"""


import numpy as np
import os
from packaging.version import Version
import logging
import collections
LOG = logging.getLogger("mass")


class MicrocalFile(object):
    """A set of data on disk containing triggered records from a microcalorimeter.

    The pulses can be noise or X-rays.  This is meant to be
    an abstract class.  Use files.LJHFile() or VirtualFile(). In the future,
    other derived classes could implement read_segment, copy, and read_trace to
    process other file types.
    """

    def __init__(self):
        # Filename of the data file
        self.filename = None
        self.channum = 99999
        self.nSamples = 0
        self.nPresamples = 0
        self.timebase = 0.0
        self.n_segments = 0
        self.data = None
        self.__cached_segment = None

    def __str__(self):
        """Summary for the print function"""
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time" % (
            self.__class__.__name__, self.filename, self.nSamples, self.nPresamples,
            1e6*self.timebase)

    def __repr__(self):
        """Compact representation of how to construct from a filename."""
        return "%s('%s')" % (self.__class__.__name__, self.filename)

    def read_segment(self, segment_num=0):
        """Read a segment of the binary data of the given number (0,1,...)."""
        raise NotImplementedError("%s is an abstract class." % self.__class__.__name__)

    def read_trace(self, trace_num=0):
        """Read a single pulse record from the binary data."""
        raise NotImplementedError("%s is an abstract class." % self.__class__.__name__)

    def copy(self):
        """Make a usable copy of self."""
        raise NotImplementedError("%s is an abstract class." % self.__class__.__name__)

    def iter_segments(self, first=0, end=-1):
        """An iterator over all segments.  Read in segments one at a time and yield triplet:
        (first pulse number, 1+last pulse number, the segment number just read, the data).

        <first> The first segment to read.
        <end>   One past the last segment to read, or -1 to read through the last segment."""

        if end <= first:
            end = self.n_segments
        for segment_num in range(first, end):
            first_pnum, end_pnum, data = self.read_segment(segment_num)
            yield first_pnum, end_pnum, segment_num, data

    def clear_cache(self):
        """File objects can cache one "segment" of raw data.  Sometimes it's nice to delete
        this from memory in order to free up unneeded cache, especially before copying a
        MicrocalFile object."""
        self.data = None
        self.__cached_segment = None


class VirtualFile(MicrocalFile):
    """Object to act like a single microcalorimeter data file on disk, though the data are all
    held only in memory.
    """

    def __init__(self, data, times=None, presamples=0):
        """Initilize with in-memory data.

        Args:
            data: a 2d ndarray of pulse records, each row being one pulse.
            times: a 1d array of pulse times (or default None)
            presamples: number samples considered presamples (default 0)
        """
        super(VirtualFile, self).__init__()
        self.data = np.asarray(data, dtype=np.int16)
        self.nSamples = data.shape[1]
        self.nPulses = data.shape[0]
        self.nPresamples = presamples
        self.filename = "virtual_file_chan1.vtf"
        self.n_segments = 1
        self.pulses_per_seg = self.nPulses
        self.segmentsize = self.pulses_per_seg * self.nSamples * 2
        self.timestamp_offset = 0
        self.timebase = 0.0

        if times is None:
            self.datatimes_float = np.zeros(self.nPulses, dtype=float)
        else:
            self.datatimes_float = np.asarray(times, dtype=float)

    def copy(self):
        """Return a copy of the object.  Handy for updating method definitions."""
        c = VirtualFile(self.data)
        c.__dict__.update(self.__dict__)
        return c

    def read_trace(self, trace_num):
        """Return the data for pulse number <trace_num>"""
        if trace_num >= self.nPulses:
            raise ValueError("This VirtualFile has only %d pulses" % self.nPulses)
        return self.data[trace_num]

    def read_segment(self, segment_num=0):
        """
        Returns:
            (first, end, data) for segment number <segment_num>, where
            <first> is the first pulse number in that segment, <end>-1 is the last,
            and <data> is a 2-d array of shape [pulses_this_segment, self.nSamples].
        """
        if segment_num > 0:
            raise ValueError("VirtualFile objects have only one segment")
        return 0, self.nPulses, self.data


def read_ljh_header(filename):
    TOO_LONG_HEADER = 256  # headers with more than this many lines are ridiculous and an error

    header_dict = collections.OrderedDict()
    with open(filename, "rb") as fp:
        i = 0
        while True:
            i += 1
            line = fp.readline()
            if line.startswith(b"#End of Header"):
                break
            elif line == b"":
                raise Exception("reached EOF before #End of Header")
            elif i > TOO_LONG_HEADER:
                raise IOError("header is too long--seems not to contain '#End of Header'\n"
                              + "in file %s" % filename)
            elif b":" in line:
                a, b = line.split(b":", 1)  # maxsplits=1, py27 doesnt support keyword
                a = a.strip()
                b = b.strip()
                if a in header_dict and a != b"Dummy":
                    print("repeated header entry {}".format(a))
                header_dict[a.strip()] = b.strip()
            else:
                continue  # ignore lines without ":"
        header_size = fp.tell()
    return header_dict, header_size


class LJHFile(MicrocalFile):
    """Read a single LJH file of version 2.2 or 2.1.

    The class is not meant to be created directly. Instead, use the class method
    `open(filename)` to return an instance of the appropriate subclass (either
    `LJHFile2_2` or `LJHFile2_1`), determined by reading the LJH header before
    creating the instance.

    usage:
    filename = "path/to/my/file_chan1.ljh"
    ljh = mass.LJHFile.open(filename)
    """

    @classmethod
    def open(cls, filename):
        """A factory-like function.

        Read the LJH header and return an instance of the appropriate subclass of `LJHFile`
        based on contents of the LJH version string.

        This is the appropriate way to create an `LJHFile` in normal usage.
        """
        header_dict, header_size = read_ljh_header(filename)
        version_str = header_dict[b'Save File Format Version']

        if Version(version_str.decode()) < Version("2.2.0"):
            return LJHFile2_1(filename, header_dict, header_size)
        else:
            return LJHFile2_2(filename, header_dict, header_size)

    def __init__(self, filename, header_dict, header_size):
        """Users shouldn't call this method directly; call class method `open` instead."""
        super().__init__()
        self.filename = filename
        self.header_dict = header_dict
        self.header_size = header_size
        self.client = None
        self.channum = int(filename.split("_chan")[1].split(".")[0])
        self.sample_usec = None
        self.timestamp_offset = None
        self.row_number = None
        self.column_number = None
        self.number_of_rows = None
        self.number_of_columns = None
        self.version_str = None
        self._parse_header()
        self.set_segment_size()

    def _parse_header(self):
        """Parse the complete `self.header_dict`, filling key attributes from it."""
        filename = self.filename
        header_dict = self.header_dict

        # extract required values from header_dict
        # use header_dict.get for default values
        self.timebase = float(header_dict[b"Timebase"])
        self.nSamples = int(header_dict[b"Total Samples"])
        self.nPresamples = int(header_dict[b"Presamples"])
        # column number and row number have entries like "Column number (from 0-0 inclusive)"
        row_number_k = [k for k in header_dict.keys() if k.startswith(b"Row number")]
        if len(row_number_k) > 0:
            self.row_number = int(header_dict[row_number_k[0]])
        col_number_k = [k for k in header_dict.keys() if k.startswith(b"Column number")]
        if len(col_number_k) > 0:
            self.row_number = int(header_dict[col_number_k[0]])
        self.client = header_dict.get(b"Software Version", b"UNKNOWN")
        self.number_of_columns = int(header_dict.get(b"Number of columns", -1))
        self.number_of_rows = int(header_dict.get(b"Number of rows", -1))
        self.timestamp_offset = float(header_dict.get(b"Timestamp offset (s)", b"-1"))

        self.version_str = header_dict[b'Save File Format Version']
        self.binary_size = os.stat(filename).st_size - self.header_size
        self.nPulses = self.binary_size // self.pulse_size_bytes

        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the "true value" of nPresamples. For now, assume that only
        # DASTARD clients have this figure correct.
        if b"DASTARD" not in self.client:
            self.nPresamples += 3

        # This used to be fatal. It prevented opening files cut short by
        # a crash of the DAQ software, so we made it just a warning.
        if self.nPulses * self.pulse_size_bytes != self.binary_size:
            LOG.warning("Warning: The binary size "
                        + "(%d) is not an integer multiple of the pulse size %d bytes" %
                        (self.binary_size, self.pulse_size_bytes))
            LOG.warning("%06s" % filename)

        # Record the sample times in microseconds
        self.sample_usec = (np.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    def set_segment_size(self, segmentsize=None):
        # Segments are no longer a critical part of how MASS handles memory, but it still makes
        # sense to offer mid-sized data chunks for data processing.
        if segmentsize is None:
            segmentsize = 2**24
        maxitems = segmentsize // self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than pulse record (%d bytes)" %
                             (segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems*self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize // self.pulse_size_bytes
        self.n_segments = 1 + (self.binary_size - 1) // self.segmentsize

    def _open_mm(self):
        self._mm = np.memmap(self.filename, offset=self.header_size, shape=(self.nPulses,),
                             dtype=self.dtype, mode="r")

    @property
    def alldata(self):
        return self._mm["data"]

    def __getitem__(self, item):
        return self.alldata[item]

    def read_trace(self, trace_num, with_timing=False):
        """Return a single data trace (number <trace_num>).

        If `with_timing` is True, return (rowcount, posix_usec, pulse_record), otherwise just pulse_record.
        This comes either from cache or by reading off disk, if needed.
        """
        pulse_record = self.alldata[trace_num]
        if with_timing:
            return (self.rowcount[trace_num], self.datatimes_raw[trace_num], pulse_record)
        return pulse_record

    def read_segment(self, segment_num=0):
        """Map segment `segment_num` to `self.data`, as it was before we started using a
        `np.memmap` to store the `self.alldata`.

        Return (first, end, data) where first is the pulse number of the first pulse read,
        end is 1+the number of the last one read, and data is the full array.

        Args:
            <segment_num> Number of the segment to read.
        """
        if segment_num > self.n_segments:
            raise ValueError("File %s has only %d segments;\n\tcannot open segment %d" %
                             (self.filename, self.n_segments, segment_num))

        first = segment_num * self.pulses_per_seg
        end = min(first+self.pulses_per_seg, self.nPulses)
        self.data = self.alldata[first:end]
        return first, end, self.data


class LJHFile2_1(LJHFile):
    def __init__(self, filename, header_dict, header_size):
        super().__init__(filename, header_dict, header_size)

        # This is for LJH file versions 2.1 (and earlier).
        # The key distinction is how pulse arrival time data is encoded. Each pulse has a
        # timestamp encoded in a weird way that contains arrival time at 4 µs resolution.
        # If the frame time is at least 4 µs, the exact frame number can be recovered.
        self.dtype = [
            ("internal_us", np.uint8),
            ("internal_unused", np.uint8),
            ("internal_ms", np.uint32),
            ("data", np.uint16, (self.nSamples,))
        ]
        self.frame_count = None
        self._open_mm()

    @property
    def pulse_size_bytes(self):
        return 6 + 2 * self.nSamples

    def _parse_times(self):
        # Time format is ugly.  From bytes 0-5 of a pulse, the bytes are uxmmmm,
        # where u is a byte giving microseconds/4, x is a reserved byte, and mmmm is a 4-byte
        # little-ending giving milliseconds.  The uu should always be in [0,999]
        # The uu was added on Sept 21, 2011, so it will be 0 on all earlier data files.

        # The old way was to store the time as a 32-bit int.  New way: double float
        # Store times as seconds in floating point.  Max value is 2^32 ms = 4.3x10^6
        NS_PER_4USEC_TICK = 4000
        NS_PER_MSEC = 1000000
        datatime_ns = NS_PER_4USEC_TICK*np.asarray(self._mm["internal_us"], dtype=np.int64)
        datatime_ns[:] += NS_PER_MSEC*np.asarray(self._mm["internal_ms"], dtype=np.int64)
        # since the timestamps is stored in 4 µs units, which are not commensurate with the actual frame rate,
        # we can be more precise if we convert to frame number, then back to time
        # this should as long as the frame rate is greater than or equal to 4 µs

        # this is integer division but rounding up
        NS_PER_FRAME = np.int64(self.timebase*1e9)
        FOURPOINT = 3  # account for 4-point triggering algorithm
        self.frame_count = (datatime_ns - 1) // NS_PER_FRAME + 1 + FOURPOINT

    @property
    def rowcount(self):
        if self.frame_count is None:
            self._parse_times()
        return np.array(self.frame_count*self.number_of_rows+self.row_number, dtype=np.int64)

    @property
    def datatimes_float(self):
        if self.frame_count is None:
            self._parse_times()
        return (self.frame_count + self.row_number / float(self.number_of_rows))*self.timebase

    @property
    def datatimes_raw(self):
        return np.asarray(self.datatimes_float / 1e-6 + 0.5, dtype=int)


class LJHFile2_2(LJHFile):
    def __init__(self, filename, header_dict, header_size):
        super().__init__(filename, header_dict, header_size)

        # Version 2.2 and later include two pieces of time-like information for each pulse.
        # 8 bytes - Int64 row count number
        # 8 bytes - Int64 posix microsecond time
        # Technically both could be read as uint64, but then you get overflows when differencing;
        # we'll give up a factor of 2 to avoid that.
        self.dtype = [
            ("rowcount", np.int64),
            ("posix_usec", np.int64),
            ("data", np.uint16, (self.nSamples,))
        ]
        self._open_mm()

    @property
    def pulse_size_bytes(self):
        return 16 + 2 * self.nSamples

    @property
    def rowcount(self):
        return self._mm["rowcount"]

    @property
    def datatimes_raw(self):
        return self._mm["posix_usec"]

    @property
    def datatimes_float(self):
        return self.datatimes_raw/1e6


def make_ljh_header(header_dict):
    """Returns a string containing an LJH header (version 2.2.0).

    Args:
        header_dict (dict): should contain the following keys: asctime, timebase,
            nPresamples, nSamples
    """

    ljh_header = """#LJH Memorial File Format
Save File Format Version: %(version_str)s
Software Version: Fake LJH file
Software Driver Version: n/a
Date: %(asctime)s GMT
Acquisition Mode: 0
Digitized Word Size in bytes: 2
Location: LANL, presumably
Cryostat: Unknown
Thermometer: Unknown
Temperature (mK): 100.0000
Bridge range: 20000
Magnetic field (mGauss): 100.0000
Detector:
Sample:
Excitation/Source:
Operator: Unknown
SYSTEM DESCRIPTION OF THIS FILE:
USER DESCRIPTION OF THIS FILE:
#End of description
Number of Digitizers: 1
Number of Active Channels: 1
Timestamp offset (s): 1304449182.876200
Digitizer: 1
Description: CS1450-1 1M ver 1.16
Master: Yes
Bits: 16
Effective Bits: 0
Anti-alias low-pass cutoff frequency (Hz): 0.000
Timebase: %(timebase).4e
Number of samples per point: 1
Presamples: %(nPresamples)d
Total Samples: %(nSamples)d
Trigger (V): 250.000000
Tigger Hysteresis: 0
Trigger Slope: +
Trigger Coupling: DC
Trigger Impedance: 1 MOhm
Trigger Source: CH A
Trigger Mode: 0 Normal
Trigger Time out: 351321
Use discrimination: No
Channel: 1.0
Description: A (Voltage)
Range: 0.500000
Offset: -0.000122
Coupling: DC
Impedance: 1 Ohms
Inverted: No
Preamp gain: 1.000000
Discrimination level (%%): 1.000000
#End of Header
""" % header_dict
    return ljh_header.encode()
