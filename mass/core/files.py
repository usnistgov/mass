"""
The `mass.core.files` module contains classes required for handling the various types
of pulse data files.  In principle, there could be multiple data file types.
But this module contains only two concrete classes, the VirtualFile and
the LJHFile (along with the abstract base class MicrocalFile).
VirtualFile is for treating an array of data as if it were a file.

If you find yourself wanting to read other file types in the future,
then make a new class that inherits from MicrocalFile and calls
MicrocalFile.__init__ to verify that it has the required interface:
* read_trace(trace_num)
* copy()

Created on Feb 16, 2011
"""

__all__ = [
    "MicrocalFile",
    "LJHFile",
    "VirtualFile"
]

import os
import logging
import collections
from deprecation import deprecated

import numpy as np
from packaging.version import Version
from mass import __version__
LOG = logging.getLogger("mass")


class MicrocalFile:
    """A set of data on disk containing triggered records from a microcalorimeter.

    The pulses can be noise or X-rays.  This is meant to be an abstract class.
    Use `LJHFile.open()` or `VirtualFile()` to create one.
    In the future, other derived classes could implement copy() and read_trace() to
    process other file types.
    """

    def __init__(self):
        self.filename = None
        self.channum = 99999
        self.nSamples = 0
        self.nPresamples = 0
        self.timebase = 0.0
        self.n_segments = 0
        self.data = None

    def __str__(self):
        """Summary for the print function"""
        return f"{self.__class__.__name__} path '{self.filename}'\n"\
            f"{self.nSamples} samples ({self.nPresamples} pretrigger) at {1e6 * self.timebase:.2f} µs sample time"

    def __repr__(self):
        """Compact representation of how to construct from a filename."""
        return f"{self.__class__.__name__}('{self.filename}')"

    def read_trace(self, trace_num):
        """Read a single pulse record from the binary data."""
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")

    def copy(self):
        """Make a usable copy of self."""
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")

    def source(self):
        """Name of the data source"""
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")


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
        super().__init__()
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
            raise ValueError(f"This VirtualFile has only {self.nPulses} pulses")
        return self.data[trace_num]
    
    @property
    def source(self):
        return "VirtualFile"


def read_ljh_header(filename):
    """Read an LJH file's ASCII header (any LJH file version).

    Opens and closes the file.

    Return `(hd, hs)`
    * `hd` is the dictionary of header information
    * `hs` is the size (bytes) of the ASCII header. Binary data starts at this offset
      in the file.
    """
    TOO_LONG_HEADER = 256  # headers with more than this many lines are ridiculous and an error

    header_dict = collections.OrderedDict()
    with open(filename, "rb") as fp:
        i = 0
        while True:
            i += 1
            line = fp.readline()
            if line.startswith(b"#End of Header"):
                break
            if line == b"":
                raise Exception("reached EOF before #End of Header")
            if i > TOO_LONG_HEADER:
                raise OSError("header is too long--seems not to contain '#End of Header'\n"
                              f"in file {filename}")
            if b":" in line:
                a, b = line.split(b":", 1)  # maxsplits=1, py27 doesnt support keyword
                a = a.strip()
                b = b.strip()
                if a in header_dict and a != b"Dummy":
                    print(f"repeated header entry {a}")
                header_dict[a.strip()] = b.strip()
            else:
                continue  # ignore lines without ":"
        header_size = fp.tell()
    return header_dict, header_size


class LJHFile(MicrocalFile):
    """Read a single LJH file of version 2.2 or 2.1.

    The class is not meant to be created directly. Instead, use the class method
    `LJHFile.open(filename)` to return an instance of the appropriate subclass (either
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
        self.row_number = 0
        self.column_number = 0
        self.number_of_rows = 1
        self.number_of_columns = 1
        self.subframe_offset = 0
        self.subframe_divisions = 1
        self.version_str = None
        self._mm = None
        self._parse_header()
        self.set_segment_size()

    def copy(self):
        """Return a copy of the object.  Handy for updating method definitions."""
        c = self.__class__(self.filename, self.header_dict, self.header_size)
        c.__dict__.update(self.__dict__)
        return c

    def __repr__(self):
        """Compact representation of how to construct from a filename."""
        return f"LJHFile.open('{self.filename}')"

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
        self.number_of_columns = int(header_dict.get(b"Number of columns", 1))
        self.number_of_rows = int(header_dict.get(b"Number of rows", 1))
        self.timestamp_offset = float(header_dict.get(b"Timestamp offset (s)", b"-1"))

        # Read the new (Feb 2024) subframe information. If missing, assume the old TDM values pertain
        # (so # of rows -> subframe divisions, and row # -> subframe offset), unless source is Abaco,
        # in which case use 64 subframe divisions and offset of 0.
        default_divisions = self.number_of_rows
        default_offset = self.row_number
        if "Abaco" in self.source:
            # The external trigger file can override this, but assume 64 divisions at first.
            default_divisions = 64
            default_offset = 0
        self.subframe_divisions = int(header_dict.get(b"Subframe divisions", default_divisions))
        self.subframe_offset = int(header_dict.get(b"Subframe offset", default_offset))

        self.version_str = header_dict[b'Save File Format Version']
        self.binary_size = os.stat(filename).st_size - self.header_size
        self.nPulses = self.binary_size // self.pulse_size_bytes

        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the nominal value of nPresamples to get the "true" value.
        # For now, assume that only DASTARD clients have this figure correct.
        # So when there are 500 samples before the first non-trivial (triggered) data
        # sample, DASTARD will say nPresamples=500, but earlier clients will say
        # nPresamples=497 in the LJH file. Correct the latter here.
        if b"DASTARD" not in self.client:
            self.nPresamples += 3

        # Files cut short by a crash of the DAQ software (or whatever) will generate
        # a warning here (unless they happened to cut off at a record boundary).
        if self.nPulses * self.pulse_size_bytes != self.binary_size:
            LOG.warning("Warning: The binary size "
                        "(%d) is not an integer multiple of the pulse size %d bytes",
                        self.binary_size, self.pulse_size_bytes)
            LOG.warning("%06s", filename)

        # It's handy to precompute the times of each sample in a record (in µs)
        self.sample_usec = (np.arange(self.nSamples) - self.nPresamples) * self.timebase * 1e6

    def set_segment_size(self, segmentsize=None):
        """Set the `segmentsize` in bytes.

        Segments are not an intrinsic part of handling MASS memory, but they can be convenient
        as a way to offer modestly sized chunks of data for certain processing.
        """
        # Segments are no longer a critical part of how MASS handles memory, but it still makes
        # sense to offer mid-sized data chunks for data processing.
        if segmentsize is None:
            segmentsize = 2**24
        maxitems = segmentsize // self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than pulse record (%d bytes)" %
                             (segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems * self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize // self.pulse_size_bytes
        self.n_segments = 1 + (self.binary_size - 1) // self.segmentsize

    def _open_mm(self):
        self._mm = np.memmap(self.filename, offset=self.header_size, shape=(self.nPulses,),
                             dtype=self.dtype, mode="r")

    @property
    def alldata(self):
        return self._mm["data"]
    
    @property
    def source(self):
        "Report the 'Data source' as found in the LJH header."
        if b"Data source" in self.header_dict:
            return self.header_dict[b"Data source"].decode()
        return "Lancero (assumed)"

    def __getitem__(self, item):
        return self.alldata[item]

    def read_trace(self, trace_num):
        """Return a single data trace (number <trace_num>)."""
        return self.alldata[trace_num]

    def read_trace_with_timing(self, trace_num):
        """Return a single data trace as (subframecount, posix_usec, pulse_record)."""
        pulse_record = self.alldata[trace_num]
        return (self.subframecount[trace_num], self.datatimes_raw[trace_num], pulse_record)


class LJHFile2_1(LJHFile):
    """Class to handle LJH version 2.1 files."""

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
        "Call this only on-demand, when row counts or record times are asked for."
        # Time format is ugly.  From bytes 0-5 of a pulse, the bytes are uxmmmm,
        # where u is a byte giving microseconds/4, x is a reserved byte, and mmmm is a 4-byte
        # little-ending giving milliseconds.  The uu should always be in [0,999]
        # The uu was added on Sept 21, 2011, so it will be 0 on all earlier data files.

        # The old way was to store the time as a 32-bit int.  New way: double float
        # Store times as seconds in floating point.  Max value is 2^32 ms = 4.3x10^6
        NS_PER_4USEC_TICK = 4000
        NS_PER_MSEC = 1000000
        datatime_ns = NS_PER_4USEC_TICK * np.asarray(self._mm["internal_us"], dtype=np.int64)
        datatime_ns[:] += NS_PER_MSEC * np.asarray(self._mm["internal_ms"], dtype=np.int64)

        NS_PER_FRAME = np.int64(self.timebase * 1e9)
        FOURPOINT = 3  # account for 4-point triggering algorithm
        self.frame_count = (1 + FOURPOINT) + (datatime_ns - 1) // NS_PER_FRAME

    @property
    def subframecount(self):
        if self.frame_count is None:
            self._parse_times()
        return np.array(self.frame_count*self.subframe_divisions + self.subframe_offset, dtype=np.int64)

    @property
    @deprecated(deprecated_in="0.8.2", details="Use subframecount, which is equivalent but better named")
    def rowcount(self):
        return self.subframecount

    @property
    def datatimes_float(self):
        if self.frame_count is None:
            self._parse_times()
        return (self.frame_count + self.subframe_offset / float(self.subframe_divisions))*self.timebase

    @property
    def datatimes_raw(self):
        return np.asarray(self.datatimes_float / 1e-6 + 0.5, dtype=int)


class LJHFile2_2(LJHFile):
    """Class to handle LJH version 2.2 files."""

    def __init__(self, filename, header_dict, header_size):
        super().__init__(filename, header_dict, header_size)

        # Version 2.2 and later include two pieces of time-like information for each pulse.
        # 8 bytes - Int64 subframe count number (for TDM, equiv to row count number)
        # 8 bytes - Int64 posix microsecond time since the epoch 1970.
        # Technically both could be read as uint64, but then you get overflows when differencing;
        # we'll happily give up a factor of 2 in dynamic range to avoid that.
        self.dtype = [
            ("subframecount", np.int64),
            ("posix_usec", np.int64),
            ("data", np.uint16, (self.nSamples,))
        ]
        self._open_mm()

    @property
    def pulse_size_bytes(self):
        return 16 + 2 * self.nSamples

    @property
    def subframecount(self):
        return self._mm["subframecount"]

    @property
    def datatimes_raw(self):
        return self._mm["posix_usec"]

    @property
    def datatimes_float(self):
        return self.datatimes_raw / 1e6


def make_ljh_header(header_dict):
    """Returns a string containing an LJH header (version 2.2.0).

    Args:
        header_dict (dict): should contain at least the following keys:
            asctime, timebase, nPresamples, nSamples
    """
    version_str = header_dict["version_str"]
    asctime = header_dict["asctime"]
    timebase = header_dict["timebase"]
    nSamples = header_dict["nSamples"]
    nPresamples = header_dict["nPresamples"]
    header_lines = [
        "#LJH Memorial File Format",
        f"Save File Format Version: {version_str}",
        f"Software Version: MASS-generated LJH file, MASS version {__version__}",
        "Software Driver Version: n/a",
        f"Date: {asctime} GMT",
        "Acquisition Mode: 0",
        "Digitized Word Size in bytes: 2",
        "Operator: Unknown",
        "SYSTEM DESCRIPTION OF THIS FILE:",
        "USER DESCRIPTION OF THIS FILE:",
        "#End of description",
        "Number of Digitizers: 1",
        "Number of Active Channels: 1",
        f"Timestamp offset (s): {1.0e9:.6f}",
        "Digitizer: 1",
        "Master: Yes",
        "Bits: 16",
        f"Timebase: {timebase:.4e}",
        "Number of samples per point: 1",
        f"Presamples: {nPresamples}",
        f"Total Samples: {nSamples}",
        "Channel: 1.0",
        "Description: A (Voltage)",
        "Range: 0.500000",
        "Offset: -0.000122",
        "Inverted: No",
        "#End of Header",
        ""  # need this to get a newline at the end of the list
    ]
    ljh_header = "\n".join(header_lines)
    return ljh_header.encode()
