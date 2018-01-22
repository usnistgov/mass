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
from distutils.version import StrictVersion


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
            self.datatimes_float = np.zeros(self.nPulses, dtype=np.float)
        else:
            self.datatimes_float = np.asarray(times, dtype=np.float)

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


class LJHFile(MicrocalFile):
    """A single LJH-format file.

    All non-LJH-specific data and methods appear in the parent MicrocalFile class.
    """

    TOO_LONG_HEADER = 100  # headers can't contain this many lines, or they are insane!

    def __init__(self, filename, segmentsize=(2**23)):
        """Open an LJH file for reading.

        Read its header.  Set the standard segment size **in bytes** so that
        read_segment() will always return segments of a fixed size.

        Args:
            <filename>   Path to the file to be read.
            <segmentsize>  Size of each segment **in bytes** that will be returned in read_segment()
                 The actual segmentsize will be rounded down to be an integer number of
                 pulses.
        """
        super(LJHFile, self).__init__()
        self.filename = filename
        self.channum = int(filename.split("_chan")[1].split(".")[0])
        self.header_lines = []
        self.sample_usec = None
        self.timestamp_offset = 0.0
        self.pulses_per_seg = 0
        self.segmentsize = 0
        self.n_segments = 0
        self.segment_pulses = 0
        self.header_size = 0
        self.pulse_size_bytes = 0
        self.row_number = -1
        self.column_number = -1
        self.number_of_rows = -1
        self.number_of_columns = -1
        self.data = None
        self.version_str = None
        self.__cached_segment = None
        self.__read_header(filename)
        self.set_segment_size(segmentsize)

        self.datatimes_float = None
        self.datatimes_float_old = None
        self.rowcount = None

        self.post22_data_dtype = np.dtype([('rowcount', np.int64),
                                           ('posix_usec', np.int64),
                                           ('data', np.uint16, self.nSamples)])

        if StrictVersion(self.version_str.decode()) >= StrictVersion("2.2.0"):
            self.__read_binary = self.__read_binary_post22
        else:
            self.__read_binary = self.__read_binary_pre22

    def copy(self):
        """Return a deep copy of the object."""
        self.clear_cache()
        c = LJHFile(self.filename, self.segmentsize)
        c.__dict__.update(self.__dict__)
        return c

    def __read_header(self, filename):
        """Read in the text header of an LJH file.

        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples

        Args:
            filename: path to the file to be opened.
        """

        fp = open(filename, "rb")

        lines = []
        while True:
            line = fp.readline()
            if line == b"":
                if len(lines) == 0:
                    raise IOError("No header found.\n   File: %s" % filename)
                break
            lines.append(line)
            if line.startswith(b"#End of Header"):
                break
            elif line.startswith(b"Timebase"):
                words = line.split()
                self.timebase = float(words[-1])
            elif line.startswith(b"Total Samples"):
                words = line.split()
                self.nSamples = int(words[-1])
            elif line.startswith(b"Presamples"):
                words = line.split()
                self.nPresamples = int(words[-1])
            elif line.startswith(b"Row number"):
                words = line.split()
                self.row_number = int(words[-1])
            elif line.startswith(b"Column number"):
                words = line.split()
                self.column_number = int(words[-1])
            elif line.startswith(b"Number of rows"):
                words = line.split()
                self.number_of_rows = int(words[-1])
            elif line.startswith(b"Number of columns"):
                words = line.split()
                self.number_of_columns = int(words[-1])
            elif line.startswith(b"Timestamp offset (s)"):
                words = line.split()
                try:
                    self.timestamp_offset = float(words[-1])
                except:
                    self.timestamp_offset = 0.0
            elif line.startswith(b"Save File Format Version:"):
                words = line.split()
                self.version_str = words[-1]

            if len(lines) > self.TOO_LONG_HEADER:
                raise IOError("header is too long--seems not to contain '#End of Header'\n" +
                              "in file %s" % filename)

        self.header_lines = lines
        self.header_size = fp.tell()
        # fp.seek(0, os.SEEK_END)
        # self.binary_size = fp.tell() - self.header_size
        self.binary_size = os.stat(filename).st_size - self.header_size
        fp.close()

        if StrictVersion(self.version_str.decode()) >= StrictVersion("2.2.0"):
            self.pulse_size_bytes = (16 + 2 * self.nSamples)
        else:
            self.pulse_size_bytes = (6 + 2 * self.nSamples)

        self.nPulses = self.binary_size // self.pulse_size_bytes

        # Check for major problems in the LJH file:
        if len(self.header_lines) < 1:
            raise IOError("No header found.\n   File: %s" % filename)
        if self.timebase is None:
            raise IOError("No 'Timebase' line found in header.\n   File: %s" % filename)
        if self.nSamples is None:
            raise IOError("No 'Total Samples' line found in header.\n   File: %s" % filename)
        if self.nPresamples is None:
            raise IOError("No 'Presamples' line found in header.\n   File: %s" % filename)
        if self.nPulses < 1:
            print("Warning: no pulses found.\n   File: %s" % filename)

        # This used to be fatal, but it prevented opening files cut short by
        # a crash of the DAQ software.
        if self.nPulses * self.pulse_size_bytes != self.binary_size:
            print("Warning: The binary size " +
                  "(%d) is not an integer multiple of the pulse size %d bytes" %
                  (self.binary_size, self.pulse_size_bytes))
            print("%06s" % filename)

        # Record the sample times in microseconds
        self.sample_usec = (np.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    def set_segment_size(self, segmentsize):
        """Set the standard segmentsize used in the read_segment() method.

        This number will
        be rounded down to equal an integer number of pulses.
        Raises ValueError if segmentsize is smaller than a single pulse.
        """
        maxitems = segmentsize // self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than pulse record (%d bytes)" %
                             (segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems*self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize // self.pulse_size_bytes
        self.n_segments = 1 + (self.binary_size - 1) // self.segmentsize
        self.__cached_segment = None

    def __getitem__(self, item):
        try:
            item = int(item)
            if item < 0 or item >= self.nPulses:
                raise ValueError("Out of range")
            return self.read_trace(item)
        except TypeError:
            pass

        first_slice = None
        second_slice = slice(None, None)

        if isinstance(item, np.ndarray):
            if item.ndim == 1 and item.dtype == np.bool:
                if item.shape[0] != self.nPulses:
                    raise ValueError("Shape doesn't match.")
                trace_range = np.arange(self.nPulses, dtype=np.int64)[item]
                num_samples = self.nSamples
        elif isinstance(item, list):
            try:
                trace_range = np.array(item, dtype=np.uint64)

                if trace_range.ndim != 1:
                    raise ValueError("Unsupported list type.")
                num_samples = self.nSamples
            except ValueError:
                raise ValueError("Unsupported list type.")
        else:
            if isinstance(item, slice):
                first_slice = item

            if isinstance(item, tuple):
                if len(item) is not 2:
                    raise ValueError("Not supported dimensions!")
                first_slice = item[0]
                second_slice = item[1]

            trace_range = range(self.nPulses)[first_slice]
            num_samples = len(range(self.nSamples)[second_slice])

        num_traces = len(trace_range)

        last_segment = trace_range[0] // self.pulses_per_seg
        self.read_segment(last_segment)

        traces = np.zeros((num_traces, num_samples), dtype=np.uint16)

        for i, j in enumerate(trace_range):
            segment_num = j // self.pulses_per_seg
            trace_idx = j % self.pulses_per_seg

            if segment_num is not last_segment:
                self.read_segment(segment_num)
                last_segment = segment_num

            traces[i] = self.data[trace_idx][second_slice]

        return traces

    def read_trace(self, trace_num):
        """Return a single data trace (number <trace_num>).

        This comes either from cache or by reading off disk, if needed.
        """
        if trace_num >= self.nPulses:
            raise ValueError("This VirtualFile has only %d pulses" % self.nPulses)

        segment_num = trace_num // self.pulses_per_seg
        self.read_segment(segment_num)
        return self.data[trace_num % self.pulses_per_seg]

    def read_segment(self, segment_num=0):
        """Read a section of the binary data of the given number (0,1,...) and size.
        It is okay to call this out of order.  The last segment might be shorter than others.

        Raises ValueError if there is no such section number.

        Return (first, end, data) where first is the pulse number of the first pulse read,
        end is 1+the number of the last one read, and data is the full array.

        Args:
            <segment_num> Number of the segment to read.
        """
        # Use cached data, if possible
        if segment_num != self.__cached_segment or self.data is None:
            if segment_num * self.segmentsize > self.binary_size:
                raise ValueError("File %s has only %d segments;\n\tcannot open segment %d" %
                                 (self.filename, self.n_segments, segment_num))

            self.__read_binary(self.header_size + segment_num*self.segmentsize, self.segmentsize,
                               error_on_partial_pulse=True)
            self.__cached_segment = segment_num
        first = segment_num * self.pulses_per_seg
        end = first + self.data.shape[0]
        return first, end, self.data

    def clear_cached_segment(self):
        super(LJHFile, self).clear_cache()
        self.datatimes_float = None
        self.datatimes_float_old = None
        self.rowcount = None

    def __read_binary(self, skip=0, max_size=(2**26), error_on_partial_pulse=True):
        """Read the binary section of an LJH file.

        Also, interpret it, and store the results in self.data and
        self.datatimes_float.  This can potentially be less than the full file
        if <max_size> is non-negative and smaller than (binary section of) the
        file.

        The binary section consists of an unspecified number of records,
        each with the same size: 6 bytes plus 2 bytes per sample.  The six contain two null bytes
        and a 4-byte (little endian) timestamp in milliseconds since the timebase (which is
        given in the text header).

        Args:
            skip: Leading bytes to seek past.  Normally this should be the header length, but it
                can be greater.
            max_size: Maximum section size to read (in bytes).  If negative, then the entire file
                will be read.  (Beware: memory filling danger if <max_size> is negative!)
            error_on_partial_pulse: Whether to raise an error when caller requests non-integer
                number of pulses.
        """
        raise NotImplementedError("The method needs to be substituted by " + self.version_str)

    def __read_binary_post22(self, skip=0, max_size=(2**26), error_on_partial_pulse=True):
        """
        Version 2.2 and later include two pieces of time information for each pulse.
        8 bytes - Int64 row count number
        8 bytes - Int64 posix microsecond time
        Technically both could be read as uint64, but then you get overflows when differencing, so
        we'll give up a factor of 2 to avoid that.
        """
        if error_on_partial_pulse and (max_size > 0) and (max_size % self.pulse_size_bytes != 0):
            msg = "__read_binary(max_size=%d) requests a non-integer number of pulses" % max_size
            raise ValueError(msg)

        with open(self.filename, "rb") as fp:
            if skip > 0:
                fp.seek(skip)
            maxitems = max_size // self.pulse_size_bytes
            # should use a platform independent spec for the order of the bytes in the ints
            array = np.fromfile(fp, dtype=self.post22_data_dtype, sep="", count=maxitems)
            # fromfile will read up to max items

        self.rowcount = array["rowcount"]
        self.datatimes_float = array["posix_usec"] * 1e-6  # convert to floating point with units of seconds
        self.datatimes_raw = np.uint64(array["posix_usec"].copy())
        self.data = array["data"]

    def __read_binary_pre22(self, skip=0, max_size=(2**26), error_on_partial_pulse=True):
        """This is for LJH file versions 2.1 and earlier.

        The key distinction is how pulse arrival time data is encoded. Pre 2.2
        each pulse has a timestamp encoded in a weird way that contains arrival
        time at 4 usec resolution.  If the frame time is greater than or equal
        to 4 usec, the exact frame number can be recovered.
        """
        if max_size >= 0:
            maxitems = max_size // self.pulse_size_bytes
            BYTES_PER_WORD = 2
            wordcount = maxitems*self.pulse_size_bytes//BYTES_PER_WORD
            if error_on_partial_pulse and wordcount*BYTES_PER_WORD != max_size:
                msg = "__read_binary(max_size=%d) requests a non-integer number of pulses" % max_size
                raise ValueError(msg)
        else:
            wordcount = -1

        with open(self.filename, "rb") as fp:
            if skip > 0:
                fp.seek(skip)
            array = np.fromfile(fp, dtype=np.uint16, sep="", count=wordcount)

        # Let's not catch an Exception, if we don't know which one to catch.
        #     print(self.filename)
        #     print('array[-4:]', array[-4:])
        #     print('wordcount', wordcount, 'skip', skip)
        #     print('arrays.size', array.size, 'array.dtype', array.dtype)
        # raise

        # If data has a fractional record at the end, truncate to make it go away.
        self.segment_pulses = len(array) // (self.pulse_size_bytes // 2)
        array = array[:self.segment_pulses * (self.pulse_size_bytes // 2)]

        try:
            self.data = array.reshape([self.segment_pulses, self.pulse_size_bytes // 2])
        except ValueError as ex:
            print(skip, max_size, self.segment_pulses, self.pulse_size_bytes, len(array))
            raise ex

        # Time format is ugly.  From bytes 0-5 of a pulse, the bytes are uxmmmm,
        # where u is a byte giving microseconds/4, x is a reserved byte, and mmmm is a 4-byte
        # little-ending giving milliseconds.  The uu should always be in [0,999]
        # The uu was added on Sept 21, 2011, so it will be 0 on all earlier data files.

        # The old way was to store the time as a 32-bit int.  New way: double float
        # Store times as seconds in floating point.  Max value is 2^32 ms = 4.3x10^6
        datatime_4usec_tics = np.array(self.data[:, 0], dtype=np.uint64)
        datatime_4usec_tics += 250*(np.array(self.data[:, 1], dtype=np.uint64) +
                                    65536 * np.array(self.data[:, 2], dtype=np.uint64))
        NS_PER_4USEC_TICK = 4000
        NS_PER_FRAME = np.int64(self.timebase*1e9)
        # since the timestamps is stored in 4 us units, which are not commensurate with the actual frame rate,
        # we can be more precise if we convert to frame number, then back to time
        # this should as long as the frame rate is greater than or equal to 4 us

        # this is integer division but rounding up
        frame_count = (datatime_4usec_tics*NS_PER_4USEC_TICK - 1) // NS_PER_FRAME + 1
        frame_count += 3  # account for 4 point triggering algorithm
        # leave in the old calculation for comparison, later this should be removed
        SECONDS_PER_4MICROSECOND_TICK = (4.0/1e6)
        SECONDS_PER_MILLISECOND = 1e-3
        self.datatimes_float_old = np.array(self.data[:, 0], dtype=np.double)*SECONDS_PER_4MICROSECOND_TICK
        self.datatimes_float_old += self.data[:, 1]*SECONDS_PER_MILLISECOND
        self.datatimes_float_old += self.data[:, 2]*(SECONDS_PER_MILLISECOND*65536.)

        self.rowcount = np.array(frame_count*self.number_of_rows+self.row_number, dtype=np.int64)
        self.datatimes_float = (frame_count+self.row_number/float(self.number_of_rows))*self.timebase

        # Cut out zeros and the timestamp, which are 3 uint16 words @ start of each pulse
        self.data = self.data[:, 3:]


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
