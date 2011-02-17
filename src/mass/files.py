'''
Created on Feb 16, 2011

@author: fowlerj
'''

import channel
import numpy
import os

class LJHFile(channel.pulseRecords):
    """Process a single LJH-format file.  All non-LJH-specific data and methods
    appear in the parent pulseRecords class"""
    
    def __init__(self, filename, segmentsize=(2**26)):
        """Open an LJH file for reading.  Read its header.  Set the standard segment
        size **in bytes** so that _read_segment() will always return segments of a
        fixed size.
        <filename>   Path to the file to be read.
        <segmentsize>  Size of each segment **in bytes** that will be returned in _read_segment()
                     The actual segmentsize will be rounded down to be an integer number of 
                     pulses.
        """
        super(LJHFile, self).__init__()
        self.filename = filename
        self._read_header(filename)
        
        self._set_segmentsize(segmentsize)

#        self.good = numpy.ones(self.nPulses, dtype=numpy.bool)
#        self.bad = numpy.zeros(self.nPulses, dtype=numpy.bool)



    def _set_segmentsize(self, segmentsize):
        """Set the standard segmentsize used in the _read_segment() method.  This number will
        be rounded down to equal an integer number of pulses.
        
        Raises ValueError if segmentsize is smaller than a single pulse."""
        pulse_size_bytes = 6 + 2*self.nSamples
        maxitems = segmentsize/pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than the pulse record (%d bytes)"
                             %(segmentsize, pulse_size_bytes))
        self.segmentsize = maxitems*pulse_size_bytes


        
    def _read_header(self, filename):
        """
        Read in the text header of an LJH file.
        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples
        
        <filename>: path to the file to be opened.
        """
        TOO_LONG_HEADER=100 # headers can't contain this many lines, or they are insane!
        fp = open(filename,"r")
        lines=[]
        while True:
            line = fp.readline()
            lines.append(line)
            if line.startswith("#End of Header"):
                break
            elif line.startswith("Timebase"):
                words = line.split()
                self.timebase = float(words[-1])
            elif line.startswith("Total Samples"):
                words = line.split()
                self.nSamples = int(words[-1])
            elif line.startswith("Presamples"):
                words = line.split()
                self.nPresamples = int(words[-1])
            
            if len(lines) > TOO_LONG_HEADER:   
                raise IOError("header is too long--seems not to contain '#End of Header'")
            
        self.header_lines = lines
        self.header_size = fp.tell()
        fp.seek(0, os.SEEK_END)
        self.binary_size = fp.tell() - self.header_size
        fp.close()
        
        self.nPulses = self.binary_size / (6+2*self.nSamples)
        assert self.nPulses * (6+2*self.nSamples) == self.binary_size
        
        if self.timebase is None:
            raise IOError("No 'Timebase' line found in header")
        if self.nSamples is None:
            raise IOError("No 'Total Samples' line found in header")
        if self.nPresamples is None:
            raise IOError("No 'Presamples' line found in header")


    
    def _read_segment(self, segment_num=0):
        """Read a section of the binary data of the given number (0,1,...) and size.
        It is okay to call this out of order.  The last segment might be shorter than others.
        
        Raises ValueError if there is no such section number.

        Params:
        -------
        <segment_num> Number of the segment to read.
        """
        if segment_num*self.segmentsize > self.binary_size:
            raise ValueError("File %s has only %d segments;\n\tcannot open segment %d"%
                             (self.filename, 1+(self.binary_size-1)/self.segmentsize, segment_num))
        self.__read_binary(self.header_size + segment_num*self.segmentsize, self.segmentsize, 
                           error_on_partial_pulse=True)
        
        
    def __read_binary(self, skip=0, max_size=(2**26), error_on_partial_pulse=True):
        """Read the binary section of an LJH file, interpret it, and store the results in
        self.data and self.datatimes.  This can potentially be less than the full file
        if <max_size> is non-negative and smaller than (binary section of) the file.

        The binary section consists of an unspecified number of records,
        each with the same size: 6 bytes plus 2 bytes per sample.  The six contain two null bytes
        and a 4-byte (little endian) timestamp in milliseconds since the timebase (which is
        given in the text header).
        
        Params:
        -------
        <skip>      Leading bytes to seek past.  Normally this should be the header length, but it 
                    can be greater.
        <max_size>  Maximum section size to read (in bytes).  If negative, then the entire file
                    will be read.  (Beware: memory filling danger if <max_size> is negative!)
        <error_on_partial_pulse> Whether to raise an error when caller requests non-integer
                                 number of pulses.
        """
        fp = open(self.filename,"rb")
        if skip>0: fp.seek(skip)
        
        pulse_size_bytes = 6 + 2*self.nSamples
        if max_size >= 0:
            maxitems = max_size/pulse_size_bytes
            BYTES_PER_WORD = 2
            wordcount = maxitems*pulse_size_bytes/BYTES_PER_WORD
            if error_on_partial_pulse and wordcount*BYTES_PER_WORD != max_size:
                raise ValueError("__read_binary(max_size=%d) requests a non-integer number of pulses"%max_size)
        else:
            wordcount = -1

        array = numpy.fromfile(fp, dtype=numpy.uint16, sep="", count=wordcount)
        fp.close()
        
        self.nPulses = len(array)/(pulse_size_bytes/2)
        self.data = array.reshape([self.nPulses, pulse_size_bytes/2])
        
        # Careful: converting 2 little-endian 16-bit words to a single 32-bit word is tricky!
        self.datatimes = numpy.array(self.data[:,2], dtype=numpy.uint32) * (1<<16) 
        self.datatimes += (self.data[:,1])
        self.data = self.data[:,3:] # cut out the zeros and the timestamp, which are 3 uint16 words at the start of each pulse
        
        # Record the sample times in microseconds
        if self.dt is None: 
            self.dt = (numpy.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6
#        self.cuts = Cuts( self.nPulses)
                

    def __str__(self):
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        return "%s('%s')"%(self.__class__.__name__, self.filename)