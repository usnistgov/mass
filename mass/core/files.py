'''
The mass.files module contains classes required for handling the various types
of pulse data files.  In principle, there are several data file types:
* LANL files
* LJH files
* PLS files

...but in practice, we are not ever using PLS files.  Therefore, this module
contains only three concrete classes, the VirtualFile, LJHFile, and the LANLFile (along with 
the abstract base class MicrocalFile).  VirtualFile is for treating an array of data as if
it were a file.

If you find yourself wanting to read PLS (or other?) file types,
then make a new class that inherits from MicrocalFile and calls 
MicrocalFile.__init__ to verify that it has the required interface:
* read_segment(segment_num)
* read_trace(trace_num)
* copy()


The utilities root2ljh_translator and root2ljh_translate_all are used to convert
a ROOT file (or all such in one directory) into LJH format for more efficient
future use within mass.

Created on Feb 16, 2011
LANLFile and translation added June 2011 by Doug Bennett and Joe Fowler 

@author: fowlerj
'''

## \file files.py
# \brief File handling classes.
#
# The class MicrocalFile defines the shared interface of all derived classes that are
# specific for a single file type: LJHFile, LANLFile, ...?

import numpy
import os
import time
import glob

# Beware that the Mac Ports install of ROOT does not add 
# /opt/local/lib/root to the PYTHONPATH.  Still, you should do it yourself. 
# If you insist on adding to the path from within Python, you do:
# >>> import sys
# >>> sys.path.append('/opt/local/lib/root/') #Folder where ROOT.py lives

try:
    import ROOT
    print 'ROOT was successfully imported into mass.'
except ImportError:
    print 'ROOT was not found.'



class MicrocalFile(object):
    """
    Encapsulate a set of data containing multiple triggered traces from
    a microcalorimeter.  The pulses can be noise or X-rays.  This is meant 
    to be an abstract class.  Use files.LJHFile(), which is currently the only 
    derived class. In the future, other derived classes could implement 
    read_segment, copy, and read_trace to process other file types."""
    
    def __init__(self):
        """"""
        # These assertions ensure that we have the proper interface, which
        # must be found in derived classes.   Roughly speaking, these make the
        # current class an abstract base class
        required_methods = ("read_segment", "read_trace", "copy")
        for req in required_methods:
            try:
                self.__getattribute__(req)
            except AttributeError:
                print self.__dict__
                raise RuntimeError("A %s.%s object requires a method %s()"%(__name__, self.__class__.__name__, req))
        
        ## Filename of the data file
        self.filename = None
        self.nSamples = 0
        self.nPresamples = 0
        self.timebase = 0.0
        self.n_segments = 0


    def __str__(self):
        """Summary for the print function"""
        return "%s path '%s'\n%d samples (%d pretrigger) at %.2f microsecond sample time"%(
                self.__class__.__name__, self.filename, self.nSamples, self.nPresamples, 
                1e6*self.timebase)
        
    def __repr__(self):
        """Compact representation of how to construct from a filename."""
        return "%s('%s')"%(self.__class__.__name__, self.filename)


    def iter_segments(self, first=0, end=-1):
        """An iterator over all segments.  Read in segments one at a time and yield triplet:
        (first pulse number, 1+last pulse number, the segment number just read, the data).
        
        <first> The first segment to read.
        <end>   One past the last segment to read, or -1 to read through the last segment."""
        
        if end <= first:
            end = self.n_segments
        for segnum in range(first, end):
            first_pnum, end_pnum, data = self.read_segment(segnum)
#            first_pnum = segnum * self.pulses_per_seg
#            end_pnum = self.datatimes.shape[0] + first_pnum
            yield first_pnum, end_pnum, segnum, data



class VirtualFile(MicrocalFile):
    """
    Object to act like a single microcalorimeter data file on disk, though the data are all
    held only in memory.
    """
    def __init__(self, data, times=None, presamples=None):
        self.data = numpy.asarray(data, dtype=numpy.int16)
        self.nSamples = data.shape[1]
        self.nPulses = data.shape[0]
        self.nPresamples = presamples
        self.filename = "virtual file"
        self.n_segments = 1
        self.pulses_per_seg = self.nPulses
        self.timestamp_offset = 0
        self.timebase = 0.0
        
        if times is None:
            self.datatimes_float = numpy.zeros(self.nPulses, dtype=numpy.float)
        else:
            self.datatimes_float = numpy.asarray(times, dtype=numpy.float)
            
        if presamples is None:
            self.nPresamples = 0
        
            
    
    def copy(self):
        """Return a copy of the object.  Handy for updating method definitions."""
        c = VirtualFile(self.data)
        c.__dict__.update( self.__dict__ )
        return c
    
    def read_trace(self, pulsenum):
        """Return the data for pulse number <pulsenum>"""
        if pulsenum > self.nPulses:
            raise ValueError("This VirtualFile has only %d pulses"% self.nPulses)
        return self.data[pulsenum]
    
    def read_segment(self, segnum=0):
        """Return <first>,<end>,<data> for segment number <segnum>, where
        <first> is the first pulse number in that segment, <end>-1 is the last,
        and <data> is a 2-d array of shape [pulses_this_segment, self.nSamples]."""
        if segnum>0:
            raise ValueError("VirtualFile objects have only one segment")
        return 0, self.nPulses, self.data


    
class LJHFile(MicrocalFile):
    """Process a single LJH-format file.  All non-LJH-specific data and methods
    appear in the parent pulseRecords class"""
    
    def __init__(self, filename, segmentsize=(2**23)):
        """Open an LJH file for reading.  Read its header.  Set the standard segment
        size **in bytes** so that read_segment() will always return segments of a
        fixed size.
        <filename>   Path to the file to be read.
        <segmentsize>  Size of each segment **in bytes** that will be returned in read_segment()
                     The actual segmentsize will be rounded down to be an integer number of 
                     pulses.
        """
        super(LJHFile, self).__init__()
        self.filename = filename
        self.__cached_segment = None
        self.__read_header(filename)
        self.set_segment_size(segmentsize)
        self.header_lines = []
        self.sample_usec = None
        self.timestmp_offset = 0.0
        self.pulses_per_seg = 0
        self.segmentsize = 0
        self.n_segments = 0
        self.segment_pulses = 0
        self.header_size = 0
        self.pulse_size_bytes = 0
        self.data = None


    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions.
        """
        c = LJHFile(self.filename, self.segmentsize)
        c.__dict__.update( self.__dict__ )
        c.__cached_segment = None
        return c


    def __read_header(self, filename):
        """
        Read in the text header of an LJH file.
        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples
        
        <filename>: path to the file to be opened.
        """
        TOO_LONG_HEADER = 100 # headers can't contain this many lines, or they are insane!
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
            elif line.startswith("Timestamp offset (s)"):
                words = line.split()
                try:
                    self.timestamp_offset = float(words[-1])
                except:
                    self.timestamp_offset = 0.0
            
            if len(lines) > TOO_LONG_HEADER:   
                raise IOError("header is too long--seems not to contain '#End of Header'")
            
        self.header_lines = lines
        self.header_size = fp.tell()
        fp.seek(0, os.SEEK_END)
        self.binary_size = fp.tell() - self.header_size
        fp.close()
        
        self.nPulses = self.binary_size / (6+2*self.nSamples)
        
        # Check for major problems in the LJH file:
        if self.timebase is None:
            raise IOError("No 'Timebase' line found in header")
        if self.nSamples is None:
            raise IOError("No 'Total Samples' line found in header")
        if self.nPresamples is None:
            raise IOError("No 'Presamples' line found in header")
        
        # I could imagine making this one non-fatal, but for now make it fatal:
        if self.nPulses * (6+2*self.nSamples) != self.binary_size:
#            pass
            raise IOError("The binary size of the file (%d) is not an integer multiple of the pulse size %d"
                          %(self.binary_size, 6+2*self.nSamples))

        # Record the sample times in microseconds
        self.sample_usec = (numpy.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    
    def set_segment_size(self, segmentsize):
        """Set the standard segmentsize used in the read_segment() method.  This number will
        be rounded down to equal an integer number of pulses.
        
        Raises ValueError if segmentsize is smaller than a single pulse."""
        self.pulse_size_bytes = 6 + 2*self.nSamples
        maxitems = segmentsize/self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than the pulse record (%d bytes)"
                             %(segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems*self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize / self.pulse_size_bytes
        self.n_segments = 1+(self.binary_size-1)/self.segmentsize
        self.__cached_segment = None

        
    def read_trace(self, trace_num):
        """Return a single data trace (number <trace_num>),
        either from cache or by reading off disk, if needed."""
        segnum = trace_num / self.pulses_per_seg
        self.read_segment(segnum)
        return self.data[trace_num % self.pulses_per_seg]
        
        
    def read_segment(self, segment_num=0):
        """Read a section of the binary data of the given number (0,1,...) and size.
        It is okay to call this out of order.  The last segment might be shorter than others.
        
        Raises ValueError if there is no such section number.

        Return (first, end, data) where first is the pulse number of the first pulse read,
        end is 1+the number of the last one read, and data is the full array.        

        Params:
        -------
        <segment_num> Number of the segment to read.
        """
        # Use cached data, if possible
        if segment_num != self.__cached_segment: 
            if segment_num*self.segmentsize > self.binary_size:
                raise ValueError("File %s has only %d segments;\n\tcannot open segment %d"%
                                 (self.filename, self.n_segments, segment_num))
                
            self.__read_binary(self.header_size + segment_num*self.segmentsize, self.segmentsize, 
                               error_on_partial_pulse=True)
            self.__cached_segment = segment_num
        first = segment_num * self.pulses_per_seg
        end = first + self.data.shape[0]
        return first, end, self.data
        
        
    def __read_binary(self, skip=0, max_size=(2**26), error_on_partial_pulse=True):
        """Read the binary section of an LJH file, interpret it, and store the results in
        self.data and self.datatimes_float.  This can potentially be less than the full file
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
        if skip > 0: fp.seek(skip)
        
        if max_size >= 0:
            maxitems = max_size/self.pulse_size_bytes
            BYTES_PER_WORD = 2
            wordcount = maxitems*self.pulse_size_bytes/BYTES_PER_WORD
            if error_on_partial_pulse and wordcount*BYTES_PER_WORD != max_size:
                raise ValueError("__read_binary(max_size=%d) requests a non-integer number of pulses"%max_size)
        else:
            wordcount = -1

        array = numpy.fromfile(fp, dtype=numpy.uint16, sep="", count=wordcount)
        fp.close()
        
        self.segment_pulses = len(array)/(self.pulse_size_bytes/2)
        self.data = array.reshape([self.segment_pulses, self.pulse_size_bytes/2])
        
        # Time format is ugly.  From bytes 0-5 of a pulse, the bytes are uxmmmm,
        # where u is a byte giving microseconds/4, x is a reserved byte, and mmmm is a 4-byte
        # little-ending giving milliseconds.  The uu should always be in [0,999]
        # The uu was added on Sept 21, 2011, so it will be 0 on all earlier data files.
 
        ##### The old way was to store the time as a 32-bit int.  New way: double float
#        # Careful: converting 2 little-endian 16-bit words to a single 32-bit word is tricky!
#        self.datatimes = numpy.array(self.data[:,2], dtype=numpy.uint32) * (1<<16) 
#        self.datatimes += (self.data[:,1])

        # Store times as seconds in floating point.  Max value is 2^32 ms = 4.3x10^6
        SECONDS_PER_4MICROSECOND_TICK = (4.0/1e6)
        SECONDS_PER_MILLISECOND = 1e-3
        self.datatimes_float = numpy.array(self.data[:, 0], dtype=numpy.double)*SECONDS_PER_4MICROSECOND_TICK
        self.datatimes_float += (self.data[:, 1] + 65536*self.data[:, 2])*SECONDS_PER_MILLISECOND

        self.data = self.data[:, 3:] # cut out the zeros and the timestamp, which are 3 uint16 words at the start of each pulse


        
class LANLFile(MicrocalFile):
    """Process a LANL ROOT file using pyROOT. """
    
    def __init__(self, filename, segmentsize=(2**24), use_noise=False):
        """Open a LANL file for reading.  Read its header.  
        <filename>   Path to the file to be read.
        """
        
        if ROOT is None:
            raise ImportError("The PyRoot library 'ROOT' could not be imported.  Check your PYTHONPATH?")
        
        super(LANLFile, self).__init__()
        self.filename = filename
        self.__cached_segment = None
        self.root_file_object = ROOT.TFile(self.filename) #@UndefinedVariable
        self.use_noise = use_noise
        if self.use_noise:
            tree_name = "ucal_noise"
        else:
            tree_name = 'ucal_data'
        self.ucal_tree = self.root_file_object.Get(tree_name) #Get the ROOT tree structure that has the data
        
#        The header file in the LANL format is in a separate ROOT files that is the same for all the channels.
#        It does not have the _chxxx designation. I will take the path passed to the class and use splitline
#        to strip of the _chxxx part.

        # Strip off the extension
        filename_array =  filename.split('.')
        if filename_array[-1] != 'root':
            raise IOError("File does not have .root extension")
        filename_noextension = filename_array[0]

        # Strip off the channel number
        separator = '_'
        self.header_filename = separator.join(filename_noextension.split(separator)[:-1])+'.root'
        
        # If this header files exists, assume it's of the form that the gamma group makes
        if os.path.isfile(self.header_filename):
            self.gamma_vector_style = True
            self.root_header_file_object = ROOT.TFile(self.header_filename) #@UndefinedVariable
            
        # If not header, assume it's the alpha group's preferred form
        else:
            self.gamma_vector_style = False
            self.header_filename = self.filename
            self.root_header_file_object = self.root_file_object

        self.__prepare_root_memory()
        self.__read_header()
        self.set_segment_size(segmentsize)
        self.raw_datatimes = numpy.zeros(self.nPulses, dtype=numpy.uint32)


    def __prepare_root_memory(self):
        """It is silly to have to create these numpy objects and then tell ROOT to
        store data into them over and over, so we move them from the read_trace method
        to here, where they can be done once and forgotten"""
        
        #Pulses are stored in vector ROOT format in the 'pulse' branch
        if self.gamma_vector_style:
            self.pdata = ROOT.std.vector(int)() # this is how gamma people do it #@UndefinedVariable
        else:
            self.pdata = ROOT.TH1I() # this is how alpha people do it #@UndefinedVariable
        self.channel = numpy.zeros(1, dtype=int)
        self.baseline = numpy.zeros(1, dtype=float)
        self.baseline_rms = numpy.zeros(1, dtype=float)
        
        self.timestamp = numpy.zeros(1, dtype=int)
        self.pulse_max = numpy.zeros(1, dtype=int)
        self.pulse_max_pos = numpy.zeros(1, dtype=int)
        self.pulse_integral = numpy.zeros(1, dtype=int)
        self.flag_pileup = numpy.zeros(1, dtype=int)
        
        # pdata is updated when the the GetEntry method to the current trace number is called
        self.ucal_tree.SetBranchAddress("baseline",self.baseline)
        self.ucal_tree.SetBranchAddress("baseline_rms",self.baseline_rms)
        self.ucal_tree.SetBranchAddress("channel",self.channel)
        if self.use_noise:
            self.ucal_tree.SetBranchAddress('noise',ROOT.AddressOf(self.pdata))  #@UndefinedVariable
        else:
            self.ucal_tree.SetBranchAddress('pulse',ROOT.AddressOf(self.pdata))  #@UndefinedVariable
            self.ucal_tree.SetBranchAddress("timestamp",self.timestamp)
            self.ucal_tree.SetBranchAddress("max",self.pulse_max)
            self.ucal_tree.SetBranchAddress("max_pos",self.pulse_max_pos)
            self.ucal_tree.SetBranchAddress("integral",self.pulse_integral)
            self.ucal_tree.SetBranchAddress("flag_pileup",self.flag_pileup)

        # The read caching seems to make no difference whatsoever, but here it is...
        # See http://root.cern.ch/drupal/content/spin-little-disk-spin for more.
        self.ucal_tree.SetCacheSize(2**23)
        self.ucal_tree.AddBranchToCache("*")
        self.ucal_tree.SetCacheLearnEntries(1)



    def copy(self):
        """Return a copy of the object.
        
        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        c = LANLFile(self.filename, self.segmentsize)
        c.__dict__.update( self.__dict__ )
        c.__cached_segment = None
        return c


    def __read_header(self):
        """
        Read the separate ROOT file that contains the header information as branches.
        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples
        
        <filename>: path to the file to be opened.
        """
        
        # Keys include record_length, pretrig_length, basetime, dac_offset, yscale, sample_rate, clocks_per_sec, npts_to_average, start_time, end_time
        find_root_quantity = lambda name : self.root_header_file_object.Get(name).GetVal()
        
        self.nSamples = find_root_quantity("record_length")
        self.nPresamples = find_root_quantity("pretrig_length")
        if self.gamma_vector_style:
            self.timebase = find_root_quantity("basetime")
            self.timestamp_msec_per_step = 1.0e-3
        else:
            # Use "sample_rate" (which is in MHz) and "npts_to_average", which is a decimation ratio
            self.timebase = find_root_quantity("npts_to_average")*1e-6/find_root_quantity("sample_rate")
            self.timestamp_msec_per_step = 1.0
            
        
        # This information is not in the root header file but is in the channel files
        self.get_nPulses() #self.nPulses now has the number of pulses
        
        # Check for major problems in the header:
        if self.timebase is None:
            raise IOError("No 'Timebase' line found in header")
        if self.nSamples is None:
            raise IOError("No 'Total Samples' line found in header")
        if self.nPresamples is None:
            raise IOError("No 'Presamples' line found in header")
        
        # Record the sample times in microseconds
        self.sample_usec = (numpy.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    def get_nPulses(self):
        """Get the numner of pulses in the current ROOT file."""
        self.nPulses = int(self.ucal_tree.GetEntries())


    def set_segment_size(self, segmentsize):
        """Set the standard segmentsize used in the read_segment() method.  This number will
        be rounded down to equal an integer number of pulses.
        
        Raises ValueError if segmentsize is smaller than a single pulse."""
        self.pulse_size_bytes = 2*self.nSamples
        maxitems = segmentsize/self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than the pulse record (%d bytes)"
                             %(segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems*self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize / self.pulse_size_bytes
        self.n_segments = 1+(self.nPulses-1)/maxitems
        self.__cached_segment = None
        
        
    def read_trace(self, trace_num):
        """Return a single data trace (number <trace_num>)."""
        
        self.ucal_tree.GetEntry(trace_num)
#        pulse = numpy.asarray(self.pdata)
#        pulse = numpy.array(self.pdata)
        if self.gamma_vector_style:
            iterator = self.pdata.begin()
        else:
            iterator = self.pdata.GetArray()
        pulse = numpy.fromiter(iterator, dtype=numpy.uint16, count=self.nSamples)
        self.raw_datatimes[trace_num] = self.timestamp[0]*self.timestamp_msec_per_step
        
#        return pulse,channel[0],timestamp[0],pulse_max[0],pulse_max_pos[0],pulse_integral[0],baseline[0],baseline_rms[0],flag_pileup[0], pdata
        return pulse

    
    def read_segment(self, segment_num=0):
        """Read a section of the binary data of the given number (0,1,...) and size.
        It is okay to call this out of order.  The last segment might be shorter than others.
        
        Raises ValueError if there is no such section number.

        Return (first, end, data) where first is the pulse number of the first pulse read,
        end is 1+the number of the last one read, and data is the full array.        

        Params:
        -------
        <segment_num> Number of the segment to read.
        """
        first = segment_num * self.pulses_per_seg
        end = first + self.pulses_per_seg

        # Use cached data, if possible
        if segment_num != self.__cached_segment: 
            if segment_num > self.n_segments:
                raise ValueError("File %s has only %d segments;\n\tCannot open segment %d"%
                                 (self.filename, self.n_segments, segment_num))
                
            if end > self.nPulses: end = self.nPulses
            print "Reading pulses [%d,%d)"%(first,end)
            self.data = numpy.array([self.read_trace(i) for i in range(first,end)])
            self.datatimes = self.raw_datatimes[first:end]
            self.__cached_segment = segment_num
        return first, end, self.data
        
        

def root2ljh_translator(rootfile, ljhfile=None, overwrite=False, segmentsize=5000000, 
                        channum=None, use_noise=False, excise_endpoints=None):
    """
    Translate a single LANL ROOT file into a single LJH file.
    
    The ROOT reader in PyROOT is rather slow, whereas LJH data can be read efficiently in large segments.
    I believe this is because ROOT cannot assume that all events are homogeneous (of course, they are).
    The point of this translator is to let us read LANL data many times without paying this
    penalty each time.
    
    Parameters: 
    -------------
    ljhfile   -- The filename of the output file.  If not given, it will be chosen by replacing
                 a trailing ".root" with ".ljh" if possible (and will fail if not possible).
    overwrite -- If the output file exists and overwrite is not True, then translation fails.
    segmentsize -- The number of ROOT file bytes to read at one gulp.  Not likely that you care about this.
    channum     -- If not set to None, then write out only data with this channel number.
    use_noise   -- We want the output to grab the ucal_noise rather than the ucal_data tree.
    excise_endpoints -- Remove the first and last few samples from each trace, optionally.  If None,
                 remove nothing.  If a single number, remove that number from each end.
                 If a 2-element-sequence (a,b), then remove a from the start and b from the end.
    """
    
    print "Attempting to translate '%s' "%rootfile,
    lanl = LANLFile(filename=rootfile, segmentsize=segmentsize, use_noise = use_noise)
    
    if isinstance(excise_endpoints, int):
        excise_endpoints = (excise_endpoints, excise_endpoints)
    if excise_endpoints[1] > 0: 
        excise_endpoints = tuple((excise_endpoints[0], -excise_endpoints[1]))
    
    if ljhfile is None:
        if not rootfile.endswith(".root"):
            raise ValueError("ljhfile argument must be supplied if rootfile name doesn't end with '.root'.")
        ljhfile = rootfile.rstrip("root")+"ljh"
        
    if os.path.exists(ljhfile) and not overwrite:
        raise IOError("The ljhfile '%s' exists and overwrite was not set to True"%ljhfile)
    
    lanl.asctime = time.asctime(time.gmtime())
    header_dict = lanl.__dict__.copy()
    header_dict['nPresamples'] -= excise_endpoints[0]
    header_dict['nSamples'] -= excise_endpoints[0]+abs(excise_endpoints[1])
    ljh_header = """#LJH Memorial File Format
Save File Format Version: 2.0.0
Software Version: Fake LJH file converted from ROOT
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
Bits: 14
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
"""% header_dict

    ljh_fp = open(ljhfile, "wb")
    ljh_fp.write(ljh_header)
    
    import struct
    prefix_fmt = "<xxL"
    BINARYFILE = ""
    for i in range(lanl.nPulses):
        trace = lanl.read_trace(i)
        if excise_endpoints is not None:
            trace = trace[excise_endpoints[0]:excise_endpoints[1]]
        if channum is not None and lanl.channel[0] != channum:
            continue
        prefix = struct.pack(prefix_fmt, int(lanl.timestamp[0]))
        ljh_fp.write(prefix)
        trace.tofile(ljh_fp, sep=BINARYFILE)
    
    ljh_fp.close()



def root2ljh_translate_all(directory):
    """Use root2ljh_translator for all files in <directory>"""
    
    for f in glob.glob("%s/*.root"%directory):
        try:
            root2ljh_translator(f, overwrite=False)
        except IOError:
            print "Could not translate '%s' .  Moving on..."% f
            