"""Functions and classes that were once used to process files written in the
LANL ROOT format, but no longer used. Moved to deprecated_root.py in
February 2017.

The utilities root2ljh_translator and root2ljh_translate_all were used to convert
a ROOT file (or all such in one directory) into LJH format for more efficient
future use within mass.

LANLFile and translation added June 2011 by Doug Bennett and Joe Fowler
"""

class LANLFile(MicrocalFile):
    """Process a LANL ROOT file using pyROOT. """

    def __init__(self, filename, segmentsize=(2**24), use_noise=False):
        """Open a LANL file for reading.  Read its header.
        <filename>   Path to the file to be read.
        """

        if ROOT is None:
            raise ImportError("The PyRoot library 'ROOT' could not be imported.  Check your PYTHONPATH?")

        super(LANLFile, self).__init__(self)
        self.filename = filename
        self.__cached_segment = None
        self.root_file_object = ROOT.TFile(self.filename)  # @UndefinedVariable
        self.use_noise = use_noise
        if self.use_noise:
            tree_name = "ucal_noise"
        else:
            tree_name = 'ucal_data'
        self.ucal_tree = self.root_file_object.Get(tree_name)  # Get the ROOT tree structure that has the data

#        The header file in the LANL format is in a separate ROOT files that is the same for all the channels.
#        It does not have the _chxxx designation. I will take the path passed to the class and use splitline
#        to strip of the _chxxx part.

        # Strip off the extension
        filename_array = filename.split('.')
        if filename_array[-1] != 'root':
            raise IOError("File does not have .root extension")
        filename_noextension = filename_array[0]

        # Strip off the channel number
        separator = '_'
        self.header_filename = separator.join(filename_noextension.split(separator)[:-1])+'.root'

        # If this header files exists, assume it's of the form that the gamma group makes
        if os.path.isfile(self.header_filename):
            self.gamma_vector_style = True
            self.root_header_file_object = ROOT.TFile(self.header_filename)  # @UndefinedVariable

        # If not header, assume it's the alpha group's preferred form
        else:
            self.gamma_vector_style = False
            self.header_filename = self.filename
            self.root_header_file_object = self.root_file_object

        self._setup()
        self.__read_header()
        self.set_segment_size(segmentsize)
        self.raw_datatimes = np.zeros(self.nPulses, dtype=np.uint32)

    def _setup(self):
        """It is silly to have to create these np objects and then tell ROOT to
        store data into them over and over, so we move them from the read_trace method
        to here, where they can be done once and forgotten"""

        # Pulses are stored in vector ROOT format in the 'pulse' branch
        if self.gamma_vector_style:
            self.pdata = ROOT.std.vector(int)()  # this is how gamma people do it #@UndefinedVariable
        else:
            self.pdata = ROOT.TH1D()  # this is how alpha people do it   #@UndefinedVariable -RDH
        self.channel = np.zeros(1, dtype=int)
        self.baseline = np.zeros(1, dtype=np.double)  # RDH
        self.baseline_rms = np.zeros(1, dtype=np.double)  # RDH

        self.timestamp = np.zeros(1, dtype=np.double)  # RDH
        self.pulse_max = np.zeros(1, dtype=np.double)  # RDH
        self.pulse_max_pos = np.zeros(1, dtype=int)
        self.pulse_integral = np.zeros(1, dtype=np.double)  # RDH
        self.flag_pileup = np.zeros(1, dtype=int)

        # pdata is updated when the the GetEntry method to the current trace number is called
        self.ucal_tree.SetBranchAddress("baseline", self.baseline)
        self.ucal_tree.SetBranchAddress("baseline_rms", self.baseline_rms)
        self.ucal_tree.SetBranchAddress("channel", self.channel)
        if self.use_noise:
            self.ucal_tree.SetBranchAddress('noise', ROOT.AddressOf(self.pdata))  # @UndefinedVariable
        else:
            self.ucal_tree.SetBranchAddress('pulse', ROOT.AddressOf(self.pdata))  # @UndefinedVariable
            self.ucal_tree.SetBranchAddress("timestamp", self.timestamp)
            self.ucal_tree.SetBranchAddress("max", self.pulse_max)
            self.ucal_tree.SetBranchAddress("max_pos", self.pulse_max_pos)
            self.ucal_tree.SetBranchAddress("integral", self.pulse_integral)
            self.ucal_tree.SetBranchAddress("flag_pileup", self.flag_pileup)

        # The read caching seems to make no difference whatsoever, but here it is...
        # See http://root.cern.ch/drupal/content/spin-little-disk-spin for more.
        self.ucal_tree.SetCacheSize(2**23)
        self.ucal_tree.AddBranchToCache("*")
        self.ucal_tree.SetCacheLearnEntries(1)

    def copy(self):
        """Return a copy of the object.

        Handy when coding and you don't want to read the whole data set back in, but
        you do want to update the method definitions."""
        self.clear_cache()
        new_rootfile = LANLFile(self.filename, self.segmentsize)
        new_rootfile.__dict__.update(self.__dict__)
        return new_rootfile

    def __read_header(self):
        """
        Read the separate ROOT file that contains the header information as branches.
        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples

        <filename>: path to the file to be opened.
        """

        # Keys include record_length, pretrig_length, basetime, dac_offset, yscale,
        # sample_rate, clocks_per_sec, npts_to_average, start_time, end_time
        find_root_quantity = lambda name : self.root_header_file_object.Get(name).GetVal()

        self.nSamples = find_root_quantity("record_length")
        self.nPresamples = find_root_quantity("pretrig_length")
        if self.gamma_vector_style:
            self.timebase = find_root_quantity("basetime")
            self.timestamp_msec_per_step = 1.0e-3
        else:
            # Use "sample_rate" (which is in MHz) and "npts_to_average", which is a decimation ratio
            # self.timebase = find_root_quantity("npts_to_average")*1e-6/find_root_quantity("sample_rate")
            # RDH - The new alpha data uses "sample_rate" in MHz
            self.timebase = 1.0/find_root_quantity("sample_rate")  # RDH
            self.timestamp_msec_per_step = 1.0

        # This information is not in the root header file but is in the channel files
        self.get_npulses()  # self.nPulses now has the number of pulses

        # Check for major problems in the header:
        if self.timebase is None:
            raise IOError("No 'Timebase' line found in header")
        if self.nSamples is None:
            raise IOError("No 'Total Samples' line found in header")
        if self.nPresamples is None:
            raise IOError("No 'Presamples' line found in header")

        # Record the sample times in microseconds
        self.sample_usec = (np.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    def get_npulses(self):
        """Get the numner of pulses in the current ROOT file."""
        self.nPulses = int(self.ucal_tree.GetEntries())

    def set_segment_size(self, segmentsize):
        """Set the standard segmentsize used in the read_segment() method.  This number will
        be rounded down to equal an integer number of pulses.

        Raises ValueError if segmentsize is smaller than a single pulse."""
        self.pulse_size_bytes = 2*self.nSamples
        maxitems = segmentsize/self.pulse_size_bytes
        if maxitems < 1:
            raise ValueError("segmentsize=%d is not permitted to be smaller than the pulse record (%d bytes)" %
                             (segmentsize, self.pulse_size_bytes))
        self.segmentsize = maxitems*self.pulse_size_bytes
        self.pulses_per_seg = self.segmentsize / self.pulse_size_bytes
        self.n_segments = 1 + (self.nPulses - 1) // maxitems
        self.__cached_segment = None

    def read_trace(self, trace_num):
        """Return a single data trace (number <trace_num>)."""

        self.ucal_tree.GetEntry(trace_num)
#        pulse = np.asarray(self.pdata)
#        pulse = np.array(self.pdata)
        if self.gamma_vector_style:
            iterator = self.pdata.begin()
        else:
            iterator = self.pdata.GetArray()

        # convert the double from LANL into an integer for IGOR
        pulsedouble = np.fromiter(iterator, dtype=np.double, count=self.nSamples)  # RDH

        pulse = np.int16(np.round(((pulsedouble + 2.0)/4.0)*2**16))  # RDH
        self.raw_datatimes[trace_num] = self.timestamp[0]*self.timestamp_msec_per_step
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
                raise ValueError("File %s has only %d segments;\n\tCannot open segment %d" %
                                 (self.filename, self.n_segments, segment_num))

            if end > self.nPulses:
                end = self.nPulses
            print("Reading pulses [%d,%d)" % (first, end))
            self.data = np.array([self.read_trace(i) for i in range(first, end)])
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

    print("Attempting to translate '%s' " % rootfile),
    lanl = LANLFile(filename=rootfile, segmentsize=segmentsize, use_noise=use_noise)
    print("Looking at channel " + str(channum))  # RDH

    if isinstance(excise_endpoints, int):
        excise_endpoints = (excise_endpoints, excise_endpoints)
    if excise_endpoints is not None and excise_endpoints[1] > 0:
        excise_endpoints = tuple((excise_endpoints[0], excise_endpoints[1]))

    if ljhfile is None:
        if not rootfile.endswith(".root"):
            raise ValueError("ljhfile argument must be supplied if rootfile name doesn't end with '.root'.")
        if use_noise:
            ljhfile = rootfile.rstrip("root")+"noi"
        else:
            ljhfile = rootfile.rstrip("root")+"ljh"

    if os.path.exists(ljhfile) and not overwrite:
        raise IOError("The ljhfile '%s' exists and overwrite was not set to True" % ljhfile)

    lanl.asctime = time.asctime(time.gmtime())
    header_dict = lanl.__dict__.copy()
    header_dict['version_str'] = '2.1.0'
    header_dict['nPresamples'] -= excise_endpoints[0]
    header_dict['nSamples'] -= excise_endpoints[0]+abs(excise_endpoints[1])

    ljh_header = make_ljh_header(header_dict)
    ljh_fp = open(ljhfile, "wb")
    ljh_fp.write(ljh_header)

    for i in range(lanl.nPulses):
        trace = lanl.read_trace(i)
        if excise_endpoints is not None:
            trace = trace[excise_endpoints[0]:len(trace)-excise_endpoints[1]]  # RDH accepts 0 as an argument
        if channum is not None and lanl.channel[0] != channum:
            continue
        prefix = struct.pack("<xxL", int(lanl.timestamp[0]))
        ljh_fp.write(prefix)
        trace.tofile(ljh_fp, sep="")

    ljh_fp.close()


def root2ljh_translate_all(directory):
    """Use root2ljh_translator for all files in <directory>"""

    for fname in glob.glob("%s/*.root" % directory):
        try:
            root2ljh_translator(fname, overwrite=False)
        except IOError:
            print("Could not translate '%s' .  Moving on..." % fname)
