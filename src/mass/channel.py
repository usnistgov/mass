"""
Created on Feb 16, 2011

@author: fowlerj
"""


class pulseRecords(object):
    """
    Encapsulate a set of data containing multiple triggered pulse traces.
    The pulses can be noise or X-rays.  This is meant to be an abstract
    class.  Use files.LJHFile(), which is currently the only derived class.
    In the future, other derived classes could implement __readHeader and 
    __readBinary to process other file types."""
    
    ( CUT_PRETRIG_MEAN,
      CUT_PRETRIG_RMS,
      CUT_RETRIGGER,
      CUT_BIAS_PULSE,
      CUT_RISETIME,
       ) = range(5)
    
    
    def __init__(self):
        self.nSamples = None
        self.nPresamples = None
        self.nPulses = 0
        self.dt = None
        self.timebase = None
        self.cuts = None
        self.timestamps = None
        self.maxPosttriggerDeriv = None
        self.riseTimes = None
        self.isBiasPulse = None
        self.hasBiasPulses = False
        self.bad = None
        self.good = None
        self.autocorrelation = None
        
        # These assertions ensure that we have the proper interface, which
        # must be found in derived classes
        required_methods=("_read_header","_read_segment")
        for rm in required_methods:
            try:
                self.__getattribute__(rm)
            except AttributeError:
                print self.__dict__
                raise RuntimeError("A %s.%s object requires a method %s()"%(__name__, self.__class__.__name__, rm))
