"""
channel_group.py

Part of the Microcalorimeter Analysis Software System (MASS).

This module defines classes that handle one or more TES data streams 
together.  While these classes are indispensable for code-
division multiplexed (CDM) systems, they are also useful for the
simpler time-division multiplexed (TDM) systems in that they allow
the same interface to handle both types of data.

That's the goal, at least.

Author: Joe Fowler, NIST

Started March 2, 2011
"""

import numpy
#import pylab

import mass.channel



class BaseChannelGroup(object):
    """
    Provides the interface for a group of one or more microcalorimeters,
    whether the detectors are multiplexed with time division or code
    division.
    
    This is an abstract base class, and the appropriate concrete class
    is the TESGroup or the CDMGroup, depending on the multiplexing scheme. 
    """
    def __init__(self, filenames):
        # Convert a single filename to a tuple of size one
        if isinstance(filenames, str):
            filenames = (filenames,)
        self.filenames = tuple(filenames)
        self.n_channels = len(self.filenames)



class TESGroup(BaseChannelGroup):
    """
    A group of one or more *independent* microcalorimeters, in that
    they are time-division multiplexed.  It might be convenient to use
    this for multiple TDM channels, or for singles.  The key is that
    this object offers the same interface as the CDMGroup object
    (which has to be more complex under the hood).
    """
    def __init__(self, filenames):

        super(self.__class__, self).__init__(filenames)
        
        channel_list = []
        for fname in self.filenames:
            chan = mass.channel.create_MicrocalDataSet(fname)
            channel_list.append(chan)
        self.channels = tuple(channel_list)
        
        
    def read_segment(self, segnum):
        for chan in self.channels:
            chan.read_segment(segnum)



class CDMGroup(BaseChannelGroup):
    """
    A group of *CDM-coupled* microcalorimeters, in that they are code-division
    multiplexing a set of calorimeters into a set of output data streams.The key
    is that this object offers the same interface as the TESGroup object (which
    is rather less complex under the hood than this object).
    """
    
    def __init__(self, filenames, walsh=None):
        
        super(self.__class__, self).__init__(filenames)

        if walsh is None:
            walsh = numpy.array(
                (( 1, 1, 1, 1),
                 (-1, 1, 1,-1),
                 (-1,-1, 1, 1),
                 (-1, 1,-1, 1)), dtype=numpy.int16)
        assert walsh.shape[0] == walsh.shape[1]
        assert walsh.shape[0] == self.n_channels
        self.walsh = walsh

        channel_list = []
        demod_list = []
        for fname in self.filenames:
            chan = mass.channel.create_MicrocalDataSet(fname)
            channel_list.append(chan)
            
            demod = mass.channel.MicrocalDataSetCDM()
            demod_list.append(demod)
        self.raw_channels = tuple(channel_list)
        self.demod_channels= tuple(demod_list)
        

    def read_segment(self, segnum):
        for chan in self.raw_channels:
            chan.read_segment(segnum)

        # Remove linear drift
        shape = self.raw_channels[0].data.shape
        mod_data = numpy.zeros([self.n_channels]+list(shape), dtype=numpy.int32)
        mod_data[0, :, :] = self.raw_channels[0].data
        for i in range(1, self.n_channels):
            mod_data[i, :, 1:] = i*self.raw_channels[i].data[:,1:]
            mod_data[i, :, 1:] += (self.n_channels-i) *self.raw_channels[i].data[:,:-1]
            mod_data[i, :, 0] = self.n_channels*self.raw_channels[i].data[:,0]
            mod_data[i, :, :] /= self.n_channels
        print mod_data.shape
        
        # Demodulate
        for chan in self.demod_channels:
            chan.data = numpy.zeros(shape, dtype=numpy.int32)
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                self.demod_channels[i].data += self.walsh[i,j]*mod_data[j, :, :]



