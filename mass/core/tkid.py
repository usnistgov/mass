"""
TKID.py

This file contains classes that will work with the TKID data.

"""
import os
import h5py
import logging
import glob

import numpy as np
import matplotlib.pylab as plt
import scipy

try:
    from collections.abc import Iterable  # Python 3
except ImportError:
    from collections import Iterable

from functools import reduce

from mass.core.files import MATFile
from mass.core.channel import PulseRecords
from mass.core.ljh_util import remove_unpaired_channel_files,filename_glob_expand
from ..common import isstr
LOG = logging.getLogger("mass")


class TKIDDataSet(object):
    """ This is a class to work with TKID data records.


      """
    def __init__(self,filenames):
        """
        Args:
        """
        self.nSamples = 0
        self.nPresamples = 0
        self.nPulses = 0
        self.timebase = 0.0
        self.data = None
        self.filenames = None
        self.peaks = None
        self.data_tails = None
        # Handle the case that either filename list is a glob pattern (e.g.,
        # "files_chan*.ljh"). Note that this will return a list, never a string,
        # even if there is only one result from the pattern matching.
        pattern = filenames
        filenames = glob.glob(pattern)

        if filenames is None or len(filenames) == 0:
            raise ValueError("Pulse filename pattern '%s' expanded to no files" % pattern)

        if isstr(filenames):
            filenames = (filenames,)

        self.filenames = tuple(filenames)

    def grab_dict(self,filename):

        pulsefile = PulseRecords(filename)

        for attr in ("nSamples", "nPresamples", "nPulses", "timebase"):
            self.__dict__[attr] = pulsefile.__dict__[attr]

    def grab_data(self):

        for i,fname in enumerate(self.filenames):

            if i == 0:
                afile = MATFile(fname)
                self.data = afile.data
                self.grab_dict(fname)
            else:
                afile = MATFile(fname)
                self.data  = np.concatenate((self.data, afile.data),axis = 0)

        return self.data

    def pick_temperature(self,temp):

        tempstr = "Tbath"+str(temp)
        temp_filenames = ()
        for i, fname in enumerate(self.filenames):

            if tempstr in fname:
                temp_filenames = temp_filenames + (fname,)

        return temp_filenames

    def pick_ATNOP(self,power):

        powerstr = "ATNOP"+str(power)
        power_filenames = ()
        for i, fname in enumerate(self.filenames):

            if powerstr in fname:
                power_filenames = power_filenames + (fname,)

        return power_filenames

    def pick_ATN1(self,power):

        powerstr = "ATN1"+str(power)
        power_filenames = ()
        for i, fname in enumerate(self.filenames):

            if powerstr in fname:
                power_filenames = power_filenames + (fname,)

        return power_filenames


    def plot_traces(self, pulsenums, axis=None, subtract_baseline=False):
          """Plot some example pulses, given by sample number.

          Args:
            <pulsenums>   A sequence of sample numbers, or a single one.
            <axis>       A plt axis to plot on.
            <residual>   Whether to show the residual between data and opt filtered model, or just raw data.
            <subtract_baseline>  Whether to subtract pretrigger mean prior to plotting the pulse
          """



    def find_peak(self,pulse):
        ipeak = scipy.signal.find_peaks(pulse, distance = self.nSamples)
        return ipeak[0][0]

    def find_peak_set(self):
        peaks = []
        for apeak in self.data:
            ipeak = self.find_peak(apeak)
            peaks.append(ipeak)
        self.peaks = peaks
        return peaks

    def grab_tails(self):
        self.find_peak_set
        avstart = sum(self.peaks)/len(self.peaks)
        mystart = avstart + 10

        dummydata = self.data[:,int(mystart):int(self.nSamples)]
        self.data_tails = dummydata
        return dummydata
