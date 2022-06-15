"""
TKID.py

This file contains classes that will work with the TKID data.

"""
import os
import h5py
import logging
import glob
import re

import numpy as np
import matplotlib.pylab as plt
import scipy
import lmfit
import pickle
from lmfit.models import ExponentialModel, LinearModel


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
        self.data_subtracted = None
        self.filenames = None
        self.filenames_list = None
        self.filename = None
        self.peaks = None
        self.data_tails = None
        self.data_noise = None
        self.data_tails_subtracted = None
        self.data_t = None
        self.noise_t = None
        self.background_slopes = None
        self.background_intercepts = None
        self.avpulse = None
        self.avtails = None
        self.avnoise = None

        self.avpulse_sub = None
        self.avtails_sub = None
        self.avnoise_sub = None

        self.double_fit_e1_decay = 0
        self.double_fit_e2_decay = 0
        self.double_fit_e1_amp = 0
        self.double_fit_e2_amp = 0

        self.avpeak = 0
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
        self.filenames_list = tuple(filenames)
        self.filename = tuple(filenames)

    def grab_dict(self,filename):

        pulsefile = PulseRecords(filename)

        for attr in ("nSamples", "nPresamples", "nPulses", "timebase"):
            self.__dict__[attr] = pulsefile.__dict__[attr]

    def grab_data(self):

        for i,fname in enumerate(self.filenames_list):

            if i == 0:
                afile = MATFile(fname)
                data = afile.data
                self.grab_dict(fname)
            else:
                afile = MATFile(fname)
                data  = np.concatenate((self.data, afile.data),axis = 0)
            self.data = data
            self.nPulses = self.data.shape[0]
            self.data_t = np.arange(self.data.shape[1])*4e-7
        return data

    def pick_temperature(self,temp):

        tempstr = "Tbath"+str(temp)
        temp_filenames = ()
        for i, fname in enumerate(self.filenames_list):

            if tempstr in fname:
                temp_filenames = temp_filenames + (fname,)

        self.filenames_list = temp_filenames
        self.filename = temp_filenames
        return temp_filenames

    def grab_temperature(self,file):

        txt = os.path.splitext(file)
        x = re.search("(?<=Tbath)\d{0,5}\.\d{0,3}", txt[0])
        temp = float(x.group(0))
        return temp

    def pick_ATNOPT(self,power):

        powerstr = "ATNOPT_"+str(power)
        power_filenames = ()
        for i, fname in enumerate(self.filenames_list):

            if powerstr in fname:
                power_filenames = power_filenames + (fname,)
        self.filenames_list = power_filenames
        self.filename = power_filenames
        return power_filenames

    def grab_ATNOPT(self,file):

        txt = os.path.splitext(file)
        x = re.search("(?<=ATNOPT_)\d{0,5}", txt[0])
        ATNOPT = float(x.group(0))
        return ATNOPT


    def pick_ATN1(self,power):

        powerstr = "ATN1_"+str(power)
        power_filenames = ()
        for i, fname in enumerate(self.filenames_list):

            if powerstr in fname:
                power_filenames = power_filenames + (fname,)
        self.filenames_list = power_filenames
        self.filename = power_filenames
        return power_filenames

    def grab_ATN1(self,file):

        txt = os.path.splitext(file)
        x = re.search("(?<=ATN1_)\d{0,5}", txt[0])
        ATN1 = float(x.group(0))
        return ATN1


    def pick_file(self,filename):

        filestr = filename
        new_filenames = ()
        for i, fname in enumerate(self.filenames_list):

            if filestr in fname:
                new_filenames = new_filenames + (fname,)
        self.filename = new_filenames
        return new_filenames

    def reset_filename(self):
        allnames = self.filenames_list
        self.filename = allnames

    def reset_filenames_list(self):
        allnames = self.filenames
        self.filenames_list = allnames

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
        self.find_peak_set()
        avstart = sum(self.peaks)/len(self.peaks)
        mystart = avstart + 50
        myend = mystart + 1000

        dummydata = self.data[:,int(mystart):int(myend)]
        self.data_tails = dummydata
        self.avpeak = avstart

        return dummydata

    def grab_noise(self):
        noiseend = self.avpeak - 50
        dataend = self.data.shape[1]
        noisestart = dataend - 1000

        prenoise = self.data[:,0:int(noiseend)]
        pret = np.arange(int(noiseend))*4e-7
        postnoise = self.data[:,int(noisestart):int(dataend)]
        postt = np.arange(int(noisestart),int(dataend))*4e-7

        dummynoise  = np.concatenate((prenoise, postnoise),axis = 1)
        myt = np.concatenate((pret,postt),axis = 0)
        self.data_noise = dummynoise
        self.noise_t = myt
        return dummynoise, myt

    def average_set(self, data):
        myav = np.zeros_like(data[0])
        for pulse in data:
            myav = np.add(myav,pulse)
        numPulse = self.nPulses
        myav = myav/numPulse
        #myav = myav/self.nPulses
        return myav

    def average_pulse(self):
        avpulse = self.average_set(self.data)
        self.avpulse = avpulse
        avtails = self.average_set(self.data_tails)
        self.avtails = avtails
        avnoise = self.average_set(self.data_noise)
        self.avnoise = avnoise
        avpulse_sub = self.average_set(self.data_subtracted)
        self.avpulse_sub = avpulse_sub
        avtails_sub = self.average_set(self.data_tails_subtracted)
        self.avtails_sub = avtails_sub

        return avpulse_sub

    def fit_background_single(self,noisedata):
        myt = np.arange(noisedata.shape[0])*4e-7
        lin = LinearModel()
        mymodel =lin

        pars = mymodel.make_params(slope = 1e-8)
        out = mymodel.fit(noisedata, pars, x=myt)

        #print(out.fit_report())
        #plt.plot(myt,avnoise)
        #plt.plot(myt,out.best_fit)
        background_slope = out.best_values['slope']
        background_intercept = out.best_values['intercept']
        return background_slope,background_intercept

    def fit_background_all(self):
        slopes = np.zeros((self.nPulses))
        intercepts = np.zeros((self.nPulses))
        for i,pulsenoise in enumerate(self.data_noise):
            [slopes[i], intercepts[i]] = self.fit_background_single(pulsenoise)

        self.background_slopes = slopes
        self.background_intercepts = intercepts

    def subtract_baseline_single(self,dataset):
        myt = np.arange(dataset.shape[1])*4e-7
        background_points = np.zeros_like(dataset)
        subtracted_data = np.copy(dataset)
        for i,pulse in enumerate(dataset):
            background_points[i] = self.background_slopes[i] * myt + self.background_intercepts[i]
            subtracted_data[i] = subtracted_data[i] - background_points[i]

        return subtracted_data

    def subtract_baseline_all(self):
        self.data_subtracted = self.subtract_baseline_single(self.data)
        self.data_tails_subtracted = self.subtract_baseline_single(self.data_tails)

    def fit_single_decay(self, plot = False, scaled = False):
        exp1 = ExponentialModel(prefix='e1_')
        mymodel = exp1
        myt = np.arange(self.avtails_sub.shape[0])*4e-7

        if scaled == True:
            pars = mymodel.make_params(e1_amplitude=7, e1_decay=1e-3)
            out = mymodel.fit(self.avtails_sub*1e7, pars, x=myt)
        else:
            pars = mymodel.make_params(e1_amplitude=7e-7, e1_decay=1e-3)
            out = mymodel.fit(self.avtails_sub, pars, x=myt)


        if plot == True:
            print(out.fit_report());
            comps = out.eval_components()
            plt.figure("single decay fit")
            plt.plot(myt, self.avtails_sub*1e7, '.')
            plt.plot(myt, comps['e1_'], '--', label='Exp1 component')

            plt.figure("residuals")
            out.plot_residuals()

        e1_amp = out.best_values['e1_amplitude']
        e1_decay = out.best_values['e1_decay']
        redchi = out.chisqr

        myvar = np.array([[e1_amp, e1_decay, redchi]])
        return myvar

    def fit_double_exp(self,plot = False, scaled = False):
        exp1 = ExponentialModel(prefix='e1_')
        exp2 = ExponentialModel(prefix='e2_')
        mymodel = exp1 + exp2
        myt = np.arange(self.avtails_sub.shape[0])*4e-7
        mymodel.set_param_hint('e1_decay', min = 0)
        mymodel.set_param_hint('e2_decay', min = 0)
        mymodel.set_param_hint('e1_amplitude',min = 0)
        mymodel.set_param_hint('e2_amplitude',min=0)
        #pars = mymodel.make_params(e1_amplitude=7e-07, e1_decay=1e-3,
                                   #e2_amplitude=8e-07, e2_decay=1e-3)
        #out = mymodel.fit(self.avtails_sub, pars, x=myt)
        if scaled == True:
            pars = mymodel.make_params(e1_amplitude=7, e1_decay=1e-3,
                                   e2_amplitude=8, e2_decay=1e-3)
            out = mymodel.fit(self.avtails_sub*1e7, pars, x=myt)
            myy = self.avtails_sub*1e7
        else:
            pars = mymodel.make_params(e1_amplitude=7e-7, e1_decay=1e-3,
                                   e2_amplitude=8e-7, e2_decay=1e-3)
            out = mymodel.fit(self.avtails_sub, pars, x=myt)
            myy = self.avtails_sub

        if plot == True:
            print(out.fit_report());
            comps = out.eval_components()
            plt.figure("double decay fit")
            plt.figure("double decay fit").clear()
            plt.plot(myt, myy, '.')
            plt.plot(myt, comps['e1_'], '--', label='Exp1 component')
            plt.plot(myt, comps['e2_'], '--', label='Exp2 component')
            plt.show()

            plt.figure("double residuals")
            plt.figure("double residuals").clear()
            out.plot()
            plt.show()

        e1_amp = out.best_values['e1_amplitude']
        e1_decay = out.best_values['e1_decay']
        e2_amp = out.best_values['e2_amplitude']
        e2_decay = out.best_values['e2_decay']
        redchi = out.chisqr

        myvar = np.array([[e1_amp, e1_decay, e2_amp, e2_decay, redchi]])
        return myvar

    def save_pickle_object(self,filename):
        strip_name = os.path.splitext(filename)[0]
        newname = strip_name + ".pkl"
        with open(newname, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def process_data(self):

        strip_name = os.path.splitext(self.filename[0])[0]
        newname = strip_name + ".pkl"
        if os.path.isfile(newname):
            #print ("Pickle File exists")
            #print(newname)
            file = open(newname, 'rb')
            mydata = pickle.load(file)
        else:
            self.grab_data();
            self.grab_tails();
            self.grab_noise();

            self.fit_background_all()
            self.subtract_baseline_all()
            self.average_pulse()

            self.save_pickle_object(self.filename[0])
        return mydata
