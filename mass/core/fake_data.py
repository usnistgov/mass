"""
fake_data - Objects to make fake data for use, e.g., in demonstration scripts.

Joe Fowler, NIST

November 7, 2011
"""

## \file fake_data.py
# \brief  Objects to make fake data for use, e.g., in demonstration scripts.
# 

__all__ = ['FakeDataGenerator']

import numpy
import os
import tempfile

import mass


class FakeDataGenerator(object):
    """"""
    
    def __init__(self, sample_time, n_samples, n_presamples=None, model_peak=None):
        # Some defaults that can be overridden before generating fake data
        self.pretrig_level = 1000
        self.rise_speed_us = 200. # in us
        self.fall_speed_us = 1200. # in us
        
        
        self.sample_time_us = sample_time # in us
        self.n_samples = n_samples
        if n_presamples is None:
            self.n_presamples = self.n_samples/4
        else:
            self.n_presamples = n_presamples
        self.compute_model(model_peak=model_peak)
        
    def compute_model(self, model_peak=None):
        dt_us = (numpy.arange(self.n_samples) - self.n_presamples-0.5) * self.sample_time_us
        self.model = numpy.exp(-dt_us/self.fall_speed_us) - numpy.exp(-dt_us/self.rise_speed_us)
        self.model[dt_us<=0] = 0
        if model_peak is not None:
            self.model = model_peak * self.model/self.model.max()
            
    
    def _generate_virtual_file(self, n_pulses, rate=1.0):
        data = numpy.zeros((n_pulses, self.n_samples), dtype=numpy.uint16)
        pulse_times = numpy.random.exponential(1.0/rate, size=n_pulses).cumsum()
        
        for i in range(n_pulses):
            data[i,:] = self.model + self.pretrig_level + (0.5+numpy.random.standard_normal(self.n_samples)*5)
        vfile = mass.files.VirtualFile(data, times=pulse_times)
        vfile.timebase = self.sample_time_us/1e6
        vfile.nPresamples = self.n_presamples
        return vfile
    
    
    def generate_microcal_dataset(self, n_pulses):
        vfile = self._generate_virtual_file(n_pulses)
        return mass.core.channel.create_pulse_and_noise_records(vfile, pulse_only=True)
    
    
    def generate_TES_group(self, n_pulses, nchan=1):
        vfiles = [self._generate_virtual_file(n_pulses) for _i in range(nchan)]
        return mass.TESGroup(vfiles, pulse_only = True)
        
    

class DONOTUSE_FakeDataGenerator(object):
    """
    Object to generate fake data sets
    """
    
    def __init__(self, ndet=1, directory=None, file_prefix=None):
        self.ndet = ndet
        if directory is None:
            self.directory = tempfile.mkdtemp("_fake_ljhfiles")
        else:
            if os.path.isdir(directory):
                self.directory = directory
            elif not os.path.exists(directory):
                os.mkdir(directory)
                self.directory = directory
            else:
                raise ValueError("Given directory='%s' is not an existing directory.")
            
        if file_prefix is None:
            self.file_prefix = "fake_data_"
        else:
            self.file_prefix = file_prefix
    
    def generate_one_dataset(self, n_pulses, n_samples, n_pretrig, resolution):
        pass