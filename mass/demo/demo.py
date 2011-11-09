"""
Demonstrations of capabilities of MASS.

This is meant to be executed in ipython as a Demo (see 
demo.help for more information).

Joe Fowler, NIST
November 9, 2011
"""


import numpy
import scipy
import pylab

import mass

# <demo> --- stop ---
# <demo> silent


def generate_mnkalpha_data(
          sample_time_us=5.12,
          n_samples=4096,
          n_presamples=1024,
          n_pulses=5000,
          n_noise=1024,
          n_chan=2,
          cts_per_ev=2.0,
          ):
    """Generate fake Manganese K-alpha fluorescence data for n_chan channels."""
    generator = mass.core.fake_data.FakeDataGenerator(sample_time=sample_time_us,
                                                      n_samples=n_samples,
                                                      n_presamples=n_presamples,
                                                      model_peak=cts_per_ev)
    
    manganese = mass.calibration.fluorescence_lines.MnKAlphaDistribution(name='why do I need a name?')
    data = generator.generate_TES_group(n_pulses=n_pulses, 
                                        n_noise=n_noise,
                                        distribution=manganese, 
                                        nchan=n_chan)
    return data

print "This silent block defines the help text and the short function used for"
print "generating fake data."
# <demo> --- stop ---

sample_time_us=5.12
n_samples=4096
n_presamples=1024
n_pulses=5000
n_noise=1024
n_chan=2

# A simple demonstration of how to use MASS
    
data = generate_mnkalpha_data(sample_time_us=sample_time_us,
                                  n_samples=n_samples,
                                  n_presamples=n_presamples,
                                  n_pulses=n_pulses,
                                  n_chan=n_chan)

# You would normally load a set of datafiles from disk, but we want to run a demo
# without assuming that you have any particular data available.  Thus, this demo
# uses simulated data instead.  The function generate_mnkalpha_data() is defined
# in a "silent block" of this script.