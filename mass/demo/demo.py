"""
Demonstrations of capabilities of MASS.

This is meant to be executed in ipython as a Demo (see 
demo.help for more information).

Joe Fowler, NIST
November 9, 2011
"""



# <demo> silent

import mass
import mass.calibration.fluorescence_lines


def generate_mnkalphabeta_data(
          sample_time_us=10.24,
          n_samples=2048,
          n_presamples=512,
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
    
    manganese = mass.calibration.fluorescence_lines.MnKAlphaDistribution(name=
                            'Why do I need a name?  This is a bug...')
    manganese2 = mass.calibration.fluorescence_lines.MnKBetaDistribution(name='x')
    data = generator.generate_tesgroup(n_pulses=n_pulses, 
                                       n_noise=n_noise,
                                       distributions=(manganese, manganese2),
                                       distribution_weights=(4,1), 
                                       nchan=n_chan)
    return data

print("This silent block defines the help text and the short function used for")
print("generating fake data.")
# <demo> --- stop ---

# A simple demonstration of how to use MASS
    
import numpy
import pylab
wasinteractive = pylab.isinteractive() # So we can go back to initial state later
pylab.ion()

sample_time_us=10.24
n_samples=2048        # samples in each data record
n_presamples=512
n_pulses=5000         # data records per channel
n_noise=1024
n_chan=2              # number of data channels (detectors)

data = generate_mnkalphabeta_data(sample_time_us=sample_time_us,
                                  n_samples=n_samples,
                                  n_presamples=n_presamples,
                                  n_pulses=n_pulses,
                                  n_chan=n_chan)

# You would normally load a set of datafiles from disk, but we want to run a demo
# without assuming that you have any particular data available.  Thus, this demo
# uses simulated data instead.  The function generate_mnkalpha_data() is defined
# in a "silent block" of this script.
# 
# Be patient: this can take several seconds...

# Now plot some traces (first 15 traces from channel 0)
pylab.figure(9, figsize=(12,8))  # resize this window if you need to
data.plot_traces(numpy.arange(15), dataset_num=0)

# <demo> --- stop ---

# Now let's compute summary information for each pulse and make a few plots.
# In real data, this will be one of only 3 passes through every byte of data,
# which is normally stored on disk and loaded only one chunk at time.
data.summarize_data()            # In real data, this requires a pass through every byte of data.
ds0 = data.datasets[0]            # We'll consider only dataset 0
ds0.plot_summaries()

# This command plots a timeseries and a histogram for each of 7 per-pulse summary
# quantities.  Pylab lets you zoom and pan with the mouse.  Check them out.
# <demo> --- stop ---

# That plot was all quantities for one channel.  If, instead, you prefer to see
# all channels for a given quantity (which you normally would do in a situation
# with more than 1 or 2 channels), you can do this:

data.plot_summaries(quantity=1, log=False)
# <demo> --- stop ---
# That was a quantity selected by number (0-6).  You can also select by name.
# For the name-number correspondence, do
print(data.plot_summaries.__doc__)
data.plot_summaries('Pulse Avg', log=False)
# <demo> --- stop ---

# Now you would normally apply some cuts.  I AM GOING TO SKIP DEMO OF CUTS FOR NOW.

# Let's compute the average pulse.  
# First we want to find the typical size of an interesting (Mn KAlpha) pulse.
# I am not totally happy with how this is done, but a decent way to handle this in
# clean fluorescence data is to look for the median pulse height
medians = numpy.array([numpy.median(ds.p_peak_value) for ds in data.datasets])
avgmed = medians.mean()
gains = medians/avgmed    # The gains will be very close to 1.00 in sim data
print('Medians: ', medians)
print('Gains:   ', gains)

# Next we make a mask to select which pulses will be used.
# For now, make a quite liberal +- 3% around the median in each channel
masks = data.make_masks(pulse_peak_ranges=[.97*avgmed, 1.03*avgmed], gains=gains)

# Finally, compute the average pulses and plot them.  This is the 2nd of 3 passes
# through all the data
data.compute_average_pulse(masks, subtract_mean=True)
ALL_CHANS=-1
data.plot_average_pulses(ALL_CHANS)
# <demo> --- stop ---

# Now we handle the noise.  The fake data generator made a noise "file" for each
# channel of pure white noise.  Let's compute the power spectrum and the autocorrelation.
data.compute_noise_spectra(n_lags = n_samples)
# We'll plot them on 2 panels of the same figure
pylab.clf()
data.plot_noise(axis=pylab.subplot(211))
data.plot_noise_autocorrelation(axis=pylab.subplot(212))
# <demo> --- stop ---

# Good.  Now we can compute optimal filters, show the expected performance,
# and plot the filters (red=Brad-style filter, gold=Fourier-style filter,
# blue and purple are 2 filters that will estimate the pretrigger level;
# we only ever use the red). 
data.compute_filters(fmax=20000)
data.summarize_filters()
data.plot_filters()

# Finally, let's filter the data.  This is the last of 3 passes through the data.
data.filter_data()
# <demo> --- stop ---

# Now let's re-estimate the peak value and histogram the data
medians = numpy.array([numpy.median(ds.p_filt_value) for ds in data.datasets])
hist_limits = numpy.array([.99,1.01])*medians.mean()
pylab.clf()
colors = 'purple', 'blue', 'green', 'gold', 'orange', 'red'
for i, (color,median,ds) in enumerate(zip(colors, medians, data.datasets)):
    good = numpy.abs(ds.p_filt_value/median-1.0)<.02
    pylab.subplot(2,1,1+i)
    pylab.hist(ds.p_filt_value[good], 200, hist_limits, 
               color=color, histtype='step', label='Channel %d'%i)
    pylab.legend(loc='best')
# <demo> --- stop ---

# We've seen the histogram, and now it's time to try fitting for the
# counts->eV scaling and for the resolution.  Here is the very easiest
# way to get started (even easier if you didn't want to have each channel
# on its own subplot...)
# 

pylab.clf()
for i, ds in enumerate(data.datasets):
    axis =pylab.subplot(211+i)
    ds.fit_spectral_line(prange=hist_limits, line='MnKAlpha',
                         plot=True, axis=axis)

# <demo> --- stop --
# Notice that it's not so easy to automate, because you had to pass in a specific
# pulse height range (we used the same limits on the previously plotted histogram).
#
# A cleaner, more automatable method would be to use the fit
# and set them into the relevant calibration object.

for median, ds in zip(medians, data.datasets):
    param, covar, fitter = ds.fit_spectral_line(prange=hist_limits, line='MnKAlpha', 
                                                plot=False)
    cal = ds.calibration['p_filt_value']
    cal.add_cal_point(param[1], 'MnKAlpha')
#    guess Mn KBeta location
    use = ds.p_filt_value_dc > (cal.name2ph('MnKBeta')+cal.name2ph('MnKAlpha'))*.5
    use = numpy.logical_and(ds.cuts.good(), use)
    mean_kbeta = ds.p_filt_value_dc[use].mean()
    cal.add_cal_point(mean_kbeta, 'MnKBeta')

# Now fit Mn Kalpha and Kbeta on channel of your choice
# Top 2 panels are the 2 lines fit in PULSE HEIGHT units.
# Bottom is the "real" fit in energy units.  Calibration curve also shown
channum=0
data.datasets[channum].fit_MnK_lines()

if not wasinteractive:
    pylab.ioff()
