"""
Demonstrations of capabilities of MASS.

This is meant to be executed in ipython as a Demo (see
demo.help for more information).

Joe Fowler, NIST
April 2017
"""

# <demo> silent

import numpy as np
import pylab as plt
import tempfile
import os
import shutil

import mass
import mass.calibration.fluorescence_lines
from mass.demo import sourceroot

wasinteractive = plt.isinteractive()  # So we can go back to initial state later
plt.ion()

DIR = tempfile.mkdtemp()
srcname = sourceroot.source_file("tests/regression_test/regress_chan1.ljh")
noiname = sourceroot.source_file("tests/regression_test/regress_noise_chan1.ljh")
shutil.copy(srcname, DIR)
shutil.copy(noiname, DIR)

print("""This silent block defines the help text and copies an example pulse
file and noise file to the temporary directory:
      %s""" % DIR)
# <demo> --- stop ---

# A simple demonstration of how to use MASS


# You create a TESGroup object "data" by giving as a 1st argument *either*
# 1) a filename
# 2) "glob pattern" (e.g., "/path/to/files/blahblah_chan*.ljh") or
# 3) a Python sequence (eg a list) containing 1 or more filenames.
# A second argument is the pattern or sequence of names corresponding to the
# noise data files.
#
# Here is a typical use that would work with lots of LJH and noise files, though
# in this case we have only one of each.

pulse_pattern = os.path.join(DIR, "regress_chan1.ljh")
noise_pattern = os.path.join(DIR, "regress_noise_chan1.ljh")
print(pulse_pattern)
print(noise_pattern)

data = mass.TESGroup(pulse_pattern, noise_pattern)

# Now plot some traces (first 15 traces from channel 1).
# Don't put the pulse summaries on the plot, b/c they are all zero still!
plt.figure(9, figsize=(12, 8))  # resize this window if you need to
data.plot_traces(np.arange(15), channum=1, pulse_summary=False)

# <demo> --- stop ---

# Now let's compute summary information for each pulse and make a few plots.
# This will be one of up to 3 passes through every byte of data,
# which is normally stored on disk and loaded only one chunk at time.
data.summarize_data()            # In real data, this requires a pass through every byte of data.
ds = data.channel[1]             # We'll consider only channel 1.
ds.plot_summaries()

# This command plots a timeseries and a histogram for each of eight per-pulse summary
# quantities.  pylab lets you zoom and pan with the mouse.  Check them out.
# <demo> --- stop ---

# That plot was all quantities for one channel.  If, instead, you prefer to see
# all channels for a given quantity (which you normally would do only in a situation
# with more than 1 or 2 channels), you can do the following:

data.plot_summaries(quantity=6, log=False)
# <demo> --- stop ---
# That was a quantity selected by number (0-7).  You can also select by name.
# For the name-number correspondence, do
print(data.plot_summaries.__doc__)
data.plot_summaries('Pulse RMS', log=False)
# <demo> --- stop ---

# Now you would normally apply some cuts. Cuts can be pretty complicated. We have just
# introduced a system to make a few anti-pileup cuts that are computed automatically.
# To try it and plot the summaries again:
data.auto_cuts()
ds.plot_summaries()
# <demo> --- stop ---

# Let's compute a reasonable average pulse.
# For the simplest case, we can simply use an "auto-mask", basically asking MASS
# to choose a wide range of pulse sizes around the median, and just average them.
# Here it is, and the result is plotted.  This is the 2nd of 3 full passes
# through all the data, but magically, it doesn't do ALL the data (instead, it
# stops the computation after some minimum number of pulses are found).
data.avg_pulses_auto_masks()
data.plot_average_pulses()
# <demo> --- stop ---

# Sometimes you have more specific needs on which pulses go into the average pulse.
# THIS IS BASICALLY A MORE COMPLICATED WAY TO DO WHAT avg_pulses_auto_masks() did,
# so most users can just ignore this bit.
medians = np.array([np.median(ds.p_peak_value) for ds in data.datasets])
avgmed = medians.mean()
gains = medians/avgmed    # The gains will be very close to 1.00 in sim data
print('Medians: ', medians)
print('Gains:   ', gains)

# Next we make a mask to select which pulses will be used.
# For now, make a quite liberal +- 5% around the median in each channel
masks = data.make_masks(pulse_peak_range=[.95*avgmed, 1.05*avgmed], gains=gains)

# Finally, compute the average pulses and plot them.
data.compute_average_pulse(masks, subtract_mean=True)
ALL_CHANS = -1
data.plot_average_pulses(ALL_CHANS)
# <demo> --- stop ---

# Now we handle the noise.  Let's compute the power spectrum and the autocorrelation.
data.compute_noise()
# We'll plot them on 2 panels of the same figure
plt.clf()
data.plot_noise(axis=plt.subplot(211))
data.plot_noise_autocorrelation(axis=plt.subplot(212))
# <demo> --- stop ---

# Good.  Now we can compute optimal filters, show the expected performance,
# and plot the filters.
data.compute_filters(f_3db=20000)
data.summarize_filters(std_energy=930.0)
data.plot_filters()

# Finally, let's filter the data.  This is the last of 3 passes through the data.
data.filter_data()
# <demo> --- stop ---

# Now drift-correct. Here's the result before and after drift correction.
# Zoom in on one plot and see the tilt in the red scatter data.
data.drift_correct()
g = ds.good()

plt.clf()
ax = plt.subplot(121)
plt.plot(ds.p_pretrig_mean[g], ds.p_filt_value[g], ".r")
plt.ylabel("Filtered pulse heights")
plt.xlabel("Pretrigger mean")
plt.title("Before drift correction", color="r")

ax2 = plt.subplot(122, sharex=ax, sharey=ax)
plt.plot(ds.p_pretrig_mean[g], ds.p_filt_value_dc[g], ".b")
plt.title("After drift correction", color="b")
plt.xlabel("Pretrigger mean")
# <demo> --- stop ---

# Now phase-correct. Here's the result before and after phase correction.
# Zoom in on one plot and see the non-flatness of the blue scatter data.
data.phase_correct()
g = ds.good()

plt.clf()
ax = plt.subplot(121)
plt.plot(ds.p_filt_phase[g], ds.p_filt_value_dc[g], ".b")
plt.ylabel("Filtered pulse heights")
plt.xlabel("'Phase' (arrival time)")
plt.title("Before phase correction", color="b")
ax2 = plt.subplot(122, sharex=ax, sharey=ax)
plt.plot(ds.p_filt_phase[g], ds.p_filt_value_phc[g], ".k")
plt.title("After phase correction")
# <demo> --- stop ---

# Here are histograms of the values
plt.clf()
for name, v in zip(["filtered value", "drift corrected", "phase_corrected"],
                   [ds.p_filt_value, ds.p_filt_value_dc, ds.p_filt_value_phc]):
    plt.hist(v[g], 35, [13400, 13500], histtype="step", label=name)
    print("Robust width estimator of %17s: %5.2f" % (name, mass.robust.shorth_range(v[g])))
plt.legend(loc="best")
if not wasinteractive:
    plt.ioff()
# <demo> --- stop ---
