"""
Demonstrations of how cuts can be used in MASS.

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
from mass.demo import sourceroot

wasinteractive = plt.isinteractive()  # So we can go back to initial state later
plt.ion()

DIR = tempfile.mkdtemp()
srcname = sourceroot.source_file("src/mass/regression_test/regress_chan1.ljh")
noiname = sourceroot.source_file("src/mass/regression_test/regress_chan1.noi")
shutil.copy(srcname, DIR)
shutil.copy(noiname, DIR)

print("""This silent block defines the help text and copies an example pulse
file and noise file to the temporary directory:
      %s""" % DIR)
# <demo> --- stop ---

# A demonstration of how to use cuts in MASS. The data file is the same small
# one used in the demo intro.py.
#
# Here is a typical use that would work with lots of LJH and noise files, though
# in this case we have only one of each.

pulse_pattern = os.path.join(DIR, "regress_chan1.ljh")
noise_pattern = os.path.join(DIR, "regress_chan1.noi")
print(pulse_pattern)
print(noise_pattern)

data = mass.TESGroup(pulse_pattern, noise_pattern)
data.summarize_data()
ds = data.channel[1]

# Plot some traces (first 15 traces from channel 1).
plt.figure(9, figsize=(12, 8))  # resize this window if you need to
data.plot_traces(np.arange(15), channum=1)

# <demo> --- stop ---

# First, use the feature "auto-cuts" to get some non-empty cuts:
data.auto_cuts()

# Which data were cut in this way? These produce vectors of record numbers.
badrecords = np.nonzero(ds.bad())[0]
goodrecords = np.nonzero(ds.good())[0]

# Plot the first 8 good records and the first 2 bad ones
# Notice that the bad ones are plotted with dashed lines. See why they are cut?
# One is clear: there's pile-up at 0.25 ms. The other is less so, but zoom in on the
# early samples during the pre-trigger. This burble is the reason it was cut.
recnums = np.hstack((goodrecords[:8], badrecords[:2]))
data.plot_traces(recnums, channum=1)

# <demo> --- stop ---

# Now figure out which criteria were the reason for the cuts. You see, cuts are performed
# on a variety of crieteria, each with a name.  These records are each cut for one reason,
# but of course records can be cut for more than one reason.
b_all = np.nonzero(ds.bad())[0]
b_rms = np.nonzero(ds.bad("pretrigger_rms"))[0]
b_ppd = np.nonzero(ds.bad("postpeak_deriv"))[0]
print("Why are records cut?")
print("\nBad for any reason:\n%s" % b_all)
print("\nBad because of pretrigger rms:\n%s" % b_rms)
print("\nBad because of post-peak derivative\n%s" % b_ppd)

# <demo> --- stop ---

# Do you want to make a cut with a name you invented? It works like this.
# First, register the cut's name:
data.register_boolean_cut_fields("low_energy")

# Now invent an array of booleans, one per record, where True means cut (i.e. True for bad)
# This means small pulses should be cut.
fails = ds.p_peak_value[:] < 10000
ds.cuts.cut("low_energy", fails)

# Here's the same plot of 10 pulses, but now notice that the smallest pulse (rec #7) is cut.
data.plot_traces(recnums, channum=1)


# <demo> --- stop ---

# Cuts are not just for rejecting bad pulses. You can also use what we call "categorical cuts",
# though a better name might be "pulse categories".  The data in this example don't have any
# natural categories, but let's pretend. Suppose pulses calibration data or science data, and the
# science data can be laser-pumped or not. That's a 3-way category. (Actually, a 4-way, because
# there is ALWAYS an implicit category called "uncategorized".)

# Again, we have to register the cut's name and its possible values (there can be many)
data.register_categorical_cut_field("source", ["calibration", "pumped", "unpumped"])

# Let's just assign randomly to the 3 named categories and the uncategorized group:
category = np.random.randint(4, size=ds.nPulses)

ds.cuts.cut("source", category)

# Now plot some key pulse values as a scatter plot, colored by category.
# Remember, categories are randomly assigned.
plt.clf()
use = ds.good(source="calibration")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".r")
use = ds.good(source="pumped")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".g")
use = ds.good(source="unpumped")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".b")

# <demo> --- stop ---

# An alternative approach to setting categories is also available. It uses booleans, and
# it might be more convenient in some situations.
# This time, let's categorize according to pretrigger mean.
data.register_categorical_cut_field("ptm_level", ["low", "medium", "high"])
islow = ds.p_pretrig_mean[:] < 2695
ishigh = ds.p_pretrig_mean[:] > 2715
neither = np.logical_and(~islow, ~ishigh)
boolean_categories = {"low": islow,
                      "medium": neither,
                      "high": ishigh}
data.channel[1].cuts.cut_categorical("ptm_level", boolean_categories)

# Now repeat the previous plot, except that the colors (categories) are not random.
plt.clf()
use = ds.good(ptm_level="low")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".b")
use = ds.good(ptm_level="medium")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".g")
use = ds.good(ptm_level="high")
plt.plot(ds.p_pretrig_mean[use], ds.p_pulse_rms[use], ".r")

if not wasinteractive:
    plt.ioff()
