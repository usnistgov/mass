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

wasinteractive = plt.isinteractive()  # So we can go back to initial state later
plt.ion()


def report(param, covar):
    labels = ("E res (FWHM)", "Peak energy", "dPH/dE", "Amplitude",
              "Const BG", "BG slope", "Tail fraction", "Tail length")
    for i, p in enumerate(param):
        txt = ""
        if covar[i, i] == 0.0:
            txt = "HELD"
        print("%-14s %8.3f +- %7.3f   %s" % (labels[i], p, covar[i, i]**0.5, txt))
print("""To run this demo, you need to have the ReferenceMicrocalFiles.jl package.
Installed.

To install it, start a terminal in the desired directory, check out the repository and
make sure that it has some LJH files:

git clone https://github.com/ggggggggg/ReferenceMicrocalFiles.jl
cd ReferenceMicrocalFiles.jl
ls -l ljh

Then within your Ipython session, set the variable like this

massdemo.run_cell('DIR = "/Users/whoever/data/ReferenceMicrocalFiles.jl"')

before you proceed to try running the rest of this demo.
""")


# <demo> --- stop ---

# Load up the data
assert(os.path.isdir(DIR))
pulse_pattern = os.path.join(DIR, "ljh", "20150707_D_chan13.ljh")
noise_pattern = os.path.join(DIR, "ljh", "20150707_C_chan13.noi")
print(pulse_pattern)
print(noise_pattern)

data = mass.TESGroup(pulse_pattern, noise_pattern)
data.summarize_data()
data.auto_cuts()

# Check out these spectra: you can see that this is Mn KAlpha
plt.figure(9, figsize=(8, 8))  # resize this window if you need to
ds = data.channel[13]
ds.plot_summaries()
g = ds.good()

# <demo> --- stop ---

# Here's the noise PSD
data.compute_noise_spectra()
data.plot_noise()
# <demo> --- stop ---

# Here's the average pulse
data.avg_pulses_auto_masks()
data.plot_average_pulses()

# <demo> --- stop ---

# Compute an optimal filter and apply it
data.compute_filters()
data.summarize_filters(std_energy=5898.8)
data.filter_data()

# Apply the usual drift- and phase-corrections.
# What do they look like?
data.drift_correct()
data.phase_correct()
KA_peak = np.median(ds.p_filt_value_dc[g])

plt.clf()
ax1 = plt.subplot(221)
plt.title("Before Drift Correction")
plt.xlabel("Pretrig mean")
plt.ylabel("Filt value")
plt.plot(ds.p_pretrig_mean[g], ds.p_filt_value[g], ".r")

ax2 = plt.subplot(222, sharey=ax1)
plt.plot(ds.p_pretrig_mean[g], ds.p_filt_value_dc[g], ".g")
plt.title("After Drift Correction")
plt.xlabel("Pretrig mean")
plt.ylabel("Filt value")

ax3 = plt.subplot(223, sharey=ax1)
plt.title("Before Phase Correction")
plt.xlabel("Phase")
plt.ylabel("Filt value")
plt.plot(ds.p_filt_phase[g], ds.p_filt_value_dc[g], ".g")

ax4 = plt.subplot(224, sharex=ax3, sharey=ax1)
plt.plot(ds.p_filt_phase[g], ds.p_filt_value_phc[g], ".b")
plt.xlim([-.65, .5])
plt.ylim(np.array([.996, 1.0025])*KA_peak)
plt.title("After Phase Correction")
plt.xlabel("Phase")
plt.ylabel("Filt value")

# <demo> --- stop ---

# Now fit for the resolution, using the Mn KAlpha data
c, b = np.histogram(ds.p_filt_value_phc[g], 120, np.array([.993, 1.003])*KA_peak)
fitter = mass.MnKAlphaFitter()
param_guess = [2.6, b[c.argmax()], 3, 10*c.max(), c.min(), 0, 0, 25]
param, covar = fitter.fit(c, b, param_guess, label="full")
report(param, covar)
# <demo> --- stop ---

# Now fit for the resolution again, with the low-energy tail allowed to vary.
param_guess = param.copy()
param_guess[-2:] = [.1, 25]
param, covar = fitter.fit(c, b, param_guess, vary_tail=True, vary_bg_slope=False, label="full")
report(param, covar)

if not wasinteractive:
    plt.ioff()
# <demo> --- stop ---
