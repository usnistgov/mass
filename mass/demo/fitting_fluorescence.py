"""
Demonstrations of fitting fluorescence lines within MASS.

This is meant to be executed in ipython as a Demo (see
demo.help for more information).

Joe Fowler, NIST
March 26, 2011
"""

# <demo> silent

import numpy as np
import pylab as plt
import mass

import logging
LOG = logging.getLogger("mass")

wasinteractive = plt.isinteractive()  # So we can go back to initial state later
plt.ion()

# <demo> stop
LOG.info("For fun, here is the Mn K-alpha complex.")
mass.MnKAlpha().plot()

# <demo> stop
# Let's generate some data distributed as if from the Mn K-alpha complex, with
# a nonzero Gaussian smearing

res_fwhm = 3.0
res_sigma = res_fwhm / 2.3548
distrib = mass.calibration.MnKAlpha()
N = 10000
energies = distrib.rvs(size=N)
energies += np.random.standard_normal(N)*res_sigma

plt.clf()
hist, bin_edges, _ = plt.hist(energies, 200, [5865, 5915], histtype="step", color="g")
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# <demo> stop
# Now fit to find the resolution, line center, and "stretch factor".
fitter = mass.MnKAlphaFitter()
param, covar = fitter.fit(hist, bin_ctr, plot=True, label="full")
# Here we're using the "full" label option to get more info on the plot.
# "H" after a parmater indicates when it was held.
# You can get the same info with:
LOG.info(fitter.result_string())

# <demo> stop
# Notice that the "stretch factor" (param 2) probably shouldn't be allowed to vary: this is a fit in
# actual, previously calibrated energy units. Let's fix this conversion factor from energy to
# the measured units to be 1.0. You should do this when you've got energy-calibrated data.
param_guess = param.copy()
param_guess[2] = 1.0
param_guess[5] = 0.5
hold = []
hold.append(fitter.param_meaning["dP_dE"])
param, covar = fitter.fit(hist, bin_ctr, param_guess, hold=hold, label="full")

# <demo> stop
# Now let's add a sloped background
expected_bg = (bin_ctr-5860)*0.4
hist += np.random.poisson(lam=expected_bg, size=len(hist))

param, covar = fitter.fit(hist, bin_ctr, param_guess, hold=hold, vary_bg_slope=True, label="full")

# <demo> stop
# Now let there be a 20% low-energy tail.
Naffected = N//5
tail_len = 10.0
energies[:Naffected] -= np.random.exponential(tail_len, size=Naffected)
hist, _ = np.histogram(energies, 200, [5865, 5915])
param_guess = [res_fwhm, 5898.9, 1.0, param_guess[3], 0, 0, 0.2, tail_len]

param, covar = fitter.fit(hist, bin_ctr, param_guess, hold=hold, vary_tail=True, label="full")

if not wasinteractive:
    plt.ioff()
