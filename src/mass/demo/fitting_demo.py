"""
Demonstrations of fitting within MASS.

This is meant to be executed in ipython as a Demo (see
demo.help for more information).

Joe Fowler, NIST
March 26, 2011
"""

# <demo> silent

import numpy as np
import pylab as plt
import mass

wasinteractive = plt.isinteractive()  # So we can go back to initial state later
plt.ion()
np.random.seed(2384792) # avoid NaN errors on galen's computer
# <demo> stop

# First, let's work with a simple Gaussian fit. We'll make some data to fit
FWHM_SIGMA_RATIO = (8*np.log(2))**0.5
N, mu, sigma = 4000, 400.0, 20.0
fwhm = FWHM_SIGMA_RATIO*sigma
d = np.random.standard_normal(size=N)*sigma + mu

# We are going to fit histograms, not raw data vectors:
hist, bin_edges = np.histogram(d, 100, [mu-4*sigma, mu+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# First, just plot the histogram.
plt.clf()
axis = plt.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr)

# <demo> stop

# Now do a fit.  All fitters for spectral peaks in mass allow for a background having a
# constant level plus a linear slope, and an energy-response function that is a Bortels
# function. That is, a Gaussian convolved with the sum of a delta function and a one-sided
# exponential tail to low energies. The parameters are:
# 0 - Gaussian resolution (FWHM)
# 1 - Pulse height (x-value) of the line peak
# 2 - Amplitude (y-value) of the line peak
# 3 - Mean background counts per bin
# 4 - Background slope (counts per bin per bin)  FIXED
# 5 - Low-energy tail fraction (0 <= f <= 1)     FIXED
# 6 - Low-energy tail scale length               FIXED
#
# Notice that the x-units are whatever you find convenient and need not be energy units.
# But of course the resolution (param 0) and the tail scale length (param 6) are in the
# same units as the x-value of the peak (param 1).
#
# Parameters 4, 5, and 6 are marked FIXED, because they are fixed by default. Soon we'll
# let them very.

fitter = mass.GaussianFitter()
guess_params = [fwhm, mu, hist.max(), 0, 0, 0.0, 25]
params, covariance = fitter.fit(hist, bin_ctr, guess_params)
for i, gp in enumerate(guess_params):
    print("Param %d: initial guess %8.4f estimate %8.4f  uncertainty %8.4f" %
          (i, gp, params[i], covariance[i, i]**.5))

# Compute the model function and plot it in red.
model = fitter.last_fit_result
plt.plot(bin_ctr, model, 'r')

# <demo> stop
# Alternativley use the fitter's plot command. The legend here shows all fit parameters,
# the fit uncertainty, and a trailing H indicates the paramter was held (aka FIXED).
fitter.plot(color="r",label="full", ph_units="arb in this demo")

# <demo> stop
# We'll repeat the fit 3 ways: (1) with zero background, (2) just like before,
# with a constant background, and finally (3) with a sloped linear background.
# To make it interesting, let's add a Poisson background of 12 counts per bin.
# Note that we get a poor fit when there IS a background but we don't let it be fit for,
# as in fit (1) here.

hist += np.random.poisson(lam=12.0, size=len(hist))

plt.clf()
axis = plt.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr, color='blue')

color = 'red', 'gold', 'green'
title = 'No BG', 'Constant BG', 'Sloped BG'
print('True parameter values: FWHM=%.4f Ctr=%.4f' % (fwhm, mu))
for nbg in (0, 1, 2):
    vary_bg_slope = (nbg == 2)
    hold = []
    if nbg == 0:
        hold.append(fitter.param_meaning["background"])
    else:
        guess_params[3] = 12.0
    params, covariance = fitter.fit(hist, bin_ctr, guess_params, hold=hold,
                                    plot=False, vary_bg_slope=vary_bg_slope)
    print("Model: %s" % title[nbg])
    for i, gp in enumerate(guess_params[:5]):
        print("Param %d: initial guess %8.4f estimate %8.4f  uncertainty %8.4f" %
              (i, gp, params[i], covariance[i, i]**.5))
    print

    # Compute the model function and plot it in red.
    plt.plot(bin_ctr, fitter.last_fit_result, color=color[nbg], label=title[nbg])
plt.legend()

# <demo> stop

# Now let's generate data from a Lorentzian (Cauchy) distribution and fit with a
# VoigtFitter. (A Voigt function is the convolution of a Lorentzian and a Gaussian).
# First, freeze the Gaussian component to a width of 1e-6 (0 causes errors), to get
# a Lorentzian fit.

mu, fullwidth = 100.0, 6.0
dc = np.random.standard_cauchy(size=N)*fullwidth*0.5 + mu
histc, bin_edges = np.histogram(dc, 200, [mu-10-4*fullwidth, mu+10+4*fullwidth])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# Fit a Lorentzian to the Lorentzian data
fitter = mass.calibration.VoigtFitter()
true_params = [1e-6, mu, fullwidth, histc.max(), 0.0, 0.0, 0.0, 25]
params, covariance = fitter.fit(histc, bin_ctr, true_params, hold=(0,), plot=True)
for i, tp in enumerate(true_params):
    print("Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f" %
          (i, tp, params[i], covariance[i, i]**.5))

# <demo> stop

# Finally, put real Gaussian smearing on the data and use the Voigt fitter again.
sigma = 3.0
dv = dc + np.random.standard_normal(size=N)*sigma
histv, bin_edges = np.histogram(dv, 200, [mu-10-4*fullwidth, mu+10+4*fullwidth])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

true_params[0] = sigma*2.3548
params, covariance = fitter.fit(histv, bin_ctr, true_params, plot=True)
for i, tp in enumerate(true_params):
    print("Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f" %
          (i, tp, params[i], covariance[i, i]**.5))

# <demo> stop

# Now let's fit two Voigt functions.
N1, N2, Nbg = 3000, 2000, 1000
mu1, mu2, sigma = 100.0, 105.0, 0.8
dc1 = np.random.standard_cauchy(size=N1)+mu1
dc2 = np.random.standard_cauchy(size=N2)+mu2
dc = np.hstack([dc1, dc2])
dc += np.random.standard_normal(size=N1+N2)*sigma

histc, bin_edges = np.histogram(dc, 200, [mu1-10-4*sigma, mu2+10+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

fitter = mass.calibration.NVoigtFitter(2)
true_params = np.array([sigma*2.3548, mu1, 2, N1*.15, mu2, 2, N2*.15, .1, 0, 0, 25])
# Those are the correct values.  Let's mess with them by 3% (more or less):
param_guess = true_params * 1+np.random.standard_normal(11)*0.03

params, covariance = fitter.fit(histc, bin_ctr, param_guess)
for i, tp in enumerate(true_params):
    print("Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f" %
          (i, tp, params[i], covariance[i, i]**.5))

if not wasinteractive:
    plt.ioff()

# See demo "fitting_fluorescence.py" for examples of fluorescence line shapes.
