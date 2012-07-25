"""
Demonstrations of fitting within MASS.

This is meant to be executed in ipython as a Demo (see 
demo.help for more information).

Joe Fowler, NIST
March 26, 2011
"""

#111 # <demo> silent

import numpy, pylab
import mass
wasinteractive = pylab.isinteractive() # So we can go back to initial state later
pylab.ion()

# First, let's work with a simple Gaussian fit
FWHM_SIGMA_RATIO = (8*numpy.log(2))**0.5
N,mu,sigma = 4000,400.0, 20.0
fwhm = FWHM_SIGMA_RATIO*sigma
d = numpy.random.standard_normal(size=N)*sigma + mu

# We are going to fit histograms, not raw data vectors:
hist, bin_edges = numpy.histogram(d, 100, [mu-4*sigma, mu+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# First, just plot the histogram.
pylab.clf()
axis = pylab.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr)

# <demo> stop

# Now do a fit.  First we'll do it the most generic, powerful, and annoying way.

# I don't particularly recommend it, but if you require a SLOPED background,
# it's the only way to go (for now).

# It's not easy to plot the fitted model.  Also, notice that it requires you
# to make initial guesses for the parameters.  Parameters are, in order:
# [FWHM, Centroid, Peak Value, Const BG Level, BG slope]
# The last is optional and can be left off

guess_params = [fwhm, mu, hist.max(), 0]
fitter = mass.fitting.MaximumLikelihoodGaussianFitter(bin_ctr, hist, guess_params)
params, covariance = fitter.fit()
for i in range(len(guess_params)):
    print "Param %d: initial guess %8.4f estimate %8.4f  uncertainty %8.4f"%(i, guess_params[i], params[i], covariance[i,i]**.5)

# Compute the model function and plot it in red.
model = fitter.theory_function(params, bin_ctr)
pylab.plot(bin_ctr, model, 'r')

# <demo> stop
# I said the mass.fitting.MaximumLikelihoodGaussianFitter method is annoying but
# powerful.  Let's see the power.

# We'll repeat that fit 3 ways: (1) with zero background, (2) just like before,
# with a constant background, and finally (3) with a sloped linear background.
# To make it interesting, let's add a Poisson background of 2 counts per bin.
# Note that we get a poor fit when there IS a background but we don't let it be fit for,
# as in fit (1) here. 

hist += numpy.random.poisson(lam=2.0, size=len(hist))

pylab.clf()
axis = pylab.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr, color='blue')

guess_params = [fwhm, mu, hist.max()]
color='red','gold','green'
title='No BG','Constant BG','Sloped BG'
print 'True parameter values: FWHM=%.4f Ctr=%.4f'%(fwhm, mu)
for nbg in (0,1,2):
    if nbg == 1 or nbg==2:
        guess_params.append(0)
    fitter = mass.fitting.MaximumLikelihoodGaussianFitter(bin_ctr, hist, guess_params)
    params, covariance = fitter.fit()
    print "Model: %s"%title[nbg]
    for i in range(len(guess_params)):
        print "Param %d: initial guess %8.4f estimate %8.4f  uncertainty %8.4f"%(i, guess_params[i], params[i], covariance[i,i]**.5)
    print
    
    # Compute the model function and plot it in red.
    model = fitter.theory_function(params, bin_ctr)
    pylab.plot(bin_ctr, model, color=color[nbg], label=title[nbg])
pylab.legend()

# <demo> stop
# Fine, that was the more powerful way, which you will probably never use.
# (If you need to use it with the sloped background a lot, then talk to me,
# and we will see if we can fit sloped BG into the simpler approach.)

# Here's the simpler, more usual way.
# We'll generate a new set of Gaussian random numbers, histogram them, and fit.

# These three lines are a repeat of what you saw earlier in this demo.
d = numpy.random.standard_normal(size=N)*sigma + mu
hist, bin_edges = numpy.histogram(d, 100, [mu-4*sigma, mu+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# Now fit the smooth way.
fitter = mass.GaussianFitter()
params, covariance = fitter.fit(hist, bin_ctr, plot=True)
true_params = [FWHM_SIGMA_RATIO*sigma, mu, N*(bin_edges[1]-bin_edges[0])/sigma/(2*numpy.pi)**0.5, 0]
for i in range(len(true_params)):
    print "Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f"%(i, 
                true_params[i], params[i], covariance[i,i]**.5)
# <demo> stop

# Now let's generate data from a Lorentzian (Cauchy) distribution
mu,sigma = 100.0, 3.0
dc = numpy.random.standard_cauchy(size=N)+mu
histc, bin_edges = numpy.histogram(dc, 200, [mu-10-4*sigma, mu+10+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# First, fit a Lorentzian to the Lorentzian data
fitter = mass.calibration.fluorescence_lines.LorentzianFitter()
params, covariance = fitter.fit(histc, bin_ctr, plot=True)
true_params = [mu, 1.0, N*(bin_edges[1]-bin_edges[0])]
for i in range(len(true_params)):
    print "Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f"%(i, 
                true_params[i], params[i], covariance[i,i]**.5)

# Notice that we could have used the VoigtFitter, and it would probably work.
# By choosing the Lorentzian fitter, we insist that the Gaussian smearing = 0.
# )You could use the VoigtFitter's hold=[0] or vary_resolution=False arguments
# to accomplish the same thing, of course.)
# <demo> stop

# Now try the general Voigt fitter, even though the Gaussian smearing is zero.
fitter = mass.calibration.fluorescence_lines.VoigtFitter()
params, covariance = fitter.fit(histc, bin_ctr, plot=True)
true_params = [0, mu, 1.0, N*(bin_edges[1]-bin_edges[0])]
for i in range(len(true_params)):
    print "Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f"%(i, 
                true_params[i], params[i], covariance[i,i]**.5)
# <demo> stop

# Finally, put real Gaussian smearing on the data and use the Voigt fitter again.
dv = dc + numpy.random.standard_normal(size=N)*sigma
histv, bin_edges = numpy.histogram(dv, 100, [mu-10-4*sigma, mu+10+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

params, covariance = fitter.fit(histv, bin_ctr, plot=True)
true_params = [FWHM_SIGMA_RATIO*sigma, mu, 1.0, N*(bin_edges[1]-bin_edges[0])/sigma/(2*numpy.pi)**0.5]
for i in range(len(true_params)):
    print "Param %d: true value %8.4f estimate %8.4f  uncertainty %8.4f"%(i, 
                true_params[i], params[i], covariance[i,i]**.5)

# <demo> stop

# Now let's fit two Voigt functions.
N1, N2, Nbg = 3000, 2000, 1000 
mu1, mu2, sigma = 100.0, 105.0, 0.5
dc1 = numpy.random.standard_cauchy(size=N1)+mu1
dc2 = numpy.random.standard_cauchy(size=N2)+mu2
dc = numpy.hstack([dc1,dc2])
dc += numpy.random.standard_normal(size=N1+N2)*sigma

histc, bin_edges = numpy.histogram(dc, 200, [mu1-10-4*sigma, mu2+10+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

fitter = mass.calibration.fluorescence_lines.TwoVoigtFitter()
param_guess= numpy.array([sigma*2.3548, mu1, 1, N1, mu2, 1, N2, .1])
# Those are the correct values.  Let's mess with them by 3% (more or less):
param_guess *= 1+numpy.random.standard_normal(8)*0.03

params,covar=fitter.fit(histc, bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0]), params=param_guess)
print params; print covar.diagonal()**0.5
