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
N,mu,sigma = 4000,400.0, 20.0
d = numpy.random.standard_normal(size=N)*sigma + mu

# We are going to fit histograms, not raw data vectors:
hist, bin_edges = numpy.histogram(d, 100, [mu-4*sigma, mu+4*sigma])
bin_ctr = 0.5*(bin_edges[1]-bin_edges[0]) + bin_edges[:-1]

# First, just plot the histogram.
pylab.clf()
axis = pylab.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr)

# <demo> stop

# Now do a fit.  First you see the most generic way.  I don't particularly recommend it.  
# It's not easy to plot the fitted model.  Also, notice that it requires you
# to make initial guesses for the parameters.  Parameters are, in order:
# [FWHM, Centroid, Peak Value, Const BG Level, BG slope]
# The last is optional and can be left off

guess_params = [2.35*sigma, mu, hist.max(), 0]
fitter = mass.fitting.MaximumLikelihoodGaussianFitter(bin_ctr, hist, guess_params)
params, covariance = fitter.fit()
for i in range(len(guess_params)):
    print "Param %d: initial guess %8.4f estimate %8.4f  uncertainty %8.4f"%(i, guess_params[i], params[i], covariance[i,i]**.5)

# Compute the model function and plot it in red.
model = fitter.theory_function(params, bin_ctr)
pylab.plot(bin_ctr, model, 'r')

# <demo> stop
# Let's repeat that fit 3 ways: first with zero background, next with a constant background, 
# and finally with a sloped linear background.
# To make it interesting, let's add a Poisson background of 2 counts per bin.
# Note that we don't get a great fit when there IS a background but we don't let it be fit for. 

hist += numpy.random.poisson(lam=2.0, size=len(hist))

pylab.clf()
axis = pylab.subplot(111)
mass.plot_as_stepped_hist(axis, hist, bin_ctr, color='blue')

guess_params = [2.35*sigma, mu, hist.max()]
color='red','gold','green'
title='No BG','Constant BG','Sloped BG'
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
