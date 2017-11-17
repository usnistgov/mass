import numpy as np
import pylab as pl
from math import pi

from mass.core.analysis_algorithms import correct_flux_jumps

pl.ion()

failed_origs = []
failed_inputs = []
failed_outputs = []

good_inputs = []
good_outputs = []

def verify(in_vals, orig_vals, corrected):
    if not np.all(np.abs(orig_vals - corrected) < 1e-6):
        print('FAILED')
        failed_origs.append(orig_vals)
        failed_inputs.append(in_vals)
        failed_outputs.append(corrected)
        pl.clf()
        pl.plot(in_vals, '.')
        pl.plot(corrected, '.')
        pl.plot(orig_vals, '.')
        pl.title("FAILED!!!!")
        #raise Exception()
    else:
        good_inputs.append(in_vals)
        good_outputs.append(corrected)


def make_trend_linear(sz):
    b = np.random.randint(0, 2**16-1)
    m = 4*np.random.rand() - 2
    trend = b + m*(sz/2.**12)*np.arange(sz)
    return trend

def make_trend_poly(sz, deg):
    max_phi0 = 2
    p = np.zeros(deg+1)
    p[:-1] = (2*max_phi0*np.random.rand(deg) - max_phi0) * 2.**12 * (1./sz)**(np.arange(deg,0,-1))
    p[-1] = 2**14 + np.random.randint(0, 2*2**14)
    trend = np.polyval(p, np.arange(sz))
    return trend

def make_trend_poly_plus_sine(sz, deg):
    max_phi0 = 2
    p = np.zeros(deg+1)
    p[:-1] = (0.1*max_phi0*np.random.rand(deg) - 0.05*max_phi0) * 2.**12 * (1./sz)**(np.arange(deg,0,-1))
    p[-1] = 2**14 + np.random.randint(0, 2*2**14)
    trend = np.polyval(p, np.arange(sz))

    phase = 2*pi*np.random.rand()
    amp = 0.1*2**12*np.random.rand()
    freq = 20*np.random.rand()

    trend += amp*np.cos(2*pi*(1.0*np.arange(sz)/sz)*freq + phase)

    return trend

def add_jumps(vals):
    njumps = 30
    for k in range(njumps):
        start = np.random.randint(1, len(vals))
        vals[start:] += 2**12 * np.random.randint(-4, 5)
    return vals

def run_tests(N):
    sz = 10000
    g = np.full(sz, True, dtype=bool)
    for k in xrange(N):
        noise = np.abs(100*np.random.randn(sz))
        #noise += 1*np.random.standard_cauchy(sz)
        vals_orig = make_trend_poly_plus_sine(sz, 2) + noise
        vals = add_jumps(vals_orig.copy())
        new_vals = correct_flux_jumps(vals, g, 2**12)
        verify(vals, vals_orig, new_vals)

'''
run_tests(100)

if len(failed_inputs) == 0:
    pl.clf()
    pl.plot(good_inputs[0], '.')
    pl.plot(good_outputs[0], '.')
'''


