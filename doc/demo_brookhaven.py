"""
demo_brookhaven.py

A super-simple demonstration script for Randy Doriese's team going to Brookhaven
national lab.

Joe Fowler, NIST

October 20, 2011
"""

import os
import numpy, pylab, scipy, mass

def demo_plots():
    """Just make sure a plot pops up."""

    pylab.figure(1)
    x=numpy.arange(10000.)
    y=numpy.sqrt(x) + numpy.random.standard_normal(10000)
    pylab.plot(x,y,',r')
    pylab.title("Scatter plot of sqrt(x) + random number: red points")


def load_data(filename="2011_07_14_A_", 
              directory="/home/pcuser/data/dixie_2011F/2011_07_14",
              channels=(1,)):
    allfiles = [os.path.join(directory, "%schan%d.ljh"%(filename, cnum))
                for cnum in channels]
    data = mass.TESGroup(allfiles)

    print "We will now make a pass through all the data, computing rise times, peak "
    print "times and heights, etc."
    data.summarize_data()
    return data

def apply_cuts(data):
    ac_cdm=mass.controller.AnalysisControl(
            pulse_average=(100,None),    # A cut against crappy little junk
            pretrigger_rms=(None,3),    # A cut against "big tails" from prior pulses
            pretrigger_mean_departure_from_median=(-20,20),
            max_posttrig_deriv=(None, 5), # A strong pileup cut
            rise_time_ms=(0,.25),         # More or less a redundant pileup cut
            peak_time_ms=(0,.5),      # More or less a redundant pileup cut
            )
            
    print 
    print "Now we apply some basic cuts"
    for ds in data.datasets:
        ds.clear_cuts()
        ds.apply_cuts(ac_cdm)
        print "Number passing: %d of %d total records"%(
            ds.cuts.good().sum(), ds.nPulses)

def demo_summaries(data, id=0, fig=1):
    """Plot summary timeseries and histograms for channel #<id> to 
    pylab figure #<fig>."""
    pylab.figure(fig, figsize=(12,8))
    data.datasets[id].plot_summaries()


def compute_avg(data):
    # Estimate gains by averaging all pulses within 4% of the median pulse
    # (since ka/kb are ~10% separated)
    gains = []
    for ds in data.datasets:
        med = numpy.median(ds.p_peak_value[ds.cuts.good()])
        mean_kalpha = ds.p_peak_value[
            numpy.logical_and(ds.cuts.good(), 
                              numpy.abs(ds.p_peak_value-med)/med<0.04)].mean()
        gains.append(mean_kalpha)
    gains = numpy.array(gains)
    mean_ph = gains.mean()
    gains /= mean_ph
    data.set_gains(gains)
    data.mean_ph=mean_ph

    masks=data.make_masks(pulse_peak_ranges=numpy.array((0.90,1.05))*mean_ph, 
                          cut_crosstalk=False, use_gains=True)
    data.compute_average_pulse(masks, subtract_mean = True)
    pylab.figure(4)
    data.plot_average_pulses(None)
    pylab.semilogy()

def filter_and_calibrate(data, fmax=None, f_3db=None):
    data.compute_filters(fmax=fmax, f_3db=f_3db)
    data.summarize_filters()
    data.filter_data()
    data.find_named_features_with_mouse(prange=[data.mean_ph*.9, data.mean_ph*1.1])
    pylab.figure(5)
    for ds in data.datasets: ds.phase_correct(plot=True)
    for ds in data.datasets: ds.auto_drift_correct_rms()
    
    for ds in data.datasets:
        ds.calibration['p_filt_value_dc'] = ds.calibration['p_filt_value'].copy('p_filt_value_dc')

    ds = data.datasets[0]
    calib = ds.calibration['p_filt_value_dc']
    ds.fit_spectral_line(prange=numpy.array((.99,1.01))*calib.name2ph('Mn Ka1'), fit_type='dc', 
                         line='MnKAlpha', plot=True)

def final_fits(data):
    #Fit one using Kalpha and Kbeta
    pylab.figure(6)
    data.datasets[0].fit_MnK_lines(plot=True)
        
    # Now fit them all!
    for ds in data.datasets: ds.fit_MnK_lines(plot=False)
    
    # Final Kalpha fit
    pylab.figure(7)
    for ds in data.datasets: _=ds.fit_spectral_line(prange=[5850,5930], 
                                                    fit_type='energy', line='MnKAlpha')
    

def all_demos():
    demo_plots()

    data = load_data()
    demo_summaries(data, 0, fig=2)
    apply_cuts(data)
    demo_summaries(data, 0, fig=3)

    compute_avg(data)

    filter_and_calibrate(data)
    final_fits(data)
