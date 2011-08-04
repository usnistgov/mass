import mass
import numpy, pylab

def load_cdm(choice="06A", demodulate=True, summarize=True):

    if demodulate:
#        demod = numpy.array((
#                         (+1, -1, +1, -1, -1, +1, -1, -1),
#                         (+1, +1, +1, +1, -1, -1, -1, +1),
#                         (+1, +1, +1, -1, +1, -1, +1, -1),
#                         (+1, -1, +1, +1, +1, +1, +1, +1),
#                         (+1, -1, -1, +1, +1, -1, -1, -1),
#                         (+1, +1, -1, -1, +1, +1, -1, +1),
#                         (+1, +1, -1, +1, -1, +1, +1, -1),
#                         (+1, -1, -1, -1, -1, -1, +1, +1))).T

        # The following comes from file 2011_05_06_E
#        demod = numpy.array([
#                [ 0.99741349,  0.99849719,  0.9978145 ,  0.97843092,  1.02064228, 1.00051355,  0.99818707,  1.00848627],
#                [-0.99780974,  1.0156951 ,  0.99467669, -0.99503637, -0.99379349, 1.00943821,  1.00665396, -0.98682619],
#                [ 0.9928538 ,  0.98966294,  1.00417909,  0.9953154 , -0.99200243,-1.00969977, -1.02158464, -0.9952904 ],
#                [-0.99890305,  0.99636633, -0.99441893,  1.02176819,  1.01593346,-0.98217259,  0.99088352, -0.99978647],
#                [-1.00275263, -0.99018161,  1.00089318,  1.00902472,  1.0196111 , 1.00608863, -0.98743624, -0.98412641],
#                [ 0.98727934, -1.00397009, -1.00858756,  0.98526407, -0.99066402, 1.01514459,  1.00493111, -1.00468611],
#                [-1.00696567, -1.00428529,  1.01057472,  0.99166728, -0.98909741,-0.98243092,  1.00455744,  1.01034975],
#                [-1.0098836 ,  0.9968309 , -0.98788222,  1.00515997, -0.99642671, 0.99550273, -0.99684002,  1.01140678]])
        # Rows mean a single TES.  Columns mean a single RS addressing line.
        
        # The following is from 2011_05_10_A
        demod = numpy.array(
          [[ 0.99695413,  0.99833182,  0.99786005,  0.97662471,  1.01988923,  1.00130642,  0.99984416,  1.00918413],
           [-0.99811238,  1.01607872,  0.99663425, -0.99050457, -0.99346857,  1.01097094,  1.00707253, -0.98706197],
           [ 0.99082581,  0.98954771,  1.00420022,  0.99733242, -0.99294089, -1.00794397, -1.02034956, -0.99736362],
           [-0.99936888,  0.99630368, -0.98823312,  1.0282532 ,  1.01287767, -0.98317271,  0.99250937, -0.9994779 ],
           [-1.00254834, -0.9860954 ,  1.00093461,  1.01858904,  1.01701903,  1.00407613, -0.98569178, -0.98549825],
           [ 0.98526344, -1.00401667, -1.00902304,  0.98672604, -0.98968943,  1.01712504,  1.00376218, -1.00490788],
           [-1.00500095, -1.00534864,  1.00661521,  0.99651946, -0.98844329, -0.98336947,  1.005574  ,  1.00903391],
           [-1.01018206,  0.99571617, -0.98637625,  1.00652913, -0.99587982,  0.9982707 , -0.99747296,  1.00953465]])
    else:  
         demod = numpy.eye(8)

    NOISEFILES = None
    if choice is None:
        DIR='/Users/fowlerj/Microcal/Data_CDM/2011_05_03/'
        DATAFILES = [DIR+"2011_05_03_pulse_A_chan%d.ljh"%i for i in 1,3,5,7,9,11,13,15]
    elif choice.startswith("05"):
        choice=choice[2:]
        DIR='/Users/fowlerj/Microcal/Data_CDM/2011_05_05/'
        DATAFILES = [DIR+"debug_%s_chan%d.ljh"%(choice,i) for i in 1,3,5,7,9,11,13,15]
    else:
        date=choice[:2]
        choice=choice[2:]
        DIR='/Users/fowlerj/Microcal/Data_CDM/2011_05_%2s/'%date
        DATAFILES = ["%s/2011_05_%s_%s_chan%d.ljh"%(DIR,date,choice,i) for i in 1,3,5,7,9,11,13,15]
        NOISEFILES = ["%s/2011_05_%s_%s_chan%d.noi"%(DIR,date,choice,i) for i in 1,3,5,7,9,11,13,15]

    if date == "06":       
        # May 6 data sets ABCD had unexplained slippage of RS 0 --> RS 5, and similar.  (or is it 5-->0?)
        if demodulate and choice in ("A","B","C","D"):
            demod = numpy.roll(demod, 5, axis=1)
            
    cdm = mass.channel_group.CDMGroup(DATAFILES, NOISEFILES, demodulation = demod,
                                        noise_only=(NOISEFILES is None))
    if summarize: cdm.summarize_data()
    
    # One example plot of 20 traces from channel 7
    i=0; cdm.plot_traces(numpy.arange(i,i+20), channum=7)
    return cdm

def cut(cdm):
    ac_cdm=mass.controller.AnalysisControl(
        peak_time_ms=(-.5,1.5),      # More or less a redundant pileup cut
        rise_time_ms=(0,1.),         # More or less a redundant pileup cut
        max_posttrig_deriv=(None, 35), # A strong pileup cut
        pretrigger_rms=(None,25),    # A cut against "big tails" from prior pulses
        pulse_average=(100,None))    # A cut against crappy little junk

    for ds in cdm.datasets: 
        ds.clear_cuts()
        ds.apply_cuts(ac_cdm)
        print 'Number passing: %d of %d total records'%(
                ds.cuts.good().sum(), ds.nPulses)

    masks=cdm.make_masks(pulse_avg_ranges=[1000,3800], cut_crosstalk=True)
    cdm.compute_average_pulse(masks, True)


def load_cdm_summer(choice="28A", demodulate=True, summarize=True):

    if demodulate:
        # The following is from 2011_05_10_A
        demod = numpy.array(
          [[ 0.99695413,  0.99833182,  0.99786005,  0.97662471,  1.01988923,  1.00130642,  0.99984416,  1.00918413],
           [-0.99811238,  1.01607872,  0.99663425, -0.99050457, -0.99346857,  1.01097094,  1.00707253, -0.98706197],
           [ 0.99082581,  0.98954771,  1.00420022,  0.99733242, -0.99294089, -1.00794397, -1.02034956, -0.99736362],
           [-0.99936888,  0.99630368, -0.98823312,  1.0282532 ,  1.01287767, -0.98317271,  0.99250937, -0.9994779 ],
           [-1.00254834, -0.9860954 ,  1.00093461,  1.01858904,  1.01701903,  1.00407613, -0.98569178, -0.98549825],
           [ 0.98526344, -1.00401667, -1.00902304,  0.98672604, -0.98968943,  1.01712504,  1.00376218, -1.00490788],
           [-1.00500095, -1.00534864,  1.00661521,  0.99651946, -0.98844329, -0.98336947,  1.005574  ,  1.00903391],
           [-1.01018206,  0.99571617, -0.98637625,  1.00652913, -0.99587982,  0.9982707 , -0.99747296,  1.00953465]])
    else:  
         demod = numpy.eye(8)

    NOISEFILES = None
    if choice is None:
        DIR='/home/pcuser/data/dixie_2011F/2011_07_01/'
        DATAFILES = [DIR+"2011_07_01_A_chan%d.dat"%i for i in 1,3,5,7,9,11,13,15]
    else:
        date=choice[:2]
        choice=choice[2:]
        if date in ("28","29","30"): 
            month='06'
        elif date in ("01","02","03","04","05","06","07","08"):
            month='07'
        DIR='/home/pcuser/data/dixie_2011F/2011_%2s_%2s/'%(month,date)
        DATAFILES = ["%s2011_%s_%s_%s_chan%d.dat"%(DIR,month,date,choice,i) for i in 1,3,5,7,9,11,13,15]
        NOISEFILES = ["%s2011_%s_%s_%s_chan%d.noi"%(DIR,month,date,choice,i) for i in 1,3,5,7,9,11,13,15]

    cdm = mass.channel_group.CDMGroup(DATAFILES, NOISEFILES, demodulation = demod,
                                        noise_only=(NOISEFILES is None))
    if summarize: cdm.summarize_data()
    
    # One example plot of 20 traces from channel 7
    i=0; cdm.plot_traces(numpy.arange(i,i+20), channum=7)
    return cdm


def cut_summer(cdm, pulse_avg_ranges=[500,650]):
    ac_cdm=mass.controller.AnalysisControl(
        peak_time_ms=(-.5,1.5),      # More or less a redundant pileup cut
        rise_time_ms=(0,1.),         # More or less a redundant pileup cut
        max_posttrig_deriv=(None, 15), # A strong pileup cut
        pretrigger_mean_departure_from_median=(-35,35),
        pretrigger_rms=(None,8),    # A cut against "big tails" from prior pulses
        pulse_average=(200,None))    # A cut against crappy little junk

    for ds in cdm.datasets: 
        ds.clear_cuts()
        ds.apply_cuts(ac_cdm)
        print 'Number passing: %d of %d total records'%(
                ds.cuts.good().sum(), ds.nPulses)

    masks=cdm.make_masks(pulse_avg_ranges=pulse_avg_ranges, cut_crosstalk=True)
    cdm.compute_average_pulse(masks, True)
    

def improve_walsh(cdm):
    cdm.plot_crosstalk(use_legend=False, xlim=[-.2,1], ylim=[-300,300])
    result =  cdm.estimate_crosstalk()
    cdm.update_demodulation(result.T)

def show_better_demod(cdm):
    # And recompute average pulses
    masks=cdm.make_masks([1500,3000], cut_crosstalk=True)
    cdm.compute_average_pulse(masks, True)
    cdm.plot_crosstalk(use_legend=False, xlim=[-.2,1], ylim=[-300,300])
    
# Greg: what we did today (Monday 5/9)
"""
import numpy, pylab, analyze8x8

cdm = analyze8x8.load_cdm("11A")
analyze8x8.cut(cdm)
#analyze8x8.improve_walsh(cdm)
cdm.compute_filters(fmax=8000) # Cut out data > 8 kHz
cdm.summarize_filters()
cdm.filter_data()

# First figure out the lovely range for the Mn Kalpha complex
ds = cdm.datasets[5]
pylab.clf(); _=pylab.hist(ds.p_filt_value, 200, [21000,22000])
# Vary the range of the 200 bins until looks right

ds.fit_mn_kalpha([21000,22000], type='filt')

# Repeat from the ds=... line for all datasets 0...7
# Voila!
"""

# May 17
"""
import numpy, pylab, analyze8x8

cdm = analyze8x8.load_cdm("18A")
analyze8x8.cut(cdm)

# Here check if avg pulse makes sense
cdm.plot_average_pulses(0)
cdm.datasets[0].plot_summaries(log=True)

cdm.compute_filters(fmax=8000) # Cut out data > 8 kHz
cdm.summarize_filters()
cdm.filter_data()

# Now work on energy calibration
cdm.find_named_features_with_mouse(name='Mn Ka1', channame='p_filt_value', xrange=[12e3,25e3])
# Optional: phase (arrival time) corrections
for ds in cdm.datasets: ds.phase_correct(prange=ds.calibration['p_filt_value'].name2ph('Mn Ka1')*numpy.array((.99,1.01)))

cdm.datasets[0].auto_drift_correct(plot=True)
for ds in cdm.datasets: 
    ds.auto_drift_correct(plot=False)
    ds.calibration['p_filt_value_dc'] = ds.calibration['p_filt_value'].copy()
    ds.calibration['p_filt_value_dc'].ph_field = 'p_filt_value_dc'

good_may18A_times=[1857410,99999999]
ds=cdm.datasets[7]
calib=ds.calibration['p_filt_value_dc']
ds.fit_spectral_line(prange=numpy.array((.99,1.01))*calib.name2ph('Mn Ka1'), type='dc', 
    line='MnKAlpha', plot=True, times=good_may18A_times)

for ds in cdm.datasets: ds.fit_MnK_lines(plot=False, times=good_may18A_times)

# Final Kalpha fit
for ds in cdm.datasets: _=ds.fit_spectral_line(prange=[5850,5930], type='energy', line='MnKAlpha')

# For plotting:
cdm.datasets[7].fit_MnK_lines(plot=True)

"""

#for ds in cdm.datasets: 
#    calib = ds.calibration['p_filt_value_dc']
#    results_a, covar = ds.fit_spectral_line(prange=numpy.array((.99,1.01))*calib.name2ph('Mn Ka1'), type='dc', line='MnKAlpha')
#    calib.add_cal_point(results_a[1], mass.energy_calibration.STANDARD_FEATURES['Mn Ka1'], name='Mn Ka1')
#    results_b, covar = ds.fit_spectral_line(prange=numpy.array((.96,1.01))*calib.name2ph('Mn Kb'), type='dc', line='MnKBeta')
#    calib.add_cal_point(results_b[1], mass.energy_calibration.STANDARD_FEATURES['Mn Kb'], name='Mn Kb')
#    # Mn Ka2
#    de = mass.energy_calibration.STANDARD_FEATURES['Mn Ka2']-mass.energy_calibration.STANDARD_FEATURES['Mn Ka1']
#    calib.add_cal_point(results_a[1]+de*results_a[2], mass.energy_calibration.STANDARD_FEATURES['Mn Ka2'], name='Mn Ka2')
#    ds.p_energy = calib(ds.p_filt_value_dc)


def compare_noise(maydata, jundata, current_units=True):
    "This compares before and after changing a feedback series resistor"
    mayfreq = maydata.datasets[0].noise_spectrum.frequencies()
    junfreq = jundata.datasets[0].noise_spectrum.frequencies()

    # First, spectrum in volts
    mayspec = [ds.noise_spectrum.spectrum()/(16384.**2) for ds in maydata.datasets]
    junspec = [ds.noise_spectrum.spectrum()/(16384.**2) for ds in jundata.datasets]
    
    # Then, if requested, convert to Amps
    if current_units:
        for m in mayspec: m/=(7438.0**2)
        for j in junspec: j/=(2197.0**2)
    
    pylab.clf()
    colors=('blue','gold','green','red','purple','cyan','#cc4400','brown')
    if current_units:
        SCALE=1e18
    else:
        SCALE=1e12
    for i,(m,j) in enumerate(zip(mayspec,junspec)):
        pylab.subplot(331+i)
        pylab.grid()
        pylab.loglog(mayfreq,m*SCALE,color='g',label='May 26 C')
        pylab.loglog(junfreq,j*SCALE,color='r',label='June 28')
        if i>=5: pylab.xlabel("Frequency (Hz)")
        if current_units:
            if i%3==0: pylab.ylabel("Power spectral density (nA$^2$/Hz)")
            pylab.ylim([.05,10])
        else:   
            if i%3==0: pylab.ylabel("Power spectral density ($\\mu$V$^2$/Hz)")
            pylab.ylim([.1,1000])
        pylab.title("TES %d"%i)
        pylab.xlim([40,1e5])
        if i==0: pylab.legend(loc='best')
        
def low_f_power_spectra(data):
    "Reshape noise arrays to get "
    fake_noise_chan=[n.copy() for n in data.noise_channels]
    pylab.clf()
    
    TOWER_FEEDBACK_RESIST=2197.0
    NA_PER_AMP=1e9
    
    for i in range(8):
        print "Working on demodulated channel %d"%i 
        nc=fake_noise_chan[i]
        nc.nPulses=2880
        nc.data = numpy.zeros((nc.nPulses,nc.nSamples),dtype=numpy.float)
        for j,n in enumerate(data.noise_channels):
            nc.data += data.demodulation[i,j]*n.data[:nc.nPulses,:]
        nc.data -= nc.data.mean()
#        nc.data.shape = (30,32*4096)
        nc.data.shape = (45,64*3072)
        nc.compute_power_spectrum(plot=False)
        ax = pylab.subplot(331+i)
        nc.plot_power_spectrum(axis=ax, color=data.colors[i], scale=NA_PER_AMP/(data.n_cdm*16384*TOWER_FEEDBACK_RESIST), sqrt_psd=True)
        pylab.title("TES %d"%i)
        if i<5: pylab.xlabel("")
        if (i%3)==0: 
            pylab.ylabel("PSD (nA/$\\sqrt{\\mathrm{Hz}}$)")
        else: 
            pylab.ylabel(" ")
        pylab.xlim([1,1e5])
        pylab.ylim([1e-2,1])
        
def show8avgpulses(data):
    pylab.clf()
    for i in range(8):
        pylab.plot(data.datasets[i].average_pulses[i],color=data.colors[i], label="TES %d"%i)
    pylab.legend(loc='upper right')



def plot_and_print_resolutions(data):
    fwhm_factor = numpy.sqrt(8*numpy.log(2))
    MNKA_ENERGY=5898.801
    predicted_res = numpy.array([fwhm_factor*numpy.sqrt(f.variances['noconst']) for f in data.filters])*MNKA_ENERGY
    
    resolutions=numpy.zeros((8,2),dtype=numpy.float)
    for i,ds in enumerate(data.datasets):
        fit,covariance = ds.fit_spectral_line(prange=[5850,5930], type='energy', line='MnKAlpha')
        resolutions[i,:]=(fit[0], numpy.sqrt(covariance[0,0]))
        
    for i in range(8):
        print 'TES %d: predicted %5.2f   achieved %5.2f +- %5.2f'%(i, predicted_res[i], resolutions[i,0], resolutions[i,1])

    pylab.clf()
    for i in range(8):
        pylab.errorbar(predicted_res[i], resolutions[i,0], yerr=resolutions[i,1], fmt='o', color=data.colors[i], label='TES %d'%i)
    pylab.legend(loc='lower right')
    
    xax=numpy.array((2.0,5.0))
    pylab.plot(xax,xax,'k--')
    pylab.plot(xax,xax*1.1,'--',color='gray')
    pylab.title("Achieved versus predicted resolutions")
    pylab.ylabel("Achieved resolution at Mn K$\\alpha$ (eV)")
    pylab.xlabel("Predicted resolution at Mn K$\\alpha$ (eV)")
    pylab.grid()
    pylab.ylim([2,5.5])


def do_all_jun30_analysis(setlabel="A"):
    data = load_cdm_summer("29"+setlabel)
    par={'A':(500,650),
         'B':(750,1000)}[setlabel]
    cut_summer(data, pulse_avg_ranges=par)
    
    data.compute_filters()
    data.summarize_filters()
    data.filter_data()
    
    do_all_jun30_analysis_secondhalf(data, setlabel)
    return data
    
def do_all_jun30_analysis_secondhalf(data, setlabel):
    xrange={'A':[4000,6000],
            'B':[5000,7000]}[setlabel]
    data.find_named_features_with_mouse(name='Mn Ka1', channame='p_filt_value', xrange=xrange)

    # Do drift correction and copy calibration from filtered to drift-corrected filtered case.
    for ds in data.datasets: 
        ds.auto_drift_correct(plot=False)
        ds.calibration['p_filt_value_dc'] = ds.calibration['p_filt_value'].copy()
        ds.calibration['p_filt_value_dc'].ph_field = 'p_filt_value_dc'

    # Just an example fit
    ds = data.datasets[3]
    calib = ds.calibration['p_filt_value_dc']
    ds.fit_spectral_line(prange=numpy.array((.99,1.01))*calib.name2ph('Mn Ka1'), type='dc', 
                         line='MnKAlpha', plot=True)

    # Now fit them all!
    for ds in data.datasets: ds.fit_MnK_lines(plot=False)
    
    # Final Kalpha fit
    for ds in data.datasets: _=ds.fit_spectral_line(prange=[5850,5930], type='energy', line='MnKAlpha')
    
    # For plotting:
    data.datasets[7].fit_MnK_lines(plot=True)
    return data


def plot_all_jun29_results(data, label):
    
    data.plot_noise(scale_factor=1e9/(8.0*16384*2197.0), sqrt_psd=True)
    pylab.title("Demodulated TES noise sqrt(power spectrum)")
    pylab.ylabel("$\\sqrt{PSD}$ (nA/$\\sqrt{\\mathrm{Hz}}$)")
    pylab.ylim([.03,.5])
    ticks = [.03,.05,.07,.1,.2,.3,.4,.5]
    pylab.yticks(ticks, ['%.2f'%g for g in ticks])
    pylab.savefig("jun29_%s_noise_onepanel.png"%label)
 
    low_f_power_spectra(data)
    pylab.savefig("jun29_%s_noise_lowfreq.png"%label)
    
    show8avgpulses(data)
    pylab.savefig("jun29_%s_average_pulses.png"%label)
    
    plot_and_print_resolutions(data)
    pylab.savefig("jun29_%s_resolutions.png"%label)
    
    
def plot_7photon_record(data):
    "data must be June 30A"
    data.plot_traces(19669) # to load into cache
    N=19669-data._cached_pnum_range[0]
    
    data.ms = (numpy.arange(data.nSamples)- data.nPresamples)*data.timebase*1e3

    pylab.clf()
    for i in range(8):
        pylab.plot(data.ms, data.raw_channels[i].data[N,:], color=data.colors[i], label="RS%d"%i)
    pylab.legend(loc='upper left')
    pylab.title("Modulated CDM data Jun 30A record 19669")
    pylab.ylabel("Feedback value")
    pylab.xlabel("Time past trigger (ms)")
    pylab.savefig("seven_photons_cdm8_modulated.png")
    pylab.savefig("seven_photons_cdm8_modulated.pdf")
    
    pylab.clf()
    for i in range(8):
        pylab.plot(data.ms, data.datasets[i].data[N,:]-data.datasets[i].p_pretrig_mean[19669], color=data.colors[i], label="TES %d"%i)
    pylab.legend(loc='upper left')
    pylab.title("Demodulated CDM data Jun 30A record 19669")
    pylab.ylabel("Feedback value")
    pylab.xlabel("Time past trigger (ms)")
    pylab.savefig("seven_photons_cdm8_demodulated.png")
    pylab.savefig("seven_photons_cdm8_demodulated.pdf")


def cut_stats(data):
    'This removes any cut#31 (against multi-hit crosstalk events), so beware'
    print 'Total pulse records divided by %d is   %d'%(data.n_cdm, data.nPulses / data.n_cdm)
    ng = ns = 0
    for i,ds in enumerate(data.datasets):
        ds.cuts.clearCut(31) 
        ngood = ds.cuts.good().sum()
        nsingle = numpy.logical_and(ds.cuts.good(), data.nhits==1).sum()
        print 'TES %d: # records %d, records w/ 1 hit %d.  %% not cut: %.3f.  %% of these single: %.3f'%(
                       i,ngood,nsingle, 100.*ngood*8/data.nPulses,100.*nsingle/ngood)
        ng += ngood
        ns += nsingle
    print '%d single hits out of %d good records. Not cut: %.3f%%  Single overall: %.3f%%'%(ns, ng, 100.*ng/data.nPulses, 100.*ns/data.nPulses)


"""
# Notes for Randy

# Standard system things
import numpy, pylab
from pylab import clf, plot

# Lives in trunk/python/mass
import mass    

# I have to send you this one
import analyze8x8

data01B=analyze8x8.load_cdm_summer("01B")

pylab.ioff()

#Maybe you want to look at some summary quantities
det=0
data01B.datasets[det].plot_summaries(log=True)
pylab.show()

# Goal: to figure out a range of pulse average (integral) that will grab all 8 Mn Ka lines.
clf()  # To clear the figure
for i in range(8): 
    pylab.subplot(331+i)
    _=pylab.hist(data01B.datasets[i].p_pulse_average,100,[100,1200])
    
# We learned that [700,1100] is good.  Use it
analyze8x8.cut_summer(data01B, pulse_avg_ranges=[700,1100])
# should comment out the last line of cut_summer (line cdm.compute...) until rel. sure of cuts

# What are the cut statistics
analyze8x8.cut_stats(data)
# Careful!  If at least 75% aren't surviving, that means you should go back to summary quantities and try again on the cuts.

# Okay, let's check average pulse.
analyze8x8.show8avgpulses(data)

#Want to update for semilog?
pylab.semilogy()

# you can see the response of det number det to photons in all 8 sensors
data.plot_average_pulses(det)  # and use mouse to zoom near zero y-values

# Let's look at noise
data.plot_noise()  # this is down to ~50 Hz

# Or the PSD rebinned to reach 1 Hz
analyze8x8.low_f_power_spectra(data)

data.compute_filters()
data.summarize_filters()
data.filter_data()

# Bravo.  We're on the home stretch.  Vary the xrange until you hit all 8 Mn Ka peaks, without having
# zoomed out too much
data.find_named_features_with_mouse(name='Mn Ka1', channame='p_filt_value', xrange=[2000,7000])
data.find_named_features_with_mouse(name='Mn Ka1', channame='p_filt_value', xrange=[3200,5000])

# Let's do a test drift correction
data.datasets[det].auto_drift_correct(plot=True)   # optionally use ds.auto_drift_correct_rms

for ds in data.datasets: 
    ds.auto_drift_correct(plot=False)   # optionally use ds.auto_drift_correct_rms
    ds.calibration['p_filt_value_dc'] = ds.calibration['p_filt_value'].copy()
    ds.calibration['p_filt_value_dc'].ph_field = 'p_filt_value_dc'

ds = data.datasets[3]
calib = ds.calibration['p_filt_value_dc']
ds.fit_spectral_line(prange=numpy.array((.99,1.01))*calib.name2ph('Mn Ka1'), type='dc', 
                     line='MnKAlpha', plot=True)

#Fit one using Kalpha and Kbeta
ds.fit_MnK_lines(plot=True)

# Now fit them all!
for ds in data.datasets: ds.fit_MnK_lines(plot=False)

# Final Kalpha fit
for ds in data.datasets: _=ds.fit_spectral_line(prange=[5850,5930], type='energy', line='MnKAlpha')

# For plotting:
data.datasets[7].fit_MnK_lines(plot=True)

# The final report on achieved vs predicted
analyze8x8.plot_and_print_resolutions(data)

# Bummer.  Not what we hoped.  But wait!  There's more!!!!
# Let's cut requiring nhits==1
analyze8x8.cut_stats(data)
HITCUT=31
for ds in data.datasets: 
    ds.cuts.cut(HITCUT, data.nhits != 1)

analyze8x8.plot_and_print_resolutions(data)

# undo 
for ds in data.datasets:
    ds.cuts.clearCut(HITCUT)

analyze8x8.plot_and_print_resolutions(data)

"""
