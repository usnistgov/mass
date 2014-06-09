'''
Created on Jun 9, 2014

@author: fowlerj
'''

import numpy as np
import scipy as sp
import pylab as plt

def MicrocalDataSet_phase_correct(self, prange=None, times=None, plot=True):
    """Apply a correction for pulse variation with arrival phase.
    Model is a parabolic correction with cups at +-180 degrees away from the "center".
    
    prange:  use only filtered values in this range for correction 
    times: if not None, use this range of p_timestamps instead of all data (units are seconds
           since server started--ugly but that's what we have to work with)
    plot:  whether to display the result
    """
    
    # Choose number and size of bins
    phase_step=.05
    nstep = int(.5+1.0/phase_step)
    phases = (0.5+np.arange(nstep))/nstep - 0.5
    phase_step = 1.0/nstep
    
    # Default: use the calibration to pick a prange
    if prange is None:
        calibration = self.calibration['p_filt_value']
        ph_estimate = calibration.name2ph('Mn Ka1')
        prange = np.array((ph_estimate*.98, ph_estimate*1.02))

    # Estimate corrections in a few different pieces
    corrections = []
    valid = self.cuts.good()
    if prange is not None:
        valid = np.logical_and(valid, self.p_filt_value<prange[1])
        valid = np.logical_and(valid, self.p_filt_value>prange[0])
    if times is not None:
        valid = np.logical_and(valid, self.p_timestamp<times[1])
        valid = np.logical_and(valid, self.p_timestamp>times[0])

    # Plot the raw filtered value vs phase
    if plot:
        plt.clf()
        plt.subplot(211)
        plt.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value[valid],',',color='orange')
        plt.xlabel("Hypothetical 'center phase'")
        plt.ylabel("Filtered PH")
        plt.xlim([-.55,.55])
        if prange is not None:
            plt.ylim(prange)
            
    for ctr_phase in phases:
        valid_ph = np.logical_and(valid,
                                     np.abs((self.p_filt_phase - ctr_phase)%1) < phase_step*0.5)
#            print valid_ph.sum(),"   ",
        mean = self.p_filt_value[valid_ph].mean()
        median = np.median(self.p_filt_value[valid_ph])
        corrections.append(mean) # not obvious that mean vs median matters
        if plot:
            plt.plot(ctr_phase, mean, 'or')
            plt.plot(ctr_phase, median, 'vk', ms=10)
    corrections = np.array(corrections)
    assert np.isfinite(corrections).all()
    
    def model(params, phase):
        "Params are (phase of center, curvature, mean peak height)"
        phase = (phase - params[0]+.5)%1 - 0.5
        return 4*params[1]*(phase**2 - 0.125) + params[2]
    errfunc = lambda p,x,y: y-model(p,x)
    
    params = (0., 4, corrections.mean())
    fitparams, _iflag = sp.optimize.leastsq(errfunc, params, args=(self.p_filt_phase[valid], self.p_filt_value[valid]))
    phases = np.arange(-0.6,0.5001,.01)
    if plot: plt.plot(phases, model(fitparams, phases), color='blue')
    
    
    self.phase_correction={'phase':fitparams[0],
                        'amplitude':fitparams[1],
                        'mean':fitparams[2]}
    fitparams[2] = 0
    correction = model(fitparams, self.p_filt_phase)
    self.p_filt_value_phc = self.p_filt_value - correction
    self.p_filt_value_dc = self.p_filt_value_phc.copy()
    print 'RMS phase correction is: %9.3f (%6.2f parts/thousand)'%(correction.std(), 
                                        1e3*correction.std()/self.p_filt_value.mean())
    
    if plot:
        plt.subplot(212)
        plt.plot((self.p_filt_phase[valid]+.5)%1-.5, self.p_filt_value_phc[valid],',b')
        plt.xlim([-.55,.55])
        if prange is not None:
            plt.ylim(prange)

def MicrocalDataSet_auto_drift_correct_rms(self, prange=None, times=None, ptrange=None, plot=False, 
                           slopes=None, line_name="MnKAlpha"):
    """Apply a correction for pulse variation with pretrigger mean, which we've found
    to be a pretty good indicator of drift.  Use the rms width of the Mn Kalpha line
    rather than actually fitting for the resolution.  (THIS IS THE OLD WAY TO DO IT.
    SUGGEST YOU USE self.auto_drift_correct instead....)
    
    prange:  use only filtered values in this range for correction 
    ptrange: use only pretrigger means in this range for correction
    times: if not None, use this range of p_timestamps instead of all data (units are seconds
           since server started--ugly but that's what we have to work with)
    plot:  whether to display the result
    line_name: Line to calibrate on, if prange is None 
    ===============================================
    returns best_slope 
    units = 
    """
    if plot:
        plt.clf()
        axis1=plt.subplot(211)
        plt.xlabel("Drift correction slope")
        plt.ylabel("RMS of selected, corrected pulse heights")
    if self.p_filt_value_phc[0] ==0:
        self.p_filt_value_phc = self.p_filt_value.copy()
    
    # Default: use the calibration to pick a prange
    if prange is None:
        calibration = self.calibration['p_filt_value']
        ph_estimate = calibration.name2ph(line_name)
        prange = np.array((ph_estimate*.99, ph_estimate*1.01))
    
    range_ctr = 0.5*(prange[0]+prange[1])
    half_range = np.abs(range_ctr-prange[0])
    valid = np.logical_and(self.cuts.good(), np.abs(self.p_filt_value_phc-range_ctr)<half_range)
    if times is not None:
        valid = np.logical_and(valid, self.p_timestamp<times[1])
        valid = np.logical_and(valid, self.p_timestamp>times[0])
        
    if ptrange is not None:
        valid = np.logical_and(valid, self.p_pretrig_mean<ptrange[1])
        valid = np.logical_and(valid, self.p_pretrig_mean>ptrange[0])

    data = self.p_filt_value_phc[valid]
    corrector = self.p_pretrig_mean[valid]
    mean_pretrig_mean = corrector.mean()
    corrector -= mean_pretrig_mean
    if slopes is None: slopes = np.arange(-.2,.9,.05)
    rms_widths=[]
    for sl in slopes:
        rms = (data+corrector*sl).std()
        rms_widths.append(rms)
#            print "%6.3f %7.2f"%(sl,rms)
        if plot: 
            plt.plot(sl,rms,'bo')
    poly_coef = sp.polyfit(slopes, rms_widths, 2)
    best_slope = -0.5*poly_coef[1]/poly_coef[0]
    print "Drift correction requires slope %6.3f"%best_slope
    self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
    
    if plot:
        plt.subplot(212)
        plt.plot(corrector, data, ',')
        xlim = plt.xlim()
        c = np.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
        plt.plot(c, -c*best_slope + data.mean(),color='green')
        plt.ylim(prange)
        axis1.plot(slopes, np.poly1d(poly_coef)(slopes),color='red')
        plt.xlabel("Pretrigger mean - mean(PT mean)")
        plt.ylabel("Selected, uncorrected pulse heights")
    return best_slope

       
def MicrocalDataSet_auto_drift_correct(self, prange=None, times=None, plot=False, slopes=None, line_name='MnKAlpha'):
    """Apply a correction for pulse variation with pretrigger mean.
    This attempts to replace the previous version by using a fit to the
    Mn K alpha complex
    
    prange:  use only filtered values in this range for correction 
    times: if not None, use this range of p_timestamps instead of all data (units are seconds
           since server started--ugly but that's what we have to work with)
    plot:  whether to display the result
    line_name: name of the element whose Kalpha complex you want to fit for drift correction
    """

    if self.p_filt_value_phc[0] ==0:
        self.p_filt_value_phc = self.p_filt_value.copy()
    
    # Default: use the calibration to pick a prange
    if prange is None:
        calibration = self.calibration['p_filt_value']
        ph_estimate = calibration.name2ph('MnKAlpha')
        prange = np.array((ph_estimate*.99, ph_estimate*1.01))
    
    range_ctr = 0.5*(prange[0]+prange[1])
    half_range = np.abs(range_ctr-prange[0])
    valid = np.logical_and(self.cuts.good(), np.abs(self.p_filt_value_phc-range_ctr)<half_range)
    if times is not None:
        valid = np.logical_and(valid, self.p_timestamp<times[1])
        valid = np.logical_and(valid, self.p_timestamp>times[0])

    data = self.p_filt_value_phc[valid]
    corrector = self.p_pretrig_mean[valid]
    mean_pretrig_mean = corrector.mean()
    corrector -= mean_pretrig_mean
    if slopes is None: slopes = np.arange(0,1.,.09)
    
    fit_resolutions=[]
    for sl in slopes:
        self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*sl
        params,_covar,_fitter = self.fit_spectral_line(prange=prange, times=times, plot=False,
                                               fit_type='dc', line=line_name, verbose=False)
#            print "%5.1f %s"%(sl, params[:4])
        fit_resolutions.append(params[0])
#        print(fit_resolutions)
    poly_coef = sp.polyfit(slopes, fit_resolutions, 2)
#        best_slope = -0.5*poly_coef[1]/poly_coef[0] # this could be better in principle, but in practice is often way worse
    # some code to check if the best slope from the quatratic fit is reasonable, like near the minimum could
    # be used to get the best of both worlds
    # or start with a sweep then add a binary search at the end
    best_slope = slopes[np.argmin(fit_resolutions)]
    best_slope_resolution = np.interp(best_slope, slopes, fit_resolutions)
    
    print "Drift correction requires slope (using min not quadratic fit) %6.3f"%best_slope
    self.p_filt_value_dc = self.p_filt_value_phc + (self.p_pretrig_mean-mean_pretrig_mean)*best_slope
    
    if plot:
        plt.clf()
        plt.subplot(211)
        plt.plot(slopes, fit_resolutions,'go')
        plt.plot(best_slope, best_slope_resolution,'bo')
        plt.plot(slopes, np.polyval(poly_coef, slopes),color='red')
        plt.xlabel("Drift correction slope")
        plt.ylabel("Fit resolution from selected, corrected pulse heights")
        plt.title('auto_drift_correct fitting %s'%line_name)
        
        plt.subplot(212)
        plt.plot(corrector, data, ',')
        xlim = plt.xlim()
        c = np.arange(0,101)*.01*(xlim[1]-xlim[0])+xlim[0]
        plt.plot(c, -c*best_slope + data.mean(),color='green')
        plt.ylim(prange)
        plt.xlabel("Pretrigger mean - mean(PT mean)")
        plt.ylabel("Selected, uncorrected pulse heights")
        
    return best_slope, mean_pretrig_mean


