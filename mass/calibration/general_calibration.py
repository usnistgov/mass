import pylab, numpy, scipy
import scipy.signal
import cPickle
import mass
import os.path

def log_and(x, y, *args):
    """A stupendously useful function that ought to be in numpy, but isn't.
    Returns the logical and of all 2+ arguments."""
    result = numpy.logical_and(x,y)
    for a in args:
        result = numpy.logical_and(result, a)
    return result



class GeneralCalibration(object):
    """Object to perform pulse height calibration using XSI source data from
    December 18, 2012.  There are 3 separate files with targets Mn, Fe, and
    Cu (this last is the nominal titanium target, but it was almost completely
    covered by copper tape).

    It's complicated, but maybe it can be a model for 
    """
    
    
    def __init__(self, noise_filename, pulse_filename, dataset_number_subset = [0,1]):
        channels = []
        # make sure file names have %d in place of a specific channel number
        pulse_filename=pulse_filename.replace(pulse_filename[pulse_filename.rfind('chan'):pulse_filename.rfind('.')],'chan%d')
        noise_filename=noise_filename.replace(noise_filename[noise_filename.rfind('chan'):noise_filename.rfind('.')],'chan%d')
        
        minSize = 2127
        for channel in numpy.array(dataset_number_subset)*2+1:
            if os.path.isfile(pulse_filename%channel) and os.path.isfile(noise_filename%channel):
                if os.path.getsize(pulse_filename%channel) >= minSize and os.path.getsize(noise_filename%channel) >= minSize:
                    channels.append(channel)
                else:
                    print('excluded channel %d because %s is size %d and %s is size %d, minSize is %d'%(channel, pulse_filename%channel,os.path.getsize(pulse_filename%channel), noise_filename%channel, os.path.getsize(noise_filename%channel), minSize ))
        noise_files=[noise_filename%c for c in channels]
        pulse_files=[pulse_filename%c for c in channels]
        
        if len(channels)>0:
            self.data = mass.TESGroup(pulse_files, noise_files)
        else:
#            channel = dataset_number_subset[0]*2+1
#            print('couldnt find '+ pulse_filename%channel + 'and/or '+ noise_filename%channel)
            print pulse_filename%(dataset_number_subset[0]*2+1)
            print noise_filename%(dataset_number_subset[0]*2+1)
            raise ValueError('WARNING no files had both noise and pulse files')

    def copy(self):
        """This trick is pretty useful if you are going to update the object (e.g.,
        by adding new methods, or fixing bugs in existing methods), but you don't
        want to lose all the computations you already did.  In some objects, this
        has to handle deep copying, or other subtleties that I forget."""
        c = generalCalibration()
        c.__dict__.update(self.__dict__)
        return c

    def do_basic_computation(self, doFilter = True, forceNew=False, **kwargs):
        """This covers all the operations required to analyze and store the data
        from the December 18, 2012 XSI calibrations."""
        self.data.summarize_data_tdm(peak_time_microsec=420.0, forceNew=forceNew)
        self.apply_cuts(**kwargs)
        # check to see if filtered data was loaded, this logic really should be in the various functions like compute_noise_spectra
        # but its a lot of work to put it in there
        if doFilter:
            numfilters = 0
            for ds in self.data:
                if ds.filter != {}:
                    numfilters+=1
            if (not numfilters == self.data.num_good_channels) or forceNew:
                self.data.compute_noise_spectra()
                self.data.plot_noise()
                self.compute_model_pulse()
                self.data.compute_filters(f_3db=6000.)
                self.data.summarize_filters()
            else:
                print('not calculating filters because they are already loaded')
            self.data.filter_data_tdm(forceNew=forceNew)
        
    def channel_histogram(self, channel=1, driftCorrected = False):
        if channel in self.data.channel.keys():
            ds = self.data.channel[channel]
        else:
            raise ValueError('channel %d doesnt exist'%channel)
        for (i, ds) in enumerate(self.data.datasets):
            if i == dataset_num:
                if driftCorrected == False:
                    data = ds.p_filt_value
                elif driftCorrected == True:
                    data = ds.p_filt_value_dc
                ph_bin_edges = numpy.arange(0,numpy.max(data),2)
                pylab.clf()
                histout = pylab.hist(data, ph_bin_edges)
                pylab.xlabel('pulse height (arb)')
                pylab.ylabel('counts per %d unit bin'%(ph_bin_edges[1]-ph_bin_edges[0]))
                
                
    def channel_findpeaks(self, channum = 1 , whichCalibration = 'p_filt_value', doPlot=False):
        ds = self.data.channel[channum]
        data = ds.__dict__[whichCalibration][ds.cuts.good()]
        histogramMax = numpy.nanmin((2**14,numpy.max(data)*1.1))
        if histogramMax <= 0 or numpy.isnan(histogramMax):
            self.data.set_chan_bad(channum, 'in channel_findpeaks with %s histogramMax = %s'%(whichCalibration, str(histogramMax)))
            return [],[]

        ph_bin_edges = numpy.arange(0,histogramMax,2)
        counts, ph_bin_edges = numpy.histogram(data, ph_bin_edges)
        ph_bin_centers = (ph_bin_edges[:-1]+ph_bin_edges[1:])/2
        peak_indexes = numpy.array(scipy.signal.find_peaks_cwt(counts, numpy.arange(1,20,1), min_snr=4))
        try:
            sort_index = numpy.argsort(counts[peak_indexes])
        except:
            self.data.set_chan_bad(channum, 'channel_findpeaks with %s had error with peak_indexes = '%whichCalibration+str(peak_indexes))
            return [], []
        peak_indexes = peak_indexes[sort_index]

        if doPlot == True:
            pylab.clf()
            pylab.plot(ph_bin_centers, counts)
            pylab.plot(ph_bin_centers[peak_indexes], counts[peak_indexes],'r.')
            pylab.xlabel('pulse height (arb)')
            pylab.ylabel('counts per %d unit bin'%(ph_bin_edges[1]-ph_bin_edges[0]))

        return ph_bin_centers[peak_indexes], counts[peak_indexes]

    def how_many_records(self):
        for i,c in enumerate(self.CHANS):
            print "Chan %3d: "%c,
            for d in self.alldata:
                ds=d.datasets[i]
                print "%6d  "%ds.nPulses,
            print
    
    def apply_cuts(self, timestampCuts = (None, None), pretrigger_departure_cuts = (-40,40), pulse_average_cuts = (5.0, None),
                   timestamp_diff_sec_cuts = (0.007, None)):
        self.cuts = mass.AnalysisControl()
        self.cuts.cuts_prm.update({
                 'max_posttrig_deriv': (None, 60.0),
                 'pretrigger_mean_departure_from_median': pretrigger_departure_cuts,
                 'pretrigger_rms': (None, 10.0),
                 'pulse_average': pulse_average_cuts,
                 'rise_time_ms': (None, None),
                 'peak_time_ms': (None, None),
                 'timestamp_sec': timestampCuts,
                 'timestamp_diff_sec': timestamp_diff_sec_cuts, 
                 'min_value': None,})

        for ds in self.data:
            try:
                ds.clear_cuts()
                ds.apply_cuts(self.cuts)
            except:
                self.data.set_chan_bad(ds.channum, 'fails apply cuts, probably havent summarized data yet')
    

    def plot_pulse_timeseries(self, channel=1, type='ph'):
        """Look at one TES over all 3 data sets.  Plot the quantity of
        interest (see code) versus time."""
        pylab.clf()

        ds = self.data.channel[channel]
        g = ds.cuts.good()
        t = ds.p_timestamp# - self.data.timestamp_offset
        if type.lower() in ('ph', 'pv'):
            y = ds.p_peak_value
        elif type.lower() in ('pa','avg'):
            y = ds.p_pulse_average
        elif type.lower() in ('filt',):
            y = ds.p_filt_value
        elif type.lower() in ('dc'):
            y = ds.p_filt_value_dc
        elif type.lower().startswith("en"):
            y = ds.p_energy
        else:
            raise ValueError("type is not in the allowed set: PH, PA, FILT, DC, or ENERGY")
        pylab.plot(t[g], y[g], '.')
            

    def compute_model_pulse(self, timelims = (0, 1e9)):
        "Find median pulses pulses"

        gains = []
        for ds in self.data:
            use = log_and(ds.cuts.good(), ds.p_timestamp>timelims[0], ds.p_timestamp<timelims[1])
            median_pulse_average = numpy.median(ds.p_pulse_average[use])
            gains.append(median_pulse_average / 1)
        
        avg_masks = self.data.make_masks([0.9,1.1], use_gains=True, gains=gains)
        self.data.compute_average_pulse(avg_masks)
        self.data.plot_average_pulses(None)

    def apply_filter(self):
        print('applying filter')
        self.data.filter_data_tdm('filt_noconst')


    
    
    def drift_correct_new(self, line_names = ['MnKAlpha'], power = 0.1, doPlot = False):
        print('starting drift_correct_new')
        # currently only uses the last item in line_names
        if type(line_names) != type(list()): line_names = [line_names]
        self.dc_slope = {}
        self.dc_meanpretrigmean = {}
        self.dc_power = power
        for ds_num, ds in enumerate(self.data):
            for i, line_name in enumerate(line_names):
                try:
                    minE,maxE =  ds.calibration['p_filt_value'].name2ph(line_name)*numpy.array([0.9905, 1.013])
                except:
                    self.data.set_chan_bad(ds.channum, 'failed name2ph for %s in drift_correct_new'%line_name)
                    break
                use = log_and(ds.cuts.good(), ds.p_filt_value>minE, ds.p_filt_value<maxE)
                mean_pretrig_mean = numpy.mean(ds.p_pretrig_mean[use])
                ds.p_filt_value[ds.p_filt_value<0]=0 # so the exponent doesn't throw an error
                corrector = (ds.p_pretrig_mean-mean_pretrig_mean)*(ds.p_filt_value**power)
                
                try:
                    pfit = numpy.polyfit(corrector[use], ds.p_filt_value[use],1)
                except:
                    self.data.set_chan_bad(ds.channum, 'failed drift_correct_new on pfit, use.sum()=%d'%(use.sum()))
                    break
                slopes = -pfit[0]*numpy.linspace(0,2,20)
                fitter = mass.__dict__['%sFitter'%line_name]()
                resolution = numpy.zeros_like(slopes)
                if doPlot: pylab.figure()
                for j,slope in enumerate(slopes):
                    corrected = ds.p_filt_value + corrector*slope
                    contents, bins = numpy.histogram(corrected[use], bins=numpy.arange(minE, maxE, 1))
                    bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
                    try:
                        param, covar = fitter.fit(contents, bin_ctrs, plot=False)
                    except:
                        self.data.set_chan_bad(ds.channum, 'failed fit for %s in drift_correct_new'%line_name)
                        break
                    #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                    #energy scale factor (counts/eV), amplitude, background level (per bin),
                    #and background slope (in counts per bin per bin) ]
                    if doPlot:
                        pylab.plot(corrector[use], corrected[use],'.', label='%f'%slope)
                        pylab.title('%f'%slope)
                    
                    res = param[0]
                    dres = covar[0,0]**0.5
                    ph = param[1]
                    dph = covar[1,1]**0.5
                    resolution[j] = res
#                    pylab.plot(corrector[use], corrected[use], '.')

            if doPlot: 
                pylab.plot(numpy.linspace(pylab.xlim()[0], pylab.xlim()[1]), pylab.polyval(pfit, numpy.linspace(pylab.xlim()[0], pylab.xlim()[1])))
                pylab.xlabel('baseline - mean baseline')
                pylab.ylabel('drift corrected pulseheight')
                pylab.legend()
                pylab.show()
                
                pylab.figure()
                pylab.plot(slopes, resolution,'o')
                pylab.ylabel('resolution (eV)')
                pylab.xlabel('slopes')
            if not ds.channum in self.data._bad_channums:
                ds.drift_correct_info = {}
                ds.drift_correct_info['type'] = 'ptmean_power'
                ds.drift_correct_info['slope'] = slopes[numpy.argmin(resolution)]
                ds.drift_correct_info['meanpretrigmean'] = mean_pretrig_mean
                ds.drift_correct_info['power'] = power
                ds.drift_correct_info['best_achieved_resolution'] = numpy.amin(resolution)
                corrector = (ds.p_pretrig_mean-ds.drift_correct_info['meanpretrigmean'])*(ds.p_filt_value**ds.drift_correct_info['power'])
                ds.p_filt_value_dc = ds.p_filt_value+corrector*ds.drift_correct_info['slope']
                print('drift_correct_new chan %d, %s, dc slope %.3f, best res %.2f, power %.1f'%(ds.channum, line_name, ds.drift_correct_info['slope'], ds.drift_correct_info['best_achieved_resolution'],power))
        
    def apply_stored_drift_correct(self, max_shift_mean_pretrig_mean = 100):
        print('apply_stored_drift_correct')
        for ds_num, ds in enumerate(self.data):
            try:
                print('chan %d, dc_meanpretrigmean %.2f, dc_slope, %.2f dc_power %.2f'%(ds.channum, ds.drift_correct_info['meanpretrigmean'], ds.drift_correct_info['slope'], ds.drift_correct_info['power']))
                current_mean_pretrig_mean = numpy.mean(ds.p_pretrig_mean[ds.cuts.good()])
                if abs(current_mean_pretrig_mean-ds.drift_correct_info['meanpretrigmean'])>max_shift_mean_pretrig_mean:
                    self.data.set_chan_bad(ds.channum, 'stored drift correct had meanpretrigmean %d, dataset has %d'%(ds.drift_correct_info['meanpretrigmean'], current_mean_pretrig_mean))
                    continue
                corrector = (ds.p_pretrig_mean-ds.drift_correct_info['meanpretrigmean'])*(ds.p_filt_value**ds.drift_correct_info['power'])
                ds.p_filt_value_dc = ds.p_filt_value+corrector*ds.drift_correct_info['slope']              
            except:
                self.data.set_chan_bad(ds.channum, 'failed apply_drift_correct')
        
    def calibrate_approximately(self, line_names = ['MnKAlpha', 'MnKBeta'], whichCalibration = 'p_filt_value', doPlot = False, minPulses = 80, append_to_cal = True):
        """Element names must be in order of peak height, only works with kAlphas for now"""
        if type(line_names) != type(list()): line_names = [line_names]
        line_known_energies = [mass.energy_calibration.STANDARD_FEATURES[line_name] for line_name in line_names]
        line_names_energy_order = [line_names[line_known_energies.index(energy)] for energy in sorted(line_known_energies)]
        for ds in self.data:
            if ds.calibration.has_key(whichCalibration) and append_to_cal:
                cal = ds.calibration[whichCalibration] # add to existing cal if it exists
            else:
                cal = mass.calibration.EnergyCalibration(whichCalibration)
                ds.calibration[whichCalibration] = cal
            if ds.cuts.good().sum() > minPulses:
                peak_location_pulseheights,peak_counts = self.channel_findpeaks(ds.channum,whichCalibration=whichCalibration, doPlot=doPlot)
            else:
                self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley with %s has %d pulses total < %d the minimum'%(whichCalibration, ds.cuts.good().sum(), minPulses))
                continue
            if len(peak_location_pulseheights) < len(line_names):
                self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley with %s, num peaks %d < %d line_names'%(whichCalibration, len(peak_location_pulseheights), len(line_names)))                
                continue
            toCalibrate = ds.__dict__[whichCalibration]
            peak_location_pulseheights = numpy.sort(peak_location_pulseheights[-len(line_names_energy_order):])
            for i, line_name in enumerate(line_names_energy_order): 
                line_location_guess = peak_location_pulseheights[i]
                use = log_and(ds.cuts.good(), numpy.abs(toCalibrate/line_location_guess-1.0)<0.012)
                if use.sum() < minPulses:
                    self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley with %s %s has %d pulses < %d the minimum'%(whichCalibration, line_name, use.sum(), minPulses))
                    break
                line_location = scipy.stats.scoreatpercentile(toCalibrate[use], 67.)
                cal.add_cal_point(line_location, '%s'%line_name, pht_error=2)
                print('calibrate_approximately %s chan %d added %s at %.2f'%(whichCalibration, ds.channum, line_name, line_location))
                try:
                    line_location_from_cal = cal.name2ph(line_name)
                except:
                    self.data.set_chan_bad(ds.channum, 'calibrate_approximatley with %s failed name2ph after %s (probably bad calibration point added, not neccesarily the most recent point)'%(whichCalibration, line_name))
                    break
                if numpy.abs(line_location/cal.name2ph(line_name)-1) > 0.01:
                    self.data.set_chan_bad(ds.channum,'calibrate_approximatley %s numpy.abs(line_location/cal.name2ph(''%s'')-1) > 0.01'%(whichCalibration, line_name))
                    break       
                
        print('calibrate_approximately with %s  %d of %d datasets survived'%(whichCalibration, self.data.num_good_channels, self.data.n_channels))

        self.convert_to_energy(whichCalibration)
        
        
    def check_interpolated_energy_cal(self, element_name = 'V', doPlot = True):
        self.all_energies = numpy.array([])
        fitter = mass.__dict__['%sKAlphaFitter'%element_name]()
        self.fitKa1EnergyError = numpy.array([])
        self.fitResolution = numpy.array([])
        numDetectorsUsed = 0
        for (ds_num,ds) in enumerate(self.data.datasets):
            numDetectorsUsed+=1
            self.all_energies = numpy.hstack((self.all_energies, ds.p_energy[ds.cuts.good()]))
            use = log_and(ds.cuts.good(), 
                  numpy.abs(ds.p_energy/mass.calibration.energy_calibration.STANDARD_FEATURES['%s Ka1'%element_name]-1)<0.003)
            contents, bins = numpy.histogram(ds.p_energy[use], 150)
            bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
            try:
                param, covar = fitter.fit(contents, bin_ctrs, plot=True)
                #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                #energy scale factor (counts/eV), amplitude, background level (per bin),
                #and background slope (in counts per bin per bin) ]
            except RuntimeError:
                print 'Cannot fit'
            self.fitResolution = numpy.hstack((self.fitResolution, param[0]))
            self.fitKa1EnergyError = numpy.hstack((self.fitKa1EnergyError, 
              param[1]-mass.calibration.energy_calibration.STANDARD_FEATURES['%s Ka1'%element_name]))
            
        use = numpy.abs(self.all_energies/mass.calibration.energy_calibration.STANDARD_FEATURES['%s Ka1'%element_name]-1)<0.003
        contents, bins = numpy.histogram(self.all_energies[use], 150)
        bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])      
        try:
            pylab.figure()
            param, covar = fitter.fit(contents, bin_ctrs, plot=True)
            pylab.title('combine %s fit, %d detectors'%(element_name,numDetectorsUsed))
            #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
            #energy scale factor (counts/eV), amplitude, background level (per bin),
            #and background slope (in counts per bin per bin) ]
        except RuntimeError:
            print 'Cannot fit'

        
            
    
    def fit_one_ka_line(self, ds_num=0, elementName = 'Ti'):
        ds = self.data.datasets[ds_num]
        fitter = mass.__dict__[elementName+'KAlphaFitter']()
        use = log_and(ds.cuts.good(),
                  numpy.abs(ds.p_filt_value/ds.calibration['p_filt_value'].name2ph(elementName+' Ka1')-1.0)<0.002)
        contents, bins = numpy.histogram(ds.p_filt_value_dc[use], 200)
        bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
        try:
            param, covar = fitter.fit(contents, bin_ctrs, plot=True)
            #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
            #energy scale factor (counts/eV), amplitude, background level (per bin),
            #and background slope (in counts per bin per bin) ]
        except RuntimeError:
            print 'Cannot fit'
            
        res = param[0]
        dres = covar[0,0]**0.5
        ph = param[1]
        dph = covar[1,1]**0.5
        print "Resolution is %.2f +- %.2f eV"%(res,dres)
        return param
    
    def calibrate_carefully(self,lines_name = ['MnKAlpha', 'MnKBeta'], whichCalibration = 'p_filt_value_dc', doPlot = False, energyRangeFracs=[0.995, 1.005], append_to_cal=True, whichFiltValue = None):
        if type(lines_name) != type(list()): lines_name = [lines_name]

        print('calibrate_carefully %s'%whichCalibration)
        for ds in self.data:
            if ds.calibration.has_key(whichCalibration) and append_to_cal:
                cal = ds.calibration[whichCalibration] # add to existing cal if it exists
            else:
                cal = mass.calibration.EnergyCalibration(whichCalibration)
                ds.calibration[whichCalibration] = cal
            if whichFiltValue is None:
                if '_scaled' in whichCalibration:
                    whichFiltValue = whichCalibration[:whichCalibration.find('_scaled')]
                else:
                    whichFiltValue = whichCalibration
            toCalibrate = ds.__dict__[whichFiltValue]
            for i, line_name in enumerate(lines_name):


                if line_name[-5:]=='Alpha' or line_name[-4:]=='Beta': # its a KBeta or KAlpha
                    if mass.energy_calibration.STANDARD_FEATURES.has_key(line_name):
                        try:
                            if whichCalibration == 'p_energy':
                                minE, maxE = mass.energy_calibration.STANDARD_FEATURES[line_name]*numpy.array(energyRangeFracs)
                            else:
                                minE, maxE = ds.calibration['p_filt_value'].name2ph('%s'%line_name)*numpy.array(energyRangeFracs)
                        except:
                            self.data.set_chan_bad(ds.channum,'failed calibrate_carefully %s %s failed name2ph (probably brentq)'%(whichCalibration, line_name))
                            break
                    else:
                        print('%s does not exist in mass.energy_calibration.STANDARD_FEATURES'%line_name)
                        break
                    if maxE-minE < 5:
                        self.data.set_chan_bad(ds.channum,'failed calibrate_carefully %s %s for too small energy range (maxE, minE)=(%.2f, %.2f)'%(whichCalibration, line_name, maxE, minE))
                        break
                    contents, bins = numpy.histogram(toCalibrate[ds.cuts.good()], bins=numpy.arange(minE, maxE, 1))
                    bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
                    fitter = mass.__dict__['%sFitter'%line_name]()
                    try:
                        energyScaleGuess = (cal.energy2ph(maxE)-cal.energy2ph(minE))/(maxE-minE)
                    except:
                        self.data.set_chan_bad(ds.channum,'calibrate_carefuly energyScaleGuess minE=%f, maxE=%f'%(minE, maxE))
                    amplitudeGuess = contents.max()/0.13
                    phGuess = bins[numpy.argmax(contents)]
                    quarterLen = len(contents)/4
                    if quarterLen <=3: # this probably wont work anyway since contents is so short
                        background = 0.1
                        background_slope = 0.0                        
                    else:
                        background = contents[0:quarterLen].mean()
                        background_slope = (contents[-quarterLen:].mean()-background)/float(len(contents))
                    try:
                        if doPlot: pylab.figure()
                        hold = []
                        if line_name[-4:]=='Beta':
                            hold = [2,4,5] # simplify the fitting for kBeta by holding energy scale factor (degenerate with resolution with only 1 line), and background level and slope
                        paramGuess = [4.0,phGuess, energyScaleGuess,amplitudeGuess, background, background_slope ]
                        param, covar = fitter.fit(contents, bin_ctrs, plot=doPlot, hold=hold, params=paramGuess)
                        #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                        #energy scale factor (pulseheights/eV), amplitude, background level (per bin),
                        #and background slope (in counts per bin per bin) ]
                    except RuntimeError:
                        self.data.set_chan_bad(ds.channum, 'failed calibrate_carefully %s %s with RuntimeError'%(whichCalibration, line_name))
                        break
                    except:
                        self.data.set_chan_bad(ds.channum, 'failed calibrate_carefully %s %s'%(whichCalibration, line_name))
                        break
                        
                    res = param[0]
                    dres = covar[0,0]**0.5
                    ph = param[1]
                    dph = covar[1,1]**0.5
                    try:
                        cal.add_cal_point(ph, line_name, pht_error=dph, info = {'resolution':res, 'dres':dres, 'fitparams':param})
                        print('%s chan %d, %s, ph %.1f, dph %.3f, resolution %.2f +- %.2f eV'%(whichCalibration, ds.channum, line_name, ph, dph, res, dres))
                    except:
                        self.data.set_chan_bad(ds.channum, 'failed add_cal_point %s ph=%s line_name=%s pht_error=%s'%(whichCalibration, str(ph), line_name, str(dph) ))
                elif line_name[-4:]=='Edge':
                    if whichCalibration == 'p_energy':
                        minE, maxE = mass.energy_calibration.STANDARD_FEATURES[line_name]*numpy.array(energyRangeFracs)
                    else:
                        try:
                            minE, maxE = ds.calibration['p_filt_value'].name2ph('%s'%line_name)*numpy.array(energyRangeFracs)
                        except:
                            self.data.set_chan_bad(ds.channum,'failed calibrate_carefully %s %s failed name2ph (probably brentq)'%(whichCalibration, line_name))
                            break
                    edge_energy = mass.energy_calibration.STANDARD_FEATURES[line_name]
                    contents, bins = numpy.histogram(toCalibrate[ds.cuts.good()], bins=numpy.arange(minE, maxE, 3))
                    bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
    #                try:
                    pfit = numpy.polyfit(bin_ctrs, contents, 3)
                    edgeGuess = numpy.roots(numpy.polyder(pfit,2))
                    preGuess, postGuess = numpy.sort(numpy.roots(numpy.polyder(pfit,1)))
#                    if not (bin_ctrs[0]<edgeGuess<bin_ctrs[-1] and numpy.polyval(numpy.polyder(pfit,1),edgeGuess)<0 and bin_ctrs[0]<preGuess<bin_ctrs[-1] and bin_ctrs[0]<postGuess<bin_ctrs[-1]):
#                        self.data.set_chan_bad(ds.channum, 'failed edge calibration rough guess')
#                        continue
    
                    pGuess = numpy.array([edgeGuess, numpy.polyval(pfit,preGuess), numpy.polyval(pfit,postGuess),10.0],dtype='float64')
                    try:
                        pOut = scipy.optimize.curve_fit(self.edgeModel, bin_ctrs, contents, pGuess)
                    except:
                        self.data.set_chan_bad(ds.channum, 'failed fit for edgeModel')
                        break
                    (edgeCenter, preHeight, postHeight, width) = pOut[0]
    #                refitEdgeModel = lambda x,edgeCenterL: self.edgeModel(x,edgeCenterL, preHeight, postHeight, width)
    #                pOut2 = scipy.optimize.curve_fit(refitEdgeModel, bin_ctrs, contents, 
    #                                [edgeCenter]) # fit again only varying the edge center
    #                edgeCenter = pOut2[0][0]
                    try:
                        dEdgeCenter = pOut[1][0,0]**0.5 
                    except:
                        print pOut, type(pOut)
                    usedStr = 'not used'
    #                print(preGuess, postGuess)
    #                print('ds_num %d, %s, pre %f, post %f'%(ds_num, line_name, preHeight, postHeight))
                    edgeInfo = {'center':edgeCenter, 'averageCounts':(preHeight+postHeight)/2,'dropCounts':preHeight-postHeight, 'uncertainty': dEdgeCenter, 'width':width, 'name': line_name}
                    if (not numpy.isnan(dEdgeCenter)) and dEdgeCenter<=20.0 and (preHeight-postHeight>10) and (abs(edgeCenter-edgeGuess)<30): 
                        cal.add_cal_point(edgeCenter, line_name, pht_error=dEdgeCenter, info=edgeInfo)
                        usedStr = 'used'
                    else:
                        self.data.set_chan_bad(ds.channum, 'calibrate_carefully_edges rejected %s'%line_name)
                    if doPlot == True:
                        pylab.figure()
                        pylab.plot(bin_ctrs, contents)
                        pylab.plot(bin_ctrs, numpy.polyval(pfit, bin_ctrs))
                        pylab.plot(edgeGuess, numpy.polyval(pfit, edgeGuess),'.')
                        pylab.plot([preGuess, postGuess], numpy.polyval(pfit,[preGuess, postGuess]),'.')       
                        pylab.plot(bin_ctrs, self.edgeModel(bin_ctrs, edgeCenter, 
                                    preHeight, postHeight, width))             
                        pylab.ylabel('counts per %4.2f unit bin'%(bin_ctrs[1]-bin_ctrs[0]))
                        pylab.xlabel(whichCalibration)
                        pylab.title('%s center=%.1f, dCenter=%.3f, width=%.3f, %s'%(line_name, edgeCenter, dEdgeCenter, width, usedStr))
                    print('cal_edge %s chan %d, %s, edgeCenter %.2f, dEdgeCenter %.3f, edgeDropCounts %.1f, %s'%(whichCalibration, ds.channum, line_name, edgeCenter, dEdgeCenter, preHeight-postHeight, usedStr))
                    
                else:
                    print('%s not recognized as a KAlpha, KBeta or Edge'%line_name)



        self.convert_to_energy(whichCalibration=whichCalibration, whichFiltValue=whichFiltValue)
        
        
#    def calibrate_carefully_edges(self,edge_names = ['TiKEdge', 'VKEdge', 'MnKEdge', 'CrKEdge'], doPlot = False, append_to_cal=True):
#        if type(edge_names) != type(list()): edge_names = [edge_names]
#        returnInfo = {}
#        for ds in (self.data):
#            if ds.calibration.has_key('p_filt_value_dc') and append_to_cal:
#                cal = ds.calibration['p_filt_value_dc'] # add to existing cal if it exists
#            else:
#                self.data.set_chan_bad(ds.channum, 'calibrate_carefully_edges cant find edges without existing p_filt_value_dc calibration ')
#                continue
#            for i, edge_name in enumerate(edge_names):
#                edge_energy = mass.energy_calibration.STANDARD_FEATURES[edge_name]
#                minE, maxE = ds.calibration['p_filt_value'].name2ph('%s'%edge_name)*numpy.array([0.98, 1.02]) 
#                contents, bins = numpy.histogram(ds.p_filt_value_dc[ds.cuts.good()], bins=numpy.arange(minE, maxE, 3))
#                bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
##                try:
#                pfit = numpy.polyfit(bin_ctrs, contents, 3)
#                edgeGuess = numpy.roots(numpy.polyder(pfit,2))
#                if edgeGuess == numpy.abs(edgeGuess):
#                    edgeGuess = numpy.abs(edgeGuess)
#                preGuess, postGuess = numpy.sort(numpy.roots(numpy.polyder(pfit,1)))
#                if not (bin_ctrs[0]<edgeGuess<bin_ctrs[-1] and numpy.polyval(numpy.polyder(pfit,1),edgeGuess)<0 and bin_ctrs[0]<preGuess<bin_ctrs[-1] and bin_ctrs[0]<postGuess<bin_ctrs[-1]):
#                    continue
#
#                pGuess = numpy.array([edgeGuess, numpy.polyval(pfit,preGuess), numpy.polyval(pfit,postGuess),10.0],dtype='float64')
#                pOut = scipy.optimize.curve_fit(self.edgeModel, bin_ctrs, contents, 
#                                                pGuess)
#                (edgeCenter, preHeight, postHeight, width) = pOut[0]
##                refitEdgeModel = lambda x,edgeCenterL: self.edgeModel(x,edgeCenterL, preHeight, postHeight, width)
##                pOut2 = scipy.optimize.curve_fit(refitEdgeModel, bin_ctrs, contents, 
##                                [edgeCenter]) # fit again only varying the edge center
##                edgeCenter = pOut2[0][0]
#                try:
#                    dEdgeCenter = pOut[1][0,0]**0.5 
#                except:
#                    print pOut, type(pOut)
#                usedStr = 'not used'
##                print(preGuess, postGuess)
##                print('ds_num %d, %s, pre %f, post %f'%(ds_num, edge_name, preHeight, postHeight))
#                if (not numpy.isnan(dEdgeCenter)) and dEdgeCenter<=20.0 and (preHeight-postHeight>10) and (abs(edgeCenter-edgeGuess)<30): 
#                    cal.add_cal_point(edgeCenter, edge_name, pht_error=dEdgeCenter)
#                    usedStr = 'used'
#                else:
#                    self.data.set_chan_bad(ds.channum, 'calibrate_carefully_edges rejected %s'%edge_name)
#                if doPlot == True:
#
#  
#                    pylab.figure()
#                    pylab.plot(bin_ctrs, contents)
#                    pylab.plot(bin_ctrs, numpy.polyval(pfit, bin_ctrs))
#                    pylab.plot(edgeGuess, numpy.polyval(pfit, edgeGuess),'.')
#                    pylab.plot([preGuess, postGuess], numpy.polyval(pfit,[preGuess, postGuess]),'.')       
#                    pylab.plot(bin_ctrs, self.edgeModel(bin_ctrs, edgeCenter, 
#                                preHeight, postHeight, width))             
#                    pylab.ylabel('counts per %4.2f unit bin'%(bin_ctrs[1]-bin_ctrs[0]))
#                    pylab.xlabel('p_filt_value_dc')
#                    pylab.title('%s center=%.1f, dCenter=%.3f, width=%.3f, %s'%(edge_name, edgeCenter, dEdgeCenter, width, usedStr))
#                print('cal_edge chan %d, %s, edgeCenter %.2f, dEdgeCenter %.3f, edgeDropCounts %.1f, %s'%(ds.channum, edge_name, edgeCenter, dEdgeCenter, preHeight-postHeight, usedStr))
#                
#                returnInfo[ds.channum] = {'used':usedStr, 'center':edgeCenter, 'averageCounts':(preHeight+postHeight)/2,'dropCounts':preHeight-postHeight, 'uncertainty': dEdgeCenter, 'width':width, 'name': edge_name}
#
##                except:
##                    print('cant find %s in channel %d'%(edge_name, ds.channum))
#        self.convert_to_energy('p_filt_value_dc')
#        return returnInfo
        
    def returnCalLocations(self, cal_features):
        if type(cal_features) != type(list()): cal_features = [cal_features]
        calLocations = {}
        for ds in self.data:
            if ds.calibration.has_key('p_filt_value_dc'):
                cal = ds.calibration['p_filt_value_dc']
            else:
                print('no cal exist for chan %d'%ds.channum)
            calLocations[ds.channum] = numpy.zeros_like(cal_features,dtype='float')
            for i, cal_feature in enumerate(cal_features):
                edge_energy = mass.energy_calibration.STANDARD_FEATURES[cal_feature]
                calLocations[ds.channum][i] = cal.name2ph(cal_feature)
        return calLocations
    
    def edgeModel(self, x, edgeCenter, preHeight, postHeight, width=1.0):
        countsOut = numpy.zeros_like(x)
        width = float(width)
        countsOut = preHeight - (numpy.tanh((x-edgeCenter)/width)/2.0+0.5)*(preHeight-postHeight)
        return countsOut
        
    def convert_to_energy(self, whichCalibration = 'p_filt_value', whichFiltValue = None):
        print('converting to energy with %s'%whichCalibration)
        commonCalibrations = ['p_pulse_average', 'p_filt_value', 'p_filt_value_dc', 'p_filt_value_phc']
        if whichCalibration not in commonCalibrations:
            print('calibration %s not a common choice, will probably break things, try one of '%whichCalibration, commonCalibrations)
        for ds in self.data:
            if not whichCalibration in ds.calibration.keys():
                self.data.set_chan_bad(ds.channum, 'failed convert_to_energy because %s not in ds.calibration.keys()'%whichCalibration)
                continue
            if whichFiltValue is None:
                if '_scaled' in whichCalibration:
                    whichFiltValue = whichCalibration[:whichCalibration.find('_scaled')]
                else:
                    whichFiltValue = whichCalibration
                if not whichFiltValue in ds.__dict__.keys():
                    self.data.set_chan_bad(ds.channum, 'failed convert_to_energy because %s not in ds.__dict__.keys()'%whichCalibration)
                    continue
            try:
                cal = ds.calibration[whichCalibration]
                ds.p_energy = cal(ds.__dict__[whichFiltValue])
                ds.p_energy[numpy.isnan(ds.p_energy)] = 0
            except:
                self.data.set_chan_bad(ds.channum, 'failed convert_to energy with %s'%whichCalibration)
            
    def plot_energy_spectra(self, erange=[5850,5950]):
        pylab.clf()
        fitter = mass.MnKAlphaFitter()
        nbins = erange[1]-erange[0]
        contents = numpy.zeros(nbins)
        for ds_num,ds in enumerate(self.data):
            c,bins = numpy.histogram(ds.p_energy[ds.cuts.good()],
                                  nbins, erange)
            contents += c
            bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
            try:
                param, covar = fitter.fit(c, bin_ctrs, plot=False)
            except RuntimeError:
                print 'Cannot fit'
                continue
            res = param[0]
            dres = covar[0,0]**0.5
            ph = param[1]
            dph = covar[1,1]**0.5
            print 'res = %.2f +/- %.2f' %(res, dres)

        axis = pylab.subplot(111)
        cmap = pylab.cm.spectral
        color = cmap(1.0*i/self.data.n_channels)
        #mass.plot_as_stepped_hist(axis, contents, bins, color=color, label="%d"%i)
        param, covar = fitter.fit(contents, bin_ctrs, plot=True, color=color)

        
    def plot_calibration(self, whichCalibration = 'p_filt_value',power=0.0):
        pylab.clf()
        ax = pylab.subplot(111)

        for ds in self.data:
            ds.calibration[whichCalibration].plot(ax, ph_rescale_power=power)
        

        
    def store_filters_and_cal(self, filename='calibration.pkl'):
        print('store_filters_and_cal -> %s'%filename)
        try:
            cals,filters, dc_slope, dc_meanpretrigmean, dc_power = load_filters_and_cal(filename)
        except:
            cals = {}
            filters = {}
            drift_correct_info = {}
            phase_correct_info = {}
        fp = open(filename, 'wb')
        pickler = cPickle.Pickler(fp, protocol=cPickle.HIGHEST_PROTOCOL)

        for ds_num,ds in enumerate(self.data.datasets):
            cals[ds.channum] = ds.calibration
            filters[ds.channum]=ds.filter
            drift_correct_info[ds.channum] = ds.drift_correct_info
            phase_correct_info[ds.channum] = ds.phase_correct_info

        pickler.dump(cals)
        pickler.dump(filters)
        pickler.dump(drift_correct_info)
        pickler.dump(phase_correct_info)
        fp.close()
        
    def load_filters_and_cal(self, filename="calibration.pkl", applyLoaded=True):
        print('load_filter_and_cal from %s'%filename)
        fp = open(filename, 'rb')
        unpickler = cPickle.Unpickler(fp)
        cals = unpickler.load()
        filters = unpickler.load()
        drift_correct_info = unpickler.load()
        phase_correct_info = unpickler.load()
        fp.close()
        for ds in self.data:
            if cals.has_key(ds.channum):
                ds.calibration = cals[ds.channum]
            else:
                self.data.set_chan_bad(ds.channum, 'loaded file %s does not have cal for channels'%filename)
                continue
            if filters.has_key(ds.channum):
                ds.filter = filters[ds.channum]
            else:
                self.data.set_chan_bad(ds.channum, 'loaded file %s does not have filter for channels'%filename)
                continue
            if drift_correct_info.has_key(ds.channum):
                ds.drift_correct_info = drift_correct_info[ds.channum]
            else:
                self.data.set_chan_bad(ds.channum, 'loaded file %s does not have drift_correct_info for channels'%filename)
                continue
            if phase_correct_info.has_key(ds.channum):
                ds.phase_correct_info = phase_correct_info[ds.channum]
            else:
                self.data.set_chan_bad(ds.channum, 'loaded file %s does not have phase_correct_info for channeles'%filename)
                continue
            try:
                if not ds.nSamples == filters[ds.channum].avg_signal.size:
                    self.data.set_chan_bad(ds.channum, 'loaded filter has %d samples, != datasets has %d samples'%(filters[ds.channum].avg_signal.size, ds.nSamples))
                    continue
            except:
                self.data.set_chan_bad(ds.channum, 'loaded filter probably doesnt have .avg_signal')
                continue
                
        if applyLoaded:
            self.apply_filter()
            self.apply_stored_drift_correct()
            if self.data.first_good_dataset.phase_correct_info.has_key('phase'):
                self.apply_stored_phase_correct()
                self.convert_to_energy('p_filt_value_phc')
            else:
                print('not applying phase correct, because it doesnt look like one got loaded')
                self.convert_to_energy('p_filt_value_dc')


    def countRateAndCuts(self, channel = 1):
        ds = self.data.channel[channel]
        totalCounts = ds.nPulses
        countsPassedCuts = ds.cuts.good().sum()
        elapsedTime = ds.p_timestamp[ds.cuts.good()][-1]-ds.p_timestamp[ds.cuts.good()][0]
        print('%d pulses cut by CUT_PRETRIG_RMS'%numpy.sum(ds.cuts.isCut(ds.CUT_PRETRIG_RMS)))
        print('%d pulses cut by CUT_BIAS_PULSE'%numpy.sum(ds.cuts.isCut(ds.CUT_BIAS_PULSE)))
        print('%d pulses cut by CUT_RISETIME'%numpy.sum(ds.cuts.isCut(ds.CUT_RISETIME)))
        print('%d pulses cut by CUT_TIMESTAMP'%numpy.sum(ds.cuts.isCut(ds.CUT_TIMESTAMP)))
        print('%d pulses cut by CUT_PRETRIG_MEAN'%numpy.sum(ds.cuts.isCut(ds.CUT_PRETRIG_MEAN)))
        print('%d pulses cut by CUT_RETRIGGER'%numpy.sum(ds.cuts.isCut(ds.CUT_RETRIGGER)))
        print('%d pulses cut by CUT_SATURATED'%numpy.sum(ds.cuts.isCut(ds.CUT_SATURATED)))
        print('%d pulses cut by CUT_UNLOCK'%numpy.sum(ds.cuts.isCut(ds.CUT_UNLOCK)))
        print('%d pulses cut by CUT_TIMESTAMP'%numpy.sum(ds.cuts.isCut(ds.CUT_TIMESTAMP)))
        print('totalCounts %d, countsPassedCuts %d, elapsedTime %f'%(totalCounts, countsPassedCuts, elapsedTime))
        
    def countRateInfo(self, usefulEnergyRange = (5300, 6000), doPlots = False, verbose = False ):
        assert(usefulEnergyRange[0]<usefulEnergyRange[1])
        countsPassedCuts = numpy.zeros(self.data.num_good_channels)
        totalCounts = numpy.zeros(self.data.num_good_channels)
        elapsedTime = numpy.zeros(self.data.num_good_channels, dtype='float')
        usefulCounts = numpy.zeros(self.data.num_good_channels)
        for i,ds in enumerate(self.data):
            
            countsPassedCuts[i] = ds.cuts.good().sum()
            totalIndex = log_and(ds.p_timestamp>ds.p_timestamp[ds.cuts.good()][0], ds.p_timestamp<ds.p_timestamp[ds.cuts.good()][-1])
            totalCounts[i] = totalIndex.sum()
            elapsedTime[i] = ds.p_timestamp[ds.cuts.good()][-1]-ds.p_timestamp[ds.cuts.good()][0]
            usefulIndex = log_and(ds.cuts.good(), ds.p_energy>usefulEnergyRange[0], ds.p_energy<usefulEnergyRange[1])
            usefulCounts[i] = usefulIndex.sum()
            if verbose:
                print('channel %d, trigger rate %4.2f/s, passed cuts rate %4.2f/s, useful rate %4.2f/s'%(ds.channum, 
                   totalCounts[i]/elapsedTime[i], countsPassedCuts[i]/elapsedTime[i], usefulCounts[i]/elapsedTime[i]))
            if doPlots:
                pylab.figure()
                bins = numpy.arange(0,16000,4)            
                pylab.hist(ds.p_energy[ds.cuts.good()],bins)
                pylab.hist(ds.p_energy[usefulIndex], bins, color='r', edgecolor='r') # I don't understand why this doesn't work, it ignore the color, but it works from ipython
                pylab.ylabel('Energy (eV) 2 line calibration')
                pylab.ylabel('counts per %4.2f eV bin'%(bins[1]-bins[0]))
                pylab.title('chan %d, trigger %4.2f s^-1, passed cuts %4.2f s^-1, useful %4.2f s^-1'%(ds.channum,
                              totalCounts[i]/elapsedTime[i], countsPassedCuts[i]/elapsedTime[i], usefulCounts[i]/elapsedTime[i]))
            
        elapsedTime = numpy.median(elapsedTime) # they should all have the same elapsed time roughly.
        if doPlots:
            pylab.figure()
            pylab.plot(totalCounts/elapsedTime, 10*usefulCounts/elapsedTime,'bs',label='10x counts in range %d eV to %d eV'%usefulEnergyRange)
            pylab.plot(totalCounts/elapsedTime, countsPassedCuts/elapsedTime,'ro',label='counts passed cuts')
            pylab.xlabel('~trigger rate s^-1')
            pylab.ylabel('other count rates s^-1')
            pylab.legend()
        
        return totalCounts/elapsedTime, countsPassedCuts/elapsedTime, usefulCounts/elapsedTime
        
    def scaleCalibration(self, referenceGenCal, whichCalibrationSelf='p_filt_value_dc', whichCalibrationReference=None, referenceFeature='MnKAlpha'):
        print('scaleCalibration currently doesnt propogate error properly, pht_error in new calibration isnt right')
        if whichCalibrationReference is None: 
            whichCalibrationReference = whichCalibrationSelf
            print('scaleCalibration: whichCalibrationReference = whichCalibrationSelf since whichCalibrationReference was None')
        scalingDict = {}
        for ds in self.data:
            try:
                origCal = ds.calibration[whichCalibrationSelf]
                newCal = mass.energy_calibration.EnergyCalibration('%s_scaled_at_%s'%(whichCalibrationSelf, referenceFeature))
                ds.calibration[newCal.ph_field] = newCal
                referenceUnscaledPulseHeight = origCal.name2ph(referenceFeature)
                referenceScaledPulseHeight = referenceGenCal.data.channel[ds.channum].calibration[whichCalibrationReference].name2ph(referenceFeature)
                for i, unscaledPulseHeight in enumerate(origCal._ph):
                    if unscaledPulseHeight > 0:
                        scaledPulseHeight = unscaledPulseHeight*referenceScaledPulseHeight/referenceUnscaledPulseHeight
                        newCal.add_cal_point(scaledPulseHeight, origCal._names[i],pht_error=origCal._stddev[i])
                scalingDict[ds.channum] = referenceScaledPulseHeight/referenceUnscaledPulseHeight
            except:
                self.data.set_chan_bad(ds.channum, 'failed scaleCalibration, probably brentq')
        return scalingDict
    
    def phase_correct(self, line_names = ['MnKAlpha'], whichCalibration = 'p_filt_value_dc', energyRangeFracs=[0.995, 1.005], times=None, doPlot=True):
        """Apply a correction for pulse variation with arrival phase.
        Model is a parabolic correction with cups at +-180 degrees away from the "center".
        
        prange:  use only filtered values in this range for correction 
        times: if not None, use this range of p_timestamps instead of all data (units are seconds
               since server started--ugly but that's what we have to work with)
        doPlot:  whether to display the result
        """
        if type(line_names) != type(list()): line_names = [line_names]
        # Choose number and size of bins
        
        phaseSpan = 1.0
        phaseMax = phaseSpan/2.0
        phases = numpy.linspace(-phaseMax,phaseMax,20)
        phase_step = phases[1]-phases[0]
        
        for ds in self.data:
            for line_name in line_names:
                calibration = ds.calibration[whichCalibration]
                ph_estimate = calibration.name2ph(line_name)
                ph_range = numpy.array(energyRangeFracs)*ph_estimate
            

                # Estimate corrections in a few different pieces
                toCorrect = ds.__dict__[whichCalibration]
                corrections = []
                valid = ds.cuts.good()
                if ph_range is not None:
                    valid = numpy.logical_and(valid, toCorrect<ph_range[1])
                    valid = numpy.logical_and(valid, toCorrect>ph_range[0])
                if times is not None:
                    valid = numpy.logical_and(valid, ds.p_timestamp<times[1])
                    valid = numpy.logical_and(valid, ds.p_timestamp>times[0])
    
                # Plot the raw filtered value vs phase
                if doPlot:
                    pylab.figure()
                    pylab.subplot(211)
                    pylab.plot((ds.p_filt_phase[valid]+phaseMax)%phaseSpan-phaseMax, toCorrect[valid],'.')
                    pylab.xlabel("Hypothetical 'center phase'")
                    pylab.ylabel(whichCalibration)
                    pylab.title('channel %d'%ds.channum)
                    pylab.xlim([-1.1*phaseMax,1.1*phaseMax])
                    if ph_range is not None:
                        pylab.ylim(ph_range)
                    
                for ctr_phase in phases:
                    valid_ph = numpy.logical_and(valid,
                                                 numpy.abs((ds.p_filt_phase - ctr_phase)%1) < phase_step*0.5)
        #            print valid_ph.sum(),"   ",
                    mean = toCorrect[valid_ph].mean()
                    median = numpy.median(toCorrect[valid_ph])
#                    robust_mean = mass.robust.bisquare_weighted_mean(toCorrect[valid_ph], 3.88)
                    corrections.append(mean) # not obvious that mean vs median matters
                    if doPlot:
                        pylab.plot(ctr_phase, mean, 'or')
#                        pylab.plot(ctr_phase, median, 'vk', ms=10)
#                        pylab.plot(ctr_phase, robust_mean, 'gd')
                corrections = numpy.array(corrections)
                if not numpy.isfinite(corrections).all():
                    self.data.set_chan_bad(ds.channum, 'phase_corrections not all finite')
                    break
            

                errfunc = lambda p,x,y: y-self.phaseCorrectionModel(p,x)
                params = (0., 4, corrections.mean())
                fitparams, _iflag = scipy.optimize.leastsq(errfunc, params, args=(ds.p_filt_phase[valid], toCorrect[valid]))
                plot_phases = numpy.linspace(-phaseMax,phaseMax,100)
            
                fitparams[2] = 0
                ds.phase_correct_info={'phase':fitparams[0],
                                    'amplitude':fitparams[1],
                                    'mean':fitparams[2],
                                    'calibration_to_correct':whichCalibration}
                correction = self.phaseCorrectionModel(fitparams, ds.p_filt_phase)
                ds.p_filt_value_phc = toCorrect - correction
    #        self.p_filt_value_dc = self.p_filt_value_phc.copy()
                print 'RMS phase correction chan %d %s is: %9.3f (%6.2f parts/thousand)'%(ds.channum, line_name, correction.std(), 
                                                    1e3*correction.std()/toCorrect.mean())
            
                if doPlot:
                    print self.phaseCorrectionModel(fitparams, plot_phases)
                    pylab.plot(plot_phases, self.phaseCorrectionModel(fitparams, plot_phases)+corrections.mean(), color='red')

                    pylab.subplot(212)
                    pylab.plot((ds.p_filt_phase[valid]+phaseMax)%phaseSpan-phaseMax, ds.p_filt_value_phc[valid],'.')
                    pylab.xlabel('p_filt_phase')
                    pylab.ylabel('p_filt_value_phc')
                    pylab.xlim([-1.1*phaseMax,1.1*phaseMax])
                    if ph_range is not None:
                        pylab.ylim(ph_range)
                        
    def phaseCorrectionModel(self, params, phase):
        "Params are (phase of center, curvature, mean peak height)"
        phase = (phase - params[0]+.5)%1 - 0.5
        return 4*params[1]*(phase**2 - 0.125) + params[2]
    
    def apply_stored_phase_correct(self):
        print('apply_stored_phase_correct')
        for ds in self.data:
            if ds.phase_correct_info.has_key('phase'):
                toCorrect = ds.__dict__[ds.phase_correct_info['calibration_to_correct']]
                fitparams = [ds.phase_correct_info['phase'], ds.phase_correct_info['amplitude'], ds.phase_correct_info['mean']]
                correction = self.phaseCorrectionModel(fitparams, ds.p_filt_phase)
                ds.p_filt_value_phc = toCorrect - correction
            else:
                self.data.set_chan_bad(ds.channum, 'phase_correct_info[''phase''] does not exist')
                
    def pickle(self):
        print('pickleing data')
        self.data.pickle()
    
    def unpickle(self):
        self.data = mass.core.channel_group.unpickle_TESGroup(self.data.first_good_dataset.filename)

        
                






