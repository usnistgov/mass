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
        for channel in numpy.array(dataset_number_subset)*2+1:
            if os.path.isfile(pulse_filename%channel) and os.path.isfile(noise_filename%channel):
                channels.append(channel)
        noise_files=[noise_filename%c for c in channels]
        pulse_files=[pulse_filename%c for c in channels]
        
        self.data = mass.TESGroup(pulse_files, noise_files)


    def copy(self):
        """This trick is pretty useful if you are going to update the object (e.g.,
        by adding new methods, or fixing bugs in existing methods), but you don't
        want to lose all the computations you already did.  In some objects, this
        has to handle deep copying, or other subtleties that I forget."""
        c = generalCalibration()
        c.__dict__.update(self.__dict__)
        return c

    def do_basic_computation(self, doFilter = True):
        """This covers all the operations required to analyze and store the data
        from the December 18, 2012 XSI calibrations."""
        self.data.summarize_data(peak_time_microsec=420.0)
        self.apply_cuts()
        if doFilter:
            self.data.compute_noise_spectra()
            self.data.plot_noise()
            self.compute_model_pulse()
            self.data.compute_filters(f_3db=6000.)
            self.data.summarize_filters()
            self.apply_filter()
#        self.calibrate_approximately()
#        self.dc_slope = self.drift_correct_multiline('Ti')
#        self.calibrate_carefully(element_names=['Ti','Cr','Mn'],doPlot = False)
#        self.store_calibration()
#        self.store_filters_and_cal()

        
    def channel_histogram(self, channel=0, driftCorrected = False):
        if channel in self.data.channel.keys():
            ds = self.data.channel[channel]
        else:
            ds = self.data.fir
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
                
                
    def channel_findpeaks(self, channum = 0 , driftCorrected = False, doPlot=False):
        ds = self.data.channel[channum]
        if driftCorrected == False:
            data = ds.p_filt_value[ds.cuts.good()]
        elif driftCorrected == True:
            data = ds.p_filt_value_dc[ds.cuts.good()]    
        histogramMax = numpy.min((2**14,numpy.max(data)))
        if histogramMax <= 0:
            return [],[]
        ph_bin_edges = numpy.arange(0,histogramMax,3)
#        print 'ph_bin_edge.shape, histogramMax %f'%histogramMax, ph_bin_edges.shape
        counts, ph_bin_edges = numpy.histogram(data, ph_bin_edges)
        ph_bin_centers = (ph_bin_edges[:-1]+ph_bin_edges[1:])/2
        peak_indexes = numpy.array(scipy.signal.find_peaks_cwt(counts, numpy.arange(1,30,3), min_snr=4))
        sort_index = numpy.argsort(counts[peak_indexes])
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
    
    def apply_cuts(self, timestampCuts = (None, None)):
        self.cuts = mass.AnalysisControl()
        self.cuts.cuts_prm.update({
                 'max_posttrig_deriv': (None, 60.0),
                 'pretrigger_mean_departure_from_median': (-40, 40),
                 'pretrigger_rms': (None, 10.0),
                 'pulse_average': (5.0, None),
                 'rise_time_ms': (None, 0.7),
                 'peak_time_ms': (None, 0.5),
                 'timestamp_sec': timestampCuts,})
        for ds in self.data:
            ds.clear_cuts()
            ds.apply_cuts(self.cuts)

    def plot_pulse_timeseries(self, channel=0, type='ph'):
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
        self.data.filter_data('filt_noconst')


    
    
    def drift_correct_new(self, line_names = ['MnKAlpha'], power = 0.1, doPlot = False):
        print('starting drift_correct_new')
        # currently only uses the last item in line_names
        if type(line_names) != type(list()): line_names = [line_names]
        self.dc_slope = {}
        self.dc_meanpretrigmean = {}
        self.dc_power = power
        for ds_num, ds in enumerate(self.data):
            for i, line_name in enumerate(line_names):
                minE,maxE =  ds.calibration['p_filt_value'].name2ph(line_name)*numpy.array([0.9905, 1.013])
                use = log_and(ds.cuts.good(), ds.p_filt_value>minE, ds.p_filt_value<maxE)
                mean_pretrig_mean = numpy.mean(ds.p_pretrig_mean[use])
                ds.p_filt_value[ds.p_filt_value<0]=0 # so the exponent doesn't throw an error
                corrector = (ds.p_pretrig_mean-mean_pretrig_mean)*(ds.p_filt_value**power)

                pfit = numpy.polyfit(corrector[use], ds.p_filt_value[use],1)
                slopes = -pfit[0]*numpy.linspace(0,2,20)
                fitter = mass.__dict__['%sFitter'%line_name]()
                resolution = numpy.zeros_like(slopes)
                if doPlot: pylab.figure()
                for j,slope in enumerate(slopes):
                    corrected = ds.p_filt_value + corrector*slope
                    contents, bins = numpy.histogram(corrected[use], bins=numpy.arange(minE, maxE, 1))
                    bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
                    param, covar = fitter.fit(contents, bin_ctrs, plot=False)
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
            ds.drift_correct_info = {}
            ds.drift_correct_info['type'] = 'ptmean_power'
            ds.drift_correct_info['slope'] = slopes[numpy.argmin(resolution)]
            ds.drift_correct_info['meanpretrigmean'] = mean_pretrig_mean
            ds.drift_correct_info['power'] = power
            ds.drift_correct_info['best_achieved_resolution'] = numpy.amin(resolution)
            print('drift_correct_new chan %d, dc slope %.3f, best resolution %.2f, power %.1f'%(ds.channum, ds.drift_correct_info['slope'], ds.drift_correct_info['best_achieved_resolution'],power))
        
    def apply_stored_drift_correct(self):
        print('applying drift correction ')
        for ds_num, ds in enumerate(self.data):
            try:
                print('chan %d, dc_meanpretrigmean %.2f, dc_slope, %.2f dc_power %.2f'%(ds.channum, ds.drift_correct_info['meanpretrigmean'], ds.drift_correct_info['slope'], ds.drift_correct_info['power']))
                corrector = (ds.p_pretrig_mean-ds.drift_correct_info['meanpretrigmean'])*(ds.p_filt_value**ds.drift_correct_info['power'])
                ds.p_filt_value_dc = ds.p_filt_value+corrector*ds.drift_correct_info['slope']
            except:
                self.data.set_chan_bad(ds.channum, 'failed apply_drift_correct')
                   
        
    def calibrate_approximately(self, line_names = ['MnKAlpha', 'MnKBeta'], doPlot = False):
        """Element names must be in order of peak height, only works with kAlphas for now"""
        minPulses = 100
        if type(line_names) != type(list()): line_names = [line_names]
        for ds in self.data:               
            cal = mass.calibration.EnergyCalibration('p_filt_value')
            ds.calibration['p_filt_value'] = cal
            if ds.cuts.good().sum() > minPulses:
                peak_location_pulseheights,peak_counts = self.channel_findpeaks(ds.channum,driftCorrected=False, doPlot=doPlot)
            else:
                self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley has %d pulses total < %d the minimum'%(ds.cuts.good().sum(), minPulses))
                continue
            if len(peak_location_pulseheights) < len(line_names):
                self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley %d peak < %d line_names'%(len(peak_location_pulseheights), len(line_names)))                
                continue
            for i, line_name in enumerate(line_names): 
                line_location_guess = peak_location_pulseheights[-(i+1)]
                use = log_and(ds.cuts.good(), numpy.abs(ds.p_filt_value/line_location_guess-1.0)<0.012)
                if use.sum() < minPulses:
                    self.data.set_chan_bad(ds.channum, 'failed calibrate_approximatley %s has %d pulses < %d the minimum'%(line_name, use.sum(), minPulses))
                    break
                else:
                    line_location = scipy.stats.scoreatpercentile(ds.p_filt_value[use], 67.)
                    cal.add_cal_point(line_location, '%s'%line_name, pht_error=2)
                    print('calibrate_approximately chan %d added %s at %.2f'%(ds.channum, line_name, line_location))
                
        print('after calibrating approximately %d of %d datasets survived'%(self.data.n_channels-len(self.data._bad_channums), self.data.n_channels))

        self.convert_to_energy(use_drift_correct=False)
        
        
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
    
    def calibrate_carefully(self,lines_name = ['MnKAlpha', 'MnKBeta'], doPlot = False):
        if type(lines_name) != type(list()): lines_name = [lines_name]

        for ds in self.data:
            if ds.calibration.has_key('p_filt_value_dc'):
                cal = ds.calibration['p_filt_value_dc'] # add to existing cal if it exists
            else:
                cal = mass.calibration.EnergyCalibration('p_filt_value_dc')
                ds.calibration['p_filt_value_dc'] = cal
            for i, line_name in enumerate(lines_name):
                minE, maxE = ds.calibration['p_filt_value'].name2ph('%s'%line_name)*numpy.array([0.995, 1.005])
                fitter = mass.__dict__['%sFitter'%line_name]()
                contents, bins = numpy.histogram(ds.p_filt_value_dc[ds.cuts.good()], bins=numpy.arange(minE, maxE, 1))
                bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
                try:
                    if doPlot: pylab.figure()
                    hold = []
                    paramGuess = None
                    if line_name[-4:]=='Beta':
                        hold = [2,4,5] # simplify the fitting for kBeta by holding energh scale factor (degenerate with resolution with only 1 line), and background level and slope
                        paramGuess = [4.0,ds.calibration['p_filt_value'].name2ph('%s'%line_name), 1.0,contents.max()/0.13, 0.0, 0.0 ]
                    param, covar = fitter.fit(contents, bin_ctrs, plot=doPlot, hold=hold)
                    #param: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
                    #energy scale factor (counts/eV), amplitude, background level (per bin),
                    #and background slope (in counts per bin per bin) ]
                except RuntimeError:
                    self.data.set_chan_bad(ds.channum, 'failed calibrate_carefully %s with RuntimeError'%line_name)
                    continue
                except:
                    self.data.set_chan_bad(ds.channum, 'failed calibrate_carefully %s'%line_name)
                    continue
                    
                res = param[0]
                dres = covar[0,0]**0.5
                ph = param[1]
                dph = covar[1,1]**0.5
                cal.add_cal_point(ph, '%s'%line_name, pht_error=dph)
#                if line_name[-5:]=='Alpha':
#                    # add an extra calibration point based on the slope
#                    offsetEnergy = 10 #eV
#                    extrapolated_ph = ph-offsetEnergy*param[2]
#                    # calculate the uncertainty in the extrapolated pulse height based on an 
#                    # equation from wikipedia
#                    d_extrapolated_ph = dph+offsetEnergy*covar[2,2]**0.5+abs(2*offsetEnergy*covar[2,1])
#                    extrapolated_energy = mass.energy_calibration.STANDARD_FEATURES[line_name]-offsetEnergy
#                    cal.add_cal_point(extrapolated_ph, extrapolated_energy,name='%s_ex'%line_name, pht_error=d_extrapolated_ph )
#                    print('ds_num %d, %s, dph %f, resolution %.2f +- %.2f eV, d_extrapolated_ph %.2f'%(ds_num, line_name, dph, res, dres, d_extrapolated_ph))
                print('cal chan %d, %s, ph %.1f, dph %.3f, resolution %.2f +- %.2f eV'%(ds.channum, line_name, ph, dph, res, dres))

        self.convert_to_energy(use_drift_correct=True)
        
        
    def calibrate_carefully_edges(self,edge_names = ['TiKEdge', 'VKEdge', 'MnKEdge', 'CrKEdge'], doPlot = False):
        if type(edge_names) != type(list()): edge_names = [edge_names]
        returnData = numpy.zeros((len(self.data.datasets), len(edge_names)))
        returnData2 = numpy.zeros((len(self.data.datasets), len(edge_names)))
        for ds_num, ds in enumerate(self.data.datasets):
            if ds.calibration.has_key('p_filt_value_dc'):
                cal = ds.calibration['p_filt_value_dc'] # add to existing cal if it exists
            else:
                print('currently not setup to find edges without existing calibration')
                raise ValueError
            for i, edge_name in enumerate(edge_names):
                print('ds_num %d, %s'%(ds_num, edge_name))
                edge_energy = mass.energy_calibration.STANDARD_FEATURES[edge_name]
                minE, maxE = ds.calibration['p_filt_value'].name2ph('%s'%edge_name)*numpy.array([0.98, 1.02]) 
                contents, bins = numpy.histogram(ds.p_filt_value_dc[ds.cuts.good()], bins=numpy.arange(minE, maxE, 3))
                bin_ctrs = bins[:-1] + 0.5*(bins[1]-bins[0])
#                try:
                pfit = numpy.polyfit(bin_ctrs, contents, 3)
                edgeGuess = numpy.roots(numpy.polyder(pfit,2))
                if edgeGuess == numpy.abs(edgeGuess):
                    edgeGuess = numpy.abs(edgeGuess)
                preGuess, postGuess = numpy.sort(numpy.roots(numpy.polyder(pfit,1)))
                if not (bin_ctrs[0]<edgeGuess<bin_ctrs[-1] and numpy.polyval(numpy.polyder(pfit,1),edgeGuess)<0 and bin_ctrs[0]<preGuess<bin_ctrs[-1] and bin_ctrs[0]<postGuess<bin_ctrs[-1]):
                    continue

                pGuess = numpy.array([edgeGuess, numpy.polyval(pfit,preGuess), numpy.polyval(pfit,postGuess),10.0],dtype='float64')
                pOut = scipy.optimize.curve_fit(self.edgeModel, bin_ctrs, contents, 
                                                pGuess)
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
#                print('ds_num %d, %s, pre %f, post %f'%(ds_num, edge_name, preHeight, postHeight))
                if (not numpy.isnan(dEdgeCenter)) and dEdgeCenter<=20.0 and (preHeight-postHeight>10) and (abs(edgeCenter-edgeGuess)<30): 
                    cal.add_cal_point(edgeCenter, edge_name, pht_error=dEdgeCenter)
                    usedStr = 'used'
#                else:
#                    print((not numpy.isnan(dEdgeCenter)), dEdgeCenter<=20.0, (preHeight-postHeight>10), (abs(edgeCenter-edgeGuess)<30))
                if doPlot == True:

  
                    pylab.figure()
                    pylab.plot(bin_ctrs, contents)
                    pylab.plot(bin_ctrs, numpy.polyval(pfit, bin_ctrs))
                    pylab.plot(edgeGuess, numpy.polyval(pfit, edgeGuess),'.')
                    pylab.plot([preGuess, postGuess], numpy.polyval(pfit,[preGuess, postGuess]),'.')       
                    pylab.plot(bin_ctrs, self.edgeModel(bin_ctrs, edgeCenter, 
                                preHeight, postHeight, width))             
                    pylab.ylabel('counts per %4.2f unit bin'%(bin_ctrs[1]-bin_ctrs[0]))
                    pylab.xlabel('p_filt_value_dc')
                    pylab.title('%s center=%.1f, dCenter=%.3f, width=%.3f, %s'%(edge_name, edgeCenter, dEdgeCenter, width, usedStr))
                print('cal ds_num %d, %s, edgeCenter %.2f, dEdgeCenter %.3f, edgeDropCounts %.1f, %s'%(ds_num, edge_name, edgeCenter, dEdgeCenter, preHeight-postHeight, usedStr))
                if usedStr == 'used':
                    returnData[ds_num, i] = edgeCenter
                returnData2[ds_num, i] = preHeight-postHeight
#                except:
#                    print('cant find %s in channel %d'%(edge_name, ds.channum))
        self.convert_to_energy(use_drift_correct=True)
        return returnData, returnData2
        
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
        
    def convert_to_energy(self, use_drift_correct=True):
        for ds in self.data.datasets:
            try:
                cal = ds.calibration['p_filt_value']
                if use_drift_correct == True:
                    ds.p_energy = cal(ds.p_filt_value_dc)
                else:
                    ds.p_energy = cal(ds.p_filt_value)
            except:
                self.data.set_chan_bad(ds.channum, 'failed convert_to energy')
            
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

        
    def plot_calibration(self, power=0.0, whichCalibration = 'p_filt_value'):
        pylab.clf()
        ax = pylab.subplot(111)
        pylab.title(whichCalibration)

        for ds in self.data:
            ds.calibration[whichCalibration].plot(ax, ph_rescale_power=power)

        
    def store_filters_and_cal(self, filename='calibration.pkl'):
        try:
            cals,filters, dc_slope, dc_meanpretrigmean, dc_power = load_filters_and_cal(filename)
        except:
            cals = {}
            filters = {}
            drift_correct_info = {}
        fp = open(filename, 'wb')
        pickler = cPickle.Pickler(fp, protocol=cPickle.HIGHEST_PROTOCOL)

        for ds_num,ds in enumerate(self.data.datasets):
            cals[ds.channum] = ds.calibration
            filters[ds.channum]=ds.filter
            drift_correct_info[ds.channum] = ds.drift_correct_info

        pickler.dump(cals)
        pickler.dump(filters)
        pickler.dump(drift_correct_info)
        fp.close()
        
    def load_filters_and_cal(self, filename="calibration.pkl", applyLoaded=True):
        fp = open(filename, 'rb')
        up = cPickle.Unpickler(fp)
        cals = up.load()
        filters = up.load()
        drift_correct_info = up.load()
        fp.close()
        for ds in self.data:
            try:
                ds.calibration = cals[ds.channum]
                ds.filter = filters[ds.channum]
                ds.drift_correct_info = drift_correct_info[ds.channum]
            except:
                self.data.set_chan_bad(ds.channum, 'failed load_filters_and_cal')
        if applyLoaded:
            self.apply_filter()
            self.apply_stored_drift_correct()

    def countRateAndCuts(self,ds_num = 0):
        ds = self.data.datasets[ds_num]
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
        
    def countRatePlot(self, usefulEnergyRange = (5300, 6000) ):
        countsPassedCuts = numpy.zeros_like(self.data.datasets)
        totalCounts = numpy.zeros_like(self.data.datasets)
        elapsedTime = numpy.zeros_like(self.data.datasets)
        usefulCounts = numpy.zeros_like(self.data.datasets)
        for i,ds in enumerate(self.data.datasets):
            
            countsPassedCuts[i] = ds.cuts.good().sum()
            totalCounts[i] = ds.nPulses-1000 # the -1000 is meant to account for the Fe55 pulses
            elapsedTime[i] = ds.p_timestamp[ds.cuts.good()][-1]-ds.p_timestamp[ds.cuts.good()][0]
            usefulIndex = log_and(ds.cuts.good(), ds.p_energy>usefulEnergyRange[0], ds.p_energy<usefulEnergyRange[1])
            usefulCounts[i] = usefulIndex.sum()
            print('channel %d, trigger rate %4.2f/s, passed cuts rate %4.2f/s, useful rate %4.2f/s'%(ds.channum, 
                   totalCounts[i]/elapsedTime[i], countsPassedCuts[i]/elapsedTime[i], usefulCounts[i]/elapsedTime[i]))
            pylab.figure()
            bins = numpy.arange(0,16000,4)
            pylab.hist(ds.p_energy[ds.cuts.good()],bins)
            pylab.hist(ds.p_energy[usefulIndex], bins, color='r', edgecolor='r') # I don't understand why this doesn't work, it ignore the color, but it works from ipython
            pylab.ylabel('Energy (eV) 2 line calibration')
            pylab.ylabel('counts per %4.2f eV bin'%(bins[1]-bins[0]))
            pylab.title('chan %d, trigger %4.2f s^-1, passed cuts %4.2f s^-1, useful %4.2f s^-1'%(ds.channum,
                          totalCounts[i]/elapsedTime[i], countsPassedCuts[i]/elapsedTime[i], usefulCounts[i]/elapsedTime[i]))
            
        
        pylab.figure()
        pylab.plot(totalCounts/elapsedTime, 10*usefulCounts/elapsedTime,'bs',label='10x counts in range %d eV to %d eV'%usefulEnergyRange)
        pylab.plot(totalCounts/elapsedTime, countsPassedCuts/elapsedTime,'ro',label='counts passed cuts')
        pylab.xlabel('~trigger rate s^-1')
        pylab.ylabel('other count rates s^-1')
        pylab.legend()
        






