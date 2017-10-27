import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py
import unittest
import fastdtw

def ds_shortname(self):
    """return a string containing part of the filename and the channel number, useful for labelling plots"""
    s = os.path.split(self.filename)[-1]
    chanstr = "chan%g"%self.channum
    if not chanstr in s:
        s+=chanstr
    return s

def data_shortname(self):
    """return a string containning part of the filename and the number of good channels"""
    ngoodchan = len([ds for ds in self])
    return os.path.split(self.datasets[0].filename)[-1]+", %g chans"%ngoodchan

def ds_hist(self,bin_edges,attr="p_energy",t0=0,tlast=1e20):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    vals = getattr(self, attr)[:]
    # sanitize the data bit
    tg = np.logical_and(self.p_timestamp[:]>t0,self.p_timestamp[:]<tlast)
    g = np.logical_and(tg,self.good())
    g = np.logical_and(g,~np.isnan(vals))

    counts, _ = np.histogram(vals[g],bin_edges)
    return bin_centers, counts

def data_hists(self,bin_edges,attr="p_energy",t0=0,tlast=1e20):
    """return a tuple of (bin_centers, countsdict). automatically filters out nan values
    where countsdict is a dictionary mapping channel numbers to numpy arrays of counts
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    countsdict = {ds.channum:ds.hist(bin_edges, attr)[1] for ds in self}
    return bin_centers, countsdict

def data_hist(self, bin_edges, attr="p_energy",t0=0,tlast=1e20):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses in all good datasets (use .hists to get the histograms individually). filters out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    """
    bin_centers, countsdict = self.hists(bin_edges, attr)
    counts = np.zeros_like(bin_centers, dtype="int")
    for (k,v) in countsdict.items():
        counts+=v
    return bin_centers, counts

def plot_hist(self,bin_edges,attr="p_energy",axis=None,label_lines=[]):
    """plot a coadded histogram from all good datasets and all good pulses
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    axis -- if None, then create a new figure, otherwise plot onto this axis
    annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
    """
    if axis is None:
        plt.figure()
        axis=plt.gca()
    x,y = self.hist(bin_edges, attr)
    axis.plot(x,y,drawstyle="steps-mid")
    axis.set_xlabel(attr)
    axis.set_ylabel("counts per %0.1f unit bin"%(bin_edges[1]-bin_edges[0]))
    axis.set_title(self.shortname())
    annotate_lines(axis, label_lines)

def annotate_lines(axis,label_lines, label_lines_color2=[],color1 = "k",color2="r"):
    """Annotate plot on axis with line names.
    label_lines -- eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    label_lines_color2 -- optional,eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    color1 -- text color for label_lines
    color2 -- text color for label_lines_color2
    """
    n=len(label_lines)+len(label_lines_color2)
    for (i,label_line) in enumerate(label_lines):
        energy = mass.STANDARD_FEATURES[label_line]
        axis.annotate(label_line, (energy, (1+i)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color1)
    for (i,label_line) in enumerate(label_lines_color2):
        energy = mass.STANDARD_FEATURES[label_line]
        axis.annotate(label_line, (energy, (1+i)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color2)

def ds_linefit(self,line_name="MnKAlpha", t0=0,tlast=1e20,axis=None,dlo=50,dhi=50,binsize=1,bin_edges=None, attr="p_energy",label="full",plot=True, guess_params=None, ph_units="eV"):
    """Do a fit to `line_name` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
    line_name -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
    dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    bin_edges -- pass the bin_edges you want as a numpy array
    attr -- default is "p_energy", you could pick "p_filt_value" or others. be sure to pass in bin_edges as well because the default calculation will probably fail for anything other than p_energy
    label -- passed to fitter.plot
    plot -- passed to fitter.fit, determine if plot happens
    guess_params -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
    ph_units -- passed to fitter.fit, used in plot label
    """
    if isinstance(line_name, mass.LineFitter):
        fitter = line_name
    else:
        fittername = line_name+"Fitter"
        fitter_class = getattr(mass,fittername)
        fitter = fitter_class()
        fitter.spect.nominal_peak_energy
    if bin_edges is None:
        bin_edges = np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    if axis is None and plot:
        plt.figure()
        axis = plt.gca()

    bin_centers, counts = self.hist(bin_edges, attr, t0, tlast)
    try:
        params, covar = fitter.fit(counts, bin_centers,params=guess_params,axis=axis,label=label, ph_units=ph_units,plot=plot)
        axis = plt.gca()
        axis.set_title(self.shortname())
    except ValueError, ex:
        fitter.success=False
        return fitter

    fitter.success = True
    return fitter

def normalizespectrum(counts): return counts/(1.0*counts.sum())
def chisqspectrumcompare(normcounts1,normcounts2): return  np.sum((normcounts1-normcounts2)**2)/len(normcounts1)
def keys_sorted_by_value(d): return np.array([k for k, v in sorted(d.iteritems(), key=lambda (k,v): (v,k))])
def rank_hists_chisq(countsdict):
    """Return chisqdict which maps channel number to chisq value. keys_sorted_by_value(chisqdict) may be useful.
    countsdict -- a dictionary mapping channel number to np arrays containing counts per bin
    """
    sumspectrum = np.zeros_like(countsdict.values()[0], dtype="int")
    for k,v in countsdict.items():
        sumspectrum+=v
    #normalize spectra
    normalizedsumspectrum = normalizespectrum(sumspectrum)
    chisqdict =  {ch:chisqspectrumcompare(normalizespectrum(spect),normalizedsumspectrum) for ch,spect in countsdict.items()}
    return chisqdict

def plot_ranked_hists(bin_centers, countsdict, chisqdict):
    channels_sorted_by_chisq = keys_sorted_by_value(chisqdict)
    plt.figure(figsize=(10,5))
    offsetsize = np.mean(normalizespectrum(countsdict[channels_sorted_by_chisq[0]]))
    for i,ch in enumerate(channels_sorted_by_chisq):
        plt.plot(bin_centers, offsetsize*i+normalizespectrum(countsdict[ch]), drawstyle="steps-mid")

def samepeaks(bin_centers, countsdict, npeaks, refchannel, gaussian_fwhm):
    refcounts = countsdict[refchannel]
    peak_locations, peak_intensities = mass.find_local_maxima(refcounts, peak_intensities)



mass.TESGroup.plot_hist = plot_hist
mass.TESGroup.hist = data_hist
mass.TESGroup.hists = data_hists
mass.TESGroup.shortname = data_shortname

mass.MicrocalDataSet.hist = ds_hist
mass.MicrocalDataSet.plot_hist = plot_hist
mass.MicrocalDataSet.shortname = ds_shortname
mass.MicrocalDataSet.linefit = ds_linefit


def ds_cut_calculated(ds):
    ds.cuts.clear_cut("pretrigger_rms")
    ds.cuts.clear_cut("postpeak_deriv")
    ds.cuts.clear_cut("pretrigger_mean")
    ds.cuts.cut_parameter(ds.p_pretrig_rms, ds.hdf5_group["calculated_cuts"]["pretrig_rms"][:], 'pretrigger_rms')
    ds.cuts.cut_parameter(ds.p_postpeak_deriv, ds.hdf5_group["calculated_cuts"]["postpeak_deriv"][:], 'postpeak_deriv')



# fitter=ds.linefit(mass.GaussianFitter(), bin_edges = np.arange(0,1000,5), attr="p_filt_value")
# if fitter.success:
#     print(fitter.last_fit_params_dict)
# else:
#     print("fitter failed")

class TestPlotAndHistMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data = mass.TESGroupHDF5("/Users/oneilg/Documents/molecular movies/Horton Wiring/horton_2017_07/20171006_B.ljh_pope.hdf5")
        self.data.set_chan_good(self.data.why_chan_bad.keys())
        for ds in self.data:
            ds_cut_calculated(ds)
            print("Chan %s, %g bad of %g"%(ds.channum, ds.bad().sum(), ds.nPulses))
        self.ds = self.data.first_good_dataset

        self.bin_edges  = np.arange(0,10000,1)

    def test_ds_hist(self):
        x,y = self.ds.hist(self.bin_edges)
        self.assertEqual(x[0], 0.5*(self.bin_edges[1]+self.bin_edges[0]))
        self.assertEqual(np.argmax(y),4511)

    def test_data_hists(self):
        x,countsdict = self.data.hists(self.bin_edges)
        self.assertTrue(all(countsdict[self.ds.channum]==self.ds.hist(self.bin_edges)[1]))
        self.assertTrue(len(x)==len(self.bin_edges)-1)
        len(countsdict.keys())==len([ds for ds in self.data])

    def test_plots(self):
        self.ds.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])
        self.data.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])

    def test_linefit(self):
        fitter = self.ds.linefit("MnKAlpha")
        self.assertTrue(fitter.success)

    def test_linefit_pass_fitter(self):
        fitter = self.ds.linefit(mass.MnKAlphaFitter(), bin_edges = np.arange(5850,5950), attr="p_energy")
        self.assertTrue(fitter.success)

    def test_rank_hists_chisq(self):
        bin_centers, countsdict = self.data.hists(np.arange(2000,10000))
        chisqdict = rank_hists_chisq(countsdict)
        print(chisqdict)
        plot_ranked_hists(bin_centers, countsdict, chisqdict)

if __name__ == "__main__":
    unittest.findTestCases("__main__").debug()
    unittest.main()
    plt.show()
