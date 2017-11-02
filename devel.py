import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py

def ds_shortname(self):
    """return a string containing part of the filename and the channel number, useful for labelling plots"""
    s = os.path.split(ds.filename)[-1]
    chanstr = "chan%g"%ds.channum
    if not chanstr in s:
        s+=chanstr
    return s

def data_shortname(self):
    """return a string containning part of the filename and the number of good channels"""
    ngoodchan = len([ds for ds in self])
    return os.path.split(self.datasets[0].filename)[-1]+", %g chans"%ngoodchan

def ds_hist(self,bin_edges,attr="p_energy",calc_centers=True):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute)
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    calc_centers -- saves some computation if false, will return None in place of actual bin_centers (this only exists to save some comutation for the version that loops over datasets)
    """
    bin_edges = np.array(bin_edges)
    if calc_centers:
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    else:
        bin_centers = None
    counts, _ = np.histogram(getattr(ds, attr)[ds.good()],bin_edges)
    return bin_centers, counts

def data_hists(self,bin_edges,attr="p_energy"):
    """return a tuple of (bin_centers, countsdict)
    where countsdict is a dictionary mapping channel numbers to numpy arrays of counts
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    countsdict = {ds.channum:ds.hist(bin_edges, attr, False)[1] for ds in data}
    return bin_centers, countsdict

def data_hist(self, bin_edges, attr="p_energy"):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses in all good datasets (use .hists to get the histograms individually)
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    """
    bin_centers, countsdict = data.hists(bin_edges, attr)
    counts = np.zeros_like(bin_centers, dtype="int")
    for (k,v) in countsdict.items():
        counts+=v
    return bin_centers, counts

def plot_hist(self,bin_edges,attr="p_energy",axis=None):
    """plot a coadded histogram from all good datasets and all good pulses
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    axis -- if None, then create a new figure, otherwise plot onto this axis
    """
    if axis is None:
        plt.figure()
        axis=plt.gca()
    x,y = self.hist(bin_edges, attr)
    axis.plot(x,y,drawstyle="steps-mid")
    axis.set_xlabel(attr)
    axis.set_ylabel("counts per %0.1f unit bin"%(bin_edges[1]-bin_edges[0]))
    axis.set_title(self.shortname())

def ds_linefit(self,line_name="MnKAlpha", t0=0,tlast=1e20,axis=None,dlo=50,dhi=50,binsize=1,bin_edges=None, attr="p_energy",label="full",plot=True):

    fittername = line_name+"Fitter"
    fitter_class = getattr(mass,fittername)
    fitter = fitter_class()
    fitter.spect.nominal_peak_energy

    if bin_edges is None:
        bin_edges = np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    if axis is None and plot:
        plt.figure()
        axis = plt.gca()
    tg = np.logical_and(ds.p_timestamp[:]>t0,ds.p_timestamp[:]<tlast)
    g = np.logical_and(tg,ds.good())
    g = np.logical_and(g,~np.isnan(ds.p_energy))

    counts,_ = np.histogram(ds.p_energy[g],bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    params, covar = fitter.fit(counts, bin_centers,axis=axis,label=label, ph_units="eV",plot=plot)

    return fitter

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
data = mass.TESGroupHDF5("/Users/oneilg/Documents/molecular movies/Horton Wiring/horton_2017_07/20171006_B.ljh_pope.hdf5")
data.set_chan_good(data.why_chan_bad.keys())
for ds in data:
    ds_cut_calculated(ds)
    print("Chan %s, %g bad of %g"%(ds.channum, ds.bad().sum(), ds.nPulses))

bin_edges = np.arange(0,40000,1)
bin_centers, countsdict = data.hists(bin_edges)

ds.plot_hist(bin_edges,"p_filt_value")
data.plot_hist(bin_edges,"p_filt_value")
