#!/usr/bin/env python
"""
asdjfhaskdfhlsdjkfh


Created on June 7, 2012

@author: fowlerj
"""

__all__ = ['create_dataset']


import os
import re
import sys
import glob
import subprocess
import pylab
import numpy

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import pyqtSlot

import mass

# Import the form created by QtDesigner.  If possible, use a *.py file created
# with the pyuic command-line tool and import as usual....
try:
    from compute_gains_form import Ui_CreateDataset
    
# ...though this is not always possible.  If necessary, use the PyQt4.uic package.
# In fact, I have not figured out how to have distutils convert the *.ui to a *.py
# file automatically, so the above will basically always fail.  Hmmm.

except ImportError:
    import PyQt4.uic
    path, _ = os.path.split(__file__)
    ui_filename = os.path.join(path, "compute_gains_form.ui")
    Ui_CreateDataset, _load_data_dialog_baseclass = PyQt4.uic.loadUiType(ui_filename)


class _DataLoader(QtGui.QDialog, Ui_CreateDataset):
    """A Qt "QDialog" object for choosing a template filename for noise and/or pulse files,
    to select which channels to load, and to hold the results.
    
    This class is meant to be used by factory function create_dataset, and not by the end
    user.  Jeez.  Why are you even reading this?
    """
    def __init__(self, parent=None, directory="", disabled_channels=()):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.choose_pulse_file.clicked.connect(self.choose_file)
        self.nchannels = 0
        self.channels_known = []
        self.chan_check_boxes = []        
        self.use_only_odd_channels = True
        self.disabled_channels = disabled_channels
        self.default_directory = directory
        self.pulse_files = {}
        self.channel_check_status = {}  # Keep a record of whether channel is wanted!
        self.line_energy = None
        self.output_file_textEdited()
        
    def accept(self):
        """Don't let user accept dialog box if doesn't accept clobbering output file...."""
        if os.path.isfile(str(self.OutputFile.text())):
            msg = QtGui.QMessageBox()
            msg.setText("This output file exists already.")
            msg.setInformativeText("Do you want to overwrite it?")
            msg.setStandardButtons(msg.Yes | msg.No)
            msg.setDefaultButton(msg.No)
            answer = msg.exec_()
            if answer == msg.No:
                self.OutputFile.setFocus()
                return

        QtGui.QDialog.accept(self)
        
    def output_file_textEdited(self): pass
#        filename = str(self.OutputFile.text())
#        direct, _base = os.path.split(filename)
#        print "Testing filename: '%s' with dir='%s', base='%s'"%(filename,direct,_base)
#        if not os.path.isdir(direct):
#            return

    @pyqtSlot(QtCore.QString)
    def energy_menu_chooser(self, item_text):
        item = str(item_text)

        # Matches digits (with optional decimal and more digits), whitespace, and 
        # 'ev', 'kev', 'mev', or 'gev'
        expression = r"(?P<energy>\d*(\.\d*)?)\s*(?P<units>[kmg]?ev)"
        rexp = re.compile(expression, re.IGNORECASE)
        match = rexp.search(item)
        if match:
#            print "energy: ", match.group('energy')
#            print "units:  ", match.group('units')
            e = float(match.group('energy'))
            units = match.group('units').lower()
            if units == 'kev':
                e *= 1000
            elif units == 'mev':
                e *= 1000000
            elif units == 'gev':
                e *= 1e9
            self.LineEnergySpin.setValue(e)
            return

    def choose_file(self, *args, **kwargs):
        filename = QtGui.QFileDialog.getOpenFileName(parent=self,
                           caption=QtCore.QString("pick a file"),
                           directory=self.default_directory,  
                           filter="LJH Files (*.ljh *.noi)")
        if len(str(filename)) > 0:
            if self.sender() == self.choose_pulse_file:
                self.pulse_file_edit.setText(filename)
                
    @pyqtSlot(QtCore.QString)
    def file_template_textEdited(self, file_string):
        filename = str(file_string)
#        print "Processing new file template: ",filename
        if len(filename) > 0 and not os.path.exists(filename):
            return
        
        if self.sender() == self.pulse_file_edit:
            self.update_known_channels(filename, self.pulse_files)
    
    def update_known_channels(self, file_example, file_dict):
        file_example = str(file_example)
        rexp = re.compile(r'chan[0-9]+')
        file_template = "chan*".join(rexp.split(file_example))
        all_files = glob.glob(file_template)
        
        channels_found = []
        file_dict.clear()
        for f in all_files:
            m = rexp.search(f)
            chanstr = m.group()
            if chanstr.startswith("chan"):
                channum = int(chanstr[4:])
                channels_found.append(channum)
                file_dict[channum] = f
        
        if set(channels_found) != self.channels_known:
            all_chan = set(self.pulse_files.keys())
            all_chan = list(all_chan)
            all_chan.sort()
            self.channels_known = all_chan
            self.nchannels = len(all_chan)
            self.update_channel_chooser_boxes()

    def update_channel_chooser_boxes(self):
        # Remove all the chan_check_boxes, storing their check/unchecked status     
#        print "Deleting %d check boxes..."%len(self.chan_check_boxes)   
        while len(self.chan_check_boxes) > 0:
            ccb = self.chan_check_boxes.pop()
            ccb.hide()
            self.channel_check_status[ccb.chan_number] = ccb.isChecked()
            self.chan_selection_layout.removeWidget(ccb)
            del ccb
        
#        np = len(self.pulse_files)
#        print "Updating the channel chooser boxes with %d files"%(np)
        
        ncol = 16
        while (self.nchannels/ncol < 8) and (ncol > 8):
            ncol -= 2
        for i, cnum in enumerate(self.channels_known):
            name = QtCore.QString("%3d" % cnum)
            box = QtGui.QCheckBox(name, parent=None)
            box.chan_number = cnum
            
            if cnum in self.channel_check_status:
                box.setChecked(self.channel_check_status[cnum])
            else:
                if cnum in self.disabled_channels:
                    box.setChecked(False)
                else:
                    box.setChecked(True)

            col, row = i % ncol, i/ncol
            self.chan_selection_layout.addWidget(box, row, col)
            self.chan_check_boxes.append(box)
        self.chan_selection_label.setEnabled(True)
        self.check_all_chan.setEnabled(True)
        self.check_all_chan.clicked.connect(self.manipulate_chan_checker)
        self.check_default_chan.setEnabled(True)
        self.check_default_chan.clicked.connect(self.manipulate_chan_checker)
        self.check_no_chan.setEnabled(True)
        self.check_no_chan.clicked.connect(self.manipulate_chan_checker)
    
    def manipulate_chan_checker(self):
        if self.nchannels <= 0:
            return
        if self.sender() == self.check_all_chan:
            for box in self.chan_check_boxes:
                box.setChecked(True)
        elif self.sender() == self.check_default_chan:
            for box in self.chan_check_boxes:
                box.setChecked(box.chan_number not in self.disabled_channels)
        elif self.sender() == self.check_no_chan:
            for box in self.chan_check_boxes:
                box.setChecked(False)

    def get_pulse_files(self):
        return self._get_files(self.pulse_files)

    def get_noise_files(self):
        return self._get_files(self.noise_files)

    def _get_files(self, file_dict):
        chan = []
        file_list = []
        for box in self.chan_check_boxes:
            if box.isChecked():
                chan.append(box.chan_number)
                try:
                    filename = file_dict[box.chan_number]
                except KeyError:
                    continue
                if os.path.exists(filename):
                    file_list.append(filename)
        return file_list


def create_dataset(default_directory="", disabled_channels=()):
    """
    Use the _DataLoader dialog class to generate lists of pulse files, noise files, or both; to
    create a mass.TESGroup object from the lists; and (optionally) to run summarize_data on it.
    
    Arguments:
    default_directory   - The directory in which to start any file opening dialog.
    disabled_channels   - A sequence of channel numbers whose default state should be not-loaded
    
    Returns: a mass.TESGroup object, or (if user cancels or selects no files) None.
    """
    dialog = _DataLoader(disabled_channels=disabled_channels)
    retval = dialog.exec_()
    if retval == _DataLoader.Rejected:
        print("User cancelled.")
        return None
    
    assert retval == _DataLoader.Accepted
    pulse_files = dialog.get_pulse_files()
    npulses = dialog.MaxPulsesSpin.value()
    if npulses <= 0:
        npulses = None
    energy = dialog.LineEnergySpin.value()
    gain_file = dialog.OutputFile.text()
    
    np = len(pulse_files)
    if np > 0:
        data = mass.TESGroup(pulse_files, pulse_only=True)
    else:
        return None
    return data, npulses, energy, gain_file


def chan_from_dataset(ds):
    suffix = ds.filename.split("_chan")[-1]
    return int(suffix.split(".")[0])


def process_data(data, npulses=None, nsamples=None):
    """Compute the relevant pulse summary quantities."""
    print("Computing pulse heights for all data...")
    first_seg = 0
    end_seg = -1
    if npulses is not None:
        end_seg = 1+data.sample2segnum(npulses)

    end_sample = data.segnum2sample_range(end_seg-1)[1]
        
    # Compute pulse peak and filtered values    
    for ds in data.datasets:
        ds.p_peak_value = numpy.array(ds.p_peak_value, dtype=numpy.float)
    
    ndet = data.n_channels
    for first, end in data.iter_segments(first_seg, end_seg):
        print('...handling pulses %6d to %6d for all %d detectors' % (first, end-1, ndet))
        for ids, ds in enumerate(data.datasets):
            if first >= ds.nPulses:
                continue
            this_end = end
            if this_end >= ds.nPulses:
                this_end = ds.nPulses

            np, _ns = ds.data.shape
            if np != (this_end-first):
                print("Weird: np=%d, first,end=%d,%d" % (np, first, this_end)),
                print(" for ds[%d]" % ids)
                continue
            
            baseline = ds.data[:, :data.nPresamples-1].mean(axis=1)
            peak = ds.data.max(axis=1)-baseline
            
            ds.p_pretrig_mean[first:this_end] = baseline
            ds.p_peak_value[first:this_end] = peak
            
            if this_end == end_sample:
                ds.p_peak_value = ds.p_peak_value[:this_end]
                ds.p_pretrig_mean = ds.p_pretrig_mean[:this_end]
    

def find_center(values, first_cut=0.01):
    """Robust method to find the "center" of the values vector.
    It looks within a factor of <first_cut> around the median, so make sure it's wide
    enough for your purposes.  Usually 1% seems good (the default), but I can imagine
    wider peaks that need a larger value.
    
    Given the data within 1+-<first_cut> times the median, it finds a dispersion estimate
    and uses that to choose the width to use in a bisquare weighted mean.
    """
    ctr = numpy.median(values)
    good = numpy.abs(values/ctr-1.0) < 0.01
    try:
        sigma = mass.robust.shorth_range(values[good], normalize=True)
        ctr = mass.robust.bisquare_weighted_mean(values[good], k=4*sigma, center=ctr)
    except:
        pass
    return ctr
    

def find_peaks(data):

    # Compute median pulse value of pulses "in" the peak
    
    # Not currently doing filtering, since I don't know how!
    
    pylab.ioff()
    pylab.clf()
#    axis1 = pylab.subplot(211)
#    axis2 = pylab.subplot(212)
    axis1 = pylab.subplot(111)
    
    peak_lim = [.98, 1.10]
#    filt_lim = [.98,1.10]
    offset = 0.06*len(data.datasets[0].p_peak_value)
    nbins = 400
    
    for i, ds in enumerate(data.datasets):
        print(numpy.median(ds.p_pretrig_mean), numpy.median(ds.p_peak_value)),
        ds.ctr_peak = find_center(ds.p_peak_value)
        print(ds.ctr_peak)
#        ds.ctr_filt = find_center(ds.p_filt_value)
#        print ds.ctr_peak, ds.ctr_filt

        contents, bins = numpy.histogram(ds.p_peak_value/ds.ctr_peak, nbins, peak_lim)
        mass.plot_as_stepped_hist(axis1, contents+i*offset, bins, color='b')
        pylab.text(.981, (i+.1)*offset, 'Chan %2d' % chan_from_dataset(ds))
        
#        contents,bins = numpy.histogram(ds.p_filt_value/ds.ctr_filt, nbins, filt_lim)
#        mass.plot_as_stepped_hist(axis2, contents+i*offset, bins, color='g')

    for ax in (axis1,):
        ax.set_xlabel("Ratio of rescaled pulse size to line center")
        ax.set_ylabel("Pulses per bin (%d total bins)" % nbins)
    axis1.set_title("Spectrum of rescaled pulse heights")
#    axis2.set_title("Spectrum of rescaled filtered pulse heights")

    # Save the image and pop it up on the screen
    imgfile = "/tmp/compute_gains_histogram.png"
    pylab.savefig(imgfile)
    if sys.platform == 'darwin':
        args = ['open', imgfile]
    elif sys.platform.startswith('linux'):
        args = ['eog', imgfile]
    else:
        args = None
    if args:
        print("Histogram file is %s (if you need to see it again)." % imgfile)
        subprocess.Popen(args)


def save_gains(data, energy, filename=None):
    if filename is not None:
        fp = open(filename, "w")
        facts = """#
# Detector gains, estimated by compute_gains.py
# These involved a search for the center of the line located at the median pulse height.
# If your data don't have a median pulse in the middle of a bright line, then these make
# no sense!  Sorry
#
# Chan  PulseHt  PHGain  FiltPH  FPHGain
#
# (Gains are pulse height-to-eV ratios, if user gave a line energy in eV.
# If not, then gains are scaled to scale all PH towards the median value
# over all detectors.)
#
"""
        fp.write(facts)
    else:
        fp = None
        
    median_ctr_peak = numpy.median([ds.ctr_peak for ds in data.datasets])
    for ds in data.datasets:
        ichan = chan_from_dataset(ds)
        if energy > 0:
            gain = ds.ctr_peak/energy
        else:
            gain = ds.ctr_peak/median_ctr_peak
        line = "%3d %8.2f %8.6f\n" % (ichan, ds.ctr_peak, gain)
        print(line),
        if fp:
            fp.write(line)
    if fp:
        fp.close()


def main():
    _app = QtGui.QApplication(sys.argv)
    result = create_dataset()
    if result is None:
        return
    
    data, npulses, energy, filename = result
    process_data(data, npulses)
    find_peaks(data)
    save_gains(data, energy, filename)


if __name__ == '__main__':
    main()
