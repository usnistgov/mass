'''
Offers the convenient GUI dialog that produces a mass.TESGroup with help from the 
filesystem's file-finding features.

Usage:

data = mass.gui.create_dataset(disabled_channels=(3,5))

Created on Apr 19, 2012

@author: fowlerj
'''

__all__ = ['create_dataset']


from PyQt4 import QtCore, QtGui
import os, re
import glob

import mass

# Import the form created by QtDesigner.  If possible, use a *.py file created
# with the pyuic command-line tool and import as usual....
try:
    from load_data_dialog_form_ui import Ui_CreateDataset
    
# ...though this is not always possible.  If necessary, use the PyQt4.uic package.
# In fact, I have not figured out how to have distutils convert the *.ui to a *.py
# file automatically, so the above will basically always fail.  Hmmm.

except ImportError:
    import PyQt4.uic
    path,_ = os.path.split(__file__)
    ui_filename = os.path.join(path, "load_data_dialog_form.ui")
    Ui_CreateDataset, _load_data_dialog_baseclass = PyQt4.uic.loadUiType(ui_filename)


class _DataLoader(QtGui.QDialog, Ui_CreateDataset):
    """A Qt "QDialog" object for choosing a template filename for noise and/or pulse files,
    to select which channels to load, and to hold the results.
    
    This class is meant to be used by factory function create_dataset, and not by the end
    user.  Jeez.  Why are you even reading this?
    """
    def __init__(self, parent=None, disabled_channels=() ):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.choose_pulse_file.clicked.connect(self.choose_file)
        self.choose_noise_file.clicked.connect(self.choose_file)
        self.nchannels=0
        self.channels_known = []
        self.chan_check_boxes = []        
        self.use_only_odd_channels=True
        self.disabled_channels = disabled_channels
        self.pulse_files={}
        self.noise_files={}

    def choose_file(self, *args, **kwargs):
        filename = QtGui.QFileDialog.getOpenFileName(parent=self,
                           caption=QtCore.QString("pick a file"),
                           directory="/Users/Shared/Data/NSLS_data/",  
                           filter="LJH Files (*.ljh *.noi)")
        if len(str(filename))>0:
            if self.sender() == self.choose_pulse_file:
                self.pulse_file_edit.setText(filename)
                self.use_pulses.setChecked(True)
                self.build_known_channels(filename, self.pulse_files)
            elif self.sender() == self.choose_noise_file:
                self.noise_file_edit.setText(filename)
                self.use_noise.setChecked(True)
                self.build_known_channels(filename, self.noise_files)
    
    def build_known_channels(self, file_example, file_dict):
        file_example = str(file_example)
        rexp = re.compile(r'chan[0-9]+')
        file_template = "chan*".join(rexp.split(file_example))
        all_files = glob.glob(file_template)
        
        channels_found = []
        file_dict.clear()
        for f in all_files:
            m=rexp.search(f)
            chanstr = m.group()
            if chanstr.startswith("chan"):
                channum = int(chanstr[4:])
                channels_found.append(channum)
                file_dict[channum] = f
        
        all_chan = set(self.channels_known)
        all_chan.update(channels_found)
        all_chan = list(all_chan)
        all_chan.sort()
        self.channels_known = all_chan
        self.nchannels = len(all_chan)

        self.chan_check_boxes = []        
        ncol = 16
        while self.nchannels/ncol < 8 and ncol>8:
            ncol -=2
        for i,cnum in enumerate(self.channels_known):
            name = QtCore.QString("%3d"%cnum)
            box =QtGui.QCheckBox(name)
            box.chan_number = cnum
            if cnum not in self.disabled_channels:
                box.setChecked(True)
            col, row = i%ncol, i/ncol
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
        if self.nchannels<=0:
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


def create_dataset(disabled_channels=()):
    """
    Use the _DataLoader dialog class to generate lists of pulse files, noise files, or both; to
    create a mass.TESGroup object from the lists; and (optionally) to run summarize_data on it.
    
    Returns: a mass.TESGroup object, or (if user cancels or selects no files) None.
    """
    dialog = _DataLoader(disabled_channels=disabled_channels)
    retval = dialog.exec_()
    if retval == _DataLoader.Rejected:
        print "User chose not to load anything."
        return None
    
    assert retval == _DataLoader.Accepted
    pulse_files = dialog.get_pulse_files()
    noise_files = dialog.get_noise_files()
    
    np, nn = len(pulse_files), len(noise_files)
    if np>0:
        if nn>0:
            data = mass.TESGroup(pulse_files, noise_files, pulse_only=False)
        else:
            data =  mass.TESGroup(pulse_files, pulse_only=True)
    else:
        if nn>0:
            data = mass.TESGroup(noise_files, noise_only=True)
        else:
            return None
    if dialog.summarize_on_load.isChecked():
        data.summarize_data()
    return data

        

