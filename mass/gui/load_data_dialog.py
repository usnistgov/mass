"""
Offers the convenient GUI dialog that produces a mass.TESGroup with help from the 
filesystem's file-finding features.

Usage:

data = mass.gui.create_dataset(disabled_channels=(3,5))

Created on Apr 19, 2012

@author: fowlerj
"""

__all__ = ['create_dataset']


from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import pyqtSlot
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
    def __init__(self, parent=None, directory="", disabled_channels=() ):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.choose_pulse_file.clicked.connect(self.choose_file)
        self.choose_noise_file.clicked.connect(self.choose_file)
        self.nchannels=0
        self.channels_known = []
        self.chan_check_boxes = []        
        self.disabled_channels = disabled_channels
        self.default_directory = directory
        self.pulse_files={}
        self.noise_files={}
        self.channel_check_status={} # Keep a record of whether channel is wanted!

    def choose_file(self, *args, **kwargs):
        filename = QtGui.QFileDialog.getOpenFileName(self,
                          QtCore.QString("pick a file"),
                          self.default_directory,
                          "LJH Files (*.ljh *.noi)")
        if len(str(filename))>0:
            if self.sender() == self.choose_pulse_file:
                self.pulse_file_edit.setText(filename)
            elif self.sender() == self.choose_noise_file:
                self.noise_file_edit.setText(filename)

    @pyqtSlot(QtCore.QString)
    def file_template_textEdited(self, file_string):
        filename = str(file_string)
        if len(filename)>0 and not os.path.exists(filename):
            return

        if self.sender() == self.pulse_file_edit:
            self.use_pulses.setChecked(True)
            self.update_known_channels(filename, self.pulse_files)
        elif self.sender() == self.noise_file_edit:
            self.use_noise.setChecked(True)
            self.update_known_channels(filename, self.noise_files)

    @pyqtSlot()
    def update_enable_error_channels(self):
        file_example = self.pulse_file_edit.text()
        self.update_known_channels(file_example, self.pulse_files)
    
    def update_known_channels(self, file_example, file_dict):
        file_example = str(file_example)
        rexp = re.compile(r'chan[0-9]+')
        file_template = "chan*".join(rexp.split(file_example))
        all_files = glob.glob(file_template)

        channels_found = []
        file_dict.clear()
        for f in all_files:
            m=rexp.search(f)
            if m is None:
                continue
            chanstr = m.group()
            if chanstr.startswith("chan"):
                channum = int(chanstr[4:])
                if (channum%2==1) or self.enable_error_channels.isChecked():
                    channels_found.append(channum)
                    file_dict[channum] = f

        if set(channels_found) != self.channels_known:
            all_chan = set(self.pulse_files.keys())
            all_chan.update(self.noise_files.keys())
            all_chan = list(all_chan)
            all_chan.sort()
            self.channels_known = all_chan
            self.nchannels = len(all_chan)
            self.update_channel_chooser_boxes()

    def update_channel_chooser_boxes(self):

        # Remove all the chan_check_boxes, storing their check/unchecked status     
        while len(self.chan_check_boxes)>0:
            ccb = self.chan_check_boxes.pop()
            ccb.hide()
            self.channel_check_status[ccb.chan_number] = ccb.isChecked()
            self.chan_selection_layout.removeWidget(ccb)
            del ccb

        np, nn = len(self.pulse_files), len(self.noise_files)

        ncol = 16
        while self.nchannels/ncol < 8 and ncol>8:
            ncol -=2
        for i,cnum in enumerate(self.channels_known):
            name = QtCore.QString("%3d"%cnum)
            box = QtGui.QCheckBox(name, None)
            box.chan_number = cnum

            if cnum in self.channel_check_status:
                box.setChecked(self.channel_check_status[cnum])
            else:
                if cnum in self.disabled_channels:
                    box.setChecked(False)
                else:
                    box.setChecked(True)

            # For channels in one list but not the other, disable the box
            if np>0 and nn>0:
                if (cnum in self.noise_files and cnum not in self.pulse_files) or\
                    (cnum not in self.noise_files and cnum  in self.pulse_files):
                    box.setChecked(False)
                    box.setEnabled(False)

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
                    if os.stat(filename).st_size > 0:
                        file_list.append(filename)
                    else:
                        print "Warning: Zero-size file ignored:\n    %s"%filename
        return file_list



def _dataset_command(pulse_files, noise_files):
    """Returns a string containing executable python code that user can save
    to re-generate this dataset in the future."""
    
    np, nn = len(pulse_files), len(noise_files)
    pulse_only = (nn==0)
    noise_only = (np==0)
    if pulse_only and noise_only:
        return ""
    dir_p = dir_n = None
    lines=[]
    channels=[]
    
    # Find file conventions for the pulse and the noise files
    if np>0:
        dir_p, name = os.path.split(pulse_files[0])
        lines.append('dir_p = "%s"'%dir_p)
        prefix,after = name.split('_chan')
        suffix = after.split(".")[1]
        lines.append('prefix_p, suffix_p = "%s","%s"'%(prefix, suffix))

    if nn>0:
        dir_n, name = os.path.split(noise_files[0])
        lines.append('dir_n = "%s"'%dir_n)
        prefix,after = name.split('_chan')
        suffix = after.split(".")[1]
        lines.append('prefix_n, suffix_n = "%s","%s"'%(prefix, suffix))
    
    # Use the pulse or noise file list to find channel numbers
    file_list = pulse_files
    if noise_only:
        file_list = noise_files
        
    for pf in file_list:
        name=os.path.split(pf)[1]
        cnum = name.split('_chan')[1].split(".")[0]
        channels.append(cnum)
    lines.append('channels=(%s)'%(','.join(channels)))

    # Now construct the pulse_files and noise_files lists and generate the command
    args = []
    extra = []
    if noise_only:
        extra.append("noise_only=True")
    else:
        lines.append(
            'pulse_files=["%s/%s_chan%d.%s"%(dir_p, prefix_p, c, suffix_p) for c in channels]')
        args.append('pulse_files')

    if pulse_only:
        extra.append("pulse_only=True")
    else:
        lines.append(
            'noise_files=["%s/%s_chan%d.%s"%(dir_n, prefix_n, c, suffix_n) for c in channels]')
        args.append('noise_files')
    args.extend(extra)
    command = 'data = mass.TESGroup(%s)'%(', '.join(args))
    lines.append(command)
    lines.append('\n')
    return "\n".join(lines)



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

    print 75*'#'
    print '#   Use the following Python code in your script to generate this data set in the future'
    print 75*'#'
    print _dataset_command(pulse_files, noise_files)
    print 75*'#'
    print '# End of code'
    print 75*'#'
    
    if dialog.summarize_on_load.isChecked():
        data.summarize_data()
    return data

