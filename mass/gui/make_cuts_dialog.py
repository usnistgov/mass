'''
Offers the convenient GUI dialog that produces a ??? CUTS OBJECT???.

Usage:

cuts = mass.gui.create_cuts(mass_data_group)

Created on Apr 23, 2012

@author: fowlerj
'''

__all__ = ['create_cuts']


from PyQt4 import QtGui ,QtCore
from PyQt4.QtCore import pyqtSlot
import os 
import numpy, pylab
import mass


# Import the form created by QtDesigner.  If possible, use a *.py file created
# with the pyuic command-line tool and import as usual....
try:
    from make_cuts_dialog_form_ui import Ui_Dialog
    
# ...though this is not always possible.  If necessary, use the PyQt4.uic package.
# In fact, I have not figured out how to have distutils convert the *.ui to a *.py
# file automatically, so the above will basically always fail.  Hmmm.

except ImportError:
    import PyQt4.uic
    path,_ = os.path.split(__file__)
    ui_filename = os.path.join(path, "make_cuts_dialog_form.ui")
    Ui_MakeCuts, _load_data_dialog_baseclass = PyQt4.uic.loadUiType(ui_filename)



def create_cuts(datagroup, existing_cuts=None):
    """
    Use the _CutsCreator dialog class to generate lists of pulse files, noise files, or both; to
    create a mass.TESGroup object from the lists; and (optionally) to run summarize_data on it.
    
    Arguments:
    
    Returns: a mass.TESGroup object, or (if user cancels or selects no files) None.
    """
    dialog = _CutsCreator(datagroup, existing_cuts=existing_cuts)
    retval = dialog.exec_()
    if retval == _CutsCreator.Rejected:
        print "User chose not to load anything."
        return None
    
    assert retval == _CutsCreator.Accepted

    cuts = dialog.generate_mass_cuts()
    if dialog.apply_cuts_check.isChecked():
        for ds in datagroup.datasets:
            ds.apply_cuts(cuts)
    return cuts


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
#        self.axes = self.fig.add_subplot(111)
#        # We want the axes cleared every time plot() is called
#        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.MinimumExpanding,
                                   QtGui.QSizePolicy.MinimumExpanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass



class CutVectorStatus(object):
    def __init__(self, name, **kwargs):
        self.use_min = False
        self.use_max = False
        self.cut_min = 0.0
        self.cut_max = 0.0
        self.actual_min = None
        self.actual_max = None
        self.__dict__.update(kwargs)
        
        self.use_hist_min = self.use_min
        self.use_hist_max = self.use_max
        self.hist_min = self.cut_min
        self.hist_max = self.cut_max
        self.__dict__.update(kwargs)

    def compute_actual_range(self, data_vectors):
        a,b = 9e99, -9e99
        for v in data_vectors:
            a = min(a, v.min())
            b = max(b, v.max())
        self.actual_min, self.actual_max = a,b

    def __repr__(self):
        d = ",".join(["%s=%s"%(k,v) for (k,v) in self.__dict__.iteritems()])
        return "CutVectorStatus(%s)"%d
    
    def get_cut_tuple(self):
        t=[None,None]
        if self.use_min:
            t[0] = self.cut_min
        if self.use_max:
            t[1] = self.cut_max
        return tuple(t)



class _CutsCreator(QtGui.QDialog, Ui_Dialog):
    """A Qt "QDialog" object for choosing a template filename for noise and/or pulse files,
    to select which channels to load, and to hold the results.
    
    This class is meant to be used by factory function make_cuts, and not by the end
    user.  Jeez.  Why are you even reading this?
    """
    def __init__(self, datagroup, existing_cuts=None ):
        QtGui.QDialog.__init__(self, parent=None)
        self.setupUi(self)
        frame_size = self.pylab_holder.frameSize()
        h,w = frame_size.height(), frame_size.width()
        dpi = 100.0
        print h,w,dpi ,'xxxx'
        self.canvas = MyMplCanvas(self.pylab_holder, w/dpi, h/dpi, dpi)
        fig = self.canvas.figure
        self.canvas.axes = [fig.add_subplot(3,1, 1),
                            fig.add_subplot(3,2,3),
                            fig.add_subplot(3,2,4),
                            fig.add_subplot(3,2,5),
                            fig.add_subplot(3,2,6),
                            ]

        self.data = datagroup
        self.n_channels = self.data.n_channels
        
        # Start with cuts at default values
        self.cuts = (CutVectorStatus("Pulse average", use_min=True, cut_min=0.0),
                     CutVectorStatus("Pretrigger RMS", use_max=True, cut_max=10.0),
                     CutVectorStatus("Pretrigger mean", use_hist_min=True, hist_min=-50,
                                     use_hist_max=True, hist_max=50),
                     CutVectorStatus("Peak Value", use_min=True, cut_min=0.0),
                     CutVectorStatus("Max posttrig dp/dt", use_max=True, cut_max=30.0), 
                     CutVectorStatus("Rise time (ms)", use_max=True, cut_max=0.7), 
                     CutVectorStatus("Peak time (ms)", use_max=True, cut_max=0.5))
        
        # If user constructed this with existing cuts, then include them here.
        if existing_cuts is not None:
            cuts=existing_cuts.cuts_prm
            for i,name in enumerate(("pulse_average", "pretrigger_rms",
                                    "pretrigger_mean_departure_from_median",
                                    "peak_value", "max_posttrig_deriv",
                                    "rise_time_ms", "peak_time_ms")):
                if name in cuts:
                    a,b = cuts[name]
                    if a is not None: 
                        self.cuts[i].use_min = True
                        self.cuts[i].cut_min = a
                    if b is not None:
                        self.cuts[i].use_max = True
                        self.cuts[i].cut_max = b
        
        for i, vector_name in enumerate(("p_pulse_average","p_pretrig_rms",
                                         "p_pretrig_mean",
                                        "p_peak_value","p_max_posttrig_deriv",
                                        "p_rise_time","p_peak_time")):
            self.cuts[i].compute_actual_range((ds.__dict__[vector_name] for ds in self.data.datasets))
            print i, self.cuts[i].actual_min, self.cuts[i].actual_max
        
        for button in (self.use_max_cut, self.use_min_cut,
                       self.use_hist_max, self.use_hist_min):
            button.clicked.connect(self.toggle_use_cut)
        self.apply_cuts.clicked.connect(self._apply_all_cuts)
        self.clear_cuts.clicked.connect(self._clear_all_cuts)
            
        self.changed_parameter_number(0)
        self.changed_dataset_count("4")


    def accept(self):
        if self.apply_cuts_check.isChecked():
            self._apply_all_cuts()
        QtGui.QDialog.accept(self)

    @pyqtSlot()
    def _apply_all_cuts(self):
        cuts = self.generate_mass_cuts()
        for ds in self.data.datasets:
            ds.apply_cuts(cuts)
    
    @pyqtSlot()
    def _clear_all_cuts(self):
        for ds in self.data.datasets:
            ds.clear_cuts()

    @pyqtSlot(float)
    def changed_cut_parameter(self, paramval): 
        print "New value: ",paramval
        sender = self.sender()
        if sender==self.cuts_min_spin:
            self.cuts[self.current_param].cut_min = self.cuts_min_spin.value()
        else:
            assert sender==self.cuts_max_spin
            self.cuts[self.current_param].cut_max = self.cuts_max_spin.value()
#        self.update_plots()
       
    def toggle_use_cut(self):
        sender = self.sender()
        state = sender.isChecked()
        if sender==self.use_max_cut:
            self.cuts[self.current_param].use_max = state
            if state:
                self.cuts[self.current_param].cut_max = self.cuts_max_spin.value()
        elif sender == self.use_min_cut:
            self.cuts[self.current_param].use_min = state
            if state:
                self.cuts[self.current_param].cut_min = self.cuts_min_spin.value()
        elif sender == self.use_hist_max:
            self.cuts[self.current_param].use_hist_max = state
            if state:
                self.cuts[self.current_param].hist_min = self.hist_min_spin.value()
        elif sender == self.use_hist_min:
            self.cuts[self.current_param].use_hist_min = state
            if state:
                self.cuts[self.current_param].hist_max = self.hist_max_spin.value()
    
    @pyqtSlot(QtCore.QString)
    def changed_dataset_count(self, newval):
        nchan_plot = int(newval)
        menu_choices = ["%d-%d"%(i, i+nchan_plot-1) for i in range(0,self.n_channels, nchan_plot)]
        
        # The last choice needs to be corrected if the n_channels isn't a multiple of nchan_plot
        if (self.n_channels%nchan_plot)==1:
            menu_choices[-1] = "%d"%(self.n_channels-1)
        elif (self.n_channels%nchan_plot) > 1:
            menu_choices[-1] = "%d-%d"%(nchan_plot*(len(menu_choices)-1), self.n_channels-1)
        
        self.dataset_chooser.clear()
        for i,m in enumerate(menu_choices):
            self.dataset_chooser.addItem(QtCore.QString(m))
        self.update_plots()
    
    @pyqtSlot(int)
    def changed_parameter_number(self, newparam):
        """Callback for when user chooses a new parameter to view."""
        self.current_param = newparam
        
        self._update_gui_cuts_limits()
        self.update_plots()
        
    def _update_gui_cuts_limits(self):
        cut = self.cuts[self.current_param]
        self.use_max_cut.setChecked(cut.use_max)
        self.cuts_max_spin.setEnabled(cut.use_max)
        self.cuts_max_spin.setValue(cut.cut_max)

        self.use_min_cut.setChecked(cut.use_min)
        self.cuts_min_spin.setEnabled(cut.use_min)
        self.cuts_min_spin.setValue(cut.cut_min)
        
        self.use_hist_max.setChecked(cut.use_hist_max)
        self.hist_max_spin.setEnabled(cut.use_hist_max)
        self.hist_max_spin.setValue(cut.hist_max)
        
        self.use_hist_min.setChecked(cut.use_hist_min)
        self.hist_min_spin.setEnabled(cut.use_hist_min)
        self.hist_min_spin.setValue(cut.hist_min)
        
    
    def color_of_channel(self, ichan):
        cm = pylab.cm.spectral   #@UndefinedVariable
        return cm(1.0*ichan/self.n_channels)
    
    @pyqtSlot()
    def update_plots(self):
        for ax in self.canvas.axes: ax.clear()
        axis = self.canvas.axes[0]

        cut = self.cuts[self.current_param]
        limits=[0,0]
        if self.use_hist_min.isChecked():
            limits[0] = self.hist_min_spin.value()
            cut.hist_min = limits[0]
        else:
            limits[0] = cut.actual_min 

        if self.use_hist_max.isChecked():
            limits[1] = self.hist_max_spin.value()
            cut.hist_max = limits[1]
        else:
            limits[1] = cut.actual_max


        hist_all=[]
        hist_good=[]
        for dsnum,ds in enumerate(self.data.datasets): 
            raw = (ds.p_pulse_average,
                   ds.p_pretrig_rms,
                   ds.p_pretrig_mean,
                   ds.p_peak_value,
                   ds.p_max_posttrig_deriv,
                   ds.p_rise_time,
                   ds.p_peak_time)[self.current_param][ds.cuts.good()]
            if self.current_param in (5,6):
                raw = raw*1000 # Convert seconds to ms
            if self.current_param == 2:
                raw = raw-numpy.median(raw)
                
            useable = numpy.ones(len(raw), dtype=numpy.bool)
            if cut.use_max:
                useable = raw<cut.cut_max
            if cut.use_min:
                useable = numpy.logical_and(useable, raw>cut.cut_min)
            
            h1, bins = numpy.histogram(raw, 150, limits)
            h2, bins = numpy.histogram(raw[useable], 150, limits)
            hist_all.append(h1)
            hist_good.append(h2)
            
        # Decide which channels go in the 4 subplots
        subaxis_number = {}
        n_per_sub = int(self.num_channel_chooser.currentText())/4
        chan_range_str = str(self.dataset_chooser.currentText())
        print "Handling ",chan_range_str
        if len(chan_range_str)==0:
            self.canvas.draw()
            return
        
        elif "-" in chan_range_str:
            chan_range = [int(s) for s in chan_range_str.split("-")]
        else:
            chan_range = int(chan_range_str), int(chan_range_str)
        assert n_per_sub*4 >=1+chan_range[1]-chan_range[0]
        
        
        for i in range(self.n_channels):
            sn = 1+(i-chan_range[0])/n_per_sub
            if sn>0 and sn<5:
                subaxis_number[i] = sn 
            
        offset = .6*numpy.max([h.max() for h in hist_all])
        n_offsets_per_sub = [0,0,0,0,0]
        
        for dsnum, (h1,h2) in enumerate(zip(hist_all,hist_good)):
            color = self.color_of_channel(dsnum)
            if (h1 != h2).any():
                mass.plot_as_stepped_hist(axis, h1+dsnum*offset, bins, color='gray')
            mass.plot_as_stepped_hist(axis, h2+dsnum*offset, bins, color=color)
            if dsnum in subaxis_number:
                subaxis = self.canvas.axes[subaxis_number[dsnum]]
                this_offset = n_offsets_per_sub[subaxis_number[dsnum]]*offset
                if (h1 != h2).any():
                    mass.plot_as_stepped_hist(subaxis, h1+this_offset, bins, color='gray')
                mass.plot_as_stepped_hist(subaxis, h2+this_offset, bins, color=color)
                subaxis.text(bins[5], 0.2*offset+this_offset, "Channel %d"%dsnum, color=color)
                n_offsets_per_sub[subaxis_number[dsnum]] += 1
            
        xlabel = ("Pulse average","Pretrigger RMS","Pretrigger mean (median subtracted)", "Peak Value",
                  "Max posttrig dp/dt", "Rise time (ms)", "Peak time (ms)")[self.current_param]
        axis.set_title(xlabel)
        self.canvas.draw()
    
    
    def generate_mass_cuts(self):
        cuts_avg = self.cuts[0].get_cut_tuple()
        cuts_rms = self.cuts[1].get_cut_tuple()
        cuts_ptm = self.cuts[2].get_cut_tuple()
    #    cuts_pkv = self.cuts[3].get_cut_tuple() # Ignore for now???
        cuts_ptd = self.cuts[4].get_cut_tuple()
        cuts_rtm = self.cuts[5].get_cut_tuple()
        cuts_pkt = self.cuts[6].get_cut_tuple()
        print cuts_avg, cuts_rms, cuts_ptm, cuts_ptd, cuts_rtm, cuts_pkt,'xxxx000'
        cuts = mass.core.controller.AnalysisControl(
                pulse_average=cuts_avg,
                pretrigger_rms=cuts_rms,
                pretrigger_mean_departure_from_median=cuts_ptm,
    #            peak_value=cuts_pkv,
                max_posttrig_deriv=cuts_ptd,
                rise_time_ms=cuts_rtm,
                peak_time_ms=cuts_pkt
                )
        return cuts
        