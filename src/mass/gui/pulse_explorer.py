#!/usr/bin/env python

import os, sys, string, time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QHeaderView, QFileDialog, QWidget
from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QPlainTextEdit, QPushButton
from PyQt5.QtCore import QTimer, QThread, QObject, pyqtSignal, pyqtSlot, QAbstractTableModel, QSettings
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt

import mass.gui.pulse_explorer_ui as pulse_explorer_ui
import mass.gui.plot_ui as plot_ui

import numpy as np
from math import pi
import scipy.signal
import glob
import sip
import pickle

import mass

def as_bold(s):
    return '<b>' + s + '</b>'

def as_bold_14pt(s):
    return '<span style="font-size:14pt">' + as_bold(s) + '</span>'

#
# Simplest possible wrapper for embedding matplotlib plots
#
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

class MPLPlot(object):
    def __init__(self, widget):

        self.widget = widget

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, widget)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.widget.setLayout(layout)


    def clear(self):
        self.figure.clear()

    def draw(self):
        self.canvas.draw()

class PulseGUI(QWidget):

    def __init__(self, ds):
        QWidget.__init__(self)

        self.ui = pulse_explorer_ui.Ui_Form()
        self.ui.setupUi(self)

        self.ds = ds

        #
        # Setup plots
        #
        def init_plot_ui(plot_container, enable_picking):
            my_plot_ui = plot_ui.Ui_Form()
            my_plot_ui.setupUi(plot_container)
            my_plot_ui.plot = MPLPlot(my_plot_ui.plot)
            my_plot_ui.plot.enable_picking = enable_picking

            my_plot_ui.cb_x_parm.currentIndexChanged.connect(lambda x: self.update_plot_ui(my_plot_ui))
            my_plot_ui.cb_y_parm.currentIndexChanged.connect(lambda x: self.update_plot_ui(my_plot_ui))

            def update_if_histogram():
                if my_plot_ui.cb_histogram.isChecked():
                    self.update_plot_ui(my_plot_ui)

            my_plot_ui.cb_histogram.stateChanged.connect(lambda x: self.update_plot_ui(my_plot_ui))
            my_plot_ui.le_range_hi.editingFinished.connect(update_if_histogram)
            my_plot_ui.le_range_lo.editingFinished.connect(update_if_histogram)
            my_plot_ui.sb_n_bins.editingFinished.connect(update_if_histogram)

            return my_plot_ui
        
        self.plot_ui_1 = init_plot_ui(self.ui.plot1_container, True)

        self.plot2 = MPLPlot(self.ui.plot2)

        #
        # Load setting from config file
        # 
        self.settings = QSettings(".pulse.ini", QSettings.IniFormat)

        self.update()

    def update(self):
        #
        # GUI stuff
        #
        attrs = sorted(self.ds.__dict__.keys())

        def setup_parm_combo_boxes(my_plot_ui, default_x_parm, default_y_parm):
            cb_x_parm = my_plot_ui.cb_x_parm
            cb_y_parm = my_plot_ui.cb_y_parm
            
            # disconnect signals
            cb_x_parm.currentIndexChanged.disconnect()
            cb_y_parm.currentIndexChanged.disconnect()

            for k in range(cb_x_parm.count()):
                cb_x_parm.removeItem(0)
            for k in range(cb_y_parm.count()):
                cb_y_parm.removeItem(0)

            for key in attrs:
                if key.startswith('p_'):
                    cb_x_parm.addItem(key)
                    cb_y_parm.addItem(key)

            k = cb_x_parm.findText(default_x_parm)
            if k > -1:
                cb_x_parm.setCurrentIndex(k)
        
            k = cb_y_parm.findText(default_y_parm)
            if k > -1:
                cb_y_parm.setCurrentIndex(k)

            # reconnect signals
            cb_x_parm.currentIndexChanged.connect(lambda x: self.update_plot_ui(my_plot_ui))
            cb_y_parm.currentIndexChanged.connect(lambda x: self.update_plot_ui(my_plot_ui))
        
        setup_parm_combo_boxes(self.plot_ui_1, 'p_pulse_average', 'p_peak_value')

        self.update_plots()

    def update_plot_ui(self, plot_ui):
        if (self.ds != None):
            plot_ui.plot.clear()
            ax = plot_ui.plot.figure.add_subplot(111)
            
            x_vals = self.ds.__dict__[str(plot_ui.cb_x_parm.currentText())]
            if plot_ui.cb_histogram.isChecked():
                range_lo = float(plot_ui.le_range_lo.text())
                range_hi = float(plot_ui.le_range_hi.text())
                nbins = plot_ui.sb_n_bins.value()
                print('hist', range_lo, range_hi, nbins)
                ax.hist(x_vals, range=[range_lo, range_hi], bins=nbins, histtype='step')
            else:
                y_vals = self.ds.__dict__[str(plot_ui.cb_y_parm.currentText())]
                g = self.ds.good()
                line, = ax.plot(x_vals, y_vals, '.', color='gray', picker=5)
                ax.plot(x_vals[g], y_vals[g], '.b', picker=5)
                if plot_ui.plot.enable_picking:
                    self.line = line
                    cid = plot_ui.plot.canvas.mpl_connect('pick_event', self.onpick)
                    
            plot_ui.plot.draw()

    def update_plots(self):
        self.update_plot_ui(self.plot_ui_1)

    def onpick(self, event):
            
        if event.artist not in [self.line]:
            return True
        
        N = len(event.ind)
        if not N: return True
        
        if event.mouseevent.button == 1:
            self.plot2.clear()

        ax = self.plot2.figure.add_subplot(111)

        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Current (R2 Units)')

        for k in event.ind:
            trace = self.ds.read_trace(k)
            if self.ds.good()[k]:
                ax.plot(trace, '-')
            else:
                ax.plot(trace, '--')

        if len(event.ind) == 1:
            self.ui.le_pulse_index.setText(str(event.ind[0]))
        else:
            self.ui.le_pulse_index.setText(str(event.ind))

        self.plot2.draw()

        return True

styleSheet = '''
QGroupBox {
    border: 2px solid black;
    border-radius: 9px;
    margin-top: 0.5em;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
'''

def explore(ds):
    window = PulseGUI(ds)
    basename = os.path.basename(ds.filename)
    window.setWindowTitle('Pulse Explorer - %s' % basename)
    window.setStyleSheet(styleSheet)
    window.show()
