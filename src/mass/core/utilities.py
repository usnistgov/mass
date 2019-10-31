"""
Various utility functions and classes:

* MouseClickReader: a class to use as a callback for reading mouse click
    locations in matplotlib plots.
* InlineUpdater: a class that loops over a generator and prints a message to
    the terminal each time it yields.
"""
import functools
import time
import sys
import glob
import os
import subprocess
import logging

import matplotlib.pylab as plt


class MouseClickReader(object):
    """A callback for reading mouse clicks in data coordinates in pylab plots.

    Stores self.b, .x, .y giving the button pressed, and the x,y data
    coordinates of the pointer when clicked.
    """

    def __init__(self, figure):
        """Connect to button press events on a pylab figure.

        Args:
            figure: The matplotlib.figure.Figure from which to capture mouse click events.
        """
        # The button number of the last mouse click inside a plot.
        self.b = 0
        # The x location of the last mouse click inside a plot.
        self.x = 0
        # The y location of the last mouse click inside a plot.
        self.y = 0
        # The Figure to whose events we are connected.
        self.fig = figure
        # The connection ID for matplotlib event handling.
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        """When called, capture the latest button number and the x,y location in
        plot units.  Store in self.b, .x, and .y.
        """
        self.b, self.x, self.y = event.button, event.xdata, event.ydata

    def __del__(self):
        """Disconnect the button press event from this object."""
        self.fig.canvas.mpl_disconnect(self.cid)


class InlineUpdater(object):

    def __init__(self, baseString):
        self.fracDone = 0.0
        self.minElapseTimeForCalc = 1.0
        self.startTime = time.time()
        self.baseString = baseString
        self.logger = logging.getLogger("mass")

    def update(self, fracDone):
        if self.logger.getEffectiveLevel() >= logging.WARNING:
            return
        self.fracDone = fracDone
        sys.stdout.write('\r' + self.baseString
                         + ' %.1f%% done, estimated %s left' % (self.fracDone * 100.0, self.timeRemainingStr))
        sys.stdout.flush()
        if fracDone >= 1:
            sys.stdout.write('\n' + self.baseString + ' finished in %s' %
                             self.elapsedTimeStr + '\n')

    @property
    def timeRemaining(self):
        if self.elapsedTimeSec > self.minElapseTimeForCalc and self.fracDone > 0:
            fracRemaining = 1 - self.fracDone
            rate = self.fracDone / self.elapsedTimeSec
            try:
                return fracRemaining / rate
            except ZeroDivisionError:
                return -1
        else:
            return -1

    @property
    def timeRemainingStr(self):
        timeRemaining = self.timeRemaining
        if timeRemaining == -1:
            return '?'
        else:
            return '%.1f min' % (timeRemaining / 60.0)

    @property
    def elapsedTimeSec(self):
        return time.time() - self.startTime

    @property
    def elapsedTimeStr(self):
        return '%.1f min' % (self.elapsedTimeSec / 60.0)


class NullUpdater(object):
    def update(self, f):
        pass


def show_progress(name):
    def decorator(func):
        @functools.wraps(func)
        def work(self, *args, **kwargs):
            try:
                print_updater = self.updater(name)
            except TypeError:
                print_updater = NullUpdater()

            for d in func(self, *args, **kwargs):
                print_updater.update(d)

        return work

    return decorator


def plot_multipage(data, subplot_shape, helper, filename_template_per_file,
                   filename_template_glob, filename_one_file, format, one_file):
    '''Helper function for multipage printing. See plot_summary_pages() for an example of how to use it. '''

    if format == 'pdf' and one_file:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(filename_one_file)

    (m, n) = subplot_shape
    plt.clf()
    for (k, ds) in enumerate(data.iter_channels()):
        ax = plt.subplot(m, n, k % (m*n) + 1)
        helper(ds, ax)

        if ((k+1) % (m*n)) == 0:
            plt.tight_layout(True)
            if format == 'pdf' and one_file:
                pdf.savefig()
            else:
                plt.savefig(filename_template_per_file % ((k+1)//(m*n)))
            plt.clf()

    # If final page is not full of plots, it hasn't yet been saved, so need to save it.
    if ((k+1) % (m*n) != 0):
        plt.tight_layout(True)
        if format == 'pdf' and one_file:
            pdf.savefig()
        else:
            plt.savefig(filename_template_per_file % ((k+1)//(m*n) + 1))

    if format == 'pdf' and one_file:
        pdf.close()

    # Convert to a single file if requested by user
    if format != 'pdf' and one_file:
        in_files = glob.glob(filename_template_glob)
        if len(in_files) > 0:
            in_files.sort()
            cmd = ['convert'] + in_files + [filename_one_file]
            ret = subprocess.call(cmd)
            if ret == 0:
                for f in in_files:
                    os.remove(f)
