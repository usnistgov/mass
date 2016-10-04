"""
Created on Jun 9, 2014

@author: fowlerj
"""

import time
import sys


class MouseClickReader(object):
    """Object to serve as a callback for reading mouse clicks in data coordinates
    in pylab plots.  Will store self.b, .x, .y giving the button pressed,
    and the x,y data coordinates of the pointer when clicked.

    Usage example (ought to be here...):
    """

    def __init__(self, figure):
        """Connect to button press events on a pylab figure.
        \param figure The matplotlib.figure.Figure from which to capture mouse click events."""
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
        plot units.  Store in self.b, .x, and .y."""
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

    def update(self, fracDone):
        self.fracDone = fracDone
        sys.stdout.write('\r' + self.baseString +
                         ' %.1f%% done, estimated %s left' % (self.fracDone * 100.0, self.timeRemainingStr))
        sys.stdout.flush()
        if fracDone >= 1:
            sys.stdout.write('\n' + self.baseString + ' finished in %s' % self.elapsedTimeStr + '\n')

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
