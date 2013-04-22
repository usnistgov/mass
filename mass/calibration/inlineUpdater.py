import time, sys

class InlineUpdater():
    def __init__(self, baseString):
        self.minElapseTimeForCalc = 1.0
        self.startTime = time.time()
        self.baseString = baseString
    def update(self, fracDone):
        self.fracDone = fracDone
        sys.stdout.write('\r'+self.baseString + ' %.1f%% done, estimated %s left'%(self.fracDone*100.0, self.timeRemainingStr))
        sys.stdout.flush()
        if fracDone >= 1:
            sys.stdout.write('\n'+self.baseString+' finished in %s'%self.elapsedTimeStr+'\n')
    @property
    def timeRemaining(self):
        if self.elapsedTimeSec > self.minElapseTimeForCalc:
            fracRemaining = 1 - self.fracDone
            rate = self.fracDone/self.elapsedTimeSec  
            return fracRemaining/rate                 
        else:
            return -1
    @property
    def timeRemainingStr(self):
        timeRemaining = self.timeRemaining
        if timeRemaining == -1:
            return '?'
        else:
            return '%.1f min'%(timeRemaining/60.0)
    @property
    def elapsedTimeSec(self):
        return time.time()-self.startTime
    @property
    def elapsedTimeStr(self):
        return '%.1f min'%(self.elapsedTimeSec/60.0)