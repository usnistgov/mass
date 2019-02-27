import json
import numpy as np
import pylab as plt
import os
import base64


def recordDtype(offVersion,nBasis):
    """ return a np.dtype matching the record datatype for the given offVersion and nBasis"""
    if offVersion == "0.1.0":
        return np.dtype([("recordSamples",np.int32),("recordPreSamples",np.int32), ("framecount", np.int64),
         ("unixnano",np.int64),("pretriggerMean",np.float32),("residualStdDev",np.float32),("coefs",np.float32,(nBasis))])
    else:
        raise Exception("dtype for OFF version {} not implemented".format(version))

def readJsonString(f):
    """look in file f for a line "}\\n" and return all contents up to that point
    for an OFF file this can be parsed by json.dumps
    and all remaining data is records"""
    s = ""
    while True:
        line = f.readline()
        s+=line
        if line == "}\n":
            return s
        elif line == "":
            raise Exception("""reached end of file without finding a line "}\\n" """)


class OffFile():
    """
    Working with an OFF file:
    off = OffFile("filename")
    print off.dtype # show the fields available
    off[0] # get record 0
    off[0]["coefs"] # get the model coefs for record 0
    x,y = off.recordXY(0)
    plot(x,y) # plot record 0
    """
    def __init__(self,filename):
        self.filename = filename
        with open(self.filename,"r") as f:
            f = open(self.filename,"r")
            self.headerString=readJsonString(f)
            self.afterHeaderPos = f.tell()
        self.header = json.loads(self.headerString)
        self.dtype = recordDtype(self.header["FileFormatVersion"], self.header["NumberOfBases"])
        self.framePeriodSeconds = float(self.header["FramePeriodSeconds"])
        self.validateHeader()
        self._updateMmap()
        self._decodeModelInfo()

    def validateHeader(self):
        if self.header["FileFormat"] != "OFF":
            raise Exception("FileFormatVersion is {}, want OFF".format(self.header["FileFormatVersion"] ))

    def _updateMmap(self):
        fileSize = os.path.getsize(self.filename)
        recordSize = fileSize-self.afterHeaderPos
        self.nRecords = recordSize//self.dtype.itemsize
        self._mmap = np.memmap(self.filename,self.dtype,mode="r",
                              offset=self.afterHeaderPos, shape=(self.nRecords,))
        self.__getitem__ = self._mmap.__getitem__ # make indexing into the off the same as indexing into the memory mapped array
        self.__len__ = self._mmap.__len__
        self.__sizeof__ = self._mmap.__sizeof__
        self.shape = self._mmap.shape

    def _decodeModelInfo(self):
        projectorsData = base64.decodestring(self.header["ModelInfo"]["Projectors"]["RowMajorFloat64ValuesBase64"])
        projectorsRows = int(self.header["ModelInfo"]["Projectors"]["Rows"])
        projectorsCols = int(self.header["ModelInfo"]["Projectors"]["Cols"])
        self.projectors = np.frombuffer(projectorsData,np.float64)
        self.projectors = self.projectors.reshape((projectorsRows,projectorsCols))
        basisData = base64.decodestring(self.header["ModelInfo"]["Basis"]["RowMajorFloat64ValuesBase64"])
        basisRows = int(self.header["ModelInfo"]["Basis"]["Rows"])
        basisCols = int(self.header["ModelInfo"]["Basis"]["Cols"])
        self.basis = np.frombuffer(basisData,np.float64)
        self.basis = self.basis.reshape((basisRows,basisCols))
        if basisRows != projectorsCols or basisCols != projectorsRows or self.header["NumberOfBases"] != projectorsRows:
            raise Exception("basis shape should be transpose of projectors shape. have basis ({},{}), projectors ({},{}), NumberOfBases {}".format(
                basisCols,basisRows,projectorsCols,projectorsRows,NumberOfBases))

    def __repr__(self):
        return "<OFF file> {}, {} records, {} length basis\n".format(self.filename,self.nRecords,self.header["NumberOfBases"])


    def sampleTimes(self,i):
        """return a vector of sample times for record i, approriate for plotting"""
        recordSamples = self[i]["recordSamples"]
        recordPreSamples = self[i]["recordPreSamples"]
        return np.arange(-recordPreSamples,recordSamples-recordPreSamples)*self.framePeriodSeconds

    def modeledPulse(self,i):
        """return a vector of the modeled pulse samples, the best available value of the actual raw samples"""
        # projectors has size (n,z) where it is (rows,cols)
        # basis has size (z,n)
        # coefs has size (n,1)
        # coefs (n,1) = projectors (n,z) * data (z,1)
        # modelData (z,1) = basis (z,n) * coefs (n,1)
        # n = number of basis (eg 3)
        # z = record length (eg 4)
        allVals = np.matmul(self.basis, self[i]["coefs"])
        return allVals

    def recordXY(self,i):
        return self.sampleTimes(i), self.modeledPulse(i)





if __name__ == "__main__":
    off = OffFile("off_test.off")
    assert off.nRecords == 1
    print off
    x = off.sampleTimes(0)
    y = off.modeledPulse(0)
    assert len(x) == len(y)
