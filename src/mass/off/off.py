import json
import numpy as np
import os
import base64


def recordDtype(offVersion, nBasis, descriptive_coefs_names=True):
    """ return a np.dtype matching the record datatype for the given offVersion and nBasis
    descriptive_coefs_names - determines how the modeled pulse coefficients are name, you usually want True
    For True, the names will be `derivLike`, `pulseLike`, and if nBasis>3, also `extraCoefs`
    For False, they will all have the single name `coefs`. False is to make implementing recordXY and other
    methods that want access to all coefs simultaneously easier"""
    if offVersion == "0.1.0" or offVersion == "0.2.0":
        # start of the dtype is identical for all cases
        dt_list = [("recordSamples", np.int32), ("recordPreSamples", np.int32), ("framecount", np.int64),
                   ("unixnano", np.int64), ("pretriggerMean", np.float32), ("residualStdDev", np.float32)]
        if descriptive_coefs_names:
            dt_list += [("pulseMean", np.float32), ("derivativeLike",
                                                    np.float32), ("filtValue", np.float32)]
            if nBasis > 3:
                dt_list += [("extraCoefs", np.float32, (nBasis-3))]
        else:
            dt_list += [("coefs", np.float32, (nBasis))]
        return np.dtype(dt_list)
    else:
        raise Exception("dtype for OFF version {} not implemented".format(offVersion))


def readJsonString(f):
    """look in file f for a line "}\\n" and return all contents up to that point
    for an OFF file this can be parsed by json.dumps
    and all remaining data is records"""
    s = ""
    while True:
        line = f.readline().decode("utf-8")
        s += line
        if line == "}\n":
            return s
        elif line == "":
            raise Exception("""reached end of file without finding a line "}\\n" """)


class OffFile(object):
    """
    Working with an OFF file:
    off = OffFile("filename")
    print off.dtype # show the fields available
    off[0] # get record 0
    off[0]["coefs"] # get the model coefs for record 0
    x,y = off.recordXY(0)
    plot(x,y) # plot record 0
    """

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "rb") as f:
            self.headerString = readJsonString(f)
            # self.headerStringLength = f.tell() # doesn't work on windows because readline uses a readahead buffer
            self.headerStringLength = len(self.headerString)
        self.header = json.loads(self.headerString)
        self.dtype = recordDtype(self.header["FileFormatVersion"], self.header["NumberOfBases"])
        self._dtype_non_descriptive = recordDtype(
            self.header["FileFormatVersion"], self.header["NumberOfBases"], descriptive_coefs_names=False)
        self.framePeriodSeconds = float(self.header["FramePeriodSeconds"])
        self.validateHeader()
        self._decodeModelInfo()  # calculates afterHeaderPos used by _updateMmap
        self._updateMmap()

    def validateHeader(self):
        with open(self.filename, "rb") as f:
            f.seek(self.headerStringLength-2)
            if not f.readline().decode("utf-8") == "}\n":
                raise Exception("failed to find end of header")
        if self.header["FileFormat"] != "OFF":
            raise Exception("FileFormatVersion is {}, want OFF".format(
                self.header["FileFormatVersion"]))

    def _updateMmap(self, _nRecords=None):
        """
        _nRecords is for testing only, mmap exaclty _nRecords records
        """
        fileSize = os.path.getsize(self.filename)
        recordSize = fileSize-self.afterHeaderPos
        if _nRecords is None:
            self.nRecords = recordSize//self.dtype.itemsize
        else: # for testing only
            self.nRecords = _nRecords
        self._mmap = np.memmap(self.filename, self.dtype, mode="r",
                               offset=self.afterHeaderPos, shape=(self.nRecords,))
        self.shape = self._mmap.shape

    def __getitem__(self, *args, **kwargs):
        # make indexing into the off the same as indexing into the memory mapped array
        return self._mmap.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self._mmap)

    def __sizeof__(self):
        return self._mmap.__sizeof__()

    def _decodeModelInfo(self):
        if "RowMajorFloat64ValuesBase64" in self.header["ModelInfo"]["Projectors"] and "RowMajorFloat64ValuesBase64" in self.header["ModelInfo"]["Basis"]:
            # should only be in version 0.1.0 files
            self._decodeModelInfoBase64()
        else:
            self._decodeModelInfoMmap()

    def _decodeModelInfoBase64(self):
        projectorsData = base64.decodestring(
            self.header["ModelInfo"]["Projectors"]["RowMajorFloat64ValuesBase64"].encode())
        projectorsRows = int(self.header["ModelInfo"]["Projectors"]["Rows"])
        projectorsCols = int(self.header["ModelInfo"]["Projectors"]["Cols"])
        self.projectors = np.frombuffer(projectorsData, np.float64)
        self.projectors = self.projectors.reshape((projectorsRows, projectorsCols))
        basisData = base64.decodestring(
            self.header["ModelInfo"]["Basis"]["RowMajorFloat64ValuesBase64"].encode())
        basisRows = int(self.header["ModelInfo"]["Basis"]["Rows"])
        basisCols = int(self.header["ModelInfo"]["Basis"]["Cols"])
        self.basis = np.frombuffer(basisData, np.float64)
        self.basis = self.basis.reshape((basisRows, basisCols))
        if basisRows != projectorsCols or basisCols != projectorsRows or self.header["NumberOfBases"] != projectorsRows:
            raise Exception("basis shape should be transpose of projectors shape. have basis ({},{}), projectors ({},{}), NumberOfBases {}".format(
                basisCols, basisRows, projectorsCols, projectorsRows, self.header["NumberOfBases"]))
        self.afterHeaderPos = self.headerStringLength

    def _decodeModelInfoMmap(self):
        projectorsRows = int(self.header["ModelInfo"]["Projectors"]["Rows"])
        projectorsCols = int(self.header["ModelInfo"]["Projectors"]["Cols"])
        basisRows = int(self.header["ModelInfo"]["Basis"]["Rows"])
        basisCols = int(self.header["ModelInfo"]["Basis"]["Cols"])
        # 8 for float64, basis and projectors have the same number of elements and therefore of bytes
        nBytes = basisCols*basisRows*8
        projectorsPos = self.headerStringLength
        basisPos = projectorsPos + nBytes
        self.afterHeaderPos = basisPos + nBytes
        self.projectors = np.memmap(self.filename, np.float64, mode="r",
                                    offset=projectorsPos, shape=(projectorsRows, projectorsCols))
        self.basis = np.memmap(self.filename, np.float64, mode="r",
                               offset=basisPos, shape=(basisRows, basisCols))
        if basisRows != projectorsCols or basisCols != projectorsRows or self.header["NumberOfBases"] != projectorsRows:
            raise Exception("basis shape should be transpose of projectors shape. have basis ({},{}), projectors ({},{}), NumberOfBases {}".format(
                basisCols, basisRows, projectorsCols, projectorsRows, self.header["NumberOfBases"]))

    def __repr__(self):
        return "<OFF file> {}, {} records, {} length basis\n".format(self.filename, self.nRecords, self.header["NumberOfBases"])

    def sampleTimes(self, i):
        """return a vector of sample times for record i, approriate for plotting"""
        recordSamples = self[i]["recordSamples"]
        recordPreSamples = self[i]["recordPreSamples"]
        return np.arange(-recordPreSamples, recordSamples-recordPreSamples)*self.framePeriodSeconds

    def modeledPulse(self, i):
        """return a vector of the modeled pulse samples, the best available value of the actual raw samples"""
        # projectors has size (n,z) where it is (rows,cols)
        # basis has size (z,n)
        # coefs has size (n,1)
        # coefs (n,1) = projectors (n,z) * data (z,1)
        # modelData (z,1) = basis (z,n) * coefs (n,1)
        # n = number of basis (eg 3)
        # z = record length (eg 4)

        # .view(self._dtype_non_descriptive) should be a copy-free way of changing the dtype so we can access the coefs all together
        allVals = np.matmul(self.basis, self._mmap_with_coefs[i]["coefs"])
        return allVals

    def recordXY(self, i):
        return self.sampleTimes(i), self.modeledPulse(i)

    @property
    def _mmap_with_coefs(self):
        return self._mmap.view(self._dtype_non_descriptive)


if __name__ == "__main__":
    off = OffFile("off_test.off")
    assert off.nRecords == 1
    print(off)
    x = off.sampleTimes(0)
    y = off.modeledPulse(0)
    assert len(x) == len(y)
