import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
import h5py
import os
import numpy as np
import pylab as plt
import collections

import unittest as ut

# this is intented to be both a test and a tutorial script
d = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=2),
                    verbose=False, channelClass=Channel, excludeStates=["START","END"])
data.setOutputDir(baseDir=d, deleteAndRecreate=True)
data.experimentStateFile.aliasState("B", "Ne")
data.experimentStateFile.aliasState("C", "W 1")
data.experimentStateFile.aliasState("D", "Os")
data.experimentStateFile.aliasState("E", "Ar")
data.experimentStateFile.aliasState("F", "Re")
data.experimentStateFile.aliasState("G", "W 2")
data.experimentStateFile.aliasState("H", "CO2")
data.experimentStateFile.aliasState("I", "Ir")
data.learnStdDevResThresholdUsingRatioToNoiseStd(ratioToNoiseStd=5, _rethrow=True)
data.learnDriftCorrection(_rethrow=True)
ds = data.firstGoodChannel()
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0, 40000, 4), "filtValue")
ds.plotHist(np.arange(0, 40000, 4), "filtValue", coAddStates=False)
ds.plotResidualStdDev()
driftCorrectInfo = ds.learnDriftCorrection(states=["W 1", "W 2"])
ds.plotCompareDriftCorrect()

ds.calibrationPlanInit("filtValueDC")
ds.calibrationPlanAddPoint(2128, "O He-Like 1s2s+1s2p", states="CO2")
ds.calibrationPlanAddPoint(2421, "O H-Like 2p", states="CO2")
ds.calibrationPlanAddPoint(2864, "O H-Like 3p", states="CO2")
ds.calibrationPlanAddPoint(3404, "Ne He-Like 1s2s+1s2p", states="Ne")
ds.calibrationPlanAddPoint(3768, "Ne H-Like 2p", states="Ne")
ds.calibrationPlanAddPoint(5716, "W Ni-2", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(6413, "W Ni-4", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(7641, "W Ni-7", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(10256, "W Ni-17", states=["W 1", "W 2"])
# ds.calibrationPlanAddPoint(10700, "W Ni-20", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(11125, "Ar He-Like 1s2s+1s2p", states="Ar")
ds.calibrationPlanAddPoint(11728, "Ar H-Like 2p", states="Ar")
# at this point energyRough should work
ds.plotHist(np.arange(0, 4000, 1), "energyRough", coAddStates=False)
fitters = ds.calibrateFollowingPlan("filtValueDC", approximate=False)
ds.linefit("Ne H-Like 2p", attr="energy", states="Ne")
ds.linefit("Ne He-Like 1s2s+1s2p", attr="energy", states="Ne")
ds.linefit("W Ni-7", attr="energy", states=["W 1", "W 2"])
ds.plotHist(np.arange(0, 4000, 4), "energy", coAddStates=False)


ds.diagnoseCalibration()
ds.learnPhaseCorrection("Ne", "derivativeLike", "filtValue", [3768])

ds3 = data[3]
data.alignToReferenceChannel(referenceChannel=ds,
                             binEdges=np.arange(500, 20000, 4), attr="filtValueDC", _rethrow=True)
aligner = ds3.aligner
aligner.samePeaksPlot()
aligner.samePeaksPlotWithAlignmentCal()

fitters = data.calibrateFollowingPlan(
    "filtValueDC", _rethrow=False, dlo=10, dhi=10, approximate=False)
data.qualityCheckDropOneErrors(thresholdAbsolute=2.5, thresholdSigmaFromMedianAbsoluteValue=6)


data.hist(np.arange(0, 4000, 1), "energy")
data.plotHist(np.arange(0, 4000, 1), "energy", coAddStates=False)
data.plotHists(np.arange(0, 16000, 4), "arbsInRefChannelUnits")
data.plotHists(np.arange(0, 4000, 1), "energy")

# histograms with peak labels
plt.figure(figsize=(12, 6))
ax = plt.gca()
data.plotHist(np.arange(1000, 4000, 1), "energy", coAddStates=False, states=["W 1", "Os"], axis=ax)
ax.set_ylim(0, 1.2*np.amax([np.amax(l.get_ydata()) for l in ax.lines]))
names = ["W Ni-{}".format(i) for i in range(1, 27)]
n = collections.OrderedDict()
# line = ax.lines[0]
for name in names:
    n[name] = mass.spectrum_classes[name].nominal_peak_energy
labelPeak(ax, "W Ni-8", n["W Ni-8"])
labelPeaks(axis=ax, names=n.keys(), energies=n.values(), line=ax.lines[0])
nos = collections.OrderedDict()
nos["Os Ni-2"] = 1680
nos["Os Ni-3"] = 1755
nos["Os Ni-4"] = 1902
nos["Os Ni-5"] = 1975
nos["Os Ni-6"] = 2155
nos["Os Ni-7"] = 2268
nos["Os Ni-8"] = 2342
nos["Os Ni-16"] = 3032
nos["Os Ni-17"] = 3102
labelPeaks(ax, names=nos.keys(), energies=nos.values(), line=ax.lines[1])

data.fitterPlot("W Ni-20", states=["W 1"])

print(data.outputHDF5)
print(os.path.abspath(data.outputDir))


with data.outputHDF5 as h5:
    fitters = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                                       worstAllowedFWHM=4.5, states="Ne", _rethrow=False,
                                       resolutionPlot=True, hdf5Group=h5)
    data.histsToHDF5(h5, np.arange(4000))
    data.recipeToHDF5(h5)
    data.energyTimestampLabelToHDF5(h5)

with h5py.File(data.outputHDF5.filename, "r") as h5:
    print(h5.keys())
    newds = Channel(ds.offFile, ds.experimentStateFile)
    newds.recipeFromHDF5(h5)


class TestSummaries(ut.TestCase):
    def test_recipeFromHDF5(self):
        self.assertTrue(newds.driftCorrection == ds.driftCorrection)

    def test_fixedBehaviors(self):
        self.assertEqual(ds.stateLabels, ["Ne", "W 1","Os", "Ar", "Re", "W 2", "CO2", "Ir"])

    def test_reading_some_items(self):
        self.assertEquals(ds.relTimeSec[0],0)
        self.assertLess(np.abs(np.median(ds.filtPhase)),0.5)
        self.assertAlmostEqual(ds.energy[3], ds.energyRough[3], delta=5)

    def test_getOffAttr_with_list_of_inds(self):
        inds = ds.getStatesIndicies(["Ne", "W 1","Os", "Ar", "Re", "W 2", "CO2", "Ir"])
        v0 = ds.getOffAttr("filtValue", inds, _listMethodSelect=0)
        v1 = ds.getOffAttr("filtValue", inds, _listMethodSelect=1)
        v2 = ds.getOffAttr("filtValue", inds, _listMethodSelect=2)
        self.assertTrue(np.allclose(v0,v2))
        self.assertTrue(np.allclose(v1,v2))
        # this is a test of correctness because
        # the implementation of method 0 is simpler than method 2 
        # method2 is the default becaue it is much faster


if __name__ == '__main__':
    ut.main()
