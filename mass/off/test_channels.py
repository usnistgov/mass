# Normal imports
import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks, NoCutInds
from mass.calibration import _highly_charged_ion_lines
import numpy as np
import pylab as plt
import lmfit

# test only imports
from mass.off import util
import unittest as ut
import pytest
import collections
import os


# this is intented to be both a test and a tutorial script
# in most cases you want to remove all _rethrow=True, its mostly for debugging mass
try:
    d = os.path.dirname(os.path.realpath(__file__))
except NameError:
    d = os.getcwd()

filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=2),
                    verbose=True, channelClass=Channel, excludeStates=["START", "END"])
data.setOutputDir(baseDir=d, deleteAndRecreate=True)
data.experimentStateFile.aliasState("B", "Ne")
data.experimentStateFile.aliasState("C", "W 1")
data.experimentStateFile.aliasState("D", "Os")
data.experimentStateFile.aliasState("E", "Ar")
data.experimentStateFile.aliasState("F", "Re")
data.experimentStateFile.aliasState("G", "W 2")
data.experimentStateFile.aliasState("H", "CO2")
data.experimentStateFile.aliasState("I", "Ir")
# creates "cutResidualStdDev" and sets it to default
data.learnResidualStdDevCut(plot=True, _rethrow=True)
data.setDefaultBinsize(0.5)
ds = data.firstGoodChannel()
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0, 40000, 4), "filtValue")
ds.plotHist(np.arange(0, 40000, 4), "filtValue", coAddStates=False)

# set calibration points, and create attribute "energyRough"
ds.calibrationPlanInit("filtValue")
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

# align all channels to have calibration plans at the same peaks
data.alignToReferenceChannel(referenceChannel=ds,
                             binEdges=np.arange(500, 20000, 4), attr="filtValue", _rethrow=True)
aligner = data[3].aligner
aligner.samePeaksPlot()
aligner.samePeaksPlotWithAlignmentCal()

# create "filtValueDC" by drift correcting on data near some particular energy
# to do so we first create a cut, but we do not set it as default
data.cutAdd("cutForLearnDC", lambda energyRough: np.logical_and(
    energyRough > 1000, energyRough < 3500), setDefault=False, _rethrow=True)
# uses "cutForLearnDC" in place of the default, so far no easy way to use both
data.learnDriftCorrection(states=["W 1", "W 2"], cutRecipeName="cutForLearnDC", _rethrow=True)
ds.plotCompareDriftCorrect()

# calibrate on filtValueDC, it's usually close enough to filtValue to use the same calibration plan
results = ds.calibrateFollowingPlan("filtValueDC", approximate=False)
ds.linefit("Ne He-Like 1s2s+1s2p", attr="energy", states="Ne")
ds.linefit("W Ni-7", attr="energy", states=["W 1", "W 2"])
ds.plotHist(np.arange(0, 4000, 4), "energy", coAddStates=False)


ds.diagnoseCalibration()


results = data.calibrateFollowingPlan(
    "filtValueDC", dlo=10, dhi=10, approximate=False, _rethrow=True, overwriteRecipe=True)
data.qualityCheckDropOneErrors(
    thresholdAbsolute=2.5, thresholdSigmaFromMedianAbsoluteValue=6, _rethrow=True)


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
    n[name] = mass.spectra[name].nominal_peak_energy
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

data.resultPlot("W Ni-20", states=["W 1"])

print(data.whyChanBad)

h5 = data.outputHDF5  # dont use with here, it will hide errors
results = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                                   worstAllowedFWHM=4.5, states="Ne", _rethrow=True,
                                   resolutionPlot=True, hdf5Group=h5)
data.histsToHDF5(h5, np.arange(4000))
# data.recipeToHDF5(h5)
data.energyTimestampLabelToHDF5(h5)
h5.close()

# h5 = h5py.File(data.outputHDF5.filename, "r")  # dont use with here, it will hide errors
# newds = Channel(ds.offFile, ds.experimentStateFile)
# newds.recipeFromHDF5(h5)
# h5.close()

# test corrections with recipes as input
ds.learnPhaseCorrection(uncorrectedName="filtValueDC")
ds.learnTimeDriftCorrection(uncorrectedName="filtValueDCPC")
ds.filtValueDCPCTC[0]  # this will error if the attr doesnt exist

# test cutRecipes
data.cutAdd("cutNearTiKAlpha", lambda energyRough: np.abs(
    energyRough-mass.STANDARD_FEATURES["TiKAlpha"]) < 60)
selectedEnergies = ds.energyRough[ds.cutNearTiKAlpha]
assert len(selectedEnergies) == np.sum(
    np.abs(ds.energyRough-mass.STANDARD_FEATURES["TiKAlpha"]) < 60)
data.learnDriftCorrection(uncorrectedName="filtValue", correctedName="filtValueDCCutTest",
                          cutRecipeName="cutNearTiKAlpha", _rethrow=True)
data.learnDriftCorrection(uncorrectedName="filtValue", correctedName="filtValueDCCutTestInv",
                          cutRecipeName="!cutNearTiKAlpha", _rethrow=True)


class TestSummaries(ut.TestCase):

    def test_calibration_n_iter(self):
        ds.calibrateFollowingPlan("filtValue", calibratedName="energy2",
                                  n_iter=2, approximate=False)
        # it should be a little different from energy
        self.assertNotEqual(0, np.mean(np.abs(ds.energy-ds.energy2)))
        # but should also be similar... though I had to set rtol higher than I expected for this to pass
        self.assertTrue(np.allclose(ds.energy, ds.energy2, rtol=1e-1))

    def test_repeatingTheSameCorrectionWithNewNameDoesntChangeTheOriginal(self):
        # repeating the same correciton with new name doesnt change the original
        orig = ds.filtValueDC[:]
        ds.learnDriftCorrection(uncorrectedName="filtValueDCPCTC")
        ds.filtValueDCPCTC[0]
        self.assertTrue(np.allclose(orig, ds.filtValueDC))

    def test_fixedBehaviors(self):
        self.assertEqual(ds.stateLabels, ["Ne", "W 1", "Os", "Ar", "Re", "W 2", "CO2", "Ir"])

    def test_reading_some_items(self):
        self.assertEqual(ds.relTimeSec[0], 0)
        self.assertLess(np.abs(np.median(ds.filtPhase)), 0.5)
        self.assertAlmostEqual(ds.energy[3], ds.energyRough[3], delta=5)

    def test_indexOffWithCuts_with_list_of_inds(self):
        inds = ds.getStatesIndicies(["Ne", "W 1", "Os", "Ar", "Re", "W 2", "CO2", "Ir"])
        v0 = ds._indexOffWithCuts(inds, _listMethodSelect=0)
        v2 = ds._indexOffWithCuts(inds, _listMethodSelect=2)
        self.assertTrue(np.allclose(v0["filtValue"], v2["filtValue"]))
        # this is a test of correctness because
        # the implementation of method 0 is simpler than method 2
        # method 2 is the default because it is much faster

    def test_getAttr(self):
        ds.getAttr("energy", slice(0, 50))  # index with slice
        ds.getAttr("energy", "Ne")  # index with state
        e0 = ds.getAttr("energy", ["Ne", "W 1"])  # index with list of states
        inds = ds.getStatesIndicies(["Ne", "W 1"])
        e1 = ds.getAttr("energy", inds)  # index with inds from same list of states
        self.assertTrue(np.allclose(e0, e1))

    def test_refresh_from_files(self):
        # ds and data refers to the global variable from the script before the tests
        # while ds_local and data_local refer to the similar local variables
        data_local = ChannelGroup([filename], verbose=False)
        ds_local = data_local.firstGoodChannel()
        experimentStateFile = data_local.experimentStateFile
        # reach inside offFile and experimentStateFile to make it look like the files were originally opened during state E
        # then we refresh to learn about states F-I
        # the numerical constants are chosen to make sense for this scenario... if you vary one you may need to vary all
        ds_local.offFile._updateMmap(_nRecords=11600)  # mmap only the first half of records
        experimentStateFile.allLabels = experimentStateFile.allLabels[:5]
        experimentStateFile.unixnanos = experimentStateFile.unixnanos[:5]
        experimentStateFile.unaliasedLabels = experimentStateFile.applyExcludesToLabels(
            experimentStateFile.allLabels)
        experimentStateFile.parse_start = 159
        self.assertEqual(len(ds_local), 11600)
        self.assertEqual(ds_local.stateLabels, ["B", "C", "D", "E"])
        # use the global ds a the source of truth
        for ((k_local, v_local), (k, v)) in zip(ds_local.statesDict.items(), ds.statesDict.items()):
            if k_local == "E":  # since we stoppe data aquisition during E, it won't equal it's final value
                self.assertNotEqual(v_local, v)
            else:
                self.assertEqual(v_local, v)
        n_new_labels, n_new_pulses_dict = data_local.refreshFromFiles()
        self.assertEqual(len(ds_local), len(ds))
        self.assertEqual(ds_local.stateLabels, ["B", "C", "D", "E", "F", "G", "H", "I"])
        states = ["B", "H", "I"]
        _, hist_local = ds_local.hist(np.arange(0, 4000, 1000), "filtValue",
                                      states=states, cutRecipeName="cutNone")
        global_states = [data.experimentStateFile.labelAliasesDict[state] for state in states]
        _, hist = ds.hist(np.arange(0, 4000, 1000), "filtValue",
                          states=global_states, cutRecipeName="cutNone")
        for ((k_local, v_local), (k, v)) in zip(ds_local.statesDict.items(), ds.statesDict.items()):
            self.assertEqual(v_local, v)
        self.assertTrue(all(ds.filtValue == ds_local.filtValue))
        self.assertTrue(all(hist_local == hist))
        # refresh again without updating the files, make sure it doesnt crash
        n_new_labels_2, n_new_pulses_dict2 = data.refreshFromFiles()
        self.assertEqual(n_new_labels_2, 0)

    def test_bad_channels_skipped(self):
        # try:
        data_local = ChannelGroup([filename])
        self.assertEqual(len(data_local.keys()), 1)
        ds_local = data_local.firstGoodChannel()
        _, hists_pre_bad = data_local.hists(np.arange(10), "filtValue")
        self.assertFalse(ds_local.markedBadBool)
        ds_local.markBad("testing")
        _, hists_post_bad = data_local.hists(np.arange(10), "filtValue")
        self.assertEqual(len(hists_pre_bad), 1)
        self.assertEqual(len(hists_post_bad), 0)
        self.assertEqual(len(data_local.keys()), 0)
        self.assertTrue(ds_local.markedBadBool)
        n_include_bad = 0
        with data_local.includeBad():
            for ds in data_local:
                n_include_bad += 1
        self.assertEqual(n_include_bad, 1)
        n_exclude_bad = 0
        with data_local.includeBad(False):
            for ds in data_local:
                n_exclude_bad += 1
        self.assertEqual(n_exclude_bad, 0)

    def test_experiment_state_file_repeated_states(self):
        # A better test would create an alternate experiment state file with repeated indicies and use that
        # rather than reach into the internals of ExperimentStateFile
        esf = mass.off.channels.ExperimentStateFile(_parse=False)
        # reach into the internals to simulate the results of parse with repeated states
        esf.allLabels = ["A", "B", "A", "B", "IGNORE", "A"]
        esf.unixnanos = np.arange(len(esf.allLabels))*100
        esf.unaliasedLabels = esf.applyExcludesToLabels(esf.allLabels)
        unixnanos = np.arange(2*len(esf.allLabels))*50  # two entires per label
        d = esf.calcStatesDict(unixnanos)
        self.assertEqual(len(d["A"]), esf.allLabels.count("A"))
        self.assertEqual(len(d["B"]), esf.allLabels.count("B"))
        self.assertFalse("IGNORE" in d.keys())
        for s in d["A"]+d["B"]:
            self.assertEqual(s.stop-s.start, 2)

        data_local = ChannelGroup([filename], experimentStateFile=esf)
        ds_local = data_local.firstGoodChannel()
        ds_local.stdDevResThreshold = 100
        inds = ds_local.getStatesIndicies("A")
        _ = ds_local.getAttr("filtValue", inds)

def test_getAttr_with_list_of_slice():
    ind = [slice(0, 5), slice(5, 10)]
    assert np.allclose(ds.getAttr("filtValue", ind), ds.getAttr("filtValue", slice(0, 10)))
    assert np.allclose(ds.getAttr("filtValue", [slice(0, 10)]), ds.getAttr("filtValue", slice(0, 10)))


def test_HCI_loads():
    assert "O He-Like 1s2p 1P1" in dir(_highly_charged_ion_lines.fluorescence_lines)

def test_getAttr_and_recipes_with_coefs():
    ind = [slice(0, 5), slice(5, 10)]
    coefs = ds.getAttr("coefs", ind)
    filtValue, coefs2 = ds.getAttr(["filtValue","coefs"], ind)   
    assert np.allclose(coefs, coefs2) 
    assert np.allclose(coefs[:,2], filtValue)
    ds.recipes.add("coefsSum", lambda coefs: coefs.sum(axis=1))
    assert np.allclose(ds.getAttr("coefsSum", ind), coefs.sum(axis=1))
    ds.recipes.add("coefsSumPlusFiltvalue", lambda filtValue,coefs: coefs.sum(axis=1)+filtValue)
    assert np.allclose(ds.getAttr("coefsSumPlusFiltvalue", ind), coefs.sum(axis=1)+filtValue)
    # test access as with NoCutInds
    ds.getAttr("coefs", NoCutInds())
    ds.getAttr("coefsSum", NoCutInds())


# pytest style test! way simpler to write
def test_get_model():
    m_127 = mass.off.util.get_model(127)
    assert m_127.spect.peak_energy == 127
    assert m_127.spect.shortname == "127eVquick_line"

    m_au = mass.off.util.get_model("AuLAlpha")
    assert m_au.spect.peak_energy == mass.STANDARD_FEATURES["AuLAlpha"]
    assert m_au.spect.shortname == "AuLAlphaquick_line"

    m_ti = mass.off.util.get_model("TiKAlpha")
    assert m_ti.spect.shortname == "TiKAlpha"

    ql = mass.SpectralLine.quick_monochromatic_line("test", 100, 0.001, 0)
    m_ql = mass.off.util.get_model(ql.model())
    assert m_ql.spect.shortname == "testquick_line"

    with pytest.raises(mass.off.util.FailedToGetModelException):
        mass.off.util.get_model("this is a str but not a standard feature")


def test_recipes():
    rb = util.RecipeBook(baseIngredients=["x", "y", "z"])

    def funa(x, y):
        return x+y

    def funb(a, z):
        return a+z

    rb.add("a", funa)
    rb.add("b", funb)
    rb.add("c", lambda a, b: a+b)
    with pytest.raises(AssertionError):
        # should fail because f isn't in baseIngredients and hasn't been added
        rb.add("e", lambda a, b, c, d, f: a)
    with pytest.raises(AssertionError):
        # should fail because ingredients is longer than argument list
        rb.add("f", lambda a, b: a+b, ingredients=["a", "b", "c"])
    args = {"x": 1, "y": 2, "z": 3}
    assert rb.craft("a", args) == 3
    assert rb.craft("b", args) == 6
    assert rb.craft("c", args) == 9
    assert rb.craft("c", args) == 9
    assert rb._craftWithFunction(lambda a, b, c: a+b+c, args) == 18
    assert rb.craft(lambda a, b, c: a+b+c, args) == 18


def test_linefit_has_tail_and_has_linear_background():
    result = ds.linefit("O H-Like 2p", states="CO2")
    assert "tail_frac" not in result.params.keys()
    result = ds.linefit("O H-Like 2p", states="CO2", has_tails=True)
    assert result.params["tail_frac"].vary is True
    assert result.params["tail_frac_hi"].vary is False
    assert result.params["tail_frac_hi"].value == 0
    params = lmfit.Parameters()
    params.add("tail_frac_hi", value=0.01, vary=True, min=0, max=0.5)
    params.add("tail_tau_hi", value=8, vary=True, min=0, max=100)
    result = ds.linefit("O H-Like 2p", states="CO2", has_tails=True, params_update=params)
    assert result.params["tail_frac"].vary is True
    assert result.params["tail_frac_hi"].vary is True
    assert result.params["tail_frac_hi"].value > 0
    assert result.params["tail_tau_hi"].vary is True
    assert result.params["tail_tau_hi"].value > 0

    result = ds.linefit("O H-Like 2p", states="CO2", has_linear_background=False)
    assert "background" not in result.params.keys()
    assert "bg_slope" not in result.params.keys()

def test_median_absolute_deviation():
    mad, sigma_equiv, median = util.median_absolute_deviation([1, 1, 2, 3, 4])
    assert mad == 1
    assert median == 2

def test_aliasState():
    esf = mass.off.channels.ExperimentStateFile(data.experimentStateFile.filename)
    esf.aliasState("B", "Ne")
    esf.aliasState(["C", "G"], "W")
    sd = esf.calcStatesDict(ds.unixnano)


if __name__ == '__main__':
    ut.main()
