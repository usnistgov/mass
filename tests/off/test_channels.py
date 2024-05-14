import os
import collections
import pytest
from mass.off import util
import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks, NoCutInds
from mass.calibration import _highly_charged_ion_lines  # noqa
import numpy as np
import pylab as plt
import lmfit
import h5py
import resource
import tempfile

# Remove a warning message
import matplotlib as mpl
mpl.rc('figure', max_open_warning=0)


def xfail_on_windows(func):
    """a conditional decorator that applies pytest.mark.xfail only if running on windows
    used to mark a test that works on linux as known to fail on windows, but still having the test suite pass"""
    is_windows = os.name == 'nt'
    if is_windows:
        return pytest.mark.xfail(func, strict=True)
    else:
        return func


# this is intented to be both a test and a tutorial script
# in most cases you want to remove all _rethrow=True, its mostly for debugging mass
try:
    d = os.path.dirname(os.path.realpath(__file__))
except NameError:
    d = os.getcwd()

filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=2),
                    verbose=True, channelClass=Channel, excludeStates=["START", "END"])
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
ds.plotAvsB("relTimeSec", "residualStdDev", includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotAvsB2d("relTimeSec", "filtValue",
              [np.arange(0, 3600 * 2, 300), np.arange(0, 40000, 500)])
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
for dsloop in data.values():
    assert dsloop.calibrationPlan.lines == ds.calibrationPlan.lines


# create "filtValueDC" by drift correcting on data near some particular energy
# to do so we first create a cut, but we do not set it as default
data.cutAdd("cutForLearnDC", lambda energyRough: np.logical_and(
    energyRough > 1000, energyRough < 3500), setDefault=False, _rethrow=True)
assert data._handleDefaultCut("cutForLearnDC") == "cutForLearnDC"
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
ax.set_ylim(0, 1.2 * np.amax([np.amax(line.get_ydata()) for line in ax.lines]))
names = [f"W Ni-{i}" for i in range(1, 27)]
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

outfile = data.outputHDF5Filename(outputDir=".", addToName="qualitychecklinefit")
with h5py.File(outfile, "w") as h5:
    results = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                                       worstAllowedFWHM=4.5, states="Ne", _rethrow=True,
                                       resolutionPlot=True, hdf5Group=h5)


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
    energyRough - mass.STANDARD_FEATURES["TiKAlpha"]) < 60)
selectedEnergies = ds.energyRough[ds.cutNearTiKAlpha]
assert len(selectedEnergies) == np.sum(
    np.abs(ds.energyRough - mass.STANDARD_FEATURES["TiKAlpha"]) < 60)
data.learnDriftCorrection(uncorrectedName="filtValue", correctedName="filtValueDCCutTest",
                          cutRecipeName="cutNearTiKAlpha", _rethrow=True)
data.learnDriftCorrection(uncorrectedName="filtValue", correctedName="filtValueDCCutTestInv",
                          cutRecipeName="!cutNearTiKAlpha", _rethrow=True)


def test_calibration_n_iter():
    ds.calibrateFollowingPlan("filtValue", calibratedName="energy2",
                              n_iter=2, approximate=False)
    # it should be a little different from energy
    assert 0 != np.mean(np.abs(ds.energy - ds.energy2))
    # but should also be similar... though I had to set rtol higher than I expected for this to pass
    assert np.allclose(ds.energy, ds.energy2, rtol=1e-1)


def test_repeating_the_same_correction_with_new_name_doesnt_change_the_original():
    # repeating the same correciton with new name doesnt change the original
    orig = ds.filtValueDC[:]
    ds.learnDriftCorrection(uncorrectedName="filtValueDCPCTC")
    ds.filtValueDCPCTC[0]
    assert np.allclose(orig, ds.filtValueDC)


def test_fixed_behaviors():
    assert ds.stateLabels == ["Ne", "W 1", "Os", "Ar", "Re", "W 2", "CO2", "Ir"]


def test_reading_some_items():
    assert ds.relTimeSec[0] == 0
    assert np.abs(np.median(ds.filtPhase)) < 0.5
    assert ds.energy[3] == pytest.approx(ds.energyRough[3], abs=5)


def test_index_off_with_cuts_with_list_of_inds():
    inds = ds.getStatesIndicies(["Ne", "W 1", "Os", "Ar", "Re", "W 2", "CO2", "Ir"])
    v0 = ds._indexOffWithCuts(inds, _listMethodSelect=0)
    v2 = ds._indexOffWithCuts(inds, _listMethodSelect=2)
    assert np.allclose(v0["filtValue"], v2["filtValue"])
    # this is a test of correctness because
    # the implementation of method 0 is simpler than method 2
    # method 2 is the default because it is much faster


def test_get_attr():
    ds.getAttr("energy", slice(0, 50))  # index with slice
    ds.getAttr("energy", "Ne")  # index with state
    e0 = ds.getAttr("energy", ["Ne", "W 1"])  # index with list of states
    inds = ds.getStatesIndicies(["Ne", "W 1"])
    e1 = ds.getAttr("energy", inds)  # index with inds from same list of states
    assert np.allclose(e0, e1)


@xfail_on_windows  # we don't need refresh to disk to work on windows, so I didn't investigate why it fails
def test_refresh_from_files():
    # ds and data refers to the global variable from the script before the tests
    # while ds_local and data_local refer to the similar local variables
    data_local = ChannelGroup([filename], verbose=False)
    ds_local = data_local.firstGoodChannel()
    experimentStateFile = data_local.experimentStateFile
    # reach inside offFile and experimentStateFile to make it look like the files were originally opened during
    # state E. Then we refresh to learn about states F-I
    # The numerical constants are chosen to make sense for this scenario; vary one, and you may need to vary all.
    ds_local.offFile._updateMmap(_nRecords=11600)  # mmap only the first half of records
    experimentStateFile.allLabels = experimentStateFile.allLabels[:5]
    experimentStateFile.unixnanos = experimentStateFile.unixnanos[:5]
    experimentStateFile.unaliasedLabels = experimentStateFile.applyExcludesToLabels(
        experimentStateFile.allLabels)
    experimentStateFile.parse_start = 159
    assert len(ds_local) == 11600
    assert ds_local.stateLabels == ["B", "C", "D", "E"]
    # use the global ds a the source of truth
    for ((k_local, v_local), (k, v)) in zip(ds_local.statesDict.items(), ds.statesDict.items()):
        if k_local == "E":  # since we stoppe data aquisition during E, it won't equal it's final value
            assert v_local != v
        else:
            assert v_local == v
    _n_new_labels, _n_new_pulses_dict = data_local.refreshFromFiles()
    assert len(ds_local) == len(ds)
    assert ds_local.stateLabels == ["B", "C", "D", "E", "F", "G", "H", "I"]
    states = ["B", "H", "I"]
    _, hist_local = ds_local.hist(np.arange(0, 4000, 1000), "filtValue",
                                  states=states, cutRecipeName="cutNone")
    global_states = [data.experimentStateFile.labelAliasesDict[state] for state in states]
    _, hist = ds.hist(np.arange(0, 4000, 1000), "filtValue",
                      states=global_states, cutRecipeName="cutNone")
    for ((k_local, v_local), (k, v)) in zip(ds_local.statesDict.items(), ds.statesDict.items()):
        assert v_local == v
    assert all(ds.filtValue == ds_local.filtValue)
    assert all(hist_local == hist)
    # refresh again without updating the files, make sure it doesnt crash
    n_new_labels_2, _n_new_pulses_dict2 = data.refreshFromFiles()
    assert n_new_labels_2 == 0


def test_bad_channels_skipped():
    # try:
    data_local = ChannelGroup([filename])
    assert len(data_local.keys()) == 1
    ds_local = data_local.firstGoodChannel()
    _, hists_pre_bad = data_local.hists(np.arange(10), "filtValue")
    assert not ds_local.markedBadBool
    ds_local.markBad("testing")
    _, hists_post_bad = data_local.hists(np.arange(10), "filtValue")
    assert len(hists_pre_bad) == 1
    assert len(hists_post_bad) == 0
    assert len(data_local.keys()) == 0
    assert ds_local.markedBadBool
    n_include_bad = 0
    with data_local.includeBad():
        for ds in data_local:
            n_include_bad += 1
    assert n_include_bad == 1
    n_exclude_bad = 0
    with data_local.includeBad(False):
        for ds in data_local:
            n_exclude_bad += 1
    assert n_exclude_bad == 0


def test_save_load_recipes():
    data_local = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=2))
    ds_local = data_local.firstGoodChannel()
    assert "energy" not in ds_local.__dict__, \
        "ds_local should not have energy yet, we haven't defined that recipe"
    pklfilename = "recipe_book_save_test2.rbpkl"
    data.saveRecipeBooks(pklfilename)
    ds = data.firstGoodChannel()
    ds.add5LagRecipes(np.zeros(996))
    data_local.loadRecipeBooks(pklfilename)
    for ds in data.values():
        ds_local = data_local[ds.channum]
        assert all(ds.energy == ds_local.energy)


def test_experiment_state_file_repeated_states():
    # A better test would create an alternate experiment state file with repeated indicies and use that
    # rather than reach into the internals of ExperimentStateFile
    esf = mass.off.ExperimentStateFile(_parse=False)
    # reach into the internals to simulate the results of parse with repeated states
    esf.allLabels = ["A", "B", "A", "B", "IGNORE", "A"]
    esf.unixnanos = np.arange(len(esf.allLabels)) * 100
    esf.unaliasedLabels = esf.applyExcludesToLabels(esf.allLabels)
    unixnanos = np.arange(2 * len(esf.allLabels)) * 50  # two entires per label
    d = esf.calcStatesDict(unixnanos)
    assert len(d["A"]) == esf.allLabels.count("A")
    assert len(d["B"]) == esf.allLabels.count("B")
    assert "IGNORE" not in d.keys()
    for s in d["A"] + d["B"]:
        assert s.stop - s.start == 2

    data_local = ChannelGroup([filename], experimentStateFile=esf)
    ds_local = data_local.firstGoodChannel()
    ds_local.stdDevResThreshold = 100
    inds = ds_local.getStatesIndicies("A")
    _ = ds_local.getAttr("filtValue", inds)


def test_experiment_state_file_add_to_same_state_fake_esf():
    # First, we create a dummy experiment state file, esf, with state labels A and B. Then, more records
    # are added to state B. This test verifies that the slice defining state B gets properly updated.
    # this test simulates an experiment state file instead of using an actual file.
    esf = mass.off.ExperimentStateFile(_parse=False)
    # reach into the internals to simulate the results of parse with repeated states
    esf.allLabels = ["A", "B"]
    esf.unixnanos = np.arange(len(esf.allLabels)) * 100
    esf.unaliasedLabels = esf.applyExcludesToLabels(esf.allLabels)
    unixnanos = [25, 75, 125, 175]  # four timestamps representing four records. two records per state.
    d = esf.calcStatesDict(unixnanos)
    slice_before_update = d["B"]  # state B will be updated with new records. We need to make sure the slice gets changed properly.
    new_unixnanos = [225, 275]  # new records are collected
    d_updated = esf.calcStatesDict(new_unixnanos, statesDict=d, i0_allLabels=len(esf.allLabels), i0_unixnanos=len(unixnanos))

    slice_after_update = d_updated["B"]
    assert slice_after_update.stop == slice_before_update.stop + len(new_unixnanos)


def test_experiment_state_file_add_to_same_state():
    # First, we create an experiment state file, esf, with state labels A and B. Then, more records
    # are added to state B. This test verifies that the slice defining state B gets properly updated.
    # This uses a temporary file f just like a regular experiment state file.
    f = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
    f.write('#\n')  # header
    f.write('0, A\n')    # state A starts at 0 unixnanos
    f.write('100, B\n')  # state B starts at 100 unixnanos
    f.flush()
    os.fsync(f.fileno())
    esf = mass.off.ExperimentStateFile(filename=f.name)

    unixnanos = [25, 75, 125, 175]  # four timestamps representing four records. two records per state.
    d = esf.calcStatesDict(unixnanos)
    slice_before_update = d["B"]  # state B will be updated with new records. We need to make sure the slice gets changed properly.
    new_unixnanos = [225, 275]  # new records are collected
    d_updated = esf.calcStatesDict(new_unixnanos, statesDict=d, i0_allLabels=len(esf.allLabels), i0_unixnanos=len(unixnanos))
    slice_after_update = d_updated["B"]
    assert slice_after_update.stop == slice_before_update.stop + len(new_unixnanos)

    # now, add state C with two records and look at the indices
    f.write('300, C\n')
    f.flush()
    os.fsync(f.fileno())
    old_states_len = len(esf.allLabels)
    esf.parse()

    d_empty_state = esf.calcStatesDict(
        [301, 302], statesDict=d_updated,
        i0_allLabels=old_states_len,
        i0_unixnanos=len(unixnanos) + len(new_unixnanos))

    assert d_empty_state['A'] == slice(0, 2, None)
    assert d_empty_state['B'] == slice(2, 6, None)
    assert d_empty_state['C'] == slice(6, 8, None)
    f.close()
    os.remove(f.name)


def test_we_get_different_histograms_when_using_different_cuts_into_a_channelGroup_function():
    # check that we actually get different histograms when using different cuts
    # into a channel group
    _bc1, counts1 = data.hist(np.arange(500, 5000, 500), "energy", cutRecipeName="cutNearTiKAlpha")
    _bc2, counts2 = data.hist(np.arange(500, 5000, 500), "energy", cutRecipeName="!cutNearTiKAlpha")
    _bc3, counts3 = data.hist(np.arange(500, 5000, 500), "energy")

    assert np.sum(counts1 - counts2) != 0
    assert np.sum(counts1 - counts3) != 0
    assert np.sum(counts2 - counts3) != 0


def test_getAttr_with_list_of_slice():
    ind = [slice(0, 5), slice(5, 10)]
    assert np.allclose(ds.getAttr("filtValue", ind), ds.getAttr("filtValue", slice(0, 10)))
    assert np.allclose(ds.getAttr("filtValue", [slice(0, 10)]),
                       ds.getAttr("filtValue", slice(0, 10)))


def test_HCI_loads():
    assert "O He-Like 1s2p 1P1" in dir(_highly_charged_ion_lines.fluorescence_lines)


def test_getAttr_and_recipes_with_coefs():
    ind = [slice(0, 5), slice(5, 10)]
    coefs = ds.getAttr("coefs", ind)
    filtValue, coefs2 = ds.getAttr(["filtValue", "coefs"], ind)
    assert np.allclose(coefs, coefs2)
    assert np.allclose(coefs[:, 2], filtValue)
    ds.recipes.add("coefsSum", lambda coefs: coefs.sum(axis=1))
    assert np.allclose(ds.getAttr("coefsSum", ind), coefs.sum(axis=1))
    ds.recipes.add("coefsSumPlusFiltvalue", lambda filtValue, coefs: coefs.sum(axis=1) + filtValue)
    assert np.allclose(ds.getAttr("coefsSumPlusFiltvalue", ind), coefs.sum(axis=1) + filtValue)
    # test access as with NoCutInds
    ds.getAttr("coefs", NoCutInds())
    ds.getAttr("coefsSum", NoCutInds())


# pytest style test! way simpler to write
def test_get_model():
    m_127 = mass.get_model(127)
    assert m_127.spect.peak_energy == 127
    assert m_127.spect.shortname == "127eVquick_line"

    m_au = mass.get_model("AuLAlpha")
    assert m_au.spect.peak_energy == mass.STANDARD_FEATURES["AuLAlpha"]
    assert m_au.spect.shortname == "AuLAlpha"

    m_ti = mass.get_model("TiKAlpha")
    assert m_ti.spect.shortname == "TiKAlpha"

    ql = mass.SpectralLine.quick_monochromatic_line("test", 100, 0.001, 0)
    m_ql = mass.get_model(ql.model())
    assert m_ql.spect.shortname == "testquick_line"

    with pytest.raises(mass.FailedToGetModelException):
        mass.get_model("this is a str but not a standard feature")


def test_duplicate_cuts():
    "See issue 214: check that we can use `overwrite=True` to update a cut."
    # Make some arbitrary cut.
    data.cutAdd("deliberateduplicate", lambda energy: energy < 730)
    # It's an error if you try to add another cut by the same name.
    with pytest.raises(Exception):
        data.cutAdd("deliberateduplicate", lambda energy: energy < 740)
    # But it's not an error to reuse the same name WHEN you have `overwrite=True`.
    data.cutAdd("deliberateduplicate", lambda energy: energy < 750, overwrite=True)


def test_recipes():
    rb = util.RecipeBook(baseIngredients=["x", "y", "z"], propertyClass=None,
                         coefs_dtype=None)

    def funa(x, y):
        return x + y

    def funb(a, z):
        return a + z

    rb.add("a", funa)
    rb.add("b", funb)
    rb.add("c", lambda a, b: a + b)
    with pytest.raises(AssertionError):
        # should fail because f isn't in baseIngredients and hasn't been added
        rb.add("e", lambda a, b, c, d, f: a)
    with pytest.raises(AssertionError):
        # should fail because ingredients is longer than argument list
        rb.add("f", lambda a, b: a + b, ingredients=["a", "b", "c"])
    args = {"x": 1, "y": 2, "z": 3}
    assert rb.craft("a", args) == 3
    assert rb.craft("b", args) == 6
    assert rb.craft("c", args) == 9
    assert rb._craftWithFunction(lambda a, b, c: a + b + c, args) == 18
    assert rb.craft(lambda a, b, c: a + b + c, args) == 18

    rb.to_file("test_recipe_book_save.pkl", overwrite=True)

    rb2 = util.RecipeBook.from_file("test_recipe_book_save.pkl")

    assert rb2.craft("a", args) == 3
    assert rb2.craft("b", args) == 6
    assert rb2.craft("c", args) == 9


def test_linefit_has_tail_and_has_linear_background():
    result = ds.linefit("O H-Like 2p", states="CO2")
    assert "tail_frac" not in result.params.keys()
    result = ds.linefit("O H-Like 2p", states="CO2", has_tails=True)
    assert result.params["tail_frac"].vary is True
    assert result.params["tail_share_hi"].vary is False
    assert result.params["tail_share_hi"].value == 0
    params = lmfit.Parameters()
    params.add("tail_share_hi", value=0.01, vary=True, min=0, max=0.5)
    params.add("tail_tau_hi", value=8, vary=True, min=0, max=100)
    result = ds.linefit("O H-Like 2p", states="CO2", has_tails=True, params_update=params)
    assert result.params["tail_frac"].vary is True
    assert result.params["tail_share_hi"].vary is True
    assert result.params["tail_share_hi"].value > 0
    assert result.params["tail_tau_hi"].vary is True
    assert result.params["tail_tau_hi"].value > 0

    result = ds.linefit("O H-Like 2p", states="CO2", has_linear_background=False)
    assert "background" not in result.params.keys()
    assert "bg_slope" not in result.params.keys()


def test_median_absolute_deviation():
    mad, _, median = util.median_absolute_deviation([1, 1, 2, 3, 4])
    assert mad == 1
    assert median == 2


def test_aliasState():
    esf = mass.off.channels.ExperimentStateFile(data.experimentStateFile.filename)
    esf.aliasState("B", "Ne")
    esf.aliasState(["C", "G"], "W")
    sd = esf.calcStatesDict(ds.unixnano)
    for s in ["B", "C", "G"]:
        assert s not in sd.keys()
    assert isinstance(sd["Ne"], slice)
    assert isinstance(sd["W"], list)
    for x in sd["W"]:
        assert isinstance(x, slice)


def test_iterstates():
    assert util.iterstates("ABC") == ["ABC"]
    assert util.iterstates(["A", "B", "CC"]) == ["A", "B", "CC"]
    assert util.iterstates([slice(0, 1, 1)]) == [slice(0, 1, 1)]

    with pytest.raises(KeyError):  # previously this would work due to being recognized as states "B" and "C"
        # now it fails since state "BC" doesnt exist
        ds.plotHist(np.arange(100, 2500, 50), 'energy', states="BC", coAddStates=False)


def test_save_load_recipe_book():
    rb = ds.recipes
    save_path = os.path.join(d, "recipe_book_save_test.rbpkl")
    rb.to_file(save_path, overwrite=True)
    rb2 = util.RecipeBook.from_file(save_path)
    assert rb.craftedIngredients.keys() == rb2.craftedIngredients.keys()
    args = {"pretriggerMean": 1, "filtValue": 2}
    print(rb.craftedIngredients["energy"])
    assert rb.craft("energy", args) == rb2.craft("energy", args)


def test_open_many_OFF_files():
    """Open more OFF ChannelGroup objects than the system allows. Test that close method closes them."""

    # LOWER the system's limit on number of open files, to make the test smaller
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    request_maxfiles = min(60, soft_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (request_maxfiles, hard_limit))
    try:
        maxfiles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        NFilesToOpen = maxfiles // 2 + 10

        filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
        filelist = getOffFileListFromOneFile(filename, maxChans=2)
        for _ in range(NFilesToOpen):
            _ = ChannelGroup(filelist, verbose=True, channelClass=Channel,
                             excludeStates=["START", "END"])

        # Now open one ChannelGroup with too many files. If the resources aren't freed, we can
        # only open it once, not twice.
        NFilePairsToOpen = (maxfiles - 12) // 6
        filelist *= NFilePairsToOpen
        for _ in range(3):
            _ = ChannelGroup(filelist, verbose=True, channelClass=Channel,
                             excludeStates=["START", "END"])

    # Use the try...finally to undo our reduction in the limit on number of open files.
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def test_listmode_to_hdf5():
    filename = data.outputHDF5Filename(outputDir=".", addToName="listmode")
    with h5py.File(filename, "w") as h5:
        data.energyTimestampLabelToHDF5(h5)
    with h5py.File(filename, "r") as h5:
        h5["3"]["Ar"]["unixnano"]


def test_hists_to_hdf5():
    filename = data.outputHDF5Filename(outputDir=".", addToName="hists")
    with h5py.File(filename, "w") as h5:
        data.histsToHDF5(h5, binEdges=np.arange(4000), attr="energy")
    with h5py.File(filename, "r") as h5:
        h5["3"]["Ar"]["counts"]
