# std lib imports
import os
import shutil
import logging
import collections

# pkg imports
import numpy as np
import pylab as plt
import fastdtw
import h5py
import lmfit
import scipy.interpolate

# local imports
import mass
from .off import OffFile
from .util import GroupLooper, add_group_loop, labelPeak, labelPeaks, Recipe, RecipeBook
from .util import annotate_lines, SilenceBar, NoCutInds, InvalidStatesException
from . import util
from . import fivelag


LOG = logging.getLogger("mass")


class ExperimentStateFile():
    def __init__(self, filename=None, datasetFilename=None, excludeStates="auto", _parse=True):
        """
        excludeStates - when "auto" it either exclude no states when START is the only state or or excludes START, END and IGNORE
        _parse is only for testing
        otherwise pass a list of states to exclude
        """
        if filename is not None:
            self.filename = filename
        elif datasetFilename is not None:
            self.filename = self.experimentStateFilenameFromDatasetFilename(datasetFilename)
        else:
            self.filename = None
        self.excludeStates = excludeStates
        self.parse_start = 0
        self.allLabels = []
        self.unixnanos = np.zeros(0)
        if _parse:
            if self.filename is None:
                raise Exception("pass filename or datasetFilename or _parse=False")
            self.parse()
        self.labelAliasesDict = {}  # map unaliasedLabels to aliasedLabels
        self._preventAliasState = False  # causes aliasState to raise an Exception when it wouldn't work as expected

    def experimentStateFilenameFromDatasetFilename(self, datasetFilename):
        basename, channum = mass.ljh_util.ljh_basename_channum(datasetFilename)
        return basename+"_experiment_state.txt"

    def parse(self):
        with open(self.filename, "r") as f:
            # if we call parse a second time, we want to add states rather than reparse the whole file
            f.seek(self.parse_start)
            lines = f.readlines()
            parse_end = f.tell()
        if self.parse_start == 0:
            header_line = lines[0]
            if header_line[0] != "#":
                raise Exception("first line should start with #, was %s" % header_line)
            lines = lines[1:]
            if len(lines) == 0:
                raise Exception("zero lines after header in file")
        if len(lines) == 0:
            return  # no new states
        unixnanos = []
        labels = []
        for line in lines:
            a, b = line.split(",")
            a = a.strip()
            b = b.strip()
            unixnano = int(a)
            label = b
            unixnanos.append(unixnano)
            labels.append(label)
        self.allLabels += labels
        self.unixnanos = np.hstack([self.unixnanos, np.array(unixnanos)])
        self.unaliasedLabels = self.applyExcludesToLabels(self.allLabels)
        self.parse_start = parse_end  # next call to parse, start from here

    def calculateAutoExcludes(self):
        if len(self.allLabels) == 1:
            return []
        else:
            return ["START", "END", "STOP", "IGNORE"]

    def applyExcludesToLabels(self, allLabels):
        """
        possible recalculate self.excludeStates
        return a list of state labels that is unique, and contains all entries in allLabels except those in self.excludeStates
        order in the returned list is that of first appearance in allLables
        """
        if self.excludeStates == "auto":
            self.excludeStates = self.calculateAutoExcludes()
        r = []
        for label in allLabels:
            if label in self.excludeStates or label in r:
                continue
            r.append(label)
        return r

    def calcStatesDict(self, unixnanos, statesDict=None, i0_allLabels=0, i0_unixnanos=0):
        """
        calculate statesDict, a dictionary mapping state name to EITHER a slice OR a boolean array with length
        equal to unixnanos. Slices are used for unique states; boolean arrays are used for repeated states.
        When updating pass in the existing statesDict and i0 must be the first label in allLabels that wasn't
        used to calculate the existing statesDict.
        """
        if statesDict is None:
            statesDict = collections.OrderedDict()
        inds = np.searchsorted(unixnanos, self.unixnanos[i0_allLabels:])+i0_unixnanos
        # the state that was active last time calcStatesDict was called may need special handling
        if len(statesDict.keys()) > 0 and len(inds) > 0:
            assert i0_allLabels > 0
            for k in statesDict.keys():
                last_key = k
            s = statesDict[last_key]
            s2 = slice(s.start, inds[0])
            statesDict[k] = s2
        # iterate over self.allLabels because it corresponds to self.unixnanos
        for i, label in enumerate(self.allLabels[i0_allLabels:]):
            if label not in self.unaliasedLabels:
                continue
            aliasedLabel = self.labelAliasesDict.get(label, label)
            if i+1 == len(self.allLabels):
                s = slice(inds[i], len(unixnanos))
            else:
                s = slice(inds[i], inds[i+1])
            if aliasedLabel in statesDict:
                # this label is unique, use a list of slices
                v = statesDict[aliasedLabel]
                if isinstance(v, slice):
                    # this label was previously unique... create the list of slices
                    statesDict[aliasedLabel] = [v, s]
                elif isinstance(v, list):
                    # this label was previously not unique... append to the list of slices
                    statesDict[aliasedLabel] = v+[s]
                else:
                    raise Exception("v should be a slice or list of slices, v is a {} for label={}, aliasedlabel={}".format(
                        type(v), label, aliasedLabel))
            else:  # this state is unique, use a slice
                statesDict[aliasedLabel] = s
        # statesDict values should be slices for unique states and lists of slices for non-unique states
        self._preventAliasState = True
        assert(len(statesDict) == len(self.unaliasedLabels))
        return statesDict

    def __repr__(self):
        return "ExperimentStateFile: "+self.filename

    def aliasState(self, unaliasedLabel, aliasedLabel):
        if self._preventAliasState:
            raise Exception("call aliasState before calculating or re-calculating statesDict")
        self.labelAliasesDict[unaliasedLabel] = aliasedLabel

    @property
    def labels(self):
        return [self.labelAliasesDict.get(label, label) for label in self.unaliasedLabels]


class DriftCorrection():
    version = 1

    def __init__(self, indicatorName, uncorrectedName, medianIndicator, slope):
        self.indicatorName = indicatorName
        self.uncorrectedName = uncorrectedName
        self.medianIndicator = medianIndicator
        self.slope = slope

    def __call__(self, indicator, uncorrected):
        return self.apply(indicator, uncorrected)

    def apply(self, indicator, uncorrected):
        gain = 1+(indicator-self.medianIndicator)*self.slope
        return gain*uncorrected

    def toHDF5(self, hdf5_group, name="driftCorrection"):
        hdf5_group["{}/indicatorName".format(name)] = self.indicatorName
        hdf5_group["{}/uncorrectedName".format(name)] = self.uncorrectedName
        hdf5_group["{}/medianIndicator".format(name)] = self.medianIndicator
        hdf5_group["{}/slope".format(name)] = self.slope
        hdf5_group["{}/version".format(name)] = self.version

    @classmethod
    def fromHDF5(cls, hdf5_group, name="driftCorrection"):
        indicatorName = hdf5_group["{}/indicatorName".format(name)][()]
        uncorrectedName = hdf5_group["{}/uncorrectedName".format(name)][()]
        medianIndicator = hdf5_group["{}/medianIndicator".format(name)][()]
        slope = hdf5_group["{}/slope".format(name)][()]
        version = hdf5_group["{}/version".format(name)][()]
        assert(version == cls.version)
        return cls(indicatorName, uncorrectedName, medianIndicator, slope)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DriftCorrection):
            return self.__dict__ == other.__dict__
        else:
            return False


class CorG():
    """
    implments methods that are shared across Channel and ChannelGroup
    """
    @property
    def stateLabels(self):
        return self.experimentStateFile.labels

    def plotHist(self, binEdges, attr, axis=None, labelLines=[], states=None, cutRecipeName=None, coAddStates=True):
        """plot a coadded histogram from all good datasets and all good pulses
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        axis -- if None, then create a new figure, otherwise plot onto this axis
        annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
        cutRecipeName -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer    """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels
        if coAddStates:
            x, y = self.hist(binEdges, attr, states=states, cutRecipeName=cutRecipeName)
            axis.plot(x, y, drawstyle="steps-mid", label=states)
        else:
            for state in states:
                x, y = self.hist(binEdges, attr, states=state, cutRecipeName=cutRecipeName)
                axis.plot(x, y, drawstyle="steps-mid", label=state)
        axis.set_xlabel(attr)
        axis.set_ylabel("counts per %0.1f unit bin" % (binEdges[1]-binEdges[0]))
        plt.legend(title="states")
        axis.set_title(self.shortName)
        annotate_lines(axis, labelLines)
        return axis

    def linefit(self, lineNameOrEnergy="MnKAlpha", attr="energy", states=None, axis=None, dlo=50, dhi=50,
                binsize=None, binEdges=None, label="full", plot=True,
                params_fixed=None, cutRecipeName=None, calibration=None, require_errorbars=True, method="leastsq_refit",
                has_linear_background=True, has_tails=False, params_update=lmfit.Parameters()):
        """Do a fit to `lineNameOrEnergy` and return the result. You can get the params results with result.params
        lineNameOrEnergy -- A string like "MnKAlpha" will get "MnKAlphaModel", your you can pass in a model like a mass.MnKAlphaModel().
        attr -- default is "energyRough". you must pass binEdges if attr is other than "energy" or "energyRough"
        states -- will be passed to hist, coAddStates will be True
        axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
        dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(model.spect.peak_energy-dlo, model.spect.peak_energy+dhi, binsize)
        binEdges -- pass the binEdges you want as a numpy array
        label -- passed to model.plot
        plot -- passed to model.fit, determine if plot happens
        params_fixed -- passed to model.fit, model.fit will guess the params on its own if this is None, in either case it will update with params_update
        cutRecipeName -- a function a function taking a MicrocalDataSet and returning a vector like ds.good() would return
        calbration -- a calibration to be passed to hist - will error if used with an "energy..." attr
        require_errorbars -- throw an error if lmfit doesn't return errorbars
        method -- fit method to use
        has_tails -- used when creating a model, will add both high and low energy tails to the model
        params_update -- after guessing params, call params.update(params_update)
        """
        model = util.get_model(lineNameOrEnergy, has_linear_background=has_linear_background, has_tails=has_tails)
        cutRecipeName = self._handleDefaultCut(cutRecipeName)
        if binEdges is None:
            if attr.startswith("energy") or calibration is not None:
                pe = model.spect.peak_energy
                binEdges = np.arange(pe-dlo, pe+dhi, self._handleDefaultBinsize(binsize))
            else:
                raise Exception(
                    "must pass binEdges if attr does not start with energy and you don't pass a calibration, also don't use energy and calibration at the same time")
        # print(f"binEdges.size={binEdges.size}, binEdges.mean()={binEdges.mean()}")
        # print(f"attr={attr},states={states}")
        bin_centers, counts = self.hist(
            binEdges, attr, states, cutRecipeName, calibration=calibration)
        # print(f"counts.size={counts.size},counts.sum()={counts.sum()}")
        if params_fixed is None:
            params = model.guess(counts, bin_centers=bin_centers)
        else:
            params = params_fixed
        if attr.startswith("energy") or calibration is not None:
            params["dph_de"].set(1.0, vary=False)
            unit_str = "eV"
        if calibration is None:
            unit_str = "arbs"
        if calibration is not None:
            attr_str = f"{attr} with cal"
        else:
            attr_str = attr
        params.update(params_update)
        # unit_str and attr_str are used by result.plotm to label the axes properly
        result = model.fit(counts, params, bin_centers=bin_centers, method=method)
        if states is None:
            states_hint = "all states"
        elif isinstance(states, list):
            states_hint = ", ".join(states)
        else:
            states_hint = states
        result.set_label_hints(binsize=bin_centers[1]-bin_centers[0], ds_shortname=self.shortName,
                               unit_str=unit_str, attr_str=attr_str, cut_hint=cutRecipeName, states_hint=states_hint)
        if plot:
            result.plotm(ax=axis)
        return result

    _default_bin_size = 1.0

    def setDefaultBinsize(self, binsize):
        """sets the default binsize in eV used by self.linefit and functions that call it"""
        self._default_bin_size = binsize

    def _handleDefaultBinsize(self, binsize):
        if binsize is None:
            return self._default_bin_size
        else:
            return binsize


# wrap up an off file with some conviencine functions
# like a TESChannel
class Channel(CorG):
    def __init__(self, offFile, experimentStateFile, verbose=True):
        self.offFile = offFile
        self.experimentStateFile = experimentStateFile
        self.markedBadBool = False
        self._statesDict = None
        self.verbose = verbose
        self.learnChannumAndShortname()
        self.recipes = RecipeBook(self._offAttrs, Channel, 
            wrapper = lambda x: util.IngredientsWrapper(x, self.offFile._dtype_non_descriptive))
        # wrapper is part of a hack to allow "coefs" and "filtValue" to be recipe ingredients
        self._defineDefaultRecipesAndProperties()  # sets _default_cut_recipe_name

    def _defineDefaultRecipesAndProperties(self):
        assert(len(self.recipes) == 0)
        t0 = self.offFile["unixnano"][0]
        self.recipes.add("relTimeSec", lambda unixnano: (unixnano-t0)*1e-9, ["unixnano"])
        self.recipes.add("filtPhase", lambda x, y: x/y, ["derivativeLike", "filtValue"])
        self.cutAdd("cutNone", lambda filtValue: np.ones(
            len(filtValue), dtype="bool"), setDefault=True)

    @add_group_loop
    def cutAdd(self, cutRecipeName, f, ingredients=None, overwrite=False, setDefault=False):
        self.recipes.add(cutRecipeName, f, ingredients, overwrite=overwrite)
        if setDefault:
            self.cutSetDefault(cutRecipeName)

    def cutSetDefault(self, cutRecipeName):
        assert cutRecipeName.startswith("cut")
        assert cutRecipeName in self.recipes.keys()
        self._default_cut_recipe_name = cutRecipeName

    def _handleDefaultCut(self, cutRecipeName):
        if cutRecipeName is None:
            return self._default_cut_recipe_name
        else:
            return cutRecipeName

    @property
    def _offAttrs(self):
        # adding ("coefs",) is part of a hack to allow "coefs" and "filtValue" to be recipe ingredients       
        return self.offFile.dtype.names+("coefs",)

    @property
    def _recipeAttrs(self):
        return self.recipes.keys()

    def isOffAttr(self, attr):
        return attr in self._offAttrs

    def isRecipeAttr(self, attr):
        return attr in self._recipeAttrs

    def learnChannumAndShortname(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFile.filename)
        self.shortName = os.path.split(basename)[-1] + " chan%g" % self.channum

    @add_group_loop
    def learnResidualStdDevCut(self, n_sigma_equiv=15, newCutRecipeName="cutResidualStdDev", binSizeFv=2000, minFv=150,
                               states=None, plot=False, setDefault=True, overwriteRecipe=False, cutRecipeName=None):
        """EXPERIMENTAL: learn a cut based on the residualStdDev. uses the median absolute deviation to estiamte a gaussian sigma
        that is robust to outliers as a function of filt Value, then uses that to set an upper limit based on n_sigma_equiv
        highly reccomend that you call it with plot=True on at least a few datasets first
        """
        # the code currently only works for a single threshold, but has some parts in place for implementing a filtValue dependent threshold
        filtValue, residualStdDev = self.getAttr(
            ["filtValue", "residualStdDev"], indsOrStates=states, cutRecipeName=cutRecipeName)
        # binEdges = np.percentile(filtValue, np.linspace(0, 100, N+1))
        binEdges = np.arange(0, np.amax(filtValue), binSizeFv)
        N = len(binEdges)-1
        sigmas, medians, fv_mids = [0], [0], [minFv]
        for i in range(N):
            lo, hi = binEdges[i], binEdges[i+1]
            inds = np.logical_and(filtValue > lo, filtValue < hi)
            if len(inds) <= 4:
                continue
            mad, sigma_equiv, median = mass.off.util.median_absolute_deviation(residualStdDev[inds])
            sigmas.append(sigma_equiv)
            medians.append(median)
            fv_mids.append((lo+hi)/2)
        if len(sigmas) < 1:
            raise Exception(f"too few pulses, len(filtValue)={len(filtValue)}")
        sigmas = np.array(sigmas)
        medians = np.array(medians)
        fv_mids = np.array(fv_mids)

        threshold = medians+n_sigma_equiv*sigmas
        threshold_func = scipy.interpolate.interp1d(fv_mids, threshold, kind="next", bounds_error=False,
                                                    fill_value=(-1, threshold[-1]))
        # the threshold for all filtValues below minFv will be -1
        # filtValues just above binFv should look to the next point since kind="next", so the precise chioce of median and sigma to pair with binFv shouldn't matter
        # filtValues above the maximum filtValue should use the same threshold as the maximum filtValue
        self.cutAdd(newCutRecipeName,
                    lambda filtValue, residualStdDev: residualStdDev < threshold_func(filtValue),
                    setDefault=setDefault, overwrite=overwriteRecipe)

        if plot:
            xmin, xmax = np.amin(filtValue), np.amax(filtValue)
            ymin, ymax = np.amin(residualStdDev), np.amax(residualStdDev)
            assert ymin > 0
            x = np.linspace(xmin, xmax, 1000)
            y = threshold_func(x)
            self.plotAvsB("filtValue", "residualStdDev", states=states, includeBad=True,
                          cutRecipeName=newCutRecipeName)  # creates a figure
            plt.plot(fv_mids, medians, "o-", label="median", lw=3)
            plt.plot(x, y, label=f"threshold", lw=3)
            plt.legend()
            plt.yscale("log")
            plt.ylim(ymin/2,ymax*2)

    def getStatesIndicies(self, states=None):
        """return a list of slices corresponding to
         the passed states
        this list is appropriate for passing to _indexOffWithCuts or getRecipeAttr
        """
        if isinstance(states, str):
            states = [states]
        if states is None:
            return [slice(0, len(self))]
        inds = []
        for state in states:
            v = self.statesDict[state]
            if isinstance(v, slice):
                inds.append(v)
            elif isinstance(v, list):
                for vv in v:
                    assert isinstance(vv, slice)
                    inds.append(vv)
            else:
                raise InvalidStatesException(
                    "v should be a list of slices or a slice, but is a {}".format(type(v)))
        return inds

    def __repr__(self):
        return "Channel based on %s" % self.offFile

    @property
    def statesDict(self):
        if self._statesDict is None:
            self._statesDict = self.experimentStateFile.calcStatesDict(self.offFile["unixnano"])
        return self._statesDict

    @property
    def nRecords(self):
        return len(self.offFile)

    @property
    def residualStdDev(self):
        return self.getAttr("residualStdDev", NoCutInds())

    @property
    def pretriggerMean(self):
        return self.getAttr("pretriggerMean", NoCutInds())

    @property
    def relTimeSec(self):
        # NoCutInds() is equivalent to indexing the whole array with :
        return self.getAttr("relTimeSec", NoCutInds())

    @property
    def unixnano(self):
        return self.getAttr("unixnano", NoCutInds())

    @property
    def pulseMean(self):
        return self.getAttr("pulseMean", NoCutInds())

    @property
    def derivativeLike(self):
        return self.getAttr("derivativeLike", NoCutInds())

    @property
    def filtPhase(self):
        """ used as input for phase correction """
        return self.getAttr("filtPhase", NoCutInds())

    @property
    def filtValue(self):
        return self.getAttr("filtValue", NoCutInds())

    def _indexOffWithCuts(self, inds, cutRecipeName=None, _listMethodSelect=2):
        """
        inds - a slice or list of slices to index into items with
        _listMethodSelect - used for debugging and testing, chooses the implmentation of this method used for lists of indicies
        _indexOffWithCuts(slice(0,10), f) is roughly equivalent to:
        g = f(offFile[0:10])
        offFile[0:10][g]
        """
        cutRecipeName = self._handleDefaultCut(cutRecipeName)
        # offAttr can be a list of offAttr names
        if isinstance(inds, slice):
            r = self.offFile[inds]
            # I'd like to be able to do either r["coefs"] to get all projection coefficients
            # or r["filtValue"] to get only the filtValue
            # IngredientsWrapper lets that work within recipes.craft
            g = self.recipes.craft(cutRecipeName, util.IngredientsWrapper(r, self.offFile._dtype_non_descriptive))
            output = r[g]
        elif isinstance(inds, list) and _listMethodSelect == 2:  # preallocate and truncate
            # testing on the 20191219_0002 TOMCAT dataset with len(inds)=432 showed this method to be more than 10x faster than repeated hstack
            # and about 2x fatster than temporary bool index, which can be found in commit 063bcce
            # make sure s.step is None so my simple length calculation will work
            assert all([isinstance(s, slice) and s.step is None for s in inds])
            max_length = np.sum([s.stop-s.start for s in inds])
            output_dtype = self.offFile.dtype  # get the dtype to preallocate with
            output_prealloc = np.zeros(max_length, output_dtype)
            ilo, ihi = 0, 0
            for s in inds:
                tmp = self._indexOffWithCuts(s, cutRecipeName)
                ilo = ihi
                ihi = ilo+len(tmp)
                output_prealloc[ilo:ihi] = tmp
            output = output_prealloc[0:ihi]
        elif isinstance(inds, list) and _listMethodSelect == 0:  # repeated hstack
            # this could be removed, along with the _listMethodSelect argument
            # this is only left in because it is useful for correctness testing for preallocate and truncate method since this is simpler
            assert all([isinstance(_inds, slice) for _inds in inds])
            output = self._indexOffWithCuts(inds[0], cutRecipeName)
            for i in range(1, len(inds)):
                output = np.hstack((output, self._indexOffWithCuts(inds[i], cutRecipeName)))
        elif isinstance(inds, NoCutInds):
            output = self.offFile
        else:
            raise Exception("type(inds)={}, should be slice or list or slices".format(type(inds)))
        return output

    def getAttr(self, attr, indsOrStates, cutRecipeName=None):
        """
        attr - may be a string or a list of strings corresponding to Attrs defined by recipes or the offFile
        inds - a slice or list of slices
        returns either a single vector or a list of vectors whose entries correspond to the entries in attr
        """
        # first
        # relies on short circuiting to not evaluate last clause unless indsOrStates is a list
        if indsOrStates is None or isinstance(indsOrStates, str) or (isinstance(indsOrStates, list) and isinstance(indsOrStates[0], str)):
            # looks like states
            try:
                inds = self.getStatesIndicies(indsOrStates)
            except InvalidStatesException:
                inds = indsOrStates
        else:
            inds = indsOrStates
        # single read from disk, read all values
        offAttrValues = self._indexOffWithCuts(inds, cutRecipeName)
        if isinstance(attr, list):
            return [self._getAttr(a, offAttrValues) for a in attr]
        else:
            return self._getAttr(attr, offAttrValues)

    def _getAttr(self, attr, offAttrValues):
        """ internal function used to implement getAttr, does no cutting """
        if self.isRecipeAttr(attr):
            return self.recipes.craft(attr, offAttrValues)
        if attr == "coefs":
            return offAttrValues.view(self.offFile._dtype_non_descriptive)["coefs"]
        elif self.isOffAttr(attr):
            return offAttrValues[attr]
        else:
            raise Exception("attr {} must be an OffAttr or a RecipeAttr or a list. OffAttrs: {}\nRecipeAttrs: {}".format(
                attr, list(self._offAttrs), list(self._recipeAttrs)))


    def plotAvsB(self, nameA, nameB, axis=None, states=None, includeBad=False, cutRecipeName=None):
        cutRecipeName = self._handleDefaultCut(cutRecipeName)
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels    
        if isinstance(nameB, list):
            self._plotAvsB_list(nameA, nameB, axis, states, includeBad, cutRecipeName)
        else:
            self._plotAvsB_single(nameA, nameB, axis, states, includeBad, cutRecipeName)
        plt.xlabel(nameA)
        plt.ylabel(nameB)
        plt.title(f"{self.shortName}\ncutRecipeName={cutRecipeName}")
        plt.legend(title="states")       
        return axis

    def _plotAvsB_list(self, nameA, nameBlist, axis, states, includeBad, cutRecipeName):
            for nameB in nameBlist:
                self._plotAvsB_single(nameA, nameB, axis, states, includeBad, cutRecipeName, prefix=nameB)

    def _plotAvsB_single(self, nameA, nameB, axis=None, states=None, includeBad=False, cutRecipeName=None, prefix=""):
        for state in states:
            A, B = self.getAttr([nameA, nameB], state, cutRecipeName)
            axis.plot(A, B, ".", label=prefix+state)
            if includeBad:
                A, B = self.getAttr([nameA, nameB], state, f"!{cutRecipeName}")
                axis.plot(A, B, "x", label=prefix+state+" bad")

    def hist(self, binEdges, attr, states=None, cutRecipeName=None, calibration=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
        cutRecipeName -- a function taking a 1d array of vales of type self.offFile.dtype and returning a vector of bool
        calibration -- if not None, transform values by val = calibration(val) before histogramming
        """
        binEdges = np.array(binEdges)
        binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vals = self.getAttr(attr, states, cutRecipeName)
        if calibration is not None:
            vals = calibration(vals)
        counts, _ = np.histogram(vals, binEdges)
        return binCenters, counts

    @add_group_loop
    def learnDriftCorrection(self, indicatorName="pretriggerMean", uncorrectedName="filtValue",
                             correctedName=None, states=None, cutRecipeName=None, overwriteRecipe=False):
        """do a linear correction between the indicator and uncorrected... """
        if correctedName is None:
            correctedName = uncorrectedName + "DC"
        indicator, uncorrected = self.getAttr(
            [indicatorName, uncorrectedName], states, cutRecipeName)
        slope, info = mass.core.analysis_algorithms.drift_correct(
            indicator, uncorrected)
        driftCorrection = DriftCorrection(
            indicatorName, uncorrectedName, info["median_pretrig_mean"], slope)
        self.recipes.add(correctedName, driftCorrection, [
            driftCorrection.indicatorName, driftCorrection.uncorrectedName], overwrite=overwriteRecipe)
        return driftCorrection

    @add_group_loop
    def learnPhaseCorrection(self, indicatorName="filtPhase", uncorrectedName="filtValue", correctedName=None, states=None,
                             linePositionsFunc=None, cutRecipeName=None):
        """
        linePositionsFunc - if None, then use self.calibrationRough._ph as the peak locations
        otherwise try to call it with self as an argument... here is an example of how you could use all but one peak from calibrationRough:
        `data.learnPhaseCorrection(linePositionsFunc = lambda ds: ds.recipes["energyRough"].f._ph`
        """
        # may need to generalize this to allow using a specific state for phase correction as a specfic line... with something like calibrationPlan
        if correctedName is None:
            correctedName = uncorrectedName + "PC"
        if linePositionsFunc is None:
            linePositions = self.recipes["energyRough"].f._ph
        else:
            linePositions = linePositionsFunc(self)
        indicator, uncorrected = self.getAttr(
            [indicatorName, uncorrectedName], states, cutRecipeName)
        phaseCorrection = mass.core.phase_correct.phase_correct(
            indicator, uncorrected, linePositions, indicatorName=indicatorName, uncorrectedName=uncorrectedName)
        self.recipes.add(correctedName, phaseCorrection.correct, [
            phaseCorrection.indicatorName, phaseCorrection.uncorrectedName])

    @add_group_loop
    def learnTimeDriftCorrection(self, indicatorName="relTimeSec", uncorrectedName="filtValue", correctedName=None,
                                 states=None, cutRecipeName=None, kernel_width=1, sec_per_degree=2000,
                                 pulses_per_degree=2000, max_degrees=20, ndeg=None, limit=None):
        """do a polynomial correction based on the indicator
        you are encouraged to change the settings that affect the degree of the polynomail
        see help in mass.core.channel.time_drift_correct for details on settings"""
        if correctedName is None:
            correctedName = uncorrectedName+"TC"
        indicator, uncorrected = self.getAttr(
            [indicatorName, uncorrectedName], states, cutRecipeName)
        info = mass.core.channel.time_drift_correct(indicator, uncorrected, kernel_width, sec_per_degree,
                                                    pulses_per_degree, max_degrees, ndeg, limit)

        def time_drift_correct(indicator, uncorrected):
            tnorm = info["normalize"](indicator)
            corrected = uncorrected*(1+info["model"](tnorm))
            return corrected
        self.recipes.add(correctedName, time_drift_correct, [indicatorName, uncorrectedName])

    def plotCompareDriftCorrect(self, axis=None, states=None, cutRecipeName=None, includeBad=False):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        recipe = self.recipes["filtValueDC"]
        indicatorName = "pretriggerMean"
        uncorrectedName = "filtValue"
        assert recipe.i2a[indicatorName] == "indicator"
        assert recipe.i2a[uncorrectedName] == "uncorrected"
        if states is None:
            states = self.stateLabels
        for state in states:
            A, B, C = self.getAttr([indicatorName, uncorrectedName,
                                    "filtValueDC"], state, cutRecipeName)
            axis.plot(A, B, ".", label=state)
            axis.plot(A, C, ".", label=state+" DC")
            if includeBad:
                A, B, C = self.getAttr([indicatorName, uncorrectedName,
                                        "filtValueDC"], state, cutRecipeName=True)
                axis.plot(A, B, "x", label=state+" bad")
                axis.plot(A, C, "x", label=state+" bad DC")
        plt.xlabel(indicatorName)
        plt.ylabel(uncorrectedName + ",filtValueDC")
        plt.title(self.shortName+" drift correct comparison")
        plt.legend(title="states")
        return axis

    def calibrationPlanInit(self, attr):
        self.calibrationPlan = CalibrationPlan()
        self.calibrationPlanAttr = attr

    def calibrationPlanAddPoint(self, uncalibratedVal, name, states=None, energy=None):
        if energy is None:
            if name in mass.spectra:
                line = mass.spectra[name]    
            elif name in mass.STANDARD_FEATURES:
                energy = mass.STANDARD_FEATURES[name]
                line = mass.SpectralLine.quick_monochromatic_line(name, energy, 0.001, 0)
            else:
                raise Exception("failed to get line")
        else:
            line = mass.SpectralLine.quick_monochromatic_line(name, energy, 0.001, 0)
        self.calibrationPlan.addCalPoint(uncalibratedVal, states, line)
        calibrationRough = self.calibrationPlan.getRoughCalibration()
        calibrationRough.uncalibratedName = self.calibrationPlanAttr
        self.recipes.add("energyRough", calibrationRough,
                         [calibrationRough.uncalibratedName], inverse=calibrationRough.energy2ph, overwrite=True)
        return self.calibrationPlan

    @add_group_loop
    def calibrateFollowingPlan(self, uncalibratedName, calibratedName="energy", curvetype="gain", approximate=False,
                               dlo=50, dhi=50, binsize=None, plan=None, n_iter=1, method="leastsq_refit", overwriteRecipe=False,
                               has_tails=False, params_update=lmfit.Parameters()):
        if plan is None:
            plan = self.calibrationPlan
        starting_cal = plan.getRoughCalibration()
        intermediate_calibrations = []
        for i in range(n_iter):
            calibration = mass.EnergyCalibration(curvetype=curvetype, approximate=approximate)
            calibration.uncalibratedName = uncalibratedName
            results = []
            for (ph,line, states) in zip(plan.uncalibratedVals, plan.lines, plan.states):
                result = self.linefit(line, uncalibratedName, states, dlo=dlo, dhi=dhi,
                                      plot=False, binsize=binsize, calibration=starting_cal, require_errorbars=False,
                                      method=method, params_update=params_update, has_tails=has_tails)

                results.append(result)
                if not result.success:
                    self.markBad(f"calibrateFollowingPlan: failed fit {line}, states {states}", 
                    extraInfo=result)
                    continue
                if not result.errorbars:
                    self.markBad(f"calibrateFollowingPlan: {line} fit without error bars, states={states}",
                    extraInfo=result)
                    continue
                ph = starting_cal.energy2ph(result.params["peak_ph"].value)
                ph_uncertainty = result.params["peak_ph"].stderr / \
                    starting_cal.energy2dedph(result.params["peak_ph"].value)
                calibration.add_cal_point(ph, line.peak_energy, line.shortname, ph_uncertainty)
            calibration.results = results
            calibration.plan = plan
            is_last_iteration = i+1 == n_iter
            if not is_last_iteration:
                intermediate_calibrations.append(calibration)
                starting_cal = calibration
        calibration.intermediate_calibrations = intermediate_calibrations
        self.recipes.add(calibratedName, calibration, 
            [calibration.uncalibratedName], overwrite=overwriteRecipe)
        return results

    def markBad(self, reason, extraInfo=None):
        self.markedBadReason = reason
        self.markedBadExtraInfo = extraInfo
        self.markedBadBool = True
        s = "\nMARK BAD {}: reason={}".format(self.shortName, reason)
        if extraInfo is not None:
            s += "\nextraInfo: {}".format(extraInfo)
        if self.verbose:
            LOG.warning(s)

    def markGood(self):
        self.markedBadReason = None
        self.markedBadExtraInfo = None
        self.markedBadBool = False

    def __len__(self):
        return len(self.offFile)

    @add_group_loop
    def alignToReferenceChannel(self, referenceChannel, attr, binEdges, cutRecipeName=None, _peakLocs=None, states=None):
        if _peakLocs is None:
            assert(len(referenceChannel.calibrationPlan.uncalibratedVals) > 0)
            peakLocs = referenceChannel.calibrationPlan.uncalibratedVals
        else:
            peakLocs = _peakLocs
        self.aligner = AlignBToA(ds_a=referenceChannel, ds_b=self,
                                 peak_xs_a=peakLocs, bin_edges=binEdges, attr=attr, states=states,
                                 cutRecipeName=cutRecipeName)
        self.calibrationArbsInRefChannelUnits = self.aligner.getCalBtoA()
        if _peakLocs is None and not (self is referenceChannel):
            self.calibrationPlanInit(referenceChannel.calibrationPlanAttr)
            refCalPlan = referenceChannel.calibrationPlan
            for (ph, energy, name, states) in zip(refCalPlan.uncalibratedVals, refCalPlan.energies,
                                                  refCalPlan.names, refCalPlan.states):
                self.calibrationPlanAddPoint(self.calibrationArbsInRefChannelUnits.energy2ph(ph),
                                             name, states, energy)
        self.recipes.add("arbsInRefChannelUnits", self.calibrationArbsInRefChannelUnits.ph2energy, [
            self.calibrationArbsInRefChannelUnits.uncalibratedName])
        return self.aligner

    @add_group_loop
    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None, worstAllowedFWHM=None,
                            positionToleranceAbsolute=None, attr="energy", states=None,
                            dlo=50, dhi=50, binsize=None, binEdges=None, guessParams=None,
                            cutRecipeName=None, holdvals=None):
        """calls ds.linefit to fit the given line
        marks self bad if the fit position is more than toleranceFitSigma*fitSigma away
        from the correct position
        """
        result = self.linefit(line, attr, states, None, dlo, dhi, binsize, binEdges,
                              guessParams, cutRecipeName, holdvals)
        fitPos = result.params["peak_ph"].value
        fitSigma = result.params["peak_ph"].stderr
        resolution = result.params["fwhm"].value
        if positionToleranceAbsolute is not None:
            if positionToleranceFitSigma is not None:
                raise Exception(
                    "specify only one of positionToleranceAbsolute or positionToleranceFitSigma")
            tolerance = positionToleranceAbsolute
        elif positionToleranceFitSigma is not None:
            tolerance = fitSigma*positionToleranceFitSigma
        else:
            tolerance = np.inf
        if np.abs(fitPos-result.model.spect.peak_energy) > tolerance:
            self.markBad("qualityCheckLinefit: for {}, want {} within {}, got {}".format(
                line, result.model.spect.peak_energy, tolerance, fitPos))
        if worstAllowedFWHM is not None and resolution > worstAllowedFWHM:
            self.markBad("qualityCheckLinefit: for {}, fit resolution {} > threshold {}".format(
                line, resolution, worstAllowedFWHM))
        return result

    @add_group_loop
    def histsToHDF5(self, h5File, binEdges, attr="energy", cutRecipeName=None):
        grp = h5File.require_group(str(self.channum))
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, cutRecipeName)
            grp["{}/bin_centers".format(state)] = binCenters
            grp["{}/counts".format(state)] = counts
        binCenters, counts = self.hist(
            binEdges, attr, cutRecipeName=cutRecipeName)  # all states hist
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    @add_group_loop
    def energyTimestampLabelToHDF5(self, h5File, cutRecipeName=None):
        grp = h5File.require_group(str(self.channum))
        if len(self.stateLabels) > 0:
            for state in self.stateLabels:
                energy, unixnano = self.getAttr(["energy", "unixnano"], state, cutRecipeName)
                grp["{}/energy".format(state)] = energy
                grp["{}/unixnano".format(state)] = unixnano
        else:
            energy, unixnano = self.getAttr(
                ["energy", "unixnano"], slice(None), cutRecipeName)
            grp["{}/energy".format(state)] = energy
            grp["{}/unixnano".format(state)] = unixnano

    @add_group_loop
    def qualityCheckDropOneErrors(self, thresholdAbsolute=None, thresholdSigmaFromMedianAbsoluteValue=None):
        calibration = self.recipes["energy"].f
        energies, errors = calibration.drop_one_errors()
        maxAbsError = np.amax(np.abs(errors))
        medianAbsoluteValue = np.median(np.abs(errors))
        k = 1.4826  # https://en.wikipedia.org/wiki/Median_absolute_deviation
        sigma = k*medianAbsoluteValue
        if thresholdAbsolute is not None:
            if maxAbsError > sigma*thresholdSigmaFromMedianAbsoluteValue:
                self.markBad("qualityCheckDropOneErrors: maximum absolute drop one error {} > theshold {} ({})".format(
                    maxAbsError, sigma*thresholdSigmaFromMedianAbsoluteValue,
                    "thresholdSigmaFromMedianAbsoluteValue"))
        if thresholdAbsolute is not None:
            if maxAbsError > thresholdAbsolute:
                self.markBad("qualityCheckDropOneErrors: maximum absolute drop one error {} > theshold {} (thresholdAbsolute)".format(
                    maxAbsError, thresholdAbsolute))

    def diagnoseCalibration(self, calibratedName="energy"):
        calibration = self.recipes[calibratedName].f
        uncalibratedName = calibration.uncalibratedName
        results = calibration.results
        n_intermediate = len(calibration.intermediate_calibrations)
        plt.figure(figsize=(20, 12))
        plt.suptitle(
            self.shortName+", cal diagnose for '{}'\n with {} intermediate calibrations".format(calibratedName, n_intermediate))
        n = int(np.ceil(np.sqrt(len(results)+2)))
        for i, result in enumerate(results):
            ax = plt.subplot(n, n, i+1)
            # pass title to suppress showing the dataset shortName on each subplot
            result.plotm(ax=ax, title=str(result.model.spect.shortname))
        ax = plt.subplot(n, n, i+2)
        calibration.plot(axis=ax)
        ax = plt.subplot(n, n, i+3)
        self.plotHist(np.arange(0, 16000, 4), uncalibratedName,
                      axis=ax, coAddStates=False)
        plt.vlines(self.calibrationPlan.uncalibratedVals, 0, plt.ylim()[1])
        plt.tight_layout()

    def add5LagRecipes(self, f):
        filter_5lag_in_basis, filter_5lag_fit_in_basis = fivelag.calc_5lag_fit_matrix(f, self.offFile.basis)
        self.recipes.add("cba5Lag", lambda coefs: np.matmul(coefs, filter_5lag_fit_in_basis))
        self.recipes.add("filtValue5Lag", lambda cba5Lag: fivelag.filtValue5Lag(cba5Lag))
        self.recipes.add("peakX5Lag", lambda cba5Lag: fivelag.peakX5Lag(cba5Lag))

def normalize(x):
    return x/float(np.sum(x))

def dtw_same_peaks(bin_edges, ph_a, ph_b, peak_inds_a, scale_by_median, normalize_before_dtw, plot=False):
    if scale_by_median:
        median_ratio_a_over_b = np.median(ph_a)/np.median(ph_b)
    else:
        median_ratio_a_over_b = 1.0
    ph_b_median_scaled = ph_b*median_ratio_a_over_b
    counts_a, _ = np.histogram(ph_a, bin_edges)
    counts_b_median_scaled, _ = np.histogram(ph_b_median_scaled, bin_edges)
    if normalize_before_dtw:
        distance, path = fastdtw.fastdtw(normalize(counts_a), 
            normalize(counts_b_median_scaled))
    else:
        distance, path = fastdtw.fastdtw(counts_a, counts_b_median_scaled)
    i_a = [x[0] for x in path]
    i_b_median_scaled = [x[1] for x in path]
    peak_inds_b_median_scaled = np.array(
        [i_b_median_scaled[i_a.index(pia)] for pia in peak_inds_a])
    peak_xs_b_median_scaled = bin_edges[peak_inds_b_median_scaled]
    peak_xs_b = peak_xs_b_median_scaled/median_ratio_a_over_b
    min_bin = bin_edges[0]
    bin_spacing = bin_edges[1]-bin_edges[0]
    peak_inds_b = np.array((peak_xs_b-min_bin)/bin_spacing, dtype="int")

    if plot:
        counts_b, _ = np.histogram(ph_b, bin_edges)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.figure()
        plt.plot(counts_a, label="counts_a")
        plt.plot(counts_b, label="counts_b")
        plt.plot(peak_inds_a, counts_a[peak_inds_a], "o")
        plt.plot(peak_inds_b, counts_b[peak_inds_b], "s")
        plt.legend()
        plt.xlabel("ind")

        plt.figure()
        plt.plot(bin_centers, counts_a, label="a")
        plt.plot(bin_centers, counts_b, label="b")
        plt.plot(bin_centers[peak_inds_a], counts_a[peak_inds_a], "o", label="a")
        plt.plot(bin_centers[peak_inds_b], counts_b[peak_inds_b], "s", label="b")
        plt.xlabel("bin_centers")
    return peak_inds_b    


class AlignBToA():
    cm = plt.cm.gist_ncar

    def __init__(self, ds_a, ds_b, peak_xs_a, bin_edges, attr, cutRecipeName, states=None,
                 scale_by_median=True, normalize_before_dtw=True):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.bin_edges = bin_edges
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        self.peak_xs_a = peak_xs_a
        self.attr = attr
        self.cutRecipeName = cutRecipeName
        self.states = states
        self.scale_by_median = scale_by_median
        self.normalize_before_dtw = normalize_before_dtw
        self.peak_inds_a = np.searchsorted(self.bin_edges, self.peak_xs_a)-1
        self.peak_inds_b = self.samePeaks()

    def samePeaks(self, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        ph_a = self.ds_a.getAttr(self.attr, self.states, cutRecipeName_a)
        ph_b = self.ds_b.getAttr(self.attr, self.states, cutRecipeName_b)
        return dtw_same_peaks(self.bin_edges, ph_a, ph_b, self.peak_inds_a, self.scale_by_median, self.normalize_before_dtw)


    def samePeaksPlot(self, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        ph_a = self.ds_a.getAttr(self.attr, self.states, cutRecipeName_a)
        ph_b = self.ds_b.getAttr(self.attr, self.states, cutRecipeName_b)
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        plt.figure()
        plt.plot(self.bin_centers, counts_a, label="a: channel %i" % self.ds_a.channum)
        for i, pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi], counts_a[pi], "o",
                     color=self.cm(float(i)/len(self.peak_inds_a)))

        plt.plot(self.bin_centers, counts_b, label="b: channel %i" % self.ds_b.channum)
        for i, pi in enumerate(self.peak_inds_b):
            plt.plot(self.bin_centers[pi], counts_b[pi], "o",
                     color=self.cm(float(i)/len(self.peak_inds_b)))
        plt.xlabel(self.attr)
        plt.ylabel("counts per %0.2f unit bin" % (self.bin_centers[1]-self.bin_centers[0]))
        plt.legend(title="channel")
        plt.title(self.ds_a.shortName+" + "+self.ds_b.shortName
                  + "\nwith same peaks noted, peaks not expected to be aligned in this plot")

    # somehow this plot is wrong... the channel a histogram is wrong somehow
    def samePeaksPlotWithAlignmentCal(self, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        ph_a = self.ds_a.getAttr(self.attr, self.states, cutRecipeName_a)
        ph_b = self.ds_b.getAttr("arbsInRefChannelUnits", self.states, cutRecipeName_b)
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        # breakpoint()
        plt.figure()
        plt.plot(self.bin_centers, counts_a, label="a: channel %i" % self.ds_a.channum)
        for i, pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi], counts_a[pi], "o",
                     color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.plot(self.bin_centers, counts_b, label="b: channel %i" % self.ds_b.channum)
        for i, pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi], counts_b[pi], "o",
                     color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.xlabel("arbsInRefChannelUnits (ref channel = {})".format(self.ds_a.channum))
        plt.ylabel("counts per %0.2f unit bin" % (self.bin_centers[1]-self.bin_centers[0]))
        plt.legend()

    def getCalBtoA(self):
        cal_b_to_a = mass.EnergyCalibration(curvetype="gain")
        for pi_a, pi_b in zip(self.peak_inds_a, self.peak_inds_b):
            cal_b_to_a.add_cal_point(self.bin_centers[pi_b], self.bin_centers[pi_a])
        cal_b_to_a.uncalibratedName = self.attr
        self.cal_b_to_a = cal_b_to_a
        return self.cal_b_to_a

    def testForGoodnessBasedOnCalCurvature(self, threshold_frac=.1):
        assert threshold_frac > 0
        threshold_hi = 1+threshold_frac
        threshold_lo = 1/threshold_hi
        # here we test the "curvature" of cal_b_to_a
        # by comparing the most extreme sloped segment to the median slope
        derivatives = self.cal_b_to_a.energy2dedph(self.cal_b_to_a._energies)
        diff_frac_hi = np.amax(derivatives)/np.median(derivatives)
        diff_frac_lo = np.amin(derivatives)/np.median(derivatives)
        return diff_frac_hi < threshold_hi and diff_frac_lo > threshold_lo

    def _laplaceEntropy(self, w=None, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        if w is None:
            w = self.bin_edges[1]-self.bin_edges[0]
        ph_a = self.ds_a.getAttr(self.attr, slice(None), cutRecipeName_a)
        ph_b = self.ds_b.getAttr(self.newattr, slice(None), cutRecipeName_b)
        entropy = mass.entropy.laplace_cross_entropy(ph_a[ph_a > self.bin_edges[0]],
                                                     ph_b[ph_b > self.bin_edges[0]], w=w)
        return entropy

    def _ksStatistic(self, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        ph_a = self.ds_a.getAttr(self.attr, slice(None), cutRecipeName_a)
        ph_b = self.ds_b.getAttr(self.newattr, slice(None), cutRecipeName_b)
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        cdf_a = np.cumsum(counts_a)/np.sum(counts_a)
        cdf_b = np.cumsum(counts_b)/np.sum(counts_b)
        ks_statistic = np.amax(np.abs(cdf_a-cdf_b))
        return ks_statistic


# calibration
class CalibrationPlan():
    def __init__(self):
        self.uncalibratedVals = np.zeros(0)
        self.states = []
        self.lines = []

    def addCalPoint(self, uncalibratedVal,  states, line):
        self.uncalibratedVals = np.hstack((self.uncalibratedVals, uncalibratedVal))
        self.states.append(states)
        self.lines.append(line)

    @property
    def energies(self):
        return np.array([line.peak_energy for line in self.lines])

    @property
    def names(self):
        return [line.shortname for line in self.lines]

    def __repr__(self):
        s = f"""CalibrationPlan with {len(self.lines)} entries
        x: {self.uncalibratedVals}
        y: {self.energies}
        states: {self.states}
        names: {self.names}"""
        return s

    def getRoughCalibration(self):
        cal = mass.EnergyCalibration(curvetype="gain")
        for (x, y, name) in zip(self.uncalibratedVals, self.energies, self.names):
            cal.add_cal_point(x, y, name)
        return cal


def getOffFileListFromOneFile(filename, maxChans=None):
    basename, _ = mass.ljh_util.ljh_basename_channum(filename)
    z = mass.ljh_util.filename_glob_expand(basename+"_chan*.off")
    if z is None:
        raise Exception("found no files while globbing {}".format(basename+"_chan*.off"))
    if maxChans is not None:
        z = z[:min(maxChans, len(z))]
    return z


class ChannelGroup(CorG, GroupLooper, collections.OrderedDict):
    """
    ChannelGroup is an OrdredDict of Channels with some additional features
    1. Most functions on a Channel can be called on a ChannelGroup, eg data.learnDriftCorrection()
    in this case it looks over all channels in the ChannelGroup, except those makred bad by ds.markBad("reason")
    2. If you want to iterate over all Channels, even those marked bad, do
    with data.includeBad:
        for (channum, ds) in data:
            print(channum) # will include bad channels
    3. data.whyChanBad returns an OrderedDict of bad channel numbers and reason
    """

    def __init__(self, offFileNames, verbose=True, channelClass=Channel, experimentStateFile=None, excludeStates="auto"):
        collections.OrderedDict.__init__(self)
        self.verbose = verbose
        self.offFileNames = offFileNames
        if experimentStateFile is None:
            self.experimentStateFile = ExperimentStateFile(
                datasetFilename=self.offFileNames[0], excludeStates=excludeStates)
        else:
            self.experimentStateFile = experimentStateFile
        self._includeBad = False
        self._channelClass = channelClass
        self.loadChannels()
        self._default_cut_recipe_name = self.firstGoodChannel()._default_cut_recipe_name

    def _handleDefaultCut(self, cutRecipeName):
        ds = self.firstGoodChannel()
        defaultCut = ds._default_cut_recipe_name
        for ds in self.values():
            if ds._default_cut_recipe_name != defaultCut:
                raise Exception("you are tyring to use the default cut from a channel group, but not all channels have the same default cut")
        return defaultCut

    @property
    def shortName(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFileNames[0])
        return os.path.split(basename)[-1] + " {} chans".format(len(self))

    def loadChannels(self):
        bar = SilenceBar('Parse OFF File Headers', max=len(
            self.offFileNames), silence=not self.verbose)
        for name in self.offFileNames:
            _, channum = mass.ljh_util.ljh_basename_channum(name)
            self[channum] = self._channelClass(OffFile(
                name), self.experimentStateFile, verbose=self.verbose)
            bar.next()
        bar.finish()

    def __repr__(self):
        return "ChannelGroup with {} channels".format(len(self))

    def firstGoodChannel(self):
        for ds in self.values():
            if not ds.markedBadBool:
                return ds
        raise Exception("no good channels")

    def refreshFromFiles(self):
        """
        refresh from files on disk to reflect new information: longer off files and new experiment states
        to be called occasionally when running something that updates in real time
        """
        ds = self.firstGoodChannel()
        i0_allLabels = len(self.experimentStateFile.allLabels)
        n_old_labels = len(self.experimentStateFile.labels)
        self.experimentStateFile.parse()
        n_new_labels = len(self.experimentStateFile.labels)-n_old_labels
        n_new_pulses_dict = collections.OrderedDict()
        for ds in self.values():
            i0_unixnanos = len(ds)
            ds.offFile._updateMmap()  # will update nRecords by mmapping more data in the offFile if available
            ds._statesDict = self.experimentStateFile.calcStatesDict(
                ds.unixnano[i0_unixnanos:], ds.statesDict, i0_allLabels, i0_unixnanos)
            n_new_pulses_dict[ds.channum] = len(ds)-i0_unixnanos
        return n_new_labels, n_new_pulses_dict

    def hist(self, binEdges, attr, states=None, cutRecipeName=None, calibration=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
        calibration -- will throw an exception if this is not None
         """
        if calibration is not None:
            raise Exception(
                "calibration is an argument only to match the api of Channel.hist, but is not valid for ChannelGroup.hist")
        binCenters, countsdict = self.hists(
            binEdges, attr, states, cutRecipeName=cutRecipeName)
        counts = np.zeros_like(binCenters, dtype="int")
        for (k, v) in countsdict.items():
            counts += v
        return binCenters, counts

    def hists(self, binEdges, attr, states=None, cutRecipeName=None, channums=None):
        binEdges = np.array(binEdges)
        binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
        countsdict = collections.OrderedDict()
        if channums is None:
            channums = self.keys()  # this should exclud bad channels by default
        for channum in channums:
            _, countsdict[channum] = self[channum].hist(binEdges, attr, states, cutRecipeName)
        return binCenters, countsdict

    @property
    def whyChanBad(self):
        with self.includeBad():
            w = collections.OrderedDict()
            for (channum, ds) in self.items():
                if ds.markedBadBool:
                    w[channum] = ds.markedBadReason
            return w

    def plotHists(self, binEdges, attr, axis=None, labelLines=[], states=None,
                  cutRecipeName=None, maxChans=8, channums=None):
        if channums is None:
            channums = list(self.keys())[:min(maxChans, len(self))]
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels
        for channum in channums:
            ds = self[channum]
            ds.plotHist(binEdges, attr, axis, [], states, cutRecipeName)
            line = axis.lines[-1]
            line.set_label("{}".format(channum))
            if ds.markedBadBool:
                line.set_dashes([2, 2, 10, 2])
        axis.set_title(self.shortName + ", states = {}".format(states))
        axis.legend(title="channel")
        annotate_lines(axis, labelLines)

    def __iter__(self):
        if self._includeBad:
            return (channum for channum in collections.OrderedDict.__iter__(self))
        else:
            return (channum for channum in collections.OrderedDict.__iter__(self) if not self[channum].markedBadBool)

    def keys(self):
        return [channum for channum in self]

    def values(self):
        return [self[channum] for channum in self.keys()]

    def items(self):
        return [(channum, self[channum]) for channum in self.keys()]

    def __len__(self):
        return len([k for k in self])

    def includeBad(self, x=True):
        """
        Use this to do iteration including bad channels temporarily, eg:

        with data.includeBad():
            for (channum, ds) in data.items():
                print(ds)
        """
        self._includeBadDesired = x
        return self

    def __enter__(self):
        self._includeBad = self._includeBadDesired

    def __exit__(self, *args):
        self._includeBad = False
        self._includeBadDesired = False

    def histsToHDF5(self, h5File, binEdges, attr="energy", cutRecipeName=None):
        for (channum, ds) in self.items():
            ds.histsToHDF5(h5File, binEdges, attr, cutRecipeName)
        grp = h5File.require_group("all_channels")
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, cutRecipeName)
            grp["{}/bin_centers".format(state)] = binCenters
            grp["{}/counts".format(state)] = counts
        binCenters, counts = self.hist(
            binEdges, attr, cutRecipeName=cutRecipeName)  # all states hist
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    def markAllGood(self):
        with self.includeBad():
            for (channum, ds) in self.items():
                ds.markGood()

    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None, worstAllowedFWHM=None, positionToleranceAbsolute=None,
                            attr='energy', states=None, dlo=50, dhi=50, binsize=None, binEdges=None,
                            guessParams=None, cutRecipeName=None, holdvals=None, resolutionPlot=True, hdf5Group=None,
                            _rethrow=False):
        """
        Here we are overwriting the qualityCheckLinefit method created by GroupLooper
        the call to _qualityCheckLinefit uses the method created by GroupLooper
        """
        results = self._qualityCheckLinefit(line, positionToleranceFitSigma, worstAllowedFWHM, positionToleranceAbsolute,
                                            attr, states, dlo, dhi, binsize, binEdges, guessParams, cutRecipeName, holdvals,
                                            _rethrow=_rethrow)
        resolutions = np.array([r.params["fwhm"].value
                                for r in results.values() if r.success])
        if resolutionPlot:
            plt.figure()
            axis = plt.gca()
            axis.hist(resolutions, bins=np.arange(0, np.amax(resolutions)+0.25, 0.25))
            axis.set_xlabel("energy resoluiton fwhm (eV)")
            axis.set_ylabel("# of channels / 0.25 eV bin")
            plt.title(self.shortName+" at {}".format(line))
        if hdf5Group is not None:
            with self.includeBad():
                for (channum, ds) in self.items():
                    grp = hdf5Group.require_group("{}/fits/{}".format(channum, line))
                    if ds.markedBadBool:
                        grp["markedBadReason"] = ds.markedBadReason
                    else:
                        result = results[channum]
                        for (k, v) in result.params.items():
                            grp[k] = v.value
                            grp[k+"_err"] = v.stderr
                        grp["states"] = str(states)
        return results

    def setOutputDir(self, baseDir=None, deleteAndRecreate=None, suffix="_output"):
        """Set the output directory to which plots and hdf5 files will go
        baseDir -- the directory in which the output directory will exist
        deleteAndRecreate (required keyword arg) -- if True, will delete the whole directory if it already exists
                (good for if you re-run the same script alot)
                if False, will attempt to create the directory,
                if it already exists (like if you rerun the same script), it will error
        suffix -- added to the first part of shortName to create the output directory name

        commonly called as
        data.setOutputDir(baseDir=os.getcwd(), deleteAndRecreate=True)
        or
        data.setOutputDir(baseDir=os.path.realpath(__file__), deleteAndRecreate=True)
        """
        self._baseDir = baseDir
        dirname = self.shortName.split(" ")[0]+suffix
        self._outputDir = os.path.join(self._baseDir, dirname)
        if deleteAndRecreate is None:
            raise Exception(
                "deleteAndRecreate should be True or False, you can't use the default value")
        if deleteAndRecreate:
            if self.verbose:
                print("deleting and recreating directory {}".format(self.outputDir))
            try:
                shutil.rmtree(self.outputDir)
            except Exception:
                pass
        os.mkdir(self.outputDir)

    @property
    def outputDir(self):
        if hasattr(self, "_outputDir"):
            return self._outputDir
        else:
            raise Exception("call setOutputDir first")

    @property
    def outputHDF5(self):
        if not hasattr(self, "_outputHDF5Filename"):
            filename = os.path.join(self.outputDir, self.shortName.split(" ")[0]+".hdf5")
            self._outputHDF5Filename = filename
            return h5py.File(self._outputHDF5Filename, "w")
        else:
            return h5py.File(self._outputHDF5Filename, "a")

    def resultPlot(self, lineName, states=None, binsize=None):
        results = [ds.linefit(lineName, plot=False, states=states, binsize=binsize)
                   for ds in self.values()]
        result = self.linefit(lineName, plot=False, states=states, binsize=binsize)
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle("{} fits to {} with states = {}".format(self.shortName, lineName, states))
        result.plotm(ax=plt.subplot(2, 2, 3))
        plt.xlabel("energy (eV)")
        plt.ylabel("counts per bin")
        resolutions = [r.params["fwhm"].value for r in results]
        positions = [r.params["peak_ph"].value for r in results]
        position_errs = [r.params["peak_ph"].stderr for r in results]
        ax = plt.subplot(2, 2, 1)
        plt.hist(resolutions)
        plt.xlabel("resolution (eV fwhm)")
        plt.ylabel("channels per bin")
        plt.text(0.5, 0.9, "median = {:.2f}".format(np.median(resolutions)), transform=ax.transAxes)
        plt.vlines(np.median(resolutions), plt.ylim()[0], plt.ylim()[1], label="median")
        ax = plt.subplot(2, 2, 2)
        plt.hist(positions)
        plt.xlabel("fit position (eV)")
        plt.ylabel("channels per bin")
        plt.text(0.5, 0.9, "median = {:.2f}\ndb position = {:.3f}".format(np.median(positions),
                                                                          result.model.spect.peak_energy), transform=ax.transAxes)
        plt.vlines(result.model.spect.peak_energy, plt.ylim()
                   [0], plt.ylim()[1], label="db position")
        ax = plt.subplot(2, 2, 4)
        plt.errorbar(np.arange(len(positions)), positions, yerr=position_errs, fmt=".")
        plt.hlines(result.model.spect.peak_energy, plt.xlim()
                   [0], plt.xlim()[1], label="db position")
        plt.legend()
        plt.xlabel("channel number")
        plt.ylabel("line position (eV)")

    def setDefaultBinsize(self, binsize):
        """sets the default binsize in eV used by self.linefit and functions that call it,
        also sets it for all children"""
        self._default_bin_size = binsize
        with self.includeBad():
            for ds in self.values():
                ds.setDefaultBinsize(binsize)

    def cutSetDefault(self, cutRecipeName):
        assert cutRecipeName.startswith("cut")
        assert cutRecipeName in self.recipes.keys()
        self._default_cut_recipe_name = cutRecipeName
        for ds in self.values():
            ds.cutSetDefault(cutRecipeName)
