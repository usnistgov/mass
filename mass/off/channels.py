import mass
from .off import OffFile
import collections
import os

import numpy as np
import pylab as plt
import progress.bar
import inspect
import fastdtw
import h5py
import shutil
import logging
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
        self._preventAliasState = False # causes aliasState to raise an Exception when it wouldn't work as expected

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
        if len(statesDict.keys()) > 0 and len(inds) > 0:  # the state that was active last time calcStatesDict was called may need special handling
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


def annotate_lines(axis, labelLines, labelLines_color2=[], color1="k", color2="r"):
    """Annotate plot on axis with line names.
    labelLines -- eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    labelLines_color2 -- optional,eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    color1 -- text color for labelLines
    color2 -- text color for labelLines_color2
    """
    n = len(labelLines)+len(labelLines_color2)
    yscale = plt.gca().get_yscale()
    for (i, labelLine) in enumerate(labelLines):
        energy = mass.STANDARD_FEATURES[labelLine]
        if yscale == "linear":
            axis.annotate(labelLine, (energy, (1+i)*plt.ylim()
                                      [1]/float(1.5*n)), xycoords="data", color=color1)
        elif yscale == "log":
            axis.annotate(labelLine, (energy, np.exp(
                (1+i)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data", color=color1)
    for (j, labelLine) in enumerate(labelLines_color2):
        energy = mass.STANDARD_FEATURES[labelLine]
        if yscale == "linear":
            axis.annotate(labelLine, (energy, (2+i+j)*plt.ylim()
                                      [1]/float(1.5*n)), xycoords="data", color=color2)
        elif yscale == "log":
            axis.annotate(labelLine, (energy, np.exp(
                (2+i+j)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data", color=color2)


class DriftCorrection():
    version = 1

    def __init__(self, indicatorName, uncorrectedName, medianIndicator, slope):
        self.indicatorName = indicatorName
        self.uncorrectedName = uncorrectedName
        self.medianIndicator = medianIndicator
        self.slope = slope

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


class GroupLooper(object):
    """A mixin class to allow ChannelGroup objects to hold methods that loop over
    their constituent channels. (Has to be a mixin, in order to break the import
    cycle that would otherwise occur.)"""
    pass


def add_group_loop(method):
    """Add MicrocalDataSet method `method` to GroupLooper (and hence, to TESGroup).

    This is a decorator to add before method definitions inside class MicrocalDataSet.
    Usage is:

    class MicrocalDataSet(...):
        ...

        @add_group_loop
        def awesome_fuction(self, ...):
            ...
    """
    method_name = method.__name__

    def wrapper(self, *args, **kwargs):
        bar = SilenceBar(method_name, max=len(self.offFileNames), silence=not self.verbose)
        rethrow = kwargs.pop("_rethrow", False)
        returnVals = collections.OrderedDict()
        for (channum, ds) in self.items():
            try:
                z = method(ds, *args, **kwargs)
                returnVals[channum] = z
            except KeyboardInterrupt as e:
                raise(e)
            except Exception as e:
                ds.markBad("{} during {}".format(e, method_name), e)
                if rethrow:
                    raise
            bar.next()
        bar.finish()
        return returnVals
    wrapper.__name__ = method_name

    # Generate a good doc-string.
    lines = ["Loop over self, calling the %s(...) method for each channel." % method_name]
    lines.append("pass _rethrow=True to see stacktrace from first error")
    try:
        argtext = inspect.signature(method)  # Python 3.3 and later
    except AttributeError:
        arginfo = inspect.getargspec(method)
        argtext = inspect.formatargspec(*arginfo)
    if method.__doc__ is None:
        lines.append("\n%s%s has no docstring" % (method_name, argtext))
    else:
        lines.append("\n%s%s docstring reads:" % (method_name, argtext))
        lines.append(method.__doc__)
    wrapper.__doc__ = "\n".join(lines)

    setattr(GroupLooper, method_name, wrapper)
    setattr(GroupLooper, "_"+method_name, wrapper)
    return method


class CorG():
    """
    implments methods that are shared across Channel and ChannelGroup
    """
    @property
    def stateLabels(self):
        return self.experimentStateFile.labels

    def plotHist(self, binEdges, attr, axis=None, labelLines=[], states=None, goodFunc=None, coAddStates=True):
        """plot a coadded histogram from all good datasets and all good pulses
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        axis -- if None, then create a new figure, otherwise plot onto this axis
        annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
        goodFunc -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer    """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels
        if coAddStates:
            x, y = self.hist(binEdges, attr, states=states, goodFunc=goodFunc)
            axis.plot(x, y, drawstyle="steps-mid", label=states)
        else:
            for state in states:
                x, y = self.hist(binEdges, attr, states=state, goodFunc=goodFunc)
                axis.plot(x, y, drawstyle="steps-mid", label=state)
        axis.set_xlabel(attr)
        axis.set_ylabel("counts per %0.1f unit bin" % (binEdges[1]-binEdges[0]))
        plt.legend(title="states")
        axis.set_title(self.shortName)
        annotate_lines(axis, labelLines)
        return axis

    def linefit(self, lineNameOrEnergy="MnKAlpha", attr="energy", states=None, axis=None, dlo=50, dhi=50,
                binsize=1, binEdges=None, label="full", plot=True,
                guessParams=None, goodFunc=None, holdvals=None):
        """Do a fit to `lineNameOrEnergy` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
        lineNameOrEnergy -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
        attr -- default is "energyRough". you must pass binEdges if attr is other than "energy" or "energyRough"
        states -- will be passed to hist, coAddStates will be True
        axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
        dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
        binEdges -- pass the binEdges you want as a numpy array
        label -- passed to fitter.plot
        plot -- passed to fitter.fit, determine if plot happens
        guessParams -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
        category -- pass {"side":"A"} or similar to use categorical cuts
        goodFunc -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
            This vector is anded with the vector calculated by the histogrammer
        """
        if isinstance(lineNameOrEnergy, mass.LineFitter):
            fitter = lineNameOrEnergy
            nominal_peak_energy = fitter.spect.nominal_peak_energy
        elif isinstance(lineNameOrEnergy, str):
            fitter = mass.fitter_classes[lineNameOrEnergy]()
            nominal_peak_energy = fitter.spect.nominal_peak_energy
        else:
            fitter = mass.GaussianFitter()
            nominal_peak_energy = float(lineNameOrEnergy)
        if binEdges is None:
            if attr.startswith("energy"):
                binEdges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)
            else:
                raise Exception("must pass binEdges if attr does not start with energy")
        if axis is None and plot:
            plt.figure()
            axis = plt.gca()
        bin_centers, counts = self.hist(binEdges, attr, states, goodFunc)
        if guessParams is None:
            guessParams = fitter.guess_starting_params(counts, bin_centers)
        if holdvals is None:
            holdvals = {}
        if attr.startswith("energy") and "dP_dE" in fitter.param_meaning:
            holdvals["dP_dE"] = 1.0
        hold = []
        for (k, v) in holdvals.items():
            i = fitter.param_meaning[k]
            guessParams[i] = v
            hold.append(i)

        params, covar = fitter.fit(counts, bin_centers, params=guessParams,
                                   axis=axis, label=label, plot=plot, hold=hold)
        if plot:
            axis.set_title(self.shortName+", {}, states = {}".format(lineNameOrEnergy, states))
            if attr.startswith("energy"):
                plt.xlabel(attr+" (eV)")
            else:
                plt.xlabel(attr + "(arbs)")

        return fitter


class NoCutInds():
    pass


class Recipe():
    """
    If `r` is a Recipe, it is a wrapper around a function `f` and the names of its arguments.
    Arguments can either be names to be provided in a dictionary `d` when `r(d)` is called, or
    argument can be Recipe.
    `r.nonRecipeArgs` is a list of the names of all the argumets `r` takes as well as all the other recipes that `r` depends upon
    `r(d)` where d is a dict mappring the names of argument to values will call `f` with the appropriate arguments, and also
    evaulate arguments which are recipes.

    The reasons this exists is so I can get a list of all the argument I need from the off file, so I can read from the off file
    a single time to evaluate a recipe that may depend on many values from the off file. My previous implementation would make multiple
    reads to the off file.
    """

    def __init__(self, f, argNames=None):
        assert not isinstance(f, Recipe)
        self.f = f
        self.args = collections.OrderedDict()  # assumes the dict preserves insertion order
        try:
            inspectedArgNames = list(inspect.signature(self.f).parameters)  # Py 3.3+ only??
        except AttributeError:
            inspectedArgNames = inspect.getargspec(self.f).args  # Pre-Py 3.3
        if "self" in inspectedArgNames:  # drop the self argument for class methods
            inspectedArgNames.remove("self")
        if argNames is None:
            for argName in inspectedArgNames:
                self.args[argName] = argName
        else:
            # i would like to do == here, but i'd need to handle optional arguments better
            assert len(inspectedArgNames) >= len(argNames)
            for argName, inspectedArgName in zip(argNames, inspectedArgNames):
                self.args[argName] = inspectedArgName

    def setArg(self, argName, r):
        assert isinstance(r, Recipe)
        assert argName in self.args
        self.args[argName] = r

    @property
    def nonRecipeArgs(self):
        # collect a unique list of all requires nonRecipe args...
        # FUTURE?: possibly I should just always index into the off files and get all offArgs all the time?
        a = [] 
        for (k, v) in self.args.items():
            if isinstance(v, Recipe):
                for vv in v.nonRecipeArgs:
                    if vv not in a:
                        a.append(vv)
            elif k not in a:
                a.append(k)
        return a

    def __call__(self, args):
        new_args = []
        for (k, v) in self.args.items():
            if isinstance(v, Recipe):
                new_args.append(v(args))
            else:
                new_args.append(args[k])
        # call functions with positional arguments so names don't need to match
        return self.f(*new_args)

    def __repr__(self, indent=0):
        s = "Recipe: f={}, args=".format(self.f)
        s += "\n" + "  "*indent + "{\n"
        for (k, v) in self.args.items():
            if isinstance(v, Recipe):
                s += "{}{}: {}\n".format("  "*(indent+1), k, v.__repr__(indent+1))
            else:
                s += "{}{}: {}\n".format("  "*(indent+1), k, v)
        s += "  "*indent + "}"
        return s


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
        self.recipes = {}
        self._defineDefaultRecipesAndProperties()

    def _defineDefaultRecipesAndProperties(self):
        assert(len(self.recipes) == 0)
        t0 = self.offFile["unixnano"][0]
        self.addRecipe("relTimeSec", lambda unixnano: (unixnano-t0)*1e9, ["unixnano"])
        self.addRecipe("filtPhase", lambda x, y: x/y, ["derivativeLike", "filtValue"])

    @property
    def _offAttrs(self):
        return self.offFile.dtype.names

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
    def learnStdDevResThresholdUsingMedianAbsoluteDeviation(self, nSigma=7):
        median = np.median(self.residualStdDev)
        mad = np.median(np.abs(self.residualStdDev-median))
        k = 1.4826  # for gaussian distribution, ratio of sigma to median absolution deviation
        sigma = mad*k
        self.stdDevResThreshold = median+nSigma*sigma

    @add_group_loop
    def learnStdDevResThresholdUsingRatioToNoiseStd(self, ratioToNoiseStd=1.5):
        self.stdDevResThreshold = self.offFile.header["ModelInfo"]["NoiseStandardDeviation"]*ratioToNoiseStd

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
                raise Exception("v should be a list of slices or a slice, but is a {}".format(type(v)))
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

    def defaultGoodFunc(self, v):
        """v must be of self.offFile.dtype"""
        g = v["residualStdDev"] < self.stdDevResThreshold
        return g

    def _indexOffWithCuts(self, inds, goodFunc=None, returnBad=False, _listMethodSelect=2):
        """
        inds - a slice or list of slices to index into items with
        goodFunc - a function called on the data read from the off file, must return a vector of bool values
        returnBad - if true, np.logical_not the goodFunc output
        _listMethodSelect - used for debugging and testing, chooses the implmentation of this method used for lists of indicies
        _indexOffWithCuts(slice(0,10), f) is roughly equivalent to:
        g = f(offFile[0:10])
        offFile[0:10][g]
        """
        if goodFunc is None:
            goodFunc = self.defaultGoodFunc
        # offAttr can be a list of offAttr names
        if isinstance(inds, slice):
            r = self.offFile[inds]
            g = goodFunc(r)
            if returnBad:
                g = np.logical_not(g)
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
                tmp = self._indexOffWithCuts(s, goodFunc, returnBad)
                ilo = ihi
                ihi = ilo+len(tmp)
                output_prealloc[ilo:ihi] = tmp
            output = output_prealloc[0:ihi]
        elif isinstance(inds, list) and _listMethodSelect == 0:  # repeated hstack
            # this could be removed, along with the _listMethodSelect argument
            # this is only left in because it is useful for correctness testing for preallocate and truncate method since this is simpler
            assert all([isinstance(_inds, slice) for _inds in inds])
            output = self._indexOffWithCuts(inds[0], goodFunc, returnBad)
            for i in range(1, len(inds)):
                output = np.hstack((output, self._indexOffWithCuts(inds[i], goodFunc, returnBad)))
        elif isinstance(inds, NoCutInds):
            output = self.offFile
        else:
            raise Exception("type(inds)={}, should be slice or list or slices".format(type(inds)))
        return output

    def getAttr(self, attr, inds, goodFunc=None, returnBad=False):
        offAttrValues = self._indexOffWithCuts(inds, goodFunc, returnBad) # single read from disk, read all values
        if isinstance(attr, list):
            return [self._getAttr(a, offAttrValues) for a in attr]
        else:
            return self._getAttr(attr, offAttrValues)

    def _getAttr(self, attr, offAttrValues):
        if self.isRecipeAttr(attr):
            recipe = self.recipes[attr]
            return recipe(offAttrValues)
        elif self.isOffAttr(attr):
            return offAttrValues[attr]
        else:
            raise Exception("attr {} must be an OffAttr or a RecipeAttr or a list. OffAttrs: {}\nRecipeAttrs: {}".format(attr, list(self._offAttrs), list(self._recipeAttrs)))

    def plotAvsB(self, nameA, nameB, axis=None, states=None, includeBad=False, goodFunc=None):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels

        def getAB(inds, goodFunc, returnBad):
            A = self.getAttr(nameA, inds, goodFunc, returnBad)
            B = self.getAttr(nameB, inds, goodFunc, returnBad)
            return A, B

        for state in states:
            inds = self.getStatesIndicies(state)
            A, B = getAB(inds, goodFunc, returnBad=False)
            axis.plot(A, B, ".", label=state)
            if includeBad:
                A, B = getAB(inds, goodFunc, returnBad=True)
                axis.plot(A, B, "x", label=state+" bad")
        plt.xlabel(nameA)
        plt.ylabel(nameB)
        plt.title(self.shortName)
        plt.legend(title="states")
        return axis

    def hist(self, binEdges, attr, states=None, goodFunc=None, returnBad=False):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
        goodFunc -- a function taking a 1d array of vales of type self.offFile.dtype and returning a vector of bool
         """
        binEdges = np.array(binEdges)
        binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
        inds = self.getStatesIndicies(states)
        vals = self.getAttr(attr, inds, goodFunc, returnBad)
        counts, _ = np.histogram(vals, binEdges)
        return binCenters, counts

    @add_group_loop
    def learnDriftCorrection(self, indicatorName="pretriggerMean", uncorrectedName="filtValue", correctedName = "filtValueDC", states=None, goodFunc=None, returnBad=False):
        inds = self.getStatesIndicies(states)
        indicator, uncorrected = self.getAttr([indicatorName, uncorrectedName], inds, goodFunc, returnBad)
        slope, info = mass.core.analysis_algorithms.drift_correct(
            indicator, uncorrected)
        self.driftCorrection = DriftCorrection(
            indicatorName, uncorrectedName, info["median_pretrig_mean"], slope)
        self.addRecipe(correctedName, self.driftCorrection.apply, [
                       self.driftCorrection.indicatorName, self.driftCorrection.uncorrectedName])
        return self.driftCorrection

    @add_group_loop
    def learnPhaseCorrection(self, indicatorName="filtPhase", uncorrectedName="filtValue", correctedName = "filtValuePC", states=None, 
    linePositionsFunc=None, goodFunc=None, returnBad=False):
        """
        linePositionsFunc - if None, then use self.calibrationRough._ph as the peak locations
        otherwise try to call it with self as an argument... here is an example of how you could use all but one peak from calibrationRough:
        `data.learnPhaseCorrection(linePositionsFunc = lambda ds: dsl.calibrationRough._ph[1:]`
        """
        # may need to generalize this to allow using a specific state for phase correction as a specfic line... with something like calibrationPlan
        if linePositionsFunc is None:
            linePositions = self.calibrationRough._ph
        else:
            linePositions = linePositionsFunc(self)
        inds = self.getStatesIndicies(states)
        indicator, uncorrected = self.getAttr([indicatorName, uncorrectedName], inds, goodFunc, returnBad)
        self.phaseCorrection = mass.core.phase_correct.phase_correct(
            indicator, uncorrected, linePositions, indicatorName=indicatorName, uncorrectedName=uncorrectedName)
        self.addRecipe(correctedName, self.phaseCorrection.correct, [
                       self.phaseCorrection.indicatorName, self.phaseCorrection.uncorrectedName])

    def loadDriftCorrection(self):
        raise Exception("not implemented")

    def hasDriftCorrection(self):
        return hasattr(self, "driftCorrection")

    def plotCompareDriftCorrect(self, axis=None, states=None, goodFunc=None, includeBad=False):
        if axis is None:
            plt.figure()
            axis = plt.gca()

        if states is None:
            states = self.stateLabels
        for state in states:
            inds = self.getStatesIndicies(state)
            A = self.getAttr(self.driftCorrection.indicatorName, inds, goodFunc)
            B = self.getAttr(self.driftCorrection.uncorrectedName, inds, goodFunc)
            C = self.getAttr("filtValueDC", inds, goodFunc)
            axis.plot(A, B, ".", label=state)
            axis.plot(A, C, ".", label=state+" DC")
            if includeBad:
                A = self.getAttr(self.driftCorrection.indicatorName, inds, goodFunc, returnBad=True)
                B = self.getAttr(self.driftCorrection.uncorrectedName,
                                 inds, goodFunc, returnBad=True)
                C = self.getAttr("filtValueDC", inds, goodFunc, returnBad=True)
                axis.plot(A, B, "x", label=state+" bad")
                axis.plot(A, C, "x", label=state+" bad DC")
        plt.xlabel(self.driftCorrection.indicatorName)
        plt.ylabel(self.driftCorrection.uncorrectedName + ",filtValueDC")
        plt.title(self.shortName+" drift correct comparison")
        plt.legend(title="states")
        return axis

    def calibrationPlanInit(self, attr):
        self.calibrationPlan = CalibrationPlan()
        self.calibrationPlanAttr = attr

    def calibrationPlanAddPoint(self, uncalibratedVal, name, states=None, energy=None):
        self.calibrationPlan.addCalPoint(uncalibratedVal, name, states, energy)
        self.calibrationRough = self.calibrationPlan.getRoughCalibration()
        self.calibrationRough.uncalibratedName = self.calibrationPlanAttr
        self.addRecipe("energyRough", self.calibrationRough.ph2energy,
                       [self.calibrationRough.uncalibratedName])
        return self.calibrationPlan

    @add_group_loop
    def calibrateFollowingPlan(self, attr, curvetype="gain", approximate=True, dlo=50, dhi=50, binsize=1):
        self.calibration = mass.EnergyCalibration(curvetype=curvetype, approximate=approximate)
        self.calibration.uncalibratedName = attr
        fitters = []
        for (ph, energy, name, states) in zip(self.calibrationPlan.uncalibratedVals, self.calibrationPlan.energies,
                                              self.calibrationPlan.names, self.calibrationPlan.states):
            if name in mass.fitter_classes:
                fitter = self.linefit(name, "energyRough", states, dlo=dlo, dhi=dhi,
                                      plot=False, binsize=binsize)
            else:
                fitter = self.linefit(energy, "energyRough", states, dlo=dlo, dhi=dhi,
                                      plot=False, binsize=binsize)
            fitters.append(fitter)
            if not fitter.fit_success:
                self.markBad("calibrateFollowingPlan: failed fit {}, states {}".format(
                    name, states), extraInfo=fitter)
                continue
            phRefined = self.calibrationRough.energy2ph(fitter.last_fit_params_dict["peak_ph"][0])
            self.calibration.add_cal_point(phRefined, energy, name)
        self.fittersFromCalibrateFollowingPlan = fitters
        self.addRecipe("energy", self.calibration.ph2energy, [self.calibration.uncalibratedName])
        return fitters

    def addRecipe(self, recipeName, f, argNames, createProperty=True):
        """
        recipeName - the name of the new Attr to create, eg "energy"
        f - the function used to caluclate the Attr
        argNames - a list of argument names, they can be OffAttrs or other recipes
        createProperty - if True will create a property such that you can access the output of the recipe as eg `ds.energy`
        """
        # add a recipe
        # 1. create the recipe
        # 2. call setArg to point at any existing recipes for argument
        # 3. add to dict with key recipeName
        assert isinstance(argNames, list)
        recipe = Recipe(f, argNames)
        for argName in argNames:
            if argName in self.recipes:
                recipe.setArg(argName, self.recipes[argName])
            elif not self.isOffAttr(argName):
                raise Exception(
                    "argName={} should be in self.recipes or be an OffAttr".format(argName))
        self.recipes[recipeName] = recipe
        # 4. create a property to access the recipe
        # recipes are added to the class, so only do it once per recipeName
        if createProperty and not hasattr(Channel, recipeName):
            setattr(Channel, recipeName, property(
                lambda argself: argself.getAttr(recipeName, NoCutInds())))

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

    def plotResidualStdDev(self, axis=None):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        x = np.sort(self.residualStdDev)/self.offFile.header["ModelInfo"]["NoiseStandardDeviation"]
        y = np.linspace(0, 1, len(x))
        inds = x > (self.stdDevResThreshold
                    / self.offFile.header["ModelInfo"]["NoiseStandardDeviation"])
        axis.plot(x, y, label="<threshold")
        axis.plot(x[inds], y[inds], "r", label=">threshold")
        axis.vlines(self.stdDevResThreshold
                    / self.offFile.header["ModelInfo"]["NoiseStandardDeviation"], 0, 1)
        axis.set_xlabel("residualStdDev/noiseStdDev")
        axis.set_ylabel("fraction of pulses with equal or lower residualStdDev")
        axis.set_title("{}, {} total pulses, {:0.3f} cut".format(
            self.shortName, len(self), float(inds.sum()/float(len(self)))))  # somehow the extra float() is required on windows?
        axis.legend()
        axis.set_xlim(max(0, x[0]), 3)
        axis.set_ylim(0, 1)

    def __len__(self):
        return len(self.offFile)

    @add_group_loop
    def alignToReferenceChannel(self, referenceChannel, attr, binEdges, _peakLocs=None, states=None):
        if _peakLocs is None:
            assert(len(referenceChannel.calibrationPlan.uncalibratedVals) > 0)
            peakLocs = referenceChannel.calibrationPlan.uncalibratedVals
        else:
            peakLocs = _peakLocs
        self.aligner = AlignBToA(ds_a=referenceChannel, ds_b=self,
                                 peak_xs_a=peakLocs, bin_edges=binEdges, attr=attr, states=states)
        self.calibrationArbsInRefChannelUnits = self.aligner.getCalBtoA()
        if _peakLocs is None and not (self is referenceChannel):
            self.calibrationPlanInit(referenceChannel.calibrationPlanAttr)
            refCalPlan = referenceChannel.calibrationPlan
            for (ph, energy, name, states) in zip(refCalPlan.uncalibratedVals, refCalPlan.energies,
                                                  refCalPlan.names, refCalPlan.states):
                self.calibrationPlanAddPoint(self.calibrationArbsInRefChannelUnits.energy2ph(ph),
                                             name, states, energy)
        self.addRecipe("arbsInRefChannelUnits", self.calibrationArbsInRefChannelUnits.ph2energy, [
                       self.calibrationArbsInRefChannelUnits.uncalibratedName])
        return self.aligner

    @add_group_loop
    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None, worstAllowedFWHM=None,
                            positionToleranceAbsolute=None, attr="energy", states=None,
                            dlo=50, dhi=50, binsize=1, binEdges=None, guessParams=None,
                            goodFunc=None, holdvals=None):
        """calls ds.linefit to fit the given line
        marks self bad if the fit position is more than toleranceFitSigma*fitSigma away
        from the correct position
        """
        fitter = self.linefit(line, attr, states, None, dlo, dhi, binsize, binEdges, False,
                              guessParams, goodFunc, holdvals)
        fitPos, fitSigma = fitter.last_fit_params_dict["peak_ph"]
        resolution, _ = fitter.last_fit_params_dict["resolution"]
        if positionToleranceAbsolute is not None:
            if positionToleranceFitSigma is not None:
                raise Exception(
                    "specify only one of positionToleranceAbsolute or positionToleranceFitSigma")
            tolerance = positionToleranceAbsolute
        elif positionToleranceFitSigma is not None:
            tolerance = fitSigma*positionToleranceFitSigma
        else:
            tolerance = np.inf
        if np.abs(fitPos-fitter.spect.peak_energy) > tolerance:
            self.markBad("qualityCheckLinefit: for {}, want {} within {}, got {}".format(
                line, fitter.spect.peak_energy, tolerance, fitPos))
        if worstAllowedFWHM is not None and resolution > worstAllowedFWHM:
            self.markBad("qualityCheckLinefit: for {}, fit resolution {} > threshold {}".format(
                line, resolution, worstAllowedFWHM))
        return fitter

    def histsToHDF5(self, h5File, binEdges, attr="energy", goodFunc=None):
        grp = h5File.require_group(str(self.channum))
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, goodFunc)
            grp["{}/bin_centers".format(state)] = binCenters
            grp["{}/counts".format(state)] = counts
        binCenters, counts = self.hist(binEdges, attr, goodFunc=goodFunc)  # all states hist
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    @add_group_loop
    def recipeToHDF5(self, h5File):
        grp = h5File.require_group(str(self.channum))
        if hasattr(self, "driftCorrection"):
            self.driftCorrection.toHDF5(grp)
        if hasattr(self, "calibration"):
            self.calibration.save_to_hdf5(grp, "calibration")
            grp["calibration/uncalibratedName"] = self.calibration.uncalibratedName
        if hasattr(self, "calibrationRough"):
            self.calibrationRough.save_to_hdf5(grp, "calibrationRough")
            grp["calibrationRough/uncalibratedName"] = self.calibrationRough.uncalibratedName
        if hasattr(self, "calibrationArbsInRefChannelUnits"):
            self.calibrationArbsInRefChannelUnits.save_to_hdf5(
                grp, "calibrationArbsInRefChannelUnits")
            grp["calibrationArbsInRefChannelUnits/uncalibratedName"] = self.calibrationArbsInRefChannelUnits.uncalibratedName
        if hasattr(self, "phaseCorrection"):
            self.phaseCorrection.toHDF5(grp)

    def recipeFromHDF5(self, h5File):
        grp = h5File.require_group(str(self.channum))
        if "driftCorrection" in grp:
            self.driftCorrection = DriftCorrection.fromHDF5(grp)
            self.addRecipe("filtValueDC", self.driftCorrection.apply, [
                           self.driftCorrection.indicatorName, self.driftCorrection.uncorrectedName])
        if "calibration" in grp:
            self.calibration = mass.EnergyCalibration.load_from_hdf5(grp, "calibration")
            self.calibration.uncalibratedName = grp["calibration/uncalibratedName"][()]
            self.addRecipe("energy", self.calibration.ph2energy,
                           [self.calibration.uncalibratedName])
        if "calibrationRough" in grp:
            self.calibrationRough = mass.EnergyCalibration.load_from_hdf5(grp, "calibrationRough")
            self.calibrationRough.uncalibratedName = grp["calibrationRough/uncalibratedName"][()]
            self.addRecipe("energyRough", self.calibrationRough.ph2energy,
                           [self.calibrationRough.uncalibratedName])
        if "calibrationArbsInRefChannelUnits" in grp:
            self.calibrationArbsInRefChannelUnits = mass.EnergyCalibration.load_from_hdf5(
                grp, "calibrationArbsInRefChannelUnits")
            self.calibrationArbsInRefChannelUnits.uncalibratedName = grp[
                "calibrationArbsInRefChannelUnits/uncalibratedName"][()]
            self.addRecipe("arbsInRefChannelUnits", self.calibrationArbsInRefChannelUnits.ph2energy, [
                self.calibrationArbsInRefChannelUnits.uncalibratedName])
        if "phase_correction" in grp:
            self.phaseCorrection = mass.core.phase_correct.PhaseCorrector.fromHDF5(grp)
            self.addRecipe("filtValuePC", self.phaseCorrection.correct, [
                           self.phaseCorrection.indicatorName, self.phaseCorrection.uncorrectedName])

    @add_group_loop
    def energyTimestampLabelToHDF5(self, h5File, goodFunc=None, returnBad=False):
        grp = h5File.require_group(str(self.channum))
        if len(self.stateLabels) > 0:
            for state in self.stateLabels:
                inds = self.getStatesIndicies(state)
                energy = self.getAttr("energy", inds, goodFunc, returnBad)
                unixnano = self.getAttr("unixnano", inds, goodFunc, returnBad)
                grp["{}/energy".format(state)] = energy
                grp["{}/unixnano".format(state)] = unixnano
        else:
            energy = self.getAttr("energy", slice(None), goodFunc, returnBad)
            unixnano = self.getAttr("unixnano", slice(None), goodFunc, returnBad)
            grp["{}/energy".format(state)] = energy
            grp["{}/unixnano".format(state)] = unixnano

    @add_group_loop
    def qualityCheckDropOneErrors(self, thresholdAbsolute=None, thresholdSigmaFromMedianAbsoluteValue=None):
        energies, errors = self.calibration.drop_one_errors()
        maxAbsError = np.amax(np.abs(errors))
        medianAbsoluteValue = np.median(np.abs(errors))
        k = 1.4826  # https://en.wikipedia.org/wiki/Median_absolute_deviation
        sigma = k*medianAbsoluteValue
        if thresholdAbsolute is not None:
            if maxAbsError > sigma*thresholdSigmaFromMedianAbsoluteValue:
                self.markBad("qualityCheckDropOneErrors: maximum absolute drop one error {} > theshold {} (thresholdSigmaFromMedianAbsoluteValue)".format(
                    maxAbsError, sigma*thresholdSigmaFromMedianAbsoluteValue))
        if thresholdAbsolute is not None:
            if maxAbsError > thresholdAbsolute:
                self.markBad("qualityCheckDropOneErrors: maximum absolute drop one error {} > theshold {} (thresholdAbsolute)".format(
                    maxAbsError, thresholdAbsolute))

    def diagnoseCalibration(self):
        plt.figure(figsize=(20, 12))
        plt.suptitle(self.shortName)
        n = int(np.ceil(np.sqrt(len(self.fittersFromCalibrateFollowingPlan)+2)))
        for i, fitter in enumerate(self.fittersFromCalibrateFollowingPlan):
            ax = plt.subplot(n, n, i+1)
            fitter.plot(axis=ax, label="full")
            if isinstance(fitter, mass.GaussianFitter):
                plt.title("GaussianFitter (energy?)")
            else:
                plt.title(type(fitter.spect).__name__)
        ax = plt.subplot(n, n, i+2)
        self.calibration.plot(axis=ax)
        ax = plt.subplot(n, n, i+3)
        self.plotHist(np.arange(0, 16000, 4), self.calibration.uncalibratedName,
                      axis=ax, coAddStates=False)
        plt.vlines(self.calibrationPlan.uncalibratedVals, 0, plt.ylim()[1])


class AlignBToA():
    cm = plt.cm.gist_ncar

    def __init__(self, ds_a, ds_b, peak_xs_a, bin_edges, attr, states=None,
                 scale_by_median=True, normalize_before_dtw=True):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.bin_edges = bin_edges
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        self.peak_xs_a = peak_xs_a
        self.attr = attr
        self.scale_by_median = scale_by_median
        self.normalize_before_dtw = normalize_before_dtw
        self.states = states
        self.peak_inds_b = self.samePeaks()

    def samePeaks(self, goodFunc_a=None, goodFunc_b=None, returnBad=False):
        ph_a = self.ds_a.getAttr(self.attr, slice(None), goodFunc_a, returnBad)
        ph_b = self.ds_b.getAttr(self.attr, slice(None), goodFunc_b, returnBad)
        if self.scale_by_median:
            median_ratio_a_over_b = np.median(ph_a)/np.median(ph_b)
        else:
            median_ratio_a_over_b = 1.0
        ph_b_median_scaled = ph_b*median_ratio_a_over_b
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b_median_scaled, _ = np.histogram(ph_b_median_scaled, self.bin_edges)
        self.peak_inds_a = self.findPeakIndsA(counts_a)
        if self.normalize_before_dtw:
            distance, path = fastdtw.fastdtw(self.normalize(
                counts_a), self.normalize(counts_b_median_scaled))
        else:
            distance, path = fastdtw.fastdtw(counts_a, counts_b_median_scaled)
        i_a = [x[0] for x in path]
        i_b_median_scaled = [x[1] for x in path]
        peak_inds_b_median_scaled = np.array(
            [i_b_median_scaled[i_a.index(pia)] for pia in self.peak_inds_a])
        peak_xs_b_median_scaled = self.bin_edges[peak_inds_b_median_scaled]
        peak_xs_b = peak_xs_b_median_scaled/median_ratio_a_over_b
        min_bin = self.bin_edges[0]
        bin_spacing = self.bin_edges[1]-self.bin_edges[0]
        peak_inds_b = map(int, (peak_xs_b-min_bin)/bin_spacing)
        return peak_inds_b

    def findPeakIndsA(self, counts_a):
        peak_inds_a = np.searchsorted(self.bin_edges, self.peak_xs_a)-1
        return peak_inds_a

    def samePeaksPlot(self, goodFunc_a=None, goodFunc_b=None, returnBad=False):
        ph_a = self.ds_a.getAttr(self.attr, slice(None), goodFunc_a, returnBad)
        ph_b = self.ds_b.getAttr(self.attr, slice(None), goodFunc_b, returnBad)
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

    def samePeaksPlotWithAlignmentCal(self, goodFunc_a=None, goodFunc_b=None, returnBad=False):
        inds_a = self.ds_a.getStatesIndicies(self.states)
        ph_a = self.ds_a.getAttr(self.attr, inds_a, goodFunc_a, returnBad)
        # inds_b = self.ds_b.getStatesIndicies(self.states)
        ph_b = self.ds_b.getAttr("arbsInRefChannelUnits", slice(None), goodFunc_b, returnBad)
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
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

    def normalize(self, x):
        return x/float(np.sum(x))

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

    def _laplaceEntropy(self, w=None, goodFunc_a=None, goodFunc_b=None, returnBad=False):
        if w is None:
            w = self.bin_edges[1]-self.bin_edges[0]
        ph_a = self.ds_a.getAttr(self.attr, slice(None), goodFunc_a, returnBad)
        ph_b = self.ds_b.getAttr(self.newattr, slice(None), goodFunc_b, returnBad)
        entropy = mass.entropy.laplace_cross_entropy(ph_a[ph_a > self.bin_edges[0]],
                                                     ph_b[ph_b > self.bin_edges[0]], w=w)
        return entropy

    def _ksStatistic(self, goodFunc_a=None, goodFunc_b=None, returnBad=False):
        ph_a = self.ds_a.getAttr(self.attr, slice(None), goodFunc_a, returnBad)
        ph_b = self.ds_b.getAttr(self.newattr, slice(None), goodFunc_b, returnBad)
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
        self.energies = np.zeros(0)
        self.states = []
        self.names = []

    def addCalPoint(self, uncalibratedVal,  name, states=None, energy=None):
        _energy = None
        if name in mass.spectrum_classes:
            _energy = mass.spectrum_classes[name]().peak_energy
        elif name in mass.STANDARD_FEATURES:
            _energy = mass.STANDARD_FEATURES[name]
        if _energy is not None:
            if (energy is not None) and (energy != _energy):
                raise(Exception("found energy={} from {}, do not pass a value to energy".format(_energy, name)))
            energy = _energy
        if energy is None:
            raise(Exception(
                "name {} not found in mass.spectrum_classes or mass.STANDARD_FEATURES, pass energy".format(name)))
        self.uncalibratedVals = np.hstack((self.uncalibratedVals, uncalibratedVal))
        self.names.append(name)
        self.energies = np.hstack((self.energies, energy))
        self.states.append(states)

    def __repr__(self):
        s = """CalibrationPlan with {} entries
        x: {}
        y: {}
        states: {}
        names: {}""".format(len(self.names), self.uncalibratedVals, self.energies, self.states, self.names)
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


class SilenceBar(progress.bar.Bar):
    "A progres bar that can be turned off by passing silence=True or by setting the log level higher than NOTSET"

    def __init__(self, message, max, silence):
        self.silence = silence
        if not silence:
            if not LOG.isEnabledFor(logging.WARN):
                self.silence = True
        if not self.silence:
            progress.bar.Bar.__init__(self, message, max=max)

    def next(self):
        if not self.silence:
            progress.bar.Bar.next(self)

    def finish(self):
        if not self.silence:
            progress.bar.Bar.finish(self)


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

    def hist(self, binEdges, attr, states=None, goodFunc=None, returnBad=False):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
         """
        binCenters, countsdict = self.hists(
            binEdges, attr, states, goodFunc=goodFunc, returnBad=returnBad)
        counts = np.zeros_like(binCenters, dtype="int")
        for (k, v) in countsdict.items():
            counts += v
        return binCenters, counts

    def hists(self, binEdges, attr, states=None, goodFunc=None, returnBad=False, channums=None):
        binEdges = np.array(binEdges)
        binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
        countsdict = collections.OrderedDict()
        if channums is None:
            channums = self.keys()  # this should exclud bad channels by default
        for channum in channums:
            _, countsdict[channum] = self[channum].hist(binEdges, attr, states, goodFunc, returnBad)
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
                  goodFunc=None, maxChans=8, channums=None):
        if channums is None:
            channums = list(self.keys())[:min(maxChans, len(self))]
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels
        for channum in channums:
            ds = self[channum]
            ds.plotHist(binEdges, attr, axis, [], states, goodFunc)
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

    def histsToHDF5(self, h5File, binEdges, attr="energy", goodFunc=None):
        for (channum, ds) in self.items():
            ds.histsToHDF5(h5File, binEdges, attr, goodFunc)
        grp = h5File.require_group("all_channels")
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, goodFunc)
            grp["{}/bin_centers".format(state)] = binCenters
            grp["{}/counts".format(state)] = counts
        binCenters, counts = self.hist(binEdges, attr, goodFunc=goodFunc)  # all states hist
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    def markAllGood(self):
        with self.includeBad():
            for (channum, ds) in self.items():
                ds.markGood()

    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None, worstAllowedFWHM=None, positionToleranceAbsolute=None,
                            attr='energy', states=None, dlo=50, dhi=50, binsize=1, binEdges=None,
                            guessParams=None, goodFunc=None, holdvals=None, resolutionPlot=True, hdf5Group=None,
                            _rethrow=False):
        fitters = self._qualityCheckLinefit(line, positionToleranceFitSigma, worstAllowedFWHM, positionToleranceAbsolute,
                                            attr, states, dlo, dhi, binsize, binEdges, guessParams, goodFunc, holdvals,
                                            _rethrow=_rethrow)
        resolutions = np.array([fitter.last_fit_params_dict["resolution"][0]
                                for fitter in fitters.values() if fitter.fit_success])
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
                        fitter = fitters[channum]
                        for (k, (v, err)) in fitter.last_fit_params_dict.items():
                            grp[k] = v
                            grp[k+"_err"] = err
                        grp["states"] = str(states)
        return fitters

    def setOutputDir(self, baseDir=None, deleteAndRecreate=None, suffix="_output"):
        """Set the output directory to which plots and hdf5 files will go
        baseDir -- the directory in which the output directory will exist
        deleteAndRecreate (required keyword arg) -- if True, will delete the whole directory if it already exists (good for if you re-run the same script alot)
        if False, will attempt to create the directory, if it already exists (like if you rerun the same script), it will error
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

    def fitterPlot(self, lineName, states=None):
        fitters = [ds.linefit(lineName, plot=False, states=states) for ds in self.values()]
        fitter = self.linefit(lineName, plot=False, states=states)
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle("{} fits to {} with states = {}".format(self.shortName, lineName, states))
        fitter.plot(label="full", axis=plt.subplot(2, 2, 3))
        plt.xlabel("energy (eV)")
        plt.ylabel("counts per bin")
        resolutions = [_f.last_fit_params_dict["resolution"][0] for _f in fitters]
        positions = [_f.last_fit_params_dict["peak_ph"][0] for _f in fitters]
        position_errs = [_f.last_fit_params_dict["peak_ph"][1] for _f in fitters]
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
                                                                          fitter.spect.peak_energy), transform=ax.transAxes)
        plt.vlines(fitter.spect.peak_energy, plt.ylim()[0], plt.ylim()[1], label="db position")
        ax = plt.subplot(2, 2, 4)
        plt.errorbar(np.arange(len(positions)), positions, yerr=position_errs, fmt=".")
        plt.hlines(fitter.spect.peak_energy, plt.xlim()[0], plt.xlim()[1], label="db position")
        plt.legend()
        plt.xlabel("channel number")
        plt.ylabel("line position (eV)")


def labelPeak(axis, name, energy, line=None, deltaELocalMaximum=5, color=None):
    if line is None:
        line = axis.lines[0]
    if color is None:
        color = line.get_color()
    ydataLocalMaximum = np.amax([np.interp(energy+de, line.get_xdata(), line.get_ydata(),
                                           right=0, left=0) for de in np.linspace(-1, 1, 10)*deltaELocalMaximum])
    plt.annotate(name, (energy, ydataLocalMaximum), (0, 10), textcoords='offset points',
                 rotation=90, verticalalignment='top', horizontalalignment="center", color=color)


def labelPeaks(axis, names, energies, line=None, deltaELocalMaximum=5, color=None):
    for name, energy in zip(names, energies):
        labelPeak(axis, name, energy, line, deltaELocalMaximum, color)
