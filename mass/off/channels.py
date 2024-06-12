# std lib imports
import os
import logging
import collections

# pkg imports
import numpy as np
import pylab as plt
import fastdtw
import lmfit
import scipy.interpolate
from mass.common import tostr
import dill
import gc
from deprecation import deprecated
import h5py

# local imports
import mass
from .off import OffFile
from .util import GroupLooper, add_group_loop, RecipeBook
from .util import annotate_lines, SilenceBar, NoCutInds, InvalidStatesException
from . import util
from . import fivelag
from .experiment_state import ExperimentStateFile
from . import recipe_classes


LOG = logging.getLogger("mass")


class DriftCorrection:
    version = 1

    def __init__(self, indicatorName, uncorrectedName, medianIndicator, slope):
        self.indicatorName = tostr(indicatorName)
        self.uncorrectedName = tostr(uncorrectedName)
        self.medianIndicator = medianIndicator
        self.slope = slope

    def __call__(self, indicator, uncorrected):
        return self.apply(indicator, uncorrected)

    def apply(self, indicator, uncorrected):
        gain = 1 + (indicator - self.medianIndicator) * self.slope
        return gain * uncorrected

    def toHDF5(self, hdf5_group, name="driftCorrection"):
        hdf5_group[f"{name}/indicatorName"] = self.indicatorName
        hdf5_group[f"{name}/uncorrectedName"] = self.uncorrectedName
        hdf5_group[f"{name}/medianIndicator"] = self.medianIndicator
        hdf5_group[f"{name}/slope"] = self.slope
        hdf5_group[f"{name}/version"] = self.version

    @classmethod
    def fromHDF5(cls, hdf5_group, name="driftCorrection"):
        indicatorName = tostr(hdf5_group[f"{name}/indicatorName"][()])
        uncorrectedName = tostr(hdf5_group[f"{name}/uncorrectedName"][()])
        medianIndicator = hdf5_group[f"{name}/medianIndicator"][()]
        slope = hdf5_group[f"{name}/slope"][()]
        version = hdf5_group[f"{name}/version"][()]
        assert (version == cls.version)
        return cls(indicatorName, uncorrectedName, medianIndicator, slope)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DriftCorrection):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash((self.indicatorName, self.uncorrectedName, self.medianIndicator, self.slope))


class CorG:
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
            # Passing a sequence as a label is deprecated in Matplotlib 3.9, so make it a string
            label = ", ".join(states)
            x, y = self.hist(binEdges, attr, states=states, cutRecipeName=cutRecipeName)
            axis.plot(x, y, drawstyle="steps-mid", label=label)
        else:
            for state in util.iterstates(states):
                x, y = self.hist(binEdges, attr, states=state, cutRecipeName=cutRecipeName)
                axis.plot(x, y, drawstyle="steps-mid", label=state)
        axis.set_xlabel(attr)
        axis.set_ylabel("counts per %0.1f unit bin" % (binEdges[1] - binEdges[0]))
        plt.legend(title="states")
        axis.set_title(self.shortName)
        annotate_lines(axis, labelLines)
        return axis

    def linefit(self, lineNameOrEnergy="MnKAlpha", attr="energy", states=None, axis=None, dlo=50, dhi=50,  # noqa: PLR0917
                binsize=None, binEdges=None, label="full", plot=True,
                params_fixed=None, cutRecipeName=None, calibration=None, require_errorbars=True, method="leastsq_refit",
                has_linear_background=True, has_tails=False, params_update=lmfit.Parameters(),
                minimum_bins_per_fwhm=None):
        """Do a fit to `lineNameOrEnergy` and return the result. You can get the params results with result.params
        lineNameOrEnergy -- A string like "MnKAlpha" will get "MnKAlphaModel"; you
            can pass in a model like a mass.MnKAlphaModel().

        attr -- default is "energyRough". you must pass binEdges if attr is other than "energy" or "energyRough"
        states -- will be passed to hist, coAddStates will be True
        axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
        dlo and dhi and binsize -- by default it tries to fit with bin edges given by
            np.arange(model.spect.peak_energy-dlo, model.spect.peak_energy+dhi, binsize)
        binEdges -- pass the binEdges you want as a numpy array
        label -- passed to model.plot
        plot -- passed to model.fit, determine if plot happens
        cutRecipeName -- a function a function taking a MicrocalDataSet and returning a vector like ds.good() would return
        calbration -- a calibration to be passed to hist - will error if used with an "energy..." attr
        require_errorbars -- throw an error if lmfit doesn't return errorbars
        method -- fit method to use
        has_tails -- used when creating a model, will add both high and low energy tails to the model
        params_update -- after guessing params, call params.update(params_update)
        minimum_bins_per_fwhm -- passed to model.fit
        """
        model = mass.get_model(
            lineNameOrEnergy, has_linear_background=has_linear_background, has_tails=has_tails)
        cutRecipeName = self._handleDefaultCut(cutRecipeName)
        attr_is_energy = attr.startswith("energy") or attr.startswith("p_energy") or calibration is not None
        if binEdges is None:
            if attr_is_energy:
                pe = model.spect.peak_energy
                binEdges = np.arange(pe - dlo, pe + dhi, self._handleDefaultBinsize(binsize))
            else:
                raise Exception(
                    "must pass binEdges if attr does not start with energy and you don't pass a calibration; "
                    "also, don't use energy and calibration at the same time")
        # print(f"binEdges.size={binEdges.size}, binEdges.mean()={binEdges.mean()}")
        # print(f"attr={attr},states={states}")
        bin_centers, counts = self.hist(
            binEdges, attr, states, cutRecipeName, calibration=calibration)
        # print(f"counts.size={counts.size},counts.sum()={counts.sum()}")
        if attr_is_energy:
            params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
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
        result = model.fit(counts, params, bin_centers=bin_centers, method=method,
                           minimum_bins_per_fwhm=minimum_bins_per_fwhm)
        if states is None:
            states_hint = "all states"
        elif isinstance(states, list):
            states_hint = ", ".join(states)
        else:
            states_hint = states
        result.set_label_hints(binsize=bin_centers[1] - bin_centers[0], ds_shortname=self.shortName,
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


class Channel(CorG):  # noqa: PLR0904
    """Wrap up an OFF file with some convience functions like a TESChannel"""

    def __init__(self, offFile, experimentStateFile, verbose=True):
        self.offFile = offFile
        self.experimentStateFile = experimentStateFile
        self.markedBadBool = False
        self._statesDict = None
        self.verbose = verbose
        self.learnChannumAndShortname()
        self.recipes = RecipeBook(self._offAttrs, propertyClass=Channel,
                                  coefs_dtype=self.offFile._dtype_non_descriptive)
        # wrapper is part of a hack to allow "coefs" and "filtValue" to be recipe ingredients
        self._defineDefaultRecipesAndProperties()  # sets _default_cut_recipe_name

    def _defineDefaultRecipesAndProperties(self):
        assert (len(self.recipes) == 0)
        t0 = self.offFile["unixnano"][0]
        self.recipes.add("relTimeSec",
                         recipe_classes.SubtractThenScale(sub=t0, scale=1e-9), ["unixnano"])
        self.recipes.add("filtPhase", recipe_classes.DivideTwo(), ["derivativeLike", "filtValue"])
        self.cutAdd("cutNone", recipe_classes.TruesOfSameSize(), ["unixnano"], setDefault=True)

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
        return self.offFile.dtype.names + ("coefs",)

    @property
    def _recipeAttrs(self):
        return self.recipes.keys()

    def isOffAttr(self, attr):
        return attr in self._offAttrs

    def isRecipeAttr(self, attr):
        return attr in self._recipeAttrs

    def learnChannumAndShortname(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFile.filename)
        self.shortName = os.path.split(basename)[-1] + f" chan{self.channum:g}"

    @add_group_loop
    def learnResidualStdDevCut(self, n_sigma_equiv=15,   # noqa: PLR0914, PLR0917
                               newCutRecipeName="cutResidualStdDev", binSizeFv=2000, minFv=150,
                               states=None, plot=False, setDefault=True, overwriteRecipe=False, cutRecipeName=None):
        """EXPERIMENTAL: learn a cut based on the residualStdDev. uses the median absolute deviation to estiamte a gaussian sigma
        that is robust to outliers as a function of filt Value, then uses that to set an upper limit based on n_sigma_equiv
        highly reccomend that you call it with plot=True on at least a few datasets first
        """
        # the code currently only works for a single threshold, but has some parts in place for
        # implementing a filtValue dependent threshold
        filtValue, residualStdDev = self.getAttr(
            ["filtValue", "residualStdDev"], indsOrStates=states, cutRecipeName=cutRecipeName)
        # binEdges = np.percentile(filtValue, np.linspace(0, 100, N+1))
        binEdges = np.arange(0, np.amax(filtValue), binSizeFv)
        N = len(binEdges) - 1
        sigmas, medians, fv_mids = [0], [0], [minFv]
        for i in range(N):
            lo, hi = binEdges[i], binEdges[i + 1]
            inds = np.logical_and(filtValue > lo, filtValue < hi)
            if len(inds) <= 4:
                continue
            _mad, sigma_equiv, median = mass.off.util.median_absolute_deviation(residualStdDev[inds])
            sigmas.append(sigma_equiv)
            medians.append(median)
            fv_mids.append((lo + hi) / 2)
        if len(sigmas) < 1:
            raise Exception(f"too few pulses, len(filtValue)={len(filtValue)}")
        sigmas = np.array(sigmas)
        medians = np.array(medians)
        fv_mids = np.array(fv_mids)

        threshold = medians + n_sigma_equiv * sigmas
        threshold_func = scipy.interpolate.interp1d(fv_mids, threshold, kind="next", bounds_error=False,
                                                    fill_value=(-1, threshold[-1]))
        # the threshold for all filtValues below minFv will be -1
        # filtValues just above binFv should look to the next point since kind="next", so the precise
        # choice of median and sigma to pair with binFv shouldn't matter.
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
            plt.plot(x, y, label=f"{threshold}", lw=3)
            plt.legend()
            plt.yscale("log")
            plt.ylim(ymin / 2, ymax * 2)

    def getStatesIndicies(self, states=None):
        """return a list of slices corresponding to the passed states
        this list is appropriate for passing to _indexOffWithCuts or getRecipeAttr
        """
        if states is None:
            return [slice(0, len(self))]
        inds = []
        for state in util.iterstates(states):
            v = self.statesDict[state]
            if isinstance(v, slice):
                inds.append(v)
            elif isinstance(v, list):
                for vv in v:
                    assert isinstance(vv, slice)
                    inds.append(vv)
            else:
                raise InvalidStatesException(
                    f"v should be a list of slices or a slice, but is a {type(v)}")
        return inds

    def __repr__(self):
        return f"Channel based on {self.offFile}"

    @property
    def statesDict(self):
        if self._statesDict is None:
            unixnano = self.getAttr("unixnano", NoCutInds())
            esf = self.experimentStateFile
            self._statesDict = esf.calcStatesDict(unixnano)
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
            g = self.recipes.craft(cutRecipeName, r)
            output = r[g]
        elif isinstance(inds, list) and _listMethodSelect == 2:  # preallocate and truncate
            # testing on the 20191219_0002 TOMCAT dataset with len(inds)=432 showed this
            # method to be more than 10x faster than repeated hstack
            # and about 2x faster than temporary bool index, which can be found in commit 063bcce
            # make sure s.step is None so my simple length calculation will work
            assert all([isinstance(s, slice) and s.step is None for s in inds])
            max_length = np.sum([s.stop - s.start for s in inds])
            output_dtype = self.offFile.dtype  # get the dtype to preallocate with
            output_prealloc = np.zeros(max_length, output_dtype)
            ilo, ihi = 0, 0
            for s in inds:
                tmp = self._indexOffWithCuts(s, cutRecipeName)
                ilo = ihi
                ihi = ilo + len(tmp)
                output_prealloc[ilo:ihi] = tmp
            output = output_prealloc[0:ihi]
        elif isinstance(inds, list) and _listMethodSelect == 0:  # repeated hstack
            # this could be removed, along with the _listMethodSelect argument
            # this is only left in because it is useful for correctness testing
            # for preallocate and truncate method since this is simpler.
            assert all([isinstance(_inds, slice) for _inds in inds])
            output = self._indexOffWithCuts(inds[0], cutRecipeName)
            for i in range(1, len(inds)):
                output = np.hstack((output, self._indexOffWithCuts(inds[i], cutRecipeName)))
        elif isinstance(inds, NoCutInds):
            output = self.offFile
        else:
            raise Exception(f"type(inds)={type(inds)}, should be slice or list or slices")
        return output

    def getAttr(self, attr, indsOrStates, cutRecipeName=None):
        """
        attr - may be a string or a list of strings corresponding to Attrs defined by recipes or the offFile
        inds - a slice or list of slices
        returns either a single vector or a list of vectors whose entries correspond to the entries in attr
        """
        # first
        # relies on short circuiting to not evaluate last clause unless indsOrStates is a list
        if indsOrStates is None or isinstance(indsOrStates, str) or \
                (isinstance(indsOrStates, list) and isinstance(indsOrStates[0], str)):
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
            raise Exception(f"attr {attr} must be an OffAttr or a RecipeAttr or a list. OffAttrs: " +
                            f"{list(self._offAttrs)}\nRecipeAttrs: {list(self._recipeAttrs)}")

    def plotAvsB2d(self, nameA, nameB, binEdgesAB, axis=None, states=None, cutRecipeName=None, norm=None):
        cutRecipeName = self._handleDefaultCut(cutRecipeName)
        if axis is None:
            plt.figure()
            axis = plt.gca()
        if states is None:
            states = self.stateLabels
        vA, vB = self.getAttr([nameA, nameB], states, cutRecipeName)
        counts, binEdgesA, binEdgesB = np.histogram2d(vA, vB, binEdgesAB)
        binCentersA = 0.5 * (binEdgesA[1:] + binEdgesA[:-1])
        binCentersB = 0.5 * (binEdgesB[1:] + binEdgesB[:-1])
        plt.contourf(binCentersA, binCentersB, counts.T, norm=norm)
        plt.xlabel(nameA)
        plt.ylabel(nameB)
        plt.title(f"{self.shortName}\ncutRecipeName={cutRecipeName}")
        return axis

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
            self._plotAvsB_single(nameA, nameB, axis, states, includeBad,
                                  cutRecipeName, prefix=nameB)

    def _plotAvsB_single(self, nameA, nameB, axis=None, states=None, includeBad=False, cutRecipeName=None, prefix=""):
        for state in util.iterstates(states):
            A, B = self.getAttr([nameA, nameB], state, cutRecipeName)
            axis.plot(A, B, ".", label=prefix + state)
            if includeBad:
                A, B = self.getAttr([nameA, nameB], state, f"!{cutRecipeName}")
                axis.plot(A, B, "x", label=prefix + state + " bad")

    def hist(self, binEdges, attr, states=None, cutRecipeName=None, calibration=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute).
        automatically filtes out nan values

        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
        cutRecipeName -- a function taking a 1d array of vales of type self.offFile.dtype and returning a vector of bool
        calibration -- if not None, transform values by val = calibration(val) before histogramming
        """
        binEdges = np.array(binEdges)
        binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
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
                             linePositionsFunc=None, cutRecipeName=None, overwriteRecipe=False):
        """
        linePositionsFunc - if None, then use self.calibrationRough._ph as the peak locations
        otherwise try to call it with self as an argument...
        Here is an example of how you could use all but one peak from calibrationRough:
        `data.learnPhaseCorrection(linePositionsFunc = lambda ds: ds.recipes["energyRough"].f._ph`
        """
        # may need to generalize this to allow using a specific state for phase correction as
        # a specfic line... with something like calibrationPlan
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
            phaseCorrection.indicatorName, phaseCorrection.uncorrectedName], overwrite=overwriteRecipe)

    @add_group_loop
    def learnTimeDriftCorrection(self, indicatorName="relTimeSec", uncorrectedName="filtValue", correctedName=None,  # noqa: PLR0917
                                 states=None, cutRecipeName=None, kernel_width=1, sec_per_degree=2000,
                                 pulses_per_degree=2000, max_degrees=20, ndeg=None, limit=None, overwriteRecipe=False):
        """do a polynomial correction based on the indicator
        you are encouraged to change the settings that affect the degree of the polynomail
        see help in mass.core.channel.time_drift_correct for details on settings"""
        if correctedName is None:
            correctedName = uncorrectedName + "TC"
        indicator, uncorrected = self.getAttr(
            [indicatorName, uncorrectedName], states, cutRecipeName)
        info = mass.core.channel.time_drift_correct(indicator, uncorrected, kernel_width, sec_per_degree,
                                                    pulses_per_degree, max_degrees, ndeg, limit)

        def time_drift_correct(indicator, uncorrected):
            tnorm = info["normalize"](indicator)
            corrected = uncorrected * (1 + info["model"](tnorm))
            return corrected
        self.recipes.add(correctedName, time_drift_correct, [indicatorName, uncorrectedName], overwrite=overwriteRecipe)

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
        for state in util.iterstates(states):
            A, B, C = self.getAttr([indicatorName, uncorrectedName,
                                    "filtValueDC"], state, cutRecipeName)
            axis.plot(A, B, ".", label=state)
            axis.plot(A, C, ".", label=state + " DC")
            if includeBad:
                A, B, C = self.getAttr([indicatorName, uncorrectedName,
                                        "filtValueDC"], state, cutRecipeName=True)
                axis.plot(A, B, "x", label=state + " bad")
                axis.plot(A, C, "x", label=state + " bad DC")
        plt.xlabel(indicatorName)
        plt.ylabel(uncorrectedName + ",filtValueDC")
        plt.title(self.shortName + " drift correct comparison")
        plt.legend(title="states")
        return axis

    def calibrationPlanInit(self, attr):
        self.calibrationPlan = CalibrationPlan()
        self.calibrationPlanAttr = attr

    def calibrationPlanAddPoint(self, uncalibratedVal, name, states=None, energy=None):
        if name in mass.spectra:
            line = mass.spectra[name]
        elif energy is None:
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
    def calibrateFollowingPlan(self, uncalibratedName, calibratedName="energy", curvetype="gain", approximate=False,  # noqa: PLR0917
                               dlo=50, dhi=50, binsize=None, plan=None, n_iter=1, method="leastsq_refit", overwriteRecipe=False,
                               has_tails=False, params_update=lmfit.Parameters(), cutRecipeName=None):
        if plan is None:
            plan = self.calibrationPlan
        starting_cal = plan.getRoughCalibration()
        intermediate_calibrations = []
        for i in range(n_iter):
            calibration = mass.EnergyCalibration(curvetype=curvetype, approximate=approximate)
            calibration.uncalibratedName = uncalibratedName
            results = []
            for (line, states) in zip(plan.lines, plan.states):
                result = self.linefit(line, uncalibratedName, states, dlo=dlo, dhi=dhi,
                                      plot=False, binsize=binsize, calibration=starting_cal, require_errorbars=False,
                                      method=method, params_update=params_update, has_tails=has_tails,
                                      cutRecipeName=cutRecipeName)

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
            is_last_iteration = i + 1 == n_iter
            if not is_last_iteration:
                intermediate_calibrations.append(calibration)
                starting_cal = calibration
        calibration.intermediate_calibrations = intermediate_calibrations
        self.recipes.add(calibratedName, calibration,
                         [calibration.uncalibratedName], overwrite=overwriteRecipe)
        return results

    @add_group_loop
    def learnCalibrationPlanFromEnergiesAndPeaks(self, attr, states, ph_fwhm, line_names, maxacc, polynomial=False):
        peak_ph_vals, _peak_heights = mass.algorithms.find_local_maxima(self.getAttr(attr, indsOrStates=states), ph_fwhm)
        if polynomial:
            _name_e, _energies_out, opt_assignments = mass.algorithms.find_opt_assignment_polynomial(peak_ph_vals, line_names, maxacc=maxacc)
        else:
            _name_e, _energies_out, opt_assignments = mass.algorithms.find_opt_assignment(peak_ph_vals, line_names, maxacc=maxacc)

        self.calibrationPlanInit(attr)
        for ph, name in zip(opt_assignments, _name_e):
            self.calibrationPlanAddPoint(ph, name, states=states)

    def markBad(self, reason, extraInfo=None):
        self.markedBadReason = reason
        self.markedBadExtraInfo = extraInfo
        self.markedBadBool = True
        s = f"\nMARK BAD {self.shortName}: reason={reason}"
        if extraInfo is not None:
            s += f"\nextraInfo: {extraInfo}"
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
            assert (len(referenceChannel.calibrationPlan.uncalibratedVals) > 0)
            peakLocs = referenceChannel.calibrationPlan.uncalibratedVals
        else:
            peakLocs = _peakLocs
        self.aligner = AlignBToA(ds_a=referenceChannel, ds_b=self,
                                 peak_xs_a=peakLocs, bin_edges=binEdges, attr=attr, states=states,
                                 cutRecipeName=cutRecipeName)
        self.calibrationArbsInRefChannelUnits = self.aligner.getCalBtoA()
        if _peakLocs is None and (self is not referenceChannel):
            self.calibrationPlanInit(referenceChannel.calibrationPlanAttr)
            refCalPlan = referenceChannel.calibrationPlan
            for (ph, energy, name, states2, line) in zip(
                refCalPlan.uncalibratedVals, refCalPlan.energies,
                    refCalPlan.names, refCalPlan.states, refCalPlan.lines):
                self.calibrationPlan.addCalPoint(
                    self.calibrationArbsInRefChannelUnits.energy2ph(ph),
                    states2, line)
        calibrationRough = self.calibrationPlan.getRoughCalibration()
        calibrationRough.uncalibratedName = self.calibrationPlanAttr
        self.recipes.add("energyRough", calibrationRough,
                         [calibrationRough.uncalibratedName], inverse=calibrationRough.energy2ph, overwrite=True)
        self.recipes.add("arbsInRefChannelUnits", self.calibrationArbsInRefChannelUnits.ph2energy, [
            self.calibrationArbsInRefChannelUnits.uncalibratedName], overwrite=True)
        return self.aligner

    @add_group_loop
    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None, worstAllowedFWHM=None,  # noqa: PLR0917
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
            tolerance = fitSigma * positionToleranceFitSigma
        else:
            tolerance = np.inf
        if np.abs(fitPos - result.model.spect.peak_energy) > tolerance:
            self.markBad(f"qualityCheckLinefit: for {line}, want {result.model.spect.peak_energy} within {tolerance}, got {fitPos}")
        if worstAllowedFWHM is not None and resolution > worstAllowedFWHM:
            self.markBad(f"qualityCheckLinefit: for {line}, fit resolution {resolution} > threshold {worstAllowedFWHM}")
        return result

    @add_group_loop
    def histsToHDF5(self, h5File, binEdges, attr="energy", cutRecipeName=None):
        grp = h5File.require_group(str(self.channum))
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, cutRecipeName)
            grp[f"{state}/bin_centers"] = binCenters
            grp[f"{state}/counts"] = counts
        binCenters, counts = self.hist(
            binEdges, attr, cutRecipeName=cutRecipeName)  # all states hist
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    @add_group_loop
    def energyTimestampLabelToHDF5(self, h5File, cutRecipeName=None):
        usedCutRecipeName = self._handleDefaultCut(cutRecipeName)
        grp = h5File.require_group(str(self.channum))
        if len(self.stateLabels) > 0:
            for state in self.stateLabels:
                energy, unixnano = self.getAttr(["energy", "unixnano"], state, usedCutRecipeName)
                grp[f"{state}/energy"] = energy
                grp[f"{state}/unixnano"] = unixnano
                state_inds = tuple(self.getStatesIndicies(states=[state]))
                cut_inds_in_state = self.getAttr(usedCutRecipeName, [state], "cutNone")
                if hasattr(self, "seconds_after_last_external_trigger"):
                    grp[f"{state}/seconds_after_last_external_trigger"] = \
                        self.seconds_after_last_external_trigger[state_inds][cut_inds_in_state]
                if hasattr(self, "seconds_until_next_external_trigger"):
                    grp[f"{state}/seconds_until_next_external_trigger"] = \
                        self.seconds_until_next_external_trigger[state_inds][cut_inds_in_state]
                if hasattr(self, "seconds_from_nearest_external_trigger"):
                    grp[f"{state}/seconds_from_nearest_external_trigger"] = \
                        self.seconds_from_nearest_external_trigger[state_inds][cut_inds_in_state]
        else:
            energy, unixnano = self.getAttr(
                ["energy", "unixnano"], slice(None), usedCutRecipeName)
            grp[f"{state}/energy"] = energy
            grp[f"{state}/unixnano"] = unixnano
        grp["off_filename"] = self.offFile.filename
        grp["used_cut_recipe_name"] = usedCutRecipeName

    @add_group_loop
    def qualityCheckDropOneErrors(self, thresholdAbsolute=None, thresholdSigmaFromMedianAbsoluteValue=None):
        calibration = self.recipes["energy"].f
        _energies, errors = calibration.drop_one_errors()
        maxAbsError = np.amax(np.abs(errors))
        medianAbsoluteValue = np.median(np.abs(errors))
        k = 1.4826  # https://en.wikipedia.org/wiki/Median_absolute_deviation
        sigma = k * medianAbsoluteValue
        if thresholdAbsolute is not None:
            if maxAbsError > sigma * thresholdSigmaFromMedianAbsoluteValue:
                self.markBad("qualityCheckDropOneErrors: maximum absolute drop one error {} > theshold {} ({})".format(
                    maxAbsError, sigma * thresholdSigmaFromMedianAbsoluteValue,
                    "thresholdSigmaFromMedianAbsoluteValue"))
        if thresholdAbsolute is not None:
            if maxAbsError > thresholdAbsolute:
                msg = f"qualityCheckDropOneErrors: maximum absolute drop one error {maxAbsError} >" + \
                    f" theshold {thresholdAbsolute} (thresholdAbsolute)"
                self.markBad(msg)

    def diagnoseCalibration(self, calibratedName="energy", fig=None, filtValuePlotBinEdges=np.arange(0, 16000, 4)):
        calibration = self.recipes[calibratedName].f
        uncalibratedName = calibration.uncalibratedName
        results = calibration.results
        n_intermediate = len(calibration.intermediate_calibrations)
        # fig can be a matplotlib.figure.Figure object or an index ("num") of the current figures (see plt.get_fignums())
        if fig is not None:
            plt.figure(fig)
        else:
            plt.figure(figsize=(20, 12))
        plt.suptitle(
            self.shortName + f", cal diagnose for '{calibratedName}'\n with {n_intermediate} intermediate calibrations")
        n = int(np.ceil(np.sqrt(len(results) + 2)))
        for i, result in enumerate(results):
            ax = plt.subplot(n, n, i + 1)
            # pass title to suppress showing the dataset shortName on each subplot
            result.plotm(ax=ax, title=str(result.model.spect.shortname))
        ax = plt.subplot(n, n, i + 2)
        calibration.plot(axis=ax)
        ax = plt.subplot(n, n, i + 3)
        self.plotHist(filtValuePlotBinEdges, uncalibratedName,
                      axis=ax, coAddStates=False)
        plt.vlines(self.calibrationPlan.uncalibratedVals, 0, plt.ylim()[1])
        plt.tight_layout()

    def add5LagRecipes(self, f):
        _filter_5lag_in_basis, filter_5lag_fit_in_basis = fivelag.calc_5lag_fit_matrix(
            f[:], self.offFile.basis)
        self.recipes.add("cba5Lag", recipe_classes.MatMulAB_FixedB(B=filter_5lag_fit_in_basis),
                         ingredients=["coefs"])
        self.recipes.add("filtValue5Lag", fivelag.filtValue5Lag, ingredients=["cba5Lag"])
        self.recipes.add("peakX5Lag", fivelag.peakX5Lag, ingredients=["cba5Lag"])

    @property
    def rowPeriodSeconds(self):
        nRows = self.offFile.header["ReadoutInfo"]["NumberOfRows"]
        return self.offFile.framePeriodSeconds / float(nRows)

    @deprecated(deprecated_in="0.8.2", details="Use subframecount, which is equivalent but better named")
    @property
    def rowcount(self):
        return self.subframecount

    @property
    def subframeDivisions(self):
        hdr = self.offFile.header["ReadoutInfo"]
        return hdr.get("SubframeDivsions", hdr["NumberOfRows"])

    @property
    def subframePeriodSeconds(self):
        nDivs = self.subframeDivisions
        return self.offFile.framePeriodSeconds / float(nDivs)

    @property
    def subframecount(self):
        return self.offFile["framecount"] * self.subframeDivisions

    @add_group_loop
    def _calcExternalTriggerTiming(self, external_trigger_subframe_count, after_last, until_next, from_nearest):
        subframes_after_last_external_trigger, subframes_until_next_external_trigger = \
            mass.core.analysis_algorithms.nearest_arrivals(self.subframecount, external_trigger_subframe_count)
        rowPeriodSeconds = self.rowPeriodSeconds
        if after_last:
            self.subframes_after_last_external_trigger = subframes_after_last_external_trigger
            self.seconds_after_last_external_trigger = subframes_after_last_external_trigger * rowPeriodSeconds
        if until_next:
            self.subframes_until_next_external_trigger = subframes_until_next_external_trigger
            self.seconds_until_next_external_trigger = subframes_until_next_external_trigger * rowPeriodSeconds
        if from_nearest:
            self.subframes_from_nearest_external_trigger = np.fmin(subframes_after_last_external_trigger,
                                                                   subframes_until_next_external_trigger)
            self.seconds_from_nearest_external_trigger = self.subframes_from_nearest_external_trigger * rowPeriodSeconds


def normalize(x):
    return x / float(np.sum(x))


def dtw_same_peaks(bin_edges, ph_a, ph_b, peak_inds_a, scale_by_median, normalize_before_dtw, plot=False):
    if scale_by_median:
        median_ratio_a_over_b = np.median(ph_a) / np.median(ph_b)
    else:
        median_ratio_a_over_b = 1.0
    ph_b_median_scaled = ph_b * median_ratio_a_over_b
    counts_a, _ = np.histogram(ph_a, bin_edges)
    counts_b_median_scaled, _ = np.histogram(ph_b_median_scaled, bin_edges)
    if normalize_before_dtw:
        _distance, path = fastdtw.fastdtw(normalize(counts_a),
                                          normalize(counts_b_median_scaled))
    else:
        _distance, path = fastdtw.fastdtw(counts_a, counts_b_median_scaled)
    i_a = [x[0] for x in path]
    i_b_median_scaled = [x[1] for x in path]
    peak_inds_b_median_scaled = np.array(
        [i_b_median_scaled[i_a.index(pia)] for pia in peak_inds_a])
    peak_xs_b_median_scaled = bin_edges[peak_inds_b_median_scaled]
    peak_xs_b = peak_xs_b_median_scaled / median_ratio_a_over_b
    min_bin = bin_edges[0]
    bin_spacing = bin_edges[1] - bin_edges[0]
    peak_inds_b = np.array((peak_xs_b - min_bin) / bin_spacing, dtype="int")

    if plot:
        counts_b, _ = np.histogram(ph_b, bin_edges)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
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


class AlignBToA:
    cm = plt.cm.gist_ncar

    def __init__(self, ds_a, ds_b, peak_xs_a, bin_edges, attr, cutRecipeName, states=None,  # noqa: PLR0917
                 scale_by_median=True, normalize_before_dtw=True):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        self.peak_xs_a = peak_xs_a
        self.attr = attr
        self.cutRecipeName = cutRecipeName
        self.states = states
        self.scale_by_median = scale_by_median
        self.normalize_before_dtw = normalize_before_dtw
        self.peak_inds_a = np.searchsorted(self.bin_edges, self.peak_xs_a) - 1
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
                     color=self.cm(float(i) / len(self.peak_inds_a)))

        plt.plot(self.bin_centers, counts_b, label="b: channel %i" % self.ds_b.channum)
        for i, pi in enumerate(self.peak_inds_b):
            plt.plot(self.bin_centers[pi], counts_b[pi], "o",
                     color=self.cm(float(i) / len(self.peak_inds_b)))
        plt.xlabel(self.attr)
        plt.ylabel("counts per %0.2f unit bin" % (self.bin_centers[1] - self.bin_centers[0]))
        plt.legend(title="channel")
        plt.title(self.ds_a.shortName + " + " + self.ds_b.shortName
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
                     color=self.cm(float(i) / len(self.peak_inds_a)))
        plt.plot(self.bin_centers, counts_b, label="b: channel %i" % self.ds_b.channum)
        for i, pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi], counts_b[pi], "o",
                     color=self.cm(float(i) / len(self.peak_inds_a)))
        plt.xlabel(f"arbsInRefChannelUnits (ref channel = {self.ds_a.channum})")
        plt.ylabel("counts per %0.2f unit bin" % (self.bin_centers[1] - self.bin_centers[0]))
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
        threshold_hi = 1 + threshold_frac
        threshold_lo = 1 / threshold_hi
        # here we test the "curvature" of cal_b_to_a
        # by comparing the most extreme sloped segment to the median slope
        derivatives = self.cal_b_to_a.energy2dedph(self.cal_b_to_a._energies)
        diff_frac_hi = np.amax(derivatives) / np.median(derivatives)
        diff_frac_lo = np.amin(derivatives) / np.median(derivatives)
        return diff_frac_hi < threshold_hi and diff_frac_lo > threshold_lo

    def _laplaceEntropy(self, w=None, cutRecipeName_a=None, cutRecipeName_b=None):
        if cutRecipeName_a is None:
            cutRecipeName_a = self.cutRecipeName
        if cutRecipeName_b is None:
            cutRecipeName_b = self.cutRecipeName
        if w is None:
            w = self.bin_edges[1] - self.bin_edges[0]
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
        cdf_a = np.cumsum(counts_a) / np.sum(counts_a)
        cdf_b = np.cumsum(counts_b) / np.sum(counts_b)
        ks_statistic = np.amax(np.abs(cdf_a - cdf_b))
        return ks_statistic


# calibration
class CalibrationPlan:
    def __init__(self):
        self.uncalibratedVals = np.zeros(0)
        self.states = []
        self.lines = []

    def addCalPoint(self, uncalibratedVal, states, line):
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
    z = mass.ljh_util.filename_glob_expand(basename + "_chan*.off")
    if z is None:
        raise Exception("found no files while globbing {}".format(basename + "_chan*.off"))
    if maxChans is not None:
        z = z[:min(maxChans, len(z))]
    return z


class ChannelGroup(CorG, GroupLooper, collections.OrderedDict):  # noqa: PLR0904
    """
    ChannelGroup is an OrdredDict of Channels with some additional features:

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

    def __del__(self):
        # We need to recover the limited resource of system file descriptors when we are done with an
        # off.ChannelGroup object. One way in practice that seems to make a difference is to run the
        # garbage collector when each one is deleted, to clean up the `np.memmap` objects held by the
        # `OffFile` objects in the `self.values()` list of `off.Channel` objects.

        # The step can take something like 1 second to run. This seems a reasonable price to pay in
        # standard usage. If a use-case arises where it's not, then we can make this step conditional?
        # See issue #212 and PR 200 for discussion.

        # Consider this a partial or temporary solution to a nagging problem.
        gc.collect()

    def _handleDefaultCut(self, cutRecipeName):
        ds = self.firstGoodChannel()
        cutRecipeName = ds._handleDefaultCut(cutRecipeName)
        for ds in self.values():
            assert cutRecipeName in ds.recipes.keys(), f"{ds} lacks cut recipe {cutRecipeName}"
        return cutRecipeName

    @property
    def shortName(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFileNames[0])
        return os.path.split(basename)[-1] + f" {len(self)} chans"

    def add5LagRecipes(self, model_hdf5_path, invert_filter_5lag=False):
        with h5py.File(model_hdf5_path, "r") as h5:
            models = {int(ch): mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
        for channum, ds in self.items():
            # define recipes for "filtValue5Lag", "peakX5Lag" and "cba5Lag"
            # where cba refers to the coefficiencts of a polynomial fit to the 5 lags of the filter
            ds.model = models[ds.channum]
            filter_5lag = ds.model.f_5lag
            if invert_filter_5lag:
                filter_5lag *= -1
            ds.add5LagRecipes(filter_5lag)
        return models

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
        return f"ChannelGroup with {len(self)} channels"

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
        n_new_labels = len(self.experimentStateFile.labels) - n_old_labels
        n_new_pulses_dict = collections.OrderedDict()
        for ds in self.values():
            i0_unixnanos = len(ds)
            ds.offFile._updateMmap()  # will update nRecords by mmapping more data in the offFile if available
            ds._statesDict = self.experimentStateFile.calcStatesDict(
                ds.unixnano[i0_unixnanos:], ds.statesDict, i0_allLabels, i0_unixnanos)
            n_new_pulses_dict[ds.channum] = len(ds) - i0_unixnanos
        return n_new_labels, n_new_pulses_dict

    def hist(self, binEdges, attr, states=None, cutRecipeName=None, calibration=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute).
        Automatically filters out nan values.

        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram, e.g. "filt_value"
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
        binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
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
            line.set_label(f"{channum}")
            if ds.markedBadBool:
                line.set_dashes([2, 2, 10, 2])
        axis.set_title(self.shortName + f", states = {states}")
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
            usedCutRecipeName = self._handleDefaultCut(cutRecipeName)
            ds.histsToHDF5(h5File, binEdges, attr, usedCutRecipeName)
        grp = h5File.require_group("all_channels")
        for state in self.stateLabels:  # hist for each state
            binCenters, counts = self.hist(binEdges, attr, state, usedCutRecipeName)
            grp[f"{state}/bin_centers"] = binCenters
            grp[f"{state}/counts"] = counts
        binCenters, counts = self.hist(
            binEdges, attr, cutRecipeName=usedCutRecipeName)  # all states hist
        grp["off_filename"] = self.offFileNames[0]
        grp["attr"] = attr
        grp["used_cut_recipe_name"] = usedCutRecipeName
        grp["bin_centers_ev"] = binCenters
        grp["counts"] = counts
        grp["name_of_energy_indicator"] = attr

    def markAllGood(self):
        with self.includeBad():
            for (channum, ds) in self.items():
                ds.markGood()

    def qualityCheckLinefit(self, line, positionToleranceFitSigma=None,   # noqa: PLR0917
                            worstAllowedFWHM=None, positionToleranceAbsolute=None,
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
            axis.hist(resolutions, bins=np.arange(0, np.amax(resolutions) + 0.25, 0.25))
            axis.set_xlabel("energy resoluiton fwhm (eV)")
            axis.set_ylabel("# of channels / 0.25 eV bin")
            plt.title(self.shortName + f" at {line}")
        if hdf5Group is not None:
            with self.includeBad():
                for (channum, ds) in self.items():
                    grp = hdf5Group.require_group(f"{channum}/fits/{line}")
                    if ds.markedBadBool:
                        grp["markedBadReason"] = ds.markedBadReason
                    else:
                        result = results[channum]
                        for (k, v) in result.params.items():
                            grp[k] = v.value
                            grp[k + "_err"] = v.stderr
                        grp["states"] = str(states)
        return results

    def outputHDF5Filename(self, outputDir, addToName=""):
        basename = self.shortName.split(" ")[0]
        filename = os.path.join(outputDir, f"{basename}_{addToName}.hdf5")
        return filename

    def resultPlot(self, lineName, states=None, binsize=None):
        results = [ds.linefit(lineName, plot=False, states=states, binsize=binsize)
                   for ds in self.values()]
        result = self.linefit(lineName, plot=False, states=states, binsize=binsize)
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f"{self.shortName} fits to {lineName} with states = {states}")
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
        plt.text(0.5, 0.9, f"median = {np.median(resolutions):.2f}", transform=ax.transAxes)
        plt.vlines(np.median(resolutions), plt.ylim()[0], plt.ylim()[1], label="median")
        ax = plt.subplot(2, 2, 2)
        plt.hist(positions)
        plt.xlabel("fit position (eV)")
        plt.ylabel("channels per bin")
        message = f"median = {np.median(positions):.2f}\ndb position = {result.model.spect.peak_energy:.3f}"
        plt.text(0.5, 0.9, message, transform=ax.transAxes)
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

    def saveRecipeBooks(self, filename):
        with open(filename, "wb") as f:
            d = {}
            for ds in self.values():
                d[ds.channum] = ds.recipes
            dill.dump(d, f)

    def loadRecipeBooks(self, filename):
        with open(filename, "rb") as f:
            d = dill.load(f)
        for channum, recipes in d.items():
            self[channum].recipes = recipes

    def _externalTriggerFilename(self):
        datasetFilename = self.offFileNames[0]
        basename, _channum = mass.ljh_util.ljh_basename_channum(datasetFilename)
        return basename + "_external_trigger.bin"

    def _externalTriggerSubframes(self, filename=None):
        if filename is None:
            filename = self._externalTriggerFilename()
        f = open(filename, "rb")
        f.readline()  # discard comment line
        external_trigger_subframe_count = np.fromfile(f, "int64")
        return external_trigger_subframe_count

    def calcExternalTriggerTiming(self, after_last=True, until_next=False, from_nearest=False):
        external_trigger_subframe_count = self._externalTriggerSubframes()
        self._calcExternalTriggerTiming(external_trigger_subframe_count, after_last, until_next, from_nearest, _rethrow=True)


class ChannelFromNpArray(Channel):
    def __init__(self, a, channum, shortname, experimentStateFile=None, verbose=True):
        self.a = a
        self.offFile = a  # to make methods from a normal channelGroup that access offFile as an array work
        self.experimentStateFile = experimentStateFile
        self.shortName = shortname
        self.channum = channum
        self.markedBadBool = False
        self._statesDict = None
        self.verbose = verbose
        self.recipes = RecipeBook(list(self.a.dtype.fields.keys()),
                                  propertyClass=ChannelFromNpArray,
                                  coefs_dtype=None)
        self._defineDefaultRecipesAndProperties()  # sets _default_cut_recipe_name

    def _defineDefaultRecipesAndProperties(self):
        assert (len(self.recipes) == 0)
        if "p_timestamp" in self.a.dtype.names:
            t0 = self.a[0]["p_timestamp"]
            self.recipes.add("relTimeSec", recipe_classes.Subtract(t0), ingredients=["p_timestamp"])
            self.cutAdd("cutNone", recipe_classes.TruesOfSameSize(),
                        ingredients=["p_timestamp"], setDefault=True)
            if "unixnano" not in self.a.dtype.names:
                # unixnano is needed for states to work
                self.recipes.add("unixnano", recipe_classes.ScalarMultAndTurnToInt64(1e9),
                                 ingredients=["p_timestamp"])
        else:
            first_field = self.a.dtype.names[0]
            self.cutAdd("cutNone", recipe_classes.TruesOfSameSize(),
                        [first_field], setDefault=True)

    def __len__(self):
        return len(self.a)

    def refreshFromFiles(self):
        raise Exception(f"not implemented for {self.__class__.__name__}")

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
            r = self.a[inds]
            # I'd like to be able to do either r["coefs"] to get all projection coefficients
            # or r["filtValue"] to get only the filtValue
            # IngredientsWrapper lets that work within recipes.craft
            g = self.recipes.craft(cutRecipeName, r)
            output = r[g]
        elif isinstance(inds, list) and _listMethodSelect == 2:  # preallocate and truncate
            # testing on the 20191219_0002 TOMCAT dataset with len(inds)=432 showed this
            # method to be more than 10x faster than repeated hstack
            # and about 2x faster than temporary bool index, which can be found in commit 063bcce
            # make sure s.step is None so my simple length calculation will work
            assert all([isinstance(s, slice) and s.step is None for s in inds])
            max_length = np.sum([s.stop - s.start for s in inds])
            output_dtype = self.a.dtype  # get the dtype to preallocate with
            output_prealloc = np.zeros(max_length, output_dtype)
            ilo, ihi = 0, 0
            for s in inds:
                tmp = self._indexOffWithCuts(s, cutRecipeName)
                ilo = ihi
                ihi = ilo + len(tmp)
                output_prealloc[ilo:ihi] = tmp
            output = output_prealloc[0:ihi]
        elif isinstance(inds, list) and _listMethodSelect == 0:  # repeated hstack
            # this could be removed, along with the _listMethodSelect argument
            # this is only left in because it is useful for correctness testing for
            # preallocate and truncate method since this is simpler
            assert all([isinstance(_inds, slice) for _inds in inds])
            output = self._indexOffWithCuts(inds[0], cutRecipeName)
            for i in range(1, len(inds)):
                output = np.hstack((output, self._indexOffWithCuts(inds[i], cutRecipeName)))
        elif isinstance(inds, NoCutInds):
            output = self.offFile
        else:
            raise Exception(f"type(inds)={type(inds)}, should be slice or list or slices")
        return output

    @property
    def _offAttrs(self):
        return self.a.dtype.names

    def __repr__(self):
        return f"{self.__class__.__name__} with shortName={self.shortName}"


class ChannelGroupFromNpArrays(ChannelGroup):
    def __init__(self, channels, shortname,
                 verbose=True, experimentStateFile=None):
        collections.OrderedDict.__init__(self)
        self._shortName = shortname
        self.verbose = verbose
        self.experimentStateFile = experimentStateFile
        self._includeBad = False
        for ds in channels:
            self[ds.channum] = ds
        self._default_cut_recipe_name = self.firstGoodChannel()._default_cut_recipe_name

    def __repr__(self):
        return f"{self.__class__.__name__} with shortName={self.shortName}"

    @property
    def shortName(self):
        return self._shortName
