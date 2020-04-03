# std lib imports
import inspect
import collections
import logging

# pkg imports
import progress.bar
import pylab as plt
import numpy as np

# user imports
import mass

LOG = logging.getLogger("mass")

class Recipe():
    """
    If `r` is a Recipe, it is a wrapper around a function `f` and the names of its arguments.
    Arguments can either be names to be provided in a dictionary `d` when `r(d)` is called, or
    argument can be Recipe.
    `r(d)` where d is a dict mappring the names of argument to values will call `f` with the appropriate arguments, and also
    evaulate arguments which are recipes.

    The reasons this exists is so I can get a list of all the argument I need from the off file, so I can read from the off file
    a single time to evaluate a recipe that may depend on many values from the off file. My previous implementation would make multiple
    reads to the off file.
    """

    def __init__(self, f, argNames=None, inverse=None):
        assert not isinstance(f, Recipe)
        self.f = f
        self.inverse = inverse
        self.args = collections.OrderedDict()  # assumes the dict preserves insertion order
        try:
            inspectedArgNames = list(inspect.signature(self.f).parameters)  # Py 3.3+ only??
        except AttributeError:
            try:
                inspectedArgNames = inspect.getargspec(self.f).args  # Pre-Py 3.3
            except TypeError:
                inspectedArgNames = inspect.getargspec(self.f.__call__).args
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

    def setArgToRecipe(self, argName, r):
        assert isinstance(r, Recipe)
        assert argName in self.args
        self.args[argName] = r

    @property
    def argsL(self):
        "return the 'left side' arguments.... aka what self.f calls them"
        return list(self.args.keys())

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


def get_model(lineNameOrEnergy):
    if isinstance(lineNameOrEnergy, mass.GenericLineModel):
        line = lineNameOrEnergy.spect
    elif isinstance(lineNameOrEnergy, str):
        if lineNameOrEnergy in mass.spectra:
            line = mass.spectra[lineNameOrEnergy]
        elif lineNameOrEnergy in mass.STANDARD_FEATURES:
            energy = mass.STANDARD_FEATURES[lineNameOrEnergy]
            line = mass.SpectralLine.quick_monochromatic_line(lineNameOrEnergy, energy, 0.001, 0)
    else:
        try:
            energy = float(lineNameOrEnergy)
        except:
            raise Exception(f"lineNameOrEnergy = {lineNameOrEnergy} is not convertable to float or a str in mass.spectra or mass.STANDARD_FEATURES")
        line = mass.SpectralLine.quick_monochromatic_line(f"{lineNameOrEnergy}eV", float(lineNameOrEnergy), 0.001, 0)
    return line.model()