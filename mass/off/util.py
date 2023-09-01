# std lib imports
import inspect
import collections
import logging
import sys

# pkg imports
import progress.bar
import pylab as plt
import numpy as np

# user imports
import mass
import dill  # use dill to allow recipe books to pickel lambdas
import os

LOG = logging.getLogger("mass")


class NoCutInds:
    pass


class InvalidStatesException(Exception):
    pass


class RecipeBook:
    def __init__(self, baseIngredients, propertyClass=None, wrapper=lambda x: x):
        """
        propertyClass - which class to add properties to, if they are to be added
        wrapper - used in Recipe.craft to allow both "coefs" and "filtValue" to be ingredients,
            while "filtValue" refers to a sub-array of "coefs"
        """
        self.craftedIngredients = collections.OrderedDict()
        self.baseIngredients = baseIngredients  # list of names of base ingredients that will be passed to craft
        self.propertyClass = propertyClass  # class that properites will be added to
        self.wrapper = wrapper

    def add(self, recipeName, f, ingredients=None, overwrite=False, inverse=None, createProperty=True):
        """
        recipeName - the name of the new ingredient to create, eg "energy", the name "__temp__" is used internall, do not use it.
        recipe names with "cut" in them are special, only use them for cuts
        f - the function used to craft (evaluate) a new output or ingredient
        ingredients - a list of igredients names, used if the names of the arguments of f are not ingredient names
        createProperty - if True will create a property such that you can access the output of the recipe as eg `ds.energy`
        """
        # add a recipe
        # 1. create the recipe
        # 2. point at other recipes for items not in baseIngredients
        # 3. add to craftedIngredients with key recipeName
        # 4. create a property in propertyClass with name recipeName

        # nomenclature:
        # the function f takes arguments
        # the recipe collects ingredients, and passes them as arguments
        # the dictionary i2a maps ingredients to arguments
        assert isinstance(ingredients, list) or ingredients is None
        inspectedArgNames = list(inspect.signature(f).parameters)
        if "self" in inspectedArgNames:  # drop the self argument for class methods
            inspectedArgNames.remove("self")
        i2a = collections.OrderedDict()
        if ingredients is None:
            # learn ingredient names from signature of f
            for argName in inspectedArgNames:
                ingredient = argName
                assert ingredient in self.baseIngredients or ingredient in self.craftedIngredients, \
                    f"ingredient='{ingredient}' must be in baseIngredients={self.baseIngredients} or "\
                    f"craftedIngredients.keys()={list(self.craftedIngredients.keys())}"
                i2a[ingredient] = argName
        else:
            # i would like to do == here, but i'd need to handle optional arguments better
            assert len(inspectedArgNames) >= len(ingredients)
            for ingredient, inspectedArgName in zip(ingredients, inspectedArgNames):
                assert ingredient in self.baseIngredients or ingredient in self.craftedIngredients
                i2a[ingredient] = inspectedArgName

        recipe = Recipe(f, i2a, inverse, recipeName, self.wrapper)
        for ingredient in i2a:
            if ingredient in self.craftedIngredients:
                recipe._setIngredientToRecipe(ingredient, self.craftedIngredients[ingredient])
        if recipeName == "__temp__":
            return recipe
        assert recipeName not in self.craftedIngredients or overwrite, \
            f"recipeName={recipeName} already in self.craftedIngredients with keys={list(self.craftedIngredients.keys())}"
        assert not recipeName.startswith("!")
        self.craftedIngredients[recipeName] = recipe
        # recipes are added to the class, so only do it once per recipeName
        if self.propertyClass is not None:
            if createProperty and not hasattr(self.propertyClass, recipeName):
                setattr(self.propertyClass, recipeName, property(
                    lambda argself: argself.getAttr(recipeName, NoCutInds())))
        return recipe

    def __len__(self):
        return len(self.craftedIngredients)

    def keys(self):
        return self.craftedIngredients.keys()

    def __getitem__(self, i):
        return self.craftedIngredients[i]

    def craft(self, recipeName, ingredientSource):
        """
        Craft (evaluate the reciple) recipeName.
        recipeName - a key in self.craftedIngredients
        ingredientSource - a dict like object with keys in self.baseIngredients
        """
        if callable(recipeName):
            return self._craftWithFunction(recipeName, ingredientSource)
        elif "cut" in recipeName:
            return self._craftCut(recipeName, ingredientSource)
        elif recipeName in self.craftedIngredients:
            r = self.craftedIngredients[recipeName]
            return r(ingredientSource)
        else:
            raise Exception(
                f"recipeName={recipeName} must be in self.cratftedIngredients or callalbe")

    def _craftWithFunction(self, f, ingredientSource, ingredients=None):
        """
        Create a temporary recipe from f and then craft it.
        f - a function to be passed to self.add
        ingredientSource - a dict like object with keys in self.baseIngredients
        """
        r = self.add("__temp__", f, ingredients)
        return r(ingredientSource)

    def _craftCut(self, cutRecipeName, ingredientSource):
        cutBaseRecipeName = cutRecipeName.lstrip("!")
        numberBang = len(cutRecipeName) - len(cutBaseRecipeName)
        assert cutBaseRecipeName in self.craftedIngredients
        r = self.craftedIngredients[cutBaseRecipeName]
        if numberBang % 2 == 0:
            return r(ingredientSource)
        else:
            return np.logical_not(r(ingredientSource))

    def __repr__(self):
        bi = ", ".join(self.baseIngredients)
        ci = ", ".join(self.craftedIngredients.keys())
        s = f"RecipeBook: baseIngedients={bi}, craftedIngredeints={ci}"
        return s

    def to_file(self, path, overwrite=False):
        if not overwrite and os.path.isfile(path):
            raise Exception(
                f"file {path} already exists, use overwrite=True if you want to clobber it")
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            r = dill.load(f)
        assert isinstance(r, cls)
        return r


class Recipe:
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

    def __init__(self, f, i2a, inverse, name, ingredientWrapper):
        assert not isinstance(f, Recipe)
        self.f = f
        self.inverse = inverse
        self.i2a = i2a
        self.name = name
        self.ingredientWrapper = ingredientWrapper

    def _setIngredientToRecipe(self, ingredient, r):
        assert isinstance(r, Recipe)
        assert ingredient in self.i2a
        self.i2a[ingredient] = r

    def __call__(self, ingredientSource):
        args = []
        for (k, v) in self.i2a.items():
            if isinstance(v, Recipe):
                args.append(v(ingredientSource))
            else:
                wrapped = self.ingredientWrapper(ingredientSource)
                args.append(wrapped[k])
        # call functions with positional arguments so names don't need to match
        return self.f(*args)

    def __repr__(self, indent=0):
        ingredients_strs = []
        for k, v in self.i2a.items():
            if isinstance(v, Recipe):
                ingredients_strs.append(f"Recipe:{k}")
            else:
                ingredients_strs.append(k)
        i = ", ".join(ingredients_strs)
        s = f"Recipe {self.name}: f={self.f}, ingredients={i}, inverse={self.inverse}"
        return s


class GroupLooper:
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
    is_running_tests = "pytest" in sys.modules

    def wrapper(self, *args, **kwargs):
        bar = SilenceBar(method_name, max=len(self.offFileNames), silence=not self.verbose)
        rethrow = kwargs.pop("_rethrow", is_running_tests)  # always rethrow during tests
        returnVals = collections.OrderedDict()
        for (channum, ds) in self.items():
            try:
                z = method(ds, *args, **kwargs)
                returnVals[channum] = z
            except KeyboardInterrupt as e:
                raise (e)
            except Exception as e:
                ds.markBad(f"{e} during {method_name}", e)
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
        lines.append(f"\n{method_name}{argtext} has no docstring")
    else:
        lines.append(f"\n{method_name}{argtext} docstring reads:")
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
    "A progress bar that can be turned off by passing silence=True or by setting the log level higher than NOTSET"

    def __init__(self, message, max, silence):
        self.silence = silence
        if not silence:
            if not LOG.isEnabledFor(logging.WARN):
                self.silence = True
        if not self.silence:
            super().__init__(message, max=max)
        else:
            # The following prevents an ugly warning in the base class destructor.
            self._hidden_cursor = False

    def next(self):
        if not self.silence:
            progress.bar.Bar.next(self)

    def finish(self):
        if not self.silence:
            progress.bar.Bar.finish(self)

#ratio between standard deviation and median absolute deviation for a gaussian distribution
SIGMA_OVER_MAD = 1/0.67449


def median_absolute_deviation(x):
    """calculate the median_absolute_deviation and its sigma equivalent assuming gaussian distribution
    returns mad, sigma_equiv, median
    """
    median = np.median(x)
    mad = np.median(np.abs(x-median))
    sigma_equiv = mad*SIGMA_OVER_MAD
    return mad, sigma_equiv, median


class IngredientsWrapper:
    """
    I'd like to be able to do either ingredient_source["coefs"] to get all projection coefficients
    or ingredient_source["filtValue"] to get only the filtValue
    IngredientsWrapper lets that work within recipes.craft
    IngredientWrapper is part of that hack

    ingredient_source - an ndarray read from an off file with dtype that defined "filtValue"
    coefs_dtype - an alternate dtype that defines "coefs"
    """

    def __init__(self, ingredient_source, coefs_dtype):
        self.ingredient_source = ingredient_source
        self.coefs_dtype = coefs_dtype

    def __getitem__(self, k):
        if k == "coefs":
            return self.ingredient_source.view(self.coefs_dtype)["coefs"]
        else:
            return self.ingredient_source[k]

    def view(self, coefs_dtype):
        return self.ingredient_source.view(coefs_dtype)


def iterstates(states):
    """
    states can be a
    str
    list of str
    in either case return a list of str
    """
    if isinstance(states, str):
        return [states]
    else:
        assert isinstance(states, list)
        return states
