import collections
from typing import Union
import mass
import numpy as np


class ExperimentStateFile:
    def __init__(self, filename: str = None, datasetFilename: str = None, excludeStates: str = "auto", _parse: bool = True):
        """
        excludeStates - when "auto" it either exclude no states (if START and STOP are the only states)
            or excludes START, STOP, END and IGNORE (if states other than START exist).
            If not "auto", should be a list of states to exclude.
        _parse - is only for testing, can be used to prevent parsing on init
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
        self.labelAliasesDict: dict[str, str] = {}  # map unaliasedLabels to aliasedLabels
        self._preventAliasState = False  # causes aliasState to raise an Exception when it wouldn't work as expected

    @staticmethod
    def experimentStateFilenameFromDatasetFilename(datasetFilename):
        basename, _channum = mass.ljh_util.ljh_basename_channum(datasetFilename)
        return basename + "_experiment_state.txt"

    def parse(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            # if we call parse a second time, we want to add states rather than reparse the whole file
            f.seek(self.parse_start)
            lines = f.readlines()
            parse_end = f.tell()
        if self.parse_start == 0:
            header_line = lines[0]
            if header_line[0] != "#":
                raise Exception(f"first line should start with #, was {header_line}")
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
        """
        What labels should be excluded by the "auto" keyword?
        In a normal experiment, where there are non-trivial experiment states, we want to exclude all
        the start/end data, hence the list normally_ignore. If, however, the normally ignored states
        are the only oes, then you only want to ignore the states explicitly named IGNORE.
        """
        normally_ignore = ["START", "STOP", "END", "IGNORE"]
        nontrivial_labels = set(self.allLabels) - set(normally_ignore)
        if len(nontrivial_labels) == 0:
            return ["IGNORE"]
        return normally_ignore

    def applyExcludesToLabels(self, allLabels):
        """
        Recalculate self.excludeStates (possibly).
        Return a list of state labels that is unique, and contains all entries in allLabels except those in self.excludeStates
        order in the returned list is that of first appearance in allLabels
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
        calculate statesDict, an ordered dictionary mapping state name to EITHER a slice OR a list of slices
        equal to unixnanos. Slices are used for unique states; list of slices are used for repeated states.
        When updating pass in the existing statesDict and i0 must be the first label in allLabels that wasn't
        used to calculate the existing statesDict.
        """
        # unixnanos = timestamps of new records
        # i0_unixnanos is how many records have been state-indexed
        if statesDict is None:
            statesDict = collections.OrderedDict()

        newLabels = self.allLabels[i0_allLabels:]

        if statesDict is None:
            statesDict = collections.OrderedDict()

        # if the statesDict already exists, and there are no new states, update the active state and return the statesDict.
        if len(statesDict.keys()) > 0 and len(newLabels) == 0:
            assert i0_allLabels > 0
            for k in statesDict.keys():
                last_key = k
                s = statesDict[last_key]
            if isinstance(s, slice):
                # set the slice from the start of the state to the last new record
                s2 = slice(s.start, i0_unixnanos + len(unixnanos))
            if isinstance(s, list):
                s_ = s[-1]  # get last instance of same state
                # set the slice from the start of the state to the last new record
                s[-1] = slice(s_.start, i0_unixnanos + len(unixnanos))
                s2 = s
            statesDict[k] = s2
            return statesDict

        # unixnanos = new record timestamps
        # self.unixnanos[i0_allLabels] is the state start times of the new states
        # i0_unixnanos is how many records were alraedy indexed
        # inds is an np.array of the indices where the new states fit
        #   in with the new records
        inds = np.searchsorted(unixnanos, self.unixnanos[i0_allLabels:]) + i0_unixnanos
        # the state that was active last time calcStatesDict was called may need special handling
        if len(statesDict.keys()) > 0 and len(newLabels) > 0:
            assert i0_allLabels > 0
            for k in statesDict.keys():
                last_key = k
            s = statesDict[last_key]
            s2 = slice(s.start, inds[0])
            statesDict[k] = s2
        # iterate over self.allLabels because it corresponds to self.unixnanos
        for i, label in enumerate(newLabels):
            if label not in self.unaliasedLabels:
                continue
            aliasedLabel = self.labelAliasesDict.get(label, label)
            if i + 1 >= len(inds):
                s = slice(inds[i], len(unixnanos) + i0_unixnanos)
            else:
                s = slice(inds[i], inds[i + 1])
            if aliasedLabel in statesDict:
                # this label is not unique; use a list of slices
                v = statesDict[aliasedLabel]
                if isinstance(v, slice):
                    # this label was previously unique... create the list of slices
                    statesDict[aliasedLabel] = [v, s]
                elif isinstance(v, list):
                    # this label was previously not unique... append to the list of slices
                    statesDict[aliasedLabel] = v + [s]
                else:
                    msg = f"v should be a slice or list of slices, v is a {type(v)} for label={label}, aliasedlabel={aliasedLabel}"
                    raise Exception(msg)
            else:  # this state is unique, use a slice
                statesDict[aliasedLabel] = s
        # statesDict values should be slices for unique states and lists of slices for non-unique states
        self._preventAliasState = True
        return statesDict

    def __repr__(self):
        return "ExperimentStateFile: " + self.filename

    def aliasState(self, unaliasedLabel: Union[str, list[str]], aliasedLabel: str) -> None:
        assert isinstance(aliasedLabel, str)
        if self._preventAliasState:
            raise Exception("call aliasState before calculating or re-calculating statesDict")
        if isinstance(unaliasedLabel, list):
            for _unaliasedLabel in unaliasedLabel:
                self.labelAliasesDict[_unaliasedLabel] = aliasedLabel
        elif isinstance(unaliasedLabel, str):
            self.labelAliasesDict[unaliasedLabel] = aliasedLabel
        else:
            raise Exception(f"invalid type for unaliasedLabel={unaliasedLabel}")

    @property
    def labels(self) -> list[str]:
        return [self.labelAliasesDict.get(label, label) for label in self.unaliasedLabels]
