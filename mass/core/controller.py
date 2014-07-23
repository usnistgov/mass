## @file  controller.py
# @brief Classes to control the behavior of a Mass analysis
#
# How we specify cuts, ... ???
# This module is not nearly finished, but it needs some imaginative new ideas.

"""
Created on Feb 16, 2011

@author: fowlerj
"""

__all__ = ['AnalysisControl', 'standardControl']

from param_dict_base import PrmDictBase

class AnalysisControl(PrmDictBase):
    """Control the behavior of an analysis operation."""

    def __init__(self, **kwargs):
        """Build a set of cuts, plans, and analysis dictionaries with default
        values."""
        super(AnalysisControl, self).__init__()

        ## Not clear what we'll use this for (placeholder).
        self.experiment_plan_prm = {}

        # Cuts follow certain heuristic rules by default (set to None).
        ## The cuts parameters.
        self.cuts_prm = {
            'peak_value':          None,
            'peak_time_ms':        None,
            'rise_time_ms':        None,
            'postpeak_deriv':      None,
            'pretrigger_rms':      None,
            'pretrigger_mean':     None,
            'pulse_average':       None,
            'min_value':           None,
            'timestamp_sec':       None,
            'pretrigger_mean_departure_from_median': None,
            'timestamp_diff_sec':  None,
            'energy':              None,
        }

        ## The analysis parameters (not used yet).
        self.analysis_prm = {
            'pulse_averaging_ranges': None,
        }

        ## The full list of parameters
        self._prm_list = [self.cuts_prm, self.analysis_prm,]# self.experiment_plan_prm]
        self._type_check.update({})
        ## No extra user parameters
        self.user_prm = None
        self.set(**kwargs)


def standardControl():
    """Create a standard set of cuts.  (Not all that useful, since cuts can very so
    hugely!)"""
    ac = AnalysisControl()
    ac.set( peak_time_ms       = (-0.5, 0.5),
            rise_time_ms       = (0, 0.2),
            postpeak_deriv     = (None, 46),
            pretrigger_rms     = (None, 12),
            pretrigger_mean    = (None, None),
            pretrigger_mean_departure_from_median = (-50, 50),
            pulse_average      = (0, None),
            peak_value         = (0, None),
            min_value          = (-50, None),
    )

    return ac
