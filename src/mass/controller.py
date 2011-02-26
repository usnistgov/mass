"""
Created on Feb 16, 2011

@author: fowlerj
"""

print 'Loading controller module.'

from param_dict_base import PrmDictBase 

class AnalysisControl(PrmDictBase):
    """Control the behavior of an analysis operation."""
    
    def __init__(self, **kwargs):
        super(AnalysisControl, self).__init__()

        self.experiment_plan_prm = {}
        
        # Cuts follow certain heuristic rules by default (set to None).
        # If the 
        self.cuts_prm = {
            'peak_time_ms':        (-0.5, 0.5),
            'rise_time_ms':        (0,.5),
            'max_posttrig_deriv':  None,
            'pretrigger_rms':      None,
            'pretrigger_mean':     None,
            'min_pulse_average':   None,
            'min_value':           None,
        }
        self.analysis_prm = {}
        
        self._prm_list = [self.cuts_prm,]# self.experiment_plan_prm, self.analysis_prm]
        self._type_check.update({})
        self.user_prm = None   # No extra user parameters
        self.set(**kwargs)
        
def standardControl():
    ac = AnalysisControl()
    ac.set( peak_time_ms       = (-0.5, 0.5),
            rise_time_ms       = (0, 0.2),
            max_posttrig_deriv = (None, 46),
            pretrigger_rms     = (None, 12),
            pretrigger_mean    = (None, 1150),
            min_pulse_average  = (0, None),
            min_value          = (-50, None),
    )

    return ac