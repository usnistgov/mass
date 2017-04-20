import numpy as np
import pylab as pl

import mass

import test_regression

cuts = mass.core.controller.AnalysisControl(
    pulse_average=(0.0, None),
    pretrigger_rms=(None, 70),
    pretrigger_mean_departure_from_median=(-50, 50),
    peak_value=(0.0, None),
    postpeak_deriv=(0, 30),
    rise_time_ms=(None, 0.2),
    peak_time_ms=(None, 0.2)
)
data = test_regression.process_file("regress", cuts)
ds = data.datasets[0]

np.savez("regress_ds0",
         p_peak_value=ds.p_peak_value,
         p_postpeak_deriv=ds.p_postpeak_deriv,
         p_peak_index=ds.p_peak_index,
         p_peak_time=ds.p_peak_time,
         p_pretrig_mean=ds.p_pretrig_mean,
         p_pretrig_rms=ds.p_pretrig_rms,
         p_pulse_average=ds.p_pulse_average,
         p_pulse_rms=ds.p_pulse_rms,
         p_rise_time=ds.p_rise_time,
         p_filt_value=ds.p_filt_value,
         p_filt_value_dc=ds.p_filt_value_dc,
         good=ds.good(),
         bad=ds.bad(),
         )
