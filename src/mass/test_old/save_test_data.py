import numpy as np
import pylab as pl

import mass

import regression as regress

cuts = mass.core.controller.AnalysisControl(
        pulse_average=(0.0, 1700.0),
        pretrigger_rms=(None, 5.5),
        pretrigger_mean_departure_from_median=(-50, 50),
        peak_value=(0.0, 17000.0),
        postpeak_deriv=(0, 3.5),
        rise_time_ms=(0.1, 0.45),
        peak_time_ms=(0.2, 0.7)
    )
data = regress.process_file("regress", cuts)
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
