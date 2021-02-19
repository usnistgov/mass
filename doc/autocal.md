# Autocal
## Automatic Energy Calibration

When you want to calibrate a TES detector using a series of X-ray emission lines
of chemical elements or any sharp peaks in a X-ray spectrum, you can use
`mass.calibration.algorithm.EnergyCalibrationAutocal`.

The simplest way to calibrate (manipulating a
`mass.calibration.energy_calibration.EnergyCalibration` object) uses the
`EnergyCalibrationAutocal.autocal` method. If you supply the name of emission
line, it tries to fit data using a corresponding `MultiLorentzianComplexFitter`
If only peak position (number in the eV units) is given, it uses the
`GaussianLineFitter` to fit the data.

```python
from mass.calibration.energy_calibration import EnergyCalibration
from mass.calibration.algorithm import EnergyCalibrationAutocal

#  Suppose we have a numpy array of pulse heights and know that the names of X-ray emission
#  lines or energies of sharp peaks in X-ray spectrum that data consist of.
#  pulse_heights (numpy.array(dtype=float)): a numpy array of pulse heights.
#  line_names (list[str or float]): names of emission lines or energies of X-ray feature in eV unit.
#  e.g. line_names = ['ScKAlpha', 4460.5, 'FeKAlpha', 'FeKBeta', 'AsKAlpha', 11726.2]

cal = EnergyCalibration()
# If you want the calibration spline to go exactly through data points.
cal.set_use_approximation(False)
auto_cal = EnergyCalibrationAutocal(cal, pulse_heights, line_names)
auto_cal.autocal()  # This method modifies the cal object underneath it.
```

Before data are fitted with corresponding line fitters,
`EnergyCalibrationAutocal.autocal` needs to determine how to build histograms
which will be subsequently fed into line fitters. Its default parameters usually
work for chemical elements from Ti to Cu on tupac. Sometimes you need to adjust
these histogram parameters before histograms are handed into line fitters. In
this case you can split `auto_cal.autocal` into `auto_cal.guess_fit_params` and
`auto_cal.fit_lines` and adjust default histogram parameters between these
method calls by changing member variables such as `auto_cal.fit_lo_hi`,
`auto_cal.binsize_ev`, or `auto_cal.ph_opt`.

```python
cal = EnergyCalibration()
cal.set_use_approximation(False)  # If you want the calibration spline to go exactly through data points.
auto_cal = EnergyCalibrationAutocal(cal, pulse_heights, line_names)

auto_cal.guess_fit_params(fit_range_ev=100, maxacc=0.24)  # Initial guess parameters are determined.
# your customizations goes below.
gakb_idx = auto_cal.line_names.index("GaKBeta")
auto_cal.fit_lo_hi[gakb_idx] = (auto_cal.ph_opt[gakb_idx] * 0.9975, auto_cal.ph_opt[gakb_idx] * 1.0025)
aska_idx = auto_cal.line_names.index("AsKAlpha")
auto_cal.fit_lo_hi[aska_idx] = (auto_cal.ph_opt[aska_idx] * 0.99, auto_cal.ph_opt[aska_idx] * 1.007)
auto_cal.binsize_ev[aska_idx] = 3.0
# your customizations are finished.
auto_cal.fit_lines()  # Histograms are constructed and fitted with corresponding line fitters.

auto_cal.diagnose()
plt.show()
```

When you need to calibrate a TES detector (`mass.core.channel.MicroDataSet`)
with any of its fields such as `p_pulse_rms`, `p_filt_value`, or
`p_filt_value_dc`, you can use `mass.core.channel.MicrocalDataSet.calibrate`
method. If you want to customize histogram parameters before any of line fitters
use these histograms, you need to supply a closure that modifies any of member
variables `mass.calibration.algorithm.EnergyCalibrationAutocal`, which will be
called before `EnergyCalibrationAutocal.fit_lines` is called. Note that
`mass.core.channel.MicrocalDataSet.calibrate` does not actually calculate
`p_energy`. It only creates a calibration spline.

```python
def param_adjust_closure(ds, auto_cal):
    gakb_idx = auto_cal.line_names.index("GaKBeta")
    auto_cal.fit_lo_hi[gakb_idx] = (auto_cal.ph_opt[gakb_idx] * 0.9975, auto_cal.ph_opt[gakb_idx] * 1.0025)
    aska_idx = auto_cal.line_names.index("AsKAlpha")
    auto_cal.fit_lo_hi[aska_idx] = (auto_cal.ph_opt[aska_idx] * 0.99, auto_cal.ph_opt[aska_idx] * 1.007)
    auto_cal.binsize_ev[aska_idx] = 3.0

line_names = ['ScKAlpha', 4460.2, 'TiKAlpha', 'TiKBeta']

# It calibrates the MicrocalDataSet ds using its ds.p_filt_value_dc[ds.good()].
ds.calibrate('p_filt_value_dc', line_names, param_adjust_closure=param_adjust_closure)
```

You can calibrate all of good channels of `mass.core.channel_group.TESGroup`
using `mass.core.channel_group.TESGroup.calibrate`. This method works just like
`mass.core.channel.MicrocalDataSet.calibrate`. But there are a couple of
differences. One is that it catches any of exceptions and set those channels bad
and proceed with next channel. And the other is that it calculates energies and
populates `MicrocalDataSet.p_energy` field.

```python
def param_adjust_closure(ds, auto_cal):
    gakb_idx = auto_cal.line_names.index("GaKBeta")
    auto_cal.fit_lo_hi[gakb_idx] = (auto_cal.ph_opt[gakb_idx] * 0.9975, auto_cal.ph_opt[gakb_idx] * 1.0025)
    aska_idx = auto_cal.line_names.index("AsKAlpha")
    auto_cal.fit_lo_hi[aska_idx] = (auto_cal.ph_opt[aska_idx] * 0.99, auto_cal.ph_opt[aska_idx] * 1.007)
    auto_cal.binsize_ev[aska_idx] = 3.0

line_names = ['ScKAlpha', 4460.2, 'TiKAlpha', 'TiKBeta']

 # It calibrates the MicrocalDataSet ds using its ds.p_filt_value_dc[ds.good()].
data.calibrate('p_filt_value_dc', line_names, param_adjust_closure=param_adjust_closure)

data.why_chan_bad  # Any of failed channels are added into this dictionary.

ds = data.channel[1]
ds.p_energy[ds.good()]  # This field is populated unless ds.channum in data.why_chan_bad
```
