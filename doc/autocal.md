When you want to calibrate a TES detector using a series of X-ray emission lines of chemical elements or any sharp features in a X-ray spectrum, you can use mass.calibration.algorithm.EnergyCalibrationAutocal.

The most simple way to calibrate is using the `EnergyCalibrationAutocal.autocal` method. If you supply the name of emission line, it tries to fit data using a corresponding `MultiLorentzianComplexFitter` And if only peak position (number in the eV unit) is given, it uses the `GuassianLineFitter`.
```python3
from mass.calibration.algorithm import EnergyCalibrationAutocal

#  Suppose we have a numpy array of pulse_height and know that the names of emission lines that these data consist of
#  or energies of sharp features in X-ray spectrum.
#  pulse_heights (numpy.array(dtype=np.float)): a numpy array of pulse heights.
#  line_names (list[str or float]): names of emission lines or energies of X-ray feature in eV unit.
#  e.g. line_names = ['ScKAlpha', 4460.5, 'FeKAlpha', 'FeKBeta', 'AsKAlpha', 11726.2]

cal = EnergyCalibrationAutocal()
cal.set_use_approximation(False)  # If you want the calibration spline to go exactly through data points.
cal.autocal(pulse_heights, line_names)
```

`EnergyCalibrationAutocal.autocal` needs to determine how to build histograms for line fitters. Its default parameters usually works for chemical elements from Ti to Cu. Sometimes you need to adjust these histogram parameters before histograms are handed into line fitters.
In this case you can split `cal.autocal` into `cal.guess_fit_params` and `cal.fit_lines` and adjust default histogram parameters before line fitting starts. 

```python3
cal = EnergyCalibrationAutocal()
cal.set_use_approximation(False)  # If you want the calibration spline to go exactly through data points.
cal.ph, cal.line_names = pulse_heights, line_names
cal.guess_fit_params(maxacc=0.04)
# your customizations goes below.
gakb_idx = cal.line_names.index("GaKBeta")
cal.fit_lo_hi[gakb_idx] = (cal.ph_opt[gakb_idx] * 0.9975, cal.ph_opt[gakb_idx] * 1.0025)
aska_idx = cal.line_names.index("AsKAlpha")
cal.fit_lo_hi[aska_idx] = (cal.ph_opt[aska_idx] * 0.99, cal.ph_opt[aska_idx] * 1.007)
cal.binsize_ev[aska_idx] = 3.0
# your customizations are finished.
cal.fit_lines()
```