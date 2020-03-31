## Fluorescence Lines

Mass includes numerous features to help you analyze and model the fluorescence emission of various elements. Mass can

1. Approximate the shape of the fluorescence line emission for certain lines (particularly the K-alpha and K-beta lines of elements from Mg to Zn, or Z=12 to 30).
2. Generate random deviates, drawn from these same energy distributions.
3. Fit a measured spectrum to on of these energy distributions.

### Examples

#### 1. Plot the distribution

Objects of the `SpectralLine` class are callable, and return their PDF given the energy as an array or scalar argument.
```python
import mass
import numpy as np
import pylab as plt
spectrum = mass.MnKAlpha()
plt.clf()
axis=plt.gca()
for fwhm in (3,4,5,6,8,10):
    spectrum.plot(axis=axis,components=False,label="{}".format(fwhm),setylim=False),instrument_gaussian_fwhm=fwhm);
plt.legend(loc="upper left")
plt.title("Mn K$\\alpha$ distribution at various resolutions")
plt.xlabel("Energy (eV)")
```

#### 2. Generate random deviates from a fluorescence line shape

Objects of the `SpectralLine` class roughly copy the API of the scipy type `scipy.stats.rv_continuous` and offer some of the methods, such as `pdf`, `rvs`.:

```python
energies0 = spectrum.rvs(size=20000, instrument_gaussian_fwhm=0)
energies3 = spectrum.rvs(size=20000, instrument_gaussian_fwhm=3) 

plt.clf()
contents0, bin_edges0, _ = plt.hist(energies0, 200, [5820,5960], histtype="step")
contents3, bin_edges3, _ = plt.hist(energies3, 200, [5820,5960], histtype="step")
plt.xlabel("Energy (eV)")
plt.ylabel("Counts per bin")
```

#### 3. Fit data to a fluorescence line model
```python
fitter = mass.MnKAlphaFitter()
fitter.fit(contents3, bin_edges3, plot=False)
fitter.plot(label="full")
ph,ph_err = fitter.last_fit_params_dict["peak_ph"]
print("peak height = {} +/- {}".format(ph,ph_err))
```

#### 4. Alternate Lookup Method
```python
spectrum2 = mass.spectrum_classes["VKAlpha"]()
fitter2 = mass.fitter_classes["ScKAlpha"]()
```
