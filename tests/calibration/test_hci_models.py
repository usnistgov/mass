import mass.calibration.hci_models
import numpy as np
import pytest

def test_hci_models():
    models = mass.calibration.hci_models.models()
    bin_centers = np.arange(10)
    counts = np.arange(10)
    for key, model in models.items():
        params = model.make_params()
        model.eval(bin_centers=bin_centers, params=params)
    model = models["O 660eV Region"] 
    result = model.fit(counts, bin_centers=bin_centers) # just testing that it runs, the input data is nonsense